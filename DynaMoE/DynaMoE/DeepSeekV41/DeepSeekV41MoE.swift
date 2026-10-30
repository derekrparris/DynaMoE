//
//  DeepSeekV41MoE.swift
//  DynaMoE
//
//  Routing, the expert arena, and the hyper-connection residual streams.
//
//  Three things here differ from the Mixtral/Qwen MoE paths in the rest of the
//  engine:
//
//    1. Routing is top-k over a SCORED logit with a per-expert selection bias, not
//       a softmax gate. `scoring_func` on this checkpoint is `sqrtsoftplus`, and a
//       softmax over 384 experts would make every logit compete with 383 others, so
//       a checkpoint trained with this gate routes to a different set than its
//       weights expect. The bias is added to the SCORED value for SELECTION only
//       and never appears in the weight, which is what keeps a rarely-chosen expert
//       from being starved forever without corrupting the mixture of the ones that
//       are chosen.
//
//    2. The mixture is rescaled by `routed_scaling_factor` after normalizing over
//       the selected experts.
//
//    3. The routed experts are MXFP4 and there are 384 of them per layer, so the
//       arena that decides which expert matrices are resident is not an
//       optimization but the difference between running and not. Its policy is
//       LRU over whole (layer, expert) pairs: one global pool, and eviction picks
//       the least recently used pair whatever layer it came from, so a slot never
//       holds half an expert.
//
//  The activation function is the checkpoint's clipped SwiGLU: `silu(min(gate,
//  limit)) * clamp(up, -limit, limit)`. The clip is part of the trained function,
//  not a stability guard, so it cannot be dropped.
//

import Foundation

// MARK: - Routing

nonisolated public struct DeepSeekV41Routing: Sendable, Equatable {
    /// Expert ids, in descending score order.
    public let expertIDs: [Int]
    /// The weight applied to each expert's output, already normalized and scaled.
    public let weights: [Float]
    /// The selection values, i.e. `score(logit / gate_temp) + bias` per expert. Kept
    /// so the tests can assert the ranking instead of inferring it. These are NOT
    /// the values the mixture weights are normalized from; those are the unbiased
    /// scored values of the selected experts.
    public let scores: [Float]

    public init(expertIDs: [Int], weights: [Float], scores: [Float]) {
        self.expertIDs = expertIDs
        self.weights = weights
        self.scores = scores
    }

    public static let empty = DeepSeekV41Routing(expertIDs: [], weights: [], scores: [])
}

nonisolated public enum DeepSeekV41Router {

    /// The logistic function, computed so a large-magnitude input cannot overflow.
    public static func sigmoid(_ value: Float) -> Float {
        if value >= 0 {
            return 1 / (1 + exp(-value))
        }
        let exponential = exp(value)
        return exponential / (1 + exponential)
    }

    /// `scoring_func` applied to one gate logit.
    ///
    /// The checkpoint ships `sqrtsoftplus`, and the three options are not
    /// rescalings of each other: each is a different monotone map, so the same
    /// logits plus the same bias rank the experts differently under `sigmoid` than
    /// under `sqrtsoftplus`. Selecting on the wrong one runs the wrong experts.
    public static func score(_ logit: Float, function: DeepSeekV41ScoringFunction) -> Float {
        switch function {
        case .sigmoid:
            return sigmoid(logit)
        case .sqrtsoftplus:
            // `softplus(x).sqrt()`. Softplus is evaluated as
            // `max(x, 0) + log1p(exp(-|x|))` so a large positive logit cannot
            // overflow `exp`.
            let softplus = max(logit, 0) + log1p(exp(-abs(logit)))
            return sqrt(softplus)
        case .softmax:
            // Softmax is a distribution over ALL experts, so it cannot be applied
            // to one logit at a time. `route` normalizes the whole vector instead.
            return logit
        }
    }

    /// Routes one token.
    ///
    /// The reference is `Gate.forward`:
    ///
    ///     scores = linear(x) / gate_temp
    ///     scores = softmax | sigmoid | sqrt(softplus) applied to scores
    ///     indices = (scores + bias).topk(topk)
    ///     weights = scores[indices]  (normalized, then *= route_scale)
    ///
    /// The bias is added to the SCORED value, not to the raw logit and not after a
    /// second monotone map, and it steers selection only: the returning weights are
    /// the unbiased scores of the experts that were chosen.
    ///
    /// - Parameters:
    ///   - logits: router output, one per routed expert.
    ///   - bias: per-expert selection bias (`ffn.gate.bias`, or `bias_vl` inside an
    ///     image span), or nil.
    ///   - topK: experts to activate.
    ///   - scalingFactor: `routed_scaling_factor`; 0 or negative disables it.
    ///   - normalise: `norm_topk_prob`. The reference renormalizes only when
    ///     `topk > 1` and adds 1e-20 rather than `norm_eps`, to match training.
    ///   - scoringFunction: `scoring_func`.
    ///   - gateTemp: `gate_temp`; the logits are divided by it first.
    public static func route(
        logits: [Float],
        bias: [Float]?,
        topK: Int,
        scalingFactor: Float,
        normalise: Bool = true,
        scoringFunction: DeepSeekV41ScoringFunction = .sqrtsoftplus,
        gateTemp: Float = 1
    ) -> DeepSeekV41Routing {
        guard !logits.isEmpty, topK > 0 else { return .empty }
        let count = min(topK, logits.count)
        let temperature = gateTemp > 0 ? gateTemp : 1

        // Scored values, in the order the reference computes them.
        var scored: [Float]
        if scoringFunction == .softmax {
            var maxLogit = -Float.greatestFiniteMagnitude
            for logit in logits { maxLogit = max(maxLogit, logit / temperature) }
            var exponentials = logits.map { exp($0 / temperature - maxLogit) }
            let total = exponentials.reduce(0, +)
            if total > 0 {
                exponentials = exponentials.map { $0 / total }
            }
            scored = exponentials
        } else {
            scored = logits.map { score($0 / temperature, function: scoringFunction) }
        }

        // Selection sees `scores + bias`; the mixture sees `scores`.
        var selection = scored
        if let bias, bias.count == logits.count {
            for index in selection.indices { selection[index] += bias[index] }
        }

        // Rank by selection value, ties broken by expert id so a reused prefix
        // routes identically.
        let ranked = selection.indices.sorted { lhs, rhs in
            selection[lhs] == selection[rhs] ? lhs < rhs : selection[lhs] > selection[rhs]
        }
        let selected = Array(ranked.prefix(count))
        var selectedWeights = selected.map { scored[$0] }

        // `norm_topk_prob and topk > 1`: a single selected expert is left alone, and
        // the epsilon is the training-time 1e-20, not `norm_eps`.
        if normalise, count > 1 {
            let total = selectedWeights.reduce(0, +) + 1e-20
            if total > 0 {
                selectedWeights = selectedWeights.map { $0 / total }
            }
        }
        if scalingFactor > 0 {
            selectedWeights = selectedWeights.map { $0 * scalingFactor }
        }
        return DeepSeekV41Routing(
            expertIDs: selected,
            weights: selectedWeights,
            scores: Array(selection)
        )
    }
}

// MARK: - Activation

nonisolated public enum DeepSeekV41Activation {

    /// The checkpoint's clipped SwiGLU.
    public static func swiglu(gate: Float, up: Float, limit: Float) -> Float {
        var gateValue = gate
        var upValue = up
        if limit > 0 {
            gateValue = min(gateValue, limit)
            upValue = min(max(upValue, -limit), limit)
        }
        return silu(gateValue) * upValue
    }

    public static func silu(_ value: Float) -> Float {
        value / (1 + exp(-value))
    }

    /// The same function over a whole intermediate vector, so the reference and the
    /// kernel share one definition.
    public static func swiglu(gates: [Float], ups: [Float], limit: Float) -> [Float] {
        let count = min(gates.count, ups.count)
        var out = [Float](repeating: 0, count: count)
        for index in 0..<count {
            out[index] = swiglu(gate: gates[index], up: ups[index], limit: limit)
        }
        return out
    }
}

// MARK: - Expert arena

/// Slot assignment for the resident routed-expert matrices.
///
/// 384 experts per layer is far past what fits, so the arena holds a fixed number
/// of slots and swaps whole (layer, expert) pairs in and out. Two rules make it
/// correct rather than merely plausible:
///
///   * a slot is owned by exactly one (layer, expert), and
///   * eviction is LRU over the pairs, never over individual matrices, because a
///     half-resident expert is a wrong answer with no error.
nonisolated public final class DeepSeekV41ExpertArena: @unchecked Sendable {

    public struct Slot: Sendable, Equatable {
        public let layer: Int
        public let expert: Int
        /// Monotonic access counter, used only for LRU ordering.
        public let lastUsed: UInt64
    }

    private let lock = NSLock()
    private let slotCount: Int
    private var slots: [Slot?]
    private var counter: UInt64 = 0
    /// Total acquisitions, hits and misses, so the load log can report whether the
    /// arena is sized sanely for the traffic.
    public private(set) var acquisitions: Int = 0
    public private(set) var hits: Int = 0
    public private(set) var evictions: Int = 0

    public init(slotCount: Int) {
        self.slotCount = max(1, slotCount)
        self.slots = Array(repeating: nil, count: max(1, slotCount))
    }

    public var capacity: Int { slotCount }

    public var residentCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return slots.compactMap { $0 }.count
    }

    /// Reserves slots for `(layer, expert)` pairs and returns the slot index of
    /// each, in the order requested.
    ///
    /// Pairs already resident keep their slot, so the returned indices are stable
    /// across calls and a caller can rely on a slot's contents for the duration of
    /// one layer. When `pairs.count` exceeds `capacity`, some indices repeat: the
    /// arena cannot hold more slots than it has, so the later pair overwrites the
    /// slot an earlier pair in the same call received. Callers must therefore keep
    /// `pairs.count <= capacity` (top-k times one arena slot per layer is the
    /// intended use).
    public func acquire(_ pairs: [(layer: Int, expert: Int)]) -> [Int] {
        acquireDetailed(pairs).map(\.slot)
    }

    /// The same placement as `acquire`, reporting whether each pair was already
    /// resident.
    ///
    /// A caller that fills slots from disk needs the distinction: a hit means the
    /// slot already holds this pair's bytes, and a miss means the slot has to be
    /// written before any kernel reads it. Re-deriving it from
    /// `slot(forLayer:expert:)` before the call would race the placement the arena
    /// itself just made.
    public func acquireDetailed(
        _ pairs: [(layer: Int, expert: Int)]
    ) -> [(slot: Int, isHit: Bool)] {
        lock.lock()
        defer { lock.unlock() }
        var out: [(slot: Int, isHit: Bool)] = []
        out.reserveCapacity(pairs.count)
        for pair in pairs {
            acquisitions += 1
            if let existing = slots.firstIndex(where: { $0?.layer == pair.layer && $0?.expert == pair.expert }) {
                hits += 1
                counter += 1
                slots[existing] = Slot(layer: pair.layer, expert: pair.expert, lastUsed: counter)
                out.append((slot: existing, isHit: true))
                continue
            }
            let free = slots.firstIndex(where: { $0 == nil })
            let index: Int
            if let free {
                index = free
            } else {
                // Evict the least recently used pair.
                guard let lru = slots.indices.min(by: { (slots[$0]?.lastUsed ?? 0) < (slots[$1]?.lastUsed ?? 0) }) else {
                    out.append((slot: 0, isHit: false))
                    continue
                }
                index = lru
                evictions += 1
            }
            counter += 1
            slots[index] = Slot(layer: pair.layer, expert: pair.expert, lastUsed: counter)
            out.append((slot: index, isHit: false))
        }
        return out
    }

    /// Forgets what a slot holds, so the next `acquire` for that pair is a miss.
    ///
    /// Needed by the streaming path: a read that fails part way leaves a slot whose
    /// identity claims bytes it does not have, and `acquire` would serve it as a hit
    /// forever. The bytes are left alone; the identity is what decides whether they
    /// are ever read.
    public func vacate(slot: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard slot >= 0, slot < slots.count else { return }
        slots[slot] = nil
    }

    /// Slot index currently holding a pair, or nil when it is not resident.
    public func slot(forLayer layer: Int, expert: Int) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return slots.firstIndex { $0?.layer == layer && $0?.expert == expert }
    }

    public func reset() {
        lock.lock()
        slots = Array(repeating: nil, count: slotCount)
        acquisitions = 0
        hits = 0
        evictions = 0
        counter = 0
        lock.unlock()
    }

    /// One-line residency report.
    public var hitRate: Double {
        lock.lock()
        defer { lock.unlock() }
        guard acquisitions > 0 else { return 0 }
        return Double(hits) / Double(acquisitions)
    }
}

// MARK: - Hyper-connections

/// The mHC residual streams.
///
/// Instead of one residual vector the block carries `hc_mult` copies of it. A
/// sub-layer does not read a fixed stream: `hc_*_fn` projects the FLATTENED copies
/// (normalized by their own RMS) onto `(2 + mult) * mult` logits, and those split
/// into three coefficient sets — `pre` (how the copies collapse into the
/// sub-layer's input), `post` (how much of the sub-layer's output each copy takes)
/// and `comb` (a doubly-stochastic matrix mixing the copies back in).
///
/// Two details here are easy to get wrong and produce plausible text when you do.
/// The coefficients a sub-layer computes are consumed by the NEXT one, not by the
/// sub-layer that computed them (`Block.forward` derives attention's `pre`/`post`/
/// `comb` from the block input, then collapses that same input with the `pre` the
/// previous sub-layer produced), and the block's output is `hc_pre(h, pre_mix)` — a
/// weighted mix of the copies, not one of them.
nonisolated public enum DeepSeekV41HyperConnections {

    /// What one `hc_*_fn` projection splits into.
    public struct Coefficients: Sendable, Equatable {
        /// `mult` weights: how the copies collapse into the sub-layer's input.
        public let pre: [Float]
        /// `mult` gains on the sub-layer's output, one per copy.
        public let post: [Float]
        /// `mult × mult`, row-major and in the reference's own orientation:
        /// `comb[j][k]` is how much of copy `j`'s residual copy `k`'s output takes.
        /// The FIRST index is the residual the entry reads, not the copy it writes,
        /// because `hc_post` sums `comb.unsqueeze(-1) * residual.unsqueeze(-2)` over
        /// its middle axis. Reading a row as one copy's mixing weights is the
        /// transpose, and with two copies a nearly symmetric Sinkhorn matrix hides
        /// it.
        public let comb: [Float]
        /// The `(2 + mult) * mult` raw logits before the split. Kept because a
        /// probe run compares against them; nothing in the forward reads them.
        public let logits: [Float]

        public init(pre: [Float], post: [Float], comb: [Float], logits: [Float] = []) {
            self.pre = pre
            self.post = post
            self.comb = comb
            self.logits = logits
        }
    }

    /// `Block.hc_mixes`: one projection of the flattened stack, split three ways.
    ///
    /// The RMS normalization is over the WHOLE flattened stack, one statistic per
    /// token, which is what keeps the projection's scale independent of how many
    /// copies the checkpoint carries.
    public static func coefficients(
        streams: [[Float]],
        projection: DeepSeekV41Matrix?,
        scale: [Float],
        base: [Float],
        normalizationEpsilon: Float,
        epsilon: Float,
        iterations: Int
    ) -> Coefficients {
        let mult = max(1, streams.count)
        let width = streams.first?.count ?? 0
        let mixCount = (2 + mult) * mult
        guard width > 0,
              let projection,
              projection.rows >= mixCount,
              projection.columns == mult * width,
              scale.count >= 3,
              base.count >= mixCount else {
            // A checkpoint without usable `hc_*` tensors degenerates to a single
            // stream rather than a guess at coefficients: `pre` picks copy 0 and
            // `post`/`comb` keep every copy's own residual.
            return degenerate(mult: mult)
        }

        var flattened = [Float]()
        flattened.reserveCapacity(mult * width)
        for stream in streams {
            flattened.append(contentsOf: stream.prefix(width))
        }
        var sumSquares: Float = 0
        for value in flattened {
            sumSquares += value * value
        }
        let rstd = 1 / sqrt(sumSquares / Float(flattened.count) + normalizationEpsilon)
        let logits = projection.apply(flattened).prefix(mixCount).map { $0 * rstd }

        var pre = [Float](repeating: 0, count: mult)
        var post = [Float](repeating: 0, count: mult)
        for copy in 0..<mult {
            // `sigmoid(...) + eps`: the floor is what keeps a collapsed copy from
            // dropping out of the residual path entirely.
            pre[copy] = sigmoid(logits[copy] * scale[0] + base[copy]) + epsilon
            post[copy] = 2 * sigmoid(logits[mult + copy] * scale[1] + base[mult + copy])
        }
        var rawComb = [Float](repeating: 0, count: mult * mult)
        for source in 0..<mult {
            for target in 0..<mult {
                let at = 2 * mult + source * mult + target
                rawComb[source * mult + target] = logits[at] * scale[2] + base[at]
            }
        }
        return Coefficients(
            pre: pre,
            post: post,
            comb: sinkhorn(
                logits: rawComb,
                dimension: mult,
                iterations: iterations,
                epsilon: epsilon
            ),
            logits: logits
        )
    }

    /// `hc_split_sinkhorn`: one row softmax, then one column normalization, then
    /// `iterations - 1` rounds of row-then-column normalization.
    ///
    /// The epsilons are ADDED to the divisor rather than flooring it, and the
    /// softmax carries no epsilon at all: the first round therefore leaves the
    /// matrix's columns summing to 1 and its rows only approximately so, which is
    /// why the schedule (and not just the eventual convergence) is what the golden
    /// test pins. Serial by design — `dimension` is `hc_mult`, four on the
    /// published checkpoint.
    public static func sinkhorn(
        logits: [Float],
        dimension: Int,
        iterations: Int,
        epsilon: Float
    ) -> [Float] {
        let n = dimension
        guard n > 0, logits.count >= n * n else { return [] }
        // Per-row max, exactly like the reference: a row whose entries are far
        // below the global max would otherwise underflow to zeros and divide by a
        // zero sum.
        var matrix = [Float](repeating: 0, count: n * n)
        for row in 0..<n {
            let base = row * n
            var rowMax = -Float.greatestFiniteMagnitude
            for column in 0..<n {
                rowMax = max(rowMax, logits[base + column])
            }
            var sum: Float = 0
            for column in 0..<n {
                let value = exp(logits[base + column] - rowMax)
                matrix[base + column] = value
                sum += value
            }
            let inverse = sum > 0 ? 1 / sum : 0
            for column in 0..<n {
                matrix[base + column] *= inverse
            }
        }
        for index in matrix.indices {
            matrix[index] += epsilon
        }
        columnNormalize(&matrix, n, epsilon)
        for _ in 0..<max(0, iterations - 1) {
            rowNormalize(&matrix, n, epsilon)
            columnNormalize(&matrix, n, epsilon)
        }
        return matrix
    }

    /// `hc_pre`: collapse the copies into the one vector a sub-layer reads.
    public static func pre(streams: [[Float]], mix: [Float]) -> [Float] {
        guard let first = streams.first else { return [] }
        var out = [Float](repeating: 0, count: first.count)
        for (copy, stream) in streams.enumerated() {
            let weight = copy < mix.count ? mix[copy] : 0
            guard weight != 0 else { continue }
            let count = min(out.count, stream.count)
            for dim in 0..<count {
                out[dim] += weight * stream[dim]
            }
        }
        return out
    }

    /// `hc_post`: expand a sub-layer's output back to every copy and mix the
    /// residual in through `comb`. Returns the new streams.
    public static func post(
        output: [Float],
        residual: [[Float]],
        gains: [Float],
        comb: [Float]
    ) -> [[Float]] {
        let mult = residual.count
        guard mult > 0 else { return residual }
        var streams = residual
        for copy in 0..<mult {
            var row = [Float](repeating: 0, count: streams[copy].count)
            // `comb[source * mult + copy]`: the matrix is indexed source-first, so
            // copy `copy`'s output reads a COLUMN of it. Walking a row instead mixes
            // every copy with the wrong neighbours, which two copies can hide and
            // four cannot.
            for source in 0..<mult where source * mult + copy < comb.count {
                let weight = comb[source * mult + copy]
                let stream = residual[source]
                for dim in 0..<min(row.count, stream.count) {
                    row[dim] += weight * stream[dim]
                }
            }
            let gain = copy < gains.count ? gains[copy] : 0
            for dim in 0..<min(row.count, output.count) {
                row[dim] += gain * output[dim]
            }
            streams[copy] = row
        }
        return streams
    }

    /// The mix a stack starts from: copy 0 only, so the first sub-layer reads the
    /// embedding and nothing else (`make_identity_pre_mix`).
    public static func identityMix(copies: Int) -> [Float] {
        var mix = [Float](repeating: 0, count: max(1, copies))
        mix[0] = 1
        return mix
    }

    private static func degenerate(mult: Int) -> Coefficients {
        var comb = [Float](repeating: 0, count: mult * mult)
        for copy in 0..<mult {
            comb[copy * mult + copy] = 1
        }
        return Coefficients(
            pre: identityMix(copies: mult),
            post: [Float](repeating: 1, count: mult),
            comb: comb
        )
    }

    private static func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private static func rowNormalize(_ matrix: inout [Float], _ n: Int, _ epsilon: Float) {
        for row in 0..<n {
            let base = row * n
            var sum: Float = 0
            for column in 0..<n {
                sum += matrix[base + column]
            }
            let inverse = 1 / (sum + epsilon)
            for column in 0..<n {
                matrix[base + column] *= inverse
            }
        }
    }

    /// Sums along the ROWS (`dim=0` in the reference), so each entry is divided by
    /// its own column's total plus epsilon.
    private static func columnNormalize(_ matrix: inout [Float], _ n: Int, _ epsilon: Float) {
        for column in 0..<n {
            var sum: Float = 0
            for row in 0..<n {
                sum += matrix[row * n + column]
            }
            let inverse = 1 / (sum + epsilon)
            for row in 0..<n {
                matrix[row * n + column] *= inverse
            }
        }
    }
}
