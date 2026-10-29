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
/// Instead of one residual vector, the block carries `hc_mult` copies. Each
/// sub-layer reads a mixing row (from `hc_*_fn`, biased by `hc_*_base` and scaled
/// by `hc_*_scale`) and combines the streams with it; the sub-layer's output is
/// added back into every stream with that stream's `hc_*_scale`. The block's
/// "hidden state" for the next sub-layer is stream 0.
///
/// The reference describes Sinkhorn normalization of the mixing matrix before it
/// is used, and `sinkhorn` below implements it, but the reference's own forward is
/// not verified to apply it and both backends currently read the matrix row as
/// published. That makes `sinkhorn` dormant rather than wrong; it stays here (and
/// as `dsv41_hc_sinkhorn`) so wiring it is a one-line change once a reference
/// vector settles the question.
nonisolated public enum DeepSeekV41HyperConnections {

    /// Sinkhorn normalization of a `dimension × dimension` matrix.
    ///
    /// Exponentiates once and then alternates row and column normalization
    /// `iterations` times, leaving rows and columns each summing to 1. Serial by
    /// design: `dimension` is `hc_mult` (4 on the published checkpoint), so the
    /// coordination a parallel version needs costs more than the loop.
    public static func sinkhorn(
        logits: [Float],
        dimension: Int,
        iterations: Int,
        epsilon: Float
    ) -> [Float] {
        let n = dimension
        guard n > 0, logits.count >= n * n else { return [] }
        // Subtract the max first: a large logit would otherwise exponentiate to
        // infinity and every later row/column sum would be `inf`.
        let shift = logits.prefix(n * n).max() ?? 0
        var matrix = (0..<(n * n)).map { exp(logits[$0] - shift) }
        let floor = max(epsilon, 1e-12)
        for _ in 0..<max(0, iterations) {
            for row in 0..<n {
                let base = row * n
                var sum: Float = 0
                for column in 0..<n { sum += matrix[base + column] }
                let inverse = 1 / max(sum, floor)
                for column in 0..<n { matrix[base + column] *= inverse }
            }
            for column in 0..<n {
                var sum: Float = 0
                for row in 0..<n { sum += matrix[row * n + column] }
                let inverse = 1 / max(sum, floor)
                for row in 0..<n { matrix[row * n + column] *= inverse }
            }
        }
        return matrix
    }

    /// Per-stream read weights departing from a mixing matrix.
    ///
    /// The matrix mixes streams; the sub-layer only consumes one vector, so the
    /// pre-mix collapses to one weighted sum per input and only the mixing matrix
    /// row matters. Written out because "take row 0" is not obvious from the
    /// matrix alone and getting it wrong silently drops half the residual path.
    public static func readWeights(matrix: [Float], streams: Int, row: Int = 0) -> [Float] {
        guard streams > 0, matrix.count >= streams * streams else {
            return [Float](repeating: 1 / Float(max(1, streams)), count: max(0, streams))
        }
        let base = max(0, min(row, streams - 1)) * streams
        return Array(matrix[base..<(base + streams)])
    }

    /// Combines the input streams with per-stream read weights, then applies the
    /// per-stream `scale` and the shared `base` bias. This is the vector a
    /// sub-layer sees.
    public static func mix(
        streams: [[Float]],
        readWeights: [Float],
        scale: [Float]?,
        bias: [Float]?
    ) -> [Float] {
        guard let first = streams.first else { return [] }
        let width = first.count
        var out = [Float](repeating: 0, count: width)
        for (index, stream) in streams.enumerated() {
            let weight = index < readWeights.count ? readWeights[index] : 0
            let streamScale = (scale?.count ?? 0) == streams.count ? scale![index] : 1
            let streamBias = (bias?.count ?? 0) == width ? bias! : nil
            let count = min(width, stream.count)
            for dim in 0..<count {
                let biasValue = streamBias?[dim] ?? 0
                out[dim] += weight * (stream[dim] * streamScale + biasValue)
            }
        }
        return out
    }

    /// Writes a sub-layer's output back into the streams without mixing them
    /// together: each stream keeps its own identity and only accumulates the
    /// sub-layer's contribution. `scale` is the write scale (`hc_*_scale`), which
    /// is what keeps a deep stack of sub-layers from amplifying one another.
    public static func write(
        streams: inout [[Float]],
        contribution: [Float],
        writeScales: [Float]?
    ) {
        guard !streams.isEmpty else { return }
        for index in streams.indices {
            let scale = (writeScales?.count ?? 0) == streams.count ? writeScales![index] : 1
            let count = min(streams[index].count, contribution.count)
            for dim in 0..<count {
                streams[index][dim] += scale * contribution[dim]
            }
        }
    }
}
