//
//  DeepSeekV41Attention.swift
//  DynaMoE
//
//  CSA2 attention: the math, as a CPU reference plus the per-layer shape the
//  Metal path binds against.
//
//  The architecture is latent-attention (MLA) with two twists that no other
//  family in this repo has:
//
//    1. One latent record per position is used as BOTH the key and the value for
//       every head. There is no per-head K/V up-projection: `attn.wkv` produces
//       the latent, every query head dots against it, and the attention output is
//       itself a `v_head_dim`-wide vector that `attn.wo_a` / `attn.wo_b` project
//       back to the hidden width. So `v_head_dim` and the latent width agree, and
//       the scored set is a single shared row.
//
//    2. Each head carries a learnable SINK logit. It enters the softmax
//       denominator with no matching value row, so it lets a head attend to
//       "nothing" (and therefore emit a near-zero output) when no cached position
//       is relevant. Dropping it does not fail loudly: it silently adds
//       probability mass to whatever rows happen to exist, which reads as a model
//       that is confidently wrong about long contexts.
//
//  Everything here is pure: no Metal, no I/O, so the reference below is what the
//  kernels are tested against.
//

import Foundation

// MARK: - Shape

/// The YaRN block one layer's rope table is built with.
///
/// The checkpoint resolves this per layer, not per model: a layer that compresses
/// its KV extrapolates with `compress_rope_theta` and the configured factor, while a
/// pure sliding-window layer resets `original_seq_len` to 0 and rotates at plain
/// `rope_theta`. Using one block for both is a silent long-context degradation
/// because only the compressed half reaches beyond the window.
nonisolated public struct DeepSeekV41RopeScaling: Sendable, Equatable {
    public let factor: Float
    public let originalLength: Int
    public let betaFast: Float
    public let betaSlow: Float

    /// No extrapolation: every frequency passes through untouched.
    public static let none = DeepSeekV41RopeScaling(
        factor: 1, originalLength: 0, betaFast: 32, betaSlow: 1
    )

    public init(factor: Float, originalLength: Int, betaFast: Float, betaSlow: Float) {
        self.factor = factor
        self.originalLength = originalLength
        self.betaFast = betaFast
        self.betaSlow = betaSlow
    }
}

/// Sizes the per-layer attention dispatch binds: how many query heads, how wide
/// the query and output are, and how the compressed rows are read.
nonisolated public struct DeepSeekV41AttentionShape: Sendable, Equatable {
    public let numHeads: Int
    public let headsPerGroup: Int
    public let qkHeadDim: Int
    public let vHeadDim: Int
    /// Values in one cached latent record (key and value width).
    public let latentWidth: Int
    public let qLoraRank: Int
    public let oLoraRank: Int
    public let isCompressed: Bool
    public let compressRatio: Int
    public let ropeTheta: Float
    /// Rope payload width. Zero means the whole latent is rotated.
    public let rotaryDim: Int
    /// The YaRN block this layer's rope tables are built with.
    public let ropeScaling: DeepSeekV41RopeScaling

    public var queryWidth: Int { numHeads * qkHeadDim }
    public var outputWidth: Int { numHeads * vHeadDim }
    /// Head groups the output projection is block-diagonal over (`o_groups`).
    /// `headsPerGroup` was already resolved from it, so the quotient recovers the
    /// group count without a second config field to keep in step.
    public var outputGroups: Int { max(1, numHeads / max(1, headsPerGroup)) }
    /// Whether this layer rotates its whole latent rather than a decoupled slice.
    public var rotatesWholeLatent: Bool { rotaryDim == 0 }

    /// Width of the rotated slice inside one query head. A decoupled payload is the
    /// TRAILING `qkRopeHeadDim` values of the head, so the `d_nope` prefix never
    /// rotates; without one the whole head is the payload.
    public var queryRopeWidth: Int {
        rotatesWholeLatent ? qkHeadDim : min(rotaryDim, qkHeadDim)
    }

    /// Width of the rotated slice inside one cache record.
    public var latentRopeWidth: Int {
        rotatesWholeLatent ? latentWidth : min(rotaryDim, latentWidth)
    }

    public var queryRopeStart: Int { qkHeadDim - queryRopeWidth }
    public var latentRopeStart: Int { latentWidth - latentRopeWidth }

    /// Whether one attention-output head is shaped like a cache record, and so
    /// carries the same rope payload its keys do. The reference conjugates that
    /// payload out of the output before the output projection.
    public var outputCarriesRope: Bool { vHeadDim == latentWidth }

    /// Start of the rotated slice inside one attention-output head.
    public var outputRopeStart: Int { vHeadDim - latentRopeWidth }

    public init(
        numHeads: Int,
        headsPerGroup: Int,
        qkHeadDim: Int,
        vHeadDim: Int,
        latentWidth: Int,
        qLoraRank: Int,
        oLoraRank: Int,
        isCompressed: Bool,
        compressRatio: Int,
        ropeTheta: Float,
        rotaryDim: Int,
        ropeScaling: DeepSeekV41RopeScaling = .none
    ) {
        self.numHeads = numHeads
        self.headsPerGroup = headsPerGroup
        self.qkHeadDim = qkHeadDim
        self.vHeadDim = vHeadDim
        self.latentWidth = latentWidth
        self.qLoraRank = qLoraRank
        self.oLoraRank = oLoraRank
        self.isCompressed = isCompressed
        self.compressRatio = compressRatio
        self.ropeTheta = ropeTheta
        self.rotaryDim = rotaryDim
        self.ropeScaling = ropeScaling
    }

    public static func make(config: DeepSeekV41Config, layer: DeepSeekV41LayerPlan) -> DeepSeekV41AttentionShape {
        DeepSeekV41AttentionShape(
            numHeads: config.numAttentionHeads,
            headsPerGroup: max(1, config.numAttentionHeads / max(1, config.oGroups)),
            qkHeadDim: config.qkHeadDim,
            vHeadDim: config.vHeadDim,
            latentWidth: config.cacheRecordWidth,
            qLoraRank: config.qLoraRank,
            oLoraRank: config.oLoraRank,
            isCompressed: layer.flavor == .compressed,
            compressRatio: layer.compressRatio,
            ropeTheta: layer.flavor == .compressed ? config.compressRopeTheta : config.ropeTheta,
            rotaryDim: config.qkRopeHeadDim,
            ropeScaling: layer.ropeTable == .compress
                ? DeepSeekV41RopeScaling(
                    factor: config.ropeFactor,
                    originalLength: config.ropeOriginalLength,
                    betaFast: config.ropeBetaFast,
                    betaSlow: config.ropeBetaSlow
                )
                : .none
        )
    }

    /// Softmax scale. Latent attention scores each head's dot product against the
    /// shared latent, so the scale is over `qkHeadDim`, not the latent width.
    public var scoreScale: Float { 1 / sqrt(Float(qkHeadDim == 0 ? 1 : qkHeadDim)) }
}

// MARK: - RoPE

/// Rotary embedding, with the YaRN-by-parts correction the checkpoint's
/// `rope_scaling` block asks for.
///
/// Everything here mirrors the checkpoint's own `inference/model.py`
/// (`precompute_freqs_cis` and `apply_rotary_emb`), because the two rules that look
/// interchangeable are not:
///
///  * The ramp runs in DIMENSION space, not wavelength space. The reference finds the
///    dimension whose wavelength completes `rotations` turns over the original
///    context (`corrected_dim`) and blends linearly between the two thresholds.
///    Ramping on the wavelength instead is the same thresholds on a different scale,
///    which divides dimensions early and quietly shortens usable context.
///  * Values pair ADJACENTLY (`v[2i]` with `v[2i + 1]`), the `view_as_complex` layout.
///    Pairing the two halves of the slice (`v[i]` with `v[i + width/2]`) is the other
///    rope convention in circulation and produces different scores for the same
///    weights, so a CPU/GPU parity test cannot catch getting this wrong.
nonisolated public enum DeepSeekV41RoPE {

    /// Per-pair inverse frequency for a width, theta and scaling block.
    public static func inverseFrequencies(
        width: Int,
        theta: Float,
        factor: Float,
        originalLength: Int,
        betaFast: Float,
        betaSlow: Float
    ) -> [Float] {
        let pairs = width / 2
        guard pairs > 0 else { return [] }
        let scaling = factor > 1 && originalLength > 0 && theta > 1
        var low = 0
        var high = 0
        if scaling {
            // The dimension whose wavelength finishes `rotations` turns inside the
            // original context: wavelength(2d / width) = 2π / theta^(2d/width), so
            // d = width * ln(length / (rotations * 2π)) / (2 * ln(theta)).
            let logTheta = 2 * log(theta)
            func correctedDimension(_ rotations: Float) -> Float {
                let numerator = log(Float(originalLength) / (max(rotations, 1) * 2 * .pi))
                return Float(width) * numerator / logTheta
            }
            low = max(Int(correctedDimension(betaFast).rounded(.down)), 0)
            high = min(Int(correctedDimension(betaSlow).rounded(.up)), width - 1)
        }
        var out: [Float] = []
        out.reserveCapacity(pairs)
        for index in 0..<pairs {
            let exponent = (2 * Float(index)) / Float(width)
            let invFreq = 1 / pow(theta, exponent)
            guard scaling else {
                out.append(invFreq)
                continue
            }
            // Linear ramp in dimension space, clamped to [0, 1]: nothing below the
            // beta_fast dimension moves, everything past the beta_slow dimension is
            // divided by the factor.
            let ramp = min(max((Float(index) - Float(low)) / max(Float(high - low), 1e-3), 0), 1)
            out.append(invFreq * (1 - ramp * (1 - 1 / factor)))
        }
        return out
    }

    /// Rotates one range of a vector in place, pairing ADJACENT values. `inverse`
    /// conjugates the rotation, which is how the attention output's rope payload is
    /// removed again so the cache can stay in one shared rotated form.
    private static func rotateRange(
        _ vector: inout [Float],
        start: Int,
        length: Int,
        position: Int,
        frequencies: [Float],
        inverse: Bool
    ) {
        guard start >= 0, length > 1, start + length <= vector.count else { return }
        let pairs = min(frequencies.count, length / 2)
        guard pairs > 0 else { return }
        let pos = Float(position) * (inverse ? -1 : 1)
        for index in 0..<pairs {
            let angle = pos * frequencies[index]
            let cosValue = cos(angle)
            let sinValue = sin(angle)
            let low = start + 2 * index
            let high = low + 1
            let lowValue = vector[low]
            let highValue = vector[high]
            vector[low] = lowValue * cosValue - highValue * sinValue
            vector[high] = lowValue * sinValue + highValue * cosValue
        }
    }

    /// Rotates one contiguous slice of a vector in place.
    public static func rotateSlice(
        _ vector: inout [Float],
        start: Int,
        length: Int,
        position: Int,
        frequencies: [Float],
        inverse: Bool = false
    ) {
        rotateRange(
            &vector,
            start: start,
            length: length,
            position: position,
            frequencies: frequencies,
            inverse: inverse
        )
    }

    /// Rotates one whole vector in place.
    public static func rotate(
        _ vector: inout [Float],
        position: Int,
        frequencies: [Float],
        inverse: Bool = false
    ) {
        let length = vector.count
        rotateRange(
            &vector,
            start: 0,
            length: length,
            position: position,
            frequencies: frequencies,
            inverse: inverse
        )
    }

    /// Rotates a concatenation of heads, each rotated inside its OWN `headDim` block.
    ///
    /// `sliceStart`/`sliceLength` select the rotated slice within each head: a decoupled
    /// rope payload is the trailing `qkRopeHeadDim` values, and rotating the `d_nope`
    /// prefix with it would mix the two subspaces the checkpoint trained separately.
    ///
    /// This is deliberately not `rotate`: the query is `numHeads * headDim` values, and
    /// rotating it as one vector pairs head 0's first value with head
    /// `numHeads/2`'s, which mixes independent heads. `dsv41_rope_latent` rotates per
    /// head, so the reference has to as well or the two backends disagree by
    /// construction.
    public static func rotateHeads(
        _ vector: inout [Float],
        heads: Int,
        headDim: Int,
        sliceStart: Int = 0,
        sliceLength: Int? = nil,
        position: Int,
        frequencies: [Float],
        inverse: Bool = false
    ) {
        guard heads > 0, headDim > 1 else { return }
        let start = min(max(sliceStart, 0), headDim - 1)
        let length = min(sliceLength ?? headDim - start, headDim - start)
        guard length > 1 else { return }
        for head in 0..<heads {
            let base = head * headDim
            guard base + headDim <= vector.count else { return }
            rotateRange(
                &vector,
                start: base + start,
                length: length,
                position: position,
                frequencies: frequencies,
                inverse: inverse
            )
        }
    }

    /// Inverse frequencies for the query rail: each head rotates its trailing
    /// `qkRopeHeadDim` values when the payload is decoupled, and the whole head
    /// otherwise. The table is built over that slice, never over the head width.
    public static func queryFrequencies(for shape: DeepSeekV41AttentionShape) -> [Float] {
        inverseFrequencies(
            width: max(1, shape.queryRopeWidth),
            theta: shape.ropeTheta,
            factor: shape.ropeScaling.factor,
            originalLength: shape.ropeScaling.originalLength,
            betaFast: shape.ropeScaling.betaFast,
            betaSlow: shape.ropeScaling.betaSlow
        )
    }

    /// Inverse frequencies for the latent rail: the same rotated slice, taken from
    /// the cache record instead of from a query head.
    public static func latentFrequencies(for shape: DeepSeekV41AttentionShape) -> [Float] {
        inverseFrequencies(
            width: max(1, shape.latentRopeWidth),
            theta: shape.ropeTheta,
            factor: shape.ropeScaling.factor,
            originalLength: shape.ropeScaling.originalLength,
            betaFast: shape.ropeScaling.betaFast,
            betaSlow: shape.ropeScaling.betaSlow
        )
    }
}

// MARK: - Attention reference

nonisolated public enum DeepSeekV41AttentionReference {

    /// CSA2 attention for one query position.
    ///
    /// - Parameters:
    ///   - query: `numHeads * qkHeadDim` values, already rotated.
    ///   - keys: one latent record per attendable position, in attention order
    ///     (window first, then compressed rows oldest to newest).
    ///   - sink: per-head sink logit, or nil to disable.
    /// - Returns: `numHeads * vHeadDim` values.
    ///
    /// The sink contributes `exp(sink - max)` to the denominator only, and the
    /// output is the softmax-weighted sum of the latent rows. A position set whose
    /// entire score mass lands on the sink yields an all-zero output, which is the
    /// behavior that makes the sink useful.
    public static func attend(
        query: [Float],
        keys: [[Float]],
        shape: DeepSeekV41AttentionShape,
        sink: [Float]? = nil,
        scale: Float? = nil
    ) -> [Float] {
        let heads = shape.numHeads
        let width = shape.latentWidth
        var output = [Float](repeating: 0, count: heads * shape.vHeadDim)
        guard heads > 0, width > 0 else { return output }
        let scoreScale = scale ?? shape.scoreScale

        for head in 0..<heads {
            let queryBase = head * shape.qkHeadDim
            let queryWidth = min(shape.qkHeadDim, width)
            var maxScore = -Float.greatestFiniteMagnitude
            var scores: [Float] = []
            scores.reserveCapacity(keys.count)
            for key in keys {
                var dot: Float = 0
                for dim in 0..<queryWidth {
                    dot += query[queryBase + dim] * key[dim]
                }
                let score = dot * scoreScale
                scores.append(score)
                if score > maxScore { maxScore = score }
            }
            // The sink is an extra logit with no value row: it can win the
            // softmax, in which case the head emits zeros.
            let sinkLogit = sink.map { $0[head] } ?? -Float.greatestFiniteMagnitude
            if sinkLogit > maxScore { maxScore = sinkLogit }
            if scores.isEmpty && sink == nil { continue }

            var denominator: Float = 0
            var accumulator = [Float](repeating: 0, count: shape.vHeadDim)
            for (index, score) in scores.enumerated() {
                let weight = exp(score - maxScore)
                denominator += weight
                let key = keys[index]
                let valueWidth = min(shape.vHeadDim, key.count)
                for dim in 0..<valueWidth {
                    accumulator[dim] += weight * key[dim]
                }
            }
            if sink != nil {
                denominator += exp(sinkLogit - maxScore)
            }
            let inverse = denominator > 0 ? 1 / denominator : 0
            let outputBase = head * shape.vHeadDim
            for dim in 0..<shape.vHeadDim {
                output[outputBase + dim] = accumulator[dim] * inverse
            }
        }
        return output
    }

    /// The softmax weights CSA2 assigns, in attention order, including the sink
    /// as a final entry. Exposed so the tests can assert the sink's share
    /// directly instead of inferring it from the output.
    public static func weights(
        query: [Float],
        head: Int,
        keys: [[Float]],
        shape: DeepSeekV41AttentionShape,
        sink: Float?
    ) -> [Float] {
        var scores: [Float] = []
        let queryBase = head * shape.qkHeadDim
        let queryWidth = min(shape.qkHeadDim, shape.latentWidth)
        for key in keys {
            var dot: Float = 0
            for dim in 0..<queryWidth {
                dot += query[queryBase + dim] * key[dim]
            }
            scores.append(dot * shape.scoreScale)
        }
        var all = scores
        if let sink { all.append(sink) }
        guard let maxScore = all.max() else { return [] }
        let exponentials = all.map { exp($0 - maxScore) }
        let total = exponentials.reduce(0, +)
        guard total > 0 else { return all.map { _ in 0 } }
        return exponentials.map { $0 / total }
    }
}

// MARK: - Compressor

/// How a run of `compress_ratio` positions is pooled into one compressed row.
///
/// The rule is `Compressor.forward` in `inference/model.py`: project the layer's
/// post-`attn_norm` input through `attn.compressor.wkv` (the latent) and through
/// `attn.compressor.wgate` (the scores), then combine the group with
/// `(kv * score.softmax(dim=2)).sum(dim=2)`.
///
/// Two details that a flat description of "softmax-gated pooling" hides, and that
/// the earlier port got wrong in both directions:
///
///  * **The gate is per DIMENSION, not per position.** `score` has the same
///    `head_dim` width as `kv`, and the softmax runs over the group axis of a
///    `[batch, groups, ratio, head_dim]` tensor, so every one of the `head_dim`
///    channels gets its own distribution across the group's positions. There is no
///    single weight per row.
///  * **The pooled row is normalised**, by `RMSNorm(head_dim)` with the
///    `attn.compressor.norm` gain, and only then rotated and stored. Skipping the
///    norm leaves the row at a scale the compressed cache never trained for.
///
/// A ratio of 1 is a plain projection: one token per group means nothing to pool,
/// and the checkpoint publishes no `wgate` for those layers at all.
nonisolated public enum DeepSeekV41Compressor {

    /// The pooled (still un-normalised, un-rotated) latent for one completed group.
    ///
    /// - Parameters:
    ///   - inputs: one entry per position in the group, in ascending position
    ///     order. These are the post-`attn_norm` block inputs, not the cached
    ///     latents: the compressor projects the layer input, it does not re-pool
    ///     what the window cache already holds.
    ///   - kvWeight: `attn.compressor.wkv`, `[head_dim][hidden]`.
    ///   - gateWeight: `attn.compressor.wgate`, `[head_dim][hidden]`, or nil for a
    ///     ratio of 1 (plain projection, no gate).
    public static func pooledLatent(
        inputs: [[Float]],
        kvWeight: DeepSeekV41Matrix?,
        gateWeight: DeepSeekV41Matrix?
    ) -> [Float] {
        guard let first = inputs.first else { return [] }
        let width = kvWeight?.rows ?? first.count
        guard width > 0 else { return [] }
        var pooled = [Float](repeating: 0, count: width)

        guard let kvWeight else { return pooled }
        let values = inputs.map { kvWeight.apply($0) }
        guard values.count > 1, let gateWeight else {
            // One position per group, or no gate published: the mean is the group.
            let inverse = 1 / Float(values.count)
            for row in values {
                let count = min(width, row.count)
                for dim in 0..<count {
                    pooled[dim] += inverse * row[dim]
                }
            }
            return pooled
        }

        let scores = inputs.map { gateWeight.apply($0) }
        var weights = [Float](repeating: 0, count: values.count)
        for dim in 0..<width {
            var maxScore = -Float.greatestFiniteMagnitude
            for row in 0..<scores.count where dim < scores[row].count {
                maxScore = max(maxScore, scores[row][dim])
            }
            guard maxScore > -Float.greatestFiniteMagnitude else { continue }
            var total: Float = 0
            for row in 0..<scores.count where dim < scores[row].count {
                weights[row] = exp(scores[row][dim] - maxScore)
                total += weights[row]
            }
            guard total > 0 else { continue }
            for row in 0..<values.count where dim < values[row].count {
                pooled[dim] += (weights[row] / total) * values[row][dim]
            }
        }
        return pooled
    }

    /// Softmax-weighted convex combination of one block of rows, with a single
    /// weight per row. Kept for the ratio-1-free cases and for the tests that pin
    /// the convex-hull property; the per-dimension rule above is the reference.
    public static func pool(latents: [[Float]], gates: [Float]) -> [Float] {
        guard let first = latents.first else { return [] }
        let width = first.count
        guard latents.count == gates.count, !gates.isEmpty else {
            return [Float](repeating: 0, count: width)
        }
        let maxGate = gates.max() ?? 0
        var weights = gates.map { exp($0 - maxGate) }
        let total = weights.reduce(0, +)
        if total > 0 {
            weights = weights.map { $0 / total }
        } else {
            weights = [Float](repeating: 1 / Float(gates.count), count: gates.count)
        }
        var out = [Float](repeating: 0, count: width)
        for (index, latent) in latents.enumerated() {
            let weight = weights[index]
            let count = min(width, latent.count)
            for dim in 0..<count {
                out[dim] += weight * latent[dim]
            }
        }
        return out
    }

    /// Every compressed row a position's history has produced, pooling each block
    /// of `ratio` latents. Partial trailing blocks are dropped: a compressed row is
    /// only complete once its whole run has been seen.
    public static func compress(
        latents: [[Float]],
        gates: [Float],
        ratio: Int
    ) -> [[Float]] {
        guard ratio > 0, latents.count >= ratio else { return [] }
        var out: [[Float]] = []
        out.reserveCapacity(latents.count / ratio)
        var start = 0
        while start + ratio <= latents.count {
            out.append(pool(
                latents: Array(latents[start..<(start + ratio)]),
                gates: Array(gates[start..<(start + ratio)])
            ))
            start += ratio
        }
        return out
    }
}

// MARK: - Indexer

/// The candidate pass: score blocks of compressed rows so attention only reads the
/// few most relevant ones.
nonisolated public enum DeepSeekV41Indexer {

    /// Block scores for one query position.
    ///
    /// Mirrors the engine's Qwen-sparse-attention scorer: each block's score is the
    /// MAX over its rows and over the indexer query heads of
    /// `dot(q, key) / sqrt(keyWidth)`. Max rather than mean because a block is only
    /// worth reading if some row in it is relevant; averaging lets one strong match
    /// be diluted by its neighbours until it loses to a uniformly-mediocre block.
    public static func blockScores(
        queries: [[Float]],
        keys: [[Float]],
        blockSize: Int,
        keyWidth: Int
    ) -> [Float] {
        guard blockSize > 0, keyWidth > 0, !keys.isEmpty else { return [] }
        let invSqrt = 1 / sqrt(Float(keyWidth))
        let blockCount = (keys.count + blockSize - 1) / blockSize
        var scores = [Float](repeating: -Float.greatestFiniteMagnitude, count: blockCount)
        for block in 0..<blockCount {
            let start = block * blockSize
            let end = min(start + blockSize, keys.count)
            var blockMax = -Float.greatestFiniteMagnitude
            for row in start..<end {
                let key = keys[row]
                let width = min(keyWidth, key.count)
                for query in queries {
                    let queryWidth = min(width, query.count)
                    var dot: Float = 0
                    for dim in 0..<queryWidth {
                        dot += query[dim] * key[dim]
                    }
                    let score = dot * invSqrt
                    if score > blockMax { blockMax = score }
                }
            }
            scores[block] = blockMax == -Float.greatestFiniteMagnitude ? 0 : blockMax
        }
        return scores
    }

    /// The `topK` best blocks, returned in ascending block order so a gathered row
    /// list is deterministic and a reused prefix addresses the same rows.
    public static func selectedBlocks(scores: [Float], topK: Int) -> [Int] {
        guard topK > 0, !scores.isEmpty else { return [] }
        if topK >= scores.count { return Array(scores.indices) }
        let ranked = scores.enumerated()
            .sorted { lhs, rhs in
                lhs.element == rhs.element ? lhs.offset < rhs.offset : lhs.element > rhs.element
            }
            .prefix(topK)
            .map(\.offset)
        return ranked.sorted()
    }

    /// Expands selected blocks into the compressed row indices they cover.
    public static func rows(
        for blocks: [Int],
        blockSize: Int,
        rowCount: Int
    ) -> [Int] {
        guard blockSize > 0, rowCount > 0 else { return [] }
        var out: [Int] = []
        out.reserveCapacity(blocks.count * blockSize)
        for block in blocks.sorted() {
            let start = block * blockSize
            guard start < rowCount else { continue }
            let end = min(start + blockSize, rowCount)
            for row in start..<end {
                out.append(row)
            }
        }
        return out
    }
}
