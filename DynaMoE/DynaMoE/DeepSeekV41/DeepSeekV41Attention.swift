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

    public var queryWidth: Int { numHeads * qkHeadDim }
    public var outputWidth: Int { numHeads * vHeadDim }
    /// Whether this layer rotates its whole latent rather than a decoupled slice.
    public var rotatesWholeLatent: Bool { rotaryDim == 0 }

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
        rotaryDim: Int
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
            rotaryDim: config.qkRopeHeadDim
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
/// `rope_scaling.factor` with `original_max_position_embeddings` is the
/// NTK-by-parts form: dimensions whose wavelength is shorter than
/// `original_length / beta_fast` extrapolate untouched, dimensions whose
/// wavelength is longer than `original_length / beta_slow` are divided by the
/// factor, and dimensions in between blend linearly. Applying the factor to every
/// dimension (or to none) both look plausible and both degrade long contexts, so
/// the ramp lives here once and is asserted in the tests.
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
        var out: [Float] = []
        out.reserveCapacity(pairs)
        let twoPi = 2 * Float.pi
        for index in 0..<pairs {
            let exponent = (2 * Float(index)) / Float(width)
            let invFreq = 1 / pow(theta, exponent)
            guard factor > 1, originalLength > 0 else {
                out.append(invFreq)
                continue
            }
            let wavelength = twoPi / invFreq
            let low = Float(originalLength) / max(betaFast, 1)
            let high = Float(originalLength) / max(betaSlow, 1)
            let scale: Float
            if wavelength < low {
                scale = 1
            } else if wavelength > high {
                scale = 1 / factor
            } else {
                // Linear ramp between the two thresholds, clamped to [0, 1].
                let denominator = max(high - low, 1e-9)
                let ramp = min(max((wavelength - low) / denominator, 0), 1)
                scale = 1 - ramp * (1 - 1 / factor)
            }
            out.append(invFreq * scale)
        }
        return out
    }

    /// Rotates one vector in place. `vector` holds `pairs * 2` values laid out as
    /// two halves (`v[i]` paired with `v[i + width/2]`), which is the layout the
    /// rest of the engine's rope kernels use.
    public static func rotate(_ vector: inout [Float], position: Int, frequencies: [Float]) {
        let pairs = min(frequencies.count, vector.count / 2)
        guard pairs > 0 else { return }
        let half = vector.count / 2
        let pos = Float(position)
        for index in 0..<pairs {
            let angle = pos * frequencies[index]
            let cosValue = cos(angle)
            let sinValue = sin(angle)
            let low = vector[index]
            let high = vector[index + half]
            vector[index] = low * cosValue - high * sinValue
            vector[index + half] = low * sinValue + high * cosValue
        }
    }

    /// Rotates a concatenation of query heads, each head rotated over its OWN
    /// `headDim` (pair `i` with `i + headDim/2` inside the head).
    ///
    /// This is deliberately not `rotate`: the query is `numHeads * qkHeadDim`
    /// values, and rotating it as one vector pairs head 0's first dimension with
    /// head `numHeads/2`'s, which mixes independent heads and leaves head 0's
    /// upper half unrotated. `dsv41_rope_latent` rotates per head, so the
    /// reference has to as well or the two backends disagree by construction.
    public static func rotateHeads(
        _ vector: inout [Float],
        heads: Int,
        headDim: Int,
        position: Int,
        frequencies: [Float]
    ) {
        guard heads > 0, headDim > 1 else { return }
        let half = headDim / 2
        let pairs = min(frequencies.count, half)
        guard pairs > 0 else { return }
        let pos = Float(position)
        for head in 0..<heads {
            let base = head * headDim
            guard base + headDim <= vector.count else { return }
            for index in 0..<pairs {
                let angle = pos * frequencies[index]
                let cosValue = cos(angle)
                let sinValue = sin(angle)
                let low = vector[base + index]
                let high = vector[base + index + half]
                vector[base + index] = low * cosValue - high * sinValue
                vector[base + index + half] = low * sinValue + high * cosValue
            }
        }
    }

    public static func frequencies(for shape: DeepSeekV41AttentionShape, config: DeepSeekV41Config) -> [Float] {
        let width = shape.rotatesWholeLatent ? shape.latentWidth : shape.rotaryDim
        return inverseFrequencies(
            width: width,
            theta: shape.ropeTheta,
            factor: config.ropeFactor,
            originalLength: config.ropeOriginalLength,
            betaFast: config.ropeBetaFast,
            betaSlow: config.ropeBetaSlow
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
/// The checkpoint ships two tensors: `attn.compressor.wkv` (the latent side) and
/// `attn.compressor.wgate` (one gate logit per position). The pooling is a convex
/// combination of the block's latents weighted by `softmax(gates)`, which is the
/// only pooling that keeps the compressed row inside the convex hull of what it
/// summarizes regardless of gate magnitude, so a badly-scaled gate cannot blow the
/// cache up.
///
/// The exact reference combination (weighted mean vs. gated sum vs. last-row
/// selection) is what `Tools/dsv41_probe.py engram` settles; until then
/// `DeepSeekV41Conformance.compressorPooling` keeps the load log honest.
nonisolated public enum DeepSeekV41Compressor {

    /// Pools one block of latents. `gates` has one entry per latent.
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
