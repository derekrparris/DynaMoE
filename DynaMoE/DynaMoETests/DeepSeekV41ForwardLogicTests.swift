//
//  DeepSeekV41ForwardLogicTests.swift
//  DynaMoE
//
//  Coverage for the parts of the DeepSeek-V4.1 forward that do not need the
//  checkpoint: attention with its sink, RoPE with the YaRN-by-parts ramp, the
//  compressor and indexer, routing, the expert arena, the hyper-connection
//  streams, the CSA2 cache layout, and the block forward end to end over
//  synthetic weights.
//
//  The end-to-end case is the important one: it is the only place that proves the
//  layer order, the residual bookkeeping and the cache addressing hang together.
//  It cannot prove the numbers are right (the weights are synthetic) but it does
//  prove that a position reads exactly the rows it wrote, that a compressed row
//  appears exactly when its block completes, and that the head sees the collapsed
//  stream.
//

import XCTest
@testable import DynaMoE

// MARK: - Shared fixtures

enum DeepSeekV41TestFixtures {

    /// A small but structurally complete config: 3 layers, real CSA2 geometry, two
    /// routed experts activated out of four, two hyper-connection streams.
    static let json = """
    {
      "architectures": ["DeepseekV41ForCausalLM"],
      "model_type": "deepseek_v41",
      "dtype": "bfloat16",
      "bos_token_id": 0,
      "eos_token_id": 1,
      "pad_token_id": 2,
      "quantization_config": {"quant_method": "fp8"},
      "text_config": {
        "num_hidden_layers": 3,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "intermediate_size": 8,
        "moe_intermediate_size": 4,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "n_activated_experts": 2,
        "routed_scaling_factor": 2.0,
        "swiglu_limit": 10.0,
        "q_lora_rank": 0,
        "head_dim": 4,
        "qk_rope_head_dim": 2,
        "v_head_dim": 4,
        "o_groups": 1,
        "o_lora_rank": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "max_position_embeddings": 4096,
        "window_size": 3,
        "compress_ratios": [0, 2, 2],
        "kv_source_layer_ids": [1],
        "index_source_layer_ids": [1],
        "candidate_source_layer_id": 1,
        "candidate_block_size": 2,
        "candidate_topk_blocks": 2,
        "index_n_heads": 1,
        "index_head_dim": 2,
        "index_topk": 4,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "engram_layer_ids": [],
        "engram_num_embeddings": []
      }
    }
    """

    static func config() throws -> DeepSeekV41Config {
        try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
    }

    static func plan() throws -> [DeepSeekV41LayerPlan] {
        try config().makeLayerPlan()
    }

    /// Deterministic pseudo-random weights, so a failure reproduces exactly.
    struct Generator {
        private var state: UInt64
        init(seed: UInt64) { self.state = seed == 0 ? 1 : seed }

        mutating func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let bits = UInt32(truncatingIfNeeded: state >> 33)
            return (Float(bits) / Float(UInt32.max)) * 2 - 1
        }

        mutating func vector(_ count: Int, scale: Float = 0.5) -> [Float] {
            (0..<count).map { _ in next() * scale }
        }

        mutating func matrix(rows: Int, columns: Int, scale: Float = 0.5) -> DeepSeekV41Matrix {
            DeepSeekV41Matrix(rows: rows, columns: columns, values: vector(rows * columns, scale: scale))
        }
    }

    /// A complete set of weights for the fixture config.
    static func weights(config: DeepSeekV41Config, plan: [DeepSeekV41LayerPlan]) -> DeepSeekV41DenseWeightSource {
        var generator = Generator(seed: 0xDE5EED)
        let source = DeepSeekV41DenseWeightSource()
        let shapes = DeepSeekV41TensorShapes(config: config)

        source.set(generator.matrix(rows: config.vocabSize, columns: shapes.hidden), role: .tokenEmbedding, layer: nil)
        source.set(generator.matrix(rows: config.vocabSize, columns: shapes.hidden), role: .lmHead, layer: nil)
        source.set(generator.vector(shapes.hidden, scale: 0.1), role: .finalNorm, layer: nil)

        for layer in plan {
            let l = layer.index
            source.set(generator.vector(shapes.hidden, scale: 0.1), role: .attnNorm, layer: l)
            source.set(generator.vector(shapes.hidden, scale: 0.1), role: .ffnNorm, layer: l)
            source.set(generator.matrix(rows: shapes.queryWidth, columns: shapes.hidden), role: .attnWqB, layer: l)
            source.set(generator.matrix(rows: shapes.latentWidth, columns: shapes.hidden), role: .attnWkv, layer: l)
            source.set(generator.vector(shapes.latentWidth, scale: 0.1), role: .attnKvNorm, layer: l)
            source.set(generator.matrix(rows: shapes.oLoraRank, columns: shapes.attentionOutputWidth), role: .attnWoA, layer: l)
            source.set(generator.matrix(rows: shapes.hidden, columns: shapes.oLoraRank), role: .attnWoB, layer: l)
            source.set(generator.vector(config.numAttentionHeads, scale: 0.5), role: .attnSink, layer: l)
            source.set(generator.matrix(rows: config.numRoutedExperts, columns: shapes.hidden), role: .routerWeight, layer: l)
            source.set(generator.vector(config.numRoutedExperts, scale: 0.2), role: .routerBias, layer: l)
            source.set(generator.matrix(rows: shapes.sharedIntermediate, columns: shapes.hidden), role: .sharedExpertW1, layer: l)
            source.set(generator.matrix(rows: shapes.sharedIntermediate, columns: shapes.hidden), role: .sharedExpertW3, layer: l)
            source.set(generator.matrix(rows: shapes.hidden, columns: shapes.sharedIntermediate), role: .sharedExpertW2, layer: l)
            // `hc_*_fn` projects the FLATTENED stack onto `(2 + mult) * mult`
            // logits, and `hc_*_scale` is the three-element per-section scale, so
            // these are the shapes `Block.hc_mixes` reads rather than per-stream
            // vectors.
            let copies = max(1, config.hcMult)
            let mixCount = (2 + copies) * copies
            source.set(generator.matrix(rows: mixCount, columns: copies * shapes.hidden), role: .hcAttnFn, layer: l)
            source.set(generator.matrix(rows: mixCount, columns: copies * shapes.hidden), role: .hcFfnFn, layer: l)
            source.set(generator.vector(3, scale: 0.05), role: .hcAttnScale, layer: l)
            source.set(generator.vector(3, scale: 0.05), role: .hcFfnScale, layer: l)
            source.set(generator.vector(mixCount, scale: 0.1), role: .hcAttnBase, layer: l)
            source.set(generator.vector(mixCount, scale: 0.1), role: .hcFfnBase, layer: l)
            for expert in 0..<config.numRoutedExperts {
                source.set(generator.matrix(rows: shapes.expertIntermediate, columns: shapes.hidden), role: .expertW1, layer: l, expert: expert)
                source.set(generator.matrix(rows: shapes.expertIntermediate, columns: shapes.hidden), role: .expertW3, layer: l, expert: expert)
                source.set(generator.matrix(rows: shapes.hidden, columns: shapes.expertIntermediate), role: .expertW2, layer: l, expert: expert)
            }
            if let table = layer.engramTable {
                let spec = config.engramTables[table]
                let columns = max(1, spec.maxNgramSize - 1) * spec.headCount
                source.set(
                    generator.matrix(
                        rows: shapes.hidden * (max(1, config.hcMult) + 1),
                        columns: columns * spec.headDimension
                    ),
                    role: .engramWkv,
                    layer: l
                )
                source.set(generator.vector(shapes.hidden * max(1, config.hcMult), scale: 0.1), role: .engramQuery, layer: l)
                source.set(generator.vector(shapes.hidden * max(1, config.hcMult), scale: 0.1), role: .engramKey, layer: l)
            }
        }
        return source
    }

    /// The fixture above with one Engram table on layer 1: two n-gram sizes times
    /// two heads, so four hash columns of two values each.
    static func engramConfig() throws -> DeepSeekV41Config {
        let json = Self.json
            .replacingOccurrences(of: "\"engram_layer_ids\": []", with: "\"engram_layer_ids\": [1]")
            .replacingOccurrences(
                of: "\"engram_num_embeddings\": []",
                with: """
                "engram_num_embeddings": [100],
                        "engram_max_ngram_size": 3,
                        "engram_n_heads": 2,
                        "engram_head_dim": 2,
                        "engram_vocab_size": 8,
                        "engram_compressed_vocab_size": 8
                """
            )
        return try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
    }
}

// MARK: - Attention

final class DeepSeekV41AttentionTests: XCTestCase {

    private func shape() -> DeepSeekV41AttentionShape {
        DeepSeekV41AttentionShape(
            numHeads: 2,
            headsPerGroup: 1,
            qkHeadDim: 4,
            vHeadDim: 4,
            latentWidth: 4,
            qLoraRank: 0,
            oLoraRank: 0,
            isCompressed: true,
            compressRatio: 2,
            ropeTheta: 10_000,
            rotaryDim: 0
        )
    }

    func testSinkTakesAllTheMassWhenNoKeyIsRelevant() {
        let shape = shape()
        // A query orthogonal to every key: every dot product is 0, so the sink,
        // set far above, must win outright and the output must collapse to zeros.
        let query: [Float] = [1, 0, 0, 0, 1, 0, 0, 0]
        let keys: [[Float]] = [
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]
        let out = DeepSeekV41AttentionReference.attend(
            query: query,
            keys: keys,
            shape: shape,
            sink: [100, 100]
        )
        XCTAssertEqual(out.count, shape.outputWidth)
        for value in out {
            XCTAssertEqual(value, 0, accuracy: 1e-6)
        }

        let weights = DeepSeekV41AttentionReference.weights(
            query: query, head: 0, keys: keys, shape: shape, sink: 100
        )
        XCTAssertEqual(weights.count, 3)
        XCTAssertEqual(weights.last!, 1, accuracy: 1e-6)
    }

    func testWeightsAreAProbabilityDistribution() {
        let shape = shape()
        let query: [Float] = [0.3, -0.2, 0.5, 0.1, 0.7, 0.2, -0.4, 0.6]
        let keys: [[Float]] = [
            [0.1, 0.2, -0.3, 0.4],
            [-0.5, 0.25, 0.1, 0.2],
            [0.7, -0.1, 0.35, 0.05]
        ]
        let weights = DeepSeekV41AttentionReference.weights(
            query: query, head: 1, keys: keys, shape: shape, sink: -2
        )
        XCTAssertEqual(weights.count, 4)
        XCTAssertEqual(weights.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertTrue(weights.allSatisfy { $0 >= 0 })
    }

    func testSingleKeyPassesItsLatentThrough() {
        let shape = shape()
        let query: [Float] = [1, 0, 0, 0, 1, 0, 0, 0]
        let key: [Float] = [0.5, -0.25, 1.5, -0.75]
        let out = DeepSeekV41AttentionReference.attend(query: query, keys: [key], shape: shape)
        // One key and no sink: the softmax is 1, so each head emits the key itself.
        for head in 0..<shape.numHeads {
            let base = head * shape.vHeadDim
            for dim in 0..<shape.vHeadDim {
                XCTAssertEqual(out[base + dim], key[dim], accuracy: 1e-6)
            }
        }
    }

    func testEmptyKeySetYieldsZeros() {
        let out = DeepSeekV41AttentionReference.attend(
            query: [1, 0, 0, 0, 1, 0, 0, 0],
            keys: [],
            shape: shape()
        )
        XCTAssertEqual(out.count, shape().outputWidth)
        XCTAssertTrue(out.allSatisfy { $0 == 0 })
    }

    func testScoreScaleUsesTheQueryWidth() {
        // qkHeadDim 4, not latentWidth 4 here, but the relation is what matters.
        XCTAssertEqual(shape().scoreScale, 0.5, accuracy: 1e-6)
    }
}

// MARK: - RoPE

final class DeepSeekV41RoPETests: XCTestCase {

    func testPlainFrequenciesWithoutScaling() {
        let frequencies = DeepSeekV41RoPE.inverseFrequencies(
            width: 8, theta: 10_000, factor: 1, originalLength: 0, betaFast: 32, betaSlow: 1
        )
        XCTAssertEqual(frequencies.count, 4)
        XCTAssertEqual(frequencies[0], 1, accuracy: 1e-6)
        // Every later pair rotates slower.
        for index in 1..<frequencies.count {
            XCTAssertLessThan(frequencies[index], frequencies[index - 1])
        }
        XCTAssertEqual(frequencies[0], 1 / pow(10_000, 0), accuracy: 1e-6)
        XCTAssertEqual(frequencies[1], 1 / pow(10_000, 2.0 / 8.0), accuracy: 1e-6)
    }

    /// Query heads rotate their own rope slice, the latent rotates its own; when
    /// those widths differ, one shared table is wrong for at least one rail, and
    /// CPU/GPU parity cannot catch it because both read the same wrong table.
    func testQueryAndLatentTablesUseTheirOwnWidths() throws {
        let shape = DeepSeekV41AttentionShape(
            numHeads: 2, headsPerGroup: 2, qkHeadDim: 8, vHeadDim: 8,
            latentWidth: 4, qLoraRank: 0, oLoraRank: 4,
            isCompressed: false, compressRatio: 0,
            ropeTheta: 10_000, rotaryDim: 0
        )
        let query = DeepSeekV41RoPE.queryFrequencies(for: shape)
        let latent = DeepSeekV41RoPE.latentFrequencies(for: shape)
        XCTAssertEqual(query.count, 4)
        XCTAssertEqual(latent.count, 2)
        XCTAssertNotEqual(query[1], latent[1], accuracy: 1e-9)
    }

    /// The checkpoint's `apply_rotary_emb` takes ADJACENT values as one complex
    /// number (`view_as_complex`), so a 4-wide vector pairs (0,1) and (2,3). Pairing
    /// the halves instead, (0,2) and (1,3), is the other rope convention and produces
    /// these different values for the same weights.
    func testAdjacentValuesPairIntoOneRotation() {
        var vector: [Float] = [1, 2, 3, 4]
        DeepSeekV41RoPE.rotate(&vector, position: 2, frequencies: [0.5, 0.25])
        XCTAssertEqual(vector[0], -1.142640, accuracy: 1e-5)
        XCTAssertEqual(vector[1], 1.922076, accuracy: 1e-5)
        XCTAssertEqual(vector[2], 0.715045, accuracy: 1e-5)
        XCTAssertEqual(vector[3], 4.948607, accuracy: 1e-5)
    }

    /// A decoupled payload is the TRAILING slice: the non-rope prefix must come out
    /// of the rotation untouched, or the two subspaces the checkpoint trained
    /// separately get mixed.
    func testDecoupledPayloadLeavesThePrefixAlone() {
        var vector: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let prefix = Array(vector[0..<4])
        DeepSeekV41RoPE.rotateSlice(
            &vector, start: 4, length: 4, position: 2, frequencies: [0.5, 0.25]
        )
        XCTAssertEqual(Array(vector[0..<4]), prefix)
        // The rotated tail is the same rotation the whole-vector form gives it.
        var tail: [Float] = [5, 6, 7, 8]
        DeepSeekV41RoPE.rotate(&tail, position: 2, frequencies: [0.5, 0.25])
        XCTAssertEqual(Array(vector[4..<8]), tail)
    }

    /// The attention output is de-rotated with the conjugate, so rotating then
    /// inverting has to be the identity.
    func testInverseRotationUndoesTheRotation() {
        let original: [Float] = [0.4, -1.2, 0.7, 2.1, -0.3, 0.9]
        var vector = original
        let frequencies: [Float] = [0.5, 0.25, 0.125]
        DeepSeekV41RoPE.rotate(&vector, position: 5, frequencies: frequencies)
        DeepSeekV41RoPE.rotate(&vector, position: 5, frequencies: frequencies, inverse: true)
        for (before, after) in zip(original, vector) {
            XCTAssertEqual(after, before, accuracy: 1e-5)
        }
    }

    /// A decoupled shape reports the trailing slice on every rail, and rotating per
    /// head over that slice matches rotating each head's tail on its own.
    func testDecoupledShapeRotatesOnlyItsTrailingSlice() {
        let shape = DeepSeekV41AttentionShape(
            numHeads: 2, headsPerGroup: 2, qkHeadDim: 8, vHeadDim: 8,
            latentWidth: 8, qLoraRank: 0, oLoraRank: 4,
            isCompressed: false, compressRatio: 0,
            ropeTheta: 10_000, rotaryDim: 4
        )
        XCTAssertFalse(shape.rotatesWholeLatent)
        XCTAssertEqual(shape.queryRopeStart, 4)
        XCTAssertEqual(shape.queryRopeWidth, 4)
        XCTAssertEqual(shape.latentRopeStart, 4)
        XCTAssertEqual(shape.latentRopeWidth, 4)
        XCTAssertEqual(shape.outputRopeStart, 4)
        XCTAssertTrue(shape.outputCarriesRope)

        var heads: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        DeepSeekV41RoPE.rotateHeads(
            &heads,
            heads: 2,
            headDim: 8,
            sliceStart: shape.queryRopeStart,
            sliceLength: shape.queryRopeWidth,
            position: 3,
            frequencies: [0.5, 0.25]
        )
        // Heads are 8 values each and only values 4..<8 of each head move.
        XCTAssertEqual(Array(heads[0..<4]), [1, 2, 3, 4])
        XCTAssertEqual(Array(heads[8..<12]), [9, 10, 11, 12])
        var firstTail: [Float] = [5, 6, 7, 8]
        DeepSeekV41RoPE.rotate(&firstTail, position: 3, frequencies: [0.5, 0.25])
        XCTAssertEqual(Array(heads[4..<8]), firstTail)
    }

    /// The real rope block (a 64-wide payload at `compress_rope_theta`, factor 16):
    /// the ramp starts at the beta_fast dimension and is complete by the beta_slow
    /// one. Pinned as ratios so the frequencies stay free to be retuned.
    func testScaledRampStartsAtTheBetaFastDimension() {
        let plain = DeepSeekV41RoPE.inverseFrequencies(
            width: 64, theta: 160_000, factor: 1, originalLength: 0, betaFast: 32, betaSlow: 1
        )
        let scaled = DeepSeekV41RoPE.inverseFrequencies(
            width: 64, theta: 160_000, factor: 16, originalLength: 65_536, betaFast: 32, betaSlow: 1
        )
        XCTAssertEqual(scaled.count, 32)
        // Dimension 14 is below the beta_fast threshold (15): untouched.
        XCTAssertEqual(scaled[14] / plain[14], 1, accuracy: 1e-6)
        // Dimension 20 sits halfway up the 15...25 ramp, so half the factor applies.
        XCTAssertEqual(scaled[20] / plain[20], 0.53125, accuracy: 1e-6)
        // Past the beta_slow threshold (25) the factor applies in full.
        XCTAssertEqual(scaled[25] / plain[25], 1 / 16, accuracy: 1e-6)
    }

    func testYarnDividesOnlyTheLongWavelengths() {
        let plain = DeepSeekV41RoPE.inverseFrequencies(
            width: 16, theta: 10_000, factor: 1, originalLength: 0, betaFast: 32, betaSlow: 1
        )
        let scaled = DeepSeekV41RoPE.inverseFrequencies(
            width: 16, theta: 10_000, factor: 8, originalLength: 131_072, betaFast: 32, betaSlow: 1
        )
        XCTAssertEqual(plain.count, scaled.count)
        // The highest-frequency pair (wavelength 2π, far below the low threshold)
        // must be untouched.
        XCTAssertEqual(scaled[0], plain[0], accuracy: 1e-9)
        // At least one dimension must be scaled down, or the ramp did nothing.
        XCTAssertTrue(zip(plain, scaled).contains { $0 > $1 * 1.0001 })
        // No dimension may be scaled UP.
        for (before, after) in zip(plain, scaled) {
            XCTAssertLessThanOrEqual(after, before + 1e-9)
        }
    }

    func testRotationAtPositionZeroIsIdentity() {
        var vector: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let original = vector
        let frequencies = DeepSeekV41RoPE.inverseFrequencies(
            width: 8, theta: 10_000, factor: 1, originalLength: 0, betaFast: 32, betaSlow: 1
        )
        DeepSeekV41RoPE.rotate(&vector, position: 0, frequencies: frequencies)
        for (before, after) in zip(original, vector) {
            XCTAssertEqual(after, before, accuracy: 1e-6)
        }
    }

    func testRotationPreservesNorm() {
        var vector: [Float] = [0.4, -1.2, 0.7, 2.1, -0.3, 0.9, 0.2, -1.5]
        let before = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        let frequencies = DeepSeekV41RoPE.inverseFrequencies(
            width: 8, theta: 10_000, factor: 1, originalLength: 0, betaFast: 32, betaSlow: 1
        )
        DeepSeekV41RoPE.rotate(&vector, position: 37, frequencies: frequencies)
        let after = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(after, before, accuracy: 1e-5)
    }
}

// MARK: - Compressor and indexer

final class DeepSeekV41CompressorTests: XCTestCase {

    func testPoolIsAConvexCombination() {
        let latents: [[Float]] = [
            [1, 0],
            [0, 1]
        ]
        let pooled = DeepSeekV41Compressor.pool(latents: latents, gates: [0, 0])
        XCTAssertEqual(pooled, [0.5, 0.5])
        for value in pooled {
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThanOrEqual(value, 1)
        }
    }

    func testPoolOfIdenticalLatentsIsThatLatent() {
        let latents: [[Float]] = [
            [0.3, -0.7],
            [0.3, -0.7],
            [0.3, -0.7]
        ]
        let pooled = DeepSeekV41Compressor.pool(latents: latents, gates: [5, -1, 2])
        XCTAssertEqual(pooled[0], 0.3, accuracy: 1e-6)
        XCTAssertEqual(pooled[1], -0.7, accuracy: 1e-6)
    }

    func testCompressDropsPartialTrailingBlock() {
        let latents: [[Float]] = (0..<5).map { [Float($0), Float($0)] }
        let gates = [Float](repeating: 0, count: 5)
        XCTAssertEqual(DeepSeekV41Compressor.compress(latents: latents, gates: gates, ratio: 2).count, 2)
        XCTAssertEqual(DeepSeekV41Compressor.compress(latents: latents, gates: gates, ratio: 3).count, 1)
        XCTAssertTrue(DeepSeekV41Compressor.compress(latents: latents, gates: gates, ratio: 0).isEmpty)
    }

    func testEmptyPoolIsZeroed() {
        XCTAssertTrue(DeepSeekV41Compressor.pool(latents: [], gates: []).isEmpty)
        XCTAssertTrue(
            DeepSeekV41Compressor.pooledLatent(inputs: [], kvWeight: nil, gateWeight: nil).isEmpty
        )
    }

    /// `score.softmax(dim=2)` runs over the GROUP axis of a
    /// `[batch, groups, ratio, head_dim]` tensor, so each of the `head_dim`
    /// channels gets its own distribution over the group's positions. A single
    /// weight per row would fail here: the two channels below disagree about which
    /// position should dominate.
    func testGateWeightsArePerDimensionNotPerPosition() {
        // head_dim 2, hidden 2. Position 0 prefers channel 0, position 1 channel 1.
        let kvWeight = DeepSeekV41Matrix(rows: 2, columns: 2, values: [
            1, 0,
            0, 1
        ])
        let gateWeight = DeepSeekV41Matrix(rows: 2, columns: 2, values: [
            100, -100,
            -100, 100
        ])
        // Stacked so the kv projection emits channel 0 = 4, channel 1 = 1 for the
        // first position and channel 0 = 1, channel 1 = 8 for the second.
        let inputs: [[Float]] = [[4, 1], [1, 8]]
        let pooled = DeepSeekV41Compressor.pooledLatent(
            inputs: inputs, kvWeight: kvWeight, gateWeight: gateWeight
        )
        // Channel 0: softmax([~0, ~0]) would be a mean only if the gate were shared.
        // Here the gate says 100 for position 0, so channel 0 takes position 0's 4.
        XCTAssertEqual(pooled[0], 4, accuracy: 1e-4)
        // Channel 1 mirrors it and takes position 1's 8.
        XCTAssertEqual(pooled[1], 8, accuracy: 1e-4)
    }

    /// A ratio of 1 publishes no `wgate`; the group is the single position and the
    /// result is its plain projection.
    func testRatioOneIsAPlainProjection() {
        let kvWeight = DeepSeekV41Matrix(rows: 2, columns: 2, values: [
            1, 1,
            0, 2
        ])
        let pooled = DeepSeekV41Compressor.pooledLatent(
            inputs: [[3, 4]], kvWeight: kvWeight, gateWeight: nil
        )
        XCTAssertEqual(pooled, [7, 8])
    }
}

// MARK: - Block-diagonal output projection

final class DeepSeekV41BlockDiagonalProjectionTests: XCTestCase {

    /// `attn.wo_a` is `groups` dense blocks stacked along the rows. Reading it as
    /// one dense matrix multiplies each group's rows into every other group's
    /// heads, so the grouped result must equal two independent GEMVs.
    func testGroupsReadOnlyTheirOwnColumns() {
        // 2 groups, 2 rows per group, 2 columns per group.
        let weight = DeepSeekV41Matrix(rows: 4, columns: 4, values: [
            1, 2, 999, 999,
            3, 4, 999, 999,
            999, 999, 5, 6,
            999, 999, 7, 8
        ])
        let input: [Float] = [1, 1, 2, 1]
        let grouped = weight.applyBlockDiagonal(input, groups: 2)

        let first = DeepSeekV41Matrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
            .apply([1, 1])
        let second = DeepSeekV41Matrix(rows: 2, columns: 2, values: [5, 6, 7, 8])
            .apply([2, 1])
        XCTAssertEqual(grouped, first + second)

        // And it must differ from the dense read, which is the bug being fixed.
        let dense = weight.apply(input)
        XCTAssertNotEqual(grouped, dense)
    }

    /// One group degenerates to the ordinary dense projection, which is what a
    /// checkpoint with `o_groups = 1` publishes.
    func testSingleGroupIsDense() {
        let weight = DeepSeekV41Matrix(rows: 2, columns: 3, values: [
            1, 0, -1,
            0, 2, 0
        ])
        let input: [Float] = [1, 2, 3]
        XCTAssertEqual(weight.applyBlockDiagonal(input, groups: 1), weight.apply(input))
    }

    /// A geometry that cannot be split is a zero block, matching `apply`'s
    /// documented behaviour for a mismatched matrix, rather than a misread.
    func testUnsplittableGeometryYieldsZeros() {
        let weight = DeepSeekV41Matrix(rows: 3, columns: 4, values: (0..<12).map { Float($0) })
        XCTAssertEqual(weight.applyBlockDiagonal([1, 1, 1, 1], groups: 2), [0, 0, 0])
    }

    /// The published `wo_a` is `[o_groups * o_lora_rank][heads * head_dim / o_groups]`
    /// and `wo_b` reads the concatenated groups, so the rank-width scratch is
    /// `o_groups * o_lora_rank`, not `o_lora_rank`.
    func testShapesSizeTheReducedWidthByGroups() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let shapes = DeepSeekV41TensorShapes(config: config)
        XCTAssertEqual(shapes.outputGroups, 1)
        XCTAssertEqual(shapes.outputReducedWidth, shapes.oLoraRank * shapes.outputGroups)
        XCTAssertEqual(shapes.attentionOutputWidth, config.numAttentionHeads * config.vHeadDim)
    }
}

final class DeepSeekV41IndexerTests: XCTestCase {

    func testBlockScoreIsTheMaxOverRowsAndHeads() {
        let queries: [[Float]] = [
            [1, 0],
            [-1, 0]
        ]
        let keys: [[Float]] = [
            [0.1, 0],   // block 0 row 0: 0.1 / sqrt(2)
            [0.9, 0],   // block 0 row 1: 0.9 / sqrt(2)
            [0, 0],     // block 1 row 0
            [-0.4, 0]   // block 1 row 1: 0.4 / sqrt(2) via the negated query
        ]
        let scores = DeepSeekV41Indexer.blockScores(queries: queries, keys: keys, blockSize: 2, keyWidth: 2)
        XCTAssertEqual(scores.count, 2)
        XCTAssertEqual(scores[0], 0.9 / sqrt(2), accuracy: 1e-5)
        XCTAssertEqual(scores[1], 0.4 / sqrt(2), accuracy: 1e-5)
    }

    func testSelectionIsDescendingByScoreButReturnedAscending() {
        let scores: [Float] = [0.1, 0.9, 0.5, 0.7]
        XCTAssertEqual(DeepSeekV41Indexer.selectedBlocks(scores: scores, topK: 2), [1, 3])
        XCTAssertEqual(DeepSeekV41Indexer.selectedBlocks(scores: scores, topK: 0), [])
        XCTAssertEqual(DeepSeekV41Indexer.selectedBlocks(scores: scores, topK: 9), [0, 1, 2, 3])
    }

    func testRowExpansionCoversWholeBlocksAndTruncatesTheLast() {
        XCTAssertEqual(DeepSeekV41Indexer.rows(for: [0, 1], blockSize: 2, rowCount: 5), [0, 1, 2, 3])
        XCTAssertEqual(DeepSeekV41Indexer.rows(for: [2], blockSize: 2, rowCount: 5), [4])
        XCTAssertEqual(DeepSeekV41Indexer.rows(for: [3], blockSize: 2, rowCount: 5), [])
    }
}

// MARK: - Routing

final class DeepSeekV41RouterTests: XCTestCase {

    func testSigmoidIsStableAtBothExtremes() {
        XCTAssertEqual(DeepSeekV41Router.sigmoid(0), 0.5, accuracy: 1e-6)
        XCTAssertEqual(DeepSeekV41Router.sigmoid(1000), 1, accuracy: 1e-6)
        XCTAssertEqual(DeepSeekV41Router.sigmoid(-1000), 0, accuracy: 1e-6)
        XCTAssertFalse(DeepSeekV41Router.sigmoid(1000).isNaN)
        XCTAssertFalse(DeepSeekV41Router.sigmoid(-1000).isNaN)
    }

    func testTopKIsSelectedByScore() {
        let routing = DeepSeekV41Router.route(
            logits: [3, -3, 1, 0.5],
            bias: nil,
            topK: 2,
            scalingFactor: 1,
            scoringFunction: .sigmoid
        )
        XCTAssertEqual(routing.expertIDs, [0, 2])
        XCTAssertEqual(routing.weights.reduce(0, +), 1, accuracy: 1e-5)
    }

    func testScalingFactorMultipliesTheMixture() {
        let plain = DeepSeekV41Router.route(
            logits: [1, 2, 3, 4], bias: nil, topK: 2, scalingFactor: 1, scoringFunction: .sigmoid
        )
        let scaled = DeepSeekV41Router.route(
            logits: [1, 2, 3, 4], bias: nil, topK: 2, scalingFactor: 2.5, scoringFunction: .sigmoid
        )
        XCTAssertEqual(plain.expertIDs, scaled.expertIDs)
        for (before, after) in zip(plain.weights, scaled.weights) {
            XCTAssertEqual(after, before * 2.5, accuracy: 1e-5)
        }
    }

    func testBiasChangesSelectionButNotTheReportedWeights() {
        // Expert 3 loses on raw logits but wins once its bias is applied.
        let unbiased = DeepSeekV41Router.route(
            logits: [2, 1, 0, -1], bias: nil, topK: 1, scalingFactor: 1, scoringFunction: .sigmoid
        )
        let biased = DeepSeekV41Router.route(
            logits: [2, 1, 0, -1],
            bias: [0, 0, 0, 100],
            topK: 1,
            scalingFactor: 1,
            scoringFunction: .sigmoid
        )
        XCTAssertEqual(unbiased.expertIDs, [0])
        XCTAssertEqual(biased.expertIDs, [3])
        // `norm_topk_prob` only renormalizes when more than one expert is selected,
        // so a lone expert keeps its own score rather than being scaled up to 1.
        XCTAssertEqual(biased.weights, [DeepSeekV41Router.sigmoid(-1)])
    }

    /// The bias ranks experts but must not change the mixture of the ones that are
    /// selected: each weight is the expert's own unbiased scored value, normalized
    /// over the selection.
    func testBiasDoesNotEnterTheMixtureWeights() {
        let routing = DeepSeekV41Router.route(
            logits: [2, 1],
            bias: [100, 0],
            topK: 2,
            scalingFactor: 1,
            scoringFunction: .sigmoid
        )
        XCTAssertEqual(routing.expertIDs, [0, 1])
        let first = DeepSeekV41Router.sigmoid(2)
        let second = DeepSeekV41Router.sigmoid(1)
        XCTAssertEqual(routing.weights[0], first / (first + second), accuracy: 1e-5)
        XCTAssertEqual(routing.weights[1], second / (first + second), accuracy: 1e-5)
        // The reported scores stay the biased selection values, so a test can assert
        // the ranking without re-deriving it.
        XCTAssertEqual(routing.scores[0], DeepSeekV41Router.sigmoid(2) + 100, accuracy: 1e-5)
        XCTAssertEqual(routing.scores[1], DeepSeekV41Router.sigmoid(1), accuracy: 1e-5)
    }

    func testEmptyAndDegenerateInputs() {
        XCTAssertEqual(DeepSeekV41Router.route(logits: [], bias: nil, topK: 2, scalingFactor: 1), .empty)
        XCTAssertEqual(DeepSeekV41Router.route(logits: [1], bias: nil, topK: 0, scalingFactor: 1), .empty)
        // topK larger than the expert count is clamped, not trusted.
        XCTAssertEqual(DeepSeekV41Router.route(logits: [1, 2], bias: nil, topK: 9, scalingFactor: 1).expertIDs.count, 2)
    }

    /// `sqrt(softplus(x))`, pinned so the family default cannot drift. The three
    /// scoring functions are different monotone maps, not rescalings, so a wrong
    /// one reorders the experts.
    func testSqrtSoftplusIsTheCheckpointScoringFunction() {
        XCTAssertEqual(DeepSeekV41Router.score(0, function: .sqrtsoftplus), 0.832_554_6, accuracy: 1e-5)
        XCTAssertEqual(DeepSeekV41Router.score(1, function: .sqrtsoftplus), 1.145_976_3, accuracy: 1e-5)
        XCTAssertEqual(DeepSeekV41Router.score(-1, function: .sqrtsoftplus), 0.559_697_9, accuracy: 1e-5)
        // Large positive logits must not overflow through exp().
        XCTAssertEqual(DeepSeekV41Router.score(1_000, function: .sqrtsoftplus), 31.622_776, accuracy: 1e-3)
        XCTAssertFalse(DeepSeekV41Router.score(1_000, function: .sqrtsoftplus).isNaN)
        // The default must be the checkpoint's function, not the V3 family's.
        XCTAssertEqual(DeepSeekV41ScoringFunction("sqrtsoftplus"), .sqrtsoftplus)
        XCTAssertEqual(DeepSeekV41ScoringFunction(nil), .sqrtsoftplus)
        XCTAssertEqual(DeepSeekV41ScoringFunction("noaux_tc"), .sigmoid)
    }

    /// The bias is added to the SCORED value, so under sqrtsoftplus a bias of 100
    /// has a completely different reach than under sigmoid. Selecting on the wrong
    /// side of the monotone map is the failure this pins.
    func testBiasIsAddedAfterScoringAndOnlyMovesTheSelection() {
        let routed = DeepSeekV41Router.route(
            logits: [2, 1],
            bias: [100, 0],
            topK: 2,
            scalingFactor: 1,
            scoringFunction: .sqrtsoftplus
        )
        XCTAssertEqual(routed.expertIDs, [0, 1])
        let first = DeepSeekV41Router.score(2, function: .sqrtsoftplus)
        let second = DeepSeekV41Router.score(1, function: .sqrtsoftplus)
        XCTAssertEqual(routed.weights[0], first / (first + second), accuracy: 1e-5)
        XCTAssertEqual(routed.weights[1], second / (first + second), accuracy: 1e-5)
        XCTAssertEqual(routed.scores[0], first + 100, accuracy: 1e-5)
        XCTAssertEqual(routed.scores[1], second, accuracy: 1e-5)
    }

    /// `gate_temp` divides the logit BEFORE the scoring function, so it is not a
    /// rescale of the reported weights.
    func testGateTemperatureDividesBeforeScoring() {
        let hot = DeepSeekV41Router.route(
            logits: [4, 0], bias: nil, topK: 2, scalingFactor: 1,
            scoringFunction: .sqrtsoftplus, gateTemp: 2
        )
        XCTAssertEqual(hot.scores[0], DeepSeekV41Router.score(2, function: .sqrtsoftplus), accuracy: 1e-5)
        XCTAssertEqual(hot.scores[1], DeepSeekV41Router.score(0, function: .sqrtsoftplus), accuracy: 1e-5)
    }

    /// `norm_topk_prob` renormalizes only when more than one expert is selected;
    /// a lone expert keeps its own score.
    func testSingleExpertIsNotRenormalized() {
        let routed = DeepSeekV41Router.route(
            logits: [3, 0], bias: nil, topK: 1, scalingFactor: 1, scoringFunction: .sqrtsoftplus
        )
        XCTAssertEqual(routed.expertIDs, [0])
        XCTAssertEqual(routed.weights, [DeepSeekV41Router.score(3, function: .sqrtsoftplus)])
    }
}

// MARK: - Activation

final class DeepSeekV41ActivationTests: XCTestCase {

    func testSwiGLUClipsBothSides() {
        // gate above the limit is clamped; up is clamped into [-limit, limit].
        let clipped = DeepSeekV41Activation.swiglu(gate: 100, up: 100, limit: 10)
        let expected = DeepSeekV41Activation.silu(10) * 10
        XCTAssertEqual(clipped, expected, accuracy: 1e-5)

        let negative = DeepSeekV41Activation.swiglu(gate: 0, up: -100, limit: 10)
        XCTAssertEqual(negative, 0, accuracy: 1e-5)
    }

    func testZeroLimitDisablesTheClip() {
        let unclipped = DeepSeekV41Activation.swiglu(gate: 2, up: 1000, limit: 0)
        let expected = DeepSeekV41Activation.silu(2) * 1000
        XCTAssertEqual(unclipped, expected, accuracy: 1e-3)
    }

    func testSiluIsMonotoneAndSigned() {
        XCTAssertLessThan(DeepSeekV41Activation.silu(-3), 0)
        XCTAssertGreaterThan(DeepSeekV41Activation.silu(3), 0)
        XCTAssertEqual(DeepSeekV41Activation.silu(0), 0, accuracy: 1e-6)
    }
}

// MARK: - Expert arena

final class DeepSeekV41ExpertArenaTests: XCTestCase {

    func testResidentPairsKeepTheirSlot() {
        let arena = DeepSeekV41ExpertArena(slotCount: 4)
        let first = arena.acquire([(layer: 0, expert: 1), (layer: 0, expert: 2)])
        let second = arena.acquire([(layer: 0, expert: 1)])
        XCTAssertEqual(first[0], second[0])
        XCTAssertEqual(arena.hits, 1)
        XCTAssertEqual(arena.residentCount, 2)
    }

    func testOldestPairIsEvictedFirst() {
        let arena = DeepSeekV41ExpertArena(slotCount: 2)
        let initial = arena.acquire([(layer: 0, expert: 0), (layer: 0, expert: 1)])
        XCTAssertEqual(Set(initial), [0, 1])

        // Touch expert 0 so expert 1 becomes the least recently used.
        _ = arena.acquire([(layer: 0, expert: 0)])
        let replacement = arena.acquire([(layer: 1, expert: 7)])

        XCTAssertEqual(arena.evictions, 1)
        XCTAssertEqual(replacement[0], initial[1])
        XCTAssertNotNil(arena.slot(forLayer: 0, expert: 0))
        XCTAssertNil(arena.slot(forLayer: 0, expert: 1))
        XCTAssertNotNil(arena.slot(forLayer: 1, expert: 7))
    }

    func testResetClearsResidencyAndCounters() {
        let arena = DeepSeekV41ExpertArena(slotCount: 2)
        _ = arena.acquire([(layer: 0, expert: 0)])
        arena.reset()
        XCTAssertEqual(arena.residentCount, 0)
        XCTAssertEqual(arena.acquisitions, 0)
        XCTAssertEqual(arena.hitRate, 0)
    }
}

// MARK: - Hyper-connections

final class DeepSeekV41HyperConnectionTests: XCTestCase {

    /// `hc_*_fn` with row `r` one-hot at column `r % columns`, so the logits are the
    /// flattened stack repeated and every coefficient is hand-checkable.
    private func repeatingProjection(rows: Int, columns: Int) -> DeepSeekV41Matrix {
        var values = [Float](repeating: 0, count: rows * columns)
        for row in 0..<rows {
            values[row * columns + row % columns] = 1
        }
        return DeepSeekV41Matrix(rows: rows, columns: columns, values: values)
    }

    private func columnSums(_ matrix: [Float], dimension: Int) -> [Float] {
        (0..<dimension).map { column in
            (0..<dimension).reduce(Float(0)) { $0 + matrix[$1 * dimension + column] }
        }
    }

    private func rowSums(_ matrix: [Float], dimension: Int) -> [Float] {
        (0..<dimension).map { row in
            (0..<dimension).reduce(Float(0)) { $0 + matrix[row * dimension + $1] }
        }
    }

    func testSinkhornProducesADoublyStochasticMatrix() {
        let logits: [Float] = [0.5, -0.2, 1.0, 0.1]
        let matrix = DeepSeekV41HyperConnections.sinkhorn(
            logits: logits,
            dimension: 2,
            iterations: 20,
            epsilon: 1e-6
        )
        XCTAssertEqual(matrix.count, 4)
        for row in rowSums(matrix, dimension: 2) {
            XCTAssertEqual(row, 1, accuracy: 1e-4)
        }
        for column in columnSums(matrix, dimension: 2) {
            XCTAssertEqual(column, 1, accuracy: 1e-4)
        }
        XCTAssertTrue(matrix.allSatisfy { $0 > 0 })
    }

    /// The reference's schedule is softmax, `+eps`, one COLUMN normalization, then
    /// `iterations - 1` rounds of row-then-column. After a single iteration the
    /// columns are exact and the rows are not, which is the asymmetry a symmetric
    /// reimplementation normalizes away.
    func testTheFirstRoundLeavesOnlyTheColumnsExact() {
        let matrix = DeepSeekV41HyperConnections.sinkhorn(
            logits: [1, 0, 0.5, 0.5],
            dimension: 2,
            iterations: 1,
            epsilon: 1e-6
        )
        for column in columnSums(matrix, dimension: 2) {
            XCTAssertEqual(column, 1, accuracy: 1e-5)
        }
        let rows = rowSums(matrix, dimension: 2)
        XCTAssertEqual(rows[0], 0.943_601, accuracy: 1e-4)
        XCTAssertEqual(rows[1], 1.056_399, accuracy: 1e-4)
    }

    func testUniformLogitsGiveTheBarycenter() {
        let matrix = DeepSeekV41HyperConnections.sinkhorn(
            logits: [Float](repeating: 0, count: 9),
            dimension: 3,
            iterations: 10,
            epsilon: 1e-6
        )
        for value in matrix {
            XCTAssertEqual(value, 1.0 / 3.0, accuracy: 1e-4)
        }
    }

    /// A large logit must not exponentiate to infinity: the row and column sums
    /// would be `inf` and the normalization would produce NaNs.
    func testLargeLogitsDoNotOverflowTheExponentiation() {
        let matrix = DeepSeekV41HyperConnections.sinkhorn(
            logits: [1_000, 0, 0, 0],
            dimension: 2,
            iterations: 20,
            epsilon: 1e-6
        )
        XCTAssertTrue(matrix.allSatisfy { $0.isFinite }, "\(matrix)")
        // The last round is a column normalization, so the columns are the ones
        // that are exact; the `+eps` floor keeps the rows a little off the
        // barycenter for a logit this extreme.
        for column in columnSums(matrix, dimension: 2) {
            XCTAssertEqual(column, 1, accuracy: 1e-4)
        }
        XCTAssertTrue(matrix[0] > matrix[1])
    }

    /// One projection of the flattened stack splits into the three coefficient
    /// sets: `pre` through a sigmoid floored at `eps`, `post` through twice a
    /// sigmoid, and `comb` through the Sinkhorn normalization.
    func testOneProjectionSplitsIntoTheThreeCoefficientSets() {
        let streams: [[Float]] = [[1, 0], [0, 1]]
        // `mean(x^2) = 0.5`, so every logit is scaled by `1 / sqrt(0.5 + 1e-6)`.
        let rstd: Float = 1 / sqrt(0.5 + 1e-6)
        let half: Float = rstd * 0.5
        let pre0: Float = 1 / (1 + exp(-half)) + 1e-6
        let post1: Float = 2 / (1 + exp(-half))
        let coefficients = DeepSeekV41HyperConnections.coefficients(
            streams: streams,
            projection: repeatingProjection(rows: 8, columns: 4),
            scale: [0.5, 0.5, 0.5],
            base: [Float](repeating: 0, count: 8),
            normalizationEpsilon: 1e-6,
            epsilon: 1e-6,
            iterations: 20
        )
        XCTAssertEqual(coefficients.logits.count, 8)
        XCTAssertEqual(coefficients.logits[0], rstd, accuracy: 1e-5)
        XCTAssertEqual(coefficients.logits[1], 0, accuracy: 1e-6)
        XCTAssertEqual(coefficients.logits[3], rstd, accuracy: 1e-5)

        XCTAssertEqual(coefficients.pre.count, 2)
        XCTAssertEqual(coefficients.pre[0], pre0, accuracy: 1e-6)
        XCTAssertEqual(coefficients.pre[1], 0.5 + 1e-6, accuracy: 1e-6)

        XCTAssertEqual(coefficients.post.count, 2)
        XCTAssertEqual(coefficients.post[0], 1, accuracy: 1e-6)
        XCTAssertEqual(coefficients.post[1], post1, accuracy: 1e-5)

        XCTAssertEqual(coefficients.comb.count, 4)
        for column in columnSums(coefficients.comb, dimension: 2) {
            XCTAssertEqual(column, 1, accuracy: 1e-4)
        }
    }

    /// A checkpoint with no usable `hc_*` tensors degenerates to a single stream
    /// rather than to a guess: copy 0 is what a sub-layer reads, and every copy
    /// keeps its own residual.
    func testAMissingProjectionCollapsesToCopyZero() {
        let coefficients = DeepSeekV41HyperConnections.coefficients(
            streams: [[1, 2], [3, 4]],
            projection: nil,
            scale: [1, 1, 1],
            base: [Float](repeating: 0, count: 8),
            normalizationEpsilon: 1e-6,
            epsilon: 1e-6,
            iterations: 4
        )
        XCTAssertEqual(coefficients.pre, [1, 0])
        XCTAssertEqual(coefficients.post, [1, 1])
        XCTAssertEqual(coefficients.comb, [1, 0, 0, 1])
    }

    func testPreCollapsesTheCopiesWithTheMix() {
        let streams: [[Float]] = [[1, 0], [0, 2]]
        XCTAssertEqual(
            DeepSeekV41HyperConnections.pre(streams: streams, mix: [0.25, 0.75]),
            [0.25, 1.5]
        )
        XCTAssertEqual(
            DeepSeekV41HyperConnections.pre(streams: streams, mix: [1, 0]),
            [1, 0]
        )
    }

    func testPostExpandsTheOutputAndMixesTheResidualThroughComb() {
        // y[j] = post[j] * x + sum_k comb[j][k] * residual[k]
        let residual: [[Float]] = [[1, 0], [0, 1]]
        let streams = DeepSeekV41HyperConnections.post(
            output: [2, 2],
            residual: residual,
            gains: [0.5, 0],
            comb: [0.25, 0.75, 0, 1]
        )
        // copy 0: 0.5 * (2, 2) + 0.25 * (1, 0) + 0.75 * (0, 1)
        XCTAssertEqual(streams[0][0], 1.25, accuracy: 1e-6)
        XCTAssertEqual(streams[0][1], 1.75, accuracy: 1e-6)
        // copy 1: 0 * output + 0 * (1, 0) + 1 * (0, 1)
        XCTAssertEqual(streams[1][0], 0, accuracy: 1e-6)
        XCTAssertEqual(streams[1][1], 1, accuracy: 1e-6)
    }
}

// MARK: - Engram combination

final class DeepSeekV41EngramCombinationTests: XCTestCase {

    /// A projection whose row `r` is one-hot at column `r % width`, so `kv` is the
    /// flattened rows repeated: one key per copy, then the shared value.
    private func identityProjection(copies: Int, dim: Int) -> DeepSeekV41Matrix {
        let rows = dim * (copies + 1)
        var values = [Float](repeating: 0, count: rows * dim)
        for row in 0..<rows { values[row * dim + row % dim] = 1 }
        return DeepSeekV41Matrix(rows: rows, columns: dim, values: values)
    }

    private func ones(_ count: Int) -> [Float] { [Float](repeating: 1, count: count) }

    /// The literals come from the checkpoint's `Engram.forward`: `kv = wkv(rows)`,
    /// a per-copy key plus one shared value, and
    /// `sigmoid(copysign(sqrt(dot * rstd * dim**-0.5), dot))` as the gate.
    func testGateAndValueFollowTheCheckpointRule() throws {
        let result = try XCTUnwrap(
            DeepSeekV41EngramCombination.contribution(
                residual: [[1, 2, 3, 4], [0, 1, 0, 1]],
                rows: [[1, 0], [0, 1]],
                wkv: identityProjection(copies: 2, dim: 4),
                queryWeight: ones(8),
                keyWeight: ones(8),
                epsilon: 1e-6
            )
        )
        XCTAssertEqual(result.gates.count, 2)
        XCTAssertEqual(result.gates[0], 0.756985, accuracy: 1e-5)
        XCTAssertEqual(result.gates[1], 0.731058, accuracy: 1e-5)
        XCTAssertEqual(result.streams[0][0], 1.756985, accuracy: 1e-5)
        XCTAssertEqual(result.streams[0][1], 2, accuracy: 1e-6)
        XCTAssertEqual(result.streams[0][2], 3, accuracy: 1e-6)
        XCTAssertEqual(result.streams[0][3], 4.756985, accuracy: 1e-5)
        XCTAssertEqual(result.streams[1][0], 0.731058, accuracy: 1e-5)
        XCTAssertEqual(result.streams[1][1], 1, accuracy: 1e-6)
        XCTAssertEqual(result.streams[1][2], 0, accuracy: 1e-6)
        XCTAssertEqual(result.streams[1][3], 1.731058, accuracy: 1e-5)
    }

    /// The gate carries the sign of the dot through `copysign`, so a stream that
    /// points away from its key lands below 0.5 rather than at 1 - nothing.
    func testGateIsSignedByTheDot() throws {
        let result = try XCTUnwrap(
            DeepSeekV41EngramCombination.contribution(
                residual: [[-1, -2, -3, -4]],
                rows: [[1, 0], [0, 1]],
                wkv: identityProjection(copies: 1, dim: 4),
                queryWeight: ones(4),
                keyWeight: ones(4),
                epsilon: 1e-6
            )
        )
        XCTAssertEqual(result.gates[0], 0.243015, accuracy: 1e-5)
        XCTAssertEqual(result.streams[0][0], -0.756985, accuracy: 1e-5)
    }

    /// `value` is shared by every copy; the gate is not. So the two deltas are
    /// the same vector at two different scales.
    func testTheValueIsSharedButTheGateIsNot() throws {
        let residual: [[Float]] = [[1, 2, 3, 4], [0, 1, 0, 1]]
        let result = try XCTUnwrap(
            DeepSeekV41EngramCombination.contribution(
                residual: residual,
                rows: [[1, 0], [0, 1]],
                wkv: identityProjection(copies: 2, dim: 4),
                queryWeight: ones(8),
                keyWeight: ones(8),
                epsilon: 1e-6
            )
        )
        let first = zip(result.streams[0], residual[0]).map { $0 - $1 }
        let second = zip(result.streams[1], residual[1]).map { $0 - $1 }
        XCTAssertNotEqual(result.gates[0], result.gates[1])
        // Both deltas are `gate * value` on the same value.
        for dimension in 0..<4 where first[dimension] != 0 {
            XCTAssertEqual(second[dimension] / first[dimension], result.gates[1] / result.gates[0], accuracy: 1e-4)
        }
    }

    /// The rows concatenate in column order, not per head or reversed: only the
    /// second row's first value feeds this key, so swapping the two rows must move
    /// the gate.
    func testRowsConcatenateInColumnOrder() throws {
        var values = [Float](repeating: 0, count: 12 * 4)
        // key 0 reads the flattened vector's third value, i.e. row 2's first.
        values[0 * 4 + 2] = 1
        values[1 * 4 + 1] = 1
        let projection = DeepSeekV41Matrix(rows: 12, columns: 4, values: values)
        let residual: [[Float]] = [[1, 2, 3, 4], [0, 1, 0, 1]]

        func gate(_ rows: [[Float]]) -> Float {
            DeepSeekV41EngramCombination.contribution(
                residual: residual,
                rows: rows,
                wkv: projection,
                queryWeight: ones(8),
                keyWeight: ones(8),
                epsilon: 1e-6
            )?.gates.first ?? 0
        }

        XCTAssertEqual(gate([[1, 2], [3, 4]]), 0.698880, accuracy: 1e-5)
        XCTAssertEqual(gate([[3, 4], [1, 2]]), 0.709463, accuracy: 1e-5)
    }

    func testMissingProjectionOrRowsYieldsNoContribution() {
        XCTAssertNil(
            DeepSeekV41EngramCombination.contribution(
                residual: [[1, 2, 3, 4]],
                rows: [[1, 0], [0, 1]],
                wkv: nil,
                queryWeight: nil,
                keyWeight: nil,
                epsilon: 1e-6
            )
        )
        XCTAssertNil(
            DeepSeekV41EngramCombination.contribution(
                residual: [[1, 2, 3, 4]],
                rows: [],
                wkv: identityProjection(copies: 1, dim: 4),
                queryWeight: nil,
                keyWeight: nil,
                epsilon: 1e-6
            )
        )
        // Two keys plus a value do not fit a projection this narrow, so the split
        // is refused rather than read past the end.
        XCTAssertNil(
            DeepSeekV41EngramCombination.contribution(
                residual: [[1, 2, 3, 4]],
                rows: [[1, 0], [0, 1]],
                wkv: DeepSeekV41Matrix(rows: 4, columns: 4, values: [Float](repeating: 1, count: 16)),
                queryWeight: nil,
                keyWeight: nil,
                epsilon: 1e-6
            )
        )
    }
}

final class DeepSeekV41EngramForwardTests: XCTestCase {

    /// Rows for every hash column of the fixture's table, so the lookup is
    /// non-empty without a ~100 GB table on disk.
    private func weights(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan],
        withRows: Bool
    ) -> DeepSeekV41DenseWeightSource {
        let source = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        guard withRows, let table = plan.first(where: { $0.engramTable != nil })?.engramTable else {
            return source
        }
        let spec = config.engramTables[table]
        let columns = max(1, spec.maxNgramSize - 1) * spec.headCount
        for column in 0..<columns {
            let row = (0..<spec.headDimension).map { Float($0 + column + 1) * 0.25 }
            source.setEngramRow(row, layer: spec.layer, column: column)
        }
        return source
    }

    /// The contribution lands in the RESIDUAL, before attention reads it. Had it
    /// been added to the post-`ffn_norm` input, the attention input would be
    /// identical with and without rows.
    func testEngramChangesWhatAttentionReads() throws {
        let config = try DeepSeekV41TestFixtures.engramConfig()
        let plan = try config.makeLayerPlan()
        let layer = try XCTUnwrap(plan.first { $0.engramTable != nil }).index

        // The in-memory store doubles as the lookup: rows a test pins stand in for
        // the ones the streaming lookup would pread.
        let empty = weights(config: config, plan: plan, withRows: false)
        let populated = weights(config: config, plan: plan, withRows: true)
        let withoutRows = DeepSeekV41CPUForward(
            config: config, plan: plan, weights: empty, engramLookup: empty
        )
        let withRows = DeepSeekV41CPUForward(
            config: config, plan: plan, weights: populated, engramLookup: populated
        )

        _ = withoutRows.forwardToken(3, position: 0)
        _ = withRows.forwardToken(3, position: 0)

        let plain = try XCTUnwrap(withoutRows.attentionInput(layer: layer, position: 0))
        let engrammed = try XCTUnwrap(withRows.attentionInput(layer: layer, position: 0))
        XCTAssertNotEqual(plain, engrammed)
        XCTAssertNil(withoutRows.engramGates[layer])

        // One gate per hyper-connection copy, recorded once per position.
        let gates = try XCTUnwrap(withRows.engramGates[layer])
        XCTAssertEqual(gates.count, 1)
        XCTAssertEqual(gates[0].count, max(1, config.hcMult))
        XCTAssertTrue(gates[0].allSatisfy { $0 > 0 && $0 < 1 })
        // A layer with no table records nothing.
        XCTAssertNil(withRows.engramGates[(layer + 1) % plan.count])
    }

    /// An image span enters the residual directly, and the positions it occupies
    /// take no Engram contribution (`Engram.forward` zeroes the gate wherever the
    /// token mask is false). Otherwise a picture would read pad rows into the
    /// stream at every Engram layer.
    func testImageSpanPositionsSkipEngram() throws {
        let config = try DeepSeekV41TestFixtures.engramConfig()
        let plan = try config.makeLayerPlan()
        let layer = try XCTUnwrap(plan.first { $0.engramTable != nil }).index

        let source = weights(config: config, plan: plan, withRows: true)
        let spanForward = DeepSeekV41CPUForward(
            config: config, plan: plan, weights: source, engramLookup: source
        )
        let plainForward = DeepSeekV41CPUForward(
            config: config, plan: plan, weights: source, engramLookup: source
        )

        let spanRow = [Float](repeating: 0.75, count: config.hiddenSize)
        spanForward.markNextTokenDead()
        _ = spanForward.forwardEmbedding(spanRow, token: 3, position: 0)
        _ = plainForward.forwardToken(3, position: 0)

        XCTAssertTrue(spanForward.deadTokens == [true])
        XCTAssertTrue(plainForward.deadTokens == [false])
        // The supplied row is what the layer read, not the token embedding.
        XCTAssertNotEqual(
            spanForward.attentionInput(layer: layer, position: 0),
            plainForward.attentionInput(layer: layer, position: 0)
        )
        XCTAssertEqual(spanForward.engramGates[layer], nil, "a dead position has no gate")
        XCTAssertNotNil(plainForward.engramGates[layer])
        // The history still carries the token id, because the hashes need it.
        XCTAssertEqual(spanForward.tokenHistory, [3])
    }
}

// MARK: - Cache layout

final class DeepSeekV41CacheLayoutTests: XCTestCase {

    func testRegionsAreDisjointAndSizedFromThePlan() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let geometry = DeepSeekV41CacheGeometry.make(
            config: config,
            plan: plan,
            contextLength: 64,
            precision: .fp8Block32
        )
        let layout = DeepSeekV41CacheLayout(geometry: geometry, plan: plan)

        XCTAssertEqual(layout.windowBase.count, 3)
        XCTAssertEqual(layout.windowBytes, 3 * geometry.windowBytesPerLayer)
        // Only layer 1 owns a compressed cache, so one slot is allocated.
        XCTAssertEqual(layout.compressedBase.keys.sorted(), [1])
        XCTAssertEqual(layout.compressedBytes, geometry.compressedBytesPerOwner)
        XCTAssertEqual(layout.indexerBase.keys.sorted(), [1])
        XCTAssertEqual(layout.candidateBase.keys.sorted(), [1])

        // Window offsets must not overlap.
        let offsets = plan.compactMap { layout.windowBase[$0.index] }.sorted()
        XCTAssertEqual(offsets, [0, geometry.windowBytesPerLayer, 2 * geometry.windowBytesPerLayer])
    }

    func testRecordBytesMatchTheFP8Footer() {
        XCTAssertEqual(DeepSeekV41CachePrecision.fp8Block32.recordBytes(width: 512), 512 + 16)
        XCTAssertEqual(DeepSeekV41CachePrecision.fp16.recordBytes(width: 512), 1024)
        // A width that is not a multiple of 32 still pays for a whole footer byte.
        XCTAssertEqual(DeepSeekV41CachePrecision.fp8Block32.recordBytes(width: 33), 33 + 2)
    }

    func testWindowRowWrapsAtTheRingLength() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let geometry = DeepSeekV41CacheGeometry.make(config: config, plan: plan, contextLength: 16)
        XCTAssertEqual(geometry.windowRows, 3)
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRow(for: 0, geometry: geometry), 0)
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRow(for: 2, geometry: geometry), 2)
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRow(for: 3, geometry: geometry), 0)
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRow(for: 7, geometry: geometry), 1)
    }

    func testWindowRangeNeverPrecedesTheSequenceStart() {
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRange(position: 0, windowSize: 3), 0..<1)
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRange(position: 1, windowSize: 3), 0..<2)
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRange(position: 5, windowSize: 3), 3..<6)
        XCTAssertEqual(DeepSeekV41CacheAddressing.windowRange(position: -1, windowSize: 3), 0..<0)
    }

    func testCompressedRowCountFollowsTheRatio() {
        XCTAssertEqual(DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: 0, ratio: 2), 0)
        XCTAssertEqual(DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: 1, ratio: 2), 1)
        XCTAssertEqual(DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: 2, ratio: 2), 1)
        XCTAssertEqual(DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: 3, ratio: 2), 2)
        XCTAssertEqual(DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: 99, ratio: 1), 100)
        XCTAssertEqual(DeepSeekV41CacheAddressing.poolRange(row: 2, ratio: 2), 4..<6)
    }

    func testCompressedAttentionRangeHonoursTheTopKBudget() {
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.compressedRowsForAttention(position: 7, ratio: 2, ownerRowCount: 100, topKRows: 2),
            2..<4
        )
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.compressedRowsForAttention(position: 7, ratio: 2, ownerRowCount: 100, topKRows: 0),
            0..<4
        )
        // The owner's own row count caps the range even when the position implies more.
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.compressedRowsForAttention(position: 99, ratio: 2, ownerRowCount: 3, topKRows: 0),
            0..<3
        )
    }

    func testAddressingRefusesRowsOutsideTheAllocation() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let geometry = DeepSeekV41CacheGeometry.make(config: config, plan: plan, contextLength: 8)
        let layout = DeepSeekV41CacheLayout(geometry: geometry, plan: plan)
        XCTAssertNil(DeepSeekV41CacheAddressing.compressedByteOffset(ownerLayer: 0, row: 0, layout: layout))
        XCTAssertNotNil(DeepSeekV41CacheAddressing.compressedByteOffset(ownerLayer: 1, row: 0, layout: layout))
        XCTAssertNil(
            DeepSeekV41CacheAddressing.compressedByteOffset(
                ownerLayer: 1,
                row: geometry.compressedRows,
                layout: layout
            )
        )
    }
}

// MARK: - Block forward

final class DeepSeekV41ForwardTests: XCTestCase {

    func testForwardProducesFiniteLogitsOverTheVocabulary() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let weights = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: weights)

        let logits = forward.prefill(tokens: [3, 5, 7])
        XCTAssertEqual(logits.count, config.vocabSize)
        XCTAssertTrue(logits.allSatisfy { $0.isFinite })
        XCTAssertTrue(logits.contains { $0 != 0 })
        XCTAssertEqual(forward.storedPositions, 3)
    }

    func testCacheRecordsEveryPositionForEveryLayer() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let weights = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: weights)
        _ = forward.prefill(tokens: [1, 2, 3, 4])

        XCTAssertEqual(forward.latents.count, plan.count)
        for layer in forward.latents {
            XCTAssertEqual(layer.count, 4)
            // head_dim = 4 with its rope payload inside it. Pinned rather
            // than compared to `config.cacheRecordWidth`, which is the same
            // helper the cache store uses.
            XCTAssertTrue(layer.allSatisfy { $0.count == 4 })
        }
    }

    func testCompressedRowsAppearExactlyWhenTheirBlockCompletes() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let weights = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: weights)

        // Layer 1 is the compressed owner with ratio 2, so rows appear at positions
        // 1, 3, 5 ... and never at an even position.
        // A ratio-2 owner pools at positions 1 and 3 only. Pinned as literals
        // rather than derived from the helper the implementation calls.
        let expectedRows = [0, 1, 1, 2]
        for (position, expected) in expectedRows.enumerated() {
            _ = forward.forwardToken(7, position: position)
            let rows = forward.compressed[1] ?? []
            XCTAssertEqual(rows.count, expected, "position \(position)")
        }
        XCTAssertEqual(forward.compressed[1]?.count, 2)
        // A window-only layer owns no compressed cache.
        XCTAssertNil(forward.compressed[0])
    }

    func testAttentionReadsExactlyTheWindowAtTheFirstPositions() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let weights = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: weights)
        _ = forward.prefill(tokens: [1, 2, 3, 4, 5])

        // Position 0 has only itself in the window.
        let first = forward.attentionKeys(layer: plan[0], position: 0)
        XCTAssertEqual(first.count, 1)
        // Position 4 has the whole window (size 3) and no compressed rows on a
        // window-only layer.
        let fifth = forward.attentionKeys(layer: plan[0], position: 4)
        XCTAssertEqual(fifth.count, 3)

        // A compressed layer adds the rows its owner pooled.
        let compressedLayer = plan[1]
        let compressedKeys = forward.attentionKeys(layer: compressedLayer, position: 4)
        XCTAssertGreaterThan(compressedKeys.count, 3)
    }

    func testResetClearsTheSequence() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let weights = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: weights)
        _ = forward.prefill(tokens: [1, 2])
        forward.reset()
        XCTAssertEqual(forward.storedPositions, 0)
        XCTAssertTrue(forward.compressed.isEmpty)
        XCTAssertTrue(forward.candidateScores.isEmpty)
    }

    func testHyperConnectionStreamsCollapseToOneVector() throws {
        let config = try DeepSeekV41TestFixtures.config()
        XCTAssertEqual(config.hcMult, 2)
        let plan = try config.makeLayerPlan()
        let weights = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: weights)
        let logits = forward.prefill(tokens: [9])
        // The head sees the collapsed stream, so the logits are the head width, not
        // one per hyper-connection stream.
        XCTAssertEqual(logits.count, config.vocabSize)
    }

    func testRoutingMatchesTheConfiguredExpertCount() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let weights = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: weights)
        let routing = try XCTUnwrap(forward.route([Float](repeating: 0.1, count: config.hiddenSize), layer: 0))
        XCTAssertEqual(routing.expertIDs.count, config.numActivatedExperts)
        XCTAssertTrue(routing.expertIDs.allSatisfy { $0 >= 0 && $0 < config.numRoutedExperts })
    }
}

// MARK: - Shapes

final class DeepSeekV41TensorShapeTests: XCTestCase {

    func testShapesComeFromTheConfig() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let shapes = DeepSeekV41TensorShapes(config: config)
        XCTAssertEqual(shapes.hidden, 8)
        XCTAssertEqual(shapes.queryWidth, 2 * 4)
        XCTAssertEqual(shapes.attentionOutputWidth, 2 * 4)
        XCTAssertEqual(shapes.latentWidth, 4)
        XCTAssertEqual(shapes.expertIntermediate, 4)
        XCTAssertEqual(shapes.sharedIntermediate, 8)
    }

    func testSharedExpertWidthFallsBackToRoutedWidthTimesCount() throws {
        let json = DeepSeekV41TestFixtures.json
            .replacingOccurrences(of: "\"intermediate_size\": 8,", with: "")
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        XCTAssertEqual(config.intermediateSize, 0)
        let shapes = DeepSeekV41TensorShapes(config: config)
        XCTAssertEqual(shapes.sharedIntermediate, 4 * 1)
    }

    func testPipelineListsEveryStageOnce() {
        let stages = DeepSeekV41ForwardStages.layerPipeline
        XCTAssertEqual(Set(stages).count, stages.count)
        // Engram memory is written into the residual at the top of the layer, so it
        // has to run before the stage that mixes the streams for attention.
        XCTAssertEqual(stages.first, .engram)
        XCTAssertEqual(stages.dropFirst().first, .mixAttentionStreams)
        XCTAssertEqual(stages.last, .collapseStreams)
        // The attention stage must come after the latent is written.
        let write = stages.firstIndex(of: .writeLatent)
        let attend = stages.firstIndex(of: .attention)
        XCTAssertNotNil(write)
        XCTAssertNotNil(attend)
        XCTAssertLessThan(write!, attend!)
    }
}

// MARK: - Chunk prefill

/// A chunk prefill measured against the walk it replaces.
///
/// `prefillChunk` runs a prompt through each layer once, all positions at a time: the
/// shape the checkpoint's own prefill uses, and the first step toward a device path
/// that stops paying for six expert weight reads per token. What has to hold is that it
/// agrees with the position-by-position walk, because a change in the order rows are
/// written or groups are pooled moves numbers without failing anywhere. Every
/// comparison here is therefore numeric and per position, and the cache the two passes
/// leave behind is compared too: a following decode step reads it.
final class DeepSeekV41ChunkForwardTests: XCTestCase {

    private func assertRowsMatch(
        _ chunked: [[Float]],
        _ walked: [[Float]],
        _ label: String
    ) {
        XCTAssertEqual(chunked.count, walked.count, "\(label) row count")
        for (position, row) in chunked.enumerated() where walked.indices.contains(position) {
            XCTAssertEqual(row.count, walked[position].count, "\(label) position \(position)")
            for dimension in row.indices where walked[position].indices.contains(dimension) {
                XCTAssertEqual(
                    row[dimension],
                    walked[position][dimension],
                    accuracy: 1e-5,
                    "\(label) position \(position) dim \(dimension)"
                )
            }
        }
    }

    private func assertLogitsMatch(
        _ chunked: [Float],
        _ walked: [Float],
        _ label: String
    ) {
        XCTAssertEqual(chunked.count, walked.count, "\(label) width")
        for dimension in chunked.indices where walked.indices.contains(dimension) {
            XCTAssertEqual(
                chunked[dimension],
                walked[dimension],
                accuracy: 1e-5,
                "\(label) logit \(dimension)"
            )
        }
    }

    /// Engram rows for every hash column, so the in-memory store can stand in for the
    /// streaming lookup the way `DeepSeekV41EngramForwardTests` does.
    private func engramSource(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan]
    ) -> DeepSeekV41DenseWeightSource {
        let source = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        guard let table = plan.first(where: { $0.engramTable != nil })?.engramTable else {
            return source
        }
        let spec = config.engramTables[table]
        let columns = max(1, spec.maxNgramSize - 1) * spec.headCount
        for column in 0..<columns {
            let row = (0..<spec.headDimension).map { Float($0 + column + 1) * 0.25 }
            source.setEngramRow(row, layer: spec.layer, column: column)
        }
        return source
    }

    func testAChunkMatchesTheWalkAtEveryPosition() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let tokens = [3, 5, 7, 9, 11]

        let walk = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        )
        var walked: [[Float]] = []
        for (position, token) in tokens.enumerated() {
            walked.append(walk.forwardToken(token, position: position))
        }

        let chunked = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        )
        let chunk = chunked.prefillChunk(tokens, capturingLogitsAt: Set(tokens.indices))

        for position in tokens.indices {
            let captured = try XCTUnwrap(chunk.captured[position], "position \(position)")
            assertLogitsMatch(captured, walked[position], "position \(position)")
        }
        // The last position's logits are the ones the caller samples from.
        XCTAssertEqual(chunk.last.count, config.vocabSize)
        XCTAssertEqual(chunk.last, chunk.captured[tokens.count - 1])

        // The cache has to be the same one, not merely a similar one: a decode step or a
        // resumed turn reads these rows.
        XCTAssertEqual(chunked.storedPositions, walk.storedPositions)
        XCTAssertEqual(chunked.tokenHistory, walk.tokenHistory)
        XCTAssertEqual(chunked.deadTokens, walk.deadTokens)
        XCTAssertEqual(chunked.candidateScores.count, walk.candidateScores.count)
        XCTAssertEqual(chunked.engramGates.keys.sorted(), walk.engramGates.keys.sorted())
        XCTAssertEqual(chunked.compressed.keys.sorted(), walk.compressed.keys.sorted())
        for (owner, rows) in walk.compressed {
            let chunkedRows = try XCTUnwrap(chunked.compressed[owner], "owner \(owner)")
            XCTAssertEqual(chunkedRows.count, rows.count, "owner \(owner) row count")
            for (row, values) in chunkedRows.enumerated() where rows.indices.contains(row) {
                for dimension in values.indices where rows[row].indices.contains(dimension) {
                    XCTAssertEqual(
                        values[dimension],
                        rows[row][dimension],
                        accuracy: 1e-5,
                        "owner \(owner) row \(row) dim \(dimension)"
                    )
                }
            }
        }
        for layer in plan {
            assertRowsMatch(
                chunked.latents[layer.index],
                walk.latents[layer.index],
                "layer \(layer.index) latent"
            )
            assertRowsMatch(
                chunked.attentionInputs[layer.index],
                walk.attentionInputs[layer.index],
                "layer \(layer.index) attention input"
            )
        }
    }

    func testDecodeContinuesFromAChunk() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let tokens = [3, 5, 7]

        let walk = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        )
        _ = walk.prefill(tokens: tokens)

        let chunked = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        )
        _ = chunked.prefillChunk(tokens)
        XCTAssertEqual(chunked.storedPositions, tokens.count)

        // The position after the chunk reads the ring and the compressed rows the chunk
        // wrote, so this is where a chunk that pooled or rotated differently shows up.
        let next = 13
        assertLogitsMatch(
            chunked.forwardToken(next, position: tokens.count),
            walk.forwardToken(next, position: tokens.count),
            "decode after the chunk"
        )
    }

    func testAChunkPoolsOnlyTheGroupsItCompletes() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let owner = try XCTUnwrap(plan.first { $0.ownsCompressedKV })
        let ratio = max(1, owner.compressRatio)
        // Five positions at ratio two: two rows, and the fifth position waits for the one
        // that completes its group.
        let tokens = [3, 5, 7, 9, 11]
        let forward = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        )
        _ = forward.prefillChunk(tokens)
        XCTAssertEqual(forward.compressed[owner.index]?.count, tokens.count / ratio)
    }

    func testTheHeadOnlyRunsForThePositionsAsked() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let source = DeepSeekV41TestFixtures.weights(config: config, plan: plan)

        let bare = DeepSeekV41CPUForward(config: config, plan: plan, weights: source)
            .prefillChunk([3, 5, 7, 9])
        XCTAssertTrue(bare.captured.isEmpty)
        XCTAssertEqual(bare.last.count, config.vocabSize)

        // 9 is past the end of a four-position chunk, so it is dropped rather than
        // reported as an empty entry.
        let kept = DeepSeekV41CPUForward(config: config, plan: plan, weights: source)
            .prefillChunk([3, 5, 7, 9], capturingLogitsAt: [1, 9])
        XCTAssertEqual(Set(kept.captured.keys), [1])
        XCTAssertEqual(kept.last.count, config.vocabSize)
    }

    func testAChunkRefusesASequenceTheCacheAlreadyHolds() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let forward = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        )
        _ = forward.prefillChunk([3, 5])
        // A resumed prefix is a walk: the reference has no chunked form of it, and a chunk
        // that guessed would fill the ring and finish groups its own way.
        let second = forward.prefillChunk([7])
        XCTAssertTrue(second.last.isEmpty)
        XCTAssertEqual(forward.storedPositions, 2)
        XCTAssertEqual(forward.tokenHistory.count, 2)
    }

    /// The Engram layers hash the token stream, so a chunk has to hand each position the
    /// history that ENDS at it. Handing over the whole chunk hashes every position as if
    /// it were the last one, which reads different table rows and still produces text.
    func testAnEngramChunkHashesTheSameNgramsAsTheWalk() throws {
        let config = try DeepSeekV41TestFixtures.engramConfig()
        let plan = try config.makeLayerPlan()
        let tokens = [3, 5, 7, 9]
        let layer = try XCTUnwrap(plan.first { $0.engramTable != nil }).index

        let walkSource = engramSource(config: config, plan: plan)
        let walk = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: walkSource,
            engramLookup: walkSource
        )
        var walked: [[Float]] = []
        for (position, token) in tokens.enumerated() {
            walked.append(walk.forwardToken(token, position: position))
        }

        let chunkSource = engramSource(config: config, plan: plan)
        let chunked = DeepSeekV41CPUForward(
            config: config,
            plan: plan,
            weights: chunkSource,
            engramLookup: chunkSource
        )
        let chunk = chunked.prefillChunk(tokens, capturingLogitsAt: Set(tokens.indices))

        for position in tokens.indices {
            let captured = try XCTUnwrap(chunk.captured[position], "position \(position)")
            assertLogitsMatch(captured, walked[position], "position \(position)")
        }
        // The gates are recorded per lookup, so a chunk that hashed the wrong history is
        // visible here even where the logits happen to be close.
        let chunkGates = try XCTUnwrap(chunked.engramGates[layer])
        let walkGates = try XCTUnwrap(walk.engramGates[layer])
        XCTAssertEqual(chunkGates.count, walkGates.count)
        for (position, gates) in chunkGates.enumerated() where walkGates.indices.contains(position) {
            for dimension in gates.indices where walkGates[position].indices.contains(dimension) {
                XCTAssertEqual(
                    gates[dimension],
                    walkGates[position][dimension],
                    accuracy: 1e-5,
                    "layer \(layer) position \(position) gate \(dimension)"
                )
            }
        }
    }
}
