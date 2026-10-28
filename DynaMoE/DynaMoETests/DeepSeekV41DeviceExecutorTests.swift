//
//  DeepSeekV41DeviceExecutorTests.swift
//  DynaMoE
//
//  Coverage for the device executor's planning half. The encoder itself needs the
//  510 GB checkpoint, so what is asserted here is everything that decides *what*
//  it dispatches: the tensor format to kernel mapping, the row plan the CPU
//  reference and the device share, and the wiring problems a checkpoint would
//  report before a single token runs.
//
//  The row-plan parity case is the important one: it is what keeps the reference
//  and the executor from attending over different rows, which is the kind of
//  divergence that produces plausible text instead of a crash.
//

import XCTest
@testable import DynaMoE

// MARK: - Format mapping

final class DeepSeekV41LinearFormatTests: XCTestCase {

    private func tensor(_ dtype: String) -> DeepSeekV41DeviceTensor {
        DeepSeekV41DeviceTensor(shardIndex: 0, byteOffset: 0, rows: 8, columns: 8, dtype: dtype)
    }

    func testFP8WithAnE8M0ScaleIsThe32x32Grid() {
        let format = DeepSeekV41DevicePlan.linearFormat(
            tensor: tensor("F8_E4M3"),
            scale: tensor("F8_E8M0")
        )
        XCTAssertEqual(format, .fp8Block32)
        XCTAssertEqual(DeepSeekV41DevicePlan.gemvKernel(format), DeepSeekV41Kernels.fp8Gemv)
        XCTAssertEqual(
            DeepSeekV41DevicePlan.swigluKernel(format, swigluLimit: 10),
            DeepSeekV41Kernels.fp8SwiGLU
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.downKernel(format),
            DeepSeekV41Kernels.fp8DownAccumulate
        )
    }

    func testFP8WithABF16ScaleIsTheV3GridAndHasNoKernel() {
        // The 128x128 family would "work" if dispatched to the 32x32 kernel and
        // read every scale at the wrong stride, so it must not resolve.
        let format = DeepSeekV41DevicePlan.linearFormat(
            tensor: tensor("F8_E4M3"),
            scale: tensor("BF16")
        )
        XCTAssertEqual(format, .unsupported)
        XCTAssertNil(DeepSeekV41DevicePlan.gemvKernel(format))
    }

    func testFP8WithoutAScaleIsUnsupported() {
        XCTAssertEqual(
            DeepSeekV41DevicePlan.linearFormat(tensor: tensor("F8_E4M3"), scale: nil),
            .unsupported
        )
    }

    func testMXFP4ResolvesToTheE2M1Kernels() {
        let format = DeepSeekV41DevicePlan.linearFormat(
            tensor: tensor("F4_E2M1"),
            scale: tensor("F8_E8M0")
        )
        XCTAssertEqual(format, .mxfp4)
        XCTAssertEqual(DeepSeekV41DevicePlan.gemvKernel(format), DeepSeekV41Kernels.mxfp4Gemv)
        XCTAssertEqual(
            DeepSeekV41DevicePlan.swigluKernel(format, swigluLimit: 10),
            DeepSeekV41Kernels.mxfp4SwiGLU
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.downKernel(format),
            DeepSeekV41Kernels.mxfp4DownAccumulate
        )
        XCTAssertEqual(format.blockWidth, 32)
    }

    func testBF16SwiGLUIsRefusedWhenTheCheckpointClips() {
        // The bf16 kernel applies an unclipped SwiGLU. `swiglu_limit` is part of the
        // trained function, so a clipped checkpoint on an unclipped kernel is a
        // wrong result rather than a coarser approximation.
        let format = DeepSeekV41DevicePlan.linearFormat(tensor: tensor("BF16"), scale: nil)
        XCTAssertEqual(format, .bf16)
        XCTAssertNil(DeepSeekV41DevicePlan.swigluKernel(format, swigluLimit: 10))
        XCTAssertEqual(
            DeepSeekV41DevicePlan.swigluKernel(format, swigluLimit: 0),
            DeepSeekV41Kernels.bf16SwiGLU
        )
        XCTAssertEqual(DeepSeekV41DevicePlan.gemvKernel(format), DeepSeekV41Kernels.bf16Gemv)
        XCTAssertEqual(DeepSeekV41DevicePlan.downKernel(format), DeepSeekV41Kernels.bf16DownAccumulate)
    }

    func testFP32DenseWeightsAreUnsupported() {
        XCTAssertEqual(
            DeepSeekV41DevicePlan.linearFormat(tensor: tensor("F32"), scale: nil),
            .unsupported
        )
    }

    func testEmbeddingLookupFollowsTheTableFormat() {
        XCTAssertEqual(
            DeepSeekV41DevicePlan.embeddingKernel(tensor: tensor("BF16"), scale: nil),
            DeepSeekV41Kernels.bf16Embedding
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.embeddingKernel(tensor: tensor("F8_E4M3"), scale: tensor("F8_E8M0")),
            DeepSeekV41Kernels.mxfp8Embedding
        )
        // A packed table cannot be read one byte per value.
        XCTAssertNil(
            DeepSeekV41DevicePlan.embeddingKernel(tensor: tensor("F4_E2M1"), scale: tensor("F8_E8M0"))
        )
    }
}

// MARK: - Row plan

final class DeepSeekV41AttentionRowPlanTests: XCTestCase {

    func testWindowOnlyLayerReadsItsRingAndNothingElse() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let geometry = DeepSeekV41CacheGeometry.make(
            config: config,
            plan: plan,
            contextLength: config.maxPositionEmbeddings,
            precision: .fp16
        )
        let rows = DeepSeekV41CacheAddressing.attentionRowPlan(
            layer: plan[0],
            position: 4,
            config: config,
            plan: plan,
            geometry: geometry
        )
        XCTAssertEqual(plan[0].flavor, .windowOnly)
        XCTAssertEqual(rows.windowPositions, [2, 3, 4])
        XCTAssertEqual(
            rows.windowRows,
            rows.windowPositions.map {
                DeepSeekV41CacheAddressing.windowRow(for: $0, geometry: geometry)
            }
        )
        XCTAssertTrue(rows.compressedRows.isEmpty)
        XCTAssertEqual(rows.totalRows, 3)
    }

    func testCompressedLayerReadsTheMostRecentRowsAscending() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let geometry = DeepSeekV41CacheGeometry.make(
            config: config,
            plan: plan,
            contextLength: config.maxPositionEmbeddings,
            precision: .fp16
        )
        // Ratio 2, position 9: five compressed rows exist (positions 1, 3, 5, 7, 9).
        let rows = DeepSeekV41CacheAddressing.attentionRowPlan(
            layer: plan[1],
            position: 9,
            config: config,
            plan: plan,
            geometry: geometry
        )
        XCTAssertEqual(plan[1].flavor, .compressed)
        XCTAssertEqual(rows.compressedRows, [1, 2, 3, 4])
        XCTAssertEqual(rows.compressedRows, rows.compressedRows.sorted())
    }

    func testRowProvisionsClampTheSelection() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let geometry = DeepSeekV41CacheGeometry.make(
            config: config,
            plan: plan,
            contextLength: 8,
            precision: .fp16
        )
        let rows = DeepSeekV41CacheAddressing.attentionRowPlan(
            layer: plan[1],
            position: 9,
            config: config,
            plan: plan,
            geometry: geometry,
            ownerRowLimit: 2
        )
        // A row past the allocation has no buffer to read, so the plan cannot name
        // it even though the sequence produced it.
        XCTAssertEqual(rows.compressedRows.last, 1)
        XCTAssertTrue(rows.compressedRows.allSatisfy { $0 < 2 })
    }

    func testReferenceAndPlanReadTheSameRowCount() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let source = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: source)
        _ = forward.prefill(tokens: [1, 2, 3, 4, 5, 6, 7])

        for position in 0..<7 {
            for layer in plan {
                let planRows = DeepSeekV41CacheAddressing.attentionRowPlan(
                    layer: layer,
                    position: position,
                    config: config,
                    plan: plan,
                    geometry: forward.geometry
                )
                // The reference drops a row it has not written yet; before position 0
                // there are none, and past it every planned row exists.
                let keys = forward.attentionKeys(layer: layer, position: position)
                XCTAssertEqual(
                    keys.count,
                    planRows.totalRows,
                    "layer \(layer.index) position \(position)"
                )
            }
        }
    }
}

// MARK: - Wiring problems

final class DeepSeekV41DevicePlanTests: XCTestCase {

    func testGatherBudgetCoversTheWindowAndTheCompressedSelection() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let budget = DeepSeekV41DevicePlan.gatherRowBudget(config: config)
        let compressed = max(config.indexTopK, config.candidateTopKBlocks * config.candidateBlockSize)
        XCTAssertEqual(budget, config.windowSize + compressed)
    }

    func testAWiderCompressRatioThanTheRingIsReported() throws {
        let json = DeepSeekV41TestFixtures.json
            .replacingOccurrences(of: "\"compress_ratios\": [0, 2, 2]", with: "\"compress_ratios\": [0, 8, 2]")
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        let plan = try config.makeLayerPlan()
        let problems = DeepSeekV41DevicePlan.problems(config: config, plan: plan)
        XCTAssertTrue(
            problems.contains { $0.contains("compress_ratio 8") },
            "expected the pool-block problem, got \(problems)"
        )
    }

    func testAnEngramLayerIsReportedRatherThanGuessed() throws {
        let json = DeepSeekV41TestFixtures.json
            .replacingOccurrences(of: "\"engram_layer_ids\": []", with: "\"engram_layer_ids\": [1]")
            .replacingOccurrences(of: "\"engram_num_embeddings\": []", with:
                "\"engram_num_embeddings\": [4096], \"engram_max_ngram_size\": 3, "
                    + "\"engram_vocab_size\": 512, \"engram_n_heads\": 2, \"engram_head_dim\": 4, "
                    + "\"engram_compressed_vocab_size\": 128")
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        let plan = try config.makeLayerPlan()
        let problems = DeepSeekV41DevicePlan.problems(config: config, plan: plan)
        XCTAssertTrue(
            problems.contains { $0.contains("Engram") && $0.contains("layers.1") },
            "expected the Engram problem, got \(problems)"
        )
    }

    func testACompressedLayerWithoutASelectionBudgetIsReported() throws {
        let json = DeepSeekV41TestFixtures.json
            .replacingOccurrences(of: "\"index_topk\": 4,", with: "\"index_topk\": 0,")
            .replacingOccurrences(of: "\"candidate_topk_blocks\": 2,", with: "\"candidate_topk_blocks\": 0,")
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        let plan = try config.makeLayerPlan()
        let problems = DeepSeekV41DevicePlan.problems(config: config, plan: plan)
        XCTAssertTrue(
            problems.contains { $0.contains("index_topk") },
            "expected the selection-budget problem, got \(problems)"
        )
    }

    func testANonBlockWidthProjectionIsReported() throws {
        // 8-wide hidden state: the block-scaled kernels need a multiple of 32.
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let problems = DeepSeekV41DevicePlan.problems(config: config, plan: plan)
        XCTAssertTrue(
            problems.contains { $0.contains("multiple of 32") },
            "expected the width problem, got \(problems)"
        )
    }

    func testTheFixtureWiringIsOtherwiseClean() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let problems = DeepSeekV41DevicePlan.problems(config: config, plan: plan)
        // Every layer of the fixture is compressed or window-only with a cache
        // owner, so the only complaint is the 8-wide width.
        XCTAssertEqual(problems.count, 1, "unexpected problems: \(problems)")
    }

    func testEveryPipelineStageIsEncoded() {
        let pipeline = DeepSeekV41ForwardStages.layerPipeline
        XCTAssertEqual(DeepSeekV41DevicePlan.encodedStages, Set(pipeline))
        XCTAssertEqual(DeepSeekV41DevicePlan.frontHalf.count + DeepSeekV41DevicePlan.backHalf.count, pipeline.count)
        XCTAssertEqual(DeepSeekV41DevicePlan.frontHalf.last, .router)
        XCTAssertEqual(DeepSeekV41DevicePlan.backHalf, [.sharedExpert, .routedExperts, .writeFFNStreams, .collapseStreams])
        XCTAssertEqual(Array(DeepSeekV41DevicePlan.frontHalf) + DeepSeekV41DevicePlan.backHalf, pipeline)
    }
}

// MARK: - Query RoPE

final class DeepSeekV41QueryRopeTests: XCTestCase {

    func testOneHeadMatchesTheWholeVectorRotation() {
        let frequencies: [Float] = [0.5, 0.25]
        var whole: [Float] = [0.4, -0.2, 0.7, 0.1]
        var perHead = whole
        DeepSeekV41RoPE.rotate(&whole, position: 3, frequencies: frequencies)
        DeepSeekV41RoPE.rotateHeads(&perHead, heads: 1, headDim: 4, position: 3, frequencies: frequencies)
        XCTAssertEqual(whole, perHead)
    }

    func testEachHeadRotatesInsideItself() {
        let frequencies: [Float] = [0.5, 0.25]
        var heads: [Float] = [
            0.4, -0.2, 0.7, 0.1,
            0.9, 0.3, -0.5, 0.2
        ]
        let secondBefore = Array(heads[4...])
        DeepSeekV41RoPE.rotateHeads(&heads, heads: 2, headDim: 4, position: 2, frequencies: frequencies)

        // Head 1 rotates with the same angles, so its own low/high halves pair up.
        var secondExpected = secondBefore
        DeepSeekV41RoPE.rotate(&secondExpected, position: 2, frequencies: frequencies)
        XCTAssertEqual(Array(heads[4...]), secondExpected)

        // And rotating the concatenation as one vector would have mixed head 0's low
        // half into head 1, which is the bug this replaced.
        var whole = [Float](repeating: 0, count: 8)
        whole.replaceSubrange(0..<4, with: [0.4, -0.2, 0.7, 0.1])
        whole.replaceSubrange(4..<8, with: secondBefore)
        DeepSeekV41RoPE.rotate(&whole, position: 2, frequencies: frequencies)
        XCTAssertNotEqual(Array(whole[4...]), secondExpected)
    }

    func testNoHeadIsTouchedWhenTheTableIsEmpty() {
        var vector: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let original = vector
        DeepSeekV41RoPE.rotateHeads(&vector, heads: 2, headDim: 4, position: 5, frequencies: [])
        XCTAssertEqual(vector, original)
    }
}
