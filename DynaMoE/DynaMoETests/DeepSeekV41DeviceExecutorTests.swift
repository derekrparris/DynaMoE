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
                // Counts agreeing is not enough: the rows must be the ones the
                // plan names. Compare the actual latents, window first then
                // compressed rows oldest-to-newest.
                for (offset, windowPosition) in planRows.windowPositions.enumerated() {
                    XCTAssertEqual(
                        keys[offset],
                        forward.latents[layer.index][windowPosition],
                        "layer \(layer.index) position \(position) window row \(windowPosition)"
                    )
                }
                let ownerRows = forward.compressed[layer.compressedKVSource] ?? []
                for (offset, row) in planRows.compressedRows.enumerated() {
                    XCTAssertEqual(
                        keys[planRows.windowPositions.count + offset],
                        ownerRows[row],
                        "layer \(layer.index) position \(position) compressed row \(row)"
                    )
                }
            }
        }
    }
}

// MARK: - Wiring problems

final class DeepSeekV41DevicePlanTests: XCTestCase {

    func testGatherBudgetCoversTheWindowAndTheCompressedSelection() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let budget = DeepSeekV41DevicePlan.gatherRowBudget(config: config)
        // A layer with an indexer keeps `min(index_topk, produced)` compressed
        // rows, so the buffer is the window plus that selection, not plus the
        // candidate-block mask.
        XCTAssertEqual(budget, config.windowSize + config.indexTopK)
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

    /// An Engram layer is served rather than refused: the rows it reads are a host
    /// batch, the `wkv` projection and the gate are kernels, and nothing about the
    /// wiring is a guess. What the conformance gate still records is that the
    /// combination rule has no numeric reference vector behind it yet.
    func testAnEngramLayerIsServedRatherThanRefused() throws {
        let json = DeepSeekV41TestFixtures.json
            .replacingOccurrences(of: "\"engram_layer_ids\": []", with: "\"engram_layer_ids\": [1]")
            .replacingOccurrences(of: "\"engram_num_embeddings\": []", with:
                "\"engram_num_embeddings\": [4096], \"engram_max_ngram_size\": 3, "
                    + "\"engram_vocab_size\": 512, \"engram_n_heads\": 2, \"engram_head_dim\": 4, "
                    + "\"engram_compressed_vocab_size\": 128")
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        let plan = try config.makeLayerPlan()
        let problems = DeepSeekV41DevicePlan.problems(config: config, plan: plan)
        XCTAssertFalse(
            problems.contains { $0.contains("Engram") },
            "an Engram layer must not be refused: \(problems)"
        )
        XCTAssertEqual(
            plan[1].engramTable, 0,
            "the plan must bind the table the config names"
        )
    }

    /// The rig can only be enabled from the device plan, so it has to be honest
    /// about the wiring: a table index the config does not describe is reported
    /// instead of being read from the wrong rows.
    func testAnEngramTableIndexOutsideTheConfigIsReported() throws {
        let json = DeepSeekV41TestFixtures.json
            .replacingOccurrences(of: "\"engram_layer_ids\": []", with: "\"engram_layer_ids\": [1]")
            .replacingOccurrences(of: "\"engram_num_embeddings\": []", with:
                "\"engram_num_embeddings\": [4096], \"engram_max_ngram_size\": 3, "
                    + "\"engram_vocab_size\": 512, \"engram_n_heads\": 2, \"engram_head_dim\": 4, "
                    + "\"engram_compressed_vocab_size\": 128")
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        var plan = try config.makeLayerPlan()
        // A plan that names a table the config never published is the shape of a
        // stale plan, and it must not fall back to table 0.
        let original = plan[1]
        plan[1] = DeepSeekV41LayerPlan(
            index: original.index,
            flavor: original.flavor,
            compressRatio: original.compressRatio,
            compressedKVSource: original.compressedKVSource,
            compressedIndexSource: original.compressedIndexSource,
            isCandidateSource: original.isCandidateSource,
            engramTable: 7,
            ropeTable: original.ropeTable
        )
        let problems = DeepSeekV41DevicePlan.problems(config: config, plan: plan)
        XCTAssertTrue(
            problems.contains { $0.contains("Engram table index 7") },
            "expected the table-index problem, got \(problems)"
        )
    }

    /// The staging slots are addressed by `layer.engramTable`, so their order has to
    /// stay `config.engramTables` order: a reordered array hands a layer its
    /// neighbour's rows and every numeric check still passes.
    func testEngramSlotsAreLaidOutInTableOrder() throws {
        let json = DeepSeekV41TestFixtures.json
            .replacingOccurrences(of: "\"engram_layer_ids\": []", with: "\"engram_layer_ids\": [1, 2]")
            .replacingOccurrences(of: "\"engram_num_embeddings\": []", with:
                "\"engram_num_embeddings\": [4096, 4096], \"engram_max_ngram_size\": 3, "
                    + "\"engram_vocab_size\": 512, \"engram_n_heads\": 2, \"engram_head_dim\": 4, "
                    + "\"engram_compressed_vocab_size\": 128")
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        let slots = DeepSeekV41EngramSlot.slots(config: config)
        XCTAssertEqual(slots.count, 2)
        XCTAssertEqual(slots.map(\.layer), [1, 2])
        XCTAssertEqual(slots.map(\.table), [0, 1])
        // Two hash columns per head (a 3-gram table has bigrams and trigrams) times
        // two heads, and 4-wide heads with no scale block past the first 32.
        XCTAssertEqual(slots[0].columns, 4)
        XCTAssertEqual(slots[0].flatWidth, 16)
        XCTAssertEqual(slots[0].rowBytes, 4 + 1)
        // The blocks are disjoint and in order, in both the raw and the decoded
        // buffer: sharing them would let one table's read land under the other's
        // dequantize.
        XCTAssertEqual(slots[0].rawByteOffset, 0)
        XCTAssertEqual(slots[1].rawByteOffset, slots[0].rawByteCount)
        XCTAssertEqual(slots[0].valueOffset, 0)
        XCTAssertEqual(slots[1].valueOffset, slots[0].valueCount)
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
        // Routed experts clear `moeOut`, so they must run BEFORE the shared
        // expert accumulates into it; the reverse order discarded the shared
        // term on every layer.
        XCTAssertEqual(DeepSeekV41DevicePlan.backHalf, [.routedExperts, .sharedExpert, .writeFFNStreams, .collapseStreams])
        XCTAssertEqual(Array(DeepSeekV41DevicePlan.frontHalf) + DeepSeekV41DevicePlan.backHalf, pipeline)
    }

    /// Which BF16 GEMV form a projection gets, at the shapes the checkpoint publishes.
    ///
    /// The two kernels differ only in where the parallelism sits, so the choice is a
    /// cost decision and not a correctness one — but it is a decision, and it was
    /// made against a measured crossover rather than a guess. These are the five
    /// dense projections of a block of eight positions, plus the decode step, which
    /// takes the threadgroup form at every size because a grid one row tall cannot
    /// amortize a row's bytes across positions.
    func testTheWideGemvGridIsChosenByHowMuchWorkTheProjectionHas() {
        // attn.wkv: 448 rows. attn.wo_a's band: 128 rows over 8,192 columns, the
        // projection that motivated the second kernel.
        XCTAssertTrue(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 448, rows: 8))
        XCTAssertTrue(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 128, rows: 8))
        // 6,144 threads still measured 2.3x faster; 12,288 measured 10% slower.
        XCTAssertTrue(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 768, rows: 8))
        XCTAssertFalse(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 1_536, rows: 8))
        // attn.wq_b: 73,728 rows, where one thread per row measured 2.4x faster.
        XCTAssertFalse(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 73_728, rows: 8))
        XCTAssertFalse(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 7_168, rows: 8))
        // A decode step is one position, so every row count takes the wide grid.
        XCTAssertTrue(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 7_168, rows: 1))
        XCTAssertTrue(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 73_728, rows: 1))
        XCTAssertTrue(DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim: 448, rows: 0))
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

// MARK: - Norm dispatch

import Metal

final class DeepSeekV41NormDispatchTests: XCTestCase {

    /// The kernel reads `threadgroup_position_in_grid` as the ROW index, so a row
    /// must be exactly one threadgroup however wide it is.
    func testARowIsOneThreadgroupAtEveryWidth() {
        for width in [64, 256, 257, 512, 7_168] {
            let grid = DeepSeekV41DevicePlan.normDispatch(width: width, lanes: 256)
            XCTAssertEqual(grid.threadgroups.width, 1, "width \(width)")
            XCTAssertEqual(grid.threadgroups.height, 1)
            XCTAssertEqual(grid.threadsPerThreadgroup.width, 256)
        }
    }

    /// Golden vector: the real kernel, dispatched the way the executor dispatches,
    /// against the CPU definition. This is the case the planning tests cannot see,
    /// because they never run a kernel.
    func testKernelMatchesTheCPUDefinitionForAWideRow() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: DeepSeekV41Kernels.rmsNorm),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the RMSNorm pipeline")
            return
        }

        let width = 1_024
        let epsilon: Float = 1e-6
        let values = (0..<width).map { Float(($0 % 17) - 8) / 8 }
        let gains = (0..<width).map { Float(1) + Float($0 % 5) * 0.1 }

        guard let input = device.makeBuffer(
                  bytes: values, length: width * MemoryLayout<Float>.stride, options: .storageModeShared),
              let gain = device.makeBuffer(
                  bytes: gains, length: width * MemoryLayout<Float>.stride, options: .storageModeShared),
              let output = device.makeBuffer(
                  length: 2 * width * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the norm buffers")
            return
        }
        // Poison the second row so a stray threadgroup writing past row 0 shows up.
        let out = output.contents().assumingMemoryBound(to: Float.self)
        for index in width..<(2 * width) { out[index] = 42 }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gain, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var dim = UInt32(width)
        var eps = epsilon
        encoder.setBytes(&dim, length: 4, index: 3)
        encoder.setBytes(&eps, length: 4, index: 4)
        let lanes = 256
        encoder.setThreadgroupMemoryLength((lanes / 32) * MemoryLayout<Float>.stride, index: 0)
        let grid = DeepSeekV41DevicePlan.normDispatch(width: width, lanes: lanes)
        encoder.dispatchThreadgroups(grid.threadgroups, threadsPerThreadgroup: grid.threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let sumOfSquares = values.reduce(Float(0)) { $0 + $1 * $1 }
        let inverse = 1 / (sumOfSquares / Float(width) + epsilon).squareRoot()
        for index in 0..<width {
            XCTAssertEqual(out[index], values[index] * inverse * gains[index], accuracy: 1e-4, "index \(index)")
        }
        for index in width..<(2 * width) {
            XCTAssertEqual(out[index], 42, "row 1 was overwritten at \(index)")
        }
    }
}

// MARK: - Gather kernel

final class DeepSeekV41GatherKernelTests: XCTestCase {

    /// The compressed half of a gather lives after the window's rows in the same
    /// index list. The kernel indexes `rowIndices[row]` from zero, so binding the
    /// list at offset 0 re-reads window rows; this test pins the offset contract.
    func testGatherReadsTheRowListFromTheBoundOffset() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_gather_latents_f16"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the gather pipeline")
            return
        }

        let width = 4
        let cacheRows = 4
        var cacheBits: [UInt16] = []
        for row in 0..<cacheRows {
            for dim in 0..<width {
                cacheBits.append(Float16(Float(row * 10 + dim)).bitPattern)
            }
        }
        // Two junk entries first, then the rows this gather should read.
        let indices: [UInt32] = [77, 77, 3, 1]
        let indexOffset = 2
        guard let cache = device.makeBuffer(
                  bytes: cacheBits,
                  length: cacheBits.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let indexBuffer = device.makeBuffer(
                  bytes: indices,
                  length: indices.count * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let dense = device.makeBuffer(
                  length: 2 * width * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the gather buffers")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(cache, offset: 0, index: 0)
        encoder.setBuffer(
            indexBuffer,
            offset: indexOffset * MemoryLayout<UInt32>.stride,
            index: 1
        )
        encoder.setBuffer(dense, offset: 0, index: 2)
        var cacheOffset: UInt64 = 0
        var dim = UInt32(width)
        var rows: UInt32 = 2
        var destination: UInt64 = 0
        encoder.setBytes(&cacheOffset, length: 8, index: 3)
        encoder.setBytes(&dim, length: 4, index: 4)
        encoder.setBytes(&rows, length: 4, index: 5)
        encoder.setBytes(&destination, length: 8, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: width, height: 2, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let out = dense.contents().assumingMemoryBound(to: Float.self)
        // Bound at offset 0 this would read rows 77 and 77 (out of the buffer).
        XCTAssertEqual(Array(UnsafeBufferPointer(start: out, count: 4)), [30, 31, 32, 33])
        XCTAssertEqual(Array(UnsafeBufferPointer(start: out + 4, count: 4)), [10, 11, 12, 13])
    }
}

// MARK: - Attention kernel

final class DeepSeekV41AttentionKernelTests: XCTestCase {

    private func makePipeline() throws -> (MTLDevice, MTLComputePipelineState, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_attention_csa2_f32"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the attention pipeline")
            throw XCTSkip("no attention pipeline")
        }
        return (device, pipeline, queue)
    }

    private func values(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { index in
            Float((index &* 37 &+ seed &* 13) % 23 - 11) / 8
        }
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self),
            count: count
        ))
    }

    /// The kernel details a head's rows and dimensions to its threadgroup, so the
    /// head's maximum and denominator are a reduction ACROSS SIMD groups. A row set
    /// wider than one group is the case that goes wrong if only the first group's
    /// partial is read, and it is the case the published checkpoint always has: 128
    /// window rows plus `index_topk` compressed ones against a 128-thread threadgroup.
    ///
    /// The value head is also wider than the record here, so the tail that no row
    /// fills has to be zeroed rather than left over from the last position.
    func testTheAttentionKernelMatchesTheHostRuleAcrossItsWholeThreadgroup() throws {
        let (device, pipeline, queue) = try makePipeline()

        let heads = 3
        let qkHeadDim = 8
        let vHeadDim = 12
        let width = 10
        let rows = 200
        let shape = DeepSeekV41AttentionShape(
            numHeads: heads,
            headsPerGroup: 1,
            qkHeadDim: qkHeadDim,
            vHeadDim: vHeadDim,
            latentWidth: width,
            qLoraRank: 0,
            oLoraRank: 0,
            isCompressed: false,
            compressRatio: 1,
            ropeTheta: 10000,
            rotaryDim: 0
        )
        let query = values(heads * qkHeadDim, seed: 1)
        let keys = values(rows * width, seed: 2)
        let sink = values(heads, seed: 3)

        guard let queryBuffer = device.makeBuffer(
                  bytes: query,
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: keys,
                  length: keys.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let sinkBuffer = device.makeBuffer(
                  bytes: sink,
                  length: sink.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let outputBuffer = device.makeBuffer(
                  length: heads * vHeadDim * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the attention buffers")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyBuffer, offset: 0, index: 1)
        encoder.setBuffer(sinkBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        var rowCount = UInt32(rows)
        var headCount = UInt32(heads)
        var headWidth = UInt32(qkHeadDim)
        var headValue = UInt32(vHeadDim)
        var recordWidth = UInt32(width)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&headCount, length: 4, index: 5)
        encoder.setBytes(&headWidth, length: 4, index: 6)
        encoder.setBytes(&headValue, length: 4, index: 7)
        encoder.setBytes(&recordWidth, length: 4, index: 8)
        encoder.setThreadgroupMemoryLength(
            (rows + 32) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        var reference = DeepSeekV41AttentionReference.attend(
            query: query,
            keys: (0..<rows).map { Array(keys[($0 * width)..<(($0 + 1) * width)]) },
            shape: shape,
            sink: sink
        )
        // No row fills the two dimensions past the record width, so they are zero by
        // construction rather than whatever the softmax produced.
        for head in 0..<heads {
            for dim in width..<vHeadDim {
                reference[head * vHeadDim + dim] = 0
            }
        }

        let output = floats(outputBuffer, count: heads * vHeadDim)
        let scale = reference.map(abs).max() ?? 1
        for index in 0..<output.count {
            XCTAssertEqual(
                output[index],
                reference[index],
                accuracy: max(scale, 1) * 1e-5,
                "the attention kernel disagrees with the host rule at \(index)"
            )
        }
    }

    /// A head whose row count is far below its threadgroup width: every idle thread
    /// must still take part in the reduction, and a softmax over one row is that row.
    func testASingleRowAttentionIsThatRow() throws {
        let (device, pipeline, queue) = try makePipeline()

        let heads = 2
        let qkHeadDim = 4
        let vHeadDim = 4
        let width = 4
        let rows = 1
        let query: [Float] = [1, 0, 0, 0, 0, 1, 0, 0]
        let keys: [Float] = [0.5, -0.25, 1.5, -2]

        guard let queryBuffer = device.makeBuffer(
                  bytes: query,
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: keys,
                  length: keys.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let outputBuffer = device.makeBuffer(
                  length: heads * vHeadDim * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the attention buffers")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        var rowCount = UInt32(rows)
        var headCount = UInt32(heads)
        var headWidth = UInt32(qkHeadDim)
        var headValue = UInt32(vHeadDim)
        var recordWidth = UInt32(width)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&headCount, length: 4, index: 5)
        encoder.setBytes(&headWidth, length: 4, index: 6)
        encoder.setBytes(&headValue, length: 4, index: 7)
        encoder.setBytes(&recordWidth, length: 4, index: 8)
        encoder.setThreadgroupMemoryLength(
            (rows + 32) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        // With one row and no sink the softmax is 1, for both heads: the head's own
        // query only scales the score, and a single weight normalizes away.
        let output = floats(outputBuffer, count: heads * vHeadDim)
        for dim in 0..<vHeadDim {
            XCTAssertEqual(output[dim], keys[dim], accuracy: 1e-6)
            XCTAssertEqual(output[vHeadDim + dim], keys[dim], accuracy: 1e-6)
        }
    }
}

// MARK: - Router kernel

final class DeepSeekV41RouterKernelTests: XCTestCase {

    /// The kernel is the GPU half of `DeepSeekV41Router.route`, so it is checked
    /// against it on the same inputs: the bias ranks, and the weights are the
    /// unbiased scores. A CPU/GPU agreement on the wrong rule would pass a
    /// self-consistency test, which is why the expected values come from the
    /// documented rule and not from a second kernel run.
    ///
    /// Driven with `scoring_func = sigmoid` so the expectation is a closed form;
    /// the sqrtsoftplus path is pinned against literals in the CPU suite.
    func testKernelMatchesTheReferenceRouting() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_router_topk"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the router pipeline")
            return
        }

        let logits: [Float] = [1.5, -0.5, 4.0, 0.25, -3.0, 2.0, 0.0, 1.0]
        let bias: [Float] = [0.0, 0.0, 0.0, 0.0, 5.0, -1.0, 0.0, 0.0]
        let topK = 3
        let scaling: Float = 2.5

        guard let logitsBuffer = device.makeBuffer(
                  bytes: logits, length: logits.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(
                  bytes: bias, length: bias.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let indexBuffer = device.makeBuffer(length: 64 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(length: 64 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the router buffers")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
        encoder.setBuffer(biasBuffer, offset: 0, index: 1)
        encoder.setBuffer(indexBuffer, offset: 0, index: 2)
        encoder.setBuffer(weightBuffer, offset: 0, index: 3)
        var expertCount = UInt32(logits.count)
        var top = UInt32(topK)
        var scale = scaling
        var normalise: UInt32 = 1
        var scoring: UInt32 = 0  // sigmoid
        var temperature: Float = 1
        // The readback buffer is a single slot here, so the stride only has to be
        // large enough for the top-k this case asks for.
        var indexStride = UInt32(4)
        var slotCount: UInt32 = 1
        encoder.setBytes(&expertCount, length: 4, index: 4)
        encoder.setBytes(&top, length: 4, index: 5)
        encoder.setBytes(&scale, length: 4, index: 6)
        encoder.setBytes(&normalise, length: 4, index: 7)
        encoder.setBytes(&scoring, length: 4, index: 8)
        encoder.setBytes(&temperature, length: 4, index: 9)
        encoder.setBytes(&indexStride, length: 4, index: 10)
        encoder.setBytes(&slotCount, length: 4, index: 11)
        encoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let indices = indexBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let weights = weightBuffer.contents().assumingMemoryBound(to: Float.self)
        let reference = DeepSeekV41Router.route(
            logits: logits, bias: bias, topK: topK, scalingFactor: scaling,
            scoringFunction: .sigmoid
        )
        // The bias is added to the SCORED value, so expert 4's `sigmoid(-3) + 5`
        // outranks expert 2's `sigmoid(4)`; adding it to the raw logit instead, as
        // the V3 rule did, would rank 2 first.
        XCTAssertEqual(reference.expertIDs, [4, 2, 0], "selection should follow sigmoid(logit) + bias")
        for rank in 0..<topK {
            XCTAssertEqual(Int(indices[rank]), reference.expertIDs[rank], "rank \(rank)")
            XCTAssertEqual(weights[rank], reference.weights[rank], accuracy: 1e-5, "rank \(rank)")
        }
        // Expert 4 is selected only because of its bias, but its weight is its own
        // unbiased sigmoid's share, not its biased score's.
        let withoutBias = DeepSeekV41Router.route(
            logits: logits, bias: nil, topK: topK, scalingFactor: scaling,
            scoringFunction: .sigmoid
        )
        XCTAssertFalse(withoutBias.expertIDs.contains(4))
        let unbiased = logits.map { DeepSeekV41Router.sigmoid($0) }
        let total = reference.expertIDs.map { unbiased[$0] }.reduce(0, +)
        for (rank, expert) in reference.expertIDs.enumerated() {
            XCTAssertEqual(weights[rank], unbiased[expert] / total * scaling, accuracy: 1e-4)
        }
        // Expert 4's selection value is sigmoid(-3) + 5 = 5.047, while its weight is
        // its unbiased sigmoid(-3) = 0.047. Weighting by the selection value instead
        // would multiply its share by over a hundred.
        let selectionValue = DeepSeekV41Router.sigmoid(-3) + 5
        XCTAssertNotEqual(weights[0], selectionValue / total * scaling, accuracy: 0.1)
    }
}

// MARK: - Rope kernel

final class DeepSeekV41RopeKernelTests: XCTestCase {

    /// The kernel is the GPU half of `DeepSeekV41RoPE.rotateSlice`, and the two
    /// properties that matter are ones a CPU/GPU parity check cannot see: the rotated
    /// slice is the TRAILING one (the non-rope prefix must not move) and values pair
    /// ADJACENTLY, the checkpoint's `view_as_complex` layout. The expected values are
    /// pinned from that rule, not from a second kernel run.
    func testKernelRotatesOnlyTheBoundSlice() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_rope_latent"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the rope pipeline")
            return
        }

        // A 8-wide record whose rope payload is the last 4 values.
        let rowWidth = 8
        let sliceStart = 4
        let pairCount = 2
        let values: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let frequencies: [Float] = [0.5, 0.25]
        guard let vector = device.makeBuffer(
                  bytes: values, length: values.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let freqs = device.makeBuffer(
                  bytes: frequencies,
                  length: frequencies.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the rope buffers")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(freqs, offset: 0, index: 1)
        var width = UInt32(rowWidth)
        var headDim: UInt32 = 0
        var pairs = UInt32(pairCount)
        var base: UInt32 = 2
        var batch: UInt32 = 1
        var offset = UInt32(sliceStart)
        var inverse: UInt32 = 0
        encoder.setBytes(&width, length: 4, index: 2)
        encoder.setBytes(&headDim, length: 4, index: 3)
        encoder.setBytes(&pairs, length: 4, index: 4)
        encoder.setBytes(&base, length: 4, index: 5)
        encoder.setBytes(&batch, length: 4, index: 6)
        encoder.setBytes(&offset, length: 4, index: 7)
        encoder.setBytes(&inverse, length: 4, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: pairCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: pairCount, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let out = vector.contents().assumingMemoryBound(to: Float.self)
        let rotated = Array(UnsafeBufferPointer(start: out, count: values.count))
        // The prefix is byte-identical: rotating from offset 0 would move it.
        XCTAssertEqual(Array(rotated[0..<4]), [1, 2, 3, 4])
        // Adjacent pairs of the tail (5,6) and (7,8), rotated at position 2.
        XCTAssertEqual(rotated[4], -2.347315, accuracy: 1e-5)
        XCTAssertEqual(rotated[5], 7.449169, accuracy: 1e-5)
        XCTAssertEqual(rotated[6], 2.307674, accuracy: 1e-5)
        XCTAssertEqual(rotated[7], 10.376639, accuracy: 1e-5)

        // And the same dispatch in the conjugate direction restores the original,
        // which is what the attention output's de-rotation relies on.
        guard let restoreBuffer = queue.makeCommandBuffer(),
              let restoreEncoder = restoreBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the second command buffer")
            return
        }
        inverse = 1
        restoreEncoder.setComputePipelineState(pipeline)
        restoreEncoder.setBuffer(vector, offset: 0, index: 0)
        restoreEncoder.setBuffer(freqs, offset: 0, index: 1)
        restoreEncoder.setBytes(&width, length: 4, index: 2)
        restoreEncoder.setBytes(&headDim, length: 4, index: 3)
        restoreEncoder.setBytes(&pairs, length: 4, index: 4)
        restoreEncoder.setBytes(&base, length: 4, index: 5)
        restoreEncoder.setBytes(&batch, length: 4, index: 6)
        restoreEncoder.setBytes(&offset, length: 4, index: 7)
        restoreEncoder.setBytes(&inverse, length: 4, index: 8)
        restoreEncoder.dispatchThreads(
            MTLSize(width: pairCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: pairCount, height: 1, depth: 1)
        )
        restoreEncoder.endEncoding()
        restoreBuffer.commit()
        restoreBuffer.waitUntilCompleted()
        XCTAssertNil(restoreBuffer.error)
        let restored = Array(UnsafeBufferPointer(start: out, count: values.count))
        for (before, after) in zip(values, restored) {
            XCTAssertEqual(after, before, accuracy: 1e-5)
        }
    }
}

// MARK: - Compressor pool kernel

final class DeepSeekV41PoolKernelTests: XCTestCase {

    /// `score.softmax(dim=2)` runs over the group axis of a
    /// `[batch, groups, ratio, head_dim]` tensor, so each CHANNEL of the pooled row
    /// gets its own distribution over the group's positions. A kernel that shared
    /// one weight per row would produce the row mean here; the two channels below
    /// disagree about which position should dominate, so the mean is distinguishable.
    func testKernelSoftmaxesTheGatePerDimension() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_pool_latents_f32"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the pool pipeline")
            return
        }

        let width = 2
        let blockSize = 2
        // Channel 0 strongly prefers position 0, channel 1 strongly prefers
        // position 1, so the per-dimension answer is [4, 8] and the mean is [2.5, 4.5].
        let latents: [Float] = [4, 1, 1, 8]
        let gates: [Float] = [100, -100, -100, 100]

        guard let latentBuffer = device.makeBuffer(
                  bytes: latents, length: latents.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let gateBuffer = device.makeBuffer(
                  bytes: gates, length: gates.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let outBuffer = device.makeBuffer(
                  length: width * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the pool buffers")
            return
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(latentBuffer, offset: 0, index: 0)
        encoder.setBuffer(gateBuffer, offset: 0, index: 1)
        encoder.setBuffer(outBuffer, offset: 0, index: 2)
        var poolWidth = UInt32(width)
        var block = UInt32(blockSize)
        encoder.setBytes(&poolWidth, length: 4, index: 3)
        encoder.setBytes(&block, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: max(1, width), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let out = outBuffer.contents().assumingMemoryBound(to: Float.self)
        XCTAssertEqual(out[0], 4, accuracy: 1e-4)
        XCTAssertEqual(out[1], 8, accuracy: 1e-4)

        // And the CPU rule must agree, so the two backends cannot drift.
        let reference = DeepSeekV41Compressor.pooledLatent(
            inputs: [[4, 1], [1, 8]],
            kvWeight: DeepSeekV41Matrix(rows: 2, columns: 2, values: [1, 0, 0, 1]),
            gateWeight: DeepSeekV41Matrix(rows: 2, columns: 2, values: [100, -100, -100, 100])
        )
        XCTAssertEqual(out[0], reference[0], accuracy: 1e-4)
        XCTAssertEqual(out[1], reference[1], accuracy: 1e-4)
    }
}

// MARK: - Engram gate kernel

final class DeepSeekV41EngramGateKernelTests: XCTestCase {

    /// The kernel is the GPU half of `DeepSeekV41EngramCombination.contribution`, so
    /// its expected numbers are the literals that suite pins from the checkpoint's
    /// `Engram.forward` — not a second run of the same arithmetic.
    ///
    /// `kv` here is what the identity projection `DeepSeekV41EngramCombinationTests`
    /// builds produces from the rows `[[1, 0], [0, 1]]`: the flattened rows repeated
    /// once per key and once for the value.
    func testKernelMatchesTheCheckpointGate() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_engram_gate_f32"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the Engram gate pipeline")
            return
        }

        let dim = 4
        let mult = 2
        let streams: [Float] = [1, 2, 3, 4, 0, 1, 0, 1]
        // key 0, key 1, then the value every copy shares.
        let kv: [Float] = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        let weights = [Float](repeating: 1, count: mult * dim)

        guard let streamBuffer = device.makeBuffer(
                  bytes: streams, length: streams.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let kvBuffer = device.makeBuffer(
                  bytes: kv, length: kv.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let weightBuffer = device.makeBuffer(
                  bytes: weights, length: weights.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let gateBuffer = device.makeBuffer(
                  length: mult * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the Engram gate buffers")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(streamBuffer, offset: 0, index: 0)
        encoder.setBuffer(kvBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(weightBuffer, offset: 0, index: 3)
        encoder.setBuffer(gateBuffer, offset: 0, index: 4)
        var width = UInt32(dim)
        var copies = UInt32(mult)
        var epsilon: Float = 1e-6
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&copies, length: 4, index: 6)
        encoder.setBytes(&epsilon, length: 4, index: 7)
        let lanes = 64
        encoder.setThreadgroupMemoryLength((lanes / 32) * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: mult, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let gate = gateBuffer.contents().assumingMemoryBound(to: Float.self)
        XCTAssertEqual(gate[0], 0.756985, accuracy: 1e-5)
        XCTAssertEqual(gate[1], 0.731058, accuracy: 1e-5)

        let out = streamBuffer.contents().assumingMemoryBound(to: Float.self)
        XCTAssertEqual(out[0], 1.756985, accuracy: 1e-4)
        XCTAssertEqual(out[1], 2, accuracy: 1e-5)
        XCTAssertEqual(out[2], 3, accuracy: 1e-5)
        XCTAssertEqual(out[3], 4.756985, accuracy: 1e-4)
        XCTAssertEqual(out[4], 0.731058, accuracy: 1e-5)
        XCTAssertEqual(out[5], 1, accuracy: 1e-5)
        XCTAssertEqual(out[6], 0, accuracy: 1e-5)
        XCTAssertEqual(out[7], 1.731058, accuracy: 1e-4)
    }

    /// `q_weight * k_weight` is the per-copy reweighting, and the checkpoint
    /// publishes both. The kernel is checked against the reference rule on a
    /// NON-ones pair, because ones cannot tell a reweighted dot from a plain one.
    func testKernelAppliesThePerCopyWeightPair() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_engram_gate_f32"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the Engram gate pipeline")
            return
        }

        let dim = 4
        let mult = 2
        let residual: [[Float]] = [[1, 2, 3, 4], [0, 1, 0, 1]]
        let streams = residual.flatMap { $0 }
        // `[[1, 0], [0, 1]]` through an identity projection, as in the CPU suite.
        let kv: [Float] = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        let query = [Float](repeating: 1, count: mult * dim)
        let key: [Float] = [2, 2, 0.5, 1, 1, 4, 1, 1]

        guard let streamBuffer = device.makeBuffer(
                  bytes: streams, length: streams.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let kvBuffer = device.makeBuffer(
                  bytes: kv, length: kv.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let queryBuffer = device.makeBuffer(
                  bytes: query, length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: key, length: key.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let gateBuffer = device.makeBuffer(
                  length: mult * MemoryLayout<Float>.stride, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the Engram gate buffers")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(streamBuffer, offset: 0, index: 0)
        encoder.setBuffer(kvBuffer, offset: 0, index: 1)
        encoder.setBuffer(queryBuffer, offset: 0, index: 2)
        encoder.setBuffer(keyBuffer, offset: 0, index: 3)
        encoder.setBuffer(gateBuffer, offset: 0, index: 4)
        var width = UInt32(dim)
        var copies = UInt32(mult)
        var epsilon: Float = 1e-6
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&copies, length: 4, index: 6)
        encoder.setBytes(&epsilon, length: 4, index: 7)
        let lanes = 64
        encoder.setThreadgroupMemoryLength((lanes / 32) * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: mult, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        // One-hot rows, so `wkv` is reproducible here without pulling the CPU
        // matrix helper in: every row reads one flattened value.
        var values = [Float](repeating: 0, count: dim * (mult + 1) * dim)
        for row in 0..<(dim * (mult + 1)) { values[row * dim + row % dim] = 1 }
        let reference = try XCTUnwrap(
            DeepSeekV41EngramCombination.contribution(
                residual: residual,
                rows: [[1, 0], [0, 1]],
                wkv: DeepSeekV41Matrix(rows: dim * (mult + 1), columns: dim, values: values),
                queryWeight: query,
                keyWeight: key,
                epsilon: epsilon
            )
        )

        let gate = gateBuffer.contents().assumingMemoryBound(to: Float.self)
        for copy in 0..<mult {
            XCTAssertEqual(gate[copy], reference.gates[copy], accuracy: 1e-4, "gate \(copy)")
        }
        let out = streamBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<(mult * dim) {
            XCTAssertEqual(out[index], reference.streams[index / dim][index % dim], accuracy: 1e-4)
        }
    }
}

/// The GPU half of the mHC split. The host rule is
/// `DeepSeekV41HyperConnections.sinkhorn`, pinned against the checkpoint's
/// `hc_split_sinkhorn` by `DeepSeekV41HyperConnectionGoldenTests`; this suite runs
/// the kernel on the same inputs and over the same schedule.
final class DeepSeekV41SinkhornKernelTests: XCTestCase {

    private func run(
        logits: [Float],
        dimension: Int,
        iterations: Int,
        epsilon: Float
    ) throws -> [Float] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        let library = try XCTUnwrap(device.makeDefaultLibrary())
        let function = try XCTUnwrap(library.makeFunction(name: "dsv41_hc_sinkhorn"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let logitsBuffer = try XCTUnwrap(
            device.makeBuffer(
                bytes: logits, length: logits.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        )
        let outputBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: dimension * dimension * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        )
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var n = UInt32(dimension)
        var rounds = UInt32(iterations)
        var eps = epsilon
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.setBytes(&rounds, length: 4, index: 3)
        encoder.setBytes(&eps, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        let pointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return (0..<(dimension * dimension)).map { pointer[$0] }
    }

    /// A single round leaves the COLUMNS exact and the rows off by the amount the
    /// hand-computed [`[1, 0], [0.5, 0.5]`] case shows, so a kernel that normalizes
    /// rows first cannot pass.
    func testOneRoundNormalizesTheColumnsOnly() throws {
        let matrix = try run(
            logits: [1, 0, 0.5, 0.5], dimension: 2, iterations: 1, epsilon: 1e-6
        )
        XCTAssertEqual(matrix[0], 0.593845, accuracy: 1e-5)
        XCTAssertEqual(matrix[1], 0.349755, accuracy: 1e-5)
        XCTAssertEqual(matrix[2], 0.406155, accuracy: 1e-5)
        XCTAssertEqual(matrix[3], 0.650245, accuracy: 1e-5)
    }

    func testConvergedRoundsMatchTheHostRule() throws {
        let logits: [Float] = [0.5, -0.2, 0.1, 1.0, -0.7, 0.3, 0.9, 0.2, -0.4]
        let kernel = try run(logits: logits, dimension: 3, iterations: 20, epsilon: 1e-6)
        let host = DeepSeekV41HyperConnections.sinkhorn(
            logits: logits, dimension: 3, iterations: 20, epsilon: 1e-6
        )
        for index in 0..<9 {
            XCTAssertEqual(kernel[index], host[index], accuracy: 1e-6, "entry \(index)")
        }
    }
}

/// The GPU half of the mHC rule. The host side is pinned against the checkpoint by
/// `DeepSeekV41HyperConnectionGoldenTests`, so these compare the kernels with
/// `DeepSeekV41HyperConnections` on the same inputs: a disagreement is a kernel bug
/// rather than a second derivation of the rule.
final class DeepSeekV41HyperConnectionKernelTests: XCTestCase {

    private let copies = 2
    private let dim = 3
    private var width: Int { copies * dim }
    private var mixCount: Int { (2 + copies) * copies }

    private let streams: [Float] = [0.2, -0.4, 0.6, 0.1, 0.3, -0.5]
    private let scale: [Float] = [0.7, -0.3, 0.5]
    private let base: [Float] = [0.05, -0.1, 0.2, 0.0, -0.05, 0.1, 0.15, -0.2]
    private let projection: [Float] = [
        0.4, -0.2, 0.1, 0.3, -0.5, 0.2,
        -0.1, 0.6, 0.2, -0.3, 0.1, 0.4,
        0.2, 0.1, -0.4, 0.5, 0.3, -0.2,
        -0.3, 0.2, 0.5, 0.1, -0.6, 0.2,
        0.1, -0.5, 0.3, 0.2, 0.4, -0.1,
        0.5, 0.3, -0.1, -0.2, 0.2, 0.6,
        -0.2, 0.4, 0.6, 0.1, -0.3, 0.3,
        0.3, -0.1, 0.2, 0.6, 0.1, -0.4,
    ]

    private var stack: [[Float]] {
        (0..<copies).map { copy in
            (0..<dim).map { streams[copy * dim + $0] }
        }
    }

    private func hostCoefficients() -> DeepSeekV41HyperConnections.Coefficients {
        DeepSeekV41HyperConnections.coefficients(
            streams: stack,
            projection: DeepSeekV41Matrix(
                rows: mixCount, columns: width, values: projection
            ),
            scale: scale,
            base: base,
            normalizationEpsilon: 1e-6,
            epsilon: 1e-6,
            iterations: 3
        )
    }

    private func environment() throws -> (
        MTLDevice, MTLLibrary, MTLCommandQueue
    ) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        let library = try XCTUnwrap(device.makeDefaultLibrary())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        return (device, library, queue)
    }

    private func pipeline(_ name: String, _ library: MTLLibrary, _ device: MTLDevice) throws
        -> MTLComputePipelineState {
        let function = try XCTUnwrap(library.makeFunction(name: name))
        return try device.makeComputePipelineState(function: function)
    }

    private func buffer(_ device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        try XCTUnwrap(
            device.makeBuffer(
                bytes: values, length: values.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        )
    }

    func testProjectionAndSplitMatchTheHostRule() throws {
        let (device, library, queue) = try environment()
        let project = try pipeline("dsv41_hc_project_f32", library, device)
        let split = try pipeline("dsv41_hc_split_f32", library, device)

        let weightBuffer = try buffer(device, projection)
        let streamBuffer = try buffer(device, streams)
        let logitsBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: mixCount * MemoryLayout<Float>.stride, options: .storageModeShared
            )
        )
        let preBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: copies * MemoryLayout<Float>.stride, options: .storageModeShared
            )
        )
        let postBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: copies * MemoryLayout<Float>.stride, options: .storageModeShared
            )
        )
        let combBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: copies * copies * MemoryLayout<Float>.stride, options: .storageModeShared
            )
        )
        let scaleBuffer = try buffer(device, scale)
        let baseBuffer = try buffer(device, base)

        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(project)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(streamBuffer, offset: 0, index: 1)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 2)
        var weightOffset = UInt64(0)
        var normEpsilon: Float = 1e-6
        var widthDimension = UInt32(width)
        var outDimension = UInt32(mixCount)
        // One slot: this case drives the kernel the way a decode step does.
        var slotCount: UInt32 = 1
        encoder.setBytes(&weightOffset, length: 8, index: 3)
        encoder.setBytes(&normEpsilon, length: 4, index: 4)
        encoder.setBytes(&widthDimension, length: 4, index: 5)
        encoder.setBytes(&outDimension, length: 4, index: 6)
        encoder.setBytes(&slotCount, length: 4, index: 7)
        encoder.setThreadgroupMemoryLength(64 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: mixCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)

        encoder.setComputePipelineState(split)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(baseBuffer, offset: 0, index: 2)
        encoder.setBuffer(preBuffer, offset: 0, index: 3)
        encoder.setBuffer(postBuffer, offset: 0, index: 4)
        encoder.setBuffer(combBuffer, offset: 0, index: 5)
        var copiesDimension = UInt32(copies)
        var iterations = UInt32(3)
        var epsilon: Float = 1e-6
        encoder.setBytes(&copiesDimension, length: 4, index: 6)
        encoder.setBytes(&iterations, length: 4, index: 7)
        encoder.setBytes(&epsilon, length: 4, index: 8)
        encoder.setBytes(&slotCount, length: 4, index: 9)
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let host = hostCoefficients()
        let logits = logitsBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<mixCount {
            XCTAssertEqual(logits[index], host.logits[index], accuracy: 1e-5, "logit \(index)")
        }
        let pre = preBuffer.contents().assumingMemoryBound(to: Float.self)
        let post = postBuffer.contents().assumingMemoryBound(to: Float.self)
        let comb = combBuffer.contents().assumingMemoryBound(to: Float.self)
        for copy in 0..<copies {
            XCTAssertEqual(pre[copy], host.pre[copy], accuracy: 1e-6, "pre \(copy)")
            XCTAssertEqual(post[copy], host.post[copy], accuracy: 1e-6, "post \(copy)")
        }
        for index in 0..<(copies * copies) {
            XCTAssertEqual(comb[index], host.comb[index], accuracy: 1e-6, "comb \(index)")
        }
    }

    func testCollapseAndExpansionMatchTheHostRule() throws {
        let (device, library, queue) = try environment()
        let prePipeline = try pipeline("dsv41_hc_pre_f32", library, device)
        let postPipeline = try pipeline("dsv41_hc_post_f32", library, device)

        let mix: [Float] = [0.6, 0.4]
        let sublayerOutput: [Float] = [0.5, -0.25, 1.0]
        let host = hostCoefficients()

        let streamBuffer = try buffer(device, streams)
        let residualBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: streams.count * MemoryLayout<Float>.stride, options: .storageModeShared
            )
        )
        let collapsedBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: dim * MemoryLayout<Float>.stride, options: .storageModeShared
            )
        )
        let mixBuffer = try buffer(device, mix)
        let outputBuffer = try buffer(device, sublayerOutput)
        let gainsBuffer = try buffer(device, host.post)
        let combBuffer = try buffer(device, host.comb)
        let descaledStreams = try buffer(device, streams)

        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(prePipeline)
        encoder.setBuffer(streamBuffer, offset: 0, index: 0)
        encoder.setBuffer(mixBuffer, offset: 0, index: 1)
        encoder.setBuffer(collapsedBuffer, offset: 0, index: 2)
        encoder.setBuffer(residualBuffer, offset: 0, index: 3)
        var widthDimension = UInt32(dim)
        var copiesDimension = UInt32(copies)
        var snapshot = UInt32(1)
        encoder.setBytes(&widthDimension, length: 4, index: 4)
        encoder.setBytes(&copiesDimension, length: 4, index: 5)
        encoder.setBytes(&snapshot, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: dim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)

        encoder.setComputePipelineState(postPipeline)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBuffer(residualBuffer, offset: 0, index: 1)
        encoder.setBuffer(gainsBuffer, offset: 0, index: 2)
        encoder.setBuffer(combBuffer, offset: 0, index: 3)
        encoder.setBuffer(descaledStreams, offset: 0, index: 4)
        var streamWidth = UInt32(dim)
        var streamCopies = UInt32(copies)
        encoder.setBytes(&streamWidth, length: 4, index: 5)
        encoder.setBytes(&streamCopies, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: dim, height: copies, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let collapsed = DeepSeekV41HyperConnections.pre(streams: stack, mix: mix)
        let collapsedOut = collapsedBuffer.contents().assumingMemoryBound(to: Float.self)
        let residual = residualBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<dim {
            XCTAssertEqual(collapsedOut[index], collapsed[index], accuracy: 1e-6, "collapse \(index)")
            // The snapshot `hc_post` mixes back in is the block input, unchanged.
            XCTAssertEqual(residual[index], streams[index], accuracy: 1e-6, "residual \(index)")
        }

        let expanded = DeepSeekV41HyperConnections.post(
            output: sublayerOutput, residual: stack, gains: host.post, comb: host.comb
        )
        let expandedOut = descaledStreams.contents().assumingMemoryBound(to: Float.self)
        for copy in 0..<copies {
            for index in 0..<dim {
                XCTAssertEqual(
                    expandedOut[copy * dim + index],
                    expanded[copy][index],
                    accuracy: 1e-6,
                    "copy \(copy) dim \(index)"
                )
            }
        }
    }

    /// The final collapse has no `hc_post` after it, so it asks for no snapshot.
    /// The flag is what says so, since the buffer is bound either way.
    func testTheCollapseLeavesTheResidualAloneWhenNotSnapshotting() throws {
        let (device, library, queue) = try environment()
        let prePipeline = try pipeline("dsv41_hc_pre_f32", library, device)

        let mix: [Float] = [0.6, 0.4]
        let sentinel = [Float](repeating: -99, count: streams.count)
        let streamBuffer = try buffer(device, streams)
        let residualBuffer = try buffer(device, sentinel)
        let collapsedBuffer = try XCTUnwrap(
            device.makeBuffer(
                length: dim * MemoryLayout<Float>.stride, options: .storageModeShared
            )
        )
        let mixBuffer = try buffer(device, mix)

        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(prePipeline)
        encoder.setBuffer(streamBuffer, offset: 0, index: 0)
        encoder.setBuffer(mixBuffer, offset: 0, index: 1)
        encoder.setBuffer(collapsedBuffer, offset: 0, index: 2)
        encoder.setBuffer(residualBuffer, offset: 0, index: 3)
        var widthDimension = UInt32(dim)
        var copiesDimension = UInt32(copies)
        var snapshot = UInt32(0)
        encoder.setBytes(&widthDimension, length: 4, index: 4)
        encoder.setBytes(&copiesDimension, length: 4, index: 5)
        encoder.setBytes(&snapshot, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: dim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let collapsed = DeepSeekV41HyperConnections.pre(streams: stack, mix: mix)
        let collapsedOut = collapsedBuffer.contents().assumingMemoryBound(to: Float.self)
        let residual = residualBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<dim {
            XCTAssertEqual(collapsedOut[index], collapsed[index], accuracy: 1e-6, "collapse \(index)")
        }
        for index in 0..<streams.count {
            XCTAssertEqual(residual[index], sentinel[index], "residual \(index)")
        }
    }
}
