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
        DeepSeekV41DeviceTensor(
            shardIndex: 0, byteOffset: 0, rows: 8, columns: 8, dtype: dtype, byteLength: 64
        )
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

    func testANilSelectionReadsEveryReachableRow() throws {
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
        // Nil is the reference's no-indexer case: every row, not the newest few.
        XCTAssertEqual(rows.compressedRows, [0, 1, 2, 3, 4])
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

    /// The rows the walk itself selected, so the plan and the keys are compared
    /// against the same list rather than against a rule restated in the test.
    private func consent(
        forward: DeepSeekV41CPUForward,
        layer: DeepSeekV41LayerPlan,
        position: Int
    ) -> [Int]? {
        guard let rows = forward.indexSelections[layer.compressedIndexSource],
              rows.indices.contains(position) else { return nil }
        return rows[position]
    }

    func testReferenceAndPlanReadTheSameRowCount() throws {
        let config = try DeepSeekV41TestFixtures.config()
        let plan = try config.makeLayerPlan()
        let source = DeepSeekV41TestFixtures.weights(config: config, plan: plan)
        let forward = DeepSeekV41CPUForward(config: config, plan: plan, weights: source)
        _ = forward.prefill(tokens: [1, 2, 3, 4, 5, 6, 7])

        for position in 0..<7 {
            for layer in plan {
                let selection = consent(forward: forward, layer: layer, position: position)
                let planRows = DeepSeekV41CacheAddressing.attentionRowPlan(
                    layer: layer,
                    position: position,
                    config: config,
                    plan: plan,
                    geometry: forward.geometry,
                    selection: selection
                )
                // The reference drops a row it has not written yet; before position 0
                // there are none, and past it every planned row exists.
                let keys = forward.attentionKeys(
                    layer: layer, position: position, selection: selection
                )
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

        // A prefilled block stages every position before its command buffer exists,
        // so the block is replicated per position and the tables stay disjoint.
        let wide = DeepSeekV41EngramSlot.slots(config: config, positions: 8)
        XCTAssertEqual(wide[0].rawByteStride, wide[0].rawByteCount * 8)
        XCTAssertEqual(wide[1].rawByteOffset, wide[0].rawByteStride)
        XCTAssertEqual(wide[0].rawBytes(atPositionSlot: 3), wide[0].rawByteCount * 3)
        XCTAssertEqual(
            wide[1].rawBytes(atPositionSlot: 7),
            wide[0].rawByteStride + wide[1].rawByteCount * 7
        )
        XCTAssertEqual(
            wide[1].valueBytes(atPositionSlot: 2),
            (wide[0].valueStride + wide[1].valueCount * 2) * MemoryLayout<Float>.stride
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

        // A full block of eight positions takes a kernel that streams an output row
        // once for every position, and it does so at every row count: it measured
        // faster than both other forms from 448 rows to 73,728, so there is no
        // threshold on the ROW count to get wrong.
        for outDim in [128, 448, 1_536, 7_168, 73_728] {
            XCTAssertEqual(
                DeepSeekV41DevicePlan.bf16GemvForm(outDim: outDim, inDim: 1_536, rows: 8),
                .block,
                "an eight-position block at \(outDim) rows and a short input row"
            )
        }
        // Which block form, though, turns on the INPUT width: a row long enough that
        // its eight position vectors dominate the stage shares one load between two
        // output rows, and a short one does not (1,536 measured 1.00x, 8,192 measured
        // 1.20-1.34x). The published attention projections are exactly that split.
        XCTAssertEqual(
            DeepSeekV41DevicePlan.bf16GemvForm(outDim: 65_536, inDim: 1_536, rows: 8), .block
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.bf16GemvForm(outDim: 8_192, inDim: 8_192, rows: 8), .blockPair
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.bf16GemvForm(outDim: 7_168, inDim: 8_192, rows: 8), .blockPair
        )
        // A shorter batch has no block kernel at all, and falls back to the two-way rule.
        XCTAssertEqual(
            DeepSeekV41DevicePlan.bf16GemvForm(outDim: 128, inDim: 8_192, rows: 1), .wideGrid
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.bf16GemvForm(outDim: 448, inDim: 8_192, rows: 3), .wideGrid
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.bf16GemvForm(outDim: 7_168, inDim: 8_192, rows: 3), .perRow
        )
        XCTAssertEqual(
            DeepSeekV41DevicePlan.bf16GemvForm(outDim: 73_728, inDim: 8_192, rows: 7), .perRow
        )
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
        encoder.setThreadgroupMemoryLength(max(16, ((lanes / 32) * 4 + 15) & ~15), index: 0)
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
                  length: 2 * width * MemoryLayout<UInt16>.stride,
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

        // The destination is the cache's own width, so this is a copy of FP16 rows
        // rather than a widening of them.
        let out = dense.contents().assumingMemoryBound(to: UInt16.self)
        func row(_ index: Int) -> [Float] {
            Array(UnsafeBufferPointer(start: out + index * width, count: width))
                .map { Float(Float16(bitPattern: $0)) }
        }
        // Bound at offset 0 this would read rows 77 and 77 (out of the buffer).
        XCTAssertEqual(row(0), [30, 31, 32, 33])
        XCTAssertEqual(row(1), [10, 11, 12, 13])
    }
}

// MARK: - Attention kernel

final class DeepSeekV41AttentionKernelTests: XCTestCase {

    /// The kernel production dispatches. The FP32 form is still in the library as
    /// the A/B baseline for `Benchmarks/PERF_FINDINGS.md` "DSV4.1 #17", but the
    /// contract that has to hold is the one the executor binds.
    private func makePipeline() throws -> (MTLDevice, MTLComputePipelineState, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_attention_csa2_f16"),
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
        // Records arrive at the cache's width. `values` produces eighths, which are
        // exact in FP16, so widening them back for the host rule is lossless.
        let keys = values(rows * width, seed: 2).map { Float16($0) }
        let widenedKeys = keys.map { Float($0) }
        let sink = values(heads, seed: 3)

        guard let queryBuffer = device.makeBuffer(
                  bytes: query,
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: keys,
                  length: keys.count * MemoryLayout<Float16>.stride,
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
        // The dispatch sizes its scores block the way the executor does: rounded up
        // to a 16-byte multiple, which Metal requires and a short row count does not
        // give on its own (1 row is 132 bytes).
        let scoreBytes = (
            (rows + 32) * MemoryLayout<Float>.stride + 15
        ) & ~15
        encoder.setThreadgroupMemoryLength(scoreBytes, index: 0)
        // Bound even for one position: the kernel takes the single-position path
        // either way, and an unbound `constant` pointer fails validation.
        var plan = DeepSeekV41AttentionSlotPlan(slotCount: 1, keyRowStride: UInt32(rows))
        encoder.setBytes(
            &plan, length: MemoryLayout<DeepSeekV41AttentionSlotPlan>.stride, index: 9
        )
        var slotRows = UInt32(rows)
        encoder.setBytes(&slotRows, length: 4, index: 10)
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
            keys: (0..<rows).map {
                Array(widenedKeys[($0 * width)..<(($0 + 1) * width)])
            },
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
        let keys: [Float16] = [0.5, -0.25, 1.5, -2]
        // The sink is disabled, not absent: an unbound `constant` pointer fails
        // Metal's validation even though the kernel tests it against nullptr.
        let sink = [Float](repeating: -1e30, count: heads)

        guard let queryBuffer = device.makeBuffer(
                  bytes: query,
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: keys,
                  length: keys.count * MemoryLayout<Float16>.stride,
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
        // The dispatch sizes its scores block the way the executor does: rounded up
        // to a 16-byte multiple, which Metal requires and a short row count does not
        // give on its own (1 row is 132 bytes).
        let scoreBytes = (
            (rows + 32) * MemoryLayout<Float>.stride + 15
        ) & ~15
        encoder.setThreadgroupMemoryLength(scoreBytes, index: 0)
        // Bound even for one position: the kernel takes the single-position path
        // either way, and an unbound `constant` pointer fails validation.
        var plan = DeepSeekV41AttentionSlotPlan(slotCount: 1, keyRowStride: UInt32(rows))
        encoder.setBytes(
            &plan, length: MemoryLayout<DeepSeekV41AttentionSlotPlan>.stride, index: 9
        )
        var slotRows = UInt32(rows)
        encoder.setBytes(&slotRows, length: 4, index: 10)
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
            XCTAssertEqual(output[dim], Float(keys[dim]), accuracy: 1e-6)
            XCTAssertEqual(output[vHeadDim + dim], Float(keys[dim]), accuracy: 1e-6)
        }
    }
}

// MARK: - Indexer score kernel

/// The GPU half of the sparse-attention rule.
///
/// Every assertion about WHICH compressed rows a query reads rests on the score
/// vector this kernel writes: the executor reads it back and hands it to
/// `DeepSeekV41Indexer.selectedRows`, so a kernel that scored differently would
/// silently select differently while the host rule's own tests kept passing. It is
/// checked against `DeepSeekV41Indexer.positionScores`, which is the function the
/// reference vectors pin.
final class DeepSeekV41IndexerKernelTests: XCTestCase {

    private func makePipeline() throws -> (MTLDevice, MTLComputePipelineState, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_indexer_scores_f32"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            XCTFail("Could not build the indexer score pipeline")
            throw XCTSkip("no indexer score pipeline")
        }
        return (device, pipeline, queue)
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self),
            count: count
        ))
    }

    /// Runs one dispatch and returns the score vector it wrote.
    private func scores(
        query: [Float],
        keys: [Float],
        weights: [Float],
        headCount: Int,
        headWidth: Int,
        rowCount: Int,
        visibleRows: Int,
        weightScale: Float,
        pipeline: MTLComputePipelineState,
        device: MTLDevice,
        queue: MTLCommandQueue
    ) throws -> [Float] {
        guard let queryBuffer = device.makeBuffer(
                  bytes: query,
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: keys,
                  length: keys.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let weightBuffer = device.makeBuffer(
                  bytes: weights,
                  length: weights.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let scoreBuffer = device.makeBuffer(
                  length: rowCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("Could not allocate the indexer score buffers")
            return []
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(keyBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(scoreBuffer, offset: 0, index: 3)
        var heads = UInt32(headCount)
        var width = UInt32(headWidth)
        var rows = UInt32(rowCount)
        var visible = UInt32(visibleRows)
        var scale = weightScale
        encoder.setBytes(&heads, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&rows, length: 4, index: 6)
        encoder.setBytes(&visible, length: 4, index: 7)
        encoder.setBytes(&scale, length: 4, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: rowCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(rowCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
        return floats(scoreBuffer, count: rowCount)
    }

    /// The rectifier is PER HEAD and comes BEFORE the weight, so a head whose weight
    /// is negative turns a positive dot negative and a head that votes against a row
    /// contributes nothing rather than subtracting. Both halves of that sentence are
    /// load-bearing and both are invisible to a fixture whose weights are all
    /// positive: on the published checkpoint most rows score exactly zero and the
    /// cut lands in the negative tail, which only happens if negative weights reach
    /// the sum.
    func testTheScoreKernelRanksLikeTheRuleWithNegativeWeights() throws {
        let (device, pipeline, queue) = try makePipeline()
        let heads = 5
        // Ten wide on purpose: the published `index_head_dim` is 128, so the kernel's
        // remainder loop never runs in production and only a width that is not a
        // multiple of four exercises it.
        let width = 10
        let rows = 13
        let query: [Float] = (0..<(heads * width)).map { index in
            Float((index * 7) % 11 - 5) / 4
        }
        let keys: [Float] = (0..<(rows * width)).map { index in
            // The last row is an all-zero key, so every head's dot with it is zero and
            // the row lands on exactly 0.0 instead of a small number that might or
            // might not survive the rectifier.
            index >= (rows - 1) * width ? 0 : Float((index * 5) % 13 - 6) / 4
        }
        // A mix of signs, so some rows lose a head's contribution by rectification
        // and others are pushed negative by a negative weight.
        let weights: [Float] = [-0.75, 0.5, -0.25, 1.25, 0.0]

        let got = try scores(
            query: query, keys: keys, weights: weights,
            headCount: heads, headWidth: width, rowCount: rows, visibleRows: rows,
            weightScale: 0.5, pipeline: pipeline, device: device, queue: queue
        )
        let want = DeepSeekV41Indexer.positionScores(
            queryHeads: (0..<heads).map { head in
                Array(query[(head * width)..<((head + 1) * width)])
            },
            keys: (0..<rows).map { row in
                Array(keys[(row * width)..<((row + 1) * width)])
            },
            headWeights: weights.map { $0 * 0.5 }
        )
        XCTAssertEqual(got.count, want.count)
        let scale = want.map(abs).max() ?? 1
        for index in 0..<want.count {
            XCTAssertEqual(
                got[index], want[index], accuracy: max(scale, 1) * 1e-5,
                "the score kernel disagrees with the rule at row \(index)"
            )
        }
        // The vectors are only evidence if they reach both states: a negative score
        // proves a rectified dot met a negative weight, and a zero proves a head
        // voting against a row removed only its own contribution.
        XCTAssertTrue(want.contains { $0 < 0 }, "no row scored negative, so the sign mix is untested")
        XCTAssertTrue(want.contains { $0 == 0 }, "no row scored exactly zero")
    }

    /// A row past the query's frontier is silenced rather than scored, because the
    /// rule drops such rows AFTER the cut and a finite score would let one take a
    /// slot the query cannot attend to. The rows the query can see are untouched by
    /// that, and a visible row that scores zero keeps a zero rather than the
    /// sentinel.
    func testTheScoreKernelSilencesOnlyTheRowsPastTheFrontier() throws {
        let (device, pipeline, queue) = try makePipeline()
        let heads = 3
        let width = 8
        let rows = 20
        let visible = 11
        let query: [Float] = (0..<(heads * width)).map { index in
            Float((index * 3) % 7 - 3) / 4
        }
        let keys: [Float] = (0..<(rows * width)).map { index in
            Float((index * 11) % 17 - 8) / 8
        }
        let weights: [Float] = [0.5, -0.5, 0.25]

        let got = try scores(
            query: query, keys: keys, weights: weights,
            headCount: heads, headWidth: width, rowCount: rows, visibleRows: visible,
            weightScale: 1, pipeline: pipeline, device: device, queue: queue
        )
        let want = DeepSeekV41Indexer.positionScores(
            queryHeads: (0..<heads).map { head in
                Array(query[(head * width)..<((head + 1) * width)])
            },
            keys: (0..<rows).map { row in
                Array(keys[(row * width)..<((row + 1) * width)])
            },
            headWeights: weights
        )
        let scale = want.map(abs).max() ?? 1
        for row in 0..<visible {
            XCTAssertEqual(
                got[row], want[row], accuracy: max(scale, 1) * 1e-5,
                "row \(row) is reachable but the kernel scored it differently"
            )
        }
        for row in visible..<rows {
            XCTAssertEqual(
                got[row], -.infinity,
                "row \(row) is past the frontier and must be silenced, not scored"
            )
        }
        // The selection the executor then makes is the reason this matters: with the
        // reachable half ranked, the silenced tail can never displace a visible row
        // no matter how high its score came out.
        let selected = DeepSeekV41Indexer.selectedRows(scores: got, visibleRows: visible, topK: 5)
        XCTAssertEqual(selected.count, 5)
        XCTAssertTrue(
            selected.allSatisfy { $0 < visible },
            "the cut selected a row past the frontier: \(selected)"
        )
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

    /// The lane-parallel top-k returns exactly what the one-thread scan does.
    ///
    /// `#63` measured the one-thread scan at 72 us per position, because each of 384
    /// experts costs three transcendentals on a single lane and the top-`k` arrays are
    /// declared at their bound. The lane form splits the scan across a SIMD group and
    /// merges each lane's candidates with the SAME total order (score descending,
    /// expert id ascending), so nothing is re-associated and the requirement is
    /// equality rather than a tolerance — which also means a wrong merge cannot hide
    /// inside a rounding bar. The last case is shorter than one SIMD group, so the
    /// lanes that own no expert must publish nothing that can be selected.
    func testTheLaneFormOfTheTopKIsTheOneThreadFormsResult() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        let library = try XCTUnwrap(device.makeDefaultLibrary())
        let slowFunction = try XCTUnwrap(library.makeFunction(name: "dsv41_router_topk"))
        let laneFunction = try XCTUnwrap(library.makeFunction(name: "dsv41_router_topk_lanes"))
        let slow = try device.makeComputePipelineState(function: slowFunction)
        let lanes = try device.makeComputePipelineState(function: laneFunction)
        let queue = try XCTUnwrap(device.makeCommandQueue())

        func run(
            _ pipeline: MTLComputePipelineState,
            lanesPerSlot: Int,
            slots: Int,
            expertCount: Int,
            topK: Int,
            logits: [Float],
            bias: [Float]
        ) throws -> (indices: [UInt32], weights: [Float]) {
            let indexStride = max(4 * topK, 16)
            let logitsBuffer = try XCTUnwrap(
                device.makeBuffer(
                    bytes: logits, length: logits.count * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
            )
            let biasBuffer = try XCTUnwrap(
                device.makeBuffer(
                    bytes: bias, length: bias.count * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
            )
            let indexBuffer = try XCTUnwrap(
                device.makeBuffer(
                    length: slots * indexStride * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared
                )
            )
            let weightBuffer = try XCTUnwrap(
                device.makeBuffer(
                    length: slots * topK * MemoryLayout<Float>.stride, options: .storageModeShared
                )
            )
            let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasBuffer, offset: 0, index: 1)
            encoder.setBuffer(indexBuffer, offset: 0, index: 2)
            encoder.setBuffer(weightBuffer, offset: 0, index: 3)
            var experts = UInt32(expertCount)
            var count = UInt32(topK)
            var scaling: Float = 2.5
            var normalise: UInt32 = 1
            var scoring: UInt32 = 1  // sqrtsoftplus, the checkpoint's
            var temperature: Float = 1
            var stride = UInt32(indexStride)
            var slotCount = UInt32(slots)
            encoder.setBytes(&experts, length: 4, index: 4)
            encoder.setBytes(&count, length: 4, index: 5)
            encoder.setBytes(&scaling, length: 4, index: 6)
            encoder.setBytes(&normalise, length: 4, index: 7)
            encoder.setBytes(&scoring, length: 4, index: 8)
            encoder.setBytes(&temperature, length: 4, index: 9)
            encoder.setBytes(&stride, length: 4, index: 10)
            encoder.setBytes(&slotCount, length: 4, index: 11)
            if lanesPerSlot > 1 {
                for index in 0..<3 {
                    encoder.setThreadgroupMemoryLength(
                        lanesPerSlot * 8 * MemoryLayout<Float>.stride, index: index
                    )
                }
                encoder.dispatchThreadgroups(
                    MTLSize(width: slots, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: lanesPerSlot, height: 1, depth: 1)
                )
            } else {
                encoder.dispatchThreads(
                    MTLSize(width: slots, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
                )
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            XCTAssertNil(commandBuffer.error)
            let indexPointer = indexBuffer.contents().assumingMemoryBound(to: UInt32.self)
            let weightPointer = weightBuffer.contents().assumingMemoryBound(to: Float.self)
            var indices: [UInt32] = []
            var weights: [Float] = []
            for slot in 0..<slots {
                indices.append(contentsOf: (0..<topK).map { indexPointer[slot * indexStride + $0] })
                weights.append(contentsOf: (0..<topK).map { weightPointer[slot * topK + $0] })
            }
            return (indices, weights)
        }

        // Deterministic logits with mixed signs plus a DUPLICATE pair, so the
        // tie-break (a lower expert id wins) has to be the same in both merges.
        func logits(_ count: Int, shift: Int) -> [Float] {
            (0..<count).map { index in
                let at = (index + shift) % count
                if at == 17 || at == 200 { return 1.25 }
                let wave = Float((at * 41) % 29) / 29 - 0.5
                return wave * 6 + (at % 3 == 0 ? 0.4 : 0)
            }
        }
        func bias(_ count: Int) -> [Float] {
            (0..<count).map { Float(($0 * 13) % 11) / 22 - 0.25 }
        }

        let shapes: [(experts: Int, topK: Int, slots: Int)] = [
            (384, 6, 1),
            (384, 6, 8),
            (5, 3, 1),
        ]
        for shape in shapes {
            var allLogits: [Float] = []
            for slot in 0..<shape.slots {
                allLogits.append(contentsOf: logits(shape.experts, shift: slot * 7))
            }
            let byThread = try run(
                slow, lanesPerSlot: 1, slots: shape.slots, expertCount: shape.experts,
                topK: shape.topK, logits: allLogits, bias: bias(shape.experts)
            )
            let byLane = try run(
                lanes, lanesPerSlot: 32, slots: shape.slots, expertCount: shape.experts,
                topK: shape.topK, logits: allLogits, bias: bias(shape.experts)
            )
            for rank in 0..<(shape.slots * shape.topK) {
                XCTAssertEqual(
                    byLane.indices[rank], byThread.indices[rank],
                    "\(shape.experts) experts, \(shape.topK) of them, rank \(rank) selected "
                        + "a different expert"
                )
                XCTAssertEqual(
                    byLane.weights[rank], byThread.weights[rank],
                    "\(shape.experts) experts, \(shape.topK) of them, rank \(rank) got a "
                        + "different weight"
                )
            }
        }
        XCTAssertTrue(DeepSeekV41DevicePlan.routerTopKWantsLanes(topK: 6))
        XCTAssertTrue(DeepSeekV41DevicePlan.routerTopKWantsLanes(topK: 1))
        XCTAssertFalse(
            DeepSeekV41DevicePlan.routerTopKWantsLanes(topK: 9),
            "a wider top-k must fall back to the one-thread scan rather than spill the merge"
        )
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
        encoder.setThreadgroupMemoryLength(max(16, ((lanes / 32) * 4 + 15) & ~15), index: 0)
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
        encoder.setThreadgroupMemoryLength(max(16, ((lanes / 32) * 4 + 15) & ~15), index: 0)
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

    /// The register-resident Sinkhorn computes the same split as the device-memory
    /// one.
    ///
    /// `#62` found the split's schedule costing 3.0 us per Sinkhorn iteration and 59 us
    /// for the checkpoint's twenty, because the 4x4 matrix makes a round trip to device
    /// memory on every one of ~160 normalization passes, on ONE thread. The replacement
    /// keeps it in registers and folds every sum in the same order, so the bar is a
    /// rounding one rather than a tolerance: the seed's product+add rounds once in a
    /// register where a store rounds it twice, which moves the coefficients by a single
    /// ulp, and the schedule is contractive afterwards. Both `mult == 4` (the fast path)
    /// and `mult == 2` (the general helper the kernel falls back to) are checked, and 20
    /// iterations is the checkpoint's own `hc_sinkhorn_iters`.
    func testTheRegisterResidentSinkhornComputesTheSameSplit() throws {
        let (device, library, queue) = try environment()
        let devicePipeline = try pipeline("dsv41_hc_split_f32", library, device)
        let registerPipeline = try pipeline("dsv41_hc_split_f32_regs", library, device)

        func run(
            copies: Int,
            logits: [Float],
            scale: [Float],
            base: [Float],
            iterations: Int,
            pipeline: MTLComputePipelineState
        ) throws -> (pre: [Float], post: [Float], comb: [Float]) {
            let logitsBuffer = try buffer(device, logits)
            let scaleBuffer = try buffer(device, scale)
            let baseBuffer = try buffer(device, base)
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
                    length: copies * copies * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
            )
            let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(baseBuffer, offset: 0, index: 2)
            encoder.setBuffer(preBuffer, offset: 0, index: 3)
            encoder.setBuffer(postBuffer, offset: 0, index: 4)
            encoder.setBuffer(combBuffer, offset: 0, index: 5)
            var mult = UInt32(copies)
            var schedule = UInt32(iterations)
            var epsilon: Float = 1e-6
            var slots: UInt32 = 1
            encoder.setBytes(&mult, length: 4, index: 6)
            encoder.setBytes(&schedule, length: 4, index: 7)
            encoder.setBytes(&epsilon, length: 4, index: 8)
            encoder.setBytes(&slots, length: 4, index: 9)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            XCTAssertNil(commandBuffer.error)
            func read(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
                Array(
                    UnsafeBufferPointer(
                        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count
                    )
                )
            }
            return (
                read(preBuffer, copies),
                read(postBuffer, copies),
                read(combBuffer, copies * copies)
            )
        }

        // Deterministic logits with mixed signs, so no row or column is uniform and a
        // transposed index or a wrong fold order cannot cancel out.
        func logits(_ count: Int) -> [Float] {
            (0..<count).map { index in
                let wave = Float((index * 37) % 23) / 23 - 0.5
                return wave * 4 + (index % 2 == 0 ? 0.3 : -0.2)
            }
        }
        let scale: [Float] = [0.7, -0.3, 0.5]
        for copies in [4, 2] {
            let count = (2 + copies) * copies
            let base = logits(count).map { $0 * 0.25 }
            let values = logits(count)
            for iterations in [0, 1, 3, 20] {
                let byDevice = try run(
                    copies: copies, logits: values, scale: scale, base: base,
                    iterations: iterations, pipeline: devicePipeline
                )
                let byRegister = try run(
                    copies: copies, logits: values, scale: scale, base: base,
                    iterations: iterations, pipeline: registerPipeline
                )
                for index in 0..<copies {
                    XCTAssertEqual(
                        byRegister.pre[index], byDevice.pre[index],
                        accuracy: 1e-6,
                        "pre \(index) moved at mult \(copies), \(iterations) iteration(s)"
                    )
                    XCTAssertEqual(
                        byRegister.post[index], byDevice.post[index],
                        accuracy: 1e-6,
                        "post \(index) moved at mult \(copies), \(iterations) iteration(s)"
                    )
                }
                for index in 0..<(copies * copies) {
                    // `comb` is a normalized matrix: every value is in (0, 1], so an
                    // absolute bar is also a relative one.
                    XCTAssertEqual(
                        byRegister.comb[index], byDevice.comb[index],
                        accuracy: 1e-7,
                        "comb \(index) moved at mult \(copies), \(iterations) iteration(s)"
                    )
                }
            }
        }
    }
}

// MARK: - FP8 block-scaled GEMV forms

/// The two shapes of one 32x32 ue8m0 dot product, checked against each other and
/// against the host rule.
///
/// `dsv41_fp8_block32_gemv` walks a row in one lane; `dsv41_fp8_block32_gemv_rows`
/// strides the row across a threadgroup. They must agree to rounding, and both must
/// agree with the arithmetic the checkpoint's scale grid describes, because a lane
/// reading the wrong scale row (the grid is 32x32, so 32 consecutive rows share one)
/// still produces numbers of a plausible size.
final class DeepSeekV41FP8GemvKernelTests: XCTestCase {

    private func environment() throws -> (MTLDevice, MTLLibrary, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary() else {
            XCTFail("the default library is missing")
            throw XCTSkip("no default library")
        }
        guard let queue = device.makeCommandQueue() else {
            XCTFail("no compute queue")
            throw XCTSkip("no compute queue")
        }
        return (device, library, queue)
    }

    /// E4M3 bytes whose exponents stay in the finite range, so a decoded value is a
    /// plain number: exponent 0 is subnormal and 15 is NaN/Inf territory.
    private func weightBytes(_ count: Int) -> [UInt8] {
        (0..<count).map { index in
            let sign: UInt8 = (index % 7 == 3) ? 0x80 : 0
            let exponent = UInt8((index / 5) % 9)
            let mantissa = UInt8(index % 8)
            return sign | (exponent << 3) | mantissa
        }
    }

    /// Block scales near 2^0, so the block sums stay in a range where a relative bar
    /// means something.
    private func scaleBytes(_ count: Int) -> [UInt8] {
        (0..<count).map { index in UInt8(124 + (index % 7)) }
    }

    private func values(_ count: Int) -> [Float] {
        (0..<count).map { index in Float((index &* 29 &+ 7) % 17 - 8) / 16 }
    }

    /// The host rule: per 32-value block, the dot product scaled by that block's
    /// E8M0 byte, summed over the row. The scale row is `row / 32` because the
    /// checkpoint's grid is 32x32.
    private func reference(
        weight: [UInt8],
        scales: [UInt8],
        inDim: Int,
        outDim: Int,
        tokens: Int,
        input: [Float],
        inputRowStride: Int,
        outputRowStride: Int
    ) -> [Float] {
        let blocks = max(1, inDim / 32)
        var out = [Float](repeating: 0, count: tokens * outputRowStride)
        for token in 0..<tokens {
            for row in 0..<outDim {
                var total: Float = 0
                for block in 0..<blocks {
                    let scale = DeepSeekV41EngramRowDequantizer.e8m0(
                        scales[(row / 32) * blocks + block]
                    )
                    var blockDot: Float = 0
                    for index in 0..<32 {
                        let at = row * inDim + block * 32 + index
                        blockDot += DeepSeekV41EngramRowDequantizer.e4m3(weight[at])
                            * input[token * inputRowStride + block * 32 + index]
                    }
                    total += blockDot * scale
                }
                out[token * outputRowStride + row] = total
            }
        }
        return out
    }

    /// Runs the shipped shape and the coalesced one over the same bytes.
    private func dispatch(
        pipeline: MTLComputePipelineState,
        wideGrid: Bool,
        device: MTLDevice,
        queue: MTLCommandQueue,
        weight: MTLBuffer,
        scale: MTLBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        inDim: Int,
        outDim: Int,
        tokens: Int,
        inputRowStride: Int,
        outputRowStride: Int
    ) throws {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("no command buffer")
            return
        }
        var weightOffset = UInt64(0)
        var scaleOffset = UInt64(0)
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)
        var inStride = UInt32(inputRowStride)
        var outStride = UInt32(outputRowStride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&weightOffset, length: 8, index: 4)
        encoder.setBytes(&scaleOffset, length: 8, index: 5)
        encoder.setBytes(&inDimension, length: 4, index: 6)
        encoder.setBytes(&outDimension, length: 4, index: 7)
        encoder.setBytes(&inStride, length: 4, index: 8)
        encoder.setBytes(&outStride, length: 4, index: 9)
        if wideGrid {
            encoder.setThreadgroupMemoryLength(
                32 * MemoryLayout<Float>.stride, index: 0
            )
            encoder.dispatchThreadgroups(
                MTLSize(width: outDim * tokens, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        } else {
            encoder.dispatchThreads(
                MTLSize(width: outDim, height: tokens, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// One shape through all three: the host rule, the one-thread form, the
    /// coalesced form. `inDim` 1,280 is 40 blocks against 64 lanes, so most lanes
    /// read one block and the tail of the threadgroup has nothing to do — the case a
    /// reduction across SIMD groups gets wrong if it reads an unwritten partial.
    private func check(
        outDim: Int,
        inDim: Int,
        tokens: Int,
        inputRowStride: Int? = nil,
        outputRowStride: Int? = nil
    ) throws {
        let (device, library, queue) = try environment()
        guard let perRowFunction = library.makeFunction(
                  name: DeepSeekV41Kernels.fp8Gemv
              ),
              let rowsFunction = library.makeFunction(
                  name: DeepSeekV41Kernels.fp8GemvRows
              ),
              let perRow = try? device.makeComputePipelineState(function: perRowFunction),
              let rows = try? device.makeComputePipelineState(function: rowsFunction) else {
            XCTFail("the FP8 GEMV kernels are not in the default library")
            return
        }
        let inStride = inputRowStride ?? inDim
        let outStride = outputRowStride ?? outDim
        let blocks = inDim / 32
        let weight = weightBytes(outDim * inDim)
        let scales = scaleBytes((outDim / 32) * blocks)
        let input = values(tokens * inStride)
        guard let weightBuffer = device.makeBuffer(
                  bytes: weight,
                  length: weight.count * MemoryLayout<UInt8>.stride,
                  options: .storageModeShared),
              let scaleBuffer = device.makeBuffer(
                  bytes: scales,
                  length: scales.count * MemoryLayout<UInt8>.stride,
                  options: .storageModeShared),
              let inputBuffer = device.makeBuffer(
                  bytes: input,
                  length: input.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let perRowOutput = device.makeBuffer(
                  length: tokens * outStride * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let rowsOutput = device.makeBuffer(
                  length: tokens * outStride * MemoryLayout<Float>.stride,
                  options: .storageModeShared) else {
            XCTFail("the FP8 GEMV buffers did not allocate")
            return
        }
        memset(perRowOutput.contents(), 0, perRowOutput.length)
        memset(rowsOutput.contents(), 0, rowsOutput.length)

        try dispatch(
            pipeline: perRow, wideGrid: false, device: device, queue: queue,
            weight: weightBuffer, scale: scaleBuffer, input: inputBuffer,
            output: perRowOutput, inDim: inDim, outDim: outDim, tokens: tokens,
            inputRowStride: inStride, outputRowStride: outStride
        )
        try dispatch(
            pipeline: rows, wideGrid: true, device: device, queue: queue,
            weight: weightBuffer, scale: scaleBuffer, input: inputBuffer,
            output: rowsOutput, inDim: inDim, outDim: outDim, tokens: tokens,
            inputRowStride: inStride, outputRowStride: outStride
        )

        let expected = reference(
            weight: weight, scales: scales, inDim: inDim, outDim: outDim, tokens: tokens,
            input: input, inputRowStride: inStride, outputRowStride: outStride
        )
        let shipped = floats(perRowOutput, count: tokens * outStride)
        let coalesced = floats(rowsOutput, count: tokens * outStride)
        var worstExpected: Float = 0
        var worstShipped: Float = 0
        for token in 0..<tokens {
            for row in 0..<outDim {
                let at = token * outStride + row
                worstExpected = max(worstExpected, abs(coalesced[at] - expected[at]))
                worstShipped = max(worstShipped, abs(shipped[at] - expected[at]))
            }
        }
        // The two forms sum the same products in a different order, and the host rule
        // in a third; a bar scaled to the values keeps it a rounding check.
        let magnitude = max(1, expected.map { abs($0) }.max() ?? 1)
        XCTAssertLessThan(
            worstExpected, magnitude * 1e-4,
            "coalesced form drifted from the host rule at \(outDim)x\(inDim), \(tokens) token(s)"
        )
        XCTAssertLessThan(
            worstShipped, magnitude * 1e-4,
            "the shipped form drifted from the host rule at \(outDim)x\(inDim)"
        )
    }

    func testTheCoalescedFormMatchesTheHostRuleAtAPublishedWidth() throws {
        try check(outDim: 96, inDim: 1_280, tokens: 1)
    }

    /// A block of positions addresses its rows with strides wider than the shape, so
    /// a kernel that ignores them writes every position on top of the first.
    func testTheCoalescedFormHonoursPositionStrides() throws {
        try check(
            outDim: 64, inDim: 2_304, tokens: 3,
            inputRowStride: 2_560, outputRowStride: 96
        )
    }

    /// Rows shorter than the block count of a lane: 512 values is 16 blocks against
    /// 64 lanes, so 48 lanes contribute a zero partial.
    func testTheCoalescedFormHandlesAShortRow() throws {
        try check(outDim: 96, inDim: 1_024, tokens: 1)
    }

    /// The rule that decides the dispatch: every published dense projection except
    /// `attn.wq_b` is wide enough to amortize the reduction barrier, and the two
    /// shapes that lose keep the one-thread form.
    func testTheFormRuleCoversThePublishedDenseShapes() {
        let published = [
            ("attn.wo_b", 5_120, 8_192),
            ("attn.wo_a band", 1_024, 4_096),
            ("attn.wq_a", 1_280, 5_120),
            ("attn.wkv", 512, 5_120),
            ("shared_experts.w1", 2_304, 5_120),
            ("shared_experts.w2", 5_120, 2_304)
        ]
        for (name, outDim, inDim) in published {
            XCTAssertTrue(
                DeepSeekV41DevicePlan.fp8GemvWantsWideGrid(
                    outDim: outDim, inDim: inDim, rows: 1
                ),
                "\(name) should take the coalesced form"
            )
        }
        // 40 blocks against 64 lanes: the one-thread form measured 1.07-1.6x faster
        // here, at one position and at a block of eight.
        XCTAssertFalse(
            DeepSeekV41DevicePlan.fp8GemvWantsWideGrid(outDim: 32_768, inDim: 1_280, rows: 1),
            "attn.wq_b has fewer blocks than lanes and keeps the one-thread form"
        )
        XCTAssertFalse(
            DeepSeekV41DevicePlan.fp8GemvWantsWideGrid(outDim: 32_768, inDim: 1_280, rows: 8),
            "attn.wq_b loses the coalesced form at a block width too"
        )
        XCTAssertFalse(
            DeepSeekV41DevicePlan.fp8GemvWantsWideGrid(outDim: 16, inDim: 512, rows: 1),
            "16 rows over 512 columns is too little work to pay for a barrier"
        )
    }

    // MARK: Two rows per threadgroup

    /// One threadgroup per (row pair, token), which is the form that halves the
    /// threadgroup count and the input reads per row.
    private func dispatchPair(
        pipeline: MTLComputePipelineState,
        queue: MTLCommandQueue,
        weight: MTLBuffer,
        scale: MTLBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        inDim: Int,
        outDim: Int,
        tokens: Int,
        inputRowStride: Int,
        outputRowStride: Int,
        lanes: Int
    ) throws {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            XCTFail("no command buffer")
            return
        }
        var weightOffset = UInt64(0)
        var scaleOffset = UInt64(0)
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)
        var inStride = UInt32(inputRowStride)
        var outStride = UInt32(outputRowStride)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&weightOffset, length: 8, index: 4)
        encoder.setBytes(&scaleOffset, length: 8, index: 5)
        encoder.setBytes(&inDimension, length: 4, index: 6)
        encoder.setBytes(&outDimension, length: 4, index: 7)
        encoder.setBytes(&inStride, length: 4, index: 8)
        encoder.setBytes(&outStride, length: 4, index: 9)
        // Two floats per SIMD group of partials, which is the most a 64-lane
        // threadgroup can need.
        encoder.setThreadgroupMemoryLength(
            2 * 32 * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: ((outDim + 1) / 2) * tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)
    }

    /// The pair form against the host rule at both lane shapes.
    private func checkPair(
        outDim: Int,
        inDim: Int,
        tokens: Int,
        inputRowStride: Int? = nil,
        outputRowStride: Int? = nil
    ) throws {
        let (device, library, queue) = try environment()
        guard let pairFunction = library.makeFunction(name: DeepSeekV41Kernels.fp8GemvRows2),
              let pair = try? device.makeComputePipelineState(function: pairFunction) else {
            XCTFail("the pair FP8 GEMV kernel is not in the default library")
            return
        }
        let inStride = inputRowStride ?? inDim
        let outStride = outputRowStride ?? outDim
        let blocks = inDim / 32
        let weight = weightBytes(outDim * inDim)
        // The checkpoint's scale grid is 32x32, so a row count that is not a multiple
        // of 32 has a partial group whose scale row must exist for the host rule and
        // the kernel to read. Every published shape is a multiple of 32; padding here
        // is what lets the odd-row case run at all.
        let scaleGroups = (outDim + 31) / 32
        let scales = scaleBytes(scaleGroups * blocks)
        let input = values(tokens * inStride)
        guard let weightBuffer = device.makeBuffer(
                  bytes: weight,
                  length: weight.count * MemoryLayout<UInt8>.stride,
                  options: .storageModeShared),
              let scaleBuffer = device.makeBuffer(
                  bytes: scales,
                  length: scales.count * MemoryLayout<UInt8>.stride,
                  options: .storageModeShared),
              let inputBuffer = device.makeBuffer(
                  bytes: input,
                  length: input.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared) else {
            XCTFail("the FP8 GEMV buffers did not allocate")
            return
        }

        let expected = reference(
            weight: weight, scales: scales, inDim: inDim, outDim: outDim, tokens: tokens,
            input: input, inputRowStride: inStride, outputRowStride: outStride
        )
        let magnitude = max(1, expected.map { abs($0) }.max() ?? 1)

        for lanes in [32, 64] {
            guard let output = device.makeBuffer(
                length: tokens * outStride * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
                XCTFail("no pair output buffer")
                return
            }
            memset(output.contents(), 0, output.length)
            try dispatchPair(
                pipeline: pair, queue: queue, weight: weightBuffer, scale: scaleBuffer,
                input: inputBuffer, output: output, inDim: inDim, outDim: outDim,
                tokens: tokens, inputRowStride: inStride, outputRowStride: outStride,
                lanes: lanes
            )
            let produced = floats(output, count: tokens * outStride)
            for token in 0..<tokens {
                for row in 0..<outDim {
                    let at = token * outStride + row
                    XCTAssertLessThan(
                        abs(produced[at] - expected[at]), magnitude * 1e-4,
                        "pair form (lanes \(lanes)) drifted at row \(row) of \(outDim)x\(inDim)"
                    )
                }
            }
        }
    }

    func testThePairFormMatchesTheHostRuleAtAPublishedWidth() throws {
        try checkPair(outDim: 96, inDim: 1_280, tokens: 1)
    }

    /// An odd row count leaves the last pair with no second row: the form must write
    /// that row once and not read material for a row past the buffer.
    func testThePairFormHandlesAnOddRowCount() throws {
        try checkPair(outDim: 2_305, inDim: 1_024, tokens: 2)
    }

    func testThePairFormHonoursPositionStrides() throws {
        try checkPair(
            outDim: 64, inDim: 2_304, tokens: 3,
            inputRowStride: 2_560, outputRowStride: 96
        )
    }
}

/// The shared expert's two FP8 kernels, in the one-thread form the graph used to run
/// and the coalesced form `#61` wired.
///
/// `#60` attributed a decoded position's device time and found the shared expert the
/// largest single stage at 109-113 GB/s, about a third of what the routed experts
/// read on the same bytes. The cause was the dispatch shape: `dsv41_fp8_block32_gemv`
/// had a coalesced sibling but the FUSED swiglu and the down accumulate did not, so
/// the one projection the model runs on every layer of every token still gave a
/// thread a whole row. This pins the two new kernels against the host rule, at both
/// lane widths, on the published shapes and on shapes that exercise the 32x32 scale
/// grid's partial group.
final class DeepSeekV41FP8SharedExpertKernelTests: XCTestCase {

    private func environment() throws -> (MTLDevice, MTLLibrary, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary() else {
            XCTFail("the default library is missing")
            throw XCTSkip("no default library")
        }
        guard let queue = device.makeCommandQueue() else {
            XCTFail("no compute queue")
            throw XCTSkip("no compute queue")
        }
        return (device, library, queue)
    }

    /// E4M3 bytes whose exponents stay in the finite range, so a decoded value is a
    /// plain number: exponent 0 is subnormal and 15 is NaN/Inf territory.
    private func weightBytes(_ count: Int) -> [UInt8] {
        (0..<count).map { index in
            let sign: UInt8 = (index % 7 == 3) ? 0x80 : 0
            let exponent = UInt8((index / 5) % 9)
            let mantissa = UInt8(index % 8)
            return sign | (exponent << 3) | mantissa
        }
    }

    /// Block scales near 2^0, so a block sum stays in a range where a relative bar
    /// means something.
    private func scaleBytes(_ count: Int) -> [UInt8] {
        (0..<count).map { index in UInt8(124 + (index % 7)) }
    }

    private func values(_ count: Int) -> [Float] {
        (0..<count).map { index in Float((index &* 29 &+ 7) % 17 - 8) / 16 }
    }

    /// The host rule for the fused pair: per 32-value block, the gate and up dots
    /// scaled by that block's E8M0 byte, summed over the row, then the checkpoint's
    /// clip and `silu(gate) * up`. The scale row is `row / 32` because the grid is
    /// 32x32, and rows past `outDim` are skipped.
    private func swigluReference(
        gate: [UInt8],
        up: [UInt8],
        gateScales: [UInt8],
        upScales: [UInt8],
        inDim: Int,
        outDim: Int,
        tokens: Int,
        input: [Float],
        limit: Float
    ) -> [Float] {
        let blocks = max(1, inDim / 32)
        var out = [Float](repeating: 0, count: tokens * outDim)
        for token in 0..<tokens {
            for row in 0..<outDim {
                var gateTotal: Float = 0
                var upTotal: Float = 0
                for block in 0..<blocks {
                    let scaleIndex = (row / 32) * blocks + block
                    let gateScale = DeepSeekV41EngramRowDequantizer.e8m0(gateScales[scaleIndex])
                    let upScale = DeepSeekV41EngramRowDequantizer.e8m0(upScales[scaleIndex])
                    var gateDot: Float = 0
                    var upDot: Float = 0
                    for index in 0..<32 {
                        let at = row * inDim + block * 32 + index
                        let x = input[token * inDim + block * 32 + index]
                        gateDot += DeepSeekV41EngramRowDequantizer.e4m3(gate[at]) * x
                        upDot += DeepSeekV41EngramRowDequantizer.e4m3(up[at]) * x
                    }
                    gateTotal += gateDot * gateScale
                    upTotal += upDot * upScale
                }
                var gateValue = gateTotal
                var upValue = upTotal
                if limit > 0 {
                    gateValue = min(gateValue, limit)
                    upValue = min(max(upValue, -limit), limit)
                }
                out[token * outDim + row] = gateValue / (1 + exp(-gateValue)) * upValue
            }
        }
        return out
    }

    /// The host rule for the down accumulate: the same block dot, then
    /// `output[row] += routingWeight * total`.
    private func downReference(
        weight: [UInt8],
        scales: [UInt8],
        inDim: Int,
        outDim: Int,
        activation: [Float],
        routing: Float
    ) -> [Float] {
        let blocks = max(1, inDim / 32)
        var out = [Float](repeating: 0, count: outDim)
        for row in 0..<outDim {
            var total: Float = 0
            for block in 0..<blocks {
                let scale = DeepSeekV41EngramRowDequantizer.e8m0(
                    scales[(row / 32) * blocks + block]
                )
                var blockDot: Float = 0
                for index in 0..<32 {
                    let at = row * inDim + block * 32 + index
                    blockDot += DeepSeekV41EngramRowDequantizer.e4m3(weight[at])
                        * activation[block * 32 + index]
                }
                total += blockDot * scale
            }
            out[row] = routing * total
        }
        return out
    }

    private func buffer(_ device: MTLDevice, bytes: [UInt8]) -> MTLBuffer? {
        device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// One shape's fused pair through the host rule, the one-thread form and the
    /// coalesced form at both lane widths. `outDim` is allowed to be off the 32-row
    /// scale boundary, which is the case a lane's scale-index arithmetic gets wrong.
    /// Returns the coalesced output so a caller can compare two limits.
    @discardableResult
    private func checkSwiGLU(
        outDim: Int,
        inDim: Int,
        tokens: Int = 1,
        limit: Float = 0
    ) throws -> [Float] {
        let (device, library, queue) = try environment()
        guard let perRowFunction = library.makeFunction(name: DeepSeekV41Kernels.fp8SwiGLU),
              let rowsFunction = library.makeFunction(name: DeepSeekV41Kernels.fp8SwiGLURows),
              let perRow = try? device.makeComputePipelineState(function: perRowFunction),
              let rows = try? device.makeComputePipelineState(function: rowsFunction) else {
            XCTFail("the FP8 shared-expert kernels are not in the default library")
            return []
        }
        let blocks = inDim / 32
        // A row count off the 32-row grid needs its partial scale group published, or
        // the host rule and both kernels read past the companion.
        let scaleGroups = (outDim + 31) / 32
        let gate = weightBytes(outDim * inDim)
        let up = weightBytes(outDim * inDim).reversed()
        let upBytes = Array(up)
        let gateScales = scaleBytes(scaleGroups * blocks)
        let upScales = scaleBytes(scaleGroups * blocks)
        let input = values(tokens * inDim)

        guard let gateBuffer = buffer(device, bytes: gate),
              let upBuffer = buffer(device, bytes: upBytes),
              let gateScaleBuffer = buffer(device, bytes: gateScales),
              let upScaleBuffer = buffer(device, bytes: upScales),
              let inputBuffer = device.makeBuffer(
                  bytes: input,
                  length: input.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared) else {
            XCTFail("the shared-expert buffers did not allocate")
            return []
        }

        let expected = swigluReference(
            gate: gate, up: upBytes, gateScales: gateScales, upScales: upScales,
            inDim: inDim, outDim: outDim, tokens: tokens, input: input, limit: limit
        )
        let magnitude = max(1, expected.map { abs($0) }.max() ?? 1)

        func dispatch(
            pipeline: MTLComputePipelineState,
            coalesced: Bool,
            lanes: Int
        ) -> [Float] {
            guard let output = device.makeBuffer(
                length: tokens * outDim * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else { return [] }
            memset(output.contents(), 0, output.length)
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                XCTFail("no command buffer")
                return []
            }
            var gateWeightOffset = UInt64(0)
            var gateScaleOffset = UInt64(0)
            var upWeightOffset = UInt64(0)
            var upScaleOffset = UInt64(0)
            var inDimension = UInt32(inDim)
            var outDimension = UInt32(outDim)
            var swiglu = limit
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gateBuffer, offset: 0, index: 0)
            encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(upBuffer, offset: 0, index: 2)
            encoder.setBuffer(upScaleBuffer, offset: 0, index: 3)
            encoder.setBuffer(inputBuffer, offset: 0, index: 4)
            encoder.setBuffer(output, offset: 0, index: 5)
            encoder.setBytes(&gateWeightOffset, length: 8, index: 6)
            encoder.setBytes(&gateScaleOffset, length: 8, index: 7)
            encoder.setBytes(&upWeightOffset, length: 8, index: 8)
            encoder.setBytes(&upScaleOffset, length: 8, index: 9)
            encoder.setBytes(&inDimension, length: 4, index: 10)
            encoder.setBytes(&outDimension, length: 4, index: 11)
            encoder.setBytes(&swiglu, length: 4, index: 12)
            if coalesced {
                encoder.setThreadgroupMemoryLength(
                    2 * 32 * MemoryLayout<Float>.stride, index: 0
                )
                encoder.dispatchThreadgroups(
                    MTLSize(width: outDim * tokens, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
                )
            } else {
                encoder.dispatchThreads(
                    MTLSize(width: outDim, height: tokens, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(outDim, 64), height: 1, depth: 1)
                )
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            XCTAssertNil(commandBuffer.error)
            return floats(output, count: tokens * outDim)
        }

        let oneThread = dispatch(pipeline: perRow, coalesced: false, lanes: 64)
        var lastProduced: [Float] = []
        for lanes in [32, 64] {
            let produced = dispatch(pipeline: rows, coalesced: true, lanes: lanes)
            lastProduced = produced
            for index in 0..<(tokens * outDim) {
                XCTAssertLessThan(
                    abs(produced[index] - expected[index]), magnitude * 1e-4,
                    "coalesced swiglu (lanes \(lanes)) drifted at \(index) of "
                        + "\(outDim)x\(inDim)"
                )
                XCTAssertLessThan(
                    abs(oneThread[index] - produced[index]), magnitude * 1e-4,
                    "the two forms disagree at \(index) of \(outDim)x\(inDim)"
                )
            }
        }
        return lastProduced
    }

    /// The same for the down accumulate, from a zeroed output so the `+=` is the
    /// whole result.
    private func checkDown(outDim: Int, inDim: Int, routing: Float = 0.75) throws {
        let (device, library, queue) = try environment()
        guard let perRowFunction = library.makeFunction(name: DeepSeekV41Kernels.fp8DownAccumulate),
              let rowsFunction = library.makeFunction(name: DeepSeekV41Kernels.fp8DownAccumulateRows),
              let perRow = try? device.makeComputePipelineState(function: perRowFunction),
              let rows = try? device.makeComputePipelineState(function: rowsFunction) else {
            XCTFail("the FP8 shared-expert kernels are not in the default library")
            return
        }
        let blocks = inDim / 32
        let scaleGroups = (outDim + 31) / 32
        let weight = weightBytes(outDim * inDim)
        let scales = scaleBytes(scaleGroups * blocks)
        let activation = values(inDim)

        guard let weightBuffer = buffer(device, bytes: weight),
              let scaleBuffer = buffer(device, bytes: scales),
              let activationBuffer = device.makeBuffer(
                  bytes: activation,
                  length: activation.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared) else {
            XCTFail("the shared-expert buffers did not allocate")
            return
        }

        let expected = downReference(
            weight: weight, scales: scales, inDim: inDim, outDim: outDim,
            activation: activation, routing: routing
        )
        let magnitude = max(1, expected.map { abs($0) }.max() ?? 1)

        func dispatch(
            pipeline: MTLComputePipelineState,
            coalesced: Bool,
            lanes: Int
        ) -> [Float] {
            guard let output = device.makeBuffer(
                length: outDim * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else { return [] }
            memset(output.contents(), 0, output.length)
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                XCTFail("no command buffer")
                return []
            }
            var weightOffset = UInt64(0)
            var scaleOffset = UInt64(0)
            var inDimension = UInt32(inDim)
            var outDimension = UInt32(outDim)
            var routingWeight = routing
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(activationBuffer, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            encoder.setBytes(&weightOffset, length: 8, index: 4)
            encoder.setBytes(&scaleOffset, length: 8, index: 5)
            encoder.setBytes(&inDimension, length: 4, index: 6)
            encoder.setBytes(&outDimension, length: 4, index: 7)
            encoder.setBytes(&routingWeight, length: 4, index: 8)
            if coalesced {
                encoder.setThreadgroupMemoryLength(
                    32 * MemoryLayout<Float>.stride, index: 0
                )
                encoder.dispatchThreadgroups(
                    MTLSize(width: outDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
                )
            } else {
                encoder.dispatchThreads(
                    MTLSize(width: outDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(outDim, 64), height: 1, depth: 1)
                )
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            XCTAssertNil(commandBuffer.error)
            return floats(output, count: outDim)
        }

        let oneThread = dispatch(pipeline: perRow, coalesced: false, lanes: 64)
        for lanes in [32, 64] {
            let produced = dispatch(pipeline: rows, coalesced: true, lanes: lanes)
            for index in 0..<outDim {
                XCTAssertLessThan(
                    abs(produced[index] - expected[index]), magnitude * 1e-4,
                    "coalesced down (lanes \(lanes)) drifted at row \(index) of "
                        + "\(outDim)x\(inDim)"
                )
                XCTAssertLessThan(
                    abs(oneThread[index] - produced[index]), magnitude * 1e-4,
                    "the two down forms disagree at row \(index)"
                )
            }
        }
    }

    func testTheCoalescedFusedPairMatchesTheHostRuleAtThePublishedShape() throws {
        try checkSwiGLU(outDim: 2_304, inDim: 5_120)
    }

    func testTheCoalescedDownMatchesTheHostRuleAtThePublishedShape() throws {
        try checkDown(outDim: 5_120, inDim: 2_304)
    }

    /// A row count off the 32-row scale boundary, and a row shorter than a lane's
    /// block count (1,024 values is 32 blocks against 64 lanes), so the partial
    /// group and the idle lanes are both exercised.
    func testTheCoalescedPairHandlesAPartialScaleGroupAndAShortRow() throws {
        try checkSwiGLU(outDim: 65, inDim: 1_024)
        try checkDown(outDim: 65, inDim: 1_024)
    }

    /// The fused kernel carries the token axis: the flat grid's token is
    /// `groupIdx / outDim`, so a row-major mix-up would give token 1 the rows of
    /// token 0.
    func testTheCoalescedFusedPairCarriesTheTokenAxis() throws {
        try checkSwiGLU(outDim: 96, inDim: 1_280, tokens: 3)
    }

    /// `swiglu_limit` is part of the trained function, so it has to survive the move:
    /// the clipped pair must match the host rule AND differ from the unlimited run, or
    /// a kernel that dropped the clip would pass.
    func testTheCoalescedFusedPairAppliesTheSwiGLULimit() throws {
        let unlimited = try checkSwiGLU(outDim: 96, inDim: 1_280)
        let clipped = try checkSwiGLU(outDim: 96, inDim: 1_280, limit: 0.25)
        let changed = zip(unlimited, clipped).filter { abs($0 - $1) > 1e-6 }.count
        XCTAssertGreaterThan(
            changed, unlimited.count / 4,
            "a 0.25 limit should clip most of a 40-block row, not \(changed) values"
        )
    }

    /// The plan rule sends both published shared-expert shapes to the coalesced form.
    func testTheRuleSendsThePublishedSharedExpertToTheCoalescedForm() {
        XCTAssertTrue(
            DeepSeekV41DevicePlan.fp8GemvWantsWideGrid(outDim: 2_304, inDim: 5_120, rows: 1),
            "the shared gate+up is 160 blocks over 64 lanes"
        )
        XCTAssertTrue(
            DeepSeekV41DevicePlan.fp8GemvWantsWideGrid(outDim: 5_120, inDim: 2_304, rows: 1),
            "the shared down is 72 blocks over 64 lanes"
        )
    }

    /// The slot forms project a BLOCK of positions from one read of the row, which is
    /// what a prefill block needs from the shared expert: its weight is the same for
    /// every position, and `#72` measured the per-position form reading it `width`
    /// times at the machine's ceiling, i.e. all of it redundant
    /// (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #72").
    ///
    /// What they must reproduce is the per-slot `_rows` form: the same bytes, the same
    /// arithmetic, one pass over the weight instead of `slotCount`. Five slots, so no
    /// form can assume a width of four or eight, and the published shapes, so the
    /// comparison is at the width that ships.
    func testTheSlotFormsProjectEveryPositionFromOneWeightRead() throws {
        let (device, library, queue) = try environment()
        guard
            let rowsFunction = library.makeFunction(name: DeepSeekV41Kernels.fp8SwiGLURows),
            let slotsFunction = library.makeFunction(
                name: DeepSeekV41Kernels.fp8SwiGLURowsSlots
            ),
            let downRowsFunction = library.makeFunction(
                name: DeepSeekV41Kernels.fp8DownAccumulateRows
            ),
            let downSlotsFunction = library.makeFunction(
                name: DeepSeekV41Kernels.fp8DownAccumulateRowsSlots
            ),
            let rowsPipeline = try? device.makeComputePipelineState(function: rowsFunction),
            let slotsPipeline = try? device.makeComputePipelineState(function: slotsFunction),
            let downRowsPipeline = try? device.makeComputePipelineState(
                function: downRowsFunction
            ),
            let downSlotsPipeline = try? device.makeComputePipelineState(
                function: downSlotsFunction
            )
        else {
            XCTFail("the shared-expert slot kernels are not in the default library")
            return
        }
        let slots = 5
        let floatStride = MemoryLayout<Float>.stride

        /// Runs `body` in one command buffer on `queue` and waits for it.
        func run(_ body: (MTLComputeCommandEncoder) -> Void) {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                XCTFail("no command buffer")
                return
            }
            body(encoder)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            XCTAssertNil(commandBuffer.error)
        }

        func check(outDim: Int, inDim: Int, againstHost: Bool) throws {
            let blocks = inDim / 32
            let scaleGroups = (outDim + 31) / 32
            let gate = weightBytes(outDim * inDim)
            let upBytes = Array(weightBytes(outDim * inDim).reversed())
            let gateScales = scaleBytes(scaleGroups * blocks)
            let upScales = scaleBytes(scaleGroups * blocks)
            let inputs = values(slots * inDim)
            guard let gateBuffer = buffer(device, bytes: gate),
                  let upBuffer = buffer(device, bytes: upBytes),
                  let gateScaleBuffer = buffer(device, bytes: gateScales),
                  let upScaleBuffer = buffer(device, bytes: upScales),
                  let inputBuffer = device.makeBuffer(
                      bytes: inputs,
                      length: inputs.count * floatStride,
                      options: .storageModeShared
                  ),
                  let perSlot = device.makeBuffer(
                      length: slots * outDim * floatStride,
                      options: .storageModeShared
                  ),
                  let batched = device.makeBuffer(
                      length: slots * outDim * floatStride,
                      options: .storageModeShared
                  ) else {
                XCTFail("the shared-expert buffers did not allocate")
                return
            }
            memset(perSlot.contents(), 0, perSlot.length)
            memset(batched.contents(), 0, batched.length)

            let bound: (MTLComputeCommandEncoder) -> Void = { encoder in
                var gateWeightOffset = UInt64(0)
                var gateScaleOffset = UInt64(0)
                var upWeightOffset = UInt64(0)
                var upScaleOffset = UInt64(0)
                var inDimension = UInt32(inDim)
                var outDimension = UInt32(outDim)
                var limit: Float = 0.5
                encoder.setBuffer(gateBuffer, offset: 0, index: 0)
                encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(upBuffer, offset: 0, index: 2)
                encoder.setBuffer(upScaleBuffer, offset: 0, index: 3)
                encoder.setBytes(&gateWeightOffset, length: 8, index: 6)
                encoder.setBytes(&gateScaleOffset, length: 8, index: 7)
                encoder.setBytes(&upWeightOffset, length: 8, index: 8)
                encoder.setBytes(&upScaleOffset, length: 8, index: 9)
                encoder.setBytes(&inDimension, length: 4, index: 10)
                encoder.setBytes(&outDimension, length: 4, index: 11)
                encoder.setBytes(&limit, length: 4, index: 12)
            }

            // The reference: one threadgroup per row, one dispatch per position.
            run { encoder in
                for slot in 0..<slots {
                    encoder.setComputePipelineState(rowsPipeline)
                    bound(encoder)
                    encoder.setBuffer(
                        inputBuffer, offset: slot * inDim * floatStride, index: 4
                    )
                    encoder.setBuffer(
                        perSlot, offset: slot * outDim * floatStride, index: 5
                    )
                    encoder.setThreadgroupMemoryLength(2 * 32 * floatStride, index: 0)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: outDim, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                    )
                }
            }
            // The block form: one dispatch, the position loop inside the threadgroup.
            run { encoder in
                encoder.setComputePipelineState(slotsPipeline)
                bound(encoder)
                encoder.setBuffer(inputBuffer, offset: 0, index: 4)
                encoder.setBuffer(batched, offset: 0, index: 5)
                var slotCount = UInt32(slots)
                encoder.setBytes(&slotCount, length: 4, index: 13)
                encoder.setThreadgroupMemoryLength(2 * 32 * floatStride, index: 0)
                encoder.dispatchThreadgroups(
                    MTLSize(width: outDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
            let reference = floats(perSlot, count: slots * outDim)
            let produced = floats(batched, count: slots * outDim)
            let magnitude = max(1, reference.map { abs($0) }.max() ?? 1)
            for index in 0..<(slots * outDim) {
                XCTAssertLessThan(
                    abs(produced[index] - reference[index]), magnitude * 1e-4,
                    "the slot fused pair drifted at \(index) of \(outDim)x\(inDim)"
                )
            }
            // Every position must land in its OWN row: a kernel that wrote slot 0 five
            // times would pass a comparison against a reference that did the same. The
            // clip saturates about half of a row at these fixture magnitudes, so this is
            // "some rows move" rather than "most", and the equality above is what
            // carries the correctness.
            for slot in 1..<slots {
                let moved = (0..<outDim).filter {
                    abs(produced[slot * outDim + $0] - produced[$0]) > 1e-5
                }.count
                XCTAssertGreaterThan(
                    moved, outDim / 4,
                    "slot \(slot) of \(outDim)x\(inDim) matched slot 0"
                )
            }
            if againstHost {
                let expected = swigluReference(
                    gate: gate, up: upBytes, gateScales: gateScales, upScales: upScales,
                    inDim: inDim, outDim: outDim, tokens: slots, input: inputs, limit: 0.5
                )
                for index in 0..<(slots * outDim) {
                    XCTAssertLessThan(
                        abs(produced[index] - expected[index]), magnitude * 1e-4,
                        "the slot fused pair drifted from the host rule at \(index)"
                    )
                }
            }

            // The same for the down projection, which accumulates into the slot's row.
            let weight = weightBytes(outDim * inDim)
            let scales = scaleBytes(scaleGroups * blocks)
            let activations = values(slots * inDim)
            guard let weightBuffer = buffer(device, bytes: weight),
                  let scaleBuffer = buffer(device, bytes: scales),
                  let activationBuffer = device.makeBuffer(
                      bytes: activations,
                      length: activations.count * floatStride,
                      options: .storageModeShared
                  ),
                  let downPerSlot = device.makeBuffer(
                      length: slots * outDim * floatStride,
                      options: .storageModeShared
                  ),
                  let downBatched = device.makeBuffer(
                      length: slots * outDim * floatStride,
                      options: .storageModeShared
                  ) else {
                XCTFail("the down buffers did not allocate")
                return
            }
            memset(downPerSlot.contents(), 0, downPerSlot.length)
            memset(downBatched.contents(), 0, downBatched.length)

            let downBound: (MTLComputeCommandEncoder) -> Void = { encoder in
                var weightOffset = UInt64(0)
                var scaleOffset = UInt64(0)
                var inDimension = UInt32(inDim)
                var outDimension = UInt32(outDim)
                var routing: Float = 0.75
                encoder.setBuffer(weightBuffer, offset: 0, index: 0)
                encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
                encoder.setBytes(&weightOffset, length: 8, index: 4)
                encoder.setBytes(&scaleOffset, length: 8, index: 5)
                encoder.setBytes(&inDimension, length: 4, index: 6)
                encoder.setBytes(&outDimension, length: 4, index: 7)
                encoder.setBytes(&routing, length: 4, index: 8)
            }
            run { encoder in
                for slot in 0..<slots {
                    encoder.setComputePipelineState(downRowsPipeline)
                    downBound(encoder)
                    encoder.setBuffer(
                        activationBuffer, offset: slot * inDim * floatStride, index: 2
                    )
                    encoder.setBuffer(
                        downPerSlot, offset: slot * outDim * floatStride, index: 3
                    )
                    encoder.setThreadgroupMemoryLength(32 * floatStride, index: 0)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: outDim, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                    )
                }
            }
            run { encoder in
                encoder.setComputePipelineState(downSlotsPipeline)
                downBound(encoder)
                encoder.setBuffer(activationBuffer, offset: 0, index: 2)
                encoder.setBuffer(downBatched, offset: 0, index: 3)
                var slotCount = UInt32(slots)
                encoder.setBytes(&slotCount, length: 4, index: 9)
                encoder.setThreadgroupMemoryLength(32 * floatStride, index: 0)
                encoder.dispatchThreadgroups(
                    MTLSize(width: outDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }
            let downExpected = floats(downPerSlot, count: slots * outDim)
            let downProduced = floats(downBatched, count: slots * outDim)
            let downMagnitude = max(1, downExpected.map { abs($0) }.max() ?? 1)
            for index in 0..<(slots * outDim) {
                XCTAssertLessThan(
                    abs(downProduced[index] - downExpected[index]),
                    downMagnitude * 1e-4,
                    "the slot down projection drifted at \(index) of \(outDim)x\(inDim)"
                )
            }
            for slot in 1..<slots {
                let moved = (0..<outDim).filter {
                    abs(downProduced[slot * outDim + $0] - downProduced[$0]) > 1e-5
                }.count
                XCTAssertGreaterThan(
                    moved, outDim / 4,
                    "down slot \(slot) of \(outDim)x\(inDim) matched slot 0"
                )
            }
            if againstHost {
                for slot in 0..<slots {
                    let row = Array(activations[(slot * inDim)..<((slot + 1) * inDim)])
                    let expected = downReference(
                        weight: weight,
                        scales: scales,
                        inDim: inDim,
                        outDim: outDim,
                        activation: row,
                        routing: 0.75
                    )
                    for index in 0..<outDim {
                        XCTAssertLessThan(
                            abs(downProduced[slot * outDim + index] - expected[index]),
                            downMagnitude * 1e-4,
                            "the slot down projection drifted from the host rule at "
                                + "slot \(slot) row \(index)"
                        )
                    }
                }
            }
        }

        try check(outDim: 96, inDim: 1_280, againstHost: true)
        try check(outDim: 2_304, inDim: 5_120, againstHost: false)
        try check(outDim: 5_120, inDim: 2_304, againstHost: false)
    }
}

/// What a committed command buffer costs to START once the queue has gone idle.
///
/// The walk's mixture drain is ~2.5 ms/layer against a ~0.7 ms device window, and
/// the late start (the gap between the host beginning to wait and the buffer's own
/// window opening) carries most of the difference. This separates the two possible
/// causes: what the queue costs when it is already fed, and what an idle gap adds
/// before the next buffer is picked up. No model, no metrics, just the driver.
final class DeepSeekV41CommandStartCostTests: XCTestCase {

    func testWhatAnIdleGapCostsACommandBuffer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary() else {
            throw XCTSkip("no default library")
        }
        guard let function = library.makeFunction(name: "clear_vector_f32") else {
            throw XCTSkip("clear_vector_f32 is missing")
        }
        guard let queue = device.makeCommandQueue() else {
            throw XCTSkip("no compute queue")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let target = device.makeBuffer(length: 4096, options: .storageModeShared) else {
            throw XCTSkip("no scratch buffer")
        }

        /// One buffer, committed after the queue has been idle for `idleMilliseconds`,
        /// reporting how long after the wait began its window opened and how long the
        /// wait itself took.
        func measure(idleMilliseconds: Double, attempts: Int = 41) -> (start: Double, drain: Double) {
            var starts: [Double] = []
            var drains: [Double] = []
            for _ in 0..<attempts {
                guard let buffer = queue.makeCommandBuffer(),
                      let encoder = buffer.makeComputeCommandEncoder() else { continue }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(target, offset: 0, index: 0)
                encoder.dispatchThreads(
                    MTLSize(width: 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1)
                )
                encoder.endEncoding()
                if idleMilliseconds > 0 {
                    Thread.sleep(forTimeInterval: idleMilliseconds / 1000)
                }
                buffer.commit()
                let begin = DispatchTime.now().uptimeNanoseconds
                buffer.waitUntilCompleted()
                let end = DispatchTime.now().uptimeNanoseconds
                let windowStart = Double(buffer.gpuStartTime) * 1_000_000_000
                starts.append(max(0, (windowStart - Double(begin)) / 1_000_000))
                drains.append(Double(end - begin) / 1_000_000)
            }
            func median(_ values: [Double]) -> Double {
                let sorted = values.sorted()
                return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
            }
            return (median(starts), median(drains))
        }

        // Warm the pipeline up so the first arm is not paying for shader setup.
        _ = measure(idleMilliseconds: 0, attempts: 5)

        let fed = measure(idleMilliseconds: 0)
        let shortGap = measure(idleMilliseconds: 2)
        let longGap = measure(idleMilliseconds: 10)
        print(
            String(
                format: "🔎 [dsv41 command start] fed: late start %.3f ms, drain %.3f ms; "
                    + "2 ms gap: late start %.3f ms, drain %.3f ms; "
                    + "10 ms gap: late start %.3f ms, drain %.3f ms",
                fed.start, fed.drain,
                shortGap.start, shortGap.drain,
                longGap.start, longGap.drain
            )
        )
    }

    /// Whether an event-gated commit lets the GPU pick the buffer up WHILE the host is
    /// still working, instead of after it stops. It does not: the answer is no.
    ///
    /// The walk commits its expert half once a fill has landed in the arena: the queue
    /// has been empty for the length of the fill, and the buffer's window opens ~0.29 ms
    /// after the commit (`DeepSeekV41WalkTiming.mixtureLateStartNanoseconds`, `#64`).
    /// If the buffer could be committed BEFORE the fill and gated on an event the fill
    /// signals, that pickup would be paid under the fill instead of after it, worth
    /// `mixture late start` per position — but only if the driver submits a gated buffer
    /// early. A gate that defers submission defeats itself (`Benchmarks/PERF_FINDINGS.md`,
    /// "DSV4.1 #65").
    ///
    /// Two arms at each gap: `idle` is the walk's shape today (queue empty, host sleeps,
    /// then commits), `gated` commits first and signals when the same sleep ends. Both
    /// report the delay from the moment the DATA was ready — the commit for `idle`, the
    /// signal for `gated` — so the two are comparable at the only instant that matters.
    ///
    /// **The gated arm does not start earlier**: at a 0.6 ms gap, 1 dispatch costs 0.122
    /// (idle) against 0.112 (gated), 20 dispatches 0.161 against 0.163, and 64
    /// dispatches 0.155 against 0.144. The gate changes WHICH call hands the work over,
    /// not how long the machine takes to begin it, which is why the walk's in-situ arm
    /// (the same experiment, on the real graph) is also a wash.
    ///
    /// Two traps this arm hit, both of which made a gate look better than it is: a wait
    /// encoded after the dispatches does not hold them, and a shared event's value never
    /// goes back down, so an arm that restarts its values at 1 finds its gate already
    /// satisfied and measures the UNGATED path under the gated arm's name. `dispatches`
    /// is swept because the walk's half is ~20, and a one-dispatch arm cannot speak for it.
    func testWhatAnEventGateSavesOnACommitIntoAnEmptyQueue() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary() else {
            throw XCTSkip("no default library")
        }
        guard let function = library.makeFunction(name: "clear_vector_f32") else {
            throw XCTSkip("clear_vector_f32 is missing")
        }
        guard let queue = device.makeCommandQueue() else {
            throw XCTSkip("no compute queue")
        }
        guard let event = device.makeSharedEvent() else {
            throw XCTSkip("no shared event")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let target = device.makeBuffer(length: 4096, options: .storageModeShared) else {
            throw XCTSkip("no scratch buffer")
        }

        /// `dispatches` copies of one tiny kernel, separated by a barrier the way a
        /// real half's stages are. The walk's expert half is ~20 dispatches over ~200
        /// bindings, so a one-dispatch arm cannot answer whether a gate helps a buffer
        /// that takes the driver real work to hand over.
        func encode(into buffer: MTLCommandBuffer, dispatches: Int = 1) {
            guard let encoder = buffer.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(target, offset: 0, index: 0)
            for _ in 0..<max(1, dispatches) {
                encoder.dispatchThreads(
                    MTLSize(width: 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1)
                )
                encoder.memoryBarrier(scope: .buffers)
            }
            encoder.endEncoding()
        }

        /// One counter for the WHOLE test, because a shared event's value never goes
        /// back down: an arm that restarted at 1 would find its gate already satisfied
        /// and measure the ungated path under the gated arm's name.
        var nextGateValue: UInt64 = 0

        /// Median of `attempts` runs of one arm. `readyToWindow` is measured from the
        /// commit (idle arm) or the signal (gated arm) to the buffer's own window
        /// opening; `readyToDone` from the same instant to the host leaving the wait, so
        /// the host's own sleep is inside neither.
        func measure(
            gapMilliseconds: Double,
            gated: Bool,
            dispatches: Int = 1,
            attempts: Int = 41
        ) -> (readyToWindow: Double, readyToDone: Double, window: Double) {
            var starts: [Double] = []
            var drains: [Double] = []
            var windows: [Double] = []
            for _ in 0..<attempts {
                guard let buffer = queue.makeCommandBuffer() else { continue }
                nextGateValue += 1
                let value = nextGateValue
                if gated { buffer.encodeWaitForEvent(event, value: value) }
                encode(into: buffer, dispatches: dispatches)
                if !gated, gapMilliseconds > 0 {
                    // The walk's shape: the host has been reading (`pread`) and the
                    // queue has been idle for the whole fill, so the commit is what
                    // hands the driver work after a gap.
                    Thread.sleep(forTimeInterval: gapMilliseconds / 1000)
                }
                buffer.commit()
                if gated, gapMilliseconds > 0 {
                    Thread.sleep(forTimeInterval: gapMilliseconds / 1000)
                }
                let ready = DispatchTime.now().uptimeNanoseconds
                if gated { event.signaledValue = value }
                buffer.waitUntilCompleted()
                let returned = DispatchTime.now().uptimeNanoseconds
                let windowStart = Double(buffer.gpuStartTime) * 1_000_000_000
                let windowEnd = Double(buffer.gpuEndTime) * 1_000_000_000
                starts.append(max(0, (windowStart - Double(ready)) / 1_000_000))
                drains.append(Double(returned - ready) / 1_000_000)
                if windowEnd > windowStart {
                    windows.append((windowEnd - windowStart) / 1_000_000)
                }
            }
            func median(_ values: [Double]) -> Double {
                let sorted = values.sorted()
                return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
            }
            return (median(starts), median(drains), median(windows))
        }

        // Warm up so the first arm is not paying for shader setup.
        _ = measure(gapMilliseconds: 0, gated: false, attempts: 5)
        _ = measure(gapMilliseconds: 0, gated: true, attempts: 5)

        for dispatches in [1, 20, 64] {
            for gap in [0.0, 0.6] {
                let idle = measure(
                    gapMilliseconds: gap, gated: false, dispatches: dispatches
                )
                let gated = measure(
                    gapMilliseconds: gap, gated: true, dispatches: dispatches
                )
                print(
                    String(
                        format: "🔎 [dsv41 gate] %d dispatch(es), gap %.1f ms: idle "
                            + "ready->window %.3f ms, ready->done %.3f ms, window %.3f ms; "
                            + "gated ready->window %.3f ms, ready->done %.3f ms, "
                            + "window %.3f ms",
                        dispatches,
                        gap,
                        idle.readyToWindow, idle.readyToDone, idle.window,
                        gated.readyToWindow, gated.readyToDone, gated.window
                    )
                )
            }
        }
    }
}

// MARK: - Where a committed buffer's start delay comes from

/// One committed buffer's time, split into the places it can go.
///
/// The walk commits 88 buffers a position and each one starts 2.1-2.9 ms after
/// `commit()` while the GPU has nothing else to do (`Benchmarks/PERF_FINDINGS.md`,
/// "DSV4.1 #47"). That single number admits three different causes, and they do not
/// share a fix, so the probe below reports all four terms for every arm instead of
/// the drain alone.
struct DeepSeekV41CommitSample {
    /// `commit()` returning to the buffer's own device window opening.
    var pickupMilliseconds: Double = 0
    /// The device window itself.
    var windowMilliseconds: Double = 0
    /// The GPU finishing to the host leaving `waitUntilCompleted`. This is the part of
    /// a blocked wait that is notification rather than device work.
    var tailMilliseconds: Double = 0
    /// `commit()` returning to the host leaving `waitUntilCompleted`.
    var drainMilliseconds: Double = 0
    /// `commit()` returning to Metal's `scheduled` handler, nil when it never fired.
    var scheduledMilliseconds: Double?
    /// The device window's own bounds, so consecutive buffers can be lined up: what a
    /// follower pays is the gap from its predecessor's end, not from its own commit.
    var windowStartNanoseconds: UInt64 = 0
    var windowEndNanoseconds: UInt64 = 0

    var hasWindow: Bool { windowMilliseconds > 0 }
}

/// Drives command buffers of chosen shapes through one queue and reports where each
/// one's time went. Nothing here is on a production path.
///
/// A fixed cost per BUFFER is fixed by merging buffers; a cost per DISPATCH is fixed
/// by cutting dispatches; a cost of the QUEUE HAVING GONE EMPTY is fixed by keeping
/// more than one buffer in flight. Only the last two are cheaper than the first, so
/// the arms have to tell them apart before anything is changed.
nonisolated final class DeepSeekV41CommitStartupProbe {
    /// The widest grid any arm dispatches, so every arm writes inside its buffer.
    static let widestGrid = 4_194_304

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    /// A kernel that READS two buffers, because every other arm here writes: if a
    /// submission starts late only for memory the host has just written, a write-only
    /// probe cannot see it.
    private let readPipeline: MTLComputePipelineState
    private let target: MTLBuffer

    static func make() throws -> DeepSeekV41CommitStartupProbe {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary() else {
            throw XCTSkip("no default library")
        }
        guard let function = library.makeFunction(name: "clear_vector_f32") else {
            throw XCTSkip("clear_vector_f32 is missing")
        }
        guard let readFunction = library.makeFunction(name: "vector_add_f32") else {
            throw XCTSkip("vector_add_f32 is missing")
        }
        return try DeepSeekV41CommitStartupProbe(
            device: device, function: function, readFunction: readFunction
        )
    }

    init(device: MTLDevice, function: MTLFunction, readFunction: MTLFunction) throws {
        guard let queue = device.makeCommandQueue() else {
            throw XCTSkip("no compute queue")
        }
        guard let target = device.makeBuffer(
            length: Self.widestGrid * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw XCTSkip("no scratch buffer")
        }
        self.device = device
        self.queue = queue
        self.pipeline = try device.makeComputePipelineState(function: function)
        self.readPipeline = try device.makeComputePipelineState(function: readFunction)
        self.target = target
    }

    /// One submission whose kernels READ `source`, with the option to have the host
    /// write every page of that memory first, the way a `pread` fills an arena slot.
    func readSample(
        source: [MTLBuffer],
        lanes: Int,
        destination: MTLBuffer,
        hostWritesFirst: Bool
    ) -> DeepSeekV41CommitSample? {
        guard !source.isEmpty else { return nil }
        if hostWritesFirst {
            for buffer in source {
                let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
                for page in 0..<max(1, buffer.length / 16_384) { pointer[page * 4_096] = 1 }
            }
        }
        guard let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else { return nil }
        encoder.setComputePipelineState(readPipeline)
        encoder.setBuffer(source[0], offset: 0, index: 0)
        encoder.setBuffer(source[1 % source.count], offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        var dim = UInt32(max(1, min(lanes, source[0].length / 4)))
        encoder.setBytes(&dim, length: 4, index: 3)
        let width = Int(dim)
        encoder.dispatchThreads(
            MTLSize(width: width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(width, readPipeline.maxTotalThreadsPerThreadgroup)),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        buffer.commit()
        let commit = DispatchTime.now().uptimeNanoseconds
        buffer.waitUntilCompleted()
        let returned = DispatchTime.now().uptimeNanoseconds
        return enriched(buffer, commit: commit, returned: returned, stamp: CommitStamp())
    }

    /// A buffer holding `dispatches` copies of the clear kernel over `threads` lanes.
    ///
    /// `pool` mimics the walk, where every dispatch binds its own weight slice out of
    /// one of ~57 shard buffers rather than re-binding a shared scratch vector: what
    /// grows is the command buffer's resource list, not the number of launches.
    ///
    /// `residency` is declared with `useResource`, which puts it on the command
    /// buffer's resource list without giving it a binding slot. That is the one axis a
    /// binding slot cannot measure on this shader (the clear kernel reads buffer 0
    /// only), and it is the axis a real half layer has: dozen of dispatches whose union
    /// is every workspace buffer, gain vector and expert slot the layer touches.
    func makeCommandBuffer(
        dispatches: Int = 1,
        threads: Int = 16,
        pool: [MTLBuffer] = [],
        residency: [MTLBuffer] = [],
        barrier: Bool = false
    ) -> MTLCommandBuffer? {
        guard let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else { return nil }
        encoder.setComputePipelineState(pipeline)
        let grid = MTLSize(width: max(1, min(threads, Self.widestGrid)), height: 1, depth: 1)
        let group = MTLSize(
            width: max(1, min(grid.width, pipeline.maxTotalThreadsPerThreadgroup)),
            height: 1,
            depth: 1
        )
        for resource in residency { encoder.useResource(resource, usage: .read) }
        for index in 0..<max(1, dispatches) {
            encoder.setBuffer(pool.isEmpty ? target : pool[index % pool.count], offset: 0, index: 0)
            encoder.dispatchThreads(grid, threadsPerThreadgroup: group)
            if barrier { encoder.memoryBarrier(scope: .buffers) }
        }
        encoder.endEncoding()
        return buffer
    }

    /// `count` buffers of `bytes` each in shared memory, held by the caller so they stay
    /// alive for the life of an arm.
    func residentPool(count: Int, bytes: Int) -> [MTLBuffer] {
        var pool: [MTLBuffer] = []
        for _ in 0..<max(1, count) {
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                continue
            }
            // Touch one float a page so the allocation is backed before it is measured,
            // otherwise the first arm pays the kernel's own page faults.
            let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
            for page in 0..<max(1, bytes / 16_384) { pointer[page * 4_096] = 0 }
            pool.append(buffer)
        }
        return pool
    }

    /// `count` buffers of `bytes` each wrapped over a sparse temp file with no copy,
    /// which is how the checkpoint's shards reach the GPU. `maps` owns the mappings.
    func mappedPool(count: Int, bytes: Int) -> (pool: [MTLBuffer], maps: [UnsafeMutableRawPointer]) {
        let path = NSTemporaryDirectory() + "dsv41_commit_probe_mapping"
        let fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0o600)
        guard fd >= 0 else { return ([], []) }
        defer { close(fd) }
        guard ftruncate(fd, off_t(bytes)) == 0 else { return ([], []) }
        var pool: [MTLBuffer] = []
        var maps: [UnsafeMutableRawPointer] = []
        for _ in 0..<max(1, count) {
            guard let raw = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0),
                  raw != UnsafeMutableRawPointer(bitPattern: -1),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: raw, length: bytes, options: .storageModeShared,
                      deallocator: nil
                  ) else { continue }
            pool.append(buffer)
            maps.append(raw)
        }
        return (pool, maps)
    }

    /// One buffer, committed on its own and waited immediately. This is the walk's
    /// shape: the host has nothing else queued when the driver is handed the work.
    ///
    /// `recordsSchedule` registers a `scheduled` handler so the arm can separate "the
    /// driver had not submitted it yet" from "the driver submitted it and the GPU was
    /// slow to open the window"; turning it off is the control that says whether
    /// registering a handler perturbs the commit path it is measuring.
    func serialSample(
        dispatches: Int = 1,
        threads: Int = 16,
        idleGapMilliseconds: Double = 0,
        recordsSchedule: Bool = true,
        pool: [MTLBuffer] = [],
        residency: [MTLBuffer] = [],
        barrier: Bool = false
    ) -> DeepSeekV41CommitSample? {
        let stamp = CommitStamp()
        guard let buffer = makeCommandBuffer(
            dispatches: dispatches, threads: threads, pool: pool,
            residency: residency, barrier: barrier
        ) else {
            return nil
        }
        if recordsSchedule {
            buffer.addScheduledHandler { _ in stamp.stamp(DispatchTime.now().uptimeNanoseconds) }
        }
        if idleGapMilliseconds > 0 {
            Thread.sleep(forTimeInterval: idleGapMilliseconds / 1_000)
        }
        buffer.commit()
        let commit = DispatchTime.now().uptimeNanoseconds
        buffer.waitUntilCompleted()
        let returned = DispatchTime.now().uptimeNanoseconds
        return enriched(buffer, commit: commit, returned: returned, stamp: stamp)
    }

    /// `depth` buffers encoded up front and committed back to back with no wait
    /// between them, then waited in order. A depth of one is the serial arm; anything
    /// wider is the queue fed before it is emptied.
    func pipelinedRound(
        depth: Int,
        dispatches: Int = 1,
        threads: Int = 16,
        pool: [MTLBuffer] = []
    ) -> [DeepSeekV41CommitSample] {
        var built: [(buffer: MTLCommandBuffer, stamp: CommitStamp)] = []
        for _ in 0..<max(1, depth) {
            let stamp = CommitStamp()
            guard let buffer = makeCommandBuffer(
                dispatches: dispatches, threads: threads, pool: pool
            ) else {
                continue
            }
            buffer.addScheduledHandler { _ in stamp.stamp(DispatchTime.now().uptimeNanoseconds) }
            built.append((buffer, stamp))
        }
        var commits: [UInt64] = []
        commits.reserveCapacity(built.count)
        for entry in built {
            entry.buffer.commit()
            commits.append(DispatchTime.now().uptimeNanoseconds)
        }
        var samples: [DeepSeekV41CommitSample] = []
        samples.reserveCapacity(built.count)
        for (index, entry) in built.enumerated() {
            entry.buffer.waitUntilCompleted()
            let returned = DispatchTime.now().uptimeNanoseconds
            samples.append(
                enriched(entry.buffer, commit: commits[index], returned: returned, stamp: entry.stamp)
            )
        }
        return samples
    }

    /// A mixture-shaped buffer running a DIFFERENT kernel from the front's: the walk
    /// switches pipeline state between the two halves of every layer, and no arm above
    /// ever submits more than one pipeline.
    func makeReadCommandBuffer(source: [MTLBuffer], destination: MTLBuffer, lanes: Int)
        -> MTLCommandBuffer? {
        guard !source.isEmpty,
              let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else { return nil }
        encoder.setComputePipelineState(readPipeline)
        encoder.setBuffer(source[0], offset: 0, index: 0)
        encoder.setBuffer(source[1 % source.count], offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        var dim = UInt32(max(1, min(lanes, source[0].length / 4)))
        encoder.setBytes(&dim, length: 4, index: 3)
        let width = max(1, min(Int(dim), readPipeline.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(
            MTLSize(width: Int(dim), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        return buffer
    }

    /// One position's submission pattern, exactly as `runLayer` writes it: a front half
    /// committed and waited immediately (the routing readback needs it finished), then a
    /// mixture half committed and left in flight, drained by the next front half's wait.
    ///
    /// `hostBytes` stands in for the host work between the two halves (the routing
    /// readback and the expert fill), written into the scratch buffer so the submission
    /// follows a real host write the way a mixture follows `pread`s.
    func alternatingRound(
        layers: Int,
        dispatches: Int = 20,
        frontPool: [MTLBuffer] = [],
        mixturePool: [MTLBuffer] = [],
        hostBytes: Int = 0,
        mixtureIsRead: Bool = false
    ) -> [DeepSeekV41CommitSample] {
        var samples: [DeepSeekV41CommitSample] = []
        var pending: [(buffer: MTLCommandBuffer, stamp: CommitStamp, commit: UInt64)] = []
        func settle() {
            for entry in pending {
                entry.buffer.waitUntilCompleted()
                let returned = DispatchTime.now().uptimeNanoseconds
                samples.append(
                    enriched(entry.buffer, commit: entry.commit, returned: returned, stamp: entry.stamp)
                )
            }
            pending.removeAll(keepingCapacity: true)
        }
        for _ in 0..<max(1, layers) {
            if let front = makeCommandBuffer(dispatches: dispatches, pool: frontPool) {
                let stamp = CommitStamp()
                front.addScheduledHandler { _ in stamp.stamp(DispatchTime.now().uptimeNanoseconds) }
                front.commit()
                pending.append((front, stamp, DispatchTime.now().uptimeNanoseconds))
            }
            settle()
            if hostBytes > 0 {
                // Written into the buffers the MIXTURE binds, not into a scratch vector:
                // a mixture follows `pread`s that land in the expert arena, so the
                // quantity under test is host writes into memory the next submission
                // reads, not host work in general.
                if mixturePool.isEmpty {
                    memset(target.contents(), 0, min(hostBytes, target.length))
                } else {
                    let each = max(1, hostBytes / mixturePool.count)
                    for buffer in mixturePool {
                        memset(buffer.contents(), 0, min(each, buffer.length))
                    }
                }
            }
            let mixture = mixtureIsRead
                ? makeReadCommandBuffer(
                    source: mixturePool.isEmpty ? [target] : mixturePool,
                    destination: target,
                    lanes: 4_000_000
                )
                : makeCommandBuffer(dispatches: dispatches, pool: mixturePool)
            if let mixture {
                let stamp = CommitStamp()
                mixture.addScheduledHandler { _ in stamp.stamp(DispatchTime.now().uptimeNanoseconds) }
                mixture.commit()
                pending.append((mixture, stamp, DispatchTime.now().uptimeNanoseconds))
            }
        }
        settle()
        return samples
    }

    /// The idle a run of submissions actually pays: the GPU with no window open. Time a
    /// buffer spends waiting behind work that is already queued is NOT idle, and a
    /// per-buffer "commit to start" figure counts it as though it were -- which is why
    /// the walk's 206.7 ms of commit-to-start is not 206.7 ms of idle.
    static func idle(_ samples: [DeepSeekV41CommitSample]) -> Double {
        let windowed = samples
            .filter { $0.hasWindow }
            .sorted { $0.windowStartNanoseconds < $1.windowStartNanoseconds }
        var total = 0.0
        var previousEnd: UInt64 = 0
        for entry in windowed {
            if previousEnd > 0, entry.windowStartNanoseconds > previousEnd {
                total += Double(entry.windowStartNanoseconds - previousEnd) / 1_000_000
            }
            previousEnd = max(previousEnd, entry.windowEndNanoseconds)
        }
        return total
    }

    /// Everything the GPU was busy for, likewise in window order.
    static func busy(_ samples: [DeepSeekV41CommitSample]) -> Double {
        samples.filter { $0.hasWindow }.reduce(0) { $0 + $1.windowMilliseconds }
    }

    /// `gpuStartTime` and `gpuEndTime` share mach's clock, which `DispatchTime` reads,
    /// so all four terms are comparable without a conversion.
    private func enriched(
        _ buffer: MTLCommandBuffer,
        commit: UInt64,
        returned: UInt64,
        stamp: CommitStamp
    ) -> DeepSeekV41CommitSample {
        var sample = DeepSeekV41CommitSample()
        let start = UInt64(max(0, Double(buffer.gpuStartTime) * 1_000_000_000))
        let end = UInt64(max(0, Double(buffer.gpuEndTime) * 1_000_000_000))
        sample.windowStartNanoseconds = start
        sample.windowEndNanoseconds = end
        if start > commit { sample.pickupMilliseconds = Double(start - commit) / 1_000_000 }
        if end > start { sample.windowMilliseconds = Double(end - start) / 1_000_000 }
        if returned > end { sample.tailMilliseconds = Double(returned - end) / 1_000_000 }
        if returned > commit { sample.drainMilliseconds = Double(returned - commit) / 1_000_000 }
        if let scheduled = stamp.value {
            sample.scheduledMilliseconds = scheduled > commit
                ? Double(scheduled - commit) / 1_000_000 : 0
        }
        return sample
    }
}

/// The first `scheduled` stamp Metal reports for one buffer, written from Metal's own
/// thread and read by the test after the buffer completed.
private nonisolated final class CommitStamp: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: UInt64?

    func stamp(_ nanoseconds: UInt64) {
        lock.lock()
        if stored == nil { stored = nanoseconds }
        lock.unlock()
    }

    var value: UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

private func commitMedian(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return 0 }
    return sorted[sorted.count / 2]
}

/// What a committed buffer's start delay is proportional to.
final class DeepSeekV41CommitPickupCostTests: XCTestCase {

    private struct Anatomy {
        var pickup: Double = 0
        var window: Double = 0
        var tail: Double = 0
        var drain: Double = 0
        var scheduled: Double = 0
    }

    private func anatomy(
        _ probe: DeepSeekV41CommitStartupProbe,
        label: String,
        dispatches: Int = 1,
        threads: Int = 16,
        idleGapMilliseconds: Double = 0,
        attempts: Int = 21,
        recordsSchedule: Bool = true,
        pool: [MTLBuffer] = [],
        residency: [MTLBuffer] = [],
        barrier: Bool = false
    ) -> Anatomy {
        var samples: [DeepSeekV41CommitSample] = []
        for _ in 0..<attempts {
            if let sample = probe.serialSample(
                dispatches: dispatches,
                threads: threads,
                idleGapMilliseconds: idleGapMilliseconds,
                recordsSchedule: recordsSchedule,
                pool: pool,
                residency: residency,
                barrier: barrier
            ) {
                samples.append(sample)
            }
        }
        XCTAssertEqual(samples.count, attempts, "\(label) dropped samples")
        XCTAssertTrue(
            samples.allSatisfy { $0.hasWindow },
            "\(label) reported a buffer that never opened a window"
        )
        // Structural, not numeric: the host cannot leave the wait before the GPU has
        // finished, so the drain has to cover the pickup and the window.
        XCTAssertTrue(
            samples.allSatisfy {
                $0.drainMilliseconds + 0.3 >= $0.pickupMilliseconds + $0.windowMilliseconds
            },
            "\(label) drained faster than it was picked up and ran"
        )
        XCTAssertTrue(
            samples.allSatisfy {
                !recordsSchedule || $0.scheduledMilliseconds != nil
            },
            "\(label) never reported its buffer scheduled"
        )
        let result = Anatomy(
            pickup: commitMedian(samples.map { $0.pickupMilliseconds }),
            window: commitMedian(samples.map { $0.windowMilliseconds }),
            tail: commitMedian(samples.map { $0.tailMilliseconds }),
            drain: commitMedian(samples.map { $0.drainMilliseconds }),
            scheduled: commitMedian(samples.compactMap { $0.scheduledMilliseconds })
        )
        print(
            String(
                format: "🔎 [dsv41 commit startup] %@: pickup %.3f ms, window %.3f ms, "
                    + "tail %.3f ms, drain %.3f ms, scheduled %.3f ms",
                label, result.pickup, result.window, result.tail, result.drain, result.scheduled
            )
        )
        return result
    }

    /// A pickup that does not move with the work inside the buffer is a per-buffer
    /// cost; one that grows with the dispatch count is per-dispatch.
    func testWhatACommittedBuffersPickupIsProportionalTo() throws {
        let probe = try DeepSeekV41CommitStartupProbe.make()

        // Warm the pipeline, the queue and the driver's own shader setup.
        for _ in 0..<5 { _ = probe.serialSample() }

        let idle = anatomy(probe, label: "one dispatch, queue idle")
        let noHandler = anatomy(
            probe, label: "one dispatch, no scheduled handler", recordsSchedule: false
        )
        let gap2 = anatomy(probe, label: "one dispatch, 2 ms idle gap first", idleGapMilliseconds: 2)
        let gap10 = anatomy(
            probe, label: "one dispatch, 10 ms idle gap first", idleGapMilliseconds: 10
        )
        // The curve, not a point: the walk commits a mixture half ~1.5 ms after the GPU
        // went idle for the expert fill, so what a gap of that length costs is the
        // number that has to line up with the walk's own pickups.
        let gap5 = anatomy(
            probe, label: "one dispatch, 5 ms idle gap first",
            idleGapMilliseconds: 5, attempts: 41
        )
        let gap20 = anatomy(
            probe, label: "one dispatch, 20 ms idle gap first",
            idleGapMilliseconds: 20, attempts: 21
        )
        let gap50 = anatomy(
            probe, label: "one dispatch, 50 ms idle gap first",
            idleGapMilliseconds: 50, attempts: 11
        )
        print(
            String(
                format: "🔎 [dsv41 commit startup] idle-gap curve: 0 ms %.3f, 2 ms %.3f, 5 ms "
                    + "%.3f, 10 ms %.3f, 20 ms %.3f, 50 ms %.3f (pickup, ms)",
                idle.pickup, gap2.pickup, gap5.pickup, gap10.pickup, gap20.pickup, gap50.pickup
            )
        )
        print(
            String(
                format: "🔎 [dsv41 commit startup] instrument check: pickup is %.3f ms with a "
                    + "scheduled handler and %.3f ms without it, so the handler costs %.3f ms",
                idle.pickup, noHandler.pickup, idle.pickup - noHandler.pickup
            )
        )

        var byDispatch: [(Int, Anatomy)] = []
        for dispatches in [8, 32, 128] {
            byDispatch.append((dispatches, anatomy(
                probe, label: "\(dispatches) dispatches", dispatches: dispatches
            )))
        }
        for (dispatches, result) in byDispatch {
            print(
                String(
                    format: "🔎 [dsv41 commit startup] %d dispatches: pickup %.3f ms for a "
                        + "%.3f ms window (%.3f ms of pickup per dispatch)",
                    dispatches, result.pickup, result.window, result.pickup / Double(dispatches)
                )
            )
        }
        print(
            String(
                format: "🔎 [dsv41 commit startup] a buffer with no work at all costs %.3f ms to "
                    + "start; one with 128 dispatches costs %.3f ms",
                idle.pickup, byDispatch.last?.1.pickup ?? 0
            )
        )

        var byGrid: [(Int, Anatomy)] = []
        for threads in [65_536, 1_048_576] {
            byGrid.append((threads, anatomy(
                probe, label: "one dispatch over \(threads) lanes",
                threads: threads, attempts: 11
            )))
        }
        for (threads, result) in byGrid {
            print(
                String(
                    format: "🔎 [dsv41 commit startup] one dispatch over %d lanes: pickup %.3f ms, "
                        + "window %.3f ms",
                    threads, result.pickup, result.window
                )
            )
        }

        // The idle-gap arms are the control for the surprising direction: if an idle
        // host makes the pickup WORSE, the delay is the driver's deferral of work it
        // does not have to start yet. If it does not move, the delay is per launch.
        print(
            String(
                format: "🔎 [dsv41 commit startup] verdict: pickup %.3f ms fed, %.3f ms after 2 ms "
                    + "idle, %.3f ms after 10 ms idle; %.3f ms with 128 dispatches inside one "
                    + "buffer, so the delay tracks %@",
                idle.pickup, gap2.pickup, gap10.pickup, byDispatch.last?.1.pickup ?? 0,
                idle.pickup > 0.5 * (byDispatch.last?.1.pickup ?? 0) ? "the buffer" : "the work"
            )
        )
    }

    /// Committing several buffers before waiting on any of them: if a follower starts
    /// as its predecessor retires, the pickup is the queue having gone empty and no
    /// per-buffer merge is needed to remove it.
    func testWhetherQueueDepthHidesACommittedBuffersPickup() throws {
        let probe = try DeepSeekV41CommitStartupProbe.make()

        for _ in 0..<5 { _ = probe.serialSample() }
        _ = probe.pipelinedRound(depth: 4)

        for depth in [1, 2, 4, 8] {
            var rounds: [[DeepSeekV41CommitSample]] = []
            for _ in 0..<11 {
                let round = probe.pipelinedRound(depth: depth)
                XCTAssertEqual(round.count, depth, "a round of \(depth) buffers built \(round.count)")
                XCTAssertTrue(
                    round.allSatisfy { $0.hasWindow },
                    "a round of \(depth) reported a buffer that never opened a window"
                )
                // One in-order queue, so a follower cannot start before its
                // predecessor retires. A violation means the windows are misread.
                for index in 1..<max(1, round.count) {
                    XCTAssertGreaterThanOrEqual(
                        Double(round[index].windowStartNanoseconds) + 200_000,
                        Double(round[index - 1].windowEndNanoseconds),
                        "buffer \(index) of \(depth) started before its predecessor finished"
                    )
                }
                rounds.append(round)
            }
            var pickups: [Double] = []
            var gaps: [Double] = []
            var windows: [Double] = []
            for index in 0..<depth {
                pickups.append(commitMedian(rounds.map { $0[index].pickupMilliseconds }))
                windows.append(commitMedian(rounds.map { $0[index].windowMilliseconds }))
                if index == 0 {
                    gaps.append(pickups[0])
                } else {
                    gaps.append(commitMedian(rounds.map { round in
                        let previousEnd = round[index - 1].windowEndNanoseconds
                        let start = round[index].windowStartNanoseconds
                        return start > previousEnd
                            ? Double(start - previousEnd) / 1_000_000 : 0
                    }))
                }
            }
            let wall = commitMedian(rounds.map { $0.map { $0.drainMilliseconds }.reduce(0, +) })
            print(
                String(
                    format: "🔎 [dsv41 commit startup] depth %d: total %.3f ms; first buffer "
                        + "pickup %.3f ms; followers' pickups %@; followers' gap from the "
                        + "predecessor's end %@; windows %@",
                    depth, wall, pickups[0],
                    pickups.dropFirst().map { String(format: "%.3f", $0) }.joined(separator: ", "),
                    gaps.dropFirst().map { String(format: "%.3f", $0) }.joined(separator: ", "),
                    windows.map { String(format: "%.3f", $0) }.joined(separator: ", ")
                )
            )
        }

        // The shape a pipelined walk would have: a predecessor still running when its
        // followers are committed. Followers cannot pay a pickup for an empty queue
        // because the queue was never empty.
        let long = probe.pipelinedRound(depth: 8, dispatches: 96, threads: 1_048_576)
        XCTAssertEqual(long.count, 8, "the long-predecessor arm built \(long.count) buffers")
        var longGaps: [Double] = []
        for index in 1..<long.count {
            let previousEnd = long[index - 1].windowEndNanoseconds
            let start = long[index].windowStartNanoseconds
            longGaps.append(start > previousEnd ? Double(start - previousEnd) / 1_000_000 : 0)
        }
        print(
            String(
                format: "🔎 [dsv41 commit startup] fed queue, 96 dispatches a buffer: windows %@; "
                    + "followers' pickups %@; followers' gap from the predecessor's end %@",
                long.map { String(format: "%.3f", $0.windowMilliseconds) }.joined(separator: ", "),
                long.dropFirst().map { String(format: "%.3f", $0.pickupMilliseconds) }
                    .joined(separator: ", "),
                longGaps.map { String(format: "%.3f", $0) }.joined(separator: ", ")
            )
        )
        print(
            String(
                format: "🔎 [dsv41 commit startup] verdict: a buffer waited on at once costs "
                    + "%.3f ms before it starts; the same buffer committed behind a running one "
                    + "starts %.3f ms after its predecessor retires",
                long.first?.pickupMilliseconds ?? 0,
                commitMedian(longGaps)
            )
        )
    }

    /// What a command buffer has to carry before a submission stops starting at once.
    ///
    /// A trivial launch starts in a tenth of a millisecond, so the walk's 2-3 ms has to
    /// come from something its buffers have: dozens of launches (a half layer), or a
    /// resource list spread over every shard the layer reads, or memory the driver has
    /// to make resident at submit time. The arms below separate those three, because
    /// only the first is affected by how many buffers a position is cut into.
    func testWhatACommandBufferHasToCarryToStartSlowly() throws {
        let probe = try DeepSeekV41CommitStartupProbe.make()
        for _ in 0..<5 { _ = probe.serialSample() }

        // 1. The control: one launch, one 16 MB scratch buffer.
        _ = anatomy(probe, label: "1 launch, 1 small resource", attempts: 11)

        // 2. Launches alone, all binding the same vector — a half layer's launch count.
        _ = anatomy(probe, label: "64 launches, 1 small resource", dispatches: 64, attempts: 11)
        _ = anatomy(
            probe, label: "64 launches at a published-width grid, 1 small resource",
            dispatches: 64, threads: 73_728, attempts: 11
        )

        // 3. Resources alone, at a constant 4 GiB: 64 slices of 64 MiB against 8 slices
        //    of 512 MiB tells resource COUNT apart from resource BYTES.
        let many = probe.residentPool(count: 64, bytes: 64 << 20)
        XCTAssertEqual(many.count, 64, "the 64-slice pool allocated \(many.count) buffers")
        _ = anatomy(
            probe, label: "64 launches, 64 resident resources of 64 MiB",
            dispatches: 64, attempts: 11, pool: many
        )
        let few = probe.residentPool(count: 8, bytes: 512 << 20)
        XCTAssertEqual(few.count, 8, "the 8-slice pool allocated \(few.count) buffers")
        _ = anatomy(
            probe, label: "64 launches, 8 resident resources of 512 MiB",
            dispatches: 64, attempts: 11, pool: few
        )

        // 4. The same shape with no copy, over a sparse file: pageable memory, which is
        //    what a shard mapping is and what the pinned chunks are not.
        let mapped = probe.mappedPool(count: 64, bytes: 64 << 20)
        XCTAssertEqual(mapped.pool.count, 64, "the mapped pool wrapped \(mapped.pool.count) buffers")
        _ = anatomy(
            probe, label: "64 launches, 64 mapped resources of 64 MiB",
            dispatches: 64, attempts: 11, pool: mapped.pool
        )

        // 5. Does feeding the queue still hide it once the buffer is heavy?
        var rounds: [[DeepSeekV41CommitSample]] = []
        for _ in 0..<11 {
            rounds.append(probe.pipelinedRound(depth: 4, dispatches: 64, pool: many))
        }
        var gaps: [Double] = []
        for index in 1..<4 {
            gaps.append(commitMedian(rounds.map { round in
                let previousEnd = round[index - 1].windowEndNanoseconds
                let start = round[index].windowStartNanoseconds
                return start > previousEnd ? Double(start - previousEnd) / 1_000_000 : 0
            }))
        }
        print(
            String(
                format: "🔎 [dsv41 commit startup] fed queue, 64 launches over 64 resident "
                    + "resources: first pickup %.3f ms, followers' gap from the predecessor's "
                    + "end %@",
                commitMedian(rounds.map { $0[0].pickupMilliseconds }),
                gaps.map { String(format: "%.3f", $0) }.joined(separator: ", ")
            )
        )

        for map in mapped.maps { munmap(map, 64 << 20) }
        try? FileManager.default.removeItem(
            atPath: NSTemporaryDirectory() + "dsv41_commit_probe_mapping"
        )
    }

    /// The one axis a binding slot cannot measure: how many DISTINCT resources a
    /// submission names. A half layer's union is every workspace buffer, gain vector,
    /// expert slot and shard slice it touches, and Metal builds that list per command
    /// buffer however few of them any single launch binds.
    func testWhatTheResourceListCostsACommittedBuffer() throws {
        let probe = try DeepSeekV41CommitStartupProbe.make()
        for _ in 0..<5 { _ = probe.serialSample() }

        // 512 slices of 4 MiB: 2 GiB in total, so raising the count does not raise the
        // bytes and the two cannot be confused.
        let slices = probe.residentPool(count: 512, bytes: 4 << 20)
        XCTAssertEqual(slices.count, 512, "the slice pool allocated \(slices.count) buffers")

        var curve: [(Int, Anatomy)] = []
        for count in [0, 32, 64, 128, 256, 512] {
            curve.append((count, anatomy(
                probe,
                label: "20 launches naming \(count) resources",
                dispatches: 20,
                attempts: 11,
                residency: Array(slices.prefix(count))
            )))
        }
        for (count, result) in curve {
            print(
                String(
                    format: "🔎 [dsv41 commit startup] 20 launches, %d resources of 4 MiB: "
                        + "pickup %.3f ms, window %.3f ms, drain %.3f ms",
                    count, result.pickup, result.window, result.drain
                )
            )
        }
        let first = curve.first?.1.pickup ?? 0
        let last = curve.last?.1.pickup ?? 0
        print(
            String(
                format: "🔎 [dsv41 commit startup] resource-list curve: %.3f ms with no "
                    + "resources, %.3f ms with 512, i.e. %.3f ms of pickup per resource",
                first, last, (last - first) / 512
            )
        )

        // Same 2 GiB, eight resources instead of 512: bytes cannot be the cause.
        let wide = probe.residentPool(count: 8, bytes: 256 << 20)
        XCTAssertEqual(wide.count, 8, "the wide pool allocated \(wide.count) buffers")
        let wideAnatomy = anatomy(
            probe, label: "20 launches naming 8 resources of 256 MiB",
            dispatches: 20, attempts: 11, residency: wide
        )
        print(
            String(
                format: "🔎 [dsv41 commit startup] 2 GiB as 8 resources: pickup %.3f ms "
                    + "against %.3f ms as 512 resources and %.3f ms as none",
                wideAnatomy.pickup, last, first
            )
        )

        // A real half layer also fences between dispatches, and it does both at once.
        let fenced = anatomy(
            probe, label: "20 launches, 128 resources, a barrier after each dispatch",
            dispatches: 20, attempts: 11,
            residency: Array(slices.prefix(128)), barrier: true
        )
        print(
            String(
                format: "🔎 [dsv41 commit startup] 20 launches, 128 resources, 20 buffer "
                    + "barriers: pickup %.3f ms, window %.3f ms",
                fenced.pickup, fenced.window
            )
        )

        // And fed: a deep queue used to hide everything measured so far.
        var rounds: [[DeepSeekV41CommitSample]] = []
        for _ in 0..<11 {
            rounds.append(probe.pipelinedRound(
                depth: 4, dispatches: 20,
                pool: Array(slices.prefix(128))
            ))
        }
        var gaps: [Double] = []
        for index in 1..<4 {
            gaps.append(commitMedian(rounds.map { round in
                let previousEnd = round[index - 1].windowEndNanoseconds
                let start = round[index].windowStartNanoseconds
                return start > previousEnd ? Double(start - previousEnd) / 1_000_000 : 0
            }))
        }
        print(
            String(
                format: "🔎 [dsv41 commit startup] fed queue of 4, 20 launches each naming "
                    + "128 resources: followers' gap from the predecessor's end %@",
                gaps.map { String(format: "%.3f", $0) }.joined(separator: ", ")
            )
        )
    }

    /// Whether a submission starts late because someone else is using the GPU.
    ///
    /// Every arm above starts in a tenth of a millisecond, so the walk's 2.3 ms has to
    /// come from something that is not the submission. The last candidate that a single
    /// process cannot rule out by shape is a co-tenant: the walk shares the GPU with
    /// the compositor and with whatever else the machine is doing, and a queue that is
    /// not the only one submitting can wait behind work no command buffer of ours
    /// accounts for.
    func testWhetherASecondQueueDelaysACommittedBuffersStart() throws {
        let measured = try DeepSeekV41CommitStartupProbe.make()
        let tenant = try DeepSeekV41CommitStartupProbe.make()

        func pickup(attempts: Int = 21) -> (median: Double, worst: Double) {
            var values: [Double] = []
            for _ in 0..<attempts {
                if let sample = measured.serialSample(dispatches: 20) {
                    values.append(sample.pickupMilliseconds)
                }
            }
            return (commitMedian(values), values.max() ?? 0)
        }

        // Warm both queues before either is measured.
        for _ in 0..<5 {
            _ = measured.serialSample()
            _ = tenant.pipelinedRound(depth: 2, dispatches: 64, threads: 1_048_576)
        }
        let alone = pickup()

        // A co-tenant whose own buffers take a few hundred microseconds, submitted
        // back to back the way a compositor submits one frame after another. It holds
        // the GPU for most of each millisecond without owning our queue.
        let flag = CoTenantFlag()
        Thread.detachNewThread {
            while flag.isRunning {
                _ = tenant.pipelinedRound(depth: 2, dispatches: 64, threads: 1_048_576)
            }
        }
        Thread.sleep(forTimeInterval: 0.05)
        let shared = pickup()
        flag.stop()
        Thread.sleep(forTimeInterval: 0.05)

        let after = pickup()
        print(
            String(
                format: "🔎 [dsv41 commit startup] alone: pickup %.3f ms (worst %.3f); "
                    + "with a co-tenant on another queue: %.3f ms (worst %.3f); after it "
                    + "stopped: %.3f ms",
                alone.median, alone.worst, shared.median, shared.worst, after.median
            )
        )
        print(
            String(
                format: "🔎 [dsv41 commit startup] the co-tenant arm reads %.3f ms against "
                    + "%.3f ms alone, i.e. a busy GPU starts a submission %@ than an idle one "
                    + "(clock ramp, not contention)",
                shared.median, alone.median,
                shared.median < alone.median ? "sooner" : "later"
            )
        )
    }

    /// The walk's own submission pattern, with none of its work: 40 layers of a front
    /// half committed and waited, then a mixture half committed and left in flight.
    ///
    /// Every arm above measures one submission in isolation or a queue fed before it is
    /// emptied. This one measures the alternation itself, which is the only thing left
    /// that a microbenchmark of the walk's own shapes has not covered.
    func testWhetherTheWalksAlternationCostsWhatNoSingleBufferDoes() throws {
        let probe = try DeepSeekV41CommitStartupProbe.make()

        // Pools shaped like the two halves' working sets, not like the checkpoint's
        // sizes: what is under test is the pattern, not the weights.
        let frontPool = probe.residentPool(count: 8, bytes: 16 << 20)
        let mixturePool = probe.residentPool(count: 6, bytes: 32 << 20)
        for _ in 0..<3 { _ = probe.alternatingRound(layers: 4) }

        func report(_ label: String, _ samples: [DeepSeekV41CommitSample]) {
            let windowed = samples.filter { $0.hasWindow }
            let first = windowed.map { $0.windowStartNanoseconds }.min() ?? 0
            let last = windowed.map { $0.windowEndNanoseconds }.max() ?? 0
            let wall = last > first ? Double(last - first) / 1_000_000 : 0
            let idle = DeepSeekV41CommitStartupProbe.idle(samples)
            print(
                String(
                    format: "🔎 [dsv41 commit startup] %@ (%d buffer(s)): GPU busy %.1f ms, "
                        + "GPU idle %.1f ms of %.1f ms (%.0f%%), per-buffer drain %.3f ms, "
                        + "per-buffer pickup %.3f ms",
                    label, samples.count,
                    DeepSeekV41CommitStartupProbe.busy(samples), idle, wall,
                    100 * idle / max(0.001, wall),
                    samples.reduce(0.0) { $0 + $1.drainMilliseconds }
                        / Double(max(1, samples.count)),
                    commitMedian(samples.map { $0.pickupMilliseconds })
                )
            )
        }

        report(
            "the walk's pattern, no host work",
            probe.alternatingRound(layers: 40, dispatches: 20)
        )
        report(
            "the walk's pattern, 8 KB of host writes a layer",
            probe.alternatingRound(
                layers: 40, dispatches: 20,
                frontPool: frontPool, mixturePool: mixturePool, hostBytes: 8 << 10
            )
        )
        report(
            "the walk's pattern, 1 MB of host writes a layer",
            probe.alternatingRound(
                layers: 40, dispatches: 20,
                frontPool: frontPool, mixturePool: mixturePool, hostBytes: 1 << 20
            )
        )
        // The mixture's own shape: the host fills the buffers it is about to read. The
        // expert arena slot is ~17.9 MB, six of them are read per layer, and the fill
        // writes all of it from `pread` before the mixture is encoded.
        report(
            "the walk's pattern, 128 MB of host writes into the mixture's own buffers",
            probe.alternatingRound(
                layers: 40, dispatches: 20,
                frontPool: frontPool, mixturePool: mixturePool, hostBytes: 128 << 20
            )
        )

        // The control: the same 80 buffers committed before any of them is waited on.
        var fed: [DeepSeekV41CommitSample] = []
        for _ in 0..<2 { fed += probe.pipelinedRound(depth: 40, dispatches: 20) }
        // The one difference left between the two halves of a layer: they run different
        // kernels. The front half and the expert half never share a pipeline, so every
        // layer switches pipeline state between them.
        report(
            "the walk's pattern, a different pipeline for the mixture",
            probe.alternatingRound(
                layers: 40, dispatches: 20,
                frontPool: frontPool, mixturePool: mixturePool, mixtureIsRead: true
            )
        )
        report("the same buffers, all queued first", fed)
    }

    /// Whether a submission starts late for memory the host has just written.
    ///
    /// A layer's expert half is the only place the walk leaves the GPU idle, and it is
    /// the one submission that follows a `pread` into the very bytes its first kernel
    /// reads. Every arm above submits kernels that only WRITE, so this is the last
    /// difference between a probe submission and a walk submission that a probe can
    /// hold constant.
    func testWhetherReadingWhatTheHostJustWroteDelaysAStart() throws {
        let probe = try DeepSeekV41CommitStartupProbe.make()
        let source = probe.residentPool(count: 2, bytes: 32 << 20)
        guard let destination = probe.residentPool(count: 1, bytes: 32 << 20).first else {
            XCTFail("no destination buffer")
            return
        }
        XCTAssertEqual(source.count, 2, "the source pool allocated \(source.count) buffers")
        let lanes = (32 << 20) / MemoryLayout<Float>.stride

        func arm(_ label: String, hostWrites: Bool, attempts: Int = 21) -> Double {
            for _ in 0..<5 {
                _ = probe.readSample(
                    source: source, lanes: lanes, destination: destination,
                    hostWritesFirst: false
                )
            }
            var pickups: [Double] = []
            var windows: [Double] = []
            for _ in 0..<attempts {
                guard let sample = probe.readSample(
                    source: source, lanes: lanes, destination: destination,
                    hostWritesFirst: hostWrites
                ) else { continue }
                pickups.append(sample.pickupMilliseconds)
                windows.append(sample.windowMilliseconds)
            }
            XCTAssertEqual(pickups.count, attempts, "\(label) dropped samples")
            XCTAssertTrue(
                windows.allSatisfy { $0 > 0 },
                "\(label) reported a buffer that never opened a window"
            )
            let pickup = commitMedian(pickups)
            print(
                String(
                    format: "🔎 [dsv41 commit startup] %@: pickup %.3f ms, window %.3f ms",
                    label, pickup, commitMedian(windows)
                )
            )
            return pickup
        }

        let cold = arm("64 MB read, nothing touched", hostWrites: false)
        let warm = arm("64 MB read, every page written by the host first", hostWrites: true)
        print(
            String(
                format: "🔎 [dsv41 commit startup] verdict: a submission over memory the host "
                    + "just wrote starts in %.3f ms against %.3f ms for the same submission "
                    + "over untouched memory",
                warm, cold
            )
        )
    }
}

/// A stop flag a detached thread can read without touching an actor.
private nonisolated final class CoTenantFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var running = true

    var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return running
    }

    func stop() {
        lock.lock()
        running = false
        lock.unlock()
    }
}
