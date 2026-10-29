//
//  DeepSeekV41LogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the DeepSeek-V4.1 port: config decoding (both the
//  multimodal wrapper and the flat inference config), the derived per-layer
//  plan, the Engram prime-bucket layout and row addressing, the tensor catalog
//  (including its fail-closed behaviour), and the snapshot preflight.
//
//  No Metal device, no weights, no MainActor required, so this runs anywhere.
//
//  The Engram assertion is a real reference vector, not a self-consistency
//  check: with `engram_vocab_size = 16_000_000`, `max_ngram_size = 4` and
//  `n_heads = 8`, the bucket arithmetic must address exactly
//  `engram_num_embeddings[0] = 384_006_168` rows of the published checkpoint.
//

import XCTest
@testable import DynaMoE

final class DeepSeekV41ConfigTests: XCTestCase {

    /// The multimodal wrapper shape: text tower under `text_config`, vision
    /// tower beside it, token ids at the root.
    static let wrapperJSON = """
    {
      "architectures": ["DeepseekV41ForCausalLM"],
      "model_type": "deepseek_v41",
      "dtype": "bfloat16",
      "bos_token_id": 0,
      "eos_token_id": 1,
      "pad_token_id": 2,
      "image_token_id": 129264,
      "quantization_config": {"quant_method": "fp8"},
      "vision_config": {"num_hidden_layers": 27, "patch_size": 16, "downsample_ratio": 3,
                        "min_pixels": 65536, "max_image_tokens": 1024},
      "text_config": {
        "model_type": "deepseek_v41",
        "num_hidden_layers": 40,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "vocab_size": 129280,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 2,
        "n_routed_experts": 384,
        "n_activated_experts": 6,
        "routed_scaling_factor": 2.5,
        "swiglu_limit": 10.0,
        "q_lora_rank": 1536,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "v_head_dim": 512,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "rope_scaling": {"factor": 8, "original_max_position_embeddings": 131072,
                         "beta_fast": 32, "beta_slow": 1},
        "max_position_embeddings": 1048576,
        "window_size": 128,
        "compress_ratios": [0, 0,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "kv_source_layer_ids": [2, 20],
        "index_source_layer_ids": [2, 20],
        "candidate_source_layer_id": 20,
        "candidate_block_size": 8,
        "candidate_topk_blocks": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39]
      }
    }
    """

    func config() throws -> DeepSeekV41Config {
        try DeepSeekV41Config.decode(jsonData: Data(Self.wrapperJSON.utf8))
    }

    func testWrapperShapeDecodes() throws {
        let config = try config()
        XCTAssertEqual(config.modelType, "deepseek_v41")
        XCTAssertEqual(config.architectures, ["DeepseekV41ForCausalLM"])
        XCTAssertEqual(config.numHiddenLayers, 40)
        XCTAssertEqual(config.hiddenSize, 7168)
        XCTAssertEqual(config.numAttentionHeads, 128)
        XCTAssertEqual(config.numRoutedExperts, 384)
        XCTAssertEqual(config.numActivatedExperts, 6)
        XCTAssertEqual(config.vocabSize, 129280)
        XCTAssertEqual(config.eosTokenIDs, [1])
        XCTAssertEqual(config.bosTokenID, 0)
        XCTAssertEqual(config.padTokenID, 2)
        XCTAssertEqual(config.imageTokenID, 129264)
        XCTAssertEqual(config.quantizationMethod, "fp8")
        XCTAssertTrue(config.hasVisionTower)
        XCTAssertEqual(config.engramLayerIDs, [1, 14])
        XCTAssertEqual(config.engramRowCounts, [384_006_168, 384_016_682])
    }

    /// The multimodal wrapper spells the sliding window `sliding_window`; the flat
    /// `inference/config.json` spells it `window_size`. Reading only one of them
    /// left every layer's ring zero rows wide, which reads as "no window" and
    /// refused the checkpoint with a compress-ratio problem on every layer.
    func testTheWrapperSlidingWindowDecodes() throws {
        let json = Self.wrapperJSON.replacingOccurrences(
            of: "\"window_size\": 128,",
            with: "\"sliding_window\": 128,"
        )
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        XCTAssertEqual(config.windowSize, 128)
        XCTAssertEqual(DeepSeekV41DevicePlan.gatherRowBudget(config: config), 128 + 512)
    }

    /// A CSA2 checkpoint always has a window ring, so a config that declares
    /// `compress_ratios` with no window is broken rather than window-less. This is
    /// the failure the alias above would otherwise hide as a note in the log.
    func testACSA2ConfigWithoutAWindowIsRejected() {
        let json = Self.wrapperJSON.replacingOccurrences(of: "\"window_size\": 128,", with: "")
        XCTAssertThrowsError(try DeepSeekV41Config.decode(jsonData: Data(json.utf8))) { error in
            guard case DeepSeekV41ConfigError.invalidField(let field, _)? = error as? DeepSeekV41ConfigError else {
                return XCTFail("expected an invalid window field, got \(error)")
            }
            XCTAssertEqual(field, "window_size")
        }
    }

    /// The latent cache widths the reference documents: 512 E4M3 values plus
    /// one E8M0 scale per 32 is 528 bytes; the designed FP4 record is 288.
    func testCacheRecordWidths() throws {
        let config = try config()
        XCTAssertEqual(config.qkHeadDim, 512)
        XCTAssertEqual(config.fp8CacheRecordBytes, 528)
        XCTAssertEqual(config.fp4CacheRecordBytes, 288)
    }

    func testLayerPlanSplitsEncoderAndDecoderHalves() throws {
        let config = try config()
        let plan = try config.makeLayerPlan()
        XCTAssertEqual(plan.count, 40)
        XCTAssertEqual(plan[0].flavor, .windowOnly)
        XCTAssertEqual(plan[1].flavor, .windowOnly)
        XCTAssertEqual(plan[2].flavor, .compressed)
        XCTAssertEqual(plan[19].flavor, .compressed)
        XCTAssertEqual(plan[20].flavor, .compressed)
        // The encoder half shares one compressed cache, the decoder half shares
        // the one layer 20 owns.
        XCTAssertEqual(plan[19].compressedKVSource, 2)
        XCTAssertEqual(plan[20].compressedKVSource, 20)
        XCTAssertEqual(plan[39].compressedKVSource, 20)
        XCTAssertEqual(plan.filter(\.ownsCompressedKV).map(\.index), [2, 20])
        XCTAssertEqual(config.compressedKVOwnerLayerIndices, [2, 20])
        // Window-only layers never attend over compressed rows.
        XCTAssertEqual(plan[0].compressedKVSource, -1)
        XCTAssertEqual(plan[1].compressedIndexSource, -1)
        // RoPE table follows the compressor: compressed layers rotate with the
        // compressed theta.
        XCTAssertEqual(plan[2].ropeTable, .compress)
        XCTAssertEqual(plan[0].ropeTable, .window)
        // Engram tables and the candidate source land where the config says.
        XCTAssertEqual(plan[1].engramTable, 0)
        XCTAssertEqual(plan[14].engramTable, 1)
        XCTAssertTrue(plan[20].isCandidateSource)
    }

    func testCompressedCacheIsSmallEvenAtFullContext() throws {
        let config = try config()
        // 2 owners x 288 bytes x 1M tokens is ~576 MB of main cache; the window
        // rings are bounded by window_size and cost nothing per token.
        XCTAssertEqual(config.cacheBytesPerToken(useFP4Main: true), 576)
        XCTAssertEqual(config.cacheBytesPerToken(useFP4Main: false), 1056)
        let mainCache = config.cacheBytesPerToken(useFP4Main: true) * 1_048_576
        XCTAssertLessThan(Double(mainCache) / 1_073_741_824.0, 1.0)
    }

    func testFlatInferenceConfigDecodes() throws {
        let flat = """
        {"model_type": "deepseek_v41", "n_layers": 40, "dim": 7168, "n_heads": 128,
         "vocab_size": 129280, "n_routed_experts": 384, "n_activated_experts": 6,
         "qk_nope_head_dim": 512, "v_head_dim": 512, "o_groups": 8,
         "window_size": 128,
         "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         "engram_layer_ids": [1, 14], "engram_num_embeddings": [384006168, 384016682],
         "engram_max_ngram_size": 4, "engram_vocab_size": 16000000, "engram_n_heads": 8,
         "engram_head_dim": 256, "engram_pad_id": 2, "engram_compressed_vocab_size": 99092,
         "dtype": "fp8", "expert_dtype": "fp4"}
        """
        let config = try DeepSeekV41Config.decode(jsonData: Data(flat.utf8))
        XCTAssertEqual(config.hiddenSize, 7168)
        XCTAssertEqual(config.numHiddenLayers, 40)
        XCTAssertEqual(config.engramPadTokenID, 2)
        // No published source list: the first compressible layer owns the cache
        // and every later compressed layer reads it.
        let plan = try config.makeLayerPlan()
        XCTAssertEqual(plan[39].compressedKVSource, 2)
    }

    func testOtherFamiliesAreRejected() {
        XCTAssertThrowsError(
            try DeepSeekV41Config.decode(jsonData: Data(#"{"model_type":"qwen3_5_moe"}"#.utf8))
        ) { error in
            XCTAssertEqual(error as? DeepSeekV41ConfigError, .notDeepSeekV41)
        }
    }

    func testMissingLayerCountIsReported() {
        XCTAssertThrowsError(
            try DeepSeekV41Config.decode(jsonData: Data(#"{"model_type":"deepseek_v41"}"#.utf8))
        ) { error in
            XCTAssertEqual(error as? DeepSeekV41ConfigError, .missingField("num_hidden_layers"))
        }
    }

    func testEngramPairMismatchIsRejected() throws {
        var config = try config()
        config.engramRowCounts = [1]
        XCTAssertThrowsError(try config.makeLayerPlan())
    }

    /// The checkpoint publishes `head_dim` 512 and `qk_rope_head_dim` 64 and no
    /// `qk_nope_head_dim`: the rope payload is the trailing 64 of each 512-wide head,
    /// so the non-rope width is the difference and the cache record stays 512 wide.
    /// Reading `qk_nope_head_dim` out of `head_dim` instead makes the head 576 wide,
    /// which no longer matches the record the cache stores.
    func testRopePayloadIsTheTrailingSliceOfTheHead() throws {
        let config = try config()
        XCTAssertEqual(config.qkRopeHeadDim, 64)
        XCTAssertEqual(config.qkNopeHeadDim, 448)
        XCTAssertEqual(config.qkHeadDim, 512)
        XCTAssertEqual(config.vHeadDim, 512)
        XCTAssertEqual(config.cacheRecordWidth, 512)
        XCTAssertEqual(config.fp8CacheRecordBytes, 528)
        XCTAssertEqual(config.fp4CacheRecordBytes, 288)
    }

    /// The published geometry of the output projection: `wo_a` is block-diagonal
    /// over `o_groups`, so it publishes `o_groups * o_lora_rank` rows and `wo_b`
    /// reads that whole concatenation.
    func testOutputProjectionGeometryIsGrouped() throws {
        let config = try config()
        XCTAssertEqual(config.oGroups, 8)
        XCTAssertEqual(config.oLoraRank, 1024)
        let shapes = DeepSeekV41TensorShapes(config: config)
        XCTAssertEqual(shapes.attentionOutputWidth, 128 * 512)
        XCTAssertEqual(shapes.outputGroups, 8)
        XCTAssertEqual(
            shapes.outputReducedWidth, 8 * 1024,
            "wo_b reads o_groups * o_lora_rank, so pricing it at o_lora_rank halves the rank"
        )
        // Per group: 16 heads (128 / 8) of 512 values.
        XCTAssertEqual(shapes.attentionOutputWidth / shapes.outputGroups, 16 * 512)
    }

    /// `Gate.forward`: `scoring_func` is decoded, the default is the checkpoint's
    /// sqrtsoftplus rather than the V3 family's sigmoid, and `gate_temp` divides
    /// the logit before the scoring function.
    func testRouterScoringConfigDecodes() throws {
        let config = try config()
        XCTAssertEqual(config.scoringFunction, .sqrtsoftplus)
        XCTAssertEqual(config.gateTemp, 1)
        XCTAssertEqual(config.normalizeTopKProb, true)

        let scored = Self.wrapperJSON
            .replacingOccurrences(
                of: "\"num_nextn_predict_layers\": 3,",
                with: "\"num_nextn_predict_layers\": 3, \"scoring_func\": \"sigmoid\", \"gate_temp\": 0.5,"
            )
        let decoded = try DeepSeekV41Config.decode(jsonData: Data(scored.utf8))
        XCTAssertEqual(decoded.scoringFunction, .sigmoid)
        XCTAssertEqual(decoded.gateTemp, 0.5, accuracy: 1e-6)

        // A non-positive temperature would hand the router infinities.
        let zeroTemp = Self.wrapperJSON
            .replacingOccurrences(
                of: "\"num_nextn_predict_layers\": 3,",
                with: "\"num_nextn_predict_layers\": 3, \"gate_temp\": 0,"
            )
        XCTAssertThrowsError(try DeepSeekV41Config.decode(jsonData: Data(zeroTemp.utf8))) { error in
            XCTAssertTrue("\(error)".contains("gate_temp"), "expected gate_temp to be named, got \(error)")
        }
    }

    /// `get_moe_config`: a DSpark stage routes its own expert count when the
    /// checkpoint publishes one, and falls back to the backbone's otherwise.
    func testDraftStagesHaveTheirOwnExpertCounts() throws {
        let config = try config()
        XCTAssertEqual(config.draftExpertCounts.routed, config.numRoutedExperts)
        XCTAssertEqual(config.draftExpertCounts.activated, config.numActivatedExperts)

        let dspark = Self.wrapperJSON
            .replacingOccurrences(
                of: "\"num_nextn_predict_layers\": 3,",
                with: "\"num_nextn_predict_layers\": 3, \"dspark_n_routed_experts\": 128, \"dspark_num_experts_per_tok\": 3,"
            )
        let decoded = try DeepSeekV41Config.decode(jsonData: Data(dspark.utf8))
        XCTAssertEqual(decoded.draftExpertCounts.routed, 128)
        XCTAssertEqual(decoded.draftExpertCounts.activated, 3)

        // A DSpark top-k wider than its own expert set can never be satisfied.
        let impossible = Self.wrapperJSON
            .replacingOccurrences(
                of: "\"num_nextn_predict_layers\": 3,",
                with: "\"num_nextn_predict_layers\": 3, \"dspark_n_routed_experts\": 4, \"dspark_num_experts_per_tok\": 9,"
            )
        XCTAssertThrowsError(try DeepSeekV41Config.decode(jsonData: Data(impossible.utf8)))
    }


    /// A declared `head_dim` that its parts do not add up to is the shape that
    /// silently widens every head, so it is rejected rather than reconciled.
    func testAHeadWidthThatDisagreesWithItsPartsIsRejected() throws {
        let json = Self.wrapperJSON
            .replacingOccurrences(of: "\"head_dim\": 512,", with: "\"head_dim\": 576, \"qk_nope_head_dim\": 448,")
        XCTAssertThrowsError(try DeepSeekV41Config.decode(jsonData: Data(json.utf8))) { error in
            XCTAssertTrue(
                "\(error)".contains("head_dim"),
                "expected the head width to be named, got \(error)"
            )
        }
    }

    func testAnOddOrOversizedRopePayloadIsRejected() {
        for rope in [63, 512, 514] {
            let json = Self.wrapperJSON
                .replacingOccurrences(of: "\"qk_rope_head_dim\": 64,", with: "\"qk_rope_head_dim\": \(rope),")
            XCTAssertThrowsError(
                try DeepSeekV41Config.decode(jsonData: Data(json.utf8)),
                "qk_rope_head_dim \(rope) should not load"
            )
        }
    }

    /// A pure sliding-window layer rotates at plain `rope_theta` with no YaRN block,
    /// which is the checkpoint's `original_seq_len = 0` branch.
    func testWindowLayersDoNotExtrapolate() throws {
        let config = try config()
        let plan = try config.makeLayerPlan()
        let window = DeepSeekV41AttentionShape.make(config: config, layer: plan[0])
        let compressed = DeepSeekV41AttentionShape.make(config: config, layer: plan[2])
        XCTAssertEqual(window.ropeScaling, .none)
        XCTAssertEqual(window.ropeTheta, 10_000)
        XCTAssertEqual(compressed.ropeScaling.factor, 8)
        XCTAssertEqual(compressed.ropeScaling.originalLength, 131_072)
        XCTAssertEqual(compressed.ropeTheta, 160_000)
        // Both rails rotate the same 64-wide slice.
        XCTAssertEqual(window.queryRopeWidth, 64)
        XCTAssertEqual(window.latentRopeWidth, 64)
    }

    /// The architecture must win before the topology heuristics: 384 experts and
    /// `maxExpertId >= 500` rules would otherwise reclassify the model as Qwen
    /// 3.8 Flash Next / Standard MoE and run the wrong forward pass.
    func testArchitectureDetectionPrefersDeepSeekV41() throws {
        let decoder = JSONDecoder()
        let root = try decoder.decode(
            ModelConfig.self,
            from: Data(#"{"model_type":"deepseek_v41","architectures":["DeepseekV41ForCausalLM"]}"#.utf8)
        )
        XCTAssertEqual(root.resolveArchitectureType(summary: nil), .deepSeekV41Flash)
        XCTAssertTrue(root.isDeepSeekV41Model)

        // The wrapper shape: identity lives in the nested text tower.
        let wrapped = try decoder.decode(
            ModelConfig.self,
            from: Data(#"{"architectures":["DeepseekV41ForCausalLM"],"text_config":{"model_type":"deepseek_v41","num_hidden_layers":40}}"#.utf8)
        )
        XCTAssertEqual(wrapped.resolveArchitectureType(summary: nil), .deepSeekV41Flash)
        XCTAssertTrue(wrapped.isDeepSeekV41Model)

        // Other families must keep their own classification.
        let qwen = try decoder.decode(
            ModelConfig.self,
            from: Data(#"{"model_type":"qwen3_5_moe","num_experts":64}"#.utf8)
        )
        XCTAssertEqual(qwen.resolveArchitectureType(summary: nil), .hybridSsmMoe)
    }

    /// Until the turn loop exists, the family is detected (so the load refuses
    /// clearly) but discovery must not advertise graded effort.
    func testGradedEffortIsNotAdvertisedWhileTheRuntimeIsUnavailable() throws {
        XCTAssertFalse(DeepSeekV41Availability.isRuntimeAvailable)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-effort-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.wrapperJSON.write(
            to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8
        )
        XCTAssertFalse(ModelConfig.supportsThinkingLevels(modelPath: dir.path))
    }
}

// MARK: - Engram

final class DeepSeekV41EngramTests: XCTestCase {

    /// The published checkpoint's own numbers: two Engram layers, `max_ngram_size`
    /// 4, 8 heads, 256-wide heads, `engram_vocab_size` 16M. The row counts differ
    /// because the prime sequence is shared across layers.
    func publishedSpecs() -> [DeepSeekV41EngramSpec] {
        [1, 14].map { layer in
            DeepSeekV41EngramSpec(
                layer: layer,
                rowCount: layer == 1 ? 384_006_168 : 384_016_682,
                maxNgramSize: 4,
                headCount: 8,
                headDimension: 256,
                padTokenID: 2,
                compressedVocabSize: 99_092,
                bucketStart: 16_000_000
            )
        }
    }

    func publishedLayouts() throws -> [Int: DeepSeekV41EngramLayout] {
        try DeepSeekV41EngramLayout.make(specs: publishedSpecs())
    }

    /// Reference vector: the two prime ranges must address exactly the published
    /// `engram_num_embeddings`, layer for layer. The second table's count is what
    /// proves the prime sequence is drawn ONCE across layers: restarting per table
    /// would land on 384_006_168 twice.
    func testBucketLayoutReproducesPublishedRowCounts() throws {
        let layouts = try publishedLayouts()
        let first = try XCTUnwrap(layouts[1])
        let second = try XCTUnwrap(layouts[14])
        XCTAssertEqual(first.hashColumnCount, 24)
        XCTAssertEqual(first.addressedRows, 384_006_168)
        XCTAssertEqual(second.addressedRows, 384_016_682)
        XCTAssertEqual(first.bucketSizes.first, 16_000_057)
        XCTAssertEqual(
            first.bucketSizes,
            [16_000_057, 16_000_079, 16_000_081, 16_000_097, 16_000_121, 16_000_129,
             16_000_133, 16_000_183, 16_000_189, 16_000_207, 16_000_211, 16_000_253,
             16_000_277, 16_000_289, 16_000_307, 16_000_321, 16_000_339, 16_000_381,
             16_000_393, 16_000_399, 16_000_403, 16_000_409, 16_000_447, 16_000_463]
        )
        // The second table continues the sequence rather than restarting it.
        XCTAssertEqual(
            Array(second.bucketSizes.prefix(3)), [16_000_477, 16_000_487, 16_000_499]
        )
        XCTAssertEqual(second.bucketSizes.last, 16_000_889)
        let all = first.bucketSizes + second.bucketSizes
        XCTAssertEqual(Set(all).count, all.count, "bucket ranges must not share a prime")
        XCTAssertTrue(all.allSatisfy(DeepSeekV41Primes.isPrime))
    }

    func testBucketRangesAreDisjointAndAscending() throws {
        let layouts = try publishedLayouts()
        for layout in layouts.values {
            XCTAssertEqual(layout.bucketOffsets.first, 0)
            for index in 1..<layout.bucketOffsets.count {
                let previousEnd = layout.bucketOffsets[index - 1] + layout.bucketSizes[index - 1]
                XCTAssertGreaterThanOrEqual(layout.bucketOffsets[index], previousEnd)
            }
        }
    }

    func testRowBytesAndTableBytes() {
        let spec = publishedSpecs()[0]
        XCTAssertEqual(spec.rowBytes, 256 + 8)
        XCTAssertEqual(spec.totalBytes, 384_006_168 * 264)
    }

    func testTableTooSmallForItsBucketsIsRejected() {
        let spec = DeepSeekV41EngramSpec(
            layer: 1, rowCount: 1_000, maxNgramSize: 4, headCount: 8,
            headDimension: 256, padTokenID: 2, compressedVocabSize: 99_092,
            bucketStart: 16_000_000
        )
        XCTAssertThrowsError(try DeepSeekV41EngramLayout.make(specs: [spec]))
    }

    func testColumnAndByteAddressing() throws {
        let layout = try XCTUnwrap(try publishedLayouts()[1])
        XCTAssertEqual(layout.column(ngramSize: 2, head: 0), 0)
        XCTAssertEqual(layout.column(ngramSize: 2, head: 7), 7)
        XCTAssertEqual(layout.column(ngramSize: 4, head: 7), 23)
        XCTAssertEqual(layout.byteOffset(column: 0, bucket: 0), 0)
        XCTAssertEqual(layout.byteOffset(column: 1, bucket: 0), layout.bucketSizes[0] * 264)
        XCTAssertEqual(layout.byteOffset(column: 0, bucket: 5), 5 * 264)
    }

    /// The hash ids are pinned against the reference in
    /// `DeepSeekV41EngramTokenMapTests`, and the combination rule next to it in
    /// `DeepSeekV41EngramGoldenTests`. Both are what these gates record.
    func testHashConformanceFollowsTheCheckedInVectors() {
        XCTAssertEqual(DeepSeekV41EngramHasher.conformance, .verifiedAgainstReference)
        XCTAssertEqual(DeepSeekV41Conformance.engramCombination, .verifiedAgainstReference)
    }

    func testSyntheticTokenMapCompression() throws {
        let map = try DeepSeekV41EngramTokenMap.synthetic(
            vocabSize: 129_280, compressedVocabSize: 99_092, padTokenID: 2
        )
        XCTAssertEqual(map.compressedToken(2), 2)
        XCTAssertEqual(map.compressedToken(99_092 + 7), 7)
        XCTAssertEqual(map.compressedToken(-1), 2)
        XCTAssertEqual(map.compressedToken(10_000_000), 2)
    }

    /// A compiled map whose largest id disagrees with the size the tokenizer
    /// reported would change every multiplier, so it is refused.
    func testTokenMapRejectsASizeThatDisagreesWithItsLookup() {
        XCTAssertThrowsError(
            try DeepSeekV41EngramTokenMap(lookup: [0, 1, 2], compressedVocabSize: 5, padID: 0)
        )
        XCTAssertThrowsError(
            try DeepSeekV41EngramTokenMap(lookup: [0, 1, 2], compressedVocabSize: 3, padID: 9)
        )
    }

    func testRowRequestsPerToken() throws {
        let layout = try XCTUnwrap(try publishedLayouts()[1])
        let map = try DeepSeekV41EngramTokenMap.synthetic(
            vocabSize: 129_280, compressedVocabSize: 99_092, padTokenID: 2
        )
        let history = [Int32(11), Int32(22), Int32(33)]
        let requests = DeepSeekV41Engram.rowRequests(
            layer: 1, history: history, tokenMap: map, layout: layout
        )
        // (max_ngram_size - 1) n-gram sizes x n_heads.
        XCTAssertEqual(requests.count, 24)
        XCTAssertEqual(requests.map(\.column), Array(0..<24))
        for request in requests {
            XCTAssertEqual(request.bucket, request.row - layout.bucketOffsets[request.column])
            XCTAssertTrue(request.row >= 0 && request.row < layout.spec.rowCount)
            XCTAssertEqual(request.byteOffset, request.row * layout.spec.rowBytes)
            XCTAssertEqual(request.byteCount, 264)
        }
        // Deterministic for the same token history, and sensitive to it.
        XCTAssertEqual(
            requests,
            DeepSeekV41Engram.rowRequests(layer: 1, history: history, tokenMap: map, layout: layout)
        )
        XCTAssertNotEqual(
            requests,
            DeepSeekV41Engram.rowRequests(
                layer: 1, history: [Int32(11), Int32(22), Int32(34)], tokenMap: map, layout: layout
            )
        )
        // A layer without this table issues no reads.
        XCTAssertTrue(
            DeepSeekV41Engram.rowRequests(layer: 2, history: history, tokenMap: map, layout: layout).isEmpty
        )
    }

    /// The hash is `XOR` of `token * multiplier` over the look-backs, reduced mod
    /// the prime that owns (n-gram size, head), plus that column's row offset. The
    /// vectors below are the checkpoint's own arithmetic on a two-column fixture:
    /// multiplier 1 for the current token, 1 for the previous one, then buckets
    /// `(7 ^ 5) % 11 = 10` for the 2-gram and `(7 ^ 5 ^ 3) % 13 = 5` for the 3-gram.
    func testHashFollowsTheReferenceRule() throws {
        let spec = DeepSeekV41EngramSpec(
            layer: 1, rowCount: 64, maxNgramSize: 3, headCount: 1,
            headDimension: 4, padTokenID: 0, compressedVocabSize: 2, bucketStart: 11
        )
        let layout = try XCTUnwrap(try DeepSeekV41EngramLayout.make(specs: [spec])[1])
        XCTAssertEqual(layout.bucketSizes, [11, 13])
        XCTAssertEqual(layout.bucketOffsets, [0, 11])

        let map = try DeepSeekV41EngramTokenMap.synthetic(
            vocabSize: 8, compressedVocabSize: 2, padTokenID: 0
        )
        // `id % 2` maps 3, 5 and 7 all onto 1, so every look-back contributes its
        // own multiplier and nothing else.
        let ids = DeepSeekV41EngramHasher.hashIDs(
            history: [Int32(3), Int32(5), Int32(7)], dead: [], tokenMap: map, layout: layout
        )
        XCTAssertEqual(ids.count, 2)
        let bigram = layout.multipliers[0] ^ layout.multipliers[1]
        let trigram = bigram ^ layout.multipliers[2]
        XCTAssertEqual(ids[0], Int(bigram % 11))
        XCTAssertEqual(ids[1], 11 + Int(trigram % 13))
    }

    /// Every look-back shares the running value, so the 3-gram hash is not a hash
    /// of its own window: it extends the 2-gram. Recomputing each size from a
    /// separate window is the mistake this pins.
    func testLongerNgramsExtendTheShorterOnes() throws {
        let spec = DeepSeekV41EngramSpec(
            layer: 1, rowCount: 64, maxNgramSize: 3, headCount: 1,
            headDimension: 4, padTokenID: 0, compressedVocabSize: 8, bucketStart: 11
        )
        let layout = try XCTUnwrap(try DeepSeekV41EngramLayout.make(specs: [spec])[1])
        let map = try DeepSeekV41EngramTokenMap.synthetic(
            vocabSize: 8, compressedVocabSize: 8, padTokenID: 0
        )
        let ids = DeepSeekV41EngramHasher.hashIDs(
            history: [Int32(2), Int32(4), Int32(6)], dead: [], tokenMap: map, layout: layout
        )
        // 2-gram: 6*m0 ^ 4*m1, the current token and the one before it.
        let bigram = Int64(6) * layout.multipliers[0] ^ Int64(4) * layout.multipliers[1]
        XCTAssertEqual(ids[0], Int(bigram % 11))
        // 3-gram: that value XOR 2*m2, exactly as the reference folds it in.
        let trigram = bigram ^ Int64(2) * layout.multipliers[2]
        XCTAssertEqual(ids[1], 11 + Int(trigram % 13))
    }

    /// Past the start of the sequence, and past a dead token, every look-back hashes
    /// as the pad token. A dead token therefore cannot be spanned by an n-gram.
    func testLookBackStopsAtTheSequenceStartAndAtDeadTokens() throws {
        let spec = DeepSeekV41EngramSpec(
            layer: 1, rowCount: 64, maxNgramSize: 3, headCount: 1,
            headDimension: 4, padTokenID: 0, compressedVocabSize: 8, bucketStart: 11
        )
        let layout = try XCTUnwrap(try DeepSeekV41EngramLayout.make(specs: [spec])[1])
        let map = try DeepSeekV41EngramTokenMap.synthetic(
            vocabSize: 8, compressedVocabSize: 8, padTokenID: 0
        )
        // One token of history: the 3-gram's second and third look-backs are pad (0).
        let short = DeepSeekV41EngramHasher.hashIDs(
            history: [Int32(6)], dead: [], tokenMap: map, layout: layout
        )
        XCTAssertEqual(short[0], Int((Int64(6) * layout.multipliers[0]) % 11))
        XCTAssertEqual(short[1], 11 + Int((Int64(6) * layout.multipliers[0]) % 13))

        // A dead token at the previous position blocks the 3-gram's second look-back,
        // so only the 2-gram contributes a real token.
        let blocked = DeepSeekV41EngramHasher.hashIDs(
            history: [Int32(2), Int32(4), Int32(6)],
            dead: [false, true, false],
            tokenMap: map,
            layout: layout
        )
        XCTAssertEqual(blocked[0], Int((Int64(6) * layout.multipliers[0]) % 11))
        XCTAssertEqual(
            blocked[1],
            11 + Int((Int64(6) * layout.multipliers[0]) % 13),
            "a dead look-back must hash as pad, not skip to the token behind it"
        )
    }
}

// MARK: - Tensor catalog

final class DeepSeekV41WeightCatalogTests: XCTestCase {

    func tensor(_ name: String, _ dtype: String = "F8_E4M3", _ shape: [Int] = [64, 64]) -> DeepSeekV41TensorRef {
        DeepSeekV41TensorRef(name: name, dtype: dtype, shape: shape)
    }

    func testModelLevelTensorsClassify() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                tensor("embed.weight", "BF16", [32, 64]),
                tensor("norm.weight", "BF16", [64]),
                tensor("head.weight", "BF16", [32, 64])
            ],
            layerCount: 4
        )
        XCTAssertEqual(catalog.tensor(.tokenEmbedding)?.name, "embed.weight")
        XCTAssertEqual(catalog.tensor(.finalNorm)?.name, "norm.weight")
        XCTAssertEqual(catalog.tensor(.lmHead)?.name, "head.weight")
        XCTAssertTrue(catalog.unconsumed.isEmpty)
    }

    func testLayerTensorsClassifyWithoutAModelPrefix() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                tensor("layers.3.attn.wq_a.weight", "F8_E4M3", [16, 64]),
                tensor("layers.3.attn.wq_a_norm.weight", "BF16", [16]),
                tensor("layers.3.attn.kv_norm.weight", "BF16", [16]),
                tensor("layers.3.attn.attn_sink", "F32", [8]),
                tensor("layers.3.ffn.gate.bias", "F32", [2]),
                tensor("layers.3.ffn.gate.bias_vl", "F32", [2]),
                tensor("layers.3.engram.embed.weight", "F8_E4M3", [1000, 8]),
                tensor("layers.3.engram.embed.scale", "F8_E8M0", [1000, 1])
            ],
            layerCount: 4
        )
        XCTAssertEqual(catalog.tensor(.attnWqA, layer: 3)?.shape, [16, 64])
        XCTAssertEqual(catalog.tensor(.attnWqANorm, layer: 3)?.dtype, "BF16")
        XCTAssertEqual(catalog.tensor(.attnKvNorm, layer: 3)?.dtype, "BF16")
        XCTAssertEqual(catalog.tensor(.attnSink, layer: 3)?.dtype, "F32")
        XCTAssertNotNil(catalog.tensor(.routerBias, layer: 3))
        XCTAssertNotNil(catalog.tensor(.routerVisionBias, layer: 3))
        XCTAssertNotNil(catalog.tensor(.engramTable, layer: 3))
        XCTAssertEqual(catalog.tensor(.engramTable, layer: 3, scale: true)?.dtype, "F8_E8M0")
        // The table and its scale both live in the streamed blob.
        XCTAssertEqual(catalog.streamedTensorNames.count, 2)
        XCTAssertTrue(catalog.unconsumed.isEmpty)
    }

    func testExpertScaleSuffixSpellings() {
        for suffix in ["weight_scale", "scale", "scales"] {
            let catalog = DeepSeekV41WeightCatalog(
                named: [tensor("layers.0.ffn.experts.7.w2.\(suffix)", "F8_E4M3", [2, 1])],
                layerCount: 4
            )
            XCTAssertNotNil(
                catalog.tensor(.expertW2, layer: 0, expert: 7, scale: true),
                "suffix \(suffix) should be recognised as a scale"
            )
        }
    }

    /// Indexer weights follow the reference's ownership split: only an index-source
    /// layer publishes them, and inside one only a layer that compresses its own KV
    /// owns the keys. Demanding the whole set on every compressed layer rejected the
    /// real checkpoint, which publishes no indexer tensors on its 30 reader layers
    /// and no `wk`/`k_norm` on the four index sources that are not KV sources.
    func testIndexerRolesFollowTheSources() throws {
        let json = DeepSeekV41ConfigTests.wrapperJSON
            .replacingOccurrences(
                of: "\"index_source_layer_ids\": [2, 20],",
                with: "\"index_source_layer_ids\": [2, 20, 24],"
            )
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        let plan = try config.makeLayerPlan()
        let byIndex = Dictionary(uniqueKeysWithValues: plan.map { ($0.index, $0) })

        XCTAssertEqual(
            DeepSeekV41WeightCatalog.requiredIndexerRoles(layer: byIndex[20]!, config: config),
            [.indexerWqB, .indexerWeights, .indexerWk, .indexerKNorm]
        )
        XCTAssertEqual(
            DeepSeekV41WeightCatalog.requiredIndexerRoles(layer: byIndex[24]!, config: config),
            [.indexerWqB, .indexerWeights]
        )
        XCTAssertEqual(
            DeepSeekV41WeightCatalog.requiredIndexerRoles(layer: byIndex[3]!, config: config),
            []
        )
    }

    func testDraftBlockTensorsClassify() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                tensor("mtp.0.main_proj.weight", "BF16", [64, 64]),
                tensor("mtp.2.norm.weight", "BF16", [64]),
                tensor("mtp.1.markov_head.head.weight", "BF16", [32, 64]),
                tensor("mtp.0.confidence_head.proj.weight", "BF16", [1, 64]),
                tensor("mtp.1.ffn.experts.5.w3.weight", "I8", [32, 64])
            ],
            layerCount: 4
        )
        XCTAssertNotNil(catalog.tensor(.draftMainProj, scope: .draft, layer: 0))
        XCTAssertNotNil(catalog.tensor(.draftNorm, scope: .draft, layer: 2))
        XCTAssertNotNil(catalog.tensor(.draftMarkovHead, scope: .draft, layer: 1))
        XCTAssertNotNil(catalog.tensor(.draftConfidence, scope: .draft, layer: 0))
        XCTAssertNotNil(catalog.tensor(.expertW3, scope: .draft, layer: 1, expert: 5))
        XCTAssertTrue(catalog.unconsumed.isEmpty)
    }

    func testUnknownTensorIsReportedAsUnconsumed() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [tensor("layers.0.ffn.surprise_gate.weight", "BF16", [8, 64])],
            layerCount: 4
        )
        XCTAssertEqual(catalog.unconsumed, ["layers.0.ffn.surprise_gate.weight"])
    }

    func testUnsupportedDtypeIsRefused() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [tensor("layers.0.attn.wq_a.weight", "Q4_G32")], layerCount: 4
        )
        let ref = catalog.tensor(.attnWqA, layer: 0)
        XCTAssertNotNil(ref)
        XCTAssertFalse(ref?.isSupportedDtype ?? true)
    }

    /// The published checkpoint ships `attn.compressor.norm` and
    /// `attn.indexer.k_norm`. Both are referenced by the reference implementation
    /// (`Compressor` normalises its pooled row, `Indexer` normalises its key) and
    /// neither used to classify, so a real load reported them as unconsumed.
    func testCompressorAndIndexerGainsClassify() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                tensor("layers.2.attn.compressor.norm.weight", "BF16", [512]),
                tensor("layers.2.attn.indexer.k_norm.weight", "BF16", [128])
            ],
            layerCount: 40
        )
        XCTAssertEqual(catalog.tensor(.compressorNorm, layer: 2)?.shape, [512])
        XCTAssertEqual(catalog.tensor(.indexerKNorm, layer: 2)?.shape, [128])
        XCTAssertTrue(catalog.unconsumed.isEmpty, "\(catalog.unconsumed)")
        // Both are per-channel gains, not quantized linears: a `.scale` companion
        // would mean the classifier split the name wrongly.
        XCTAssertEqual(catalog.streamedTensorNames.count, 0)
    }

    /// `engram.embed` is the n-gram table (one ~98 GB row block); `engram.wkv` is
    /// a 25 600 x 6 144 projection over the retrieved rows. Mapping the
    /// projection as the table would hand ~150 MB to a reader that preads rows
    /// out of a table, so the two must never swap.
    func testEngramTableIsTheEmbeddingAndWkvIsAProjection() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                tensor("layers.1.engram.embed.weight", "F8_E4M3", [384_006_168, 256]),
                tensor("layers.1.engram.embed.scale", "F8_E8M0", [384_006_168, 8]),
                tensor("layers.1.engram.wkv.weight", "F8_E4M3", [25_600, 6_144]),
                tensor("layers.1.engram.wkv.scale", "F8_E8M0", [800, 192]),
                tensor("layers.1.engram.q_weight", "BF16", [4, 5_120]),
                tensor("layers.1.engram.k_weight", "BF16", [4, 5_120])
            ],
            layerCount: 40
        )
        XCTAssertEqual(catalog.tensor(.engramTable, layer: 1)?.name, "layers.1.engram.embed.weight")
        XCTAssertEqual(catalog.tensor(.engramWkv, layer: 1)?.name, "layers.1.engram.wkv.weight")
        XCTAssertNotNil(catalog.tensor(.engramWkv, layer: 1, scale: true))
        XCTAssertEqual(catalog.tensor(.engramQuery, layer: 1)?.shape, [4, 5_120])
        // Only the table and its scale are streamed; the projection is a dense
        // matrix the loader may map.
        XCTAssertEqual(
            catalog.streamedTensorNames,
            ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale"]
        )
    }

    /// A DSpark stage is a full backbone block under `mtp.N.*`. Its tensors must
    /// land in the draft namespace: `mtp.0.attn.wkv` and `layers.0.attn.wkv` are
    /// different weights, and a shared slot would drop one of them.
    func testDraftStagesCarryBackboneBlocksInTheirOwnNamespace() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                tensor("layers.0.attn.wkv.weight", "F8_E4M3", [512, 5120]),
                tensor("layers.0.attn.wkv.scale", "F8_E8M0", [16, 160]),
                tensor("mtp.0.attn.wkv.weight", "F8_E4M3", [512, 5120]),
                tensor("mtp.0.attn.wkv.scale", "F8_E8M0", [16, 160]),
                tensor("mtp.0.attn.wq_b.weight", "F8_E4M3", [32768, 1280]),
                tensor("mtp.0.ffn.gate.bias", "F32", [384]),
                tensor("mtp.0.hc_attn_fn", "BF16", [4, 4])
            ],
            layerCount: 40
        )
        XCTAssertEqual(catalog.unconsumed, [])
        XCTAssertEqual(catalog.duplicateRoles.count, 0, "\(catalog.duplicateRoles)")
        XCTAssertEqual(catalog.tensor(.attnWkv, layer: 0)?.name, "layers.0.attn.wkv.weight")
        XCTAssertEqual(
            catalog.tensor(.attnWkv, scope: .draft, layer: 0)?.name, "mtp.0.attn.wkv.weight"
        )
        XCTAssertEqual(
            catalog.tensor(.attnWkv, scope: .draft, layer: 0, scale: true)?.name,
            "mtp.0.attn.wkv.scale"
        )
        XCTAssertNotNil(catalog.tensor(.attnWqB, scope: .draft, layer: 0))
        XCTAssertNotNil(catalog.tensor(.routerBias, scope: .draft, layer: 0))
        XCTAssertNotNil(catalog.tensor(.hcAttnFn, scope: .draft, layer: 0))
        // Draft blocks are not backbone layers, so a block index must never be
        // range-checked against the backbone layer count.
        XCTAssertEqual(catalog.outOfRangeLayers, [])
    }

    /// The vision tower, its projector, and the three image markers ship in the
    /// same checkpoint. None of them classifies into the text graph, but leaving
    /// them unconsumed would refuse a text-only load of a complete checkpoint.
    func testVisionTowerTensorsDoNotReadAsUnconsumed() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                tensor("vision.patch_embed.proj.weight", "BF16", [1_024, 3, 16, 16]),
                tensor("vision.blocks.3.attn.wqkv.weight", "BF16", [3_072, 1_024]),
                tensor("vision.norm.weight", "BF16", [1_024]),
                tensor("aligner.w1.weight", "BF16", [5_120, 4_096]),
                tensor("image_start", "BF16", [5_120]),
                tensor("image_newline", "BF16", [5_120]),
                tensor("image_end", "BF16", [5_120])
            ],
            layerCount: 40
        )
        XCTAssertEqual(catalog.unconsumed, [])
        XCTAssertEqual(catalog.slots.filter { $0.role.isVision }.count, 7)
        // Each tower tensor gets its own role, because the forward has to address
        // `wqkv` separately from `wo` in the same block.
        XCTAssertEqual(
            catalog.slots.filter { $0.role == .visionBlockWqkv && $0.layer == 3 }.count, 1
        )
        XCTAssertEqual(catalog.slots.filter { $0.role == .visionPatchEmbed }.count, 1)
        XCTAssertEqual(catalog.slots.filter { $0.role == .visionFinalNorm }.count, 1)
        XCTAssertEqual(catalog.slots.filter { $0.role == .visionAlignerW1 }.count, 1)
        XCTAssertEqual(catalog.slots.filter { $0.role == .visionImageStart }.count, 1)
        XCTAssertEqual(catalog.slots.filter { $0.role == .visionImageNewline }.count, 1)
        XCTAssertEqual(catalog.slots.filter { $0.role == .visionImageEnd }.count, 1)
        // A vision block index is not a backbone layer index.
        XCTAssertEqual(catalog.outOfRangeLayers, [])
    }

    /// Every tensor the published checkpoint ships for the tower must classify,
    /// with no two of them sharing a slot. A collapse here would leave the tower
    /// unable to address its own weights.
    func testEveryPublishedVisionTensorClassifiesDistinctly() {
        var names = ["vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias",
                     "vision.norm.weight",
                     "aligner.w1.weight", "aligner.w1.bias",
                     "aligner.w2.weight", "aligner.w2.bias",
                     "image_start", "image_newline", "image_end"]
        for stage in ["norm1.weight", "attn.wqkv.weight", "attn.wqkv.bias",
                      "attn.wo.weight", "attn.wo.bias", "norm2.weight",
                      "mlp.w1.weight", "mlp.w2.weight"] {
            names.append("vision.blocks.7.\(stage)")
        }
        let catalog = DeepSeekV41WeightCatalog(named: names.map { tensor($0, "BF16", [8, 8]) }, layerCount: 40)
        XCTAssertEqual(catalog.unconsumed, [], "every tower tensor must classify")
        let keys = Set(catalog.slots.map(\.key))
        XCTAssertEqual(keys.count, names.count, "two tower tensors share a slot")
    }
}

// MARK: - Conformance gates

/// Pins every gate. A gate that flips without the reference vectors and a test
/// update would otherwise be invisible: the load log is the only place it shows.
final class DeepSeekV41ConformanceGateTests: XCTestCase {

    func testOnlyTheRulesWithReferenceVectorsAreVerified() {
        // Checked-in vectors, from `Tools/dsv41_goldens.py` and its test suites:
        // Engram indexing, the Engram combination, the compressor's pooling and the
        // router's scoring.
        XCTAssertEqual(DeepSeekV41EngramHasher.conformance, .verifiedAgainstReference)
        XCTAssertEqual(DeepSeekV41Conformance.engramCombination, .verifiedAgainstReference)
        XCTAssertEqual(DeepSeekV41Conformance.compressorPooling, .verifiedAgainstReference)
        XCTAssertEqual(DeepSeekV41Conformance.routerScoring, .verifiedAgainstReference)
        // Still transcribed-only: no golden has been generated for these yet.
        XCTAssertEqual(DeepSeekV41Conformance.candidateSelection, .unverified)
        XCTAssertEqual(DeepSeekV41Conformance.hyperConnections, .unverified)
        XCTAssertEqual(DeepSeekV41Conformance.dsparkDrafting, .unverified)
        XCTAssertEqual(DeepSeekV41Conformance.visionPatching, .unverified)
        XCTAssertEqual(DeepSeekV41Conformance.unverifiedSummary.count, 4)
        XCTAssertFalse(DeepSeekV41Conformance.unverifiedSummary.contains("Engram hashing"))
        XCTAssertFalse(DeepSeekV41Conformance.unverifiedSummary.contains("Engram combination"))
        XCTAssertFalse(DeepSeekV41Conformance.unverifiedSummary.contains("compressor pooling"))
        XCTAssertFalse(DeepSeekV41Conformance.unverifiedSummary.contains("router scoring"))
    }
}

// MARK: - Preflight

final class DeepSeekV41PreflightTests: XCTestCase {

    /// A valid safetensors file with a readable header and `dataBytes` of data.
    private func writeShard(at url: URL, dataBytes: Int) throws {
        let header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[\(dataBytes)],"
            + "\"data_offsets\":[0,\(dataBytes)]}}"
        var data = Data()
        var length = UInt64(header.utf8.count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(contentsOf: header.utf8)
        data.append(Data(repeating: 1, count: dataBytes))
        try data.write(to: url)
    }

    private func shardSize(dataBytes: Int) -> Int {
        let header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[\(dataBytes)],"
            + "\"data_offsets\":[0,\(dataBytes)]}}"
        return 8 + header.utf8.count + dataBytes
    }

    func testIncompleteDownloadIsReported() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-preflight-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let presentSize = shardSize(dataBytes: 1_000)
        let index = """
        {"metadata": {"total_size": \(presentSize * 2)},
         "weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}
        """
        try index.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )
        try writeShard(
            at: dir.appendingPathComponent("model-00001-of-00002.safetensors"),
            dataBytes: 1_000
        )

        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertEqual(report.shardCount, 2)
        XCTAssertEqual(report.presentShardCount, 1)
        XCTAssertEqual(report.bytesPresent, presentSize)
        XCTAssertEqual(report.bytesExpected, presentSize * 2)
        XCTAssertFalse(report.isComplete)
        XCTAssertEqual(report.issues.map(\.name), ["model-00002-of-00002.safetensors"])
        XCTAssertTrue(report.humanSummary.hasPrefix("1/2 shards"), report.humanSummary)
        XCTAssertTrue(report.humanSummary.contains("50%"), report.humanSummary)
    }

    func testCompleteDownloadReportsComplete() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-preflight-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let index = """
        {"metadata": {"total_size": 10},
         "weight_map": {"a": "model-00001-of-00001.safetensors"}}
        """
        try index.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )
        try writeShard(
            at: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            dataBytes: 10
        )

        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertTrue(report.isComplete, "\(report.issues)")
        XCTAssertTrue(report.issues.isEmpty)
    }

    func testPackedRepackHasNothingToCheck() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-preflight-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(DeepSeekV41Preflight.inspect(snapshotDir: dir))
    }
}

// MARK: - Memory plan

final class DeepSeekV41MemoryPlanTests: XCTestCase {

    func wrapperConfig() throws -> DeepSeekV41Config {
        try DeepSeekV41Config.decode(jsonData: Data(DeepSeekV41ConfigTests.wrapperJSON.utf8))
    }

    /// The headline fact about this architecture: the KV cache is not the
    /// constraint. Two compressed caches at 288 bytes/token is 576 MB for a
    /// million tokens, and the window rings are bounded by `window_size`.
    func testCachesStayUnderTwoGigabytesAtOneMillionTokens() throws {
        let config = try wrapperConfig()
        let plan = try config.makeLayerPlan()
        let memory = DeepSeekV41MemoryPlan.make(
            config: config, plan: plan, catalog: nil, contextLength: 1_048_576
        )
        // Each owner is priced from its own ratio: layer 2 is ratio 2 and layer
        // 20 is ratio 1, and each has one trailing partial-block row.
        XCTAssertEqual(
            memory.compressedCacheBytes,
            ((1_048_576 / 2 + 1) + (1_048_576 + 1)) * 288
        )
        // The rings are priced at the same record width as the compressed caches,
        // which is the FP4 design width by default here.
        XCTAssertEqual(memory.windowRingBytes, 40 * 128 * config.fp4CacheRecordBytes)
        // Main caches (576 MB) plus one index key per compressed row (two
        // owners x 128 FP16 values per row) stays well under 2 GB, which is
        // what makes a 1M-token context affordable on this machine.
        // Index keys are per COMPRESSED row, so the ratio-2 owner stores half
        // as many as the ratio-1 owner.
        XCTAssertEqual(
            memory.indexerBytes,
            (1_048_576 / 2 + 1) * 128 * 2 + (1_048_576 + 1) * 128 * 2
        )
        XCTAssertLessThan(memory.cacheBytes, 2 * 1_073_741_824, "caches must fit in 2 GB")
    }

    func testExpertArenaAndStagingAreSmall() throws {
        let config = try wrapperConfig()
        let plan = try config.makeLayerPlan()
        let memory = DeepSeekV41MemoryPlan.make(
            config: config, plan: plan, catalog: nil, contextLength: 8_192
        )
        // 3 * 7168 * 2048 values at MXFP4 width (two values per byte plus one
        // E8M0 scale per 32).
        // MXFP4 stores two values per byte plus one E8M0 scale per 32.
        let values = 3 * 7_168 * 2_048
        XCTAssertEqual(memory.bytesPerExpert, Int((Double(values) * 0.53125).rounded(.up)))
        XCTAssertLessThan(memory.bytesPerExpert, 64 * 1024 * 1024)
        // 64 staged rows of both tables, double buffered.
        XCTAssertEqual(memory.engramStagingBytes, 64 * (264 + 264) * 2)
    }

    /// The arena is a cache, not a requirement: what matters is that a big machine
    /// holds a working set rather than the `numActivatedExperts * 2` placeholder.
    func testAffordableExpertSlotsScaleWithRam() throws {
        let config = try wrapperConfig()
        let plan = try config.makeLayerPlan()
        let memory = DeepSeekV41MemoryPlan.make(
            config: config, plan: plan, catalog: nil, contextLength: 4_096
        )
        let slots = DeepSeekV41MemoryPlan.affordableExpertSlots(plan: memory, physicalGB: 128)
        // 0.6 * 128 GB over a ~23 MB expert is a few thousand pairs, and the
        // checkpoint publishes 384 * 40 = 15,360 of them.
        XCTAssertGreaterThan(slots, 1_000, "128 GB must hold a working set")
        XCTAssertLessThan(slots, config.numRoutedExperts * config.numHiddenLayers)
        XCTAssertLessThan(
            DeepSeekV41MemoryPlan.affordableExpertSlots(plan: memory, physicalGB: 16),
            slots
        )
        XCTAssertEqual(DeepSeekV41MemoryPlan.affordableExpertSlots(plan: memory, physicalGB: 0), 1)
    }

    func testRequiredBytesGrowsWithResidentExperts() throws {
        let config = try wrapperConfig()
        let plan = try config.makeLayerPlan()
        let memory = DeepSeekV41MemoryPlan.make(
            config: config, plan: plan, catalog: nil, contextLength: 32_768
        )
        XCTAssertEqual(
            memory.requiredBytes(expertSlots: 32) - memory.requiredBytes(expertSlots: 16),
            16 * memory.bytesPerExpert
        )
        XCTAssertTrue(memory.describe(expertSlots: 32).contains("total"))
    }

    /// Dense weights come from the catalog, so a checkpoint with extra tensors
    /// is counted rather than assumed.
    func testResidentWeightsComeFromTheCatalog() throws {
        let config = try wrapperConfig()
        let plan = try config.makeLayerPlan()
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                DeepSeekV41TensorRef(name: "embed.weight", dtype: "BF16", shape: [1000, 64]),
                DeepSeekV41TensorRef(name: "norm.weight", dtype: "BF16", shape: [64]),
                DeepSeekV41TensorRef(name: "head.weight", dtype: "F8_E4M3", shape: [1000, 64]),
                // Expert and streamed tensors must not be counted as resident.
                DeepSeekV41TensorRef(name: "layers.0.ffn.experts.0.w1.weight", dtype: "I8", shape: [64, 64]),
                DeepSeekV41TensorRef(name: "layers.1.engram.embed.weight", dtype: "F8_E4M3", shape: [64, 8])
            ],
            layerCount: 2
        )
        let memory = DeepSeekV41MemoryPlan.make(
            config: config, plan: plan, catalog: catalog, contextLength: 1_024
        )
        let expected = 1000 * 64 * 2
            + 64 * 2
            + Int((Double(1000 * 64) * (1.0 + 1.0 / 32.0)).rounded(.up))
        XCTAssertEqual(memory.residentWeightBytes, expected)
    }
}

// MARK: - Engram reader

final class DeepSeekV41EngramReaderTests: XCTestCase {

    /// A small but structurally real table: 4 hash columns, 9 bytes per row.
    func spec(rowCount: Int = 4_096) -> DeepSeekV41EngramSpec {
        DeepSeekV41EngramSpec(
            layer: 1, rowCount: rowCount, maxNgramSize: 3, headCount: 2,
            headDimension: 8, padTokenID: 2, compressedVocabSize: 64, bucketStart: 20
        )
    }

    /// Writes a table whose row `r` starts with its little-endian row index and
    /// is filled with byte `r & 0xFF`, so a misread row is identifiable.
    func makeTable(rows: Int, rowBytes: Int) throws -> URL {
        var data = Data(capacity: rows * rowBytes)
        for row in 0..<rows {
            var bytes = [UInt8](repeating: UInt8(row & 0xFF), count: rowBytes)
            let index = UInt32(row)
            bytes[0] = UInt8(truncatingIfNeeded: index)
            bytes[1] = UInt8(truncatingIfNeeded: index >> 8)
            bytes[2] = UInt8(truncatingIfNeeded: index >> 16)
            bytes[3] = UInt8(truncatingIfNeeded: index >> 24)
            data.append(contentsOf: bytes)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-engram-\(UUID().uuidString).bin")
        try data.write(to: url)
        return url
    }

    func testReadsTheRowsTheLayoutAddresses() throws {
        let spec = spec()
        let layout = try XCTUnwrap(try DeepSeekV41EngramLayout.make(specs: [spec])[1])
        let url = try makeTable(rows: spec.rowCount, rowBytes: spec.rowBytes)
        defer { try? FileManager.default.removeItem(at: url) }

        let map = try DeepSeekV41EngramTokenMap.synthetic(
            vocabSize: 128, compressedVocabSize: 64, padTokenID: 2
        )
        let requests = DeepSeekV41Engram.rowRequests(
            layer: 1, history: [Int32(5), Int32(9), Int32(17)], tokenMap: map, layout: layout
        )
        XCTAssertEqual(requests.count, 4)

        let reader = try DeepSeekV41EngramReader(fileURL: url, spec: spec)
        defer { reader.closeNow() }
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: requests.count * spec.rowBytes,
            alignment: MemoryLayout<UInt64>.alignment
        )
        defer { buffer.deallocate() }
        let bytesRead = try reader.readRows(requests, into: buffer)
        XCTAssertEqual(bytesRead, requests.count * spec.rowBytes)

        for (index, request) in requests.enumerated() {
            let row = request.byteOffset / spec.rowBytes
            let slice = buffer.advanced(by: index * spec.rowBytes)
                .assumingMemoryBound(to: UInt8.self)
            var encoded: UInt32 = 0
            for byte in 0..<4 {
                encoded |= UInt32(slice[byte]) << (8 * UInt32(byte))
            }
            XCTAssertEqual(Int(encoded), row, "request \(index) read the wrong row")
            XCTAssertEqual(slice[4], UInt8(row & 0xFF))
        }
    }

    /// The table is one tensor inside a shard, so requests are table-relative
    /// while `pread` is file-relative. Forgetting the tensor's offset reads the
    /// shard's own header bytes as if they were memory rows.
    func testReadsAreOffsetByTheTensorsPositionInTheShard() throws {
        let spec = spec()
        let plain = try makeTable(rows: spec.rowCount, rowBytes: spec.rowBytes)
        defer { try? FileManager.default.removeItem(at: plain) }
        let table = try Data(contentsOf: plain)
        let base = 64
        var shard = Data(repeating: 0xAB, count: base)
        shard.append(table)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-engram-shard-\(UUID().uuidString).bin")
        try shard.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try DeepSeekV41EngramReader(fileURL: url, spec: spec, baseOffset: base)
        defer { reader.closeNow() }
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: spec.rowBytes, alignment: 8)
        defer { buffer.deallocate() }
        let row = 3
        let request = DeepSeekV41EngramRowRequest(
            layer: 1, column: 0, bucket: 0, row: row,
            byteOffset: row * spec.rowBytes, byteCount: spec.rowBytes
        )
        _ = try reader.readRows([request], into: buffer)
        let slice = buffer.assumingMemoryBound(to: UInt8.self)
        var encoded: UInt32 = 0
        for byte in 0..<4 {
            encoded |= UInt32(slice[byte]) << (8 * UInt32(byte))
        }
        XCTAssertEqual(Int(encoded), row)
        XCTAssertNotEqual(slice[0], 0xAB)
    }

    func testTruncatedTableIsRefused() throws {
        let spec = spec()
        let url = try makeTable(rows: 8, rowBytes: spec.rowBytes)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try DeepSeekV41EngramReader(fileURL: url, spec: spec))
    }

    func testRowFromAnotherLayerIsRefused() throws {
        let spec = spec()
        let url = try makeTable(rows: spec.rowCount, rowBytes: spec.rowBytes)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try DeepSeekV41EngramReader(fileURL: url, spec: spec)
        defer { reader.closeNow() }
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: spec.rowBytes, alignment: 8)
        defer { buffer.deallocate() }
        let request = DeepSeekV41EngramRowRequest(
            layer: 14, column: 0, bucket: 0, row: 0, byteOffset: 0, byteCount: spec.rowBytes
        )
        XCTAssertThrowsError(try reader.readRows([request], into: buffer)) { error in
            XCTAssertEqual(
                error as? DeepSeekV41EngramReadError,
                .layerMismatch(request: 14, table: 1)
            )
        }
    }

    func testOutOfRangeRowIsRefused() throws {
        let spec = spec()
        let url = try makeTable(rows: spec.rowCount, rowBytes: spec.rowBytes)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try DeepSeekV41EngramReader(fileURL: url, spec: spec)
        defer { reader.closeNow() }
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: spec.rowBytes, alignment: 8)
        defer { buffer.deallocate() }
        let request = DeepSeekV41EngramRowRequest(
            layer: 1, column: 0, bucket: 0, row: 0,
            byteOffset: spec.totalBytes, byteCount: spec.rowBytes
        )
        XCTAssertThrowsError(try reader.readRows([request], into: buffer))
    }
}

// MARK: - Preflight shard sizes

final class DeepSeekV41PreflightSizeTests: XCTestCase {

    private func makeDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-preflight-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Writes a shard whose safetensors header declares more data than the file
    /// carries, which is what a half-finished download looks like.
    private func writeShard(at url: URL, declaredDataBytes: Int, actualDataBytes: Int) throws {
        let header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[\(declaredDataBytes)],"
            + "\"data_offsets\":[0,\(declaredDataBytes)]}}"
        var data = Data()
        var length = UInt64(header.utf8.count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(contentsOf: header.utf8)
        data.append(Data(repeating: 1, count: actualDataBytes))
        try data.write(to: url)
    }

    func testShortShardIsReportedByItsOwnHeader() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let name = "model-00001-of-00001.safetensors"
        try writeShard(at: dir.appendingPathComponent(name), declaredDataBytes: 1_000, actualDataBytes: 100)
        try """
        {"metadata": {"total_size": 1008}, "weight_map": {"a": "\(name)"}}
        """.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )

        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertFalse(report.isComplete)
        let issue = try XCTUnwrap(report.issues.first)
        XCTAssertEqual(issue.name, name)
        // The header declares 1_000 data bytes and only 100 were written.
        let actual = try XCTUnwrap(issue.actualBytes)
        let expected = try XCTUnwrap(issue.expectedBytes)
        XCTAssertEqual(expected, actual + 900)
        XCTAssertTrue(issue.humanDescription.contains("short"), issue.humanDescription)
    }

    /// A zero-length (or header-less) shard is exactly the half-written file the
    /// preflight exists to catch; it must not validate as complete just because
    /// its size could not be compared to a header.
    func testShardWithAnUnreadableHeaderIsReported() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let name = "model-00001-of-00001.safetensors"
        try Data().write(to: dir.appendingPathComponent(name))
        try """
        {"weight_map": {"a": "\(name)"}}
        """.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )
        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertFalse(report.isComplete)
        let issue = try XCTUnwrap(report.issues.first)
        XCTAssertFalse(issue.headerReadable)
        XCTAssertTrue(issue.humanDescription.contains("header unreadable"), issue.humanDescription)
    }

    /// A Hugging Face snapshot is a directory of symlinks into `blobs/`. The
    /// preflight used `attributesOfItem`, which does not follow the link, so a
    /// complete 510 GB download reported "48/48 shards, 4 KB (0%)".
    func testSymlinkedShardIsMeasuredThroughTheLink() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let name = "model-00001-of-00001.safetensors"
        let blob = dir.appendingPathComponent("blob.safetensors")
        try writeShard(at: blob, declaredDataBytes: 1_000, actualDataBytes: 1_000)
        try FileManager.default.createSymbolicLink(
            at: dir.appendingPathComponent(name),
            withDestinationURL: blob
        )
        try """
        {"metadata": {"total_size": 1008}, "weight_map": {"a": "\(name)"}}
        """.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )

        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertTrue(report.isComplete, "a link to a complete blob is complete: \(report.issues)")
        XCTAssertEqual(report.presentShardCount, 1)
        // The link is 21ish bytes; the measurement must be the blob's own size.
        XCTAssertGreaterThan(report.bytesPresent, 1_000, report.humanSummary)
    }

    func testShardMatchingItsHeaderIsComplete() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let name = "model-00001-of-00001.safetensors"
        try writeShard(at: dir.appendingPathComponent(name), declaredDataBytes: 64, actualDataBytes: 64)
        try """
        {"weight_map": {"a": "\(name)"}}
        """.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )
        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertTrue(report.isComplete, "\(report.issues)")
        XCTAssertTrue(report.issues.isEmpty)
    }

    /// A 6 GB shard is past what an `Int32` byte count can hold. The file is
    /// sparse, so the test costs no disk.
    func testByteCountsAreNotTruncatedTo32Bits() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let name = "model-00001-of-00001.safetensors"
        let url = dir.appendingPathComponent(name)
        try Data(repeating: 0, count: 8).write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 6_000_000_000)
        try handle.close()
        try """
        {"metadata": {"total_size": 12000000000}, "weight_map": {"a": "\(name)"}}
        """.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )

        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertEqual(report.bytesPresent, 6_000_000_000)
        XCTAssertEqual(report.bytesExpected, 12_000_000_000)
        XCTAssertTrue(report.humanSummary.contains("50%"), report.humanSummary)
    }
}

// MARK: - Compressor group

/// The compressor's per-position state is per OWNER, not shared.
///
/// It used to be one block addressed by `position % ratio` alone. Owners run in
/// index order at every position, so the earlier slots of a block were last
/// written by whichever owner ran latest at the previous position, and a ratio-2
/// owner pooled its own latent together with another layer's.
final class DeepSeekV41CompressorGroupTests: XCTestCase {

    private func plan(owners: [Int], ratio: Int) -> [DeepSeekV41LayerPlan] {
        let source = owners.first ?? -1
        return (0..<8).map { index in
            let isOwner = owners.contains(index)
            return DeepSeekV41LayerPlan(
                index: index,
                flavor: isOwner ? .compressed : .windowOnly,
                compressRatio: isOwner ? ratio : 0,
                compressedKVSource: isOwner ? index : source,
                compressedIndexSource: -1,
                isCandidateSource: false,
                engramTable: nil,
                ropeTable: .compress
            )
        }
    }

    func testEachOwnerGetsItsOwnBlockInIndexOrder() {
        let slots = DeepSeekV41CacheAddressing.compressorOwnerSlots(
            plan: plan(owners: [5, 1, 3], ratio: 2)
        )
        XCTAssertEqual(slots, [1: 0, 3: 1, 5: 2])
        XCTAssertNil(slots[0], "a layer that owns no compressed rows fills no block")
        XCTAssertNil(slots[2])
    }

    func testTwoOwnersNeverShareASlot() throws {
        let width = 4
        let groupWidth = 2
        let first = try XCTUnwrap(
            DeepSeekV41CacheAddressing.compressorSlotByteOffset(
                owner: 0, slot: 1, groupWidth: groupWidth, width: width
            )
        )
        let second = try XCTUnwrap(
            DeepSeekV41CacheAddressing.compressorSlotByteOffset(
                owner: 1, slot: 0, groupWidth: groupWidth, width: width
            )
        )
        XCTAssertEqual(first, width * MemoryLayout<Float>.stride)
        XCTAssertEqual(second, groupWidth * width * MemoryLayout<Float>.stride)
        XCTAssertNotEqual(first, second, "an owner's block must not overlap the next one's")
    }

    /// A narrow owner's unused slots are the difference between the group width and
    /// its own ratio, and they stay inside its own block.
    func testANarrowOwnerLeavesItsSpareSlotsUnused() {
        XCTAssertNil(
            DeepSeekV41CacheAddressing.compressorSlotByteOffset(
                owner: 0, slot: 2, groupWidth: 2, width: 4
            )
        )
        XCTAssertNotNil(
            DeepSeekV41CacheAddressing.compressorSlotByteOffset(
                owner: 1, slot: 1, groupWidth: 2, width: 4
            )
        )
    }
}

// MARK: - Cache reuse

final class DeepSeekV41CacheReuseTests: XCTestCase {

    func testAStoredPrefixIsReusable() {
        XCTAssertTrue(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 64, storedPositions: 100
            )
        )
    }

    /// The window ring is not a bound, because a resumed turn never replays what it
    /// skips. Every window row a later position reads is one the ring still holds,
    /// and the compressed rows are retained rather than ringed.
    func testAPrefixPastTheWindowRingIsReusable() {
        XCTAssertTrue(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 200, storedPositions: 300
            )
        )
        XCTAssertTrue(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 300, storedPositions: 300
            )
        )
    }

    func testPrefixPastWhatIsStoredIsNotReusable() {
        XCTAssertFalse(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 40, storedPositions: 20
            )
        )
        XCTAssertTrue(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 0, storedPositions: 20
            )
        )
    }

    /// The pin and the cache can disagree in both directions, and only exact
    /// agreement is a skip.
    func testTheReusablePrefixIsTheResidentSequenceExactly() {
        // The pin leads the cache: a stop sequence ended the last turn on a token
        // that never went through the block. The cache's own count wins.
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 500, storedPositions: 300, promptTokens: 1_000
            ),
            300
        )
        // The prompt diverged inside the resident sequence, so its rows past that
        // point belong to tokens this prompt does not contain: no reuse, full prefill.
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 100, storedPositions: 300, promptTokens: 1_000
            ),
            0
        )
        // The last prompt token always runs: its logits are the first sample.
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 300, storedPositions: 300, promptTokens: 300
            ),
            0
        )
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 0, storedPositions: 300, promptTokens: 1_000
            ),
            0
        )
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 10, storedPositions: 0, promptTokens: 1_000
            ),
            0
        )
    }

    /// A length is not identity: a pin left by another model's turn can propose a
    /// prefix whose length fits a sequence this cache never held.
    func testAResidentSequenceOfOtherTokensIsNotReused() {
        let prompt: [Int] = [10, 11, 12, 13, 14]
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 3,
                storedPositions: 3,
                promptTokenIds: prompt,
                residentTokens: [10, 11, 12]
            ),
            3
        )
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 3,
                storedPositions: 3,
                promptTokenIds: prompt,
                residentTokens: [10, 99, 12]
            ),
            0,
            "a resident position holding a different token is not this prompt"
        )
        // The history has to describe every position the cache claims.
        XCTAssertEqual(
            DeepSeekV41CacheAddressing.reusablePrefix(
                proposed: 3,
                storedPositions: 3,
                promptTokenIds: prompt,
                residentTokens: [10, 11]
            ),
            0
        )
    }

    /// The window and the compressed rows cover the newest positions twice when the
    /// ratio is no wider than the window. That overlap is the reference behavior, so
    /// it is asserted here rather than left to be rediscovered.
    func testWindowAndCompressedRowsOverlapForANarrowRatio() throws {
        let config = try DeepSeekV41Config.decode(jsonData: Data(DeepSeekV41ConfigTests.wrapperJSON.utf8))
        let plan = try config.makeLayerPlan()
        let geometry = DeepSeekV41CacheGeometry.make(config: config, plan: plan, contextLength: 1_024)
        let layer = try XCTUnwrap(plan.first { $0.flavor == .compressed && $0.index == 2 })
        let rows = DeepSeekV41CacheAddressing.attentionRowPlan(
            layer: layer, position: 9, config: config, plan: plan, geometry: geometry
        )
        // Position 9 pools into compressed row 4 (positions 8...9) and the window
        // still holds positions 0...9, so those two positions are in both sets.
        XCTAssertEqual(rows.compressedRows.last, 4)
        XCTAssertTrue(rows.windowPositions.contains(8))
        XCTAssertTrue(rows.windowPositions.contains(9))
        XCTAssertEqual(rows.totalRows, rows.windowPositions.count + rows.compressedRows.count)
    }
}

// MARK: - Dtype widths

final class DeepSeekV41DtypeWidthTests: XCTestCase {

    func testEightBitIntegersAreOneByte() {
        XCTAssertEqual(DeepSeekV41DtypeWidth.bytesPerValue(dtype: "I8"), 1, accuracy: 1e-9)
        XCTAssertEqual(DeepSeekV41DtypeWidth.bytesPerValue(dtype: "U8"), 1, accuracy: 1e-9)
        XCTAssertEqual(DeepSeekV41DtypeWidth.bytesPerValue(dtype: "INT8"), 1, accuracy: 1e-9)
    }

    func testMXFP4IsNamedExplicitly() {
        XCTAssertEqual(
            DeepSeekV41DtypeWidth.bytesPerValue(dtype: "MXFP4"),
            0.5 + 1.0 / 32.0,
            accuracy: 1e-9
        )
        XCTAssertEqual(
            DeepSeekV41DtypeWidth.bytesPerValue(dtype: "F4_E2M1"),
            0.5 + 1.0 / 32.0,
            accuracy: 1e-9
        )
    }

    func testFloatWidths() {
        XCTAssertEqual(DeepSeekV41DtypeWidth.bytesPerValue(dtype: "BF16"), 2, accuracy: 1e-9)
        XCTAssertEqual(DeepSeekV41DtypeWidth.bytesPerValue(dtype: "F16"), 2, accuracy: 1e-9)
        XCTAssertEqual(DeepSeekV41DtypeWidth.bytesPerValue(dtype: "F32"), 4, accuracy: 1e-9)
        XCTAssertEqual(DeepSeekV41DtypeWidth.bytesPerValue(dtype: "F8_E4M3"), 1 + 1.0 / 32.0, accuracy: 1e-9)
    }
}

// MARK: - Memory plan honesty

final class DeepSeekV41MemoryPlanHonestyTests: XCTestCase {

    private func memory(recordBytesOverride: Int? = nil) throws -> DeepSeekV41MemoryPlan {
        let config = try DeepSeekV41Config.decode(jsonData: Data(DeepSeekV41ConfigTests.wrapperJSON.utf8))
        let plan = try config.makeLayerPlan()
        return DeepSeekV41MemoryPlan.make(
            config: config,
            plan: plan,
            catalog: nil,
            contextLength: 4_096,
            recordBytesOverride: recordBytesOverride
        )
    }

    func testRequiredBytesIncludesEngramStaging() throws {
        let plan = try memory()
        let expected = plan.cacheBytes
            + plan.residentWeightBytes
            + 8 * plan.bytesPerExpert
            + plan.engramStagingBytes
        XCTAssertEqual(plan.requiredBytes(expertSlots: 8), expected)
    }

    func testFP16OverridePricesTheWindowRingsToo() throws {
        let fp16 = try memory(recordBytesOverride: 1_024)
        let config = try DeepSeekV41Config.decode(jsonData: Data(DeepSeekV41ConfigTests.wrapperJSON.utf8))
        XCTAssertEqual(
            fp16.compressedCacheBytes,
            ((4_096 / 2 + 1) + (4_096 + 1)) * 1_024
        )
        XCTAssertEqual(fp16.windowRingBytes, 40 * config.windowSize * 1_024)
    }
}

// MARK: - Encoding and catalog edges

final class DeepSeekV41ConfigEdgeTests: XCTestCase {

    private func decode(_ json: String) throws -> DeepSeekV41Config {
        try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
    }

    private static let minimalJSON = """
    {"model_type": "deepseek_v41",
     "num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 4,
     "num_key_value_heads": 1, "vocab_size": 1024,
     "n_routed_experts": 8, "n_activated_experts": 2,
     "qk_nope_head_dim": 64, "v_head_dim": 64, "o_groups": 1,
     "window_size": 8, "compress_ratios": [0, 2],
     "engram_layer_ids": [0], "engram_num_embeddings": [3000000000],
     "engram_max_ngram_size": 2, "engram_vocab_size": 1000,
     "engram_n_heads": 1, "engram_head_dim": 8,
     "engram_compressed_vocab_size": 100, "engram_pad_token_id": 2}
    """

    /// A decoupled rope payload loads, and the record it caches is the head width
    /// rather than the head width plus the payload: 64 non-rope values with a 64-wide
    /// payload is a 128-wide record, not a 192-wide one.
    func testDecoupledRopePayloadLoadsAsATrailingSlice() throws {
        let json = Self.minimalJSON.replacingOccurrences(
            of: "\"qk_nope_head_dim\": 64",
            with: "\"qk_nope_head_dim\": 64, \"qk_rope_head_dim\": 64"
        )
        let config = try decode(json)
        XCTAssertEqual(config.qkRopeHeadDim, 64)
        XCTAssertEqual(config.qkHeadDim, 128)
        XCTAssertEqual(config.cacheRecordWidth, 128)
    }

    /// Engram row counts are ~4x10^8 and can exceed 32 bits, so the reader must not
    /// truncate them.
    func testLargeEngramRowCountsDecodeExactly() throws {
        let config = try decode(Self.minimalJSON)
        XCTAssertEqual(config.engramRowCounts, [3_000_000_000])
        XCTAssertEqual(config.engramTables.first?.rowCount, 3_000_000_000)
    }

    func testOutOfRangePadTokenIsRefusedInsteadOfTrapping() {
        XCTAssertThrowsError(
            try DeepSeekV41EngramTokenMap.synthetic(
                vocabSize: 100, compressedVocabSize: 10, padTokenID: 100
            )
        )
    }

    func testTwoNamesForOneRoleAreReported() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                DeepSeekV41TensorRef(name: "layers.0.attn.kv_norm.weight", dtype: "BF16", shape: [64]),
                DeepSeekV41TensorRef(name: "layers.0.attn.wkv_norm.weight", dtype: "BF16", shape: [64])
            ],
            layerCount: 1
        )
        XCTAssertEqual(catalog.duplicateRoles.count, 1, "\(catalog.duplicateRoles)")
    }

    func testNegativeIndicesAreNotTensorsOfALayer() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                DeepSeekV41TensorRef(name: "layers.-1.attn_norm.weight", dtype: "BF16", shape: [64]),
                DeepSeekV41TensorRef(name: "layers.0.ffn.experts.-2.w1.weight", dtype: "BF16", shape: [64, 64])
            ],
            layerCount: 1
        )
        XCTAssertEqual(catalog.slots.count, 0)
        XCTAssertEqual(catalog.unconsumed.count, 2)
    }

    /// `Int(Double)` traps at 2^63, and `Double(Int.max)` is exactly 2^63, so
    /// the old bound accepted the one value that traps.
    func testAnIntegerPastIntMaxIsRejectedInsteadOfTrapping() {
        let json = Self.minimalJSON.replacingOccurrences(
            of: "\"vocab_size\": 1024",
            with: "\"vocab_size\": 9223372036854775808"
        )
        XCTAssertThrowsError(try decode(json))
    }

    func testNegativeNextNPredictLayerCountIsRejected() throws {
        let json = Self.minimalJSON.replacingOccurrences(
            of: "\"window_size\": 8",
            with: "\"window_size\": 8, \"num_nextn_predict_layers\": -1"
        )
        let config = try decode(json)
        XCTAssertThrowsError(try config.makeLayerPlan()) { error in
            guard case DeepSeekV41ConfigError.invalidField(let field, _)? = error as? DeepSeekV41ConfigError else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(field, "num_nextn_predict_layers")
        }
    }

    /// `engramTables` used to `zip` the two arrays, silently pairing a short
    /// row-count list with the first layers and dropping the rest.
    func testMismatchedEngramArraysDoNotPairSilently() throws {
        let json = Self.minimalJSON.replacingOccurrences(
            of: "\"engram_layer_ids\": [0]",
            with: "\"engram_layer_ids\": [0, 1]"
        )
        let config = try decode(json)
        XCTAssertTrue(config.engramTables.isEmpty)
        XCTAssertThrowsError(try config.makeLayerPlan())
    }

    /// `hc_*_scale` is a plain role, not a quantized companion: its name ends in
    /// `_scale`, not `.scale`, so the catalog resolves it through the normal
    /// lookup and the vector bank never falls back to the identity value.
    func testHyperConnectionScaleResolvesAsAPlainVector() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                DeepSeekV41TensorRef(name: "layers.0.hc_attn_scale", dtype: "BF16", shape: [4])
            ],
            layerCount: 1
        )
        XCTAssertNotNil(catalog.tensor(.hcAttnScale, layer: 0))
    }

    /// The loader keys "never map this" off its own classifier, not the Rust
    /// display string.
    func testEngramNamesAreStreamedByTheSwiftClassifier() {
        XCTAssertTrue(DeepSeekV41WeightCatalog.isStreamedName("layers.1.engram.embed.weight"))
        XCTAssertTrue(DeepSeekV41WeightCatalog.isStreamedName("layers.14.engram.embed.scale"))
        XCTAssertFalse(DeepSeekV41WeightCatalog.isStreamedName("layers.1.attn.wkv.weight"))
    }

    /// `layerCount` was accepted and never read, so a manifest placing a
    /// backbone tensor past the last layer validated clean.
    func testTensorsPastTheLastLayerAreReported() {
        let catalog = DeepSeekV41WeightCatalog(
            named: [
                DeepSeekV41TensorRef(name: "layers.5.attn_norm.weight", dtype: "BF16", shape: [64])
            ],
            layerCount: 2
        )
        XCTAssertEqual(catalog.slots.count, 1)
        XCTAssertEqual(catalog.outOfRangeLayers.count, 1, "\(catalog.outOfRangeLayers)")
    }
}

// MARK: - Vision placeholder expansion

final class DeepSeekV41VisionExpandTests: XCTestCase {

    private let spec = DeepSeekV41VisionSpec(
        patchSize: 16, downsampleRatio: 1, minPixels: 0, maxImageTokens: 0,
        numHiddenLayers: 1, hiddenSize: 16, numAttentionHeads: 1,
        intermediateSize: 16, imageTokenID: 7
    )

    func testEmptyMarkerDoesNotLoopForever() {
        let markers = DeepSeekV41VisionMarkers(imagePlaceholder: "", imageToken: "x")
        XCTAssertEqual(
            DeepSeekV41VisionLayout.expand(text: "hi", images: [(32, 32)], spec: spec, markers: markers),
            "hi"
        )
    }

    func testExtraPlaceholderIsKeptVisibly() {
        let markers = DeepSeekV41VisionMarkers(imagePlaceholder: "<img>", imageToken: "x")
        let out = DeepSeekV41VisionLayout.expand(
            text: "<img> and <img>", images: [(32, 32)], spec: spec, markers: markers
        )
        XCTAssertTrue(out.contains(" and <img>"), out)
        // 32 x 32 at patch 16 is a 2 x 2 grid: 2 rows of (2 + 1) plus two delimiters.
        XCTAssertTrue(out.hasPrefix(String(repeating: "x", count: 8) + " and"), out)
    }

    func testExtraImageIsReportedInsteadOfDroppedSilently() {
        let markers = DeepSeekV41VisionMarkers(imagePlaceholder: "<img>", imageToken: "x")
        let expansion = DeepSeekV41VisionLayout.expandReporting(
            text: "<img>", images: [(32, 32), (32, 32)], spec: spec, markers: markers
        )
        XCTAssertEqual(expansion.text, String(repeating: "x", count: 8))
        XCTAssertEqual(expansion.report.unusedImageCount, 1)
        XCTAssertFalse(expansion.report.isBalanced)
    }

    func testPlaceholderWithoutImagesIsReported() {
        let markers = DeepSeekV41VisionMarkers(imagePlaceholder: "<img>", imageToken: "x")
        let expansion = DeepSeekV41VisionLayout.expandReporting(
            text: "<img>lost", images: [], spec: spec, markers: markers
        )
        XCTAssertEqual(expansion.text, "<img>lost")
        XCTAssertEqual(expansion.report.unfilledPlaceholderCount, 1)
        XCTAssertFalse(expansion.report.isBalanced)
    }

    func testExtraPlaceholderIsReported() {
        let markers = DeepSeekV41VisionMarkers(imagePlaceholder: "<img>", imageToken: "x")
        let expansion = DeepSeekV41VisionLayout.expandReporting(
            text: "<img><img>", images: [(32, 32)], spec: spec, markers: markers
        )
        XCTAssertEqual(expansion.text, String(repeating: "x", count: 8) + "<img>")
        XCTAssertEqual(expansion.report.unfilledPlaceholderCount, 1)
    }
}
