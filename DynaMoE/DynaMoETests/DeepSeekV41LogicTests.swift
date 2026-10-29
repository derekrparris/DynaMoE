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
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 512,
        "qk_rope_head_dim": 0,
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

    func publishedSpec(layer: Int = 1, rows: Int = 384_006_168) -> DeepSeekV41EngramSpec {
        DeepSeekV41EngramSpec(
            layer: layer,
            rowCount: rows,
            maxNgramSize: 4,
            headCount: 8,
            headDimension: 256,
            padTokenID: 2,
            compressedVocabSize: 99_092,
            bucketStart: 16_000_000
        )
    }

    /// Reference vector: the prime ranges must address exactly the published
    /// `engram_num_embeddings[0]`, which is what proves the bucket arithmetic
    /// (`cursor = prime + 1`, 24 columns) matches the checkpoint.
    func testBucketLayoutReproducesPublishedRowCount() throws {
        let layout = try DeepSeekV41EngramLayout(spec: publishedSpec())
        XCTAssertEqual(layout.hashColumnCount, 24)
        XCTAssertEqual(layout.addressedRows, 384_006_168)
        XCTAssertEqual(layout.bucketSizes.first, 16_000_057)
        XCTAssertTrue(layout.bucketSizes.allSatisfy(DeepSeekV41Primes.isPrime))
    }

    func testBucketRangesAreDisjointAndAscending() throws {
        let layout = try DeepSeekV41EngramLayout(spec: publishedSpec())
        XCTAssertEqual(layout.bucketOffsets.first, 0)
        for index in 1..<layout.bucketOffsets.count {
            let previousEnd = layout.bucketOffsets[index - 1] + layout.bucketSizes[index - 1]
            XCTAssertGreaterThanOrEqual(layout.bucketOffsets[index], previousEnd)
        }
    }

    func testRowBytesAndTableBytes() throws {
        let spec = publishedSpec()
        XCTAssertEqual(spec.rowBytes, 256 + 8)
        XCTAssertEqual(spec.totalBytes, 384_006_168 * 264)
    }

    func testSecondTableFitsItsLargerRowBudget() throws {
        let first = try DeepSeekV41EngramLayout(spec: publishedSpec(layer: 14, rows: 384_016_682))
        XCTAssertLessThanOrEqual(first.addressedRows, 384_016_682)
    }

    func testTableTooSmallForItsBucketsIsRejected() {
        let spec = DeepSeekV41EngramSpec(
            layer: 1, rowCount: 1_000, maxNgramSize: 4, headCount: 8,
            headDimension: 256, padTokenID: 2, compressedVocabSize: 99_092,
            bucketStart: 16_000_000
        )
        XCTAssertThrowsError(try DeepSeekV41EngramLayout(spec: spec))
    }

    func testColumnAndByteAddressing() throws {
        let layout = try DeepSeekV41EngramLayout(spec: publishedSpec())
        XCTAssertEqual(layout.column(ngramSize: 2, head: 0), 0)
        XCTAssertEqual(layout.column(ngramSize: 2, head: 7), 7)
        XCTAssertEqual(layout.column(ngramSize: 4, head: 7), 23)
        XCTAssertEqual(layout.byteOffset(column: 0, bucket: 0), 0)
        XCTAssertEqual(layout.byteOffset(column: 1, bucket: 0), layout.bucketSizes[0] * 264)
        XCTAssertEqual(layout.byteOffset(column: 0, bucket: 5), 5 * 264)
    }

    func testHashFinalizerIsDeterministic() {
        XCTAssertEqual(DeepSeekV41EngramHasher.mix(0), 0)
        // Same input, same output; different input, different output.
        XCTAssertEqual(DeepSeekV41EngramHasher.mix(12345), DeepSeekV41EngramHasher.mix(12345))
        XCTAssertNotEqual(DeepSeekV41EngramHasher.mix(12345), DeepSeekV41EngramHasher.mix(12346))
    }

    /// Until a probe run confirms the token-id combination against the
    /// checkpoint's own `inference/engram.py`, the runtime must not claim the
    /// hash is verified. This test fails the day someone flips the flag without
    /// the reference vectors behind it.
    func testHashConformanceIsDeclaredUnverified() {
        XCTAssertEqual(DeepSeekV41EngramHasher.conformance, .unverified)
    }

    func testTokenMapCompression() throws {
        let map = try DeepSeekV41EngramTokenMap(
            vocabSize: 129_280, compressedVocabSize: 99_092, padTokenID: 2
        )
        XCTAssertEqual(map.compressedToken(2), 2)
        XCTAssertEqual(map.compressedToken(99_092 + 7), 7)
        XCTAssertEqual(map.compressedToken(-1), 2)
        XCTAssertEqual(map.compressedToken(10_000_000), 2)
    }

    func testRowRequestsPerToken() throws {
        let layout = try DeepSeekV41EngramLayout(spec: publishedSpec())
        let map = try DeepSeekV41EngramTokenMap(
            vocabSize: 129_280, compressedVocabSize: 99_092, padTokenID: 2
        )
        let history = [Int32(11), Int32(22), Int32(33)]
        let requests = DeepSeekV41Engram.rowRequests(
            layer: 1, history: history, tokenMap: map, layout: layout
        )
        // (max_ngram_size - 1) n-gram sizes x n_heads.
        XCTAssertEqual(requests.count, 24)
        XCTAssertTrue(requests.allSatisfy { $0.bucket >= 0 && $0.bucket < layout.bucketSizes[$0.column] })
        XCTAssertTrue(requests.allSatisfy { $0.byteCount == 264 })
        XCTAssertTrue(requests.allSatisfy { $0.byteOffset + $0.byteCount <= layout.spec.totalBytes })
        XCTAssertEqual(requests.map(\.column), Array(0..<24))
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
            DeepSeekV41Engram.rowRequests(layer: 14, history: history, tokenMap: map, layout: layout).isEmpty
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
                tensor("layers.3.engram.wkv.weight", "F8_E4M3", [1000, 8]),
                tensor("layers.3.engram.wkv.scale", "F8_E8M0", [1000, 1])
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
        XCTAssertNotNil(catalog.tensor(.draftMainProj, layer: 0))
        XCTAssertNotNil(catalog.tensor(.draftNorm, layer: 2))
        XCTAssertNotNil(catalog.tensor(.draftMarkovHead, layer: 1))
        XCTAssertNotNil(catalog.tensor(.draftConfidence, layer: 0))
        XCTAssertNotNil(catalog.tensor(.draftExpertW3, layer: 1, expert: 5))
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
}

// MARK: - Preflight

final class DeepSeekV41PreflightTests: XCTestCase {

    func testIncompleteDownloadIsReported() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-preflight-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let index = """
        {"metadata": {"total_size": 2000},
         "weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}
        """
        try index.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8
        )
        try Data(repeating: 0, count: 1000)
            .write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))

        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertEqual(report.shardCount, 2)
        XCTAssertEqual(report.presentShardCount, 1)
        XCTAssertEqual(report.bytesPresent, 1000)
        XCTAssertEqual(report.bytesExpected, 2000)
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
        try Data(repeating: 0, count: 10)
            .write(to: dir.appendingPathComponent("model-00001-of-00001.safetensors"))

        let report = try XCTUnwrap(DeepSeekV41Preflight.inspect(snapshotDir: dir))
        XCTAssertTrue(report.isComplete)
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
                DeepSeekV41TensorRef(name: "layers.1.engram.wkv.weight", dtype: "F8_E4M3", shape: [64, 8])
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
        let layout = try DeepSeekV41EngramLayout(spec: spec)
        let url = try makeTable(rows: spec.rowCount, rowBytes: spec.rowBytes)
        defer { try? FileManager.default.removeItem(at: url) }

        let map = try DeepSeekV41EngramTokenMap(
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
            layer: 1, column: 0, bucket: 0,
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
            layer: 14, column: 0, bucket: 0, byteOffset: 0, byteCount: spec.rowBytes
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
            layer: 1, column: 0, bucket: 0,
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

// MARK: - Cache reuse

final class DeepSeekV41CacheReuseTests: XCTestCase {

    func testPrefixInsideTheWindowRingIsReusable() {
        XCTAssertTrue(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 64, storedPositions: 100, windowRows: 128
            )
        )
    }

    func testPrefixPastTheWindowRingIsNotReusable() {
        // The position count says every row exists, but the ring has overwritten the
        // oldest window rows, so a hit would attend stale context.
        XCTAssertFalse(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 200, storedPositions: 300, windowRows: 128
            )
        )
        // The prefix itself fits in the ring, but the ring has already wrapped:
        // rows 0...3 now hold positions 128...131, not the prefix's positions.
        XCTAssertFalse(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 4, storedPositions: 300, windowRows: 128
            )
        )
    }

    func testPrefixPastWhatIsStoredIsNotReusable() {
        XCTAssertFalse(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 40, storedPositions: 20, windowRows: 128
            )
        )
        XCTAssertTrue(
            DeepSeekV41CacheAddressing.prefixIsReusable(
                prefixCount: 0, storedPositions: 20, windowRows: 128
            )
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

    /// A decoupled rope payload is refused instead of loading a checkpoint whose two
    /// backends would rotate the latent differently.
    func testDecoupledRopePayloadIsRefused() {
        let json = Self.minimalJSON.replacingOccurrences(
            of: "\"qk_nope_head_dim\": 64",
            with: "\"qk_nope_head_dim\": 64, \"qk_rope_head_dim\": 64"
        )
        XCTAssertThrowsError(try decode(json)) { error in
            guard case DeepSeekV41ConfigError.invalidField(let field, _)? = error as? DeepSeekV41ConfigError else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(field, "qk_rope_head_dim")
        }
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
            try DeepSeekV41EngramTokenMap(
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
}

// MARK: - Vision placeholder expansion

final class DeepSeekV41VisionExpandTests: XCTestCase {

    private let spec = DeepSeekV41VisionSpec(
        patchSize: 16, downsampleRatio: 1, minPixels: 0, maxImageTokens: 0,
        numHiddenLayers: 1, hiddenSize: 16, numAttentionHeads: 1,
        intermediateSize: 16, imageTokenID: 7
    )

    func testEmptyMarkerDoesNotLoopForever() {
        let markers = DeepSeekV41VisionMarkers(
            imagePlaceholder: "", imageOpen: "<", imageClose: ">", imageToken: "x"
        )
        XCTAssertEqual(
            DeepSeekV41VisionLayout.expand(text: "hi", images: [(32, 32)], spec: spec, markers: markers),
            "hi"
        )
    }

    func testExtraPlaceholderIsKeptVisibly() {
        let markers = DeepSeekV41VisionMarkers(
            imagePlaceholder: "<img>", imageOpen: "<", imageClose: ">", imageToken: "x"
        )
        let out = DeepSeekV41VisionLayout.expand(
            text: "<img> and <img>", images: [(32, 32)], spec: spec, markers: markers
        )
        XCTAssertTrue(out.contains(" and <img>"), out)
        XCTAssertFalse(out.contains("xx and"), out)
    }

    func testExtraImageIsReportedInsteadOfDroppedSilently() {
        let markers = DeepSeekV41VisionMarkers(
            imagePlaceholder: "<img>", imageOpen: "<", imageClose: ">", imageToken: "x"
        )
        let expansion = DeepSeekV41VisionLayout.expandReporting(
            text: "<img>", images: [(32, 32), (32, 32)], spec: spec, markers: markers
        )
        XCTAssertEqual(expansion.text, "<xxxx>")
        XCTAssertEqual(expansion.report.unusedImageCount, 1)
        XCTAssertFalse(expansion.report.isBalanced)
    }

    func testPlaceholderWithoutImagesIsReported() {
        let markers = DeepSeekV41VisionMarkers(
            imagePlaceholder: "<img>", imageOpen: "<", imageClose: ">", imageToken: "x"
        )
        let expansion = DeepSeekV41VisionLayout.expandReporting(
            text: "<img>lost", images: [], spec: spec, markers: markers
        )
        XCTAssertEqual(expansion.text, "<img>lost")
        XCTAssertEqual(expansion.report.unfilledPlaceholderCount, 1)
        XCTAssertFalse(expansion.report.isBalanced)
    }

    func testExtraPlaceholderIsReported() {
        let markers = DeepSeekV41VisionMarkers(
            imagePlaceholder: "<img>", imageOpen: "<", imageClose: ">", imageToken: "x"
        )
        let expansion = DeepSeekV41VisionLayout.expandReporting(
            text: "<img><img>", images: [(32, 32)], spec: spec, markers: markers
        )
        XCTAssertEqual(expansion.text, "<xxxx><img>")
        XCTAssertEqual(expansion.report.unfilledPlaceholderCount, 1)
    }
}
