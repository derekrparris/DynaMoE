//
//  RepackedManifestShapeTests.swift
//  DynaMoE
//
//  Coverage for what a FlashMoE repack is allowed to forget about a tensor.
//
//  The packer writes `model_weights.json` and `packed_experts/layout.json`, and for a
//  long time it recorded `shape: []` for every entry. That erased the only record of a
//  tensor's width, so `buildCachedLayers` fell back to config widths. For Qwen 3.8
//  Flash Next, whose config publishes `moe_intermediate_size` (640) but no dense
//  `intermediate_size`, the fallback landed on the hard-coded 512 default and the
//  expert MLP ran 512-wide against 640-wide weights: every repacked turn came back as
//  gibberish while the same weights streamed from safetensors decoded normally.
//
//  These tests pin both halves of the fix: the packer now records real shapes, and the
//  layer builder resolves a routed expert's width from `moe_intermediate_size` when the
//  manifest has no shape to give (which also rescues models repacked before the fix).
//

import XCTest
@testable import DynaMoE

final class RepackedManifestShapeTests: XCTestCase {

    /// Excerpt of `Qwen3.8-Flash-Next-FP8`'s published `config.json` (snapshot
    /// `236dfdf2...`) with the keys this resolution depends on. Note the absence of a
    /// dense `intermediate_size`: the checkpoint only ships `moe_intermediate_size`.
    static func qwen38ConfigJSON(layerCount: Int = 48) -> Data {
        let layerTypes = (0..<layerCount).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
        let json = """
        {
          "architectures": ["Qwen4ExpForConditionalGeneration"],
          "model_type": "qwen4_exp",
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": \(layerCount),
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "rms_norm_eps": 1e-06,
            "vocab_size": 248320,
            "layer_types": [\(layerTypes.map { "\"\($0)\"" }.joined(separator: ","))]
          }
        }
        """
        return Data(json.utf8)
    }

    /// One packed layer's tensors as the Rust packed loader synthesizes them: routed
    /// experts named `model.layers.<l>.mlp.experts.<e>.*`, every entry shape-less.
    static func packedLayerTensors(layer: UInt32 = 0, includeShapes: Bool = false) -> [TensorMetadata] {
        let gateShape = includeShapes ? "[640, 2560]" : "[]"
        var tensors: [TensorMetadata] = [
            TensorMetadata(
                name: "model.language_model.layers.\(layer).mlp.gate.weight",
                shapeDisplay: includeShapes ? "[512, 2560]" : "[]", dtype: "BF16", sizeMb: 2.5,
                shardIndex: 0, offsetStart: 0, offsetEnd: 2_621_440,
                category: "MoE Router", layerIndex: layer, expertId: nil
            ),
            TensorMetadata(
                name: "model.language_model.layers.\(layer).mlp.shared_expert.gate_proj.weight",
                shapeDisplay: gateShape, dtype: "BF16", sizeMb: 3.1,
                shardIndex: 0, offsetStart: 2_621_440, offsetEnd: 5_242_880,
                category: "Shared Expert", layerIndex: layer, expertId: nil
            ),
            TensorMetadata(
                name: "model.language_model.layers.\(layer).mlp.shared_expert.up_proj.weight",
                shapeDisplay: gateShape, dtype: "BF16", sizeMb: 3.1,
                shardIndex: 0, offsetStart: 5_242_880, offsetEnd: 7_864_320,
                category: "Shared Expert", layerIndex: layer, expertId: nil
            ),
            TensorMetadata(
                name: "model.language_model.layers.\(layer).mlp.shared_expert.down_proj.weight",
                shapeDisplay: gateShape, dtype: "BF16", sizeMb: 3.1,
                shardIndex: 0, offsetStart: 7_864_320, offsetEnd: 10_485_760,
                category: "Shared Expert", layerIndex: layer, expertId: nil
            )
        ]

        let components = ["gate_proj.weight", "gate_proj.weight_scale_inv",
                          "up_proj.weight", "up_proj.weight_scale_inv",
                          "down_proj.weight", "down_proj.weight_scale_inv"]
        var offset: UInt64 = 10_485_760
        for expert in 0..<2 {
            for component in components {
                let isScale = component.contains("scale")
                tensors.append(TensorMetadata(
                    name: "model.layers.\(layer).mlp.experts.\(expert).\(component)",
                    shapeDisplay: includeShapes ? (isScale ? "[5, 20]" : "[640, 2560]") : "[]",
                    dtype: isScale ? "BF16" : "F8_E4M3",
                    sizeMb: isScale ? 0.0002 : 1.56,
                    shardIndex: 1, offsetStart: offset, offsetEnd: offset + (isScale ? 200 : 1_638_400),
                    category: "Routed Expert #\(expert)", layerIndex: layer, expertId: UInt32(expert)
                ))
                offset += isScale ? 200 : 1_638_400
            }
        }
        return tensors
    }

    static func summary(tensors: [TensorMetadata], layerCount: UInt32 = 48, maxExpertId: UInt32 = 511) -> ModelSummary {
        ModelSummary(
            sizeGb: 0.1,
            tensorCount: UInt32(tensors.count),
            layerCount: layerCount,
            maxExpertId: maxExpertId,
            shards: [ShardMetadata(index: 0, filename: "model_weights.bin", baseAddress: 0x1_0000_0000, length: 1 << 20),
                     ShardMetadata(index: 1, filename: "packed_experts/layer_00.bin", baseAddress: 0x2_0000_0000, length: 1 << 20)],
            tensors: tensors,
            layers: []
        )
    }

    // MARK: - The packer records shapes

    func testParseShapeDisplayRoundTripsSafetensorsShapes() {
        XCTAssertEqual(ExpertRepacker.parseShapeDisplay("[640, 2560]"), [640, 2560])
        XCTAssertEqual(ExpertRepacker.parseShapeDisplay("[248320, 2560]"), [248320, 2560])
        XCTAssertEqual(ExpertRepacker.parseShapeDisplay("[5, 20]"), [5, 20])
        XCTAssertEqual(ExpertRepacker.parseShapeDisplay("[2560]"), [2560])
        XCTAssertEqual(ExpertRepacker.parseShapeDisplay("[]"), [], "A scalar/empty shape must stay empty, not become [0]")
    }

    // MARK: - A shape-less manifest still resolves the expert width

    func testShapelessManifestResolvesExpertWidthFromMoeIntermediateSize() throws {
        let config = try JSONDecoder().decode(ModelConfig.self, from: Self.qwen38ConfigJSON())
        XCTAssertEqual(config.effectiveIntermediateSize, 0, "The published Qwen 3.8 config has no dense intermediate_size")

        let layers = InferenceEngine.shared.buildCachedLayers(
            summary: Self.summary(tensors: Self.packedLayerTensors(includeShapes: false)),
            config: config
        )

        XCTAssertEqual(layers.count, 48)
        XCTAssertEqual(layers[0].mlpType, .moeExperts)
        XCTAssertEqual(layers[0].intermediateDim, 640,
                       "A shape-less packed expert must be priced at moe_intermediate_size, not the 512 default")
    }

    func testPackedShapesAreUsedWhenTheManifestHasThem() throws {
        let config = try JSONDecoder().decode(ModelConfig.self, from: Self.qwen38ConfigJSON())
        let layers = InferenceEngine.shared.buildCachedLayers(
            summary: Self.summary(tensors: Self.packedLayerTensors(includeShapes: true)),
            config: config
        )
        XCTAssertEqual(layers[0].intermediateDim, 640, "A repack that records shapes must drive the width from them")
    }

    func testExpertWidthPrefersMoeIntermediateSizeOverDenseWidth() throws {
        // Checkpoints that publish both widths (Qwen 3 MoE and friends) keep the dense
        // width for their non-MoE layers; a routed expert is never that wide.
        let json = """
        {
          "architectures": ["Qwen3MoeForCausalLM"],
          "model_type": "qwen3_moe",
          "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "intermediate_size": 18944,
            "moe_intermediate_size": 768,
            "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"]
          }
        }
        """
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
        let layers = InferenceEngine.shared.buildCachedLayers(
            summary: Self.summary(tensors: Self.packedLayerTensors(includeShapes: false), layerCount: 4, maxExpertId: 127),
            config: config
        )
        XCTAssertEqual(layers[0].mlpType, .moeExperts)
        XCTAssertEqual(layers[0].intermediateDim, 768, "The 18944-wide dense MLP is not the expert width")
    }

    func testDenseLayerStillUsesTheDenseWidth() throws {
        let json = """
        {
          "architectures": ["Gemma4ForConditionalGeneration"],
          "model_type": "gemma4",
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 176,
            "intermediate_size": 704,
            "layer_types": ["full_attention", "full_attention"]
          }
        }
        """
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(json.utf8))
        let dense = TensorMetadata(
            name: "model.language_model.layers.0.mlp.gate_proj.weight",
            shapeDisplay: "[]", dtype: "BF16", sizeMb: 3.9,
            shardIndex: 0, offsetStart: 0, offsetEnd: 3_964_928,
            category: "Dense MLP", layerIndex: 0, expertId: nil
        )
        let layers = InferenceEngine.shared.buildCachedLayers(
            summary: Self.summary(tensors: [dense], layerCount: 2, maxExpertId: 0),
            config: config
        )
        XCTAssertEqual(layers[0].mlpType, .denseMlp)
        XCTAssertEqual(layers[0].intermediateDim, 704, "A layer with no routed experts keeps the dense MLP width")
    }

    // MARK: - The model this bug was found on

    /// Runs only where the repacked snapshot is on disk; it is metadata-only, so it does
    /// not touch the 121 GB of weights. Skipped (not failed) elsewhere.
    func testRepackedQwen38FlashNextResolvesItsExpertWidth() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
        guard FileManager.default.fileExists(atPath: snapshotDir + "/packed_experts/layout.json") else {
            print("Repacked Qwen 3.8 snapshot not found, skipping.")
            return
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        XCTAssertEqual(config?.modelType?.contains("qwen4_exp"), true)
        XCTAssertEqual(summary.maxExpertId, 512, "All 512 routed experts must survive the repack")

        let layers = InferenceEngine.shared.buildCachedLayers(summary: summary, config: config)
        XCTAssertEqual(layers.count, 48)
        XCTAssertEqual(layers[0].mlpType, .moeExperts)
        XCTAssertEqual(layers[0].intermediateDim, 640,
                       "The repacked checkpoint's expert MLP is moe_intermediate_size (640) wide")
        XCTAssertEqual(layers[3].intermediateDim, 640)
    }

    /// Packing is not allowed to change what the engine sees. This loads the same snapshot
    /// twice: once through its packed manifest, and once through a symlink farm that
    /// exposes only the safetensors (no `model_weights.json`, so the engine takes the raw
    /// path), then requires the resolved layer graph to match tensor for tensor.
    func testRepackedQwen38GraphMatchesItsSafetensorsSource() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
        guard FileManager.default.fileExists(atPath: snapshotDir + "/packed_experts/layout.json") else {
            print("Repacked Qwen 3.8 snapshot not found, skipping.")
            return
        }

        let fileManager = FileManager.default
        let rawDir = fileManager.temporaryDirectory
            .appendingPathComponent("dynamoe-unpacked-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: rawDir, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: rawDir) }

        let sourceDir = URL(fileURLWithPath: snapshotDir)
        for name in try fileManager.contentsOfDirectory(atPath: snapshotDir) {
            guard name == "config.json" || name == "model.safetensors.index.json" || name.hasSuffix(".safetensors") else { continue }
            try fileManager.createSymbolicLink(at: rawDir.appendingPathComponent(name),
                                               withDestinationURL: sourceDir.appendingPathComponent(name))
        }
        XCTAssertFalse(fileManager.fileExists(atPath: rawDir.appendingPathComponent("model_weights.json").path),
                       "The raw farm must not shadow the safetensors with a packed manifest")

        let config = ModelConfig.load(from: sourceDir)
        let packedSummary = try DynaMoeEngine(filePath: snapshotDir).getSummary()
        let rawSummary = try DynaMoeEngine(filePath: rawDir.path).getSummary()
        XCTAssertEqual(packedSummary.maxExpertId, rawSummary.maxExpertId)
        XCTAssertEqual(packedSummary.layerCount, rawSummary.layerCount)

        let packed = InferenceEngine.shared.buildCachedLayers(summary: packedSummary, config: config)
        let raw = InferenceEngine.shared.buildCachedLayers(summary: rawSummary, config: config)
        XCTAssertEqual(packed.count, raw.count)
        var divergences: [String] = []
        for layer in 0..<min(packed.count, raw.count) {
            let (packedSig, rawSig) = (Self.signature(packed[layer]), Self.signature(raw[layer]))
            for key in Set(packedSig.keys).union(rawSig.keys).sorted() where packedSig[key] != rawSig[key] {
                divergences.append("layer \(layer) \(key): packed=\(packedSig[key] ?? "nil") raw=\(rawSig[key] ?? "nil")")
            }
        }
        XCTAssertEqual(divergences, [], "Packing changed which tensors the decoder resolves")
    }

    /// What a layer will actually run: every singleton tensor it resolved (by name, dtype
    /// and byte size) plus, for routed experts, the key set and per-component sizes. The
    /// packed loader synthesizes expert names under a different prefix, so expert maps are
    /// compared by what the decoder dispatches on rather than by name.
    static func signature(_ layer: EngineCachedLayer) -> [String: String] {
        var parts: [String: String] = [:]
        for child in Mirror(reflecting: layer).children {
            let label = child.label ?? "?"
            if let tensor = child.value as? TensorMetadata {
                parts[label] = "\(tensor.name)|\(tensor.dtype)|\(tensor.offsetEnd - tensor.offsetStart)"
            } else if let experts = child.value as? [Int: TensorMetadata] {
                let encoded = experts.keys.sorted().map { id in
                    let t = experts[id]!
                    return "\(id):\(t.dtype):\(t.offsetEnd - t.offsetStart)"
                }.joined(separator: ",")
                parts[label] = "[\(experts.count)]\(encoded)"
            } else {
                parts[label] = String(describing: child.value)
            }
        }
        return parts
    }
}
