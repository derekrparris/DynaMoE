//
//  ModelConfig.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/25/26.
//

import Foundation

public enum ModelArchitectureType: String, CaseIterable, Identifiable, Codable {
    case hybridSsmMoe = "Hybrid SSM + MoE (Ornith 1.5 / Qwen 3.5 MoE)"
    case standardMoe = "Standard MoE (Qwen MoE / Mixtral)"
    case denseTransformer = "Dense Transformer (Qwen Dense / LLaMA / Mistral)"

    public var id: String { rawValue }

    public var shortName: String {
        switch self {
        case .hybridSsmMoe: return "Hybrid SSM + MoE"
        case .standardMoe: return "Standard MoE"
        case .denseTransformer: return "Dense Transformer"
        }
    }

    public var icon: String {
        switch self {
        case .hybridSsmMoe: return "cpu.fill"
        case .standardMoe: return "square.grid.3x3.fill"
        case .denseTransformer: return "cube.fill"
        }
    }
}

public enum LayerAttentionType: String, Codable {
    case linearAttention = "Linear Attention (GatedDeltaNet SSM)"
    case fullAttention = "Full Attention (GQA)"
}

public enum LayerMlpType: String, Codable {
    case denseMlp = "Dense MLP (Single SwiGLU)"
    case moeExperts = "Sparse MoE (Routed + Shared Experts)"
}

public struct RopeScalingConfig: Codable {
    public var ropeType: String?
    public var type: String?
    public var factor: Float?
    public var originalMaxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
        case ropeType = "rope_type"
        case type
        case factor
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
}

public struct RopeParametersConfig: Codable {
    public var type: String?
    public var ropeTheta: Float?
    public var partialRotaryFactor: Float?
    public var mropeInterleaved: Bool?

    enum CodingKeys: String, CodingKey {
        case type
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case mropeInterleaved = "mrope_interleaved"
    }
}

public struct NestedTextConfig: Codable {
    public var hiddenSize: Int?
    public var numHiddenLayers: Int?
    public var numAttentionHeads: Int?
    public var numKeyValueHeads: Int?
    public var headDim: Int?
    public var intermediateSize: Int?
    public var vocabSize: Int?
    public var numExperts: Int?
    public var numExpertsPerTok: Int?
    public var layerTypes: [String]?
    public var maxPositionEmbeddings: Int?
    public var rmsNormEps: Float?
    public var partialRotaryFactor: Float?
    public var ropeParameters: RopeParametersConfig?
    public var ropeScaling: RopeScalingConfig?
    public var ropeTheta: Float?
    public var numLoops: Int?
    public var eosTokenId: Int?
    public var bosTokenId: Int?
    public var tieWordEmbeddings: Bool?
    public var skipLoopFinalNorm: Bool?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case layerTypes = "layer_types"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
        case numLoops = "num_loops"
        case eosTokenId = "eos_token_id"
        case bosTokenId = "bos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case skipLoopFinalNorm = "skip_loop_final_norm"
    }
}

public struct ModelConfig: Codable {
    public var architectures: [String]?
    public var modelType: String?
    public var hiddenSize: Int?
    public var numHiddenLayers: Int?
    public var numAttentionHeads: Int?
    public var numKeyValueHeads: Int?
    public var headDim: Int?
    public var intermediateSize: Int?
    public var vocabSize: Int?
    public var numExperts: Int?
    public var numExpertsPerTok: Int?
    public var layerTypes: [String]?
    public var maxPositionEmbeddings: Int?
    public var rmsNormEps: Float?
    public var partialRotaryFactor: Float?
    public var ropeParameters: RopeParametersConfig?
    public var ropeScaling: RopeScalingConfig?
    public var ropeTheta: Float?
    public var numLoops: Int?
    public var eosTokenId: Int?
    public var bosTokenId: Int?
    public var tieWordEmbeddings: Bool?
    public var skipLoopFinalNorm: Bool?
    public var textConfig: NestedTextConfig?

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case layerTypes = "layer_types"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
        case numLoops = "num_loops"
        case eosTokenId = "eos_token_id"
        case bosTokenId = "bos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case skipLoopFinalNorm = "skip_loop_final_norm"
        case textConfig = "text_config"
    }

    // Resolved Properties (handling text_config fallback)
    public var effectiveHiddenSize: Int {
        return textConfig?.hiddenSize ?? hiddenSize ?? 2048
    }

    public var effectiveNumHiddenLayers: Int {
        return textConfig?.numHiddenLayers ?? numHiddenLayers ?? 40
    }

    public var effectiveNumAttentionHeads: Int {
        return textConfig?.numAttentionHeads ?? numAttentionHeads ?? 16
    }

    public var effectiveNumKeyValueHeads: Int {
        return textConfig?.numKeyValueHeads ?? numKeyValueHeads ?? 2
    }

    public var effectiveHeadDim: Int {
        if let h = textConfig?.headDim ?? headDim {
            return h
        }
        let heads = effectiveNumAttentionHeads
        return heads > 0 ? (effectiveHiddenSize / heads) : 128
    }

    public var effectiveNumLoops: Int {
        return textConfig?.numLoops ?? numLoops ?? 1
    }

    public var effectiveVocabSize: Int {
        return textConfig?.vocabSize ?? vocabSize ?? 248320
    }

    public var effectiveNumExperts: Int {
        return textConfig?.numExperts ?? numExperts ?? 0
    }

    public var effectiveNumExpertsPerTok: Int {
        return textConfig?.numExpertsPerTok ?? numExpertsPerTok ?? (effectiveNumExperts > 0 ? 8 : 0)
    }

    public var effectiveRmsNormEps: Float {
        return textConfig?.rmsNormEps ?? rmsNormEps ?? 1e-6
    }

    public var effectiveMaxPositionEmbeddings: Int {
        return textConfig?.maxPositionEmbeddings ?? maxPositionEmbeddings ?? 262144
    }

    public var effectiveRopeTheta: Float {
        if let t = textConfig?.ropeParameters?.ropeTheta { return t }
        if let t = ropeParameters?.ropeTheta { return t }
        if let t = textConfig?.ropeTheta ?? ropeTheta { return t }
        return 10000000.0
    }

    public var effectiveEosTokenId: Int {
        return textConfig?.eosTokenId ?? eosTokenId ?? 248044
    }

    public var effectiveRotaryDim: Int {
        let headDim = effectiveHeadDim
        if let factor = textConfig?.partialRotaryFactor ?? partialRotaryFactor ?? textConfig?.ropeParameters?.partialRotaryFactor ?? ropeParameters?.partialRotaryFactor {
            return max(32, Int(Float(headDim > 0 ? headDim : 128) * factor))
        }
        if textConfig != nil || (modelType ?? "").contains("qwen") || (modelType ?? "").contains("ornith") {
            return max(32, Int(Float(headDim > 0 ? headDim : 256) * 0.25))
        }
        return headDim
    }

    public var effectiveLayerTypesStrings: [String]? {
        return textConfig?.layerTypes ?? layerTypes
    }

    /// Auto-detect the architecture type from config and topology summary
    public func resolveArchitectureType(summary: ModelSummary?) -> ModelArchitectureType {
        let rawType = (textConfig != nil ? "qwen3_5_moe" : (modelType ?? "")).lowercased()
        let archs = architectures?.map { $0.lowercased() } ?? []

        let isSsmModel = rawType.contains("qwen3_5") || rawType.contains("ornith") || rawType.contains("deltanet") || rawType.contains("mamba") || archs.contains(where: { $0.contains("qwen3_5") || $0.contains("deltanet") })
        let hasMoEExperts = effectiveNumExperts > 1 || (summary != nil && summary!.maxExpertId > 0)

        if isSsmModel && hasMoEExperts {
            return .hybridSsmMoe
        } else if hasMoEExperts {
            return .standardMoe
        } else {
            return .denseTransformer
        }
    }

    /// Resolves per-layer attention type (linear vs full attention)
    public func resolveLayerAttentionTypes(totalLayers: Int) -> [LayerAttentionType] {
        if let types = effectiveLayerTypesStrings, !types.isEmpty {
            return (0..<totalLayers).map { l in
                if l < types.count {
                    let str = types[l].lowercased()
                    if str.contains("linear") { return .linearAttention }
                    if str.contains("full") || str.contains("attn") { return .fullAttention }
                }
                return (l % 4 == 3) ? .fullAttention : .linearAttention
            }
        }

        // Default Hybrid pattern for Ornith / Qwen 3.5: every 4th layer is full attention (3, 7, 11, ...)
        let arch = resolveArchitectureType(summary: nil)
        if arch == .hybridSsmMoe {
            return (0..<totalLayers).map { ($0 % 4 == 3) ? .fullAttention : .linearAttention }
        } else {
            return Array(repeating: .fullAttention, count: totalLayers)
        }
    }

    // Static Loaders
    public static func load(from directoryUrl: URL) -> ModelConfig? {
        let configUrl = directoryUrl.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configUrl.path) else { return nil }
        return load(fromFilePath: configUrl.path)
    }

    public static func load(fromFilePath path: String) -> ModelConfig? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        let decoder = JSONDecoder()
        return try? decoder.decode(ModelConfig.self, from: data)
    }
}
