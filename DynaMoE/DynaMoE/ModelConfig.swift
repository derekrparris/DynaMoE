//
//  ModelConfig.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/25/26.
//

import Foundation

public enum ModelArchitectureType: String, CaseIterable, Identifiable, Codable {
    case qwen38FlashNext = "Hybrid GDN + QSA MoE (Qwen 3.8 Flash Next)"
    case hybridSsmMoe = "Hybrid SSM + MoE (Ornith 1.5 / Qwen 3.5 MoE)"
    case hybridSsmDense = "Hybrid SSM + Dense (Ornith 1.5 9B / Qwen 3.5 Dense)"
    case standardMoe = "Standard MoE (Qwen MoE / Mixtral)"
    case denseTransformer = "Dense Transformer (Qwen Dense / LLaMA / Mistral)"

    public var id: String { rawValue }

    public var shortName: String {
        switch self {
        case .qwen38FlashNext: return "Qwen 3.8 Flash Next"
        case .hybridSsmMoe: return "Hybrid SSM + MoE"
        case .hybridSsmDense: return "Hybrid SSM + Dense"
        case .standardMoe: return "Standard MoE"
        case .denseTransformer: return "Dense Transformer"
        }
    }

    public var icon: String {
        switch self {
        case .qwen38FlashNext: return "bolt.horizontal.fill"
        case .hybridSsmMoe: return "cpu.fill"
        case .hybridSsmDense: return "cpu"
        case .standardMoe: return "square.grid.3x3.fill"
        case .denseTransformer: return "cube.fill"
        }
    }

    public var isHybridSsm: Bool {
        return self == .qwen38FlashNext || self == .hybridSsmMoe || self == .hybridSsmDense
    }

    public var isQwen38: Bool {
        return self == .qwen38FlashNext
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

public struct TokenIdOrArray: Codable {
    public var single: Int?
    public var array: [Int]?

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let singleVal = try? container.decode(Int.self) {
            self.single = singleVal
            self.array = [singleVal]
        } else if let arrayVal = try? container.decode([Int].self) {
            self.array = arrayVal
            self.single = arrayVal.first
        } else {
            self.single = nil
            self.array = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let single = single {
            try container.encode(single)
        } else if let array = array {
            try container.encode(array)
        }
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
    public var eosTokenId: TokenIdOrArray?
    public var bosTokenId: TokenIdOrArray?
    public var tieWordEmbeddings: Bool?
    public var skipLoopFinalNorm: Bool?

    public var linearNumValueHeads: Int?
    public var linearNumKeyHeads: Int?
    public var linearValueHeadDim: Int?
    public var linearKeyHeadDim: Int?
    public var linearConvKernelDim: Int?
    public var moeIntermediateSize: Int?
    public var hcCount: Int?
    public var hcLowrank: Int?
    public var indexerNHeads: Int?
    public var indexerKvHeads: Int?
    public var indexerHeadDim: Int?
    public var attnOutputGate: Bool?
    public var outputGateType: String?

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
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case moeIntermediateSize = "moe_intermediate_size"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case indexerNHeads = "indexer_n_heads"
        case indexerKvHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case attnOutputGate = "attn_output_gate"
        case outputGateType = "output_gate_type"
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
    public var eosTokenId: TokenIdOrArray?
    public var bosTokenId: TokenIdOrArray?
    public var tieWordEmbeddings: Bool?
    public var skipLoopFinalNorm: Bool?
    public var linearNumValueHeads: Int?
    public var linearNumKeyHeads: Int?
    public var linearValueHeadDim: Int?
    public var linearKeyHeadDim: Int?
    public var linearConvKernelDim: Int?
    public var moeIntermediateSize: Int?
    public var hcCount: Int?
    public var hcLowrank: Int?
    public var indexerNHeads: Int?
    public var indexerKvHeads: Int?
    public var indexerHeadDim: Int?
    public var attnOutputGate: Bool?
    public var outputGateType: String?
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
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case moeIntermediateSize = "moe_intermediate_size"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case indexerNHeads = "indexer_n_heads"
        case indexerKvHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case attnOutputGate = "attn_output_gate"
        case outputGateType = "output_gate_type"
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

    public var isMoE: Bool {
        return effectiveNumExperts > 0
    }

    public var effectiveNumExpertsPerTok: Int {
        if let topK = textConfig?.numExpertsPerTok ?? numExpertsPerTok {
            return topK
        }
        if effectiveNumExperts >= 512 {
            return 10
        }
        return effectiveNumExperts > 0 ? 8 : 0
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
        return textConfig?.eosTokenId?.single ?? eosTokenId?.single ?? 248044
    }

    public var effectiveEosTokenIds: [Int] {
        if let textArray = textConfig?.eosTokenId?.array, !textArray.isEmpty {
            return textArray
        }
        if let topArray = eosTokenId?.array, !topArray.isEmpty {
            return topArray
        }
        return [248044, 248046]
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

    public var effectiveLinearNumValueHeads: Int {
        return textConfig?.linearNumValueHeads ?? linearNumValueHeads ?? 48
    }

    public var effectiveLinearNumKeyHeads: Int {
        return textConfig?.linearNumKeyHeads ?? linearNumKeyHeads ?? 16
    }

    public var effectiveLinearValueHeadDim: Int {
        return textConfig?.linearValueHeadDim ?? linearValueHeadDim ?? 128
    }

    public var effectiveLinearKeyHeadDim: Int {
        return textConfig?.linearKeyHeadDim ?? linearKeyHeadDim ?? 128
    }

    public var effectiveMoeIntermediateSize: Int {
        return textConfig?.moeIntermediateSize ?? moeIntermediateSize ?? 640
    }

    public var effectiveHcCount: Int {
        return textConfig?.hcCount ?? hcCount ?? 4
    }

    public var effectiveHcLowrank: Int {
        return textConfig?.hcLowrank ?? hcLowrank ?? 320
    }

    public var effectiveLayerTypesStrings: [String]? {
        return textConfig?.layerTypes ?? layerTypes
    }

    /// Whether full attention layers use Gated Attention (with sigmoid output gate)
    public var effectiveAttnOutputGate: Bool {
        let arch = resolveArchitectureType(summary: nil)
        return textConfig?.attnOutputGate ?? attnOutputGate ?? (arch.isQwen38 || arch.isHybridSsm)
    }

    /// Output gate activation type for Gated DeltaNet / linear recurrence (default: "silu")
    public var effectiveOutputGateType: String {
        return textConfig?.outputGateType ?? outputGateType ?? "silu"
    }

    /// Auto-detect the architecture type from config and topology summary
    public func resolveArchitectureType(summary: ModelSummary?) -> ModelArchitectureType {
        let rawType = (modelType ?? (textConfig != nil ? "qwen3_5_moe" : "")).lowercased()
        let archs = architectures?.map { $0.lowercased() } ?? []

        let isQwen38Model = rawType.contains("qwen3_8") || rawType.contains("qwen38") || rawType.contains("flash_next") || rawType.contains("qwen4") || archs.contains(where: { $0.contains("qwen3_8") || $0.contains("flash_next") || $0.contains("qwen4") }) || effectiveNumExperts >= 512 || (summary != nil && summary!.maxExpertId >= 500)
        if isQwen38Model {
            return .qwen38FlashNext
        }

        let isSsmModel = rawType.contains("qwen3_5") || rawType.contains("ornith") || rawType.contains("deltanet") || rawType.contains("mamba") || archs.contains(where: { $0.contains("qwen3_5") || $0.contains("deltanet") || $0.contains("ornith") })
        let hasMoEExperts = effectiveNumExperts > 1 || (summary != nil && summary!.maxExpertId > 0)

        if isSsmModel && hasMoEExperts {
            return .hybridSsmMoe
        } else if isSsmModel {
            return .hybridSsmDense
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
        if arch.isHybridSsm {
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

    /// Whether the model architecture uses 0-mean unit-offset RMSNorm weights (output = x * (1 + weight))
    /// NOTE: Gemma models, Qwen 3.5 MoE / Ornith 35B models, and Qwen 4 / Next models (e.g. Qwen 3.8 Flash Next, qwen4_exp)
    /// initialize their attention and layer norms to 0-mean and use unit offset (x * (1 + w)).
    /// MLX-converted Ornith 1.5 9B and dense Qwen 3.5 use standard 1-mean weights and must NOT use unit offset.
    public var isRMSNormUnitOffset: Bool {
        let archs = architectures?.map { $0.lowercased() } ?? []
        let modelTypeName = (modelType ?? (textConfig != nil ? "qwen3_5_moe" : "")).lowercased()
        let isGemma = modelTypeName.contains("gemma") || archs.contains(where: { $0.contains("gemma") })
        let isQwen4OrNext = modelTypeName.contains("qwen4") || modelTypeName.contains("next") ||
                            archs.contains(where: { $0.contains("qwen4") || $0.contains("next") || $0.contains("qwen4exp") })
        let isQwen35Moe = modelTypeName.contains("qwen3_5_moe") || archs.contains(where: { $0.contains("qwen3_5moe") || $0.contains("qwen3_5_moe") })
        let isOrnith35B = modelTypeName.contains("35b") || archs.contains(where: { $0.contains("35b") })
        let isOrnithOrQwen35Dense = (modelTypeName.contains("ornith") || modelTypeName.contains("qwen3_5") ||
                               archs.contains(where: { $0.contains("ornith") || $0.contains("qwen3_5") })) && !isQwen35Moe && !isOrnith35B
        return isGemma || isQwen35Moe || isOrnith35B || (isQwen4OrNext && !isOrnithOrQwen35Dense)
    }

    public static let userDefaultSystemPromptKey = "dynamoe_user_default_system_prompt"

    /// Reads the saved user default system prompt from UserDefaults
    public static func getUserDefaultSystemPrompt() -> String {
        let saved = UserDefaults.standard.string(forKey: userDefaultSystemPromptKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let saved = saved, !saved.isEmpty {
            return saved
        }
        return "You are a helpful AI assistant."
    }

    /// Persists the user's default system prompt to UserDefaults
    public static func setUserDefaultSystemPrompt(_ prompt: String) {
        UserDefaults.standard.set(prompt.trimmingCharacters(in: .whitespacesAndNewlines), forKey: userDefaultSystemPromptKey)
    }

    /// Returns the required system prompt mandatory for specific models to function properly (e.g. Nanbeige instruct tuning)
    public static func resolveRequiredSystemPrompt(
        config: ModelConfig?,
        summary: ModelSummary?,
        modelName: String? = nil,
        modelPath: String? = nil
    ) -> String {
        let nameLower = (modelName ?? "").lowercased()
        let pathLower = (modelPath ?? "").lowercased()
        let typeStr = (config?.modelType ?? "").lowercased()
        let archStr = config?.architectures?.joined(separator: " ").lowercased() ?? ""

        let isNanbeige = nameLower.contains("nanbeige") ||
                         pathLower.contains("nanbeige") ||
                         typeStr.contains("nanbeige") ||
                         archStr.contains("nanbeige") ||
                         (summary?.maxExpertId == 0 && (
                            summary?.layerCount == 22 ||
                            (summary?.tensors.contains(where: { $0.name.contains("dense_gate_up_proj") }) == true) ||
                            (summary?.tensors.contains(where: { $0.name.hasPrefix("model.layers.0.mlp.gate_proj") }) == true && summary?.layerCount == 22)
                         ))

        if isNanbeige {
            return "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。"
        }

        // Ornith models MUST be checked before Qwen fallback since Ornith configs inherit "qwen3_5_moe"
        let isOrnith = nameLower.contains("ornith") ||
                       pathLower.contains("ornith") ||
                       typeStr.contains("ornith") ||
                       archStr.contains("ornith") ||
                       (summary?.tensors.contains(where: { $0.name.contains("linear_attn") }) == true &&
                        summary?.tensors.contains(where: { $0.name.contains("hc_norm") || $0.name.contains("hyper_connection") }) == false)

        if isOrnith {
            // Ornith chat template defaults to clean instruct without mandatory prefix
            return ""
        }

        if nameLower.contains("qwen") || pathLower.contains("qwen") || typeStr.contains("qwen") || archStr.contains("qwen") {
            return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }

        if nameLower.contains("deepseek") || pathLower.contains("deepseek") || typeStr.contains("deepseek") || archStr.contains("deepseek") {
            return "You are a helpful and harmless AI assistant."
        }

        if nameLower.contains("llama") || pathLower.contains("llama") || typeStr.contains("llama") || archStr.contains("llama") {
            return "You are a helpful, respectful and honest assistant."
        }

        return ""
    }

    /// Dynamically determines the suggested default system prompt for the active model architecture
    public static func resolveDefaultSystemPrompt(
        config: ModelConfig?,
        summary: ModelSummary?,
        modelName: String? = nil,
        modelPath: String? = nil
    ) -> String {
        let required = resolveRequiredSystemPrompt(config: config, summary: summary, modelName: modelName, modelPath: modelPath)
        if !required.isEmpty {
            return required
        }
        return getUserDefaultSystemPrompt()
    }

    /// Combines the model's required system prompt with the user's custom system prompt in conjunction
    public static func buildEffectiveSystemPrompt(
        userPrompt: String,
        config: ModelConfig?,
        summary: ModelSummary?,
        modelName: String? = nil,
        modelPath: String? = nil
    ) -> String {
        let required = resolveRequiredSystemPrompt(config: config, summary: summary, modelName: modelName, modelPath: modelPath).trimmingCharacters(in: .whitespacesAndNewlines)
        let user = userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)

        if !required.isEmpty && !user.isEmpty {
            if user.contains(required) {
                return user
            }
            return "\(required)\n\n\(user)"
        } else if !required.isEmpty {
            return required
        } else {
            return user
        }
    }

    /// Determines whether a given model architecture/configuration supports native thinking / reasoning tokens
    public static func supportsThinking(
        config: ModelConfig? = nil,
        summary: ModelSummary? = nil,
        modelName: String? = nil,
        modelPath: String? = nil
    ) -> Bool {
        let nameLower = (modelName ?? "").lowercased()
        let pathLower = (modelPath ?? "").lowercased()
        let typeStr = (config?.modelType ?? "").lowercased()
        let archStr = config?.architectures?.joined(separator: " ").lowercased() ?? ""

        if nameLower.contains("nanbeige") || typeStr.contains("nanbeige") || archStr.contains("nanbeige") || pathLower.contains("nanbeige") {
            return true
        }
        if nameLower.contains("ornith") || typeStr.contains("ornith") || archStr.contains("ornith") || pathLower.contains("ornith") {
            return true
        }
        if nameLower.contains("qwen") || typeStr.contains("qwen") || archStr.contains("qwen") || pathLower.contains("qwen") {
            return true
        }
        if nameLower.contains("deepseek") || typeStr.contains("deepseek") || archStr.contains("deepseek") || pathLower.contains("deepseek") {
            return true
        }
        if nameLower.contains("r1") || nameLower.contains("reason") || nameLower.contains("qwq") || typeStr.contains("qwq") || archStr.contains("qwq") || pathLower.contains("qwq") {
            return true
        }
        if nameLower.contains("glm") || typeStr.contains("glm") || archStr.contains("glm") || pathLower.contains("glm") {
            return true
        }
        if nameLower.contains("nemotron") || typeStr.contains("nemotron") || archStr.contains("nemotron") || pathLower.contains("nemotron") {
            return true
        }
        if nameLower.contains("think") || pathLower.contains("think") {
            return true
        }
        // Inspect small config files if path provided
        if let path = modelPath, !path.isEmpty {
            let fileMgr = FileManager.default
            let dirUrl = URL(fileURLWithPath: path)
            for fName in ["chat_template.jinja", "tokenizer_config.json"] {
                let fUrl = dirUrl.appendingPathComponent(fName)
                if fileMgr.fileExists(atPath: fUrl.path), let content = try? String(contentsOf: fUrl, encoding: .utf8) {
                    if content.contains("<think>") || content.contains("enable_thinking") || content.contains("<|thought|>") || content.contains("reasoning_content") {
                        return true
                    }
                }
            }
        }
        if summary?.maxExpertId == 0 && (
            summary?.layerCount == 22 ||
            (summary?.tensors.contains(where: { $0.name.contains("dense_gate_up_proj") }) == true) ||
            (summary?.tensors.contains(where: { $0.name.hasPrefix("model.layers.0.mlp.gate_proj") }) == true && summary?.layerCount == 22)
        ) {
            return true
        }
        return false
    }
}
