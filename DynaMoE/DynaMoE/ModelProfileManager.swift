//
//  ModelProfileManager.swift
//  DynaMoE
//
//  Manages persistent model-specific generation profiles (Coder and Assistant)
//  with customizable inference parameters per model.
//

import Foundation
import Combine

public enum ModelProfileType: String, Codable, CaseIterable, Identifiable {
    case coder = "Coder"
    case assistant = "Assistant"

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .coder: return "curlybraces"
        case .assistant: return "bubble.left.and.bubble.right.fill"
        }
    }

    public var description: String {
        switch self {
        case .coder:
            return "Optimized for programming, code generation, refactoring, and deterministic syntax."
        case .assistant:
            return "Optimized for natural conversation, creative writing, explanations, and general assistance."
        }
    }

    public func profileDisplayName(for modelName: String? = nil) -> String {
        let nameLower = (modelName ?? "").lowercased()
        if nameLower.contains("qwen") {
            switch self {
            case .coder: return "Coding & Agentic (Thinking Mode)"
            case .assistant: return "General Assistant (Instruct Mode)"
            }
        } else if nameLower.contains("ornith") {
            switch self {
            case .coder: return "Precise Coding & Tool Calling"
            case .assistant: return "General Chat / Agent Loops"
            }
        }
        return self.rawValue
    }

    public func description(for modelName: String? = nil) -> String {
        let nameLower = (modelName ?? "").lowercased()
        if nameLower.contains("qwen") {
            switch self {
            case .coder:
                return "Official Qwen Thinking Mode: High-entropy exploration (Temp 1.0, Pres 0.0) optimized for SWE-bench, complex code, and multi-tool execution."
            case .assistant:
                return "Official Qwen Instruct Mode: Direct non-reasoning responses (Temp 0.70, Pres 1.5) optimized for quick chat and summarization."
            }
        } else if nameLower.contains("ornith") {
            switch self {
            case .coder:
                return "Official Ornith Precise Profile: Low entropy (Temp 0.60, Pres 0.0) preventing code syntax drift and schema violation."
            case .assistant:
                return "Official Ornith Agentic Loop Profile: Dynamic variety (Temp 1.00, Pres 1.5) for open-ended benchmark reproduction and chat."
            }
        }
        return self.description
    }
}

public struct GenerationProfileSettings: Codable, Equatable, Hashable {
    public var temperature: Float
    public var topP: Float
    public var minP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var presencePenalty: Float
    public var maxNewTokens: Int
    public var systemPrompt: String
    public var jetSpecEnabled: Bool
    public var jetSpecMaxDepth: Int
    public var jetSpecBranchingFactor: Int
    public var jetSpecMaxExpertCap: Int

    public init(
        temperature: Float,
        topP: Float,
        minP: Float,
        topK: Int,
        repetitionPenalty: Float,
        presencePenalty: Float = 0.0,
        maxNewTokens: Int,
        systemPrompt: String,
        jetSpecEnabled: Bool = true,
        jetSpecMaxDepth: Int = 3,
        jetSpecBranchingFactor: Int = 2,
        jetSpecMaxExpertCap: Int = 8
    ) {
        self.temperature = temperature
        self.topP = topP
        self.minP = minP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.maxNewTokens = maxNewTokens
        self.systemPrompt = systemPrompt
        self.jetSpecEnabled = jetSpecEnabled
        self.jetSpecMaxDepth = jetSpecMaxDepth
        self.jetSpecBranchingFactor = jetSpecBranchingFactor
        self.jetSpecMaxExpertCap = jetSpecMaxExpertCap
    }

    public static func defaultFor(type: ModelProfileType, modelName: String? = nil) -> GenerationProfileSettings {
        let nameLower = (modelName ?? "").lowercased()
        let isQwen = nameLower.contains("qwen")
        let isOrnith = nameLower.contains("ornith")

        if isQwen {
            switch type {
            case .coder:
                // Official Qwen 3.8 Flash Next Coding & Agentic Profile (Thinking Mode)
                return GenerationProfileSettings(
                    temperature: 1.00,
                    topP: 0.95,
                    minP: 0.00,
                    topK: 20,
                    repetitionPenalty: 1.00,
                    presencePenalty: 0.00,
                    maxNewTokens: 8192,
                    systemPrompt: "You are an expert software engineer and agentic coding assistant. Provide clear, correct, and high-performance code with minimal filler.",
                    jetSpecEnabled: true,
                    jetSpecMaxDepth: 3,
                    jetSpecBranchingFactor: 2,
                    jetSpecMaxExpertCap: 8
                )
            case .assistant:
                // Official Qwen 3.8 Flash Next General Assistant Profile (Instruct / Direct Mode)
                return GenerationProfileSettings(
                    temperature: 0.70,
                    topP: 0.80,
                    minP: 0.00,
                    topK: 20,
                    repetitionPenalty: 1.00,
                    presencePenalty: 1.50,
                    maxNewTokens: 4096,
                    systemPrompt: "You are a helpful, respectful, and honest AI assistant.",
                    jetSpecEnabled: true,
                    jetSpecMaxDepth: 3,
                    jetSpecBranchingFactor: 2,
                    jetSpecMaxExpertCap: 8
                )
            }
        }

        if isOrnith {
            switch type {
            case .coder:
                // Official Ornith-1.5-9B Precise Coding & Tool Calling Profile
                return GenerationProfileSettings(
                    temperature: 0.60,
                    topP: 0.95,
                    minP: 0.00,
                    topK: 20,
                    repetitionPenalty: 1.00,
                    presencePenalty: 0.00,
                    maxNewTokens: 8192,
                    systemPrompt: "You are an expert software engineer and programming assistant. Provide clear, correct, and high-performance code with minimal filler.",
                    jetSpecEnabled: false,
                    jetSpecMaxDepth: 3,
                    jetSpecBranchingFactor: 2,
                    jetSpecMaxExpertCap: 8
                )
            case .assistant:
                // Official Ornith-1.5-9B General Chat / Agent Loops Profile
                return GenerationProfileSettings(
                    temperature: 1.00,
                    topP: 0.95,
                    minP: 0.00,
                    topK: 20,
                    repetitionPenalty: 1.00,
                    presencePenalty: 1.50,
                    maxNewTokens: 4096,
                    systemPrompt: "You are a helpful, respectful, and honest AI assistant.",
                    jetSpecEnabled: false,
                    jetSpecMaxDepth: 3,
                    jetSpecBranchingFactor: 2,
                    jetSpecMaxExpertCap: 8
                )
            }
        }

        // Generic fallback defaults
        switch type {
        case .coder:
            return GenerationProfileSettings(
                temperature: 0.60,
                topP: 0.95,
                minP: 0.00,
                topK: 20,
                repetitionPenalty: 1.00,
                presencePenalty: 0.00,
                maxNewTokens: 8192,
                systemPrompt: "You are an expert software engineer and programming assistant. Provide clear, correct, and high-performance code with minimal filler.",
                jetSpecEnabled: true,
                jetSpecMaxDepth: 3,
                jetSpecBranchingFactor: 2,
                jetSpecMaxExpertCap: 8
            )
        case .assistant:
            return GenerationProfileSettings(
                temperature: 0.70,
                topP: 0.90,
                minP: 0.05,
                topK: 50,
                repetitionPenalty: 1.10,
                presencePenalty: 0.00,
                maxNewTokens: 4096,
                systemPrompt: "You are a helpful, respectful, and honest AI assistant.",
                jetSpecEnabled: true,
                jetSpecMaxDepth: 3,
                jetSpecBranchingFactor: 2,
                jetSpecMaxExpertCap: 8
            )
        }
    }
}

public final class ModelProfileManager: ObservableObject {
    public static let shared = ModelProfileManager()

    private let profilesKey = "dynamoe_model_profiles_v1"
    private let activeProfilesKey = "dynamoe_model_active_profiles_v1"
    private let globalActiveProfileKey = "dynamoe_global_active_profile_v1"

    @Published public private(set) var profilesByModel: [String: [String: GenerationProfileSettings]] = [:]
    @Published public private(set) var activeProfileByModel: [String: String] = [:]
    @Published public var globalActiveProfile: ModelProfileType = .coder {
        didSet {
            UserDefaults.standard.set(globalActiveProfile.rawValue, forKey: globalActiveProfileKey)
        }
    }

    public init() {
        loadFromStorage()
        seedOfficialProfiles()
    }

    public func normalizeKey(_ modelIdentifier: String) -> String {
        let trimmed = modelIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()

        // 1. If it's a HuggingFace hub path with "models--"
        if lower.contains("models--") {
            let components = trimmed.split(separator: "/")
            if let modelsDir = components.first(where: { $0.lowercased().hasPrefix("models--") }) {
                let stripped = String(modelsDir).replacingOccurrences(of: "models--", with: "")
                return stripped.replacingOccurrences(of: "--", with: "/").lowercased()
            }
        }

        // 2. If it's a path or repoId containing "/"
        if trimmed.contains("/") {
            let components = trimmed.split(separator: "/")
            // Check snapshot directory ending in commit hash: .../snapshots/<hash>
            if let last = components.last, last.count >= 32, components.count >= 3, components[components.count - 2].lowercased() == "snapshots" {
                let repoDir = components[components.count - 3]
                let stripped = String(repoDir).replacingOccurrences(of: "models--", with: "")
                return stripped.replacingOccurrences(of: "--", with: "/").lowercased()
            }
            if components.count == 2 {
                return trimmed.lowercased()
            }
            if let last = components.last, !last.isEmpty {
                return String(last).lowercased()
            }
        }
        return trimmed.lowercased()
    }

    public func getProfile(for modelId: String?, type: ModelProfileType) -> GenerationProfileSettings {
        guard let modelId = modelId, !modelId.isEmpty else {
            return GenerationProfileSettings.defaultFor(type: type)
        }
        let key = normalizeKey(modelId)

        // 1. Exact match on normalized key
        if let modelProfiles = profilesByModel[key],
           let settings = modelProfiles[type.rawValue] {
            return settings
        }

        // 2. Exact match on raw modelId
        if let modelProfiles = profilesByModel[modelId],
           let settings = modelProfiles[type.rawValue] {
            return settings
        }

        // 3. Short repo name (e.g. qwen3.8-flash-next-fp8 from qwen/qwen3.8-flash-next-fp8)
        if key.contains("/") {
            let short = String(key.split(separator: "/").last ?? "")
            if !short.isEmpty, let modelProfiles = profilesByModel[short], let settings = modelProfiles[type.rawValue] {
                return settings
            }
        }

        // 4. Model family matching for known profiles
        let keyLower = key.lowercased()
        let idLower = modelId.lowercased()
        let isOrnith = keyLower.contains("ornith") || idLower.contains("ornith")
        let isQwen = keyLower.contains("qwen") || idLower.contains("qwen")

        if isOrnith {
            for (candidateKey, profiles) in profilesByModel where candidateKey.lowercased().contains("ornith") {
                if let settings = profiles[type.rawValue] {
                    return settings
                }
            }
        } else if isQwen {
            for (candidateKey, profiles) in profilesByModel where candidateKey.lowercased().contains("qwen") {
                if let settings = profiles[type.rawValue] {
                    return settings
                }
            }
        }

        // 5. Fallback to model-specific default
        return GenerationProfileSettings.defaultFor(type: type, modelName: modelId)
    }

    public func saveProfile(for modelId: String?, type: ModelProfileType, settings: GenerationProfileSettings) {
        let key = normalizeKey(modelId ?? "default")
        var current = profilesByModel[key] ?? [:]
        current[type.rawValue] = settings
        profilesByModel[key] = current
        saveToStorage()
    }

    public func getActiveProfile(for modelId: String?) -> ModelProfileType {
        guard let modelId = modelId, !modelId.isEmpty else {
            return globalActiveProfile
        }
        let key = normalizeKey(modelId)
        if let raw = activeProfileByModel[key], let type = ModelProfileType(rawValue: raw) {
            return type
        }
        return globalActiveProfile
    }

    public func setActiveProfile(for modelId: String?, type: ModelProfileType) {
        globalActiveProfile = type
        if let modelId = modelId, !modelId.isEmpty {
            let key = normalizeKey(modelId)
            activeProfileByModel[key] = type.rawValue
            UserDefaults.standard.set(activeProfileByModel, forKey: activeProfilesKey)
        }
    }

    public func resetProfile(for modelId: String?, type: ModelProfileType) {
        let key = normalizeKey(modelId ?? "default")
        let defaultSettings = GenerationProfileSettings.defaultFor(type: type, modelName: modelId)

        let lower = (modelId ?? "").lowercased()
        if lower.contains("ornith") || lower.contains("qwen") {
            if profilesByModel[key] == nil {
                profilesByModel[key] = [:]
            }
            profilesByModel[key]?[type.rawValue] = defaultSettings
        } else {
            profilesByModel[key]?.removeValue(forKey: type.rawValue)
            if profilesByModel[key]?.isEmpty == true {
                profilesByModel.removeValue(forKey: key)
            }
        }
        saveToStorage()
    }

    public func seedOfficialProfiles(force: Bool = false) {
        let ornithCoder = GenerationProfileSettings.defaultFor(type: .coder, modelName: "ornith")
        let ornithAssistant = GenerationProfileSettings.defaultFor(type: .assistant, modelName: "ornith")

        let ornithKeys = [
            "mlx-community/ornith-1.5-9b-optiq-4bit",
            "ornith-1.5-9b-optiq-4bit",
            "ornith"
        ]

        let qwenCoder = GenerationProfileSettings.defaultFor(type: .coder, modelName: "qwen")
        let qwenAssistant = GenerationProfileSettings.defaultFor(type: .assistant, modelName: "qwen")

        let qwenKeys = [
            "qwen/qwen3.8-flash-next-fp8",
            "qwen3.8-flash-next-fp8",
            "qwen"
        ]

        let seedFlagKey = "dynamoe_seeded_official_profiles_v3"
        let alreadySeeded = UserDefaults.standard.bool(forKey: seedFlagKey)

        var changed = false
        if !alreadySeeded || force {
            for key in ornithKeys {
                profilesByModel[key] = [
                    ModelProfileType.coder.rawValue: ornithCoder,
                    ModelProfileType.assistant.rawValue: ornithAssistant
                ]
            }
            for key in qwenKeys {
                profilesByModel[key] = [
                    ModelProfileType.coder.rawValue: qwenCoder,
                    ModelProfileType.assistant.rawValue: qwenAssistant
                ]
            }
            UserDefaults.standard.set(true, forKey: seedFlagKey)
            changed = true
        } else {
            for key in ornithKeys {
                if profilesByModel[key] == nil {
                    profilesByModel[key] = [
                        ModelProfileType.coder.rawValue: ornithCoder,
                        ModelProfileType.assistant.rawValue: ornithAssistant
                    ]
                    changed = true
                }
            }
            for key in qwenKeys {
                if profilesByModel[key] == nil {
                    profilesByModel[key] = [
                        ModelProfileType.coder.rawValue: qwenCoder,
                        ModelProfileType.assistant.rawValue: qwenAssistant
                    ]
                    changed = true
                }
            }
        }

        if changed {
            saveToStorage()
        }
    }

    private func loadFromStorage() {
        if let savedGlobal = UserDefaults.standard.string(forKey: globalActiveProfileKey),
           let profile = ModelProfileType(rawValue: savedGlobal) {
            self.globalActiveProfile = profile
        }

        if let savedActive = UserDefaults.standard.dictionary(forKey: activeProfilesKey) as? [String: String] {
            self.activeProfileByModel = savedActive
        }

        if let data = UserDefaults.standard.data(forKey: profilesKey),
           let decoded = try? JSONDecoder().decode([String: [String: GenerationProfileSettings]].self, from: data) {
            self.profilesByModel = decoded
        }
    }

    private func saveToStorage() {
        if let encoded = try? JSONEncoder().encode(profilesByModel) {
            UserDefaults.standard.set(encoded, forKey: profilesKey)
        }
    }
}
