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
        let isOrnith = (modelName ?? "").localizedCaseInsensitiveContains("ornith")
        let isJetSpecDefault = !isOrnith
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
                jetSpecEnabled: isJetSpecDefault,
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
                jetSpecEnabled: isJetSpecDefault,
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
    }

    public func normalizeKey(_ modelIdentifier: String) -> String {
        let trimmed = modelIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.contains("/") {
            // Can be snapshot path like /Users/.../snapshots/xxx or repoId like mlx-community/Ornith-1.5-9B
            let components = trimmed.split(separator: "/")
            if let last = components.last, !last.isEmpty {
                if last.count >= 32, components.count >= 2 {
                    let parent = components[components.count - 2]
                    return String(parent).replacingOccurrences(of: "models--", with: "").replacingOccurrences(of: "--", with: "/").lowercased()
                }
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
        if let modelProfiles = profilesByModel[key],
           let settings = modelProfiles[type.rawValue] {
            return settings
        }
        if let modelProfiles = profilesByModel[modelId],
           let settings = modelProfiles[type.rawValue] {
            return settings
        }
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
        profilesByModel[key]?.removeValue(forKey: type.rawValue)
        if profilesByModel[key]?.isEmpty == true {
            profilesByModel.removeValue(forKey: key)
        }
        saveToStorage()
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
