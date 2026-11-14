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

    private enum LocalFamily {
        case qwen
        case ornith
    }

    private static func localFamily(for modelName: String?) -> LocalFamily? {
        guard !CloudModelAddress.isCloudModelId(modelName),
              !CloudModelAddress.isCloudPath(modelName) else {
            return nil
        }
        let nameLower = (modelName ?? "").lowercased()
        if nameLower.contains("qwen") { return .qwen }
        if nameLower.contains("ornith") { return .ornith }
        return nil
    }

    public func profileDisplayName(for modelName: String? = nil) -> String {
        switch Self.localFamily(for: modelName) {
        case .qwen:
            switch self {
            case .coder: return "Coding & Agentic (Thinking Mode)"
            case .assistant: return "General Assistant (Instruct Mode)"
            }
        case .ornith:
            switch self {
            case .coder: return "Precise Coding & Tool Calling"
            case .assistant: return "General Chat / Agent Loops"
            }
        case nil:
            return self.rawValue
        }
    }

    public func description(for modelName: String? = nil) -> String {
        switch Self.localFamily(for: modelName) {
        case .qwen:
            switch self {
            case .coder:
                return "Official Qwen Thinking Mode: High-entropy exploration (Temp 1.0, Pres 0.0) optimized for SWE-bench, complex code, and multi-tool execution."
            case .assistant:
                return "Official Qwen Instruct Mode: Direct non-reasoning responses (Temp 0.70, Pres 1.5) optimized for quick chat and summarization."
            }
        case .ornith:
            switch self {
            case .coder:
                return "Official Ornith Precise Profile: Low entropy (Temp 0.60, Pres 0.0) preventing code syntax drift and schema violation."
            case .assistant:
                return "Official Ornith Agentic Loop Profile: Dynamic variety (Temp 1.00, Pres 1.5) for open-ended benchmark reproduction and chat."
            }
        case nil:
            return self.description
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

    /// `modelPath` is asked about only where the NAME cannot answer, which is
    /// Nanbeige 4.3: its identity is structural (`ModelConfig.isNanbeige43Shape`),
    /// so a renamed copy would otherwise fall through to the generic profile.
    /// Every other family here is matched by name because its name is the only
    /// discriminator that exists.
    ///
    /// A cloud model id is asked about NEITHER. The name of a provider model
    /// id describes someone else's checkpoint on someone else's hardware: Zen
    /// ships `qwen3-coder` and Synthetic serves `hf:Qwen/...` as first-class
    /// ids, and matching `qwen` in those names handed a remote model this app's
    /// 8,192-token local budget plus a `temperature 1.00` /
    /// `presencePenalty 1.50` pair tuned for a checkpoint the provider is not
    /// running. Name-gating is a statement about LOCAL families, so the whole
    /// gate is skipped for a cloud address and the generic (cloud) branch below
    /// answers instead.
    ///
    /// Both cloud address forms are recognised, and the sentinel is not merely
    /// decorative: `ContentView.applyProfile` is handed `cloud://<provider>/<model>`
    /// (`activeLoadedModelPath`) after every cloud activation, while the registry id
    /// `cloud-model-<provider>-<model>` is what `switchModel` passes. A gate that
    /// only understood the id form would leave the path the app actually uses on
    /// the family branch.
    public static func defaultFor(
        type: ModelProfileType,
        modelName: String? = nil,
        modelPath: String? = nil
    ) -> GenerationProfileSettings {
        let isCloud = CloudModelAddress.isCloudModelId(modelName)
            || CloudModelAddress.isCloudPath(modelName)
            || CloudModelAddress.isCloudPath(modelPath)
        let nameLower = isCloud ? "" : (modelName ?? "").lowercased()
        let isQwen = nameLower.contains("qwen")
        let isOrnith = nameLower.contains("ornith")
        let isNanbeige = nameLower.contains("nanbeige")
            || (!isCloud && (modelPath.map { ModelConfig.isNanbeige43Snapshot($0) } ?? false))

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
                // Official Ornith-1.5-9B Precise Coding & Tool Calling Profile.
                // Note: an earlier revision raised repetitionPenalty here (1.10 /
                // 1.20) to fight template loops, but that is counterproductive —
                // the penalty also perturbs VERBATIM copying (the model mangled
                // `/Users/derekparris` to `derekeparris`/`derek Harris`), and it did
                // not stop the loops anyway. Kept at the official 1.00; loop damage
                // is handled by the parser repair + failure caps instead.
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

        // Nanbeige 4.3 is a dense 33B tuned for agentic tool use. Its card quotes
        // temperature 1.0 / top_p 0.95 for agentic work and 0.6 / 0.95 for plain
        // reasoning, and NO repetition penalty (unlike the R1-style families).
        if isNanbeige {
            switch type {
            case .coder:
                return GenerationProfileSettings(
                    temperature: 1.00,
                    topP: 0.95,
                    minP: 0.00,
                    topK: 20,
                    repetitionPenalty: 1.00,
                    presencePenalty: 0.00,
                    maxNewTokens: 65536,
                    systemPrompt: "You are a helpful, respectful, and honest AI assistant.",
                    jetSpecEnabled: false,
                    jetSpecMaxDepth: 3,
                    jetSpecBranchingFactor: 2,
                    jetSpecMaxExpertCap: 8
                )
            case .assistant:
                return GenerationProfileSettings(
                    temperature: 0.60,
                    topP: 0.95,
                    minP: 0.00,
                    topK: 20,
                    repetitionPenalty: 1.00,
                    presencePenalty: 0.00,
                    maxNewTokens: 262144,
                    systemPrompt: "You are a helpful, respectful, and honest AI assistant.",
                    jetSpecEnabled: false,
                    jetSpecMaxDepth: 3,
                    jetSpecBranchingFactor: 2,
                    jetSpecMaxExpertCap: 8
                )
            }
        }

        // Generic fallback defaults.
        //
        // This branch is the CLOUD profile, and it is reached by every cloud
        // model because the family gate above skips a cloud address entirely —
        // not because a provider's model id happens to match no family name
        // (several do: `qwen3-coder` is a shipped Zen id).
        //
        // It is also the default for an ungated LOCAL checkpoint (Llama, Gemma,
        // Spark, DeepSeek), which is why these numbers cannot be cloud-only
        // reasoning: a local turn resolves them as its reply budget too, and
        // `turnFit` lowers the allocation rather than refusing the turn when the
        // KV budget cannot hold `prompt + this`. On the 16 GB reference machine
        // a long local prompt therefore clamps more often than it did at 8,192.
        //
        // They were 8,192 / 4,096, a budget sized for a local checkpoint's KV
        // appetite, which crippled a capable remote model's ability to use the
        // harness (a long tool step or a full file rewrite truncated mid-answer).
        // Doubled, and deliberately NOT raised to the policy ceiling: the slider
        // is the ceiling and the cloud path derives it per window
        // (`CloudContextPolicy.maximumOutputTokens`), while this default has to be
        // safe for a provider nobody recorded a window for — which is exactly
        // `CloudContextPolicy.unknownWindowReplyCeiling`.
        //
        // That constant IS the coder budget, read from it rather than restated: a
        // literal here would be a second copy of a number whose whole job is to
        // be the budget an unrecorded-window turn is clamped to, and the two
        // could drift apart with nothing to notice. The assistant budget is half
        // of it, the 2:1 coder:assistant ratio the shipped profiles keep.
        let cloudReplyCeiling = CloudContextPolicy.unknownWindowReplyCeiling
        switch type {
        case .coder:
            return GenerationProfileSettings(
                temperature: 0.60,
                topP: 0.95,
                minP: 0.00,
                topK: 20,
                repetitionPenalty: 1.00,
                presencePenalty: 0.00,
                maxNewTokens: cloudReplyCeiling,
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
                maxNewTokens: cloudReplyCeiling / 2,
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

    /// Where these profiles are persisted.
    ///
    /// Injectable because the app's domain (`DRP.DynaMoE`) is shared with the
    /// test host: a test that saves through `shared` rewrites the developer's own
    /// profiles, and no write is distinguishable from a real one afterwards. A
    /// test that exercises a WRITE path builds a manager over its own suite
    /// (`ModelProfileManager(defaults:)`) instead, the way
    /// `VisionAttachmentLogicTests` already does for its own keys.
    private let defaults: UserDefaults

    @Published public private(set) var profilesByModel: [String: [String: GenerationProfileSettings]] = [:]
    @Published public private(set) var activeProfileByModel: [String: String] = [:]
    @Published public var globalActiveProfile: ModelProfileType = .coder {
        didSet {
            defaults.set(globalActiveProfile.rawValue, forKey: globalActiveProfileKey)
        }
    }

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        loadFromStorage()
        seedOfficialProfiles()
    }

    /// The storage key for one address.
    ///
    /// Cloud addresses come FIRST, before any of the local path rules, because
    /// those rules are what broke them: a sentinel `cloud://zen/qwen` contains
    /// `/`, so the local rules collapsed it to its last path component
    /// (`qwen`) — the key of the seeded LOCAL Qwen profile. `getProfile` then
    /// answered a cloud activation with the local checkpoint's 8,192-token
    /// budget and its `presencePenalty 1.50`, and `saveProfile` through the same
    /// sentinel overwrote that local profile with cloud sampling values.
    ///
    /// Both cloud forms collapse to ONE canonical key
    /// (`CloudModelAddress.profileKey`), so a read and a write through different
    /// forms of the same model meet. A cloud-shaped address whose provider has
    /// gone unresolvable keeps its own key — the whole lowercased address —
    /// rather than degrading to a local family name it must never inherit.
    public func normalizeKey(_ modelIdentifier: String) -> String {
        let trimmed = modelIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)

        if CloudModelAddress.isCloudModelId(trimmed) || CloudModelAddress.isCloudPath(trimmed) {
            return CloudModelAddress.profileKey(trimmed) ?? trimmed.lowercased()
        }

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
        // A cloud address answers only to a cloud key. Every rule below this
        // point is a guess about LOCAL checkpoints — a short repo name, a family
        // name — and both of them are wrong for a provider's model id, whose name
        // describes someone else's checkpoint (`qwen3-coder` is a shipped Zen id).
        // The check is made once, before the guesses, because the guesses are
        // what used to answer: `cloud://zen/qwen` collapsed to `qwen` and matched
        // the seeded 9B profile at step 1 outright, and an unresolvable sentinel
        // would reach step 3's `/` rule and land on the same key.
        let isCloudAddress = CloudModelAddress.isCloudModelId(modelId)
            || CloudModelAddress.isCloudPath(modelId)
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
        if !isCloudAddress, key.contains("/") {
            let short = String(key.split(separator: "/").last ?? "")
            if !short.isEmpty, let modelProfiles = profilesByModel[short], let settings = modelProfiles[type.rawValue] {
                return settings
            }
        }

        // 4. Model family matching for known profiles.
        //
        // Skipped for a cloud address for the same reason `defaultFor` skips it:
        // `cloud-model-zen-qwen3-coder` contains "qwen" and would otherwise
        // inherit the seeded 9B Qwen profile — an 8,192-token budget and a
        // sampling pair tuned for a checkpoint the provider is not serving.
        // Steps 1-3 still apply, so a cloud id with a profile saved against it
        // is honoured: those are matched by exact key, not by substring.
        let keyLower = isCloudAddress ? "" : key.lowercased()
        let idLower = isCloudAddress ? "" : modelId.lowercased()
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
        } else if keyLower.contains("nanbeige") || idLower.contains("nanbeige") {
            for (candidateKey, profiles) in profilesByModel where candidateKey.lowercased().contains("nanbeige") {
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
            defaults.set(activeProfileByModel, forKey: activeProfilesKey)
        }
    }

    public func resetProfile(for modelId: String?, type: ModelProfileType) {
        let key = normalizeKey(modelId ?? "default")
        let defaultSettings = GenerationProfileSettings.defaultFor(type: type, modelName: modelId)

        // A cloud address takes the REMOVAL branch even when its upstream id
        // names a local family: the family check below exists to keep a seeded
        // official profile in place for the checkpoints it was written for, and
        // a remote model id was never one of them. Storing the generic profile
        // under a cloud key would also outlive the reset, because `defaultFor`
        // is what step 5 would have answered with anyway.
        let isCloudAddress = CloudModelAddress.isCloudModelId(modelId)
            || CloudModelAddress.isCloudPath(modelId)

        let lower = isCloudAddress ? "" : (modelId ?? "").lowercased()
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

        let nanbeigeCoder = GenerationProfileSettings.defaultFor(type: .coder, modelName: "nanbeige")
        let nanbeigeAssistant = GenerationProfileSettings.defaultFor(type: .assistant, modelName: "nanbeige")

        let nanbeigeKeys = [
            "nanbeige/nanbeige4.3-33b",
            "nanbeige4.3-33b",
            "nanbeige"
        ]

        // Deliberately NOT bumped for Nanbeige. The `else` branch below adds a
        // profile for any key that has none, so a new family needs no flag change;
        // bumping it would make `alreadySeeded` false for every existing user and
        // overwrite customised ornith and qwen generation settings.
        let seedFlagKey = "dynamoe_seeded_official_profiles_v3"
        let alreadySeeded = defaults.bool(forKey: seedFlagKey)

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
            for key in nanbeigeKeys {
                profilesByModel[key] = [
                    ModelProfileType.coder.rawValue: nanbeigeCoder,
                    ModelProfileType.assistant.rawValue: nanbeigeAssistant
                ]
            }
            defaults.set(true, forKey: seedFlagKey)
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
            for key in nanbeigeKeys {
                if profilesByModel[key] == nil {
                    profilesByModel[key] = [
                        ModelProfileType.coder.rawValue: nanbeigeCoder,
                        ModelProfileType.assistant.rawValue: nanbeigeAssistant
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
        if let savedGlobal = defaults.string(forKey: globalActiveProfileKey),
           let profile = ModelProfileType(rawValue: savedGlobal) {
            self.globalActiveProfile = profile
        }

        if let savedActive = defaults.dictionary(forKey: activeProfilesKey) as? [String: String] {
            self.activeProfileByModel = savedActive
        }

        if let data = defaults.data(forKey: profilesKey),
           let decoded = try? JSONDecoder().decode([String: [String: GenerationProfileSettings]].self, from: data) {
            self.profilesByModel = decoded
        }
    }

    private func saveToStorage() {
        if let encoded = try? JSONEncoder().encode(profilesByModel) {
            defaults.set(encoded, forKey: profilesKey)
        }
    }
}
