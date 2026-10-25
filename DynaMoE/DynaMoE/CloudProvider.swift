//
//  CloudProvider.swift
//  DynaMoE
//
//  Connected cloud providers (Synthetic integration).
//
//  DynaMoE is local-weights-first: this file is where REMOTE inference
//  providers enter the model registry. A cloud provider is a value-only
//  description (endpoints, docs URL); its models appear as virtual
//  `DiscoveredModel` entries with a `cloud://` sentinel path, mirroring the
//  Apple Foundation Model's virtual-entry contract, so the chat UI, session
//  persistence, and the model switch engine all keep working without ever
//  touching a local snapshot.
//
//  Models are not hardcoded: the registered list for a provider comes from
//  the `dynamoe_cloud_<provider>_models` UserDefaults key (see
//  `CloudModelRegistry`). An entry registered there must re-resolve at send
//  time against the live provider, and the Keychain holds credentials —
//  nothing secret ever lives in UserDefaults or on disk.
//

import Foundation

// MARK: - API Flavor

/// The wire dialect a provider's endpoint speaks. Synthetic offers both an
/// OpenAI-compatible and an Anthropic-compatible surface; the chat client
/// implements the OpenAI flavor and keeps the Anthropic flavor behind the
/// same seam so a later flavor does not disturb the registry or UI.
public enum CloudApiFlavor: String, CaseIterable, Identifiable, Codable, Sendable {
    case openAI = "OpenAI"
    case anthropic = "Anthropic"

    public var id: String { rawValue }
}

// MARK: - Provider Record

/// A remote inference provider. Value-only: credentials live in the
/// Keychain (`CloudSecretStore`), never here.
public nonisolated struct CloudProvider: Identifiable, Codable, Equatable, Sendable {
    public let id: String
    public let displayName: String
    public let docsURL: URL
    /// Base URL of the OpenAI-compatible API, including the `/v1` suffix
    /// ("https://api.synthetic.new/openai/v1").
    public let openAIBaseURL: URL
    /// Base of the Anthropic-compatible API when the provider offers one
    /// ("https://api.synthetic.new/anthropic"; its client appends `/v1`).
    public let anthropicBaseURL: URL?

    public init(id: String, displayName: String, docsURL: URL, openAIBaseURL: URL, anthropicBaseURL: URL?) {
        self.id = id
        self.displayName = displayName
        self.docsURL = docsURL
        self.openAIBaseURL = openAIBaseURL
        self.anthropicBaseURL = anthropicBaseURL
    }

    public var chatCompletionsURL: URL {
        openAIBaseURL.appendingPathComponent("chat/completions")
    }

    /// GET endpoint used by the Settings "Test Connection" button: a good
    /// key returns the provider's model list, a bad one returns 401 without
    /// ever spending a completion.
    public var modelsURL: URL {
        openAIBaseURL.appendingPathComponent("models")
    }
}

/// The set of cloud providers DynaMoE knows how to reach. Adding a provider
/// means appending one record here; the Settings UI, registry, and picker
/// all render from this list.
public nonisolated enum CloudProviderRegistry {
    public static let synthetic = CloudProvider(
        id: "synthetic",
        displayName: "Synthetic",
        docsURL: URL(string: "https://dev.synthetic.new/docs/api/overview")!,
        openAIBaseURL: URL(string: "https://api.synthetic.new/openai/v1")!,
        anthropicBaseURL: URL(string: "https://api.synthetic.new/anthropic")!
    )

    public static let all: [CloudProvider] = [synthetic]

    public static func provider(withId id: String) -> CloudProvider? {
        all.first { $0.id == id }
    }
}

// MARK: - Sentinel Addressing

/// How a cloud model is *addressed* inside DynaMoE's weights-centric
/// plumbing. Sessions persist `selectedModelId` / `selectedModelPath`
/// verbatim and re-install them on relaunch, so both forms must round-trip
/// back to (provider, upstream model id):
///
/// - registry id: `cloud-model-<providerId>-<upstreamModelId>`
/// - sentinel path: `cloud://<providerId>/<upstreamModelId>`
///
/// The sentinel path deliberately carries a URL scheme — like the Apple
/// Foundation Model's `apple-foundation-model://on-device` — so it can never
/// collide with a real snapshot directory on disk.
public nonisolated enum CloudModelAddress {
    public static let idPrefix = "cloud-model-"
    public static let pathScheme = "cloud://"

    public static func modelId(providerId: String, providerModelId: String) -> String {
        "\(idPrefix)\(providerId)-\(providerModelId)"
    }

    public static func sentinelPath(providerId: String, providerModelId: String) -> String {
        "\(pathScheme)\(providerId)/\(providerModelId)"
    }

    public static func isCloudModelId(_ id: String?) -> Bool {
        id?.hasPrefix(idPrefix) ?? false
    }

    public static func isCloudPath(_ path: String?) -> Bool {
        path?.hasPrefix(pathScheme) ?? false
    }

    /// Whether a persisted selection claims a cloud model but no longer
    /// resolves (unknown provider, corrupted defaults). Callers must refuse
    /// this instead of falling through to local weights: the turn would run on
    /// a different model than the conversation names.
    public static func selectionIsUnresolved(modelId: String?, modelPath: String?) -> Bool {
        if let modelId, isCloudModelId(modelId), parseModelId(modelId) == nil {
            return true
        }
        if let modelPath, isCloudPath(modelPath), parsePath(modelPath) == nil {
            return true
        }
        return false
    }

    /// Splits a registry id back into its provider and upstream model id.
    /// The provider id is matched against the known registry (so
    /// `cloud-model-synthetic-hf:org/name` resolves provider `synthetic`,
    /// model `hf:org/name`), which keeps provider ids free to use `-` too.
    public static func parseModelId(_ id: String) -> (provider: CloudProvider, providerModelId: String)? {
        guard id.hasPrefix(idPrefix) else { return nil }
        let rest = String(id.dropFirst(idPrefix.count))
        for provider in CloudProviderRegistry.all where rest.hasPrefix(provider.id + "-") {
            let upstream = String(rest.dropFirst(provider.id.count + 1))
            guard !upstream.isEmpty else { return nil }
            return (provider, upstream)
        }
        return nil
    }

    /// Splits a sentinel path `cloud://<providerId>/<upstreamModelId>`.
    public static func parsePath(_ path: String) -> (provider: CloudProvider, providerModelId: String)? {
        guard path.hasPrefix(pathScheme) else { return nil }
        let rest = String(path.dropFirst(pathScheme.count))
        for provider in CloudProviderRegistry.all where rest.hasPrefix(provider.id + "/") {
            let upstream = String(rest.dropFirst(provider.id.count + 1))
            guard !upstream.isEmpty else { return nil }
            return (provider, upstream)
        }
        return nil
    }
}

// MARK: - Registered Cloud Models

/// One chat model a provider can be asked to run. This is a registration
/// record, not provider metadata: capabilities (thinking, context) are
/// editable here because the provider's `/models` listing is only consulted
/// opportunistically.
public nonisolated struct CloudModelSpec: Equatable, Sendable {
    public let providerId: String
    /// Upstream model id exactly as the API expects it ("hf:deepseek-ai/…").
    public let providerModelId: String
    public let displayName: String
    public let supportsThinking: Bool
    /// Whether the provider's endpoint accepts a `tools` array for this model.
    /// Defaults to true; a model registered under the provider's no-tools key
    /// is marked false so a chat turn runs in plain mode instead of failing
    /// mid-loop with an unexplained HTTP 400.
    public let supportsTools: Bool

    public init(
        providerId: String,
        providerModelId: String,
        displayName: String,
        supportsThinking: Bool,
        supportsTools: Bool = true
    ) {
        self.providerId = providerId
        self.providerModelId = providerModelId
        self.displayName = displayName
        self.supportsThinking = supportsThinking
        self.supportsTools = supportsTools
    }
}

/// Builds `DiscoveredModel` entries for cloud providers, appended after the
/// local scan like the Apple Foundation Model entry.
///
/// Registration source: per-provider UserDefaults list. A full model
/// picker/manager is the follow-up model branch's job; until then, power
/// users register models with:
///
///     defaults write DRP.DynaMoE dynamoe_cloud_synthetic_models \
///         "hf:deepseek-ai/DeepSeek-V4.1-Flash; hf:minimax/M2"
///
/// Entries are `;`-separated upstream model ids; display names derive from
/// the id tail and carry the provider name so the chat picker labels them
/// unmistakably. `supportsThinking` is inferred from a "thinking" hint in
/// the id — a deliberate convenience heuristic, not provider truth.
public nonisolated enum CloudModelRegistry {
    public static func modelsDefaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_models"
    }

    /// The UserDefaults key that enables one debug model even when the main
    /// list is empty — the end-to-end smoke-test seam for this branch.
    public static let debugModelDefaultsKey = "dynamoe_cloud_debug_model"

    /// Upstream model ids for which this provider rejects a `tools` array
    /// (same `;`/`,`/newline separated format as the registry list). A model
    /// listed here runs in plain chat with a visible notice instead of
    /// failing mid-loop on an HTTP 400.
    public static func noToolsDefaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_no_tools"
    }

    /// The upstream model ids currently registered for one provider, in
    /// storage order, deduplicated.
    public static func registeredModelIds(
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) -> [String] {
        var ids: [String] = []
        let raw = defaults.string(forKey: modelsDefaultsKey(for: provider.id)) ?? ""
        for chunk in raw.split(whereSeparator: { $0 == ";" || $0 == "," || $0 == "\n" }) {
            let id = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
            if !id.isEmpty { ids.append(id) }
        }
        var seen = Set<String>()
        return ids.filter { seen.insert($0).inserted }
    }

    /// Replaces one provider's registered model list. Written by the Settings
    /// "Sync Models" action from inside the app process, so the value is
    /// always visible to the running app even when an external
    /// `defaults write` landed in a preference cache this process never
    /// re-read.
    public static func setRegisteredModelIds(
        _ modelIds: [String],
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) {
        var seen = Set<String>()
        let cleaned = modelIds
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
        defaults.set(cleaned.joined(separator: "; "), forKey: modelsDefaultsKey(for: provider.id))
    }

    /// Upstream model ids the provider serves without function calling.
    public static func modelIdsWithoutToolSupport(
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) -> Set<String> {
        let raw = defaults.string(forKey: noToolsDefaultsKey(for: provider.id)) ?? ""
        return Set(
            raw.split(whereSeparator: { $0 == ";" || $0 == "," || $0 == "\n" })
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        )
    }

    /// Whether a `/models` catalog entry looks like a chat completion model.
    /// Providers publish embeddings, rerankers, speech, moderation, and image
    /// generators in the same catalog; registering those as chat models puts
    /// pickable entries in the composer that are guaranteed to fail on send.
    public static func isChatModelCandidate(_ providerModelId: String) -> Bool {
        let lower = providerModelId.lowercased()
        let nonChatMarkers = [
            "embed", "rerank", "whisper", "text-to-speech", "speech-to-text",
            "moderation", "transcri", "diffusion", "dall-e", "sdxl", "flux",
            "upscal", "text-to-image", "image-to-image", "image-generation",
            "txt2img", "img2img", "inpaint", "bg-removal"
        ]
        return !nonChatMarkers.contains { lower.contains($0) }
    }

    /// Filters a provider catalog down to chat models. An empty result means
    /// the heuristic recognized nothing rather than that nothing is chattable,
    /// so the unfiltered list wins over registering zero models.
    public static func chatModelIds(from providerModelIds: [String]) -> [String] {
        let chat = providerModelIds.filter { isChatModelCandidate($0) }
        return chat.isEmpty ? providerModelIds : chat
    }

    public static func registeredSpecs(
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) -> [CloudModelSpec] {
        var ids = registeredModelIds(for: provider, defaults: defaults)
        if ids.isEmpty,
           let debugId = defaults.string(forKey: debugModelDefaultsKey)?
               .trimmingCharacters(in: .whitespacesAndNewlines),
           !debugId.isEmpty {
            ids.append(debugId)
        }
        let withoutTools = modelIdsWithoutToolSupport(for: provider, defaults: defaults)
        return ids.map { id in
            CloudModelSpec(
                providerId: provider.id,
                providerModelId: id,
                displayName: defaultDisplayName(provider: provider, providerModelId: id),
                supportsThinking: id.lowercased().contains("thinking"),
                supportsTools: !withoutTools.contains(id)
            )
        }
    }

    /// Human-facing name for an upstream id.
    ///
    /// - `hf:<org>/<model>` ids use the model tail:
    ///   `DeepSeek-V4.1-Flash-1017 (Synthetic)`.
    /// - Provider aliases (`syn:large:text`) are humanized segment by segment:
    ///   `Syn Large Text (Synthetic)`.
    ///
    /// The provider suffix is always present so a cloud row can never be
    /// mistaken for a local one; the chat picker keys its cloud icon off it
    /// (see `ChatDetailView.modelIconName`).
    public static func defaultDisplayName(provider: CloudProvider, providerModelId: String) -> String {
        let trimmed = providerModelId.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.lowercased().hasPrefix("hf:") {
            let tail = trimmed.split(separator: "/").last.map(String.init)
                ?? String(trimmed.dropFirst(3))
            return "\(tail) (\(provider.displayName))"
        }
        if trimmed.contains(":") {
            let humanized = trimmed
                .split(separator: ":")
                .map { $0.replacingOccurrences(of: "_", with: " ").capitalized }
                .joined(separator: " ")
            return "\(humanized) (\(provider.displayName))"
        }
        return "\(trimmed) (\(provider.displayName))"
    }

    public static func makeDiscoveredModel(for spec: CloudModelSpec) -> DiscoveredModel {
        DiscoveredModel(
            id: CloudModelAddress.modelId(providerId: spec.providerId, providerModelId: spec.providerModelId),
            displayName: spec.displayName,
            author: spec.providerId,
            repoId: spec.providerModelId,
            snapshotPath: CloudModelAddress.sentinelPath(providerId: spec.providerId, providerModelId: spec.providerModelId),
            weightsEntryPath: CloudModelAddress.sentinelPath(providerId: spec.providerId, providerModelId: spec.providerModelId),
            sizeBytes: 0,
            formattedSize: "Cloud",
            architectureName: "CloudRemoteModel",
            isMoE: false,
            hasTokenizer: false,
            // A fixed epoch, like the AFM entry: with no file on disk, a
            // per-call Date() would make every scan look newly changed.
            lastModified: Date(timeIntervalSince1970: 0),
            quantization: nil,
            rawModelType: "cloud-\(spec.providerId)",
            supportsThinking: spec.supportsThinking
        )
    }

    public static func makeDiscoveredModel(
        provider: CloudProvider,
        providerModelId: String
    ) -> DiscoveredModel {
        makeDiscoveredModel(for: CloudModelSpec(
            providerId: provider.id,
            providerModelId: providerModelId,
            displayName: defaultDisplayName(provider: provider, providerModelId: providerModelId),
            supportsThinking: providerModelId.lowercased().contains("thinking")
        ))
    }

    /// All registered entries across all known providers. Appended, never
    /// prepended: `LocalModelManager.getDefaultOrFirstModel()` requires
    /// `discoveredModels.first` to be a weights model.
    ///
    /// Entries are only offered once the provider has a stored credential:
    /// without one every turn is guaranteed to fail, so listing those models
    /// would put unpickable wishes in the composer.
    public static func makeDiscoveredModels(
        defaults: UserDefaults = .standard,
        hasCredential: (String) -> Bool = { CloudSecretStore.shared.exists(providerId: $0) }
    ) -> [DiscoveredModel] {
        CloudProviderRegistry.all.flatMap { provider in
            let specs = registeredSpecs(for: provider, defaults: defaults)
            guard !specs.isEmpty, hasCredential(provider.id) else { return [] as [DiscoveredModel] }
            return specs.map { makeDiscoveredModel(for: $0) }
        }
    }

    /// Entry built from a bare (provider, model) pair — the session-restore
    /// fallback when a persisted cloud id is no longer in the registered
    /// list, so an old conversation still routes its turns instead of
    /// silently falling back to a weights load of a phantom path.
    public static func adHocDiscoveredModel(
        provider: CloudProvider,
        providerModelId: String
    ) -> DiscoveredModel {
        makeDiscoveredModel(provider: provider, providerModelId: providerModelId)
    }
}

// MARK: - DiscoveredModel Probing

public extension DiscoveredModel {
    /// True for any virtual cloud-provider registry entry.
    var isCloudModel: Bool {
        CloudModelAddress.isCloudModelId(id) || CloudModelAddress.isCloudPath(snapshotPath)
    }

    /// Resolves this entry to its (provider, upstream model id), or nil for
    /// local weights / Apple Foundation Model entries.
    var cloudResolution: (provider: CloudProvider, providerModelId: String)? {
        if let parsed = CloudModelAddress.parseModelId(id) { return parsed }
        return CloudModelAddress.parsePath(snapshotPath)
    }
}
