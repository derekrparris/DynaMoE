//
//  CloudProvider.swift
//  DynaMoE
//
//  Connected cloud providers (built-in and user-configured).
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
    /// The provider's own documentation. Optional because a provider added
    /// from Settings may not have a link worth keeping.
    public let docsURL: URL?
    /// Base URL of the OpenAI-compatible API, including the `/v1` suffix
    /// ("https://api.synthetic.new/openai/v1").
    public let openAIBaseURL: URL
    /// Base of the Anthropic-compatible API when the provider offers one
    /// ("https://api.synthetic.new/anthropic"; its client appends `/v1`).
    public let anthropicBaseURL: URL?
    /// What this endpoint is willing to be told about prompt caching. Optional
    /// so a record written before the field existed still decodes; nil means
    /// the default (`affinity`): headers an HTTP server is required to ignore
    /// when it does not recognise them, so they change nothing for an endpoint
    /// that does not route on them.
    public let cacheHints: CloudCacheHints?

    public init(
        id: String,
        displayName: String,
        docsURL: URL? = nil,
        openAIBaseURL: URL,
        anthropicBaseURL: URL? = nil,
        cacheHints: CloudCacheHints? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.docsURL = docsURL
        self.openAIBaseURL = openAIBaseURL
        self.anthropicBaseURL = anthropicBaseURL
        self.cacheHints = cacheHints
    }

    /// The capability in force: an unset record gets session-affinity headers,
    /// which every HTTP server ignores safely, and nothing else.
    public var resolvedCacheHints: CloudCacheHints { cacheHints ?? .affinity }

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

/// The set of cloud providers DynaMoE knows how to reach: the built-ins below
/// plus every provider the user added in Settings. Adding a provider no longer
/// means appending a record here — Settings writes one into
/// `CloudProviderCatalog` — but a built-in stays a literal so the registry can
/// never be empty.
public nonisolated enum CloudProviderRegistry {
    public static let synthetic = CloudProvider(
        id: "synthetic",
        displayName: "Synthetic",
        docsURL: URL(string: "https://dev.synthetic.new/docs/api/overview")!,
        openAIBaseURL: URL(string: "https://api.synthetic.new/openai/v1")!,
        anthropicBaseURL: URL(string: "https://api.synthetic.new/anthropic")!,
        // Synthetic fronts OpenAI-shaped backends, so the documented
        // `prompt_cache_key` routing hint is the one body field worth sending;
        // a rejection is retried once without it.
        cacheHints: .openAI
    )

    /// Built-ins first, then the runtime-configured list in the order it was
    /// added. Read through `CloudProviderRuntime` because the model scan and
    /// sentinel parsing call this off the main actor.
    public static var all: [CloudProvider] { CloudProviderRuntime.shared.providers }

    public static func provider(withId id: String) -> CloudProvider? {
        all.first { $0.id == id }
    }
}

// MARK: - Provider Validation

/// Why a provider draft cannot be saved. Every case is actionable in the
/// Settings form, and `errorDescription` is the text that field shows.
public nonisolated enum CloudProviderValidationError: LocalizedError, Equatable {
    case emptyDisplayName
    case invalidIdentifier
    case duplicateIdentifier(String)
    case reservedIdentifier(String)
    case invalidBaseURL
    case insecureBaseURL
    case baseURLIncludesEndpoint
    case baseURLIncludesQuery
    case credentialsInBaseURL
    case invalidDocsURL
    case credentialsInDocsURL
    case docsURLIncludesQuery

    public var errorDescription: String? {
        switch self {
        case .emptyDisplayName:
            return "Give the provider a display name."
        case .invalidIdentifier:
            return "Use 1-\(CloudProviderCatalog.maximumIdentifierLength) lowercase letters, digits, or dashes, starting and ending with a letter or digit."
        case .duplicateIdentifier(let identifier):
            return "A provider with the identifier \"\(identifier)\" already exists."
        case .reservedIdentifier(let identifier):
            return "\"\(identifier)\" belongs to a built-in provider."
        case .invalidBaseURL:
            return "Enter an absolute URL, for example https://opencode.ai/zen/v1."
        case .insecureBaseURL:
            return "Use https://. Plain http:// is accepted only for 127.0.0.1, localhost, and ::1."
        case .baseURLIncludesEndpoint:
            return "Enter the API base, not the whole endpoint: stop before /chat/completions or /models."
        case .baseURLIncludesQuery:
            return "Remove the query string from the API base URL. DynaMoE appends the endpoint path itself, so a query such as ?api-version=… would be dropped and the request would fail later."
        case .credentialsInBaseURL:
            return "Remove the user:password from the URL. The API key goes in the key field and rides the Authorization header, never the URL."
        case .invalidDocsURL:
            return "Enter a documentation link that starts with https:// or http://."
        case .credentialsInDocsURL:
            return "Remove the user:password from the documentation link. It is stored in plain text with the provider record, so a credential there would be written to disk."
        case .docsURLIncludesQuery:
            return "Remove the query string from the documentation link. DynaMoE stores the link verbatim, so a query can carry a credential onto disk."
        }
    }
}

// MARK: - Provider Catalog

/// Built-in plus user-configured providers, persisted in UserDefaults as a
/// JSON array under `dynamoe_cloud_custom_providers`.
///
/// Every function takes the defaults it may read or write, which keeps the
/// validation and persistence rules testable without mutating the running
/// app's list. `CloudProviderRuntime` is the cached view the app reads.
public nonisolated enum CloudProviderCatalog {
    public static let customDefaultsKey = "dynamoe_cloud_custom_providers"
    public static let maximumIdentifierLength = 32

    /// Providers that ship with the app. They cannot be edited or removed, so
    /// the registry can never be empty and `synthetic` always resolves.
    public static var builtIns: [CloudProvider] { [CloudProviderRegistry.synthetic] }

    public static func customProviders(defaults: UserDefaults = .standard) -> [CloudProvider] {
        guard let data = defaults.data(forKey: customDefaultsKey) else { return [] }
        return decodeCustomProviders(from: data)
    }

    public static func all(defaults: UserDefaults = .standard) -> [CloudProvider] {
        builtIns + customProviders(defaults: defaults)
    }

    public static func provider(withId id: String, defaults: UserDefaults = .standard) -> CloudProvider? {
        all(defaults: defaults).first { $0.id == id }
    }

    public static func saveCustomProviders(
        _ providers: [CloudProvider],
        defaults: UserDefaults = .standard
    ) {
        var seen = Set(builtIns.map(\.id))
        let cleaned = providers.filter { seen.insert($0.id).inserted && isValidIdentifier($0.id) }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(cleaned) else { return }
        defaults.set(data, forKey: customDefaultsKey)
    }

    /// `"OpenCode Zen"` -> `"opencode-zen"`. Every run of punctuation or
    /// non-ASCII becomes one dash, so the result is always a valid identifier.
    public static func slug(forDisplayName name: String) -> String {
        var slug = ""
        var pendingDash = false
        for scalar in name.lowercased().unicodeScalars {
            let value = scalar.value
            let isDigit = value >= 48 && value <= 57
            let isLowercaseLetter = value >= 97 && value <= 122
            if isDigit || isLowercaseLetter {
                if pendingDash && !slug.isEmpty { slug.append("-") }
                slug.unicodeScalars.append(scalar)
                pendingDash = false
            } else if !slug.isEmpty {
                pendingDash = true
            }
        }
        if slug.count > maximumIdentifierLength {
            slug = String(slug.prefix(maximumIdentifierLength))
            while slug.hasSuffix("-") { slug.removeLast() }
        }
        return slug
    }

    /// The one place a draft becomes a provider. `existing` is the list the
    /// identifier must not collide with; a caller editing a provider passes
    /// the list without that provider so its own identifier stays available.
    /// `preserving` carries the fields the edit form does not own — the
    /// Anthropic base and the prompt-caching capability — so a rename or a new
    /// docs link cannot silently reset them to their defaults.
    public static func validate(
        displayName: String,
        identifier: String,
        baseURL: String,
        docsURL: String = "",
        existing: [CloudProvider]? = nil,
        preserving preserved: CloudProvider? = nil
    ) -> Result<CloudProvider, CloudProviderValidationError> {
        let name = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return .failure(.emptyDisplayName) }

        let id = identifier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard isValidIdentifier(id) else { return .failure(.invalidIdentifier) }
        if builtIns.contains(where: { $0.id == id }) { return .failure(.reservedIdentifier(id)) }
        if (existing ?? all()).contains(where: { $0.id == id }) {
            return .failure(.duplicateIdentifier(id))
        }

        let normalizedBase: URL
        switch checkBaseURL(baseURL) {
        case .ok(let value):
            normalizedBase = value
        case .invalid:
            return .failure(.invalidBaseURL)
        case .insecure:
            return .failure(.insecureBaseURL)
        case .includesEndpoint:
            return .failure(.baseURLIncludesEndpoint)
        case .includesQuery:
            return .failure(.baseURLIncludesQuery)
        case .includesCredentials:
            return .failure(.credentialsInBaseURL)
        }

        let rawDocs = docsURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let docs: URL?
        if rawDocs.isEmpty {
            docs = nil
        } else {
            switch checkDocsURL(rawDocs) {
            case .ok(let url):
                docs = url
            case .invalid:
                return .failure(.invalidDocsURL)
            case .includesCredentials:
                return .failure(.credentialsInDocsURL)
            case .includesQuery:
                return .failure(.docsURLIncludesQuery)
            }
        }

        return .success(CloudProvider(
            id: id,
            displayName: name,
            docsURL: docs,
            openAIBaseURL: normalizedBase,
            anthropicBaseURL: preserved?.anthropicBaseURL,
            cacheHints: preserved?.cacheHints
        ))
    }

    // MARK: Identifier & URL rules

    static func isValidIdentifier(_ id: String) -> Bool {
        guard !id.isEmpty, id.count <= maximumIdentifierLength else { return false }
        guard id.first != "-", id.last != "-" else { return false }
        return id.unicodeScalars.allSatisfy { scalar in
            let value = scalar.value
            return (value >= 48 && value <= 57)      // 0-9
                || (value >= 97 && value <= 122)     // a-z
                || value == 45                       // -
        }
    }

    private enum BaseURLCheck {
        case ok(URL)
        case invalid
        case insecure
        case includesEndpoint
        case includesQuery
        case includesCredentials
    }

    /// Accepts an absolute URL, prefixes a missing scheme with https, strips a
    /// trailing slash, and refuses the whole endpoint instead of the base.
    /// User-info is refused rather than dropped: a silently stripped credential
    /// would 401 later with nothing naming the cause. A query is refused for
    /// the same reason a credential in the path would be: the request URL is
    /// built by appending a path, so a base that needs `?api-version=…` would
    /// have its query dropped and fail later with nothing naming the cause.
    /// The scheme and host are lowercased, so the stored URL is the one the row
    /// shows and the one URLSession sees. The fragment is dropped: it is never
    /// sent to a server.
    private static func checkBaseURL(_ raw: String) -> BaseURLCheck {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .invalid }
        let candidate = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
        guard var components = URLComponents(string: candidate),
              let scheme = components.scheme?.lowercased(),
              let host = components.host, !host.isEmpty else { return .invalid }
        guard scheme == "https" || scheme == "http" else { return .invalid }
        if scheme == "http", !isLoopbackHost(host) { return .insecure }
        if components.user != nil || components.password != nil {
            return .includesCredentials
        }
        if components.query != nil { return .includesQuery }

        var path = components.path
        while path.hasSuffix("/") { path.removeLast() }
        let lower = path.lowercased()
        if lower.hasSuffix("/chat/completions") || lower.hasSuffix("/models") {
            return .includesEndpoint
        }

        components.scheme = scheme
        components.host = host.lowercased()
        components.path = path
        components.query = nil
        components.fragment = nil
        guard let url = components.url, url.host != nil else { return .invalid }
        return .ok(url)
    }

    private enum DocsURLCheck {
        case ok(URL)
        case invalid
        case includesCredentials
        case includesQuery
    }

    /// The docs link is displayed and opened from Settings and persisted
    /// verbatim in the plaintext provider list, so it refuses user-info and a
    /// query for the same reason the base URL does: neither may carry a secret
    /// onto disk. The fragment is kept (a section anchor is meaningful and
    /// never sent to a server) and the URL is otherwise not normalized, since
    /// documentation paths are case-sensitive.
    private static func checkDocsURL(_ raw: String) -> DocsURLCheck {
        guard let components = URLComponents(string: raw),
              let scheme = components.scheme?.lowercased(),
              scheme == "https" || scheme == "http",
              let host = components.host, !host.isEmpty else { return .invalid }
        if components.user != nil || components.password != nil {
            return .includesCredentials
        }
        if components.query != nil { return .includesQuery }
        guard let url = components.url else { return .invalid }
        return .ok(url)
    }

    private static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .lowercased()
        return normalized == "127.0.0.1" || normalized == "::1" || normalized == "localhost"
    }

    // MARK: Lenient persistence

    /// One stored entry that may fail to decode without taking the list with
    /// it: a hand-edited or half-written record must cost that provider, not
    /// every provider.
    private struct FailableProvider: Decodable {
        let provider: CloudProvider?

        init(from decoder: Decoder) throws {
            provider = try? CloudProvider(from: decoder)
        }
    }

    static func decodeCustomProviders(from data: Data) -> [CloudProvider] {
        guard let decoded = try? JSONDecoder().decode([FailableProvider].self, from: data) else {
            return []
        }
        var seen = Set(builtIns.map(\.id))
        var providers: [CloudProvider] = []
        for entry in decoded {
            guard let stored = entry.provider,
                  isValidIdentifier(stored.id),
                  seen.insert(stored.id).inserted,
                  let restored = restoredProvider(from: stored) else { continue }
            providers.append(restored)
        }
        return providers
    }

    /// A persisted record is re-validated when it is READ, not only when it is
    /// written: the bytes may predate the URL rules or have been hand-edited,
    /// and a turn must never run on a plaintext `http://` base or one carrying
    /// a credential. The record is normalized the same way a fresh draft is, so
    /// a stored trailing slash cannot make the row disagree with the request.
    /// A record that fails costs itself, not the list.
    private static func restoredProvider(from stored: CloudProvider) -> CloudProvider? {
        guard case .ok(let base) = checkBaseURL(stored.openAIBaseURL.absoluteString) else {
            return nil
        }
        let anthropic: URL?
        if let raw = stored.anthropicBaseURL {
            guard case .ok(let normalized) = checkBaseURL(raw.absoluteString) else {
                return nil
            }
            anthropic = normalized
        } else {
            anthropic = nil
        }
        let docs: URL?
        if let raw = stored.docsURL {
            guard case .ok(let normalized) = checkDocsURL(raw.absoluteString) else {
                return nil
            }
            docs = normalized
        } else {
            docs = nil
        }
        return CloudProvider(
            id: stored.id,
            displayName: stored.displayName,
            docsURL: docs,
            openAIBaseURL: base,
            anthropicBaseURL: anthropic,
            cacheHints: stored.cacheHints
        )
    }
}

// MARK: - Runtime Snapshot

/// The provider list the app is actually using. `CloudModelAddress` and the
/// model scan read it off the main actor while Settings writes it, so the
/// snapshot is a lock-guarded array of `Sendable` values rather than a
/// main-actor property.
public nonisolated final class CloudProviderRuntime: @unchecked Sendable {
    public static let shared = CloudProviderRuntime()

    private let lock = NSLock()
    private var snapshot: [CloudProvider]

    private init(defaults: UserDefaults = .standard) {
        snapshot = CloudProviderCatalog.all(defaults: defaults)
    }

    public var providers: [CloudProvider] {
        lock.lock()
        defer { lock.unlock() }
        return snapshot
    }

    /// Re-reads the persisted list. Called after every Settings mutation so a
    /// background scan and the next turn see the same providers.
    public func refresh(defaults: UserDefaults = .standard) {
        let loaded = CloudProviderCatalog.all(defaults: defaults)
        lock.lock()
        snapshot = loaded
        lock.unlock()
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
    /// The provider id is matched against the configured providers (so
    /// `cloud-model-synthetic-hf:org/name` resolves provider `synthetic`,
    /// model `hf:org/name`), which keeps provider ids free to use `-` too.
    ///
    /// Providers are tried LONGEST id first: with `my` and `my-provider`
    /// configured, `cloud-model-my-provider-x` belongs to `my-provider`. A
    /// match that leaves an empty upstream id is not a candidate, so
    /// `cloud-model-my-provider` still resolves to `my` + `provider`.
    public static func parseModelId(
        _ id: String,
        providers: [CloudProvider] = CloudProviderRegistry.all
    ) -> (provider: CloudProvider, providerModelId: String)? {
        guard id.hasPrefix(idPrefix) else { return nil }
        let rest = String(id.dropFirst(idPrefix.count))
        for provider in longestFirst(providers) where rest.hasPrefix(provider.id + "-") {
            let upstream = String(rest.dropFirst(provider.id.count + 1))
            if !upstream.isEmpty { return (provider, upstream) }
        }
        return nil
    }

    /// Splits a sentinel path `cloud://<providerId>/<upstreamModelId>`.
    public static func parsePath(
        _ path: String,
        providers: [CloudProvider] = CloudProviderRegistry.all
    ) -> (provider: CloudProvider, providerModelId: String)? {
        guard path.hasPrefix(pathScheme) else { return nil }
        let rest = String(path.dropFirst(pathScheme.count))
        for provider in longestFirst(providers) where rest.hasPrefix(provider.id + "/") {
            let upstream = String(rest.dropFirst(provider.id.count + 1))
            if !upstream.isEmpty { return (provider, upstream) }
        }
        return nil
    }

    private static func longestFirst(_ providers: [CloudProvider]) -> [CloudProvider] {
        providers.sorted { $0.id.count > $1.id.count }
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
    /// Whether this model accepts image content parts. The provider's `/models`
    /// catalog does not publish modalities, so this is an explicit opt-in
    /// registration fact (the vision equivalent of the no-tools list). A model
    /// left off the vision list is treated as text-only: the composer disables
    /// attachment and a rejected image turn retries once without it.
    public let supportsVision: Bool
    /// What this model's context window is believed to be, or nil when nobody
    /// recorded one for THIS model. A registration fact, not provider truth:
    /// the catalog publishes no window, and a wrong guess would compact at the
    /// wrong time. Nil falls back to the global
    /// `dynamoe_cloud_context_window_tokens` default; only when that is unset
    /// too is auto-compaction off for the model.
    public let contextWindowTokens: Int?

    public init(
        providerId: String,
        providerModelId: String,
        displayName: String,
        supportsThinking: Bool,
        supportsTools: Bool = true,
        supportsVision: Bool = false,
        contextWindowTokens: Int? = nil
    ) {
        self.providerId = providerId
        self.providerModelId = providerModelId
        self.displayName = displayName
        self.supportsThinking = supportsThinking
        self.supportsTools = supportsTools
        self.supportsVision = supportsVision
        self.contextWindowTokens = contextWindowTokens
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
    /// list is empty — the end-to-end smoke-test seam for this branch. It is a
    /// single global value rather than a per-provider one, so a per-provider
    /// prune leaves it alone; clear it with `defaults delete`.
    public static let debugModelDefaultsKey = "dynamoe_cloud_debug_model"

    /// Upstream model ids for which this provider rejects a `tools` array
    /// (same `;`/`,`/newline separated format as the registry list). A model
    /// listed here runs in plain chat with a visible notice instead of
    /// failing mid-loop on an HTTP 400.
    public static func noToolsDefaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_no_tools"
    }

    /// Upstream model ids that accept image content parts. Opt-in per model:
    /// the catalog publishes no modality information, and guessing wrong costs
    /// a failed turn where the model answers as if no image was sent.
    public static func visionDefaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_vision"
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
        parsedModelIdList(noToolsDefaultsKey(for: provider.id), defaults: defaults)
    }

    /// Replaces the set of models that run without a `tools` array.
    public static func setModelIdsWithoutToolSupport(
        _ modelIds: Set<String>,
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) {
        let cleaned = modelIds
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .sorted()
        defaults.set(cleaned.joined(separator: "; "), forKey: noToolsDefaultsKey(for: provider.id))
    }

    /// Upstream model ids registered as accepting image parts.
    public static func modelIdsWithVisionSupport(
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) -> Set<String> {
        parsedModelIdList(visionDefaultsKey(for: provider.id), defaults: defaults)
    }

    /// Replaces the set of models marked vision-capable. Written by Settings when
    /// the user marks a model, and by the chat path when a provider rejects an
    /// image for a model we had marked capable (so a later turn does not fail
    /// the same way).
    public static func setModelIdsWithVisionSupport(
        _ modelIds: Set<String>,
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) {
        let cleaned = modelIds
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .sorted()
        defaults.set(cleaned.joined(separator: "; "), forKey: visionDefaultsKey(for: provider.id))
    }

    /// Marks or unmarks one model as vision-capable without disturbing the rest
    /// of the list.
    public static func setVisionSupport(
        _ supported: Bool,
        forModel providerModelId: String,
        provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) {
        var ids = modelIdsWithVisionSupport(for: provider, defaults: defaults)
        if supported {
            ids.insert(providerModelId)
        } else {
            ids.remove(providerModelId)
        }
        setModelIdsWithVisionSupport(ids, for: provider, defaults: defaults)
    }

    private static func parsedModelIdList(_ key: String, defaults: UserDefaults) -> Set<String> {
        let raw = defaults.string(forKey: key) ?? ""
        return Set(
            raw.split(whereSeparator: { $0 == ";" || $0 == "," || $0 == "\n" })
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        )
    }

    /// Drops every registration fact about the SERVER at the provider's base
    /// URL: the model list, the no-tools list and the vision marks. Used when
    /// the base URL changes, where the old ids and capability marks describe an
    /// endpoint that is no longer being talked to.
    ///
    /// Prices and context windows deliberately survive, because they describe
    /// the MODEL rather than the endpoint. They are typed by hand from the
    /// provider's own price sheet, neither has a Settings control to retype
    /// one, and re-pointing a provider (a proxy, a regional host, a corrected
    /// path) says nothing about what its models cost or how much context they
    /// hold. Deleting them silently also deleted the spend figure and the
    /// auto-compaction ceiling for every model on it.
    public static func clearEndpointState(
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) {
        defaults.removeObject(forKey: modelsDefaultsKey(for: provider.id))
        defaults.removeObject(forKey: noToolsDefaultsKey(for: provider.id))
        defaults.removeObject(forKey: visionDefaultsKey(for: provider.id))
    }

    /// Drops every per-provider registration fact: the model list, the
    /// no-tools list, the vision marks, the context windows and the prices.
    /// Used when the provider itself is removed, so a re-added identifier
    /// starts clean.
    public static func clearProviderState(
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) {
        clearEndpointState(for: provider, defaults: defaults)
        defaults.removeObject(forKey: CloudContextPolicy.defaultsKey(for: provider.id))
        defaults.removeObject(forKey: CloudPricingStore.defaultsKey(for: provider.id))
    }

    /// Unregisters one model without disturbing the rest of the list. A
    /// catalog can list models the endpoint does not serve on the wire
    /// dialect in use, so pruning one has to be possible without Clear-all.
    ///
    /// The price and window removals go through their own key-by-key helpers
    /// rather than through the readers: those readers drop entries they cannot
    /// decode, so rewriting their output deleted a neighbouring model's
    /// hand-typed value that merely had a typo.
    public static func removeRegisteredModel(
        _ providerModelId: String,
        for provider: CloudProvider,
        defaults: UserDefaults = .standard
    ) {
        setRegisteredModelIds(
            registeredModelIds(for: provider, defaults: defaults).filter { $0 != providerModelId },
            for: provider,
            defaults: defaults
        )
        var withoutTools = modelIdsWithoutToolSupport(for: provider, defaults: defaults)
        if withoutTools.remove(providerModelId) != nil {
            setModelIdsWithoutToolSupport(withoutTools, for: provider, defaults: defaults)
        }
        var withVision = modelIdsWithVisionSupport(for: provider, defaults: defaults)
        if withVision.remove(providerModelId) != nil {
            setModelIdsWithVisionSupport(withVision, for: provider, defaults: defaults)
        }
        CloudContextPolicy.removeOverride(
            model: providerModelId,
            providerId: provider.id,
            defaults: defaults
        )
        CloudPricingStore.removePrice(
            forModel: providerModelId,
            providerId: provider.id,
            defaults: defaults
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
        let withVision = modelIdsWithVisionSupport(for: provider, defaults: defaults)
        let windows = CloudContextPolicy.overrides(for: provider.id, defaults: defaults)
        return ids.map { id in
            CloudModelSpec(
                providerId: provider.id,
                providerModelId: id,
                displayName: defaultDisplayName(provider: provider, providerModelId: id),
                supportsThinking: id.lowercased().contains("thinking"),
                supportsTools: !withoutTools.contains(id),
                supportsVision: withVision.contains(id),
                contextWindowTokens: windows[id]
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
            supportsThinking: spec.supportsThinking,
            supportsVision: spec.supportsVision
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
            supportsThinking: providerModelId.lowercased().contains("thinking"),
            supportsVision: modelIdsWithVisionSupport(for: provider).contains(providerModelId)
        ))
    }

    /// All registered entries across all configured providers. Appended,
    /// never prepended: `LocalModelManager.getDefaultOrFirstModel()` requires
    /// `discoveredModels.first` to be a weights model.
    ///
    /// Entries are only offered once the provider has a stored credential:
    /// without one every turn is guaranteed to fail, so listing those models
    /// would put unpickable wishes in the composer.
    public static func makeDiscoveredModels(
        defaults: UserDefaults = .standard,
        providers: [CloudProvider] = CloudProviderRegistry.all,
        hasCredential: (String) -> Bool = { CloudSecretStore.shared.exists(providerId: $0) }
    ) -> [DiscoveredModel] {
        providers.flatMap { provider in
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
