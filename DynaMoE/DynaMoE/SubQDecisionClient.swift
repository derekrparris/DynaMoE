//
//  SubQDecisionClient.swift
//  DynaMoE
//
//  Phase 8 — Skills.
//
//  The first skill that needs a real capability: SubQ's decisions-only model
//  (`subq-snap-preview`). It is NOT a chat model — it answers typed questions
//  about a `state` over `POST https://api.subq.ai/v1/systemone` — so it cannot
//  be registered as a Cloud Provider model. `subq_decide` is the native tool
//  that stands in for the Python helper a shell-based agent would run, and the
//  key never enters a prompt or a log line. `ControlledProcessRunner` strips
//  `SUBQ_API_KEY` from every spawned tool process, because the environment
//  fallback below is live exactly when `shell_run` could otherwise print it.
//
//  The wire contract is documented in the companion skill. This file owns the
//  request shape, the three answer types, the error mapping, and the key
//  resolution order (an existing SubQ Cloud Provider first, then the Skills
//  Keychain entry, then the process environment).
//

import Foundation
import Security

// MARK: - Errors

public nonisolated enum SubQDecisionError: LocalizedError, Equatable {
    case missingKey
    /// A credential exists but this build may not read it (a dismissed
    /// Keychain dialog, a locked keychain, a legacy-only item). Reporting that
    /// as "no key configured" sent the operator to add a second one.
    case keyUnreadable(status: OSStatus)
    case badRequest(String)
    case unauthorized(String)
    case paymentRequired(String)
    case forbidden(String)
    case invalidRequest(String)
    case rateLimited(String)
    case unavailable(String)
    case http(status: Int, detail: String)
    case transport(String)
    case decoding(String)

    public var errorDescription: String? {
        switch self {
        case .missingKey:
            return "No SubQ API key is configured. Add one in Settings → Agent & Tools → Skills (SubQ Decisions), or add a Cloud Provider with base URL https://api.subq.ai/v1 and a key. Create a key at console.subq.ai."
        case .keyUnreadable(let status):
            return "A SubQ API key is stored, but this build could not read it (Keychain error \(status)). If macOS showed an access prompt, allow it and retry; otherwise import the credential from Settings → Stored Credentials."
        case .badRequest(let detail):
            return "subq_decide could not build a request: \(detail)"
        case .unauthorized(let detail):
            return "SubQ rejected the API key (401). Check the key in Settings → Agent & Tools → Skills, or the SubQ Cloud Provider's key.\(detail.isEmpty ? "" : " \(detail)")"
        case .paymentRequired(let detail):
            return "SubQ reports insufficient credits (402). Top up at console.subq.ai.\(detail.isEmpty ? "" : " \(detail)")"
        case .forbidden(let detail):
            return "This SubQ key is not allowed to use the model (403).\(detail.isEmpty ? "" : " \(detail)")"
        case .invalidRequest(let detail):
            return "SubQ rejected the request (422): \(detail)"
        case .rateLimited(let detail):
            return "SubQ rate-limited the request (429). Retry shortly.\(detail.isEmpty ? "" : " \(detail)")"
        case .unavailable(let detail):
            return "SubQ is temporarily unavailable (503). Retry later.\(detail.isEmpty ? "" : " \(detail)")"
        case .http(let status, let detail):
            return "SubQ returned HTTP \(status).\(detail.isEmpty ? "" : " \(detail)")"
        case .transport(let detail):
            return "Could not reach SubQ: \(detail)"
        case .decoding(let detail):
            return "SubQ answered in a shape this build could not read: \(detail)"
        }
    }

    /// Maps a status code to the actionable message the skill promises.
    public static func forStatus(_ status: Int, detail: String) -> SubQDecisionError {
        switch status {
        case 400: return .badRequest(detail)
        case 401: return .unauthorized(detail)
        case 402: return .paymentRequired(detail)
        case 403: return .forbidden(detail)
        case 422: return .invalidRequest(detail)
        case 429: return .rateLimited(detail)
        case 500...599: return .unavailable(detail)
        default: return .http(status: status, detail: detail)
        }
    }
}

// MARK: - Settings

/// Defaults shared by the tool and the Settings card. Endpoint and model are
/// overridable so a future snapshot model does not need a new build.
public nonisolated enum SubQDecisionSettings {
    public static let endpointKey = "dynamoe_subq_decide_endpoint"
    public static let modelKey = "dynamoe_subq_decide_model"
    public static let defaultEndpoint = "https://api.subq.ai/v1/systemone"
    public static let defaultModel = "subq-snap-preview"

    public static func endpoint(defaults: UserDefaults = .standard) -> URL {
        let raw = (defaults.string(forKey: endpointKey) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = URL(string: defaultEndpoint)!
        guard !raw.isEmpty, let url = URL(string: raw),
              let scheme = url.scheme?.lowercased(), let host = url.host, !host.isEmpty,
              scheme == "https" || (scheme == "http" && isLoopback(host)) else {
            return fallback
        }
        return url
    }

    /// The API key travels as a Bearer token, so plaintext HTTP is allowed only
    /// for a loopback endpoint (a local proxy), matching the MCP client's
    /// posture. Any other `http://` value is refused and the official endpoint
    /// is used instead; the Settings card explains why.
    public static func isLoopback(_ host: String) -> Bool {
        let lowered = host.lowercased()
        return lowered == "localhost" || lowered == "127.0.0.1" || lowered == "::1"
            || lowered == "[::1]"
    }

    /// Canonical origin: `scheme://host:port`, with the scheme's default port
    /// made explicit. A HOST is not an origin: `https://api.subq.ai` and
    /// `https://api.subq.ai:8443` are different destinations for a Bearer
    /// token, and comparing hosts treated the second as SubQ's own.
    public static func origin(_ url: URL?) -> String? {
        guard let url,
              let scheme = url.scheme?.lowercased(), !scheme.isEmpty,
              let host = url.host?.lowercased(), !host.isEmpty else { return nil }
        return "\(scheme)://\(host):\(effectivePort(url))"
    }

    /// The port a URL actually reaches, with the scheme's default made
    /// explicit. `url.port` is nil for `https://host`, which is 443.
    public static func effectivePort(_ url: URL) -> Int {
        let scheme = url.scheme?.lowercased()
        return url.port ?? (scheme == "https" ? 443 : (scheme == "http" ? 80 : -1))
    }

    /// The origin this build treats as SubQ's own. Everything else is a custom
    /// destination that needs explicit approval.
    public static var officialOrigin: String? {
        origin(URL(string: defaultEndpoint))
    }

    /// Non-nil when the configured endpoint is not SubQ's own origin. The
    /// request carries the API key as a Bearer token, so a custom URL sends
    /// that secret wherever it points; Settings shows this before a test does
    /// and the approval card names it before a call runs.
    public static func customEndpointWarning(defaults: UserDefaults = .standard) -> String? {
        let configured = endpoint(defaults: defaults)
        guard origin(configured) != officialOrigin else { return nil }
        let destination = origin(configured) ?? configured.absoluteString
        return "This endpoint is not api.subq.ai. The API key is sent as a Bearer token to \(destination); use a custom endpoint only with a server you trust."
    }

    /// Non-nil when the raw endpoint value was rejected rather than used, so
    /// Settings can show the reason instead of silently substituting the
    /// official one.
    public static func endpointRejection(defaults: UserDefaults = .standard) -> String? {
        let raw = (defaults.string(forKey: endpointKey) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return nil }
        guard let url = URL(string: raw), let scheme = url.scheme?.lowercased(),
              let host = url.host, !host.isEmpty else {
            return "'\(raw)' is not an http(s) URL with a host; the official endpoint is used instead."
        }
        if scheme == "https" { return nil }
        if scheme == "http" && isLoopback(host) { return nil }
        return "'\(raw)' would send the API key over plaintext to \(host). Use https://, or a loopback host for a local proxy; the official endpoint is used instead."
    }

    public static func model(defaults: UserDefaults = .standard) -> String {
        let raw = (defaults.string(forKey: modelKey) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return raw.isEmpty ? defaultModel : raw
    }
}

// MARK: - Key Resolution

public nonisolated enum SubQDecisionKeySource {
    public static let skillName = "subq-decisions"
    public static let apiKeyField = "apikey"
    /// A provider registered against this host is assumed to carry a key that
    /// works for the Decisions API too (SubQ issues one key for both).
    public static let providerHost = "api.subq.ai"
    public static let environmentName = "SUBQ_API_KEY"

    public static var account: String {
        SkillsSecretStore.account(skill: skillName, field: apiKeyField)
    }

    public static func matchingProviders(defaults: UserDefaults = .standard) -> [CloudProvider] {
        CloudProviderCatalog.all(defaults: defaults).filter {
            $0.openAIBaseURL.host?.lowercased() == providerHost
        }
    }

    /// What a call can do with the credentials this build can see.
    public enum Resolution: Sendable, Equatable {
        case found(key: String, label: String)
        case missing
        /// A credential is stored but the read was refused or the keychain was
        /// unavailable. A turn must say that rather than "add a key".
        case unreadable(OSStatus)
    }

    /// The key and where it came from, for the Settings readout and the tool's
    /// error text. Ordered so the user's existing Cloud Provider setup wins.
    ///
    /// Distinguishes "not stored" from "could not read it": `get` folds both
    /// into nil, which reported a denied Keychain read as a missing key.
    public static func resolveDetail(defaults: UserDefaults = .standard) -> Resolution {
        #if DEBUG
        if let override = resolutionOverrideForTesting {
            guard let resolved = override() else { return .missing }
            return .found(key: resolved.key, label: resolved.label)
        }
        #endif
        var unreadable: OSStatus?
        for provider in matchingProviders(defaults: defaults) {
            switch CloudSecretStore.shared.read(providerId: provider.id) {
            case .found(let key) where !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
                return .found(key: key, label: "Cloud Provider '\(provider.displayName)'")
            case .accessDenied(let status), .unavailable(let status):
                unreadable = unreadable ?? status
            default:
                break
            }
        }
        switch SkillsSecretStore.shared.read(account: account) {
        case .found(let key) where !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
            return .found(key: key, label: "Skills Keychain")
        case .accessDenied(let status), .unavailable(let status):
            unreadable = unreadable ?? status
        default:
            break
        }
        if let key = ProcessInfo.processInfo.environment[environmentName],
           !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .found(key: key, label: "\(environmentName) environment variable")
        }
        if let unreadable { return .unreadable(unreadable) }
        return .missing
    }

    /// The key alone, for callers that only need a value. Prefer
    /// `resolveDetail` where the difference between missing and unreadable
    /// changes what the user is told.
    public static func resolve(defaults: UserDefaults = .standard) -> (key: String, label: String)? {
        guard case .found(let key, let label) = resolveDetail(defaults: defaults) else { return nil }
        return (key, label)
    }

    /// Attributes only where possible, so this is safe on the prompt-seeding
    /// path: no query here can raise a Keychain dialog.
    ///
    /// It answers the SAME question `resolve` does: a credential the build
    /// actually reads. `exists` covers both keychains, so a legacy-only item
    /// would report "Ready" while every call failed with a missing key until
    /// the Stored Credentials import ran.
    public static func hasAnyKey(defaults: UserDefaults = .standard) -> Bool {
        #if DEBUG
        if let override = hasAnyKeyOverrideForTesting { return override }
        #endif
        for provider in matchingProviders(defaults: defaults) where CloudSecretStore.shared.hasCurrentCredential(providerId: provider.id) {
            return true
        }
        if SkillsSecretStore.shared.hasCurrentCredential(account: account) { return true }
        if let key = ProcessInfo.processInfo.environment[environmentName], !key.isEmpty { return true }
        return false
    }

    /// Test seams for `resolve` and `hasAnyKey`: without them a case that
    /// exercises the tool would read (and depend on) the developer's real
    /// Keychain or shell environment.
    #if DEBUG
    nonisolated(unsafe) public static var resolutionOverrideForTesting: (() -> (key: String, label: String)?)?
    nonisolated(unsafe) public static var hasAnyKeyOverrideForTesting: Bool?
    #endif

    /// True when a key exists only in the keychain an earlier version wrote.
    /// The Settings card uses it to point at the Stored Credentials import
    /// rather than saying "No Key" about a key that is sitting right there.
    public static func legacyKeyPending(defaults: UserDefaults = .standard) -> Bool {
        for provider in matchingProviders(defaults: defaults)
        where CloudSecretStore.shared.exists(providerId: provider.id)
            && !CloudSecretStore.shared.hasCurrentCredential(providerId: provider.id) {
            return true
        }
        return SkillsSecretStore.shared.exists(account: account)
            && !SkillsSecretStore.shared.hasCurrentCredential(account: account)
    }

    /// What the Settings card can say about the credential WITHOUT reading it.
    ///
    /// `describeWithoutPrompt` folded "denied" into "no key", which is the one
    /// screen a user consults before deleting a good credential: a dismissed
    /// Keychain dialog read as "No Key" and invited them to delete the item
    /// that was actually there. The distinction is attribute-only, so it still
    /// raises no dialog.
    public enum KeyStatus: Sendable, Equatable {
        /// A key resolves from a provider, this card, or the environment.
        case ready(source: String)
        /// Nothing is stored anywhere this build can see.
        case missing
        /// An item exists but the Keychain refused the attribute read (a
        /// dismissed dialog, a locked keychain). The advice is "authorize",
        /// not "add a key".
        case unreadable(OSStatus)
        /// A pre-import credential is in the legacy keychain only.
        case legacyPending
    }

    public static func keyStatus(defaults: UserDefaults = .standard) -> KeyStatus {
        #if DEBUG
        if hasAnyKeyOverrideForTesting == true { return .ready(source: "a test key") }
        if hasAnyKeyOverrideForTesting == false { return .missing }
        #endif
        let providers = matchingProviders(defaults: defaults)
        var unreadable: OSStatus?
        // A legacy-only item is NOT `.ready`: `read()` consults only the
        // resolved backend, so the card would promise a key every call then
        // fails to find. It falls through to `.legacyPending` below, which is
        // the answer that points at the Stored Credentials import.
        var legacyPending = false
        for provider in providers {
            switch CloudSecretStore.shared.credentialStatus(providerId: provider.id) {
            case .found:
                return .ready(source: "Cloud Provider '\(provider.displayName)'")
            case .legacyOnly:
                legacyPending = true
            case .unreadable(let status):
                unreadable = unreadable ?? status
            case .missing:
                break
            }
        }
        switch SkillsSecretStore.shared.credentialStatus(account: account) {
        case .found:
            return .ready(source: "this card")
        case .legacyOnly:
            legacyPending = true
        case .unreadable(let status):
            unreadable = unreadable ?? status
        case .missing:
            break
        }
        if let key = ProcessInfo.processInfo.environment[environmentName], !key.isEmpty {
            return .ready(source: environmentName)
        }
        if let unreadable { return .unreadable(unreadable) }
        // `credentialStatus` already established this; the second probe is
        // only for the case where a provider list was empty.
        if legacyPending || legacyKeyPending(defaults: defaults) { return .legacyPending }
        return .missing
    }

    /// Same answer as `resolve`, without reading a secret. The Settings card
    /// renders this on every body evaluation, and a read is the one Keychain
    /// call that can prompt for access on the legacy backend.
    public static func describeWithoutPrompt(defaults: UserDefaults = .standard) -> String {
        switch keyStatus(defaults: defaults) {
        case .ready(let source):
            return "Using the key from \(source)."
        case .unreadable(let status):
            return "A SubQ key is stored, but the Keychain refused the lookup (\(status)). If macOS showed an access prompt, allow it and reopen this card; otherwise import the credential from Settings → Stored Credentials. Do not delete the key first — it is probably still there."
        case .legacyPending:
            return "A SubQ key is stored in the legacy keychain. Import it from Settings → Stored Credentials; this build cannot read it until then."
        case .missing:
            if let provider = matchingProviders(defaults: defaults).first {
                return "Cloud Provider '\(provider.displayName)' exists but has no stored key."
            }
            return "No SubQ API key. Add one here, or add a Cloud Provider for api.subq.ai."
        }
    }
}

// MARK: - Wire Types

public nonisolated struct SubQDecisionUsage: Decodable, Equatable, Sendable {
    public let inputTokens: Int?
    public let outputTokens: Int?

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }
}

public nonisolated enum SubQDecisionAnswer: Equatable, Sendable {
    case noul(probability: Double)
    case choice(choice: String, probabilities: [String: Double], confidence: Double?)
    case score(score: Double, legend: [String: String], probabilities: [String: Double], confidence: Double?)
    case other(type: String)

    /// One-line, model-facing reading. Numbers only: the thresholds that turn
    /// them into a decision belong to the skill, not to the client.
    public var readableDescription: String {
        switch self {
        case .noul(let probability):
            return "noul \(Self.number(probability)) (probability of yes)"
        case .choice(let choice, let probabilities, let confidence):
            var parts = ["choice \"\(choice)\""]
            if let confidence { parts.append("confidence \(Self.number(confidence))") }
            let top = probabilities.sorted { $0.value > $1.value }.prefix(3)
                .map { "\($0.key) \(Self.number($0.value))" }
            if !top.isEmpty { parts.append(top.joined(separator: ", ")) }
            return parts.joined(separator: " · ")
        case .score(let score, let legend, let probabilities, let confidence):
            var parts = ["score \(Self.number(score))"]
            // `Int(exactly:)` rather than `Int(...)`: a hostile endpoint's
            // `1e300` trapped the app here, before any validation ran.
            if let level = Int(exactly: score.rounded()), let name = legend[String(level)] {
                parts.append("= \(name)")
            }
            if let confidence { parts.append("confidence \(Self.number(confidence))") }
            let top = probabilities.sorted { $0.value > $1.value }.prefix(3)
                .map { entry in
                    let name = legend[entry.key].map { " \($0)" } ?? ""
                    return "\(entry.key)\(name) \(Self.number(entry.value))"
                }
            if !top.isEmpty { parts.append(top.joined(separator: ", ")) }
            return parts.joined(separator: " · ")
        case .other(let type):
            return "\(type.isEmpty ? "unrecognised" : type) answer (unknown shape)"
        }
    }

    static func number(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

extension SubQDecisionAnswer: Decodable {
    private enum CodingKeys: String, CodingKey {
        case type, noul, choice, probabilities, confidence, score, legend
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = (try? container.decode(String.self, forKey: .type)) ?? ""
        switch type {
        case "noul":
            // A gate whose whole output is one number must not invent it: a
            // missing/renamed/string field used to decode as 0.0, i.e. the
            // strongest possible "no", and report `status: success`. Schema
            // drift throws so the tool refuses instead of answering a
            // distribution the API never produced.
            let probability = try Self.unit(container, .noul)
            self = .noul(probability: probability)
        case "choice":
            // A choice with no chosen label used to render as `choice "none"`
            // and report success. That is a missing answer, not the option
            // named "none", so schema drift throws like the noul gate above.
            let choice = try container.decode(String.self, forKey: .choice)
            self = .choice(
                choice: choice,
                probabilities: try Self.probabilities(container),
                confidence: try Self.confidence(container)
            )
        case "score":
            // Same rule: a missing score printed `score 0.00`, the strongest
            // possible low answer, as a success.
            let score = try Self.finite(container, .score)
            self = .score(
                score: score,
                legend: (try? container.decode([String: String].self, forKey: .legend)) ?? [:],
                probabilities: try Self.probabilities(container),
                confidence: try Self.confidence(container)
            )
        default:
            self = .other(type: type)
        }
    }

    /// A number this client can print. `1e300` used to reach
    /// `Int(score.rounded())` and trap the app; NaN would compare false
    /// against every threshold without saying so.
    private static func finite(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys
    ) throws -> Double {
        let value = try container.decode(Double.self, forKey: key)
        guard value.isFinite else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "\(key.stringValue) is not a finite number"
            )
        }
        return value
    }

    private static func unit(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys
    ) throws -> Double {
        let value = try finite(container, key)
        guard (-1e-9)...(1.0 + 1e-9) ~= value else {
            throw DecodingError.dataCorruptedError(
                forKey: key,
                in: container,
                debugDescription: "\(key.stringValue) is \(value), outside 0...1"
            )
        }
        return min(max(value, 0), 1)
    }

    private static func probabilities(
        _ container: KeyedDecodingContainer<CodingKeys>
    ) throws -> [String: Double] {
        guard let map = try container.decodeIfPresent([String: Double].self, forKey: .probabilities) else {
            return [:]
        }
        for (name, value) in map {
            guard value.isFinite, (-1e-9)...(1.0 + 1e-9) ~= value else {
                throw DecodingError.dataCorruptedError(
                    forKey: .probabilities,
                    in: container,
                    debugDescription: "probability for '\(name)' is \(value), outside 0...1"
                )
            }
        }
        return map.mapValues { min(max($0, 0), 1) }
    }

    private static func confidence(
        _ container: KeyedDecodingContainer<CodingKeys>
    ) throws -> Double? {
        guard let value = try container.decodeIfPresent(Double.self, forKey: .confidence) else { return nil }
        guard value.isFinite, (-1e-9)...(1.0 + 1e-9) ~= value else {
            throw DecodingError.dataCorruptedError(
                forKey: .confidence,
                in: container,
                debugDescription: "confidence is \(value), outside 0...1"
            )
        }
        return min(max(value, 0), 1)
    }
}

public nonisolated struct SubQDecisionResponse: Decodable, Equatable, Sendable {
    public let model: String?
    public let answers: [String: SubQDecisionAnswer]
    public let usage: SubQDecisionUsage?

    private enum CodingKeys: String, CodingKey {
        case model, answers, usage
    }

    /// An empty `answers` map used to render "No answers were returned" and
    /// report `status: success`, i.e. a missing decision presented as one.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        answers = try container.decode([String: SubQDecisionAnswer].self, forKey: .answers)
        usage = try container.decodeIfPresent(SubQDecisionUsage.self, forKey: .usage)
        guard !answers.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .answers,
                in: container,
                debugDescription: "the response carried no answers"
            )
        }
    }

    public var readableSummary: String {
        guard !answers.isEmpty else { return "No answers were returned." }
        // `compactMap` rather than `answers[name]!`: the subscript is safe today
        // only because the keys come from the same dictionary, and this file
        // otherwise treats every input as hostile.
        return answers.keys.sorted().compactMap { name in
            answers[name].map { "- \(name): \($0.readableDescription)" }
        }.joined(separator: "\n")
    }

    public var usageLine: String {
        let input = usage?.inputTokens ?? 0
        let output = usage?.outputTokens ?? 0
        return "input tokens \(input) · output tokens \(output) (output is free)"
    }
}

public nonisolated struct SubQImage: Sendable {
    public let mediaType: String
    public let base64: String

    public init(mediaType: String, base64: String) {
        self.mediaType = mediaType
        self.base64 = base64
    }

    public static let maximumBytes = 4 * 1024 * 1024
    public static let maximumImages = 4

    /// Loads one image from disk, refusing anything the API would reject:
    /// over 4 MiB decoded, a format it does not accept, or anything that is
    /// not a regular file.
    ///
    /// `workingDirectory` confines the read. This is the one native tool that
    /// both opens a local file and POSTs it to a third party, so a path outside
    /// the turn's workspace is refused rather than uploaded — `web_fetch` has
    /// no equivalent capability.
    public static let supportedMediaTypes: Set<String> = ["image/png", "image/jpeg", "image/webp"]

    public static func load(
        path: String,
        workingDirectory: URL? = nil,
        mediaTypeOverride: String? = nil
    ) throws -> SubQImage {
        let expanded = (path as NSString).expandingTildeInPath
        let candidate: URL
        if expanded.hasPrefix("/") {
            candidate = URL(fileURLWithPath: expanded)
        } else if let workingDirectory {
            candidate = workingDirectory.appendingPathComponent(expanded)
        } else {
            candidate = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(expanded)
        }
        let url = candidate.resolvingSymlinksInPath().standardizedFileURL
        if let workingDirectory {
            let base = workingDirectory.resolvingSymlinksInPath().standardizedFileURL.path
            guard url.path == base || url.path.hasPrefix(base + "/") else {
                throw SubQDecisionError.badRequest(
                    "image \(path) is outside the working directory \(base). Pass a path inside the folder this turn is working in."
                )
            }
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SubQDecisionError.badRequest("image not found at \(url.path)")
        }
        let data = try readBytes(at: url, maximumBytes: maximumBytes)
        let mediaType: String
        // An explicit `media_type` is honoured rather than silently dropped:
        // the schema advertises it, and ignoring it uploaded a different type
        // than the caller declared. It is still validated against the API's
        // accepted set.
        if let requested = mediaTypeOverride?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !requested.isEmpty {
            guard supportedMediaTypes.contains(requested) else {
                throw SubQDecisionError.badRequest(
                    "image \(url.lastPathComponent) declares media_type '\(requested)'; use image/png, image/jpeg or image/webp."
                )
            }
            mediaType = requested
        } else {
            switch url.pathExtension.lowercased() {
            case "png": mediaType = "image/png"
            case "jpg", "jpeg": mediaType = "image/jpeg"
            case "webp": mediaType = "image/webp"
            default:
                throw SubQDecisionError.badRequest(
                    "image \(url.lastPathComponent) is not a supported format; use PNG, JPEG or WebP."
                )
            }
        }
        return SubQImage(mediaType: mediaType, base64: data.base64EncodedString())
    }

    /// The same discipline as `SkillDiscovery.readTextCapped`: classify the
    /// descriptor that is actually read, not a path checked a moment earlier.
    /// `O_NONBLOCK` means a FIFO renamed into place cannot stall the caller.
    private static func readBytes(at url: URL, maximumBytes: Int) throws -> Data {
        let descriptor = open(url.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw SubQDecisionError.badRequest(
                "image could not be opened at \(url.path): \(String(cString: strerror(errno)))"
            )
        }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else {
            throw SubQDecisionError.badRequest(
                "image could not be inspected at \(url.path): \(String(cString: strerror(errno)))"
            )
        }
        guard (info.st_mode & S_IFMT) == S_IFREG else {
            throw SubQDecisionError.badRequest("image is not a regular file at \(url.path)")
        }
        guard info.st_size <= maximumBytes else {
            throw SubQDecisionError.badRequest(
                "image \(url.lastPathComponent) is \(info.st_size) bytes; the API accepts at most \(maximumBytes) decoded bytes per image."
            )
        }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = buffer.withUnsafeMutableBytes { raw in
                read(descriptor, raw.baseAddress, raw.count)
            }
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                throw SubQDecisionError.badRequest(
                    "image could not be read at \(url.path): \(String(cString: strerror(errno)))"
                )
            }
            data.append(contentsOf: buffer[0..<count])
            if data.count > maximumBytes {
                throw SubQDecisionError.badRequest(
                    "image \(url.lastPathComponent) grew past \(maximumBytes) bytes; the API accepts at most \(maximumBytes) decoded bytes per image."
                )
            }
        }
        return data
    }
}

// MARK: - Redirects

/// Refuses a redirect to a different host. The request carries the API key as a
/// Bearer token, so following a 3xx to somebody else's host would hand them the
/// secret. A refused redirect is reported as the 3xx status it was.
nonisolated final class SubQRedirectGuard: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(Self.isAllowedRedirect(from: response.url, to: request.url) ? request : nil)
    }

    static func isAllowedRedirect(from origin: URL?, to destination: URL?) -> Bool {
        guard let origin, let destination,
              let originHost = origin.host?.lowercased(), !originHost.isEmpty,
              let destinationHost = destination.host?.lowercased(), !destinationHost.isEmpty else {
            return false
        }
        guard originHost == destinationHost else { return false }
        // Host alone was not the origin: the same host on another PORT is a
        // different service, and the Bearer token would go with it.
        guard SubQDecisionSettings.effectivePort(origin) == SubQDecisionSettings.effectivePort(destination) else {
            return false
        }
        // A same-host redirect is still a downgrade when it drops TLS: the
        // Bearer token would leave in clear on the second hop. Refuse it.
        let originScheme = origin.scheme?.lowercased()
        let destinationScheme = destination.scheme?.lowercased()
        if originScheme == "https", destinationScheme != "https" { return false }
        return true
    }
}

// MARK: - Client

public nonisolated struct SubQDecisionClient: Sendable {
    public static let maximumQuestions = 3
    public static let maximumStateCharacters = 200_000
    /// A decision response is a handful of numbers. A server that streams
    /// megabytes at us is a misconfigured endpoint (or worse), not an answer.
    public static let maximumResponseBytes = 2 * 1024 * 1024
    public static let defaultTimeout: TimeInterval = 30

    public let endpoint: URL
    public let apiKey: String
    public let model: String
    public let timeout: TimeInterval
    /// Injectable so an HTTP-path test can answer without the network. The
    /// default session has a resource ceiling and a same-host redirect guard.
    public let session: URLSession

    public init(
        endpoint: URL = SubQDecisionSettings.endpoint(),
        apiKey: String,
        model: String = SubQDecisionSettings.model(),
        timeout: TimeInterval = SubQDecisionClient.defaultTimeout,
        session: URLSession = SubQDecisionClient.defaultSession
    ) {
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.model = model
        self.timeout = timeout
        self.session = session
    }

    /// `timeoutIntervalForRequest` is per request (see `send`), but a resource
    /// ceiling cannot be, so it is set here. Three default timeouts lets the
    /// retry below finish inside one call.
    public static let defaultSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = SubQDecisionClient.defaultTimeout
        configuration.timeoutIntervalForResource = SubQDecisionClient.defaultTimeout * 3
        configuration.httpMaximumConnectionsPerHost = 2
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        return URLSession(
            configuration: configuration,
            delegate: SubQRedirectGuard(),
            delegateQueue: nil
        )
    }()

    private struct RequestBody: Encodable {
        let model: String
        let state: AnyCodable
        let questions: [String: AnyCodable]
        let images: [ImagePayload]?
    }

    private struct ImagePayload: Encodable {
        let mediaType: String
        let base64: String
    }

    private struct ErrorEnvelope: Decodable {
        struct Detail: Decodable {
            let errorType: String?
            let message: String?
            let detail: String?

            private enum CodingKeys: String, CodingKey {
                case errorType = "error_type"
                case message
                case detail
            }
        }

        let detail: Detail?
        let message: String?

        /// Best-effort human text from either documented envelope shape.
        var summary: String {
            if let detail {
                let parts = [detail.errorType, detail.message ?? detail.detail].compactMap { $0 }
                if !parts.isEmpty { return parts.joined(separator: ": ") }
            }
            return message ?? ""
        }
    }

    public func decide(
        state: AnyCodable,
        questions: [String: AnyCodable],
        images: [SubQImage] = []
    ) async throws -> SubQDecisionResponse {
        guard !questions.isEmpty else {
            throw SubQDecisionError.badRequest("at least one question is required.")
        }
        guard questions.count <= Self.maximumQuestions else {
            throw SubQDecisionError.badRequest(
                "\(questions.count) questions were sent; the API accepts at most \(Self.maximumQuestions) per call."
            )
        }
        guard images.count <= SubQImage.maximumImages else {
            throw SubQDecisionError.badRequest(
                "\(images.count) images were sent; the API accepts at most \(SubQImage.maximumImages)."
            )
        }

        let body = RequestBody(
            model: model,
            state: state,
            questions: questions,
            images: images.isEmpty ? nil : images.map { ImagePayload(mediaType: $0.mediaType, base64: $0.base64) }
        )
        let payload: Data
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            payload = try encoder.encode(body)
        } catch {
            throw SubQDecisionError.badRequest("the request could not be encoded as JSON: \(error.localizedDescription)")
        }

        var lastError: SubQDecisionError?
        for attempt in 0..<2 {
            do {
                return try await send(payload: payload)
            } catch let error as SubQDecisionError {
                lastError = error
                switch error {
                case .rateLimited, .unavailable:
                    if attempt == 0 {
                        // `try?` swallowed cancellation and slept through it, so
                        // the retry ran after the turn had been cancelled.
                        try await Task.sleep(nanoseconds: 1_000_000_000)
                        continue
                    }
                    throw error
                default:
                    throw error
                }
            }
        }
        throw lastError ?? SubQDecisionError.transport("no response")
    }

    /// What one error body is allowed to contribute to a message.
    ///
    /// The body is written by a remote server and reaches the transcript
    /// verbatim: it is persisted in the session store and re-sent in every
    /// later prompt. A proxy that echoes the `Authorization` header back in a
    /// 401, or a gateway that dumps a page of HTML, therefore used to write the
    /// operator's key to disk. Errors carry a capped, redacted, single-line
    /// excerpt instead, and the raw body only reaches `DiagLog` redacted.
    public static let maximumErrorDetailCharacters = 240

    /// Redacts the credential, collapses whitespace/newlines (so one body
    /// cannot forge extra `key: value` result lines) and caps the length.
    static func redactedErrorDetail(_ raw: String, apiKey: String) -> String {
        var text = raw
        if !apiKey.isEmpty {
            for secret in [apiKey, "Bearer \(apiKey)"] where !secret.isEmpty {
                text = text.replacingOccurrences(of: secret, with: "[redacted]")
            }
        }
        text = text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if text.count > maximumErrorDetailCharacters {
            text = String(text.prefix(maximumErrorDetailCharacters)) + "… (truncated)"
        }
        return text
    }

    private func send(payload: Data) async throws -> SubQDecisionResponse {
        guard let scheme = endpoint.scheme?.lowercased(),
              endpoint.host?.isEmpty == false else {
            throw SubQDecisionError.badRequest(
                "the configured endpoint \(endpoint.absoluteString) is not an http(s) URL with a host."
            )
        }
        guard scheme == "https" || (scheme == "http" && SubQDecisionSettings.isLoopback(endpoint.host ?? "")) else {
            throw SubQDecisionError.badRequest(
                "the configured endpoint \(endpoint.absoluteString) would send the API key over plaintext. Use https://, or a loopback host for a local proxy."
            )
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = payload

        // Streamed rather than buffered: `session.data(for:)` materialises the
        // whole body before the ceiling is consulted, so a hostile endpoint
        // could exhaust memory first. The cap is now enforced as bytes arrive.
        let data: Data
        let response: URLResponse
        do {
            let (bytes, urlResponse) = try await session.bytes(for: request)
            var collected = Data()
            collected.reserveCapacity(16 * 1024)
            for try await byte in bytes {
                collected.append(byte)
                if collected.count > Self.maximumResponseBytes {
                    throw SubQDecisionError.decoding(
                        "the response exceeded \(Self.maximumResponseBytes) bytes; this client reads at most that."
                    )
                }
            }
            data = collected
            response = urlResponse
        } catch is CancellationError {
            // A cancelled turn is not a transport failure; reporting it as one
            // made an interrupted generation look like a network problem.
            throw CancellationError()
        } catch let error as SubQDecisionError {
            throw error
        } catch {
            throw SubQDecisionError.transport(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw SubQDecisionError.transport("the response was not HTTP")
        }
        guard (200...299).contains(http.statusCode) else {
            let summary = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data))?.summary ?? ""
            // Read a bounded prefix BEFORE decoding as text: `data` can be the
            // whole 2 MiB ceiling, and an error body is not worth that.
            let fallback = String(decoding: data.prefix(2048), as: UTF8.self)
            let detail = Self.redactedErrorDetail(
                summary.isEmpty ? fallback : summary,
                apiKey: apiKey
            )
            // The structure (status → actionable case) is what the skill
            // promises; the body only fills in the detail, and only after it
            // has been made safe to persist.
            DiagLog.warning(
                "subq",
                "HTTP \(http.statusCode) from \(endpoint.host ?? "endpoint"): \(detail)"
            )
            throw SubQDecisionError.forStatus(http.statusCode, detail: detail)
        }
        do {
            return try JSONDecoder().decode(SubQDecisionResponse.self, from: data)
        } catch {
            throw SubQDecisionError.decoding(error.localizedDescription)
        }
    }
}

// MARK: - Tool

public final class SubQDecideTool: AgentTool {
    public let definition = ToolDefinition(
        name: "subq_decide",
        description: "Asks SubQ's decisions-only model (subq-snap-preview) one to three typed questions about a state and returns probabilities plus confidence. Use it for yes/no gates, routing or classification, triage and ordered scoring. Do NOT use it for open-ended analysis, drafting, summarizing, or extracting fields — use a chat model for those. Load the subq-decisions skill first for framing rules and thresholds. The state and questions are sent to SubQ's API (or the endpoint configured in Settings); never put secrets or credentials in the state. A call that uploads images or targets a non-SubQ endpoint asks for approval first, and a subagent is refused because it cannot ask.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "state": [
                    "type": ["string", "object", "array"],
                    "description": "Everything the questions must see. Facts only — the actual ticket, text, values or objects, not a summary or an opinion about them."
                ],
                "questions": [
                    "type": "object",
                    "description": "One to three named questions. Each name maps to {type: 'noul'|'choice'|'score', instructions: string, criteria: ...}. noul takes criteria.true/false; choice takes a map of option name to description; score takes an ordered array of level names, low to high.",
                    "additionalProperties": true
                ],
                "images": [
                    "type": "array",
                    "description": "Optional up to 4 images to judge alongside the state. Each entry is {path: 'shot.png', media_type: 'image/png'} — PNG, JPEG or WebP under 4 MiB, resolved against this turn's working directory. The file must be inside that folder; the tool refuses an image outside it because the bytes are uploaded. A call with images asks for approval first and is refused inside a subagent.",
                    "items": [
                        "type": "object",
                        "properties": [
                            "path": [
                                "type": "string",
                                "description": "Path to the image file, relative to this turn's working directory (or an absolute path that resolves inside it)."
                            ],
                            "media_type": [
                                "type": "string",
                                "description": "Optional MIME type; inferred from the extension when omitted."
                            ]
                        ],
                        "required": ["path"]
                    ]
                ]
            ]),
            "required": AnyCodable(["state", "questions"])
        ]
    )

    public init() {}

    #if DEBUG
    /// Lets a test answer `execute` without the network. The redirect guard,
    /// status mapping and sanitizing are pinned through it.
    nonisolated(unsafe) public static var clientFactoryForTesting: ((String) -> SubQDecisionClient)?
    /// Observes the endpoint `execute` resolved, so a test can prove the
    /// APPROVED url is used rather than the stored setting.
    nonisolated(unsafe) public static var endpointObserverForTesting: ((URL) -> Void)?
    /// The pinned endpoint, readable by a test's injected client factory so it
    /// can build a client aimed at the APPROVED destination instead of
    /// hardcoding one (which made the URL assertion vacuous).
    public static var pinnedEndpointForTesting: URL? { pinnedEndpoint }
    #endif

    public static let maximumLabels = 64
    public static let maximumLabelCharacters = 256

    /// The endpoint the operator approved for THIS call. Bound by
    /// `AgentHarness.executeTool` from the value the approval card displayed,
    /// because re-reading `UserDefaults` here let a `defaults write` between
    /// the card and the call retarget the key to a host nobody approved.
    @TaskLocal public static var pinnedEndpoint: URL?

    private func failure(_ message: String) -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
    }

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let key: (key: String, label: String)
        switch SubQDecisionKeySource.resolveDetail() {
        case .found(let value, let label):
            key = (value, label)
        case .missing:
            let error = SubQDecisionError.missingKey
            return failure(error.localizedDescription)
        case .unreadable(let status):
            let error = SubQDecisionError.keyUnreadable(status: status)
            return failure(error.localizedDescription)
        }

        guard let rawState = arguments["state"] else {
            return failure("Missing 'state' in subq_decide: it must carry the facts the questions judge.")
        }
        let state = AnyCodable(rawState)
        if let problem = Self.stateProblem(state) {
            return failure("subq_decide could not send this state: \(problem)")
        }

        guard let rawQuestions = arguments["questions"] as? [String: Any], !rawQuestions.isEmpty else {
            return failure("Missing or empty 'questions' in subq_decide. Send one to three named questions, for example {\"needs_human\": {\"type\": \"noul\", \"instructions\": \"…\", \"criteria\": {\"true\": \"…\", \"false\": \"…\"}}}.")
        }

        if let problem = Self.questionProblem(rawQuestions) {
            return failure("subq_decide could not use these questions: \(problem)")
        }

        var images: [SubQImage] = []
        if let value = arguments["images"], !(value is NSNull) {
            guard let rawImages = value as? [Any] else {
                return failure("'images' must be an array of {path: ...} objects.")
            }
            if !rawImages.isEmpty {
                // Without a working directory an absolute path could name any
                // file on this Mac, and the bytes leave the machine. A turn
                // with no workspace therefore gets no image upload.
                guard let workingDirectory else {
                    return failure("subq_decide can only read images inside this turn's working directory, and none is set. Send the state as text, or run the call from a workspace.")
                }
                // Silently ignoring the fifth image (or a malformed entry)
                // answered a different request than the caller made.
                guard rawImages.count <= SubQImage.maximumImages else {
                    return failure("\(rawImages.count) images were sent; subq_decide accepts at most \(SubQImage.maximumImages). Split the call.")
                }
                for entry in rawImages {
                    guard let dict = entry as? [String: Any],
                          let path = dict["path"] as? String,
                          !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        let described = AgentHarness.truncateText(
                            AgentHarness.sanitizeText(String(describing: entry)),
                            limit: 120
                        )
                        return failure("Every image entry must be an object with a non-empty 'path'. One entry was: \(described)")
                    }
                    let mediaType = dict["media_type"] as? String
                    // Up to 4 MiB per image is read off the main actor: the
                    // tool itself is MainActor-bound and this is a local file
                    // read.
                    let outcome = await Task.detached(priority: .userInitiated) { () -> (image: SubQImage?, error: String?) in
                        do {
                            return (try SubQImage.load(
                                path: path,
                                workingDirectory: workingDirectory,
                                mediaTypeOverride: mediaType
                            ), nil)
                        } catch {
                            return (nil, (error as? LocalizedError)?.errorDescription ?? error.localizedDescription)
                        }
                    }.value
                    if let image = outcome.image {
                        images.append(image)
                    } else {
                        return failure(outcome.error ?? "the image could not be read.")
                    }
                }
            }
        }

        let endpoint = Self.pinnedEndpoint ?? SubQDecisionSettings.endpoint()
        #if DEBUG
        Self.endpointObserverForTesting?(endpoint)
        #endif
        let client: SubQDecisionClient
        #if DEBUG
        client = Self.clientFactoryForTesting?(key.key)
            ?? SubQDecisionClient(endpoint: endpoint, apiKey: key.key)
        #else
        client = SubQDecisionClient(endpoint: endpoint, apiKey: key.key)
        #endif
        do {
            let response = try await client.decide(
                state: state,
                questions: rawQuestions.mapValues { AnyCodable($0) },
                images: images
            )

            if let problem = Self.answerProblem(response, questions: rawQuestions) {
                return failure("subq_decide answered in a shape this call cannot use: \(problem)")
            }

            var answersJSON: [String: Any] = [:]
            for (name, answer) in response.answers {
                answersJSON[Self.label(name)] = Self.jsonObject(answer)
            }

            // The answer's labels (choice names, legend rows) are echoed text,
            // so they go through the same marker pass as every other tool
            // result before they reach the transcript. `jsonObject` sanitizes
            // the structured copy; this covers the readable one.
            let summary = AgentHarness.truncateText(
                AgentHarness.sanitizeText(response.readableSummary),
                limit: max(0, maxOutputLength)
            )
            let servedModel = Self.label(response.model ?? client.model)
            // A Cloud Provider's display name is operator text, but it lands in
            // the transcript like any other label, so it gets the same pass.
            let keyLabel = Self.label(key.label)
            let readable = AgentHarness.truncateText("""
            SubQ decision (\(servedModel), key: \(keyLabel))
            \(summary)
            \(response.usageLine)
            """, limit: max(0, maxOutputLength))

            var data: [String: Any] = [
                "model": servedModel,
                "answers": answersJSON,
                "readable": summary,
                "key_source": keyLabel
            ]
            if let usage = response.usage {
                var usageJSON: [String: Any] = [:]
                if let input = usage.inputTokens { usageJSON["input_tokens"] = input }
                if let output = usage.outputTokens { usageJSON["output_tokens"] = output }
                data["usage"] = usageJSON
            }
            let res = AgentHarness.toolSuccessJSON(tool: "subq_decide", data: data)
            return (res, readable, nil, false)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as SubQDecisionError {
            return failure(error.localizedDescription)
        } catch {
            return failure("subq_decide failed: \(error.localizedDescription)")
        }
    }

    // MARK: Validation

    /// Question names from the raw `questions` argument, for a tool card's
    /// title. The card used to print the entire JSON blob as the summary line.
    static func questionNames(from raw: String?) -> [String] {
        guard let raw, let data = raw.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }
        return object.keys.sorted()
    }

    /// `maximumStateCharacters` was documented and never enforced, so `state`
    /// was effectively uncapped. Measuring the encoded body is the same number
    /// the request will spend on the wire.
    static func stateProblem(_ state: AnyCodable) -> String? {
        let encoder = JSONEncoder()
        guard let data = try? encoder.encode(state) else {
            return "the state is not JSON-encodable."
        }
        guard data.count <= SubQDecisionClient.maximumStateCharacters else {
            return "the state is \(data.count) bytes; the limit is \(SubQDecisionClient.maximumStateCharacters). Split the batch into smaller calls."
        }
        return nil
    }

    /// Catches the request the API would reject with a 422 before the round
    /// trip, with text that says how to fix it. Deliberately light: the API is
    /// the authority on criteria semantics.
    static func questionProblem(_ questions: [String: Any]) -> String? {
        if questions.isEmpty {
            return "No questions were sent; send one to three."
        }
        if questions.count > SubQDecisionClient.maximumQuestions {
            return "\(questions.count) questions were sent; the API accepts at most \(SubQDecisionClient.maximumQuestions)."
        }
        for (name, value) in questions {
            guard let question = value as? [String: Any] else {
                return "'\(name)' is not an object with type and instructions."
            }
            let type = ((question["type"] as? String) ?? "").lowercased()
            guard ["noul", "choice", "score"].contains(type) else {
                return "'\(name)' has type '\(type)'; use noul, choice or score."
            }
            let instructions = (question["instructions"] as? String) ?? ""
            guard !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return "'\(name)' has no instructions; say what the question is asking."
            }
            if type == "choice" {
                guard let criteria = question["criteria"] as? [String: Any], criteria.count >= 2 else {
                    return "'\(name)' is a choice question and needs criteria with at least two named options."
                }
            }
            if type == "score" {
                guard let criteria = question["criteria"] as? [Any], (2...10).contains(criteria.count) else {
                    return "'\(name)' is a score question and needs criteria as an ordered array of 2-10 level names."
                }
            }
        }
        return nil
    }

    /// The API is the authority on criteria semantics, but an answer about a
    /// label that was never offered is not an answer to the question asked, and
    /// a missing answer is not a decision. Both used to report success.
    static func answerProblem(_ response: SubQDecisionResponse, questions: [String: Any]) -> String? {
        for (name, answer) in response.answers.sorted(by: { $0.key < $1.key }) {
            guard let raw = questions[name], let question = raw as? [String: Any] else {
                return "it answered '\(name)', which was not one of the questions."
            }
            switch answer {
            case .noul(let probability):
                guard probability.isFinite, (0.0...1.0).contains(probability) else {
                    return "'\(name)' returned probability \(probability), outside 0...1."
                }
            case .choice(let choice, let probabilities, let confidence):
                if let criteria = question["criteria"] as? [String: Any],
                   !criteria.keys.contains(choice) {
                    return "'\(name)' chose '\(choice)', which is not one of the offered options (\(criteria.keys.sorted().joined(separator: ", ")))."
                }
                for (key, value) in probabilities where !value.isFinite || !(0.0...1.0).contains(value) {
                    return "'\(name)' returned probability \(value) for '\(key)', outside 0...1."
                }
                if let confidence, !confidence.isFinite || !(0.0...1.0).contains(confidence) {
                    return "'\(name)' returned confidence \(confidence), outside 0...1."
                }
            case .score(let score, let legend, let probabilities, let confidence):
                guard score.isFinite else {
                    return "'\(name)' returned a non-finite score."
                }
                // `isFinite` is not enough: 1e300 passes it and then traps in
                // `Int(_:)`, the same crash `readableDescription` already fixed
                // with `Int(exactly:)`. This site runs inside `execute`, after
                // the network call, so the trap takes the app down instead of
                // failing one tool result.
                guard let level = Int(exactly: score.rounded(.toNearestOrAwayFromZero)) else {
                    return "'\(name)' scored \(score), which is not a level any legend or criteria list can name."
                }
                if !legend.isEmpty {
                    guard level >= 0, legend[String(level)] != nil else {
                        return "'\(name)' scored \(score), which is not a level its legend offers (\(legend.keys.sorted().joined(separator: ", ")))."
                    }
                } else if let criteria = question["criteria"] as? [Any], !criteria.isEmpty {
                    guard level >= 0, level < criteria.count else {
                        return "'\(name)' scored \(score), outside 0..<\(criteria.count)."
                    }
                }
                for (key, value) in probabilities where !value.isFinite || !(0.0...1.0).contains(value) {
                    return "'\(name)' returned probability \(value) for '\(key)', outside 0...1."
                }
                if let confidence, !confidence.isFinite || !(0.0...1.0).contains(confidence) {
                    return "'\(name)' returned confidence \(confidence), outside 0...1."
                }
            case .other(let type):
                return "'\(name)' answered with an unrecognised shape '\(type)'."
            }
        }
        if let missing = questions.keys.sorted().first(where: { response.answers[$0] == nil }) {
            return "no answer was returned for '\(missing)'."
        }
        return nil
    }

    /// Bounded, sanitized label. A hostile endpoint controls every string it
    /// echoes back, and the response ceiling alone still allows megabytes of
    /// labels in the transcript.
    static func label(_ text: String) -> String {
        let safe = AgentHarness.sanitizeText(text)
        guard safe.count > maximumLabelCharacters else { return safe }
        return String(safe.prefix(maximumLabelCharacters)) + "…"
    }

    /// What the approval card must show: the ORIGIN the state (and any image
    /// bytes) go to, plus the files themselves. `endpointString` is the
    /// destination the turn loop pinned beside the record, which is what the
    /// call will actually use.
    public static func approvalDetail(rawArguments: String?, endpointString: String?) -> String {
        let endpoint = endpointString.flatMap(URL.init(string:)) ?? SubQDecisionSettings.endpoint()
        let origin = SubQDecisionSettings.origin(endpoint) ?? endpoint.absoluteString
        var names: [String] = []
        // The state can be 200 KB; only pay to parse when an upload is possible.
        if let rawArguments, rawArguments.contains("\"images\""),
           let data = rawArguments.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let images = object["images"] as? [Any] {
            names = images.compactMap { entry in
                guard let path = (entry as? [String: Any])?["path"] as? String, !path.isEmpty else { return nil }
                return (path as NSString).lastPathComponent
            }
        }
        guard !names.isEmpty else { return "Sends the state to \(origin)." }
        return "Sends the state and \(names.count) image(s) to \(origin): \(names.joined(separator: ", "))"
    }

    static func jsonObject(_ answer: SubQDecisionAnswer) -> [String: Any] {
        switch answer {
        case .noul(let probability):
            return ["type": "noul", "probability": probability]
        case .choice(let choice, let probabilities, let confidence):
            var object: [String: Any] = [
                "type": "choice",
                "choice": AgentHarness.sanitizeText(choice),
                "probabilities": sanitized(probabilities)
            ]
            if let confidence { object["confidence"] = confidence }
            return object
        case .score(let score, let legend, let probabilities, let confidence):
            var object: [String: Any] = [
                "type": "score",
                "score": score,
                "legend": sanitized(legend),
                "probabilities": sanitized(probabilities)
            ]
            if let confidence { object["confidence"] = confidence }
            return object
        case .other(let type):
            return ["type": type.isEmpty ? "unknown" : label(type)]
        }
    }

    /// A hostile endpoint controls every label it echoes back, so the choice
    /// names, legend rows and question keys are untrusted text. Sanitizing them
    /// here (not only the human-readable summary) is what keeps a forged
    /// `<|im_start|>system` key out of the `resultJSON` the model is shown.
    static func sanitized(_ probabilities: [String: Double]) -> [String: Double] {
        probabilities.sorted { $0.key < $1.key }.prefix(maximumLabels).reduce(into: [:]) { result, entry in
            result[label(entry.key)] = entry.value
        }
    }

    static func sanitized(_ legend: [String: String]) -> [String: String] {
        legend.sorted { $0.key < $1.key }.prefix(maximumLabels).reduce(into: [:]) { result, entry in
            result[label(entry.key)] = label(entry.value)
        }
    }
}
