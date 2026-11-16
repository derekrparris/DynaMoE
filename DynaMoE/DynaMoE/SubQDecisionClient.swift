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

// MARK: - Errors

public nonisolated enum SubQDecisionError: LocalizedError, Equatable {
    case missingKey
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
              let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http",
              let host = url.host, !host.isEmpty else {
            return fallback
        }
        return url
    }

    /// Non-nil when the configured endpoint is not SubQ's own host. The request
    /// carries the API key as a Bearer token, so a custom URL sends that secret
    /// wherever it points; the Settings card shows this before a test does.
    public static func customEndpointWarning(defaults: UserDefaults = .standard) -> String? {
        let configured = endpoint(defaults: defaults)
        let officialHost = URL(string: defaultEndpoint)?.host?.lowercased()
        guard configured.host?.lowercased() != officialHost else { return nil }
        let destination = configured.host ?? configured.absoluteString
        return "This endpoint is not api.subq.ai. The API key is sent as a Bearer token to \(destination); use a custom endpoint only with a server you trust."
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

    /// The key and where it came from, for the Settings readout and the tool's
    /// error text. Ordered so the user's existing Cloud Provider setup wins.
    public static func resolve(defaults: UserDefaults = .standard) -> (key: String, label: String)? {
        for provider in matchingProviders(defaults: defaults) {
            if let key = CloudSecretStore.shared.get(providerId: provider.id),
               !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return (key, "Cloud Provider '\(provider.displayName)'")
            }
        }
        if let key = SkillsSecretStore.shared.get(account: account),
           !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return (key, "Skills Keychain")
        }
        if let key = ProcessInfo.processInfo.environment[environmentName],
           !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return (key, "\(environmentName) environment variable")
        }
        return nil
    }

    /// Attributes only where possible, so this is safe on the prompt-seeding
    /// path: no query here can raise a Keychain dialog.
    ///
    /// It answers the SAME question `resolve` does: a credential the build
    /// actually reads. `exists` covers both keychains, so a legacy-only item
    /// would report "Ready" while every call failed with a missing key until
    /// the Stored Credentials import ran.
    public static func hasAnyKey(defaults: UserDefaults = .standard) -> Bool {
        for provider in matchingProviders(defaults: defaults) where CloudSecretStore.shared.hasCurrentCredential(providerId: provider.id) {
            return true
        }
        if SkillsSecretStore.shared.hasCurrentCredential(account: account) { return true }
        if let key = ProcessInfo.processInfo.environment[environmentName], !key.isEmpty { return true }
        return false
    }

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

    /// Same answer as `resolve`, without reading a secret. The Settings card
    /// renders this on every body evaluation, and a read is the one Keychain
    /// call that can prompt for access on the legacy backend.
    public static func describeWithoutPrompt(defaults: UserDefaults = .standard) -> String {
        let providers = matchingProviders(defaults: defaults)
        if let provider = providers.first(where: { CloudSecretStore.shared.hasCurrentCredential(providerId: $0.id) }) {
            return "Using the key stored for Cloud Provider '\(provider.displayName)'."
        }
        if SkillsSecretStore.shared.hasCurrentCredential(account: account) {
            return "Using the key stored in this card."
        }
        if let key = ProcessInfo.processInfo.environment[environmentName], !key.isEmpty {
            return "Using \(environmentName) from the app environment."
        }
        if legacyKeyPending(defaults: defaults) {
            return "A SubQ key is stored in the legacy keychain. Import it from Settings → Stored Credentials; this build cannot read it until then."
        }
        if let provider = providers.first {
            return "Cloud Provider '\(provider.displayName)' exists but has no stored key."
        }
        return "No SubQ API key. Add one here, or add a Cloud Provider for api.subq.ai."
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
    case choice(choice: String?, probabilities: [String: Double], confidence: Double?)
    case score(score: Double?, legend: [String: String], probabilities: [String: Double], confidence: Double?)
    case other(type: String)

    /// One-line, model-facing reading. Numbers only: the thresholds that turn
    /// them into a decision belong to the skill, not to the client.
    public var readableDescription: String {
        switch self {
        case .noul(let probability):
            return "noul \(Self.number(probability)) (probability of yes)"
        case .choice(let choice, let probabilities, let confidence):
            var parts = ["choice \"\(choice ?? "none")\""]
            if let confidence { parts.append("confidence \(Self.number(confidence))") }
            let top = probabilities.sorted { $0.value > $1.value }.prefix(3)
                .map { "\($0.key) \(Self.number($0.value))" }
            if !top.isEmpty { parts.append(top.joined(separator: ", ")) }
            return parts.joined(separator: " · ")
        case .score(let score, let legend, let probabilities, let confidence):
            var parts = ["score \(Self.number(score ?? 0))"]
            if let score {
                let level = String(Int(score.rounded()))
                if let name = legend[level] { parts.append("= \(name)") }
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
            let probability = try container.decode(Double.self, forKey: .noul)
            self = .noul(probability: probability)
        case "choice":
            self = .choice(
                choice: try? container.decodeIfPresent(String.self, forKey: .choice),
                probabilities: (try? container.decode([String: Double].self, forKey: .probabilities)) ?? [:],
                confidence: try? container.decodeIfPresent(Double.self, forKey: .confidence)
            )
        case "score":
            self = .score(
                score: try? container.decodeIfPresent(Double.self, forKey: .score),
                legend: (try? container.decode([String: String].self, forKey: .legend)) ?? [:],
                probabilities: (try? container.decode([String: Double].self, forKey: .probabilities)) ?? [:],
                confidence: try? container.decodeIfPresent(Double.self, forKey: .confidence)
            )
        default:
            self = .other(type: type)
        }
    }
}

public nonisolated struct SubQDecisionResponse: Decodable, Equatable, Sendable {
    public let model: String?
    public let answers: [String: SubQDecisionAnswer]
    public let usage: SubQDecisionUsage?

    public var readableSummary: String {
        guard !answers.isEmpty else { return "No answers were returned." }
        return answers.keys.sorted().map { name in
            "- \(name): \(answers[name]!.readableDescription)"
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
    public static func load(path: String, workingDirectory: URL? = nil) throws -> SubQImage {
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
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values?.isRegularFile != false else {
            throw SubQDecisionError.badRequest("image is not a regular file at \(url.path)")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SubQDecisionError.badRequest("image not found at \(url.path)")
        }
        let size = values?.fileSize ?? 0
        guard size <= maximumBytes else {
            throw SubQDecisionError.badRequest(
                "image \(url.lastPathComponent) is \(size) bytes; the API accepts at most \(maximumBytes) decoded bytes per image."
            )
        }
        let data = try Data(contentsOf: url)
        let mediaType: String
        switch url.pathExtension.lowercased() {
        case "png": mediaType = "image/png"
        case "jpg", "jpeg": mediaType = "image/jpeg"
        case "webp": mediaType = "image/webp"
        default:
            throw SubQDecisionError.badRequest(
                "image \(url.lastPathComponent) is not a supported format; use PNG, JPEG or WebP."
            )
        }
        return SubQImage(mediaType: mediaType, base64: data.base64EncodedString())
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
        guard let originHost = origin?.host?.lowercased(), !originHost.isEmpty,
              let destinationHost = destination?.host?.lowercased(), !destinationHost.isEmpty else {
            return false
        }
        return originHost == destinationHost
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
                        try? await Task.sleep(nanoseconds: 1_000_000_000)
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

    private func send(payload: Data) async throws -> SubQDecisionResponse {
        guard let scheme = endpoint.scheme?.lowercased(), scheme == "https" || scheme == "http",
              endpoint.host?.isEmpty == false else {
            throw SubQDecisionError.badRequest(
                "the configured endpoint \(endpoint.absoluteString) is not an http(s) URL with a host."
            )
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = payload

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw SubQDecisionError.transport(error.localizedDescription)
        }
        guard data.count <= Self.maximumResponseBytes else {
            throw SubQDecisionError.decoding(
                "the response was \(data.count) bytes; this client reads at most \(Self.maximumResponseBytes)."
            )
        }

        guard let http = response as? HTTPURLResponse else {
            throw SubQDecisionError.transport("the response was not HTTP")
        }
        guard (200...299).contains(http.statusCode) else {
            let summary = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data))?.summary ?? ""
            let fallback = String(data: data.prefix(400), encoding: .utf8) ?? ""
            throw SubQDecisionError.forStatus(
                http.statusCode,
                detail: summary.isEmpty ? fallback.trimmingCharacters(in: .whitespacesAndNewlines) : summary
            )
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
        description: "Asks SubQ's decisions-only model (subq-snap-preview) one to three typed questions about a state and returns probabilities plus confidence. Use it for yes/no gates, routing or classification, triage and ordered scoring. Do NOT use it for open-ended analysis, drafting, summarizing, or extracting fields — use a chat model for those. Load the subq-decisions skill first for framing rules and thresholds. The state and questions are sent to SubQ's API; never put secrets or credentials in the state.",
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
                    "description": "Optional up to 4 images to judge alongside the state. Each entry is {path: '/absolute/file.png'} — PNG, JPEG or WebP under 4 MiB. The file must be inside this turn's working directory; the tool refuses an image outside it because the bytes are uploaded to SubQ.",
                    "items": [
                        "type": "object",
                        "properties": [
                            "path": [
                                "type": "string",
                                "description": "Absolute path to the image file."
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

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let key = SubQDecisionKeySource.resolve() else {
            let error = SubQDecisionError.missingKey
            return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: error.localizedDescription), nil, error.localizedDescription, false)
        }

        guard let rawState = arguments["state"] else {
            let message = "Missing 'state' in subq_decide: it must carry the facts the questions judge."
            return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
        }
        let state = AnyCodable(rawState)
        if let problem = Self.stateProblem(state) {
            let message = "subq_decide could not send this state: \(problem)"
            return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
        }

        guard let rawQuestions = arguments["questions"] as? [String: Any], !rawQuestions.isEmpty else {
            let message = "Missing or empty 'questions' in subq_decide. Send one to three named questions, for example {\"needs_human\": {\"type\": \"noul\", \"instructions\": \"…\", \"criteria\": {\"true\": \"…\", \"false\": \"…\"}}}."
            return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
        }

        if let problem = Self.questionProblem(rawQuestions) {
            let message = "subq_decide could not use these questions: \(problem)"
            return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
        }

        var images: [SubQImage] = []
        if let rawImages = arguments["images"] as? [Any] {
            for entry in rawImages.prefix(SubQImage.maximumImages) {
                guard let dict = entry as? [String: Any],
                      let path = dict["path"] as? String else { continue }
                do {
                    images.append(try SubQImage.load(path: path, workingDirectory: workingDirectory))
                } catch {
                    let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                    return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
                }
            }
        }

        let client = SubQDecisionClient(apiKey: key.key)
        do {
            let response = try await client.decide(
                state: state,
                questions: rawQuestions.mapValues { AnyCodable($0) },
                images: images
            )

            var answersJSON: [String: Any] = [:]
            for (name, answer) in response.answers {
                answersJSON[name] = Self.jsonObject(answer)
            }

            // The answer's labels (choice names, legend rows) are echoed text,
            // so they go through the same marker pass as every other tool
            // result before they reach the transcript.
            let summary = AgentHarness.sanitizeText(response.readableSummary)
            let readable = """
            SubQ decision (\(response.model ?? client.model), key: \(key.label))
            \(summary)
            \(response.usageLine)
            """

            var data: [String: Any] = [
                "model": response.model ?? client.model,
                "answers": answersJSON,
                "readable": summary,
                "key_source": key.label
            ]
            if let usage = response.usage {
                var usageJSON: [String: Any] = [:]
                if let input = usage.inputTokens { usageJSON["input_tokens"] = input }
                if let output = usage.outputTokens { usageJSON["output_tokens"] = output }
                data["usage"] = usageJSON
            }
            let res = AgentHarness.toolSuccessJSON(tool: "subq_decide", data: data)
            return (res, readable, nil, false)
        } catch let error as SubQDecisionError {
            let message = error.localizedDescription
            return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
        } catch {
            let message = "subq_decide failed: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "subq_decide", error: message), nil, message, false)
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

    static func jsonObject(_ answer: SubQDecisionAnswer) -> [String: Any] {
        switch answer {
        case .noul(let probability):
            return ["type": "noul", "probability": probability]
        case .choice(let choice, let probabilities, let confidence):
            var object: [String: Any] = ["type": "choice", "probabilities": probabilities]
            if let choice { object["choice"] = choice }
            if let confidence { object["confidence"] = confidence }
            return object
        case .score(let score, let legend, let probabilities, let confidence):
            var object: [String: Any] = ["type": "score", "legend": legend, "probabilities": probabilities]
            if let score { object["score"] = score }
            if let confidence { object["confidence"] = confidence }
            return object
        case .other(let type):
            return ["type": type.isEmpty ? "unknown" : type]
        }
    }
}
