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
//  that stands in for the Python helper a shell-based agent would run, which
//  also keeps the API key out of the model's reach: DynaMoE's `shell_run` does
//  not inherit the user's shell environment, and a secret in a process
//  environment is a secret the model can print.
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
        return URL(string: raw.isEmpty ? defaultEndpoint : raw) ?? URL(string: defaultEndpoint)!
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
    public static func hasAnyKey(defaults: UserDefaults = .standard) -> Bool {
        for provider in matchingProviders(defaults: defaults) where CloudSecretStore.shared.exists(providerId: provider.id) {
            return true
        }
        if SkillsSecretStore.shared.exists(account: account) { return true }
        if let key = ProcessInfo.processInfo.environment[environmentName], !key.isEmpty { return true }
        return false
    }

    /// Same answer as `resolve`, without reading a secret. The Settings card
    /// renders this on every body evaluation, and a read is the one Keychain
    /// call that can prompt for access on the legacy backend.
    public static func describeWithoutPrompt(defaults: UserDefaults = .standard) -> String {
        let providers = matchingProviders(defaults: defaults)
        if let provider = providers.first(where: { CloudSecretStore.shared.exists(providerId: $0.id) }) {
            return "Using the key stored for Cloud Provider '\(provider.displayName)'."
        }
        if SkillsSecretStore.shared.exists(account: account) {
            return "Using the key stored in this card."
        }
        if let key = ProcessInfo.processInfo.environment[environmentName], !key.isEmpty {
            return "Using \(environmentName) from the app environment."
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
            let probability = (try? container.decode(Double.self, forKey: .noul)) ?? 0
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
    /// over 4 MiB decoded, or a format it does not accept.
    public static func load(path: String) throws -> SubQImage {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SubQDecisionError.badRequest("image not found at \(url.path)")
        }
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
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

// MARK: - Client

public nonisolated struct SubQDecisionClient: Sendable {
    public static let maximumQuestions = 3
    public static let maximumStateCharacters = 200_000
    public static let defaultTimeout: TimeInterval = 30

    public let endpoint: URL
    public let apiKey: String
    public let model: String
    public let timeout: TimeInterval

    public init(
        endpoint: URL = SubQDecisionSettings.endpoint(),
        apiKey: String,
        model: String = SubQDecisionSettings.model(),
        timeout: TimeInterval = SubQDecisionClient.defaultTimeout
    ) {
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.model = model
        self.timeout = timeout
    }

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
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw SubQDecisionError.transport(error.localizedDescription)
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
        description: "Asks SubQ's decisions-only model (subq-snap-preview) one to three typed questions about a state and returns probabilities plus confidence. Use it for yes/no gates, routing or classification, triage and ordered scoring. Do NOT use it for open-ended analysis, drafting, summarizing, or extracting fields — use a chat model for those. Load the subq-decisions skill first for framing rules and thresholds.",
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
                    "description": "Optional up to 4 images to judge alongside the state. Each entry is {path: '/absolute/file.png'} — PNG, JPEG or WebP under 4 MiB.",
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
                    images.append(try SubQImage.load(path: path))
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

            let readable = """
            SubQ decision (\(response.model ?? client.model), key: \(key.label))
            \(response.readableSummary)
            \(response.usageLine)
            """

            var data: [String: Any] = [
                "model": response.model ?? client.model,
                "answers": answersJSON,
                "readable": response.readableSummary,
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
