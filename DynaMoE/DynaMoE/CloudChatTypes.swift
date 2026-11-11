//
//  CloudChatTypes.swift
//  DynaMoE
//
//  Wire types, stream parser, failures, and message assembly for connected
//  cloud providers. Pure values and pure functions only: the networking
//  lives in CloudChatClient.swift, so every mapping below is unit-testable
//  without a network or a running server.
//

import Foundation

// MARK: - Wire Messages

/// One assistant function call in the OpenAI Chat-Completions shape
/// (`{"id":"call_…","type":"function","function":{"name":…,"arguments":"<json>"}}`).
/// `arguments` is the provider's raw JSON string, exactly as emitted.
public nonisolated struct CloudChatToolCall: Codable, Equatable, Sendable {
    public struct Function: Codable, Equatable, Sendable {
        public let name: String
        public let arguments: String

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }

    public let id: String
    public let type: String
    public let function: Function

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.type = "function"
        self.function = Function(name: name, arguments: arguments)
    }
}

/// One Chat-Completions message. `content` is nil for an assistant turn that
/// only issued tool calls, and `toolCalls`/`toolCallId` carry the OpenAI
/// tool-exchange shape so a reloaded conversation replays faithfully.
///
/// Images ride as `imageParts` (data URLs) and change only the *shape* of
/// `content` on the wire: with parts it is encoded as
/// `[{"type":"text",…},{"type":"image_url","image_url":{"url":…}}]`, without
/// them as a bare string, byte-identical to the pre-image request. That keeps
/// every text-only turn (and every existing request-shape test) unchanged.
public nonisolated struct CloudChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: String?
    public let toolCalls: [CloudChatToolCall]?
    public let toolCallId: String?
    /// Data URLs for images attached to this message, in display order.
    public let imageParts: [String]?
    /// Metadata-only labels (`"[image: shot.png 1024×768 412 KB]"`) for prompt
    /// renders and request logs. Deliberately **not** encoded onto the wire: the
    /// provider gets the bytes, not our filenames.
    public let imageDescriptions: [String]?

    public init(
        role: String,
        content: String?,
        toolCalls: [CloudChatToolCall]? = nil,
        toolCallId: String? = nil,
        imageParts: [String]? = nil,
        imageDescriptions: [String]? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.imageParts = imageParts
        self.imageDescriptions = imageDescriptions
    }

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.role = try container.decode(String.self, forKey: .role)
        self.content = try container.decodeIfPresent(String.self, forKey: .content)
        self.toolCalls = try container.decodeIfPresent([CloudChatToolCall].self, forKey: .toolCalls)
        self.toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
        // Wire-only type: image bytes are rebuilt per turn from the session's
        // attachment refs, so a decoded message carries none. `encode(to:)` also
        // omits them on purpose. Persisting this type would need its own coder,
        // not these keys, because the OpenAI request shape forbids extra fields.
        self.imageParts = nil
        self.imageDescriptions = nil
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        if let imageParts, !imageParts.isEmpty {
            var parts = container.nestedUnkeyedContainer(forKey: .content)
            if let content, !content.isEmpty {
                try parts.encode(WireContentPart.text(content))
            }
            for url in imageParts {
                try parts.encode(WireContentPart.image(url))
            }
        } else {
            try container.encodeIfPresent(content, forKey: .content)
        }
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
        try container.encodeIfPresent(toolCallId, forKey: .toolCallId)
    }
}

/// One entry in the multi-part `content` array. Optional fields are omitted by
/// the synthesized encoder, so a text part carries no `image_url` and an image
/// part carries no `text`.
private nonisolated struct WireContentPart: Encodable {
    struct ImageURL: Encodable {
        let url: String
    }

    let type: String
    let text: String?
    let imageURL: ImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }

    static func text(_ text: String) -> WireContentPart {
        WireContentPart(type: "text", text: text, imageURL: nil)
    }

    static func image(_ url: String) -> WireContentPart {
        WireContentPart(type: "image_url", text: nil, imageURL: ImageURL(url: url))
    }
}

/// A tool call assembled from a streamed `delta.tool_calls[]` fragment
/// sequence. Arguments arrive split across chunks, so they are only decoded
/// once the stream ends.
public nonisolated struct CloudToolCall: Equatable, Sendable {
    public let id: String
    public let name: String
    public let argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

// MARK: - Sampling

/// What a provider is willing to be told about prompt caching.
///
/// The prefix is byte-identical on every step of an agent run either way; the
/// difference is whether the request ASKS for the reuse. `affinity` is the
/// safe default: two custom `x-…` headers, which no HTTP server is obliged to
/// understand and therefore cannot break a turn. `openAI` adds
/// `prompt_cache_key`, the documented OpenAI routing hint, and is opt-in
/// because a strict compatible endpoint is allowed to reject unknown body
/// fields (the request path retries once without it if that happens).
public nonisolated enum CloudCacheHints: String, Codable, Equatable, Hashable, Sendable, CaseIterable {
    /// Send nothing. For an endpoint that is known to reject extra headers.
    case none
    /// Session-affinity headers only.
    case affinity
    /// Affinity headers plus `prompt_cache_key`.
    case openAI

    /// A session-affinity header is what lets a router keep one conversation on
    /// one replica, which is the only way an automatic per-instance prefix
    /// cache ever hits.
    public var sendsAffinityHeaders: Bool { self != .none }

    /// `prompt_cache_key` is an OpenAI-documented field; other compatible
    /// endpoints may ignore it or reject the body, so it is opt-in.
    public var sendsPromptCacheKey: Bool { self == .openAI }

    public var displayName: String {
        switch self {
        case .none: return "Off"
        case .affinity: return "Session affinity"
        case .openAI: return "Session affinity + cache key"
        }
    }
}

/// Sampling knobs the OpenAI-compatible surface understands. DynaMoE also
/// samples with Min-P, Top-K, and a repetition penalty; remote providers do
/// not expose those, so they are deliberately not forwarded (same silent-gap
/// contract the Apple Foundation Model path documents).
public nonisolated struct CloudChatGenerationParams: Equatable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var presencePenalty: Double?
    public var maxTokens: Int?
    /// Stable per-conversation identifier carrying the cache hint. Hashed
    /// before it leaves the machine (see `CloudSessionKey`) so a provider never
    /// receives a raw DynaMoE session UUID.
    public var sessionKey: String?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        presencePenalty: Double? = nil,
        maxTokens: Int? = nil,
        sessionKey: String? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.maxTokens = maxTokens
        self.sessionKey = sessionKey
    }
}

/// The opaque value a provider sees instead of a conversation UUID.
///
/// Cache affinity only needs the value to be stable for one conversation and
/// unlikely to collide, so it is a 64-bit FNV-1a digest in hex: no dependency,
/// no key material, and nothing that identifies the user or the machine. It is
/// deliberately not the session UUID — that would write the app's identifiers
/// into a third party's logs for no benefit.
public nonisolated enum CloudSessionKey {
    public static func hashed(_ sessionId: UUID) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in sessionId.uuidString.lowercased().utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01B3
        }
        return String(hash, radix: 16)
    }
}

// MARK: - Usage & Results

/// The usage block a provider reports at the end of a stream.
///
/// `cachedTokens` is the part of `promptTokens` the provider served from its
/// prompt cache. It is the single number that tells a multi-step agent run
/// apart from a conversation that re-prefills everything: the same prompt
/// costs a fraction as much when most of it is a cache read. Three wire shapes
/// carry it — OpenAI's nested `prompt_tokens_details.cached_tokens`, DeepSeek's
/// top-level `prompt_cache_hit_tokens`, and Anthropic's
/// `cache_read_input_tokens` — so all three are decoded here rather than
/// guessed at the call site.
public nonisolated struct CloudUsage: Codable, Equatable, Sendable {
    public let promptTokens: Int?
    public let completionTokens: Int?
    public let totalTokens: Int?
    /// Input tokens served from the provider's prompt cache.
    public let cachedTokens: Int?
    /// Tokens written into the provider's cache by this request
    /// (Anthropic-style providers bill this separately from a cache read).
    public let cacheWriteTokens: Int?

    public init(
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        totalTokens: Int? = nil,
        cachedTokens: Int? = nil,
        cacheWriteTokens: Int? = nil
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.cachedTokens = cachedTokens
        self.cacheWriteTokens = cacheWriteTokens
    }

    /// The input tokens actually billed at the full rate: everything the
    /// provider processed minus what it served from cache. Nil when the
    /// provider reported no prompt count at all (an endpoint that drops usage).
    public var billableInputTokens: Int? {
        guard let promptTokens else { return nil }
        return max(0, promptTokens - (cachedTokens ?? 0))
    }

    /// Fraction of the prompt served from cache, 0...1, or nil when the
    /// provider reported no cache split at all. A reported zero is a real
    /// answer ("nothing was cached"); silence is not.
    public var cacheReadFraction: Double? {
        guard let promptTokens, promptTokens > 0, let cachedTokens else { return nil }
        return Double(min(cachedTokens, promptTokens)) / Double(promptTokens)
    }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case cachedTokens = "cached_tokens"
        case promptCacheHitTokens = "prompt_cache_hit_tokens"
        case cacheReadInputTokens = "cache_read_input_tokens"
        case cacheCreationInputTokens = "cache_creation_input_tokens"
        case cacheWriteTokens = "cache_write_tokens"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }

    private struct PromptTokensDetails: Decodable {
        let cachedTokens: Int?

        enum CodingKeys: String, CodingKey { case cachedTokens = "cached_tokens" }
    }

    /// The first key that decodes to a value. A provider that spells a field
    /// differently should not cost the whole usage block, so an unreadable key
    /// falls through to the next spelling instead of throwing.
    private static func firstInt(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ keys: [CodingKeys]
    ) -> Int? {
        for key in keys {
            if let value = try? container.decodeIfPresent(Int.self, forKey: key) {
                return value
            }
        }
        return nil
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let details = try? container.decodeIfPresent(PromptTokensDetails.self, forKey: .promptTokensDetails)
        let openAIPrompt = Self.firstInt(container, [.promptTokens])
        let inputTokens = Self.firstInt(container, [.inputTokens])
        let cachedTokens = Self.firstInt(
            container,
            [.promptCacheHitTokens, .cachedTokens, .cacheReadInputTokens]
        ) ?? details?.cachedTokens
        let cacheWriteTokens = Self.firstInt(
            container,
            [.cacheCreationInputTokens, .cacheWriteTokens]
        )
        // Anthropic splits the prompt three ways: `input_tokens` is only the
        // share that was neither read from nor written to its cache, and the
        // other two columns carry the rest. Its `usage` block is recognised by
        // those Anthropic-only cache keys, so a provider that publishes
        // `input_tokens` as the total (OpenAI's Responses shape) keeps that
        // meaning when no cache columns ride beside it. Normalising to the
        // total here is what lets `billableInputTokens` and `CloudPricing`
        // subtract a cache read without clamping the prompt to zero.
        let usesAnthropicCacheColumns = openAIPrompt == nil
            && inputTokens != nil
            && (container.contains(.cacheReadInputTokens)
                || container.contains(.cacheCreationInputTokens))
        if usesAnthropicCacheColumns, let inputTokens {
            self.promptTokens = inputTokens + (cachedTokens ?? 0) + (cacheWriteTokens ?? 0)
        } else {
            self.promptTokens = openAIPrompt ?? inputTokens
        }
        self.completionTokens = Self.firstInt(container, [.completionTokens, .outputTokens])
        self.totalTokens = Self.firstInt(container, [.totalTokens])
        self.cachedTokens = cachedTokens
        self.cacheWriteTokens = cacheWriteTokens
    }

    /// Encodes the canonical OpenAI shape. Nothing in the app persists a usage
    /// block, so this exists only so the type stays round-trippable in tests.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(promptTokens, forKey: .promptTokens)
        try container.encodeIfPresent(completionTokens, forKey: .completionTokens)
        try container.encodeIfPresent(totalTokens, forKey: .totalTokens)
        if let cachedTokens {
            var details = container.nestedContainer(keyedBy: PromptTokensDetails.CodingKeys.self, forKey: .promptTokensDetails)
            try details.encode(cachedTokens, forKey: .cachedTokens)
        }
        try container.encodeIfPresent(cacheWriteTokens, forKey: .cacheCreationInputTokens)
    }
}

public nonisolated struct CloudTurnResult: Equatable, Sendable {
    public let text: String
    public let thinking: String
    public let finishReason: String?
    public let usage: CloudUsage?
    public let toolCalls: [CloudToolCall]

    public init(
        text: String,
        thinking: String,
        finishReason: String?,
        usage: CloudUsage?,
        toolCalls: [CloudToolCall] = []
    ) {
        self.text = text
        self.thinking = thinking
        self.finishReason = finishReason
        self.usage = usage
        self.toolCalls = toolCalls
    }
}

/// Everything one streamed cloud turn can report to the chat surface.
public nonisolated enum CloudTurnEvent: Equatable, Sendable {
    case started
    case thinkingDelta(String)
    case contentDelta(String)
    case finished(CloudTurnResult)
    case failed(CloudTurnFailure)
}

// MARK: - Failures

public nonisolated enum CloudTurnFailure: Equatable, Sendable, LocalizedError {
    case missingAPIKey(providerName: String)
    /// The key is stored but this build could not read it (dismissed dialog,
    /// locked keychain). Distinct from `missingAPIKey` because the fix differs.
    case credentialAccessDenied(providerName: String)
    /// The key still lives in the older keychain and has not been imported.
    case credentialNeedsImport(providerName: String)
    case unsupportedFlavor(flavor: String)
    case httpStatus(code: Int, summary: String)
    case transport(String)
    case invalidResponse(String)

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey(let providerName):
            return "No API key is stored for \(providerName). Add one in Settings ▸ Cloud Providers."
        case .credentialAccessDenied(let providerName):
            return "DynaMoE could not read the stored \(providerName) API key. Approve the Keychain dialog (Always Allow) and send again, or re-enter the key in Settings ▸ Cloud Providers."
        case .credentialNeedsImport(let providerName):
            return "The \(providerName) API key is still in the older keychain. Open Settings ▸ Cloud Providers and choose Import stored credentials."
        case .unsupportedFlavor(let flavor):
            return "The \(flavor) API flavor is not implemented yet."
        case .httpStatus(let code, let summary):
            switch code {
            case 401, 403:
                return "Authentication failed (HTTP \(code)). Check the API key in Settings ▸ Cloud Providers."
            case 429:
                return "The provider is rate limiting requests (HTTP 429). Wait a moment, then send again."
            default:
                let detail = summary.trimmingCharacters(in: .whitespacesAndNewlines)
                return detail.isEmpty
                    ? "The provider returned HTTP \(code)."
                    : "The provider returned HTTP \(code): \(detail)"
            }
        case .transport(let message):
            return "Network error reaching the provider: \(message)"
        case .invalidResponse(let message):
            return message
        }
    }

    /// Whether a provider's 400/422 body is a context-window rejection rather
    /// than a schema or request-shape complaint. Providers word it differently,
    /// and the same status code covers "tools are unsupported", so the body has
    /// to decide which hint the chat surface shows.
    public nonisolated static func indicatesContextOverflow(_ summary: String) -> Bool {
        let lower = summary.lowercased()
        let markers = [
            "context_length", "context length", "maximum context", "context window",
            "too many tokens", "token limit", "prompt is too long", "input is too long",
            "input too long", "exceeds the maximum", "reduce the length"
        ]
        return markers.contains { lower.contains($0) }
    }

    /// Whether a provider's 400/422 body is complaining about the image content
    /// parts rather than about tools, schemas, or the context window. Registration
    /// of vision capability is an editorial fact (no catalog publishes
    /// modalities), so a model we marked capable can still refuse; the chat path
    /// uses this to un-mark it and retry the turn once without the pixels instead
    /// of failing the conversation.
    public nonisolated static func indicatesImageRejection(_ summary: String) -> Bool {
        let lower = summary.lowercased()
        // Deliberately narrow: these phrases name the image content part itself.
        // Broad words like "vision", "modality", or "content parts" also appear in
        // unrelated 400 bodies, and a false match persists `supportsVision = false`
        // for a working model, so they are not markers.
        let markers = [
            "image_url", "image url", "image input", "image content", "image part",
            "images are not", "does not support image", "doesn't support image",
            "unsupported image", "not support vision", "invalid image", "image format",
            "no image"
        ]
        return markers.contains { lower.contains($0) }
    }
}

// MARK: - Subagent Relay
/// The coordinator directive that carries completed subagent reports into a
/// cloud agent turn. The cloud path rebuilds its messages from the session on
/// every step, so unlike the weights path the reports cannot be appended to a
/// rendered prompt: they ride the leading system message instead, which
/// keeps the transcript free of synthetic user turns.
public nonisolated enum CloudSubagentRelay {
    public static func directive(reports: [String]) -> String {
        let context = reports.joined(separator: "\n\n")
        return """
        <system>
        A delegated subagent has finished its work. Its final report is below. You are the coordinator: the user only talks to you, and you are responsible for presenting subagent results.
        Using the report below, write your complete final answer to the user's original question NOW. Do not emit any tool calls.

        \(context)
        </system>
        """
    }
}

/// The tools-disabled final-answer pass a cloud agent run falls back to when
/// its loop guards trip (failed/repeated calls) or the step budget runs out.
/// Without it the run ends on an assistant turn whose content is empty beside
/// its tool records, so the user is left with no answer at all.
public nonisolated enum CloudAgentSynthesis {
    public static let directive = """
    <system>
    IMPORTANT: All tool use is now DISABLED for this task. Your recent tool calls failed, repeated, or exhausted the step budget and could not advance the task, so the agent loop was forcibly ended.
    Using ONLY the tool outputs already shown above in this conversation (plus your own knowledge), write your complete, self-contained final answer to the user's original question NOW.
    Do not emit any tool calls. Do not call another tool. Just answer.
    </system>
    """
}

// MARK: - Stream Chunks

/// One decoded `chat.completion.chunk` frame. Both `reasoning_content` (the
/// extension DynaMoE's own server emits) and `reasoning` (other providers'
/// spelling) are accepted for the thinking channel.
nonisolated struct CloudChatStreamChunk: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            struct ToolCall: Decodable {
                struct Function: Decodable {
                    let name: String?
                    let arguments: String?
                }

                let index: Int?
                let id: String?
                let function: Function?
            }

            let role: String?
            let content: String?
            let reasoning: String?
            let toolCalls: [ToolCall]?

            enum CodingKeys: String, CodingKey {
                case role, content, reasoning
                case reasoningContent = "reasoning_content"
                case toolCalls = "tool_calls"
            }

            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                self.role = try container.decodeIfPresent(String.self, forKey: .role)
                self.content = try container.decodeIfPresent(String.self, forKey: .content)
                self.reasoning = try container.decodeIfPresent(String.self, forKey: .reasoningContent)
                    ?? container.decodeIfPresent(String.self, forKey: .reasoning)
                self.toolCalls = try container.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
            }
        }

        let delta: Delta?
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case delta
            case finishReason = "finish_reason"
        }
    }

    struct WireError: Decodable {
        let message: String?
    }

    let choices: [Choice]?
    let usage: CloudUsage?
    let error: WireError?
}

// MARK: - Stream Parser

/// Accumulates SSE data payloads into turn text. One parser per turn; the
/// transport feeds it each dispatched `data:` line and relays whatever
/// deltas it returns. `[DONE]` detection and transport errors stay in the
/// transport, so this type remains pure.
nonisolated struct CloudStreamParser {
    private(set) var text: String = ""
    private(set) var thinking: String = ""
    private(set) var finishReason: String?
    private(set) var usage: CloudUsage?
    /// How many payloads carried something usable: a content or reasoning
    /// delta, a tool-call fragment, a finish reason, or usage. Zero after a
    /// clean stream end means the endpoint answered 200 with nothing usable —
    /// a case worth reporting instead of surfacing an empty reply. A chunk that
    /// merely decodes (`{"error":…}`, an unknown envelope) does not count.
    private(set) var decodedChunkCount: Int = 0

    /// Tool-call fragments accumulated by their stream `index`, plus the order
    /// in which each index first appeared (providers may repeat, omit, or
    /// interleave them; argument deltas are pure text to concatenate).
    private(set) var toolCalls: [CloudToolCall] = []
    private var toolCallOrder: [Int] = []
    private var toolCallIds: [Int: String] = [:]
    private var toolCallNames: [Int: String] = [:]
    private var toolCallArguments: [Int: String] = [:]

    init() {}

    /// One decoder per parser instead of one per SSE payload: a decode runs for
    /// every streamed token, and allocation showed up on that hot path.
    private let decoder = JSONDecoder()

    /// Decodes one payload and returns the deltas it introduced.
    mutating func consume(dataPayload: String) throws -> [CloudTurnEvent] {
        guard let data = dataPayload.data(using: .utf8) else {
            throw CloudTurnFailure.invalidResponse("The provider sent a response that was not valid UTF-8.")
        }
        let chunk: CloudChatStreamChunk
        do {
            chunk = try decoder.decode(CloudChatStreamChunk.self, from: data)
        } catch {
            throw CloudTurnFailure.invalidResponse(
                "The provider sent a response chunk DynaMoE could not decode."
            )
        }
        if let reported = chunk.error {
            let detail = reported.message?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            throw CloudTurnFailure.invalidResponse(
                detail.isEmpty
                    ? "The provider reported an error inside a 200 response."
                    : "The provider reported an error: \(detail)"
            )
        }

        var events: [CloudTurnEvent] = []
        if let delta = chunk.choices?.first?.delta {
            if let reasoning = delta.reasoning, !reasoning.isEmpty {
                thinking += reasoning
                events.append(.thinkingDelta(reasoning))
            }
            if let content = delta.content, !content.isEmpty {
                text += content
                events.append(.contentDelta(content))
            }
            if let fragments = delta.toolCalls, !fragments.isEmpty {
                absorb(fragments)
            }
        }
        var carriedPayload = !events.isEmpty || !(chunk.choices?.first?.delta?.toolCalls ?? []).isEmpty
        if let finish = chunk.choices?.first?.finishReason {
            finishReason = finish
            carriedPayload = true
        }
        if let reported = chunk.usage {
            usage = reported
            carriedPayload = true
        }
        if carriedPayload { decodedChunkCount += 1 }
        return events
    }

    /// Folds one `delta.tool_calls[]` batch into the accumulators. Each entry
    /// carries a stream `index`; the first fragment for an index supplies the
    /// `id` and often only part of `function.name`, and every later fragment
    /// appends to `function.arguments`.
    private mutating func absorb(_ fragments: [CloudChatStreamChunk.Choice.Delta.ToolCall]) {
        for fragment in fragments {
            let index = fragment.index ?? unindexedFragmentIndex(
                fragment,
                continuing: toolCallOrder.last ?? -1
            )
            guard index >= 0 else { continue }
            if !toolCallOrder.contains(index) { toolCallOrder.append(index) }
            if let id = fragment.id, !id.isEmpty { toolCallIds[index] = id }
            if let name = fragment.function?.name, !name.isEmpty {
                // Name fragments normally arrive once, but some providers repeat
                // the whole name on every fragment; appending blindly then yields
                // "file_readfile_read". An identical repeat is not a new fragment.
                let existing = toolCallNames[index] ?? ""
                if existing.isEmpty {
                    toolCallNames[index] = name
                } else if existing != name {
                    toolCallNames[index] = existing + name
                }
            }
            if let arguments = fragment.function?.arguments, !arguments.isEmpty {
                toolCallArguments[index, default: ""] += arguments
            }
        }
        rebuildToolCalls()
    }

    /// Resolves the stream index for a fragment the provider sent without one.
    ///
    /// An index-less fragment normally continues the call before it (some
    /// providers omit `index` on argument continuations). A non-empty `id` that
    /// differs from the previous call's id is a NEW call starting in parallel
    /// though: folding it into the previous accumulator would append one call's
    /// arguments to another and overwrite its id, producing a call that cannot
    /// be executed. A fresh index keeps both intact.
    private func unindexedFragmentIndex(
        _ fragment: CloudChatStreamChunk.Choice.Delta.ToolCall,
        continuing previousIndex: Int
    ) -> Int {
        guard previousIndex >= 0 else { return 0 }
        if let id = fragment.id,
           !id.isEmpty,
           let existing = toolCallIds[previousIndex],
           !existing.isEmpty,
           existing != id {
            // Above every index seen so far, so it can never collide with one
            // the provider does send explicitly.
            return (toolCallOrder.max() ?? previousIndex) + 1
        }
        return previousIndex
    }

    private mutating func rebuildToolCalls() {
        toolCalls = toolCallOrder.compactMap { index in
            let name = (toolCallNames[index] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            let id = toolCallIds[index] ?? CloudToolCall.synthesizedId(index: index)
            return CloudToolCall(
                id: id,
                name: name,
                argumentsJSON: toolCallArguments[index] ?? ""
            )
        }
    }

    func result() -> CloudTurnResult {
        CloudTurnResult(
            text: text,
            thinking: thinking,
            finishReason: finishReason,
            usage: usage,
            toolCalls: toolCalls
        )
    }
}

// MARK: - Tool Call Decoding

public extension CloudToolCall {
    /// Some OpenAI-compatible providers stream calls without an `id`. One is
    /// synthesized so the assistant call and its tool result can still be
    /// paired on the next request.
    nonisolated static func synthesizedId(index: Int) -> String {
        "call_cloud_\(index)"
    }

    /// Decodes the streamed `arguments` JSON object into the typed dictionary
    /// the harness executes. An empty string is a legitimate `{}` (a call with
    /// no parameters); anything malformed yields nil so the caller can report
    /// a broken call instead of executing it with guessed inputs.
    nonisolated var decodedArguments: [String: Any]? {
        let trimmed = argumentsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [:] }
        guard let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any] else { return nil }
        return dictionary
    }
}

// MARK: - Agent Loop Policy

/// The rules that decide whether a native-function-calling agent step runs
/// another turn and what that turn is allowed to do. Kept pure so termination
/// is testable without a provider: a step continues only when the model
/// actually asked for tools, nothing else ended the run (`complete`, a
/// detected failure loop, cancellation), and a step remains in the budget.
public nonisolated enum CloudAgentLoopPolicy {
    public static func shouldContinue(
        stepCallCount: Int,
        step: Int,
        maxSteps: Int,
        ranComplete: Bool,
        failureLoop: Bool,
        cancelled: Bool
    ) -> Bool {
        guard !cancelled, !ranComplete, !failureLoop, stepCallCount > 0 else { return false }
        return step + 1 < maxSteps
    }

    /// Whether the run must spend one final tools-disabled turn instead of
    /// ending on a tool-only assistant message. `failureLoop` is the guard
    /// that stopped it; `budgetExhausted` means it still wanted another tool
    /// round but had no steps left.
    public static func shouldForceSynthesis(
        failureLoop: Bool,
        budgetExhausted: Bool,
        ranComplete: Bool,
        alreadyForced: Bool
    ) -> Bool {
        guard !ranComplete, !alreadyForced else { return false }
        return failureLoop || budgetExhausted
    }
}

/// The per-step request/execution decisions the cloud agent loop makes. Both
/// used to be implicit in the loop body, where a tools-disabled synthesis turn
/// could still execute streamed calls the request never asked for.
public nonisolated enum CloudAgentStepPlan {
    /// Tools ride the request only on an agent step that is not the final
    /// synthesis pass.
    public static func offersTools(agentTurn: Bool, isSynthesisTurn: Bool) -> Bool {
        agentTurn && !isSynthesisTurn
    }

    /// A provider can stream `tool_calls` even when the request carried no
    /// `tools` array. Executing those would let a tools-disabled synthesis turn
    /// re-enter the loop, so calls only run when the step actually offered tools.
    public static func shouldExecuteToolCalls(offeredToolCount: Int, callCount: Int) -> Bool {
        offeredToolCount > 0 && callCount > 0
    }
}

// MARK: - Message Assembly

/// How much of a conversation's pictures one request carries.
///
/// A stateless API has to be told about an image on every step that should
/// still see it, and those tokens are billed every time. Attaching a picture
/// to one question and then asking ten follow-ups re-sends it ten times. The
/// cheaper policy sends the newest turn's pictures and replaces older ones
/// with a one-line notice, which is also prefix-stable: the choice depends on
/// the conversation, not on the step.
public nonisolated enum CloudImageDelivery: Equatable, Sendable {
    /// Every picture in the conversation rides every request (the default).
    case all
    /// Only the newest user turn's pictures ride; older ones become notices.
    case newestUserTurnOnly
}

/// Builds the outbound messages array from a conversation. The provider owns
/// chat templating, so history travels as messages; tool turns are replayed in
/// the OpenAI exchange shape (assistant `tool_calls` followed by one `tool`
/// message per result) because a conversation that loses that shape — on a
/// reload, or on the next agent step — leaves the model unable to pair a call
/// with its output. System-role messages are UI notices (the same rule the
/// Apple Foundation Model path applies) and the in-flight assistant
/// placeholder is excluded by id.
public nonisolated enum CloudMessageAssembler {
    public static func assemble(
        from messages: [ChatMessage],
        excludingMessageIds: Set<UUID> = [],
        imageDataURLs: [UUID: [String]] = [:],
        imagesSupported: Bool = true,
        imageDelivery: CloudImageDelivery = .all
    ) -> [CloudChatMessage] {
        // Resolved before the loop: "newest" must mean the newest in the
        // conversation, not the newest seen so far.
        var newestImageTurnId: UUID?
        if imageDelivery == .newestUserTurnOnly {
            newestImageTurnId = messages
                .last(where: { $0.role == .user && !($0.attachments ?? []).isEmpty })?
                .id
        }
        var assembled: [CloudChatMessage] = []
        for message in messages {
            if excludingMessageIds.contains(message.id) { continue }
            guard message.role != .system else { continue }
            let content = AppleFoundationModelService.stripLeadingSpeakerLabel(message.content)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .assistant, let records = message.toolCalls, !records.isEmpty {
                assembled.append(contentsOf: toolExchange(content: content, records: records))
                continue
            }
            let attachments = message.role == .user ? (message.attachments ?? []) : []
            let omittedByPolicy = newestImageTurnId != nil && message.id != newestImageTurnId
            let resolvedURLs = (attachments.isEmpty || omittedByPolicy)
                ? []
                : (imageDataURLs[message.id] ?? [])
            let (text, parts, descriptions) = attachmentCarryingContent(
                content: content,
                attachments: attachments,
                resolvedURLs: resolvedURLs,
                imagesSupported: imagesSupported,
                omittedForPolicy: omittedByPolicy
            )
            // A user turn that carries only images is still a real turn: sending
            // nothing for it would silently drop the picture from the history.
            guard !text.isEmpty || !parts.isEmpty else { continue }
            assembled.append(CloudChatMessage(
                role: message.role == .user ? "user" : "assistant",
                content: text.isEmpty ? nil : text,
                imageParts: parts.isEmpty ? nil : parts,
                imageDescriptions: descriptions.isEmpty ? nil : descriptions
            ))
        }
        return assembled
    }

    /// Splits one user turn into the text the model reads and the image parts it
    /// is sent. Attachments are re-sent with their original message on every
    /// step (history is rebuilt per step), so an agent run keeps the model's
    /// visual context instead of forgetting the picture after turn one.
    ///
    /// When the model cannot take images — or an asset is missing on disk, or
    /// the turn is older than the delivery policy allows — the turn carries a
    /// metadata-only notice instead of silently losing the picture: a model
    /// told "an image was attached" answers about it, while a model that never
    /// hears about it invents an answer about nothing.
    static func attachmentCarryingContent(
        content: String,
        attachments: [ImageAttachmentRef],
        resolvedURLs: [String],
        imagesSupported: Bool,
        omittedForPolicy: Bool = false
    ) -> (text: String, parts: [String], descriptions: [String]) {
        guard !attachments.isEmpty else { return (content, [], []) }
        let descriptions = attachments.map(\.logDescription)
        guard imagesSupported else {
            return (appendingNotice(
                omittedImagesNotice(count: attachments.count, reason: "model has no vision support"),
                to: content
            ), [], descriptions)
        }
        if omittedForPolicy {
            return (appendingNotice(
                omittedImagesNotice(count: attachments.count, reason: "older turn, not re-sent"),
                to: content
            ), [], descriptions)
        }
        // A partially resolved turn (one asset pruned while another survives)
        // must say which pictures are missing rather than silently sending fewer
        // images than the transcript shows.
        if resolvedURLs.count < attachments.count {
            return (appendingNotice(
                omittedImagesNotice(
                    count: attachments.count - resolvedURLs.count,
                    reason: "image data unavailable"
                ),
                to: content
            ), resolvedURLs, descriptions)
        }
        return (content, resolvedURLs, descriptions)
    }

    /// The text a turn carries in place of pixels it could not send.
    public static func omittedImagesNotice(count: Int, reason: String) -> String {
        let noun = count == 1 ? "image" : "images"
        return "[\(count) \(noun) omitted: \(reason)]"
    }

    private static func appendingNotice(_ notice: String, to content: String) -> String {
        content.isEmpty ? notice : content + "\n\n" + notice
    }

    /// One assistant turn that issued calls, followed by one `tool` message per
    /// call. Every call gets exactly one result — an unanswered `tool_calls`
    /// entry is a hard API error on the next request — so a record whose run
    /// never produced output (cancelled mid-flight) still reports its status.
    private static func toolExchange(content: String, records: [ToolCallRecord]) -> [CloudChatMessage] {
        let calls = records.map { record in
            CloudChatToolCall(
                id: wireCallId(for: record),
                name: record.name,
                arguments: wireArguments(for: record)
            )
        }
        var exchange: [CloudChatMessage] = [
            CloudChatMessage(
                role: "assistant",
                content: content.isEmpty ? nil : content,
                toolCalls: calls
            )
        ]
        for record in records {
            exchange.append(CloudChatMessage(
                role: "tool",
                content: toolResultContent(for: record),
                toolCallId: wireCallId(for: record)
            ))
        }
        return exchange
    }

    /// Stable per-record call id. `ToolCallRecord.id` is persisted, so a call
    /// and its result keep the same id across every re-render of the turn.
    public static func wireCallId(for record: ToolCallRecord) -> String {
        "call_" + record.id.uuidString.lowercased()
    }

    /// The arguments string replayed to the provider. The model's own raw JSON
    /// is preferred (it preserves numbers, booleans, and nesting); stored
    /// string arguments are only a fallback for records that lost it.
    public static func wireArguments(for record: ToolCallRecord) -> String {
        let raw = record.rawArguments.trimmingCharacters(in: .whitespacesAndNewlines)
        if !raw.isEmpty,
           let data = raw.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data)) is [String: Any] {
            return raw
        }
        guard !record.arguments.isEmpty else { return "{}" }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try? encoder.encode(record.arguments),
           let json = String(data: data, encoding: .utf8) {
            return json
        }
        return "{}"
    }

    /// What the model sees for one call's result.
    public static func toolResultContent(for record: ToolCallRecord) -> String {
        if let output = record.output, !output.isEmpty { return output }
        if let error = record.error, !error.isEmpty { return error }
        switch record.status {
        case .rejected: return "Action rejected by user."
        case .withheld: return "Withheld by the agent mode; this call was not executed."
        case .awaitingApproval: return "Awaiting approval."
        case .running: return "The call did not finish before the turn ended."
        case .error: return "The tool call failed."
        case .success: return "Completed."
        }
    }

    /// Character-based input estimate for the turn log (providers report real
    /// usage only in the final streamed chunk, which may never arrive). Images
    /// contribute a flat allowance rather than their base64 length: a data URL
    /// counted as text would report tens of thousands of phantom tokens.
    public static func approximatePromptTokens(system: String?, messages: [CloudChatMessage]) -> Int {
        let imageTokens = messages.reduce(0) { total, message in
            total + (message.imageParts?.count ?? 0) * ImageAttachmentPreparer.estimatedTokensPerImage
        }
        let characters = (system ?? "").count + messages.reduce(0) { total, message in
            total + (message.content ?? "").count
                + (message.toolCalls ?? []).reduce(0) { $0 + $1.function.name.count + $1.function.arguments.count }
        }
        return max(1, characters / 4 + imageTokens)
    }

    /// The prompt as the provider sees it, for the Settings ▸ Server log's
    /// request-body column.
    public static func renderedPromptForLog(system: String?, messages: [CloudChatMessage]) -> String {
        var blocks: [String] = []
        let persona = (system ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !persona.isEmpty { blocks.append(persona) }
        for message in messages {
            if message.role == "tool" {
                let content = (message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                blocks.append("Tool result: \(content)")
                continue
            }
            let rawContent = (message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            // Images are rendered as metadata only. A data URL here would put
            // megabytes of base64 into the request log's SQLite row.
            let imageLabels = message.imageDescriptions ?? []
            let content = ([rawContent] + imageLabels)
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
            let label: String
            switch message.role {
            case "assistant": label = "Assistant"
            case "system": label = "System"
            default: label = "User"
            }
            if let calls = message.toolCalls, !calls.isEmpty {
                let names = calls.map { "\($0.function.name)(\($0.function.arguments))" }
                    .joined(separator: ", ")
                blocks.append("\(label) [tool calls: \(names)]\(content.isEmpty ? "" : "\n\(content)")")
            } else if !content.isEmpty {
                blocks.append("\(label): \(content)")
            }
        }
        return blocks.joined(separator: "\n\n")
    }
}
