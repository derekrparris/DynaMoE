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
/// safe default: two custom `x-…` headers, which HTTP requires a server to
/// ignore when it does not know them, so under the protocol they change
/// nothing for an endpoint that does not route on them. A gateway that rejects
/// unrecognised headers anyway is not conforming, so the request path retries
/// once without them — but only when the rejection body NAMES a header, which
/// is the one signal that separates "we do not take this header" from every
/// other request-shape complaint. `none` is for an endpoint known to reject
/// them, and skips the wasted round trip. `openAI` adds `prompt_cache_key`,
/// the documented OpenAI routing hint, and is opt-in because a strict
/// compatible endpoint is allowed to reject unknown body fields (the request
/// path retries once without it if that happens).
public nonisolated enum CloudCacheHints: String, Codable, Equatable, Hashable, Sendable, CaseIterable {
    /// Send nothing. For an endpoint that is known to reject extra headers.
    case none
    /// Session-affinity headers only.
    case affinity
    /// Affinity headers plus `prompt_cache_key`.
    case openAI

    /// An unrecognised spelling decodes to the safe default rather than
    /// failing the whole provider record. The value lives in hand-editable
    /// JSON and a newer build may write one this build does not know; losing
    /// the provider from the list over a label is worse than falling back to
    /// headers every HTTP server is free to ignore. A value that is not even a
    /// string (`"cacheHints": 1`) takes the same path: refusing to decode it
    /// would drop the provider for a field the fallback already answers.
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = (try? container.decode(String.self)) ?? ""
        self = CloudCacheHints(rawValue: raw) ?? .affinity
    }

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
    /// (Anthropic-style providers bill this separately from a cache read,
    /// and at a premium the price list can record as `cache_write_input`).
    public let cacheWriteTokens: Int?

    /// Whether `promptTokens` was rebuilt from Anthropic's three-way split,
    /// i.e. whether the wire `input_tokens` column was the UNCACHED share
    /// rather than the prompt total.
    ///
    /// It describes how the block was DECODED, which is the one thing a stream
    /// holding several usage blocks has to know: a later block that repeats
    /// `input_tokens` without the cache columns cannot be compared with — or
    /// laid over — a total that included them (`merging`).
    var inputTokensExcludeCache: Bool

    /// Whether the prompt count came from the wire's `input_tokens` rather than
    /// from `prompt_tokens`. Both spellings exist, and only this one can be an
    /// uncached share; it is what lets `merging` recognise a later block that
    /// repeats the column without repeating the cache split.
    var inputTokensFromInputColumn: Bool

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
        self.inputTokensExcludeCache = false
        self.inputTokensFromInputColumn = false
    }

    /// Fraction of the prompt served from cache, 0...1, or nil when the
    /// provider reported no cache split at all. A reported zero is a real
    /// answer ("nothing was cached"); silence is not.
    public var cacheReadFraction: Double? {
        guard let promptTokens, promptTokens > 0, let cachedTokens else { return nil }
        return Double(min(cachedTokens, promptTokens)) / Double(promptTokens)
    }

    /// Equality is the NUMBERS, not the wire shape they arrived in.
    ///
    /// `inputTokensExcludeCache` and `inputTokensFromInputColumn` describe how a
    /// block was decoded, so two blocks carrying the same counts from different
    /// dialects are the same usage. The synthesized memberwise comparison made
    /// them unequal, which is a trap for the next test or dedupe that compares
    /// usage blocks instead of reading their columns.
    public static func == (lhs: CloudUsage, rhs: CloudUsage) -> Bool {
        lhs.promptTokens == rhs.promptTokens
            && lhs.completionTokens == rhs.completionTokens
            && lhs.totalTokens == rhs.totalTokens
            && lhs.cachedTokens == rhs.cachedTokens
            && lhs.cacheWriteTokens == rhs.cacheWriteTokens
    }

    /// This block with `newer`'s columns laid over it, field by field.
    ///
    /// A stream can report usage more than once, and the later block is not
    /// obliged to repeat every column: some endpoints send a running
    /// `output_tokens` mid-stream and the prompt count only at the end (or the
    /// other way round). Replacing the accumulated block with the newest one
    /// dropped the columns the newest omitted, which silently turned a
    /// measured prompt and its cache split into the app's estimate — exactly
    /// where the cache reporting matters. A nil column means "not reported
    /// here", so it inherits the earlier value instead of erasing it.
    ///
    /// The prompt count is the exception a nil check cannot cover. Anthropic's
    /// `message_start` carries the cache columns and its `message_delta` may
    /// repeat `input_tokens` alone, which decodes to the UNCACHED share; laying
    /// that over a total that included the cached and written tokens shrinks
    /// the prompt by exactly the cache split, so the billed total and the
    /// `cached` figure beside it stop adding up. A later block that names no
    /// cache columns therefore leaves an earlier three-way split's total alone.
    public func merging(_ newer: CloudUsage) -> CloudUsage {
        // A later block that repeats `input_tokens` without a non-zero cache
        // split (either omitting the cache keys or emitting them as zero)
        // cannot replace an earlier Anthropic total or zero out its split. An
        // OpenAI-shaped `prompt_tokens` IS the total and still wins, as does an
        // `input_tokens` block carrying a positive cache count.
        let newerRepeatsInputWithoutSplit = newer.inputTokensFromInputColumn
            && !newer.inputTokensExcludeCache
            && (newer.cachedTokens ?? 0) == 0
            && (newer.cacheWriteTokens ?? 0) == 0
        let keepsEarlierSplit = inputTokensExcludeCache && newerRepeatsInputWithoutSplit
        let keepsEarlierPrompt = newer.promptTokens == nil || keepsEarlierSplit
        var merged = CloudUsage(
            promptTokens: keepsEarlierPrompt ? promptTokens : newer.promptTokens,
            completionTokens: newer.completionTokens ?? completionTokens,
            totalTokens: newer.totalTokens ?? totalTokens,
            cachedTokens: keepsEarlierSplit ? (cachedTokens ?? newer.cachedTokens) : (newer.cachedTokens ?? cachedTokens),
            cacheWriteTokens: keepsEarlierSplit ? (cacheWriteTokens ?? newer.cacheWriteTokens) : (newer.cacheWriteTokens ?? cacheWriteTokens)
        )
        merged.inputTokensExcludeCache = keepsEarlierPrompt
            ? inputTokensExcludeCache
            : newer.inputTokensExcludeCache
        merged.inputTokensFromInputColumn = keepsEarlierPrompt
            ? inputTokensFromInputColumn
            : newer.inputTokensFromInputColumn
        return merged
    }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case inputTokensDetails = "input_tokens_details"
        case cachedTokens = "cached_tokens"
        case promptCacheHitTokens = "prompt_cache_hit_tokens"
        case cacheReadInputTokens = "cache_read_input_tokens"
        case cacheCreationInputTokens = "cache_creation_input_tokens"
        case cacheWriteTokens = "cache_write_tokens"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }

    private struct TokenDetails: Decodable {
        let cachedTokens: Int?

        enum CodingKeys: String, CodingKey { case cachedTokens = "cached_tokens" }

        /// The same tolerance as `firstInt`: one stringified number inside a
        /// detail object must not cost the whole nested block.
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let value = try? container.decodeIfPresent(Int.self, forKey: .cachedTokens) {
                cachedTokens = value
            } else if let text = try? container.decodeIfPresent(String.self, forKey: .cachedTokens) {
                cachedTokens = Int(text.trimmingCharacters(in: .whitespaces))
            } else if let number = try? container.decodeIfPresent(Double.self, forKey: .cachedTokens),
                      number.isFinite,
                      number.magnitude < 9e15 {
                cachedTokens = Int(number.rounded())
            } else {
                cachedTokens = nil
            }
        }
    }

    /// The first key that yields a count. A provider that spells a field
    /// differently, or stringifies its numbers, should not cost the whole
    /// usage block, so an unreadable key falls through to the next spelling
    /// instead of throwing. Numbers written as strings matter in practice:
    /// losing them marks a measured usage block as estimated.
    private static func firstInt(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ keys: [CodingKeys]
    ) -> Int? {
        for key in keys {
            if let value = try? container.decodeIfPresent(Int.self, forKey: key) {
                return value
            }
            if let text = try? container.decodeIfPresent(String.self, forKey: key),
               let value = Int(text.trimmingCharacters(in: .whitespaces)) {
                return value
            }
            if let number = try? container.decodeIfPresent(Double.self, forKey: key),
               number.isFinite,
               number.magnitude < 9e15 {
                return Int(number.rounded())
            }
        }
        return nil
    }

    /// Adds wire counts without trapping. A broken or hostile usage block can
    /// carry `Int.max` in every column; an overflow is clamped, never a crash.
    private static func summed(_ values: [Int]) -> Int {
        values.reduce(0) { total, value in
            let (sum, overflow) = total.addingReportingOverflow(max(0, value))
            return overflow ? Int.max : sum
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let details = try? container.decodeIfPresent(TokenDetails.self, forKey: .promptTokensDetails)
        let inputDetails = try? container.decodeIfPresent(TokenDetails.self, forKey: .inputTokensDetails)
        let openAIPrompt = Self.firstInt(container, [.promptTokens])
        let inputTokens = Self.firstInt(container, [.inputTokens])
        // The nested detail is the documented OpenAI spelling and wins over a
        // stray flat `cached_tokens`, which is a generic column some
        // compatible endpoints use for something else. DeepSeek's dedicated
        // top-level name still wins over both, and Anthropic's column is its
        // own protocol's only spelling. The Anthropic names are read once so
        // the dialect check below can ask what they SAY, not only whether the
        // keys exist.
        let anthropicCacheRead = Self.firstInt(container, [.cacheReadInputTokens])
        let anthropicCacheWrite = Self.firstInt(container, [.cacheCreationInputTokens])
        let cachedCandidates = [
            Self.firstInt(container, [.promptCacheHitTokens]),
            details?.cachedTokens,
            inputDetails?.cachedTokens,
            anthropicCacheRead,
            Self.firstInt(container, [.cachedTokens])
        ].compactMap { $0 }
        let cachedTokens = cachedCandidates.first(where: { $0 > 0 })
            ?? cachedCandidates.first(where: { $0 >= 0 })
            ?? cachedCandidates.first
        let cacheWriteCandidates = [
            anthropicCacheWrite,
            Self.firstInt(container, [.cacheWriteTokens])
        ].compactMap { $0 }
        let cacheWriteTokens = cacheWriteCandidates.first(where: { $0 > 0 })
            ?? cacheWriteCandidates.first(where: { $0 >= 0 })
            ?? cacheWriteCandidates.first
        // Anthropic splits the prompt three ways: `input_tokens` is only the
        // share that was neither read from nor written to its cache, and the
        // other two columns carry the rest. Its `usage` block is recognised by
        // those Anthropic-only cache keys, so a provider that publishes
        // `input_tokens` as the total (OpenAI's Responses shape) keeps that
        // meaning when no cache columns ride beside it. Normalising to the
        // total here is what lets `CloudPricing` subtract a cache read without
        // clamping the prompt to zero.
        //
        // The signal is a cache column with a NON-ZERO value, not the key's
        // presence. Some OpenAI-shaped gateways emit the Anthropic spelling as
        // an explicit zero while publishing `input_tokens` as the total;
        // treating that zero as proof normalised 1,000 (total) + 600 (cached
        // detail) + 0 into 1,600 and billed a prompt the provider never had.
        // A zero column also carries no split to normalise: the total is
        // already the answer. No transport this app speaks publishes
        // `input_tokens` as a total beside a POSITIVE Anthropic column:
        // Anthropic defines the field as the uncached share, and an
        // OpenAI-shaped response uses `prompt_tokens` for the total, which
        // wins above. A provider that ever does needs an explicit flavor
        // signal, not a second heuristic on which keys happen to be present.
        let usesAnthropicCacheColumns = openAIPrompt == nil
            && inputTokens != nil
            && ((anthropicCacheRead ?? 0) > 0 || (anthropicCacheWrite ?? 0) > 0)
        if usesAnthropicCacheColumns, let inputTokens {
            self.promptTokens = Self.summed([inputTokens, cachedTokens ?? 0, cacheWriteTokens ?? 0])
        } else {
            self.promptTokens = (openAIPrompt ?? inputTokens).map { max(0, $0) }
        }
        self.inputTokensExcludeCache = usesAnthropicCacheColumns
        self.inputTokensFromInputColumn = openAIPrompt == nil && inputTokens != nil
        self.completionTokens = Self.firstInt(container, [.completionTokens, .outputTokens])
            .map { max(0, $0) }
        self.totalTokens = Self.firstInt(container, [.totalTokens]).map { max(0, $0) }
        self.cachedTokens = cachedTokens.map { max(0, $0) }
        self.cacheWriteTokens = cacheWriteTokens.map { max(0, $0) }
    }

    /// Encodes the canonical OpenAI shape. Nothing in the app persists a usage
    /// block, so this exists only so the type stays round-trippable in tests.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(promptTokens, forKey: .promptTokens)
        try container.encodeIfPresent(completionTokens, forKey: .completionTokens)
        try container.encodeIfPresent(totalTokens, forKey: .totalTokens)
        if let cachedTokens {
            var details = container.nestedContainer(keyedBy: TokenDetails.CodingKeys.self, forKey: .promptTokensDetails)
            try details.encode(cachedTokens, forKey: .cachedTokens)
        }
        try container.encodeIfPresent(cacheWriteTokens, forKey: .cacheCreationInputTokens)
    }
}

/// One streamed turn's collected result.
///
/// `terminated` says the stream reached a marker the protocol defines as the
/// end of a completion — the `[DONE]` sentinel or a non-nil `finish_reason`.
/// A connection that closes cleanly in the middle of a reply looks identical
/// to a finished one at the transport level, and the difference matters where
/// the text is about to delete something: a summary committed from a partial
/// answer permanently removes the messages it never covered.
public nonisolated struct CloudTurnResult: Equatable, Sendable {
    public let text: String
    public let thinking: String
    public let finishReason: String?
    public let usage: CloudUsage?
    public let toolCalls: [CloudToolCall]
    /// Whether the stream ended with a terminal marker. Defaults to true so a
    /// hand-built result in a test or a non-streaming source is not reported
    /// as truncated by omission.
    public let terminated: Bool

    public init(
        text: String,
        thinking: String,
        finishReason: String?,
        usage: CloudUsage?,
        toolCalls: [CloudToolCall] = [],
        terminated: Bool = true
    ) {
        self.text = text
        self.thinking = thinking
        self.finishReason = finishReason
        self.usage = usage
        self.toolCalls = toolCalls
        self.terminated = terminated
    }
}

/// One collected text completion (compaction's summarizer passes). The usage
/// rides along because those requests are billed like any other: a summary
/// whose cost vanished would make the session's spend disagree with the
/// provider's invoice.
///
/// `finishReason` is the provider's own reason for stopping. It exists so a
/// caller can tell a finished summary from one that was cut off: `length` and
/// `max_tokens` mean the text ran out of budget, and a compaction that
/// committed it would delete messages the summary never covered.
/// `CloudFinishDisposition` classifies the reason; `finishedCleanly` is the
/// allowlist the destructive path checks.
public nonisolated struct CloudTextCompletion: Equatable, Sendable {
    public let text: String
    public let usage: CloudUsage?
    public let finishReason: String?

    public init(text: String, usage: CloudUsage?, finishReason: String? = nil) {
        self.text = text
        self.usage = usage
        self.finishReason = finishReason
    }

    /// Whether the provider stopped because the reply ran out of budget.
    public var stoppedAtOutputLimit: Bool {
        CloudFinishDisposition(finishReason: finishReason) == .outputLimit
    }

    /// Whether the reply is known to be the whole answer.
    ///
    /// Deliberately the ALLOWLIST, not "anything but `length`": only the
    /// spellings this app recognises as a finished reply count, and an unknown
    /// reason is treated as a cut-off. That direction is the safe one where the
    /// answer decides whether a compaction may delete messages — a dialect that
    /// says `MAX_TOKENS` or `content_filter` must not be read as "done" just
    /// because it is not one of the two words the app knows.
    ///
    /// A nil reason counts as complete: a stream only yields a result once it
    /// saw `[DONE]` or a `finish_reason`, so an absent reason means the
    /// protocol's own end marker arrived, which is the strongest completion
    /// signal there is (`CloudTurnResult.terminated`). A connection that closed
    /// with neither is refused before a completion is built.
    public var finishedCleanly: Bool {
        finishReason == nil || CloudFinishDisposition(finishReason: finishReason) == .complete
    }
}

/// What a provider's `finish_reason` says about the text it accompanied.
///
/// Both wire dialects are covered (`length`/`max_tokens`, and Anthropic's
/// `end_turn`/`stop_sequence`/`tool_use`), matched case-insensitively after
/// trimming, because a compatible endpoint is free to spell its own reason and
/// the old exact-match test read `MAX_TOKENS` as a finished reply.
public nonisolated enum CloudFinishDisposition: Equatable, Sendable {
    /// The reply ended because the model finished (or handed off to a tool).
    case complete
    /// The reply ran out of its output budget.
    case outputLimit
    /// The provider stopped the reply for its own reason, so the text is not
    /// the answer that was asked for.
    case halted
    /// A reason this app does not recognise.
    case unrecognized

    /// The reasons on both protocols that mean "the model finished".
    static let completeReasons: Set<String> = [
        "stop", "end_turn", "stop_sequence", "tool_calls", "tool_use", "function_call"
    ]

    /// The reasons that mean the reply was stopped before the conversation
    /// ended, for a reason other than the output budget.
    static let haltedReasons: Set<String> = [
        "content_filter", "refusal", "pause_turn"
    ]

    static let outputLimitReasons: Set<String> = ["length", "max_tokens"]

    public init(finishReason: String?) {
        guard let finishReason else {
            self = .unrecognized
            return
        }
        let normalized = finishReason.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if Self.completeReasons.contains(normalized) {
            self = .complete
        } else if Self.outputLimitReasons.contains(normalized) {
            self = .outputLimit
        } else if Self.haltedReasons.contains(normalized) {
            self = .halted
        } else {
            self = .unrecognized
        }
    }
}

/// Everything one streamed cloud turn can report to the chat surface.
public nonisolated enum CloudTurnEvent: Equatable, Sendable {
    case started
    case thinkingDelta(String)
    case contentDelta(String)
    case finished(CloudTurnResult)
    case failed(CloudTurnFailure)

    /// Whether this event carries a failure.
    ///
    /// The chat loop reads a failure even after Stop, because a failure is the
    /// only thing that can say the request never left the Mac — and the cancel
    /// path prices everything it cannot prove was unsent. Text arriving after
    /// Stop belongs to a turn nobody is waiting for.
    public var isFailure: Bool {
        if case .failed = self { return true }
        return false
    }

    /// Whether this event ends a turn: a completed result or a failure.
    ///
    /// The chat loop treats these as readable after Stop, because a completed
    /// result still carries the provider's measured usage and a failure still
    /// carries the send classification. Deltas that arrive after Stop are not.
    public var isTerminal: Bool {
        if case .finished = self { return true }
        return isFailure
    }
}

// MARK: - Turn receipt

/// What the transport concluded about one turn, readable after its stream ends.
///
/// Stop cancels the consumer task, and a cancelled `for await` over an
/// `AsyncStream` ends immediately with no terminal event of any kind — the
/// repo's own cancellation case pins that. So the chat loop's only evidence
/// that a request never reached the provider (a 4xx refusal, an offline send,
/// a local encode fault) could not arrive exactly when the user pressed Stop,
/// and the cancel path priced the request anyway. A result that raced Stop was
/// dropped the same way, losing the provider's measured usage and cache split.
///
/// The transport writes its last classification here, and the caller reads it
/// after the loop instead of waiting for a delivery that may never happen. One
/// receipt belongs to one turn; every field is lock-guarded because the
/// transport writes from its own detached task. `spentRequest` is tri-state on
/// purpose: nil means the transport never reached a decision (a scripted one
/// that only yields events), which leaves the caller its conservative default.
public nonisolated final class CloudTurnReceipt: @unchecked Sendable {
    private let lock = NSLock()
    private var spent: Bool?
    private var terminalResult: CloudTurnResult?
    private var terminalFailure: CloudTurnFailure?

    public init() {}

    /// Whether the request reached the provider's inference path, as far as
    /// the transport could tell. Nil when nothing decided it.
    public var spentRequest: Bool? {
        lock.lock()
        defer { lock.unlock() }
        return spent
    }

    /// The terminal result the provider produced, when it produced one before
    /// the stream was cut off. Present even when Stop discarded the event that
    /// carried it: the provider still measured the turn.
    public var result: CloudTurnResult? {
        lock.lock()
        defer { lock.unlock() }
        return terminalResult
    }

    /// The terminal failure the transport classified, when one arrived before
    /// the stream was cut off.
    public var failure: CloudTurnFailure? {
        lock.lock()
        defer { lock.unlock() }
        return terminalFailure
    }

    /// Records the transport's current answer. A later attempt or a terminal
    /// event can overwrite it; the last word describes the request the provider
    /// answered.
    func record(spent: Bool) {
        lock.lock()
        self.spent = spent
        lock.unlock()
    }

    /// Records a completed answer. A finished result is proof the prompt
    /// reached inference, so it also settles the billing question. Any failure
    /// an earlier attempt classified is cleared: a retry that succeeded ends
    /// the turn successfully, and a stale failure must not make the caller read
    /// a completed turn as an error.
    func record(result: CloudTurnResult) {
        lock.lock()
        spent = true
        terminalResult = result
        terminalFailure = nil
        lock.unlock()
    }

    /// Records a classified failure. `spentRequest` is the same distinction
    /// the event carries, so the receipt and the event cannot disagree.
    func record(failure: CloudTurnFailure) {
        lock.lock()
        spent = failure.spentRequest
        terminalFailure = failure
        lock.unlock()
    }
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
    /// A failure below HTTP: no connection, a connection that died, or a local
    /// fault before the request could be encoded. `billed` says whether the
    /// provider could have processed the prompt. A transport error is not
    /// evidence of a send — the app produced one for "offline" and for a
    /// `JSONEncoder` throw before URLSession was touched — so the flag travels
    /// with the failure instead of being inferred from its case.
    case transport(String, billed: Bool)
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
        case .transport(let message, let billed):
            // A failure the client classified as pre-send is not "reaching the
            // provider": the request never left the Mac, and saying otherwise
            // sends the reader looking at the wrong end of the connection.
            return billed
                ? "Network error reaching the provider: \(message)"
                : "The request could not be sent: \(message)"
        case .invalidResponse(let message):
            return message
        }
    }

    /// Whether this failure means the provider processed the prompt, and so
    /// billed for it.
    ///
    /// A missing or unreadable key, and a flavor with no implementation, fail
    /// before anything is built. A 4xx is a REJECTION: the provider refused the
    /// request without running the model — an expired key, an unknown model, a
    /// rate limit, a schema complaint — and none of those are invoiced. A 5xx
    /// and a malformed stream come back from a request the provider had
    /// already accepted, and a prompt that reached its inference path is
    /// charged whether or not the answer arrived. A transport failure carries
    /// its own answer because "the connection failed" can mean the request
    /// never left this Mac (`notConnectedToInternet`, a local encode throw) or
    /// that a request the provider had accepted died on the way back.
    ///
    /// The distinction is what stops the chat path reporting a spend for a turn
    /// the provider threw away — and, on the other side, from omitting the
    /// prompt of a turn it did process. It is the same posture the compaction
    /// path takes with `CloudCompactionFailure.spentRequest`.
    public var spentRequest: Bool {
        switch self {
        case .missingAPIKey, .credentialAccessDenied, .credentialNeedsImport, .unsupportedFlavor:
            return false
        case .httpStatus(let code, _):
            return code >= 500
        case .transport(_, let billed):
            return billed
        case .invalidResponse:
            return true
        }
    }

    /// Classifies an error thrown on the send path.
    ///
    /// The caller knows one thing this helper cannot: whether it had finished
    /// building and submitting the request. Before that point nothing left the
    /// Mac, so the request cannot have been billed; after it, only a failure
    /// that PROVES no connection was established is refunded. A timeout or a
    /// lost connection is billed, because both can happen after the provider
    /// accepted the prompt and under-reporting a real invoice is the error
    /// that cannot be corrected afterwards.
    public nonisolated static func transportFailure(
        _ error: Error,
        requestWasSent: Bool
    ) -> CloudTurnFailure {
        if let failure = error as? CloudTurnFailure { return failure }
        let billed = requestWasSent && !neverReachedProvider(error)
        return .transport(error.localizedDescription, billed: billed)
    }

    /// Whether a URL loading error happened before any HTTP request could be
    /// delivered: no route, no host, no DNS answer, no TLS session, or a
    /// connection policy that refused to open one at all. Each of these is a
    /// connection that never carried the prompt, so the provider had nothing to
    /// bill. Errors that can arrive after the request was sent (a timeout, a
    /// lost connection) are deliberately absent.
    ///
    /// The three policy codes — no cellular data, international roaming off,
    /// a call holding the connection — are pre-send on the same argument: the
    /// system declines to open a connection, so nothing reached the provider.
    /// Leaving them out priced each of those sends as if a prompt had arrived.
    nonisolated static func neverReachedProvider(_ error: Error) -> Bool {
        guard let urlError = error as? URLError else { return false }
        switch urlError.code {
        case .notConnectedToInternet,
             .cannotFindHost,
             .cannotConnectToHost,
             .dnsLookupFailed,
             .badURL,
             .unsupportedURL,
             .dataNotAllowed,
             .internationalRoamingOff,
             .callIsActive,
             .secureConnectionFailed,
             .serverCertificateUntrusted,
             .serverCertificateHasBadDate,
             .serverCertificateNotYetValid,
             .serverCertificateHasUnknownRoot,
             .clientCertificateRejected,
             .clientCertificateRequired,
             .appTransportSecurityRequiresSecureConnection:
            return true
        default:
            return false
        }
    }

    /// Whether any error a sent request can throw means it was billed.
    ///
    /// The summarizer path catches an arbitrary `Error`, not just this type:
    /// `completeText` reports a provider failure as `CloudTurnFailure` and an
    /// abandoned stream as `CancellationError`, and a pass that gave up
    /// mid-answer was still charged for its prompt. A raw URL loading error is
    /// classified the same way the transport classifies it — an offline or
    /// unroutable connection carried nothing — and everything else is treated
    /// as sent, because under-reporting a real invoice is the error that cannot
    /// be corrected afterwards.
    public nonisolated static func spentRequest(for error: Error) -> Bool {
        if let failure = error as? CloudTurnFailure { return failure.spentRequest }
        if let refusal = error as? CloudCompactionFailure { return refusal.spentRequest }
        // A `URLError` that proves the prompt never left the Mac was not
        // billed. This is the same predicate the transport uses, so a failure
        // that reached this path unclassified still gets the same answer.
        if let urlError = error as? URLError {
            return !neverReachedProvider(urlError)
        }
        return true
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
/// rendered prompt: they travel as a TRAILING user message
/// (`CloudStepPrompt.assemble`), which keeps the transcript free of synthetic
/// turns and still leaves the leading cached prefix byte-identical.
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

        enum CodingKeys: String, CodingKey {
            case message, detail, error
        }

        init(message: String?) {
            self.message = message
        }

        init(from decoder: Decoder) throws {
            if let single = try? decoder.singleValueContainer(),
               let text = try? single.decode(String.self) {
                self.message = text
                return
            }
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.message = (try? container.decodeIfPresent(String.self, forKey: .message))
                ?? (try? container.decodeIfPresent(String.self, forKey: .detail))
                ?? (try? container.decodeIfPresent(String.self, forKey: .error))
        }
    }

    let choices: [Choice]?
    let usage: CloudUsage?
    let error: WireError?

    enum CodingKeys: String, CodingKey {
        case choices, usage, error
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.choices = try container.decodeIfPresent([Choice].self, forKey: .choices)
        self.error = Self.decodeError(from: container)
        // Usage is also decoded leniently. It rides the last chunks of a stream,
        // and a provider that spells one column wrong (`"usage": "n/a"`, a
        // nested detail that is not an object) used to throw away the DELTAS in
        // the same chunk — the reply lost text because its token accounting was
        // malformed. An unreadable usage block costs the app's estimate and the
        // `(est.)` mark, nothing else.
        self.usage = (try? container.decodeIfPresent(CloudUsage.self, forKey: .usage)) ?? nil
    }

    private static func decodeError(from container: KeyedDecodingContainer<CodingKeys>) -> WireError? {
        guard container.contains(.error),
              (try? container.decodeNil(forKey: .error)) == false else {
            return nil
        }
        if let flag = try? container.decode(Bool.self, forKey: .error) {
            return flag ? WireError(message: nil) : nil
        }
        if let text = try? container.decode(String.self, forKey: .error) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : WireError(message: text)
        }
        if let wire = try? container.decode(WireError.self, forKey: .error) {
            return wire
        }
        if let number = try? container.decode(Double.self, forKey: .error), number.isFinite {
            return WireError(message: String(number))
        }
        return nil
    }
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
            // Merge rather than replace: a later chunk may carry only part of
            // the block (see `CloudUsage.merging`).
            usage = usage?.merging(reported) ?? reported
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

    /// The accumulated turn. `terminated` comes from the transport, which is the
    /// only layer that sees the `[DONE]` sentinel; it defaults to true so a
    /// parser-level caller that never looks at the flag reads what it expects.
    func result(terminated: Bool = true) -> CloudTurnResult {
        CloudTurnResult(
            text: text,
            thinking: thinking,
            finishReason: finishReason,
            usage: usage,
            toolCalls: toolCalls,
            terminated: terminated
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
    ///
    /// `streamTerminated` is the third condition, because a stream that closed
    /// without a terminal marker never proves it finished writing its call set.
    /// The parser hands back whole call objects either way, so a cut-off stream
    /// and a complete one are indistinguishable at that layer, and the calls
    /// this loop runs include irreversible writes. Refusing them costs a retry;
    /// running one the model had not finished can cost the file.
    public static func shouldExecuteToolCalls(
        offeredToolCount: Int,
        callCount: Int,
        streamTerminated: Bool
    ) -> Bool {
        offeredToolCount > 0 && callCount > 0 && streamTerminated
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

/// Builds one agent step's request. The ordering is the whole point: the
/// system message is the request's first token, so a mid-run coordinator
/// directive is APPENDED as its own user turn rather than merged into the
/// system prompt. Rewriting the prefix would throw away the provider's cached
/// prompt for this step and the one after it; a trailing user turn extends the
/// prefix instead.
public nonisolated enum CloudStepPrompt {
    public static func assemble(
        persona: String,
        history: [CloudChatMessage],
        directive: String
    ) -> [CloudChatMessage] {
        var outbound: [CloudChatMessage] = []
        if !persona.isEmpty {
            outbound.append(CloudChatMessage(role: "system", content: persona))
        }
        outbound.append(contentsOf: history)
        if !directive.isEmpty {
            outbound.append(CloudChatMessage(role: "user", content: directive))
        }
        return outbound
    }
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
            // The NEWEST user turn, not the newest image-bearing one. Selecting
            // an older picture because a later question carried no file defeats
            // the setting: the user asked for text-only follow-ups not to keep
            // paying for a picture they already stopped asking about.
            newestImageTurnId = messages.last(where: { $0.role == .user })?.id
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

    /// How many images the first request of a turn will actually carry, before
    /// any bytes are encoded. Mirrors `assemble`'s rules: attachments travel
    /// only on user turns, only when the model takes images, and under
    /// `.newestUserTurnOnly` only from the newest user turn. A pre-turn estimate
    /// then prices what will be sent instead of every attachment the transcript
    /// happens to hold.
    public static func carriedImageCount(
        from messages: [ChatMessage],
        delivery: CloudImageDelivery,
        imagesSupported: Bool
    ) -> Int {
        guard imagesSupported else { return 0 }
        let carrying: [ChatMessage]
        switch delivery {
        case .all:
            carrying = messages
        case .newestUserTurnOnly:
            carrying = messages.last(where: { $0.role == .user }).map { [$0] } ?? []
        }
        return carrying.reduce(0) { total, message in
            guard message.role == .user else { return total }
            return total + (message.attachments?.count ?? 0)
        }
    }

    /// Splits one user turn into the text the model reads and the image parts it
    /// is sent. History is rebuilt per step, so whether a historical turn's
    /// attachments ride again is decided by `CloudImageDelivery` (Request
    /// Behaviour → Re-send Images): `.all` keeps the model's visual context for
    /// the whole run, `.newestUserTurnOnly` sends only the newest user turn's
    /// pictures and turns older ones into the notice below rather than billing
    /// them on every step.
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
    ///
    /// `tools` is not decoration. The provider serialises every schema into the
    /// prompt, so a loaded agent toolset adds one to three thousand tokens to
    /// EVERY request of the loop, and a pre-flight estimate that left them out
    /// read a tool-heavy run at 80% of the window while the real prompt was
    /// already past it. The schemas are counted at three characters per token
    /// rather than the prose ratio below, because they are JSON: the same
    /// density that makes a code block cost more than its length suggests, and
    /// erring high is what keeps the pre-flight on the safe side of the wall.
    public static func approximatePromptTokens(
        system: String?,
        messages: [CloudChatMessage],
        tools: [ToolDefinition]? = nil
    ) -> Int {
        let imageTokens = messages.reduce(0) { total, message in
            total + (message.imageParts?.count ?? 0) * ImageAttachmentPreparer.estimatedTokensPerImage
        }
        let characters = (system ?? "").count + messages.reduce(0) { total, message in
            total + (message.content ?? "").count
                + (message.toolCalls ?? []).reduce(0) { $0 + $1.function.name.count + $1.function.arguments.count }
        }
        return max(1, characters / 4 + toolSchemaCharacters(tools) / 3 + imageTokens)
    }

    /// Characters the `tools` array occupies in the encoded request body. Zero
    /// for no tools, and zero rather than a guess if the schemas cannot be
    /// encoded: an unencodable tool set fails the request itself, so the
    /// estimate must not invent a size for it.
    static func toolSchemaCharacters(_ tools: [ToolDefinition]?) -> Int {
        guard let tools, !tools.isEmpty else { return 0 }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(tools) else { return 0 }
        return data.count
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
