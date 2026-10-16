//
//  AnthropicAdapter.swift
//  DynaMoE
//
//  Anthropic-compatible Messages protocol: Codable request/response types
//  plus the pure prompt-assembly logic that maps an Anthropic message list
//  onto the engine's per-family ChatPromptFormatter templating. Minimal pure
//  state — all networking lives in ServerController.
//
//  Protocol notes:
//  - `system` is a top-level field (string or text-block array), not a
//    message role; content blocks carry tool_use / tool_result turns.
//  - `max_tokens` is required by the spec; validation happens server-side.
//  - Extended sampling (`min_p`, `repetition_penalty`, `presence_penalty`)
//    is not part of the Anthropic wire format — the shared turn plan applies
//    engine defaults for those lanes.
//  - `thinking: {type: "enabled"}` is opt-in (Anthropic default), unlike the
//    OpenAI adapter's opt-out `enable_thinking`.
//

import Foundation
import Hummingbird

// MARK: - Request types

struct AnthropicMessagesRequest: Decodable {
    struct Message: Decodable {
        let role: String
        /// Bare string content, or the concatenated text blocks. Nil when the
        /// message carries only tool_use / tool_result blocks.
        let contentText: String?
        /// Reasoning a client echoed back in a `thinking` block. Clients that
        /// keep the block (SDKs do, once it is signed) let the next turn
        /// reproduce the pinned generation instead of only its visible tail.
        let reasoningText: String?
        let toolUses: [ToolUseIn]?
        let toolResults: [ToolResultIn]?

        init(role: String, contentText: String?, toolUses: [ToolUseIn]? = nil, toolResults: [ToolResultIn]? = nil, reasoningText: String? = nil) {
            self.role = role
            self.contentText = contentText
            self.reasoningText = reasoningText
            self.toolUses = toolUses
            self.toolResults = toolResults
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.role = try container.decode(String.self, forKey: .role)
            var text: String?
            var reasoning: String?
            var toolUses: [ToolUseIn] = []
            var toolResults: [ToolResultIn] = []
            if let s = try? container.decode(String.self, forKey: .content) {
                text = s.isEmpty ? nil : s
            } else if let blocks = try? container.decode([ContentBlockIn].self, forKey: .content) {
                let joined = blocks.compactMap(\.text).filter { !$0.isEmpty }.joined(separator: "\n")
                text = joined.isEmpty ? nil : joined
                let thoughts = blocks.compactMap(\.thinking).filter { !$0.isEmpty }.joined(separator: "\n")
                reasoning = thoughts.isEmpty ? nil : thoughts
                toolUses = blocks.compactMap(\.toolUse)
                toolResults = blocks.compactMap(\.toolResult)
            }
            self.contentText = text
            self.reasoningText = reasoning
            self.toolUses = toolUses.isEmpty ? nil : toolUses
            self.toolResults = toolResults.isEmpty ? nil : toolResults
        }

        enum CodingKeys: String, CodingKey { case role, content }
    }

    struct ContentBlockIn: Decodable {
        let text: String?
        let thinking: String?
        let toolUse: ToolUseIn?
        let toolResult: ToolResultIn?

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decodeIfPresent(String.self, forKey: .type) ?? ""
            self.text = type == "text" ? try container.decodeIfPresent(String.self, forKey: .text) : nil
            self.thinking = type == "thinking" ? try container.decodeIfPresent(String.self, forKey: .thinking) : nil
            self.toolUse = type == "tool_use" ? try ToolUseIn(from: decoder) : nil
            self.toolResult = type == "tool_result" ? try ToolResultIn(from: decoder) : nil
        }

        enum CodingKeys: String, CodingKey { case type, text, thinking }
    }

    struct ToolUseIn: Decodable {
        let id: String?
        let name: String
        /// Raw JSON input object; preserved as pre-encoded JSON text so the
        /// transcript can embed it verbatim.
        let inputJSON: String?

        enum CodingKeys: String, CodingKey { case id, name, input }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.id = try container.decodeIfPresent(String.self, forKey: .id)
            self.name = try container.decode(String.self, forKey: .name)
            if let input = try? container.decode(AnyCodable.self, forKey: .input) {
                self.inputJSON = try? String(data: JSONEncoder().encode(input), encoding: .utf8)
            } else {
                self.inputJSON = nil
            }
        }
    }

    struct ToolResultIn: Decodable {
        let toolUseId: String?
        /// String content, or the concatenated text blocks of the result.
        let content: String?

        enum CodingKeys: String, CodingKey { case toolUseId = "tool_use_id", content }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.toolUseId = try container.decodeIfPresent(String.self, forKey: .toolUseId)
            if let s = try? container.decode(String.self, forKey: .content) {
                self.content = s.isEmpty ? nil : s
            } else if let blocks = try? container.decode([ContentBlockIn].self, forKey: .content) {
                let joined = blocks.compactMap(\.text).filter { !$0.isEmpty }.joined(separator: "\n")
                self.content = joined.isEmpty ? nil : joined
            } else {
                self.content = nil
            }
        }
    }

    struct Tool: Decodable {
        let name: String
        let description: String?
        /// Raw `input_schema` JSON object, pre-encoded so we can embed it
        /// verbatim in the tool-declaration prompt block.
        let inputSchemaData: Data?

        enum CodingKeys: String, CodingKey { case name, description, inputSchema = "input_schema" }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.name = try container.decode(String.self, forKey: .name)
            self.description = try container.decodeIfPresent(String.self, forKey: .description)
            if let schema = try? container.decode(AnyCodable.self, forKey: .inputSchema) {
                self.inputSchemaData = try? JSONEncoder().encode(schema)
            } else {
                self.inputSchemaData = nil
            }
        }
    }

    struct ThinkingConfig: Decodable {
        let type: String?
        let budgetTokens: Int?

        enum CodingKeys: String, CodingKey {
            case type
            case budgetTokens = "budget_tokens"
        }
    }

    struct ToolChoice: Decodable {
        let type: String?
        let name: String?
    }

    /// Top-level `system` field: bare string or array of text blocks.
    let systemText: String?
    let model: String?
    let messages: [Message]
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    let stopSequences: [String]?
    let stream: Bool?
    let tools: [Tool]?
    let toolChoice: ToolChoice?
    let thinking: ThinkingConfig?

    enum CodingKeys: String, CodingKey {
        case system, model, messages, stream, tools, thinking
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case stopSequences = "stop_sequences"
        case toolChoice = "tool_choice"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let s = try? container.decode(String.self, forKey: .system) {
            self.systemText = s.isEmpty ? nil : s
        } else if let blocks = try? container.decode([ContentBlockIn].self, forKey: .system) {
            let joined = blocks.compactMap(\.text).filter { !$0.isEmpty }.joined(separator: "\n\n")
            self.systemText = joined.isEmpty ? nil : joined
        } else {
            self.systemText = nil
        }
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
        self.messages = try container.decode([Message].self, forKey: .messages)
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        self.topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        self.stopSequences = try container.decodeIfPresent([String].self, forKey: .stopSequences)
        self.stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        self.tools = try container.decodeIfPresent([Tool].self, forKey: .tools)
        // Keep the whole choice, not just its `type`: a forced `{"type":"tool",
        // "name":"get_weather"}` must be able to name the tool it demands
        // (silently dropping the name would make it behave like `auto`).
        self.toolChoice = try? container.decode(ToolChoice.self, forKey: .toolChoice)
        self.thinking = try container.decodeIfPresent(ThinkingConfig.self, forKey: .thinking)
    }

    /// The `tool_choice.type` value, for callers (and tests) that only need the
    /// mode rather than the full choice.
    var toolChoiceType: String? { toolChoice?.type }
}

// MARK: - Response types

struct AnthropicUsage: Encodable {
    let inputTokens: Int
    let outputTokens: Int

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }
}

/// A single assistant content block in the non-streaming response.
enum AnthropicContentBlock: Encodable {
    case text(String)
    case thinking(String)
    case toolUse(id: String, name: String, inputJSON: String)

    enum CodingKeys: String, CodingKey { case type, text, thinking, signature, id, name, input }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .thinking(let thinking):
            try container.encode("thinking", forKey: .type)
            try container.encode(thinking, forKey: .thinking)
            // Anthropic clients treat `signature` as an opaque provider token,
            // echo it back verbatim, and drop unsigned thinking blocks. Without
            // it a client's reasoning never survives the round trip.
            try container.encode(AnthropicAdapter.thinkingSignature(for: thinking), forKey: .signature)
        case .toolUse(let id, let name, let inputJSON):
            try container.encode("tool_use", forKey: .type)
            try container.encode(id, forKey: .id)
            try container.encode(name, forKey: .name)
            let input: [String: AnyCodable]
            if let data = inputJSON.data(using: .utf8),
               let decoded = try? JSONDecoder().decode([String: AnyCodable].self, from: data) {
                input = decoded
            } else {
                input = [:]
            }
            try container.encode(input, forKey: .input)
        }
    }
}

struct AnthropicMessageResponse: ResponseEncodable {
    let id: String
    let type: String = "message"
    let role: String = "assistant"
    let model: String
    let content: [AnthropicContentBlock]
    let stopReason: String?
    let stopSequence: String?
    let usage: AnthropicUsage

    enum CodingKeys: String, CodingKey {
        case id, type, role, model, content, usage
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
    }
}

struct AnthropicErrorEvent: Encodable {
    struct Body: Encodable {
        let type: String
        let message: String
    }
    let type: String = "error"
    let error: Body
}

// MARK: - Streaming event payloads

/// The `message` object embedded in message_start.
struct AnthropicStreamMessageBase: Encodable {
    let id: String
    let type: String = "message"
    let role: String = "assistant"
    let model: String
    let content: [AnthropicContentBlock] = []
    let stopReason: String? = nil
    let stopSequence: String? = nil
    let usage: AnthropicUsage

    enum CodingKeys: String, CodingKey {
        case id, type, role, model, content, usage
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
    }
}

struct AnthropicMessageStartEvent: Encodable {
    let type: String = "message_start"
    let message: AnthropicStreamMessageBase
}

/// The `content_block` object embedded in content_block_start.
enum AnthropicStreamStartBlock: Encodable {
    case text
    case thinking
    case toolUse(id: String, name: String)

    enum CodingKeys: String, CodingKey { case type, text, thinking, id, name, input }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text:
            try container.encode("text", forKey: .type)
            try container.encode("", forKey: .text)
        case .thinking:
            try container.encode("thinking", forKey: .type)
            try container.encode("", forKey: .thinking)
        case .toolUse(let id, let name):
            try container.encode("tool_use", forKey: .type)
            try container.encode(id, forKey: .id)
            try container.encode(name, forKey: .name)
            try container.encode([String: String](), forKey: .input)
        }
    }
}

struct AnthropicContentBlockStartEvent: Encodable {
    let type: String = "content_block_start"
    let index: Int
    let contentBlock: AnthropicStreamStartBlock

    enum CodingKeys: String, CodingKey {
        case type, index
        case contentBlock = "content_block"
    }
}

enum AnthropicStreamDelta: Encodable {
    case text(String)
    case thinking(String)
    case signature(String)
    case inputJSON(String)

    enum CodingKeys: String, CodingKey { case type, text, thinking, signature, partialJSON = "partial_json" }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let delta):
            try container.encode("text_delta", forKey: .type)
            try container.encode(delta, forKey: .text)
        case .thinking(let delta):
            try container.encode("thinking_delta", forKey: .type)
            try container.encode(delta, forKey: .thinking)
        case .signature(let signature):
            try container.encode("signature_delta", forKey: .type)
            try container.encode(signature, forKey: .signature)
        case .inputJSON(let json):
            try container.encode("input_json_delta", forKey: .type)
            try container.encode(json, forKey: .partialJSON)
        }
    }
}

struct AnthropicContentBlockDeltaEvent: Encodable {
    let type: String = "content_block_delta"
    let index: Int
    let delta: AnthropicStreamDelta
}

struct AnthropicContentBlockStopEvent: Encodable {
    let type: String = "content_block_stop"
    let index: Int
}

struct AnthropicMessageDeltaEvent: Encodable {
    struct StopDelta: Encodable {
        let stopReason: String?
        let stopSequence: String?

        enum CodingKeys: String, CodingKey {
            case stopReason = "stop_reason"
            case stopSequence = "stop_sequence"
        }
    }
    struct OutputUsage: Encodable {
        let outputTokens: Int

        enum CodingKeys: String, CodingKey {
            case outputTokens = "output_tokens"
        }
    }
    let type: String = "message_delta"
    let delta: StopDelta
    let usage: OutputUsage
}

struct AnthropicMessageStopEvent: Encodable {
    let type: String = "message_stop"
}

struct AnthropicPingEvent: Encodable {
    let type: String = "ping"
}

// MARK: - Pure protocol mapping

/// Pure protocol-mapping logic for the Anthropic adapter. Every function is a
/// free static so it is unit-testable without a live engine or socket.
enum AnthropicAdapter {

    /// Opaque, deterministic signature for a thinking block. Anthropic treats
    /// `signature` as provider-private: clients echo it back verbatim when they
    /// return the assistant turn, and SDKs drop unsigned thinking blocks — so
    /// without one a client's reasoning never survives the round trip, and the
    /// pinned prompt+generation cannot be reproduced on the next turn. Stable
    /// over the block's text so a re-sent turn stays self-consistent.
    static func thinkingSignature(for text: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in text.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x100000001b3
        }
        var digest = [UInt8]()
        digest.reserveCapacity(24)
        var current = hash
        for _ in 0..<3 {
            for shift in stride(from: 56, through: 0, by: -8) {
                digest.append(UInt8(truncatingIfNeeded: current >> UInt64(shift)))
            }
            current = current &* 0x100000001b3 ^ 0x9e3779b97f4a7c15
        }
        return Data(digest).base64EncodedString()
    }

    /// Whether tools may be offered for this request. Anthropic's forced
    /// choices (`any`, or a named `tool`) demand a generation constraint this
    /// server cannot enforce; silently treating them as `auto` would let the
    /// model answer with a different tool or no tool at all. Reject them
    /// explicitly instead.
    static func toolUseEnabled(_ choice: AnthropicMessagesRequest.ToolChoice?) throws -> Bool {
        guard let type = choice?.type else { return true }
        switch type {
        case "auto":
            return true
        case "none":
            return false
        case "any":
            throw ServerAPIError.badRequest("tool_choice type \"any\" is not supported: this server cannot force a tool call. Use \"auto\" or \"none\".")
        case "tool":
            throw ServerAPIError.badRequest("tool_choice for a specific tool (\"\(choice?.name ?? "")\") is not supported: this server cannot force a tool call. Use \"auto\" or \"none\".")
        default:
            return true
        }
    }

    /// Maps the engine's terminal reasons onto Anthropic stop_reason values.
    static func finishReasonString(_ reason: TurnFinishReason, hasToolCalls: Bool) -> String? {
        switch reason {
        case .lengthCapped:
            return "max_tokens"
        case .stoppedBySequence:
            return hasToolCalls ? "tool_use" : "stop_sequence"
        case .toolFreeze, .eos, .endTag, .degenerateCycle, .stopped:
            return hasToolCalls ? "tool_use" : "end_turn"
        }
    }

    /// Renders the Anthropic tool schemas as a compact declaration block that
    /// is appended to the effective system prompt. Same shape as the OpenAI
    /// adapter's block (a JSON array of function definitions) so
    /// grammar-masked tool-call models recognize the vocabulary.
    ///
    /// The instruction teaches the engine's canonical tagged call markup, not
    /// bare JSON: the bare-JSON fallback only recognizes built-in tool names,
    /// so a client-defined tool emitted as bare JSON would be returned as
    /// ordinary text. The tagged markup parses for any tool name and is the
    /// exact shape the grammar masks enforce, so constrained sampling stays
    /// compatible.
    static func toolDeclarationText(_ tools: [AnthropicMessagesRequest.Tool]) -> String? {
        guard !tools.isEmpty else { return nil }
        var definitions: [[String: String]] = []
        for tool in tools {
            var def: [String: String] = ["name": tool.name]
            if let desc = tool.description, !desc.isEmpty {
                def["description"] = desc
            }
            if let schema = tool.inputSchemaData,
               let schemaJSON = String(data: schema, encoding: .utf8) {
                def["parameters"] = schemaJSON
            }
            definitions.append(def)
        }
        guard let data = try? JSONSerialization.data(withJSONObject: definitions, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else { return nil }
        let open = StreamingToolParser.qwenToolCallOpen
        let close = StreamingToolParser.qwenToolCallClose
        let fnOpen = StreamingToolParser.qwenFunctionOpen
        let fnClose = StreamingToolParser.qwenFunctionClose
        let paramOpen = "<parameter="
        let paramClose = "</parameter>"
        return """
        You have access to the following functions. When you need to use one, emit a single call in the tagged form below, then stop; the client will execute the call and send the result back before your next turn. Do not fabricate results.

        \(open)\(fnOpen)FUNCTION_NAME>\(paramOpen)ARG_NAME>ARG_VALUE\(paramClose)\(fnClose)\(close)

        Wrap each argument as \(paramOpen)ARG_NAME>ARG_VALUE\(paramClose); omit the parameter markup entirely for a call with no arguments. Emit the call alone, with no extra prose after the closing tag.

        FUNCTIONS: \(json)
        """
    }

    /// Rejects a request that ends with an assistant message carrying text or
    /// reasoning. Anthropic reads that as a response prefill — "start the answer
    /// with this" — but the engine generates a complete assistant turn from its
    /// own prompt prefix and the server reports only what was generated, so
    /// honoring it silently would return a second answer rather than a
    /// continuation of the supplied text. Rejecting names the limitation where
    /// the client can see it; a trailing assistant message with no content of
    /// its own (tool calls, or an empty block) is just the usual generation
    /// request and stays accepted.
    static func rejectUnsupportedPrefill(_ messages: [AnthropicMessagesRequest.Message]) throws {
        guard let last = messages.last, last.role == "assistant" else { return }
        let prefilled = [last.contentText, last.reasoningText]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .contains { !$0.isEmpty }
        guard prefilled else { return }
        throw ServerAPIError.badRequest(
            "A request may not end with an assistant message that carries text: this server does not support response prefills (the continuation cannot be reported as an Anthropic message). End the request with a user message."
        )
    }

    /// Renders the message list into role-tagged turns. The Anthropic system
    /// prompt lives in the top-level `system` field, so only the conversation
    /// body is assembled here.
    ///
    /// History stays role-structured (rather than being flattened into one
    /// user block with "User:"/"Assistant:" labels) so the server can render it
    /// with the model family's native role boundaries. A flattened transcript
    /// re-tokenizes the whole conversation every turn, so each turn's prompt
    /// can no longer extend the previous turn's pinned prompt+generation and
    /// multi-turn requests fully re-prefill.
    static func splitMessages(_ messages: [AnthropicMessagesRequest.Message]) throws -> [ChatPromptFormatter.ConversationTurn] {
        guard !messages.isEmpty else {
            throw ServerAPIError.badRequest("messages must contain at least one message")
        }
        let invalidRole = messages.first { $0.role != "user" && $0.role != "assistant" }?.role
        if let invalidRole = invalidRole {
            throw ServerAPIError.badRequest("messages roles must be \"user\" or \"assistant\", got \"\(invalidRole)\"")
        }
        try rejectUnsupportedPrefill(messages)

        var turns: [ChatPromptFormatter.ConversationTurn] = []
        for message in messages {
            switch message.role {
            case "user":
                var parts: [String] = []
                if let text = message.contentText, !text.isEmpty {
                    parts.append(text)
                }
                for result in message.toolResults ?? [] {
                    let label = result.toolUseId ?? "tool"
                    parts.append("Tool result (\(label)):\n\(result.content ?? "(no result)")")
                }
                let body = parts.joined(separator: "\n\n")
                turns.append(.init(role: .user, text: body.isEmpty ? "(no content)" : body))
            case "assistant":
                var parts: [String] = []
                // Reasoning a client echoed back is re-rendered ahead of the
                // answer, closed with the canonical tag, so the turn reproduces
                // what was generated rather than only its visible tail.
                if let reasoning = message.reasoningText?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !reasoning.isEmpty {
                    parts.append(reasoning + TurnTextSplitter.canonicalReasoningCloseTag)
                }
                if let text = message.contentText, !text.isEmpty {
                    parts.append(text)
                }
                if let uses = message.toolUses, !uses.isEmpty {
                    let rendered = uses.map { use in
                        var obj: [String: String] = ["name": use.name]
                        // Keep the opaque tool_use id: the tool_result turns
                        // below reference it, and with several calls (or
                        // reordered results) the id is the only thing tying a
                        // result back to the call it answers.
                        if let id = use.id, !id.isEmpty {
                            obj["id"] = id
                        }
                        if let input = use.inputJSON, !input.isEmpty {
                            obj["arguments"] = input
                        }
                        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
                           let json = String(data: data, encoding: .utf8) {
                            return json
                        }
                        return "{\"name\": \"\(use.name)\"}"
                    }
                    .joined(separator: "\n")
                    parts.append(ChatPromptFormatter.anthropicToolCallEchoPrefix + "\n" + rendered)
                }
                let body = parts.joined(separator: "\n\n")
                turns.append(.init(role: .assistant, text: body.isEmpty ? "(no content)" : body))
            default:
                if let content = message.contentText, !content.isEmpty {
                    turns.append(.init(role: .user, text: "\(message.role.capitalized): \(content)"))
                }
            }
        }
        guard !turns.isEmpty else {
            throw ServerAPIError.badRequest("messages must contain at least one message")
        }
        return turns
    }

    /// Stable prefix-cache session key, mirroring the OpenAI adapter scheme:
    /// identity is anchored on the conversation's FIRST USER message
    /// (stateless clients resend the full history, and the opening message
    /// never changes as a conversation grows) plus the model. Deterministic
    /// UUID (FNV-1a over two lanes); pure and testable. Two conversations that
    /// open identically still collide, so reuse is additionally gated on exact
    /// token-prefix equality inside PrefixCacheManager.findCommonPrefix and on
    /// `ServerTurnMemory` re-validating the rendered history.
    static func sessionKey(for messages: [AnthropicMessagesRequest.Message], model: String?) -> UUID {
        var lane0: UInt64 = 0xcbf29ce484222325
        var lane1: UInt64 = 0x9e3779b97f4a7c15
        func mix(_ byte: UInt8) {
            lane0 = (lane0 ^ UInt64(byte)) &* 0x100000001b3
            lane1 = (lane1 ^ UInt64(byte)) &* 0x00000100000001b3
            lane1 = (lane1 << 7) | (lane1 >> 57)
        }
        func mixString(_ s: String) {
            for byte in s.utf8 { mix(byte) }
            mix(0x1f)
        }
        mixString(model ?? "")
        if let anchor = messages.first(where: { $0.role == "user" }) ?? messages.first {
            mixString(anchor.role)
            mixString(anchor.contentText ?? "")
            for use in anchor.toolUses ?? [] {
                mixString("call:" + use.name)
                mixString(use.inputJSON ?? "")
            }
            for result in anchor.toolResults ?? [] {
                mixString("result:" + (result.toolUseId ?? ""))
                mixString(result.content ?? "")
            }
        }

        var bytes = [UInt8](repeating: 0, count: 16)
        withUnsafeMutableBytes(of: &lane0) { bytes.replaceSubrange(0..<8, with: $0) }
        withUnsafeMutableBytes(of: &lane1) { bytes.replaceSubrange(8..<16, with: $0) }
        // RFC 4122 v4 bits so downstream UUID consumers parse it cleanly.
        bytes[6] = (bytes[6] & 0x0f) | 0x40
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        return UUID(uuid: (bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]))
    }
}
