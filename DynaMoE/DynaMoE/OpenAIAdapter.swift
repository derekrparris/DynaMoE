//
//  OpenAIAdapter.swift
//  DynaMoE
//
//  OpenAI-compatible chat completion protocol: Codable request/response
//  types plus the pure prompt-assembly logic that maps an OpenAI message
//  list onto the engine's per-family ChatPromptFormatter templating.
//  Minimal pure state — all networking lives in ServerController.
//
//  Protocol notes:
//  - `top_k` / `min_p` / `repetition_penalty` / `presence_penalty` follow the
//    llama.cpp server extras so local clients (Crush, opencode, Continue) can
//    reach the engine's full sampler.
//  - `enable_thinking` follows the llama.cpp chat-template-kwargs convention.
//

import Foundation
import CryptoKit
import Hummingbird

// MARK: - Request types

struct OpenAIChatRequest: Decodable {
    struct Message: Decodable {
        let role: String
        /// String form, or the concatenated text parts when the client sends
        /// the multi-part content array. Nil for assistant tool-call turns.
        let contentText: String?
        /// Reasoning a client echoed back (the `reasoning_content` extension
        /// this server emits). Clients that keep it let the next turn
        /// reproduce the pinned generation instead of only its visible tail.
        let reasoningContent: String?
        let name: String?
        let toolCalls: [AssistantToolCallIn]?
        let toolCallId: String?

        enum CodingKeys: String, CodingKey {
            case role, content, name
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
            case toolCallId = "tool_call_id"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.role = try container.decode(String.self, forKey: .role)
            self.name = try container.decodeIfPresent(String.self, forKey: .name)
            self.toolCalls = try container.decodeIfPresent([AssistantToolCallIn].self, forKey: .toolCalls)
            self.toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
            self.reasoningContent = try container.decodeIfPresent(String.self, forKey: .reasoningContent)
            if let s = try? container.decode(String.self, forKey: .content) {
                self.contentText = s
            } else if let parts = try? container.decode([ContentPart].self, forKey: .content) {
                let joined = parts.compactMap(\.text).filter { !$0.isEmpty }.joined(separator: "\n")
                self.contentText = joined.isEmpty ? nil : joined
            } else {
                self.contentText = nil
            }
        }

        /// Convenience constructor for tests and internal assembly.
        init(role: String, contentText: String?, name: String? = nil, toolCalls: [AssistantToolCallIn]? = nil, toolCallId: String? = nil, reasoningContent: String? = nil) {
            self.role = role
            self.contentText = contentText
            self.reasoningContent = reasoningContent
            self.name = name
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
        }
    }

    struct ContentPart: Decodable {
        let type: String?
        let text: String?
    }

    struct AssistantToolCallIn: Decodable {
        struct Fn: Decodable {
            let name: String
            let arguments: String?
        }
        let id: String?
        let type: String?
        let function: Fn
    }

    struct Tool: Decodable {
        struct Function: Decodable {
            let name: String
            let description: String?
            /// Raw JSON schema object; preserved as pre-encoded JSON text so we
            /// can embed it verbatim in the tool-declaration prompt block.
            let parametersData: Data?
            enum CodingKeys: String, CodingKey { case name, description, parameters }
            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                self.name = try container.decode(String.self, forKey: .name)
                self.description = try container.decodeIfPresent(String.self, forKey: .description)
                if let params = try? container.decode(AnyCodable.self, forKey: .parameters) {
                    self.parametersData = try? JSONEncoder().encode(params)
                } else {
                    self.parametersData = nil
                }
            }
        }
        let type: String?
        let function: Function
    }

    /// `tool_choice` accepts either a bare string (`"none"` / `"auto"` /
    /// `"required"`) or a `{"type":"function","function":{"name":"..."}}`
    /// object. Both forms are decoded so a valid named-function request is not
    /// rejected as an invalid body; the mode is honored (or explicitly
    /// rejected) in `OpenAIAdapter.toolUseEnabled`.
    enum ToolChoice: Decodable, Equatable {
        case auto
        case none
        case required
        case function(name: String)

        private enum CodingKeys: String, CodingKey { case type, function }
        private enum FunctionKeys: String, CodingKey { case name }

        init(from decoder: Decoder) throws {
            if let container = try? decoder.singleValueContainer(),
               let raw = try? container.decode(String.self) {
                switch raw {
                case "none": self = .none
                case "required": self = .required
                default: self = .auto
                }
                return
            }
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decodeIfPresent(String.self, forKey: .type)
            if type == "function",
               let fn = try? container.nestedContainer(keyedBy: FunctionKeys.self, forKey: .function),
               let name = try? fn.decode(String.self, forKey: .name) {
                self = .function(name: name)
            } else {
                self = .auto
            }
        }
    }

    let model: String?
    let messages: [Message]
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    let minP: Float?
    let repetitionPenalty: Float?
    let presencePenalty: Float?
    let maxTokens: Int?
    let stream: Bool?
    let stop: [String]?
    let tools: [Tool]?
    let toolChoice: ToolChoice?
    let enableThinking: Bool?
    let reasoningEffort: String?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, stop, tools
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case maxTokens = "max_tokens"
        case toolChoice = "tool_choice"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
        self.messages = try container.decode([Message].self, forKey: .messages)
        self.temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        self.topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        self.minP = try container.decodeIfPresent(Float.self, forKey: .minP)
        self.repetitionPenalty = try container.decodeIfPresent(Float.self, forKey: .repetitionPenalty)
        self.presencePenalty = try container.decodeIfPresent(Float.self, forKey: .presencePenalty)
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        // OpenAI allows `stop` as a bare string or an array.
        if let single = try? container.decode(String.self, forKey: .stop) {
            self.stop = [single]
        } else {
            self.stop = try container.decodeIfPresent([String].self, forKey: .stop)
        }
        self.tools = try container.decodeIfPresent([Tool].self, forKey: .tools)
        // OpenAI allows a bare string or an object-valued tool_choice; decoding
        // only the string form rejected valid named-function requests as
        // invalid bodies before generation.
        self.toolChoice = try container.decodeIfPresent(ToolChoice.self, forKey: .toolChoice)
        self.enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking)
        self.reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
    }
}

// MARK: - Response types

struct OpenAIUsage: Encodable {
    let promptTokens: Int
    let completionTokens: Int
    var totalTokens: Int { promptTokens + completionTokens }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(promptTokens, forKey: .promptTokens)
        try container.encode(completionTokens, forKey: .completionTokens)
        try container.encode(totalTokens, forKey: .totalTokens)
    }
}

struct OpenAIToolCallOut: Encodable {
    struct Fn: Encodable {
        let name: String
        let arguments: String
    }
    /// Streaming-only index (nil in the full-response object form).
    let index: Int?
    let id: String
    let type: String
    let function: Fn
}

struct OpenAIResponseMessage: Encodable {
    let role: String = "assistant"
    let content: String?
    let reasoningContent: String?
    let toolCalls: [OpenAIToolCallOut]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case reasoningContent = "reasoning_content"
    }
}

struct OpenAIChatResponse: ResponseEncodable {
    struct Choice: Encodable {
        let index: Int
        let message: OpenAIResponseMessage
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: OpenAIUsage
}

struct OpenAIStreamDelta: Encodable {
    let role: String?
    let content: String?
    let reasoningContent: String?
    let toolCalls: [OpenAIToolCallOut]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case reasoningContent = "reasoning_content"
    }
}

struct OpenAIChatChunk: Encodable {
    struct Choice: Encodable {
        let index: Int
        let delta: OpenAIStreamDelta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    let id: String
    let object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [Choice]
}

struct OpenAIModelList: ResponseEncodable {
    struct Model: Encodable {
        let id: String
        let object: String = "model"
        let created: Int
        let ownedBy: String = "dynamoe"

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }

    let object: String = "list"
    let data: [Model]
}

struct OpenAIErrorEnvelope: Encodable {
    struct Body: Encodable {
        let message: String
        let type: String
        let code: String?
    }
    let error: Body
}

// MARK: - Pure protocol mapping

/// Pure protocol-mapping logic for the OpenAI adapter. Every function is a
/// free static so it is unit-testable without a live engine or socket.
enum OpenAIAdapter {

    /// Whether tools may be offered for this request. OpenAI's forced choices
    /// (`"required"`, or a named-function object) demand a generation
    /// constraint this server cannot enforce; silently treating them as `auto`
    /// would let the model answer with a different tool or no tool at all.
    /// Reject them explicitly instead.
    static func toolUseEnabled(_ choice: OpenAIChatRequest.ToolChoice?) throws -> Bool {
        switch choice {
        case nil, .some(.auto):
            return true
        case .some(.none):
            return false
        case .some(.required):
            throw ServerAPIError.badRequest("tool_choice \"required\" is not supported: this server cannot force a tool call. Use \"auto\" or \"none\".")
        case .some(.function(let name)):
            throw ServerAPIError.badRequest("tool_choice for a specific function (\"\(name)\") is not supported: this server cannot force a tool call. Use \"auto\" or \"none\".")
        }
    }

    /// Maps the engine's terminal reasons onto OpenAI finish_reason values.
    static func finishReasonString(_ reason: TurnFinishReason, hasToolCalls: Bool) -> String {
        switch reason {
        case .lengthCapped:
            return "length"
        case .toolFreeze:
            return hasToolCalls ? "tool_calls" : "stop"
        case .eos, .endTag, .degenerateCycle, .stoppedBySequence, .stopped:
            return hasToolCalls ? "tool_calls" : "stop"
        }
    }

    /// Renders the OpenAI tool schemas as a compact declaration block that is
    /// appended to the effective system prompt. Mirrors the shape the in-app
    /// agent system prompt uses (a JSON array of function definitions) so
    /// grammar-masked tool-call models recognize the vocabulary.
    ///
    /// The instruction teaches the engine's canonical tagged call markup, not
    /// bare JSON: the bare-JSON fallback only recognizes built-in tool names,
    /// so a client-defined tool emitted as bare JSON would be returned as
    /// ordinary text. The tagged markup parses for any tool name and is the
    /// exact shape the grammar masks enforce, so constrained sampling stays
    /// compatible.
    static func toolDeclarationText(_ tools: [OpenAIChatRequest.Tool]) -> String? {
        let fns = tools.filter { ($0.type ?? "function") == "function" }
        guard !fns.isEmpty else { return nil }
        var definitions: [[String: String]] = []
        for tool in fns {
            var def: [String: String] = ["name": tool.function.name]
            if let desc = tool.function.description, !desc.isEmpty {
                def["description"] = desc
            }
            if let params = tool.function.parametersData,
               let paramsJSON = String(data: params, encoding: .utf8) {
                def["parameters"] = paramsJSON
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

    /// Splits the message list into the system prompt and role-tagged turns.
    ///
    /// History stays role-structured (rather than being flattened into one
    /// user block with "User:"/"Assistant:" labels) so the server can render it
    /// with the model family's native role boundaries. A flattened transcript
    /// re-tokenizes the whole conversation every turn, so each turn's prompt
    /// can no longer extend the previous turn's pinned prompt+generation and
    /// multi-turn requests fully re-prefill.
    static func splitMessages(_ messages: [OpenAIChatRequest.Message]) throws -> (systemPrompt: String, turns: [ChatPromptFormatter.ConversationTurn]) {
        let systemTexts = messages
            .filter { $0.role == "system" || $0.role == "developer" }
            .compactMap(\.contentText)
            .filter { !$0.isEmpty }
        let systemPrompt = systemTexts.joined(separator: "\n\n")

        let conversation = messages.filter { $0.role != "system" && $0.role != "developer" }
        guard !conversation.isEmpty else {
            throw ServerAPIError.badRequest("messages must contain at least one non-system message")
        }

        var turns: [ChatPromptFormatter.ConversationTurn] = []
        for message in conversation {
            switch message.role {
            case "user":
                turns.append(.init(role: .user, text: message.contentText ?? ""))
            case "assistant":
                var lines: [String] = []
                // Reasoning a client echoed back is re-rendered ahead of the
                // answer, closed with the canonical tag, so the turn reproduces
                // what was generated rather than only its visible tail.
                if let reasoning = message.reasoningContent?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !reasoning.isEmpty {
                    lines.append(reasoning + TurnTextSplitter.canonicalReasoningCloseTag)
                }
                if let content = message.contentText, !content.isEmpty {
                    lines.append("\(content)")
                }
                if let calls = message.toolCalls, !calls.isEmpty {
                    let rendered = calls.map { call in
                        var obj: [String: String] = ["name": call.function.name]
                        // Keep the call id: the tool-result turns below
                        // reference it via tool_call_id, and with several calls
                        // (or repeated calls to the same function) the id is the
                        // only thing tying a result back to its arguments.
                        if let id = call.id, !id.isEmpty {
                            obj["id"] = id
                        }
                        if let args = call.function.arguments, !args.isEmpty {
                            obj["arguments"] = args
                        }
                        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
                           let json = String(data: data, encoding: .utf8) {
                            return json
                        }
                        return "{\"name\": \"\(call.function.name)\"}"
                    }
                    .joined(separator: "\n")
                    lines.append(ChatPromptFormatter.openAIToolCallEchoPrefix + "\n" + rendered)
                }
                let body = lines.joined(separator: "\n\n")
                turns.append(.init(role: .assistant, text: body.isEmpty ? "(no content)" : body))
            case "tool", "function":
                let label = message.name ?? message.toolCallId ?? "tool"
                let args = message.contentText ?? "(no result)"
                turns.append(.init(role: .user, text: "Function result (\(label)):\n\(args)"))
            default:
                if let content = message.contentText, !content.isEmpty {
                    turns.append(.init(role: .user, text: "\(message.role.capitalized): \(content)"))
                }
            }
        }
        guard !turns.isEmpty else {
            throw ServerAPIError.badRequest("messages must contain at least one non-system message")
        }
        return (systemPrompt, turns)
    }

    /// Stable prefix-cache session key. The OpenAI protocol is stateless, so
    /// identity is anchored on the conversation's FIRST USER message plus the
    /// model: stateless clients resend the full history, that message never
    /// changes as a conversation grows, and — unlike the leading system
    /// message, which every conversation a client opens usually shares — it is
    /// what distinguishes unrelated conversations. Deterministic UUID (FNV-1a
    /// over two lanes); pure and testable. Two conversations that open
    /// identically still collide, so reuse is additionally gated on exact
    /// token-prefix equality inside PrefixCacheManager.findCommonPrefix and on
    /// `ServerTurnMemory` re-validating the rendered history.
    static func sessionKey(for messages: [OpenAIChatRequest.Message], model: String?) -> UUID {
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
            for call in anchor.toolCalls ?? [] {
                mixString("call:" + call.function.name)
                mixString(call.function.arguments ?? "")
            }
            mixString(anchor.name ?? "")
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
