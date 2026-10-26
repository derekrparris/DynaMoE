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
public nonisolated struct CloudChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: String?
    public let toolCalls: [CloudChatToolCall]?
    public let toolCallId: String?

    public init(
        role: String,
        content: String?,
        toolCalls: [CloudChatToolCall]? = nil,
        toolCallId: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
    }

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
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

/// Sampling knobs the OpenAI-compatible surface understands. DynaMoE also
/// samples with Min-P, Top-K, and a repetition penalty; remote providers do
/// not expose those, so they are deliberately not forwarded (same silent-gap
/// contract the Apple Foundation Model path documents).
public nonisolated struct CloudChatGenerationParams: Equatable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var presencePenalty: Double?
    public var maxTokens: Int?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        presencePenalty: Double? = nil,
        maxTokens: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.maxTokens = maxTokens
    }
}

// MARK: - Usage & Results

public nonisolated struct CloudUsage: Codable, Equatable, Sendable {
    public let promptTokens: Int?
    public let completionTokens: Int?
    public let totalTokens: Int?

    public init(promptTokens: Int? = nil, completionTokens: Int? = nil, totalTokens: Int? = nil) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
    }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
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
    case unsupportedFlavor(flavor: String)
    case httpStatus(code: Int, summary: String)
    case transport(String)
    case invalidResponse(String)

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey(let providerName):
            return "No API key is stored for \(providerName). Add one in Settings ▸ Cloud Providers."
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
        excludingMessageIds: Set<UUID> = []
    ) -> [CloudChatMessage] {
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
            guard !content.isEmpty else { continue }
            assembled.append(CloudChatMessage(
                role: message.role == .user ? "user" : "assistant",
                content: content
            ))
        }
        return assembled
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
        case .awaitingApproval: return "Awaiting approval."
        case .running: return "The call did not finish before the turn ended."
        case .error: return "The tool call failed."
        case .success: return "Completed."
        }
    }

    /// Character-based input estimate for the turn log (providers report real
    /// usage only in the final streamed chunk, which may never arrive).
    public static func approximatePromptTokens(system: String?, messages: [CloudChatMessage]) -> Int {
        let characters = (system ?? "").count + messages.reduce(0) { total, message in
            total + (message.content ?? "").count
                + (message.toolCalls ?? []).reduce(0) { $0 + $1.function.name.count + $1.function.arguments.count }
        }
        return max(1, characters / 4)
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
            let content = (message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
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
