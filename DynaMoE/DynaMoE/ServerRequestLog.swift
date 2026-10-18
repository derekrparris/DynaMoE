//
//  ServerRequestLog.swift
//  DynaMoE
//
//  Request logging for the local inference server: the row model, the
//  metadata-only unified-log line, and the recorder that owns per-request
//  log boxes. The wire-level middleware (ServerWireLogMiddleware, defined
//  alongside the router in ServerController.swift) creates a box per request;
//  route handlers and streaming body closures enrich it with the facts only
//  they know (model, tokens, finish reason, bodies); whoever holds the box
//  when the request's outcome is known composes the row and records it.
//
//  Privacy model (modeled on osaurus): default metadata-only. Body capture is
//  opt-in via `dynamoe_server_log_store_bodies`; when off, chat/message rows
//  carry the `contentWithheldMarker` placeholder in place of bodies so "not
//  stored" is never read as "should have been there". Authorization headers
//  and the configured API key are never recorded anywhere.
//

import Foundation

// MARK: - Row model

/// One recorded API-server request, persisted as a JSON payload by
/// `ServerRequestLogStore` and rendered by the Server settings dashboard.
struct ServerRequestLog: Identifiable, Codable, Sendable, Equatable, Hashable {
    enum Proto: String, Codable, Sendable {
        case openai
        case anthropic
        case other
    }

    /// How a streaming response ended. Non-streaming rows are `nil`.
    /// A stream's HTTP status is always 200 (errors travel as SSE frames),
    /// so this is the row's real error signal.
    enum StreamTermination: String, Codable, Sendable {
        case completed
        case errorFrame
        case clientDisconnected
    }

    struct ToolCallRecord: Codable, Sendable, Equatable, Hashable {
        var name: String
        /// Client-visible arguments JSON. Captured only when body capture is
        /// enabled; otherwise the withheld marker.
        var arguments: String
    }

    var id: UUID = UUID()
    var timestamp: Date = Date()
    var method: String = ""
    var path: String = ""
    var proto: Proto = .other
    var wasStreaming: Bool = false
    /// Wire status. For streaming rows always 200 — see `streamTermination`.
    var httpStatus: Int = 200
    var streamTermination: StreamTermination?
    var durationMs: Double = 0
    /// Time from request start to the first streamed token (streaming only).
    var firstTokenMs: Double?
    /// Time the request spent waiting in the FIFO for the engine's
    /// single-flight slot, from request arrival to slot-open. Nil when the
    /// turn never reached the slot (queue timeout, cancellation while
    /// queued) and never zero-filled, so "no wait" and "never ran" stay
    /// distinguishable.
    var queueWaitMs: Double?
    var model: String?
    var inputTokens: Int?
    var outputTokens: Int?
    var temperature: Double?
    var maxTokens: Int?
    var finishReason: String?
    var toolCalls: [ToolCallRecord]?
    var errorMessage: String?
    /// Decoded request size in bytes (Content-Length), when reported.
    var requestBytes: Int?
    var requestBody: String?
    var responseBody: String?
    /// Reason encoding cancelled-by-queue-timeout and similar launch-failure
    /// frames is included in `errorMessage`.

    /// Placeholder written in place of bodies when body capture is disabled,
    /// so the row stays self-describing about what was withheld (osaurus's
    /// contentWithheldMarker convention).
    static let contentWithheldMarker = "[content withheld — metadata only]"

    /// Bodies are clipped so a runaway request can never bloat the database.
    static let maxBodyCaptureLength = 32 * 1024
    static let maxToolArgumentCaptureLength = 4 * 1024

    var isError: Bool { httpStatus >= 400 || streamTermination == .errorFrame }

    /// Short correlation id printed in the unified-log line: the leading 8
    /// hex digits of the row id. Enough to pivot from a Console.app line to
    /// the dashboard row (the text filter is a substring match over the
    /// persisted payload), without flooding the line with a full UUID.
    var shortId: String {
        String(id.uuidString.prefix(8)).lowercased()
    }

    /// True when at least one body field holds captured content rather than
    /// the withheld placeholder or nil.
    var hasCapturedContent: Bool {
        [requestBody, responseBody].contains { $0 != nil && $0 != Self.contentWithheldMarker }
            || (toolCalls?.contains { $0.arguments != Self.contentWithheldMarker } ?? false)
    }

    static func clippedBody(_ text: String) -> String {
        guard text.count > maxBodyCaptureLength else { return text }
        return String(text.prefix(maxBodyCaptureLength))
    }

    static func clippedArguments(_ text: String) -> String {
        guard text.count > maxToolArgumentCaptureLength else { return text }
        return String(text.prefix(maxToolArgumentCaptureLength))
    }
}

// MARK: - Recorder

/// Owns the per-request box registry and is the single funnel through which a
/// completed row reaches the unified log and the persisted store. Thread-safe:
/// route handlers run on arbitrary executor threads.
final class ServerRequestLogRecorder: @unchecked Sendable {
    static let shared = ServerRequestLogRecorder()

    static let captureBodiesKey = "dynamoe_server_log_store_bodies"
    static let requestCategory = "server.request"

    private let lock = NSLock()
    private var boxes: [String: ServerRequestLogBox] = [:]
    /// Test indirection: recording normally appends to the shared on-disk
    /// store; tests point this at an in-memory store first so app-hosted runs
    /// never write to the developer's real request log.
    private var overrideStore: ServerRequestLogStore?

    private init() {}

    /// Whether the user opted into storing request/response content.
    nonisolated static var captureBodiesEnabled: Bool {
        UserDefaults.standard.bool(forKey: captureBodiesKey)
    }

    /// Installs (or with nil, removes) the store override used by tests.
    nonisolated func useOverrideStore(_ store: ServerRequestLogStore?) {
        lock.lock()
        overrideStore = store
        lock.unlock()
    }

    // MARK: Box registry

    /// Creates the box for one inbound request. Wire fields only; enrichment
    /// happens in the route as facts resolve. The row id reuses the wire
    /// request id when it is UUID-shaped (Hummingbird's `context.id`), so
    /// the store's row and any other log keyed by the same request share one
    /// identity; a non-UUID registry key (tests, alternate request-id
    /// sources) falls back to a fresh id.
    nonisolated func beginBox(requestId: String, method: String, path: String) -> ServerRequestLogBox {
        var wire = ServerRequestLog()
        wire.method = method
        wire.path = path
        wire.id = UUID(uuidString: requestId) ?? wire.id
        wire.proto = Self.proto(forPath: path)
        let box = ServerRequestLogBox(builder: wire)
        lock.lock()
        boxes[requestId] = box
        lock.unlock()
        return box
    }

    /// The box for an in-flight request, if its middleware has begun one.
    nonisolated func box(for requestId: String) -> ServerRequestLogBox? {
        lock.lock()
        defer { lock.unlock() }
        return boxes[requestId]
    }

    /// Drops the registry entry. The box itself stays alive as long as the
    /// route or streaming body closure still holds a strong reference.
    nonisolated func endBox(requestId: String) {
        lock.lock()
        boxes[requestId] = nil
        lock.unlock()
    }

    nonisolated static func proto(forPath path: String) -> ServerRequestLog.Proto {
        if path.hasPrefix("/v1/messages") { return .anthropic }
        if path.hasPrefix("/v1/chat") || path.hasPrefix("/v1/completions") { return .openai }
        return .other
    }

    // MARK: Recording

    /// Emits the unified-log line and appends the row to the persisted store.
    /// Fire-and-forget from the caller's perspective: never blocks the
    /// request path.
    nonisolated func record(_ row: ServerRequestLog) {
        let line = Self.composeLine(row)
        if row.isError {
            DiagLog.error(Self.requestCategory, line)
        } else {
            DiagLog.info(Self.requestCategory, line)
        }
        lock.lock()
        let store = overrideStore
        lock.unlock()
        (store ?? ServerRequestLogStore.shared).append(row)
    }

    /// The metadata-only Console.app line. Structured key=value so a single
    /// request is traceable without flipping a debug flag; never includes
    /// bodies (they live only in the persisted row, which is opt-in). Leads
    /// with the row's short id so a Console line can be pivoted to the
    /// dashboard row by search.
    nonisolated static func composeLine(_ row: ServerRequestLog) -> String {
        var parts: [String] = [
            "id=\(row.shortId)",
            "\(row.method) \(row.path)",
            "→ \(row.httpStatus)",
            String(format: "%.0fms", row.durationMs),
        ]
        if let wait = row.queueWaitMs, wait > 0 {
            parts.append("queue=\(Int(wait.rounded()))ms")
        }
        if row.wasStreaming {
            parts.append("stream=\(row.streamTermination.map { "\($0.rawValue)" } ?? "open")")
            if let ttft = row.firstTokenMs {
                parts.append("first=\(Int(ttft.rounded()))ms")
            }
        }
        if let model = row.model, !model.isEmpty {
            parts.append("model=\(model)")
        }
        if let input = row.inputTokens {
            parts.append("in=\(input)")
        }
        if let output = row.outputTokens {
            parts.append("out=\(output)")
        }
        if let toolCalls = row.toolCalls, !toolCalls.isEmpty {
            let names = toolCalls.map(\.name).joined(separator: ",")
            parts.append("tools=\(names)")
        }
        if let finish = row.finishReason {
            parts.append("finish=\(finish)")
        }
        if let message = row.errorMessage, !message.isEmpty {
            parts.append("error=\(message)")
        }
        return parts.joined(separator: " ")
    }

    /// Extracts the recordable error message for a failed request. Strips
    /// nothing today, but concentrates the policy decision (message strings
    /// are wire error messages, never credential material) in one place.
    nonisolated static func errorMessage(for error: Error) -> String {
        if let apiError = error as? ServerAPIError {
            return apiError.message
        }
        if let localized = error as? LocalizedError, let message = localized.errorDescription {
            return message
        }
        return error.localizedDescription
    }
}

// MARK: - Per-request box

/// Mutable, lock-guarded staging for the row of one in-flight request. The
/// middleware owns its lifetime via the recorder's registry; streaming body
/// closures retain it past the registry entry (the HTTP response is already
/// 200 by the time the stream drains, so the row is finalized when the
/// stream does).
final class ServerRequestLogBox: @unchecked Sendable {
    private let lock = NSLock()
    private var builder: ServerRequestLog
    private var claimed = false
    private let startedAt = Date()

    init(builder: ServerRequestLog) {
        self.builder = builder
    }

    /// Applies route/handler enrichment to the staged row.
    nonisolated func modify(_ mutate: (inout ServerRequestLog) -> Void) {
        lock.lock()
        mutate(&builder)
        lock.unlock()
    }

    /// Claims responsibility for recording: the first caller wins and every
    /// later claim returns false. The wire middleware records only unclaimed
    /// boxes; a streaming route claims its box before returning the 200
    /// response so the stream's terminal facts (not the empty response
    /// object) make it into the row.
    nonisolated func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if claimed { return false }
        claimed = true
        return true
    }

    var isClaimed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return claimed
    }

    /// Milliseconds the request has been in flight (for duration at
    /// finalization and TTFT at the first streamed token).
    nonisolated func elapsedMs() -> Double {
        Date().timeIntervalSince(startedAt) * 1000
    }

    /// Stamps the moment the queue's single-flight slot opened for this
    /// request, as a wait measured from request arrival into `queueWaitMs`.
    /// First caller wins: exactly one queue wait per row, and requests
    /// that never reach the slot stay nil instead of reading as a zero
    /// wait.
    nonisolated func noteSlotOpened() {
        lock.lock()
        if builder.queueWaitMs == nil {
            builder.queueWaitMs = Date().timeIntervalSince(startedAt) * 1000
        }
        lock.unlock()
    }

    /// Copy of the staged row as-is.
    nonisolated func snapshot() -> ServerRequestLog {
        lock.lock()
        defer { lock.unlock() }
        return builder
    }

    /// Stamps wire status, duration, and (for failures) the recordable error
    /// message, returning the finalized row for `record`.
    nonisolated func composeFinal(status: Int, error: (any Error)?) -> ServerRequestLog {
        lock.lock()
        builder.httpStatus = status
        builder.durationMs = Date().timeIntervalSince(startedAt) * 1000
        let row = builder
        lock.unlock()
        var finalized = row
        if let error {
            finalized.errorMessage = ServerRequestLogRecorder.errorMessage(for: error)
        }
        return finalized
    }
}

// MARK: - Request body summaries

/// The captured "request body" for inference rows is a normalized JSON
/// summary of the decoded request — the wire types are Decodable-only (their
/// decoding is deliberately lossy-normalizing), so raw bytes are re-encoded
/// back and re-decoding loses information anyway. The summary keeps every
/// field a client-format debugging session needs. Built only when body
/// capture is enabled.
extension ServerRequestLog {
    nonisolated static func openAISummary(_ request: OpenAIChatRequest) -> String? {
        var payload: [String: Any] = [:]
        payload["model"] = request.model
        payload["stream"] = request.stream
        payload["temperature"] = request.temperature
        payload["top_p"] = request.topP
        payload["top_k"] = request.topK
        payload["max_tokens"] = request.maxTokens
        payload["stop"] = request.stop
        payload["enable_thinking"] = request.enableThinking
        payload["messages"] = request.messages.map { message in
            var entry: [String: Any] = ["role": message.role]
            if let content = message.contentText { entry["content"] = content }
            if let name = message.name { entry["name"] = name }
            if let reasoning = message.reasoningContent { entry["reasoning_content"] = reasoning }
            if let toolCallId = message.toolCallId { entry["tool_call_id"] = toolCallId }
            if let calls = message.toolCalls {
                entry["tool_calls"] = calls.map { call in
                    var callEntry: [String: Any] = [
                        "id": call.id as Any,
                        "type": call.type as Any
                    ]
                    callEntry["function"] = ["name": call.function.name, "arguments": call.function.arguments as Any]
                    return callEntry
                }
            }
            return entry
        }
        if let tools = request.tools {
            payload["tools"] = tools.map { ["type": $0.type as Any, "name": $0.function.name] }
        }
        if let choice = request.toolChoice {
            switch choice {
            case .auto: payload["tool_choice"] = "auto"
            case .none: payload["tool_choice"] = "none"
            case .required: payload["tool_choice"] = "required"
            case .function(let name): payload["tool_choice"] = ["function": ["name": name]]
            }
        }
        return serializeSummary(payload)
    }

    nonisolated static func anthropicSummary(_ request: AnthropicMessagesRequest) -> String? {
        var payload: [String: Any] = [:]
        payload["model"] = request.model
        payload["system"] = request.systemText
        payload["stream"] = request.stream
        payload["temperature"] = request.temperature
        payload["top_p"] = request.topP
        payload["top_k"] = request.topK
        payload["max_tokens"] = request.maxTokens
        payload["stop_sequences"] = request.stopSequences
        payload["messages"] = request.messages.map { message in
            var entry: [String: Any] = ["role": message.role]
            if let content = message.contentText { entry["content"] = content }
            if let reasoning = message.reasoningText { entry["thinking"] = reasoning }
            if let uses = message.toolUses {
                entry["tool_use"] = uses.map { ["id": $0.id as Any, "name": $0.name, "input": $0.inputJSON as Any] }
            }
            if let results = message.toolResults {
                entry["tool_result"] = results.map { ["tool_use_id": $0.toolUseId as Any, "content": $0.content as Any] }
            }
            return entry
        }
        if let tools = request.tools {
            payload["tools"] = tools.map { ["name": $0.name] }
        }
        if let thinking = request.thinking {
            payload["thinking"] = ["type": thinking.type as Any, "budget_tokens": thinking.budgetTokens as Any]
        }
        if let choice = request.toolChoice {
            payload["tool_choice"] = ["type": choice.type as Any, "name": choice.name as Any]
        }
        return serializeSummary(payload)
    }

    nonisolated private static func serializeSummary(_ payload: [String: Any]) -> String? {
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return clippedBody(json)
    }
}
