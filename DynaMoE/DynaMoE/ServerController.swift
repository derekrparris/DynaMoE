//
//  ServerController.swift
//  DynaMoE
//
//  Local OpenAI-compatible inference server (Milestones M2). Binds a
//  Hummingbird 2 HTTP server to 127.0.0.1 and adapts POST /v1/chat/completions
//  onto the headless CompletionEngine: FIFO request queueing over the
//  engine's single-flight turn slot, SSE streaming with reasoning_content
//  deltas, disconnect-cancel, and optional bearer-token auth.
//
//  Concurrency layout:
//  - `ServerController` (MainActor) owns the ServiceGroup lifecycle only.
//  - `TurnExecutionQueue` is an actor that serializes queued work; every
//    request waits for the previous engine turn before launching its own.
//  - Route handlers are @Sendable closures; they touch only lock-guarded
//    singletons (CompletionEngine, UserDefaults) and the queue actor.
//

import Foundation
import Combine
import Hummingbird
import HTTPTypes
import NIOCore
import NIOFoundationEssentialsCompat
import ServiceLifecycle

// MARK: - Request queue / streaming relay

/// Events relayed from a queued turn to the SSE body writer (or the
/// non-streaming aggregator).
enum QueuedTurnEvent: @unchecked Sendable {
    /// Server-level comment ping (queue wait / prefill silence).
    case ping(comment: String?)
    case engine(CompletionEvent)
    /// Synchronous launch failure (model not loaded, engine busy, etc.).
    case launchFailed(String)
}

/// FIFO queue over the engine's single-flight generation slot. Each request
/// enqueues its whole turn; the work only begins once every earlier turn has
/// fully drained. Cancellation propagates to the owner task, which cancels
/// the engine turn handle.
actor TurnExecutionQueue {
    private var tail: Task<Void, Never> = Task {}
    private var enqueuedOperations = 0
    private var completedOperations = 0

    /// Turns enqueued but not yet drained (queued or currently running) —
    /// surfaced by the Server settings dashboard.
    var pendingCount: Int {
        max(0, enqueuedOperations - completedOperations)
    }

    /// Enqueues streaming work. Returns the event relay stream plus the task
    /// that owns the work (cancel it to abort a queued/running turn).
    /// `queueWaitTimeout` bounds how long the turn may sit behind earlier
    /// turns; when it expires the relay emits a launchFailed event instead of
    /// running the work. Nil (the default) waits indefinitely.
    func streamTurn(
        _ work: @escaping @Sendable (_ emit: @escaping @Sendable (QueuedTurnEvent) -> Void) async -> Void,
        queueWaitTimeout: TimeInterval? = nil
    ) -> (AsyncStream<QueuedTurnEvent>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream.makeStream(of: QueuedTurnEvent.self, bufferingPolicy: .unbounded)
        let previous = tail
        enqueuedOperations += 1
        let deadline = queueWaitTimeout.map { Date().addingTimeInterval($0) }
        let heartbeat = Task {
            // Queue-wait heartbeat so clients don't idle-timeout while a long
            // turn ahead of us holds the single-flight slot.
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard !Task.isCancelled else { return }
                continuation.yield(.ping(comment: nil))
            }
        }
        let task = Task<Void, Never> { [previous, heartbeat, deadline] in
            defer { self.completedOperations += 1 }
            await previous.value
            heartbeat.cancel()
            guard !Task.isCancelled else {
                continuation.finish()
                return
            }
            if let deadline, Date() >= deadline {
                continuation.yield(.launchFailed("Request timed out waiting in the generation queue"))
                continuation.finish()
                return
            }
            await work { continuation.yield($0) }
            continuation.finish()
        }
        // Keep the chain element alive regardless of this task's outcome.
        tail = task
        return (stream, task)
    }

    /// Enqueues value-returning work (non-streaming requests and other
    /// engine-facing endpoints). `queueWaitTimeout` bounds how long the work
    /// may sit behind earlier turns; when it expires the perform throws a
    /// request-timeout error instead of running the work. Nil (the default)
    /// waits indefinitely.
    func perform<T: Sendable>(
        _ work: @escaping @Sendable () async throws -> T,
        queueWaitTimeout: TimeInterval? = nil
    ) async throws -> T {
        let previous = tail
        enqueuedOperations += 1
        let deadline = queueWaitTimeout.map { Date().addingTimeInterval($0) }
        let task = Task<T, Error> { [previous, deadline] in
            defer { self.completedOperations += 1 }
            await previous.value
            if let deadline, Date() >= deadline {
                throw ServerAPIError.requestTimeout("Request timed out waiting in the generation queue")
            }
            return try await work()
        }
        tail = Task { _ = try? await task.value }
        return try await task.value
    }
}

// MARK: - Request statistics

/// Lock-guarded request counters shared between the nonisolated static route
/// handlers (arbitrary executor threads) and the MainActor Server dashboard.
final class ServerStats: @unchecked Sendable {
    struct Snapshot: Sendable, Equatable {
        var totalRequests: Int
        var activeRequests: Int
    }

    private let lock = NSLock()
    private var total = 0
    private var active = 0

    func beginRequest() {
        lock.lock()
        total += 1
        active += 1
        lock.unlock()
    }

    func endRequest() {
        lock.lock()
        active = max(0, active - 1)
        lock.unlock()
    }

    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(totalRequests: total, activeRequests: active)
    }
}

// MARK: - ServerController

/// Tiny lock-guarded int so the queued turn task can hand the resolved
/// prompt-token count back to the SSE writer closure without a data race.
final class PromptCountBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func set(_ newValue: Int) {
        lock.lock()
        value = newValue
        lock.unlock()
    }

    func get() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

// MARK: - Wire-shaped errors

/// Server error that renders as the correct JSON envelope for the calling
/// protocol: the Anthropic `{"type":"error","error":{...}}` shape under
/// /v1/messages, the OpenAI `{"error":{...}}` shape everywhere else. Conforms
/// to HTTPResponseError so the router renders it directly with its status —
/// no route-level do/catch mapping required.
struct ServerAPIError: Error, HTTPResponseError, LocalizedError, Sendable {
    enum WireType: String, Sendable {
        case invalidRequest = "invalid_request_error"
        case authentication = "authentication_error"
        case requestTimeout = "request_timeout"
        case overloaded = "overloaded_error"
        case api = "api_error"
    }

    let status: HTTPResponse.Status
    let wireType: WireType
    let message: String

    var errorDescription: String? { message }

    static func badRequest(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .badRequest, wireType: .invalidRequest, message: message)
    }

    static func unauthorized(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .unauthorized, wireType: .authentication, message: message)
    }

    static func requestTimeout(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .requestTimeout, wireType: .requestTimeout, message: message)
    }

    static func serviceUnavailable(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .serviceUnavailable, wireType: .overloaded, message: message)
    }

    static func internalError(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .internalServerError, wireType: .api, message: message)
    }

    func response(from request: Request, context: some RequestContext) throws -> Response {
        var response: Response
        if request.uri.path.hasPrefix("/v1/messages") {
            response = try context.responseEncoder.encode(
                AnthropicErrorEvent(error: .init(type: wireType.rawValue, message: message)),
                from: request,
                context: context
            )
        } else {
            response = try context.responseEncoder.encode(
                OpenAIErrorEnvelope(error: .init(message: message, type: wireType.rawValue, code: nil)),
                from: request,
                context: context
            )
        }
        response.status = self.status
        return response
    }
}

final class ServerController: ObservableObject {
    static let shared = ServerController()

    @Published private(set) var isRunning = false
    @Published private(set) var boundPort: Int = 0
    @Published private(set) var lastError: String?

    /// Request counters read by the route handlers (nonisolated) and the
    /// Server settings dashboard (MainActor).
    nonisolated static let stats = ServerStats()

    private var serviceGroup: ServiceGroup?
    private let queue = TurnExecutionQueue()

    static let enabledKey = "dynamoe_server_enabled"
    static let portKey = "dynamoe_server_port"
    static let apiKeyKey = "dynamoe_server_api_key"
    static let queueTimeoutKey = "dynamoe_server_queue_timeout"
    static let maxPromptTokensKey = "dynamoe_server_max_prompt_tokens"
    static let defaultPort = 8787
    static let bindHost = "127.0.0.1"

    private init() {}

    // MARK: Settings (read off the default values, tolerating first-run absence)

    nonisolated static var resolvedPort: Int {
        let stored = UserDefaults.standard.integer(forKey: portKey)
        return (1...65535).contains(stored) ? stored : defaultPort
    }

    /// Seconds a request may wait in the generation queue before it is
    /// rejected. Absent (first run) → 120; explicit 0 → wait indefinitely.
    nonisolated static var resolvedQueueWaitTimeout: TimeInterval? {
        guard let stored = UserDefaults.standard.object(forKey: queueTimeoutKey) as? Double else { return 120 }
        return stored > 0 ? stored : nil
    }

    /// Token ceiling for request prompts; longer prompts are rejected with
    /// 400 before prefill can pin gigabytes of KV cache. Absent → 262,144;
    /// explicit 0 → unlimited.
    nonisolated static var resolvedMaxPromptTokens: Int {
        guard let stored = UserDefaults.standard.object(forKey: maxPromptTokensKey) as? Int else { return 262_144 }
        return stored > 0 ? stored : Int.max
    }

    nonisolated private static var resolvedAPIKey: String? {
        let raw = UserDefaults.standard.string(forKey: apiKeyKey) ?? ""
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? nil : trimmed
    }

    // MARK: Lifecycle

    /// Queue depth snapshot for the Server settings dashboard (turns queued or
    /// currently generating toward the engine's single-flight slot).
    nonisolated func pendingQueueDepth() async -> Int {
        await queue.pendingCount
    }

    func start() {
        guard !isRunning else { return }
        lastError = nil
        let port = Self.resolvedPort
        let router = Self.buildRouter(queue: queue)
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(Self.bindHost, port: port), serverName: "DynaMoE")
        )
        let group = ServiceGroup(services: [app])
        serviceGroup = group
        boundPort = port
        isRunning = true
        Task.detached { [weak self] in
            do {
                try await group.run()
            } catch {
                await self?.noteError("Server stopped with error: \(error.localizedDescription)")
            }
        }
        NSLog("🌐 [SERVER] listening on %@:%d", Self.bindHost, port)
    }

    func stop() async {
        guard let group = serviceGroup else { return }
        serviceGroup = nil
        CompletionEngine.shared.cancelActiveTurn()
        await group.triggerGracefulShutdown()
        isRunning = false
        boundPort = 0
        print("🌐 [SERVER] stopped")
    }

    func setRunning(_ running: Bool) async {
        if running && !isRunning {
            start()
        } else if !running && isRunning {
            await stop()
        }
    }

    private func noteError(_ message: String) {
        lastError = message
        isRunning = false
        boundPort = 0
        NSLog("🌐 [SERVER] error: %@", message)
    }

    // MARK: Router

    nonisolated static func buildRouter(queue: TurnExecutionQueue) -> Router<BasicRequestContext> {
        let router = Router()

        router.get("health") { _, _ in
            ServerController.healthBody()
        }

        router.get("v1/models") { request, context in
            try ServerController.authorizeOrThrow(request)
            if let info = CompletionEngine.shared.loadedModelInfo {
                return OpenAIModelList(data: [OpenAIModelList.Model(id: info.displayName, created: Int(Date().timeIntervalSince1970))])
            }
            return OpenAIModelList(data: [])
        }

        router.post("v1/chat/completions") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            try ServerController.authorizeOrThrow(request)
            let apiRequest: OpenAIChatRequest
            do {
                apiRequest = try await request.decode(as: OpenAIChatRequest.self, context: context)
            } catch {
                throw ServerAPIError.badRequest("Invalid request body: \(error.localizedDescription)")
            }
            if apiRequest.stream == true {
                // Streaming keeps the 200 + text/event-stream contract even when
                // the engine can't accept the turn: errors arrive as SSE frames.
                return ServerController.streamingResponse(apiRequest: apiRequest, queue: queue)
            }
            let plan = try ServerController.buildTurnPlan(apiRequest)
            let result = try await ServerController.nonStreamingResponse(plan: plan, queue: queue)
            return try context.responseEncoder.encode(result, from: request, context: context)
        }

        router.post("v1/messages") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            try ServerController.authorizeOrThrow(request)
            let apiRequest: AnthropicMessagesRequest
            do {
                apiRequest = try await request.decode(as: AnthropicMessagesRequest.self, context: context)
            } catch {
                throw ServerAPIError.badRequest("Invalid request body: \(error.localizedDescription)")
            }
            if apiRequest.stream == true {
                // Streaming keeps the 200 + text/event-stream contract even when
                // the engine can't accept the turn: errors arrive as SSE frames.
                return ServerController.anthropicStreamingResponse(apiRequest: apiRequest, queue: queue)
            }
            let plan = try ServerController.buildTurnPlan(apiRequest)
            let result = try await ServerController.anthropicNonStreamingResponse(plan: plan, queue: queue)
            return try context.responseEncoder.encode(result, from: request, context: context)
        }

        return router
    }

    // MARK: Auth

    nonisolated private static func authorizeOrThrow(_ request: Request) throws {
        guard let apiKey = resolvedAPIKey else { return }
        let header = request.headers[.authorization] ?? ""
        var presented = header.hasPrefix("Bearer ") ? String(header.dropFirst("Bearer ".count)).trimmingCharacters(in: .whitespaces) : ""
        // Anthropic clients authenticate with the x-api-key header instead
        // of a bearer token; accept either presentation shape.
        if presented.isEmpty, let apiKeyName = HTTPField.Name("x-api-key") {
            presented = (request.headers[apiKeyName] ?? "").trimmingCharacters(in: .whitespaces)
        }
        guard !presented.isEmpty, presented == apiKey else {
            throw ServerAPIError.unauthorized("Invalid or missing API key")
        }
    }

    // MARK: Turn plan

    struct TurnPlan: Sendable {
        var request: CompletionTurnRequest
        var promptTokenCount: Int
        var modelName: String
        var hasTools: Bool
    }

    nonisolated private static func requireLoadedModel() throws -> CompletionEngine.LoadedModelInfo {
        guard let info = CompletionEngine.shared.loadedModelInfo else {
            throw ServerAPIError.serviceUnavailable("No model is loaded. Load a model in the DynaMoE app first.")
        }
        return info
    }

    /// Adapter-neutral inputs to the shared turn-plan builder. Each protocol
    /// adapter resolves its own wire-format defaults (thinking opt-in vs
    /// opt-out, sampler extras, stop-sequence spelling, session identity)
    /// before handing the resolved values to the shared core.
    struct TurnPlanInputs: Sendable {
        var systemPrompt: String
        var userPrompt: String
        var temperature: Float?
        var topP: Float?
        var topK: Int?
        var minP: Float?
        var repetitionPenalty: Float?
        var presencePenalty: Float?
        var maxTokens: Int?
        var stopSequences: [String]
        var thinkingRequested: Bool
        var hasTools: Bool
        var toolDeclaration: String?
        var sessionId: UUID
    }

    nonisolated private static func buildTurnPlan(_ inputs: TurnPlanInputs) throws -> TurnPlan {
        let engine = CompletionEngine.shared
        let info = try requireLoadedModel()

        var cleanSystem = ModelConfig.buildEffectiveSystemPrompt(
            userPrompt: inputs.systemPrompt,
            config: engine.promptModelConfig,
            summary: engine.loadedSummary,
            modelName: info.displayName,
            modelPath: info.path,
            currentDate: Date()
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        if inputs.hasTools, let declaration = inputs.toolDeclaration {
            cleanSystem = cleanSystem.isEmpty ? declaration : cleanSystem + "\n\n" + declaration
        }

        let thinkingEnabled = info.supportsThinking && inputs.thinkingRequested
        let formattedPrompt = ChatPromptFormatter.formatPrompt(
            prompt: inputs.userPrompt,
            cleanSystem: cleanSystem,
            thinkingEnabled: thinkingEnabled,
            modelSupportsThinking: info.supportsThinking,
            modelConfig: engine.promptModelConfig
        )

        let promptTokenIds: [UInt32]
        do {
            promptTokenIds = try engine.tokenize(formattedPrompt)
        } catch {
            throw ServerAPIError.badRequest("Tokenizer failed to encode the prompt: \(error.localizedDescription)")
        }
        guard !promptTokenIds.isEmpty else {
            throw ServerAPIError.badRequest("Prompt tokenization produced 0 tokens.")
        }
        let promptTokenCap = Self.resolvedMaxPromptTokens
        if promptTokenIds.count > promptTokenCap {
            throw ServerAPIError.badRequest("Prompt is too long: \(promptTokenIds.count) tokens exceeds the server's \(promptTokenCap)-token limit (dynamoe_server_max_prompt_tokens).")
        }

        let defaults = UserDefaults.standard
        let defaultMaxTokens = (defaults.object(forKey: "dynamoe_max_tokens") as? Int) ?? 8192
        let requestedMax = inputs.maxTokens ?? defaultMaxTokens
        let maxTokens = max(1, min(requestedMax, 262_144))

        let sampling = TurnSampling(
            temperature: inputs.temperature ?? 0.0,
            topP: inputs.topP ?? 0.9,
            minP: inputs.minP ?? 0.05,
            topK: inputs.topK ?? 50,
            repetitionPenalty: inputs.repetitionPenalty ?? 1.1,
            presencePenalty: inputs.presencePenalty ?? 0.0,
            maxTokens: maxTokens
        )

        let kvRaw = defaults.string(forKey: "dynamoe_kv_cache_precision") ?? ""
        let kvPrecision = KVCachePrecision(rawValue: kvRaw) ?? .fp16
        let memory = TurnMemorySettings(executionMode: .autoDetect, budgetMode: .balanced16GB, kvPrecision: kvPrecision)

        let turnRequest = CompletionTurnRequest(
            promptTokenIds: promptTokenIds,
            formattedPrompt: formattedPrompt,
            sessionId: inputs.sessionId,
            sampling: sampling,
            memory: memory,
            agentToolsEnabled: inputs.hasTools,
            thinkingEnabled: thinkingEnabled,
            agentStep: 0,
            forceSynthesis: false,
            isSplicedContinuation: false,
            stopSequences: inputs.stopSequences
        )
        return TurnPlan(request: turnRequest, promptTokenCount: promptTokenIds.count, modelName: info.displayName, hasTools: inputs.hasTools)
    }

    nonisolated private static func buildTurnPlan(_ apiRequest: OpenAIChatRequest) throws -> TurnPlan {
        let info = try requireLoadedModel()
        let (systemPrompt, userPrompt) = try OpenAIAdapter.splitMessages(apiRequest.messages)
        let hasTools = !(apiRequest.tools ?? []).isEmpty && apiRequest.toolChoice != "none"
        return try buildTurnPlan(TurnPlanInputs(
            systemPrompt: systemPrompt,
            userPrompt: userPrompt,
            temperature: apiRequest.temperature,
            topP: apiRequest.topP,
            topK: apiRequest.topK,
            minP: apiRequest.minP,
            repetitionPenalty: apiRequest.repetitionPenalty,
            presencePenalty: apiRequest.presencePenalty,
            maxTokens: apiRequest.maxTokens,
            stopSequences: apiRequest.stop ?? [],
            thinkingRequested: apiRequest.enableThinking ?? true,
            hasTools: hasTools,
            toolDeclaration: hasTools ? OpenAIAdapter.toolDeclarationText(apiRequest.tools ?? []) : nil,
            sessionId: OpenAIAdapter.sessionKey(for: apiRequest.messages, model: info.displayName)
        ))
    }

    nonisolated private static func buildTurnPlan(_ apiRequest: AnthropicMessagesRequest) throws -> TurnPlan {
        guard let maxTokens = apiRequest.maxTokens else {
            throw ServerAPIError.badRequest("max_tokens is a required field")
        }
        let info = try requireLoadedModel()
        let userPrompt = try AnthropicAdapter.splitMessages(apiRequest.messages)
        let hasTools = !(apiRequest.tools ?? []).isEmpty && apiRequest.toolChoiceType != "none"
        return try buildTurnPlan(TurnPlanInputs(
            systemPrompt: apiRequest.systemText ?? "",
            userPrompt: userPrompt,
            temperature: apiRequest.temperature,
            topP: apiRequest.topP,
            topK: apiRequest.topK,
            minP: nil,
            repetitionPenalty: nil,
            presencePenalty: nil,
            maxTokens: maxTokens,
            stopSequences: apiRequest.stopSequences ?? [],
            thinkingRequested: apiRequest.thinking?.type == "enabled",
            hasTools: hasTools,
            toolDeclaration: hasTools ? AnthropicAdapter.toolDeclarationText(apiRequest.tools ?? []) : nil,
            sessionId: AnthropicAdapter.sessionKey(for: apiRequest.messages, model: info.displayName)
        ))
    }

    // MARK: Non-streaming handling

    /// Runs a planned turn through the FIFO queue to completion. Shared by
    /// both adapters' non-streaming paths.
    nonisolated private static func runTurn(plan: TurnPlan, queue: TurnExecutionQueue) async throws -> TurnResult {
        try await queue.perform(queueWaitTimeout: resolvedQueueWaitTimeout) {
            () async throws -> TurnResult in
            let launch: CompletionLaunch
            do {
                launch = try CompletionEngine.shared.launchTurn(plan.request)
            } catch {
                throw ServerAPIError.serviceUnavailable((error as? CompletionLaunchError)?.errorDescription ?? "\(error.localizedDescription)")
            }
            return try await withTaskCancellationHandler {
                for await event in launch.stream {
                    switch event {
                    case .failedTurn(let failure):
                        if failure.cancelled { throw ServerAPIError.serviceUnavailable("Generation cancelled") }
                        throw ServerAPIError.internalError(failure.message)
                    case .finished(let result):
                        return result
                    case .started, .priming, .prefillProgress, .frame:
                        continue
                    }
                }
                throw ServerAPIError.internalError("Generation stream closed without a result")
            } onCancel: {
                launch.handle.cancel()
            }
        }
    }

    nonisolated private static func nonStreamingResponse(plan: TurnPlan, queue: TurnExecutionQueue) async throws -> OpenAIChatResponse {
        let result = try await runTurn(plan: plan, queue: queue)

        let toolCalls = result.finishReason == .toolFreeze || !result.parsedToolCalls.isEmpty
            ? result.parsedToolCalls.map { call in
                OpenAIToolCallOut(index: nil, id: "call-\(UUID().uuidString.prefix(8))", type: "function", function: .init(name: call.name, arguments: Self.toolArgumentsJSON(call)))
            }
            : nil

        let message = OpenAIResponseMessage(
            content: result.responseText,
            reasoningContent: result.thinkingText.isEmpty ? nil : result.thinkingText,
            toolCalls: (toolCalls?.isEmpty == false) ? toolCalls : nil
        )
        let finishReason = OpenAIAdapter.finishReasonString(result.finishReason, hasToolCalls: !(toolCalls ?? []).isEmpty)
        return OpenAIChatResponse(
            id: "chatcmpl-\(UUID().uuidString.prefix(24))",
            created: Int(Date().timeIntervalSince1970),
            model: plan.modelName,
            choices: [.init(index: 0, message: message, finishReason: finishReason)],
            usage: .init(promptTokens: plan.promptTokenCount, completionTokens: result.tokensGenerated)
        )
    }

    /// The engine parses arguments into a dictionary; re-serialize for the
    /// OpenAI `function.arguments` JSON-string field (raw text when valid).
    nonisolated private static func toolArgumentsJSON(_ call: ParsedToolCall) -> String {
        if !call.rawArguments.isEmpty, call.rawArguments.contains(":") || call.rawArguments.contains("{") {
            return call.rawArguments
        }
        if !call.arguments.isEmpty,
           let data = try? JSONSerialization.data(withJSONObject: call.arguments, options: [.sortedKeys]),
           let json = String(data: data, encoding: .utf8) {
            return json
        }
        return "{}"
    }

    // MARK: Streaming handling

    nonisolated private static func streamingResponse(apiRequest: OpenAIChatRequest, queue: TurnExecutionQueue) -> Response {
        let encoder = JSONEncoder()
        let id = "chatcmpl-\(UUID().uuidString.prefix(24))"
        let created = Int(Date().timeIntervalSince1970)
        let model = apiRequest.model ?? CompletionEngine.shared.loadedModelInfo?.displayName ?? "dynamoe"

        func chunkBuffer(_ delta: OpenAIStreamDelta, finishReason: String?) throws -> ByteBuffer {
            let chunk = OpenAIChatChunk(id: id, created: created, model: model, choices: [.init(index: 0, delta: delta, finishReason: finishReason)])
            return ByteBuffer(data: try encoder.encode(chunk))
        }

        let body = ResponseBody { writer in
            func sse(_ buffer: ByteBuffer) async throws {
                var frame = ByteBuffer(string: "data: ")
                frame.writeImmutableBuffer(buffer)
                frame.writeString("\n\n")
                try await writer.write(frame)
            }
            func sseRaw(_ text: String) async throws {
                var frame = ByteBuffer(string: "data: ")
                frame.writeString(text)
                frame.writeString("\n\n")
                try await writer.write(frame)
            }
            func ping(_ comment: String?) async throws {
                try await writer.write(ByteBuffer(string: ":\(comment ?? "ping")\n\n"))
            }

            var sentRole = false
            var sentThinking = ""
            var sentContent = ""
            let promptTokensBox = PromptCountBox()
            let (relay, task) = await queue.streamTurn(queueWaitTimeout: resolvedQueueWaitTimeout) { emit in
                do {
                    let plan = try ServerController.buildTurnPlan(apiRequest)
                    promptTokensBox.set(plan.promptTokenCount)
                    let launch = try CompletionEngine.shared.launchTurn(plan.request)
                    if Task.isCancelled {
                        launch.handle.cancel()
                        return
                    }
                    await withTaskCancellationHandler {
                        for await event in launch.stream {
                            emit(.engine(event))
                        }
                    } onCancel: {
                        launch.handle.cancel()
                    }
                } catch {
                    let message = (error as? ServerAPIError)?.message
                        ?? (error as? CompletionLaunchError)?.errorDescription
                        ?? error.localizedDescription
                    emit(.launchFailed(message))
                }
            }
            defer { task.cancel() }
            var finishedCleanly = false
            for await event in relay {
                switch event {
                case .ping(let comment):
                    try await ping(comment)
                case .launchFailed(let message):
                    let envelope = OpenAIErrorEnvelope(error: .init(message: message, type: "server_error", code: nil))
                    if let data = try? encoder.encode(envelope) {
                        try await sse(ByteBuffer(data: data))
                    }
                    try await sseRaw("[DONE]")
                    finishedCleanly = true
                case .engine(.started):
                    if !sentRole {
                        sentRole = true
                        try await sse(chunkBuffer(.init(role: "assistant", content: nil, reasoningContent: nil, toolCalls: nil), finishReason: nil))
                    }
                case .engine(.priming):
                    if !sentRole {
                        sentRole = true
                        try await sse(chunkBuffer(.init(role: "assistant", content: nil, reasoningContent: nil, toolCalls: nil), finishReason: nil))
                    }
                    try await ping("priming weights")
                case .engine(.prefillProgress(let status, _, _)):
                    if let status = status {
                        try await ping("prefill: \(status)")
                    }
                case .engine(.frame(let frame)):
                    if !sentRole {
                        sentRole = true
                        try await sse(chunkBuffer(.init(role: "assistant", content: nil, reasoningContent: nil, toolCalls: nil), finishReason: nil))
                    }
                    if frame.thinkingText.count > sentThinking.count && frame.thinkingText.hasPrefix(sentThinking) {
                        let delta = String(frame.thinkingText.dropFirst(sentThinking.count))
                        sentThinking = frame.thinkingText
                        try await sse(chunkBuffer(.init(role: nil, content: nil, reasoningContent: delta, toolCalls: nil), finishReason: nil))
                    } else if frame.thinkingText.count > sentThinking.count {
                        // UTF-8 repair mid-stream: re-render rather than fork.
                        sentThinking = frame.thinkingText
                        try await sse(chunkBuffer(.init(role: nil, content: nil, reasoningContent: frame.thinkingText, toolCalls: nil), finishReason: nil))
                    }
                    if frame.responseText.count > sentContent.count && frame.responseText.hasPrefix(sentContent) {
                        let delta = String(frame.responseText.dropFirst(sentContent.count))
                        sentContent = frame.responseText
                        try await sse(chunkBuffer(.init(role: nil, content: delta, reasoningContent: nil, toolCalls: nil), finishReason: nil))
                    } else if frame.responseText.count > sentContent.count {
                        sentContent = frame.responseText
                        try await sse(chunkBuffer(.init(role: nil, content: frame.responseText, reasoningContent: nil, toolCalls: nil), finishReason: nil))
                    }
                case .engine(.failedTurn(let failure)):
                    let envelope = OpenAIErrorEnvelope(error: .init(
                        message: failure.cancelled ? "Generation cancelled" : failure.message,
                        type: failure.cancelled ? "cancelled" : "server_error",
                        code: nil
                    ))
                    if let data = try? encoder.encode(envelope) {
                        try await sse(ByteBuffer(data: data))
                    }
                    try await sseRaw("[DONE]")
                    finishedCleanly = true
                case .engine(.finished(let result)):
                    // Tail text the throttle never emitted.
                    if result.thinkingText.count > sentThinking.count && result.thinkingText.hasPrefix(sentThinking) {
                        try await sse(chunkBuffer(.init(role: nil, content: nil, reasoningContent: String(result.thinkingText.dropFirst(sentThinking.count)), toolCalls: nil), finishReason: nil))
                    }
                    if result.responseText.count > sentContent.count && result.responseText.hasPrefix(sentContent) {
                        try await sse(chunkBuffer(.init(role: nil, content: String(result.responseText.dropFirst(sentContent.count)), reasoningContent: nil, toolCalls: nil), finishReason: nil))
                    }
                    let toolCalls: [OpenAIToolCallOut]? = {
                        guard !result.parsedToolCalls.isEmpty || result.finishReason == .toolFreeze else { return nil }
                        guard !result.parsedToolCalls.isEmpty else { return nil }
                        return result.parsedToolCalls.enumerated().map { index, call in
                            OpenAIToolCallOut(index: index, id: "call-\(UUID().uuidString.prefix(8))", type: "function", function: .init(name: call.name, arguments: toolArgumentsJSON(call)))
                        }
                    }()
                    if let toolCalls, !toolCalls.isEmpty {
                        try await sse(chunkBuffer(.init(role: nil, content: nil, reasoningContent: nil, toolCalls: toolCalls), finishReason: nil))
                    }
                    let finishReason = OpenAIAdapter.finishReasonString(result.finishReason, hasToolCalls: !(toolCalls ?? []).isEmpty)
                    try await sse(chunkBuffer(.init(role: nil, content: nil, reasoningContent: nil, toolCalls: nil), finishReason: finishReason))
                    // Final usage chunk (OpenAI includes this with stream_options.include_usage).
                    let usageTail = OpenAIUsage(promptTokens: promptTokensBox.get(), completionTokens: result.tokensGenerated)
                    if let data = try? JSONSerialization.data(withJSONObject: [
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [],
                        "usage": [
                            "prompt_tokens": usageTail.promptTokens,
                            "completion_tokens": usageTail.completionTokens,
                            "total_tokens": usageTail.totalTokens,
                        ],
                    ]) {
                        try await sse(ByteBuffer(data: data))
                    }
                    try await sseRaw("[DONE]")
                    finishedCleanly = true
                }
                if finishedCleanly { break }
            }
            if !finishedCleanly {
                // Client disconnect or queue cancellation: abort the engine turn.
                task.cancel()
                throw CancellationError()
            }
            try await writer.finish(nil)
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-store"
        headers[.connection] = "keep-alive"
        return Response(status: .ok, headers: headers, body: body)
    }

    // MARK: Anthropic non-streaming handling

    nonisolated private static func anthropicNonStreamingResponse(plan: TurnPlan, queue: TurnExecutionQueue) async throws -> AnthropicMessageResponse {
        let result = try await runTurn(plan: plan, queue: queue)

        var blocks: [AnthropicContentBlock] = []
        if !result.thinkingText.isEmpty {
            blocks.append(.thinking(result.thinkingText))
        }
        if !result.responseText.isEmpty {
            blocks.append(.text(result.responseText))
        }
        for call in result.parsedToolCalls {
            blocks.append(.toolUse(id: "toolu-\(UUID().uuidString.prefix(12))", name: call.name, inputJSON: toolArgumentsJSON(call)))
        }
        if blocks.isEmpty {
            blocks.append(.text(""))
        }

        return AnthropicMessageResponse(
            id: "msg-\(UUID().uuidString.prefix(24))",
            model: plan.modelName,
            content: blocks,
            stopReason: AnthropicAdapter.finishReasonString(result.finishReason, hasToolCalls: !result.parsedToolCalls.isEmpty),
            stopSequence: nil,
            usage: .init(inputTokens: plan.promptTokenCount, outputTokens: result.tokensGenerated)
        )
    }

    // MARK: Anthropic streaming handling

    nonisolated private static func anthropicStreamingResponse(apiRequest: AnthropicMessagesRequest, queue: TurnExecutionQueue) -> Response {
        let encoder = JSONEncoder()
        let id = "msg-\(UUID().uuidString.prefix(24))"
        let model = apiRequest.model ?? CompletionEngine.shared.loadedModelInfo?.displayName ?? "dynamoe"

        let body = ResponseBody { writer in
            func sse<E: Encodable>(_ eventName: String, _ payload: E) async throws {
                var frame = ByteBuffer(string: "event: \(eventName)\n")
                frame.writeImmutableBuffer(ByteBuffer(data: try encoder.encode(payload)))
                frame.writeString("\n\n")
                try await writer.write(frame)
            }
            func ping() async throws {
                try await sse("ping", AnthropicPingEvent())
            }
            func errorEvent(_ message: String) async throws {
                try await sse("error", AnthropicErrorEvent(error: .init(type: "api_error", message: message)))
            }

            var sentStart = false
            var sentThinking = ""
            var sentContent = ""
            var thinkingIndex: Int?
            var contentIndex: Int?
            var nextIndex = 0
            let promptTokensBox = PromptCountBox()

            func ensureStart() async throws {
                if !sentStart {
                    sentStart = true
                    try await sse("message_start", AnthropicMessageStartEvent(message: .init(
                        id: id,
                        model: model,
                        usage: .init(inputTokens: promptTokensBox.get(), outputTokens: 0)
                    )))
                }
            }
            func openThinking() async throws {
                guard thinkingIndex == nil else { return }
                thinkingIndex = nextIndex
                nextIndex += 1
                try await sse("content_block_start", AnthropicContentBlockStartEvent(index: thinkingIndex ?? 0, contentBlock: .thinking))
            }
            func openContent() async throws {
                guard contentIndex == nil else { return }
                contentIndex = nextIndex
                nextIndex += 1
                try await sse("content_block_start", AnthropicContentBlockStartEvent(index: contentIndex ?? 0, contentBlock: .text))
            }
            func closeOpenBlocks() async throws {
                if let ti = thinkingIndex {
                    try await sse("content_block_stop", AnthropicContentBlockStopEvent(index: ti))
                }
                if let ci = contentIndex {
                    try await sse("content_block_stop", AnthropicContentBlockStopEvent(index: ci))
                }
            }

            let (relay, task) = await queue.streamTurn(queueWaitTimeout: resolvedQueueWaitTimeout) { emit in
                do {
                    let plan = try ServerController.buildTurnPlan(apiRequest)
                    promptTokensBox.set(plan.promptTokenCount)
                    let launch = try CompletionEngine.shared.launchTurn(plan.request)
                    if Task.isCancelled {
                        launch.handle.cancel()
                        return
                    }
                    await withTaskCancellationHandler {
                        for await event in launch.stream {
                            emit(.engine(event))
                        }
                    } onCancel: {
                        launch.handle.cancel()
                    }
                } catch {
                    let message = (error as? ServerAPIError)?.message
                        ?? (error as? CompletionLaunchError)?.errorDescription
                        ?? error.localizedDescription
                    emit(.launchFailed(message))
                }
            }
            defer { task.cancel() }
            var finishedCleanly = false
            for await event in relay {
                switch event {
                case .ping:
                    try await ping()
                case .launchFailed(let message):
                    try await ensureStart()
                    try await closeOpenBlocks()
                    try await errorEvent(message)
                    finishedCleanly = true
                case .engine(.started):
                    try await ensureStart()
                case .engine(.priming):
                    try await ensureStart()
                    try await ping()
                case .engine(.prefillProgress(let status, _, _)):
                    if status != nil {
                        try await ping()
                    }
                case .engine(.frame(let frame)):
                    try await ensureStart()
                    if frame.thinkingText.count > sentThinking.count && frame.thinkingText.hasPrefix(sentThinking) {
                        try await openThinking()
                        let delta = String(frame.thinkingText.dropFirst(sentThinking.count))
                        sentThinking = frame.thinkingText
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: thinkingIndex ?? 0, delta: .thinking(delta)))
                    } else if frame.thinkingText.count > sentThinking.count {
                        // UTF-8 repair mid-stream: re-render rather than fork.
                        try await openThinking()
                        sentThinking = frame.thinkingText
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: thinkingIndex ?? 0, delta: .thinking(frame.thinkingText)))
                    }
                    if frame.responseText.count > sentContent.count && frame.responseText.hasPrefix(sentContent) {
                        try await openContent()
                        let delta = String(frame.responseText.dropFirst(sentContent.count))
                        sentContent = frame.responseText
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: contentIndex ?? 0, delta: .text(delta)))
                    } else if frame.responseText.count > sentContent.count {
                        try await openContent()
                        sentContent = frame.responseText
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: contentIndex ?? 0, delta: .text(frame.responseText)))
                    }
                case .engine(.failedTurn(let failure)):
                    try await ensureStart()
                    try await closeOpenBlocks()
                    try await errorEvent(failure.cancelled ? "Generation cancelled" : failure.message)
                    finishedCleanly = true
                case .engine(.finished(let result)):
                    try await ensureStart()
                    // Tail text the throttle never emitted.
                    if result.thinkingText.count > sentThinking.count && result.thinkingText.hasPrefix(sentThinking) {
                        try await openThinking()
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: thinkingIndex ?? 0, delta: .thinking(String(result.thinkingText.dropFirst(sentThinking.count)))))
                        sentThinking = result.thinkingText
                    }
                    if result.responseText.count > sentContent.count && result.responseText.hasPrefix(sentContent) {
                        try await openContent()
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: contentIndex ?? 0, delta: .text(String(result.responseText.dropFirst(sentContent.count)))))
                        sentContent = result.responseText
                    }
                    try await closeOpenBlocks()
                    for call in result.parsedToolCalls {
                        let toolIndex = nextIndex
                        nextIndex += 1
                        let toolId = "toolu-\(UUID().uuidString.prefix(12))"
                        try await sse("content_block_start", AnthropicContentBlockStartEvent(index: toolIndex, contentBlock: .toolUse(id: toolId, name: call.name)))
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: toolIndex, delta: .inputJSON(toolArgumentsJSON(call))))
                        try await sse("content_block_stop", AnthropicContentBlockStopEvent(index: toolIndex))
                    }
                    let stopReason = AnthropicAdapter.finishReasonString(result.finishReason, hasToolCalls: !result.parsedToolCalls.isEmpty)
                    try await sse("message_delta", AnthropicMessageDeltaEvent(
                        delta: .init(stopReason: stopReason, stopSequence: nil),
                        usage: .init(outputTokens: result.tokensGenerated)
                    ))
                    try await sse("message_stop", AnthropicMessageStopEvent())
                    finishedCleanly = true
                }
                if finishedCleanly { break }
            }
            if !finishedCleanly {
                // Client disconnect or queue cancellation: abort the engine turn.
                task.cancel()
                throw CancellationError()
            }
            try await writer.finish(nil)
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-store"
        headers[.connection] = "keep-alive"
        return Response(status: .ok, headers: headers, body: body)
    }

    // MARK: Health

    nonisolated private static func healthBody() -> ByteBuffer {
        let loaded = CompletionEngine.shared.hasLoadedModel
        let name = CompletionEngine.shared.loadedModelInfo?.displayName ?? ""
        let payload: [String: Any] = [
            "status": "ok",
            "model_loaded": loaded,
            "model": name,
            "engine_active": CompletionEngine.shared.isActive,
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data()
        return ByteBuffer(data: data)
    }
}
