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
    /// Carries the wire-shaped error so both protocol adapters render their
    /// native envelope (status/type included).
    case launchFailed(ServerAPIError)
}

/// FIFO queue over the engine's single-flight generation slot. Each request
/// enqueues its whole turn; the work only begins once every earlier turn has
/// fully drained. Cancellation propagates to the owner task, which cancels
/// the engine turn handle. Queue-wait timeouts are delivered at their
/// deadline even when a predecessor is stalled; the timed-out turn leaves
/// the queue without running work, while turns queued behind it still wait
/// for their predecessors' drain.
actor TurnExecutionQueue {
    /// Chain element for successor turns: resolves only once every earlier
    /// operation's work has drained. A timed-out operation contributes no
    /// work, so its chain element resolves when its own predecessor drains.
    private var tail: Task<Void, Never> = Task {}
    private var enqueuedOperations = 0
    private var completedOperations = 0
    /// Live server-owned operations, each with a cancel invoker. Server
    /// shutdown cancels exactly these; engine turns launched by the chat UI
    /// never route through this queue and must not be touched.
    private var liveCancelBoxes: [UUID: ServerTurnCancelBox] = [:]

    /// Turns enqueued but not yet drained (queued or currently running) —
    /// surfaced by the Server settings dashboard.
    var pendingCount: Int {
        max(0, enqueuedOperations - completedOperations)
    }

    /// Cancels every live server-owned turn (queued or running). Called on
    /// server shutdown only.
    func cancelLiveWork() {
        for box in liveCancelBoxes.values {
            box.invoke()
        }
    }

    private func registerCancelBox(_ box: ServerTurnCancelBox) -> UUID {
        let id = UUID()
        liveCancelBoxes[id] = box
        return id
    }

    private func deregisterCancelBox(_ id: UUID) {
        liveCancelBoxes[id] = nil
    }

    /// Waits for the predecessor to drain, but gives up at `deadline` so a
    /// stalled or long-running leader cannot hold a timed-out request past
    /// its deadline. `Task.value` awaits ignore cancellation, so the
    /// predecessor's completion is observed by a detached watcher and raced
    /// against the deadline on a short poll cadence; the watcher is
    /// abandoned on timeout and completes on its own. Returning `true` means
    /// the slot is genuinely open — the FIFO barrier for real work is
    /// untouched, since callers run work only after this returns true.
    nonisolated private static func waitForSlot(_ previous: Task<Void, Never>, deadline: Date) async throws -> Bool {
        let drained = DrainFlag()
        let watcher = Task.detached(priority: .utility) { [previous, drained] in
            _ = await previous.value
            drained.mark()
        }
        defer { watcher.cancel() }
        while !drained.isDrained {
            try Task.checkCancellation()
            if Date() >= deadline {
                return false
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return true
    }

    /// Enqueues streaming work. Returns the event relay stream plus the task
    /// that owns the work (cancel it to abort a queued/running turn).
    /// `queueWaitTimeout` bounds how long the turn may sit behind earlier
    /// turns; when it expires the relay emits a launchFailed event instead of
    /// running the work — promptly, even if an earlier turn is stalled. Nil
    /// (the default) waits indefinitely.
    func streamTurn(
        _ work: @escaping @Sendable (_ emit: @escaping @Sendable (QueuedTurnEvent) -> Void) async -> Void,
        queueWaitTimeout: TimeInterval? = nil
    ) -> (AsyncStream<QueuedTurnEvent>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream.makeStream(of: QueuedTurnEvent.self, bufferingPolicy: .unbounded)
        let previous = tail
        enqueuedOperations += 1
        let deadline = queueWaitTimeout.map { Date().addingTimeInterval($0) }
        let cancelBox = ServerTurnCancelBox()
        let hookID = registerCancelBox(cancelBox)
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
            defer {
                // The heartbeat also serves the RUNNING turn: cancel it here, not
                // right after the previous turn drains, so the SSE stream keeps
                // emitting ping frames while a long prefill runs and clients
                // (and any intermediate proxies) never see a silent socket.
                self.completedOperations += 1
                self.deregisterCancelBox(hookID)
                heartbeat.cancel()
            }
            do {
                let slotOpened: Bool
                if let deadline {
                    slotOpened = try await Self.waitForSlot(previous, deadline: deadline)
                } else {
                    await previous.value
                    slotOpened = true
                }
                guard slotOpened else {
                    continuation.yield(.launchFailed(ServerAPIError.requestTimeout("Request timed out waiting in the generation queue")))
                    continuation.finish()
                    return
                }
            } catch {
                // waitForSlot throws only on cancellation.
                continuation.finish()
                return
            }
            guard !Task.isCancelled else {
                continuation.finish()
                return
            }
            await work { continuation.yield($0) }
            continuation.finish()
        }
        cancelBox.set { task.cancel() }
        // Successors wait for this turn's full drain, not merely its task
        // completion: a turn that timed out without running still holds its
        // predecessors' drain for the turns queued behind it.
        let gate = Task<Void, Never> { [previous, task] in
            await previous.value
            _ = await task.value
        }
        // Keep the chain element alive regardless of this task's outcome.
        tail = gate
        return (stream, task)
    }

    /// Enqueues value-returning work (non-streaming requests and other
    /// engine-facing endpoints). `queueWaitTimeout` bounds how long the work
    /// may sit behind earlier turns; when it expires the perform throws a
    /// request-timeout error instead of running the work — promptly, even if
    /// an earlier turn is stalled. Nil (the default) waits indefinitely.
    /// Cancelling the caller cancels the queued work: a cancelled request
    /// neither launches after its predecessor drains nor leaves a running
    /// engine turn unreachable by the work's own cancellation hook.
    func perform<T: Sendable>(
        _ work: @escaping @Sendable () async throws -> T,
        queueWaitTimeout: TimeInterval? = nil
    ) async throws -> T {
        let previous = tail
        enqueuedOperations += 1
        let deadline = queueWaitTimeout.map { Date().addingTimeInterval($0) }
        let cancelBox = ServerTurnCancelBox()
        let hookID = registerCancelBox(cancelBox)
        let task = Task<T, Error> { [previous, deadline] in
            defer {
                self.completedOperations += 1
                self.deregisterCancelBox(hookID)
            }
            let slotOpened: Bool
            if let deadline {
                slotOpened = try await Self.waitForSlot(previous, deadline: deadline)
            } else {
                await previous.value
                slotOpened = true
            }
            guard slotOpened else {
                throw ServerAPIError.requestTimeout("Request timed out waiting in the generation queue")
            }
            try Task.checkCancellation()
            return try await work()
        }
        cancelBox.set { task.cancel() }
        let gate = Task<Void, Never> { [previous, task] in
            await previous.value
            _ = try? await task.value
        }
        tail = gate
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }
}

/// Lock-guarded one-shot flag so a queue slot wait can poll its
/// predecessor's drain without a data race.
private final class DrainFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var drained = false

    func mark() {
        lock.lock()
        drained = true
        lock.unlock()
    }

    var isDrained: Bool {
        lock.lock()
        defer { lock.unlock() }
        return drained
    }
}

/// Cancel invoker for a queued turn's shutdown registry entry. The work task
/// does not exist yet at enqueue time, so the real cancel closure is
/// installed right after task creation; `invoke` before that is a no-op.
private final class ServerTurnCancelBox: @unchecked Sendable {
    private let lock = NSLock()
    private var cancel: (() -> Void)?

    func set(_ newCancel: @escaping () -> Void) {
        lock.lock()
        cancel = newCancel
        lock.unlock()
    }

    func invoke() {
        lock.lock()
        let target = cancel
        lock.unlock()
        target?()
    }
}

// MARK: - Completed-turn memory

/// Memory of the last completed server turn, mirroring the engine's
/// single-slot prefix pin: one session's prompt+generation is pinned at a
/// time, so one record is all that can ever be reused.
///
/// The pin covers the *generated* tokens, which include the model's reasoning.
/// A stateless API client echoes back only the visible answer, so without this
/// record a follow-up turn cannot reproduce the pinned prefix and every
/// multi-turn request re-prefills from scratch — on a hybrid (GDN) model that
/// is the whole prompt, because its recurrent state only matches at the exact
/// pinned boundary. The in-app chat loop has this information first-hand (it
/// re-sends `TurnResult.rawDecodedText`); the server has to keep it.
final class ServerTurnMemory: @unchecked Sendable {
    static let shared = ServerTurnMemory()

    struct Record: Sendable {
        var sessionId: UUID
        var modelName: String
        /// The turn exactly as generated, reasoning and control markers included.
        var rawDecodedText: String
        /// What the client is expected to echo back as its assistant content.
        var visibleResponse: String
        var promptTokenIds: [UInt32]
        var generatedTokenIds: [UInt32]
    }

    private let lock = NSLock()
    private var record: Record?

    init() {}

    func store(_ new: Record) {
        lock.lock()
        record = new
        lock.unlock()
    }

    func clear() {
        lock.lock()
        record = nil
        lock.unlock()
    }

    /// The recorded raw turn text for `sessionId`, available only when the
    /// record clearly belongs to this turn and the client's echoed assistant
    /// text is compatible with what was generated.
    ///
    /// `turnCloser` is the family's assistant-turn close tag. It is dropped
    /// before comparing and returning: the model emits it as tokens, but a
    /// client never sees it (it is not part of `content`), and the renderer
    /// appends it again when the turn is re-rendered.
    func recordedRawTurn(sessionId: UUID, modelName: String, echoedByClient: String, turnCloser: String?) -> String? {
        lock.lock()
        let current = record
        lock.unlock()
        guard let current,
              current.sessionId == sessionId,
              current.modelName == modelName else { return nil }
        let raw = Self.droppingTrailingCloser(current.rawDecodedText, closer: turnCloser)
        guard Self.isCompatible(echo: echoedByClient, visibleResponse: current.visibleResponse, rawTurn: raw) else {
            return nil
        }
        return raw
    }

    /// Drops a trailing assistant-turn closer, if one is present.
    nonisolated static func droppingTrailingCloser(_ text: String, closer: String?) -> String {
        guard let closer, !closer.isEmpty, text.hasSuffix(closer) else { return text }
        return String(text.dropLast(closer.count))
    }

    /// Whether a client's echo of the assistant turn is consistent with the
    /// recorded turn. A client may echo the visible answer, the full raw turn
    /// (when it kept the reasoning), or nothing at all (assistant tool-call
    /// turns carry the call structure instead of text). Anything else means the
    /// history was edited or the session key collided, so the record is not safe
    /// to splice in.
    nonisolated static func isCompatible(echo: String, visibleResponse: String, rawTurn: String) -> Bool {
        let client = echo.trimmingCharacters(in: .whitespacesAndNewlines)
        if client.isEmpty { return true }
        if client == visibleResponse.trimmingCharacters(in: .whitespacesAndNewlines) { return true }
        return rawTurn.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix(client)
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
    static let inlineReasoningKey = "dynamoe_server_inline_reasoning"
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

    /// When on, the OpenAI endpoint folds reasoning into `content` inside the
    /// canonical think tags instead of reporting it out-of-band as
    /// `reasoning_content`. Most OpenAI-compatible clients ignore that
    /// extension, so inlining is what makes reasoning visible in them — and
    /// because the tags then travel in `content`, a client echoes them back and
    /// the next turn can reproduce the pinned generation. Absent → off (the
    /// out-of-band field stays the default, matching DeepSeek-style clients).
    nonisolated static var resolvedInlineReasoning: Bool {
        UserDefaults.standard.bool(forKey: inlineReasoningKey)
    }

    /// Composes the inline reasoning form of a response: the reasoning followed
    /// by the canonical close tag and the visible answer. The *open* tag is
    /// deliberately omitted — it already sits at the end of the prompt, so
    /// emitting it here would double it when the turn is re-rendered out of
    /// history and the pinned prefix would no longer match. This shape is the
    /// model's own generation, so a client that echoes it back reproduces the
    /// pin exactly. Returns the answer untouched when there is no reasoning.
    nonisolated static func inlineReasoningText(thinking: String, response: String) -> String {
        let think = thinking.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !think.isEmpty else { return response }
        let answer = response.trimmingCharacters(in: .whitespacesAndNewlines)
        let tagged = think + TurnTextSplitter.canonicalReasoningCloseTag
        return answer.isEmpty ? tagged : tagged + "\n\n" + answer
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
        // Cancel only turns the server owns. The chat UI drives the engine
        // directly (not through this queue), so its in-flight response must
        // survive a server restart or settings change.
        await queue.cancelLiveWork()
        // The recorded turn belongs to the server's lifetime; a restart starts
        // a fresh session history anyway.
        ServerTurnMemory.shared.clear()
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

    /// Maps synchronous launch failures onto the protocol-correct error:
    /// request-shape problems (prompt too long for this machine's KV budget)
    /// are 4xx, engine state problems are 503.
    nonisolated private static func wireLaunchError(_ error: Error) -> ServerAPIError {
        if let apiError = error as? ServerAPIError { return apiError }
        if let launchError = error as? CompletionLaunchError {
            let message = launchError.errorDescription ?? error.localizedDescription
            return launchError.isClientError ? ServerAPIError.badRequest(message) : ServerAPIError.serviceUnavailable(message)
        }
        return ServerAPIError.internalError(error.localizedDescription)
    }

    /// Adapter-neutral inputs to the shared turn-plan builder. Each protocol
    /// adapter resolves its own wire-format defaults (thinking opt-in vs
    /// opt-out, sampler extras, stop-sequence spelling, session identity)
    /// before handing the resolved values to the shared core.
    struct TurnPlanInputs: Sendable {
        var systemPrompt: String
        /// Role-tagged conversation history. The server renders it with the
        /// model family's native role boundaries so later turns extend the
        /// previous turn's pinned prefix instead of re-prefilling.
        var conversation: [ChatPromptFormatter.ConversationTurn]
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
            // The chat UI threads a per-conversation date for prefix reuse; the
            // server needs the same discipline. A fresh Date() here renders a
            // new "at 3:41 PM" into the prompt's temporal block every turn,
            // splitting the common token prefix a sentence in and forcing
            // hybrid (GDN) sessions — which cannot rewind linear state — into
            // a full ~30 s re-prefill on every agent step.
            currentDate: PrefixCacheManager.shared.asOfDate(sessionId: inputs.sessionId)
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        if inputs.hasTools, let declaration = inputs.toolDeclaration {
            cleanSystem = cleanSystem.isEmpty ? declaration : cleanSystem + "\n\n" + declaration
        }

        let thinkingEnabled = info.supportsThinking && inputs.thinkingRequested
        // Splice in the raw generation recorded for this session (reasoning
        // included) so the rendered prompt reproduces the pinned
        // prompt+generation; a client's echo of the visible answer alone cannot.
        let conversation = Self.substitutingRecordedTurn(
            inputs.conversation,
            sessionId: inputs.sessionId,
            modelName: info.displayName,
            modelConfig: engine.promptModelConfig
        )
        // A single plain user turn renders exactly as before (native template
        // with zero wrapper noise). A real history renders with the family's
        // native role boundaries, so this turn's formatted prompt is a strict
        // extension of the previous turn's prompt+generation and the pinned
        // KV prefix stays reusable across turns.
        let formattedPrompt: String
        if conversation.count == 1, conversation[0].role == .user {
            formattedPrompt = ChatPromptFormatter.formatPrompt(
                prompt: conversation[0].text,
                cleanSystem: cleanSystem,
                thinkingEnabled: thinkingEnabled,
                modelSupportsThinking: info.supportsThinking,
                modelConfig: engine.promptModelConfig
            )
        } else {
            formattedPrompt = ChatPromptFormatter.formatConversation(
                turns: conversation,
                cleanSystem: cleanSystem,
                thinkingEnabled: thinkingEnabled,
                modelSupportsThinking: info.supportsThinking,
                modelConfig: engine.promptModelConfig
            )
        }

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

        let memory = TurnMemorySettings.persisted(defaults)

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
        let (systemPrompt, turns) = try OpenAIAdapter.splitMessages(apiRequest.messages)
        let hasTools = !(apiRequest.tools ?? []).isEmpty && apiRequest.toolChoice != "none"
        return try buildTurnPlan(TurnPlanInputs(
            systemPrompt: systemPrompt,
            conversation: turns,
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
        let turns = try AnthropicAdapter.splitMessages(apiRequest.messages)
        let hasTools = !(apiRequest.tools ?? []).isEmpty && apiRequest.toolChoiceType != "none"
        return try buildTurnPlan(TurnPlanInputs(
            systemPrompt: apiRequest.systemText ?? "",
            conversation: turns,
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

    /// Remembers one completed turn so the next turn for the same session can
    /// reproduce its generated tokens — reasoning included — instead of trusting
    /// the client's echo of the visible answer alone.
    nonisolated private static func recordCompletedTurn(plan: TurnPlan, result: TurnResult) {
        // No session identity means no pin to extend, so there is nothing to
        // reuse (the server always sets one from the adapter's session key).
        guard let sessionId = plan.request.sessionId else { return }
        ServerTurnMemory.shared.store(.init(
            sessionId: sessionId,
            modelName: plan.modelName,
            rawDecodedText: result.rawDecodedText,
            visibleResponse: result.responseText,
            promptTokenIds: result.promptTokenIds,
            generatedTokenIds: result.generatedTokenIds
        ))
    }

    /// Replaces the client's echo of the last assistant turn with the raw
    /// generation recorded for this session, so the rendered prompt reproduces
    /// the pinned prompt+generation (and the prefix stays reusable). Falls back
    /// to the client's own text whenever the record does not clearly belong to
    /// this turn — see `ServerTurnMemory.isCompatible`.
    nonisolated private static func substitutingRecordedTurn(
        _ conversation: [ChatPromptFormatter.ConversationTurn],
        sessionId: UUID,
        modelName: String,
        modelConfig: ModelConfig?
    ) -> [ChatPromptFormatter.ConversationTurn] {
        let closer = ChatPromptFormatter.assistantTurnCloseTag(modelConfig: modelConfig)
        guard let index = conversation.lastIndex(where: { $0.role == .assistant }),
              let recorded = ServerTurnMemory.shared.recordedRawTurn(
                  sessionId: sessionId,
                  modelName: modelName,
                  echoedByClient: conversation[index].text,
                  turnCloser: closer
              ) else { return conversation }
        var turns = conversation
        turns[index].text = recorded
        return turns
    }

    /// Runs a planned turn through the FIFO queue to completion. Shared by
    /// both adapters' non-streaming paths.
    nonisolated private static func runTurn(plan: TurnPlan, queue: TurnExecutionQueue) async throws -> TurnResult {
        try await queue.perform(queueWaitTimeout: resolvedQueueWaitTimeout) {
            () async throws -> TurnResult in
            let launch: CompletionLaunch
            do {
                launch = try CompletionEngine.shared.launchTurn(plan.request)
            } catch {
                throw wireLaunchError(error)
            }
            return try await withTaskCancellationHandler {
                for await event in launch.stream {
                    switch event {
                    case .failedTurn(let failure):
                        if failure.cancelled { throw ServerAPIError.serviceUnavailable("Generation cancelled") }
                        throw ServerAPIError.internalError(failure.message)
                    case .finished(let result):
                        Self.recordCompletedTurn(plan: plan, result: result)
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

        // Inline-reasoning mode folds the reasoning into `content` (wrapped in
        // the canonical think tags) so clients that ignore `reasoning_content`
        // still display it, and echo it back on the next turn.
        let inlineReasoning = Self.resolvedInlineReasoning
        let message = OpenAIResponseMessage(
            content: inlineReasoning
                ? Self.inlineReasoningText(thinking: result.thinkingText, response: result.responseText)
                : result.responseText,
            reasoningContent: (!inlineReasoning && !result.thinkingText.isEmpty) ? result.thinkingText : nil,
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
    /// OpenAI `function.arguments` JSON-string field. `rawArguments` can hold
    /// parser artifacts (a Swift dictionary description, or the whole hermetic
    /// call envelope), so the parsed dictionary wins whenever it is available
    /// and raw text is trusted only if it parses back as a JSON object.
    nonisolated static func toolArgumentsJSON(_ call: ParsedToolCall) -> String {
        if !call.arguments.isEmpty,
           let data = try? JSONSerialization.data(withJSONObject: call.arguments, options: [.sortedKeys]),
           let json = String(data: data, encoding: .utf8) {
            return json
        }
        if !call.rawArguments.isEmpty,
           let data = call.rawArguments.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data)) is [String: Any] {
            return call.rawArguments
        }
        return "{}"
    }

    /// Length of the longest suffix of `text` that is a prefix of (or exactly)
    /// one of `stopSequences` — the portion that could still grow into a
    /// match the engine will trim, so it must not reach an append-only
    /// streaming client yet.
    nonisolated static func unsafeSuffixLength(_ text: String, stopSequences: [String]) -> Int {
        var longest = 0
        for stop in stopSequences where !stop.isEmpty {
            var length = min(text.count, stop.count)
            while length > longest {
                if stop.hasPrefix(text.suffix(length)) {
                    longest = length
                    break
                }
                length -= 1
            }
        }
        return longest
    }

    /// Append-safe delta of `accumulated` beyond the already-delivered `sent`.
    /// Empty when the engine's snapshot no longer extends what was sent: a
    /// streaming client can only append, so re-sending a repaired snapshot
    /// would duplicate the delivered prefix. Also holds back the suffix that
    /// could still complete a configured stop sequence, so the engine's
    /// eventual trim cannot leave phantom text in the client's assembled
    /// message. Pass empty `stopSequences` at terminal flush, where the
    /// engine has already trimmed matches and every remaining character is
    /// final.
    nonisolated static func safeStreamDelta(accumulated: String, sent: String, stopSequences: [String]) -> String {
        guard accumulated.hasPrefix(sent) else { return "" }
        let delta = accumulated.dropFirst(sent.count)
        guard !delta.isEmpty else { return "" }
        let hold = min(unsafeSuffixLength(accumulated, stopSequences: stopSequences), delta.count)
        return String(delta.dropLast(hold))
    }

    // MARK: Streaming handling

    nonisolated private static func streamingResponse(apiRequest: OpenAIChatRequest, queue: TurnExecutionQueue) -> Response {
        let encoder = JSONEncoder()
        let id = "chatcmpl-\(UUID().uuidString.prefix(24))"
        let created = Int(Date().timeIntervalSince1970)
        let stopSequences = (apiRequest.stop ?? []).filter { !$0.isEmpty }
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

            // Inline-reasoning mode: reasoning is streamed as `content` inside
            // the canonical think tags, so clients that ignore the
            // `reasoning_content` extension still display it — and because the
            // tags travel in `content`, they come back on the next turn and the
            // pinned generation can be reproduced.
            let inlineReasoning = Self.resolvedInlineReasoning
            var inlineReasoningSeen = false
            var inlineReasoningClosed = false

            func emitReasoning(_ delta: String) async throws {
                // Once answer text has gone out, the reasoning can no longer be
                // closed ahead of it; fall back to the out-of-band field rather
                // than emitting the tag out of order.
                guard inlineReasoning, sentContent.isEmpty else {
                    try await sse(chunkBuffer(.init(role: nil, content: nil, reasoningContent: delta, toolCalls: nil), finishReason: nil))
                    return
                }
                inlineReasoningSeen = true
                try await sse(chunkBuffer(.init(role: nil, content: delta, reasoningContent: nil, toolCalls: nil), finishReason: nil))
            }

            /// Closes the inline reasoning before any answer text, so the tag
            /// never straddles the response. Emitting only the close tag keeps
            /// the stream byte-identical to the generation the prompt setup
            /// implies (the open tag is already the end of the prompt).
            func closeInlineReasoning() async throws {
                guard inlineReasoning, inlineReasoningSeen, !inlineReasoningClosed else { return }
                inlineReasoningClosed = true
                let close = TurnTextSplitter.canonicalReasoningCloseTag
                try await sse(chunkBuffer(.init(role: nil, content: close + "\n\n", reasoningContent: nil, toolCalls: nil), finishReason: nil))
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
                            if case .finished(let result) = event {
                                ServerController.recordCompletedTurn(plan: plan, result: result)
                            }
                            emit(.engine(event))
                        }
                    } onCancel: {
                        launch.handle.cancel()
                    }
                } catch {
                    emit(.launchFailed(wireLaunchError(error)))
                }
            }
            defer { task.cancel() }
            var finishedCleanly = false
            for await event in relay {
                switch event {
                case .ping(let comment):
                    try await ping(comment)
                case .launchFailed(let apiError):
                    let envelope = OpenAIErrorEnvelope(error: .init(message: apiError.message, type: apiError.wireType.rawValue, code: nil))
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
                    // Append-only safety: a snapshot that no longer extends the
                    // delivered text is skipped (clients cannot replace), and
                    // any suffix that could still grow into a stop sequence is
                    // withheld until later frames prove it safe.
                    let thinkingDelta = Self.safeStreamDelta(accumulated: frame.thinkingText, sent: sentThinking, stopSequences: stopSequences)
                    if !thinkingDelta.isEmpty {
                        sentThinking += thinkingDelta
                        try await emitReasoning(thinkingDelta)
                    }
                    let contentDelta = Self.safeStreamDelta(accumulated: frame.responseText, sent: sentContent, stopSequences: stopSequences)
                    if !contentDelta.isEmpty {
                        try await closeInlineReasoning()
                        sentContent += contentDelta
                        try await sse(chunkBuffer(.init(role: nil, content: contentDelta, reasoningContent: nil, toolCalls: nil), finishReason: nil))
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
                    // Tail text the throttle never emitted. The engine has
                    // already trimmed stop matches from the final text, so no
                    // hold-back applies here.
                    let tailThinking = Self.safeStreamDelta(accumulated: result.thinkingText, sent: sentThinking, stopSequences: [])
                    if !tailThinking.isEmpty {
                        try await emitReasoning(tailThinking)
                    }
                    try await closeInlineReasoning()
                    let tailContent = Self.safeStreamDelta(accumulated: result.responseText, sent: sentContent, stopSequences: [])
                    if !tailContent.isEmpty {
                        try await sse(chunkBuffer(.init(role: nil, content: tailContent, reasoningContent: nil, toolCalls: nil), finishReason: nil))
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
            stopSequence: result.matchedStopSequence,
            usage: .init(inputTokens: plan.promptTokenCount, outputTokens: result.tokensGenerated)
        )
    }

    // MARK: Anthropic streaming handling

    nonisolated private static func anthropicStreamingResponse(apiRequest: AnthropicMessagesRequest, queue: TurnExecutionQueue) -> Response {
        let encoder = JSONEncoder()
        let id = "msg-\(UUID().uuidString.prefix(24))"
        let model = apiRequest.model ?? CompletionEngine.shared.loadedModelInfo?.displayName ?? "dynamoe"
        let stopSequences = (apiRequest.stopSequences ?? []).filter { !$0.isEmpty }

        let body = ResponseBody { writer in
            func sse<E: Encodable>(_ eventName: String, _ payload: E) async throws {
                var frame = ByteBuffer(string: "event: \(eventName)\ndata: ")
                frame.writeImmutableBuffer(ByteBuffer(data: try encoder.encode(payload)))
                frame.writeString("\n\n")
                try await writer.write(frame)
            }
            func ping() async throws {
                try await sse("ping", AnthropicPingEvent())
            }
            func errorEvent(_ message: String, type: String = "api_error") async throws {
                try await sse("error", AnthropicErrorEvent(error: .init(type: type, message: message)))
            }

            var sentStart = false
            var sentThinking = ""
            var sentContent = ""
            var thinkingIndex: Int?
            var contentIndex: Int?
            // Anthropic blocks follow a strict sequential start/delta/stop
            // lifecycle; the flags keep terminal cleanup from stopping a
            // block twice.
            var thinkingClosed = false
            var contentClosed = false
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
            func closeThinkingBlock() async throws {
                if let ti = thinkingIndex, !thinkingClosed {
                    // Anthropic closes a thinking block with a signature_delta:
                    // clients echo it back so the reasoning survives the round
                    // trip (SDKs drop unsigned thinking blocks).
                    if !sentThinking.isEmpty {
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(
                            index: ti,
                            delta: .signature(AnthropicAdapter.thinkingSignature(for: sentThinking))
                        ))
                    }
                    try await sse("content_block_stop", AnthropicContentBlockStopEvent(index: ti))
                    thinkingClosed = true
                }
            }
            func openContent() async throws {
                // Blocks are strictly sequential: the thinking block must
                // stop before the text block starts.
                try await closeThinkingBlock()
                guard contentIndex == nil else { return }
                contentIndex = nextIndex
                nextIndex += 1
                try await sse("content_block_start", AnthropicContentBlockStartEvent(index: contentIndex ?? 0, contentBlock: .text))
            }
            func closeOpenBlocks() async throws {
                try await closeThinkingBlock()
                if let ci = contentIndex, !contentClosed {
                    try await sse("content_block_stop", AnthropicContentBlockStopEvent(index: ci))
                    contentClosed = true
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
                            if case .finished(let result) = event {
                                ServerController.recordCompletedTurn(plan: plan, result: result)
                            }
                            emit(.engine(event))
                        }
                    } onCancel: {
                        launch.handle.cancel()
                    }
                } catch {
                    emit(.launchFailed(wireLaunchError(error)))
                }
            }
            defer { task.cancel() }
            var finishedCleanly = false
            for await event in relay {
                switch event {
                case .ping:
                    try await ping()
                case .launchFailed(let apiError):
                    try await ensureStart()
                    try await closeOpenBlocks()
                    try await errorEvent(apiError.message, type: apiError.wireType.rawValue)
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
                    // Append-only safety, mirroring the OpenAI writer: skip
                    // non-extending snapshots, hold back possible stop-sequence
                    // suffixes.
                    let thinkingDelta = Self.safeStreamDelta(accumulated: frame.thinkingText, sent: sentThinking, stopSequences: stopSequences)
                    if !thinkingDelta.isEmpty {
                        try await openThinking()
                        sentThinking += thinkingDelta
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: thinkingIndex ?? 0, delta: .thinking(thinkingDelta)))
                    }
                    let contentDelta = Self.safeStreamDelta(accumulated: frame.responseText, sent: sentContent, stopSequences: stopSequences)
                    if !contentDelta.isEmpty {
                        try await openContent()
                        sentContent += contentDelta
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: contentIndex ?? 0, delta: .text(contentDelta)))
                    }
                case .engine(.failedTurn(let failure)):
                    try await ensureStart()
                    try await closeOpenBlocks()
                    try await errorEvent(failure.cancelled ? "Generation cancelled" : failure.message)
                    finishedCleanly = true
                case .engine(.finished(let result)):
                    try await ensureStart()
                    // Tail text the throttle never emitted; the engine has
                    // already trimmed stop matches, so no hold-back applies.
                    let tailThinking = Self.safeStreamDelta(accumulated: result.thinkingText, sent: sentThinking, stopSequences: [])
                    if !tailThinking.isEmpty {
                        try await openThinking()
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: thinkingIndex ?? 0, delta: .thinking(tailThinking)))
                        sentThinking += tailThinking
                    }
                    let tailContent = Self.safeStreamDelta(accumulated: result.responseText, sent: sentContent, stopSequences: [])
                    if !tailContent.isEmpty {
                        try await openContent()
                        try await sse("content_block_delta", AnthropicContentBlockDeltaEvent(index: contentIndex ?? 0, delta: .text(tailContent)))
                        sentContent += tailContent
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
                    // Surface which requested sequence ended generation; nil
                    // (the JSON default) for every other stop reason.
                    let matchedStop = result.finishReason == .stoppedBySequence ? result.matchedStopSequence : nil
                    try await sse("message_delta", AnthropicMessageDeltaEvent(
                        delta: .init(stopReason: stopReason, stopSequence: matchedStop),
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
