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
/// fully drained AND the engine's one generation slot has been CLAIMED — the chat
/// UI and the agent harness launch turns straight through the engine, so they
/// occupy that slot without appearing in this FIFO and a request waits for them
/// the same way it waits for its predecessor. The claim is taken in the same step
/// that decides the slot is free and is handed to the turn's launch, so the
/// preparation between the two cannot lose the slot to a chat turn that starts in
/// between (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #84"). Cancellation propagates
/// to the owner task, which cancels the engine turn handle. Queue-wait timeouts
/// are delivered at their deadline even when a predecessor is stalled; the
/// timed-out turn leaves the queue without running work, while turns queued
/// behind it still wait for their predecessors' drain.
actor TurnExecutionQueue {
    /// How the queue asks whether the engine's single-flight generation slot is
    /// free. A probe rather than a direct read so the wait for an in-app
    /// generation can be tested without loading a model; production passes the
    /// default. It exists only to tell the two queue-wait timeouts apart — the
    /// claim below is what actually decides.
    private let slotProbe: @Sendable () async -> Bool

    /// How the queue takes the slot. Nil means somebody else holds it, in which
    /// case the turn keeps waiting rather than failing: the request has already
    /// invested its queue wait and a claim is bounded work, not a refusal.
    private let slotReserve: @Sendable () async -> TurnSlotClaim?

    init(
        slotProbe: @escaping @Sendable () async -> Bool = TurnExecutionQueue.engineSlotIsFree,
        slotReserve: @escaping @Sendable () async -> TurnSlotClaim? = TurnExecutionQueue.claimEngineSlot
    ) {
        self.slotProbe = slotProbe
        self.slotReserve = slotReserve
    }

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

    /// Why a turn stopped waiting for the engine's generation slot.
    private enum SlotWaitOutcome {
        /// The slot was claimed: run the work with this claim.
        case opened(TurnSlotClaim)
        /// An earlier SERVER turn was still draining at the deadline.
        case timedOutBehindTurns
        /// The deadline passed with a generation the app started holding the slot.
        case timedOutBehindInAppTurn
    }

    /// The engine's single-flight slot as it stands right now.
    ///
    /// Read through the engine rather than tracked here, because the chat UI and
    /// the agent harness call `CompletionEngine.launchTurn` directly and never
    /// route through this queue — so a slot this queue believes free is not
    /// necessarily free. A claim another caller holds reads as busy here too.
    nonisolated private static func engineSlotIsFree() async -> Bool {
        let active = await CompletionEngine.shared.isActive
        return !active
    }

    /// Takes the engine's generation slot, or nil if a turn already holds it.
    ///
    /// This is the queue's only way in: the turn is launched with the claim it
    /// returns, so the slot stays ours across the preparation between here and
    /// `CompletionEngine.launchTurn`.
    nonisolated private static func claimEngineSlot() async -> TurnSlotClaim? {
        await CompletionEngine.shared.reserveTurnSlot()
    }

    /// Waits for the engine's single-flight generation slot to be free: first for
    /// every earlier queued turn to drain, then for whoever else holds it.
    ///
    /// There is ONE slot, and an in-app generation occupies it without passing
    /// through this FIFO. Waiting for it rather than failing is what makes a
    /// request that arrives mid-answer queue instead of returning 503: the client
    /// asked first, the resource is busy for a bounded time, and the queue already
    /// has a deadline and a heartbeat. `deadline` bounds both waits, so a stalled
    /// predecessor or a very long in-app generation cannot hold a request past its
    /// budget — it is told, with a message that names which of the two it waited
    /// on. Returning `.opened` means the slot is genuinely free; callers run work
    /// only then, so the FIFO barrier for real work is untouched.
    nonisolated private static func waitForSlot(
        _ previous: Task<Void, Never>,
        deadline: Date?,
        slotProbe: @Sendable () async -> Bool,
        slotReserve: @Sendable () async -> TurnSlotClaim?
    ) async throws -> SlotWaitOutcome {
        let drained = DrainFlag()
        let watcher = Task.detached(priority: .utility) { [previous, drained] in
            _ = await previous.value
            drained.mark()
        }
        defer { watcher.cancel() }
        // `Task.value` awaits ignore cancellation, so an unbounded wait polls the
        // drained flag rather than awaiting the predecessor directly. With no
        // deadline (the user asked to wait indefinitely) there is nothing to race,
        // so await it properly and spend nothing.
        if let deadline {
            while !drained.isDrained {
                try Task.checkCancellation()
                if Date() >= deadline {
                    return .timedOutBehindTurns
                }
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
        } else {
            await previous.value
        }
        // Same deadline, same cadence: only reached while a generation is actually
        // running, so the idle case costs one MainActor hop. A probe that says free
        // followed by a claim that fails means somebody else took the slot in
        // between — that caller now owns it, so this one goes back to waiting
        // instead of launching into a refusal.
        while true {
            try Task.checkCancellation()
            if await slotProbe(), let claim = await slotReserve() {
                return .opened(claim)
            }
            if let deadline, Date() >= deadline {
                return .timedOutBehindInAppTurn
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    /// Enqueues streaming work. Returns the event relay stream plus the task
    /// that owns the work (cancel it to abort a queued/running turn).
    /// `queueWaitTimeout` bounds how long the turn may sit behind earlier
    /// turns OR behind a generation the app started (the queue is a FIFO over
    /// one engine slot, and the chat UI occupies that slot without passing
    /// through it); when it expires the relay emits a launchFailed event
    /// instead of running the work — promptly, even if an earlier turn is
    /// stalled. Nil (the default) waits indefinitely. `onSlotOpened` fires
    /// exactly once, the instant the FIFO slot opens and immediately before
    /// the work runs (never on the timeout or cancellation paths), mirroring
    /// `perform`: the streaming writers stamp their box's `queueWaitMs`
    /// through the hook — at the producer — because the relay can deliver an
    /// event late, so an SSE writer blocked on an earlier heartbeat would
    /// otherwise stamp generation time as queue wait.
    ///
    /// The work closure receives the engine-slot claim the queue took on its
    /// behalf; a turn launched from here must pass it to
    /// `CompletionEngine.launchTurn(_:claim:)` or the launch will be refused.
    func streamTurn(
        _ work: @escaping @Sendable (_ claim: TurnSlotClaim, _ emit: @escaping @Sendable (QueuedTurnEvent) -> Void) async -> Void,
        queueWaitTimeout: TimeInterval? = nil,
        onSlotOpened: (@Sendable () -> Void)? = nil
    ) -> (AsyncStream<QueuedTurnEvent>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream.makeStream(of: QueuedTurnEvent.self, bufferingPolicy: .unbounded)
        // Read and advance the chain with no suspension in between, exactly as
        // `perform` does: the gate is built first and waits on `drained`, which the
        // turn's own task resolves in its `defer`.
        let previous = tail
        enqueuedOperations += 1
        let deadline = queueWaitTimeout.map { Date().addingTimeInterval($0) }
        let drained = SlotDrain()
        let gate = Task<Void, Never> { [previous, drained] in
            await previous.value
            await drained.wait()
        }
        tail = gate

        let cancelBox = ServerTurnCancelBox()
        let hookID = registerCancelBox(cancelBox)
        let heartbeat = Task {
            // Queue-wait heartbeat so clients don't idle-timeout while a long
            // turn ahead of us — or a generation the app is running — holds the
            // single-flight slot.
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard !Task.isCancelled else { return }
                continuation.yield(.ping(comment: nil))
            }
        }
        let task = Task<Void, Never> { [previous, heartbeat, deadline, onSlotOpened, slotProbe, slotReserve, drained] in
            var heldClaim: TurnSlotClaim?
            defer {
                // A claim this turn never redeemed (a timeout, a cancellation, a
                // work closure that threw first) goes back here; one the launch
                // already released is a no-op.
                heldClaim?.release()
                // The heartbeat also serves the RUNNING turn: cancel it here, not
                // right after the previous turn drains, so the SSE stream keeps
                // emitting ping frames while a long prefill runs and clients
                // (and any intermediate proxies) never see a silent socket.
                self.completedOperations += 1
                self.deregisterCancelBox(hookID)
                heartbeat.cancel()
                drained.resolve()
            }
            let claim: TurnSlotClaim
            do {
                switch try await Self.waitForSlot(
                    previous, deadline: deadline, slotProbe: slotProbe, slotReserve: slotReserve
                ) {
                case .opened(let opened):
                    claim = opened
                case .timedOutBehindTurns:
                    continuation.yield(.launchFailed(ServerAPIError.requestTimeout(
                        "Request timed out waiting in the generation queue")))
                    continuation.finish()
                    return
                case .timedOutBehindInAppTurn:
                    continuation.yield(.launchFailed(ServerAPIError.requestTimeout(
                        "Request timed out waiting for a generation running in the DynaMoE app")))
                    continuation.finish()
                    return
                }
            } catch {
                // waitForSlot throws only on cancellation.
                continuation.finish()
                return
            }
            heldClaim = claim
            guard !Task.isCancelled else {
                continuation.finish()
                return
            }
            onSlotOpened?()
            await work(claim) { continuation.yield($0) }
            continuation.finish()
        }
        cancelBox.set { task.cancel() }
        return (stream, task)
    }

    /// Enqueues value-returning work (non-streaming requests and other
    /// engine-facing endpoints). `queueWaitTimeout` bounds how long the work
    /// may sit behind earlier turns OR behind a generation the app started;
    /// when it expires the perform throws a request-timeout error instead of
    /// running the work — promptly, even if an earlier turn is stalled. Nil
    /// (the default) waits indefinitely.
    /// `onSlotOpened` fires exactly once, the instant the FIFO slot opens and
    /// immediately before the work runs (never on the timeout or
    /// cancellation paths), so callers can split queue wait from generation
    /// time.
    /// Cancelling the caller cancels the queued work: a cancelled request
    /// neither launches after its predecessor drains nor leaves a running
    /// engine turn unreachable by the work's own cancellation hook.
    ///
    /// The work closure receives the engine-slot claim the queue took on its
    /// behalf; a turn launched from here must pass it to
    /// `CompletionEngine.launchTurn(_:claim:)` or the launch will be refused.
    func perform<T: Sendable>(
        _ work: @escaping @Sendable (_ claim: TurnSlotClaim) async throws -> T,
        queueWaitTimeout: TimeInterval? = nil,
        onSlotOpened: (@Sendable () -> Void)? = nil
    ) async throws -> T {
        // The chain is READ AND ADVANCED WITHOUT A SUSPENSION between the two, and
        // only value construction and a nonisolated lock box sit between them. That
        // atomicity is the queue: a successor that read a half-advanced `tail` would
        // chain off the turn BEFORE this one and run alongside it, which the engine
        // then refuses as a second active turn. Building the gate before the work
        // task is what makes it possible — the gate waits on `drained`, which the
        // task resolves in its own `defer`.
        let previous = tail
        enqueuedOperations += 1
        let deadline = queueWaitTimeout.map { Date().addingTimeInterval($0) }
        let drained = SlotDrain()
        let gate = Task<Void, Never> { [previous, drained] in
            await previous.value
            await drained.wait()
        }
        tail = gate

        let cancelBox = ServerTurnCancelBox()
        let hookID = registerCancelBox(cancelBox)
        let task = Task<T, Error> { [previous, deadline, slotProbe, slotReserve, drained] in
            var heldClaim: TurnSlotClaim?
            defer {
                // A claim this turn never redeemed — a timeout, a cancellation, a
                // work closure that threw — goes back here; one the launch already
                // released is a no-op.
                heldClaim?.release()
                self.completedOperations += 1
                self.deregisterCancelBox(hookID)
                drained.resolve()
            }
            let outcome = try await Self.waitForSlot(
                previous, deadline: deadline, slotProbe: slotProbe, slotReserve: slotReserve
            )
            let claim: TurnSlotClaim
            switch outcome {
            case .opened(let opened):
                claim = opened
            case .timedOutBehindTurns:
                throw ServerAPIError.requestTimeout("Request timed out waiting in the generation queue")
            case .timedOutBehindInAppTurn:
                throw ServerAPIError.requestTimeout(
                    "Request timed out waiting for a generation running in the DynaMoE app"
                )
            }
            heldClaim = claim
            try Task.checkCancellation()
            onSlotOpened?()
            return try await work(claim)
        }
        cancelBox.set { task.cancel() }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }
}

/// Lock-guarded one-shot flag so a queue slot wait can poll its
/// predecessor's drain without a data race.
///
/// `nonisolated` because every one of these boxes is touched from the queue
/// actor and from a nonisolated static wait: the target builds with
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so leaving them unannotated
/// makes them main-actor types called across an isolation boundary from the
/// queue's critical path.
private nonisolated final class DrainFlag: @unchecked Sendable {
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

/// One-shot completion signal for a queued turn.
///
/// The FIFO gate has to exist BEFORE the turn's task does — that is what lets the
/// queue read and advance its chain without a suspension in between — so the gate
/// waits on this instead of on `Task.value`. Resolving twice is free, and a
/// resolve that lands before the wait still releases it.
private nonisolated final class SlotDrain: @unchecked Sendable {
    private let lock = NSLock()
    private var resolved = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func resolve() {
        lock.lock()
        guard !resolved else {
            lock.unlock()
            return
        }
        resolved = true
        let pending = waiters
        waiters = []
        lock.unlock()
        for waiter in pending { waiter.resume() }
    }

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if resolved {
                lock.unlock()
                continuation.resume()
                return
            }
            waiters.append(continuation)
            lock.unlock()
        }
    }
}

/// Cancel invoker for a queued turn's shutdown registry entry. The work task
/// does not exist yet at enqueue time, so the real cancel closure is
/// installed right after task creation; `invoke` before that is a no-op.
///
/// `nonisolated` for the same reason as `DrainFlag`: the queue actor installs
/// and invokes it, and `cancelLiveWork()` must be able to run it from server
/// shutdown without crossing an actor.
private nonisolated final class ServerTurnCancelBox: @unchecked Sendable {
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
        /// Whether the recorded turn emitted tool calls. Such a turn is echoed
        /// back as the adapters' decorated call description rather than the raw
        /// tagged markup, so compatibility must accept that shape (for these
        /// turns only).
        var hadToolCalls: Bool = false
        /// How many leading conversation turns the record's prompt was rendered
        /// from, and a digest over them plus the rendered system prompt. The
        /// session key is only a hash of the conversation's opening message, so
        /// two unrelated conversations can share it; this is what proves the
        /// record actually belongs to the history on the wire before its raw
        /// generation is spliced in.
        var historyTurnCount: Int = 0
        var historyDigest: UInt64 = 0
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
    ///
    /// `conversation` and `systemPrompt` are the history the request actually
    /// carries. The session key alone cannot identify a conversation (it hashes
    /// only the opening message, which unrelated conversations commonly share),
    /// so the recorded history must match before the raw generation is trusted:
    /// a colliding conversation whose history diverges is rejected instead of
    /// having another conversation's private reasoning spliced into its prompt.
    func recordedRawTurn(
        sessionId: UUID,
        modelName: String,
        echoedByClient: String,
        turnCloser: String?,
        conversation: [ChatPromptFormatter.ConversationTurn] = [],
        systemPrompt: String = ""
    ) -> String? {
        lock.lock()
        let current = record
        lock.unlock()
        guard let current,
              current.sessionId == sessionId,
              current.modelName == modelName else { return nil }
        guard Self.historyMatches(
            systemPrompt: systemPrompt,
            conversation: conversation,
            turnCount: current.historyTurnCount,
            digest: current.historyDigest
        ) else { return nil }
        let raw = Self.droppingTrailingCloser(current.rawDecodedText, closer: turnCloser)
        let echoMatches = Self.isCompatible(echo: echoedByClient, visibleResponse: current.visibleResponse, rawTurn: raw)
            || (current.hadToolCalls && Self.isToolCallEcho(echoedByClient))
        guard echoMatches else { return nil }
        return raw
    }

    /// Whether the request's history still carries the leading turns the
    /// record's prompt was rendered from. A record with no recorded history
    /// (non-conversational callers) imposes no constraint.
    nonisolated static func historyMatches(
        systemPrompt: String,
        conversation: [ChatPromptFormatter.ConversationTurn],
        turnCount: Int,
        digest: UInt64
    ) -> Bool {
        guard turnCount > 0 else { return true }
        guard conversation.count >= turnCount else { return false }
        return historyDigest(systemPrompt: systemPrompt, turns: conversation[..<turnCount]) == digest
    }

    /// FNV-1a digest of the leading conversation turns plus the rendered system
    /// prompt. Cheap, order-sensitive, and stable across the turns of one
    /// conversation, which is all the turn memory needs to tell a follow-up
    /// from an unrelated conversation that merely shares a session key.
    nonisolated static func historyDigest(
        systemPrompt: String,
        turns: ArraySlice<ChatPromptFormatter.ConversationTurn>
    ) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        func mix(_ byte: UInt8) { hash = (hash ^ UInt64(byte)) &* 0x100000001b3 }
        func mixString(_ s: String) {
            for byte in s.utf8 { mix(byte) }
            mix(0x1f)
        }
        mixString(systemPrompt)
        for turn in turns {
            mixString(turn.role == .user ? "user" : "assistant")
            mixString(turn.text)
        }
        return hash
    }

    /// Whether a client's echoed assistant turn is the adapters' decorated
    /// tool-call description rather than model text. A faithful echo of a
    /// tool-call turn can only take this form, so it neither equals the
    /// recorded visible response nor tail-matches the raw tagged generation.
    nonisolated static func isToolCallEcho(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return [ChatPromptFormatter.openAIToolCallEchoPrefix, ChatPromptFormatter.anthropicToolCallEchoPrefix]
            .contains { trimmed.hasPrefix($0) }
    }

    /// Drops a trailing assistant-turn closer, if one is present.
    nonisolated static func droppingTrailingCloser(_ text: String, closer: String?) -> String {
        guard let closer, !closer.isEmpty, text.hasSuffix(closer) else { return text }
        return String(text.dropLast(closer.count))
    }

    /// Whether a client's echo of the assistant turn is consistent with the
    /// recorded turn. A client may echo the visible answer, the full raw turn
    /// (when it kept the reasoning), or nothing at all (assistant tool-call
    /// turns carry the call structure instead of text; those are recognized
    /// separately via `isToolCallEcho`). Anything else means the history was
    /// edited, so the record is not safe to splice in.
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

// MARK: - Request logging middleware

/// Wire-level request logging. Begins one log box per inbound request keyed
/// by the request context's id, records the row for every request the route
/// handler does NOT claim (auth failures, decode failures, queue timeouts,
/// non-streaming completions), and steps aside for streaming responses: the
/// streaming body closure claims its box before the route returns the 200
/// and records with the stream's terminal facts instead. Health probes get a
/// debug-level unified-log line only, so polling never floods the store.
struct ServerWireLogMiddleware<Context: RequestContext>: RouterMiddleware {
    @concurrent
    func handle(
        _ request: Request,
        context: Context,
        next: @concurrent (Request, Context) async throws -> Response
    ) async throws -> Response {
        let requestId = context.id
        let box = ServerRequestLogRecorder.shared.beginBox(
            requestId: requestId,
            method: request.method.rawValue,
            path: request.uri.path
        )
        defer { ServerRequestLogRecorder.shared.endBox(requestId: requestId) }
        do {
            let response = try await next(request, context)
            Self.finalize(box: box, path: request.uri.path, status: response.status.code, error: nil)
            return response
        } catch {
            let status = ServerController.recordedStatus(for: error)
            Self.finalize(box: box, path: request.uri.path, status: status, error: error)
            throw error
        }
    }

    nonisolated private static func finalize(box: ServerRequestLogBox, path: String, status: Int, error: (any Error)?) {
        if path == "/health" {
            DiagLog.debug(ServerRequestLogRecorder.requestCategory, "GET /health → \(status)")
            return
        }
        guard !box.isClaimed else { return }
        ServerRequestLogRecorder.shared.record(box.composeFinal(status: status, error: error))
    }
}

extension ServerController {
    /// The wire status an error rendered to the client: every
    /// `HTTPResponseError` — `ServerAPIError`, Hummingbird's own `HTTPError`,
    /// any conformer the router meets — carries its status on itself, so the
    /// row mirrors what the client actually received. Anything else still
    /// records as 500, matching the fallback the router itself uses.
    nonisolated static func recordedStatus(for error: any Error) -> Int {
        (error as? any HTTPResponseError)?.status.code ?? 500
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

/// Which protocol a request is answered in.
///
/// The two protocols share `GET /v1/models`: identical path, different body. A
/// route therefore cannot decide from its own path, and the request's headers are
/// the only evidence available, so this type names the decision once and every
/// caller that has to render a shape consults it — the models routes for their
/// body and their log row, and the error renderer for its envelope.
///
/// `nonisolated` because it is a value the routing decisions compare and render in
/// contexts that are not the main actor; without it the project's default isolation
/// makes the comparison itself a main-actor call.
nonisolated enum ServerWireProtocol: Sendable {
    case openai
    case anthropic

    /// The request log's spelling of the same decision.
    var rowProtocol: ServerRequestLog.Proto {
        switch self {
        case .openai: return .openai
        case .anthropic: return .anthropic
        }
    }
}

/// Server error that renders as the correct JSON envelope for the calling
/// protocol: the Anthropic `{"type":"error","error":{...}}` shape under
/// /v1/messages, the OpenAI `{"error":{...}}` shape everywhere else. Conforms
/// to HTTPResponseError so the router renders it directly with its status —
/// no route-level do/catch mapping required.
struct ServerAPIError: Error, HTTPResponseError, LocalizedError, Sendable {
    enum WireType: String, Sendable {
        case invalidRequest = "invalid_request_error"
        case authentication = "authentication_error"
        case notFound = "not_found_error"
        case requestTimeout = "request_timeout"
        case overloaded = "overloaded_error"
        case api = "api_error"
    }

    let status: HTTPResponse.Status
    let wireType: WireType
    let message: String
    /// The log-safe rendering of this error for the persisted row and the
    /// unified log. The client-facing `message` may echo request values back
    /// to the client (an invalid role, a demanded tool name) so the client can
    /// debug its own wire usage — content the request log must not persist
    /// when body capture is off and must never print publicly — so errors
    /// that reflect such values carry a log-safe variant here; every other
    /// error is its own log-safe message.
    let logSafeMessage: String?
    /// Protocol-level error code (OpenAI's `error.code`). A client can classify
    /// a rejection from this instead of pattern-matching prose; nil keeps the
    /// previous envelope shape. The Anthropic envelope has no code field.
    let code: String?
    /// What the request log and the unified log record for this error.
    var recordableMessage: String { logSafeMessage ?? message }

    var errorDescription: String? { message }

    static func badRequest(_ message: String, logSafeMessage: String? = nil) -> ServerAPIError {
        ServerAPIError(status: .badRequest, wireType: .invalidRequest, message: message, logSafeMessage: logSafeMessage, code: nil)
    }

    /// The turn could not be served at the context size it asked for: the KV
    /// cache this machine can provision has no room for the prompt (or for the
    /// prompt plus usable generation). OpenAI clients recognize this code.
    static func contextLengthExceeded(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .badRequest, wireType: .invalidRequest, message: message, logSafeMessage: nil, code: "context_length_exceeded")
    }

    /// The request carried content this server cannot process at all (image
    /// parts against a local weights model with no vision tower). A distinct
    /// code so clients can tell it apart from a malformed body.
    static func unsupportedContent(_ message: String) -> ServerAPIError {
        ServerAPIError(
            status: .badRequest,
            wireType: .invalidRequest,
            message: message,
            logSafeMessage: nil,
            code: "unsupported_content"
        )
    }

    static func unauthorized(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .unauthorized, wireType: .authentication, message: message, logSafeMessage: nil, code: nil)
    }

    /// The named model is not the one this server serves. A models route answers
    /// 404 rather than an empty list, because a client that asked for a model by
    /// name and silently received a different one is the failure the route exists
    /// to prevent.
    static func modelNotFound(_ model: String) -> ServerAPIError {
        ServerAPIError(
            status: .notFound,
            wireType: .notFound,
            message: "model \"\(model)\" does not exist",
            logSafeMessage: "the requested model is not the one this server serves",
            code: nil
        )
    }

    static func requestTimeout(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .requestTimeout, wireType: .requestTimeout, message: message, logSafeMessage: nil, code: nil)
    }

    static func serviceUnavailable(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .serviceUnavailable, wireType: .overloaded, message: message, logSafeMessage: nil, code: nil)
    }

    static func internalError(_ message: String) -> ServerAPIError {
        ServerAPIError(status: .internalServerError, wireType: .api, message: message, logSafeMessage: nil, code: nil)
    }

    /// The OpenAI spelling of this error's `type`. The protocols disagree on how a
    /// missing model is named — Anthropic has `not_found_error`, OpenAI keeps
    /// `invalid_request_error` and says `model_not_found` in `code` — and the
    /// status is 404 either way, so the difference belongs in the envelope rather
    /// than in the error.
    var openAIWireType: String {
        wireType == .notFound ? WireType.invalidRequest.rawValue : wireType.rawValue
    }

    /// The OpenAI spelling of this error's `code`, for the same reason.
    var openAICode: String? {
        code ?? (wireType == .notFound ? "model_not_found" : nil)
    }

    func response(from request: Request, context: some RequestContext) throws -> Response {
        var response: Response
        // Decided from the request, not from a route: the models routes are one
        // path serving two protocols, so a path check here would answer an
        // Anthropic client's 404 in the OpenAI envelope.
        if ServerController.wireProtocol(for: request) == .anthropic {
            response = try context.responseEncoder.encode(
                AnthropicErrorEvent(error: .init(type: wireType.rawValue, message: message)),
                from: request,
                context: context
            )
        } else {
            response = try context.responseEncoder.encode(
                OpenAIErrorEnvelope(error: .init(message: message, type: openAIWireType, code: openAICode)),
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

    /// Identity of the newest lifecycle transition. A `stop()` that is still
    /// draining when a newer `start()` lands must not publish its terminal state
    /// over the server that start brought up.
    private var lifecycleGeneration: UInt64 = 0

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

    /// Whether the settings toggle currently asks for the server to be running.
    /// `@AppStorage` writes `dynamoe_server_enabled` before the change handler
    /// runs, so this is the intent a lifecycle transition has to satisfy when it
    /// lands — not the state it happened to observe when it started.
    nonisolated static var isEnabledRequested: Bool {
        UserDefaults.standard.bool(forKey: enabledKey)
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
        lifecycleGeneration &+= 1
        lastError = nil
        let port = Self.resolvedPort
        let router = Self.buildRouter(queue: queue)
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(Self.bindHost, port: port), serverName: "DynaMoE")
        )
        let group = ServiceGroup(services: [app])
        let transition = lifecycleGeneration
        serviceGroup = group
        boundPort = port
        isRunning = true
        Task.detached { [weak self] in
            do {
                try await group.run()
            } catch {
                await self?.noteError("Server stopped with error: \(error.localizedDescription)", transition: transition)
            }
        }
        DiagLog.info("server.lifecycle", "listening on \(Self.bindHost):\(port)")
    }

    func stop() async {
        guard let group = serviceGroup else { return }
        lifecycleGeneration &+= 1
        let transition = lifecycleGeneration
        serviceGroup = nil
        // Cancel only turns the server owns. The chat UI drives the engine
        // directly (not through this queue), so its in-flight response must
        // survive a server restart or settings change.
        await queue.cancelLiveWork()
        // The recorded turn belongs to the server's lifetime; a restart starts
        // a fresh session history anyway.
        ServerTurnMemory.shared.clear()
        await group.triggerGracefulShutdown()
        // A newer transition (a toggle that went straight back on, or a restart)
        // owns the published state now: this shutdown must not report the server
        // as stopped on top of the server that started meanwhile.
        guard transition == lifecycleGeneration else { return }
        isRunning = false
        boundPort = 0
        DiagLog.info("server.lifecycle", "stopped")
    }

    /// Applies the requested state (`dynamoe_server_enabled`), which the
    /// settings toggle persists *before* it asks for the change. Reconciling to
    /// the persisted value rather than to the caller's argument is what makes the
    /// newest request win when two toggles race — and the check after a shutdown
    /// is what honors a toggle that flipped back on while that shutdown was still
    /// draining, instead of letting the shutdown decide the final state.
    func applyRequestedState() async {
        if isRunning && !Self.isEnabledRequested {
            await stop()
        }
        if Self.isEnabledRequested && !isRunning {
            start()
        }
    }

    /// A server that has already been replaced must not publish its own failure
    /// (or its shutdown) as the current server's state.
    private func noteError(_ message: String, transition: UInt64) {
        guard transition == lifecycleGeneration else { return }
        lastError = message
        isRunning = false
        boundPort = 0
        DiagLog.error("server.lifecycle", "server error: \(message)")
    }

    // MARK: Router

    nonisolated static func buildRouter(queue: TurnExecutionQueue) -> Router<BasicRequestContext> {
        let router = Router()
        router.add(middleware: ServerWireLogMiddleware<BasicRequestContext>())

        router.get("health") { _, _ in
            ServerController.healthBody()
        }

        // `GET /v1/models` is both protocols' own route: identical path, different
        // body. OpenAI answers `object: "list"` with an epoch `created`; Anthropic
        // answers `type: "model"` with an RFC 3339 `created_at`, a human
        // `display_name` and pagination cursors. The headers decide which, and the
        // same decision stamps the log row's protocol, so a dashboard shows which
        // shape a client was handed.
        router.get("v1/models") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            let logBox = ServerRequestLogRecorder.shared.box(for: context.id)
            try ServerController.authorizeOrThrow(request)
            let wire = ServerController.wireProtocol(for: request)
            logBox?.modify { $0.proto = wire.rowProtocol }
            let now = Date()
            guard let info = CompletionEngine.shared.loadedModelInfo else {
                // No loaded checkpoint is an empty list, not an error: a client
                // polling for a model to appear must not read a 404 as "wrong URL".
                switch wire {
                case .openai:
                    return try context.responseEncoder.encode(
                        OpenAIModelList(data: []), from: request, context: context
                    )
                case .anthropic:
                    return try context.responseEncoder.encode(
                        AnthropicModelList(data: [], hasMore: false, firstId: nil, lastId: nil),
                        from: request,
                        context: context
                    )
                }
            }
            logBox?.modify { $0.model = info.displayName }
            // What this server refuses is a fact about the server, and what the
            // weights support is a fact about the checkpoint; `Int.max` (the "no
            // cap" sentinel) is reported as nil so neither is mistaken for the
            // other after a client's JSON float conversion.
            let promptCap = ServerController.resolvedMaxPromptTokens
            switch wire {
            case .openai:
                return try context.responseEncoder.encode(
                    OpenAIModelList(data: [
                        OpenAIModelList.entry(
                            for: info,
                            contextLength: CompletionEngine.shared.promptModelConfig?
                                .effectiveMaxPositionEmbeddings,
                            maxPromptTokens: promptCap == Int.max ? nil : promptCap,
                            created: Int(now.timeIntervalSince1970)
                        )
                    ]),
                    from: request,
                    context: context
                )
            case .anthropic:
                return try context.responseEncoder.encode(
                    AnthropicModelList(
                        data: [AnthropicAdapter.entry(
                            id: info.displayName,
                            displayName: AnthropicAdapter.displayName(
                                servedId: info.displayName, snapshotPath: info.path
                            ),
                            createdAt: now
                        )],
                        hasMore: false,
                        firstId: info.displayName,
                        lastId: info.displayName
                    ),
                    from: request,
                    context: context
                )
            }
        }

        // Both protocols define a retrieve route beside their list. It answers the
        // same object the list carries, so a client that resolved an id from the list
        // reads the same facts here. An id this server does not serve is a 404 in the
        // protocol's own envelope, because answering the loaded model for a name the
        // client did not ask for would hide a typo or a stale config.
        router.get("v1/models/:model_id") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            let logBox = ServerRequestLogRecorder.shared.box(for: context.id)
            try ServerController.authorizeOrThrow(request)
            let wire = ServerController.wireProtocol(for: request)
            logBox?.modify { $0.proto = wire.rowProtocol }
            let info = try ServerController.requireLoadedModel()
            logBox?.modify { $0.model = info.displayName }
            // The path parameter lives on the request's context, not on the request.
            let requested = context.parameters.get("model_id")
            guard let requested, requested == info.displayName else {
                throw ServerAPIError.modelNotFound(requested ?? "")
            }
            let promptCap = ServerController.resolvedMaxPromptTokens
            let now = Date()
            switch wire {
            case .openai:
                return try context.responseEncoder.encode(
                    OpenAIModelList.entry(
                        for: info,
                        contextLength: CompletionEngine.shared.promptModelConfig?
                            .effectiveMaxPositionEmbeddings,
                        maxPromptTokens: promptCap == Int.max ? nil : promptCap,
                        created: Int(now.timeIntervalSince1970)
                    ),
                    from: request,
                    context: context
                )
            case .anthropic:
                return try context.responseEncoder.encode(
                    AnthropicAdapter.entry(
                        id: info.displayName,
                        displayName: AnthropicAdapter.displayName(
                            servedId: info.displayName, snapshotPath: info.path
                        ),
                        createdAt: now
                    ),
                    from: request,
                    context: context
                )
            }
        }

        router.post("v1/chat/completions") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            let logBox = ServerRequestLogRecorder.shared.box(for: context.id)
            logBox?.modify { $0.requestBytes = Int(request.headers[.contentLength] ?? "") }
            try ServerController.authorizeOrThrow(request)
            let apiRequest: OpenAIChatRequest
            do {
                apiRequest = try await request.decode(as: OpenAIChatRequest.self, context: context)
            } catch {
                throw ServerAPIError.badRequest("Invalid request body: \(error.localizedDescription)")
            }
            logBox?.modify {
                $0.wasStreaming = apiRequest.stream == true
                $0.temperature = apiRequest.temperature.map { Double($0) }
                $0.maxTokens = apiRequest.maxTokens
                // Building the normalized summary walks and serializes the
                // whole conversation. Skip that work entirely when body
                // capture is off and stage the withheld marker directly:
                // the produced summary would be discarded on the next line.
                $0.requestBody = ServerRequestLogRecorder.captureBodiesEnabled
                    ? ServerRequestLog.capturedRequestBody(ServerRequestLog.openAISummary(apiRequest))
                    : ServerRequestLog.contentWithheldMarker
            }
            if apiRequest.stream == true {
                // Streaming keeps the 200 + text/event-stream contract even when
                // the engine can't accept the turn: errors arrive as SSE frames.
                return ServerController.streamingResponse(apiRequest: apiRequest, queue: queue, logBox: logBox)
            }
            let plan = try ServerController.buildTurnPlan(apiRequest)
            logBox?.modify {
                $0.model = plan.modelName
                $0.inputTokens = plan.promptTokenCount
            }
            let result = try await ServerController.nonStreamingResponse(plan: plan, queue: queue, logBox: logBox)
            return try context.responseEncoder.encode(result, from: request, context: context)
        }

        // The protocol's legacy text endpoint. It is the same engine and the same
        // planner as the chat route: the prompt is rendered by the one renderer, so
        // a plain prompt is answered after the checkpoint's own assistant prefix and
        // an already-templated one is continued byte for byte. What it does not
        // share is a shape: no messages in, `text_completion` with `choices[].text`
        // out. `ServerRequestLog.proto(forPath:)` has always classified this path as
        // OpenAI, so before this route existed the log vocabulary named a path the
        // router did not serve.
        router.post("v1/completions") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            let logBox = ServerRequestLogRecorder.shared.box(for: context.id)
            logBox?.modify { $0.requestBytes = Int(request.headers[.contentLength] ?? "") }
            try ServerController.authorizeOrThrow(request)
            let apiRequest: OpenAICompletionsRequest
            do {
                apiRequest = try await request.decode(as: OpenAICompletionsRequest.self, context: context)
            } catch {
                throw ServerAPIError.badRequest("Invalid request body: \(error.localizedDescription)")
            }
            logBox?.modify {
                $0.wasStreaming = apiRequest.stream == true
                $0.temperature = apiRequest.temperature.map { Double($0) }
                $0.maxTokens = apiRequest.maxTokens
                // Same gating as the sibling routes: never build the summary just
                // to discard it when body capture is off.
                $0.requestBody = ServerRequestLogRecorder.captureBodiesEnabled
                    ? ServerRequestLog.capturedRequestBody(ServerRequestLog.openAICompletionsSummary(apiRequest))
                    : ServerRequestLog.contentWithheldMarker
            }
            if apiRequest.stream == true {
                // Streaming keeps the 200 + text/event-stream contract even when the
                // engine cannot accept the turn: errors arrive as SSE frames.
                return ServerController.completionsStreamingResponse(apiRequest: apiRequest, queue: queue, logBox: logBox)
            }
            let plan = try ServerController.buildTurnPlan(apiRequest)
            logBox?.modify {
                $0.model = plan.modelName
                $0.inputTokens = plan.promptTokenCount
            }
            let result = try await ServerController.completionsNonStreamingResponse(
                plan: plan,
                apiRequest: apiRequest,
                queue: queue,
                logBox: logBox
            )
            return try context.responseEncoder.encode(result, from: request, context: context)
        }

        router.post("v1/messages") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            let logBox = ServerRequestLogRecorder.shared.box(for: context.id)
            logBox?.modify { $0.requestBytes = Int(request.headers[.contentLength] ?? "") }
            try ServerController.authorizeOrThrow(request)
            let apiRequest: AnthropicMessagesRequest
            do {
                apiRequest = try await request.decode(as: AnthropicMessagesRequest.self, context: context)
            } catch {
                throw ServerAPIError.badRequest("Invalid request body: \(error.localizedDescription)")
            }
            logBox?.modify {
                $0.wasStreaming = apiRequest.stream == true
                $0.temperature = apiRequest.temperature.map { Double($0) }
                $0.maxTokens = apiRequest.maxTokens
                // Same gating as the OpenAI route: never build the summary
                // just to discard it when body capture is off.
                $0.requestBody = ServerRequestLogRecorder.captureBodiesEnabled
                    ? ServerRequestLog.capturedRequestBody(ServerRequestLog.anthropicSummary(apiRequest))
                    : ServerRequestLog.contentWithheldMarker
            }
            if apiRequest.stream == true {
                // Streaming keeps the 200 + text/event-stream contract even when
                // the engine can't accept the turn: errors arrive as SSE frames.
                return ServerController.anthropicStreamingResponse(apiRequest: apiRequest, queue: queue, logBox: logBox)
            }
            let plan = try ServerController.buildTurnPlan(apiRequest)
            logBox?.modify {
                $0.model = plan.modelName
                $0.inputTokens = plan.promptTokenCount
            }
            let result = try await ServerController.anthropicNonStreamingResponse(plan: plan, queue: queue, logBox: logBox)
            return try context.responseEncoder.encode(result, from: request, context: context)
        }

        // The protocol's pre-flight. Anthropic clients count a conversation
        // before sending it, so a client that cannot ask this question either
        // fails its own pre-flight or guesses.
        //
        // The number comes from the messages route's own planner, which is the
        // point: it is the prompt that route would report as `usage.input_tokens`
        // for the same body — image spans expanded, a recorded generation spliced
        // back in, the system block rendered with the same session date. A second
        // estimator here would answer a number the turn it precedes never matches.
        // It runs no generation and claims no slot, so it answers while another
        // turn is generating.
        router.post("v1/messages/count_tokens") { request, context in
            ServerController.stats.beginRequest()
            defer { ServerController.stats.endRequest() }
            let logBox = ServerRequestLogRecorder.shared.box(for: context.id)
            logBox?.modify { $0.requestBytes = Int(request.headers[.contentLength] ?? "") }
            try ServerController.authorizeOrThrow(request)
            let apiRequest: AnthropicMessagesRequest
            do {
                apiRequest = try await request.decode(as: AnthropicMessagesRequest.self, context: context)
            } catch {
                throw ServerAPIError.badRequest("Invalid request body: \(error.localizedDescription)")
            }
            logBox?.modify {
                $0.requestBody = ServerRequestLogRecorder.captureBodiesEnabled
                    ? ServerRequestLog.capturedRequestBody(ServerRequestLog.anthropicSummary(apiRequest))
                    : ServerRequestLog.contentWithheldMarker
            }
            // `max_tokens` is required on the messages route and absent from this
            // one by protocol; the reply budget never moves the prompt, so the
            // planner is told not to demand it rather than this route inventing a
            // plan of its own.
            let plan = try ServerController.buildTurnPlan(apiRequest, maxTokensRequired: false)
            logBox?.modify {
                $0.model = plan.modelName
                $0.inputTokens = plan.promptTokenCount
            }
            let count = AnthropicTokenCount(inputTokens: plan.promptTokenCount)
            return try context.responseEncoder.encode(count, from: request, context: context)
        }

        return router
    }

    // MARK: Protocol

    /// Which protocol's wire shape a request is answered in.
    ///
    /// Where the two protocols have different paths, the path decides. `GET
    /// /v1/models` is the case they share — same path, different body — so there the
    /// request's own headers decide, and they are unambiguous evidence: an Anthropic
    /// client sends `x-api-key` or `anthropic-version`, and no OpenAI client sends
    /// either. A client that presents neither gets the OpenAI shape, which is what
    /// this route has always answered, so nothing that works today changes.
    ///
    /// The cost of guessing here is a decode failure inside a client's own SDK, which
    /// is why the answer is keyed on headers the protocol defines rather than on a
    /// `User-Agent` string or a query parameter the SDK never sends.
    nonisolated static func wireProtocol(for request: Request) -> ServerWireProtocol {
        let path = request.uri.path
        if path.hasPrefix("/v1/messages") { return .anthropic }
        if path.hasPrefix("/v1/chat") || path.hasPrefix("/v1/completions") { return .openai }
        for name in ["anthropic-version", "x-api-key"] {
            guard let field = HTTPField.Name(name),
                  let value = request.headers[field],
                  !value.trimmingCharacters(in: .whitespaces).isEmpty
            else { continue }
            return .anthropic
        }
        return .openai
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
        /// Prompt positions the model will actually attend over, i.e. AFTER an
        /// attachment's span is expanded. One image renders as one token and runs
        /// several hundred, so this is not the tokenized prompt's length whenever a
        /// picture is attached.
        var promptTokenCount: Int
        var modelName: String
        var hasTools: Bool
        /// How many leading conversation turns this turn's prompt was rendered
        /// from, plus a digest over them and the rendered system prompt. Kept on
        /// the plan so the completed turn can record what a follow-up must
        /// reproduce before its raw generation may be spliced back in.
        var historyTurnCount: Int
        var historyDigest: UInt64
        /// Client tool schemas for the grammar mask (empty when the turn offers
        /// no tools). Installed around the generation so the mask constrains the
        /// model to the client's own tool vocabulary rather than the in-app
        /// agent catalog.
        var maskSchemas: [GrammarToolSchema] = []
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
    nonisolated static func wireLaunchError(_ error: Error) -> ServerAPIError {
        if let apiError = error as? ServerAPIError { return apiError }
        if let launchError = error as? CompletionLaunchError {
            let message = launchError.errorDescription ?? error.localizedDescription
            // A KV-capacity rejection is a context-length error, not a generic
            // bad request: clients (Crush, opencode, Continue) act on the code.
            if case .promptTooLong = launchError {
                return ServerAPIError.contextLengthExceeded(message)
            }
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
        /// OpenAI's `frequency_penalty`, already range-checked by the adapter.
        var frequencyPenalty: Float?
        /// OpenAI's `logit_bias`, already converted from the wire's string keys and
        /// range-checked by the adapter.
        var logitBias: [Int32: Float]?
        var maxTokens: Int?
        var stopSequences: [String]
        var thinkingRequested: Bool
        /// The client's requested graded level, already validated by the
        /// adapter. Ignored for models without graded levels; the template's
        /// high default is applied below when thinking is on.
        var thinkingEffort: ThinkingEffort?
        var hasTools: Bool
        var toolDeclaration: String?
        var maskSchemas: [GrammarToolSchema]
        var sessionId: UUID
        /// Images attached to specific turns, keyed by that turn's index in
        /// `conversation`. The planner writes one prompt marker per image at the
        /// head of its turn and hands the decoded pixels to the engine, because
        /// only it knows whether the loaded checkpoint publishes a tower.
        var images: [Int: [ServerImagePart]] = [:]
    }

    /// Resolves the level a turn renders. Level-capable models always render a
    /// level while thinking is on: the template defaults to high, so an API
    /// request that omitted the knob must not render the in-app variant's
    /// other prompt.
    nonisolated static func resolvedThinkingEffort(
        requested: ThinkingEffort?,
        thinkingEnabled: Bool,
        levelCapable: Bool
    ) -> ThinkingEffort? {
        guard thinkingEnabled, levelCapable else { return nil }
        return requested ?? .high
    }

    /// The one refusal a checkpoint with no tower gets, on every path.
    nonisolated private static func noVisionTower(_ info: CompletionEngine.LoadedModelInfo) -> ServerAPIError {
        ServerAPIError.unsupportedContent(
            "The loaded model (\(info.displayName)) publishes no vision tower, so image content parts "
                + "cannot be processed. Send text only, or load a vision-capable checkpoint."
        )
    }

    /// A turn's attachments, resolved to what the engine consumes.
    nonisolated struct ResolvedTurnImages {
        /// The conversation with one prompt marker written at the head of every
        /// turn that carried a picture.
        var turns: [ChatPromptFormatter.ConversationTurn]
        /// The decoded pixels, in prompt order.
        var images: [TurnImage]
    }

    /// Writes one marker per attachment into its turn and decodes the pixels.
    ///
    /// This is the server's whole image gate. A checkpoint with no tower is refused
    /// here, by name, rather than downstream where the engine would refuse an
    /// expanded prompt whose placeholders have no rows; a URL is refused rather
    /// than fetched; and the marker is the checkpoint's own image token, so the
    /// prompt the renderer produces is the same one the in-app composer produces.
    nonisolated private static func resolveTurnImages(
        _ inputs: TurnPlanInputs,
        info: CompletionEngine.LoadedModelInfo
    ) throws -> ResolvedTurnImages {
        guard inputs.images.values.contains(where: { !$0.isEmpty }) else {
            return ResolvedTurnImages(turns: inputs.conversation, images: [])
        }
        guard info.supportsVision,
              let marker = ServerImageParts.visionMarker(
                  modelPath: info.path,
                  config: CompletionEngine.shared.promptModelConfig
              ) else {
            throw noVisionTower(info)
        }
        return try attachImageMarkers(turns: inputs.conversation, images: inputs.images, marker: marker)
    }

    /// The placement half of the gate, split out so it can be tested without a
    /// loaded model: which turn a marker lands in, how many markers precede a
    /// turn's text, and what a part that carries no bytes reports.
    nonisolated static func attachImageMarkers(
        turns: [ChatPromptFormatter.ConversationTurn],
        images: [Int: [ServerImagePart]],
        marker: String
    ) throws -> ResolvedTurnImages {
        var turns = turns
        var decoded: [TurnImage] = []
        for index in images.keys.sorted() {
            let parts = images[index] ?? []
            guard !parts.isEmpty else { continue }
            guard index >= 0, index < turns.count else {
                throw ServerAPIError.badRequest("An image was attached to a conversation turn that does not exist.")
            }
            var fragment = ""
            for part in parts {
                guard let data = part.bytes else {
                    switch part.payload {
                    case let .remoteURL(url):
                        throw ServerAPIError.unsupportedContent(
                            "This server will not fetch an image from a URL (\(url)). Send the picture "
                                + "inline as base64 (a `data:image/png;base64,...` URL), which is what the "
                                + "OpenAI and Anthropic SDKs send."
                        )
                    case let .malformed(reason):
                        throw ServerAPIError.unsupportedContent("An attached image could not be read: \(reason).")
                    case .inline:
                        throw ServerAPIError.unsupportedContent("An attached image carried no bytes.")
                    }
                }
                let pixels: DeepSeekV41ImagePixels
                do {
                    pixels = try DeepSeekV41ImagePixels.decode(data)
                } catch {
                    throw ServerAPIError.unsupportedContent(
                        "An attached image could not be decoded: \(error.localizedDescription)"
                    )
                }
                // The picture goes FIRST inside its turn: that is where the
                // checkpoint's own prompts place an image relative to the question
                // about it, and what the in-app composer writes.
                fragment += marker + "\n"
                decoded.append(
                    TurnImage(
                        rgb: pixels.rgb,
                        width: pixels.width,
                        height: pixels.height,
                        sha256: ServerImageParts.digest([part])
                    )
                )
            }
            turns[index].text = fragment + turns[index].text
        }
        return ResolvedTurnImages(turns: turns, images: decoded)
    }

    nonisolated private static func buildTurnPlan(_ inputs: TurnPlanInputs) throws -> TurnPlan {
        let engine = CompletionEngine.shared
        let info = try requireLoadedModel()

        // Pictures are resolved first: everything below (the history digest, the
        // recorded-turn splice, the tokenized prompt and the cache the turn runs
        // on) has to see the expanded turn rather than the picture-less one.
        let vision = try resolveTurnImages(inputs, info: info)
        let conversationTurns = vision.turns

        // The temporal context this prompt is rendered with. Threaded through the
        // turn request so the completed turn pins the date its prompt actually
        // carries: a KV-precision change can invalidate the pin after planning,
        // and pinning a completion-time date instead would make the next turn
        // render a different timestamp and lose the prefix.
        let renderedAsOfDate = PrefixCacheManager.shared.asOfDate(sessionId: inputs.sessionId)

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
            currentDate: renderedAsOfDate
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        if inputs.hasTools, let declaration = inputs.toolDeclaration {
            cleanSystem = cleanSystem.isEmpty ? declaration : cleanSystem + "\n\n" + declaration
        }

        let thinkingEnabled = info.supportsThinking && inputs.thinkingRequested
        let thinkingEffort = Self.resolvedThinkingEffort(
            requested: inputs.thinkingEffort,
            thinkingEnabled: thinkingEnabled,
            levelCapable: info.supportsThinkingLevels
        )
        // The digest must cover the system text the model actually sees: changing
        // the level between turns has to invalidate a recorded raw turn, just as
        // changing the system does. Both graded-effort families render the level
        // into the System block, so `renderedSystemBody` is the authority here.
        let memorySystem = ChatPromptFormatter.renderedSystemBody(
            cleanSystem: cleanSystem,
            thinkingEnabled: thinkingEnabled,
            thinkingEffort: thinkingEffort,
            modelConfig: engine.promptModelConfig,
            promptDialect: engine.promptDialect
        )
        // What a follow-up request has to reproduce before its own raw turn can
        // be trusted: the leading history as the client sent it, plus the system
        // prompt it was rendered against. Computed before substitution so both
        // sides of the comparison see the same client-authored turns.
        let historyDigest = ServerTurnMemory.historyDigest(
            systemPrompt: memorySystem,
            turns: conversationTurns[...]
        )
        // Splice in the raw generation recorded for this session (reasoning
        // included) so the rendered prompt reproduces the pinned
        // prompt+generation; a client's echo of the visible answer alone cannot.
        let conversation = Self.substitutingRecordedTurn(
            conversationTurns,
            sessionId: inputs.sessionId,
            modelName: info.displayName,
            modelConfig: engine.promptModelConfig,
            systemPrompt: memorySystem
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
                modelConfig: engine.promptModelConfig,
                thinkingEffort: thinkingEffort,
                promptDialect: engine.promptDialect
            )
        } else {
            formattedPrompt = ChatPromptFormatter.formatConversation(
                turns: conversation,
                cleanSystem: cleanSystem,
                thinkingEnabled: thinkingEnabled,
                modelSupportsThinking: info.supportsThinking,
                modelConfig: engine.promptModelConfig,
                thinkingEffort: thinkingEffort,
                promptDialect: engine.promptDialect
            )
        }

        let promptTokenIds: [UInt32]
        do {
            promptTokenIds = try engine.tokenize(formattedPrompt)
        } catch {
            throw ServerAPIError.badRequest(
                "Tokenizer failed to encode the prompt: \(error.localizedDescription)",
                // The cause can quote the offending prompt text (an unknown
                // special token), so the row keeps the category only.
                logSafeMessage: "Tokenizer failed to encode the prompt"
            )
        }
        guard !promptTokenIds.isEmpty else {
            throw ServerAPIError.badRequest("Prompt tokenization produced 0 tokens.")
        }
        // An image prompt renders one token per attachment and then runs the whole
        // span, so the number the budget is checked against and the number the
        // client is told must be the EXPANDED size. `price` reads no weight: the
        // resize plan alone says how many positions the spans occupy. It also
        // re-checks the placeholder/image agreement, so a prompt built with a stray
        // marker is a 400 here instead of an engine error later.
        var promptTokenCount = promptTokenIds.count
        if !vision.images.isEmpty {
            guard let spec = ModelConfig.visionSpec(modelPath: info.path) else {
                throw noVisionTower(info)
            }
            do {
                promptTokenCount = try DeepSeekV41TurnImages.price(
                    tokenIds: promptTokenIds,
                    images: vision.images.map {
                        DeepSeekV41ImageInput(
                            rgb: $0.rgb, width: $0.width, height: $0.height, sha256: $0.sha256
                        )
                    },
                    spec: spec
                ).promptTokens
            } catch {
                throw ServerAPIError.badRequest(
                    "The attached image could not be placed in the prompt: \(error.localizedDescription)"
                )
            }
        }
        let promptTokenCap = Self.resolvedMaxPromptTokens
        if promptTokenCount > promptTokenCap {
            throw ServerAPIError.badRequest("Prompt is too long: \(promptTokenCount) tokens exceeds the server's \(promptTokenCap)-token limit (dynamoe_server_max_prompt_tokens).")
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
            frequencyPenalty: inputs.frequencyPenalty ?? 0.0,
            logitBias: inputs.logitBias ?? [:],
            maxTokens: maxTokens
        )

        let memory = TurnMemorySettings.persisted(defaults)

        // The dialect's stop markers belong to the sampler, not just to the
        // renderer: without them a generation that emits the next role marker
        // keeps going until maxTokens instead of ending the turn.
        var stopSequences = inputs.stopSequences
        if engine.promptModelConfig?.isDeepSeekV41Model == true {
            for stop in engine.promptDialect.stopSequences
            where !stop.isEmpty && !stopSequences.contains(stop) {
                stopSequences.append(stop)
            }
        }

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
            stopSequences: stopSequences,
            asOfDate: renderedAsOfDate,
            images: vision.images
        )
        return TurnPlan(
            request: turnRequest,
            promptTokenCount: promptTokenCount,
            modelName: info.displayName,
            hasTools: inputs.hasTools,
            historyTurnCount: conversationTurns.count,
            historyDigest: historyDigest,
            maskSchemas: inputs.maskSchemas
        )
    }

    /// Installs the turn's own tool schemas into the grammar mask for the
    /// duration of its generation, then restores the registry that was live
    /// before it. The mask resolves tool names and parameter keys from the
    /// in-app agent catalog; a client-defined API tool is not in it, so without
    /// this a masked sample could rewrite the client's tool name into a built-in
    /// one (or withhold the client's parameter keys) and the call could never be
    /// emitted. The engine is single-flight and this queue serializes server
    /// turns, so the installed registry is never shared by two live turns.
    nonisolated private static func withTurnToolMask<T>(
        _ schemas: [GrammarToolSchema],
        _ body: () async throws -> T
    ) async rethrows -> T {
        guard !schemas.isEmpty else { return try await body() }
        let previous = GrammarConstrainedSampler.shared.snapshotRegistry()
        GrammarConstrainedSampler.shared.registerToolSchemas(schemas)
        defer { GrammarConstrainedSampler.shared.restoreRegistry(previous) }
        return try await body()
    }

    nonisolated private static func buildTurnPlan(_ apiRequest: OpenAIChatRequest) throws -> TurnPlan {
        // Refuse content parts the engine cannot process at all, before touching
        // engine state. Images are not on this list: whether they can be read is a
        // property of the loaded checkpoint, which the shared planner checks.
        if apiRequest.hasUnsupportedContent {
            throw ServerAPIError.unsupportedContent(
                "This server is serving a local weights model, which has no reader for file or audio content parts. Send text and images only."
            )
        }
        // Reject a forced tool choice before touching engine state, so the
        // client gets a clear error instead of a silently unconstrained turn.
        let toolsEnabled = try OpenAIAdapter.toolUseEnabled(apiRequest.toolChoice)
        // Same posture for the fields that ask for an answer this server cannot
        // produce: structured output, several choices, or token log-probabilities.
        // Ignoring them answered a different request than the one sent, with a 200.
        try OpenAIAdapter.refuseUnsupportedResponseShape(apiRequest)
        let info = try requireLoadedModel()
        let split = try OpenAIAdapter.splitMessages(apiRequest.messages)
        let hasTools = !(apiRequest.tools ?? []).isEmpty && toolsEnabled
        let thinkingRequested = apiRequest.enableThinking ?? true
        let thinkingEffort = try OpenAIAdapter.thinkingEffort(
            from: apiRequest.reasoningEffort,
            levelCapable: info.supportsThinkingLevels,
            thinkingRequested: thinkingRequested
        )
        return try buildTurnPlan(TurnPlanInputs(
            systemPrompt: split.systemPrompt,
            conversation: split.turns,
            temperature: apiRequest.temperature,
            topP: apiRequest.topP,
            topK: apiRequest.topK,
            minP: apiRequest.minP,
            repetitionPenalty: apiRequest.repetitionPenalty,
            presencePenalty: apiRequest.presencePenalty,
            frequencyPenalty: try OpenAIAdapter.frequencyPenalty(apiRequest.frequencyPenalty),
            logitBias: try OpenAIAdapter.logitBias(from: apiRequest.logitBias),
            // Either spelling of the reply budget, resolved here exactly as the
            // refusal above validated it.
            maxTokens: try OpenAIAdapter.replyBudget(
                maxTokens: apiRequest.maxTokens,
                maxCompletionTokens: apiRequest.maxCompletionTokens
            ),
            stopSequences: apiRequest.stop ?? [],
            thinkingRequested: thinkingRequested,
            thinkingEffort: thinkingEffort,
            hasTools: hasTools,
            toolDeclaration: hasTools ? OpenAIAdapter.toolDeclarationText(apiRequest.tools ?? []) : nil,
            maskSchemas: hasTools ? (apiRequest.tools ?? []).filter { ($0.type ?? "function") == "function" }.map {
                GrammarToolSchema(name: $0.function.name, schemaJSON: $0.function.parametersData)
            } : [],
            sessionId: OpenAIAdapter.sessionKey(for: apiRequest.messages, model: info.displayName),
            images: split.images
        ))
    }

    /// The legacy text-completions plan. One prompt, one user turn, no tools: this
    /// protocol has no message list and no vocabulary for a call, so the planner has
    /// nothing to resolve beyond the refusal of the fields it cannot honour.
    ///
    /// The refusals run before `requireLoadedModel()` for the reason the sibling
    /// planners' do: a request this server cannot answer at all should say so with a
    /// 400 rather than a 503 that blames the missing model.
    nonisolated private static func buildTurnPlan(_ apiRequest: OpenAICompletionsRequest) throws -> TurnPlan {
        try OpenAIAdapter.refuseUnsupportedCompletionsShape(apiRequest)
        let info = try requireLoadedModel()
        let split = OpenAIAdapter.completionsSplit(apiRequest)
        let thinkingRequested = OpenAIAdapter.completionsThinkingRequested(apiRequest)
        let thinkingEffort = try OpenAIAdapter.thinkingEffort(
            from: apiRequest.reasoningEffort,
            levelCapable: info.supportsThinkingLevels,
            thinkingRequested: thinkingRequested
        )
        return try buildTurnPlan(TurnPlanInputs(
            systemPrompt: split.systemPrompt,
            conversation: split.turns,
            temperature: apiRequest.temperature,
            topP: apiRequest.topP,
            topK: apiRequest.topK,
            minP: apiRequest.minP,
            repetitionPenalty: apiRequest.repetitionPenalty,
            presencePenalty: apiRequest.presencePenalty,
            frequencyPenalty: try OpenAIAdapter.frequencyPenalty(apiRequest.frequencyPenalty),
            logitBias: try OpenAIAdapter.logitBias(from: apiRequest.logitBias),
            maxTokens: apiRequest.maxTokens,
            stopSequences: apiRequest.stop ?? [],
            thinkingRequested: thinkingRequested,
            thinkingEffort: thinkingEffort,
            hasTools: false,
            toolDeclaration: nil,
            maskSchemas: [],
            sessionId: OpenAIAdapter.sessionKey(forPrompt: apiRequest.promptText ?? "", model: info.displayName),
            images: split.images
        ))
    }

    /// - Parameter maxTokensRequired: the messages route requires `max_tokens`
    ///   and the `count_tokens` body has no such field by protocol. The reply
    ///   budget never moves the prompt, so the count path takes the smallest legal
    ///   one instead of refusing or building a plan of its own.
    nonisolated private static func buildTurnPlan(
        _ apiRequest: AnthropicMessagesRequest,
        maxTokensRequired: Bool = true
    ) throws -> TurnPlan {
        let declaredMaxTokens = apiRequest.maxTokens ?? (maxTokensRequired ? nil : 1)
        guard let maxTokens = declaredMaxTokens else {
            throw ServerAPIError.badRequest("max_tokens is a required field")
        }
        // Same explicit refusal as the OpenAI surface for the block kinds the
        // engine has no reader for. Images are resolved by the shared planner,
        // which knows whether the loaded checkpoint publishes a tower.
        if apiRequest.hasUnsupportedContent {
            throw ServerAPIError.unsupportedContent(
                "This server is serving a local weights model, which has no reader for document or audio content blocks. Send text and images only."
            )
        }
        // Reject a forced tool choice before touching engine state, so the
        // client gets a clear error instead of a silently unconstrained turn.
        let toolsEnabled = try AnthropicAdapter.toolUseEnabled(apiRequest.toolChoice)
        let info = try requireLoadedModel()
        let split = try AnthropicAdapter.splitMessages(apiRequest.messages)
        let hasTools = !(apiRequest.tools ?? []).isEmpty && toolsEnabled
        let thinkingRequested = apiRequest.thinking?.type == "enabled"
        let thinkingEffort = try AnthropicAdapter.thinkingEffort(
            from: apiRequest.thinking,
            levelCapable: info.supportsThinkingLevels
        )
        return try buildTurnPlan(TurnPlanInputs(
            systemPrompt: apiRequest.systemText ?? "",
            conversation: split.turns,
            temperature: apiRequest.temperature,
            topP: apiRequest.topP,
            topK: apiRequest.topK,
            minP: nil,
            repetitionPenalty: nil,
            presencePenalty: nil,
            maxTokens: maxTokens,
            stopSequences: apiRequest.stopSequences ?? [],
            thinkingRequested: thinkingRequested,
            thinkingEffort: thinkingEffort,
            hasTools: hasTools,
            toolDeclaration: hasTools ? AnthropicAdapter.toolDeclarationText(apiRequest.tools ?? []) : nil,
            maskSchemas: hasTools ? (apiRequest.tools ?? []).map {
                GrammarToolSchema(name: $0.name, schemaJSON: $0.inputSchemaData)
            } : [],
            sessionId: AnthropicAdapter.sessionKey(for: apiRequest.messages, model: info.displayName),
            images: split.images
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
            generatedTokenIds: result.generatedTokenIds,
            hadToolCalls: Self.turnEmittedToolCalls(result),
            historyTurnCount: plan.historyTurnCount,
            historyDigest: plan.historyDigest
        ))
    }

    /// Whether a finished turn emitted tool calls: a tool freeze, or any call
    /// the parser recovered. Such a turn reaches a client as a call structure
    /// rather than as text, which is what the turn memory has to expect back.
    nonisolated static func turnEmittedToolCalls(_ result: TurnResult) -> Bool {
        result.finishReason == .toolFreeze || !result.parsedToolCalls.isEmpty
    }

    /// Replaces the client's echo of the last assistant turn with the raw
    /// generation recorded for this session, so the rendered prompt reproduces
    /// the pinned prompt+generation (and the prefix stays reusable). Falls back
    /// to the client's own text whenever the record does not clearly belong to
    /// this turn — see `ServerTurnMemory.recordedRawTurn`.
    nonisolated private static func substitutingRecordedTurn(
        _ conversation: [ChatPromptFormatter.ConversationTurn],
        sessionId: UUID,
        modelName: String,
        modelConfig: ModelConfig?,
        systemPrompt: String
    ) -> [ChatPromptFormatter.ConversationTurn] {
        let closer = ChatPromptFormatter.assistantTurnCloseTag(
            modelConfig: modelConfig,
            promptDialect: CompletionEngine.shared.promptDialect
        )
        guard let index = conversation.lastIndex(where: { $0.role == .assistant }),
              let recorded = ServerTurnMemory.shared.recordedRawTurn(
                  sessionId: sessionId,
                  modelName: modelName,
                  echoedByClient: conversation[index].text,
                  turnCloser: closer,
                  conversation: conversation,
                  systemPrompt: systemPrompt
              ) else { return conversation }
        var turns = conversation
        turns[index].text = recorded
        return turns
    }

    /// Runs a planned turn through the FIFO queue to completion. Shared by
    /// both adapters' non-streaming paths. The slot-open signal stamps the
    /// box's queue wait so it can be split from generation time.
    nonisolated private static func runTurn(plan: TurnPlan, queue: TurnExecutionQueue, logBox: ServerRequestLogBox?) async throws -> TurnResult {
        try await queue.perform({ claim in
            try await withTurnToolMask(plan.maskSchemas) {
                let launch: CompletionLaunch
                do {
                    launch = try CompletionEngine.shared.launchTurn(plan.request, claim: claim)
                } catch {
                    throw wireLaunchError(error)
                }
                return try await withTaskCancellationHandler {
                    for await event in launch.stream {
                        switch event {
                        case .failedTurn(let failure):
                            // The throw unwinds before any success-path stamping, so the
                            // tokens generated before the failure must reach the row here.
                            Self.recordFailedTurnTokenCount(logBox: logBox, failure: failure)
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
        }, queueWaitTimeout: resolvedQueueWaitTimeout, onSlotOpened: { logBox?.noteSlotOpened() })
    }

    nonisolated private static func nonStreamingResponse(plan: TurnPlan, queue: TurnExecutionQueue, logBox: ServerRequestLogBox?) async throws -> OpenAIChatResponse {
        let result = try await runTurn(plan: plan, queue: queue, logBox: logBox)

        let toolCalls = Self.turnEmittedToolCalls(result)
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
        Self.recordTurnOutcome(logBox: logBox, result: result, finishReason: finishReason)
        return OpenAIChatResponse(
            id: "chatcmpl-\(UUID().uuidString.prefix(24))",
            created: Int(Date().timeIntervalSince1970),
            model: plan.modelName,
            choices: [.init(index: 0, message: message, finishReason: finishReason)],
            usage: .init(promptTokens: plan.promptTokenCount, completionTokens: result.tokensGenerated)
        )
    }

    // MARK: Legacy text completions

    nonisolated private static func completionsNonStreamingResponse(
        plan: TurnPlan,
        apiRequest: OpenAICompletionsRequest,
        queue: TurnExecutionQueue,
        logBox: ServerRequestLogBox?
    ) async throws -> OpenAICompletionsResponse {
        let result = try await runTurn(plan: plan, queue: queue, logBox: logBox)
        let text = Self.completionsText(
            thinking: result.thinkingText,
            response: result.responseText,
            echoPrompt: apiRequest.echo == true ? (apiRequest.promptText ?? "") : nil
        )
        // This protocol cannot call a tool, so no tool call can be the finish
        // reason even if the family's dialect would have frozen on one.
        let finishReason = OpenAIAdapter.finishReasonString(result.finishReason, hasToolCalls: false)
        Self.recordTurnOutcome(logBox: logBox, result: result, finishReason: finishReason)
        return OpenAICompletionsResponse(
            id: "cmpl-\(UUID().uuidString.prefix(24))",
            created: Int(Date().timeIntervalSince1970),
            model: plan.modelName,
            choices: [.init(text: text, index: 0, finishReason: finishReason)],
            usage: .init(promptTokens: plan.promptTokenCount, completionTokens: result.tokensGenerated)
        )
    }

    /// The completion text this protocol returns, and why it is folded.
    ///
    /// `text_completion` has no `reasoning_content` field and no content-block
    /// list, so the reasoning channel is folded into `text` with the canonical
    /// close tag rather than dropped: a client displaying `choices[0].text` would
    /// otherwise show an answer whose reasoning silently vanished, and a streaming
    /// client would see dead air for the whole reasoning phase. The fold is the one
    /// the chat route's inline-reasoning mode emits, so the two agree there.
    ///
    /// `echo` is honoured rather than refused: upstream prepends the prompt to the
    /// completion so a client can display or score the pair, and the prompt here is
    /// the client's own text (never the rendered template, which the client never
    /// sent and could not re-derive).
    nonisolated static func completionsText(thinking: String, response: String, echoPrompt: String?) -> String {
        let body = inlineReasoningText(thinking: thinking, response: response)
        guard let echoPrompt, !echoPrompt.isEmpty else { return body }
        return echoPrompt + body
    }

    /// Stamp the terminal inference facts (tokens, finish reason, tool calls,
    /// captured response body) onto a completed non-streaming turn's log box.
    /// `finishReason` is optional because the Anthropic adapter maps some
    /// stop reasons to a JSON-absent nil.
    nonisolated private static func recordTurnOutcome(logBox: ServerRequestLogBox?, result: TurnResult, finishReason: String?) {
        let toolRecords = ServerRequestLog.capturedToolCalls(
            result.parsedToolCalls,
            serializeArguments: Self.toolArgumentsJSON
        )
        let responseBody = ServerRequestLog.capturedResponseBody(
            thinking: result.thinkingText,
            response: result.responseText
        )
        logBox?.modify { row in
            row.outputTokens = result.tokensGenerated
            row.finishReason = finishReason
            row.toolCalls = toolRecords
            row.responseBody = responseBody
        }
    }

    /// Stamp the partial facts of a streaming turn that died after emitting
    /// output (failure frame, disconnect, mid-tail writer throw): the latest
    /// frame's accumulated text and the failure's exact token count. No tool
    /// calls or finish reason — the row's `streamTermination` and error
    /// message already carry how it ended. Without this, a failure after
    /// output would record neither the tokens that streamed nor the text.
    nonisolated static func recordStreamedPartialOutcome(logBox: ServerRequestLogBox?, frame: CompletionFrame, tokenCount: Int) {
        let responseBody = ServerRequestLog.capturedResponseBody(
            thinking: frame.thinkingText,
            response: frame.responseText
        )
        logBox?.modify { row in
            row.outputTokens = tokenCount
            row.responseBody = responseBody
        }
    }

    /// Stamp a non-streaming turn failure's generated-token count onto the
    /// log row. `recordTurnOutcome` runs only on success and `runTurn` throws
    /// on `.failedTurn`, so without this a failure after output would leave
    /// the persisted error row's output-token count nil — mirroring the
    /// streaming paths' `recordStreamedPartialOutcome` minus the streamed
    /// text, which the non-streaming pipeline does not retain per row.
    nonisolated static func recordFailedTurnTokenCount(logBox: ServerRequestLogBox?, failure: TurnFailure) {
        logBox?.modify { row in
            row.outputTokens = failure.tokensGenerated
        }
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

    nonisolated private static func streamingResponse(apiRequest: OpenAIChatRequest, queue: TurnExecutionQueue, logBox: ServerRequestLogBox?) -> Response {
        let encoder = JSONEncoder()
        let id = "chatcmpl-\(UUID().uuidString.prefix(24))"
        let created = Int(Date().timeIntervalSince1970)
        let stopSequences = (apiRequest.stop ?? []).filter { !$0.isEmpty }
        let model = apiRequest.model ?? CompletionEngine.shared.loadedModelInfo?.displayName ?? "dynamoe"
        // The protocol asks the client to opt into the trailing usage chunk, and it is
        // the one chunk whose `choices` is empty. Sending it regardless is not a
        // harmless extra: a client that indexes `choices[0]` on every chunk — a common
        // shape in OpenAI SDK wrappers — throws on it, and only ever against a server
        // that volunteers the chunk. So the request decides, as it does upstream.
        let includeUsage = apiRequest.streamOptions?.includeUsage ?? false

        // This request's row is finalized when the stream drains (the HTTP
        // response is already a bare 200 by then), so claim the box: the wire
        // middleware steps aside and the body closure records below with the
        // stream's terminal facts.
        if let logBox {
            // claim() must run even in release builds (assert() elides its
            // autoclosure argument under -O, which would skip the claim and
            // let the middleware double-record); the assert just turns a
            // future double-record regression into a test-time failure.
            let claimed = logBox.claim()
            assert(claimed, "the streaming writer must win its request-log box claim exactly once")
            logBox.modify { row in
                row.wasStreaming = true
            }
        }

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
            let (relay, task) = await queue.streamTurn({ claim, emit in
                do {
                    let plan = try ServerController.buildTurnPlan(apiRequest)
                    promptTokensBox.set(plan.promptTokenCount)
                    logBox?.modify { row in
                        // The row names the model that actually ran, not the
                        // client's label: the plan resolves the loaded model
                        // behind an alias (or a model switch while queued)
                        // once the slot drains, mirroring the non-streaming
                        // rows. A turn that never plans carries no model.
                        row.model = plan.modelName
                    }
                    try await ServerController.withTurnToolMask(plan.maskSchemas) {
                        let launch = try CompletionEngine.shared.launchTurn(plan.request, claim: claim)
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
                    }
                } catch {
                    emit(.launchFailed(wireLaunchError(error)))
                }
            }, queueWaitTimeout: resolvedQueueWaitTimeout, onSlotOpened: { logBox?.noteSlotOpened() })
            defer { task.cancel() }

            // Request-log terminal state. `clientDisconnected` is the default:
            // any path that ends the stream without one of the terminal cases
            // below (a writer throw on a dropped socket) is a disconnect. The
            // defer records exactly one row no matter how the stream ends.
            var logTermination: ServerRequestLog.StreamTermination = .clientDisconnected
            var logErrorMessage: String?
            var logFirstTokenMs: Double?
            var logFinish: TurnResult?
            // A turn that fails — or a socket that drops — after output still
            // owes its row the partial response and token count: retain the
            // latest frame for the defer's fallback enrichment.
            var logFailure: TurnFailure?
            var logPartialFrame: CompletionFrame?
            defer {
                if let logBox {
                    logBox.modify { row in
                        row.httpStatus = 200
                        row.durationMs = logBox.elapsedMs()
                        row.firstTokenMs = logFirstTokenMs
                        row.streamTermination = logTermination
                        if promptTokensBox.get() > 0 { row.inputTokens = promptTokensBox.get() }
                        if let message = logErrorMessage { row.errorMessage = message }
                    }
                    if let result = logFinish {
                        recordTurnOutcome(
                            logBox: logBox,
                            result: result,
                            finishReason: OpenAIAdapter.finishReasonString(result.finishReason, hasToolCalls: !result.parsedToolCalls.isEmpty)
                        )
                    } else if let frame = logPartialFrame {
                        recordStreamedPartialOutcome(
                            logBox: logBox,
                            frame: frame,
                            tokenCount: logFailure?.tokensGenerated ?? frame.tokensGenerated
                        )
                    } else if let failure = logFailure {
                        // A prefill failure emits `.failedTurn` with no frame and
                        // no response text; the failure still knows how many
                        // tokens were generated, so the row keeps that count
                        // instead of falling back to nil.
                        recordFailedTurnTokenCount(logBox: logBox, failure: failure)
                    }
                    ServerRequestLogRecorder.shared.record(logBox.snapshot())
                }
            }
            var finishedCleanly = false
            for await event in relay {
                switch event {
                case .ping(let comment):
                    try await ping(comment)
                case .launchFailed(let apiError):
                    logTermination = .errorFrame
                    // The row records the log-safe rendering; the wire envelope
                    // below keeps echoing the client's values back to the client.
                    logErrorMessage = apiError.recordableMessage
                    let envelope = OpenAIErrorEnvelope(error: .init(message: apiError.message, type: apiError.wireType.rawValue, code: apiError.code))
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
                    if logFirstTokenMs == nil { logFirstTokenMs = logBox?.elapsedMs() }
                    logPartialFrame = frame
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
                    logTermination = .errorFrame
                    logFailure = failure
                    logErrorMessage = failure.cancelled ? "Generation cancelled" : failure.message
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
                    // Capture the final facts before any fallible terminal
                    // write: a write throwing after this point (a dropped
                    // socket, say) still records the final token count,
                    // response, and tool calls instead of falling back to the
                    // last partial frame — or worse, recording none at all.
                    // The `.completed` transition below still marks the
                    // stream only after the terminal writes succeed.
                    logFinish = result
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
                    // Final usage chunk, when the client asked for it with
                    // `stream_options.include_usage`.
                    let usageTail = OpenAIUsage(promptTokens: promptTokensBox.get(), completionTokens: result.tokensGenerated)
                    if includeUsage, let data = try? JSONSerialization.data(withJSONObject: [
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
                    logTermination = .completed
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

    /// The legacy route's stream: `text_completion` frames carrying `choices[].text`
    /// deltas, terminated by `[DONE]`.
    ///
    /// It keeps the chat writer's three disciplines because they are properties of
    /// this server rather than of that protocol: append-only deltas (a repaired
    /// snapshot is never re-sent), the suffix that could still grow into a stop
    /// sequence is withheld until it is safe, and the queue's heartbeats keep a
    /// waiting client from timing out. Reasoning has no field here, so it is folded
    /// into `text` exactly as the non-streaming body folds it, which also means the
    /// stream is never silent through the reasoning phase.
    nonisolated private static func completionsStreamingResponse(
        apiRequest: OpenAICompletionsRequest,
        queue: TurnExecutionQueue,
        logBox: ServerRequestLogBox?
    ) -> Response {
        let encoder = JSONEncoder()
        let id = "cmpl-\(UUID().uuidString.prefix(24))"
        let created = Int(Date().timeIntervalSince1970)
        let stopSequences = (apiRequest.stop ?? []).filter { !$0.isEmpty }
        let model = apiRequest.model ?? CompletionEngine.shared.loadedModelInfo?.displayName ?? "dynamoe"
        // Same opt-in rule as the chat route, for the same reason: the usage chunk is
        // the one chunk whose `choices` is empty, and a client that indexes
        // `choices[0]` on every chunk throws on it.
        let includeUsage = apiRequest.streamOptions?.includeUsage ?? false
        let echoPrompt = apiRequest.echo == true ? (apiRequest.promptText ?? "") : nil

        if let logBox {
            let claimed = logBox.claim()
            assert(claimed, "the streaming writer must win its request-log box claim exactly once")
            logBox.modify { row in
                row.wasStreaming = true
            }
        }

        func chunkBuffer(_ text: String, finishReason: String?) throws -> ByteBuffer {
            let chunk = OpenAICompletionsChunk(
                id: id,
                created: created,
                model: model,
                choices: [.init(text: text, index: 0, finishReason: finishReason)]
            )
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
            func sseChunk(_ text: String, finishReason: String? = nil) async throws {
                try await sse(try chunkBuffer(text, finishReason: finishReason))
            }

            var sentThinking = ""
            var sentContent = ""
            var reasoningClosed = false
            let promptTokensBox = PromptCountBox()

            /// Closes the folded reasoning before any answer text, so the tag never
            /// straddles the response and the streamed text matches the shape the
            /// non-streaming body produces.
            func closeReasoning() async throws {
                guard !reasoningClosed, !sentThinking.isEmpty else { return }
                reasoningClosed = true
                try await sseChunk(TurnTextSplitter.canonicalReasoningCloseTag + "\n\n")
            }

            if let echoPrompt, !echoPrompt.isEmpty {
                try await sseChunk(echoPrompt)
            }

            let (relay, task) = await queue.streamTurn({ claim, emit in
                do {
                    let plan = try ServerController.buildTurnPlan(apiRequest)
                    promptTokensBox.set(plan.promptTokenCount)
                    logBox?.modify { row in
                        // The row names the model that actually ran, not the
                        // client's label: the plan resolves the loaded model behind
                        // an alias (or a model switch while queued) once the slot
                        // drains. A turn that never plans carries no model.
                        row.model = plan.modelName
                    }
                    try await ServerController.withTurnToolMask(plan.maskSchemas) {
                        let launch = try CompletionEngine.shared.launchTurn(plan.request, claim: claim)
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
                    }
                } catch {
                    emit(.launchFailed(wireLaunchError(error)))
                }
            }, queueWaitTimeout: resolvedQueueWaitTimeout, onSlotOpened: { logBox?.noteSlotOpened() })
            defer { task.cancel() }

            var logTermination: ServerRequestLog.StreamTermination = .clientDisconnected
            var logErrorMessage: String?
            var logFirstTokenMs: Double?
            var logFinish: TurnResult?
            var logFailure: TurnFailure?
            var logPartialFrame: CompletionFrame?
            defer {
                if let logBox {
                    logBox.modify { row in
                        row.httpStatus = 200
                        row.durationMs = logBox.elapsedMs()
                        row.firstTokenMs = logFirstTokenMs
                        row.streamTermination = logTermination
                        if promptTokensBox.get() > 0 { row.inputTokens = promptTokensBox.get() }
                        if let message = logErrorMessage { row.errorMessage = message }
                    }
                    if let result = logFinish {
                        recordTurnOutcome(
                            logBox: logBox,
                            result: result,
                            finishReason: OpenAIAdapter.finishReasonString(result.finishReason, hasToolCalls: false)
                        )
                    } else if let frame = logPartialFrame {
                        recordStreamedPartialOutcome(
                            logBox: logBox,
                            frame: frame,
                            tokenCount: logFailure?.tokensGenerated ?? frame.tokensGenerated
                        )
                    } else if let failure = logFailure {
                        recordFailedTurnTokenCount(logBox: logBox, failure: failure)
                    }
                    ServerRequestLogRecorder.shared.record(logBox.snapshot())
                }
            }
            var finishedCleanly = false
            for await event in relay {
                switch event {
                case .ping(let comment):
                    try await ping(comment)
                case .launchFailed(let apiError):
                    logTermination = .errorFrame
                    logErrorMessage = apiError.recordableMessage
                    let envelope = OpenAIErrorEnvelope(error: .init(
                        message: apiError.message,
                        type: apiError.wireType.rawValue,
                        code: apiError.code
                    ))
                    if let data = try? encoder.encode(envelope) {
                        try await sse(ByteBuffer(data: data))
                    }
                    try await sseRaw("[DONE]")
                    finishedCleanly = true
                case .engine(.started):
                    // This protocol has no opening role chunk: the first frame that
                    // carries text is the first frame.
                    continue
                case .engine(.priming):
                    try await ping("priming weights")
                case .engine(.prefillProgress(let status, _, _)):
                    if let status = status {
                        try await ping("prefill: \(status)")
                    }
                case .engine(.frame(let frame)):
                    if logFirstTokenMs == nil { logFirstTokenMs = logBox?.elapsedMs() }
                    logPartialFrame = frame
                    let thinkingDelta = Self.safeStreamDelta(
                        accumulated: frame.thinkingText,
                        sent: sentThinking,
                        stopSequences: stopSequences
                    )
                    if !thinkingDelta.isEmpty {
                        sentThinking += thinkingDelta
                        try await sseChunk(thinkingDelta)
                    }
                    let contentDelta = Self.safeStreamDelta(
                        accumulated: frame.responseText,
                        sent: sentContent,
                        stopSequences: stopSequences
                    )
                    if !contentDelta.isEmpty {
                        try await closeReasoning()
                        sentContent += contentDelta
                        try await sseChunk(contentDelta)
                    }
                case .engine(.failedTurn(let failure)):
                    logTermination = .errorFrame
                    logFailure = failure
                    logErrorMessage = failure.cancelled ? "Generation cancelled" : failure.message
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
                    // Capture the final facts before any fallible terminal write, so
                    // a dropped socket still records the token count and response
                    // rather than falling back to the last partial frame.
                    logFinish = result
                    let tailThinking = Self.safeStreamDelta(
                        accumulated: result.thinkingText,
                        sent: sentThinking,
                        stopSequences: []
                    )
                    if !tailThinking.isEmpty {
                        sentThinking += tailThinking
                        try await sseChunk(tailThinking)
                    }
                    try await closeReasoning()
                    let tailContent = Self.safeStreamDelta(
                        accumulated: result.responseText,
                        sent: sentContent,
                        stopSequences: []
                    )
                    if !tailContent.isEmpty {
                        try await sseChunk(tailContent)
                    }
                    // No tool channel on this protocol, so the terminal frame names
                    // one of the two reasons a text completion can have.
                    try await sseChunk("", finishReason: OpenAIAdapter.finishReasonString(result.finishReason, hasToolCalls: false))
                    let completionTokens = result.tokensGenerated
                    if includeUsage, let data = try? JSONSerialization.data(withJSONObject: [
                        "id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [],
                        "usage": [
                            "prompt_tokens": promptTokensBox.get(),
                            "completion_tokens": completionTokens,
                            // The chat route's usage object always carries the sum;
                            // this one is built by hand because its chunk has an
                            // empty `choices`, so the two must not disagree.
                            "total_tokens": promptTokensBox.get() + completionTokens,
                        ],
                    ]) {
                        try await sse(ByteBuffer(data: data))
                    }
                    try await sseRaw("[DONE]")
                    logTermination = .completed
                    finishedCleanly = true
                }
                if finishedCleanly { break }
            }
            if !finishedCleanly {
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

    nonisolated private static func anthropicNonStreamingResponse(plan: TurnPlan, queue: TurnExecutionQueue, logBox: ServerRequestLogBox?) async throws -> AnthropicMessageResponse {
        let result = try await runTurn(plan: plan, queue: queue, logBox: logBox)

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

        Self.recordTurnOutcome(
            logBox: logBox,
            result: result,
            finishReason: AnthropicAdapter.finishReasonString(result.finishReason, hasToolCalls: !result.parsedToolCalls.isEmpty)
        )
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

    nonisolated private static func anthropicStreamingResponse(apiRequest: AnthropicMessagesRequest, queue: TurnExecutionQueue, logBox: ServerRequestLogBox?) -> Response {
        let encoder = JSONEncoder()
        let id = "msg-\(UUID().uuidString.prefix(24))"
        let model = apiRequest.model ?? CompletionEngine.shared.loadedModelInfo?.displayName ?? "dynamoe"
        let stopSequences = (apiRequest.stopSequences ?? []).filter { !$0.isEmpty }

        // Same ownership hand-off as the OpenAI writer: the stream's terminal
        // facts (not the bare 200) are what the row records.
        if let logBox {
            // claim() must run even in release builds (assert() elides its
            // autoclosure argument under -O, which would skip the claim and
            // let the middleware double-record); the assert just turns a
            // future double-record regression into a test-time failure.
            let claimed = logBox.claim()
            assert(claimed, "the streaming writer must win its request-log box claim exactly once")
            logBox.modify { row in
                row.wasStreaming = true
            }
        }

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

            let (relay, task) = await queue.streamTurn({ claim, emit in
                do {
                    let plan = try ServerController.buildTurnPlan(apiRequest)
                    promptTokensBox.set(plan.promptTokenCount)
                    logBox?.modify { row in
                        // Same contract as the OpenAI writer: the row names the
                        // resolved model, and a turn that never plans carries
                        // no model at all.
                        row.model = plan.modelName
                    }
                    try await ServerController.withTurnToolMask(plan.maskSchemas) {
                        let launch = try CompletionEngine.shared.launchTurn(plan.request, claim: claim)
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
                    }
                } catch {
                    emit(.launchFailed(wireLaunchError(error)))
                }
            }, queueWaitTimeout: resolvedQueueWaitTimeout, onSlotOpened: { logBox?.noteSlotOpened() })
            defer { task.cancel() }

            // Request-log terminal state, mirroring the OpenAI writer: the
            // defer records exactly one row no matter how the stream ends.
            var logTermination: ServerRequestLog.StreamTermination = .clientDisconnected
            var logErrorMessage: String?
            var logFirstTokenMs: Double?
            var logFinish: TurnResult?
            // Fallback facts for a turn that fails or disconnects after
            // output, mirroring the OpenAI writer.
            var logFailure: TurnFailure?
            var logPartialFrame: CompletionFrame?
            defer {
                if let logBox {
                    logBox.modify { row in
                        row.httpStatus = 200
                        row.durationMs = logBox.elapsedMs()
                        row.firstTokenMs = logFirstTokenMs
                        row.streamTermination = logTermination
                        if promptTokensBox.get() > 0 { row.inputTokens = promptTokensBox.get() }
                        if let message = logErrorMessage { row.errorMessage = message }
                    }
                    if let result = logFinish {
                        recordTurnOutcome(
                            logBox: logBox,
                            result: result,
                            finishReason: AnthropicAdapter.finishReasonString(result.finishReason, hasToolCalls: !result.parsedToolCalls.isEmpty)
                        )
                    } else if let frame = logPartialFrame {
                        recordStreamedPartialOutcome(
                            logBox: logBox,
                            frame: frame,
                            tokenCount: logFailure?.tokensGenerated ?? frame.tokensGenerated
                        )
                    } else if let failure = logFailure {
                        // Mirrors the OpenAI writer: a prefill failure with no
                        // frame still knows its generated-token count.
                        recordFailedTurnTokenCount(logBox: logBox, failure: failure)
                    }
                    ServerRequestLogRecorder.shared.record(logBox.snapshot())
                }
            }
            var finishedCleanly = false
            for await event in relay {
                switch event {
                case .ping:
                    try await ping()
                case .launchFailed(let apiError):
                    logTermination = .errorFrame
                    // The row records the log-safe rendering; the wire envelope
                    // below keeps echoing the client's values back to the client.
                    logErrorMessage = apiError.recordableMessage
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
                    if logFirstTokenMs == nil { logFirstTokenMs = logBox?.elapsedMs() }
                    logPartialFrame = frame
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
                    logTermination = .errorFrame
                    logFailure = failure
                    logErrorMessage = failure.cancelled ? "Generation cancelled" : failure.message
                    try await ensureStart()
                    try await closeOpenBlocks()
                    try await errorEvent(failure.cancelled ? "Generation cancelled" : failure.message)
                    finishedCleanly = true
                case .engine(.finished(let result)):
                    // Same ordering contract as the OpenAI writer: the row's
                    // final facts are captured before the first fallible
                    // write, `.completed` marks only a fully written stream.
                    logFinish = result
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
                    logTermination = .completed
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
