//
//  ServerQueueLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the server's FIFO turn-queue bookkeeping and
//  request statistics — no sockets, no models, no MainActor required.
//

import XCTest
import HTTPTypes
@testable import DynaMoE

final class ServerQueueLogicTests: XCTestCase {

    /// Continuation gate that holds queued work open until the test opens it.
    private final class Gate: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Void, Never>?
        private var opened = false

        func wait() async {
            await withCheckedContinuation { cont in
                lock.lock()
                if opened {
                    lock.unlock()
                    cont.resume()
                    return
                }
                continuation = cont
                lock.unlock()
            }
        }

        func open() {
            lock.lock()
            let cont = continuation
            continuation = nil
            opened = true
            lock.unlock()
            cont?.resume()
        }
    }

    private func waitUntil(
        timeout: TimeInterval = 5,
        _ condition: @escaping @Sendable () async -> Bool
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await condition() { return }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("Condition not met within \(timeout)s")
    }

    /// Lock-guarded bool for cross-task assertions (a plain captured var
    /// would race between the queue task and the test task).
    private final class FlagBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false

        func set(_ newValue: Bool) {
            lock.lock()
            value = newValue
            lock.unlock()
        }

        func get() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    func testPendingCountTracksQueuedPerformWork() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()

        let first = Task { try await queue.perform { await gate.wait(); return 1 } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform { return 2 } }

        try await waitUntil { await queue.pendingCount == 2 }
        let observedPending = await queue.pendingCount
        XCTAssertEqual(observedPending, 2, "both turns must count as pending while the leader is gated")

        gate.open()
        let firstValue = try await first.value
        let secondValue = try await second.value
        XCTAssertEqual(firstValue, 1)
        XCTAssertEqual(secondValue, 2, "FIFO order: the gated turn must complete before the queued one")

        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testPendingCountDrainsCancelledStreamingTurns() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()

        // Turn 1 runs immediately and hangs on the gate; turn 2 queues behind it.
        let (firstStream, firstTask) = await queue.streamTurn { _ in
            await gate.wait()
        }
        _ = firstStream
        let (secondStream, secondTask) = await queue.streamTurn { _ in }
        _ = secondStream
        let twoQueued = await queue.pendingCount
        XCTAssertEqual(twoQueued, 2, "both the running and the queued turn count as pending")

        // Cancel the queued turn while it waits for the single-flight slot.
        secondTask.cancel()

        // Release the running turn; the cancelled turn must bail at its
        // cancellation guard and drain the queue instead of running its work.
        gate.open()
        try await waitUntil { await queue.pendingCount == 0 }
        XCTAssertEqual(firstTask.isCancelled, false)
        XCTAssertEqual(secondTask.isCancelled, true)
    }

    func testPerformThrowsWhenQueueWaitTimeoutExpires() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()

        // Turn 1 runs immediately and hangs on the gate; turn 2 queues behind
        // it with a short queue-wait timeout. Await the leader's enqueue first so
        // the two are never ordered the wrong way under scheduler load.
        let first = Task { try await queue.perform { await gate.wait(); return 1 } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform({ () async throws -> Int in 2 }, queueWaitTimeout: 0.2) }

        try await waitUntil { await queue.pendingCount == 2 }

        // Let the queued turn's deadline pass while it waits, then release.
        try await Task.sleep(nanoseconds: 350_000_000)
        gate.open()

        do {
            let value = try await second.value
            XCTFail("the queued turn must time out, not run (got \(value))")
        } catch let error as ServerAPIError {
            XCTAssertEqual(error.status, .requestTimeout)
            XCTAssertTrue(error.message.contains("timed out"), "the timeout error must explain itself")
        }
        let firstValue = try await first.value
        XCTAssertEqual(firstValue, 1, "the leader must complete normally")

        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testStreamTurnEmitsLaunchFailedWhenQueueWaitTimeoutExpires() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()

        // Turn 1 runs immediately and hangs on the gate; turn 2 queues behind
        // it with a short queue-wait timeout.
        let (firstStream, firstTask) = await queue.streamTurn { _ in
            await gate.wait()
        }
        _ = firstStream
        let (secondStream, secondTask) = await queue.streamTurn({ _ in }, queueWaitTimeout: 0.2)

        try await waitUntil { await queue.pendingCount == 2 }

        try await Task.sleep(nanoseconds: 350_000_000)
        gate.open()

        var sawTimeoutFailure = false
        for await event in secondStream {
            if case .launchFailed(let apiError) = event {
                sawTimeoutFailure = apiError.message.contains("timed out")
            }
        }
        XCTAssertTrue(sawTimeoutFailure, "the queued streaming turn must surface a timeout launch failure instead of running")
        XCTAssertEqual(firstTask.isCancelled, false)
        XCTAssertEqual(secondTask.isCancelled, false)
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testPerformTimeoutFiresWhileLeaderGateClosed() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()

        let first = Task { try await queue.perform { await gate.wait(); return 1 } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform({ () async throws -> Int in 2 }, queueWaitTimeout: 0.2) }
        try await waitUntil { await queue.pendingCount == 2 }

        // The leader's gate stays shut well past the follower's deadline: the
        // timeout must still be delivered instead of waiting out the leader.
        let started = Date()
        do {
            let value = try await second.value
            XCTFail("the queued turn must time out without the leader draining (got \(value))")
        } catch let error as ServerAPIError {
            XCTAssertEqual(error.status, .requestTimeout)
        }
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertLessThan(elapsed, 1.5, "the timeout must not wait for the stalled predecessor")

        gate.open()
        let firstValue = try await first.value
        XCTAssertEqual(firstValue, 1)
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testTimedOutTurnKeepsFIFOBarrierForSuccessors() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let box = FlagBox()

        let first = Task { try await queue.perform { () async throws -> Int in
            await gate.wait()
            box.set(true)
            return 1
        } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform({ () async throws -> Int in 2 }, queueWaitTimeout: 0.2) }
        try await waitUntil { await queue.pendingCount == 2 }
        let third = Task { try await queue.perform { () async throws -> Int in
            guard box.get() else {
                XCTFail("the successor must not run before the leader's work drained")
                return -1
            }
            return 3
        } }
        try await waitUntil { await queue.pendingCount == 3 }

        // The middle turn times out while the leader is gated.
        do {
            _ = try await second.value
            XCTFail("the middle turn must time out")
        } catch let error as ServerAPIError {
            XCTAssertEqual(error.status, .requestTimeout)
        }

        gate.open()
        let firstValue = try await first.value
        let thirdValue = try await third.value
        XCTAssertEqual(firstValue, 1)
        XCTAssertEqual(thirdValue, 3, "the turn behind a timed-out turn must still run after the leader drains")
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testCancellingCallerPreventsQueuedWorkFromLaunching() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let box = FlagBox()

        let first = Task { try await queue.perform { await gate.wait(); return 1 } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform { () async throws -> Int in
            box.set(true)
            return 2
        } }
        try await waitUntil { await queue.pendingCount == 2 }

        // The caller goes away while its turn is still queued.
        second.cancel()
        try await Task.sleep(nanoseconds: 100_000_000)

        gate.open()
        do {
            _ = try await second.value
            XCTFail("the cancelled caller's queued work must not run")
        } catch is CancellationError {
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        let firstValue = try await first.value
        XCTAssertEqual(firstValue, 1)
        XCTAssertFalse(box.get(), "cancelled work must never launch")
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testCancelLiveWorkCancelsServerOwnedTurnsOnly() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let cancelReachedWork = FlagBox()
        let workRan = FlagBox()

        // The running turn models the engine contract: its work observes
        // cancellation through a cancellation handler (like runTurn's
        // onCancel → engine handle cancel).
        let first = Task { try await queue.perform { () async throws -> Int in
            try await withTaskCancellationHandler {
                await gate.wait()
                try Task.checkCancellation()
                workRan.set(true)
                return 1
            } onCancel: {
                cancelReachedWork.set(true)
            }
        } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform { () async throws -> Int in
            workRan.set(true)
            return 2
        } }
        try await waitUntil { await queue.pendingCount == 2 }

        await queue.cancelLiveWork()

        gate.open()
        do {
            _ = try await first.value
            XCTFail("the running turn must be cancelled by shutdown")
        } catch is CancellationError {
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        do {
            _ = try await second.value
            XCTFail("the queued turn must be cancelled by shutdown")
        } catch is CancellationError {
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        XCTAssertTrue(cancelReachedWork.get(), "cancellation must reach the running work (engine handle analog)")
        XCTAssertFalse(workRan.get(), "no server-owned work may run after cancelLiveWork")
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testServerStatsCountsActiveAndTotalRequests() throws {
        let stats = ServerStats()
        XCTAssertEqual(stats.snapshot(), ServerStats.Snapshot(totalRequests: 0, activeRequests: 0))

        stats.beginRequest()
        stats.beginRequest()
        XCTAssertEqual(stats.snapshot(), ServerStats.Snapshot(totalRequests: 2, activeRequests: 2))

        stats.endRequest()
        stats.endRequest()
        stats.endRequest()
        let snapshot = stats.snapshot()
        XCTAssertEqual(snapshot.totalRequests, 2)
        XCTAssertEqual(snapshot.activeRequests, 0, "active count must clamp at zero")
    }

    // MARK: Slot-open signaling (queue wait vs generation time)

    func testStreamTurnInvokesSlotOpenedHookOncePredecessorDrains() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let workRan = FlagBox()
        let slotOpened = FlagBox()
        let hookRanBeforeWork = FlagBox()

        // Turn 1 runs immediately and hangs on the gate; turn 2 queues behind it.
        let (firstStream, firstTask) = await queue.streamTurn { _ in
            await gate.wait()
        }
        let (secondStream, secondTask) = await queue.streamTurn({ emit in
            // The queue invokes the hook immediately before the work runs,
            // so it must already be true here — and this records the ordering
            // without racing the producer task.
            hookRanBeforeWork.set(slotOpened.get())
            workRan.set(true)
            emit(.ping(comment: "work began"))
        }, onSlotOpened: { slotOpened.set(true) })
        try await waitUntil { await queue.pendingCount == 2 }
        XCTAssertFalse(workRan.get(), "the queued turn must not run behind the gated leader")
        XCTAssertFalse(slotOpened.get(), "the hook must not fire while the leader holds the slot")

        gate.open()

        var sawWorkPing = false
        for await event in secondStream {
            if case .ping(comment: "work began") = event { sawWorkPing = true }
        }
        for await _ in firstStream {}

        XCTAssertTrue(sawWorkPing, "the queued turn's work must run and relay its ping")
        XCTAssertTrue(workRan.get(), "the work must run after the slot opened")
        XCTAssertTrue(slotOpened.get(), "the hook must fire exactly once, when the slot opens")
        XCTAssertTrue(hookRanBeforeWork.get(), "the hook must fire before the work it unblocks, not from a later relay consumer")
        XCTAssertEqual(firstTask.isCancelled, false)
        XCTAssertEqual(secondTask.isCancelled, false)
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testStreamTurnSkipsSlotOpenedHookWhenQueueWaitTimesOut() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let workRan = FlagBox()
        let slotOpened = FlagBox()

        let (firstStream, firstTask) = await queue.streamTurn { _ in
            await gate.wait()
        }
        let (secondStream, secondTask) = await queue.streamTurn(
            { emit in
                workRan.set(true)
                emit(.ping(comment: "work began"))
            },
            queueWaitTimeout: 0.2,
            onSlotOpened: { slotOpened.set(true) }
        )

        try await waitUntil { await queue.pendingCount == 2 }
        try await Task.sleep(nanoseconds: 350_000_000)

        var sawTimeoutFailure = false
        for await event in secondStream {
            if case .launchFailed(let apiError) = event {
                sawTimeoutFailure = apiError.message.contains("timed out")
            }
        }
        XCTAssertFalse(slotOpened.get(), "a timed-out turn never got the slot, so the hook must not fire and the row must not claim a queue wait")
        XCTAssertTrue(sawTimeoutFailure)
        XCTAssertFalse(workRan.get())

        gate.open()
        for await _ in firstStream {}
        XCTAssertEqual(firstTask.isCancelled, false)
        XCTAssertEqual(secondTask.isCancelled, false)
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testPerformInvokesSlotOpenedOnlyWhenSlotOpens() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let slotOpened = FlagBox()

        let first = Task { try await queue.perform { await gate.wait(); return 1 } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform({ () async throws -> Int in 2 }, onSlotOpened: { slotOpened.set(true) }) }

        try await waitUntil { await queue.pendingCount == 2 }
        XCTAssertFalse(slotOpened.get(), "the hook must not fire while the leader holds the slot")

        gate.open()
        let secondValue = try await second.value
        XCTAssertEqual(secondValue, 2, "FIFO order: the queued turn runs after the leader drains")
        XCTAssertTrue(slotOpened.get(), "the hook must fire exactly once, when the slot opens")
        let firstValue = try await first.value
        XCTAssertEqual(firstValue, 1)
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testPerformSkipsSlotOpenedWhenQueueWaitTimesOut() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let slotOpened = FlagBox()

        let first = Task { try await queue.perform { await gate.wait(); return 1 } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform({ () async throws -> Int in 2 }, queueWaitTimeout: 0.2, onSlotOpened: { slotOpened.set(true) }) }
        try await waitUntil { await queue.pendingCount == 2 }

        do {
            _ = try await second.value
            XCTFail("the queued turn must time out, not run")
        } catch let error as ServerAPIError {
            XCTAssertEqual(error.status, .requestTimeout)
        }
        XCTAssertFalse(slotOpened.get(), "a timed-out turn never got the slot, so the hook must not fire")

        gate.open()
        let firstValue = try await first.value
        XCTAssertEqual(firstValue, 1)
        try await waitUntil { await queue.pendingCount == 0 }
    }

    /// Tracks how many turns were inside their work at the same time.
    private final class OverlapBox: @unchecked Sendable {
        private let lock = NSLock()
        private var live = 0
        private var peak = 0
        private var finished = 0

        func enter() {
            lock.lock()
            live += 1
            peak = max(peak, live)
            lock.unlock()
        }

        func leave() {
            lock.lock()
            live -= 1
            finished += 1
            lock.unlock()
        }

        func stats() -> (peak: Int, finished: Int) {
            lock.lock()
            defer { lock.unlock() }
            return (peak, finished)
        }
    }

    /// The queue's chain is READ AND ADVANCED WITHOUT A SUSPENSION in between, and
    /// that atomicity IS the queue: a turn that read a half-advanced tail would chain
    /// off the turn before it and run alongside it. The engine then refuses the second
    /// as "a generation is already active", so the symptom is a 503 under load rather
    /// than anything visible in this queue.
    ///
    /// A burst of independent enqueues is the case that exposes it, and the property
    /// it pins is peak concurrency rather than timing: the pre-fix queue reached a
    /// peak above 1 here, and the shipped one stays at 1 over five consecutive runs.
    func testConcurrentEnqueuesNeverRunTwoTurnsAtOnce() async throws {
        // Both paths through the wait, because they get their exclusivity from
        // different places: no deadline relies on the chain alone, and a deadline
        // adds the engine-slot probe on top.
        for queueWaitTimeout in [nil, 5.0] as [TimeInterval?] {
            let queue = TurnExecutionQueue(slotProbe: { true })
            let box = OverlapBox()
            let turns = 24

            await withTaskGroup(of: Void.self) { group in
                for _ in 0..<turns {
                    group.addTask {
                        _ = try? await queue.perform({ () async throws -> Int in
                            box.enter()
                            // Long enough for a second turn to overlap if the chain
                            // lets it, short enough to keep the case fast.
                            try? await Task.sleep(nanoseconds: 1_000_000)
                            box.leave()
                            return 0
                        }, queueWaitTimeout: queueWaitTimeout)
                    }
                }
                await group.waitForAll()
            }

            let stats = box.stats()
            let label = queueWaitTimeout == nil ? "unbounded" : "deadline"
            XCTAssertEqual(stats.finished, turns, "\(label): every enqueued turn must run")
            XCTAssertEqual(
                stats.peak, 1,
                "\(label): two turns must never hold the engine at once"
            )
            try await waitUntil { await queue.pendingCount == 0 }
        }
    }

    // MARK: An in-app generation holding the engine slot

    /// Stands in for `CompletionEngine.isActive` in the queue's slot probe.
    private final class SlotBox: @unchecked Sendable {
        private let lock = NSLock()
        private var free = false

        func setFree(_ newValue: Bool) {
            lock.lock()
            free = newValue
            lock.unlock()
        }

        func isFree() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return free
        }
    }

    /// Lock-guarded list so the order turns actually ran in can be asserted from
    /// the test task while the queue's tasks append.
    private final class OrderBox: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [Int] = []

        func append(_ value: Int) {
            lock.lock()
            items.append(value)
            lock.unlock()
        }

        func snapshot() -> [Int] {
            lock.lock()
            defer { lock.unlock() }
            return items
        }
    }

    /// The chat UI launches turns straight through `CompletionEngine.launchTurn`
    /// and never routes them through this queue, so an in-app generation holds the
    /// engine's ONE generation slot where the FIFO cannot see it. A request that
    /// arrives mid-answer must therefore queue behind it: the client asked first,
    /// the resource is busy for a bounded time, and the queue already has a
    /// deadline and a heartbeat. Before this, such a request was refused with 503
    /// before it had waited at all.
    func testPerformWaitsForAnInAppGenerationInsteadOfRefusing() async throws {
        let slot = SlotBox()
        let queue = TurnExecutionQueue(slotProbe: { slot.isFree() })
        let ran = FlagBox()

        let turn = Task { try await queue.perform { ran.set(true); return 7 } }
        try await waitUntil { await queue.pendingCount == 1 }
        try await Task.sleep(nanoseconds: 150_000_000)

        XCTAssertFalse(ran.get(), "the work must not run while the app holds the engine slot")
        XCTAssertFalse(turn.isCancelled)

        slot.setFree(true)
        let value = try await turn.value
        XCTAssertEqual(value, 7, "the queued request runs once the app's generation ends")
        XCTAssertTrue(ran.get())
        try await waitUntil { await queue.pendingCount == 0 }
    }

    /// The wait is bounded by the same deadline as every other queue wait, and it
    /// says which of the two things it waited on: a client debugging a 408 needs
    /// to know whether the queue was long or the app was busy.
    func testPerformTimesOutWaitingForAnInAppGeneration() async throws {
        let slot = SlotBox()
        let queue = TurnExecutionQueue(slotProbe: { slot.isFree() })
        let ran = FlagBox()
        let slotOpened = FlagBox()

        do {
            _ = try await queue.perform(
                { ran.set(true); return 1 },
                queueWaitTimeout: 0.2,
                onSlotOpened: { slotOpened.set(true) }
            )
            XCTFail("the request must be told it could not get the slot")
        } catch let error as ServerAPIError {
            XCTAssertEqual(error.status, .requestTimeout)
            XCTAssertTrue(error.message.contains("timed out"), "the timeout error must explain itself")
            XCTAssertTrue(
                error.message.contains("DynaMoE app"),
                "the message must name the in-app generation, not a queue of other requests"
            )
        }
        XCTAssertFalse(ran.get(), "a timed-out request never reaches the engine")
        XCTAssertFalse(slotOpened.get(), "no queue wait is billed, because no turn ran")
        try await waitUntil { await queue.pendingCount == 0 }
    }

    /// The same rule on the streaming path, where the refusal travels as an error
    /// frame rather than an HTTP status.
    func testStreamTurnEmitsLaunchFailedWhileAnInAppGenerationHoldsTheSlot() async throws {
        let slot = SlotBox()
        let queue = TurnExecutionQueue(slotProbe: { slot.isFree() })
        let ran = FlagBox()

        let (stream, _) = await queue.streamTurn({ _ in ran.set(true) }, queueWaitTimeout: 0.2)
        var failure: ServerAPIError?
        for await event in stream {
            if case .launchFailed(let apiError) = event { failure = apiError }
        }
        XCTAssertEqual(failure?.status, .requestTimeout)
        XCTAssertTrue(failure?.message.contains("DynaMoE app") == true)
        XCTAssertFalse(ran.get())
        try await waitUntil { await queue.pendingCount == 0 }
    }

    /// Queue wait is measured from request arrival, and the wait for an in-app
    /// generation is queue wait: the slot-open hook is what the log rows stamp
    /// `queueWaitMs` from, so firing it early would count the app's generation as
    /// this request's own.
    func testSlotOpenedHookWaitsForTheInAppGenerationToo() async throws {
        let slot = SlotBox()
        let queue = TurnExecutionQueue(slotProbe: { slot.isFree() })
        let slotOpened = FlagBox()

        let turn = Task {
            try await queue.perform(
                { () async throws -> Int in 3 },
                onSlotOpened: { slotOpened.set(true) }
            )
        }
        try await waitUntil { await queue.pendingCount == 1 }
        try await Task.sleep(nanoseconds: 150_000_000)
        XCTAssertFalse(slotOpened.get(), "the hook must not fire while the app holds the slot")

        slot.setFree(true)
        let value = try await turn.value
        XCTAssertEqual(value, 3)
        XCTAssertTrue(slotOpened.get(), "the hook fires when the slot genuinely opens")
        try await waitUntil { await queue.pendingCount == 0 }
    }

    /// One slot, two ways to hold it: a turn queued behind an in-app generation
    /// still queues behind the earlier SERVER turn first, and the FIFO order among
    /// requests survives the new wait.
    func testFIFOOrderSurvivesWaitingForAnInAppGeneration() async throws {
        let slot = SlotBox()
        let queue = TurnExecutionQueue(slotProbe: { slot.isFree() })
        let order = OrderBox()

        let first = Task { try await queue.perform { order.append(1); return 1 } }
        try await waitUntil { await queue.pendingCount == 1 }
        let second = Task { try await queue.perform { order.append(2); return 2 } }
        try await waitUntil { await queue.pendingCount == 2 }

        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(order.snapshot(), [], "nothing runs while the app holds the slot")

        slot.setFree(true)
        _ = try await first.value
        _ = try await second.value
        XCTAssertEqual(order.snapshot(), [1, 2], "requests keep the order they arrived in")
        try await waitUntil { await queue.pendingCount == 0 }
    }
}
