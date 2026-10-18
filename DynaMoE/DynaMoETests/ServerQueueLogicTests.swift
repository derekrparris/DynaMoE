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

    func testStreamTurnRelaysSlotOpenedOncePredecessorDrains() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let workRan = FlagBox()

        // Turn 1 runs immediately and hangs on the gate; turn 2 queues behind it.
        let (firstStream, firstTask) = await queue.streamTurn { _ in
            await gate.wait()
        }
        let (secondStream, secondTask) = await queue.streamTurn { emit in
            workRan.set(true)
            emit(.ping(comment: "work began"))
        }
        try await waitUntil { await queue.pendingCount == 2 }
        XCTAssertFalse(workRan.get(), "the queued turn must not run behind the gated leader")

        gate.open()

        var secondEvents: [QueuedTurnEvent] = []
        for await event in secondStream {
            secondEvents.append(event)
        }
        var firstEvents: [QueuedTurnEvent] = []
        for await event in firstStream {
            firstEvents.append(event)
        }

        // Heartbeat pings may legitimately precede the slot event under
        // scheduler load, but the slot-opened event itself must precede the
        // work it unblocks.
        func isSlotOpened(_ event: QueuedTurnEvent) -> Bool {
            if case .slotOpened = event { return true }
            return false
        }
        func isWorkPing(_ event: QueuedTurnEvent) -> Bool {
            if case .ping(comment: "work began") = event { return true }
            return false
        }
        let slotIndex = secondEvents.firstIndex(where: isSlotOpened)
        let workIndex = secondEvents.firstIndex(where: isWorkPing)
        XCTAssertNotNil(slotIndex, "the queued turn must relay exactly one slot-opened event")
        XCTAssertNotNil(workIndex, "the queued turn's work must run and relay its ping")
        if let slotIndex, let workIndex {
            XCTAssertLessThan(slotIndex, workIndex, "slot-opened must precede the work it unblocks")
        }
        XCTAssertEqual(secondEvents.filter(isSlotOpened).count, 1, "exactly one slot-opened event per turn")
        XCTAssertTrue(workRan.get(), "the work must run after the slot opened")
        XCTAssertTrue(firstEvents.contains(where: isSlotOpened), "the leader's slot also opened and must relay it")
        XCTAssertEqual(firstTask.isCancelled, false)
        XCTAssertEqual(secondTask.isCancelled, false)
        try await waitUntil { await queue.pendingCount == 0 }
    }

    func testStreamTurnNeverRelaysSlotOpenedWhenQueueWaitTimesOut() async throws {
        let queue = TurnExecutionQueue()
        let gate = Gate()
        let workRan = FlagBox()

        let (firstStream, firstTask) = await queue.streamTurn { _ in
            await gate.wait()
        }
        let (secondStream, secondTask) = await queue.streamTurn({ emit in
            workRan.set(true)
            emit(.ping(comment: "work began"))
        }, queueWaitTimeout: 0.2)

        try await waitUntil { await queue.pendingCount == 2 }
        try await Task.sleep(nanoseconds: 350_000_000)

        var sawSlotOpened = false
        var sawTimeoutFailure = false
        for await event in secondStream {
            switch event {
            case .slotOpened:
                sawSlotOpened = true
            case .launchFailed(let apiError):
                sawTimeoutFailure = apiError.message.contains("timed out")
            default:
                break
            }
        }
        XCTAssertFalse(sawSlotOpened, "a timed-out turn never got the slot, so it must not claim a queue wait")
        XCTAssertTrue(sawTimeoutFailure)
        XCTAssertFalse(workRan.get())

        gate.open()
        var firstSawSlotOpened = false
        for await event in firstStream {
            if case .slotOpened = event { firstSawSlotOpened = true }
        }
        XCTAssertTrue(firstSawSlotOpened, "the leader's slot-opened event must still be relayed")
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
}
