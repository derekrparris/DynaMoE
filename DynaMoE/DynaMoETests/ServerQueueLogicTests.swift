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
}
