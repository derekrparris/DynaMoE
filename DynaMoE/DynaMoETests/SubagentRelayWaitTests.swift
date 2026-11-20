//
//  SubagentRelayWaitTests.swift
//  DynaMoETests
//
//  The Stop-sensitive teardown wait for a conversation's delegated runs.
//  Pure logic: the predicate is injected, so no subagent is spawned.
//

import XCTest
@testable import DynaMoE

final class SubagentRelayWaitTests: XCTestCase {
    /// A counter the injected predicate can write from any actor.
    private final class Calls: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func note() {
            lock.lock()
            count += 1
            lock.unlock()
        }
        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
    }

    /// The wait is awaited by a generation task the operator can Stop. A wait
    /// that ignored cancellation kept that task alive for up to two minutes and
    /// then relayed a report into the stopped conversation.
    func testAStoppedWaitReturnsImmediately() async {
        let started = Date()
        let task = Task {
            await SubagentRelayWait.wait(sessionId: UUID(), timeout: 60) { _ in true }
        }
        try? await Task.sleep(nanoseconds: 150_000_000)
        task.cancel()
        await task.value

        XCTAssertLessThan(
            Date().timeIntervalSince(started), 2,
            "cancellation must end the wait, not run it to the deadline"
        )
    }

    /// The predicate is what decides the wait, so a timing-only assertion
    /// would pass for a wait that returned early for any other reason. This
    /// one asserts the CALL: a predicate answering false on its second ask
    /// must have been asked twice, and the wait must end there rather than at
    /// the deadline it was given.
    func testTheWaitReturnsAsSoonAsNoRunIsActive() async {
        let calls = Calls()
        let started = Date()
        await SubagentRelayWait.wait(sessionId: UUID(), timeout: 30) { _ in
            calls.note()
            // Active once, then done: only a wait that CONSULTED the
            // predicate again can end before the deadline.
            return calls.value < 2
        }
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertEqual(calls.value, 2, "the predicate must be re-asked until it answers false")
        XCTAssertLessThan(elapsed, 2, "it must return once the runs are done, not at the deadline")
    }

    func testTheWaitStopsAtItsDeadline() async {
        let started = Date()
        await SubagentRelayWait.wait(sessionId: UUID(), timeout: 0.6) { _ in true }
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertGreaterThanOrEqual(elapsed, 0.5)
        XCTAssertLessThan(elapsed, 3)
    }
}
