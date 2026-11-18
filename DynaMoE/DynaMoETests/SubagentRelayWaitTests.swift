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

    func testTheWaitReturnsAsSoonAsNoRunIsActive() async {
        let started = Date()
        await SubagentRelayWait.wait(sessionId: UUID(), timeout: 30) { _ in false }
        XCTAssertLessThan(Date().timeIntervalSince(started), 1)
    }

    func testTheWaitStopsAtItsDeadline() async {
        let started = Date()
        await SubagentRelayWait.wait(sessionId: UUID(), timeout: 0.6) { _ in true }
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertGreaterThanOrEqual(elapsed, 0.5)
        XCTAssertLessThan(elapsed, 3)
    }
}
