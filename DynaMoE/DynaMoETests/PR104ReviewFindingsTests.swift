//
//  PR104ReviewFindingsTests.swift
//  DynaMoETests
//
//  Regressions for the nine minors in the PR #104 review. Each case fails on
//  the code as reviewed and passes on the fix, and none needs a model.
//

import Foundation
import XCTest
@testable import DynaMoE

final class PR104ReviewFindingsTests: XCTestCase {

    // MARK: - #1: a released claim must not read as preempted

    /// The watermark this replaced answered "at or below the highest cancelled
    /// ticket", so a claim that was RELEASED normally and then outlived by a
    /// later preemption reported `wasCancelled == true`. `LocalAgentBackend`
    /// asks exactly that after a launch fails, and a real failure (a model
    /// swapped mid-run) would have been reported as "Cancelled" with its error
    /// discarded.
    func testAReleasedClaimIsNotReportedAsPreemptedByALaterOne() throws {
        let ledger = TurnSlotLedger()
        let released = try XCTUnwrap(ledger.tryClaim())
        released.release()

        let later = try XCTUnwrap(ledger.tryClaim())
        XCTAssertTrue(ledger.cancelClaim(ticket: later.ticket), "a background claim can be preempted")

        XCTAssertFalse(
            released.wasCancelled,
            "a claim that was released normally must not inherit a later claim's preemption"
        )
        XCTAssertTrue(later.wasCancelled, "while the preempted claim does report it")
    }

    /// The other direction: the flag is carried by the claim, so it survives
    /// any number of later cancellations — the ring's 64-entry limit used to
    /// be the point past which the answer degraded to the watermark.
    func testAPreemptedClaimStillKnowsAfterManyLaterPreemptions() throws {
        let ledger = TurnSlotLedger()
        let victim = try XCTUnwrap(ledger.tryClaim(priority: .background))
        XCTAssertTrue(ledger.cancelClaim(ifPriority: .background))
        for _ in 0..<100 {
            let claim = try XCTUnwrap(ledger.tryClaim(priority: .background))
            _ = ledger.cancelClaim(ticket: claim.ticket)
        }
        XCTAssertTrue(victim.wasCancelled, "the answer must not age out")
    }

    // MARK: - #5: the deadline and the work settling together

    /// Work that finishes on its own must be reported as finished even when
    /// the timer is dequeued first — the tie the old code resolved as
    /// `.timedOut`, failing work that had already completed.
    func testWorkSettlingAtTheDeadlineIsReportedAsCompleted() async {
        let raced = await HeadlessAgentLoop.firstResult(deadlineNanoseconds: 0) { 42 }
        switch raced {
        case .completed(let value):
            XCTAssertEqual(value, 42)
        case .timedOut:
            XCTFail("work that returned must not be reported as timed out")
        }
    }

    /// The other half, which the tie-break must not swallow: a deadline that
    /// really expires still reports `.timedOut`, and still carries whatever
    /// the cancelled work returned.
    func testAnExpiredDeadlineIsStillATimeoutAndKeepsThePartial() async {
        let raced = await HeadlessAgentLoop.firstResult(deadlineNanoseconds: 20_000_000) {
            try? await Task.sleep(nanoseconds: 30_000_000_000)
            return "partial"
        }
        switch raced {
        case .completed:
            XCTFail("work the deadline interrupted is a timeout")
        case .timedOut(let partial):
            XCTAssertEqual(partial, "partial", "the record the tool managed to write must not be dropped")
        }
    }

    /// A nil answer is an answer: `T` may be an Optional, so the race must not
    /// use "the value is nil" to mean "no value arrived".
    func testANilAnswerFromWorkIsStillAnAnswer() async {
        let raced = await HeadlessAgentLoop.firstResult(deadlineNanoseconds: 5_000_000_000) { () -> Int? in nil }
        switch raced {
        case .completed(let value):
            XCTAssertNil(value)
        case .timedOut:
            XCTFail("a nil answer is still an answer")
        }
    }

    // MARK: - #8: the notifier's withdraw from a nonisolated context

    /// `withdraw` runs from the store's IO queue and must not touch the
    /// operator's notification state under a test run. The guard is what makes
    /// the `assumeIsolated` body unreachable here, and this pins that it stays
    /// a no-op rather than reaching the notification daemon from a suite.
    func testWithdrawIsANoOpUnderATestRun() {
        SubagentNotifier.withdraw(runId: UUID())
    }
}
