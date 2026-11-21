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
    /// the deadline EXPIRED first and had already cancelled it -- the tie the
    /// old code resolved as `.timedOut`, failing work that had already
    /// completed.
    ///
    /// The race itself cannot be forced from outside: once the deadline arm
    /// wins, `cancelAll()` makes the work observe the cancellation, so any
    /// real race ends in `.timedOut` and a case built on one is a coin flip
    /// (the original zero-deadline version reproduced its failure once in
    /// 20,000). What decides the tie is the CLASSIFICATION, so that is what is
    /// pinned here, and the end-to-end orderings are covered by the two cases
    /// below.
    func testWorkSettlingAtTheDeadlineIsReportedAsCompleted() {
        // `RacedResult` is not Equatable (its payload is generic), so each
        // expectation is spelled as a pattern.
        func isCompleted(_ raced: HeadlessAgentLoop.RacedResult<Int>, _ expected: Int) -> Bool {
            if case .completed(let value) = raced { return value == expected }
            return false
        }
        func isTimedOut(_ raced: HeadlessAgentLoop.RacedResult<Int>, _ expected: Int?) -> Bool {
            if case .timedOut(let value) = raced { return value == expected }
            return false
        }
        XCTAssertTrue(
            isCompleted(
                HeadlessAgentLoop.racedResult(expired: true, workSawCancellation: false, workValue: 42),
                42
            ),
            "an answer that never observed the cancellation is a finished result"
        )
        XCTAssertTrue(
            isTimedOut(
                HeadlessAgentLoop.racedResult(expired: true, workSawCancellation: true, workValue: 42),
                42
            ),
            "an answer the deadline interrupted is the deadline's casualty, and keeps its partial"
        )
        XCTAssertTrue(isCompleted(
            HeadlessAgentLoop.racedResult(expired: false, workSawCancellation: false, workValue: 7),
            7
        ))
        // A nil answer is an answer: `T` may be an Optional, and the race must
        // not read "the value is nil" as "no value arrived".
        let nilAnswer = HeadlessAgentLoop.racedResult(
            expired: false,
            workSawCancellation: false,
            workValue: Optional<Int>.none
        )
        guard case .completed(let answered) = nilAnswer else {
            return XCTFail("a nil answer must still be .completed")
        }
        XCTAssertNil(answered)
        // `T` is `Int?` here, so the payload is a nil answer rather than an
        // absent one: the race reports the timeout but still carries it.
        let interruptedNil: HeadlessAgentLoop.RacedResult<Int?> = HeadlessAgentLoop.racedResult(
            expired: true,
            workSawCancellation: true,
            workValue: Optional<Int>.none
        )
        guard case .timedOut(let carried) = interruptedNil else {
            return XCTFail("the deadline interrupted it, so it is a timeout")
        }
        // `carried` is `Int??`: the point is that the inner answer ARRIVED and
        // is nil, so the outer level must be `.some`, not `.none`.
        guard let answer = carried else {
            return XCTFail("the nil answer it produced must be kept, not dropped as an absent one")
        }
        XCTAssertNil(answer)
    }

    /// The other half, which the tie-break must not swallow: a deadline that
    /// really expires still reports `.timedOut`, and still carries whatever
    /// the cancelled work returned.
    func testAnExpiredDeadlineIsStillATimeoutAndKeepsThePartial() async {
        let raced = await HeadlessAgentLoop.firstResult(deadlineNanoseconds: 20_000_000) {
            // Its own, shorter than the deadline's sleep: the work is the one
            // producing "partial" AFTER the deadline expired, and it must not
            // hold the suite for the deadline's full span.
            try? await Task.sleep(nanoseconds: 60_000_000)
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
