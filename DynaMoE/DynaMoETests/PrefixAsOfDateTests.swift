//
//  PrefixAsOfDateTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the pinned temporal context: the date rendered into
//  a turn's system prompt must stay byte-stable for a session while its prefix
//  pin lives. `# Current Date & Time` carries minute+second precision, so a
//  fresh Date() per turn splits the common token prefix a sentence in and
//  forces hybrid (GDN) sessions — which cannot rewind linear state — into a
//  full re-prefill on every agent step.
//

import XCTest
@testable import DynaMoE

final class PrefixAsOfDateTests: XCTestCase {

    private var enabledBefore: Bool = true

    override func setUp() {
        super.setUp()
        enabledBefore = PrefixCacheManager.shared.isEnabled
        PrefixCacheManager.shared.invalidate()
        PrefixCacheManager.shared.isEnabled = true
    }

    override func tearDown() {
        PrefixCacheManager.shared.invalidate()
        PrefixCacheManager.shared.isEnabled = enabledBefore
        super.tearDown()
    }

    func testDateIsStableAcrossTurnsOfTheSameSession() {
        let session = UUID()
        let first = PrefixCacheManager.shared.asOfDate(sessionId: session)
        let second = PrefixCacheManager.shared.asOfDate(sessionId: session)
        XCTAssertEqual(first, second, "a session must render the byte-identical temporal block every turn")
    }

    func testReusingTurnsDoNotReMintAfterRecording() {
        // The opencode flow: turn 1 mints, records prompt+generated as the pin;
        // turn 2+ must embed the SAME date so the tokenized prompt matches the
        // pin all the way to the end.
        let session = UUID()
        let minted = PrefixCacheManager.shared.asOfDate(sessionId: session)
        PrefixCacheManager.shared.recordTurn(promptTokenIds: [1, 2, 3], generatedTokenIds: [], sessionId: session)
        XCTAssertEqual(PrefixCacheManager.shared.asOfDate(sessionId: session), minted,
                       "a recorded pin must keep its date for subsequent reuse-turns")
    }

    func testDifferentSessionMintsFreshDate() {
        // Single-slot pin: a different conversation's question must not render
        // another session's pinned temporal context.
        let _ = PrefixCacheManager.shared.asOfDate(sessionId: UUID())
        Thread.sleep(forTimeInterval: 0.01)
        XCTAssertNotEqual(
            PrefixCacheManager.shared.asOfDate(sessionId: UUID()),
            PrefixCacheManager.shared.asOfDate(sessionId: UUID()),
            "each new session's first turn mints its own date"
        )
    }

    func testInvalidationResetsTheMarkedDate() {
        let session = UUID()
        let minted = PrefixCacheManager.shared.asOfDate(sessionId: session)
        Thread.sleep(forTimeInterval: 0.01)
        PrefixCacheManager.shared.invalidate(sessionId: session)
        XCTAssertNotEqual(PrefixCacheManager.shared.asOfDate(sessionId: session), minted,
                          "a cleared pin must not carry the old temporal context into the next prefill")
    }

    func testDisabledPinningAlwaysMintsFreshDates() {
        // With pinning off nothing is reusable, so there is no reason to serve
        // a growing-stale date; every ask renders the real current time.
        PrefixCacheManager.shared.isEnabled = false
        let stale = PrefixCacheManager.shared.asOfDate(sessionId: UUID())
        Thread.sleep(forTimeInterval: 0.01)
        XCTAssertNotEqual(PrefixCacheManager.shared.asOfDate(sessionId: UUID()), stale)
    }

    func testRejectedRequestFromAnotherSessionLeavesLiveDateIntact() {
        // A's turn mints its date and pins it. A request from B is then planned
        // (minting a date) but rejected before it runs: it must not take over
        // A's pinned temporal context, or A's next request loses its prefix.
        let a = UUID()
        let dateA = PrefixCacheManager.shared.asOfDate(sessionId: a)
        PrefixCacheManager.shared.recordTurn(promptTokenIds: [1, 2, 3], generatedTokenIds: [], sessionId: a)

        let b = UUID()
        let _ = PrefixCacheManager.shared.asOfDate(sessionId: b)
        XCTAssertEqual(
            PrefixCacheManager.shared.asOfDate(sessionId: a),
            dateA,
            "a rejected request from another session must not re-mint the live pin's date"
        )
    }

    func testAdmittedTurnKeepsTheDateItRendered() {
        // B is admitted and pins: it must keep the date it rendered so its own
        // next turn renders the byte-identical temporal block.
        let a = UUID()
        let _ = PrefixCacheManager.shared.asOfDate(sessionId: a)
        PrefixCacheManager.shared.recordTurn(promptTokenIds: [1], generatedTokenIds: [], sessionId: a)

        let b = UUID()
        let dateB = PrefixCacheManager.shared.asOfDate(sessionId: b)
        Thread.sleep(forTimeInterval: 0.01)
        PrefixCacheManager.shared.recordTurn(promptTokenIds: [9], generatedTokenIds: [], sessionId: b)
        XCTAssertEqual(
            PrefixCacheManager.shared.asOfDate(sessionId: b),
            dateB,
            "an admitted turn must keep the date it rendered"
        )
    }

    func testPrecisionChangeStillPinsTheDateThePromptRendered() {
        // A KV-precision change invalidates the pin from inside the turn, after
        // the prompt was already rendered with `rendered`. Pinning a
        // completion-time date instead would make the next turn render a
        // different timestamp, so the pin that was just written could never be
        // reused (a full re-prefill on a hybrid model).
        let session = UUID()
        let rendered = PrefixCacheManager.shared.asOfDate(sessionId: session)
        PrefixCacheManager.shared.invalidate()
        Thread.sleep(forTimeInterval: 0.01)
        PrefixCacheManager.shared.recordTurn(
            promptTokenIds: [1, 2, 3],
            generatedTokenIds: [],
            sessionId: session,
            asOfDate: rendered
        )
        XCTAssertEqual(
            PrefixCacheManager.shared.asOfDate(sessionId: session),
            rendered,
            "the pin must carry the date its prompt actually rendered"
        )
    }

    func testUntrackedSessionClaimsNoDateInsteadOfMintingOne() {
        // Recording a date no prompt ever rendered would make the next turn
        // diverge from the pin that was just written, so a session the manager
        // never handed a date to leaves the live pin's date alone.
        let live = UUID()
        let liveDate = PrefixCacheManager.shared.asOfDate(sessionId: live)
        PrefixCacheManager.shared.recordTurn(promptTokenIds: [1], generatedTokenIds: [], sessionId: live)

        let other = UUID()
        Thread.sleep(forTimeInterval: 0.01)
        PrefixCacheManager.shared.recordTurn(promptTokenIds: [2], generatedTokenIds: [], sessionId: other)
        XCTAssertEqual(
            PrefixCacheManager.shared.asOfDate(sessionId: live),
            liveDate,
            "another session must not take over the live pin's date"
        )
        XCTAssertNotEqual(
            PrefixCacheManager.shared.asOfDate(sessionId: other),
            liveDate,
            "the untracked session renders its own date"
        )
    }
}
