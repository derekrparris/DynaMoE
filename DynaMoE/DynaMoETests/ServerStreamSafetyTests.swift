//
//  ServerStreamSafetyTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the append-only streaming writers' delta safety:
//  snapshots that no longer extend the delivered text must not re-send it
//  (clients can only append), and suffixes that could still grow into a
//  configured stop sequence must be withheld until proven safe.
//

import XCTest
@testable import DynaMoE

final class ServerStreamSafetyTests: XCTestCase {

    // MARK: - unsafeSuffixLength

    func testNoStopSequencesMeansNoHoldBack() {
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello EN", stopSequences: []), 0)
    }

    func testPartialStopPrefixIsHeldBack() {
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello EN", stopSequences: ["END"]), 2)
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello E", stopSequences: ["END"]), 1)
        // "N" sits mid-sequence, not at its head: a suffix must be a prefix
        // of the stop to be unsafe.
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello N", stopSequences: ["END"]), 0)
    }

    func testCompleteStopMatchIsHeldBack() {
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello END", stopSequences: ["END"]), 3)
    }

    func testHoldBackCapsAtStopLength() {
        // "DEND" ends with "END" (3) but no suffix longer than the stop
        // itself can be a prefix of it.
        XCTAssertEqual(ServerController.unsafeSuffixLength("DEND", stopSequences: ["END"]), 3)
    }

    func testUnrelatedSuffixIsSafe() {
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello world", stopSequences: ["END"]), 0)
        // Matching is case-sensitive, mirroring the engine's range(of:) check.
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello end", stopSequences: ["END"]), 0)
    }

    func testLongestStopWins() {
        // "AB" is a prefix of "ABX" (2 chars) and "B" a prefix of "B" (1).
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello AB", stopSequences: ["END", "ABX", "B"]), 2)
        XCTAssertEqual(ServerController.unsafeSuffixLength("hello B", stopSequences: ["END", "ABX", "B"]), 1)
    }

    // MARK: - safeStreamDelta

    func testEmitsOnlyTheAppendDelta() {
        let delta = ServerController.safeStreamDelta(accumulated: "hello world", sent: "hello ", stopSequences: [])
        XCTAssertEqual(delta, "world")
    }

    func testDivergentSnapshotEmitsNothing() {
        // The engine repaired its text so it no longer extends what was
        // already delivered; re-sending would duplicate the delivered prefix.
        XCTAssertEqual(ServerController.safeStreamDelta(accumulated: "hallo world", sent: "hello wor", stopSequences: []), "")
    }

    func testWithholdsPartialStopSuffix() {
        let delta = ServerController.safeStreamDelta(accumulated: "hello E", sent: "hello", stopSequences: ["END"])
        XCTAssertEqual(delta, " ", "the unsafe 'E' must be withheld")
    }

    func testWithholdsCompleteStopMatch() {
        // The engine truncates AT the match start, so text before "END" (the
        // space) is final and safe; only the match itself is withheld.
        let delta = ServerController.safeStreamDelta(accumulated: "hello END", sent: "hello", stopSequences: ["END"])
        XCTAssertEqual(delta, " ")
    }

    func testTerminalFlushIgnoresHoldBack() {
        // The engine trimmed matches from the final text; everything beyond
        // the delivered prefix is final, including a lookalike suffix.
        let delta = ServerController.safeStreamDelta(accumulated: "hello world END", sent: "hello", stopSequences: [])
        XCTAssertEqual(delta, " world END")
    }

    func testHoldBackSpanningSentBoundaryStillGuardsDelta() {
        // "EN" was already delivered; the fresh "D" completes the match, so
        // the delta is withheld in full even though the match began earlier.
        let delta = ServerController.safeStreamDelta(accumulated: "hello END", sent: "hello EN", stopSequences: ["END"])
        XCTAssertEqual(delta, "")
    }

    func testEmittedProgressTracksDeliveredTextOnly() {
        var sent = ""
        var emitted = ""
        let stops = ["END"]

        let first = ServerController.safeStreamDelta(accumulated: "hello E", sent: sent, stopSequences: stops)
        XCTAssertEqual(first, "hello ")
        sent += first
        emitted += first

        // The engine's next snapshot completes the match and trims it.
        let second = ServerController.safeStreamDelta(accumulated: "hello", sent: sent, stopSequences: stops)
        XCTAssertEqual(second, "", "nothing more may be emitted once the engine trims the match")
        let terminal = ServerController.safeStreamDelta(accumulated: "hello", sent: sent, stopSequences: [])
        XCTAssertEqual(terminal, "", "the client's assembled text must equal the trimmed engine text")
        emitted += terminal
        XCTAssertEqual(emitted, "hello ")
    }
}
