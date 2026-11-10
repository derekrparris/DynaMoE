//
//  StatusNoticeLogicTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for the app's transient notices, which the composer and
//  the chat window both draw as overlays rather than rows. The rules worth
//  pinning are the ones a layout cannot express: which kinds clear themselves,
//  when a repeated message is allowed to re-run its fade, and which severity a
//  message authored with a leading glyph is read as.
//

import XCTest
@testable import DynaMoE

final class StatusNoticePolicyTests: XCTestCase {

    /// Builds a notice the way a surface does: a serial plus the surface that
    /// minted it.
    private func notice(
        _ serial: UInt64,
        from source: StatusNotice.Source = .window,
        kind: StatusNotice.Kind = .warning,
        text: String
    ) -> StatusNotice {
        StatusNotice(
            id: StatusNotice.Identity(source: source, serial: serial),
            kind: kind,
            text: text
        )
    }

    func testOnlyOutcomesClearThemselves() {
        XCTAssertTrue(StatusNotice.Kind.warning.autoDismisses)
        XCTAssertTrue(StatusNotice.Kind.info.autoDismisses)
        XCTAssertTrue(StatusNotice.Kind.success.autoDismisses)
        // Progress describes work still running: a timer would clear the message
        // while the thing it belongs to is still happening.
        XCTAssertFalse(StatusNotice.Kind.progress.autoDismisses)
    }

    func testNothingIsARepeatOfNothing() {
        XCTAssertFalse(
            StatusNoticePolicy.isRepeat(text: "nope", kind: .warning, source: .composer, current: nil)
        )
    }

    func testTheSameMessageIsARepeat() {
        let current = notice(7, from: .composer, text: "nope")
        XCTAssertTrue(
            StatusNoticePolicy.isRepeat(text: "nope", kind: .warning, source: .composer, current: current)
        )
    }

    func testAChangedMessageIsNotARepeat() {
        let current = notice(7, text: "nope")
        XCTAssertFalse(
            StatusNoticePolicy.isRepeat(text: "different", kind: .warning, source: .window, current: current)
        )
        // Same text, different meaning: a warning that becomes progress (or the
        // reverse) has to replace the notice rather than pass for the old one.
        XCTAssertFalse(
            StatusNoticePolicy.isRepeat(text: "nope", kind: .progress, source: .window, current: current)
        )
        // Same text on the other surface is a different pill with its own
        // identity, so it is not a repeat of this one.
        XCTAssertFalse(
            StatusNoticePolicy.isRepeat(text: "nope", kind: .warning, source: .composer, current: current)
        )
    }

    func testIdentityCarriesTheNoticeGeneration() {
        // The view animates on `id`, so two notices with the same text but
        // different generations must not compare equal.
        let first = notice(1, text: "nope")
        let second = notice(2, text: "nope")
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(first, notice(1, text: "nope"))
    }

    /// The layer renders both surfaces in ONE `ForEach` and each mints its serials
    /// from zero, so the first notice from each is serial 1. The surface is part
    /// of the identity for exactly that reason: without it SwiftUI sees one row
    /// and drops a pill or animates it onto the other's slot.
    func testTheTwoSurfacesMintDistinctIdentities() {
        let window = notice(1, from: .window, text: "Engine load failed")
        let composer = notice(1, from: .composer, text: "This model cannot see images.")
        XCTAssertNotEqual(window.id, composer.id)

        let stacked = StatusNoticePolicy.stacked(window: window, composer: composer)
        XCTAssertEqual(stacked.count, 2)
        XCTAssertEqual(Set(stacked.map(\.id)).count, 2)

        // The source does not widen the dedupe: one sentence is still one pill,
        // whichever surface raised it.
        XCTAssertEqual(
            StatusNoticePolicy.stacked(
                window: notice(1, from: .window, text: "Engine load failed"),
                composer: notice(1, from: .composer, text: "Engine load failed")
            ).count,
            1
        )
    }

    func testPreparingTextIsProgressCopy() {
        XCTAssertFalse(StatusNoticePolicy.preparingText.isEmpty)
        XCTAssertTrue(StatusNoticePolicy.preparingText.hasSuffix("…"))
    }

    /// Severity is read off the leading glyph every existing writer already uses,
    /// so the tint cannot disagree with the message.
    func testSeverityComesFromTheLeadingGlyph() {
        XCTAssertEqual(StatusNoticePolicy.kind(for: "⚠️ Stop the running turn"), .warning)
        XCTAssertEqual(StatusNoticePolicy.kind(for: "❌ Engine load failed"), .warning)
        // A stop the user asked for is neutral, not a fault.
        XCTAssertEqual(StatusNoticePolicy.kind(for: "⏹ Generation stopped by user."), .info)
        XCTAssertEqual(StatusNoticePolicy.kind(for: "✅ Task complete"), .success)
        XCTAssertEqual(StatusNoticePolicy.kind(for: "✨ Generated 12 tokens"), .success)
        // A tool step is an event, not a wait: it must clear itself.
        XCTAssertEqual(StatusNoticePolicy.kind(for: "⚙️ Executing 2 tool call(s)..."), .info)
        XCTAssertTrue(StatusNoticePolicy.kind(for: "⚙️ Executing 2 tool call(s)...").autoDismisses)
        XCTAssertEqual(StatusNoticePolicy.kind(for: "☁️ Provider active."), .info)
        XCTAssertEqual(StatusNoticePolicy.kind(for: "plain text"), .info)
    }

    /// The glyph in the message and the `kind:` a call site passes are two halves
    /// of one decision. Nothing stops an author from tinting a warning green, so
    /// the disagreement has to be detectable: this is what the call sites assert
    /// on, and it is the only thing that notices a message whose leading glyph was
    /// reworded into another severity.
    func testAGlyphThatDisagreesWithTheCallSiteIsDetected() {
        XCTAssertTrue(
            StatusNoticePolicy.statesContradictorySeverity(text: "⚠️ Cannot send", kind: .success)
        )
        XCTAssertTrue(
            StatusNoticePolicy.statesContradictorySeverity(text: "✅ Saved", kind: .warning)
        )
        XCTAssertFalse(
            StatusNoticePolicy.statesContradictorySeverity(text: "⚠️ Cannot send", kind: .warning)
        )
        // A message with no glyph states nothing, so a call site is free to tint
        // it: that is the only way to mark copy written without one.
        XCTAssertFalse(
            StatusNoticePolicy.statesContradictorySeverity(text: "Cannot send", kind: .warning)
        )
        // `.progress` beside a glyph is a contradiction too: a glyph announces an
        // event, and a progress pill is waiting on one.
        XCTAssertTrue(
            StatusNoticePolicy.statesContradictorySeverity(text: "❌ Failed", kind: .progress)
        )
        // The stop glyph is table-listed so a call site cannot tint the user's own
        // Stop as a fault, which is how the turn-failure path used to report it.
        XCTAssertTrue(
            StatusNoticePolicy.statesContradictorySeverity(
                text: "⏹ Generation stopped by user.",
                kind: .warning
            )
        )
        XCTAssertFalse(
            StatusNoticePolicy.statesContradictorySeverity(
                text: "⏹ Generation stopped by user.",
                kind: .info
            )
        )
    }

    /// Two surfaces, one sentence: the layer shows it once, and the composer's
    /// copy is the one that survives because the composer is what it is about.
    func testTheSameSentenceIsShownOnce() {
        let window = notice(1, text: "This model cannot see images.")
        let composer = notice(2, from: .composer, text: "This model cannot see images.")
        let stacked = StatusNoticePolicy.stacked(window: window, composer: composer)
        XCTAssertEqual(stacked.map(\.id), [composer.id])

        // Different sentences are two facts, and order is furthest-first so the
        // window's copy reads above the composer's.
        let other = notice(3, text: "Engine load failed")
        XCTAssertEqual(StatusNoticePolicy.stacked(window: other, composer: composer).map(\.id), [other.id, composer.id])
        XCTAssertEqual(StatusNoticePolicy.stacked(window: other, composer: nil).map(\.id), [other.id])
        XCTAssertEqual(StatusNoticePolicy.stacked(window: nil, composer: composer).map(\.id), [composer.id])
        XCTAssertTrue(StatusNoticePolicy.stacked(window: nil, composer: nil).isEmpty)
    }

    /// The two writers that describe work still running pass `.progress` with
    /// copy from this policy, so the copy and the rule that reads it are pinned
    /// together: neither message may lead with a severity glyph, because a glyph
    /// announces an event while a progress pill is waiting on one.
    func testProgressCopyCarriesNoSeverityGlyph() {
        let copy = [
            StatusNoticePolicy.compactionText(messageCount: 3),
            StatusNoticePolicy.appleFoundationModelGeneratingText
        ]
        for text in copy {
            XCTAssertFalse(
                StatusNoticePolicy.statesContradictorySeverity(text: text, kind: .progress),
                text
            )
        }
        // The count is the only part of the compaction line that moves, so the
        // sentence stays one notice identity across a multi-pass compaction.
        XCTAssertEqual(
            StatusNoticePolicy.compactionText(messageCount: 3),
            StatusNoticePolicy.compactionText(messageCount: 3)
        )
        XCTAssertNotEqual(
            StatusNoticePolicy.compactionText(messageCount: 3),
            StatusNoticePolicy.compactionText(messageCount: 4)
        )
    }
}
