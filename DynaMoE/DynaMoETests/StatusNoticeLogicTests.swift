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
            StatusNoticePolicy.isRepeat(text: "nope", kind: .warning, current: nil)
        )
    }

    func testTheSameMessageIsARepeat() {
        let current = StatusNotice(id: 7, kind: .warning, text: "nope")
        XCTAssertTrue(
            StatusNoticePolicy.isRepeat(text: "nope", kind: .warning, current: current)
        )
    }

    func testAChangedMessageIsNotARepeat() {
        let current = StatusNotice(id: 7, kind: .warning, text: "nope")
        XCTAssertFalse(
            StatusNoticePolicy.isRepeat(text: "different", kind: .warning, current: current)
        )
        // Same text, different meaning: a warning that becomes progress (or the
        // reverse) has to replace the notice rather than pass for the old one.
        XCTAssertFalse(
            StatusNoticePolicy.isRepeat(text: "nope", kind: .progress, current: current)
        )
    }

    func testIdentityCarriesTheNoticeGeneration() {
        // The view animates on `id`, so two notices with the same text but
        // different generations must not compare equal.
        let first = StatusNotice(id: 1, kind: .warning, text: "nope")
        let second = StatusNotice(id: 2, kind: .warning, text: "nope")
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(first, StatusNotice(id: 1, kind: .warning, text: "nope"))
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
    }

    /// Two surfaces, one sentence: the layer shows it once, and the composer's
    /// copy is the one that survives because the composer is what it is about.
    func testTheSameSentenceIsShownOnce() {
        let window = StatusNotice(id: 1, kind: .warning, text: "This model cannot see images.")
        let composer = StatusNotice(id: 2, kind: .warning, text: "This model cannot see images.")
        let stacked = StatusNoticePolicy.stacked(window: window, composer: composer)
        XCTAssertEqual(stacked.map(\.id), [composer.id])

        // Different sentences are two facts, and order is furthest-first so the
        // window's copy reads above the composer's.
        let other = StatusNotice(id: 3, kind: .warning, text: "Engine load failed")
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
