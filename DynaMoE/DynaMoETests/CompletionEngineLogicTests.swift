//
//  CompletionEngineLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the headless completion engine's movable parts —
//  no Metal, no model weights, no hardware. These run anywhere.
//
//  Note on tag-bearing literals: model-family control tags (some multibyte)
//  are assembled from codepoints, never retyped from a rendered view
//  (AGENTS.md "Reading files safely").
//

import XCTest
@testable import DynaMoE

final class CompletionEngineLogicTests: XCTestCase {

    // MARK: - ChatPromptFormatter

    func testFormatterDefaultQwenPathEmitsSystemUserAssistant() {
        let prompt = ChatPromptFormatter.formatPrompt(
            prompt: "Hello",
            cleanSystem: "You are helpful.",
            thinkingEnabled: false,
            modelSupportsThinking: false,
            modelConfig: nil
        )
        XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\nYou are helpful.<|im_end|>\n"), "system block missing: \(String(prompt.prefix(60)))")
        XCTAssertTrue(prompt.contains("<|im_start|>user\nHello<|im_end|>"))
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    func testFormatterThinkingCapableButDisabledClosesReasoningImmediately() {
        // modelSupportsThinking == true with thinking off must append an
        // empty open+close pair so the answer streams as response text, not reasoning.
        let prompt = ChatPromptFormatter.formatPrompt(
            prompt: "Hello",
            cleanSystem: "",
            thinkingEnabled: false,
            modelSupportsThinking: true,
            modelConfig: nil
        )
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n" + Self.thinkOpenTag() + "\n\n" + Self.thinkCloseTag() + "\n\n"), "expected empty open/close pair after assistant header, got: \(prompt.suffix(48))")
    }

    func testFormatterPreservesCustomContinuationVerbatim() {
        // Agent tool-response turns arrive pre-templated: the formatter must
        // detect the markers and pass the text through untouched.
        let custom = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant"
        let out = ChatPromptFormatter.formatPrompt(
            prompt: custom,
            cleanSystem: "sys",
            thinkingEnabled: false,
            modelSupportsThinking: false,
            modelConfig: nil
        )
        XCTAssertEqual(out, custom)
    }

    // MARK: - TurnTextSplitter

    func testSplittingClosedThinkingBlock() {
        let rawFull = Self.thinkOpenTag() + "step by step" + Self.thinkCloseTag() + "\n\nAnswer: 42"
        let split = TurnTextSplitter.splitThinkingAndResponse(raw: rawFull, promptRequestsThinking: false)
        XCTAssertTrue(split.thinkClose, "close tag should mark the boundary")
        XCTAssertFalse(split.thinkOpen)
        XCTAssertEqual(split.think, "step by step")
        XCTAssertEqual(split.resp, "Answer: 42")
    }

    func testSplittingOpenThinkingBlockStreamsThinkOnly() {
        let raw = Self.thinkOpenTag() + "partial reasoning so far"
        let split = TurnTextSplitter.splitThinkingAndResponse(raw: raw, promptRequestsThinking: false)
        XCTAssertTrue(split.thinkOpen)
        XCTAssertFalse(split.thinkClose)
        XCTAssertEqual(split.think, "partial reasoning so far")
        XCTAssertEqual(split.resp, "")
    }

    func testUntaggedDirectResponseClassifiesAsResp() {
        let split = TurnTextSplitter.splitThinkingAndResponse(raw: "just an answer", promptRequestsThinking: false)
        XCTAssertFalse(split.thinkOpen)
        XCTAssertFalse(split.thinkClose)
        XCTAssertEqual(split.think, "")
        XCTAssertEqual(split.resp, "just an answer")
    }

    func testPromptRequestsThinkingRejectsPlainPrompts() {
        XCTAssertFalse(TurnTextSplitter.promptRequestsThinking("<|im_start|>assistant"))
        XCTAssertFalse(TurnTextSplitter.promptRequestsThinking("plain text"))
    }

    // MARK: - Degenerate-cycle guard

    func testDetectDegenerateCycleCatchesRepeatRunWithLetterUnit() {
        // 30 identical repetitions of a 2-token unit carrying a letter: a
        // genuine degeneration.
        var ids: [UInt32] = []
        let unit: [UInt32] = [65, 66] // "A", "B"
        for _ in 0..<30 { ids.append(contentsOf: unit) }
        let period = TurnTextSplitter.detectDegenerateCycle(tokenIds: ids) { tokenIds in
            String(tokenIds.map { Character(UnicodeScalar($0)!) })
        }
        XCTAssertEqual(period, 2)
    }

    func testDetectDegenerateCycleIgnoresFormattingRuns() {
        // A run of newlines is legitimate formatting, not a loop: the guard
        // requires a letter (or digit) in the cycle's decoded unit.
        let ids = [UInt32](repeating: 10, count: 200)
        let period = TurnTextSplitter.detectDegenerateCycle(tokenIds: ids) { tokenIds in
            String(tokenIds.map { Character(UnicodeScalar($0)!) })
        }
        XCTAssertNil(period)
    }

    func testDetectDegenerateCycleIgnoresShortSequences() {
        let ids: [UInt32] = [72, 73]
        XCTAssertNil(TurnTextSplitter.detectDegenerateCycle(tokenIds: ids) { _ in "HI" })
    }

    // MARK: - Odometer seeding rule (turn-start, mirrored from old prefill seam)

    func testCumulativeTokenBaseCountsOnlyDeltaForMeteredConversation() {
        XCTAssertEqual(ChatSession.cumulativeTokenBase(previousTotal: 120, promptTokenCount: 100, prefixTokensReused: 60), 160)
    }

    func testCumulativeTokenBaseSeedsFullPromptForUnmeteredConversation() {
        XCTAssertEqual(ChatSession.cumulativeTokenBase(previousTotal: nil, promptTokenCount: 100, prefixTokensReused: 60), 100)
    }

    func testCumulativeTokenBaseToleratesPrefixLongerThanPrompt() {
        XCTAssertEqual(ChatSession.cumulativeTokenBase(previousTotal: 50, promptTokenCount: 100, prefixTokensReused: 140), 50)
    }

    // MARK: - Tag assembly (codepoint-exact)

    /// openTags[0] from TurnTextSplitter, assembled from codepoints:
    /// [0x3c, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e]
    private static func thinkOpenTag() -> String {
        [0x3c, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e].reduce(into: "") { acc, cp in
            acc.unicodeScalars.append(Unicode.Scalar(cp)!)
        }
    }

    /// closeTags[0] from TurnTextSplitter, assembled from codepoints:
    /// [0x3c, 0x2f, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e]
    private static func thinkCloseTag() -> String {
        [0x3c, 0x2f, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e].reduce(into: "") { acc, cp in
            acc.unicodeScalars.append(Unicode.Scalar(cp)!)
        }
    }
}
