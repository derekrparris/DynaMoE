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

    func testTheOdometerIsThePromptSizeNotARunningSum() {
        // One meaning for every backend: the conversation's context size. A
        // previous total is not an input, so a turn cannot count the same
        // history twice, and a chat that moved here from a cloud model (whose
        // stored figure is also a context size) does not jump by a whole
        // conversation on the switch.
        //
        // Asserted through the seed a turn actually calls, with a large stored
        // total in place: an additive regression (the old bug) fails here,
        // where comparing the identity function with itself could not.
        var session = ChatSession(title: "seeded")
        session.cumulativeTokenCount = 120_000
        session.seedContextTotal(promptTokenCount: 1_200)
        XCTAssertEqual(session.cumulativeTokenCount, 1_200)

        session.cumulativeTokenCount = 120_000
        session.seedContextTotal(promptTokenCount: 0)
        XCTAssertEqual(session.cumulativeTokenCount, 0, "an empty prompt seeds zero, not the old total")
    }

    func testTheOdometerBaseIsNeverNegative() {
        XCTAssertEqual(ChatSession.contextTokenBase(promptTokenCount: -5), 0)
    }

    func testTheOdometerAddsGeneratedTokensOnTopOfThePrompt() {
        let seeded = ChatSession.contextTokenBase(promptTokenCount: 1200)
        XCTAssertEqual(seeded + 300, 1500)
    }

    // MARK: - Sampler penalties (frequency_penalty / logit_bias)

    /// A logits vector with two candidates and a floor everywhere else, so a greedy
    /// draw returns the winner of the ARITHMETIC rather than of a sample.
    private static let samplerVocabSize = 2_000

    private func withSamplerLogits(
        _ body: (UnsafeMutablePointer<Float>) throws -> Void
    ) rethrows {
        let logits = UnsafeMutablePointer<Float>.allocate(capacity: Self.samplerVocabSize)
        logits.initialize(repeating: -20, count: Self.samplerVocabSize)
        logits[1000] = 20.0
        logits[1005] = 19.5
        defer {
            logits.deinitialize(count: Self.samplerVocabSize)
            logits.deallocate()
        }
        try body(logits)
    }

    private func greedy(
        _ logits: UnsafeMutablePointer<Float>,
        context: [UInt32],
        repetition: Float = 1.0,
        presence: Float = 0.0,
        frequency: Float = 0.0,
        bias: [Int32: Float] = [:]
    ) -> UInt32 {
        InferenceEngine.sampleNextToken(
            logits: logits,
            vocabSize: Self.samplerVocabSize,
            contextTokens: context,
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 20,
            repetitionPenalty: repetition,
            presencePenalty: presence,
            frequencyPenalty: frequency,
            logitBias: bias
        )
    }

    /// `frequency_penalty` is charged per OCCURRENCE, `presence_penalty` once per
    /// distinct token. The same context and the same magnitude must therefore pick
    /// opposite candidates here, which is the one thing that separates the two knobs.
    func testFrequencyPenaltyIsChargedPerOccurrenceNotPerPresence() {
        withSamplerLogits { logits in
            let repeated: [UInt32] = [1000, 1000, 1000]

            // 20.0 - 0.2 = 19.8 against 19.5: presence cannot demote it.
            XCTAssertEqual(greedy(logits, context: repeated, presence: 0.2), 1000)
            // 20.0 - 0.2 * 3 = 19.4 against 19.5: three occurrences can.
            XCTAssertEqual(greedy(logits, context: repeated, frequency: 0.2), 1005)
            XCTAssertEqual(logits[1000], 20.0, "the penalty must be restored by the defer")
            XCTAssertEqual(logits[1005], 19.5)
        }
    }

    /// A negative frequency penalty rewards repetition, as upstream defines it.
    func testANegativeFrequencyPenaltyRewardsRepetition() {
        withSamplerLogits { logits in
            logits[1000] = 19.0
            logits[1005] = 19.2
            let repeated: [UInt32] = [1000, 1000]
            XCTAssertEqual(greedy(logits, context: repeated), 1005, "unpenalised, 19.2 wins")
            // 19.0 + 0.2 * 2 = 19.4, which overtakes 19.2.
            XCTAssertEqual(greedy(logits, context: repeated, frequency: -0.2), 1000)
            XCTAssertEqual(logits[1000], 19.0, "a negative penalty must be restored too")
        }
    }

    /// `logit_bias` is additive on a named id whether or not the context holds it,
    /// which is how a client bans a token it has not seen yet.
    func testLogitBiasSuppressesAndPromotesTokensOutsideTheContext() {
        withSamplerLogits { logits in
            let unrelated: [UInt32] = [7, 8]
            XCTAssertEqual(greedy(logits, context: unrelated), 1000)
            // 20.0 - 1.0 = 19.0 against 19.5.
            XCTAssertEqual(greedy(logits, context: unrelated, bias: [1000: -1.0]), 1005)
            // 19.5 + 1.0 = 20.5 against 20.0.
            XCTAssertEqual(greedy(logits, context: unrelated, bias: [1005: 1.0]), 1005)
            XCTAssertEqual(logits[1000], 20.0)
            XCTAssertEqual(logits[1005], 19.5, "a biased logit must be restored exactly")
        }
    }

    /// The bias composes with the penalties on the same id, and the restore is not
    /// applied twice (a double restore would put the un-biased value back first and
    /// then "restore" the biased one, leaving the vector changed).
    func testLogitBiasComposesWithThePenaltiesOnTheSameToken() {
        withSamplerLogits { logits in
            // 20.0 - 0.2 (presence) - 1.0 (bias) = 18.8 against 19.5.
            let chosen = greedy(logits, context: [1000], presence: 0.2, bias: [1000: -1.0])
            XCTAssertEqual(chosen, 1005)
            XCTAssertEqual(logits[1000], 20.0, "penalty and bias must both be undone")
        }
    }

    /// An id the loaded checkpoint cannot emit has no candidate to bias, so it is
    /// skipped rather than treated as an error.
    func testLogitBiasOnAnIdOutsideTheVocabularyChangesNothing() {
        withSamplerLogits { logits in
            XCTAssertEqual(greedy(logits, context: [], bias: [999_999: -1_000]), 1000)
            XCTAssertEqual(greedy(logits, context: [], bias: [-5: 1_000]), 1000)
        }
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
