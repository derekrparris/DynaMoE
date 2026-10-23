//
//  ChatPromptFormatterConversationTests.swift
//  DynaMoE
//
//  Pure-logic coverage for multi-turn prompt assembly. History must keep the
//  model family's native role boundaries so each turn's prompt extends the
//  previous turn's prompt+generation — otherwise every multi-turn request
//  re-prefills (and hybrid/GDN models, whose recurrent state only matches at
//  the exact pinned boundary, cannot reuse the pin at all).
//

import XCTest
@testable import DynaMoE

final class ChatPromptFormatterConversationTests: XCTestCase {

    private let system = "You are helpful."

    /// Builds a ModelConfig carrying just enough to select a prompt family.
    private func config(_ modelType: String) throws -> ModelConfig {
        try JSONDecoder().decode(ModelConfig.self, from: Data("{\"model_type\":\"\(modelType)\"}".utf8))
    }

    private func families() throws -> [(name: String, config: ModelConfig?)] {
        [
            ("default", nil),
            ("ling", try config("bailing")),
            ("spark", try config("spark")),
            ("gemma4", try config("gemma4")),
        ]
    }

    // MARK: - Single-turn equivalence

    func testSingleTurnConversationRendersExactlyLikeSingleTurnTemplate() throws {
        // One user turn must render byte-identically to the existing
        // single-turn template, so today's behavior is unchanged for the
        // common case (and the family markers can never drift apart).
        for (name, family) in try families() {
            for thinking in [true, false] {
                for supports in [true, false] {
                    for effort in [nil, .low, .high] as [ThinkingEffort?] {
                        let expected = ChatPromptFormatter.formatPrompt(
                            prompt: "hello",
                            cleanSystem: system,
                            thinkingEnabled: thinking,
                            modelSupportsThinking: supports,
                            modelConfig: family,
                            thinkingEffort: effort
                        )
                        let actual = ChatPromptFormatter.formatConversation(
                            turns: [.init(role: .user, text: "hello")],
                            cleanSystem: system,
                            thinkingEnabled: thinking,
                            modelSupportsThinking: supports,
                            modelConfig: family,
                            thinkingEffort: effort
                        )
                        XCTAssertEqual(
                            actual,
                            expected,
                            "\(name) thinking=\(thinking) supports=\(supports) effort=\(String(describing: effort))"
                        )
                    }
                }
            }
        }
    }

    // MARK: - Prefix preservation across turns

    func testMultiTurnPromptExtendsPreviousTurnPlusGeneration() throws {
        // Turn 1 (single user) then turn 2 resending the history with the
        // assistant's reply: the rendered prompt for turn 2 must start with
        // turn 1's full prompt followed by the generated text, which is what
        // makes the engine's prompt+generation pin reusable.
        let generation = "ANSWER"
        for (name, family) in try families() {
            let turn1 = ChatPromptFormatter.formatPrompt(
                prompt: "hello",
                cleanSystem: system,
                thinkingEnabled: false,
                modelSupportsThinking: false,
                modelConfig: family
            )
            let turn2 = ChatPromptFormatter.formatConversation(
                turns: [
                    .init(role: .user, text: "hello"),
                    .init(role: .assistant, text: generation),
                    .init(role: .user, text: "again"),
                ],
                cleanSystem: system,
                thinkingEnabled: false,
                modelSupportsThinking: false,
                modelConfig: family
            )
            XCTAssertTrue(
                turn2.hasPrefix(turn1 + generation),
                "\(name): turn 2 must extend turn 1's prompt+generation"
            )
        }
    }

    func testThinkingModeStillReproducesThePromptPrefix() throws {
        // With thinking on, the client only echoes the visible answer, so the
        // generated reasoning cannot be reproduced; the system + history
        // prefix (including the assistant-open markers) still must be.
        for (name, family) in try families() {
            let turn1 = ChatPromptFormatter.formatPrompt(
                prompt: "hello",
                cleanSystem: system,
                thinkingEnabled: true,
                modelSupportsThinking: true,
                modelConfig: family
            )
            let turn2 = ChatPromptFormatter.formatConversation(
                turns: [
                    .init(role: .user, text: "hello"),
                    .init(role: .assistant, text: "ANSWER"),
                    .init(role: .user, text: "again"),
                ],
                cleanSystem: system,
                thinkingEnabled: true,
                modelSupportsThinking: true,
                modelConfig: family
            )
            XCTAssertTrue(turn2.hasPrefix(turn1), "\(name): the prompt prefix must survive thinking mode")
        }
    }

    func testHistoryUsesNativeRoleBoundariesNotFlatLabels() throws {
        let prompt = ChatPromptFormatter.formatConversation(
            turns: [
                .init(role: .user, text: "hello"),
                .init(role: .assistant, text: "hi there"),
                .init(role: .user, text: "again"),
            ],
            cleanSystem: "",
            thinkingEnabled: false,
            modelSupportsThinking: false,
            modelConfig: nil
        )
        XCTAssertTrue(prompt.contains("<|im_start|>user\nhello<|im_end|>\n"), "user turn must use native markers")
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\nhi there<|im_end|>\n"), "assistant turn must use native markers")
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n"), "must end at the assistant-open prefix")
        XCTAssertFalse(prompt.contains("User:"), "no synthetic speaker labels")
        XCTAssertFalse(prompt.contains("Assistant:"), "no synthetic speaker labels")
    }

    func testEmptyConversationRendersNothing() {
        XCTAssertEqual(
            ChatPromptFormatter.formatConversation(
                turns: [],
                cleanSystem: system,
                thinkingEnabled: false,
                modelSupportsThinking: false,
                modelConfig: nil
            ),
            ""
        )
    }

    // MARK: - Completed-turn shape

    func testCompletedAssistantTurnClosesReasoningBlock() throws {
        // A client echoes only the visible answer; with thinking on it must not
        // be rendered as an unfinished thought inside an open reasoning block.
        let close = TurnTextSplitter.canonicalReasoningCloseTag
        for (name, family) in try families() where name != "gemma4" {
            let prompt = ChatPromptFormatter.formatConversation(
                turns: [
                    .init(role: .user, text: "hello"),
                    .init(role: .assistant, text: "ANSWER"),
                    .init(role: .user, text: "again"),
                ],
                cleanSystem: system,
                thinkingEnabled: true,
                modelSupportsThinking: true,
                modelConfig: family
            )
            XCTAssertTrue(prompt.contains(close + "ANSWER"), "\(name): a completed answer must sit outside the reasoning block")
        }
    }

    func testRecordedReasoningIsNotDoubleClosed() throws {
        // The server splices the recorded raw generation (reasoning + close +
        // answer); re-rendering must not add a second close marker, so the
        // pinned prompt+generation prefix stays reusable.
        let close = TurnTextSplitter.canonicalReasoningCloseTag
        let raw = "reasoning" + close + "ANSWER"
        let turn1 = ChatPromptFormatter.formatPrompt(
            prompt: "hello",
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: nil
        )
        let prompt = ChatPromptFormatter.formatConversation(
            turns: [
                .init(role: .user, text: "hello"),
                .init(role: .assistant, text: raw),
                .init(role: .user, text: "again"),
            ],
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: nil
        )
        XCTAssertEqual(
            prompt.components(separatedBy: close).count - 1,
            1,
            "the reasoning close marker must appear exactly once"
        )
        XCTAssertTrue(prompt.hasPrefix(turn1 + raw), "the recorded raw generation must still extend the pinned prefix")
    }

    // MARK: - Gemma reasoning channel

    func testGemmaCompletedTurnRewrapsEchoedReasoningInItsChannelBlock() throws {
        // Gemma's reasoning lives in a native `<|channel>` block. The adapters
        // fold an echoed reasoning into `reasoning + canonical close + answer`,
        // which must be re-wrapped here: replayed as bare text the turn neither
        // reproduces the pinned generation nor reads as a finished answer.
        let gemma = try config("gemma4")
        let open = ChatPromptFormatter.gemmaReasoningOpen
        let close = ChatPromptFormatter.gemmaReasoningClose
        let prompt = ChatPromptFormatter.formatConversation(
            turns: [
                .init(role: .user, text: "hello"),
                .init(role: .assistant, text: "why" + TurnTextSplitter.canonicalReasoningCloseTag + "ANSWER"),
                .init(role: .user, text: "again"),
            ],
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: gemma
        )
        XCTAssertTrue(
            prompt.hasPrefix(try gemmaTurnOne() + open + "why" + close + "ANSWER"),
            "echoed reasoning must be re-wrapped in Gemma's native channel block"
        )
        XCTAssertFalse(
            prompt.contains(TurnTextSplitter.canonicalReasoningCloseTag),
            "the canonical marker belongs to the other families"
        )
    }

    func testGemmaRecordedRawGenerationIsNotRewrapped() throws {
        // A server-spliced raw generation already carries the native markers;
        // adding a second channel block would break the pinned prefix.
        let gemma = try config("gemma4")
        let raw = ChatPromptFormatter.gemmaReasoningOpen + "why" + ChatPromptFormatter.gemmaReasoningClose + "ANSWER"
        let prompt = ChatPromptFormatter.formatConversation(
            turns: [
                .init(role: .user, text: "hello"),
                .init(role: .assistant, text: raw),
                .init(role: .user, text: "again"),
            ],
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: gemma
        )
        XCTAssertTrue(prompt.hasPrefix(try gemmaTurnOne() + raw), "the recorded raw generation must extend the pinned prefix")
    }

    func testGemmaCompletedAnswerWithoutReasoningKeepsAnEmptyChannel() throws {
        // Thinking was on but the client dropped the reasoning (and the server
        // had no record to splice): the channel stays explicitly closed and
        // empty, exactly as the non-thinking template emits it, rather than
        // leaving the answer inside an unfinished thought.
        let gemma = try config("gemma4")
        let prompt = ChatPromptFormatter.formatConversation(
            turns: [
                .init(role: .user, text: "hello"),
                .init(role: .assistant, text: "ANSWER"),
                .init(role: .user, text: "again"),
            ],
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: gemma
        )
        XCTAssertTrue(
            prompt.hasPrefix(
                try gemmaTurnOne()
                    + ChatPromptFormatter.gemmaReasoningOpen
                    + ChatPromptFormatter.gemmaReasoningClose
                    + "ANSWER"
            ),
            "a completed answer must sit outside an explicitly closed channel block"
        )
    }

    /// The single-turn template for a Gemma thinking conversation: the prompt
    /// every follow-up turn must extend.
    private func gemmaTurnOne() throws -> String {
        ChatPromptFormatter.formatPrompt(
            prompt: "hello",
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: try config("gemma4")
        )
    }

    // MARK: - Reasoning across a thinking toggle

    func testHistoricalReasoningStaysInItsBlockWhenThinkingIsTurnedOff() throws {
        // A client that kept the reasoning of an earlier thinking turn sends it
        // again after thinking has been disabled for the follow-up. That earlier
        // turn must still render with its reasoning block open, and Gemma's must
        // render in its native channel: placing the reasoning after an
        // already-closed empty block would put it outside the block it belongs to
        // and re-render the turn differently from the pinned prompt.
        let canonicalClose = TurnTextSplitter.canonicalReasoningCloseTag
        for (name, family) in try families() {
            let prompt = ChatPromptFormatter.formatConversation(
                turns: [
                    .init(role: .user, text: "hello"),
                    .init(role: .assistant, text: "why" + canonicalClose + "\n\nANSWER"),
                    .init(role: .user, text: "again"),
                ],
                cleanSystem: system,
                thinkingEnabled: false,
                modelSupportsThinking: true,
                modelConfig: family
            )
            let open = name == "gemma4"
                ? ChatPromptFormatter.gemmaReasoningOpen
                : TurnTextSplitter.canonicalReasoningOpenTag
            let close = name == "gemma4"
                ? ChatPromptFormatter.gemmaReasoningClose
                : canonicalClose
            let opened = try XCTUnwrap(prompt.range(of: open), "\(name): the reasoning block must be opened")
            let afterOpen = prompt[opened.upperBound...]
            let closed = try XCTUnwrap(afterOpen.range(of: close), "\(name): the open block must be closed again")
            let reasoning = try XCTUnwrap(afterOpen.range(of: "why"), "\(name): the echoed reasoning must survive")
            XCTAssertLessThan(
                reasoning.lowerBound,
                closed.lowerBound,
                "\(name): the reasoning must sit inside its block, not after a closed empty one"
            )
            if name == "gemma4" {
                XCTAssertFalse(
                    prompt.contains(canonicalClose),
                    "gemma4: the echoed reasoning must use the native channel markers"
                )
            }
        }
    }

    func testSparkAssistantTurnsCarryEndOfSentenceMarker() throws {
        let spark = try config("spark")
        let prompt = ChatPromptFormatter.formatConversation(
            turns: [
                .init(role: .user, text: "hello"),
                .init(role: .assistant, text: "ANSWER"),
                .init(role: .user, text: "again"),
            ],
            cleanSystem: system,
            thinkingEnabled: false,
            modelSupportsThinking: false,
            modelConfig: spark
        )
        XCTAssertTrue(
            prompt.contains("ANSWER" + ChatPromptFormatter.sparkEndOfSentence),
            "a completed Spark assistant turn must end with its native terminator"
        )
        XCTAssertFalse(
            prompt.hasSuffix(ChatPromptFormatter.sparkEndOfSentence),
            "the final generation prefix must stay open"
        )
        let turn1 = ChatPromptFormatter.formatPrompt(
            prompt: "hello",
            cleanSystem: system,
            thinkingEnabled: false,
            modelSupportsThinking: false,
            modelConfig: spark
        )
        XCTAssertTrue(prompt.hasPrefix(turn1), "the prompt prefix must survive the added terminator")
    }

    // MARK: - Spark X2.6 graded thinking levels (reasoning_effort)

    func testSparkEffortInstructionMatchesTheOfficialTemplate() {
        // Golden string, verbatim from Spark-X2.6-4B's chat_template.jinja
        // (reasoning_instructions). Pure ASCII with backticks; if a render
        // ever strips them, dump codepoints instead of retyping.
        let lead = "\n`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low` and `high`.\nNow the system is invoked with `thinking_effort="
        XCTAssertEqual(ChatPromptFormatter.sparkThinkingEffortInstructions(level: .high), lead + "high`.")
        XCTAssertEqual(ChatPromptFormatter.sparkThinkingEffortInstructions(level: .low), lead + "low`.")
    }

    func testSparkEffortLineRidesTheSystemBlockWhenThinkingIsOn() throws {
        let spark = try config("spark")
        for level in [ThinkingEffort.low, .high] {
            let prompt = ChatPromptFormatter.formatPrompt(
                prompt: "hello",
                cleanSystem: system,
                thinkingEnabled: true,
                modelSupportsThinking: true,
                modelConfig: spark,
                thinkingEffort: level
            )
            XCTAssertTrue(
                prompt.contains(
                    "you are a helpful assistant."
                        + ChatPromptFormatter.sparkThinkingEffortInstructions(level: level)
                        + "\n\n"
                        + system
                ),
                "\(level.rawValue): the effort line must sit in the default sentence, before the client system"
            )
            XCTAssertTrue(prompt.contains("`thinking_effort=\(level.rawValue)`"), "\(level.rawValue): the level must be spelled out")
        }
    }

    func testSparkEffortLineOmittedWhenUnsetOrThinkingOff() throws {
        let spark = try config("spark")
        let unset = ChatPromptFormatter.formatPrompt(
            prompt: "hello",
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: spark
        )
        XCTAssertFalse(unset.contains("thinking_effort"), "no chosen level means no effort line")

        let thinkingOff = ChatPromptFormatter.formatPrompt(
            prompt: "hello",
            cleanSystem: system,
            thinkingEnabled: false,
            modelSupportsThinking: true,
            modelConfig: spark,
            thinkingEffort: .high
        )
        XCTAssertFalse(thinkingOff.contains("thinking_effort"), "thinking off must never carry the effort line")
    }

    func testSparkEmptySystemWithEffortCarriesDefaultSentencePlusLine() throws {
        // With no custom system, the System block must carry the default
        // sentence followed by the effort instruction, matching what
        // Spark-X2.6's chat template renders for the same kwargs.
        let spark = try config("spark")
        let prompt = ChatPromptFormatter.formatPrompt(
            prompt: "hello",
            cleanSystem: "",
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: spark,
            thinkingEffort: .high
        )
        XCTAssertTrue(
            prompt.contains("you are a helpful assistant." + ChatPromptFormatter.sparkThinkingEffortInstructions(level: .high)),
            "the empty-system render must carry the default sentence plus the effort line"
        )
    }

    func testSparkEffortLinePrecedesTheToolsSection() throws {
        // The template keeps the instruction with the default sentence, before
        // any tools block. The in-app and server system texts append tools
        // differently, so both shapes must keep the line ahead of them.
        let spark = try config("spark")
        let effort = ChatPromptFormatter.sparkThinkingEffortInstructions(level: .high)
        for tools in ["\n\n# Tools\n<tools></tools>", "\n\nYou have access to the following functions."] {
            let prompt = ChatPromptFormatter.formatPrompt(
                prompt: "hello",
                cleanSystem: "System" + tools,
                thinkingEnabled: true,
                modelSupportsThinking: true,
                modelConfig: spark,
                thinkingEffort: .high
            )
            let effortIndex = try XCTUnwrap(prompt.range(of: effort))
            let toolsIndex = try XCTUnwrap(prompt.range(of: tools.trimmingCharacters(in: .whitespacesAndNewlines)))
            XCTAssertLessThan(effortIndex.lowerBound, toolsIndex.lowerBound, "the effort line must precede tools")
        }
    }

    func testNonSparkFamiliesIgnoreThinkingEffort() throws {
        // The level is only consumed by the Spark branch. Emitting it for
        // another family's template would make the menu lie.
        for (name, family) in try families() where name != "spark" {
            for thinking in [true, false] {
                let plain = ChatPromptFormatter.formatPrompt(
                    prompt: "hello",
                    cleanSystem: system,
                    thinkingEnabled: thinking,
                    modelSupportsThinking: true,
                    modelConfig: family
                )
                let withEffort = ChatPromptFormatter.formatPrompt(
                    prompt: "hello",
                    cleanSystem: system,
                    thinkingEnabled: thinking,
                    modelSupportsThinking: true,
                    modelConfig: family,
                    thinkingEffort: .high
                )
                XCTAssertEqual(withEffort, plain, "\(name) thinking=\(thinking)")
                XCTAssertFalse(withEffort.contains("thinking_effort"), "\(name) must not emit a Spark effort line")
            }
        }
    }

    func testSingleTurnConversationWithEffortRendersLikeSingleTurnTemplate() throws {
        // The single-turn/multi-turn equivalence invariant holds with an effort
        // in play, so a stateless follow-up still extends the pinned prefix.
        let spark = try config("spark")
        let expected = ChatPromptFormatter.formatPrompt(
            prompt: "hello",
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: spark,
            thinkingEffort: .low
        )
        let actual = ChatPromptFormatter.formatConversation(
            turns: [.init(role: .user, text: "hello")],
            cleanSystem: system,
            thinkingEnabled: true,
            modelSupportsThinking: true,
            modelConfig: spark,
            thinkingEffort: .low
        )
        XCTAssertEqual(actual, expected)
    }
}
