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
                    let expected = ChatPromptFormatter.formatPrompt(
                        prompt: "hello",
                        cleanSystem: system,
                        thinkingEnabled: thinking,
                        modelSupportsThinking: supports,
                        modelConfig: family
                    )
                    let actual = ChatPromptFormatter.formatConversation(
                        turns: [.init(role: .user, text: "hello")],
                        cleanSystem: system,
                        thinkingEnabled: thinking,
                        modelSupportsThinking: supports,
                        modelConfig: family
                    )
                    XCTAssertEqual(actual, expected, "\(name) thinking=\(thinking) supports=\(supports)")
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
}
