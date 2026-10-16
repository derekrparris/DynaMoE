//
//  ReasoningRoundTripTests.swift
//  DynaMoE
//
//  Pure-logic coverage for keeping the model's reasoning alive across turns:
//  signing outbound Anthropic thinking blocks so SDK clients echo them, folding
//  reasoning a client does echo back into the assistant turn, and deciding when
//  the server may substitute its own record of the last generated turn.
//
//  Note on tag-bearing literals: reasoning markers come from
//  `TurnTextSplitter`, never retyped (AGENTS.md).
//

import XCTest
@testable import DynaMoE

final class ReasoningRoundTripTests: XCTestCase {

    private let reasoningClose = TurnTextSplitter.canonicalReasoningCloseTag

    private func config(_ modelType: String) throws -> ModelConfig {
        try JSONDecoder().decode(ModelConfig.self, from: Data("{\"model_type\":\"\(modelType)\"}".utf8))
    }

    // MARK: - Signed thinking blocks

    func testThinkingSignatureIsStableAndContentDependent() {
        let first = AnthropicAdapter.thinkingSignature(for: "step by step")
        XCTAssertFalse(first.isEmpty)
        XCTAssertEqual(first, AnthropicAdapter.thinkingSignature(for: "step by step"), "must be stable for the same text")
        XCTAssertNotEqual(first, AnthropicAdapter.thinkingSignature(for: "step by step "), "must depend on the text")
    }

    func testThinkingContentBlockEncodesSignature() throws {
        let data = try JSONEncoder().encode(AnthropicContentBlock.thinking("weigh the options"))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["type"] as? String, "thinking")
        XCTAssertEqual(json["thinking"] as? String, "weigh the options")
        XCTAssertEqual(
            json["signature"] as? String,
            AnthropicAdapter.thinkingSignature(for: "weigh the options"),
            "unsigned thinking blocks are dropped by SDK clients"
        )
    }

    func testSignatureDeltaEncodesForStreaming() throws {
        let data = try JSONEncoder().encode(AnthropicStreamDelta.signature("sig-123"))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["type"] as? String, "signature_delta")
        XCTAssertEqual(json["signature"] as? String, "sig-123")
    }

    // MARK: - Inbound reasoning is folded back into the assistant turn

    func testOpenAIEchoedReasoningFoldsIntoAssistantTurn() throws {
        let body = """
        {"messages":[
            {"role":"user","content":"hello"},
            {"role":"assistant","content":"the answer","reasoning_content":"let me think"},
            {"role":"user","content":"again"}
        ]}
        """
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.utf8))
        let (_, turns) = try OpenAIAdapter.splitMessages(request.messages)
        XCTAssertEqual(turns.map(\.role), [.user, .assistant, .user])
        XCTAssertTrue(
            turns[1].text.contains("let me think" + reasoningClose + "\n\nthe answer"),
            "echoed reasoning must be re-rendered ahead of the answer: \(turns[1].text)"
        )
    }

    func testAnthropicEchoedThinkingBlockFoldsIntoAssistantTurn() throws {
        let body = """
        {"max_tokens":10,"messages":[
            {"role":"user","content":"hello"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"let me think","signature":"sig"},
                {"type":"text","text":"the answer"}
            ]},
            {"role":"user","content":"again"}
        ]}
        """
        let request = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(body.utf8))
        let turns = try AnthropicAdapter.splitMessages(request.messages)
        XCTAssertEqual(turns.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(turns[1].text, "let me think" + reasoningClose + "\n\nthe answer")
        XCTAssertEqual(turns[0].text, "hello", "thinking blocks must not leak into the visible text")
    }

    func testAssistantTurnWithoutReasoningIsUnchanged() throws {
        let body = """
        {"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"plain"},{"role":"user","content":"x"}]}
        """
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.utf8))
        let (_, turns) = try OpenAIAdapter.splitMessages(request.messages)
        XCTAssertEqual(turns[1].text, "plain")
    }

    // MARK: - Substitution safety

    func testRecordIsCompatibleWithTheEchoesClientsActuallySend() {
        let raw = "thinking" + reasoningClose + "the answer"
        // Clients that drop reasoning echo the visible answer.
        XCTAssertTrue(ServerTurnMemory.isCompatible(echo: "the answer", visibleResponse: "the answer", rawTurn: raw))
        // Clients that keep it (signed thinking blocks, reasoning_content) echo more.
        XCTAssertTrue(ServerTurnMemory.isCompatible(echo: raw, visibleResponse: "the answer", rawTurn: raw))
        // Assistant tool-call turns carry no text at all.
        XCTAssertTrue(ServerTurnMemory.isCompatible(echo: "", visibleResponse: "the answer", rawTurn: raw))
        XCTAssertTrue(ServerTurnMemory.isCompatible(echo: "  \n ", visibleResponse: "the answer", rawTurn: raw))
    }

    func testRecordIsRejectedWhenHistoryWasEdited() {
        let raw = "thinking" + reasoningClose + "the answer"
        XCTAssertFalse(ServerTurnMemory.isCompatible(echo: "a different answer", visibleResponse: "the answer", rawTurn: raw))
        XCTAssertFalse(ServerTurnMemory.isCompatible(echo: "the answer, edited", visibleResponse: "the answer", rawTurn: raw))
    }

    func testStoreKeepsOnlyTheLatestRecord() {
        let memory = ServerTurnMemory()
        let session = UUID()
        memory.store(.init(
            sessionId: session,
            modelName: "m",
            rawDecodedText: "first",
            visibleResponse: "first",
            promptTokenIds: [],
            generatedTokenIds: []
        ))
        XCTAssertEqual(
            memory.recordedRawTurn(sessionId: session, modelName: "m", echoedByClient: "first", turnCloser: nil),
            "first"
        )

        // A different model or session must not be able to reuse the record.
        XCTAssertNil(memory.recordedRawTurn(sessionId: session, modelName: "other", echoedByClient: "first", turnCloser: nil))
        XCTAssertNil(memory.recordedRawTurn(sessionId: UUID(), modelName: "m", echoedByClient: "first", turnCloser: nil))

        memory.clear()
        XCTAssertNil(memory.recordedRawTurn(sessionId: session, modelName: "m", echoedByClient: "first", turnCloser: nil))
    }

    func testTrailingTurnCloserIsDroppedFromTheRecordedTurn() {
        let memory = ServerTurnMemory()
        let session = UUID()
        memory.store(.init(
            sessionId: session,
            modelName: "m",
            rawDecodedText: "why" + reasoningClose + "the answer<|im_end|>",
            visibleResponse: "the answer",
            promptTokenIds: [],
            generatedTokenIds: []
        ))
        XCTAssertEqual(
            memory.recordedRawTurn(sessionId: session, modelName: "m", echoedByClient: "the answer", turnCloser: "<|im_end|>"),
            "why" + reasoningClose + "the answer",
            "the closer the client never saw must not block reuse"
        )
    }

    // MARK: - Inline reasoning (the OpenAI server setting)

    func testInlineReasoningTextMirrorsTheGeneration() {
        // Only the close tag is emitted: the open tag is already the end of the
        // prompt, and repeating it would double the tag when the turn is
        // re-rendered out of history.
        let text = ServerController.inlineReasoningText(thinking: "  why  ", response: "the answer\n")
        XCTAssertEqual(text, "why" + reasoningClose + "\n\nthe answer")
        XCTAssertFalse(text.contains(TurnTextSplitter.canonicalReasoningOpenTag), "the open tag belongs to the prompt")
    }

    func testInlineReasoningTextLeavesAnswersWithoutReasoningUntouched() {
        XCTAssertEqual(ServerController.inlineReasoningText(thinking: "", response: "  the answer \n"), "  the answer \n")
        XCTAssertEqual(ServerController.inlineReasoningText(thinking: " \n ", response: "x"), "x")
    }

    func testInlineReasoningTextWithNoVisibleAnswer() {
        XCTAssertEqual(
            ServerController.inlineReasoningText(thinking: "why", response: "  "),
            "why" + reasoningClose
        )
    }

    func testInlinedReasoningEchoesBackToAReusableTurn() throws {
        // Inline mode puts the reasoning in `content`, so a client that ignores
        // `reasoning_content` still echoes it back — and the echoed turn is
        // exactly the generation the prompt expects to see again.
        let inline = ServerController.inlineReasoningText(thinking: "why", response: "the answer")
        let messages = [
            OpenAIChatRequest.Message(role: "user", contentText: "hello", name: nil, toolCalls: nil, toolCallId: nil),
            OpenAIChatRequest.Message(role: "assistant", contentText: inline, name: nil, toolCalls: nil, toolCallId: nil),
            OpenAIChatRequest.Message(role: "user", contentText: "again", name: nil, toolCalls: nil, toolCallId: nil),
        ]
        let (_, turns) = try OpenAIAdapter.splitMessages(messages)
        XCTAssertEqual(turns[1].text, "why" + reasoningClose + "\n\nthe answer")

        // The recorded generation carries the family's turn closer (which the
        // client never sees); it is dropped before the echo is compared, so the
        // turn still validates as reusable.
        let memory = ServerTurnMemory()
        let session = UUID()
        memory.store(.init(
            sessionId: session,
            modelName: "m",
            rawDecodedText: "why" + reasoningClose + "\n\nthe answer<|im_end|>",
            visibleResponse: "the answer",
            promptTokenIds: [],
            generatedTokenIds: []
        ))
        XCTAssertEqual(
            memory.recordedRawTurn(sessionId: session, modelName: "m", echoedByClient: turns[1].text, turnCloser: "<|im_end|>"),
            "why" + reasoningClose + "\n\nthe answer"
        )
    }

    func testInlineReasoningSettingDefaultsOffAndHonorsOverride() {
        let defaults = UserDefaults.standard
        let saved = defaults.object(forKey: ServerController.inlineReasoningKey)
        defer {
            if let saved {
                defaults.set(saved, forKey: ServerController.inlineReasoningKey)
            } else {
                defaults.removeObject(forKey: ServerController.inlineReasoningKey)
            }
        }
        defaults.removeObject(forKey: ServerController.inlineReasoningKey)
        XCTAssertFalse(ServerController.resolvedInlineReasoning, "out-of-band reasoning_content stays the default")

        defaults.set(true, forKey: ServerController.inlineReasoningKey)
        XCTAssertTrue(ServerController.resolvedInlineReasoning)
    }

    // MARK: - Family turn closers
    func testAssistantTurnCloseTagPerFamily() throws {
        XCTAssertEqual(ChatPromptFormatter.assistantTurnCloseTag(modelConfig: nil), "<|im_end|>")
        XCTAssertEqual(ChatPromptFormatter.assistantTurnCloseTag(modelConfig: try config("bailing")), "<|role_end|>")
        XCTAssertEqual(ChatPromptFormatter.assistantTurnCloseTag(modelConfig: try config("gemma4")), "<turn|>")
        XCTAssertNil(ChatPromptFormatter.assistantTurnCloseTag(modelConfig: try config("spark")))
    }
}
