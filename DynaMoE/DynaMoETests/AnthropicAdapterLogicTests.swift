//
//  AnthropicAdapterLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the Anthropic adapter's protocol mapping: request
//  decoding (string + block content, system field, tools, thinking config),
//  message splitting, tool declarations, finish-reason mapping, and stable
//  session keys.
//

import XCTest
@testable import DynaMoE

final class AnthropicAdapterLogicTests: XCTestCase {

    private func decodeRequest(_ json: String) throws -> AnthropicMessagesRequest {
        let decoder = JSONDecoder()
        return try decoder.decode(AnthropicMessagesRequest.self, from: Data(json.utf8))
    }

    // MARK: - Request decoding

    func testDecodesMinimalRequestWithStringContent() throws {
        let request = try decodeRequest("""
        {"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}
        """)
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].contentText, "hi")
        XCTAssertEqual(request.maxTokens, 100)
        XCTAssertEqual(request.stream, nil)
        XCTAssertEqual(request.systemText, nil)
    }

    func testDecodesSystemAsBareStringAndBlockArray() throws {
        let stringForm = try decodeRequest("""
        {"max_tokens":10,"system":"be terse","messages":[{"role":"user","content":"hi"}]}
        """)
        XCTAssertEqual(stringForm.systemText, "be terse")

        let blockForm = try decodeRequest("""
        {"max_tokens":10,"system":[{"type":"text","text":"part one"},{"type":"text","text":"part two"}],"messages":[{"role":"user","content":"hi"}]}
        """)
        XCTAssertEqual(blockForm.systemText, "part one\n\npart two")
    }

    func testDecodesMultipartContentBlocks() throws {
        let request = try decodeRequest("""
        {"max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"part one"},{"type":"text","text":"part two"}]}]}
        """)
        XCTAssertEqual(request.messages[0].contentText, "part one\npart two")
    }

    func testDecodesToolUseAndToolResultBlocks() throws {
        let request = try decodeRequest("""
        {"max_tokens":10,"messages":[
            {"role":"assistant","content":[
                {"type":"text","text":"I'll run that."},
                {"type":"tool_use","id":"toolu_1","name":"shell_run","input":{"cmd":"make test"}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"all green"}]}
            ]}
        ]}
        """)
        XCTAssertEqual(request.messages[0].contentText, "I'll run that.")
        let use = try XCTUnwrap(request.messages[0].toolUses?.first)
        XCTAssertEqual(use.name, "shell_run")
        XCTAssertEqual(use.id, "toolu_1")
        XCTAssertTrue(use.inputJSON?.contains("make test") == true)
        let result = try XCTUnwrap(request.messages[1].toolResults?.first)
        XCTAssertEqual(result.toolUseId, "toolu_1")
        XCTAssertEqual(result.content, "all green")
    }

    func testDecodesSamplerAndStreamExtras() throws {
        let request = try decodeRequest("""
        {"max_tokens":123,"temperature":0.3,"top_p":0.8,"top_k":40,"stop_sequences":["END"],"stream":true,"messages":[{"role":"user","content":"hi"}]}
        """)
        XCTAssertEqual(request.maxTokens, 123)
        XCTAssertEqual(request.temperature ?? -1, 0.3, accuracy: 0.0001)
        XCTAssertEqual(request.topP ?? -1, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.topK, 40)
        XCTAssertEqual(request.stopSequences, ["END"])
        XCTAssertEqual(request.stream, true)
    }

    func testDecodesToolSchemasAndThinkingConfig() throws {
        let request = try decodeRequest("""
        {"max_tokens":10,"thinking":{"type":"enabled","budget_tokens":2048},"tool_choice":{"type":"auto"},"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"messages":[{"role":"user","content":"hi"}]}
        """)
        XCTAssertEqual(request.tools?.count, 1)
        XCTAssertEqual(request.tools?[0].name, "get_weather")
        let schemaJSON = try XCTUnwrap(request.tools?[0].inputSchemaData).flatMap { String(data: $0, encoding: .utf8) }
        XCTAssertTrue(schemaJSON?.contains("\"city\"") == true, "schema must embed parameter names")
        XCTAssertEqual(request.thinking?.type, "enabled")
        XCTAssertEqual(request.thinking?.budgetTokens, 2048)
        XCTAssertEqual(request.toolChoiceType, "auto")
    }

    // MARK: - Message splitting

    func testSingleTurnPassesUserTextThroughVerbatim() throws {
        let messages = [
            AnthropicMessagesRequest.Message(role: "user", contentText: "what is 2+2", toolUses: nil, toolResults: nil),
        ]
        let userPrompt = try AnthropicAdapter.splitMessages(messages)
        XCTAssertEqual(userPrompt, "what is 2+2")
    }

    func testToolContinuationRendersAsTranscript() throws {
        let request = try decodeRequest("""
        {"max_tokens":10,"messages":[
            {"role":"user","content":"run the tests"},
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"shell_run","input":{"cmd":"make test"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"all green"}]}
        ]}
        """)
        let userPrompt = try AnthropicAdapter.splitMessages(request.messages)
        XCTAssertTrue(userPrompt.contains("User: run the tests"))
        XCTAssertTrue(userPrompt.contains("shell_run"))
        XCTAssertTrue(userPrompt.contains("make test"))
        XCTAssertTrue(userPrompt.contains("Tool result (toolu_1)"))
        XCTAssertTrue(userPrompt.contains("all green"))
    }

    func testRejectsEmptyMessagesAndInvalidRoles() {
        XCTAssertThrowsError(try AnthropicAdapter.splitMessages([]))
        let invalid = [AnthropicMessagesRequest.Message(role: "system", contentText: "no", toolUses: nil, toolResults: nil)]
        XCTAssertThrowsError(try AnthropicAdapter.splitMessages(invalid))
    }

    // MARK: - Finish reasons

    func testFinishReasonMapping() {
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.eos, hasToolCalls: false), "end_turn")
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.endTag, hasToolCalls: false), "end_turn")
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.stopped, hasToolCalls: false), "end_turn")
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.degenerateCycle, hasToolCalls: false), "end_turn")
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.lengthCapped, hasToolCalls: false), "max_tokens")
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.stoppedBySequence, hasToolCalls: false), "stop_sequence")
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.toolFreeze, hasToolCalls: true), "tool_use")
        XCTAssertEqual(AnthropicAdapter.finishReasonString(.eos, hasToolCalls: true), "tool_use")
    }

    // MARK: - Tool declarations

    func testToolDeclarationEmbedsSchemasAsValidJSON() throws {
        let request = try decodeRequest("""
        {"max_tokens":10,"tools":[{"name":"list_files","input_schema":{"type":"object","properties":{}}}],"messages":[{"role":"user","content":"hi"}]}
        """)
        let text = try XCTUnwrap(AnthropicAdapter.toolDeclarationText(try XCTUnwrap(request.tools)))
        XCTAssertTrue(text.contains("list_files"))
        XCTAssertTrue(text.contains("FUNCTIONS: ["))
        let jsonStart = try XCTUnwrap(text.range(of: "FUNCTIONS: "))
        let jsonText = String(text[jsonStart.upperBound...])
        let parsed = try JSONSerialization.jsonObject(with: Data(jsonText.utf8))
        XCTAssertTrue(parsed is [[String: Any]], "declaration must embed a JSON array")
    }

    func testToolDeclarationNilWithoutTools() {
        XCTAssertNil(AnthropicAdapter.toolDeclarationText([]))
    }

    // MARK: - Session keys

    func testSessionKeyStableAcrossConversationTurns() throws {
        let first = try decodeRequest("""
        {"max_tokens":10,"messages":[{"role":"user","content":"turn one"}]}
        """)
        let second = try decodeRequest("""
        {"max_tokens":10,"messages":[{"role":"user","content":"turn one"},{"role":"assistant","content":"answer one"},{"role":"user","content":"turn two"}]}
        """)
        XCTAssertEqual(
            AnthropicAdapter.sessionKey(for: first.messages, model: "m"),
            AnthropicAdapter.sessionKey(for: second.messages, model: "m"),
            "appending turns must keep the session identity stable"
        )
    }

    func testSessionKeyDiffersAcrossConversationsAndModels() throws {
        let a = try decodeRequest(#"{"max_tokens":10,"messages":[{"role":"user","content":"hello"}]}"#)
        let b = try decodeRequest(#"{"max_tokens":10,"messages":[{"role":"user","content":"different"}]}"#)
        XCTAssertNotEqual(
            AnthropicAdapter.sessionKey(for: a.messages, model: "m"),
            AnthropicAdapter.sessionKey(for: b.messages, model: "m")
        )
        XCTAssertNotEqual(
            AnthropicAdapter.sessionKey(for: a.messages, model: "m1"),
            AnthropicAdapter.sessionKey(for: a.messages, model: "m2")
        )
    }

    // MARK: - Response encoding

    func testContentBlockEncodingRoundTrips() throws {
        let textData = try JSONEncoder().encode(AnthropicContentBlock.text("hello"))
        let textJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: textData) as? [String: Any])
        XCTAssertEqual(textJSON["type"] as? String, "text")
        XCTAssertEqual(textJSON["text"] as? String, "hello")

        let toolData = try JSONEncoder().encode(AnthropicContentBlock.toolUse(id: "toolu_1", name: "shell_run", inputJSON: #"{"cmd":"ls"}"#))
        let toolJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: toolData) as? [String: Any])
        XCTAssertEqual(toolJSON["type"] as? String, "tool_use")
        XCTAssertEqual(toolJSON["name"] as? String, "shell_run")
        let input = try XCTUnwrap(toolJSON["input"] as? [String: Any])
        XCTAssertEqual(input["cmd"] as? String, "ls")
    }

    func testStreamDeltaEncoding() throws {
        let textDelta = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(AnthropicStreamDelta.text("abc"))) as? [String: Any])
        XCTAssertEqual(textDelta["type"] as? String, "text_delta")
        XCTAssertEqual(textDelta["text"] as? String, "abc")

        let jsonDelta = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(AnthropicStreamDelta.inputJSON(#"{"a":1}"#))) as? [String: Any])
        XCTAssertEqual(jsonDelta["type"] as? String, "input_json_delta")
        XCTAssertEqual(jsonDelta["partial_json"] as? String, #"{"a":1}"#)
    }
}
