//
//  OpenAIAdapterLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the OpenAI adapter's protocol mapping: request
//  decoding (string + array content, string stop), message splitting, tool
//  declarations, finish-reason mapping, and stable session keys.
//
//  Note on tag-bearing literals: model-family control tags are assembled
//  from codepoints, never retyped from a rendered view (AGENTS.md).
//

import XCTest
@testable import DynaMoE

final class OpenAIAdapterLogicTests: XCTestCase {

    private func decodeRequest(_ json: String) throws -> OpenAIChatRequest {
        let decoder = JSONDecoder()
        return try decoder.decode(OpenAIChatRequest.self, from: Data(json.utf8))
    }

    // MARK: - Request decoding

    func testDecodesMinimalRequestWithStringContent() throws {
        let request = try decodeRequest("""
        {"model":"m","messages":[{"role":"user","content":"hi"}]}
        """)
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].contentText, "hi")
        XCTAssertEqual(request.stream, nil)
        XCTAssertEqual(request.stop, nil)
    }

    func testDecodesMultipartContentArray() throws {
        let request = try decodeRequest("""
        {"messages":[{"role":"user","content":[{"type":"text","text":"part one"},{"type":"text","text":"part two"}]}]}
        """)
        XCTAssertEqual(request.messages[0].contentText, "part one\npart two")
    }

    func testDecodesBareStringStop() throws {
        let request = try decodeRequest("""
        {"messages":[],"stop":"END"}
        """)
        XCTAssertEqual(request.stop, ["END"])
    }

    func testDecodesSamplerExtras() throws {
        let request = try decodeRequest("""
        {"messages":[],"temperature":0.3,"top_k":40,"min_p":0.02,"repetition_penalty":1.05,"presence_penalty":0.1,"max_tokens":123,"stream":true,"stop":["a","b"]}
        """)
        XCTAssertEqual(request.temperature, 0.3)
        XCTAssertEqual(request.topK, 40)
        XCTAssertEqual(request.minP ?? -1, 0.02, accuracy: 0.0001)
        XCTAssertEqual(request.repetitionPenalty ?? -1, 1.05, accuracy: 0.0001)
        XCTAssertEqual(request.presencePenalty ?? -1, 0.1, accuracy: 0.0001)
        XCTAssertEqual(request.maxTokens, 123)
        XCTAssertEqual(request.stream, true)
        XCTAssertEqual(request.stop, ["a", "b"])
    }

    func testDecodesToolSchemas() throws {
        let request = try decodeRequest("""
        {"messages":[],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}
        """)
        XCTAssertEqual(request.tools?.count, 1)
        XCTAssertEqual(request.tools?[0].function.name, "get_weather")
        let argsJSON = try XCTUnwrap(request.tools?[0].function.parametersData).flatMap { String(data: $0, encoding: .utf8) }
        XCTAssertTrue(argsJSON?.contains("\"city\"") == true, "schema must embed parameter names")
    }

    // MARK: - Message splitting

    func testSingleTurnPassesUserTextThroughVerbatim() throws {
        let messages = [
            OpenAIChatRequest.Message(role: "system", contentText: "sys prompt", name: nil, toolCalls: nil, toolCallId: nil),
            OpenAIChatRequest.Message(role: "user", contentText: "what is 2+2", name: nil, toolCalls: nil, toolCallId: nil),
        ]
        let (system, user) = try OpenAIAdapter.splitMessages(messages)
        XCTAssertEqual(system, "sys prompt")
        XCTAssertEqual(user, "what is 2+2")
    }

    func testAssistantToolCallRendersAsJSON() throws {
        let body = """
        {"messages":[
            {"role":"user","content":"run the tests"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell_run","arguments":"{\\"cmd\\":\\"make test\\"}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"all green"}
        ]}
        """
        let request = try decodeRequest(body)
        let (system, user) = try OpenAIAdapter.splitMessages(request.messages)
        XCTAssertEqual(system, "")
        XCTAssertTrue(user.contains("User: run the tests"))
        XCTAssertTrue(user.contains("shell_run"))
        XCTAssertTrue(user.contains("make test"))
        XCTAssertTrue(user.contains("Function result (call_1)"))
        XCTAssertTrue(user.contains("all green"))
    }

    // MARK: - Finish reasons

    func testFinishReasonMapping() {
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.eos, hasToolCalls: false), "stop")
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.endTag, hasToolCalls: false), "stop")
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.stoppedBySequence, hasToolCalls: false), "stop")
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.stopped, hasToolCalls: false), "stop")
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.degenerateCycle, hasToolCalls: false), "stop")
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.lengthCapped, hasToolCalls: false), "length")
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.toolFreeze, hasToolCalls: true), "tool_calls")
        XCTAssertEqual(OpenAIAdapter.finishReasonString(.eos, hasToolCalls: true), "tool_calls")
    }

    // MARK: - Tool declarations

    func testToolDeclarationEmbedsSchemasAsValidJSON() throws {
        let request = try decodeRequest("""
        {"messages":[],"tools":[{"type":"function","function":{"name":"list_files","parameters":{"type":"object","properties":{}}}}]}
        """)
        let text = try XCTUnwrap(OpenAIAdapter.toolDeclarationText(try XCTUnwrap(request.tools)))
        XCTAssertTrue(text.contains("list_files"))
        XCTAssertTrue(text.contains("FUNCTIONS: ["))
        // The embedded array itself must parse back out as JSON.
        let jsonStart = try XCTUnwrap(text.range(of: "FUNCTIONS: "))
        let jsonText = String(text[jsonStart.upperBound...])
        let parsed = try JSONSerialization.jsonObject(with: Data(jsonText.utf8))
        XCTAssertTrue(parsed is [[String: Any]], "declaration must embed a JSON array")
    }

    func testToolDeclarationNilWithoutFunctions() throws {
        let request = try decodeRequest("""
        {"messages":[],"tools":[{"type":"function","function":{"name":"x","parameters":null}}]}
        """)
        XCTAssertNotNil(OpenAIAdapter.toolDeclarationText(request.tools!))
        XCTAssertNil(OpenAIAdapter.toolDeclarationText([]))
    }

    // MARK: - Session keys

    func testSessionKeyStableAcrossConversationTurns() throws {
        // The key is anchored on the conversation's first message, which
        // stateless clients resend verbatim with every turn.
        let first = try decodeRequest("""
        {"messages":[{"role":"system","content":"you are helpful"},{"role":"user","content":"turn one"}]}
        """)
        let second = try decodeRequest("""
        {"messages":[{"role":"system","content":"you are helpful"},{"role":"user","content":"turn one"},{"role":"assistant","content":"answer one"},{"role":"user","content":"turn two"}]}
        """)
        let keyOne = OpenAIAdapter.sessionKey(for: first.messages, model: "m")
        let keyTwo = OpenAIAdapter.sessionKey(for: second.messages, model: "m")
        XCTAssertEqual(keyOne, keyTwo, "appending turns must keep the session identity stable")
    }

    func testSessionKeyDiffersAcrossConversationsAndModels() throws {
        let a = try decodeRequest(#"{"messages":[{"role":"user","content":"hello"}]}"#)
        let b = try decodeRequest(#"{"messages":[{"role":"user","content":"different"}]}"#)
        XCTAssertNotEqual(
            OpenAIAdapter.sessionKey(for: a.messages, model: "m"),
            OpenAIAdapter.sessionKey(for: b.messages, model: "m")
        )
        XCTAssertNotEqual(
            OpenAIAdapter.sessionKey(for: a.messages, model: "m1"),
            OpenAIAdapter.sessionKey(for: a.messages, model: "m2")
        )
    }
}
