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
        let argsJSON = try String(data: XCTUnwrap(request.tools?[0].function.parametersData), encoding: .utf8)
        XCTAssertTrue(argsJSON?.contains("\"city\"") == true, "schema must embed parameter names")
    }

    // MARK: - Message splitting

    func testSingleTurnPassesUserTextThroughVerbatim() throws {
        let messages = [
            OpenAIChatRequest.Message(role: "system", contentText: "sys prompt", name: nil, toolCalls: nil, toolCallId: nil),
            OpenAIChatRequest.Message(role: "user", contentText: "what is 2+2", name: nil, toolCalls: nil, toolCallId: nil),
        ]
        let (system, turns) = try OpenAIAdapter.splitMessages(messages)
        XCTAssertEqual(system, "sys prompt")
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].role, .user)
        XCTAssertEqual(turns[0].text, "what is 2+2")
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
        let (system, turns) = try OpenAIAdapter.splitMessages(request.messages)
        XCTAssertEqual(system, "")
        XCTAssertEqual(turns.map(\.role), [.user, .assistant, .user], "roles must stay native, not flattened into one user block")
        XCTAssertEqual(turns[0].text, "run the tests")
        XCTAssertTrue(turns[1].text.contains("shell_run"))
        XCTAssertTrue(turns[1].text.contains("make test"))
        XCTAssertTrue(turns[2].text.contains("Function result (call_1)"))
        XCTAssertTrue(turns[2].text.contains("all green"))
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

    // MARK: - Response wire format

    func testResponseMessageEncodesSnakeCaseToolCalls() throws {
        let call = OpenAIToolCallOut(
            index: nil,
            id: "call_1",
            type: "function",
            function: .init(name: "get_weather", arguments: "{\"city\":\"Oslo\"}"),
        )
        let response = OpenAIChatResponse(
            id: "chatcmpl_1",
            created: 1,
            model: "m",
            choices: [.init(index: 0, message: .init(content: nil, reasoningContent: nil, toolCalls: [call]), finishReason: "tool_calls")],
            usage: .init(promptTokens: 1, completionTokens: 1),
        )
        let data = try JSONEncoder().encode(response)
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertTrue(json.contains("\"tool_calls\""), "non-streaming message must use the OpenAI wire key")
        XCTAssertFalse(json.contains("toolCalls"), "Swift property name must not leak into the payload")
    }

    func testStreamDeltaEncodesSnakeCaseToolCalls() throws {
        let call = OpenAIToolCallOut(
            index: 0,
            id: "call_1",
            type: "function",
            function: .init(name: "get_weather", arguments: "{\"city\":\"Oslo\"}"),
        )
        let chunk = OpenAIChatChunk(
            id: "chatcmpl_1",
            created: 1,
            model: "m",
            choices: [.init(index: 0, delta: .init(role: nil, content: nil, reasoningContent: nil, toolCalls: [call]), finishReason: nil)],
        )
        let data = try JSONEncoder().encode(chunk)
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertTrue(json.contains("\"tool_calls\""), "streaming deltas must use the OpenAI wire key")
        XCTAssertFalse(json.contains("toolCalls"), "Swift property name must not leak into the payload")
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

    func testClientToolCallEmittedPerDeclarationIsParsed() throws {
        // The declaration must teach the parser-recognized tagged markup: a
        // client-defined tool is not in the built-in catalog, so a bare-JSON
        // emission would be returned as plain text instead of a tool call.
        let request = try decodeRequest("""
        {"messages":[],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}
        """)
        let text = try XCTUnwrap(OpenAIAdapter.toolDeclarationText(try XCTUnwrap(request.tools)))
        XCTAssertTrue(text.contains(StreamingToolParser.qwenToolCallOpen), "declaration must teach the tagged call format")
        XCTAssertTrue(text.contains(StreamingToolParser.qwenFunctionOpen), "declaration must teach the function tag")

        let emitted = StreamingToolParser.qwenToolCallOpen
            + "<function=get_weather><parameter=city>Oslo</parameter></function>"
            + StreamingToolParser.qwenToolCallClose
        let parsed = StreamingToolParser.shared.parseStreamingToolCalls(from: emitted, format: .qwenXML)
        XCTAssertEqual(parsed.calls.count, 1, "a client tool emitted per the declaration must parse as a call")
        XCTAssertEqual(parsed.calls.first?.name, "get_weather")
        XCTAssertEqual(parsed.calls.first?.arguments["city"] as? String, "Oslo")
    }

    // MARK: - Session keys

    func testSessionKeyStableAcrossConversationTurns() throws {
        // The key is anchored on the conversation's first USER message, which
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

    func testSessionKeyDistinguishesConversationsSharingASystemMessage() throws {
        // Every conversation a client opens usually carries the same system
        // message, so anchoring on the first message (rather than on the first
        // user turn) handed unrelated conversations a single session identity.
        let a = try decodeRequest("""
        {"messages":[{"role":"system","content":"you are helpful"},{"role":"user","content":"first question"}]}
        """)
        let b = try decodeRequest("""
        {"messages":[{"role":"system","content":"you are helpful"},{"role":"user","content":"another question"}]}
        """)
        XCTAssertNotEqual(
            OpenAIAdapter.sessionKey(for: a.messages, model: "m"),
            OpenAIAdapter.sessionKey(for: b.messages, model: "m"),
            "conversations that share only their system message must not share a session"
        )
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

    // MARK: - Tool choice

    func testObjectValuedToolChoiceDecodesAndForcedChoiceIsRejected() throws {
        // OpenAI's object form must decode rather than failing the body.
        let named = try decodeRequest("""
        {"messages":[],"tools":[{"type":"function","function":{"name":"get_weather"}}],"tool_choice":{"type":"function","function":{"name":"get_weather"}}}
        """)
        XCTAssertEqual(named.toolChoice, .function(name: "get_weather"))
        XCTAssertThrowsError(try OpenAIAdapter.toolUseEnabled(named.toolChoice)) { error in
            XCTAssertTrue("\(error)".contains("get_weather"), "the rejection must name the forced function")
        }

        let none = try decodeRequest(#"{"messages":[],"tool_choice":"none"}"#)
        XCTAssertEqual(none.toolChoice, OpenAIChatRequest.ToolChoice.none)
        XCTAssertFalse(try OpenAIAdapter.toolUseEnabled(none.toolChoice))

        let required = try decodeRequest(#"{"messages":[],"tool_choice":"required"}"#)
        XCTAssertThrowsError(try OpenAIAdapter.toolUseEnabled(required.toolChoice))

        let auto = try decodeRequest(#"{"messages":[],"tool_choice":"auto"}"#)
        XCTAssertEqual(auto.toolChoice, .auto)
        XCTAssertTrue(try OpenAIAdapter.toolUseEnabled(auto.toolChoice))
    }

    // MARK: - Tool IDs in history

    func testAssistantHistoryKeepsToolCallIds() throws {
        let body = """
        {"messages":[
            {"role":"user","content":"go"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_abc","type":"function","function":{"name":"shell_run","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call_abc","content":"ok"}
        ]}
        """
        let request = try decodeRequest(body)
        let (_, turns) = try OpenAIAdapter.splitMessages(request.messages)
        XCTAssertTrue(turns[1].text.contains("call_abc"), "the call id must reach history so results can be matched to calls")
    }

    // MARK: - Null preservation in tool schemas

    func testToolSchemaPreservesNullValues() throws {
        let request = try decodeRequest("""
        {"messages":[],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"unit":{"enum":["c","f"],"default":null}},"required":["unit"]}}}]}
        """)
        let data = try XCTUnwrap(request.tools?[0].function.parametersData)
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertTrue(json.contains("\"default\":null"), "a null schema default must survive, not become \"\"")
        XCTAssertFalse(json.contains("\"default\":\"\""))
    }
}
