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
        let split = try OpenAIAdapter.splitMessages(messages)
        let system = split.systemPrompt
        let turns = split.turns
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
        let split = try OpenAIAdapter.splitMessages(request.messages)
        let system = split.systemPrompt
        let turns = split.turns
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

    // MARK: - GET /v1/models

    /// A cost block as `CompletionEngine` publishes it for a real tower, carrying
    /// the published checkpoint's own numbers (`DeepSeekV41VisionCostTests` holds
    /// those to the resize plan). Rounding is the wire layer's job, so these stay
    /// unrounded.
    private func visionBounds() -> DeepSeekV41VisionPricing.Bounds {
        DeepSeekV41VisionPricing.Bounds(
            minPatchTokens: 1_521,
            maxPatchTokens: 9_216,
            minSpanPositions: 184,
            maxSpanPositions: 1_024,
            minSeconds: 0.3202,
            maxSeconds: 7.8407
        )
    }

    /// A load row of a vision-capable family, as `CompletionEngine` publishes it.
    private func loadedInfo(
        vision: Bool, thinking: Bool, levels: Bool,
        visionCost: DeepSeekV41VisionPricing.Bounds? = nil
    ) -> CompletionEngine.LoadedModelInfo {
        CompletionEngine.LoadedModelInfo(
            displayName: "DeepSeek-V4.1-Flash-1017",
            path: "/nonexistent/snapshot",
            supportsThinking: thinking,
            supportsThinkingLevels: levels,
            supportsVision: vision,
            visionCost: visionCost,
            weightsSizeGB: 475.25
        )
    }

    private func encodedModel(_ model: OpenAIModelList.Model) throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: try JSONEncoder().encode(model)) as? [String: Any]
        )
    }

    /// `/v1/models` is the only way a client can learn that an attachment will be
    /// read rather than refused, so the capability it reports has to be the fact
    /// the planner uses. `#78`/`#79`/`#80` each flagged its absence.
    ///
    /// `vision: true` is not by itself actionable: a picture can be 184 prompt
    /// positions and a third of a second or 1,024 and nearly eight, and a client
    /// cannot tell from a boolean which end it is sizing for. So the cost rides
    /// beside the flag, on the keys a client would query.
    func testModelListAdvertisesTheCapabilitiesThePlannerGatesOn() throws {
        let model = OpenAIModelList.entry(
            for: loadedInfo(
                vision: true, thinking: true, levels: true, visionCost: visionBounds()
            ),
            contextLength: 163_840,
            maxPromptTokens: 262_144,
            created: 1_700_000_000
        )
        let json = try encodedModel(model)
        XCTAssertEqual(json["id"] as? String, "DeepSeek-V4.1-Flash-1017")
        XCTAssertEqual(json["object"] as? String, "model")
        XCTAssertEqual(json["owned_by"] as? String, "dynamoe")
        XCTAssertEqual(json["created"] as? Int, 1_700_000_000)
        XCTAssertEqual(json["context_length"] as? Int, 163_840)
        XCTAssertEqual(json["max_prompt_tokens"] as? Int, 262_144)
        let capabilities = try XCTUnwrap(json["capabilities"] as? [String: Any])
        XCTAssertEqual(capabilities["vision"] as? Bool, true)
        XCTAssertEqual(capabilities["thinking"] as? Bool, true)
        XCTAssertEqual(capabilities["thinking_levels"] as? Bool, true)
        let cost = try XCTUnwrap(capabilities["vision_cost"] as? [String: Any])
        XCTAssertEqual(cost["min_prompt_positions"] as? Int, 184)
        XCTAssertEqual(cost["max_prompt_positions"] as? Int, 1_024)
        XCTAssertEqual(cost["min_patch_tokens"] as? Int, 1_521)
        XCTAssertEqual(cost["max_patch_tokens"] as? Int, 9_216)
        // Two decimals: it is a model of a machine class, not a promise, and a
        // client must not read 0.32020000000000004 as a measurement.
        XCTAssertEqual(cost["min_encode_seconds"] as? Double, 0.32)
        XCTAssertEqual(cost["max_encode_seconds"] as? Double, 7.84)
        // The wire must not leak Swift property names, at either level.
        let encoded = String(decoding: try JSONEncoder().encode(model), as: UTF8.self)
        for leaked in ["thinkingLevels", "visionCost", "minPromptPositions",
                       "maxPatchTokens", "minEncodeSeconds", "minSpanPositions"] {
            XCTAssertFalse(encoded.contains(leaked), "\(leaked) leaked into the payload")
        }
    }

    /// The flags must be copied, not hardcoded: a tower-less family must advertise
    /// `vision: false` or a client sends a picture that is refused with a 400, and
    /// a family with a plain on/off switch must not claim graded levels.
    ///
    /// The cost block follows the same rule, and it is the case that matters here:
    /// a checkpoint can publish a `vision_config` and a resize plan and still not
    /// ship the tower's weights, so the bounds exist while `supportsVision` is
    /// false. Advertising them would sell an attachment the server refuses.
    func testModelListCapabilitiesFollowTheLoadedModelRatherThanTheWire() throws {
        let json = try encodedModel(OpenAIModelList.entry(
            for: loadedInfo(
                vision: false, thinking: true, levels: false, visionCost: visionBounds()
            ),
            contextLength: nil,
            maxPromptTokens: nil,
            created: 0
        ))
        let capabilities = try XCTUnwrap(json["capabilities"] as? [String: Any])
        XCTAssertEqual(capabilities["vision"] as? Bool, false)
        XCTAssertEqual(capabilities["thinking"] as? Bool, true)
        XCTAssertEqual(capabilities["thinking_levels"] as? Bool, false)
        XCTAssertNil(capabilities["vision_cost"], "no tower, no cost to advertise")
        // A model with no architectural ceiling and a server with no cap both
        // report nothing rather than a sentinel a client would read as a number.
        XCTAssertNil(json["context_length"])
        XCTAssertNil(json["max_prompt_tokens"])
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

    // MARK: - Request fields this server cannot honour

    func testTheFieldsTheServerCannotHonourAreDecodedRatherThanIgnored() throws {
        // Before `#88` these keys were absent from the decoder, so Swift's
        // unknown-key tolerance dropped them and the turn was built as if the client
        // had asked for none of it. Decoding them is what lets the planner refuse
        // them by name.
        let request = try decodeRequest("""
        {"model":"m","messages":[{"role":"user","content":"hi"}],
         "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}},
         "n":3,"logprobs":true,"top_logprobs":2,
         "stream_options":{"include_usage":true}}
        """)
        XCTAssertEqual(request.responseFormat?.type, "json_schema")
        XCTAssertEqual(request.n, 3)
        XCTAssertEqual(request.logprobs, true)
        XCTAssertEqual(request.topLogprobs, 2)
        XCTAssertEqual(request.streamOptions?.includeUsage, true)
    }

    func testStructuredOutputIsRefusedByNameRatherThanAnsweredInProse() throws {
        for type in ["json_object", "json_schema"] {
            let request = try decodeRequest("""
            {"messages":[],"response_format":{"type":"\(type)"}}
            """)
            XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedResponseShape(request)) { error in
                XCTAssertTrue(
                    "\(error)".contains("response_format"),
                    "the refusal must name the field the client sent, got: \(error)"
                )
                XCTAssertTrue(
                    "\(error)".contains(type),
                    "the refusal must name the mode it cannot honour"
                )
            }
        }

        // OpenAI's default mode is a no-op, and an absent field is every request that
        // does not ask.
        for body in [#"{"messages":[]}"#, #"{"messages":[],"response_format":{"type":"text"}}"#] {
            XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedResponseShape(try decodeRequest(body)))
        }

        let unknown = try decodeRequest(#"{"messages":[],"response_format":{"type":"yaml"}}"#)
        XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedResponseShape(unknown)) { error in
            XCTAssertTrue("\(error)".contains("yaml"))
        }
    }

    func testMultipleChoicesAndTokenLogprobsAreRefusedByName() throws {
        XCTAssertNoThrow(
            try OpenAIAdapter.refuseUnsupportedResponseShape(try decodeRequest(#"{"messages":[],"n":1}"#)),
            "n = 1 is the protocol's own default"
        )
        for body in [#"{"messages":[],"n":2}"#, #"{"messages":[],"n":0}"#] {
            XCTAssertThrowsError(
                try OpenAIAdapter.refuseUnsupportedResponseShape(try decodeRequest(body))
            ) { error in
                XCTAssertTrue("\(error)".contains("n ="), "the refusal must name the field, got: \(error)")
            }
        }

        XCTAssertNoThrow(
            try OpenAIAdapter.refuseUnsupportedResponseShape(try decodeRequest(#"{"messages":[],"logprobs":false}"#)),
            "asking NOT to receive log-probabilities is already the only thing this server does"
        )
        for body in [#"{"messages":[],"logprobs":true}"#, #"{"messages":[],"top_logprobs":5}"#] {
            XCTAssertThrowsError(
                try OpenAIAdapter.refuseUnsupportedResponseShape(try decodeRequest(body))
            ) { error in
                XCTAssertTrue("\(error)".contains("logprobs"), "got: \(error)")
            }
        }
    }

    func testStreamOptionsAreReadBecauseTheirOneMeaningfulFieldIsHonoured() throws {
        // The usage tail is opt-in by protocol, so this field is decoded and obeyed
        // rather than refused: the streaming writer emits the chunk only for a client
        // that asked for it.
        let absent = try decodeRequest(#"{"messages":[]}"#)
        XCTAssertNil(absent.streamOptions)

        let declined = try decodeRequest(#"{"messages":[],"stream_options":{"include_usage":false}}"#)
        XCTAssertEqual(declined.streamOptions?.includeUsage, false)

        let asked = try decodeRequest(#"{"messages":[],"stream_options":{"include_usage":true}}"#)
        XCTAssertEqual(asked.streamOptions?.includeUsage, true)
        XCTAssertNoThrow(
            try OpenAIAdapter.refuseUnsupportedResponseShape(asked),
            "a field this server can honour must never be refused"
        )
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
        let split = try OpenAIAdapter.splitMessages(request.messages)
        let turns = split.turns
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

    // MARK: - Graded thinking levels

    func testReasoningEffortAcceptsOnlySupportedLevels() throws {
        XCTAssertNil(try OpenAIAdapter.thinkingEffort(from: nil, levelCapable: true, thinkingRequested: true))
        XCTAssertEqual(try OpenAIAdapter.thinkingEffort(from: "low", levelCapable: true, thinkingRequested: true), .low)
        XCTAssertEqual(try OpenAIAdapter.thinkingEffort(from: " HIGH ", levelCapable: true, thinkingRequested: true), .high)
        XCTAssertThrowsError(try OpenAIAdapter.thinkingEffort(from: "medium", levelCapable: true, thinkingRequested: true))
        XCTAssertThrowsError(try OpenAIAdapter.thinkingEffort(from: "low", levelCapable: false, thinkingRequested: true))
    }

    func testReasoningEffortIsMootWhenThinkingIsOff() throws {
        XCTAssertNil(try OpenAIAdapter.thinkingEffort(from: "high", levelCapable: true, thinkingRequested: false))
        XCTAssertNil(try OpenAIAdapter.thinkingEffort(from: "high", levelCapable: false, thinkingRequested: false))
    }
}
