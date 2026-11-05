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

    private func decodeCompletions(_ json: String) throws -> OpenAICompletionsRequest {
        let decoder = JSONDecoder()
        return try decoder.decode(OpenAICompletionsRequest.self, from: Data(json.utf8))
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
            displayName: "1bae0e1089",
            path: "/nonexistent/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1089",
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

    // MARK: - The legacy text-completions route

    func testTheLegacyCompletionsRequestDecodesItsOwnPromptShapes() throws {
        let single = try decodeCompletions(#"{"prompt":"The capital of France is"}"#)
        XCTAssertEqual(single.promptText, "The capital of France is")
        XCTAssertEqual(single.promptCount, 1)
        XCTAssertFalse(single.promptIsTokenIDs)

        let batch = try decodeCompletions(#"{"prompt":["a","b"]}"#)
        XCTAssertEqual(batch.promptCount, 2, "a batch is decoded so the planner can refuse it by count rather than silently running the first")

        let ids = try decodeCompletions(#"{"prompt":[1,2,3]}"#)
        XCTAssertTrue(ids.promptIsTokenIDs)
        XCTAssertNil(ids.promptText)

        let idBatch = try decodeCompletions(#"{"prompt":[[1,2],[3]]}"#)
        XCTAssertTrue(idBatch.promptIsTokenIDs)

        let shape = try decodeCompletions("""
        {"prompt":"x","stop":"END","echo":true,"logprobs":5,"best_of":1,"seed":7,
         "user":"someone","stream_options":{"include_usage":true}}
        """)
        XCTAssertEqual(shape.stop, ["END"], "the protocol allows a bare string stop")
        XCTAssertEqual(shape.echo, true)
        XCTAssertTrue(shape.logprobsRequested)
        XCTAssertEqual(shape.bestOf, 1)
        XCTAssertEqual(shape.streamOptions?.includeUsage, true)

        // `logprobs` is a Bool on the chat route and an Int here; either spelling is a
        // request for token log-probabilities.
        let logprobsAsBool = try decodeCompletions(#"{"prompt":"x","logprobs":true}"#)
        XCTAssertTrue(logprobsAsBool.logprobsRequested)
        let logprobsAsZero = try decodeCompletions(#"{"prompt":"x","logprobs":0}"#)
        XCTAssertFalse(logprobsAsZero.logprobsRequested)
        let logprobsAbsent = try decodeCompletions(#"{"prompt":"x"}"#)
        XCTAssertFalse(logprobsAbsent.logprobsRequested)

        XCTAssertThrowsError(
            try decodeCompletions(#"{"prompt":{"text":"x"}}"#),
            "a prompt shape the protocol does not define is a decode error, not an empty completion"
        )
    }

    func testTheCompletionsFieldsThisServerCannotHonourAreRefusedByName() throws {
        func refuse(_ body: String, naming needle: String, file: StaticString = #filePath, line: UInt = #line) throws {
            let request = try decodeCompletions(body)
            XCTAssertThrowsError(
                try OpenAIAdapter.refuseUnsupportedCompletionsShape(request),
                body,
                file: file,
                line: line
            ) { error in
                XCTAssertTrue(
                    "\(error)".contains(needle),
                    "the refusal must name \(needle), got: \(error)",
                    file: file,
                    line: line
                )
            }
        }

        try refuse(#"{"prompt":[1,2,3]}"#, naming: "token ids")
        try refuse(#"{"prompt":["a","b"]}"#, naming: "2 prompts")
        try refuse(#"{"prompt":""}"#, naming: "empty")
        try refuse(#"{"prompt":"x","suffix":"y"}"#, naming: "suffix")
        try refuse(#"{"prompt":"x","n":2}"#, naming: "n = 2")
        try refuse(#"{"prompt":"x","logprobs":1}"#, naming: "logprobs")
        try refuse(#"{"prompt":"x","best_of":3}"#, naming: "best_of")

        // Everything this route honours, and the two fields that cannot move an
        // answer (`seed` is a best-effort hint and `user` is an abuse label), is
        // accepted. The sampler fields are honoured now, so they are resolved
        // rather than refused: see `testTheSamplerFieldsAreResolvedRatherThanIgnored`.
        for body in [
            #"{"prompt":"x"}"#,
            #"{"prompt":"x","n":1,"logprobs":0,"best_of":1}"#,
            #"{"prompt":"x","frequency_penalty":0,"logit_bias":{}}"#,
            #"{"prompt":"x","echo":true,"presence_penalty":0.2,"temperature":0.4,"stop":["\n"]}"#,
            #"{"prompt":"x","seed":42,"user":"someone","stream":true,"stream_options":{"include_usage":true}}"#,
        ] {
            XCTAssertNoThrow(
                try OpenAIAdapter.refuseUnsupportedCompletionsShape(try decodeCompletions(body)),
                body
            )
        }
    }

    /// The reply budget has two spellings in OpenAI's own protocol, and a client that
    /// sends only the newer one must not be answered with the app's default budget.
    ///
    /// `max_completion_tokens` was not in the decoder at all before this, so
    /// `{"max_completion_tokens": 512}` ran to the app default of up to 8192 tokens.
    /// A request that carries both spellings with DIFFERENT values is refused rather
    /// than resolved by preference: whichever one the server picked, the answer would
    /// be a budget the client did not ask for.
    func testTheReplyBudgetIsResolvedAcrossItsTwoSpellings() throws {
        let olderOnly = try decodeRequest(#"{"messages":[],"max_tokens":128}"#)
        XCTAssertEqual(olderOnly.maxTokens, 128)
        XCTAssertNil(olderOnly.maxCompletionTokens)
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedResponseShape(olderOnly))
        let olderBudget = try OpenAIAdapter.replyBudget(
            maxTokens: olderOnly.maxTokens,
            maxCompletionTokens: olderOnly.maxCompletionTokens
        )
        XCTAssertEqual(olderBudget, 128)

        let newerOnly = try decodeRequest(#"{"messages":[],"max_completion_tokens":512}"#)
        XCTAssertNil(newerOnly.maxTokens)
        XCTAssertEqual(newerOnly.maxCompletionTokens, 512)
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedResponseShape(newerOnly))
        let newerBudget = try OpenAIAdapter.replyBudget(
            maxTokens: newerOnly.maxTokens,
            maxCompletionTokens: newerOnly.maxCompletionTokens
        )
        XCTAssertEqual(newerBudget, 512, "the newer field alone is the whole budget, not a hint")

        let agreeing = try decodeRequest(#"{"messages":[],"max_tokens":64,"max_completion_tokens":64}"#)
        XCTAssertNoThrow(
            try OpenAIAdapter.refuseUnsupportedResponseShape(agreeing),
            "two spellings that agree are not a conflict"
        )
        let agreeingBudget = try OpenAIAdapter.replyBudget(
            maxTokens: agreeing.maxTokens,
            maxCompletionTokens: agreeing.maxCompletionTokens
        )
        XCTAssertEqual(agreeingBudget, 64)

        let conflicting = try decodeRequest(#"{"messages":[],"max_tokens":64,"max_completion_tokens":512}"#)
        XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedResponseShape(conflicting)) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("max_tokens"), "the refusal must name the older field: \(text)")
            XCTAssertTrue(
                text.contains("max_completion_tokens"),
                "the refusal must name the newer field: \(text)"
            )
        }

        let absent = try OpenAIAdapter.replyBudget(maxTokens: nil, maxCompletionTokens: nil)
        XCTAssertNil(absent, "neither field means the app default, which the planner owns")
    }

    /// The fields both protocols accept and deliberately do not honour now reach the
    /// request model, so the help guide and the journal can name them instead of
    /// `Decodable` dropping them silently.
    func testTheAcceptedAndIgnoredFieldsAreDecodedRatherThanVanishing() throws {
        let chat = try decodeRequest("""
        {"messages":[],"parallel_tool_calls":false,"seed":7,"user":"someone"}
        """)
        XCTAssertEqual(chat.parallelToolCalls, false)
        XCTAssertNoThrow(
            try OpenAIAdapter.refuseUnsupportedResponseShape(chat),
            "an ignored field is documented, not refused: a 400 would break clients that always send it"
        )
    }

    /// OpenAI's `store` is the one field in this protocol whose entire meaning is a
    /// promise about the server rather than about the answer, and its two values are
    /// deliberately not symmetric: `false` is honoured per request (that row keeps its
    /// bodies out of the log), `true` cannot force capture because retention on this
    /// Mac is the operator's decision. Absent must stay distinguishable from both,
    /// because a client that never spoke gets the operator's default.
    func testTheRetentionPromiseIsDecodedOnBothOpenAIRoutes() throws {
        let refusing = try decodeRequest(#"{"messages":[],"store":false}"#)
        XCTAssertEqual(refusing.store, false)
        let allowing = try decodeRequest(#"{"messages":[],"store":true}"#)
        XCTAssertEqual(allowing.store, true)
        let silent = try decodeRequest(#"{"messages":[]}"#)
        XCTAssertNil(silent.store, "absent is not the same request as either value: it leaves retention to the operator")

        let legacy = try decodeCompletions(#"{"prompt":"x","store":false}"#)
        XCTAssertEqual(legacy.store, false, "both routes take the same retention promise")

        // Retention is a privacy promise, not a shape promise, so it is honoured
        // rather than refused: no new 400 appears on either path.
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedResponseShape(refusing))
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedCompletionsShape(legacy))
    }

    /// A row whose bodies were withheld has to explain itself, and the captured
    /// request summary is where a debugging session looks: `store` must survive
    /// into it, so "withheld at the client's request" is readable from the row
    /// alone rather than inferred from the switch's current value.
    func testTheCapturedRequestSummaryRecordsTheRetentionPromise() throws {
        func recordedStore(_ json: String) throws -> Bool? {
            let summary = try XCTUnwrap(ServerRequestLog.openAISummary(try decodeRequest(json)))
            let object = try JSONSerialization.jsonObject(with: Data(summary.utf8)) as? [String: Any]
            return object?["store"] as? Bool
        }
        XCTAssertEqual(try recordedStore(#"{"messages":[],"store":false}"#), false)
        XCTAssertEqual(try recordedStore(#"{"messages":[],"store":true}"#), true)
        XCTAssertNil(
            try recordedStore(#"{"messages":[]}"#),
            "a client that sent no promise records none: absent is not store: true"
        )

        let legacy = try decodeCompletions(#"{"prompt":"x","store":false}"#)
        let legacySummary = try XCTUnwrap(ServerRequestLog.openAICompletionsSummary(legacy))
        let legacyObject = try JSONSerialization.jsonObject(with: Data(legacySummary.utf8)) as? [String: Any]
        XCTAssertEqual(legacyObject?["store"] as? Bool, false)
    }

    /// OpenAI's two output-modality fields promise a SPOKEN reply, and this server has
    /// no text-to-speech at all: left out of the decoder they were dropped by unknown
    /// key tolerance, so a request for audio was answered as prose with no error and no
    /// `message.audio` for the client's parser to find. They are refused by name
    /// instead, and the one value the engine DOES produce is served rather than refused
    /// — a client naming `["text"]`, the protocol's default, must not meet a new 400.
    func testAudioOutputIsRefusedByNameAndPlainTextModalitiesIsServed() throws {
        // Silent, and explicit about the default: both are served.
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedResponseShape(try decodeRequest(#"{"messages":[]}"#)))
        let explicitText = try decodeRequest(#"{"messages":[],"modalities":["text"]}"#)
        XCTAssertEqual(explicitText.modalities, ["text"])
        XCTAssertTrue(OpenAIAdapter.unservableModalities(explicitText).isEmpty)
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedResponseShape(explicitText))

        // A spoken reply, in the shape the SDKs actually send.
        let spoken = try decodeRequest("""
        {"messages":[],"modalities":["text","audio"],"audio":{"voice":"alloy","format":"mp3","speed":1.1}}
        """)
        XCTAssertEqual(spoken.audio?.voice, "alloy")
        XCTAssertEqual(spoken.audio?.format, "mp3")
        XCTAssertEqual(OpenAIAdapter.unservableModalities(spoken), ["audio"])
        XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedResponseShape(spoken)) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("audio output is not supported"), "got: \(text)")
            XCTAssertTrue(text.contains("mp3"), "the refusal should name the format the client asked for, got: \(text)")
            XCTAssertTrue(text.contains("text-to-speech"), "the refusal should say why, got: \(text)")
        }

        // An `audio` object with no `modalities` is the same request: the settings are
        // only meaningful for a reply the client has been promised.
        let audioOnly = try decodeRequest(#"{"messages":[],"audio":{"voice":"nova"}}"#)
        XCTAssertEqual(OpenAIAdapter.unservableModalities(audioOnly), [])
        XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedResponseShape(audioOnly)) { error in
            XCTAssertTrue("\(error)".contains("audio output is not supported"), "got: \(error)")
        }

        // Any non-text modality, not just audio: the engine has exactly one output, so
        // the rule is "text or nothing" rather than a list of known audio values.
        let image = try decodeRequest(#"{"messages":[],"modalities":["text","image"]}"#)
        XCTAssertEqual(OpenAIAdapter.unservableModalities(image), ["image"])
        XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedResponseShape(image)) { error in
            XCTAssertTrue("\(error)".contains("image"), "the refusal must name the modality, got: \(error)")
        }
    }

    /// A refused request still gets a row, and that row's only body is the request
    /// summary — so the fields the refusal was ABOUT have to survive into it, or the
    /// 400 is unexplainable from the log.
    func testTheCapturedRequestSummaryRecordsTheRefusedModalityFields() throws {
        let spoken = try decodeRequest("""
        {"messages":[],"modalities":["text","audio"],"audio":{"voice":"alloy","format":"mp3"}}
        """)
        let summary = try XCTUnwrap(ServerRequestLog.openAISummary(spoken))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(summary.utf8)) as? [String: Any])
        XCTAssertEqual(object["modalities"] as? [String], ["text", "audio"])
        let audio = try XCTUnwrap(object["audio"] as? [String: Any])
        XCTAssertEqual(audio["voice"] as? String, "alloy")
        XCTAssertEqual(audio["format"] as? String, "mp3")
    }

    /// The two sampler fields OpenAI defines and this engine can honour are decoded,
    /// range-checked and resolved into the turn's sampling instead of ignored.
    ///
    /// Both routes carry them, and a value either route accepts has to survive the
    /// decode intact: a silently dropped `logit_bias` answers a request the client
    /// did not send, which is the class of lie `#88` and `#89` exist to remove.
    func testTheSamplerFieldsAreResolvedRatherThanIgnored() throws {
        let chat = try decodeRequest("""
        {"messages":[],"frequency_penalty":0.5,"logit_bias":{"1000":-100,"1005":5,"7":0},"seed":7,"user":"someone"}
        """)
        XCTAssertEqual(chat.frequencyPenalty, 0.5)
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedResponseShape(chat))
        let chatFrequency = try OpenAIAdapter.frequencyPenalty(chat.frequencyPenalty)
        XCTAssertEqual(chatFrequency, 0.5, accuracy: 1e-6)
        let chatBias = try OpenAIAdapter.logitBias(from: chat.logitBias)
        XCTAssertEqual(chatBias, [1000: -100, 1005: 5], "a zero bias changes nothing and is dropped")

        let legacy = try decodeCompletions(#"{"prompt":"x","frequency_penalty":-1.5,"logit_bias":{"1000":-100}}"#)
        XCTAssertEqual(legacy.frequencyPenalty, -1.5)
        XCTAssertNoThrow(try OpenAIAdapter.refuseUnsupportedCompletionsShape(legacy))
        let legacyFrequency = try OpenAIAdapter.frequencyPenalty(legacy.frequencyPenalty)
        XCTAssertEqual(legacyFrequency, -1.5, accuracy: 1e-6)
        let legacyBias = try OpenAIAdapter.logitBias(from: legacy.logitBias)
        XCTAssertEqual(legacyBias, [1000: -100])

        // Absent means "no penalty" on both, not an error.
        let absent = try OpenAIAdapter.frequencyPenalty(nil)
        XCTAssertEqual(absent, 0)
        let noBias = try OpenAIAdapter.logitBias(from: nil)
        XCTAssertTrue(noBias.isEmpty)
    }

    /// The values neither route can honour are refused BY NAME, before the model
    /// gate, rather than clamped: a clamped bias answers a distribution the client
    /// did not ask for.
    func testTheSamplerFieldsAreRefusedByNameWhereTheyCannotBeHonoured() throws {
        func refuseChat(_ body: String, naming needle: String, file: StaticString = #filePath, line: UInt = #line) throws {
            let request = try decodeRequest(body)
            XCTAssertThrowsError(
                try OpenAIAdapter.refuseUnsupportedResponseShape(request), body, file: file, line: line
            ) { error in
                XCTAssertTrue(
                    "\(error)".contains(needle), "the refusal must name \(needle), got: \(error)",
                    file: file, line: line
                )
            }
        }

        try refuseChat(#"{"messages":[],"frequency_penalty":5}"#, naming: "frequency_penalty")
        try refuseChat(#"{"messages":[],"logit_bias":{"50256":-500}}"#, naming: "logit_bias")
        try refuseChat(#"{"messages":[],"logit_bias":{"abc":1}}"#, naming: "logit_bias")
        // A negative key is not a token id either.
        try refuseChat(#"{"messages":[],"logit_bias":{"-4":1}}"#, naming: "logit_bias")

        // The same two rules on the legacy route, through its own refusal.
        let legacyFrequency = try decodeCompletions(#"{"prompt":"x","frequency_penalty":2.5}"#)
        XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedCompletionsShape(legacyFrequency)) { error in
            XCTAssertTrue("\(error)".contains("frequency_penalty"), "got: \(error)")
        }
        let legacyBias = try decodeCompletions(#"{"prompt":"x","logit_bias":{"nope":-1}}"#)
        XCTAssertThrowsError(try OpenAIAdapter.refuseUnsupportedCompletionsShape(legacyBias)) { error in
            XCTAssertTrue("\(error)".contains("logit_bias"), "got: \(error)")
        }

        // A value that is effectively zero is no penalty rather than a refusal, so a
        // client that always sends the field is unaffected.
        let effectivelyZero = try OpenAIAdapter.frequencyPenalty(0.0004)
        XCTAssertEqual(effectivelyZero, 0)
    }

    func testTheLegacyRouteReasonsOnlyWhenTheClientAsks() throws {
        // The opposite default from the chat route on purpose: this protocol has no
        // field to carry reasoning, so a turn that reasons first spends its budget
        // there and returns a trace rather than a continuation.
        let plain = try decodeCompletions(#"{"prompt":"x"}"#)
        XCTAssertFalse(OpenAIAdapter.completionsThinkingRequested(plain))
        let asked = try decodeCompletions(#"{"prompt":"x","enable_thinking":true}"#)
        XCTAssertTrue(OpenAIAdapter.completionsThinkingRequested(asked))
        let declined = try decodeCompletions(#"{"prompt":"x","enable_thinking":false}"#)
        XCTAssertFalse(OpenAIAdapter.completionsThinkingRequested(declined))
    }

    func testTheLegacyPromptIsOneUserTurnWithNoSystemMessage() throws {
        let request = try decodeCompletions(#"{"prompt":"The capital of France is","model":"m"}"#)
        let split = OpenAIAdapter.completionsSplit(request)
        XCTAssertEqual(split.systemPrompt, "", "this protocol has no system field, so the family's own default block is what renders")
        XCTAssertEqual(split.turns.count, 1)
        XCTAssertEqual(split.turns[0].role, .user)
        XCTAssertEqual(split.turns[0].text, "The capital of France is")
        XCTAssertFalse(split.hasImages, "there is no image field on this protocol")
    }

    func testTheCompletionsSessionKeyIsThePromptAndTheModel() {
        let key = OpenAIAdapter.sessionKey(forPrompt: "The capital of France is", model: "m")
        XCTAssertEqual(
            OpenAIAdapter.sessionKey(forPrompt: "The capital of France is", model: "m"),
            key,
            "a stateless client resends the same prompt, so identity has to be stable"
        )
        XCTAssertNotEqual(OpenAIAdapter.sessionKey(forPrompt: "The capital of France is ", model: "m"), key)
        XCTAssertNotEqual(OpenAIAdapter.sessionKey(forPrompt: "The capital of France is", model: "other"), key)
        // One prompt and one chat message with the same text are different
        // conversations, so they must not share a pinned prefix.
        let chat = OpenAIAdapter.sessionKey(
            for: [.init(role: "user", contentText: "The capital of France is")],
            model: "m"
        )
        XCTAssertNotEqual(chat, key)
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
