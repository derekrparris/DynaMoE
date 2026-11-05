//
//  DeepSeekV41ToolDialectTests.swift
//  DynaMoETests
//
//  The tool dialect is measured from the published checkpoint
//  (`DeepSeekV41PublishedTurnTests`), so what these cases pin is the transcription.
//  The checkpoint answers in TWO dialects depending on the prompt it is shown, and
//  both are covered here: the DSML tags it emits for the agent-shaped prompt this
//  app builds (and which the instructions therefore teach), and the plain-text
//  name/arguments tags it emits when it is shown that schematic. Either becomes the
//  canonical call block the shared parser already extracts; a truncated call never
//  executes; generation freezes on the closer rather than on the opener; and the
//  result turn uses the checkpoint's own output pair.
//
//  All pure logic: no model, no device, no snapshot.
//

import XCTest
@testable import DynaMoE

final class DeepSeekV41ToolDialectTests: XCTestCase {

    /// The exact text the checkpoint emitted for a `get_current_time(city:)` prompt
    /// (recorded in `PERF_FINDINGS.md`, "DSV4.1 #27"). The preamble is part of the
    /// turn and has to survive canonicalization: it is what the user sees.
    private let publishedTurn = """
    I'll check the current time in Paris for you.

    <tool_call_name>get_current_time</tool_call_name>
    <tool_call_arguments>{"city": "Paris"}</tool_call_arguments>
    """

    private func calls(_ text: String) -> [ParsedToolCall] {
        let canonical = DeepSeekV41ToolDialect.canonicalized(text)
        return StreamingToolParser.shared
            .parseStreamingToolCalls(from: canonical, format: .qwenXML)
            .calls
    }

    // MARK: - The outbound half

    func testTheCheckpointsOwnCallBecomesAnExecutableCall() {
        let parsed = calls(publishedTurn)
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed.first?.name, "get_current_time")
        XCTAssertEqual(parsed.first?.arguments["city"] as? String, "Paris")
    }

    func testTheCallsPreambleSurvives() {
        let canonical = DeepSeekV41ToolDialect.canonicalized(publishedTurn)
        XCTAssertTrue(canonical.hasPrefix("I'll check the current time in Paris for you."))
    }

    /// Shown no schematic the model emits the JSON form instead, so the parser has
    /// to understand that one too.
    func testTheJSONCallFormBecomesAnExecutableCall() {
        let text = """
        Let me look that up.

        <tool_call>
        {"name": "get_current_time", "arguments": {"city": "Paris"}}
        </tool_call>
        """
        let parsed = calls(text)
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed.first?.name, "get_current_time")
        XCTAssertEqual(parsed.first?.arguments["city"] as? String, "Paris")
    }

    func testACanonicalBlockIsLeftAlone() {
        let text = #"<tool_call><function=shell_run>{"command": "ls"}</function></tool_call>"#
        let parsed = calls(text)
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed.first?.name, "shell_run")
        XCTAssertEqual(parsed.first?.arguments["command"] as? String, "ls")
    }

    /// Two calls in one turn: the model chains name/arguments pairs when it needs
    /// more than one tool, so both must survive.
    func testTwoChainedCallsBecomeTwoCalls() {
        let text = """
        <tool_call_name>file_read</tool_call_name>
        <tool_call_arguments>{"path": "/tmp/a"}</tool_call_arguments>
        <tool_call_name>file_read</tool_call_name>
        <tool_call_arguments>{"path": "/tmp/b"}</tool_call_arguments>
        """
        let parsed = calls(text)
        XCTAssertEqual(parsed.count, 2)
        XCTAssertEqual(parsed.map { $0.arguments["path"] as? String }, ["/tmp/a", "/tmp/b"])
    }

    /// A zero-argument tool is written with no arguments block at all.
    func testANameWithoutArgumentsIsStillACall() {
        let text = "<tool_call_name>list_subagents</tool_call_name>"
        let parsed = calls(text)
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed.first?.name, "list_subagents")
        XCTAssertTrue(parsed.first?.arguments.isEmpty ?? false)
    }

    /// A generation that froze mid-call must not execute a partial name.
    func testAnUnterminatedCallIsNotExecuted() {
        let text = "<tool_call_name>shell_run"
        XCTAssertEqual(calls(text).count, 0)
    }

    func testAnUnterminatedArgumentsBlockIsNotExecuted() {
        let text = """
        <tool_call_name>shell_run</tool_call_name>
        <tool_call_arguments>{"command": "ls
        """
        XCTAssertEqual(calls(text).count, 0)
    }

    /// The vocabulary's own container is not what the checkpoint emits, but the
    /// tokens exist and an API caller can send it.
    func testTheVocabularyContainerIsUnderstood() {
        let m = DeepSeekV41PromptMarkers.familyDefault
        let text = m.toolCallsBegin + m.toolCallBegin + "shell_run" + m.toolCallSeparator
            + #"{"command": "ls -la"}"# + m.toolCallEnd + m.toolCallsEnd
        let parsed = calls(text)
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed.first?.name, "shell_run")
        XCTAssertEqual(parsed.first?.arguments["command"] as? String, "ls -la")
    }

    // MARK: - DSML, the dialect the agent turn receives

    /// The exact text the published agent turn emitted (`PERF_FINDINGS.md`,
    /// "DSV4.1 #27"), tag for tag. Built from the dialect's own tag constants rather
    /// than retyped, because a rendered view strips the fullwidth bars in the DSML
    /// token and the literal would then look right and match nothing.
    private var publishedAgentTurn: String {
        let d = DeepSeekV41ToolDialect.self
        return "\n\n" + d.dsmlCallsOpen + "\n"
            + d.dsmlInvokeTag(name: "shell_run") + "\n"
            + d.dsmlParameterTag(name: "command") + "ls -la /tmp" + d.dsmlParameterClose + "\n"
            + d.dsmlInvokeClose + "\n" + d.dsmlCallsClose
    }

    func testThePublishedAgentTurnBecomesAnExecutableCall() {
        let parsed = calls(publishedAgentTurn)
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed.first?.name, "shell_run")
        XCTAssertEqual(parsed.first?.arguments["command"] as? String, "ls -la /tmp")
    }

    /// The container is machinery, not prose: it must not reach the bubble.
    func testTheCallsContainerIsRemoved() {
        let canonical = DeepSeekV41ToolDialect.canonicalized(publishedAgentTurn)
        XCTAssertFalse(canonical.contains(DeepSeekV41ToolDialect.dsmlCallsOpen))
        XCTAssertFalse(canonical.contains(DeepSeekV41ToolDialect.dsmlCallsClose))
        XCTAssertTrue(canonical.contains("<function=shell_run>"))
    }

    /// The model chains invokes inside ONE container when it needs several tools.
    func testTwoInvokesBecomeTwoCalls() {
        let d = DeepSeekV41ToolDialect.self
        let text = d.dsmlCallsOpen
            + d.dsmlInvokeTag(name: "file_read")
            + d.dsmlParameterTag(name: "path") + "/tmp/a" + d.dsmlParameterClose
            + d.dsmlInvokeClose
            + d.dsmlInvokeTag(name: "file_read")
            + d.dsmlParameterTag(name: "path") + "/tmp/b" + d.dsmlParameterClose
            + d.dsmlInvokeClose
            + d.dsmlCallsClose
        let parsed = calls(text)
        XCTAssertEqual(parsed.count, 2)
        XCTAssertEqual(parsed.map { $0.arguments["path"] as? String }, ["/tmp/a", "/tmp/b"])
    }

    /// A call with no parameters still has to parse: the tag says nothing.
    func testAnInvokeWithoutParametersIsStillACall() {
        let d = DeepSeekV41ToolDialect.self
        let text = d.dsmlCallsOpen + d.dsmlInvokeTag(name: "list_subagents")
            + d.dsmlInvokeClose + d.dsmlCallsClose
        let parsed = calls(text)
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed.first?.name, "list_subagents")
        XCTAssertTrue(parsed.first?.arguments.isEmpty ?? false)
    }

    /// `string="false"` marks a JSON argument, so a numeric limit stays numeric
    /// instead of arriving as the string "40".
    func testANonStringParameterKeepsItsType() {
        let d = DeepSeekV41ToolDialect.self
        let text = d.dsmlCallsOpen + d.dsmlInvokeTag(name: "file_read")
            + d.dsmlParameterTag(name: "path") + "/tmp/a" + d.dsmlParameterClose
            + d.dsmlParameterTag(name: "limit", isString: false) + "40"
            + d.dsmlParameterClose
            + d.dsmlInvokeClose + d.dsmlCallsClose
        let parsed = calls(text)
        XCTAssertEqual(parsed.first?.arguments["limit"] as? Int, 40)
        XCTAssertEqual(parsed.first?.arguments["path"] as? String, "/tmp/a")
    }

    /// A generation that stopped inside a parameter must not execute a half-built
    /// argument list: the tool would run with an argument the model never finished.
    func testAnUnterminatedParameterIsNotExecuted() {
        let d = DeepSeekV41ToolDialect.self
        let text = d.dsmlCallsOpen + d.dsmlInvokeTag(name: "shell_run")
            + d.dsmlParameterTag(name: "command") + "ls -la"
        XCTAssertEqual(calls(text).count, 0)
    }

    func testAnUnterminatedInvokeIsNotExecuted() {
        let d = DeepSeekV41ToolDialect.self
        let text = d.dsmlInvokeTag(name: "shell_run")
            + d.dsmlParameterTag(name: "command") + "ls" + d.dsmlParameterClose
        XCTAssertEqual(calls(text).count, 0)
    }

    func testGenerationFreezesOnTheCallsContainerClose() {
        XCTAssertTrue(DeepSeekV41ToolDialect.shouldFreeze(publishedAgentTurn))
    }

    /// A single bare invoke, with no container, is complete at its own closer.
    func testGenerationFreezesOnABareInvokeClose() {
        let d = DeepSeekV41ToolDialect.self
        let text = d.dsmlInvokeTag(name: "list_subagents") + d.dsmlInvokeClose
        XCTAssertTrue(DeepSeekV41ToolDialect.shouldFreeze(text))
    }

    func testGenerationDoesNotFreezeInsideADSMLCall() {
        let d = DeepSeekV41ToolDialect.self
        let partial = d.dsmlCallsOpen + d.dsmlInvokeTag(name: "shell_run")
            + d.dsmlParameterTag(name: "command") + "ls -la /tmp"
        XCTAssertFalse(DeepSeekV41ToolDialect.shouldFreeze(partial))
    }

    /// The DSML token is one vocabulary id (128825), so its spelling is pinned by
    /// codepoints: U+FF5C, "DSML", U+FF5C, with no word-joining space inside.
    func testTheDSMLTokenIsTheCheckpointsOwnSpelling() {
        XCTAssertEqual(
            Array(DeepSeekV41ToolDialect.dsmlTag.unicodeScalars.map(\.value)),
            [0xFF5C, 0x44, 0x53, 0x4D, 0x4C, 0xFF5C]
        )
        XCTAssertEqual(
            DeepSeekV41ToolDialect.dsmlCallsOpen,
            "<" + DeepSeekV41ToolDialect.dsmlTag + " calls>"
        )
    }

    // MARK: - Freezing

    func testGenerationDoesNotFreezeBeforeTheCallCloses() {
        let partial = """
        I'll check.

        <tool_call_name>get_current_time</tool_call_name>
        <tool_call_arguments>{"city": "Par
        """
        XCTAssertFalse(DeepSeekV41ToolDialect.shouldFreeze(partial))
    }

    func testGenerationFreezesWhenTheArgumentsClose() {
        XCTAssertTrue(DeepSeekV41ToolDialect.shouldFreeze(publishedTurn))
    }

    func testGenerationFreezesOnAJSONCallBlock() {
        let text = #"<tool_call>{"name": "list_subagents", "arguments": {}}</tool_call>"#
        XCTAssertTrue(DeepSeekV41ToolDialect.shouldFreeze(text))
    }

    /// A `</tool_call>` that does not close a JSON call is prose and must not stop
    /// the turn: the Qwen parser's closer shares the spelling.
    func testACloserWithoutACallDoesNotFreeze() {
        XCTAssertFalse(DeepSeekV41ToolDialect.shouldFreeze("A stray </tool_call> in prose."))
    }

    func testGenerationFreezesOnTheVocabularyContainer() {
        let m = DeepSeekV41PromptMarkers.familyDefault
        XCTAssertTrue(DeepSeekV41ToolDialect.shouldFreeze(m.toolCallsEnd))
    }

    // MARK: - The inbound half

    /// The measured result shape: the payload inside the checkpoint's own
    /// `tool▁output` pair, whose spelling is pinned by codepoints because a rendered
    /// view strips the fullwidth bars.
    func testAResultTurnUsesTheCheckpointsOutputPair() {
        let turn = DeepSeekV41ToolDialect.resultTurn(payload: #"{"local_time": "14:05"}"#)
        let m = DeepSeekV41PromptMarkers.familyDefault
        XCTAssertEqual(turn, m.toolOutputBegin + #"{"local_time": "14:05"}"# + m.toolOutputEnd)
        XCTAssertEqual(
            Array(m.toolOutputBegin.unicodeScalars.map(\.value)),
            [0x3C, 0xFF5C, 0x74, 0x6F, 0x6F, 0x6C, 0x2581, 0x6F, 0x75, 0x74,
             0x70, 0x75, 0x74, 0x2581, 0x62, 0x65, 0x67, 0x69, 0x6E, 0xFF5C, 0x3E]
        )
    }

    // MARK: - Instructions

    /// The instruction block has to describe the dialect the model actually emits
    /// for this prompt: the DSML tags. A Qwen `<function=>` example teaches a call it
    /// will not make.
    func testTheInstructionBlockUsesTheModelTags() {
        let block = DeepSeekV41ToolDialect.instructions()
        XCTAssertTrue(block.contains(DeepSeekV41ToolDialect.dsmlCallsOpen))
        XCTAssertTrue(block.contains(DeepSeekV41ToolDialect.dsmlCallsClose))
        XCTAssertTrue(block.contains(DeepSeekV41ToolDialect.dsmlInvokeTag(
            name: "example_function_name"
        )))
        XCTAssertTrue(block.contains(DeepSeekV41ToolDialect.dsmlParameterTag(
            name: "example_parameter_1"
        )))
        XCTAssertFalse(block.contains("<function="))
        XCTAssertFalse(block.contains("<|tool"))
    }

    // MARK: - Markers

    /// Every tool marker is one token in the checkpoint's vocabulary
    /// (`tokenizer.json` added tokens, ids 128806-128814). The spellings are read
    /// from the markers, never retyped, so this pins the parts rather than the whole.
    func testTheFamilyMarkersAreTheCheckpointsSpellings() {
        let m = DeepSeekV41PromptMarkers.familyDefault
        let expected: [(String, [String])] = [
            (m.toolCallsBegin, ["tool", "calls", "begin"]),
            (m.toolCallBegin, ["tool", "call", "begin"]),
            (m.toolCallSeparator, ["tool", "sep"]),
            (m.toolCallEnd, ["tool", "call", "end"]),
            (m.toolCallsEnd, ["tool", "calls", "end"]),
            (m.toolOutputBegin, ["tool", "output", "begin"]),
            (m.toolOutputEnd, ["tool", "output", "end"]),
            (m.toolOutputsBegin, ["tool", "outputs", "begin"]),
            (m.toolOutputsEnd, ["tool", "outputs", "end"])
        ]
        for (marker, parts) in expected {
            XCTAssertEqual(marker, DeepSeekV41PromptMarkers.marker(parts))
        }
    }
}
