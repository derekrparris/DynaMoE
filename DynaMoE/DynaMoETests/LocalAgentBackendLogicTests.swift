//
//  LocalAgentBackendLogicTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for the local subagent backend: prompt rendering, tool
//  mask install/restore, the result mapping and the re-prefill cost refusal.
//  No model is loaded and no turn is launched.
//

import XCTest
@testable import DynaMoE

@MainActor
final class LocalAgentBackendLogicTests: XCTestCase {

    private var savedRegistry: GrammarToolRegistry!

    override func setUp() {
        super.setUp()
        savedRegistry = GrammarConstrainedSampler.shared.snapshotRegistry()
    }

    override func tearDown() {
        GrammarConstrainedSampler.shared.restoreRegistry(savedRegistry)
        PrefixCacheManager.shared.updatePinnedTokens(tokens: [], sessionId: nil)
        PrefillThroughput.resetForTesting()
        super.tearDown()
    }

    // MARK: - Conversation turns

    func testToolResultsBecomeToolResultTurnsNotOrdinaryUserTurns() {
        let turns = LocalAgentBackend.conversationTurns(from: [
            .user("read the file"),
            .assistant(text: "reading", thinking: "", calls: []),
            .toolResult(callId: "call_1", name: "file_read", content: "[file_read] ok", isError: false)
        ])
        XCTAssertEqual(turns.count, 3)
        XCTAssertEqual(turns[0].role, .user)
        XCTAssertEqual(turns[1].role, .assistant)
        XCTAssertEqual(turns[2].role, .user)
        XCTAssertEqual(turns[2].kind, .toolResult(name: "file_read"))
        XCTAssertTrue(turns[2].text.contains("[file_read] ok"))
        XCTAssertTrue(turns[2].text.contains("Function result (file_read)"))
    }

    func testACompletedTurnKeepsItsReasoningBlockClosed() {
        let turns = LocalAgentBackend.conversationTurns(from: [
            .assistant(text: "the answer", thinking: "the reasoning", calls: [])
        ])
        XCTAssertEqual(turns.count, 1)
        XCTAssertTrue(turns[0].text.contains("the reasoning"))
        XCTAssertTrue(turns[0].text.contains(TurnTextSplitter.canonicalReasoningCloseTag))
        XCTAssertTrue(turns[0].text.hasSuffix("the answer"))
    }

    func testACallMissingFromTheAssistantTextIsEchoed() {
        let call = AgentToolCall(id: "call_1", name: "file_read", arguments: ["path": "a"], rawArguments: "{\"path\":\"a\"}")
        let turns = LocalAgentBackend.conversationTurns(from: [
            .assistant(text: "I will read it.", thinking: "", calls: [call])
        ])
        XCTAssertTrue(turns[0].text.contains(ChatPromptFormatter.openAIToolCallEchoPrefix))
        XCTAssertTrue(turns[0].text.contains("file_read"))
        XCTAssertTrue(turns[0].text.contains("arguments"), "the echoed record must carry the call's arguments, not only its name")
    }

    func testACallAlreadyInTheTextIsNotEchoedTwice() {
        let call = AgentToolCall(id: "call_1", name: "file_read", arguments: ["path": "a"])
        let text = "x \(StreamingToolParser.qwenToolCallOpen)<function=file_read></function>\(StreamingToolParser.qwenToolCallClose)"
        let turns = LocalAgentBackend.conversationTurns(from: [
            .assistant(text: text, thinking: "", calls: [call])
        ])
        XCTAssertFalse(
            turns[0].text.contains(ChatPromptFormatter.openAIToolCallEchoPrefix),
            "the model's own call block is already the call; a second record would teach it to call twice"
        )
    }

    func testEveryKnownCallDialectCountsAsMarkup() {
        XCTAssertTrue(LocalAgentBackend.carriesCallMarkup(DeepSeekV41ToolDialect.dsmlTag))
        XCTAssertTrue(LocalAgentBackend.carriesCallMarkup(StreamingToolParser.qwenToolCallOpen))
        XCTAssertTrue(LocalAgentBackend.carriesCallMarkup(StreamingToolParser.gemmaToolCallOpen))
        XCTAssertTrue(LocalAgentBackend.carriesCallMarkup(StreamingToolParser.llamaTagOpen))
        XCTAssertFalse(LocalAgentBackend.carriesCallMarkup("just prose"))
    }

    // MARK: - Prompt rendering

    func testASingleUserTurnRendersWithTheFamilyTemplateAndAnOpenAssistantTurn() {
        let prompt = LocalAgentBackend.renderPrompt(
            systemPrompt: "be careful",
            messages: [.user("hello")],
            thinkingEnabled: false,
            modelSupportsThinking: false,
            modelConfig: nil
        )
        XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\nbe careful<|im_end|>"))
        XCTAssertTrue(prompt.contains("<|im_start|>user\nhello<|im_end|>"))
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n"), "the model answers after the assistant prefix, not after the user turn")
    }

    func testAHistoryIsRenderedThroughTheConversationFormatter() {
        let prompt = LocalAgentBackend.renderPrompt(
            systemPrompt: "sys",
            messages: [
                .user("first"),
                .assistant(text: "reply", thinking: "", calls: []),
                .user("second")
            ],
            thinkingEnabled: false,
            modelSupportsThinking: false,
            modelConfig: nil
        )
        // Both user turns survive as their own boundaries, and the assistant turn
        // renders as a completed turn rather than being flattened into the last
        // user message.
        XCTAssertTrue(prompt.contains("<|im_start|>user\nfirst<|im_end|>"))
        XCTAssertTrue(prompt.contains("<|im_start|>assistant\nreply<|im_end|>"))
        XCTAssertTrue(prompt.contains("<|im_start|>user\nsecond<|im_end|>"))
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    // MARK: - Tool mask

    func testMaskSchemasCarryTheParametersAndRequiredKeys() {
        let tool = ToolDefinition(
            name: "file_read",
            description: "read",
            // Plain JSON values: a nested schema is a plain dictionary, not a
            // nested `AnyCodable` (which has no encoding of its own).
            parameters: [
                "type": AnyCodable("object"),
                "properties": AnyCodable(["path": ["type": "string"]]),
                "required": AnyCodable(["path"])
            ]
        )
        let schemas = LocalAgentBackend.maskSchemas(for: [tool])
        XCTAssertEqual(schemas.count, 1)
        XCTAssertEqual(schemas[0].name, "file_read")
        XCTAssertEqual(schemas[0].parameterKeys, ["path"])
        XCTAssertEqual(schemas[0].requiredKeys, ["path"])
    }

    func testInstallingRunSchemasRestoresTheRegistryAfterwards() async {
        let base = GrammarToolSchema(name: "app_tool", parameterKeys: ["a"], requiredKeys: [])
        GrammarConstrainedSampler.shared.registerToolSchemas([base])
        XCTAssertEqual(GrammarConstrainedSampler.shared.snapshotRegistry().names, ["app_tool"])

        let runSchemas = [GrammarToolSchema(name: "run_tool", parameterKeys: ["b"], requiredKeys: ["b"])]
        let inside = await GrammarConstrainedSampler.withTurnToolMask(runSchemas) {
            GrammarConstrainedSampler.shared.snapshotRegistry()
        }
        XCTAssertEqual(inside.names, ["run_tool"])
        XCTAssertEqual(inside.requiredKeys["run_tool"], ["b"])

        let after = GrammarConstrainedSampler.shared.snapshotRegistry()
        XCTAssertEqual(after.names, ["app_tool"], "a run's vocabulary must not outlive its turn")
        XCTAssertEqual(after.parameterKeys["app_tool"], ["a"])
    }

    func testARunWithNoToolsLeavesTheRegistryAlone() async {
        GrammarConstrainedSampler.shared.registerToolSchemas([GrammarToolSchema(name: "app_tool")])
        let names = await GrammarConstrainedSampler.withTurnToolMask([]) {
            GrammarConstrainedSampler.shared.snapshotRegistry().names
        }
        XCTAssertEqual(names, ["app_tool"])
    }

    // MARK: - Stream mapping

    func testTheSuffixDeltaIsWhatArrivedSinceTheLastFrame() {
        XCTAssertEqual(LocalAgentBackend.suffixDelta(from: "ab", to: "abcd"), "cd")
        XCTAssertEqual(LocalAgentBackend.suffixDelta(from: "", to: "abcd"), "abcd")
        XCTAssertEqual(LocalAgentBackend.suffixDelta(from: "abcd", to: "abcd"), "")
        XCTAssertEqual(LocalAgentBackend.suffixDelta(from: "abcd", to: ""), "")
        XCTAssertEqual(
            LocalAgentBackend.suffixDelta(from: "abc\u{FFFD}", to: "abcé"),
            "abcé",
            "a repaired UTF-8 boundary is re-emitted rather than dropped"
        )
    }

    func testOnlyActionableCallsReachTheLoop() {
        let result = makeResult(
            parsedCalls: [
                ParsedToolCall(name: "file_read", arguments: ["path": "a"], rawArguments: "{\"path\":\"a\"}", rawText: "call"),
                ParsedToolCall(name: "noop", arguments: [:], rawArguments: "{}", rawText: "gesture")
            ],
            actionable: [0],
            promptTokens: 12,
            generated: [7, 8, 9],
            reused: 5
        )
        let turn = LocalAgentBackend.turn(from: result)
        XCTAssertEqual(turn.calls.map(\.name), ["file_read"])
        XCTAssertEqual(turn.calls.first?.id, "call_local_0")
        XCTAssertEqual(turn.usage.promptTokens, 12)
        XCTAssertEqual(turn.usage.completionTokens, 3)
        XCTAssertEqual(turn.usage.cachedTokens, 5)
        XCTAssertEqual(turn.finishReason, TurnFinishReason.toolFreeze.rawValue)
    }

    func testAResultThatWasCutForLengthReportsThatRatherThanSuccess() {
        let result = makeResult(finish: .lengthCapped)
        XCTAssertEqual(LocalAgentBackend.turn(from: result).finishReason, TurnFinishReason.lengthCapped.rawValue)
    }

    // MARK: - Cost estimate and refusal

    func testTheCoordinatorCostIsItsPinnedPrefixOverTheMeasuredRate() {
        let estimate = LocalSubagentCostEstimate.make(
            coordinatorPrefixTokens: 4_000,
            runPromptTokens: 500,
            tokensPerSecond: 8
        )
        XCTAssertEqual(estimate.coordinatorReprefillSeconds, 500, accuracy: 0.001)
        XCTAssertEqual(estimate.runPrefillSeconds, 62.5, accuracy: 0.001)
        XCTAssertTrue(estimate.exceedsLimit)
        XCTAssertFalse(estimate.admits(operatorOverride: false))
        XCTAssertTrue(estimate.admits(operatorOverride: true))
    }

    func testARunInsideTheLimitIsAdmittedAndSaysWhy() {
        let estimate = LocalSubagentCostEstimate.make(
            coordinatorPrefixTokens: 100,
            runPromptTokens: 50,
            tokensPerSecond: 10,
            limitSeconds: 120
        )
        XCTAssertFalse(estimate.exceedsLimit)
        XCTAssertNil(estimate.refusalMessage)
        XCTAssertTrue(estimate.admits(operatorOverride: false))
        XCTAssertTrue(estimate.summary.contains("100 tokens"))
    }

    func testTheRefusalNamesTheTokensTheSecondsAndTheLimit() throws {
        let estimate = LocalSubagentCostEstimate.make(
            coordinatorPrefixTokens: 9_000,
            runPromptTokens: 0,
            tokensPerSecond: 5,
            limitSeconds: 120
        )
        let message = try XCTUnwrap(estimate.refusalMessage)
        XCTAssertTrue(message.contains("9000 tokens"))
        XCTAssertTrue(message.contains("1800 s"))
        XCTAssertTrue(message.contains("120 s"))
        XCTAssertTrue(message.contains("cloud backend") || message.contains("raise the limit"))
    }

    func testAnUnmeasuredProcessPricesWithThePessimisticFloor() {
        PrefillThroughput.resetForTesting()
        XCTAssertEqual(PrefillThroughput.resolvedTokensPerSecond(), PrefillThroughput.fallbackTokensPerSecond)
        PrefillThroughput.note(tokensPerSecond: 42)
        XCTAssertEqual(PrefillThroughput.resolvedTokensPerSecond(), 42)
        PrefillThroughput.note(tokensPerSecond: 0)
        XCTAssertEqual(PrefillThroughput.resolvedTokensPerSecond(), 42, "a zero rate is not a measurement")
    }

    func testANonPositiveRateIsTreatedAsOneTokenPerSecondNotAsInfinity() {
        let estimate = LocalSubagentCostEstimate.make(
            coordinatorPrefixTokens: 10,
            runPromptTokens: 0,
            tokensPerSecond: -4
        )
        XCTAssertEqual(estimate.tokensPerSecond, 1)
        XCTAssertEqual(estimate.coordinatorReprefillSeconds, 10, accuracy: 0.001)
    }

    func testTheLimitComesFromDefaultsAndFallsBackWhenUnset() {
        let suiteName = "LocalAgentBackendLogicTests-\(UUID().uuidString)"
        let suite = UserDefaults(suiteName: suiteName)!
        defer { suite.removePersistentDomain(forName: suiteName) }
        XCTAssertEqual(LocalSubagentCostEstimate.configuredLimitSeconds(defaults: suite), 120)
        suite.set(30.0, forKey: LocalSubagentCostEstimate.limitDefaultsKey)
        XCTAssertEqual(LocalSubagentCostEstimate.configuredLimitSeconds(defaults: suite), 30)
        suite.set(0.0, forKey: LocalSubagentCostEstimate.limitDefaultsKey)
        XCTAssertEqual(LocalSubagentCostEstimate.configuredLimitSeconds(defaults: suite), 120)
    }

    func testOnlyAPrefixOwnedByAnotherSessionIsPricedAsDisplaced() {
        let runSession = UUID()
        let coordinator = UUID()
        PrefixCacheManager.shared.updatePinnedTokens(tokens: Array(repeating: 1, count: 300), sessionId: coordinator)

        let displaced = LocalAgentBackend.costEstimate(
            runSessionId: runSession,
            promptTokenEstimate: 10,
            tokensPerSecond: 10,
            limitSeconds: 120
        )
        XCTAssertEqual(displaced.coordinatorPrefixTokens, 300)

        let own = LocalAgentBackend.costEstimate(
            runSessionId: coordinator,
            promptTokenEstimate: 10,
            tokensPerSecond: 10,
            limitSeconds: 120
        )
        XCTAssertEqual(own.coordinatorPrefixTokens, 0, "a resumed run already owns the pin")
    }

    func testALocalRunIsRefusedWhenNoModelIsLoaded() {
        let estimate = LocalSubagentCostEstimate.make(coordinatorPrefixTokens: 0, runPromptTokens: 0, tokensPerSecond: 10)
        XCTAssertThrowsError(try LocalAgentBackend.admission(hasLoadedModel: false, estimate: estimate, operatorOverride: true)) { error in
            let message = (error as? LocalizedError)?.errorDescription ?? ""
            XCTAssertTrue(message.contains("loaded model"), "an unloadable backend is named, not a run that fails on its first step")
        }
    }

    func testAnOverLimitRunIsRefusedUnlessTheOperatorOverridesIt() {
        let estimate = LocalSubagentCostEstimate.make(
            coordinatorPrefixTokens: 9_000,
            runPromptTokens: 0,
            tokensPerSecond: 5,
            limitSeconds: 120
        )
        XCTAssertThrowsError(
            try LocalAgentBackend.admission(hasLoadedModel: true, estimate: estimate, operatorOverride: false)
        ) { error in
            guard let backendError = error as? AgentBackendError,
                  case .refused(let message) = backendError else {
                return XCTFail("expected a refusal, got \(error)")
            }
            XCTAssertTrue(message.contains("9000 tokens"))
            XCTAssertTrue(message.contains("re-prefill"))
        }
        XCTAssertNoThrow(
            try LocalAgentBackend.admission(hasLoadedModel: true, estimate: estimate, operatorOverride: true)
        )
    }

    // MARK: - Fixtures

    private func makeResult(
        finish: TurnFinishReason = .toolFreeze,
        parsedCalls: [ParsedToolCall] = [],
        actionable: [Int] = [],
        promptTokens: Int = 0,
        generated: [UInt32] = [],
        reused: Int = 0
    ) -> TurnResult {
        TurnResult(
            finishReason: finish,
            formattedPrompt: "prompt",
            promptTokenIds: Array(repeating: 1, count: promptTokens),
            generatedTokenIds: generated,
            rawDecodedText: "raw",
            thinkingText: "",
            responseText: "answer",
            tokensGenerated: generated.count,
            elapsedMs: 1,
            tokensPerSecond: 1,
            timeToFirstTokenSeconds: nil,
            thinkingDurationSeconds: nil,
            jetSpecActive: false,
            jetSpecMeanTau: 0,
            jetSpecDraftAccepted: 0,
            jetSpecDraftProposed: 0,
            residentMemoryGB: 0,
            residentExpertCount: 0,
            totalExpertCount: 0,
            cacheHitRatePercent: 0,
            prefetchEfficiencyPercent: 0,
            pagingLatencyMs: 0,
            prefixTokensReused: reused,
            parsedToolCalls: parsedCalls,
            actionableCallIndices: actionable,
            gestureSkipNotices: [:],
            brokenFragments: []
        )
    }
}
