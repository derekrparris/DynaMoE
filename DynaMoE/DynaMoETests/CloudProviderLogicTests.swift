//
//  CloudProviderLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the connected cloud provider layer: sentinel
//  addressing, model registration, SSE chunk parsing, request assembly,
//  provider failure messaging, and Keychain-backed secret storage.
//

import XCTest
@testable import DynaMoE

// MARK: - Addressing

final class CloudModelAddressTests: XCTestCase {

    func testModelIdRoundTrip() throws {
        let id = CloudModelAddress.modelId(providerId: "synthetic", providerModelId: "hf:deepseek-ai/DeepSeek-V4.1-Flash")
        XCTAssertEqual(id, "cloud-model-synthetic-hf:deepseek-ai/DeepSeek-V4.1-Flash")

        let parsed = try XCTUnwrap(CloudModelAddress.parseModelId(id))
        XCTAssertEqual(parsed.provider.id, "synthetic")
        XCTAssertEqual(parsed.providerModelId, "hf:deepseek-ai/DeepSeek-V4.1-Flash")
    }

    func testSentinelPathRoundTrip() throws {
        let path = CloudModelAddress.sentinelPath(providerId: "synthetic", providerModelId: "hf:moonshotai/Kimi-K2-Thinking")
        XCTAssertEqual(path, "cloud://synthetic/hf:moonshotai/Kimi-K2-Thinking")

        let parsed = try XCTUnwrap(CloudModelAddress.parsePath(path))
        XCTAssertEqual(parsed.provider.id, "synthetic")
        XCTAssertEqual(parsed.providerModelId, "hf:moonshotai/Kimi-K2-Thinking")
    }

    func testUnknownProviderDoesNotResolve() {
        XCTAssertNil(CloudModelAddress.parseModelId("cloud-model-unknown-hf:org/model"))
        XCTAssertNil(CloudModelAddress.parsePath("cloud://unknown/hf:org/model"))
        XCTAssertNil(CloudModelAddress.parsePath("/Users/me/models/local"))
    }

    func testEmptyUpstreamIdDoesNotResolve() {
        XCTAssertNil(CloudModelAddress.parseModelId("cloud-model-synthetic-"))
        XCTAssertNil(CloudModelAddress.parsePath("cloud://synthetic/"))
    }

    func testFlags() {
        XCTAssertTrue(CloudModelAddress.isCloudModelId("cloud-model-synthetic-hf:org/model"))
        XCTAssertTrue(CloudModelAddress.isCloudPath("cloud://synthetic/hf:org/model"))
        XCTAssertFalse(CloudModelAddress.isCloudModelId("apple-foundation-model-ondevice"))
        XCTAssertFalse(CloudModelAddress.isCloudPath(nil))
    }

    func testUnresolvedCloudSelectionIsDistinguishedFromLocalAndResolvedCloud() {
        // A prefixed id/path that no longer resolves must refuse the send
        // instead of falling through to local weights.
        XCTAssertTrue(CloudModelAddress.selectionIsUnresolved(
            modelId: "cloud-model-unknown-hf:org/model",
            modelPath: nil
        ))
        XCTAssertTrue(CloudModelAddress.selectionIsUnresolved(
            modelId: nil,
            modelPath: "cloud://unknown/hf:org/model"
        ))
        // Resolved cloud selections and ordinary local selections are fine.
        XCTAssertFalse(CloudModelAddress.selectionIsUnresolved(
            modelId: "cloud-model-synthetic-hf:org/model",
            modelPath: nil
        ))
        XCTAssertFalse(CloudModelAddress.selectionIsUnresolved(
            modelId: nil,
            modelPath: "/Users/me/models/local"
        ))
        XCTAssertFalse(CloudModelAddress.selectionIsUnresolved(
            modelId: nil,
            modelPath: "apple-foundation-model://on-device"
        ))
    }

    func testDiscoveredModelProbes() {
        let entry = CloudModelRegistry.makeDiscoveredModel(
            provider: CloudProviderRegistry.synthetic,
            providerModelId: "hf:deepseek-ai/DeepSeek-V4.1-Flash"
        )
        XCTAssertTrue(entry.isCloudModel)
        XCTAssertFalse(entry.isAppleFoundationModel)
        XCTAssertEqual(entry.cloudResolution?.provider.id, "synthetic")
        XCTAssertEqual(entry.cloudResolution?.providerModelId, "hf:deepseek-ai/DeepSeek-V4.1-Flash")
        XCTAssertEqual(entry.snapshotPath, "cloud://synthetic/hf:deepseek-ai/DeepSeek-V4.1-Flash")
        XCTAssertEqual(entry.sizeBytes, 0)
        XCTAssertFalse(entry.hasTokenizer)
    }
}

// MARK: - Registration

final class CloudModelRegistryTests: XCTestCase {
    private var defaults: UserDefaults!
    private let suiteName = "dynamoe.cloud.tests.\(UUID().uuidString)"

    override func setUp() {
        super.setUp()
        defaults = UserDefaults(suiteName: suiteName)
        defaults.removePersistentDomain(forName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        super.tearDown()
    }

    func testParsesSemicolonCommaAndNewlineSeparatedIds() {
        defaults.set(
            "hf:org/a ; hf:org/b, hf:org/c\nhf:org/d",
            forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic")
        )
        let specs = CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults)
        XCTAssertEqual(specs.map(\.providerModelId), ["hf:org/a", "hf:org/b", "hf:org/c", "hf:org/d"])
        XCTAssertTrue(specs.allSatisfy { $0.providerId == "synthetic" })
    }

    func testDeduplicatesRepeatedIds() {
        defaults.set("hf:org/a; hf:org/a", forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic"))
        XCTAssertEqual(CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults).count, 1)
    }

    func testDisplayNameCarriesProviderAndModelTail() {
        defaults.set("hf:deepseek-ai/DeepSeek-V4.1-Flash", forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic"))
        let spec = try? XCTUnwrap(CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults).first)
        XCTAssertEqual(spec?.displayName, "DeepSeek-V4.1-Flash (Synthetic)")
    }

    func testThinkingHintInfersCapability() {
        defaults.set("hf:moonshotai/Kimi-K2-Thinking; hf:org/plain", forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic"))
        let specs = CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults)
        XCTAssertEqual(specs.first?.supportsThinking, true)
        XCTAssertEqual(specs.last?.supportsThinking, false)
    }

    func testDebugKeyRegistersOneModelWhenListIsEmpty() {
        defaults.set("hf:org/debug-model", forKey: CloudModelRegistry.debugModelDefaultsKey)
        let specs = CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults)
        XCTAssertEqual(specs.map(\.providerModelId), ["hf:org/debug-model"])
    }

    func testEmptyDefaultsRegisterNothing() {
        XCTAssertTrue(CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults).isEmpty)
        XCTAssertTrue(CloudModelRegistry.makeDiscoveredModels(defaults: defaults).isEmpty)
    }

    func testDiscoveredModelsUseRegistryAddressing() {
        defaults.set("hf:org/a", forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic"))
        let models = CloudModelRegistry.makeDiscoveredModels(
            defaults: defaults,
            hasCredential: { _ in true }
        )
        XCTAssertEqual(models.count, 1)
        XCTAssertEqual(models.first?.id, "cloud-model-synthetic-hf:org/a")
        XCTAssertEqual(models.first?.formattedSize, "Cloud")
        XCTAssertEqual(models.first?.architectureName, "CloudRemoteModel")
    }

    func testDiscoveredModelsAreHiddenWithoutAStoredCredential() {
        defaults.set("hf:org/a", forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic"))
        XCTAssertTrue(CloudModelRegistry.makeDiscoveredModels(
            defaults: defaults,
            hasCredential: { _ in false }
        ).isEmpty)
        XCTAssertEqual(CloudModelRegistry.makeDiscoveredModels(
            defaults: defaults,
            hasCredential: { $0 == "synthetic" }
        ).count, 1)
    }

    func testSetRegisteredModelIdsReplacesAndDeduplicates() {
        CloudModelRegistry.setRegisteredModelIds(
            ["hf:org/a", "hf:org/b", "hf:org/a", "  "],
            for: CloudProviderRegistry.synthetic,
            defaults: defaults
        )
        XCTAssertEqual(
            CloudModelRegistry.registeredModelIds(for: CloudProviderRegistry.synthetic, defaults: defaults),
            ["hf:org/a", "hf:org/b"]
        )
        XCTAssertEqual(
            CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults).count,
            2
        )

        CloudModelRegistry.setRegisteredModelIds([], for: CloudProviderRegistry.synthetic, defaults: defaults)
        XCTAssertTrue(
            CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults).isEmpty
        )
    }

    func testModelsSupportToolsByDefault() {
        defaults.set("hf:org/a", forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic"))
        let spec = CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults).first
        XCTAssertEqual(spec?.supportsTools, true)
    }

    func testNoToolsKeyMarksOnlyTheListedModel() {
        defaults.set(
            "hf:org/a; hf:org/b",
            forKey: CloudModelRegistry.modelsDefaultsKey(for: "synthetic")
        )
        defaults.set(
            "hf:org/b",
            forKey: CloudModelRegistry.noToolsDefaultsKey(for: "synthetic")
        )
        let specs = CloudModelRegistry.registeredSpecs(for: CloudProviderRegistry.synthetic, defaults: defaults)
        XCTAssertEqual(specs.map(\.providerModelId), ["hf:org/a", "hf:org/b"])
        XCTAssertEqual(specs.map(\.supportsTools), [true, false])
        XCTAssertEqual(
            CloudModelRegistry.modelIdsWithoutToolSupport(for: CloudProviderRegistry.synthetic, defaults: defaults),
            ["hf:org/b"]
        )
    }

    func testAliasDisplayNameIsHumanized() {
        XCTAssertEqual(
            CloudModelRegistry.defaultDisplayName(
                provider: CloudProviderRegistry.synthetic,
                providerModelId: "syn:large:text"
            ),
            "Syn Large Text (Synthetic)"
        )
        XCTAssertEqual(
            CloudModelRegistry.defaultDisplayName(
                provider: CloudProviderRegistry.synthetic,
                providerModelId: "hf:deepseek-ai/DeepSeek-V4.1-Flash-1017"
            ),
            "DeepSeek-V4.1-Flash-1017 (Synthetic)"
        )
    }

    func testSyncFilterKeepsChatModelsAndDropsCatalogNoise() {
        let ids = [
            "hf:org/chat-model",
            "text-embedding-3-large",
            "bge-reranker-v2",
            "whisper-large-v3",
            "black-forest-labs/flux-dev",
            "some-image-generation-model",
            "speech-to-text-fast",
            "gpt-4o-mini-moderation",
        ]
        XCTAssertEqual(CloudModelRegistry.chatModelIds(from: ids), ["hf:org/chat-model"])
    }

    func testSyncFilterFallsBackWhenNoEntryLooksChatty() {
        let ids = ["hf:org/one", "hf:org/two"]
        XCTAssertEqual(CloudModelRegistry.chatModelIds(from: ids), ids)
        XCTAssertEqual(CloudModelRegistry.chatModelIds(from: []), [])
    }
}

// MARK: - Stream Parsing

final class CloudStreamParserTests: XCTestCase {

    private func contentChunk(_ text: String) -> String {
        #"{"choices":[{"index":0,"delta":{"content":"\#(text)"},"finish_reason":null}]}"#
    }

    private func reasoningChunk(_ text: String) -> String {
        #"{"choices":[{"index":0,"delta":{"reasoning_content":"\#(text)"},"finish_reason":null}]}"#
    }

    func testContentDeltasAccumulate() throws {
        var parser = CloudStreamParser()
        let first = try parser.consume(dataPayload: contentChunk("Hel"))
        let second = try parser.consume(dataPayload: contentChunk("lo"))
        XCTAssertEqual(first, [.contentDelta("Hel")])
        XCTAssertEqual(second, [.contentDelta("lo")])
        XCTAssertEqual(parser.result().text, "Hello")
        XCTAssertEqual(parser.result().thinking, "")
        XCTAssertEqual(parser.decodedChunkCount, 2)
    }

    func testReasoningContentMapsToThinkingChannel() throws {
        var parser = CloudStreamParser()
        let events = try parser.consume(dataPayload: reasoningChunk("considering"))
        XCTAssertEqual(events, [.thinkingDelta("considering")])
        XCTAssertEqual(parser.result().thinking, "considering")
        XCTAssertEqual(parser.result().text, "")
    }

    func testReasoningAliasIsAccepted() throws {
        var parser = CloudStreamParser()
        let payload = #"{"choices":[{"delta":{"reasoning":"plan"},"finish_reason":null}]}"#
        let events = try parser.consume(dataPayload: payload)
        XCTAssertEqual(events, [.thinkingDelta("plan")])
    }

    func testFinishReasonAndUsageAreCaptured() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: contentChunk("done"))
        let payload = #"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#
        _ = try parser.consume(dataPayload: payload)
        let result = parser.result()
        XCTAssertEqual(result.finishReason, "stop")
        XCTAssertEqual(result.usage?.promptTokens, 11)
        XCTAssertEqual(result.usage?.completionTokens, 7)
        XCTAssertEqual(result.usage?.totalTokens, 18)
    }

    func testUsageOnlyChunkDoesNotProduceDeltas() throws {
        var parser = CloudStreamParser()
        let payload = #"{"choices":[],"usage":{"completion_tokens":3}}"#
        XCTAssertEqual(try parser.consume(dataPayload: payload), [])
        XCTAssertEqual(parser.result().usage?.completionTokens, 3)
    }

    func testUndecodableChunkThrowsTypedFailure() {
        var parser = CloudStreamParser()
        XCTAssertThrowsError(try parser.consume(dataPayload: "not json")) { error in
            guard case CloudTurnFailure.invalidResponse = error else {
                return XCTFail("expected invalidResponse, got \(error)")
            }
        }
    }

    func testUnknownEnvelopeDoesNotCountAsContent() throws {
        var parser = CloudStreamParser()
        XCTAssertEqual(try parser.consume(dataPayload: #"{"id":"x","object":"unexpected"}"#), [])
        XCTAssertEqual(parser.decodedChunkCount, 0)
        XCTAssertEqual(parser.result().text, "")
    }

    func testRoleOnlyChunkDoesNotCountAsContent() throws {
        var parser = CloudStreamParser()
        XCTAssertEqual(try parser.consume(dataPayload: #"{"choices":[{"delta":{"role":"assistant"}}]}"#), [])
        XCTAssertEqual(parser.decodedChunkCount, 0)
    }

    func testErrorPayloadIsReportedInsteadOfBecomingEmptySuccess() {
        var parser = CloudStreamParser()
        XCTAssertThrowsError(try parser.consume(
            dataPayload: #"{"error":{"message":"model not found"}}"#
        )) { error in
            guard case CloudTurnFailure.invalidResponse(let message) = error else {
                return XCTFail("expected invalidResponse, got \(error)")
            }
            XCTAssertTrue(message.contains("model not found"))
        }
        XCTAssertEqual(parser.decodedChunkCount, 0)
    }
}

// MARK: - Streamed Tool Call Assembly

final class CloudToolCallFragmentTests: XCTestCase {

    private func callFragment(
        index: Int?,
        id: String? = nil,
        name: String? = nil,
        arguments: String? = nil
    ) -> String {
        var function: [String: Any] = [:]
        if let name { function["name"] = name }
        if let arguments { function["arguments"] = arguments }
        var fragment: [String: Any] = ["function": function]
        if let index { fragment["index"] = index }
        if let id { fragment["id"] = id }
        let chunk: [String: Any] = [
            "choices": [
                [
                    "index": 0,
                    "delta": ["tool_calls": [fragment]],
                    "finish_reason": NSNull()
                ]
            ]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: chunk)) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    func testSplitArgumentDeltasAreConcatenated() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(
            index: 0, id: "call_1", name: "shell_run", arguments: #"{"comm"#
        ))
        _ = try parser.consume(dataPayload: callFragment(index: 0, arguments: #"and":"ls"}"#))

        let calls = parser.result().toolCalls
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.id, "call_1")
        XCTAssertEqual(calls.first?.name, "shell_run")
        XCTAssertEqual(calls.first?.argumentsJSON, #"{"command":"ls"}"#)
        XCTAssertEqual(calls.first?.decodedArguments?["command"] as? String, "ls")
    }

    func testParallelCallsAreKeptInFirstSeenOrder() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(index: 0, id: "call_a", name: "file_read", arguments: #"{"path":"#))
        _ = try parser.consume(dataPayload: callFragment(index: 1, id: "call_b", name: "grep_search", arguments: #"{"pattern":"#))
        _ = try parser.consume(dataPayload: callFragment(index: 1, arguments: #""foo"}"#))
        _ = try parser.consume(dataPayload: callFragment(index: 0, arguments: #""/tmp/a"}"#))

        let calls = parser.result().toolCalls
        XCTAssertEqual(calls.map(\.id), ["call_a", "call_b"])
        XCTAssertEqual(calls[0].argumentsJSON, #"{"path":"/tmp/a"}"#)
        XCTAssertEqual(calls[1].argumentsJSON, #"{"pattern":"foo"}"#)
    }

    func testSplitFunctionNameIsReassembled() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(index: 0, id: "call_1", name: "file_", arguments: ""))
        _ = try parser.consume(dataPayload: callFragment(index: 0, name: "read"))
        XCTAssertEqual(parser.result().toolCalls.first?.name, "file_read")
    }

    func testRepeatedFullFunctionNameIsNotAppendedTwice() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(index: 0, id: "call_1", name: "file_read", arguments: "{"))
        _ = try parser.consume(dataPayload: callFragment(index: 0, name: "file_read", arguments: #""path":"/tmp/a"}"#))
        XCTAssertEqual(parser.result().toolCalls.first?.name, "file_read")
        XCTAssertEqual(parser.result().toolCalls.first?.argumentsJSON, #"{"path":"/tmp/a"}"#)
    }

    func testMissingIdIsSynthesizedPerIndex() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(index: 0, name: "shell_run", arguments: "{}"))
        _ = try parser.consume(dataPayload: callFragment(index: 1, name: "complete", arguments: "{}"))
        let calls = parser.result().toolCalls
        XCTAssertEqual(calls.map(\.id), [
            CloudToolCall.synthesizedId(index: 0),
            CloudToolCall.synthesizedId(index: 1)
        ])
        XCTAssertNotEqual(calls[0].id, calls[1].id)
    }

    func testEmptyArgumentsMeanNoParameters() {
        let call = CloudToolCall(id: "call_1", name: "git_status", argumentsJSON: "")
        XCTAssertEqual(call.decodedArguments?.count, 0)
    }

    func testMalformedArgumentsAreRejectedRatherThanGuessed() {
        let call = CloudToolCall(id: "call_1", name: "shell_run", argumentsJSON: #"{"command":"#)
        XCTAssertNil(call.decodedArguments)
    }

    func testFinishReasonToolCallsIsCaptured() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(index: 0, id: "call_1", name: "complete", arguments: "{}"))
        _ = try parser.consume(dataPayload: #"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#)
        XCTAssertEqual(parser.result().finishReason, "tool_calls")
        XCTAssertEqual(parser.result().toolCalls.count, 1)
    }

    func testTextAndToolCallsCoexist() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: contentChunk("Checking the file…"))
        _ = try parser.consume(dataPayload: callFragment(index: 0, id: "call_1", name: "file_read", arguments: #"{"path":"/tmp/a"}"#))
        let result = parser.result()
        XCTAssertEqual(result.text, "Checking the file…")
        XCTAssertEqual(result.toolCalls.count, 1)
    }

    private func contentChunk(_ text: String) -> String {
        #"{"choices":[{"index":0,"delta":{"content":"\#(text)"},"finish_reason":null}]}"#
    }
}

// MARK: - Agent Loop Termination

final class CloudAgentLoopPolicyTests: XCTestCase {

    func testContinuesWhileStepsRemain() {
        XCTAssertTrue(CloudAgentLoopPolicy.shouldContinue(
            stepCallCount: 1, step: 0, maxSteps: 3,
            ranComplete: false, failureLoop: false, cancelled: false
        ))
    }

    func testStopsAtStepBudget() {
        XCTAssertFalse(CloudAgentLoopPolicy.shouldContinue(
            stepCallCount: 1, step: 2, maxSteps: 3,
            ranComplete: false, failureLoop: false, cancelled: false
        ))
    }

    func testStopsWithoutToolCalls() {
        XCTAssertFalse(CloudAgentLoopPolicy.shouldContinue(
            stepCallCount: 0, step: 0, maxSteps: 5,
            ranComplete: false, failureLoop: false, cancelled: false
        ))
    }

    func testStopsOnCompleteFailureLoopOrCancellation() {
        for (ranComplete, failureLoop, cancelled) in [(true, false, false), (false, true, false), (false, false, true)] {
            XCTAssertFalse(CloudAgentLoopPolicy.shouldContinue(
                stepCallCount: 1, step: 0, maxSteps: 5,
                ranComplete: ranComplete, failureLoop: failureLoop, cancelled: cancelled
            ))
        }
    }

    func testApprovalGateAppliesToStateChangingCloudCalls() {
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "shell_run"))
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "file_write"))
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "file_edit"))
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "git_commit"))
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "mcp__server__anything"))
        XCTAssertFalse(AgentHarness.isStateChanging(toolName: "file_read"))
        XCTAssertFalse(AgentHarness.isStateChanging(toolName: "grep_search"))
    }
}

// MARK: - Step Planning & Forced Synthesis

final class CloudAgentStepPlanTests: XCTestCase {

    func testSynthesisTurnDoesNotOfferTools() {
        XCTAssertTrue(CloudAgentStepPlan.offersTools(agentTurn: true, isSynthesisTurn: false))
        XCTAssertFalse(CloudAgentStepPlan.offersTools(agentTurn: true, isSynthesisTurn: true))
        XCTAssertFalse(CloudAgentStepPlan.offersTools(agentTurn: false, isSynthesisTurn: false))
    }

    func testStreamedCallsAreIgnoredWhenTheRequestOfferedNoTools() {
        // The provider can emit tool_calls on a tools-disabled turn; executing
        // them would re-enter the loop the synthesis pass was meant to end.
        XCTAssertTrue(CloudAgentStepPlan.shouldExecuteToolCalls(offeredToolCount: 3, callCount: 1))
        XCTAssertFalse(CloudAgentStepPlan.shouldExecuteToolCalls(offeredToolCount: 0, callCount: 2))
        XCTAssertFalse(CloudAgentStepPlan.shouldExecuteToolCalls(offeredToolCount: 1, callCount: 0))
    }

    func testForcedSynthesisTripsOnFailureLoopOrBudgetOnly() {
        XCTAssertTrue(CloudAgentLoopPolicy.shouldForceSynthesis(
            failureLoop: true, budgetExhausted: false, ranComplete: false, alreadyForced: false
        ))
        XCTAssertTrue(CloudAgentLoopPolicy.shouldForceSynthesis(
            failureLoop: false, budgetExhausted: true, ranComplete: false, alreadyForced: false
        ))
        XCTAssertFalse(CloudAgentLoopPolicy.shouldForceSynthesis(
            failureLoop: false, budgetExhausted: false, ranComplete: false, alreadyForced: false
        ))
    }

    func testForcedSynthesisIsSpentAtMostOnceAndNeverAfterComplete() {
        XCTAssertFalse(CloudAgentLoopPolicy.shouldForceSynthesis(
            failureLoop: true, budgetExhausted: true, ranComplete: false, alreadyForced: true
        ))
        XCTAssertFalse(CloudAgentLoopPolicy.shouldForceSynthesis(
            failureLoop: true, budgetExhausted: true, ranComplete: true, alreadyForced: false
        ))
    }

    func testSynthesisDirectiveForbidsFurtherCalls() {
        XCTAssertTrue(CloudAgentSynthesis.directive.contains("Do not emit any tool calls."))
        XCTAssertTrue(CloudAgentSynthesis.directive.hasPrefix("<system>"))
        XCTAssertTrue(CloudAgentSynthesis.directive.hasSuffix("</system>"))
    }
}

// MARK: - SSE Framing

final class CloudSSEFramingTests: XCTestCase {

    /// Feeds a chunked byte stream through the shared MCP framing utilities the
    /// transport uses, exactly as the client does, and returns the assembled
    /// data payloads.
    private func payloads(from chunks: [String]) -> [String] {
        var buffer = MCPLineBuffer(limit: 1_048_576)
        var frames = MCPSSEFrameParser(limit: 1_048_576)
        var payloads: [String] = []
        for chunk in chunks {
            for line in buffer.append(Data(chunk.utf8)) {
                if let frame = frames.consume(line: line) { payloads.append(frame.data) }
            }
        }
        if let trailing = buffer.flush(), let frame = frames.consume(line: trailing) {
            payloads.append(frame.data)
        }
        if let frame = frames.finish() { payloads.append(frame.data) }
        return payloads
    }

    func testFramesAreDispatchedOnBlankLine() {
        let result = payloads(from: ["data: {\"a\":1}\n\n", "data: {\"b\":2}\n\n"])
        XCTAssertEqual(result, [#"{"a":1}"#, #"{"b":2}"#])
    }

    func testSplitChunksReassembleOneFrame() {
        let result = payloads(from: ["data: {\"a\"", ":1}\n", "\n"])
        XCTAssertEqual(result, [#"{"a":1}"#])
    }

    func testDoneSentinelArrivesAsItsOwnPayload() {
        let result = payloads(from: ["data: [DONE]\n\n"])
        XCTAssertEqual(result, ["[DONE]"])
    }
}

// MARK: - Request Assembly

final class CloudChatRequestTests: XCTestCase {

    func testRequestCarriesBearerAuthAndStreamOptions() throws {
        let service = OpenAICloudChatService(
            chatURL: CloudProviderRegistry.synthetic.chatCompletionsURL,
            modelsURL: CloudProviderRegistry.synthetic.modelsURL,
            apiKey: "sk-test-key",
            requestTimeout: 5
        )
        let request = try service.makeRequest(
            model: "hf:deepseek-ai/DeepSeek-V4.1-Flash",
            messages: [
                CloudChatMessage(role: "system", content: "You are helpful."),
                CloudChatMessage(role: "user", content: "Hello")
            ],
            params: CloudChatGenerationParams(temperature: 0.6, topP: 0.9, maxTokens: 128)
        )

        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.url?.absoluteString, "https://api.synthetic.new/openai/v1/chat/completions")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer sk-test-key")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "text/event-stream")

        let body = try XCTUnwrap(request.httpBody)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertEqual(json["model"] as? String, "hf:deepseek-ai/DeepSeek-V4.1-Flash")
        XCTAssertEqual(json["stream"] as? Bool, true)
        XCTAssertEqual(json["temperature"] as? Double, 0.6)
        XCTAssertEqual(json["top_p"] as? Double, 0.9)
        XCTAssertEqual(json["max_tokens"] as? Int, 128)

        let streamOptions = try XCTUnwrap(json["stream_options"] as? [String: Any])
        XCTAssertEqual(streamOptions["include_usage"] as? Bool, true)

        let messages = try XCTUnwrap(json["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.map { $0["role"] as? String }, ["system", "user"])
        XCTAssertEqual(messages.map { $0["content"] as? String }, ["You are helpful.", "Hello"])
    }

    func testToolSchemasRideTheRequestBodyWithAutoChoice() throws {
        let service = OpenAICloudChatService(
            chatURL: CloudProviderRegistry.synthetic.chatCompletionsURL,
            modelsURL: CloudProviderRegistry.synthetic.modelsURL,
            apiKey: "sk-test-key",
            requestTimeout: 5
        )
        let tools = [
            ToolDefinition(
                name: "file_read",
                description: "Read a file",
                parameters: [
                    "type": AnyCodable("object"),
                    "properties": AnyCodable([
                        "path": [
                            "type": "string",
                            "description": "Absolute path to read."
                        ]
                    ]),
                    "required": AnyCodable(["path"])
                ]
            )
        ]
        let request = try service.makeRequest(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "read it")],
            params: CloudChatGenerationParams(),
            tools: tools
        )

        let body = try XCTUnwrap(request.httpBody)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertEqual(json["tool_choice"] as? String, "auto")

        let encodedTools = try XCTUnwrap(json["tools"] as? [[String: Any]])
        XCTAssertEqual(encodedTools.count, 1)
        XCTAssertEqual(encodedTools.first?["type"] as? String, "function")
        let function = try XCTUnwrap(encodedTools.first?["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "file_read")
        XCTAssertEqual(function["description"] as? String, "Read a file")
        let parameters = try XCTUnwrap(function["parameters"] as? [String: Any])
        XCTAssertEqual(parameters["type"] as? String, "object")
        let properties = try XCTUnwrap(parameters["properties"] as? [String: Any])
        let path = try XCTUnwrap(properties["path"] as? [String: Any])
        XCTAssertEqual(path["type"] as? String, "string")
    }

    func testPlainChatRequestOmitsTools() throws {
        let service = OpenAICloudChatService(
            chatURL: CloudProviderRegistry.synthetic.chatCompletionsURL,
            modelsURL: CloudProviderRegistry.synthetic.modelsURL,
            apiKey: "sk-test-key",
            requestTimeout: 5
        )
        let request = try service.makeRequest(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(),
            tools: []
        )
        let body = try XCTUnwrap(request.httpBody)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertNil(json["tools"])
        XCTAssertNil(json["tool_choice"])
    }

    func testUsageExtensionCanBeOmittedForProvidersThatRejectIt() throws {
        let service = OpenAICloudChatService(
            chatURL: CloudProviderRegistry.synthetic.chatCompletionsURL,
            modelsURL: CloudProviderRegistry.synthetic.modelsURL,
            apiKey: "sk-test-key",
            requestTimeout: 5
        )
        let request = try service.makeRequest(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(),
            includeUsage: false
        )
        let body = try XCTUnwrap(request.httpBody)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertNil(json["stream_options"])
    }

    func testStreamOptionsRejectionIsRecognized() {
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutStreamOptions(
            status: 400,
            body: #"{"error":"Unsupported parameter: stream_options"}"#
        ))
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutStreamOptions(
            status: 422,
            body: "include_usage is not supported"
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutStreamOptions(
            status: 401,
            body: "stream_options"
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutStreamOptions(
            status: 400,
            body: #"{"error":{"code":"context_length_exceeded"}}"#
        ))
    }

    func testToolExchangeMessagesEncodeToolCallsAndResults() throws {
        let call = CloudChatMessage(
            role: "assistant",
            content: nil,
            toolCalls: [CloudChatToolCall(id: "call_1", name: "file_read", arguments: #"{"path":"/tmp/a"}"#)]
        )
        let result = CloudChatMessage(role: "tool", content: "ok", toolCallId: "call_1")
        let data = try JSONEncoder().encode([call, result])
        let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])

        XCTAssertEqual(decoded[0]["role"] as? String, "assistant")
        XCTAssertNil(decoded[0]["content"])
        let calls = try XCTUnwrap(decoded[0]["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(calls.first?["id"] as? String, "call_1")
        XCTAssertEqual(calls.first?["type"] as? String, "function")
        let function = try XCTUnwrap(calls.first?["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "file_read")
        XCTAssertEqual(function["arguments"] as? String, #"{"path":"/tmp/a"}"#)

        XCTAssertEqual(decoded[1]["role"] as? String, "tool")
        XCTAssertEqual(decoded[1]["tool_call_id"] as? String, "call_1")
        XCTAssertEqual(decoded[1]["content"] as? String, "ok")
    }

    func testAnthropicFlavorFailsExplicitly() async {
        let client = CloudChatClient(provider: CloudProviderRegistry.synthetic, flavor: .anthropic, apiKey: "sk-test")
        var seen: [CloudTurnEvent] = []
        for await event in client.streamTurn(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams()
        ) {
            seen.append(event)
        }
        XCTAssertEqual(seen, [.failed(.unsupportedFlavor(flavor: CloudApiFlavor.anthropic.rawValue))])
    }

    func testTestConnectionReportsUnsupportedFlavor() async {
        let client = CloudChatClient(provider: CloudProviderRegistry.synthetic, flavor: .anthropic, apiKey: "sk-test")
        let result = await client.testConnection()
        guard case .failure(let failure) = result else {
            return XCTFail("expected failure")
        }
        XCTAssertEqual(failure, .unsupportedFlavor(flavor: "Anthropic"))
    }
}

// MARK: - Message Assembly

final class CloudMessageAssemblerTests: XCTestCase {

    func testSkipsSystemAndEmptyMessages() {
        let messages = [
            ChatMessage(role: .system, content: "UI notice"),
            ChatMessage(role: .user, content: "  first  "),
            ChatMessage(role: .assistant, content: "   "),
            ChatMessage(role: .assistant, content: "reply")
        ]
        let assembled = CloudMessageAssembler.assemble(from: messages)
        XCTAssertEqual(assembled, [
            CloudChatMessage(role: "user", content: "first"),
            CloudChatMessage(role: "assistant", content: "reply")
        ])
    }

    func testExcludesTheStreamingPlaceholderById() {
        let placeholderId = UUID()
        let messages = [
            ChatMessage(role: .user, content: "hi"),
            ChatMessage(id: placeholderId, role: .assistant, content: "")
        ]
        let assembled = CloudMessageAssembler.assemble(from: messages, excludingMessageIds: [placeholderId])
        XCTAssertEqual(assembled.count, 1)
        XCTAssertEqual(assembled.first?.content, "hi")
    }

    func testStripsLeadingSpeakerLabels() {
        let messages = [ChatMessage(role: .assistant, content: "Assistant: Hello there")]
        let assembled = CloudMessageAssembler.assemble(from: messages)
        XCTAssertEqual(assembled.first?.content, "Hello there")
    }

    func testApproximateTokensAndRenderedLog() {
        let messages = [CloudChatMessage(role: "user", content: "12345678")]
        let tokens = CloudMessageAssembler.approximatePromptTokens(system: "1234", messages: messages)
        XCTAssertEqual(tokens, 3)

        let rendered = CloudMessageAssembler.renderedPromptForLog(system: "sys", messages: messages)
        XCTAssertEqual(rendered, "sys\n\nUser: 12345678")
    }

    func testEmptyPromptStillCountsOneToken() {
        XCTAssertEqual(CloudMessageAssembler.approximatePromptTokens(system: "", messages: []), 1)
    }

    func testToolTurnsRebuildTheOpenAIToolExchange() {
        let record = ToolCallRecord(
            name: "file_read",
            arguments: ["path": "/tmp/a"],
            rawArguments: #"{"path":"/tmp/a"}"#,
            status: .success,
            output: "[file_read] success\n--- content ---\nhello"
        )
        let messages = [
            ChatMessage(role: .user, content: "read it"),
            ChatMessage(role: .assistant, content: "Reading now.", toolCalls: [record]),
            ChatMessage(role: .assistant, content: "It says hello.")
        ]

        let assembled = CloudMessageAssembler.assemble(from: messages)
        XCTAssertEqual(assembled.map(\.role), ["user", "assistant", "tool", "assistant"])

        XCTAssertEqual(assembled[1].content, "Reading now.")
        XCTAssertEqual(assembled[1].toolCalls?.count, 1)
        XCTAssertEqual(assembled[1].toolCalls?.first?.id, CloudMessageAssembler.wireCallId(for: record))
        XCTAssertEqual(assembled[1].toolCalls?.first?.function.arguments, #"{"path":"/tmp/a"}"#)

        XCTAssertEqual(assembled[2].toolCallId, CloudMessageAssembler.wireCallId(for: record))
        XCTAssertEqual(assembled[2].content, "[file_read] success\n--- content ---\nhello")

        XCTAssertEqual(assembled[3].content, "It says hello.")
    }

    func testToolOnlyAssistantTurnKeepsNilContent() {
        let record = ToolCallRecord(
            name: "git_status",
            arguments: [:],
            rawArguments: "",
            status: .success,
            output: "clean"
        )
        let assembled = CloudMessageAssembler.assemble(from: [
            ChatMessage(role: .assistant, content: "", toolCalls: [record])
        ])
        XCTAssertEqual(assembled.map(\.role), ["assistant", "tool"])
        XCTAssertNil(assembled[0].content)
        XCTAssertEqual(assembled[0].toolCalls?.count, 1)
        XCTAssertEqual(assembled[0].toolCalls?.first?.function.arguments, "{}")
    }

    func testInterruptedCallStillGetsAResultMessage() {
        let record = ToolCallRecord(name: "shell_run", arguments: ["command": "ls"], status: .running)
        let assembled = CloudMessageAssembler.assemble(from: [
            ChatMessage(role: .assistant, content: "", toolCalls: [record])
        ])
        XCTAssertEqual(assembled.map(\.role), ["assistant", "tool"])
        XCTAssertEqual(assembled[1].content, "The call did not finish before the turn ended.")
    }

    func testEveryPersistedCallGetsExactlyOneResult() {
        let records = [
            ToolCallRecord(name: "file_read", arguments: ["path": "/a"], status: .success, output: "a"),
            ToolCallRecord(name: "shell_run", arguments: ["command": "ls"], status: .error, error: "boom"),
            ToolCallRecord(name: "file_write", arguments: ["path": "/b"], status: .rejected)
        ]
        let assembled = CloudMessageAssembler.assemble(from: [
            ChatMessage(role: .assistant, content: "going", toolCalls: records)
        ])
        let toolMessages = assembled.filter { $0.role == "tool" }
        XCTAssertEqual(toolMessages.count, records.count)
        XCTAssertEqual(Set(toolMessages.compactMap(\.toolCallId)).count, records.count)
        XCTAssertEqual(toolMessages[1].content, "boom")
        XCTAssertEqual(toolMessages[2].content, "Action rejected by user.")
    }

    func testToolTurnAssemblyIsStableAcrossRepeatedRenders() {
        let record = ToolCallRecord(
            name: "file_read",
            arguments: ["path": "/tmp/a"],
            rawArguments: #"{"path":"/tmp/a"}"#,
            status: .success,
            output: "hello"
        )
        let history = [
            ChatMessage(role: .user, content: "read it"),
            ChatMessage(role: .assistant, content: "", toolCalls: [record])
        ]
        let first = CloudMessageAssembler.assemble(from: history)
        let second = CloudMessageAssembler.assemble(from: history)
        XCTAssertEqual(first, second)
    }

    func testRenderedLogNamesToolCalls() {
        let rendered = CloudMessageAssembler.renderedPromptForLog(
            system: "sys",
            messages: [
                CloudChatMessage(
                    role: "assistant",
                    content: nil,
                    toolCalls: [CloudChatToolCall(id: "call_1", name: "file_read", arguments: #"{"path":"/a"}"#)]
                ),
                CloudChatMessage(role: "tool", content: "ok", toolCallId: "call_1")
            ]
        )
        XCTAssertTrue(rendered.contains("file_read"))
        XCTAssertTrue(rendered.contains("Tool result: ok"))
    }
}

// MARK: - Failure Messaging

final class CloudTurnFailureTests: XCTestCase {

    func testAuthenticationFailurePointsAtSettings() {
        let failure = CloudTurnFailure.httpStatus(code: 401, summary: "Unauthorized")
        XCTAssertTrue(failure.localizedDescription.contains("Settings ▸ Cloud Providers"))
    }

    func testRateLimitFailureIsActionable() {
        let failure = CloudTurnFailure.httpStatus(code: 429, summary: "")
        XCTAssertTrue(failure.localizedDescription.lowercased().contains("rate limiting"))
    }

    func testOtherStatusKeepsProviderDetail() {
        let failure = CloudTurnFailure.httpStatus(code: 500, summary: "upstream exploded")
        XCTAssertTrue(failure.localizedDescription.contains("upstream exploded"))
    }

    func testMissingKeyNamesProviderAndPlace() {
        let failure = CloudTurnFailure.missingAPIKey(providerName: "Synthetic")
        XCTAssertTrue(failure.localizedDescription.contains("Synthetic"))
        XCTAssertTrue(failure.localizedDescription.contains("Settings ▸ Cloud Providers"))
    }

    func testContextOverflowBodiesAreRecognized() {
        XCTAssertTrue(CloudTurnFailure.indicatesContextOverflow(
            "This model's maximum context length is 131072 tokens, however you requested 140000 tokens."
        ))
        XCTAssertTrue(CloudTurnFailure.indicatesContextOverflow(
            "prompt is too long: 210000 tokens > 128000 maximum"
        ))
        XCTAssertTrue(CloudTurnFailure.indicatesContextOverflow(
            #"{"error":{"code":"context_length_exceeded"}}"#
        ))
    }

    func testToolSchemaRejectionsAreNotMistakenForOverflow() {
        XCTAssertFalse(CloudTurnFailure.indicatesContextOverflow(
            "This model does not support tools or function calling."
        ))
        XCTAssertFalse(CloudTurnFailure.indicatesContextOverflow("invalid request body"))
    }
}

// MARK: - Subagent Relay

final class CloudSubagentRelayTests: XCTestCase {

    func testDirectiveCarriesEveryReportAndForbidsMoreCalls() {
        let directive = CloudSubagentRelay.directive(reports: [
            "[SUBAGENT_RESULT] role=Codebase Researcher id=1 status=completed\nfirst",
            "[SUBAGENT_RESULT] role=Test Runner id=2 status=completed\nsecond"
        ])
        XCTAssertTrue(directive.contains("first"))
        XCTAssertTrue(directive.contains("second"))
        XCTAssertTrue(directive.contains("Do not emit any tool calls."))
        XCTAssertTrue(directive.hasPrefix("<system>"))
        XCTAssertTrue(directive.hasSuffix("</system>"))
    }
}

// MARK: - Keychain Store

final class CloudSecretStoreTests: XCTestCase {
    private var store: CloudSecretStore!

    override func setUp() {
        super.setUp()
        store = CloudSecretStore(service: "DRP.DynaMoE.cloud.tests.\(UUID().uuidString)")
    }

    override func tearDown() {
        store?.delete(providerId: "synthetic")
        store = nil
        super.tearDown()
    }

    func testAccountNameIsDeterministic() {
        XCTAssertEqual(CloudSecretStore.account(providerId: "synthetic"), "dynamoe.cloud.synthetic.apikey")
    }

    func testSetGetDeleteRoundTrip() throws {
        let secret = "sk-test-\(UUID().uuidString)"
        do {
            try store.set(secret, providerId: "synthetic")
        } catch {
            throw XCTSkip("Keychain is not reachable from this test host: \(error.localizedDescription)")
        }
        XCTAssertEqual(store.get(providerId: "synthetic"), secret)
        XCTAssertTrue(store.exists(providerId: "synthetic"))

        store.delete(providerId: "synthetic")
        XCTAssertNil(store.get(providerId: "synthetic"))
    }

    func testRefusesEmptySecret() {
        XCTAssertThrowsError(try store.set("", providerId: "synthetic"))
    }
}

// MARK: - Stream Transport

/// Scripted response for the injected URLSession below. The app never uses a
/// stub session; it exists so the transport's status handling, SSE framing, and
/// empty-stream guard can be exercised without a provider.
private final class CloudStubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var status = 200
    nonisolated(unsafe) static var body = Data()
    nonisolated(unsafe) static var requestBodies: [Data] = []

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        if let httpBody = request.httpBody {
            Self.requestBodies.append(httpBody)
        } else if let stream = request.httpBodyStream {
            stream.open()
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let read = stream.read(&buffer, maxLength: buffer.count)
                if read <= 0 { break }
                data.append(buffer, count: read)
            }
            stream.close()
            Self.requestBodies.append(data)
        }
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "https://example.invalid")!,
            statusCode: Self.status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class CloudStreamTransportTests: XCTestCase {

    private func makeService() -> OpenAICloudChatService {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CloudStubURLProtocol.self]
        return OpenAICloudChatService(
            chatURL: CloudProviderRegistry.synthetic.chatCompletionsURL,
            modelsURL: CloudProviderRegistry.synthetic.modelsURL,
            apiKey: "sk-test-key",
            requestTimeout: 5,
            session: URLSession(configuration: configuration)
        )
    }

    private func collect(_ service: OpenAICloudChatService) async -> [CloudTurnEvent] {
        var events: [CloudTurnEvent] = []
        for await event in service.streamTurn(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(),
            tools: nil
        ) {
            events.append(event)
        }
        return events
    }

    override func setUp() {
        super.setUp()
        CloudStubURLProtocol.status = 200
        CloudStubURLProtocol.body = Data()
        CloudStubURLProtocol.requestBodies = []
    }

    func testContentStreamFinishesWithAccumulatedText() async {
        CloudStubURLProtocol.body = Data(
            """
            data: {"choices":[{"delta":{"content":"Hel"}}]}

            data: {"choices":[{"delta":{"content":"lo"}}]}

            data: [DONE]

            """.utf8
        )
        let events = await collect(makeService())
        guard case .finished(let result)? = events.last else {
            return XCTFail("expected a finished event, got \(String(describing: events.last))")
        }
        XCTAssertEqual(result.text, "Hello")
        XCTAssertTrue(events.contains(.contentDelta("Hel")))
    }

    func testUnknownJSONBodyFailsInsteadOfFinishingEmpty() async {
        CloudStubURLProtocol.body = Data("data: {\"unexpected\":true}\n\n".utf8)
        let events = await collect(makeService())
        guard case .failed(let failure)? = events.last else {
            return XCTFail("expected a failure, got \(String(describing: events.last))")
        }
        XCTAssertTrue(failure.localizedDescription.contains("without sending any content"))
    }

    func testErrorEnvelopeInABodySurfacesTheProviderMessage() async {
        CloudStubURLProtocol.body = Data(
            "data: {\"error\":{\"message\":\"model not found\"}}\n\n".utf8
        )
        let events = await collect(makeService())
        guard case .failed(let failure)? = events.last else {
            return XCTFail("expected a failure, got \(String(describing: events.last))")
        }
        XCTAssertTrue(failure.localizedDescription.contains("model not found"))
    }

    func testRejectedUsageExtensionIsRetriedOnceWithoutIt() async {
        CloudStubURLProtocol.status = 400
        CloudStubURLProtocol.body = Data(#"{"error":"Unsupported parameter: stream_options"}"#.utf8)
        let events = await collect(makeService())
        guard case .failed(let failure)? = events.last else {
            return XCTFail("expected a failure, got \(String(describing: events.last))")
        }
        // The stub rejects both attempts, so the turn fails, but the retry must
        // have been attempted: two request bodies, the second without the field.
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 2)
        let first = try? JSONSerialization.jsonObject(with: CloudStubURLProtocol.requestBodies[0]) as? [String: Any]
        let second = try? JSONSerialization.jsonObject(with: CloudStubURLProtocol.requestBodies[1]) as? [String: Any]
        XCTAssertNotNil(first?["stream_options"])
        XCTAssertNil(second?["stream_options"])
        guard case .httpStatus(let code, _) = failure else {
            return XCTFail("expected an HTTP failure, got \(failure)")
        }
        XCTAssertEqual(code, 400)
    }
}
