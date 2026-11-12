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

    func testUnindexedFragmentWithoutAnIdContinuesThePreviousCall() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(
            index: nil, id: "call_1", name: "shell_run", arguments: #"{"comm"#
        ))
        _ = try parser.consume(dataPayload: callFragment(index: nil, arguments: #"and":"ls"}"#))

        let calls = parser.result().toolCalls
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.argumentsJSON, #"{"command":"ls"}"#)
    }

    func testUnindexedFragmentWithAFreshIdStartsANewCall() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: callFragment(
            index: nil, id: "call_a", name: "file_read", arguments: #"{"path":"/tmp/a"}"#
        ))
        _ = try parser.consume(dataPayload: callFragment(
            index: nil, id: "call_b", name: "grep_search", arguments: #"{"pattern":"foo"}"#
        ))

        let calls = parser.result().toolCalls
        XCTAssertEqual(calls.count, 2)
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
            try skipIfKeychainUnreachable(error)
        }
        XCTAssertEqual(store.get(providerId: "synthetic"), secret)
        XCTAssertEqual(store.read(providerId: "synthetic"), .found(secret))
        XCTAssertTrue(store.exists(providerId: "synthetic"))
        XCTAssertTrue(store.hasCurrentCredential(providerId: "synthetic"))

        store.delete(providerId: "synthetic")
        XCTAssertNil(store.get(providerId: "synthetic"))
        XCTAssertEqual(
            store.read(providerId: "synthetic"),
            .missing,
            "a deleted key must read as missing rather than as a refusal, so the reply is 'add one'"
        )
    }

    func testRefusesEmptySecret() {
        XCTAssertThrowsError(try store.set("", providerId: "synthetic"))
    }
}

// MARK: - Stored key outcome

/// The reasons a cloud turn cannot start, which the chat surface turns into
/// different messages. `makeStoredResult` is this rule plus a keychain read, so
/// every branch is exercised here without a keychain.
final class CloudStoredKeyLogicTests: XCTestCase {

    private func result(
        _ read: SecretAccessResult,
        legacyCopy: Bool = false
    ) -> Result<String, CloudTurnFailure> {
        CloudChatClient.storedKeyResult(
            providerName: "Synthetic",
            read: read,
            existsInEitherKeychain: { legacyCopy }
        )
    }

    func testAStoredKeyIsUsedAndTheOlderKeychainIsNeverAsked() {
        var asked = false
        let result = CloudChatClient.storedKeyResult(
            providerName: "Synthetic",
            read: .found("sk-1"),
            existsInEitherKeychain: {
                asked = true
                return true
            }
        )
        XCTAssertEqual(try? result.get(), "sk-1")
        XCTAssertFalse(asked, "a readable key settles it; the existence check is a second keychain call")
    }

    func testAKeyOnlyInTheOlderKeychainAsksForAnImport() {
        XCTAssertEqual(
            result(.missing, legacyCopy: true),
            .failure(.credentialNeedsImport(providerName: "Synthetic"))
        )
    }

    func testNoKeyAnywhereAsksForOne() {
        XCTAssertEqual(
            result(.missing),
            .failure(.missingAPIKey(providerName: "Synthetic"))
        )
    }

    /// The defect this pins: `.found` with an empty value joined the
    /// needs-import branch, and the existence check found that same empty item,
    /// so a key that could never work was reported as one waiting to be imported.
    func testAnEmptyStoredItemIsNotAnImportRequest() {
        XCTAssertEqual(
            result(.found(""), legacyCopy: true),
            .failure(.missingAPIKey(providerName: "Synthetic"))
        )
    }

    func testARefusedReadIsReportedAsDeniedRatherThanMissing() {
        XCTAssertEqual(
            result(.accessDenied(errSecUserCanceled)),
            .failure(.credentialAccessDenied(providerName: "Synthetic"))
        )
        XCTAssertEqual(
            result(.unavailable(errSecMissingEntitlement)),
            .failure(.credentialAccessDenied(providerName: "Synthetic")),
            "an unentitled build holding a key is not a missing key"
        )
    }
}

// MARK: - Stream Transport

/// Scripted response for the injected URLSession below. The app never uses a
/// stub session; it exists so the transport's status handling, SSE framing, and
/// empty-stream guard can be exercised without a provider.
private final class CloudStubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var status = 200
    nonisolated(unsafe) static var body = Data()
    /// Consumed first, one entry per request, so a test can script a rejection
    /// followed by an acceptance on the same endpoint.
    nonisolated(unsafe) static var responses: [(status: Int, body: Data)] = []
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
        let reply = Self.responses.isEmpty
            ? (status: Self.status, body: Self.body)
            : Self.responses.removeFirst()
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "https://example.invalid")!,
            statusCode: reply.status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: reply.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class CloudStreamTransportTests: XCTestCase {

    private func makeService(
        host: String = "provider.test",
        // The SHIPPED built-in (`CloudProviderRegistry.synthetic`) sends the
        // OpenAI hint, so the default request shape under test is the one a
        // real turn produces: both optional body fields present.
        hints: CloudCacheHints = .openAI
    ) -> OpenAICloudChatService {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CloudStubURLProtocol.self]
        let base = "https://\(host)/openai/v1"
        return OpenAICloudChatService(
            chatURL: URL(string: "\(base)/chat/completions")!,
            modelsURL: URL(string: "\(base)/models")!,
            apiKey: "sk-test-key",
            requestTimeout: 5,
            session: URLSession(configuration: configuration),
            cacheHints: hints
        )
    }

    private func collect(
        _ service: OpenAICloudChatService,
        sessionKey: String? = nil
    ) async -> [CloudTurnEvent] {
        var events: [CloudTurnEvent] = []
        for await event in service.streamTurn(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(sessionKey: sessionKey),
            tools: nil
        ) {
            events.append(event)
        }
        return events
    }

    private func requestBody(_ index: Int) -> [String: Any]? {
        guard index < CloudStubURLProtocol.requestBodies.count else { return nil }
        return try? JSONSerialization.jsonObject(
            with: CloudStubURLProtocol.requestBodies[index]
        ) as? [String: Any]
    }

    override func setUp() {
        super.setUp()
        CloudStubURLProtocol.status = 200
        CloudStubURLProtocol.body = Data()
        CloudStubURLProtocol.responses = []
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
        // The shipped provider sends the cache key as well, and that is the
        // request shape this retry has to preserve: dropping the usage
        // extension must not also drop the routing hint.
        let events = await collect(makeService(), sessionKey: "usage-retry")
        guard case .failed(let failure)? = events.last else {
            return XCTFail("expected a failure, got \(String(describing: events.last))")
        }
        // The stub rejects every attempt, so the turn fails, but both retries
        // must have been attempted. Attempt 2 drops only the usage extension,
        // which is the shape that proves the cache key survives it; its
        // continued 400 then earns the blind cache-key retry, and attempt 3
        // carries neither optional field.
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 3)
        let first = try? JSONSerialization.jsonObject(with: CloudStubURLProtocol.requestBodies[0]) as? [String: Any]
        let second = try? JSONSerialization.jsonObject(with: CloudStubURLProtocol.requestBodies[1]) as? [String: Any]
        let third = try? JSONSerialization.jsonObject(with: CloudStubURLProtocol.requestBodies[2]) as? [String: Any]
        XCTAssertNotNil(first?["stream_options"])
        XCTAssertEqual(first?["prompt_cache_key"] as? String, "usage-retry")
        XCTAssertNil(second?["stream_options"])
        XCTAssertEqual(second?["prompt_cache_key"] as? String, "usage-retry")
        XCTAssertNil(third?["stream_options"])
        XCTAssertNil(third?["prompt_cache_key"])
        guard case .httpStatus(let code, _) = failure else {
            return XCTFail("expected an HTTP failure, got \(failure)")
        }
        XCTAssertEqual(code, 400)
    }

    func testAUsageRejectionThenACacheKeyRejectionTakesTwoRetries() async {
        // The two optional fields are dropped one at a time, and the second
        // attempt still carries the cache key. A retry that cleared both at
        // once would look identical to a turn that never asked for the cache.
        let ok = Data("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".utf8)
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"Unsupported parameter: stream_options"}"#.utf8)),
            (400, Data(#"{"error":"unexpected property prompt_cache_key"}"#.utf8)),
            (200, ok)
        ]
        // A fresh endpoint per invocation keeps the process-wide latch out of
        // the picture, so a rerun starts on its first attempt again.
        let host = "two-stage-\(UUID().uuidString).test"
        let service = makeService(host: host)
        guard case .finished? = await collect(service, sessionKey: "session-two-stage").last else {
            return XCTFail("the third attempt should succeed")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 3)
        XCTAssertNotNil(requestBody(0)?["stream_options"])
        XCTAssertEqual(requestBody(0)?["prompt_cache_key"] as? String, "session-two-stage")
        XCTAssertNil(requestBody(1)?["stream_options"])
        XCTAssertEqual(requestBody(1)?["prompt_cache_key"] as? String, "session-two-stage")
        XCTAssertNil(requestBody(2)?["stream_options"])
        XCTAssertNil(requestBody(2)?["prompt_cache_key"])
        XCTAssertFalse(
            OpenAICloudChatService.sendsCacheKey(
                hints: .openAI,
                sessionKey: "session-two-stage",
                endpoint: "https://\(host)/openai/v1/chat/completions"
            ),
            "the accepted retry without the field is what proves it was refused"
        )
    }

    func testACacheKeyIsRememberedOnlyAfterTheRetrySucceeds() async {
        let ok = Data("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".utf8)
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"unexpected property prompt_cache_key"}"#.utf8)),
            (200, ok),
            // Second turn: the field is already known to be refused, so its
            // first attempt is the only request this turn makes.
            (200, ok)
        ]
        // The latch is process-lifetime and keyed by endpoint, so a fixed host
        // would carry one run's answer into the next: a repeat-on-failure
        // rerun (or a second pass at this case) would find the key already
        // withheld and fail its first assertion. A fresh endpoint per
        // invocation is the isolation, without mutating the shared latch.
        let service = makeService(host: "cache-latch-\(UUID().uuidString).test", hints: .openAI)
        guard case .finished? = await collect(service, sessionKey: "session-a").last else {
            return XCTFail("the first turn should succeed on its retry")
        }
        guard case .finished? = await collect(service, sessionKey: "session-a").last else {
            return XCTFail("the second turn should succeed on its first attempt")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 3)
        XCTAssertNotNil(requestBody(0)?["prompt_cache_key"])
        XCTAssertNil(requestBody(1)?["prompt_cache_key"])
        XCTAssertNil(requestBody(2)?["prompt_cache_key"])
    }

    func testAnUnrelatedRejectionDoesNotRememberTheCacheKeyAsRefused() async {
        // Every attempt fails on a model error, so nothing proves the optional
        // field was the problem. The next turn must still ask for the cache.
        let failure = Data(#"{"error":"model not found"}"#.utf8)
        CloudStubURLProtocol.responses = [(400, failure), (400, failure)]
        let service = makeService(host: "cache-kept-\(UUID().uuidString).test", hints: .openAI)
        guard case .failed? = await collect(service, sessionKey: "session-b").last else {
            return XCTFail("the first turn should fail")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 2)

        CloudStubURLProtocol.responses = [(400, failure), (400, failure)]
        guard case .failed? = await collect(service, sessionKey: "session-b").last else {
            return XCTFail("the second turn should fail")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 4)
        XCTAssertNotNil(requestBody(2)?["prompt_cache_key"])
        XCTAssertNil(requestBody(3)?["prompt_cache_key"])
    }
}

// MARK: - Cache-aware usage

final class CloudUsageSplitTests: XCTestCase {
    private func decode(_ json: String) throws -> CloudUsage {
        try JSONDecoder().decode(CloudUsage.self, from: Data(json.utf8))
    }

    func testOpenAINestedCacheDetailIsDecoded() throws {
        let usage = try decode(
            #"{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1050,"prompt_tokens_details":{"cached_tokens":800}}"#
        )
        XCTAssertEqual(usage.promptTokens, 1000)
        XCTAssertEqual(usage.cachedTokens, 800)
        XCTAssertEqual(usage.billableInputTokens, 200)
        XCTAssertEqual(usage.cacheReadFraction, 0.8)
    }

    func testDeepSeekTopLevelCacheHitIsDecoded() throws {
        let usage = try decode(#"{"prompt_tokens":900,"completion_tokens":10,"prompt_cache_hit_tokens":600}"#)
        XCTAssertEqual(usage.cachedTokens, 600)
        XCTAssertEqual(usage.billableInputTokens, 300)
    }

    func testDeepSeekTopLevelHitBeatsANestedCachedCount() throws {
        // A gateway that fronts DeepSeek can publish both spellings, and they
        // are not the same number. The dedicated top-level field is DeepSeek's
        // own; the nested detail is OpenAI's generic one, so the dedicated
        // name wins. Reading the nested value here would under-report the
        // cache hit and over-report what the turn billed.
        let usage = try decode(
            #"{"prompt_tokens":1000,"completion_tokens":10,"prompt_cache_hit_tokens":600,"prompt_tokens_details":{"cached_tokens":200}}"#
        )
        XCTAssertEqual(usage.cachedTokens, 600)
        XCTAssertEqual(usage.billableInputTokens, 400)
    }

    func testAnthropicCacheColumnsAreDecoded() throws {
        let usage = try decode(
            #"{"input_tokens":500,"output_tokens":20,"cache_read_input_tokens":400,"cache_creation_input_tokens":100}"#
        )
        // Anthropic's `input_tokens` is only the uncached share; the prompt is
        // all three columns. 500 + 400 + 100, not 500.
        XCTAssertEqual(usage.promptTokens, 1_000)
        XCTAssertEqual(usage.cachedTokens, 400)
        XCTAssertEqual(usage.cacheWriteTokens, 100)
        XCTAssertEqual(usage.billableInputTokens, 600)
        XCTAssertEqual(usage.cacheReadFraction, 0.4)
    }

    func testAnthropicWithoutCacheColumnsKeepsItsPromptTotal() throws {
        // Without the cache keys the block is indistinguishable from the
        // Responses shape, where `input_tokens` already is the total.
        let usage = try decode(#"{"input_tokens":500,"output_tokens":20}"#)
        XCTAssertEqual(usage.promptTokens, 500)
        XCTAssertNil(usage.cachedTokens)
        XCTAssertEqual(usage.billableInputTokens, 500)
    }

    func testAnthropicTotalsPriceLikeAnyOtherPrompt() throws {
        let usage = try decode(
            #"{"input_tokens":500,"output_tokens":20,"cache_read_input_tokens":400,"cache_creation_input_tokens":100}"#
        )
        let pricing = CloudPricing(input: 3, cachedInput: 0.3, output: 15)
        // 600 fresh (500 uncached + 100 cache write) at $3/M plus 400 cached
        // at $0.30/M plus 20 out at $15/M.
        XCTAssertEqual(try XCTUnwrap(pricing.cost(for: usage)), 0.0018 + 0.00012 + 0.0003, accuracy: 1e-12)
    }

    func testAProviderWithoutACacheSplitStillReportsBillableInput() throws {
        let usage = try decode(#"{"prompt_tokens":250,"completion_tokens":5}"#)
        XCTAssertNil(usage.cachedTokens)
        XCTAssertEqual(usage.billableInputTokens, 250)
        XCTAssertNil(usage.cacheReadFraction)
    }

    func testNoPromptCountMeansNoBillableNumber() throws {
        let usage = try decode(#"{"completion_tokens":5}"#)
        XCTAssertNil(usage.billableInputTokens)
        XCTAssertNil(usage.cacheReadFraction)
    }

    func testCachedCannotExceedThePrompt() throws {
        let usage = try decode(#"{"prompt_tokens":10,"cached_tokens":50}"#)
        XCTAssertEqual(usage.billableInputTokens, 0)
        XCTAssertEqual(usage.cacheReadFraction, 1.0)
    }

    func testTheNestedOpenAIDetailBeatsAFlatColumn() throws {
        // `cached_tokens` at the top level is a generic column some compatible
        // endpoints use for something else. OpenAI's documented home is the
        // nested detail, so that is what a block carrying both must report.
        let usage = try decode(
            #"{"prompt_tokens":1000,"cached_tokens":999,"prompt_tokens_details":{"cached_tokens":800}}"#
        )
        XCTAssertEqual(usage.cachedTokens, 800)
        XCTAssertEqual(usage.billableInputTokens, 200)
    }

    func testAResponsesShapedInputDetailIsAlsoRead() throws {
        let usage = try decode(
            #"{"input_tokens":1000,"output_tokens":50,"input_tokens_details":{"cached_tokens":600}}"#
        )
        XCTAssertEqual(usage.promptTokens, 1_000)
        XCTAssertEqual(usage.cachedTokens, 600)
        XCTAssertEqual(usage.completionTokens, 50)
    }

    func testStringifiedCountsAreStillDecoded() throws {
        // Losing a stringified number silently marks a measured block as an
        // estimate, which then misreports the run's totals.
        let usage = try decode(
            #"{"prompt_tokens":"1200","completion_tokens":"56","prompt_tokens_details":{"cached_tokens":"1000"}}"#
        )
        XCTAssertEqual(usage.promptTokens, 1_200)
        XCTAssertEqual(usage.completionTokens, 56)
        XCTAssertEqual(usage.cachedTokens, 1_000)
    }

    func testNonsenseAndOverflowingCountsAreClamped() throws {
        let negative = try decode(#"{"prompt_tokens":-5,"completion_tokens":-2}"#)
        XCTAssertEqual(negative.promptTokens, 0)
        XCTAssertEqual(negative.completionTokens, 0)
        // Anthropic's three columns are summed; a hostile block can put Int.max
        // in every one of them and used to trap in the addition.
        let overflow = try decode(
            #"{"input_tokens":9223372036854775807,"cache_read_input_tokens":9223372036854775807,"cache_creation_input_tokens":1}"#
        )
        XCTAssertEqual(overflow.promptTokens, Int.max)
        XCTAssertEqual(overflow.billableInputTokens, 0)
    }

    func testTheUsageBlockRoundTrips() throws {
        let original = CloudUsage(
            promptTokens: 100,
            completionTokens: 7,
            totalTokens: 107,
            cachedTokens: 40,
            cacheWriteTokens: 5
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(CloudUsage.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testALaterPartialUsageBlockKeepsTheColumnsItOmits() {
        // Some endpoints send a running `output_tokens` mid-stream and the
        // prompt only at the end (or the other way round). Replacing the
        // accumulated block with the newest one dropped the prompt and its
        // cache split, which silently turned a measured block into an estimate
        // — exactly where the cache reporting matters.
        let full = CloudUsage(
            promptTokens: 1_000,
            completionTokens: 5,
            totalTokens: nil,
            cachedTokens: 800,
            cacheWriteTokens: 100
        )
        let later = CloudUsage(promptTokens: nil, completionTokens: 50, totalTokens: nil)
        let merged = full.merging(later)
        XCTAssertEqual(merged.promptTokens, 1_000)
        XCTAssertEqual(merged.cachedTokens, 800)
        XCTAssertEqual(merged.cacheWriteTokens, 100)
        XCTAssertEqual(merged.completionTokens, 50)
    }

    func testALaterUsageBlockOverwritesTheColumnsItReports() {
        let earlier = CloudUsage(promptTokens: 10, completionTokens: 1, cachedTokens: 4)
        let later = CloudUsage(promptTokens: 1_000, completionTokens: 50, cachedTokens: 900)
        let merged = earlier.merging(later)
        XCTAssertEqual(merged.promptTokens, 1_000)
        XCTAssertEqual(merged.completionTokens, 50)
        XCTAssertEqual(merged.cachedTokens, 900)
    }

    func testTheStreamParserMergesUsageAcrossChunks() throws {
        var parser = CloudStreamParser()
        _ = try parser.consume(
            dataPayload: #"{"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":900,"completion_tokens":0,"prompt_tokens_details":{"cached_tokens":700}}}"#
        )
        // A trailing chunk that carries only the final output count.
        _ = try parser.consume(
            dataPayload: #"{"choices":[],"usage":{"completion_tokens":42}}"#
        )
        let usage = try XCTUnwrap(parser.usage)
        XCTAssertEqual(usage.promptTokens, 900)
        XCTAssertEqual(usage.cachedTokens, 700)
        XCTAssertEqual(usage.completionTokens, 42)
    }
}

// MARK: - Cache hints on the wire

final class CloudCacheHintRequestTests: XCTestCase {
    private func makeRequest(
        hints: CloudCacheHints,
        sessionKey: String? = "abc123",
        host: String = "provider.test"
    ) throws -> URLRequest {
        let base = "https://\(host)/openai/v1"
        let service = OpenAICloudChatService(
            chatURL: URL(string: "\(base)/chat/completions")!,
            modelsURL: URL(string: "\(base)/models")!,
            apiKey: "sk-test",
            requestTimeout: 5,
            cacheHints: hints
        )
        return try service.makeRequest(
            model: "m",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(sessionKey: sessionKey)
        )
    }

    private func body(_ request: URLRequest) throws -> [String: Any] {
        let data = try XCTUnwrap(request.httpBody)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testAffinityHeadersRideTheRequestWithoutACacheKey() throws {
        let request = try makeRequest(hints: .affinity, host: "affinity.test")
        XCTAssertEqual(request.value(forHTTPHeaderField: "x-session-id"), "abc123")
        XCTAssertEqual(request.value(forHTTPHeaderField: "x-session-affinity"), "abc123")
        XCTAssertNil(try body(request)["prompt_cache_key"])
    }

    func testTheOpenAIHintAddsTheDocumentedCacheKey() throws {
        let request = try makeRequest(hints: .openAI, host: "openai-key.test")
        XCTAssertEqual(try body(request)["prompt_cache_key"] as? String, "abc123")
        XCTAssertEqual(request.value(forHTTPHeaderField: "x-session-affinity"), "abc123")
    }

    func testNoSessionKeySendsNoCacheHintAtAll() throws {
        let request = try makeRequest(hints: .openAI, sessionKey: nil, host: "no-session.test")
        XCTAssertNil(request.value(forHTTPHeaderField: "x-session-id"))
        XCTAssertNil(request.value(forHTTPHeaderField: "x-session-affinity"))
        XCTAssertNil(try body(request)["prompt_cache_key"])
    }

    func testAnEmptySessionKeySendsNoAffinityHeadersEither() throws {
        // An empty header value is a value routers may treat as a real (shared)
        // session, so the headers are withheld on the same condition as the
        // body key rather than set to "".
        let request = try makeRequest(hints: .openAI, sessionKey: "", host: "empty-session.test")
        XCTAssertNil(request.value(forHTTPHeaderField: "x-session-id"))
        XCTAssertNil(request.value(forHTTPHeaderField: "x-session-affinity"))
        XCTAssertNil(try body(request)["prompt_cache_key"])
    }

    func testTheOffHintSendsNothing() throws {
        let request = try makeRequest(hints: .none, host: "off.test")
        XCTAssertNil(request.value(forHTTPHeaderField: "x-session-id"))
        XCTAssertNil(try body(request)["prompt_cache_key"])
    }

    func testTheBodyIsDeterministicAsWellAsCorrect() throws {
        // Sorted JSON keys are a cache requirement, not a style choice: tool
        // schemas are Swift dictionaries, and an order that flips between
        // requests invalidates the provider's whole prompt prefix.
        let request = try makeRequest(hints: .affinity, host: "sorted.test")
        let data = try XCTUnwrap(request.httpBody)
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))
        let messages = try XCTUnwrap(text.range(of: "\"messages\""))
        let model = try XCTUnwrap(text.range(of: "\"model\""))
        let stream = try XCTUnwrap(text.range(of: "\"stream\""))
        XCTAssertLessThan(messages.lowerBound, model.lowerBound)
        XCTAssertLessThan(model.lowerBound, stream.lowerBound)
    }

    func testOnlyClientErrorsAreRetriedWithoutTheCacheKey() {
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutCacheKey(status: 400, body: "invalid request"))
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutCacheKey(status: 422, body: "unexpected property"))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(status: 401, body: "unauthorized"))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(status: 429, body: "slow down"))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(status: 500, body: "boom"))
        // A 400 about the request's CONTENT is not about the cache field, and
        // retrying it re-uploads the same megabytes before the turn's own
        // overflow or image recovery even starts.
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: #"{"error":{"code":"context_length_exceeded"}}"#
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: #"{"error":"image_url is not supported by this model"}"#
        ))
    }

    func testALatchedCacheKeyIsNotRetriedAndIsNotSent() throws {
        // After the process latched the field off, a request body no longer
        // carries it. The retry predicate must agree: otherwise every 4xx for
        // the rest of the process re-sends the identical body.
        let endpoint = "https://latch-\(UUID().uuidString).test/openai/v1/chat/completions"
        let service = OpenAICloudChatService(
            chatURL: URL(string: endpoint)!,
            modelsURL: URL(string: "https://latch.test/openai/v1/models")!,
            apiKey: "sk-test",
            requestTimeout: 5,
            cacheHints: .openAI
        )
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint
        ))
        CloudCacheHintMemory.shared.declinePromptCacheKey(endpoint: endpoint)
        XCTAssertFalse(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint
        ))
        let request = try service.makeRequest(
            model: "m",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(sessionKey: "abc123")
        )
        XCTAssertNil(try body(request)["prompt_cache_key"])
    }

    func testTheLatchIsScopedToItsOwnEndpoint() {
        // One provider refusing the body field says nothing about another
        // endpoint. A set that was not keyed would switch the hint off for
        // every provider the app talks to after one strict server said no.
        let first = "https://latch-a-\(UUID().uuidString).test/openai/v1/chat/completions"
        let second = "https://latch-b-\(UUID().uuidString).test/openai/v1/chat/completions"
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: first
        ))
        CloudCacheHintMemory.shared.declinePromptCacheKey(endpoint: first)
        XCTAssertFalse(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: first
        ))
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: second
        ))
        // The reset seam restores the first endpoint, so a repeat-on-failure
        // rerun of this case starts where it did the first time.
        CloudCacheHintMemory.shared.reset()
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: first
        ))
    }

    func testAnUnknownCacheHintDecodesToTheSafeDefault() throws {
        // The value is hand-editable and a newer build may write one this build
        // does not know. Failing the whole record silently removed the
        // provider from the list; session-affinity headers break nothing.
        let json = #"""
        {"id":"custom","displayName":"Custom","openAIBaseURL":"https://custom.test/v1","cacheHints":"telepathy"}
        """#
        let provider = try JSONDecoder().decode(CloudProvider.self, from: Data(json.utf8))
        XCTAssertEqual(provider.cacheHints, .affinity)
        XCTAssertEqual(provider.resolvedCacheHints, .affinity)
    }

    func testACacheHintThatIsNotEvenAStringStillDecodes() throws {
        // The same tolerance as an unrecognised spelling. A number here threw
        // out of the provider's decoder, which `FailableProvider` turned into a
        // dropped provider — a whole account lost over one wrong value in a
        // label the app can default.
        let json = #"""
        {"id":"custom","displayName":"Custom","openAIBaseURL":"https://custom.test/v1","cacheHints":1}
        """#
        let provider = try JSONDecoder().decode(CloudProvider.self, from: Data(json.utf8))
        XCTAssertEqual(provider.cacheHints, .affinity)
    }

    func testTheLatchedHintIsInjectedRatherThanGlobal() throws {
        // The latch is keyed by endpoint and lives for the process. A service
        // that holds its own instance keeps one case (or one app run) from
        // leaving a state the next one did not set up.
        let endpoint = "https://injected-\(UUID().uuidString).test/openai/v1/chat/completions"
        let memory = CloudCacheHintMemory()
        let service = OpenAICloudChatService(
            chatURL: URL(string: endpoint)!,
            modelsURL: URL(string: "https://injected.test/openai/v1/models")!,
            apiKey: "sk-test",
            requestTimeout: 5,
            cacheHints: .openAI,
            cacheHintMemory: memory
        )
        memory.declinePromptCacheKey(endpoint: endpoint)
        let request = try service.makeRequest(
            model: "m",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(sessionKey: "abc123")
        )
        XCTAssertNil(try body(request)["prompt_cache_key"])
        // The process-wide latch never saw this endpoint, so another service
        // (which defaults to it) still sends the key.
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint
        ))
    }
}

final class CloudSessionKeyTests: XCTestCase {
    func testTheKeyIsStableOpaqueAndPerConversation() {
        let first = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!
        let second = UUID(uuidString: "22222222-2222-2222-2222-222222222222")!
        let key = CloudSessionKey.hashed(first)
        XCTAssertEqual(key, CloudSessionKey.hashed(first))
        XCTAssertNotEqual(key, CloudSessionKey.hashed(second))
        XCTAssertFalse(key.contains("-"))
        XCTAssertTrue(key.allSatisfy { $0.isHexDigit })
        // A GOLDEN value, not a second call to the same function. The provider
        // keys its prefix cache on this string, so changing the FNV offset,
        // prime or input spelling must fail here: every upgraded install would
        // otherwise silently lose the affinity of conversations already in
        // progress, with no test noticing.
        XCTAssertEqual(key, "c155f09940d24ca1")
    }
}

// MARK: - Image delivery policy

final class CloudImageDeliveryPolicyTests: XCTestCase {
    private func ref(_ name: String) -> ImageAttachmentRef {
        ImageAttachmentRef(
            fileName: name,
            pixelWidth: 10,
            pixelHeight: 10,
            byteCount: 10,
            sha256: name,
            relativePath: "images/\(name)",
            mimeType: "image/png"
        )
    }

    private func conversation() -> [ChatMessage] {
        [
            ChatMessage(role: .user, content: "first picture", attachments: [ref("a.png")]),
            ChatMessage(role: .assistant, content: "saw it"),
            ChatMessage(role: .user, content: "second picture", attachments: [ref("b.png")])
        ]
    }

    private func urls(for messages: [ChatMessage]) -> [UUID: [String]] {
        var map: [UUID: [String]] = [:]
        for message in messages {
            guard let attachments = message.attachments else { continue }
            map[message.id] = attachments.map { "data:image/png;base64,\($0.fileName)" }
        }
        return map
    }

    func testTheDefaultKeepsEveryPicture() {
        let messages = conversation()
        let assembled = CloudMessageAssembler.assemble(
            from: messages,
            imageDataURLs: urls(for: messages)
        )
        XCTAssertEqual(assembled.filter { ($0.imageParts?.count ?? 0) > 0 }.count, 2)
    }

    func testNewestTurnOnlyDropsOlderPicturesWithANotice() {
        let messages = conversation()
        let assembled = CloudMessageAssembler.assemble(
            from: messages,
            imageDataURLs: urls(for: messages),
            imageDelivery: .newestUserTurnOnly
        )
        // Only the newest user turn carries pixels.
        XCTAssertEqual(assembled.filter { ($0.imageParts?.count ?? 0) > 0 }.count, 1)
        let firstTurn = assembled.first { $0.role == "user" }
        XCTAssertNil(firstTurn?.imageParts)
        XCTAssertTrue(firstTurn?.content?.contains("older turn, not re-sent") ?? false)
        // The conversation is otherwise unchanged, so the prefix still extends.
        XCTAssertEqual(assembled.count, 3)
    }

    func testATextFollowUpDoesNotResendTheEarlierPicture() {
        // The switch is about re-sending, so the decision is the newest USER
        // turn even when it has no attachment of its own. Picking the newest
        // image-bearing turn instead kept an older picture on every request,
        // including every step of an agent run, which is what the setting was
        // supposed to stop.
        let picture = ChatMessage(role: .user, content: "see this", attachments: [ref("a.png")])
        let followUp = ChatMessage(role: .user, content: "and now in text?")
        let messages = [
            picture,
            ChatMessage(role: .assistant, content: "saw it"),
            followUp
        ]
        let assembled = CloudMessageAssembler.assemble(
            from: messages,
            imageDataURLs: urls(for: messages),
            imageDelivery: .newestUserTurnOnly
        )
        XCTAssertFalse(assembled.contains { ($0.imageParts?.count ?? 0) > 0 })
        XCTAssertTrue(
            assembled.first { $0.role == "user" }?.content?.contains("older turn, not re-sent") ?? false
        )
        // The newest turn itself is text-only and unchanged.
        XCTAssertEqual(assembled.last?.content, "and now in text?")
    }

    func testThePolicyIsStableAcrossRepeatedRenders() {
        // Prefix stability means an EXTENDED history re-renders the earlier
        // turns identically. Comparing two renders of the same array proves
        // nothing: a policy that ignored `imageDelivery` or inverted it would
        // pass that and still invalidate the provider's cached prefix on every
        // step of an agent run.
        let messages = conversation()
        let map = urls(for: messages)
        let first = CloudMessageAssembler.assemble(
            from: messages,
            imageDataURLs: map,
            imageDelivery: .newestUserTurnOnly
        )
        let extended = CloudMessageAssembler.assemble(
            from: messages + [ChatMessage(role: .assistant, content: "next")],
            imageDataURLs: map,
            imageDelivery: .newestUserTurnOnly
        )
        XCTAssertEqual(Array(extended.prefix(first.count)), first)
        XCTAssertEqual(extended.count, first.count + 1)
        // And the policy is actually ON: keeping every picture must produce a
        // different prefix, or the comparison above would hold for a no-op.
        let keepAll = CloudMessageAssembler.assemble(
            from: messages,
            imageDataURLs: map,
            imageDelivery: .all
        )
        XCTAssertNotEqual(keepAll, first)
    }

    func testThePreTurnImageCountMatchesWhatTheFirstRequestCarries() {
        // The odometer is seeded before the turn's bytes are encoded, so the
        // flat per-image allowance has to price the same pictures `assemble`
        // will actually send. Comparing against the assembled parts is the
        // drift check: a policy added on one side but not the other fails here.
        let messages = conversation()
        let map = urls(for: messages)
        let cases: [(delivery: CloudImageDelivery, supported: Bool)] = [
            (.all, true),
            (.newestUserTurnOnly, true),
            (.all, false)
        ]
        for item in cases {
            let assembled = CloudMessageAssembler.assemble(
                from: messages,
                imageDataURLs: map,
                imagesSupported: item.supported,
                imageDelivery: item.delivery
            )
            let carried = assembled.reduce(0) { $0 + ($1.imageParts?.count ?? 0) }
            XCTAssertEqual(
                CloudMessageAssembler.carriedImageCount(
                    from: messages,
                    delivery: item.delivery,
                    imagesSupported: item.supported
                ),
                carried,
                "delivery \(item.delivery), supported \(item.supported)"
            )
        }
    }

    func testATextFollowUpCarriesNoImageUnderTheResendOffPolicy() {
        let messages = conversation() + [ChatMessage(role: .user, content: "and in text?")]
        XCTAssertEqual(
            CloudMessageAssembler.carriedImageCount(
                from: messages,
                delivery: .newestUserTurnOnly,
                imagesSupported: true
            ),
            0
        )
        XCTAssertEqual(
            CloudMessageAssembler.carriedImageCount(
                from: messages,
                delivery: .all,
                imagesSupported: true
            ),
            2
        )
    }

    func testThePreTurnImageCountIgnoresAttachmentsOnNonUserTurns() {
        let messages = [
            ChatMessage(role: .assistant, content: "look", attachments: [ref("a.png")]),
            ChatMessage(role: .user, content: "hello")
        ]
        XCTAssertEqual(
            CloudMessageAssembler.carriedImageCount(
                from: messages,
                delivery: .all,
                imagesSupported: true
            ),
            0
        )
    }
}

// MARK: - Prefix stability

final class CloudPrefixStabilityTests: XCTestCase {
    func testAStepExtendsThePreviousPromptInsteadOfRewritingIt() {
        let record = ToolCallRecord(
            name: "file_read",
            arguments: ["path": "/a"],
            rawArguments: #"{"path":"/a"}"#,
            status: .success,
            output: "hello"
        )
        let history = [
            ChatMessage(role: .user, content: "read it"),
            ChatMessage(role: .assistant, content: "Reading.", toolCalls: [record]),
            ChatMessage(role: .assistant, content: "It says hello.")
        ]
        let firstStep = CloudMessageAssembler.assemble(from: Array(history.prefix(2)))
        let secondStep = CloudMessageAssembler.assemble(from: history)
        XCTAssertEqual(Array(secondStep.prefix(firstStep.count)), firstStep)
        XCTAssertEqual(secondStep.count, firstStep.count + 1)
    }
}

// MARK: - Step prompt ordering

final class CloudStepPromptTests: XCTestCase {
    private let history = [
        CloudChatMessage(role: "user", content: "hello"),
        CloudChatMessage(role: "assistant", content: "hi")
    ]

    func testACoordinatorDirectiveRidesTheEndAsItsOwnUserTurn() {
        let outbound = CloudStepPrompt.assemble(
            persona: "You are the coordinator.",
            history: history,
            directive: "[subagent report]"
        )
        XCTAssertEqual(outbound.map(\.role), ["system", "user", "assistant", "user"])
        XCTAssertEqual(outbound.first?.content, "You are the coordinator.")
        XCTAssertEqual(outbound.last?.content, "[subagent report]")
    }

    func testADirectiveExtendsThePrefixInsteadOfRewritingIt() {
        let without = CloudStepPrompt.assemble(persona: "persona", history: history, directive: "")
        let with = CloudStepPrompt.assemble(persona: "persona", history: history, directive: "[report]")
        // The system message and every history turn are identical, which is
        // what lets the provider reuse its cached prefix; only the trailing
        // turn is new.
        XCTAssertEqual(Array(with.prefix(without.count)), without)
        XCTAssertEqual(with.count, without.count + 1)
        XCTAssertEqual(with.last?.role, "user")
        // The directive is never merged into the system prompt.
        XCTAssertEqual(with.first?.content, "persona")
    }

    func testAnEmptyPersonaStillLeadsWithHistory() {
        let outbound = CloudStepPrompt.assemble(persona: "", history: history, directive: "")
        XCTAssertEqual(outbound, history)
        XCTAssertFalse(outbound.contains { $0.role == "system" })
    }
}

// MARK: - Context budget

final class CloudCompactionAttemptsTests: XCTestCase {
    func testAFailedPreflightPassIsNotRetriedOnEveryStep() {
        var attempts = CloudCompactionAttempts()
        XCTAssertTrue(attempts.shouldAttemptPreflight)
        attempts.beginPreflight()
        // The pass failed (empty summary, truncated summary, provider error):
        // it already spent its request, so no later step may spend another.
        XCTAssertFalse(attempts.shouldAttemptPreflight)
        XCTAssertFalse(attempts.historyWasRewritten)
        // A provider overflow is different evidence and keeps its own attempt.
        XCTAssertTrue(attempts.shouldAttemptOverflowRecovery)
    }

    func testEachPassHasOneAttemptAndARewriteClosesThePreflightGate() {
        var attempts = CloudCompactionAttempts()
        attempts.beginPreflight()
        attempts.recordRewrite()
        XCTAssertFalse(attempts.shouldAttemptPreflight)
        XCTAssertTrue(attempts.historyWasRewritten)
        XCTAssertTrue(attempts.shouldAttemptOverflowRecovery)
        attempts.beginOverflowRecovery()
        XCTAssertFalse(attempts.shouldAttemptOverflowRecovery)
    }

    func testAnAttemptThatSentNoRequestGivesItsGateBack() {
        // A pass refused because there was nothing to fold away sent no
        // request and was billed nothing; one more assistant step can make the
        // tail evictable, so the attempt must not be spent.
        var attempts = CloudCompactionAttempts()
        attempts.beginPreflight()
        attempts.abandonPreflight()
        XCTAssertTrue(attempts.shouldAttemptPreflight)
        // A rewrite is final: there is nothing left to re-summarize.
        attempts.beginPreflight()
        attempts.recordRewrite()
        attempts.abandonPreflight()
        XCTAssertFalse(attempts.shouldAttemptPreflight)
    }
}

final class CloudContextPolicyTests: XCTestCase {
    private func scratchDefaults() -> UserDefaults {
        let suite = "CloudContextPolicyTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    func testOverridesParseBothSpellingsAndDropNonsense() {
        let defaults = scratchDefaults()
        CloudContextPolicy.setOverrides(
            ["b": 32_000, "a": 128_000, "units": 12],
            for: "p",
            defaults: defaults
        )
        // The WRITE side is pinned literally, not through its own reader: a
        // writer and parser that agreed on a wrong format would round-trip
        // perfectly, and the value is hand-edited JSON an operator reads.
        XCTAssertEqual(
            defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "p")),
            "a=128000; b=32000"
        )
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "p", defaults: defaults),
            ["a": 128_000, "b": 32_000]
        )
        defaults.set(
            "model@64000; other=8192; bad=12; =5",
            forKey: CloudContextPolicy.defaultsKey(for: "q")
        )
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "q", defaults: defaults),
            ["model": 64_000, "other": 8_192]
        )
        // Nothing usable means no stored key at all, so a later reader cannot
        // mistake an empty string for "recorded, but zero".
        CloudContextPolicy.setOverrides(["units": 12], for: "r", defaults: defaults)
        XCTAssertNil(defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "r")))
    }

    func testTheThresholdIsEightyPercentOfAKnownWindow() {
        XCTAssertFalse(CloudContextPolicy.shouldCompact(promptTokens: 79_999, window: 100_000))
        XCTAssertTrue(CloudContextPolicy.shouldCompact(promptTokens: 80_000, window: 100_000))
        XCTAssertFalse(CloudContextPolicy.shouldCompact(promptTokens: 1_000_000, window: nil))
        XCTAssertFalse(CloudContextPolicy.shouldCompact(promptTokens: 1_000, window: 512))
    }

    func testTheSpecWindowWinsOverTheProviderFallback() {
        let defaults = scratchDefaults()
        defaults.set(64_000, forKey: CloudContextPolicy.globalWindowKey)
        let spec = CloudModelSpec(
            providerId: "p",
            providerModelId: "m",
            displayName: "m",
            supportsThinking: false,
            contextWindowTokens: 200_000
        )
        XCTAssertEqual(CloudContextPolicy.window(for: spec, defaults: defaults), 200_000)
        XCTAssertEqual(CloudContextPolicy.window(for: nil, defaults: defaults), 64_000)
        defaults.removeObject(forKey: CloudContextPolicy.globalWindowKey)
        XCTAssertNil(CloudContextPolicy.window(for: nil, defaults: defaults))
    }

    func testTheModelLookupAgreesWithTheRegistrySpec() throws {
        // The chat header resolves a window straight from the recorded
        // overrides so it does not rebuild the provider's whole registry on
        // every frame. The other path builds the spec through
        // `registeredSpecs`. This compares the two for real: a comment that
        // claimed they agree without calling the registry would pass even if
        // the spec stopped reading the override map.
        let defaults = scratchDefaults()
        let provider = CloudProvider(
            id: "p",
            displayName: "P",
            openAIBaseURL: URL(string: "https://p.test/v1")!
        )
        CloudModelRegistry.setRegisteredModelIds(["m", "other"], for: provider, defaults: defaults)
        CloudContextPolicy.setOverrides(
            ["m": 200_000, "other": 8_192],
            for: provider.id,
            defaults: defaults
        )
        let specs = CloudModelRegistry.registeredSpecs(for: provider, defaults: defaults)
        let spec = try XCTUnwrap(specs.first { $0.providerModelId == "m" })
        XCTAssertEqual(spec.contextWindowTokens, 200_000)
        XCTAssertEqual(CloudContextPolicy.window(for: spec, defaults: defaults), 200_000)
        XCTAssertEqual(
            CloudContextPolicy.window(forModelId: "m", providerId: "p", defaults: defaults),
            200_000
        )
        // A model with no recorded window falls through to the global default,
        // which is provider-agnostic by design.
        defaults.set(64_000, forKey: CloudContextPolicy.globalWindowKey)
        XCTAssertEqual(
            CloudContextPolicy.window(forModelId: "unrecorded", providerId: "p", defaults: defaults),
            64_000
        )
        XCTAssertEqual(
            CloudContextPolicy.window(for: nil, defaults: defaults),
            64_000
        )
        defaults.removeObject(forKey: CloudContextPolicy.globalWindowKey)
        XCTAssertNil(
            CloudContextPolicy.window(forModelId: "unrecorded", providerId: "p", defaults: defaults)
        )
        // A sub-plausible value is a units mistake, not a window.
        defaults.set(
            "m=12",
            forKey: CloudContextPolicy.defaultsKey(for: "p")
        )
        XCTAssertNil(CloudContextPolicy.window(forModelId: "m", providerId: "p", defaults: defaults))
    }

    func testPercentagesAndChunkBudgetsStaySane() {
        XCTAssertEqual(CloudContextPolicy.usagePercent(promptTokens: 64_000, window: 128_000), 50)
        XCTAssertNil(CloudContextPolicy.usagePercent(promptTokens: 64_000, window: nil))
        XCTAssertEqual(CloudContextPolicy.chunkCharacterBudget(window: 128_000), 168_960)
        // Unknown windows still get a usable chunk rather than zero.
        XCTAssertEqual(CloudContextPolicy.chunkCharacterBudget(window: nil), 24_000)
    }

    func testAnAbsurdWindowCannotTrapTheChunkBudget() {
        // The window is hand-typed. `Int(Double(window) * 0.33) * 4` trapped on
        // a value near Int.max; a budget that large is a units mistake, so it
        // clamps instead of crashing the /compact command.
        XCTAssertEqual(
            CloudContextPolicy.chunkCharacterBudget(window: Int.max),
            CloudContextPolicy.maximumChunkCharacters
        )
        XCTAssertEqual(
            CloudContextPolicy.chunkCharacterBudget(window: Int.max / 2),
            CloudContextPolicy.maximumChunkCharacters
        )
        XCTAssertLessThanOrEqual(
            CloudContextPolicy.summaryResponseTokens(window: Int.max),
            8_000
        )
        // The sensible range is untouched.
        XCTAssertEqual(CloudContextPolicy.chunkCharacterBudget(window: 1_048_576), 1_384_120)
    }

    func testTheSummarizerBudgetScalesWithTheChunk() {
        // A fixed 2,000-token cap is too small for a third-of-window chunk:
        // the summary stops at its limit, is billed, and is refused. Short
        // transcripts keep the cap they always had.
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: nil), 2_000)
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: 32_000), 2_000)
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: 200_000), 6_600)
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: 1_048_576), 8_000)
    }
}

final class CloudCompactionBoundaryTests: XCTestCase {
    func testTheKeptTailStartsAtAUserTurn() {
        let messages = [
            ChatMessage(role: .user, content: "one"),
            ChatMessage(role: .assistant, content: "a"),
            ChatMessage(role: .user, content: "two"),
            ChatMessage(role: .assistant, content: "b")
        ]
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 2), 2)
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 1), 2)
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 0), 4)
        XCTAssertNil(CloudCompactionBoundary.tailStart(
            messages: [ChatMessage(role: .assistant, content: "a")],
            keepRecentMessages: 1
        ))
        XCTAssertNil(CloudCompactionBoundary.tailStart(messages: [], keepRecentMessages: 1))
    }

    func testACompactedAgentRunCompactsAgain() throws {
        // The head is the system-role card and marker a previous compaction
        // left behind, and a long agent run has exactly ONE user turn — at
        // index 2, behind that head. Cutting there evicted the two system rows
        // and nothing else, so the second compaction reported a failure and
        // every later overflow was unrecoverable. The cut must fall through to
        // the requested boundary and fold away the run itself.
        var messages = [
            ChatMessage(role: .system, content: "card"),
            ChatMessage(role: .system, content: "marker"),
            ChatMessage(role: .user, content: "build the thing")
        ]
        for step in 0..<20 {
            messages.append(ChatMessage(role: .assistant, content: "step \(step)"))
        }
        XCTAssertEqual(messages.count, 23)
        XCTAssertEqual(
            CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 8),
            15
        )
        let plan = try XCTUnwrap(
            CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 8)
        )
        // The anchor rule pulls the run's question back into the tail, so the
        // evicted half is the two system rows plus twelve assistant steps. A
        // bound like `>= 2` would pass a regression that re-kept most of the
        // run and shrank nothing, which is the failure this case exists for.
        XCTAssertEqual(ChatSession.compactableMessageCount(in: plan.evicted), 12)
        XCTAssertEqual(plan.evicted.count, 14)
        // The run's question stays as the request's opening user turn.
        XCTAssertEqual(plan.kept.first?.content, "build the thing")
        XCTAssertEqual(plan.kept.count, 9)
        // Nothing is both summarized and kept.
        XCTAssertTrue(Set(plan.kept.map(\.id)).isDisjoint(with: Set(plan.evicted.map(\.id))))
    }

    func testAShortCompactedHistorySaysThereIsNothingToCompact() {
        // The only cut a compacted head offers is the two system rows. A plan
        // that evicts them rewrites the history for nothing and reports a
        // failure, so the honest answer is nil — not an unusable plan.
        let messages = [
            ChatMessage(role: .system, content: "card"),
            ChatMessage(role: .system, content: "marker"),
            ChatMessage(role: .user, content: "continue"),
            ChatMessage(role: .assistant, content: "ok")
        ]
        XCTAssertNil(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 8))
        XCTAssertNil(CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 8))
    }

    func testAPlanNeverFoldsAwayOnlySystemRows() throws {
        // The anchor rule can put back the only conversation row the cut had,
        // which leaves a plan whose summary replaces nothing.
        let messages = [
            ChatMessage(role: .user, content: "the only question"),
            ChatMessage(role: .assistant, content: "the only answer")
        ]
        let plan = CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 1)
        XCTAssertNil(plan)
        // The manual form still folds the whole conversation away.
        let manual = try XCTUnwrap(
            CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 0)
        )
        XCTAssertEqual(manual.evicted.count, 2)
        XCTAssertTrue(manual.kept.isEmpty)
    }

    func testAnAgentRunCutsAtTheRequestedBoundary() throws {
        // One user turn, then N assistant turns: an agent run has no later user
        // turn to back up to, so a user-turn-only rule always cut at index 0
        // and evicted nothing: the exact history auto-compaction exists to
        // shorten could never be compacted, and the turn ended at the context
        // wall instead.
        var messages = [ChatMessage(role: .user, content: "build the thing")]
        for step in 0..<12 {
            var turn = ChatMessage(role: .assistant, content: "step \(step)")
            turn.toolCalls = [ToolCallRecord(name: "shell_run", arguments: ["command": "ls"])]
            messages.append(turn)
        }
        XCTAssertEqual(messages.count, 13)
        // keep 4 -> cut 9 messages back, which lands inside the run.
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 4), 9)
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 12), 1)
        // What the bound actually promises is checked on the PLAN, not on the
        // cut: a cut is a message boundary by construction, so
        // `(1..<count).contains(cut)` could only fail at the ends and says
        // nothing about the tool exchange. Every result the evicted steps
        // emitted is evicted with its call, and every kept call keeps its
        // results — that is the rule the comment claims.
        let plan = try XCTUnwrap(
            CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 4)
        )
        XCTAssertEqual(plan.evicted.flatMap { $0.toolCalls ?? [] }.count, 8)
        XCTAssertEqual(plan.kept.flatMap { $0.toolCalls ?? [] }.count, 4)
        XCTAssertEqual(plan.kept.count, 5)
        XCTAssertEqual(plan.kept.first?.content, "build the thing")
        XCTAssertTrue(Set(plan.kept.map(\.id)).isDisjoint(with: Set(plan.evicted.map(\.id))))
    }

    func testTheCutStillPrefersAUserTurnWhenOneIsInRange() {
        // The agent-run fallback must not override the user-turn preference: a
        // conversation with several questions still cuts at the question.
        let messages = [
            ChatMessage(role: .user, content: "one"),
            ChatMessage(role: .assistant, content: "a"),
            ChatMessage(role: .assistant, content: "b"),
            ChatMessage(role: .user, content: "two"),
            ChatMessage(role: .assistant, content: "c"),
            ChatMessage(role: .assistant, content: "d")
        ]
        // Target 2 lands on an assistant turn; backing up finds the user turn
        // at 0, which is the head of the conversation, so the raw cut wins.
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 4), 2)
        // Target 1 backs up to no user turn before it, and the next user turn
        // is at 3. The raw cut is still the answer, and it evicts something.
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 5), 1)
        // A target that lands inside a run still backs up to the user turn that
        // precedes it when that turn is not the head of the conversation.
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 2), 3)
    }

    func testAnAgentRunKeepsItsQuestionSoTheRequestOpensWithAUserTurn() throws {
        // The cut the agent-run fallback chooses lands on an assistant turn,
        // so the plan has to bring the question back: a request whose first
        // message is an assistant turn is rejected by some endpoints, and the
        // question is what the evicted work answered.
        var messages = [ChatMessage(role: .user, content: "build the thing")]
        for step in 0..<12 {
            messages.append(ChatMessage(role: .assistant, content: "step \(step)"))
        }
        XCTAssertEqual(CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 4), 9)

        let plan = try XCTUnwrap(
            CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 4)
        )
        XCTAssertEqual(plan.kept.first?.role, .user)
        XCTAssertEqual(plan.kept.first?.content, "build the thing")
        XCTAssertEqual(plan.kept.count, 5)
        XCTAssertEqual(plan.evicted.count, 8)
        XCTAssertFalse(
            plan.evicted.contains { $0.id == messages[0].id },
            "a kept message must not be summarized as well"
        )
        XCTAssertEqual(plan.kept.map(\.id), [messages[0].id] + messages[9...].map(\.id))
    }

    func testAUserTurnAtTheCutIsKeptOnce() throws {
        let messages = [
            ChatMessage(role: .user, content: "one"),
            ChatMessage(role: .assistant, content: "a"),
            ChatMessage(role: .user, content: "two"),
            ChatMessage(role: .assistant, content: "b")
        ]
        let plan = try XCTUnwrap(
            CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 2)
        )
        XCTAssertEqual(plan.kept.map(\.content), ["two", "b"])
        XCTAssertEqual(plan.evicted.map(\.content), ["one", "a"])
    }

    func testAManualCompactionStillKeepsNothing() throws {
        let messages = [
            ChatMessage(role: .user, content: "one"),
            ChatMessage(role: .assistant, content: "a")
        ]
        let plan = try XCTUnwrap(
            CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 0)
        )
        XCTAssertTrue(plan.kept.isEmpty)
        XCTAssertEqual(plan.evicted.map(\.id), messages.map(\.id))
    }

    func testALongRunDoesNotPullTheCutBackToItsQuestion() throws {
        // [u1, a1, u2, ...200 assistant steps] with keep 8: the newest user
        // turn is 195 messages behind the requested cut. Following it there
        // evicted two rows, billed a summary and shrank nothing, so the cut
        // stays at the target.
        var messages = [
            ChatMessage(role: .user, content: "one"),
            ChatMessage(role: .assistant, content: "a"),
            ChatMessage(role: .user, content: "two")
        ]
        for step in 0..<200 {
            messages.append(ChatMessage(role: .assistant, content: "step \(step)"))
        }
        XCTAssertEqual(messages.count, 203)
        XCTAssertEqual(
            CloudCompactionBoundary.tailStart(messages: messages, keepRecentMessages: 8),
            195
        )
        let plan = try XCTUnwrap(
            CloudCompactionBoundary.plan(messages: messages, keepRecentMessages: 8)
        )
        XCTAssertEqual(plan.evicted.count, 194)
        XCTAssertEqual(plan.kept.count, 9)
        // The anchor rule still opens the kept tail with the run's question.
        XCTAssertEqual(plan.kept.first?.content, "two")
    }
}

// MARK: - Compaction outcomes

final class CloudCompactionFailureTests: XCTestCase {
    func testOnlyABilledFailureSpendsTheAttempt() {
        // Refusals that never reached the provider cost nothing and are not
        // evidence about the next step.
        for reason in [
            CloudCompactionFailure.conversationTooShort,
            .nothingToEvict,
            .conversationUnavailable,
            .noCredential("ignored"),
            .noTranscript
        ] {
            XCTAssertFalse(reason.spentRequest, "\(reason) sent no request")
        }
        for reason in [
            CloudCompactionFailure.emptySummary,
            .truncatedSummary,
            .cancelled,
            .provider("HTTP 429")
        ] {
            XCTAssertTrue(reason.spentRequest, "\(reason) was billed")
        }
        // A refusal that sent no request and depends on the history's shape
        // gives the attempt back, because a later step can change that shape.
        for reason in [
            CloudCompactionFailure.conversationTooShort,
            .nothingToEvict,
            .noTranscript
        ] {
            XCTAssertTrue(reason.isRetryableOnNextStep, "\(reason) is retryable")
        }
        for reason in [
            CloudCompactionFailure.conversationUnavailable,
            .noCredential("ignored"),
            .emptySummary,
            .truncatedSummary,
            .cancelled,
            .provider("HTTP 429")
        ] {
            XCTAssertFalse(reason.isRetryableOnNextStep, "\(reason) is not retryable")
        }
    }

    func testAPreflightRefusalDoesNotBlameTheProvider() {
        // "\(provider) could not summarize this conversation" was printed for
        // every refusal, including the ones that never sent a request. The two
        // bulletins and the empty transcript are the caller's own state.
        for reason in [
            CloudCompactionFailure.conversationTooShort,
            .nothingToEvict,
            .noTranscript
        ] {
            let notice = reason.notice(providerName: "Synthetic")
            XCTAssertTrue(notice.hasPrefix("ℹ️"), notice)
            XCTAssertFalse(notice.contains("Synthetic"), notice)
        }
    }

    func testAMissingKeySaysWhereTheKeyLives() {
        let failure = CloudTurnFailure.missingAPIKey(providerName: "Synthetic")
        let notice = CloudCompactionFailure.noCredential(failure.localizedDescription)
            .notice(providerName: "Synthetic")
        XCTAssertTrue(notice.contains("No API key is stored for Synthetic"), notice)
        XCTAssertTrue(notice.contains("Settings"), notice)
    }

    func testAProviderFaultIsNamedAsOne() {
        let notice = CloudCompactionFailure.provider("The provider returned HTTP 429.")
            .notice(providerName: "Synthetic")
        XCTAssertTrue(notice.hasPrefix("⚠️"))
        XCTAssertTrue(notice.contains("HTTP 429"))
        XCTAssertTrue(notice.hasSuffix("it was left unchanged."))
    }

    func testATruncatedSummaryIsRefusedWithItsReason() {
        let reason = CloudCompactionFailure.truncatedSummary
        XCTAssertTrue(reason.notice(providerName: "Synthetic").contains("output limit"))
        XCTAssertTrue(reason.clause(providerName: "Synthetic").contains("output limit"))
    }

    func testAStopReadsAsAnOutcomeRatherThanAFault() {
        let notice = CloudCompactionFailure.cancelled.notice(providerName: "Synthetic")
        XCTAssertTrue(notice.hasPrefix("⏹"), notice)
        XCTAssertTrue(notice.contains("left unchanged"), notice)
    }

    func testEveryClauseCanEndASentence() {
        // The overflow error embeds the clause: it must be one line and must
        // not carry the pill's glyph.
        for reason in [
            CloudCompactionFailure.conversationTooShort,
            .nothingToEvict,
            .conversationUnavailable,
            .noCredential("ignored"),
            .noTranscript,
            .emptySummary,
            .truncatedSummary,
            .cancelled,
            .provider("HTTP 400")
        ] {
            let clause = reason.clause(providerName: "Synthetic")
            XCTAssertFalse(clause.isEmpty)
            XCTAssertFalse(clause.hasPrefix("⚠️") || clause.hasPrefix("ℹ️") || clause.hasPrefix("⏹"))
            XCTAssertFalse(clause.contains("\n"), clause)
        }
    }

    func testASecondOverflowNamesTheCompactionItAlreadyRan() {
        // A run that compacted once and still overflowed used to be told the
        // history "could not be compacted automatically", which was true of no
        // attempt at all and false of this one.
        let alreadyRan = CloudOverflowAdvice.text(
            alreadyCompacted: true,
            failure: .nothingToEvict,
            providerName: "Synthetic"
        )
        XCTAssertTrue(alreadyRan.contains("already compacted older history once"), alreadyRan)
        XCTAssertFalse(alreadyRan.contains("could not be compacted"), alreadyRan)

        // A failed attempt names its cause instead of the generic line.
        let failed = CloudOverflowAdvice.text(
            alreadyCompacted: false,
            failure: .nothingToEvict,
            providerName: "Synthetic"
        )
        XCTAssertTrue(failed.contains("left the history unchanged"), failed)
        XCTAssertTrue(failed.contains("no older history to fold away"), failed)

        // No attempt at all keeps the original advice.
        let untouched = CloudOverflowAdvice.text(
            alreadyCompacted: false,
            failure: nil,
            providerName: "Synthetic"
        )
        XCTAssertTrue(untouched.contains("could not be compacted automatically"), untouched)
    }

    func testAReplyThatHitItsLimitIsDistinguishable() {
        let complete = CloudTextCompletion(
            text: "done",
            usage: nil,
            finishReason: "stop"
        )
        XCTAssertFalse(complete.stoppedAtOutputLimit)
        let truncated = CloudTextCompletion(
            text: "half a sent",
            usage: nil,
            finishReason: "length"
        )
        XCTAssertTrue(truncated.stoppedAtOutputLimit)
        XCTAssertTrue(CloudTextCompletion(
            text: "half a sent",
            usage: nil,
            finishReason: "max_tokens"
        ).stoppedAtOutputLimit)
        // A provider that reports no reason is not assumed to have truncated.
        XCTAssertFalse(CloudTextCompletion(text: "done", usage: nil).stoppedAtOutputLimit)
    }
}

// MARK: - Summarizer completions

/// A transport with a scripted stream, for the terminal-event rules a real
/// provider cannot be asked to reproduce on demand.
private nonisolated final class ScriptedCloudService: CloudChatServicing, @unchecked Sendable {
    private let script: @Sendable (AsyncStream<CloudTurnEvent>.Continuation) -> Void

    init(script: @escaping @Sendable (AsyncStream<CloudTurnEvent>.Continuation) -> Void) {
        self.script = script
    }

    func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]?
    ) -> AsyncStream<CloudTurnEvent> {
        AsyncStream { continuation in
            script(continuation)
        }
    }

    func fetchModelIds() async throws -> [String] { [] }
}

final class CloudTextCompletionTests: XCTestCase {
    private func client(_ script: @escaping @Sendable (AsyncStream<CloudTurnEvent>.Continuation) -> Void) -> CloudChatClient {
        CloudChatClient(
            provider: CloudProvider(
                id: "p",
                displayName: "Synthetic",
                openAIBaseURL: URL(string: "https://example.invalid/v1")!
            ),
            service: ScriptedCloudService(script: script)
        )
    }

    private func complete(_ client: CloudChatClient) async throws -> CloudTextCompletion {
        try await client.completeText(
            model: "m",
            systemPrompt: "summarize",
            userPrompt: "the chunk",
            maxTokens: 2_000,
            sessionKey: nil
        )
    }

    func testAFinishedStreamReturnsItsTextUsageAndReason() async throws {
        let usage = CloudUsage(promptTokens: 100, completionTokens: 20)
        let client = client { continuation in
            continuation.yield(.started)
            continuation.yield(.contentDelta("a summary"))
            continuation.yield(.finished(CloudTurnResult(
                text: "a summary",
                thinking: "",
                finishReason: "stop",
                usage: usage
            )))
            continuation.finish()
        }
        let completion = try await complete(client)
        XCTAssertEqual(completion.text, "a summary")
        XCTAssertEqual(completion.usage, usage)
        XCTAssertEqual(completion.finishReason, "stop")
        XCTAssertFalse(completion.stoppedAtOutputLimit)
    }

    func testAStreamThatEndsWithoutATerminalEventThrows() async {
        // A provider that closes the stream mid-reply must not be read as
        // having finished one: the fragment would replace the messages a
        // compaction is about to delete.
        let client = client { continuation in
            continuation.yield(.started)
            continuation.yield(.contentDelta("half a sent"))
            continuation.finish()
        }
        do {
            let completion = try await complete(client)
            XCTFail("expected a refusal, got \(completion.text)")
        } catch let failure as CloudTurnFailure {
            guard case .invalidResponse = failure else {
                return XCTFail("unexpected failure: \(failure)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testACancelledSummaryThrowsInsteadOfReturningItsFragment() async {
        // `for await` over an AsyncStream ends the moment its task is
        // cancelled, with no event of any kind. Stop used to be indistinguishable
        // from a finished summary, so a half-written one was committed.
        let client = client { continuation in
            continuation.yield(.started)
            continuation.yield(.contentDelta("half a sent"))
            let held = Task {
                try? await Task.sleep(nanoseconds: 30_000_000_000)
            }
            continuation.onTermination = { _ in held.cancel() }
        }
        let task = Task { try await complete(client) }
        try? await Task.sleep(nanoseconds: 200_000_000)
        task.cancel()
        do {
            let completion = try await task.value
            XCTFail("expected cancellation, got \(completion.text)")
        } catch is CancellationError {
            // The expected outcome.
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testAProviderFailureIsRaisedRatherThanReturnedAsText() async {
        let client = client { continuation in
            continuation.yield(.started)
            continuation.yield(.contentDelta("half"))
            continuation.yield(.failed(.httpStatus(code: 429, summary: "slow down")))
            continuation.finish()
        }
        do {
            _ = try await complete(client)
            XCTFail("expected the provider failure to be raised")
        } catch let failure as CloudTurnFailure {
            XCTAssertEqual(failure, .httpStatus(code: 429, summary: "slow down"))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }
}

// MARK: - Context odometer

final class CloudContextOdometerTests: XCTestCase {
    func testTheFigureTracksTheLastRequestRatherThanTheSumOfSteps() {
        // A cloud run re-sends the whole conversation on every step. Summing
        // the prompts (what a provider with no cache split bills) would read as
        // several conversations; the last request IS the context.
        let stepPrompts = [1_000, 1_040, 1_100, 1_150]
        let total = CloudContextOdometer.total(promptTokens: stepPrompts.last!, generatedTokens: 40)
        XCTAssertEqual(total, 1_190)
        // The four prompts summed is 4,290. Naming it keeps the intent in the
        // assertion instead of restating the addition above.
        XCTAssertNotEqual(total, 4_290)
        // A hostile usage block can carry `Int.max` in the prompt column; the
        // addition must clamp rather than trap.
        XCTAssertEqual(CloudContextOdometer.total(promptTokens: .max, generatedTokens: 1), .max)
    }

    func testNegativeCountsAreClampedRatherThanDisplayed() {
        XCTAssertEqual(CloudContextOdometer.total(promptTokens: -5, generatedTokens: 0), 0)
        XCTAssertEqual(CloudContextOdometer.total(promptTokens: 10, generatedTokens: -7), 10)
    }
}

// MARK: - Pricing

final class CloudPricingTests: XCTestCase {
    func testCachedInputIsDiscountedAndAnUnknownRateFallsBackToInput() throws {
        let usage = CloudUsage(promptTokens: 1_000_000, completionTokens: 100_000, cachedTokens: 800_000)
        let priced = CloudPricing(input: 2, cachedInput: 0.5, output: 10)
        // 200k fresh * $2 + 800k cached * $0.50 + 100k out * $10.
        XCTAssertEqual(try XCTUnwrap(priced.cost(for: usage)), 1.8, accuracy: 1e-9)
        let noCacheRate = CloudPricing(input: 2, output: 10)
        XCTAssertEqual(try XCTUnwrap(noCacheRate.cost(for: usage)), 3.0, accuracy: 1e-9)
    }

    func testNoPromptCountMeansNoCost() {
        let usage = CloudUsage(completionTokens: 10)
        XCTAssertNil(CloudPricing(input: 1, output: 1).cost(for: usage))
    }

    func testACacheWritePremiumIsPricedWhenRecorded() throws {
        // Anthropic bills a cache write at 1.25x input. Charging it at the
        // fresh rate understates that column; recording the rate fixes it.
        let usage = CloudUsage(
            promptTokens: 1_000,
            completionTokens: 0,
            cachedTokens: 400,
            cacheWriteTokens: 100
        )
        let priced = CloudPricing(input: 3, cachedInput: 0.3, output: 15, cacheWriteInput: 3.75)
        // 500 fresh * $3 + 400 cached * $0.30 + 100 written * $3.75.
        XCTAssertEqual(
            try XCTUnwrap(priced.cost(for: usage)),
            (1_500 + 120 + 375) / 1_000_000,
            accuracy: 1e-12
        )
        // Without the field the write costs what a fresh token costs.
        let withoutPremium = CloudPricing(input: 3, cachedInput: 0.3, output: 15)
        XCTAssertEqual(
            try XCTUnwrap(withoutPremium.cost(for: usage)),
            (1_500 + 120 + 300) / 1_000_000,
            accuracy: 1e-12
        )
    }

    func testACacheWriteCannotBeChargedBeyondWhatTheCacheReadLeft() throws {
        // A hostile or buggy usage block can claim more written tokens than the
        // prompt has left after the cache read. The clamp is
        // `min(written, prompt - cached)`, so 900 written over a 1,000 prompt
        // with 400 cached can only bill 600 — and the fresh share is zero, not
        // negative (which would discount the turn).
        let usage = CloudUsage(
            promptTokens: 1_000,
            completionTokens: 0,
            cachedTokens: 400,
            cacheWriteTokens: 900
        )
        let priced = CloudPricing(input: 3, cachedInput: 0.3, output: 15, cacheWriteInput: 3.75)
        // 400 cached * $0.30 + 600 written * $3.75, all per million tokens.
        XCTAssertEqual(
            try XCTUnwrap(priced.cost(for: usage)),
            (120 + 2_250) / 1_000_000,
            accuracy: 1e-12
        )
        // The same block with no cache-write rate bills the clamp at input.
        let withoutPremium = CloudPricing(input: 3, cachedInput: 0.3, output: 15)
        XCTAssertEqual(
            try XCTUnwrap(withoutPremium.cost(for: usage)),
            (120 + 1_800) / 1_000_000,
            accuracy: 1e-12
        )
    }

    func testANegativeRateCannotProduceANegativeSpend() throws {
        let usage = CloudUsage(promptTokens: 1_000, completionTokens: 100)
        let negative = CloudPricing(input: -2, output: 10)
        // Only the output column is billed: 100 * $10/M. Comparing against a
        // second `CloudPricing` would pass for any shared bug in the pricing.
        XCTAssertEqual(
            try XCTUnwrap(negative.cost(for: usage)),
            0.001,
            accuracy: 1e-12
        )
    }

    func testOneBadPriceEntryDoesNotDisableTheWholeList() {
        let suite = "CloudPricingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        defaults.set(
            #"{"good":{"input":1,"output":2},"typo":{"input":"cheap","output":2},"negative":{"input":-1,"output":2}}"#,
            forKey: CloudPricingStore.defaultsKey(for: "p")
        )
        let prices = CloudPricingStore.pricing(for: "p", defaults: defaults)
        XCTAssertEqual(prices["good"], CloudPricing(input: 1, output: 2))
        XCTAssertNil(prices["typo"])
        XCTAssertNil(prices["negative"])
    }

    func testTheStoreRoundTripsAndIgnoresGarbage() {
        let suite = "CloudPricingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        CloudPricingStore.setPricing(
            ["m": CloudPricing(input: 0.27, cachedInput: 0.07, output: 1.1)],
            for: "p",
            defaults: defaults
        )
        XCTAssertEqual(
            CloudPricingStore.pricing(forModel: "m", providerId: "p", defaults: defaults),
            CloudPricing(input: 0.27, cachedInput: 0.07, output: 1.1)
        )
        defaults.set("{not json", forKey: CloudPricingStore.defaultsKey(for: "p"))
        XCTAssertTrue(CloudPricingStore.pricing(for: "p", defaults: defaults).isEmpty)
    }

    func testSmallAmountsStayVisible() {
        XCTAssertEqual(CloudPricingStore.formatted(0.00012), "$0.0001")
        XCTAssertEqual(CloudPricingStore.formatted(1.234), "$1.23")
        XCTAssertEqual(CloudPricingStore.formatted(0), "$0")
        // A cost too small to render at four places used to print "$0.0000",
        // which reads as free on a turn that was billed.
        XCTAssertEqual(CloudPricingStore.formatted(0.00004), "<$0.0001")
        // 0.009999 rounds to "$0.0100", claiming a cent it is under.
        XCTAssertEqual(CloudPricingStore.formatted(0.009999), "<$0.0100")
        XCTAssertEqual(CloudPricingStore.formatted(0.0094), "$0.0094")
        XCTAssertEqual(CloudPricingStore.formatted(-1), "$0")
        XCTAssertEqual(CloudPricingStore.formatted(.nan), "$0")
    }
}

final class CloudTurnReportTests: XCTestCase {
    func testTheSuffixNamesCachedTokensAndCost() {
        let report = CloudTurnReport(
            usage: CloudUsage(promptTokens: 1204, completionTokens: 56, cachedTokens: 1000),
            estimatedPromptTokens: 999,
            outputTokens: 50,
            cost: 0.0031
        )
        XCTAssertEqual(report.statusSuffix, "1,204 in (1,000 cached) / 56 out · $0.0031")
    }

    func testAMissingCompletionCountIsMarkedAsEstimated() {
        // The output figure falls back to the character estimate, so the
        // suffix has to say so even though the prompt count was reported.
        let report = CloudTurnReport(
            usage: CloudUsage(promptTokens: 1204, completionTokens: nil),
            estimatedPromptTokens: 0,
            outputTokens: 50,
            cost: nil
        )
        XCTAssertTrue(report.wasEstimated)
        XCTAssertEqual(report.outputTokens, 50)
        XCTAssertEqual(report.statusSuffix, "1,204 in / 50 out (est.)")
    }

    func testAnEndpointThatDropsUsageIsMarkedAsEstimated() {
        let report = CloudTurnReport(
            usage: nil,
            estimatedPromptTokens: 1200,
            outputTokens: 40,
            cost: nil
        )
        XCTAssertTrue(report.wasEstimated)
        XCTAssertEqual(report.statusSuffix, "1,200 in / 40 out (est.)")
    }

    func testARunReportsEveryBilledStepsTotals() {
        // A 20-step agent run is 20 billed requests; the summary must match the
        // session's cumulative spend, not the last step's usage block.
        let report = CloudTurnReport(
            promptTokens: 24_000,
            cachedTokens: 18_000,
            outputTokens: 350,
            cost: 0.0031,
            wasEstimated: false
        )
        XCTAssertEqual(report.statusSuffix, "24,000 billed in (18,000 of it cached) / 350 out · $0.0031")
    }

    func testARunWithOneEstimatedStepIsStillMarked() {
        let report = CloudTurnReport(
            promptTokens: 24_000,
            cachedTokens: 0,
            outputTokens: 350,
            cost: nil,
            wasEstimated: true
        )
        XCTAssertEqual(report.statusSuffix, "24,000 billed in / 350 out (est.)")
    }
}

// MARK: - Session spend

final class ChatSessionCostPersistenceTests: XCTestCase {
    func testReportedSpendSurvivesARoundTrip() throws {
        let session = ChatSession(title: "priced", cumulativeCost: 0.1234)
        let data = try JSONEncoder().encode(session)
        let decoded = try JSONDecoder().decode(ChatSession.self, from: data)
        XCTAssertEqual(try XCTUnwrap(decoded.cumulativeCost), 0.1234, accuracy: 1e-9)
    }

    func testASessionWrittenBeforeTheFieldDecodesWithNoCost() throws {
        let session = ChatSession(title: "legacy", cumulativeCost: 0.5)
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: try JSONEncoder().encode(session)) as? [String: Any]
        )
        object.removeValue(forKey: "cumulativeCost")
        let data = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(ChatSession.self, from: data)
        XCTAssertNil(decoded.cumulativeCost)
    }
}
