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

    func testStringErrorPayloadAndFalseErrorFlagAreHandledLeniently() throws {
        var parser = CloudStreamParser()
        XCTAssertThrowsError(try parser.consume(
            dataPayload: #"{"error":"rate limited"}"#
        )) { error in
            guard case CloudTurnFailure.invalidResponse(let message) = error else {
                return XCTFail("expected invalidResponse, got \(error)")
            }
            XCTAssertTrue(message.contains("rate limited"))
        }

        // A false/empty error marker alongside deltas must not discard the reply.
        let events = try parser.consume(
            dataPayload: #"{"choices":[{"delta":{"content":"still alive"}}],"error":false}"#
        )
        XCTAssertEqual(events, [.contentDelta("still alive")])
        XCTAssertEqual(parser.text, "still alive")
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
        XCTAssertTrue(CloudAgentStepPlan.shouldExecuteToolCalls(
            offeredToolCount: 3, callCount: 1, streamTerminated: true
        ))
        XCTAssertFalse(CloudAgentStepPlan.shouldExecuteToolCalls(
            offeredToolCount: 0, callCount: 2, streamTerminated: true
        ))
        XCTAssertFalse(CloudAgentStepPlan.shouldExecuteToolCalls(
            offeredToolCount: 1, callCount: 0, streamTerminated: true
        ))
    }

    func testACutOffStreamRunsNoneOfItsCalls() {
        // A stream that closed without `[DONE]` or a finish reason proves
        // nothing about whether the model finished writing its call set, and
        // the parser cannot tell a cut-off call from a whole one. The write it
        // did emit is irreversible, so the step runs nothing and the turn ends
        // with the cut-off notice.
        XCTAssertFalse(CloudAgentStepPlan.shouldExecuteToolCalls(
            offeredToolCount: 4, callCount: 1, streamTerminated: false
        ))
        XCTAssertFalse(CloudAgentStepPlan.shouldExecuteToolCalls(
            offeredToolCount: 4, callCount: 0, streamTerminated: false
        ))
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

    func testToolSchemasArePartOfTheEstimate() {
        // The provider serialises every schema into the prompt, so a loaded
        // agent toolset is prompt the pre-flight has to see. The old estimate
        // counted only text and call arguments, which read a tool-heavy run at
        // 80% of the window while the real prompt was already past it.
        let messages = [CloudChatMessage(role: "user", content: "list the files")]
        let withoutTools = CloudMessageAssembler.approximatePromptTokens(system: nil, messages: messages)
        let schema: [String: AnyCodable] = [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "path": [
                    "type": "string",
                    "description": "Absolute path to search. Must be inside the workspace root."
                ]
            ])
        ]
        let tools = [
            ToolDefinition(
                name: "find_files",
                description: "Find files by name under a directory. Returns every match in a sorted list.",
                parameters: schema
            )
        ]
        let withTools = CloudMessageAssembler.approximatePromptTokens(
            system: nil,
            messages: messages,
            tools: tools
        )
        XCTAssertGreaterThan(withTools, withoutTools + 10)
        // The default argument keeps every pre-existing call site measuring the
        // same thing it did before.
        XCTAssertEqual(
            CloudMessageAssembler.approximatePromptTokens(system: nil, messages: messages),
            withoutTools
        )
        XCTAssertEqual(
            CloudMessageAssembler.approximatePromptTokens(system: nil, messages: messages, tools: []),
            withoutTools
        )
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
    /// The affinity headers each request carried, `""` when absent.
    nonisolated(unsafe) static var affinityHeaders: [String] = []
    nonisolated(unsafe) static var sessionIdHeaders: [String] = []
    /// Runs while a request is in flight, before its scripted reply. A test
    /// uses it to act on shared state MID-REQUEST, which is the only way to
    /// reproduce a latch closing between a send and its response.
    nonisolated(unsafe) static var onRequest: ((URLRequest) -> Void)?

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
        Self.affinityHeaders.append(request.value(forHTTPHeaderField: "x-session-affinity") ?? "")
        Self.sessionIdHeaders.append(request.value(forHTTPHeaderField: "x-session-id") ?? "")
        let reply = Self.responses.isEmpty
            ? (status: Self.status, body: Self.body)
            : Self.responses.removeFirst()
        Self.onRequest?(request)
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
        hints: CloudCacheHints = .openAI,
        cacheHintMemory: CloudCacheHintMemory = .shared
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
            cacheHints: hints,
            cacheHintMemory: cacheHintMemory
        )
    }

    private func collect(
        _ service: OpenAICloudChatService,
        sessionKey: String? = nil,
        receipt: CloudTurnReceipt? = nil
    ) async -> [CloudTurnEvent] {
        var events: [CloudTurnEvent] = []
        for await event in service.streamTurn(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "hi")],
            params: CloudChatGenerationParams(sessionKey: sessionKey),
            tools: nil,
            receipt: receipt
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
        CloudStubURLProtocol.affinityHeaders = []
        CloudStubURLProtocol.sessionIdHeaders = []
        CloudStubURLProtocol.onRequest = nil
    }

    func testAFinishReasonAloneCountsAsTerminal() async {
        // Not every endpoint sends `[DONE]`; a non-nil `finish_reason` is the
        // other marker the protocol defines. The hand-built result this used to
        // assert (`terminated: true` written by the test itself) tested nothing
        // about the transport's rule, so this drives a real SSE body with no
        // sentinel and lets `streamTurn` decide.
        CloudStubURLProtocol.body = Data(
            """
            data: {"choices":[{"delta":{"content":"a summary"},"finish_reason":"stop"}]}

            """.utf8
        )
        let events = await collect(makeService())
        guard case .finished(let result)? = events.last else {
            return XCTFail("expected a finished event, got \(String(describing: events.last))")
        }
        XCTAssertEqual(result.text, "a summary")
        XCTAssertEqual(result.finishReason, "stop")
        XCTAssertTrue(result.terminated, "finish_reason is the protocol's other end marker")
    }

    func testACleanEOFWithoutAnyMarkerIsNotTerminal() async {
        // The negative case of the same rule: a body that carries text and
        // closes without `[DONE]` and without a `finish_reason` is a prefix,
        // and a compaction that committed it would delete messages the summary
        // never covered.
        CloudStubURLProtocol.body = Data(
            "data: {\"choices\":[{\"delta\":{\"content\":\"half a sent\"}}]}\n\n".utf8
        )
        let events = await collect(makeService())
        guard case .finished(let result)? = events.last else {
            return XCTFail("expected a finished event, got \(String(describing: events.last))")
        }
        XCTAssertEqual(result.text, "half a sent")
        XCTAssertNil(result.finishReason)
        XCTAssertFalse(result.terminated, "a clean EOF is not a finished answer")
    }

    func testARejectedAffinityHeaderIsRetriedWithoutItAndLatched() async {
        // Affinity is the DEFAULT hint and is sent to providers configured
        // before the setting existed, so an endpoint that rejects unknown
        // headers used to fail every turn for a hint nobody chose. The retry is
        // earned only by a body that NAMES a header, which is what makes it a
        // targeted fix rather than a blind re-upload.
        let ok = Data("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".utf8)
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"unexpected header x-session-affinity"}"#.utf8)),
            (200, ok),
            // Second turn: the latch is closed, so this turn's first attempt
            // already carries no affinity headers.
            (200, ok)
        ]
        // A fresh endpoint per invocation keeps the process-wide latch out of
        // the picture, so a repeat-on-failure rerun starts clean.
        let host = "affinity-retry-\(UUID().uuidString).test"
        let endpoint = "https://\(host)/openai/v1/chat/completions"
        let memory = CloudCacheHintMemory()
        let service = makeService(host: host, hints: .affinity, cacheHintMemory: memory)
        defer { memory.allowCacheHints(endpoint: endpoint) }

        guard case .finished? = await collect(service, sessionKey: "affinity-a").last else {
            return XCTFail("the retry should succeed without the headers")
        }
        XCTAssertEqual(CloudStubURLProtocol.affinityHeaders, ["affinity-a", ""])
        XCTAssertEqual(
            CloudStubURLProtocol.sessionIdHeaders,
            ["affinity-a", "affinity-a"],
            "rejecting x-session-affinity alone must keep x-session-id on the retry"
        )
        XCTAssertFalse(
            memory.acceptsAffinityHeader(.sessionAffinity, endpoint: endpoint),
            "the accepted retry should have latched x-session-affinity off"
        )
        XCTAssertTrue(
            memory.acceptsAffinityHeader(.sessionId, endpoint: endpoint),
            "x-session-id was never rejected and must stay enabled"
        )
        XCTAssertFalse(
            memory.acceptsAffinityHeaders(endpoint: endpoint),
            "acceptsAffinityHeaders reflects that at least one header was latched off"
        )

        guard case .finished? = await collect(service, sessionKey: "affinity-a").last else {
            return XCTFail("the second turn should succeed on its first attempt")
        }
        XCTAssertEqual(
            CloudStubURLProtocol.affinityHeaders,
            ["affinity-a", "", ""],
            "the latch is what makes the second turn's first attempt free of x-session-affinity"
        )
        XCTAssertEqual(
            CloudStubURLProtocol.sessionIdHeaders,
            ["affinity-a", "affinity-a", "affinity-a"],
            "x-session-id stays active on subsequent turns"
        )

        // Forget Refusals re-tries both headers and, if x-session-affinity is
        // still rejected, re-latches only x-session-affinity while keeping x-session-id.
        memory.allowCacheHints(endpoint: endpoint)
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"unexpected header x-session-affinity"}"#.utf8)),
            (200, ok)
        ]
        guard case .finished? = await collect(service, sessionKey: "affinity-a").last else {
            return XCTFail("clearing refusals should recover cleanly while preserving x-session-id")
        }
        XCTAssertEqual(CloudStubURLProtocol.affinityHeaders, ["affinity-a", "", "", "affinity-a", ""])
        XCTAssertEqual(CloudStubURLProtocol.sessionIdHeaders, ["affinity-a", "affinity-a", "affinity-a", "affinity-a", "affinity-a"])
        XCTAssertTrue(memory.acceptsAffinityHeader(.sessionId, endpoint: endpoint))
        XCTAssertFalse(memory.acceptsAffinityHeader(.sessionAffinity, endpoint: endpoint))
    }

    func testAnUnrelatedRejectionDoesNotDropTheAffinityHeaders() {
        // The predicate is the whole safety argument: a 400 that does not name a
        // header could be about the model, the tools or the prompt, and
        // re-uploading the same megabytes without a header it never mentioned
        // would fail identically.
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutAffinity(
            status: 400,
            body: "unexpected header x-session-affinity"
        ))
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutAffinity(
            status: 422,
            body: "header x-session-id is not allowed"
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutAffinity(
            status: 400,
            body: "invalid request"
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutAffinity(
            status: 401,
            body: "x-session-affinity"
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutAffinity(
            status: 500,
            body: "x-session-affinity"
        ))
    }

    func testAnEmptySessionKeyEarnsNoAffinityRetryAndNoLatch() async {
        // `applyCacheAffinity` sends no header for an empty/missing key, so a
        // body that names one cannot be about a header this request carried.
        // Retrying would re-upload the prompt for nothing, and latching would
        // switch affinity off for headers that were never sent.
        let host = "affinity-empty-\(UUID().uuidString).test"
        let endpoint = "https://\(host)/openai/v1/chat/completions"
        let memory = CloudCacheHintMemory()
        let service = makeService(host: host, hints: .affinity, cacheHintMemory: memory)
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"unexpected header x-session-affinity"}"#.utf8))
        ]
        guard case .failed? = await collect(service, sessionKey: nil).last else {
            return XCTFail("the turn should fail")
        }
        XCTAssertEqual(
            CloudStubURLProtocol.requestBodies.count,
            1,
            "the rejected request carried no affinity header, so there was nothing to drop and retry"
        )
        XCTAssertTrue(
            memory.acceptsAffinityHeaders(endpoint: endpoint),
            "nothing left the Mac to be refused"
        )
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

    func testATrailingSentinelWithNoNewlineStillCountsAsTerminal() async {
        // The final line of a stream reaches the parser through
        // `lineBuffer.flush()`, not through the line splitter. Its answer used
        // to be discarded, so a `[DONE]` that arrived without the blank line
        // after it left the sentinel flag unset: a complete answer was reported
        // as a connection that closed early (a spurious "cut off" row), and a
        // compaction — which refuses a non-terminal stream — refused its own
        // finished summary.
        CloudStubURLProtocol.body = Data(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]".utf8
        )
        let events = await collect(makeService())
        guard case .finished(let result)? = events.last else {
            return XCTFail("expected a finished event, got \(String(describing: events.last))")
        }
        XCTAssertEqual(result.text, "ok")
        XCTAssertTrue(result.terminated, "a trailing [DONE] is the protocol's end marker")
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
        // A fresh endpoint AND a private latch: the case must neither inherit
        // another run's refusal nor leave one behind for the app's requests.
        let host = "two-stage-\(UUID().uuidString).test"
        let memory = CloudCacheHintMemory()
        let service = makeService(host: host, cacheHintMemory: memory)
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
                endpoint: "https://\(host)/openai/v1/chat/completions",
                model: "hf:org/model",
                memory: memory
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
        // withheld and fail its first assertion. A private latch is the
        // isolation, and it keeps this case from leaving state behind for the
        // app's own requests.
        let memory = CloudCacheHintMemory()
        let service = makeService(
            host: "cache-latch-\(UUID().uuidString).test",
            hints: .openAI,
            cacheHintMemory: memory
        )
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

    func testAnUnrelatedRejectionIsNotRetriedAndDoesNotLatch() async {
        // Every attempt fails on a model error, so the optional field cannot be
        // the problem. Re-uploading the prompt to prove that was a wasted
        // request, and latching on it would switch the hint off for a server
        // that never objected to it.
        let failure = Data(#"{"error":"model not found"}"#.utf8)
        CloudStubURLProtocol.responses = [(400, failure)]
        let memory = CloudCacheHintMemory()
        let service = makeService(
            host: "cache-kept-\(UUID().uuidString).test",
            hints: .openAI,
            cacheHintMemory: memory
        )
        guard case .failed? = await collect(service, sessionKey: "session-b").last else {
            return XCTFail("the first turn should fail")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 1)

        CloudStubURLProtocol.responses = [(400, failure)]
        guard case .failed? = await collect(service, sessionKey: "session-b").last else {
            return XCTFail("the second turn should fail")
        }
        // The turn still ASKED for the cache: nothing was latched.
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 2)
        XCTAssertNotNil(requestBody(0)?["prompt_cache_key"])
        XCTAssertNotNil(requestBody(1)?["prompt_cache_key"])
    }

    func testAToolSchemaRejectionIsNotRetriedWithoutTheCacheKey() {
        // The upload is the whole prompt. A body about the tools would fail
        // identically the second time, and the chat path has its own advice for
        // it (turn Agent Tools off, or register the no-tools key).
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: #"{"error":"tools are not supported for this model"}"#
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 422,
            body: #"{"error":"function calling is not enabled"}"#
        ))
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: #"{"error":"unsupported parameter: prompt_cache_key"}"#
        ))
    }

    func testACacheKeyRejectionIsRetriedEvenIfAnotherRequestLatchesMeanwhile() async {
        // The latch is process-wide. Re-reading it after the send let a
        // concurrent request declare the endpoint indifferent in between, which
        // suppressed the retry this request had earned and failed the turn over
        // a field it had, in fact, sent. The decision is taken before the send
        // and reused, so the retry happens regardless.
        let endpointHost = "latch-race-\(UUID().uuidString).test"
        let endpoint = "https://\(endpointHost)/openai/v1/chat/completions"
        let ok = Data("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".utf8)
        // A private latch shared by the stub and the service: the callback
        // mimicking a concurrent request must close the SAME latch this one
        // reads.
        let memory = CloudCacheHintMemory()
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"unexpected property prompt_cache_key"}"#.utf8)),
            (200, ok)
        ]
        CloudStubURLProtocol.onRequest = { _ in
            if CloudStubURLProtocol.requestBodies.count == 1 {
                memory.declinePromptCacheKey(endpoint: endpoint, model: "hf:org/model")
            }
        }
        defer {
            CloudStubURLProtocol.onRequest = nil
            memory.allowCacheHints(endpoint: endpoint)
        }
        let service = makeService(host: endpointHost, cacheHintMemory: memory)
        guard case .finished? = await collect(service, sessionKey: "session-race").last else {
            return XCTFail("the retry should succeed even though the latch closed meanwhile")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 2)
        XCTAssertNotNil(requestBody(0)?["prompt_cache_key"])
        XCTAssertNil(requestBody(1)?["prompt_cache_key"])
    }

    func testAFailedRetryDoesNotLatchTheCacheKey() async {
        // The latch closes only when a request WITHOUT the field is ACCEPTED.
        // A retry that fails too proves nothing about the field — the endpoint
        // may have been down, or complaining about something else — so the next
        // turn must still ask for the cache.
        let host = "cache-latch-failed-\(UUID().uuidString).test"
        let endpoint = "https://\(host)/openai/v1/chat/completions"
        let memory = CloudCacheHintMemory()
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"unexpected property prompt_cache_key"}"#.utf8)),
            (500, Data(#"{"error":"upstream exploded"}"#.utf8))
        ]
        let service = makeService(host: host, cacheHintMemory: memory)
        guard case .failed? = await collect(service, sessionKey: "session-failed-latch").last else {
            return XCTFail("the retry should fail too")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 2)
        XCTAssertNil(requestBody(1)?["prompt_cache_key"])
        XCTAssertTrue(
            memory.acceptsPromptCacheKey(endpoint: endpoint, model: "hf:org/model"),
            "a retry that did not succeed is not proof the field was the problem"
        )

        // And the next turn still sends it.
        CloudStubURLProtocol.responses = [(
            200,
            Data("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".utf8)
        )]
        guard case .finished? = await collect(service, sessionKey: "session-failed-latch").last else {
            return XCTFail("the next turn should succeed on its first attempt")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 3)
        XCTAssertNotNil(requestBody(2)?["prompt_cache_key"])
    }

    func testAVague400IsRetriedButDoesNotLatch() async {
        // The retry is free (a rejected request is not billed), so a vague 400
        // earns one attempt without the field. The LATCH is what must not
        // happen: the body never named `prompt_cache_key`, so the accepted
        // retry proves only that the endpoint accepted the second request.
        let host = "cache-latch-vague-\(UUID().uuidString).test"
        let endpoint = "https://\(host)/openai/v1/chat/completions"
        let memory = CloudCacheHintMemory()
        CloudStubURLProtocol.responses = [
            (400, Data(#"{"error":"invalid request"}"#.utf8)),
            (200, Data("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".utf8))
        ]
        let service = makeService(host: host, cacheHintMemory: memory)
        guard case .finished? = await collect(service, sessionKey: "session-vague").last else {
            return XCTFail("the retry should succeed")
        }
        XCTAssertEqual(CloudStubURLProtocol.requestBodies.count, 2)
        XCTAssertNil(requestBody(1)?["prompt_cache_key"])
        XCTAssertTrue(
            memory.acceptsPromptCacheKey(endpoint: endpoint, model: "hf:org/model"),
            "a vague 400 never named the field, so nothing could be latched"
        )
    }

    func testAParameterRefusalIsNotRetriedWithoutTheCacheKey() {
        // Re-uploading the whole prompt (base64 images and all) to drop a field
        // the body never mentioned fails identically and pays the upload per
        // agent step. The sampling parameters this app sends are named here for
        // the same reason the model and tool names already were.
        for body in [
            #"{"error":"max_tokens is too large"}"#,
            #"{"error":"max_completion_tokens must be positive"}"#,
            #"{"error":"temperature must be between 0 and 2"}"#,
            #"{"error":"invalid value for top_p"}"#,
            #"{"error":"presence_penalty out of range"}"#
        ] {
            XCTAssertFalse(
                OpenAICloudChatService.shouldRetryWithoutCacheKey(status: 400, body: body),
                "a parameter-level refusal is not about the cache field: \(body)"
            )
        }
    }

    func testA4xxReceiptSaysTheRequestWasNotBilledAndA5xxSaysItWas() async {
        // The enum's `spentRequest` is pinned elsewhere; this drives a real
        // request through the transport, so the classification under test is
        // the one a chat turn receives — on the event AND on the receipt that
        // survives a cancelled stream.
        let refusedReceipt = CloudTurnReceipt()
        CloudStubURLProtocol.responses = [
            (429, Data(#"{"error":"rate limit exceeded"}"#.utf8))
        ]
        let refusedService = makeService(host: "billing-4xx-\(UUID().uuidString).test")
        guard case .failed(let refusal)? = await collect(
            refusedService,
            sessionKey: "billing-4xx",
            receipt: refusedReceipt
        ).last else {
            return XCTFail("a 429 should fail the turn")
        }
        XCTAssertFalse(refusal.spentRequest)
        XCTAssertEqual(refusedReceipt.spentRequest, false)

        let failedReceipt = CloudTurnReceipt()
        CloudStubURLProtocol.responses = [
            (500, Data(#"{"error":"upstream exploded"}"#.utf8))
        ]
        let failedService = makeService(host: "billing-5xx-\(UUID().uuidString).test")
        guard case .failed(let failure)? = await collect(
            failedService,
            sessionKey: "billing-5xx",
            receipt: failedReceipt
        ).last else {
            return XCTFail("a 500 should fail the turn")
        }
        XCTAssertTrue(failure.spentRequest)
        XCTAssertEqual(failedReceipt.spentRequest, true)
    }

    func testAnOfflineSendIsRecordedOnTheReceipt() async {
        // A consumer whose task is cancelled cannot receive the failure event,
        // so the receipt is the cancel path's only evidence. This drives the
        // same failure a real offline Mac produces and asserts the receipt
        // carries it even though the event path is separate.
        final class OfflineProtocol: URLProtocol {
            override class func canInit(with request: URLRequest) -> Bool { true }
            override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
            override func startLoading() {
                client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
            }
            override func stopLoading() {}
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [OfflineProtocol.self]
        let base = "https://offline-\(UUID().uuidString).test/openai/v1"
        let service = OpenAICloudChatService(
            chatURL: URL(string: "\(base)/chat/completions")!,
            modelsURL: URL(string: "\(base)/models")!,
            apiKey: "sk-test",
            requestTimeout: 5,
            session: URLSession(configuration: configuration),
            cacheHints: .openAI,
            cacheHintMemory: CloudCacheHintMemory()
        )
        let receipt = CloudTurnReceipt()
        guard case .failed(let failure)? = await collect(
            service,
            sessionKey: "offline",
            receipt: receipt
        ).last else {
            return XCTFail("an offline send should fail")
        }
        XCTAssertFalse(failure.spentRequest)
        XCTAssertEqual(receipt.spentRequest, false)
        XCTAssertEqual(receipt.failure?.spentRequest, false)
    }

    func testAFinishedTurnIsRecordedOnTheReceipt() async {
        // Usage that arrived just as Stop was pressed is dropped by the
        // cancelled consumer's `for await`, so the receipt keeps it. The
        // provider still measured and billed that turn.
        let usage = #"{"prompt_tokens":1000,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":800}}"#
        let body = Data(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}],\"usage\":\(usage)}\n\ndata: [DONE]\n\n".utf8
        )
        CloudStubURLProtocol.responses = [(200, body)]
        let receipt = CloudTurnReceipt()
        let service = makeService(host: "receipt-finished-\(UUID().uuidString).test")
        guard case .finished(let result)? = await collect(
            service,
            sessionKey: "receipt-finished",
            receipt: receipt
        ).last else {
            return XCTFail("the stream should finish")
        }
        XCTAssertEqual(receipt.spentRequest, true)
        XCTAssertEqual(receipt.result, result)
        XCTAssertEqual(receipt.result?.usage?.cachedTokens, 800)
    }

    func testACancelledRequestKeepsTheReceiptsLastClassification() async {
        // The property this whole receipt exists for: Stop cancels the
        // CONSUMER, its `for await` ends with no event at all, and the only
        // thing left is what the transport wrote before it noticed. A request
        // that was in flight when Stop arrived cannot be proven unsent, so it
        // stays billed — the conservative answer — even though no event
        // carried it.
        final class HeldProtocol: URLProtocol {
            nonisolated(unsafe) static var didStart = false
            override class func canInit(with request: URLRequest) -> Bool { true }
            override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
            override func startLoading() {
                // Never answers: the turn is cancelled while waiting for the
                // response HEADERS, which is exactly the window this case is
                // about.
                Self.didStart = true
            }
            override func stopLoading() {}
        }
        HeldProtocol.didStart = false
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [HeldProtocol.self]
        let base = "https://cancel-\(UUID().uuidString).test/openai/v1"
        let service = OpenAICloudChatService(
            chatURL: URL(string: "\(base)/chat/completions")!,
            modelsURL: URL(string: "\(base)/models")!,
            apiKey: "sk-test",
            requestTimeout: 30,
            session: URLSession(configuration: configuration),
            cacheHints: .openAI,
            cacheHintMemory: CloudCacheHintMemory()
        )
        let receipt = CloudTurnReceipt()
        let task = Task {
            await collect(service, sessionKey: "cancel", receipt: receipt)
        }
        for _ in 0..<400 where !HeldProtocol.didStart {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertTrue(HeldProtocol.didStart, "the request should reach the transport")
        task.cancel()
        _ = await task.value
        // The producer records its classification on its own task, so allow it
        // to settle after the cancelled consumer has returned.
        for _ in 0..<400 where receipt.spentRequest == nil {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(
            receipt.spentRequest,
            true,
            "a request in flight when Stop arrived cannot be proven unsent"
        )
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
        XCTAssertEqual(usage.cacheReadFraction, 0.8)
    }

    func testDeepSeekTopLevelCacheHitIsDecoded() throws {
        let usage = try decode(#"{"prompt_tokens":900,"completion_tokens":10,"prompt_cache_hit_tokens":600}"#)
        XCTAssertEqual(usage.cachedTokens, 600)
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
        XCTAssertEqual(usage.cacheReadFraction, 0.4)
    }

    func testAnthropicWithoutCacheColumnsKeepsItsPromptTotal() throws {
        // Without the cache keys the block is indistinguishable from the
        // Responses shape, where `input_tokens` already is the total.
        let usage = try decode(#"{"input_tokens":500,"output_tokens":20}"#)
        XCTAssertEqual(usage.promptTokens, 500)
        XCTAssertNil(usage.cachedTokens)
        XCTAssertNil(usage.cacheWriteTokens)
    }

    func testAnExplicitZeroAnthropicCacheColumnIsNotDialectProof() throws {
        // Some OpenAI-shaped gateways emit the Anthropic spelling as an
        // explicit zero while publishing `input_tokens` as the TOTAL and the
        // cache split in OpenAI's nested detail. Treating the key's presence as
        // proof read 1,000 + 600 + 0 as 1,600 and billed a prompt the provider
        // never had. A zero column also carries no split: the total stands.
        let usage = try decode(
            #"{"input_tokens":1000,"output_tokens":10,"prompt_tokens_details":{"cached_tokens":600},"cache_creation_input_tokens":0}"#
        )
        XCTAssertEqual(usage.promptTokens, 1_000, "input_tokens is the total, not the uncached share")
        XCTAssertEqual(usage.cachedTokens, 600)
        XCTAssertEqual(usage.cacheWriteTokens, 0)
    }

    func testAZeroAnthropicColumnBesideAZeroCacheReadKeepsTheTotal() throws {
        // The same rule when both Anthropic columns are explicit zeroes: there
        // is no uncached-versus-cached split to add back.
        let usage = try decode(
            #"{"input_tokens":500,"output_tokens":20,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}"#
        )
        XCTAssertEqual(usage.promptTokens, 500)
    }

    func testAPositiveAnthropicColumnStillNormalisesTheTotal() throws {
        // The guard is about a ZERO column, not a missing dialect: a positive
        // cache read is still Anthropic's uncached-plus-cached split.
        let usage = try decode(
            #"{"input_tokens":500,"output_tokens":20,"cache_read_input_tokens":400,"cache_creation_input_tokens":0}"#
        )
        XCTAssertEqual(usage.promptTokens, 900)
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

    func testAProviderWithoutACacheSplitReportsNoFraction() throws {
        let usage = try decode(#"{"prompt_tokens":250,"completion_tokens":5}"#)
        XCTAssertNil(usage.cachedTokens)
        XCTAssertEqual(usage.promptTokens, 250)
        XCTAssertNil(usage.cacheReadFraction)
    }

    func testNoPromptCountMeansNoFraction() throws {
        let usage = try decode(#"{"completion_tokens":5}"#)
        XCTAssertNil(usage.promptTokens)
        XCTAssertNil(usage.cacheReadFraction)
    }

    func testCachedCannotExceedThePrompt() throws {
        let usage = try decode(#"{"prompt_tokens":10,"completion_tokens":0,"cached_tokens":50}"#)
        XCTAssertEqual(usage.promptTokens, 10)
        XCTAssertEqual(usage.cacheReadFraction, 1.0)
        let priced = CloudPricing(input: 1, output: 1)
        // The clamps keep the fresh share non-negative: a cache figure larger
        // than the prompt is charged as a full cache read (10 tokens at $1/M),
        // never as a negative input cost.
        XCTAssertEqual(try XCTUnwrap(priced.cost(for: usage)), 0.00001, accuracy: 1e-12)
    }

    func testTheNestedOpenAIDetailBeatsAFlatColumn() throws {
        // `cached_tokens` at the top level is a generic column some compatible
        // endpoints use for something else. OpenAI's documented home is the
        // nested detail, so that is what a block carrying both must report.
        let usage = try decode(
            #"{"prompt_tokens":1000,"cached_tokens":999,"prompt_tokens_details":{"cached_tokens":800}}"#
        )
        XCTAssertEqual(usage.cachedTokens, 800)
        XCTAssertEqual(usage.promptTokens, 1000)
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

    func testAnUncachedShareCannotOverwriteAnEarlierAnthropicTotal() throws {
        // Anthropic's `message_start` carries the cache columns and its
        // `message_delta` may repeat `input_tokens` alone. Decoded in
        // isolation that second block is the UNCACHED share; laying it over the
        // accumulated total shrank the prompt by exactly the cached and written
        // columns, so the billed total no longer covered the `cached` figure
        // reported beside it.
        let start = try decode(
            #"{"input_tokens":500,"output_tokens":1,"cache_read_input_tokens":400,"cache_creation_input_tokens":100}"#
        )
        XCTAssertEqual(start.promptTokens, 1_000)
        let delta = try decode(#"{"input_tokens":500,"output_tokens":20}"#)
        let merged = start.merging(delta)
        XCTAssertEqual(merged.promptTokens, 1_000)
        XCTAssertEqual(merged.cachedTokens, 400)
        XCTAssertEqual(merged.cacheWriteTokens, 100)
        XCTAssertEqual(merged.completionTokens, 20)
        XCTAssertEqual(merged.cacheReadFraction, 0.4)

        // An OpenAI-shaped block, whose prompt IS the total, still wins.
        let openAI = try decode(#"{"prompt_tokens":1200,"completion_tokens":30}"#)
        XCTAssertEqual(start.merging(openAI).promptTokens, 1_200)
    }

    func testAStreamedAnthropicTurnKeepsItsSplitAcrossChunks() throws {
        // The same sequence through the parser, which is where the two blocks
        // actually arrive.
        var parser = CloudStreamParser()
        _ = try parser.consume(
            dataPayload: #"{"choices":[{"delta":{"content":"hi"}}],"usage":{"input_tokens":500,"output_tokens":1,"cache_read_input_tokens":400,"cache_creation_input_tokens":100}}"#
        )
        _ = try parser.consume(
            dataPayload: #"{"choices":[{"delta":{},"finish_reason":"end_turn"}],"usage":{"input_tokens":500,"output_tokens":20}}"#
        )
        let usage = try XCTUnwrap(parser.usage)
        XCTAssertEqual(usage.promptTokens, 1_000)
        XCTAssertEqual(usage.cachedTokens, 400)
        XCTAssertEqual(usage.cacheWriteTokens, 100)
        XCTAssertEqual(usage.completionTokens, 20)
    }

    func testAMalformedUsageBlockCostsTheEstimateNotTheChunk() throws {
        // The token accounting is an optional convenience; the reply is the
        // payload. A block spelled as a string threw and dropped the deltas
        // that rode the same chunk.
        var parser = CloudStreamParser()
        let events = try parser.consume(
            dataPayload: #"{"choices":[{"delta":{"content":"half a reply"}}],"usage":"n/a"}"#
        )
        XCTAssertEqual(events, [.contentDelta("half a reply")])
        XCTAssertEqual(parser.text, "half a reply")
        XCTAssertNil(parser.usage)
        XCTAssertEqual(parser.decodedChunkCount, 1)
    }

    func testExplicitZeroTopLevelCacheColumnDoesNotHidePositiveNestedCount() throws {
        let usage = try decode(
            #"{"prompt_tokens":1000,"completion_tokens":20,"prompt_cache_hit_tokens":0,"prompt_tokens_details":{"cached_tokens":800},"cache_creation_input_tokens":0,"cache_write_tokens":150}"#
        )
        XCTAssertEqual(usage.cachedTokens, 800)
        XCTAssertEqual(usage.cacheWriteTokens, 150)

        let genuineZero = try decode(
            #"{"prompt_tokens":1000,"completion_tokens":20,"prompt_cache_hit_tokens":0}"#
        )
        XCTAssertEqual(genuineZero.cachedTokens, 0)
    }

    func testTrailingAnthropicDeltaWithExplicitZeroCacheColumnsPreservesEarlierSplit() throws {
        let messageStart = try decode(
            #"{"input_tokens":100,"cache_read_input_tokens":500,"cache_creation_input_tokens":50,"output_tokens":5}"#
        )
        let trailingDelta = try decode(
            #"{"input_tokens":100,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":42}"#
        )
        let merged = messageStart.merging(trailingDelta)
        XCTAssertEqual(merged.promptTokens, 650)
        XCTAssertEqual(merged.cachedTokens, 500)
        XCTAssertEqual(merged.cacheWriteTokens, 50)
        XCTAssertEqual(merged.completionTokens, 42)
        XCTAssertTrue(merged.inputTokensExcludeCache)

        // An OpenAI-style prompt_tokens block still overrides the Anthropic total.
        let openAIOverride = try decode(
            #"{"prompt_tokens":900,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":300}}"#
        )
        let overridden = merged.merging(openAIOverride)
        XCTAssertEqual(overridden.promptTokens, 900)
        XCTAssertEqual(overridden.cachedTokens, 300)
        XCTAssertEqual(overridden.completionTokens, 50)
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
        // requests invalidates the provider's whole prompt prefix. Comparing
        // two independent encodes is the property; the key-order assertion
        // below only says which order it is.
        let first = try XCTUnwrap(
            makeRequest(hints: .affinity, host: "sorted.test").httpBody
        )
        let second = try XCTUnwrap(
            makeRequest(hints: .affinity, host: "sorted.test").httpBody
        )
        XCTAssertEqual(first, second, "two encodes of the same request must be byte-identical")
        let text = try XCTUnwrap(String(data: first, encoding: .utf8))
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
        // A transient gateway condition is not a schema complaint: dropping
        // the hint cannot fix "concurrency limit exceeded", and re-sending the
        // request twice neither helps the user nor says anything about the
        // endpoint.
        for body in [
            "concurrency limit exceeded",
            "rate limit reached, try again",
            "service temporarily unavailable",
            "overloaded, please retry",
            "quota exceeded"
        ] {
            XCTAssertFalse(
                OpenAICloudChatService.shouldRetryWithoutCacheKey(status: 400, body: body),
                body
            )
        }
        // The latch is a different rule from the retry, and needs the field's
        // own name: only that is evidence about this endpoint's schema.
        XCTAssertTrue(OpenAICloudChatService.namesCacheKeyField(
            #"{"error":"unexpected property prompt_cache_key"}"#
        ))
        XCTAssertTrue(OpenAICloudChatService.namesCacheKeyField("unsupported field: prompt cache key"))
        XCTAssertFalse(OpenAICloudChatService.namesCacheKeyField("invalid request"))
        XCTAssertFalse(OpenAICloudChatService.namesCacheKeyField("concurrency limit exceeded"))
    }

    func testABodyNamingTheCacheKeyOutranksATransientOrModelWord() {
        // A body that names the field is the strongest evidence there is about
        // this endpoint's schema, and it is the one thing the retry can fix.
        // "unavailable" and "model" used to claim these bodies first, so an
        // endpoint saying exactly what was wrong failed every turn.
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: #"{"error":"prompt_cache_key is unavailable for this model"}"#
        ))
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 422,
            body: "The prompt cache key field is temporarily unavailable."
        ))
        XCTAssertTrue(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: "unsupported tool schema alongside prompt_cache_key"
        ))
        // Without the field's own name those readings still win.
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: "this model is temporarily unavailable"
        ))
        XCTAssertFalse(OpenAICloudChatService.shouldRetryWithoutCacheKey(
            status: 400,
            body: "tools are unavailable for this model"
        ))
        // Once affinity was already dropped/withheld, a 400 naming an affinity
        // header must fail rather than trigger an extra full-prompt upload.
        for marker in ["x-session-affinity", "x-session-id"] {
            XCTAssertFalse(
                OpenAICloudChatService.shouldRetryWithoutCacheKey(
                    status: 400,
                    body: "unsupported header \(marker)"
                ),
                marker
            )
        }
    }

    func testALatchedCacheKeyIsNotRetriedAndIsNotSent() throws {
        // After the process latched the field off, a request body no longer
        // carries it. The retry predicate must agree: otherwise every 4xx for
        // the rest of the process re-sends the identical body.
        let endpoint = "https://latch-\(UUID().uuidString).test/openai/v1/chat/completions"
        // A private memory, as the type intends: the shared latch is keyed by
        // endpoint and a case that latched it would be seen by the app's own
        // requests and by the next case.
        let memory = CloudCacheHintMemory()
        let service = OpenAICloudChatService(
            chatURL: URL(string: endpoint)!,
            modelsURL: URL(string: "https://latch.test/openai/v1/models")!,
            apiKey: "sk-test",
            requestTimeout: 5,
            cacheHints: .openAI,
            cacheHintMemory: memory
        )
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint,
            model: "m",
            memory: memory
        ))
        memory.declinePromptCacheKey(endpoint: endpoint, model: "m")
        XCTAssertFalse(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint,
            model: "m",
            memory: memory
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
        // A private latch: this case used to reset the process-wide one, which
        // cleared every other endpoint a concurrent case or the app held.
        let memory = CloudCacheHintMemory()
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: first,
            model: "m",
            memory: memory
        ))
        memory.declinePromptCacheKey(endpoint: first, model: "m")
        XCTAssertFalse(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: first,
            model: "m",
            memory: memory
        ))
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: second,
            model: "m",
            memory: memory
        ))
        // The reset seam restores the first endpoint, so a repeat-on-failure
        // rerun of this case starts where it did the first time.
        memory.reset()
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: first,
            model: "m",
            memory: memory
        ))
    }

    func testTheLatchIsScopedToTheModelThatRefused() {
        // A compatible gateway fronts several backends under one base URL (the
        // built-in Synthetic provider says so itself), so a 400 from one model
        // says nothing about the model beside it. An endpoint-only latch
        // switched the hint off for every one of them, and a wrongly latched
        // built-in needed a relaunch before the Settings row could be used
        // again (the row now carries a Forget Refusals control).
        let endpoint = "https://latch-model-\(UUID().uuidString).test/openai/v1/chat/completions"
        // A private latch: this case used to reset the process-wide one, which
        // cleared every other endpoint a concurrent case or the app held.
        let memory = CloudCacheHintMemory()
        memory.declinePromptCacheKey(endpoint: endpoint, model: "strict-model")
        XCTAssertFalse(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint,
            model: "strict-model",
            memory: memory
        ))
        XCTAssertTrue(OpenAICloudChatService.sendsCacheKey(
            hints: .openAI,
            sessionKey: "abc123",
            endpoint: endpoint,
            model: "neighbour-model",
            memory: memory
        ), "one model's refusal must not silence its neighbour on the same endpoint")
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
        memory.declinePromptCacheKey(endpoint: endpoint, model: "m")
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
            endpoint: endpoint,
            model: "m"
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
        XCTAssertTrue(attempts.shouldAttemptPreflight(projectedTokens: 90_000))
        attempts.beginPreflight()
        // The pass failed (empty summary, truncated summary, provider error):
        // it already spent its request, so no later step may spend another.
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 90_000))
        XCTAssertFalse(attempts.historyWasRewritten)
        // A provider overflow is different evidence and keeps its own attempt.
        XCTAssertTrue(attempts.shouldAttemptOverflowRecovery)
    }

    func testEachPassHasOneAttemptAndARewriteRecordsItsSize() {
        var attempts = CloudCompactionAttempts()
        attempts.beginPreflight()
        attempts.recordRewrite(promptTokens: 80_000)
        // The pass that succeeded is not a pending question, so its attempt is
        // released; what keeps a second one from running is the size it left
        // behind, not a spent budget.
        XCTAssertTrue(attempts.historyWasRewritten)
        XCTAssertEqual(attempts.promptTokensAtLastRewrite, 80_000)
        XCTAssertEqual(attempts.rewriteCount, 1)
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 80_000))
        XCTAssertTrue(attempts.shouldAttemptPreflight(projectedTokens: 80_001))
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
        XCTAssertTrue(attempts.shouldAttemptPreflight(projectedTokens: 90_000))
        // A rewrite records the size it left behind, and a pass that later
        // sends nothing gives its own attempt back without erasing that.
        attempts.beginPreflight()
        attempts.recordRewrite(promptTokens: 80_000)
        attempts.abandonPreflight()
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 80_000))
        XCTAssertTrue(attempts.shouldAttemptPreflight(projectedTokens: 80_001))
    }

    func testARewriteReopensTheGateOnlyOnceTheHistoryGrowsBack() {
        // An uncapped agent run can refill the window after a pass. Closing the
        // gate for the whole run meant the third wall ended the turn with no
        // further attempt; reopening it immediately would re-summarize the
        // digest the pass just wrote. Growth is the boundary.
        var attempts = CloudCompactionAttempts()
        attempts.beginPreflight()
        attempts.recordRewrite(promptTokens: 80_000)
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 40_000))
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 79_999))
        XCTAssertTrue(attempts.shouldAttemptPreflight(projectedTokens: 80_001))

        // Each reopen costs a fresh attempt, and the run spends at most
        // `maximumRewritesPerRun` rewrites rather than compacting without end.
        for _ in 1..<CloudCompactionAttempts.maximumRewritesPerRun {
            attempts.beginPreflight()
            attempts.recordRewrite(promptTokens: 80_000)
        }
        XCTAssertEqual(attempts.rewriteCount, CloudCompactionAttempts.maximumRewritesPerRun)
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 900_000))
        XCTAssertFalse(attempts.allowsPreflightCheck)
    }

    func testTheCheapGateAgreesWithTheFullOne() {
        // `allowsPreflightCheck` exists so an exhausted run does not assemble
        // and tokenise its history every step just to decline. It must never
        // deny a pass the full rule would allow, or close while the full rule
        // is still open.
        var attempts = CloudCompactionAttempts()
        XCTAssertTrue(attempts.allowsPreflightCheck)
        XCTAssertTrue(attempts.shouldAttemptPreflight(projectedTokens: 0))

        attempts.beginPreflight()
        attempts.recordRewrite(promptTokens: 80_000)
        XCTAssertTrue(attempts.allowsPreflightCheck)
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 80_000))
        XCTAssertTrue(attempts.shouldAttemptPreflight(projectedTokens: 200_000))

        // A second rewrite uses the second attempt; the cheap gate stays open
        // because the run still has one left.
        attempts.beginPreflight()
        attempts.recordRewrite(promptTokens: 80_000)
        XCTAssertEqual(attempts.rewriteCount, 2)
        XCTAssertTrue(attempts.allowsPreflightCheck)

        // The third rewrite spends the budget: both answers close together.
        attempts.beginPreflight()
        attempts.recordRewrite(promptTokens: 80_000)
        XCTAssertEqual(attempts.rewriteCount, CloudCompactionAttempts.maximumRewritesPerRun)
        XCTAssertFalse(attempts.allowsPreflightCheck)
        XCTAssertFalse(attempts.shouldAttemptPreflight(projectedTokens: 200_000))

        // An in-flight attempt also closes the cheap gate, which is the other
        // reason it exists: no pass may start twice from one step.
        var one = CloudCompactionAttempts()
        one.beginPreflight()
        XCTAssertFalse(one.allowsPreflightCheck)
        XCTAssertFalse(one.shouldAttemptPreflight(projectedTokens: 200_000))
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
        // A revision-pinned id contains the same `@` the historical
        // `model@window` spelling used, so the boundary is the LAST separator:
        // splitting at the first one read `model@rev=8192` as model `model`
        // with window `rev=8192`, dropped it, and left it un-removable.
        defaults.set(
            "deepseek-ai/DeepSeek-V4.1@rev-2=131072; plain=8192",
            forKey: CloudContextPolicy.defaultsKey(for: "s")
        )
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "s", defaults: defaults),
            ["deepseek-ai/DeepSeek-V4.1@rev-2": 131_072, "plain": 8_192]
        )
        CloudContextPolicy.removeOverride(
            model: "deepseek-ai/DeepSeek-V4.1@rev-2",
            providerId: "s",
            defaults: defaults
        )
        XCTAssertEqual(
            defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "s")),
            "plain=8192"
        )
        // Nothing usable means no stored key at all, so a later reader cannot
        // mistake an empty string for "recorded, but zero".
        CloudContextPolicy.setOverrides(["units": 12], for: "r", defaults: defaults)
        XCTAssertNil(defaults.string(forKey: CloudContextPolicy.defaultsKey(for: "r")))
    }

    func testTheThresholdIsEightyPercentOfAKnownWindow() {
        // The reserve is a REQUIRED argument: the call site cannot drop the
        // reply budget and still compile, which is the only pin a pure test
        // cannot give.
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 79_999, window: 100_000, reservedTokens: 0
        ))
        XCTAssertTrue(CloudContextPolicy.shouldCompact(
            promptTokens: 80_000, window: 100_000, reservedTokens: 0
        ))
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 1_000_000, window: nil, reservedTokens: 0
        ))
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 1_000, window: 512, reservedTokens: 0
        ))
    }

    func testThePreflightReservesRoomForTheReply() {
        // A 16k window with the app's default 8,192-token reply: a prompt of
        // 12,800 is exactly the fraction, but the request that carries it is
        // 20,992 tokens and cannot fit. The old check ignored the reply budget,
        // so it compacted too late to help and spent the run's one pre-flight
        // attempt on a wall it could not clear.
        XCTAssertTrue(CloudContextPolicy.shouldCompact(
            promptTokens: 12_000,
            window: 16_000,
            reservedTokens: 8_192
        ), "12,000 + 8,192 does not fit a 16k window")
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 6_000,
            window: 16_000,
            reservedTokens: 8_192
        ), "6,000 + 8,192 does fit a 16k window")
        // The fraction still applies on its own, whatever the reply budget.
        XCTAssertTrue(CloudContextPolicy.shouldCompact(
            promptTokens: 80_000,
            window: 100_000,
            reservedTokens: 100
        ))
        // A big window keeps the old answer: the reply budget is nowhere near
        // the remaining room, so nothing here compacts a healthy conversation.
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 100_000,
            window: 1_048_576,
            reservedTokens: 8_192
        ))
        // No window is still no answer, and a negative reserve cannot buy room.
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 100_000,
            window: nil,
            reservedTokens: 8_192
        ))
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 1_000,
            window: 16_000,
            reservedTokens: -50_000
        ))
    }

    func testAnImpossibleReplyBudgetNeverTriggersCompaction() {
        // A reply budget at or above the window can never fit beside a prompt,
        // however much history is summarized. Reporting "compact" here paid for
        // up to three summarizer passes on EVERY send of a model whose window
        // is smaller than the app's Max Output Tokens, and still overflowed.
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 0,
            window: 8_192,
            reservedTokens: 8_192
        ), "a reply as large as the window leaves no room for a prompt")
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 100,
            window: 8_192,
            reservedTokens: 9_000
        ))
        // Below the window the reserve still legitimately triggers compaction.
        XCTAssertTrue(CloudContextPolicy.shouldCompact(
            promptTokens: 1_000,
            window: 8_192,
            reservedTokens: 8_191
        ), "1,000 + 8,191 cannot fit, but compaction can still make it fit")
    }

    func testTheReplyBudgetIsClampedToTheRoomTheThresholdLeaves() {
        // 20% of the window is the room an 80%-of-window prompt leaves. The
        // clamp is a ceiling on `max_tokens`, so lowering it cannot break a
        // turn that would otherwise have fitted; leaving it above the window
        // guarantees a refusal.
        XCTAssertEqual(CloudContextPolicy.replyBudget(requested: 8_192, window: 16_000), 3_200)
        XCTAssertEqual(CloudContextPolicy.replyBudget(requested: 8_192, window: 1_024), 205)
        // Nothing to clamp: a request already under the room passes through.
        XCTAssertEqual(CloudContextPolicy.replyBudget(requested: 8_192, window: nil), 8_192)
        XCTAssertEqual(CloudContextPolicy.replyBudget(requested: 1_000, window: 16_000), 1_000)
        XCTAssertEqual(CloudContextPolicy.replyBudget(requested: 8_192, window: 512), 8_192)
        // A caller that asked for no budget keeps asking for none.
        XCTAssertEqual(CloudContextPolicy.replyBudget(requested: 0, window: 16_000), 0)
        // The clamped pair no longer reports a wall compaction cannot clear.
        let budget = CloudContextPolicy.replyBudget(requested: 8_192, window: 16_000)
        XCTAssertFalse(CloudContextPolicy.shouldCompact(
            promptTokens: 12_000,
            window: 16_000,
            reservedTokens: budget
        ), "12,000 + 3,200 fits a 16k window, and 12,000 is under the threshold")
    }

    /// The regression this whole change exists for.
    ///
    /// `replyBudget` used to hand back the caller's request untouched when no
    /// window was recorded. That was harmless only while the settings slider
    /// topped out at a budget most providers accept: once the Max Output Tokens
    /// ceiling became the model's real capacity, an unrecorded provider would
    /// have received an arbitrary `max_tokens` — a 400, or an expensive success,
    /// with no local evidence behind it.
    func testAnUnrecordedWindowIsCappedRatherThanPassedThrough() {
        XCTAssertEqual(
            CloudContextPolicy.replyBudget(requested: 131_072, window: nil),
            CloudContextPolicy.unknownWindowReplyCeiling
        )
        // A sub-plausible window is a units mistake, not capacity, so it takes
        // the same ceiling instead of the (tiny) "window" arithmetic.
        XCTAssertEqual(
            CloudContextPolicy.replyBudget(requested: 131_072, window: 512),
            CloudContextPolicy.unknownWindowReplyCeiling
        )
        XCTAssertEqual(
            CloudContextPolicy.replyBudget(requested: 131_072, window: 1_023),
            CloudContextPolicy.unknownWindowReplyCeiling
        )
        // Under the ceiling nothing is lowered: the cap is a ceiling, not a floor.
        XCTAssertEqual(
            CloudContextPolicy.replyBudget(requested: 4_096, window: nil),
            4_096
        )
        // A recorded window still wins over the fallback.
        XCTAssertEqual(
            CloudContextPolicy.replyBudget(requested: 131_072, window: 200_000),
            40_000
        )
    }

    /// The Settings slider must advertise exactly what the send path enforces.
    /// A slider that computed its own range would eventually show a budget the
    /// clamp silently lowered, which reads as the app ignoring the setting.
    func testTheAdvertisedCeilingIsTheEnforcedBudget() {
        // No window: the conservative ceiling, in both directions.
        XCTAssertEqual(
            CloudContextPolicy.maximumOutputTokens(window: nil),
            CloudContextPolicy.unknownWindowReplyCeiling
        )
        XCTAssertEqual(
            CloudContextPolicy.maximumOutputTokens(window: 512),
            CloudContextPolicy.unknownWindowReplyCeiling
        )

        // A recorded window: 20% of it, identical to the clamp.
        XCTAssertEqual(CloudContextPolicy.maximumOutputTokens(window: 200_000), 40_000)
        XCTAssertEqual(
            CloudContextPolicy.maximumOutputTokens(window: 200_000),
            CloudContextPolicy.replyBudget(requested: Int.max, window: 200_000),
            "the slider and the clamp cannot disagree"
        )

        // A window small enough that 20% is under the fallback yields the smaller
        // number: capacity, not a floor, is what binds.
        XCTAssertEqual(CloudContextPolicy.maximumOutputTokens(window: 32_000), 6_400)
    }

    /// `maxTokenPresets(upTo:)` feeds `ForEach(_:id: \.self)`, so a duplicate is a
    /// SwiftUI runtime crash rather than a cosmetic glitch, and a preset above
    /// the ceiling would set a value the control immediately clamps.
    func testMaxTokenPresetsRespectTheCeilingAndStayUnique() {
        for ceiling in [16_384, 32_768, 40_000, 262_144] {
            let presets = maxTokenPresets(upTo: ceiling)
            XCTAssertEqual(presets, Array(Set(presets)).sorted(), "unique and ascending at \(ceiling)")
            XCTAssertTrue(presets.allSatisfy { $0 <= ceiling }, "no preset above the ceiling at \(ceiling)")
            XCTAssertEqual(presets.filter { $0 == ceiling }.count, 1, "the ceiling is offered exactly once at \(ceiling)")
            XCTAssertFalse(presets.isEmpty)
        }
        // The default ceiling keeps every value the old fixed list offered.
        let standard = maxTokenPresets(upTo: 32_768)
        for legacy in [1024, 4096, 8192] {
            XCTAssertTrue(standard.contains(legacy))
        }
        XCTAssertEqual(standard.last, 32_768)
        // A cloud ceiling that is not a rung is still offered verbatim.
        XCTAssertEqual(maxTokenPresets(upTo: 40_000).last, 40_000)
        XCTAssertFalse(maxTokenPresets(upTo: 40_000).contains(65_536))
    }

    func testAContextPercentageAboveTheWindowStopsAtOneHundred() {
        // Switching a long conversation to a smaller-window model can leave the
        // carried context larger than the new ceiling. "125% of 16,000" reads
        // as a broken meter; the token count beside it still shows the truth.
        XCTAssertEqual(CloudContextPolicy.usagePercent(promptTokens: 20_000, window: 16_000), 100)
        XCTAssertEqual(CloudContextPolicy.usagePercent(promptTokens: 8_000, window: 16_000), 50)
        // The overflow/units guards answer the same way they always did.
        XCTAssertEqual(CloudContextPolicy.usagePercent(promptTokens: Int.max, window: 1), 100)
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

    func testATinyWindowCannotTrapThePercentage() {
        // The window is hand-typed and `usagePercent` is public, so a caller can
        // hand it a window of 1. `Int.max` tokens over that overflows a Double
        // into `+inf`, and converting that (or anything above `Int.max`) to Int
        // traps. The share of a window is capped at 100, which is the same
        // answer for every one of those cases.
        XCTAssertEqual(CloudContextPolicy.usagePercent(promptTokens: 1, window: 1), 100)
        XCTAssertEqual(CloudContextPolicy.usagePercent(promptTokens: Int.max, window: 1), 100)
        XCTAssertEqual(CloudContextPolicy.usagePercent(promptTokens: -5, window: 1_024), 0)
        XCTAssertNil(CloudContextPolicy.usagePercent(promptTokens: 10, window: 0))
    }

    func testAnAbsurdWindowCannotTrapTheChunkBudget() {
        // The window is hand-typed. `Int(Double(window) * 0.33) * 4` trapped on
        // a value near Int.max; a budget that large is a units mistake, so it
        // clamps instead of crashing the /compact command.
        XCTAssertEqual(
            CloudContextPolicy.chunkCharacterBudget(window: Int.max),
            8_000_000
        )
        XCTAssertEqual(
            CloudContextPolicy.chunkCharacterBudget(window: Int.max / 2),
            8_000_000
        )
        // The reply budget is bounded by `maximumSummaryResponseTokens` so
        // gateways that cap `max_tokens` at 8–32k do not answer 400 on 1M-window
        // models, while absurd windows cannot trap the quarter-window guard.
        XCTAssertEqual(
            CloudContextPolicy.summaryResponseTokens(window: Int.max),
            CloudContextPolicy.maximumSummaryResponseTokens
        )
        // The sensible range is untouched.
        XCTAssertEqual(CloudContextPolicy.chunkCharacterBudget(window: 1_048_576), 1_384_120)
    }

    func testTheSummarizerBudgetScalesWithTheChunkAndRespectsTheOutputCeiling() {
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: nil), 2_000)
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: 2_000), 500)
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: 32_000), 2_000)
        XCTAssertEqual(CloudContextPolicy.summaryResponseTokens(window: 200_000), 6_600)
        XCTAssertEqual(
            CloudContextPolicy.summaryResponseTokens(window: 1_000_000),
            CloudContextPolicy.maximumSummaryResponseTokens
        )
        XCTAssertEqual(
            CloudContextPolicy.summaryResponseTokens(window: 1_048_576),
            CloudContextPolicy.maximumSummaryResponseTokens
        )
    }

    func testWindowOverridesTolerateThousandsSeparatorsAndPruneCleanly() {
        let defaults = scratchDefaults()
        let key = CloudContextPolicy.defaultsKey(for: "p")
        defaults.set("m=1,048,576; n=128,000, o=64,000", forKey: key)
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "p", defaults: defaults),
            ["m": 1_048_576, "n": 128_000, "o": 64_000]
        )

        CloudContextPolicy.removeOverride(model: "m", providerId: "p", defaults: defaults)
        XCTAssertEqual(
            defaults.string(forKey: key),
            "n=128,000; o=64,000",
            "removing a grouped-number entry must not leave orphan comma groups behind"
        )
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "p", defaults: defaults),
            ["n": 128_000, "o": 64_000]
        )
    }

    /// Saving one model's window must not delete another's — including an entry
    /// this build cannot parse, which is the operator's to fix rather than the
    /// Save button's to erase.
    ///
    /// The UI writer used to read `overrides(for:)` and write the dictionary
    /// back. That reader drops every entry it cannot parse, so saving `a` in
    /// `a=200000; typo=12k; b=128000` left `a` and `b` and silently deleted the
    /// typo. `removeOverride` exists because unregister had the same bug; the
    /// single-model writer had taken the unsafe route back.
    func testSavingOneWindowLeavesItsNeighboursExactlyAsTyped() {
        let defaults = scratchDefaults()
        let key = CloudContextPolicy.defaultsKey(for: "p")
        defaults.set("a=200000; typo=12k; b=128000", forKey: key)

        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "a", window: 300_000, providerId: "p", defaults: defaults)
        )
        let afterSave = defaults.string(forKey: key) ?? ""
        XCTAssertTrue(afterSave.contains("typo=12k"), "an unparseable neighbour must survive a save")
        XCTAssertTrue(afterSave.contains("b=128000"), "a valid neighbour must survive a save")
        XCTAssertEqual(
            CloudContextPolicy.overrides(for: "p", defaults: defaults),
            ["a": 300_000, "b": 128_000],
            "the parse view still reports only what it can read"
        )

        // A model with no entry is appended, and the neighbours still stand.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "c", window: 64_000, providerId: "p", defaults: defaults)
        )
        let afterAdd = defaults.string(forKey: key) ?? ""
        XCTAssertTrue(afterAdd.contains("typo=12k"))
        XCTAssertTrue(afterAdd.contains("b=128000"))
        XCTAssertEqual(CloudContextPolicy.overrides(for: "p", defaults: defaults)["c"], 64_000)

        // Saving the same model again REPLACES rather than duplicating: a
        // duplicate would leave the reader's last-one-wins merge to decide.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "c", window: 96_000, providerId: "p", defaults: defaults)
        )
        let afterReplace = defaults.string(forKey: key) ?? ""
        XCTAssertEqual(afterReplace.components(separatedBy: "c=").count - 1, 1, "one entry per model")
        XCTAssertEqual(CloudContextPolicy.overrides(for: "p", defaults: defaults)["c"], 96_000)
        XCTAssertTrue(afterReplace.contains("typo=12k"))

        // Clearing goes through the same in-place path.
        XCTAssertTrue(
            CloudContextPolicy.setOverride(model: "b", window: nil, providerId: "p", defaults: defaults)
        )
        let afterClear = defaults.string(forKey: key) ?? ""
        XCTAssertFalse(afterClear.contains("b=128000"))
        XCTAssertTrue(afterClear.contains("typo=12k"), "clearing one window must not touch the typo either")
        XCTAssertTrue(afterClear.contains("a=300000"))
    }

    /// A sub-plausible window is refused rather than stored: the reader would
    /// drop it, so storing it would look saved and read back as absent.
    func testAnImplausibleWindowIsRefusedRatherThanStored() {
        let defaults = scratchDefaults()
        let key = CloudContextPolicy.defaultsKey(for: "p")
        defaults.set("a=200000", forKey: key)

        XCTAssertFalse(
            CloudContextPolicy.setOverride(model: "a", window: 512, providerId: "p", defaults: defaults)
        )
        XCTAssertEqual(
            defaults.string(forKey: key),
            "a=200000",
            "a refused write must not modify the stored list at all"
        )
        XCTAssertFalse(
            CloudContextPolicy.setOverride(model: "", window: 200_000, providerId: "p", defaults: defaults)
        )
    }

    /// The Cloud tab's row and the send path must report one window. The row used
    /// to read the per-model override alone, so with a global fallback recorded it
    /// advertised 16,384 for a model the turn resolved to 40,000.
    func testTheRowAndTheSendPathAgreeOnTheWindow() {
        let defaults = scratchDefaults()
        defaults.set("a=200000", forKey: CloudContextPolicy.defaultsKey(for: "p"))
        defaults.set(200_000, forKey: CloudContextPolicy.globalWindowKey)

        // Per-model wins.
        XCTAssertEqual(
            CloudContextPolicy.window(forModelId: "a", providerId: "p", defaults: defaults),
            200_000
        )
        // A model with no entry falls through to the single global value, and the
        // ceiling that follows is the one the send path enforces.
        let fallbackWindow = CloudContextPolicy.window(forModelId: "b", providerId: "p", defaults: defaults)
        XCTAssertEqual(fallbackWindow, 200_000)
        XCTAssertEqual(CloudContextPolicy.globalWindow(defaults: defaults), 200_000)
        XCTAssertEqual(
            CloudContextPolicy.maximumOutputTokens(window: fallbackWindow),
            40_000,
            "a fallback-resolved window must advertise the fallback budget, not the no-window ceiling"
        )
        // With neither recorded, the no-window ceiling is the answer.
        XCTAssertNil(CloudContextPolicy.globalWindow(defaults: scratchDefaults()))
    }

    func testCloudCompactionChunkBudgetReservesInstructionsDigestAndReplyCap() {
        // On a small window (< 3k), fixed occupants shrink the chunk budget or
        // report 0 rather than returning the 4,000-character floor and failing with 400.
        let smallBudget = CloudContextPolicy.chunkCharacterBudget(
            window: 1_024,
            previousSummary: nil,
            previousRecap: nil,
            instructions: String(repeating: "x", count: 1_100),
            responseTokens: CloudContextPolicy.summaryResponseTokens(window: 1_024)
        )
        XCTAssertGreaterThan(smallBudget, 0)
        XCTAssertLessThan(smallBudget, 2_000)

        let exhausted = CloudContextPolicy.chunkCharacterBudget(
            window: 1_024,
            previousSummary: String(repeating: "s", count: 1_500),
            previousRecap: String(repeating: "r", count: 1_500),
            instructions: String(repeating: "x", count: 1_100),
            responseTokens: CloudContextPolicy.summaryResponseTokens(window: 1_024)
        )
        XCTAssertEqual(exhausted, 0)
    }
}

final class CloudCompactionBoundaryTests: XCTestCase {
    func testARewriteKeepsAddedRowsAndDropsDeletedOnes() {
        // A long compaction suspends for minutes, and the transcript it
        // rewrites from is the one it snapshotted. Two rows can change while it
        // runs, and both used to be handled wrongly: a row appended since the
        // snapshot was deleted by the rewrite, and a row the user deleted from
        // that snapshot was resurrected by it.
        let first = ChatMessage(role: .user, content: "one")
        let second = ChatMessage(role: .assistant, content: "two")
        let third = ChatMessage(role: .user, content: "three")
        let snapshot = [first, second, third]
        let appended = ChatMessage(role: .system, content: "a /plan notice")
        // The live list: `second` was deleted, `appended` arrived.
        let live = [first, third, appended]

        XCTAssertEqual(
            ChatSession.messagesAdded(to: live, since: snapshot).map(\.content),
            ["a /plan notice"]
        )
        XCTAssertEqual(
            ChatSession.stillPresent(snapshot, in: live).map(\.content),
            ["one", "three"],
            "the row deleted during the pass must not come back"
        )
    }

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
            .noTranscript,
            .digestExceedsWindow
        ] {
            XCTAssertFalse(reason.spentRequest, "\(reason) sent no request")
        }
        for reason in [
            CloudCompactionFailure.emptySummary,
            .truncatedSummary,
            .cancelled,
            .provider("HTTP 500", spent: true)
        ] {
            XCTAssertTrue(reason.spentRequest, "\(reason) was billed")
        }
        // A refusal the provider ANSWERED with a 4xx was never invoiced: an
        // expired key, a rate limit or a schema complaint never reaches
        // inference, and charging the prompt for one made the spend grow on
        // every failed turn. Only a request the provider accepted — a 5xx, a
        // dropped connection, a stream that died mid-answer — is billed.
        // Classified THROUGH the production mapper, not by hand-building an
        // outcome whose `spent` flag is the answer being asserted.
        let refused = CloudCompactionFailure.fromProviderError(
            CloudTurnFailure.httpStatus(code: 401, summary: "expired")
        )
        XCTAssertEqual(refused, .provider("Authentication failed (HTTP 401). Check the API key in Settings ▸ Cloud Providers.", spent: false))
        XCTAssertFalse(refused.spentRequest)
        let accepted = CloudCompactionFailure.fromProviderError(
            CloudTurnFailure.httpStatus(code: 500, summary: "boom")
        )
        XCTAssertEqual(accepted, .provider("The provider returned HTTP 500: boom", spent: true))
        XCTAssertTrue(accepted.spentRequest)
        XCTAssertFalse(CloudCompactionFailure.fromProviderError(
            CloudTurnFailure.transport("offline", billed: false)
        ).spentRequest)
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
            .digestExceedsWindow,
            .emptySummary,
            .truncatedSummary,
            .cancelled,
            .provider("HTTP 500", spent: true)
        ] {
            XCTAssertFalse(reason.isRetryableOnNextStep, "\(reason) is not retryable")
        }
    }

    func testAChatFailureKnowsWhetherItSpentARequest() {
        // The same rule the summarizer path applies, for the turn path: a
        // credential refusal the client reports before connecting is not a
        // request, and a 4xx is a rejection the provider never invoiced.
        // Pricing either one invented a spend that grew on every failed turn.
        for reason in [
            CloudTurnFailure.missingAPIKey(providerName: "Synthetic"),
            .credentialAccessDenied(providerName: "Synthetic"),
            .credentialNeedsImport(providerName: "Synthetic"),
            .unsupportedFlavor(flavor: "anthropic")
        ] {
            XCTAssertFalse(reason.spentRequest, "\(reason) never left the machine")
        }
        for code in [400, 401, 403, 404, 422, 429] {
            XCTAssertFalse(
                CloudTurnFailure.httpStatus(code: code, summary: "refused").spentRequest,
                "HTTP \(code) is a rejection, not an invoice"
            )
        }
        // A status the provider answered AFTER accepting the request, a
        // transport failure that happened after the request was submitted, and
        // a malformed stream are all billed: the prompt reached the inference
        // path and nothing came back to say otherwise.
        for reason in [
            CloudTurnFailure.httpStatus(code: 500, summary: "boom"),
            .httpStatus(code: 503, summary: "overloaded"),
            .transport("connection reset", billed: true),
            .invalidResponse("the stream ended early")
        ] {
            XCTAssertTrue(reason.spentRequest, "\(reason) was sent")
        }
        // A transport failure the client itself classified as pre-send is not
        // an invoice. This is the offline case: assuming "transport" meant
        // "accepted" charged an estimated prompt for a request that never left
        // the Mac.
        XCTAssertFalse(CloudTurnFailure.transport("offline", billed: false).spentRequest)
        // The summarizer catches an arbitrary `Error`, not just this type: a
        // cancelled stream arrives as `CancellationError` and an unclassified
        // fault is treated as sent rather than silently refunded.
        XCTAssertTrue(CloudTurnFailure.spentRequest(for: CancellationError()))
        XCTAssertTrue(CloudTurnFailure.spentRequest(
            for: CloudTurnFailure.transport("reset", billed: true)
        ))
        XCTAssertFalse(CloudTurnFailure.spentRequest(
            for: CloudTurnFailure.httpStatus(code: 401, summary: "expired")
        ))
        // A raw URL loading error is classified the way the transport
        // classifies it. An offline failure racing Stop used to arrive here as
        // a `CancellationError` (the client dropped it) and was charged a
        // prompt the provider never saw; once the client reports it, this is
        // the second gate that has to agree.
        XCTAssertFalse(CloudTurnFailure.spentRequest(for: URLError(.notConnectedToInternet)))
        XCTAssertFalse(CloudTurnFailure.spentRequest(for: URLError(.cannotFindHost)))
        XCTAssertTrue(CloudTurnFailure.spentRequest(for: URLError(.timedOut)))
        XCTAssertTrue(CloudTurnFailure.spentRequest(for: URLError(.networkConnectionLost)))
        struct UnknownFault: Error {}
        XCTAssertTrue(CloudTurnFailure.spentRequest(for: UnknownFault()))
    }

    func testAFailureEventIsReadableAfterStop() {
        // The chat loop reads a failure even when the user has just stopped the
        // turn: it is the only event that can say the request never left the
        // Mac, and the cancel path prices whatever it cannot prove was unsent.
        XCTAssertTrue(CloudTurnEvent.failed(.transport("offline", billed: false)).isFailure)
        XCTAssertFalse(CloudTurnEvent.started.isFailure)
        XCTAssertFalse(CloudTurnEvent.contentDelta("hi").isFailure)
        XCTAssertFalse(CloudTurnEvent.thinkingDelta("hm").isFailure)
        XCTAssertFalse(CloudTurnEvent.finished(CloudTurnResult(
            text: "ok",
            thinking: "",
            finishReason: "stop",
            usage: nil
        )).isFailure)
    }

    func testAnOfflineSendIsNotBilledAndALostResponseIs() {
        // The classification the client applies to an unclassified error. A
        // URL loading error that proves no connection was delivered is a
        // request the provider never saw; a timeout or a lost connection can
        // happen after it accepted the prompt, so those stay billed.
        let offline: [URLError.Code] = [
            .notConnectedToInternet, .cannotFindHost, .cannotConnectToHost,
            .dnsLookupFailed, .secureConnectionFailed, .badURL,
            // Connection policy: the system declined to open a connection, so
            // the prompt never left the Mac either. These were billed before.
            .dataNotAllowed, .internationalRoamingOff, .callIsActive
        ]
        for code in offline {
            XCTAssertTrue(CloudTurnFailure.neverReachedProvider(URLError(code)), "\(code)")
            XCTAssertFalse(CloudTurnFailure.transportFailure(
                URLError(code),
                requestWasSent: true
            ).spentRequest, "\(code) never reached the provider")
        }
        for code in [URLError.Code.timedOut, .networkConnectionLost] {
            XCTAssertFalse(CloudTurnFailure.neverReachedProvider(URLError(code)))
            XCTAssertTrue(CloudTurnFailure.transportFailure(
                URLError(code),
                requestWasSent: true
            ).spentRequest, "\(code) may have reached the provider")
        }
        // A local throw before URLSession was touched is never billed, whatever
        // kind of error it is: the encoder's failure is not evidence of a send.
        struct LocalFault: Error {}
        XCTAssertFalse(CloudTurnFailure.transportFailure(
            LocalFault(),
            requestWasSent: false
        ).spentRequest)
        XCTAssertTrue(CloudTurnFailure.transportFailure(
            LocalFault(),
            requestWasSent: true
        ).spentRequest)
        // A classified failure passes through unchanged.
        XCTAssertEqual(
            CloudTurnFailure.transportFailure(
                CloudTurnFailure.httpStatus(code: 401, summary: "expired"),
                requestWasSent: true
            ),
            .httpStatus(code: 401, summary: "expired")
        )
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
        let notice = CloudCompactionFailure.provider("The provider returned HTTP 500.", spent: true)
            .notice(providerName: "Synthetic")
        XCTAssertTrue(notice.hasPrefix("⚠️"))
        XCTAssertTrue(notice.contains("HTTP 500"))
        XCTAssertTrue(notice.hasSuffix("it was left unchanged."))
    }

    func testATruncatedSummaryIsRefusedWithItsReason() {
        // The GATE, not just the wording: this is the branch that decides
        // whether a summary replaces the messages it was written from, so the
        // assertion has to be about what blocks the rewrite.
        XCTAssertEqual(
            CloudCompactionGate.refusal(for: CloudTextCompletion(
                text: "half a sentence", usage: nil, finishReason: "length"
            )),
            .truncatedSummary
        )
        XCTAssertEqual(
            CloudCompactionGate.refusal(for: CloudTextCompletion(
                text: "cut", usage: nil, finishReason: "MAX_TOKENS"
            )),
            .truncatedSummary,
            "the wire spelling is the provider's, so an exact-match test reads this as a finished reply"
        )
        XCTAssertEqual(
            CloudCompactionGate.refusal(for: CloudTextCompletion(
                text: "cut", usage: nil, finishReason: "content_filter"
            )),
            .truncatedSummary,
            "a provider-side filter also stops before the text covered the chunk"
        )
        XCTAssertEqual(
            CloudCompactionGate.refusal(for: CloudTextCompletion(
                text: "   ", usage: nil, finishReason: "stop"
            )),
            .emptySummary
        )
        XCTAssertNil(CloudCompactionGate.refusal(for: CloudTextCompletion(
            text: "the conversation so far", usage: nil, finishReason: "stop"
        )))
        XCTAssertNil(CloudCompactionGate.refusal(for: CloudTextCompletion(
            text: "the conversation so far", usage: nil, finishReason: "end_turn"
        )))
        // A stream that ended on `[DONE]` reports no reason, and that is the
        // protocol's own completion marker.
        XCTAssertNil(CloudCompactionGate.refusal(for: CloudTextCompletion(
            text: "the conversation so far", usage: nil
        )))

        let reason = CloudCompactionFailure.truncatedSummary
        XCTAssertTrue(reason.notice(providerName: "Synthetic").contains("did not finish cleanly"))
        XCTAssertTrue(reason.clause(providerName: "Synthetic").contains("did not finish cleanly"))
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
            .provider("HTTP 400", spent: false)
        ] {
            let clause = reason.clause(providerName: "Synthetic")
            XCTAssertFalse(clause.isEmpty)
            XCTAssertFalse(clause.hasPrefix("⚠️") || clause.hasPrefix("ℹ️") || clause.hasPrefix("⏹"))
            XCTAssertFalse(clause.contains("\n"), clause)
        }
    }

    func testASecondOverflowNamesTheCompactionItAlreadyRan() {
        // A run that compacted and still overflowed used to be told the
        // history "could not be compacted automatically", which was true of no
        // attempt at all and false of this one. The sentence names that a pass
        // ran without claiming a COUNT: a run may spend up to the rewrite
        // budget, so "once" was wrong after the second pass.
        let alreadyRan = CloudOverflowAdvice.text(
            alreadyCompacted: true,
            failure: nil,
            providerName: "Synthetic"
        )
        XCTAssertTrue(alreadyRan.contains("already compacted older history"), alreadyRan)
        XCTAssertFalse(alreadyRan.contains(" once"), alreadyRan)
        XCTAssertFalse(alreadyRan.contains("could not be compacted"), alreadyRan)

        // A recorded failure is the MOST RECENT attempt's cause, so it wins over
        // the earlier rewrite: "it compacted and then the summary came back
        // empty" calls for a different next step than "the result still did not
        // fit", and the old order reported the wrong one of the two.
        let failedAfterARewrite = CloudOverflowAdvice.text(
            alreadyCompacted: true,
            failure: .emptySummary,
            providerName: "Synthetic"
        )
        XCTAssertTrue(failedAfterARewrite.contains("compacted older history"), failedAfterARewrite)
        XCTAssertFalse(failedAfterARewrite.contains(" once"), failedAfterARewrite)
        XCTAssertTrue(failedAfterARewrite.contains("returned an empty summary"), failedAfterARewrite)

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
        // Case and spacing are the provider's, not ours.
        XCTAssertTrue(CloudTextCompletion(
            text: "half a sent",
            usage: nil,
            finishReason: " MAX_TOKENS "
        ).stoppedAtOutputLimit)
        // A provider that reports no reason is not assumed to have truncated.
        XCTAssertFalse(CloudTextCompletion(text: "done", usage: nil).stoppedAtOutputLimit)
        XCTAssertFalse(CloudTextCompletion(text: "done", usage: nil).finishedCleanly == false)
        // A clean stop is the only thing the destructive gate accepts.
        XCTAssertTrue(CloudTextCompletion(text: "done", usage: nil, finishReason: "stop").finishedCleanly)
        XCTAssertFalse(CloudTextCompletion(text: "done", usage: nil, finishReason: "content_filter").finishedCleanly)
        XCTAssertFalse(CloudTextCompletion(text: "done", usage: nil, finishReason: "some_new_dialect").finishedCleanly)
    }
}

// MARK: - Summarizer completions

/// A transport with a scripted stream, for the terminal-event rules a real
/// provider cannot be asked to reproduce on demand.
private nonisolated final class ScriptedCloudService: CloudChatServicing, @unchecked Sendable {
    private let script: @Sendable (AsyncStream<CloudTurnEvent>.Continuation) -> Void
    /// The params of the last request, so a case can prove the caller's
    /// budget reached the wire instead of only testing the pure function that
    /// computed it.
    private let recorder: (@Sendable (CloudChatGenerationParams) -> Void)?

    init(
        recorder: (@Sendable (CloudChatGenerationParams) -> Void)? = nil,
        script: @escaping @Sendable (AsyncStream<CloudTurnEvent>.Continuation) -> Void
    ) {
        self.recorder = recorder
        self.script = script
    }

    func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]?,
        receipt: CloudTurnReceipt?
    ) -> AsyncStream<CloudTurnEvent> {
        recorder?(params)
        return AsyncStream { continuation in
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
            budget: CloudSummarizerBudget(window: nil),
            sessionKey: nil
        )
    }

    func testTheSummarizerBudgetIsWhatTheRequestCarries() async throws {
        // The cap is pinned by `summaryResponseTokens`' own case, but that case
        // cannot see whether the caller passed it. A budget the type can only
        // build from a window, carried onto the wire here, closes the gap: a
        // constant at the summarizer call site no longer compiles.
        final class ParamsBox: @unchecked Sendable {
            private let lock = NSLock()
            private var maxTokens: Int?
            func record(_ params: CloudChatGenerationParams) {
                lock.lock()
                maxTokens = params.maxTokens
                lock.unlock()
            }
            var recordedMaxTokens: Int? {
                lock.lock()
                defer { lock.unlock() }
                return maxTokens
            }
        }
        let box = ParamsBox()
        let client = CloudChatClient(
            provider: CloudProvider(
                id: "p",
                displayName: "Synthetic",
                openAIBaseURL: URL(string: "https://example.invalid/v1")!
            ),
            service: ScriptedCloudService(recorder: { box.record($0) }) { continuation in
                continuation.yield(.finished(CloudTurnResult(
                    text: "summary",
                    thinking: "",
                    finishReason: "stop",
                    usage: nil
                )))
                continuation.finish()
            }
        )
        let budget = CloudSummarizerBudget(window: 200_000)
        XCTAssertEqual(budget.maxTokens, 6_600)
        _ = try await client.completeText(
            model: "m",
            systemPrompt: "summarize",
            userPrompt: "the chunk",
            budget: budget,
            sessionKey: nil
        )
        XCTAssertEqual(box.recordedMaxTokens, 6_600)
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

    func testAFinishedResultWithoutATerminalMarkerIsRefused() async {
        // A clean EOF is not a finished answer: the provider wrote a prefix and
        // closed the connection. The transport reports what it decoded, and
        // `terminated` is the protocol's own marker; a compaction that
        // committed this would delete every message the summary never covered.
        let client = client { continuation in
            continuation.yield(.started)
            continuation.yield(.contentDelta("half a sent"))
            continuation.yield(.finished(CloudTurnResult(
                text: "half a sent",
                thinking: "",
                finishReason: nil,
                usage: CloudUsage(promptTokens: 100, completionTokens: 3),
                terminated: false
            )))
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

    func testACancelledSummaryForwardsReceiptForRefusedAttempt() async {
        let service = ReceiptAwareCloudService { continuation, receipt in
            receipt?.record(spent: false)
            continuation.yield(.started)
            let held = Task {
                try? await Task.sleep(nanoseconds: 30_000_000_000)
            }
            continuation.onTermination = { _ in held.cancel() }
        }
        let client = CloudChatClient(
            provider: CloudProvider(
                id: "p",
                displayName: "Synthetic",
                openAIBaseURL: URL(string: "https://example.invalid/v1")!
            ),
            service: service
        )
        let receipt = CloudTurnReceipt()
        let task = Task {
            try await client.completeText(
                model: "m",
                systemPrompt: "summarize",
                userPrompt: "the chunk",
                budget: CloudSummarizerBudget(window: nil),
                sessionKey: nil,
                receipt: receipt
            )
        }
        try? await Task.sleep(nanoseconds: 200_000_000)
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected CancellationError")
        } catch is CancellationError {
            XCTAssertEqual(receipt.spentRequest, false)
            XCTAssertNil(receipt.result)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testACancelledSummaryPreservesReceiptUsageWhenTerminalResultRacesCancellation() async {
        let measuredUsage = CloudUsage(
            promptTokens: 320,
            completionTokens: 18,
            cachedTokens: 200,
            cacheWriteTokens: 0
        )
        let terminalResult = CloudTurnResult(
            text: "summary",
            thinking: "",
            finishReason: "stop",
            usage: measuredUsage
        )
        let service = ReceiptAwareCloudService { continuation, receipt in
            // Mirrors OpenAICloudChatService, which records the terminal result
            // on the receipt before yielding `.finished` / cancelling.
            receipt?.record(result: terminalResult)
            continuation.yield(.finished(terminalResult))
            continuation.finish()
        }
        let client = CloudChatClient(
            provider: CloudProvider(
                id: "p",
                displayName: "Synthetic",
                openAIBaseURL: URL(string: "https://example.invalid/v1")!
            ),
            service: service
        )
        let receipt = CloudTurnReceipt()
        let task = Task {
            try await client.completeText(
                model: "m",
                systemPrompt: "summarize",
                userPrompt: "the chunk",
                budget: CloudSummarizerBudget(window: nil),
                sessionKey: nil,
                receipt: receipt
            )
        }
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected CancellationError even when terminal result raced")
        } catch is CancellationError {
            XCTAssertEqual(receipt.spentRequest, true)
            XCTAssertEqual(receipt.result?.usage, measuredUsage)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }
}

private nonisolated struct ReceiptAwareCloudService: CloudChatServicing {
    let script: @Sendable (AsyncStream<CloudTurnEvent>.Continuation, CloudTurnReceipt?) -> Void

    func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]?,
        receipt: CloudTurnReceipt?
    ) -> AsyncStream<CloudTurnEvent> {
        AsyncStream { continuation in
            script(continuation, receipt)
        }
    }

    func fetchModelIds() async throws -> [String] {
        []
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

    func testNoCompletionCountMeansNoCostEither() {
        // Pricing the input alone showed a definite figure beside `(est.)` that
        // silently charged nothing for the reply — an understatement rendered
        // as if it were the whole answer. A cost needs both halves or neither.
        let usage = CloudUsage(promptTokens: 1_000)
        XCTAssertNil(CloudPricing(input: 1, output: 1).cost(for: usage))
        // An explicitly reported zero IS a count: the compaction path prices a
        // request it knows returned nothing.
        XCTAssertEqual(
            try XCTUnwrap(CloudPricing(input: 1, output: 1).cost(
                for: CloudUsage(promptTokens: 1_000, completionTokens: 0)
            )),
            0.001,
            accuracy: 1e-12
        )
    }

    func testAnUnknownModelIsSimplyUnpriced() {
        // No recorded price means no cost figure, not a guessed one: the status
        // line shows the token counts without a `$`, which is what the app did
        // before any price list existed. Asserting only that the resolved
        // pricing is nil would pass even if the report invented a price, so the
        // check is the RENDERED line.
        let suite = "CloudPricingTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        CloudPricingStore.setPricing(
            ["known": CloudPricing(input: 1, output: 2)],
            for: "p",
            defaults: defaults
        )
        let known = CloudPricingStore.pricing(
            forModel: "known",
            providerId: "p",
            defaults: defaults
        )
        XCTAssertNotNil(known)
        let unknown = CloudPricingStore.pricing(forModel: "unknown", providerId: "p", defaults: defaults)
        XCTAssertNil(unknown)
        let usage = CloudUsage(promptTokens: 1_000, completionTokens: 10)
        let priced = CloudTurnReport(
            usage: usage,
            estimatedPromptTokens: 0,
            outputTokens: 0,
            cost: known?.cost(for: usage)
        )
        XCTAssertTrue(priced.statusSuffix.contains("$"), priced.statusSuffix)
        let unpriced = CloudTurnReport(
            usage: usage,
            estimatedPromptTokens: 0,
            outputTokens: 0,
            cost: unknown?.cost(for: usage)
        )
        XCTAssertFalse(unpriced.statusSuffix.contains("$"), unpriced.statusSuffix)
        XCTAssertFalse(unpriced.wasEstimated, unpriced.statusSuffix)
        XCTAssertEqual(unpriced.statusSuffix, "1,000 in / 10 out")
    }

    func testAnAbsurdRateIsRefusedRatherThanPoisoningTheSpend() {
        // A finite 1e300 passes a finiteness check but overflows the product
        // with any real token count: the cost is `+inf`, the display renders it
        // as "$0", and one addition makes `cumulativeCost` infinite for the
        // rest of the conversation.
        let absurd = CloudPricing(input: 1e300, output: 1e300)
        XCTAssertFalse(absurd.isValid)
        XCTAssertFalse(CloudPricing(input: 1, output: 1e300).isValid)
        XCTAssertFalse(CloudPricing(input: 1, cachedInput: 1e300, output: 2).isValid)
        XCTAssertTrue(CloudPricing(input: 3, cachedInput: 0.3, output: 15, cacheWriteInput: 3.75).isValid)
        // A rate an order of magnitude above the largest list price is accepted;
        // the bound exists for units mistakes, not for expensive models.
        XCTAssertTrue(CloudPricing(input: 1_000, output: 2).isValid)
    }

    func testAnInfiniteCostIsNotReported() {
        // Directly constructed, bypassing `isValid`: the arithmetic guard is
        // the second half of the same rule.
        let runaway = CloudPricing(input: .greatestFiniteMagnitude, output: .greatestFiniteMagnitude)
        let cost = runaway.cost(for: CloudUsage(promptTokens: .max, completionTokens: .max))
        XCTAssertNil(cost)
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

    func testAnOverReportedCacheShareIsClampedToThePrompt() {
        // Cached input is a SUBSET of the prompt. Pricing already clamps it, so
        // a gateway that counts a cache write in both columns used to print
        // "1,000 in (5,000 cached)" beside a correctly discounted price: the
        // status line disagreed with the invoice.
        let one = CloudTurnReport(
            usage: CloudUsage(promptTokens: 1_000, completionTokens: 10, cachedTokens: 5_000),
            estimatedPromptTokens: 0,
            outputTokens: 10,
            cost: nil
        )
        XCTAssertEqual(one.cachedTokens, 1_000)
        XCTAssertEqual(one.statusSuffix, "1,000 in (1,000 cached) / 10 out")

        // The same rule on a run total, which accumulates one request at a
        // time: a single over-reported step cannot make the sum exceed the
        // prompt it is part of.
        var totals = CloudRunTotals()
        totals.add(
            usage: CloudUsage(promptTokens: 1_000, completionTokens: 10, cachedTokens: 5_000),
            estimatedPromptTokens: 0,
            estimatedOutputTokens: 0,
            cost: nil
        )
        XCTAssertEqual(totals.cachedTokens, 1_000)
        XCTAssertEqual(totals.report.statusSuffix, "1,000 billed in (1,000 of it cached) / 10 out")
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

    func testARunTotalSumsEveryStepRatherThanReportingTheLastOne() throws {
        // The regression this covers: the status line used to report the LAST
        // iteration's usage, naming a fraction of a long run while the header
        // pill and the session spend had already accumulated the whole thing.
        var totals = CloudRunTotals()
        totals.add(
            usage: CloudUsage(promptTokens: 1_000, completionTokens: 50, cachedTokens: 800),
            estimatedPromptTokens: 0,
            estimatedOutputTokens: 0,
            cost: 0.001
        )
        totals.add(
            usage: CloudUsage(promptTokens: 1_200, completionTokens: 60, cachedTokens: 900),
            estimatedPromptTokens: 0,
            estimatedOutputTokens: 0,
            cost: 0.0012
        )
        // A summarizer pass spends money without being an agent step, and its
        // tokens join the run's counts exactly as its cost joins its spend.
        totals.add(promptTokens: 4_000, outputTokens: 0, cost: 0.0004, estimated: false)
        XCTAssertEqual(totals.promptTokens, 6_200)
        XCTAssertEqual(totals.cachedTokens, 1_700)
        XCTAssertEqual(totals.outputTokens, 110)
        XCTAssertEqual(try XCTUnwrap(totals.cost), 0.0026, accuracy: 1e-12)
        XCTAssertFalse(totals.wasEstimated)
        XCTAssertEqual(
            totals.report.statusSuffix,
            "6,200 billed in (1,700 of it cached) / 110 out · $0.0026"
        )
    }

    func testAnEstimatedStepMarksTheRunAndDoesNotInventACost() throws {
        var totals = CloudRunTotals()
        totals.add(usage: nil, estimatedPromptTokens: 900, estimatedOutputTokens: 40, cost: nil)
        XCTAssertTrue(totals.wasEstimated)
        XCTAssertNil(totals.cost, "an estimate on its own does not produce a price")

        // A later measured step accumulates, and the estimate flag stays: the
        // line has to say one figure on it was estimated.
        totals.add(
            usage: CloudUsage(promptTokens: 1_000, completionTokens: 10),
            estimatedPromptTokens: 0,
            estimatedOutputTokens: 0,
            cost: 0.5
        )
        XCTAssertTrue(totals.wasEstimated)
        XCTAssertEqual(totals.promptTokens, 1_900)
        XCTAssertEqual(totals.outputTokens, 50)
        XCTAssertEqual(try XCTUnwrap(totals.cost), 0.5, accuracy: 1e-12)

        // The ONE case where estimated tokens carry a cost: a request that was
        // sent and never answered, priced from the estimate on purpose.
        totals.add(promptTokens: 500, outputTokens: 3, cost: 0.0002, estimated: true)
        XCTAssertEqual(try XCTUnwrap(totals.cost), 0.5002, accuracy: 1e-12)
    }

    func testARunTotalWithNoPricedStepReportsNoCost() {
        var totals = CloudRunTotals()
        XCTAssertTrue(totals.isEmpty)
        totals.add(
            usage: CloudUsage(promptTokens: 10, completionTokens: 2),
            estimatedPromptTokens: 0,
            estimatedOutputTokens: 0,
            cost: nil
        )
        XCTAssertFalse(totals.isEmpty)
        XCTAssertNil(totals.cost)
        XCTAssertFalse(totals.wasEstimated)
        // A provider that returns a nonsense (negative) cost is never credited.
        totals.add(promptTokens: 1, outputTokens: 1, cost: -5, estimated: false)
        XCTAssertNil(totals.cost)
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

    func testAnAbsurdUsageColumnSaturatesInsteadOfTrapping() {
        // Every column is provider-reported and unchecked. `+=` traps on
        // overflow, and the status line is rendered after the fact, so a
        // gateway repeating `Int.max` could crash the run it was reporting.
        var totals = CloudRunTotals()
        totals.add(
            promptTokens: Int.max,
            cachedTokens: Int.max,
            outputTokens: Int.max,
            cost: nil,
            estimated: false
        )
        totals.add(
            promptTokens: Int.max,
            cachedTokens: Int.max,
            outputTokens: Int.max,
            cost: nil,
            estimated: false
        )
        XCTAssertEqual(totals.promptTokens, Int.max)
        XCTAssertEqual(totals.cachedTokens, Int.max)
        XCTAssertEqual(totals.outputTokens, Int.max)
        // The cached column is still a subset of the prompt, so the displayed
        // figure cannot read as more cached than input.
        XCTAssertEqual(totals.report.cachedTokens, Int.max)
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
