//
//  CloudAgentBackendLogicTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for the cloud subagent backend. No network and no key:
//  the same `CloudChatServicing` seam the provider tests use drives scripted
//  streams through `CloudAgentBackend.runTurn`.
//

import XCTest
@testable import DynaMoE

@MainActor
final class CloudAgentBackendLogicTests: XCTestCase {

    // MARK: - Helpers

    private final class RequestRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [([CloudChatMessage], [ToolDefinition]?, String, CloudChatGenerationParams)] = []

        func record(
            model: String,
            messages: [CloudChatMessage],
            params: CloudChatGenerationParams,
            tools: [ToolDefinition]?
        ) {
            lock.lock()
            calls.append((messages, tools, model, params))
            lock.unlock()
        }

        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return calls.count
        }

        var messages: [[CloudChatMessage]] {
            lock.lock()
            defer { lock.unlock() }
            return calls.map(\.0)
        }

        var tools: [[ToolDefinition]?] {
            lock.lock()
            defer { lock.unlock() }
            return calls.map(\.1)
        }
    }

    private final class ScriptedService: CloudChatServicing, @unchecked Sendable {
        let recorder: RequestRecorder
        private let script: @Sendable (Int, AsyncStream<CloudTurnEvent>.Continuation) -> Void

        init(
            recorder: RequestRecorder,
            script: @escaping @Sendable (Int, AsyncStream<CloudTurnEvent>.Continuation) -> Void
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
            recorder.record(model: model, messages: messages, params: params, tools: tools)
            let attempt = recorder.count
            return AsyncStream { continuation in
                script(attempt, continuation)
            }
        }

        func fetchModelIds() async throws -> [String] { [] }
    }

    private static let provider = CloudProvider(
        id: "test-provider",
        displayName: "Test Provider",
        openAIBaseURL: URL(string: "https://example.invalid/v1")!
    )

    private static func backend(script: @escaping @Sendable (Int, AsyncStream<CloudTurnEvent>.Continuation) -> Void)
        -> (CloudAgentBackend, RequestRecorder)
    {
        let recorder = RequestRecorder()
        let client = CloudChatClient(provider: provider, service: ScriptedService(recorder: recorder, script: script))
        let backend = CloudAgentBackend(
            provider: provider,
            providerModelId: "test-model",
            client: client,
            displayName: "Test Model"
        )
        return (backend, recorder)
    }

    private static func request(
        messages: [AgentBackendMessage] = [.user("hello")],
        tools: [ToolDefinition] = [],
        onDelta: (@Sendable (String, Bool) -> Void)? = nil
    ) -> AgentBackendRequest {
        AgentBackendRequest(
            systemPrompt: "you are a worker",
            messages: messages,
            tools: tools,
            sampling: .default,
            maxTokens: 128,
            stopSequences: [],
            thinking: false,
            sessionId: UUID(),
            onDelta: onDelta
        )
    }

    // MARK: - Wire mapping

    func testWireMessagesMapEveryRole() {
        let call = AgentToolCall(id: "call_1", name: "file_read", arguments: ["path": "a.swift"], rawArguments: "{\"path\":\"a.swift\"}")
        let messages: [AgentBackendMessage] = [
            .user("find the bug"),
            .assistant(text: "reading", thinking: "think", calls: [call]),
            .toolResult(callId: "call_1", name: "file_read", content: "[file_read] ok", isError: false)
        ]
        let wire = CloudAgentBackend.wireMessages(systemPrompt: "sys", messages: messages)
        XCTAssertEqual(wire.map(\.role), ["system", "user", "assistant", "tool"])
        XCTAssertEqual(wire[0].content, "sys")
        XCTAssertEqual(wire[1].content, "find the bug")
        XCTAssertEqual(wire[2].content, "reading")
        XCTAssertEqual(wire[2].toolCalls?.count, 1)
        XCTAssertEqual(wire[2].toolCalls?.first?.id, "call_1")
        XCTAssertEqual(wire[2].toolCalls?.first?.function.name, "file_read")
        XCTAssertEqual(wire[2].toolCalls?.first?.function.arguments, "{\"path\":\"a.swift\"}")
        XCTAssertEqual(wire[3].toolCallId, "call_1")
        XCTAssertEqual(wire[3].content, "[file_read] ok")
    }

    func testAssistantToolCallArgumentsAreReEncodedWhenTheRawTextIsGone() {
        let call = AgentToolCall(
            id: "call_2",
            name: "grep_search",
            arguments: ["pattern": "todo", "path": "src"],
            rawArguments: ""
        )
        let wire = CloudAgentBackend.wireMessages(systemPrompt: "s", messages: [.assistant(text: "", thinking: "", calls: [call])])
        XCTAssertNil(wire[1].content, "an empty assistant body must stay empty, not become a blank string")
        let arguments = wire[1].toolCalls?.first?.function.arguments.data(using: .utf8)
        let object = arguments.flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: String]
        XCTAssertEqual(object?["pattern"], "todo")
        XCTAssertEqual(object?["path"], "src")
    }

    func testAnUnparseableAssembledArgumentBlockIsReportedRatherThanGuessedAt() {
        // No synthetic argument: the tool sees the arguments it was given
        // (none) and reports the missing required field by name, while the raw
        // JSON stays on the call's `rawArguments` for the transcript.
        XCTAssertTrue(CloudAgentBackend.arguments(from: "{\"path\":").isEmpty)
        XCTAssertTrue(CloudAgentBackend.arguments(from: "").isEmpty)
        XCTAssertTrue(CloudAgentBackend.arguments(from: "[]").isEmpty)
        XCTAssertEqual(CloudAgentBackend.arguments(from: "{\"a\":1}")["a"] as? Int, 1)
    }

    func testFragmentedToolCallArgumentsAssembleBeforeTheBackendSeesThem() throws {
        // The provider streams one call as a name fragment plus three argument
        // fragments; the backend must receive the assembly, because a half
        // object would execute a tool with guessed inputs.
        var parser = CloudStreamParser()
        _ = try parser.consume(dataPayload: #"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"file_read","arguments":"{\"pa"}}]}}]}"#)
        _ = try parser.consume(dataPayload: #"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"main"}}]}}]}"#)
        _ = try parser.consume(dataPayload: #"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":".swift\"}"}}]}}]}"#)
        let result = parser.result()
        let call = try XCTUnwrap(result.toolCalls.first)
        XCTAssertEqual(call.name, "file_read")
        XCTAssertEqual(call.argumentsJSON, "{\"path\":\"main.swift\"}")
        let decoded = CloudAgentBackend.arguments(from: call.argumentsJSON)
        XCTAssertEqual(decoded["path"] as? String, "main.swift")
    }

    // MARK: - Retry rules

    func testARateLimitedTurnIsRetriedOnce() async throws {
        let (backend, recorder) = Self.backend { attempt, continuation in
            if attempt == 1 {
                continuation.yield(.failed(.httpStatus(code: 429, summary: "slow down")))
            } else {
                continuation.yield(.finished(CloudTurnResult(text: "ok", thinking: "", finishReason: "stop", usage: nil)))
            }
            continuation.finish()
        }
        let turn = try await backend.runTurn(Self.request())
        XCTAssertEqual(turn.text, "ok")
        XCTAssertEqual(recorder.count, 2)
    }

    func testAClientErrorIsNotRetried() async throws {
        let (backend, recorder) = Self.backend { _, continuation in
            continuation.yield(.failed(.httpStatus(code: 401, summary: "no key")))
            continuation.finish()
        }
        do {
            _ = try await backend.runTurn(Self.request())
            XCTFail("a 401 must fail the step")
        } catch {
            XCTAssertEqual(recorder.count, 1, "a 4xx is not retryable: retrying it only delays the report")
        }
    }

    func testATransportFailureIsRetriedOnlyWhenThePromptProvablyNeverLeft() async throws {
        // A send that never reached the provider is safe to repeat.
        let (unbilledBackend, unbilledRecorder) = Self.backend { attempt, continuation in
            if attempt == 1 {
                continuation.yield(.failed(.transport("offline", billed: false)))
            } else {
                continuation.yield(.finished(CloudTurnResult(text: "ok", thinking: "", finishReason: "stop", usage: nil)))
            }
            continuation.finish()
        }
        let turn = try await unbilledBackend.runTurn(Self.request())
        XCTAssertEqual(turn.text, "ok")
        XCTAssertEqual(unbilledRecorder.count, 2)

        // A request that MAY have reached inference is not: a retry can be
        // invoiced twice for a turn the caller never sees.
        let (billedBackend, billedRecorder) = Self.backend { _, continuation in
            continuation.yield(.failed(.transport("connection dropped after send", billed: true)))
            continuation.finish()
        }
        do {
            _ = try await billedBackend.runTurn(Self.request())
            XCTFail("a possibly-invoiced transport failure must fail the step")
        } catch {
            XCTAssertEqual(billedRecorder.count, 1)
        }
    }

    func testAMissingKeyIsNotRetried() async throws {
        let (backend, recorder) = Self.backend { _, continuation in
            continuation.yield(.failed(.missingAPIKey(providerName: "Test Provider")))
            continuation.finish()
        }
        do {
            _ = try await backend.runTurn(Self.request())
            XCTFail("a missing key cannot be fixed by retrying")
        } catch {
            XCTAssertEqual(recorder.count, 1)
        }
    }

    func testRetryabilityIsKeyedOnStatusAndBillingOnly() {
        XCTAssertTrue(CloudAgentBackend.isRetryable(.httpStatus(code: 429, summary: "")))
        // A 5xx is not retryable at ANY receipt value. The live transport
        // records `spent: true` for every 5xx, so the old `spentRequest ==
        // false` arm could never fire; a 5xx means the provider accepted (and
        // may have run) the prompt, and a retry can be invoiced twice.
        XCTAssertFalse(
            CloudAgentBackend.isRetryable(.httpStatus(code: 503, summary: ""), spentRequest: false),
            "no receipt value makes a 5xx safe to re-send"
        )
        XCTAssertFalse(
            CloudAgentBackend.isRetryable(.httpStatus(code: 503, summary: ""), spentRequest: true),
            "a 5xx after the provider ran the prompt can be billed twice"
        )
        XCTAssertFalse(
            CloudAgentBackend.isRetryable(.httpStatus(code: 503, summary: ""), spentRequest: nil),
            "an undecided receipt is not evidence of a free retry"
        )
        XCTAssertFalse(CloudAgentBackend.isRetryable(.httpStatus(code: 400, summary: "")))
        XCTAssertFalse(CloudAgentBackend.isRetryable(.httpStatus(code: 404, summary: "")))
        XCTAssertFalse(CloudAgentBackend.isRetryable(.transport("x", billed: true)))
        XCTAssertTrue(CloudAgentBackend.isRetryable(.transport("x", billed: false)))
        XCTAssertFalse(CloudAgentBackend.isRetryable(.invalidResponse("x")))
        XCTAssertFalse(CloudAgentBackend.isRetryable(.unsupportedFlavor(flavor: "anthropic")))
    }

    // MARK: - Terminal rules

    func testAStreamThatNeverTerminatesIsAFailure() async throws {
        let (backend, _) = Self.backend { _, continuation in
            continuation.yield(.contentDelta("half a sentence"))
            continuation.finish()
        }
        do {
            _ = try await backend.runTurn(Self.request())
            XCTFail("a closed stream with no terminal event is not a completed turn")
        } catch let error as AgentBackendError {
            guard case .malformedResponse = error else {
                return XCTFail("expected a malformed-response error, got \(error)")
            }
        }
    }

    func testAnUnterminatedResultIsAFailure() async throws {
        let (backend, _) = Self.backend { _, continuation in
            continuation.yield(.finished(CloudTurnResult(
                text: "truncated",
                thinking: "",
                finishReason: nil,
                usage: nil,
                terminated: false
            )))
            continuation.finish()
        }
        do {
            _ = try await backend.runTurn(Self.request())
            XCTFail("a result without a completion marker is a fragment")
        } catch let error as AgentBackendError {
            guard case .malformedResponse = error else {
                return XCTFail("expected a malformed-response error, got \(error)")
            }
        }
    }

    func testDeltasArriveSplitByChannel() async throws {
        final class DeltaBox: @unchecked Sendable {
            private let lock = NSLock()
            private var entries: [(String, Bool)] = []
            func add(_ text: String, _ thinking: Bool) {
                lock.lock()
                entries.append((text, thinking))
                lock.unlock()
            }
            var all: [(String, Bool)] {
                lock.lock()
                defer { lock.unlock() }
                return entries
            }
        }
        let box = DeltaBox()
        let (backend, _) = Self.backend { _, continuation in
            continuation.yield(.thinkingDelta("weighing"))
            continuation.yield(.contentDelta("answer"))
            continuation.yield(.finished(CloudTurnResult(text: "answer", thinking: "weighing", finishReason: "stop", usage: nil)))
            continuation.finish()
        }
        let turn = try await backend.runTurn(Self.request(onDelta: { text, thinking in box.add(text, thinking) }))
        XCTAssertEqual(turn.text, "answer")
        XCTAssertEqual(turn.thinking, "weighing")
        XCTAssertEqual(box.all.map(\.0), ["weighing", "answer"])
        XCTAssertEqual(box.all.map(\.1), [true, false])
    }

    // MARK: - Usage and cost

    func testUsageAndCostComeFromTheProviderBlockAndThePriceList() {
        let pricing = CloudPricing(input: 2.0, output: 8.0)
        let usage = CloudUsage(promptTokens: 1_000, completionTokens: 500, totalTokens: 1_500)
        let mapped = CloudAgentBackend.usage(from: usage, pricing: pricing)
        XCTAssertEqual(mapped.promptTokens, 1_000)
        XCTAssertEqual(mapped.completionTokens, 500)
        // 1,000 fresh input at $2/M plus 500 output at $8/M.
        XCTAssertEqual(mapped.costUSD, 0.006, accuracy: 1e-9)
    }

    func testNoPricingMeansNoCostRatherThanAFreeTurn() {
        let mapped = CloudAgentBackend.usage(from: CloudUsage(promptTokens: 10, completionTokens: 10), pricing: nil)
        XCTAssertEqual(mapped.promptTokens, 10)
        XCTAssertEqual(mapped.completionTokens, 10)
        XCTAssertEqual(mapped.costUSD, 0)
        XCTAssertTrue(mapped.costIsUnpriced, "a 0.00 spend on a billed token count is unknown, not free")
    }

    func testAPricedTurnIsNotMarkedUnpriced() {
        let mapped = CloudAgentBackend.usage(
            from: CloudUsage(promptTokens: 1_000, completionTokens: 500),
            pricing: CloudPricing(input: 2.0, output: 8.0)
        )
        XCTAssertFalse(mapped.costIsUnpriced)
        XCTAssertGreaterThan(mapped.costUSD, 0)
    }

    func testTheUnpricedFlagSurvivesAMerge() {
        var first = AgentUsage()
        first.costIsUnpriced = true
        var merged = AgentUsage()
        merged.merge(first)
        merged.merge(AgentUsage())
        XCTAssertTrue(merged.costIsUnpriced)
    }

    func testExplicitZeroCountsAreAMeasuredZero() {
        // A block that reports literal zeros measured a zero-cost turn.
        let priced = CloudAgentBackend.usage(
            from: CloudUsage(promptTokens: 0, completionTokens: 0),
            pricing: CloudPricing(input: 2.0, output: 8.0)
        )
        XCTAssertEqual(priced.promptTokens, 0)
        XCTAssertEqual(priced.completionTokens, 0)
        XCTAssertEqual(priced.costUSD, 0)
        XCTAssertFalse(priced.costIsUnpriced, "a reported zero-count block IS a measured zero")
    }

    func testAPartialUsageBlockIsUnknownRatherThanFree() {
        // A provider that reports only `total_tokens` bills a turn the app
        // cannot price. Reading it as a measured zero made `max_cost_usd` see
        // "$0 spent" forever, so the ceiling never bound.
        let partial = CloudAgentBackend.usage(
            from: CloudUsage(totalTokens: 1_500),
            pricing: CloudPricing(input: 2.0, output: 8.0)
        )
        XCTAssertEqual(partial.costUSD, 0)
        XCTAssertTrue(partial.costIsUnpriced, "half a usage block is an unpriced bill, not a free turn")

        // An empty block is no measurement at all: the same answer.
        let empty = CloudAgentBackend.usage(from: CloudUsage(), pricing: CloudPricing(input: 2.0, output: 8.0))
        XCTAssertTrue(empty.costIsUnpriced, "a block without counts cannot prove a $0 turn either")
    }

    func testAMissingUsageBlockIsUnknownRatherThanFree() {
        // A Stop or a dropped terminal frame leaves a billed request with no
        // usage block. A max_cost_usd ceiling must see "unknown", not "$0".
        let mapped = CloudAgentBackend.usage(from: nil, pricing: CloudPricing(input: 2.0, output: 8.0))
        XCTAssertEqual(mapped.costUSD, 0)
        XCTAssertTrue(mapped.costIsUnpriced)
    }

    // MARK: - Request shape

    func testTheRunToolSchemasAndBudgetReachTheWire() async throws {
        let tool = ToolDefinition(
            name: "file_read",
            description: "read a file",
            parameters: ["type": AnyCodable("object")]
        )
        let (backend, recorder) = Self.backend { _, continuation in
            continuation.yield(.finished(CloudTurnResult(text: "done", thinking: "", finishReason: "stop", usage: nil)))
            continuation.finish()
        }
        _ = try await backend.runTurn(Self.request(tools: [tool]))
        XCTAssertEqual(recorder.tools.first.flatMap { $0 }?.first?.function.name, "file_read")
        XCTAssertEqual(recorder.count, 1)
    }

    // MARK: - Preferred-model resolution

    private func freshDefaults(_ name: String) -> UserDefaults {
        UserDefaults.standard.removePersistentDomain(forName: name)
        let defaults = UserDefaults(suiteName: name)!
        // A suite's plist is created in Preferences/ on first write and
        // nothing removed it, so every case that wrote to its suite leaked a
        // domain for the rest of this machine's life. The teardown block runs
        // in the case that created it.
        addTeardownBlock {
            defaults.removePersistentDomain(forName: name)
        }
        return defaults
    }

    func testAnEmptyCloudModelSettingResolvesNothingStored() {
        // no stored setting, no providers: nothing resolves, and the caller
        // refuses with the "no cloud model is configured" message.
        let defaults = freshDefaults("dynamoe-tests-cloud-empty")
        XCTAssertEqual(
            SubagentBackendSettings.preferredCloudModel(
                defaults: defaults, providers: []
            ),
            .noneConfigured
        )
    }

    /// Settings promises "empty means a cloud spawn refuses by name", and the
    /// old code fell back to a registered tool-capable model instead — billing
    /// the operator on a provider they never named, with a card that named no
    /// model. A configured provider in the list must not change the answer.
    func testAnEmptyCloudModelSettingRefusesRatherThanFallingBack() {
        let defaults = freshDefaults("dynamoe-tests-cloud-empty-fallback")
        XCTAssertEqual(
            SubagentBackendSettings.preferredCloudModel(
                defaults: defaults, providers: [Self.provider]
            ),
            .noneConfigured
        )
    }

    func testRetryAfterIsHonouredInBothSpellingsAndClamped() {
        func response(_ value: String) -> HTTPURLResponse {
            HTTPURLResponse(
                url: URL(string: "https://example.invalid/v1/chat/completions")!,
                statusCode: 429,
                httpVersion: nil,
                headerFields: ["Retry-After": value]
            )!
        }
        XCTAssertEqual(CloudRetryAfter.seconds(from: response("2")), 2)
        XCTAssertEqual(CloudRetryAfter.seconds(from: response("0")), 0)
        XCTAssertEqual(
            CloudRetryAfter.seconds(from: response("3600")),
            60,
            "one retry must not park the run for the provider's full window"
        )
        XCTAssertNil(CloudRetryAfter.seconds(from: response("not-a-date")))
        XCTAssertNil(
            CloudRetryAfter.seconds(
                from: HTTPURLResponse(
                    url: URL(string: "https://example.invalid")!,
                    statusCode: 429,
                    httpVersion: nil,
                    headerFields: nil
                )!
            )
        )
    }

    func testAStoredCloudModelThatParsesIsThePreference() {
        let defaults = freshDefaults("dynamoe-tests-cloud-parses")
        let address = CloudModelAddress.modelId(providerId: "test-provider", providerModelId: "test-model")
        defaults.set(address, forKey: SubagentBackendSettings.cloudModelKey)
        XCTAssertEqual(
            SubagentBackendSettings.preferredCloudModel(
                defaults: defaults, providers: [Self.provider]
            ),
            .configured(address)
        )
    }

    func testAStoredCloudModelThatDoesNotParseIsRefusedByName() {
        // The old behavior silently substituted the provider default here,
        // billing runs on a model the operator never chose.
        let defaults = freshDefaults("dynamoe-tests-cloud-broken")
        defaults.set("not-a-cloud-model", forKey: SubagentBackendSettings.cloudModelKey)
        XCTAssertEqual(
            SubagentBackendSettings.preferredCloudModel(
                defaults: defaults, providers: [Self.provider]
            ),
            .unresolvable("not-a-cloud-model")
        )
    }

    func testASpawnRefusesAnUnresolvableStoredCloudModelInsteadOfSubstituting() async {
        let defaults = freshDefaults("dynamoe-tests-cloud-spawn-broken")
        defaults.set("not-a-cloud-model", forKey: SubagentBackendSettings.cloudModelKey)
        do {
            _ = try await AgentSpawnResolver.resolve(
                AgentSpawnRequest(
                    role: "Codebase Researcher",
                    taskDescription: "Find the scheduler.",
                    backendPreference: .cloud,
                    explicitToolNames: ["file_read"],
                    workingDirectory: FileManager.default.temporaryDirectory
                ),
                runId: UUID(),
                sessionId: UUID(),
                defaults: defaults
            )
            XCTFail("a spawn must refuse an unresolvable stored cloud model")
        } catch {
            guard case .noBackend(let message)? = error as? AgentSpawnError else {
                return XCTFail("expected a noBackend refusal, got \(error)")
            }
            XCTAssertTrue(message.contains("not-a-cloud-model"), message)
        }
    }

    // MARK: - Opt-in real turn

    /// One real cloud turn, on the first configured model that will serve it.
    ///
    /// Opt-in: `touch /tmp/dynamoe_subagent_cloud_turn`. It walks the spawn's own
    /// candidates because which ones exist is machine state: the preferred model
    /// first, then every discovered model (skipping `infinite:` ones, which need
    /// a plan subscription this Mac may not have). A provider that refuses with
    /// an HTTP status is a candidate that cannot serve the run, not a defect, so
    /// the case SKIPS when they all refuse and names the last refusal; any other
    /// error fails. It asserts the SHAPE a real provider produces — text, token
    /// counts — because wording and counts are not deterministic.
    func testARealCloudSubagentTurnAnswers() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_subagent_cloud_turn"),
            "opt-in: touch /tmp/dynamoe_subagent_cloud_turn"
        )
        var candidates: [String] = []
        if case .configured(let preferred) = SubagentBackendSettings.preferredCloudModel() {
            candidates.append(preferred)
        }
        for model in CloudModelRegistry.makeDiscoveredModels() where !model.id.lowercased().contains("infinite") {
            if !candidates.contains(model.id) { candidates.append(model.id) }
        }
        guard !candidates.isEmpty else {
            throw XCTSkip("no configured cloud model with tool support")
        }

        let request = AgentBackendRequest(
            systemPrompt: "You are a subagent. Answer in exactly one short sentence.",
            messages: [.user("What is the capital of France?")],
            tools: [],
            sampling: AgentSampling(temperature: 0, topP: 1, maxTokens: 32),
            maxTokens: 32,
            stopSequences: [],
            thinking: false,
            sessionId: UUID(),
            onDelta: nil
        )

        var refusals: [String] = []
        for address in candidates {
            let backend: CloudAgentBackend
            switch CloudAgentBackend.make(address: address) {
            case .success(let made): backend = made
            case .failure: continue
            }
            do {
                let turn = try await backend.runTurn(request)
                XCTAssertFalse(turn.text.isEmpty)
                XCTAssertGreaterThan(turn.usage.completionTokens, 0)
                XCTAssertGreaterThan(turn.usage.promptTokens, 0)
                print(
                    "🔎 [subagent real] cloud turn on \(backend.descriptor.displayName): "
                        + "\(turn.usage.promptTokens) prompt / \(turn.usage.completionTokens) generated, "
                        + "cost \(String(format: "%.6f", turn.usage.costUSD)) USD, "
                        + "text: \(turn.text.prefix(160))"
                )
                return
            } catch let error as AgentBackendError {
                let message = error.errorDescription ?? ""
                // `runTurn` wraps every provider failure as a transport error, so
                // the HTTP status only survives in the message. A 4xx is the
                // provider declining this model; anything else is a real failure.
                if message.contains("HTTP 4") {
                    refusals.append("\(backend.descriptor.displayName): \(message)")
                    continue
                }
                throw error
            }
        }
        throw XCTSkip("every configured cloud model refused this turn: \(refusals.suffix(2).joined(separator: " | "))")
    }

    /// One real CLOUD subagent RUN through `SubagentManager`.
    ///
    /// The turn test above pins the wire; this pins the manager, the loop, the
    /// run-scoped harness and the cloud scheduler class around it. Kept on its
    /// own marker so the turn test can be used without spending a run.
    func testARealCloudSubagentRunAnswersThroughTheManager() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_subagent_cloud_run"),
            "opt-in: touch /tmp/dynamoe_subagent_cloud_run"
        )
        var candidates: [String] = []
        if case .configured(let preferred) = SubagentBackendSettings.preferredCloudModel() {
            candidates.append(preferred)
        }
        for model in CloudModelRegistry.makeDiscoveredModels() where !model.id.lowercased().contains("infinite") {
            if !candidates.contains(model.id) { candidates.append(model.id) }
        }
        guard !candidates.isEmpty else {
            throw XCTSkip("no configured cloud model with tool support")
        }

        let workingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubagentCloudRun_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
        defer {
            SubagentManager.shared.cancelAll()
            try? FileManager.default.removeItem(at: workingDirectory)
        }

        var refusals: [String] = []
        for address in candidates {
            let backend: CloudAgentBackend
            switch CloudAgentBackend.make(address: address) {
            case .success(let made): backend = made
            case .failure: continue
            }
            let providerId = backend.descriptor.providerId ?? "cloud"
            let runId = UUID()
            let plan = AgentSpawnPlan(
                role: "Codebase Researcher",
                taskDescription: "In one short sentence, what is the capital of France?",
                contextSummary: nil,
                preset: .explore,
                backendDescriptor: backend.descriptor,
                schedulingClass: .cloud(providerId: providerId),
                backend: backend,
                // No tools on purpose: this case prices the manager/loop/wire
                // seam, and a tool-less turn is the shortest one a provider
                // answers with text.
                toolNames: [],
                budget: AgentBudget(
                    maxSteps: 2,
                    maxCompletionTokens: 400,
                    maxWallClockSeconds: 60,
                    maxCostUSD: nil
                ),
                approvalPolicy: .auto,
                isolation: .none,
                thinking: false,
                // 256, not the 96 the first cut used: a small model on the
                // real worker prompt can spend a hundred tokens before its
                // first visible word, and a step cut at its own cap is a
                // finished answer to the loop (finishReason is advisory
                // there), so the cap has to leave room for one sentence.
                sampling: AgentSampling(temperature: 0, topP: 1, maxTokens: 256),
                priority: .userRequested,
                parentSessionId: nil,
                parentTurnId: nil,
                workingDirectory: workingDirectory
            )
            let subagent = SubagentManager.shared.spawnModelBacked(plan, id: runId)
            let summary = await subagent.waitForCompletion()
            if subagent.status == .completed {
                XCTAssertTrue(summary.lowercased().contains("paris"), "summary: \(summary)")
                XCTAssertEqual(subagent.report?.backend, backend.descriptor)
                print(
                    "🔎 [subagent real] cloud run on \(backend.descriptor.displayName): "
                        + "\(subagent.usage.promptTokens) prompt / \(subagent.usage.completionTokens) generated, "
                        + "cost \(String(format: "%.6f", subagent.usage.costUSD)) USD, "
                        + "summary: \(summary.prefix(160))"
                )
                return
            }
            // `runTurn` wraps every provider failure as a transport error, so the
            // HTTP status only survives in the message. A 4xx is the provider
            // declining this model; anything else is a real failure.
            if summary.contains("HTTP 4") {
                refusals.append("\(backend.descriptor.displayName): \(summary)")
                continue
            }
            XCTFail("cloud run failed: \(summary)")
            return
        }
        throw XCTSkip("every configured cloud model refused this run: \(refusals.suffix(2).joined(separator: " | "))")
    }

    func testTheDescriptorCarriesTheModelIdentityAndKind() {
        let (backend, _) = Self.backend { _, continuation in
            continuation.finish()
        }
        XCTAssertEqual(backend.descriptor.kind, .cloud)
        XCTAssertEqual(backend.descriptor.modelId, "test-model")
        XCTAssertEqual(backend.descriptor.displayName, "Test Model")
        XCTAssertTrue(backend.descriptor.isRemote)
    }
}
