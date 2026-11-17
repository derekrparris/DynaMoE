//
//  LocalAgentBackend.swift
//  DynaMoE
//
//  Local-engine backend for a subagent run. See SUBAGENTS.md §6.
//

import Foundation

/// A subagent turn served by the app's own model.
///
/// The run is rendered with the loaded family's formatter and launched through
/// the same engine path the chat UI and the local API server use, so prompt
/// templating, prefix pinning, tool masking and mid-stream tool-call freezing are
/// the app's own behaviour rather than a second implementation.
///
/// **The engine slot is taken per step, not per run, and Phase 2 kept it that
/// way.** `launchTurn` consumes the `TurnSlotClaim` it is handed as soon as it
/// has minted the generation task, and the ledger holds one claim, so a claim
/// cannot be carried across steps without changing the mechanism the server
/// queue and the chat send path depend on. A lifetime lease was measured and
/// rejected (§8), because holding the only slot for a whole run blocks every
/// chat turn. The consequence is stated plainly rather than hidden: a
/// coordinator turn that lands between two steps moves the global prefix pin to
/// its own session, so the run's next step re-prefills. `AgentScheduler` owns
/// admission and cancellation; the turn is `.background`, so an operator's send
/// CANCELS it at its next boundary and the run reports what it lost. It is not
/// paused and not resumed.
nonisolated final class LocalAgentBackend: AgentModelBackend, @unchecked Sendable {
    let descriptor: AgentBackendDescriptor

    private let thinkingEnabled: Bool
    private let runId: UUID

    init(descriptor: AgentBackendDescriptor, thinkingEnabled: Bool, runId: UUID) {
        self.descriptor = descriptor
        self.thinkingEnabled = thinkingEnabled
        self.runId = runId
    }

    /// Builds a backend for a run, refusing by name when the app cannot serve it.
    ///
    /// The cost estimate is consulted HERE, not in the spawn path, so no caller
    /// can start a local run without pricing it; `operatorOverride` is the one
    /// thing that turns a refusal into an admission, and it is the spawn card's
    /// decision (invariant 10).
    @MainActor
    static func make(
        runId: UUID,
        sessionId: UUID,
        thinking: Bool,
        promptTokenEstimate: Int = 0,
        operatorOverride: Bool = false,
        defaults: UserDefaults = .standard
    ) throws -> LocalAgentBackend {
        let engine = CompletionEngine.shared
        guard let info = engine.loadedModelInfo, engine.promptModelConfig != nil else {
            throw AgentBackendError.unavailable(
                "A local subagent run needs a loaded model. Load a checkpoint in Settings first, or send this work to a cloud backend."
            )
        }
        let estimate = costEstimate(
            runSessionId: sessionId,
            promptTokenEstimate: promptTokenEstimate,
            defaults: defaults
        )
        try admission(hasLoadedModel: true, estimate: estimate, operatorOverride: operatorOverride)
        return LocalAgentBackend(
            descriptor: AgentBackendDescriptor(
                kind: .local,
                providerId: nil,
                modelId: info.servedID,
                displayName: info.servedID
            ),
            thinkingEnabled: thinking,
            runId: runId
        )
    }

    /// The two ways a local run is refused, as a pure function so both are
    /// testable without a loaded checkpoint.
    static func admission(
        hasLoadedModel: Bool,
        estimate: LocalSubagentCostEstimate,
        operatorOverride: Bool
    ) throws {
        guard hasLoadedModel else {
            throw AgentBackendError.unavailable(
                "A local subagent run needs a loaded model. Load a checkpoint in Settings first, or send this work to a cloud backend."
            )
        }
        guard estimate.admits(operatorOverride: operatorOverride) else {
            throw AgentBackendError.refused(
                (estimate.refusalMessage ?? "This local run exceeds the configured cost limit.")
                    + " Estimated cost: " + estimate.summary + "."
            )
        }
    }

    /// What this spawn would cost the coordinator. Pure of engine state except the
    /// prefix pin, so the spawn card can show the number before deciding.
    static func costEstimate(
        runSessionId: UUID,
        promptTokenEstimate: Int,
        tokensPerSecond: Double? = nil,
        limitSeconds: Double? = nil,
        defaults: UserDefaults = .standard
    ) -> LocalSubagentCostEstimate {
        let pin = PrefixCacheManager.shared
        // Only a pin owned by ANOTHER session is displaced by this run. When the
        // pin is already the run's own session (a resumed run), the coordinator's
        // next turn pays for the run's session either way, and the pin length is
        // not the coordinator's prefix.
        let displaced = pin.currentPinnedSession != runSessionId ? pin.currentPinnedCount : 0
        return LocalSubagentCostEstimate.make(
            coordinatorPrefixTokens: displaced,
            runPromptTokens: promptTokenEstimate,
            tokensPerSecond: tokensPerSecond ?? PrefillThroughput.resolvedTokensPerSecond(),
            limitSeconds: limitSeconds ?? LocalSubagentCostEstimate.configuredLimitSeconds(defaults: defaults)
        )
    }

    func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn {
        let prepared = try await MainActor.run { try Self.prepare(request, thinkingEnabled: thinkingEnabled) }
        do {
            return try await GrammarConstrainedSampler.withTurnToolMask(prepared.maskSchemas) {
                let launch = try await MainActor.run { () -> CompletionLaunch in
                    try CompletionEngine.shared.launchTurn(prepared.turn, claim: prepared.claim)
                }
                return try await Self.collect(launch: launch, request: request)
            }
        } catch {
            // `launchTurn` releases the claim it was handed on every path it
            // reaches; this covers the window before it is called at all.
            prepared.claim.release()
            throw error
        }
    }

    // MARK: - Turn preparation

    struct Prepared {
        var turn: CompletionTurnRequest
        var claim: TurnSlotClaim
        var maskSchemas: [GrammarToolSchema]
    }

    @MainActor
    static func prepare(_ request: AgentBackendRequest, thinkingEnabled: Bool) throws -> Prepared {
        let engine = CompletionEngine.shared
        guard let info = engine.loadedModelInfo, let config = engine.promptModelConfig else {
            throw AgentBackendError.unavailable(
                "A local subagent run needs a loaded model. Load a checkpoint in Settings first."
            )
        }
        let prompt = renderPrompt(
            systemPrompt: request.systemPrompt,
            messages: request.messages,
            thinkingEnabled: thinkingEnabled && info.supportsThinking,
            modelSupportsThinking: info.supportsThinking,
            modelConfig: config,
            promptDialect: engine.promptDialect
        )
        let promptTokenIds: [UInt32]
        do {
            promptTokenIds = try engine.tokenize(prompt)
        } catch {
            throw AgentBackendError.malformedResponse(
                "The tokenizer could not encode the subagent prompt: \(error.localizedDescription)"
            )
        }
        guard !promptTokenIds.isEmpty else {
            throw AgentBackendError.malformedResponse("The subagent prompt tokenized to zero tokens.")
        }
        guard let claim = engine.reserveTurnSlot() else {
            throw AgentBackendError.unavailable(
                "The local engine is busy with another turn. A local subagent run needs the generation slot; retry when the turn finishes."
            )
        }
        let turn = CompletionTurnRequest(
            promptTokenIds: promptTokenIds,
            formattedPrompt: prompt,
            sessionId: request.sessionId,
            sampling: TurnSampling(
                temperature: Float(request.sampling.temperature),
                topP: Float(request.sampling.topP),
                minP: 0,
                topK: 40,
                repetitionPenalty: 1.0,
                maxTokens: max(1, request.maxTokens)
            ),
            memory: TurnMemorySettings.persisted(),
            agentToolsEnabled: !request.tools.isEmpty,
            thinkingEnabled: thinkingEnabled && info.supportsThinking,
            agentStep: 0,
            forceSynthesis: false,
            isSplicedContinuation: false,
            stopSequences: request.stopSequences,
            asOfDate: nil,
            images: [],
            imageSpans: [],
            turnPriority: .background
        )
        return Prepared(turn: turn, claim: claim, maskSchemas: maskSchemas(for: request.tools))
    }

    // MARK: - Stream

    private static func collect(
        launch: CompletionLaunch,
        request: AgentBackendRequest
    ) async throws -> AgentBackendTurn {
        var emittedText = ""
        var emittedThinking = ""
        for await event in launch.stream {
            switch event {
            case .started, .priming, .prefillProgress:
                continue
            case .frame(let frame):
                if let onDelta = request.onDelta {
                    let thinkingDelta = suffixDelta(from: emittedThinking, to: frame.thinkingText)
                    if !thinkingDelta.isEmpty { onDelta(thinkingDelta, true) }
                    let textDelta = suffixDelta(from: emittedText, to: frame.responseText)
                    if !textDelta.isEmpty { onDelta(textDelta, false) }
                }
                emittedThinking = frame.thinkingText
                emittedText = frame.responseText
            case .failedTurn(let failure):
                if failure.cancelled || Task.isCancelled { throw AgentBackendError.cancelled }
                throw AgentBackendError.transport(failure.message)
            case .finished(let result):
                // `TurnResult` carries main-actor-isolated value types, so its
                // mapping is read there; the stream itself is consumed here.
                return await MainActor.run { Self.turn(from: result) }
            }
        }
        throw AgentBackendError.malformedResponse("The local engine's stream closed without a result.")
    }

    /// What arrived since the last frame. Frames carry the whole text so far, so a
    /// delta is the suffix the previous snapshot did not have; a rewritten prefix
    /// (a repaired UTF-8 boundary) is emitted whole rather than dropped.
    static func suffixDelta(from previous: String, to current: String) -> String {
        guard !current.isEmpty else { return "" }
        if current.hasPrefix(previous) {
            return String(current.dropFirst(previous.count))
        }
        return current
    }

    @MainActor
    static func turn(from result: TurnResult) -> AgentBackendTurn {
        var calls: [AgentToolCall] = []
        for index in result.actionableCallIndices where result.parsedToolCalls.indices.contains(index) {
            let call = result.parsedToolCalls[index]
            calls.append(AgentToolCall(
                id: "call_local_\(index)",
                name: call.name,
                arguments: call.arguments,
                rawArguments: call.rawArguments,
                rawText: call.rawText
            ))
        }
        var usage = AgentUsage()
        usage.promptTokens = result.promptTokenIds.count
        usage.completionTokens = result.generatedTokenIds.count
        usage.cachedTokens = result.prefixTokensReused
        return AgentBackendTurn(
            text: result.responseText,
            thinking: result.thinkingText,
            calls: calls,
            rawText: result.rawDecodedText,
            finishReason: result.finishReason.rawValue,
            usage: usage
        )
    }

    // MARK: - Rendering (pure)

    /// Renders the run's conversation with the loaded family's own template.
    ///
    /// The shape deliberately matches the local API server's stateless path
    /// (`OpenAIAdapter.splitMessages` + `ServerController`'s formatter call): a
    /// run's history is rebuilt from its own messages every step, so a step's
    /// prompt is an extension of the previous step's prompt+generation whenever
    /// the renderer is deterministic — which is what keeps the run's own prefix
    /// reusable between steps.
    @MainActor
    static func renderPrompt(
        systemPrompt: String,
        messages: [AgentBackendMessage],
        thinkingEnabled: Bool,
        modelSupportsThinking: Bool,
        modelConfig: ModelConfig?,
        promptDialect: DeepSeekV41PromptDialect = DeepSeekV41PromptDialect()
    ) -> String {
        let turns = conversationTurns(from: messages)
        guard let first = turns.first, turns.count == 1, first.role == .user else {
            return ChatPromptFormatter.formatConversation(
                turns: turns,
                cleanSystem: systemPrompt,
                thinkingEnabled: thinkingEnabled,
                modelSupportsThinking: modelSupportsThinking,
                modelConfig: modelConfig,
                promptDialect: promptDialect
            )
        }
        return ChatPromptFormatter.formatPrompt(
            prompt: first.text,
            cleanSystem: systemPrompt,
            thinkingEnabled: thinkingEnabled,
            modelSupportsThinking: modelSupportsThinking,
            modelConfig: modelConfig,
            promptDialect: promptDialect
        )
    }

    /// The run's messages as conversation turns.
    ///
    /// A tool result is a TOOL RESULT turn, not an ordinary user turn: the family
    /// decides how one is rendered and only the turn's kind can say so (`role` is
    /// `.user` for both). A call whose markup is not already in the assistant
    /// text is echoed with the server's own plain-text call record rather than a
    /// dialect this backend cannot spell for every family.
    @MainActor
    static func conversationTurns(from messages: [AgentBackendMessage]) -> [ChatPromptFormatter.ConversationTurn] {
        messages.map { message in
            switch message {
            case .user(let text):
                return ChatPromptFormatter.ConversationTurn(role: .user, text: text)
            case .assistant(let text, let thinking, let calls):
                var lines: [String] = []
                let reasoning = thinking.trimmingCharacters(in: .whitespacesAndNewlines)
                if !reasoning.isEmpty {
                    lines.append(reasoning + TurnTextSplitter.canonicalReasoningCloseTag)
                }
                if !text.isEmpty { lines.append(text) }
                if !calls.isEmpty, !carriesCallMarkup(text) {
                    lines.append(ChatPromptFormatter.openAIToolCallEchoPrefix + "\n" + callEcho(calls))
                }
                let body = lines.joined(separator: "\n\n")
                return ChatPromptFormatter.ConversationTurn(role: .assistant, text: body.isEmpty ? "(no content)" : body)
            case .toolResult(_, let name, let content, _):
                return ChatPromptFormatter.ConversationTurn(
                    role: .user,
                    text: "Function result (\(name)):\n\(content)",
                    kind: .toolResult(name: name)
                )
            }
        }
    }

    @MainActor
    static func carriesCallMarkup(_ text: String) -> Bool {
        text.contains(DeepSeekV41ToolDialect.dsmlTag)
            || text.contains(StreamingToolParser.qwenToolCallOpen)
            || text.contains(StreamingToolParser.gemmaToolCallOpen)
            || text.contains(StreamingToolParser.llamaTagOpen)
    }

    @MainActor
    static func callEcho(_ calls: [AgentToolCall]) -> String {
        calls.map { call in
            var object: [String: String] = ["name": call.name]
            if !call.id.isEmpty { object["id"] = call.id }
            let arguments = call.rawArguments.isEmpty
                ? Self.argumentsJSON(call.arguments)
                : call.rawArguments
            if !arguments.isEmpty, arguments != "{}" { object["arguments"] = arguments }
            if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
               let json = String(data: data, encoding: .utf8) {
                return json
            }
            return "{\"name\": \"\(call.name)\"}"
        }
        .joined(separator: "\n")
    }

    @MainActor
    static func argumentsJSON(_ arguments: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(arguments),
              let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    /// The run's tool schemas, as the grammar sampler's mask vocabulary. Empty
    /// when the run was given no tools, which leaves the sampler's registry
    /// untouched (see `GrammarConstrainedSampler.withTurnToolMask`).
    @MainActor
    static func maskSchemas(for tools: [ToolDefinition]) -> [GrammarToolSchema] {
        tools.map { tool in
            // Encode the PARAMETERS DICTIONARY, not a wrapper around it: each
            // value is already an `AnyCodable`, and wrapping the dictionary in a
            // second one gives it no encoding of its own (every branch in
            // `AnyCodable.encode` misses), so the schema would decode as an empty
            // object and the mask would constrain nothing.
            let schemaData = try? JSONEncoder().encode(tool.function.parameters)
            return GrammarToolSchema(name: tool.function.name, schemaJSON: schemaData)
        }
    }

    /// The worker prompt with the family's own tool-call syntax. The spawn path
    /// builds this before the run starts; the flags are the loaded checkpoint's.
    ///
    /// `harness` is the RUN's own instance, never `AgentHarness.shared`: the
    /// builder reads the run's tool section, and a run that read the
    /// coordinator's would describe tools it does not have.
    @MainActor
    static func workerSystemPrompt(
        role: String,
        baseSystem: String,
        mode: AgentMode,
        harness: AgentHarness,
        config: ModelConfig?
    ) -> String {
        harness.buildWorkerSystemPrompt(
            baseSystem: baseSystem,
            role: role,
            mode: mode,
            isLingModel: config?.isLingModel == true,
            isGemma4Model: config?.isGemma4Model == true,
            isDeepSeekV41Model: config?.isDeepSeekV41Model == true
        )
    }
}
