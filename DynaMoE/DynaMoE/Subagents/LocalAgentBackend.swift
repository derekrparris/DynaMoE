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
        estimatedCost: LocalSubagentCostEstimate? = nil,
        operatorOverride: Bool = false,
        approvedEstimate: LocalSubagentCostEstimate? = nil,
        defaults: UserDefaults = .standard
    ) throws -> LocalAgentBackend {
        let engine = CompletionEngine.shared
        guard let info = engine.loadedModelInfo, engine.promptModelConfig != nil else {
            throw AgentBackendError.unavailable(
                "A local subagent run needs a loaded model. Load a checkpoint in Settings first, or send this work to a cloud backend."
            )
        }
        // The caller's estimate is the one the spawn card showed, computed
        // from the same pin and rate a moment earlier. Recomputing here is
        // only the fallback for callers with no card.
        let estimate = estimatedCost ?? costEstimate(
            runSessionId: sessionId,
            promptTokenEstimate: promptTokenEstimate,
            defaults: defaults
        )
        try admission(
            hasLoadedModel: true,
            estimate: estimate,
            operatorOverride: operatorOverride,
            approvedEstimate: approvedEstimate
        )
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
        operatorOverride: Bool,
        approvedEstimate: LocalSubagentCostEstimate? = nil
    ) throws {
        guard hasLoadedModel else {
            throw AgentBackendError.unavailable(
                "A local subagent run needs a loaded model. Load a checkpoint in Settings first, or send this work to a cloud backend."
            )
        }
        if operatorOverride,
           let approvedEstimate,
           let drift = estimate.approvalDriftMessage(approved: approvedEstimate) {
            throw AgentBackendError.refused(drift)
        }
        guard estimate.admits(
            operatorOverride: operatorOverride,
            approvedEstimate: approvedEstimate
        ) else {
            throw AgentBackendError.refused(
                (estimate.refusalMessage ?? "This local run exceeds the configured cost limit.")
                    + " Estimated cost: " + estimate.summary + "."
            )
        }
    }

    /// The prompt size a spawn card prices. One function so the estimate the
    /// card SHOWS and the estimate `make` compares against can never be
    /// computed from different inputs.
    static func spawnPromptTokenEstimate(task: String, contextSummary: String?) -> Int {
        max(1, (task.count + (contextSummary?.count ?? 0)) / 4 + 600)
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

    @concurrent func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn {
        let prepared = try await Self.prepare(request, thinkingEnabled: thinkingEnabled)
        // A run cancelled while it prepared (rendering, tokenizing, waiting for
        // the slot) must not burn a full turn nobody will process: a DeepSeek
        // launch rebuilds the runtime (measured 17.2 s) and moves the global
        // prefix pin on its way through. The claim is given back here, which
        // is what keeps the refusal from wedging the single-flight slot.
        if Task.isCancelled {
            prepared.claim.release()
            throw AgentBackendError.cancelled
        }
        do {
            return try await GrammarConstrainedSampler.withTurnToolMask(prepared.maskSchemas) {
                let launch = try await MainActor.run { () -> CompletionLaunch in
                    // Cancellation can land during the hop to the main actor,
                    // after the check above and before this closure runs. The
                    // check must be INSIDE the hop: a launch resets the KV
                    // cache and moves the prefix pin, so a stopped run that
                    // still launched paid for a turn nobody would consume.
                    if Task.isCancelled { throw AgentBackendError.cancelled }
                    do {
                        return try CompletionEngine.shared.launchTurn(prepared.turn, claim: prepared.claim)
                    } catch {
                        // A claim dropped by an operator's preemption is a
                        // cancelled run, not a broken engine: the refusal is
                        // the cancel arriving through the slot. Reporting it as
                        // a failure contradicts the "Stopping…" the operator was
                        // shown and throws away the run's partial report.
                        if prepared.claim.wasCancelled { throw AgentBackendError.cancelled }
                        throw error
                    }
                }
                return try await withTaskCancellationHandler {
                    try await Self.collect(launch: launch, request: request)
                } onCancel: {
                    // The engine's layer loops run on `runOnLargeStack`'s plain
                    // thread, where `Task.isCancelled` is always false and only
                    // `activeTurnCancelled` stops them. Cancelling this handle
                    // alone would leave the turn generating for minutes while it
                    // holds the single slot, so the cancellation is forwarded to
                    // the engine by generation id. `collect` then awaits that
                    // turn, which is what keeps the tool mask below this closure
                    // installed until the engine has stopped reading it.
                    launch.handle.cancel()
                    Task { @MainActor in
                        _ = CompletionEngine.shared.cancelTurn(id: launch.handle.generationId)
                    }
                }
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
    static func prepare(_ request: AgentBackendRequest, thinkingEnabled: Bool) async throws -> Prepared {
        let engine = CompletionEngine.shared
        guard engine.loadedModelInfo != nil, engine.promptModelConfig != nil else {
            throw AgentBackendError.unavailable(
                "A local subagent run needs a loaded model. Load a checkpoint in Settings first."
            )
        }
        // The engine has ONE generation slot and the chat UI and API server use
        // it directly. A background run's next step is not a reason to fail:
        // any chat turn running when that step starts would otherwise kill the
        // run permanently. The run waits out the holder — its own wall-clock
        // budget bounds the wait — instead of treating "busy" as "unavailable".
        return try await withStepClaim(
            acquire: { try await waitForEngineSlot(engine) },
            prepare: { claim in
                // Re-read the loaded model snapshot after acquiring the slot so
                // a model switch or unload while waiting does not render with a
                // stale checkpoint.
                guard let currentInfo = engine.loadedModelInfo,
                      let currentConfig = engine.promptModelConfig else {
                    throw AgentBackendError.unavailable(
                        "A local subagent run needs a loaded model. Load a checkpoint in Settings first."
                    )
                }
                return try await prepare(
                    request,
                    thinkingEnabled: thinkingEnabled,
                    engine: engine,
                    info: currentInfo,
                    config: currentConfig,
                    claim: claim
                )
            }
        )
    }

    /// The claim lifecycle of one step, with the acquisition and preparation
    /// pluggable so the release rule is provable without a loaded checkpoint.
    ///
    /// Between the grant and `launchTurn` sit rendering, tokenizing and prompt
    /// assembly, each of which can throw. An early throw that kept the claim
    /// would leave the ONLY generation slot held by a run that will never
    /// launch: `isActive` answers true for every later prober, so chat sends,
    /// server requests and other runs' steps are refused for the rest of the
    /// session. Whatever `prepare` does, the ledger is told.
    @MainActor
    static func withStepClaim(
        acquire: () async throws -> TurnSlotClaim,
        prepare: (TurnSlotClaim) async throws -> Prepared
    ) async throws -> Prepared {
        let claim = try await acquire()
        do {
            return try await prepare(claim)
        } catch {
            claim.release()
            throw error
        }
    }

    /// Polls the single-flight ledger until a claim is granted. A claim, not a
    /// probe: the slot cannot be taken between the check and the launch.
    ///
    /// The wait is bounded only by the run's own cancellation and wall-clock
    /// budget (the step runs inside `HeadlessAgentLoop`'s deadline). The old
    /// 120 s queue-admission deadline reused the SERVER's timeout, and a
    /// coordinator turn longer than that — routine for a local model — killed
    /// the child permanently at its next step instead of letting it wait out
    /// the turn that held the engine.
    @MainActor
    private static func waitForEngineSlot(_ engine: CompletionEngine) async throws -> TurnSlotClaim {
        while true {
            // `.background` because a subagent step IS one: the claim is the
            // only evidence a turn in preparation leaves, and without the
            // priority an operator's send cannot tell this preparation (which
            // it may stop) from a chat turn's or a server request's (which it
            // may not).
            if let claim = engine.reserveTurnSlot(priority: .background) {
                // The claim is granted by the ledger, not by this run's state:
                // check cancellation AFTER taking it, so a run stopped while
                // it waited hands the slot straight back instead of launching
                // a full turn nobody will process.
                if Task.isCancelled {
                    claim.release()
                    throw AgentBackendError.cancelled
                }
                return claim
            }
            if Task.isCancelled { throw AgentBackendError.cancelled }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    @MainActor
    private static func prepare(
        _ request: AgentBackendRequest,
        thinkingEnabled: Bool,
        engine: CompletionEngine,
        info: CompletionEngine.LoadedModelInfo,
        config: ModelConfig,
        claim: TurnSlotClaim
    ) async throws -> Prepared {
        // The render is pure; the tokenize is a long serial pass over a prompt
        // that can be 160k characters. BOTH stay on this actor: the tokenizer
        // bindings `CompletionEngine.tokenize` calls are themselves MainActor
        // isolated (the generated code is unannotated, so this target's default
        // isolation applies), and a `Task.detached` around the call was an
        // isolation violation that could not have moved the work off the actor
        // anyway. Moving it would have to start at the bindings.
        // The per-request flag is the caller's answer for THIS run; the stored
        // one is only the backend's default. Ignoring it made a run that asked
        // for thinking off think anyway, and pay for it.
        let thinking = request.thinking && thinkingEnabled && info.supportsThinking
        let prompt = renderPrompt(
            systemPrompt: request.systemPrompt,
            messages: request.messages,
            thinkingEnabled: thinking,
            modelSupportsThinking: info.supportsThinking,
            modelConfig: config,
            promptDialect: engine.promptDialect
        )
        let promptTokenIds: [UInt32]
        do {
            // A direct call, NOT `Task.detached`: `tokenize` is MainActor
            // isolated (as are the tokenizer bindings it calls), so a detached
            // call was an isolation violation, and this method is already on
            // that actor — the encoding always ran there.
            promptTokenIds = try engine.tokenize(prompt)
        } catch {
            throw AgentBackendError.malformedResponse(
                "The tokenizer could not encode the subagent prompt: \(error.localizedDescription)"
            )
        }
        guard !promptTokenIds.isEmpty else {
            throw AgentBackendError.malformedResponse("The subagent prompt tokenized to zero tokens.")
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
            thinkingEnabled: thinking,
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
        // A Stop can land after part of the answer streamed. The loop appends
        // whatever a cancelled turn produced, so the text travels with the
        // error instead of being discarded with it. Tool-call markup is removed
        // first: a turn frozen mid-call leaves raw `<tool_call>`/DSML bytes that
        // are not prose and used to be appended to the run's partial report.
        func partialText() -> (text: String, thinking: String) {
            (
                Self.strippingToolCallBlocks(emittedText),
                Self.strippingToolCallBlocks(emittedThinking)
            )
        }
        // Usage an attempt spent without returning a turn travels through the
        // ledger: the run's usage is otherwise the operator's only spend figure,
        // and a cancelled or failed step counted a step with zero tokens.
        func billCompletion(_ generated: Int) {
            guard generated > 0 else { return }
            var delta = AgentUsage()
            delta.completionTokens = generated
            delta.latestCompletionTokens = generated
            delta.costIsUnpriced = true
            delta.costUnpricedReason = .noPriceRecorded
            request.usageLedger?.add(delta)
        }
        func cancellationError() -> AgentBackendError {
            let partial = partialText()
            if partial.text.isEmpty && partial.thinking.isEmpty {
                return .cancelled
            }
            return .cancelledAfterPartial(text: partial.text, thinking: partial.thinking)
        }
        for await event in launch.stream {
            // The run's task is cancelled from outside this loop. The stream
            // itself may already have ended, so the check has to come before
            // the switch: the failure this used to report was a malformed
            // response, which reads as the model breaking rather than a Stop.
            if Task.isCancelled {
                await stop(launch)
                throw cancellationError()
            }
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
                billCompletion(failure.tokensGenerated)
                if failure.cancelled || Task.isCancelled {
                    await launch.handle.task.value
                    throw cancellationError()
                }
                // The task is awaited on EVERY exit from this loop, not just
                // the success and cancellation ones: `withTurnToolMask`
                // restores the registry when this call unwinds, and a turn
                // still unwinding would read a vocabulary that no longer
                // belongs to it.
                await launch.handle.task.value
                // A transport failure after the model already streamed part of
                // its answer is not evidence it said nothing. The loop appends
                // `partialTurn` for a cancelled turn; a failed one deserves the
                // same, or the run's report fell back to an earlier step.
                if !emittedText.isEmpty || !emittedThinking.isEmpty {
                    let partial = partialText()
                    if !partial.text.isEmpty || !partial.thinking.isEmpty {
                        throw AgentBackendError.failedAfterPartial(
                            text: partial.text,
                            thinking: partial.thinking,
                            message: failure.message
                        )
                    }
                }
                throw AgentBackendError.transport(failure.message)
            case .finished(let result):
                // `TurnResult` carries main-actor-isolated value types, so its
                // mapping is read there; the stream itself is consumed here.
                // The engine's task is also AWAITED on this path: a result is
                // emitted before the detached generation task finishes its own
                // teardown, and the guarantee below — that no turn outlives the
                // tool mask this call is nested in — is a claim about the task,
                // not about the stream. It used to hold only on the error and
                // cancellation paths, which is exactly where the comment
                // already was; the success path silently restored the mask
                // while the task was still unwinding.
                let turn = await MainActor.run { Self.turn(from: result) }
                await launch.handle.task.value
                return turn
            }
        }
        if Task.isCancelled {
            await stop(launch)
            throw cancellationError()
        }
        // The stream only ends without a result when the engine's task is gone,
        // so awaiting it is immediate here and guarantees no turn outlives the
        // tool mask this call is nested in.
        await launch.handle.task.value
        if !emittedText.isEmpty || !emittedThinking.isEmpty {
            let partial = partialText()
            if !partial.text.isEmpty || !partial.thinking.isEmpty {
                throw AgentBackendError.failedAfterPartial(
                    text: partial.text,
                    thinking: partial.thinking,
                    message: "The local engine's stream closed without a result."
                )
            }
        }
        throw AgentBackendError.malformedResponse("The local engine's stream closed without a result.")
    }

    /// Cancels the turn this run started and waits for the engine to release
    /// it. The caller restores the grammar registry when this returns, so a
    /// turn still generating would consult a vocabulary that no longer belongs
    /// to it.
    private static func stop(_ launch: CompletionLaunch) async {
        launch.handle.cancel()
        await MainActor.run {
            _ = CompletionEngine.shared.cancelTurn(id: launch.handle.generationId)
        }
        await launch.handle.task.value
    }

    /// What arrived since the last frame. Frames carry the whole text so far, so a
    /// delta is the suffix the previous snapshot did not have; a rewritten prefix
    /// (a repaired UTF-8 boundary) is emitted whole rather than dropped.
    static func suffixDelta(from previous: String, to current: String) -> String {
        guard !current.isEmpty else { return "" }
        if current.hasPrefix(previous) {
            return String(current.dropFirst(previous.count))
        }
        // A SHRINK is a reclassification, not new text: a late think-close
        // moves part of the answer into the reasoning channel, so the response
        // snapshot gets shorter. Emitting the whole shorter string appended a
        // duplicate of the answer to the live transcript.
        if previous.hasPrefix(current) {
            return ""
        }
        // A rewritten prefix (a repaired UTF-8 boundary) is emitted whole
        // rather than dropped.
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
        // A local turn spends the operator's own hardware, so nothing here can
        // name a dollar figure — and leaving costUSD at 0 with costIsUnpriced
        // false made a cost ceiling silently inert (it compared 0 >= maxCost
        // forever). Marking it unpriced is what makes the loop REFUSE a ceiling
        // it cannot enforce instead of pretending to.
        usage.costIsUnpriced = true
        usage.costUnpricedReason = .noPriceRecorded
        // `responseText` is the raw response channel, which still contains a
        // call's markup when the model emitted one. The markup is an
        // instruction, not prose: the report builder reads this text for
        // findings and the summary, and was quoting `<parameter=content>` as
        // an assertion the run made. The calls themselves travel separately.
        //
        // Removing each call's own `rawText` is not enough, and the two ways it
        // fails both hurt the NEXT prompt: the Qwen dialect's rawText is the
        // inner `<function=…>` without the outer `<tool_call>` shell, so the
        // shell survived and `carriesCallMarkup` then skipped the echo of the
        // calls in this run's own history; and a canonicalized call (DSML,
        // Ling) has a rawText that no longer appears in the text at all, so the
        // dialect markup leaked into the report. Strip whole blocks.
        var visibleText = Self.strippingToolCallBlocks(result.responseText)
        for call in result.parsedToolCalls where !call.rawText.isEmpty {
            visibleText = visibleText.replacingOccurrences(of: call.rawText, with: "")
        }
        visibleText = visibleText.trimmingCharacters(in: .whitespacesAndNewlines)
        // A GESTURE-ONLY turn is a finished run, not a broken one. The local
        // parser freezes on the closing tag of a tool call, so a model that
        // signals "done" with a no-op call (`echo`, `true`, `:`) reports
        // `toolFreeze`; `actionableCallIndices` then drops every call, and the
        // loop's `emptyToolFreeze` guard read that as "a tool-call boundary
        // that produced no complete call" and failed a run that had actually
        // completed. The gesture's own skip notice is already in the text, so
        // reporting the turn as a normal stop keeps the answer.
        let gestureOnly = calls.isEmpty && !result.parsedToolCalls.isEmpty
        // An unclosed tool-call block is removed to EOF by the strip above, so
        // prose followed by a truncated call would otherwise read as a finished
        // answer. It is a fragment: report it as a truncation, which the loop
        // already treats as unusable, instead of letting `eos` vouch for it.
        let hasBrokenFragment = !result.brokenFragments.isEmpty
        return AgentBackendTurn(
            text: visibleText,
            thinking: result.thinkingText,
            calls: calls,
            rawText: result.rawDecodedText,
            finishReason: hasBrokenFragment
                ? TurnFinishReason.lengthCapped.rawValue
                : (gestureOnly ? TurnFinishReason.eos.rawValue : result.finishReason.rawValue),
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

    /// Removes whole tool-call blocks, across the dialects a local family can
    /// emit, so what is left is prose.
    ///
    /// A block runs from its opener to its closer; an opener with no closer is
    /// the pre-execution freeze's shape, and the tail is dropped because a
    /// half-call is not prose either. Paired tags are handled generically
    /// (`<tool_call>`/`</tool_call>`, the DSML `calls` container, Gemma's
    /// `<|tool_call>`/`<tool_call|>`), then the bare openers that carry no
    /// container (Llama's `<|python_tag|>`, a stray `<tool_call>`, a DSML
    /// `invoke` the freeze stopped before its container closed).
    static func strippingToolCallBlocks(_ text: String) -> String {
        nonisolatedMarkupStrip(text)
    }

    private static func nonisolatedMarkupStrip(_ text: String) -> String {
        var out = text
        let pairs: [(open: String, close: String)] = [
            (StreamingToolParser.qwenToolCallOpen, StreamingToolParser.qwenToolCallClose),
            (DeepSeekV41ToolDialect.dsmlCallsOpen, DeepSeekV41ToolDialect.dsmlCallsClose),
            (StreamingToolParser.gemmaToolCallOpen, StreamingToolParser.gemmaToolCallClose),
            // The dialect's plain-tag call has no container: a bare
            // name/arguments pair was left in a cancelled run's partial text.
            (DeepSeekV41ToolDialect.callNameOpen, DeepSeekV41ToolDialect.callNameClose),
            (DeepSeekV41ToolDialect.callArgumentsOpen, DeepSeekV41ToolDialect.callArgumentsClose)
        ]
        for pair in pairs {
            while let open = out.range(of: pair.open) {
                if let close = out.range(of: pair.close, range: open.upperBound..<out.endIndex) {
                    out.removeSubrange(open.lowerBound..<close.upperBound)
                } else {
                    out.removeSubrange(open.lowerBound..<out.endIndex)
                }
            }
        }
        // A DSML `invoke` with no `calls` around it is the shape the
        // pre-execution freeze can leave behind, and the container loop above
        // never sees it.
        while let open = out.range(of: DeepSeekV41ToolDialect.dsmlInvokeOpen) {
            if let close = out.range(
                of: DeepSeekV41ToolDialect.dsmlInvokeClose,
                range: open.upperBound..<out.endIndex
            ) {
                out.removeSubrange(open.lowerBound..<close.upperBound)
            } else {
                out.removeSubrange(open.lowerBound..<out.endIndex)
            }
        }
        for bare in [StreamingToolParser.llamaTagOpen] {
            out = out.replacingOccurrences(of: bare, with: "")
        }
        return out
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
    ///
    /// A schema that cannot be encoded must NOT drop the whole mask: an empty
    /// mask installs nothing, so the coordinator's global vocabulary stayed
    /// active for a run that has its own tools. The fallback is a strict empty
    /// object — the tool is callable with no arguments — which constrains more
    /// than the real schema, never less.
    @MainActor
    static func maskSchemas(for tools: [ToolDefinition]) -> [GrammarToolSchema] {
        var schemas: [GrammarToolSchema] = []
        schemas.reserveCapacity(tools.count)
        for tool in tools {
            // Encode the PARAMETERS DICTIONARY, not a wrapper around it: each
            // value is already an `AnyCodable`, and wrapping the dictionary in a
            // second one gives it no encoding of its own (every branch in
            // `AnyCodable.encode` misses), so the schema would decode as an empty
            // object and the mask would constrain nothing.
            let schemaData = (try? JSONEncoder().encode(tool.function.parameters))
                ?? Self.emptyObjectSchema
            schemas.append(GrammarToolSchema(name: tool.function.name, schemaJSON: schemaData))
        }
        return schemas
    }

    /// `{"type":"object","properties":{}}`, the strict fallback for a tool
    /// whose schema cannot be encoded: callable with no arguments, nothing
    /// else.
    private static let emptyObjectSchema = Data(
        #"{"type":"object","properties":{}}"#.utf8
    )

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
            isDeepSeekV41Model: config?.isDeepSeekV41Model == true,
            isInkling2Model: config?.isInkling2Model == true
        )
    }
}
