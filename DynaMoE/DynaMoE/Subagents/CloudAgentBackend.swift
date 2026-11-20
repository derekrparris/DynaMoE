//
//  CloudAgentBackend.swift
//  DynaMoE
//
//  Cloud-provider backend for a subagent run. See SUBAGENTS.md §6.
//

import Foundation

nonisolated final class CloudAgentBackend: AgentModelBackend, @unchecked Sendable {
    let descriptor: AgentBackendDescriptor
    private let client: CloudChatClient
    private let providerModelId: String
    private let pricing: CloudPricing?

    init(
        provider: CloudProvider,
        providerModelId: String,
        client: CloudChatClient,
        displayName: String? = nil
    ) {
        self.client = client
        self.providerModelId = providerModelId
        self.pricing = CloudPricingStore.pricing(forModel: providerModelId, providerId: provider.id)
        self.descriptor = AgentBackendDescriptor(
            kind: .cloud,
            providerId: provider.id,
            modelId: providerModelId,
            displayName: displayName ?? CloudModelRegistry.defaultDisplayName(provider: provider, providerModelId: providerModelId)
        )
    }

    static func make(address: String, providers: [CloudProvider] = CloudProviderRegistry.all) -> Result<CloudAgentBackend, AgentBackendError> {
        guard let parsed = CloudModelAddress.parseModelId(address, providers: providers)
            ?? CloudModelAddress.parsePath(address, providers: providers) else {
            return .failure(.unavailable("'\(address)' is not a configured cloud model."))
        }
        // `resolveDefault` only ever picks a model recorded as tool-capable, but
        // an explicit address bypassed that check. A run whose model cannot call
        // tools has no way to do anything: it answers the task in prose and
        // never reaches `complete`.
        if let spec = CloudModelRegistry.registeredSpecs(for: parsed.provider)
            .first(where: { $0.providerModelId == parsed.providerModelId }),
           !spec.supportsTools {
            return .failure(.unavailable(
                "'\(parsed.providerModelId)' is recorded as not supporting tool calling in Settings ▸ Cloud Providers, and a subagent run needs tools. Choose a tool-capable model."
            ))
        }
        switch CloudChatClient.makeStoredResult(provider: parsed.provider) {
        case .success(let client):
            return .success(CloudAgentBackend(
                provider: parsed.provider,
                providerModelId: parsed.providerModelId,
                client: client
            ))
        case .failure(let failure):
            return .failure(.unavailable(failure.errorDescription ?? "The cloud provider is not configured."))
        }
    }

    func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn {
        var attempt = 0
        while true {
            attempt += 1
            // Every attempt gets its own receipt. A failure or a Stop can end
            // the event stream without a terminal event, and the receipt is the
            // only evidence left of what the provider may have billed.
            let receipt = CloudTurnReceipt()
            do {
                // Only the FIRST attempt streams deltas. The drawer's live
                // text only ever appends, and a retry re-emits the turn from
                // its beginning, so attempt 2's opening text landed after
                // attempt 1's partial — the operator watched a duplicated
                // stream while the persisted transcript kept just the final
                // turn. Later attempts still produce the turn; only the live
                // commentary is silenced.
                var streamed = request
                if attempt > 1 {
                    streamed.onDelta = nil
                }
                return try await performTurn(streamed, receipt: receipt)
            } catch let failure as CloudTurnFailure
                where attempt < 2 && Self.isRetryable(failure, spentRequest: receipt.spentRequest) {
                Self.recordSpentUsage(receipt, pricing: pricing, into: request.usageLedger)
                await Self.backoffBeforeRetry(failure, receipt: receipt)
                if Task.isCancelled { throw AgentBackendError.cancelled }
                continue
            } catch let failure as CloudTurnFailure {
                Self.recordSpentUsage(receipt, pricing: pricing, into: request.usageLedger)
                throw Self.agentError(from: failure)
            } catch {
                Self.recordSpentUsage(receipt, pricing: pricing, into: request.usageLedger)
                throw error
            }
        }
    }

    /// The run-facing error for a provider failure.
    ///
    /// Every `CloudTurnFailure` used to be wrapped as `AgentBackendError
    /// .transport`, so a missing key, an unreadable credential and an
    /// unimplemented flavor all read as a network fault and sent the operator
    /// looking at the wrong end of the connection. The message was always the
    /// failure's own; the TYPE now says what kind of failure it was too.
    static func agentError(from failure: CloudTurnFailure) -> AgentBackendError {
        let message = failure.errorDescription ?? "The cloud request failed."
        switch failure {
        case .missingAPIKey, .credentialAccessDenied, .credentialNeedsImport, .unsupportedFlavor:
            return .unavailable(message)
        case .httpStatus, .transport, .invalidResponse:
            return .transport(message)
        }
    }

    private func performTurn(
        _ request: AgentBackendRequest,
        receipt: CloudTurnReceipt
    ) async throws -> AgentBackendTurn {
        // A Stop between two steps cancels the run's task; without this the
        // turn still built and ISSUED the request, billing the operator for an
        // answer nothing would read. The retry loop checks before a retry for
        // the same reason.
        if Task.isCancelled { throw AgentBackendError.cancelled }
        let messages = Self.wireMessages(systemPrompt: request.systemPrompt, messages: request.messages)
        let params = CloudChatGenerationParams(
            temperature: request.sampling.temperature,
            topP: request.sampling.topP,
            presencePenalty: nil,
            maxTokens: request.maxTokens,
            stop: request.stopSequences.isEmpty ? nil : request.stopSequences,
            sessionKey: CloudSessionKey.hashed(request.sessionId)
        )
        let stream = client.streamTurn(
            model: providerModelId,
            messages: messages,
            params: params,
            tools: request.tools.isEmpty ? nil : request.tools,
            receipt: receipt
        )

        var result: CloudTurnResult?
        var failure: CloudTurnFailure?
        var streamedText = ""
        var streamedThinking = ""
        for await event in stream {
            switch event {
            case .started:
                break
            case .thinkingDelta(let text):
                streamedThinking += text
                request.onDelta?(text, true)
            case .contentDelta(let text):
                streamedText += text
                request.onDelta?(text, false)
            case .finished(let finished):
                result = finished
            case .failed(let failed):
                failure = failed
            }
        }

        if Task.isCancelled {
            if streamedText.isEmpty && streamedThinking.isEmpty {
                throw AgentBackendError.cancelled
            }
            throw AgentBackendError.cancelledAfterPartial(
                text: streamedText,
                thinking: streamedThinking
            )
        }
        if let failure {
            // Whatever the provider streamed before it failed is the run's
            // work, exactly as with a Stop. The loop appends `partialTurn` for
            // a cancelled turn only, so the partial has to travel WITH the
            // failure or the report falls back to an earlier step.
            if !streamedText.isEmpty || !streamedThinking.isEmpty {
                throw AgentBackendError.failedAfterPartial(
                    text: streamedText,
                    thinking: streamedThinking,
                    message: failure.errorDescription ?? "The provider failed mid-stream."
                )
            }
            throw failure
        }
        guard let result else {
            if !streamedText.isEmpty || !streamedThinking.isEmpty {
                throw AgentBackendError.failedAfterPartial(
                    text: streamedText,
                    thinking: streamedThinking,
                    message: "The provider closed the stream without a terminal event."
                )
            }
            throw AgentBackendError.malformedResponse("The provider closed the stream without a terminal event.")
        }
        guard result.terminated else {
            if !streamedText.isEmpty || !streamedThinking.isEmpty {
                throw AgentBackendError.failedAfterPartial(
                    text: streamedText,
                    thinking: streamedThinking,
                    message: "The provider's stream ended without a completion marker."
                )
            }
            throw AgentBackendError.malformedResponse("The provider's stream ended without a completion marker.")
        }

        let usage = Self.usage(from: result.usage, pricing: pricing)
        let calls = result.toolCalls.map { call -> AgentToolCall in
            let decoded = Self.arguments(from: call.argumentsJSON)
            return AgentToolCall(
                id: call.id,
                name: call.name,
                arguments: decoded ?? [:],
                rawArguments: call.argumentsJSON,
                rawText: call.name,
                argumentsMalformed: decoded == nil
            )
        }
        return AgentBackendTurn(
            text: result.text,
            thinking: result.thinking,
            calls: calls,
            finishReason: result.finishReason ?? "stop",
            usage: usage
        )
    }

    static func usage(from usage: CloudUsage?, pricing: CloudPricing?) -> AgentUsage {
        var result = AgentUsage()
        guard let usage else {
            // No usage block is not proof the turn was free. The provider may
            // have billed it and omitted the block (a Stop, a dropped terminal
            // frame, an endpoint that never reports usage), so a cost ceiling
            // must see "unknown", not "$0". The reason is recorded separately
            // so the refusal does not send the operator to record a price that
            // is already there.
            result.costIsUnpriced = true
            result.costUnpricedReason = .providerOmittedUsage
            return result
        }
        // A provider may report only some of the counts, and a missing one used
        // to decode as 0 — which reads as "this turn generated nothing" and
        // silently disables the completion cap, the one budget that bounds a
        // run by what it spends rather than by steps or the clock.
        //
        // Cache WRITES are prompt tokens the provider billed as input, so they
        // belong in the prompt total (a cache read is already inside it on the
        // providers that report both).
        var prompt = usage.promptTokens
        var completion = usage.completionTokens
        if let total = usage.totalTokens {
            // `total` is the only count some providers send. Derive whichever
            // side is missing from it rather than dropping it.
            if prompt == nil, let completion { prompt = max(0, total - completion) }
            if completion == nil, let prompt { completion = max(0, total - prompt) }
        }
        if let writes = usage.cacheWriteTokens, writes > 0 {
            prompt = (prompt ?? 0) + writes
        }
        result.promptTokens = prompt ?? 0
        result.completionTokens = completion ?? 0
        result.cachedTokens = usage.cachedTokens ?? 0
        if let pricing, let cost = pricing.cost(for: usage) {
            result.costUSD = cost
        } else {
            // The block could not be priced: no price is recorded for this
            // model, or the provider reported only PART of the counts (a bare
            // `total_tokens`, which `cost(for:)` refuses to price). Either way
            // the dollar figure is unknown rather than zero — a cost ceiling
            // reading "$0 spent" here would never bind, and the run would
            // spend an amount the app cannot name. A recorded price with an
            // incomplete count is still the provider's omission, not a
            // missing price.
            result.costIsUnpriced = true
            result.costUnpricedReason = pricing == nil ? .noPriceRecorded : .providerOmittedUsage
        }
        return result
    }

    static func isRetryable(_ failure: CloudTurnFailure, spentRequest: Bool? = nil) -> Bool {
        switch failure {
        case .transport(_, let billed):
            // A request that provably never left the Mac is safe to send again.
            // One that may have reached inference is not: the retry can be
            // invoiced a second time for a turn the caller will never see.
            return !billed && spentRequest != true
        case .httpStatus(let code, _):
            if code == 429 { return true }
            // A 5xx is never retried. It is an answer FROM the provider after
            // the request was accepted, so the prompt may already have run and
            // a retry can be invoiced twice. The old arm here required
            // `spentRequest == false`, which the transport can never report for
            // a 5xx (it records `spent: statusCode >= 500`), so the branch was
            // dead; removing it is the honest answer, not retrying.
            return false
        case .invalidResponse:
            return false
        case .missingAPIKey, .credentialAccessDenied, .credentialNeedsImport, .unsupportedFlavor:
            return false
        }
    }

    /// A rate-limited request is retried once, but not immediately: an
    /// immediate retry is the request the provider's window just refused.
    static func retryDelaySeconds(for failure: CloudTurnFailure) -> Double {
        if case .httpStatus(let code, _) = failure, code == 429 { return 0.75 }
        return 0
    }

    private static func backoffBeforeRetry(
        _ failure: CloudTurnFailure,
        receipt: CloudTurnReceipt? = nil
    ) async {
        // The provider's Retry-After outranks the fixed fallback: it is the
        // window that actually refused the request.
        let delay = receipt?.retryAfterSeconds ?? retryDelaySeconds(for: failure)
        guard delay > 0 else { return }
        try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
    }

    /// Prices an attempt that produced no turn. A billed attempt whose provider
    /// sent no usage block is recorded as unpriced rather than as free, so a
    /// cost ceiling cannot read an invisible request as $0.
    private static func recordSpentUsage(
        _ receipt: CloudTurnReceipt,
        pricing: CloudPricing?,
        into ledger: AgentUsageLedger?
    ) {
        guard let ledger else { return }
        if let result = receipt.result {
            var usage = Self.usage(from: result.usage, pricing: pricing)
            usage.backendRequests = 1
            ledger.add(usage)
            return
        }
        guard receipt.spentRequest == true else { return }
        var usage = AgentUsage()
        usage.backendRequests = 1
        usage.costIsUnpriced = true
        usage.costUnpricedReason = .providerOmittedUsage
        ledger.add(usage)
    }

    static func wireMessages(systemPrompt: String, messages: [AgentBackendMessage]) -> [CloudChatMessage] {
        var wire: [CloudChatMessage] = [
            CloudChatMessage(role: "system", content: systemPrompt)
        ]
        for message in messages {
            switch message {
            case .user(let text):
                wire.append(CloudChatMessage(role: "user", content: text))
            case .assistant(let text, _, let calls):
                let wireCalls = calls.map { call in
                    CloudChatToolCall(
                        id: call.id,
                        name: call.name,
                        arguments: call.rawArguments.isEmpty ? argumentsJSON(call.arguments) : call.rawArguments
                    )
                }
                wire.append(CloudChatMessage(
                    role: "assistant",
                    content: text.isEmpty ? nil : text,
                    toolCalls: wireCalls.isEmpty ? nil : wireCalls
                ))
            case .toolResult(let callId, _, let content, _):
                wire.append(CloudChatMessage(role: "tool", content: content, toolCallId: callId))
            }
        }
        return wire
    }

    static func argumentsJSON(_ arguments: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(arguments),
              let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    static func arguments(from json: String) -> [String: Any]? {
        // Empty/whitespace arguments are a valid zero-argument call (`{}`).
        // Any non-empty string that does not decode to a JSON object (such as a
        // truncated payload) returns nil so the loop refuses to execute it
        // instead of silently running the tool with empty arguments.
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [:] }
        guard let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return object
    }
}

nonisolated enum SubagentBackendSettings {
    static let cloudModelKey = "dynamoe_subagent_cloud_model"
    static let backendKey = "dynamoe_subagent_backend"
    /// The operator's standing dollar ceiling for a cloud run. Applied by
    /// `AgentSpawnResolver` to every cloud spawn that did not set its own
    /// `max_cost_usd`, which is what makes a model-authored budget bounded by
    /// something the operator can see before the run exists: presets carry no
    /// ceiling and a read-only cloud spawn raises no card of its own.
    static let cloudCostCeilingKey = "dynamoe_subagent_cloud_max_cost_usd"
    /// A ceiling above this is refused by the setter rather than accepted: a
    /// mistyped "10000" would otherwise read as a budget no run can exceed.
    static let maximumCloudCostUSD: Double = 100
    /// The cap a cloud run gets before the operator changes anything. A cap
    /// that started at "none" would leave the default cloud configuration
    /// exactly as unbounded as before the Setting existed; the local analogue
    /// (`LocalSubagentCostEstimate.defaultLimitSeconds`) is a number out of the
    /// box too. 0 in the field is how the operator clears it.
    static let defaultCloudCostUSD: Double = 5
    /// Per-conversation aggregate cloud spend cap. Bounding only a single run
    /// still lets a coordinator spawn arbitrarily many read-only cloud runs
    /// with no approval card. Defaults to 4× the per-run cap ($20 out of the
    /// box) unless overridden or cleared.
    static let sessionCloudCostCeilingKey = "dynamoe_subagent_cloud_session_max_cost_usd"
    static let defaultSessionCloudCostMultiplier: Double = 4
    static let maximumSessionCloudCostUSD: Double = 500

    /// `nil` means "no per-conversation aggregate ceiling".
    static func sessionCloudCostCeiling(defaults: UserDefaults = .standard) -> Double? {
        if let stored = defaults.object(forKey: sessionCloudCostCeilingKey) as? NSNumber {
            let ceiling = stored.doubleValue
            guard ceiling.isFinite, ceiling > 0 else { return nil }
            return min(ceiling, maximumSessionCloudCostUSD)
        }
        guard let perRun = cloudCostCeiling(defaults: defaults) else { return nil }
        return min(perRun * defaultSessionCloudCostMultiplier, maximumSessionCloudCostUSD)
    }

    static func setSessionCloudCostCeiling(_ value: Double?, defaults: UserDefaults = .standard) {
        guard let value else {
            defaults.removeObject(forKey: sessionCloudCostCeilingKey)
            return
        }
        guard value.isFinite, value > 0 else {
            defaults.set(0, forKey: sessionCloudCostCeilingKey)
            return
        }
        defaults.set(min(value, maximumSessionCloudCostUSD), forKey: sessionCloudCostCeilingKey)
    }

    /// `nil` means "no operator ceiling". A stored 0, a negative or a
    /// non-finite value are the same answer: a $0 ceiling fails every run at
    /// step 0, which is never what an operator meant by clearing the field.
    static func cloudCostCeiling(defaults: UserDefaults = .standard) -> Double? {
        if let stored = defaults.object(forKey: cloudCostCeilingKey) as? NSNumber {
            let ceiling = stored.doubleValue
            guard ceiling.isFinite, ceiling > 0 else { return nil }
            return min(ceiling, maximumCloudCostUSD)
        }
        return defaultCloudCostUSD
    }

    /// Stores the operator's ceiling, clamped to the bound the Setting shows.
    /// Shared by the Settings field and any programmatic writer so a value
    /// written past the bound cannot outlive the UI that named it. `nil`
    /// restores the built-in default; 0 is the "no cap" value.
    static func setCloudCostCeiling(_ value: Double?, defaults: UserDefaults = .standard) {
        guard let value else {
            defaults.removeObject(forKey: cloudCostCeilingKey)
            return
        }
        guard value.isFinite, value > 0 else {
            defaults.set(0, forKey: cloudCostCeilingKey)
            return
        }
        defaults.set(min(value, maximumCloudCostUSD), forKey: cloudCostCeilingKey)
    }

    /// Folds a typed value onto the stored range: a negative field is "no
    /// ceiling", an inflated one is the documented maximum.
    static func clampedCloudCostCeiling(_ value: Double) -> Double {
        guard value.isFinite, value > 0 else { return 0 }
        return min(value, maximumCloudCostUSD)
    }

    /// The outcome of resolving the subagent Cloud model setting.
    ///
    /// A stored value that does not name a configured cloud model is
    /// REFUSED by name rather than silently replaced with the default: a
    /// run would otherwise be billed on a provider the operator never chose
    /// (the same rule as `CloudModelAddress.selectionIsUnresolved` for the
    /// chat model picker). Resolution is PARSE-level on purpose so this stays
    /// pure; a stored address whose provider has no stored key still reaches
    /// `CloudAgentBackend.make` one step later and refuses there with the
    /// provider named, so no path falls through to the default either.
    ///
    /// An EMPTY setting is `.noneConfigured`, not a fallback. Settings says
    /// "empty means a cloud spawn refuses by name", and the old fallback to
    /// the first registered tool-capable model billed the operator on a
    /// provider they never named, with an approval card that named no model.
    enum PreferredCloudModel: Equatable {
        /// Nothing is stored, so a cloud spawn is refused with the "no cloud
        /// model is configured" message.
        case noneConfigured
        /// The stored setting parses to a configured cloud model.
        case configured(String)
        /// The stored setting does not name a configured cloud model. The
        /// payload is the stored text so the refusal can name it.
        case unresolvable(String)
    }

    static func preferredCloudModel(
        defaults: UserDefaults = .standard,
        providers: [CloudProvider] = CloudProviderRegistry.all
    ) -> PreferredCloudModel {
        let stored = (defaults.string(forKey: cloudModelKey) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stored.isEmpty else { return .noneConfigured }
        guard CloudModelAddress.parseModelId(stored, providers: providers) != nil
            || CloudModelAddress.parsePath(stored, providers: providers) != nil else {
            return .unresolvable(stored)
        }
        return .configured(stored)
    }
}
