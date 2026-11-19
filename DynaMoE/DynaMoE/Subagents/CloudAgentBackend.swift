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
        let messages = Self.wireMessages(systemPrompt: request.systemPrompt, messages: request.messages)
        let params = CloudChatGenerationParams(
            temperature: request.sampling.temperature,
            topP: request.sampling.topP,
            presencePenalty: nil,
            maxTokens: request.maxTokens,
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
        if let failure { throw failure }
        guard let result else {
            throw AgentBackendError.malformedResponse("The provider closed the stream without a terminal event.")
        }
        guard result.terminated else {
            throw AgentBackendError.malformedResponse("The provider's stream ended without a completion marker.")
        }

        let usage = Self.usage(from: result.usage, pricing: pricing)
        let calls = result.toolCalls.map { call -> AgentToolCall in
            let arguments = Self.arguments(from: call.argumentsJSON)
            return AgentToolCall(
                id: call.id,
                name: call.name,
                arguments: arguments,
                rawArguments: call.argumentsJSON,
                rawText: call.name
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
        result.promptTokens = usage.promptTokens ?? 0
        result.completionTokens = usage.completionTokens ?? 0
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

    static func arguments(from json: String) -> [String: Any] {
        // An unparseable block answers with NO arguments, not a fabricated
        // `_raw_arguments` key: the raw text already travels on the call's own
        // `rawArguments` (and from there into the transcript), and a synthetic
        // key turned a malformed call into one with a plausible-looking
        // argument name that no tool declares.
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        return object
    }
}

nonisolated enum SubagentBackendSettings {
    static let cloudModelKey = "dynamoe_subagent_cloud_model"
    static let backendKey = "dynamoe_subagent_backend"

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
