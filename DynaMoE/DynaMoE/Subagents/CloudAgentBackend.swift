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

    /// The model a spawn uses when the caller did not name one: the
    /// coordinator's own cloud model when it has one, otherwise the first
    /// registered cloud model that supports tools and has a stored key.
    static func resolveDefault(coordinateModelId: String?, providers: [CloudProvider] = CloudProviderRegistry.all) -> String? {
        if let coordinateModelId,
           let parsed = CloudModelAddress.parseModelId(coordinateModelId, providers: providers) {
            let specs = CloudModelRegistry.registeredSpecs(for: parsed.provider)
            if specs.contains(where: { $0.providerModelId == parsed.providerModelId && $0.supportsTools }) {
                return CloudModelAddress.modelId(providerId: parsed.provider.id, providerModelId: parsed.providerModelId)
            }
        }
        let available = CloudModelRegistry.makeDiscoveredModels(providers: providers)
        for model in available {
            guard let resolution = model.cloudResolution else { continue }
            let specs = CloudModelRegistry.registeredSpecs(for: resolution.provider)
            if specs.contains(where: { $0.providerModelId == resolution.providerModelId && $0.supportsTools }) {
                return model.id
            }
        }
        return nil
    }

    func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn {
        var attempt = 0
        while true {
            attempt += 1
            do {
                return try await performTurn(request)
            } catch let failure as CloudTurnFailure where attempt < 2 && Self.isRetryable(failure) {
                continue
            } catch let failure as CloudTurnFailure {
                throw AgentBackendError.transport(failure.errorDescription ?? "The cloud request failed.")
            }
        }
    }

    private func performTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn {
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
            tools: request.tools.isEmpty ? nil : request.tools
        )

        var result: CloudTurnResult?
        var failure: CloudTurnFailure?
        for await event in stream {
            switch event {
            case .started:
                break
            case .thinkingDelta(let text):
                request.onDelta?(text, true)
            case .contentDelta(let text):
                request.onDelta?(text, false)
            case .finished(let finished):
                result = finished
            case .failed(let failed):
                failure = failed
            }
        }

        if Task.isCancelled { throw AgentBackendError.cancelled }
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

    private static func usage(from usage: CloudUsage?, pricing: CloudPricing?) -> AgentUsage {
        var result = AgentUsage()
        guard let usage else { return result }
        result.promptTokens = usage.promptTokens ?? 0
        result.completionTokens = usage.completionTokens ?? 0
        result.cachedTokens = usage.cachedTokens ?? 0
        if let pricing, let cost = pricing.cost(for: usage) {
            result.costUSD = cost
        }
        return result
    }

    private static func isRetryable(_ failure: CloudTurnFailure) -> Bool {
        switch failure {
        case .transport(_, let billed):
            return billed
        case .httpStatus(let code, _):
            return code == 429 || code >= 500
        case .invalidResponse:
            return false
        case .missingAPIKey, .credentialAccessDenied, .credentialNeedsImport, .unsupportedFlavor:
            return false
        }
    }

    private static func wireMessages(systemPrompt: String, messages: [AgentBackendMessage]) -> [CloudChatMessage] {
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

    private static func argumentsJSON(_ arguments: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(arguments),
              let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    private static func arguments(from json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ["_raw_arguments": json]
        }
        return object
    }
}

nonisolated enum SubagentBackendSettings {
    static let cloudModelKey = "dynamoe_subagent_cloud_model"
    static let backendKey = "dynamoe_subagent_backend"

    static func preferredCloudModel(coordinateModelId: String?, defaults: UserDefaults = .standard) -> String? {
        if let stored = defaults.string(forKey: cloudModelKey)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !stored.isEmpty,
           case .success = CloudAgentBackend.make(address: stored) {
            return stored
        }
        return CloudAgentBackend.resolveDefault(coordinateModelId: coordinateModelId)
    }
}
