//
//  AgentModelBackend.swift
//  DynaMoE
//
//  The model boundary for a subagent run. See SUBAGENTS.md §6.
//

import Foundation

nonisolated struct AgentBackendRequest: @unchecked Sendable {
    var systemPrompt: String
    var messages: [AgentBackendMessage]
    var tools: [ToolDefinition]
    var sampling: AgentSampling
    var maxTokens: Int
    var stopSequences: [String]
    var thinking: Bool
    var sessionId: UUID
    var onDelta: (@Sendable (String, Bool) -> Void)?
}

nonisolated struct AgentBackendTurn: @unchecked Sendable {
    var text: String
    var thinking: String
    var calls: [AgentToolCall]
    var rawText: String
    var finishReason: String
    var usage: AgentUsage

    init(
        text: String,
        thinking: String = "",
        calls: [AgentToolCall] = [],
        rawText: String? = nil,
        finishReason: String = "stop",
        usage: AgentUsage = AgentUsage()
    ) {
        self.text = text
        self.thinking = thinking
        self.calls = calls
        self.rawText = rawText ?? text
        self.finishReason = finishReason
        self.usage = usage
    }
}

nonisolated protocol AgentModelBackend: AnyObject, Sendable {
    var descriptor: AgentBackendDescriptor { get }
    func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn
}

nonisolated enum AgentBackendError: LocalizedError, Equatable {
    case unavailable(String)
    case refused(String)
    case transport(String)
    case malformedResponse(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .unavailable(let message): return message
        case .refused(let message): return message
        case .transport(let message): return message
        case .malformedResponse(let message): return message
        case .cancelled: return "The subagent run was cancelled."
        }
    }
}

nonisolated final class ScriptedBackend: AgentModelBackend, @unchecked Sendable {
    let descriptor: AgentBackendDescriptor

    struct ScriptedTurn {
        var text: String
        var thinking: String
        var calls: [AgentToolCall]
        var finishReason: String
        var usage: AgentUsage
        var error: AgentBackendError?
        var delayNanoseconds: UInt64

        init(
            text: String,
            thinking: String = "",
            calls: [AgentToolCall] = [],
            finishReason: String = "stop",
            usage: AgentUsage = AgentUsage(),
            error: AgentBackendError? = nil,
            delayNanoseconds: UInt64 = 0
        ) {
            self.text = text
            self.thinking = thinking
            self.calls = calls
            self.finishReason = finishReason
            self.usage = usage
            self.error = error
            self.delayNanoseconds = delayNanoseconds
        }
    }

    private let lock = NSLock()
    private var turns: [ScriptedTurn]
    private var recordedRequests: [AgentBackendRequest] = []
    private var overflowTurn: ScriptedTurn?

    init(turns: [ScriptedTurn], descriptor: AgentBackendDescriptor = .scripted) {
        self.turns = turns
        self.descriptor = descriptor
    }

    var requests: [AgentBackendRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recordedRequests
    }

    var remainingTurnCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return turns.count
    }

    private func takeTurn(for request: AgentBackendRequest) -> ScriptedTurn {
        lock.lock()
        defer { lock.unlock() }
        recordedRequests.append(request)
        if turns.isEmpty {
            return overflowTurn ?? ScriptedTurn(text: "No further scripted turns.", finishReason: "error")
        }
        return turns.removeFirst()
    }

    func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn {
        let turn = takeTurn(for: request)

        if turn.delayNanoseconds > 0 {
            try? await Task.sleep(nanoseconds: turn.delayNanoseconds)
        }
        if Task.isCancelled {
            throw AgentBackendError.cancelled
        }
        if let error = turn.error {
            throw error
        }
        if let onDelta = request.onDelta {
            if !turn.thinking.isEmpty { onDelta(turn.thinking, true) }
            if !turn.text.isEmpty { onDelta(turn.text, false) }
        }
        return AgentBackendTurn(
            text: turn.text,
            thinking: turn.thinking,
            calls: turn.calls,
            finishReason: turn.finishReason,
            usage: turn.usage
        )
    }
}
