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
    /// Spend a backend incurred on an attempt that never returned a turn.
    /// A failed or cancelled cloud request can still be invoiced, and the
    /// normal `AgentBackendTurn.usage` path cannot carry it because there is
    /// no turn. The loop merges whatever this holds into the run's usage.
    var usageLedger: AgentUsageLedger? = nil
}

/// A lock-guarded place for out-of-band usage. Only a backend that can be
/// billed without producing a turn writes here; a successful turn's usage
/// still travels on the turn itself, so nothing is counted twice.
nonisolated final class AgentUsageLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var usage = AgentUsage()

    init() {}

    func add(_ delta: AgentUsage) {
        lock.lock()
        usage.merge(delta)
        lock.unlock()
    }

    func drain() -> AgentUsage {
        lock.lock()
        defer { lock.unlock() }
        let drained = usage
        usage = AgentUsage()
        return drained
    }

    var snapshot: AgentUsage {
        lock.lock()
        defer { lock.unlock() }
        return usage
    }
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
    /// A Stop that arrived after the model had already streamed part of its
    /// answer. The partial text is the run's work; dropping it lost the
    /// transcript entry (and the report fell back to an earlier step) even
    /// though the operator had watched it stream.
    case cancelledAfterPartial(text: String, thinking: String)

    /// Whether this error means "the run was stopped", with or without a
    /// partial turn. Comparisons against `.cancelled` alone miss the partial
    /// case and reported it as a failure.
    var isCancellation: Bool {
        switch self {
        case .cancelled, .cancelledAfterPartial: return true
        default: return false
        }
    }

    /// What the model managed to say before the Stop, when there is any.
    var partialTurn: (text: String, thinking: String)? {
        guard case .cancelledAfterPartial(let text, let thinking) = self else { return nil }
        return (text, thinking)
    }

    var errorDescription: String? {
        switch self {
        case .unavailable(let message): return message
        case .refused(let message): return message
        case .transport(let message): return message
        case .malformedResponse(let message): return message
        case .cancelled, .cancelledAfterPartial: return "The subagent run was cancelled."
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
