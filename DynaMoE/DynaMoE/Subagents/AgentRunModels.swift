//
//  AgentRunModels.swift
//  DynaMoE
//
//  Value model for the model-backed subagent runtime. See SUBAGENTS.md.
//

import Foundation

nonisolated enum AgentBackendKind: String, Codable, Equatable, Sendable {
    case cloud
    case local
    case appleFoundationModel
    case scripted

    var displayName: String {
        switch self {
        case .cloud: return "Cloud"
        case .local: return "Local"
        case .appleFoundationModel: return "Apple Foundation Model"
        case .scripted: return "Scripted"
        }
    }
}

nonisolated struct AgentBackendDescriptor: Codable, Equatable, Sendable {
    var kind: AgentBackendKind
    var providerId: String?
    var modelId: String?
    var displayName: String

    var isRemote: Bool { kind == .cloud }

    static let scripted = AgentBackendDescriptor(
        kind: .scripted,
        providerId: nil,
        modelId: nil,
        displayName: "Scripted"
    )
}

nonisolated enum AgentRunStatus: String, Codable, Equatable, Sendable, CaseIterable {
    case pending
    case preparing
    case running
    case waitingForApproval
    case completed
    case failed
    case cancelled
    case blocked

    var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled, .blocked: return true
        case .pending, .preparing, .running, .waitingForApproval: return false
        }
    }

    var displayName: String {
        switch self {
        case .pending: return "Pending"
        case .preparing: return "Preparing"
        case .running: return "Running"
        case .waitingForApproval: return "Awaiting Approval"
        case .completed: return "Completed"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        case .blocked: return "Blocked"
        }
    }
}

nonisolated enum AgentRolePreset: String, Codable, Equatable, Sendable, CaseIterable {
    case explore
    case implementer
    case verifier
    case reviewer
    case general
    case custom

    var displayName: String {
        switch self {
        case .explore: return "Explore"
        case .implementer: return "Implementer"
        case .verifier: return "Verifier"
        case .reviewer: return "Reviewer"
        case .general: return "General"
        case .custom: return "Custom"
        }
    }
}

nonisolated struct AgentBudget: Codable, Equatable, Sendable {
    var maxSteps: Int
    var maxCompletionTokens: Int
    var maxWallClockSeconds: Int
    var maxCostUSD: Double?

    static let standard = AgentBudget(
        maxSteps: 24,
        maxCompletionTokens: 24_000,
        maxWallClockSeconds: 600,
        maxCostUSD: nil
    )

    var isValid: Bool {
        maxSteps > 0 && maxCompletionTokens > 0 && maxWallClockSeconds > 0
    }
}

nonisolated enum AgentIsolation: Codable, Equatable, Sendable {
    case none
    case worktree(path: String, branch: String)

    var isIsolated: Bool {
        if case .worktree = self { return true }
        return false
    }

    var worktreePath: String? {
        if case .worktree(let path, _) = self { return path }
        return nil
    }
}

nonisolated public enum AgentApprovalPolicy: String, Codable, Equatable, Sendable {
    case auto
    case spawnOnly
    case everyStateChange
    case deny

    var displayName: String {
        switch self {
        case .auto: return "Automatic"
        case .spawnOnly: return "Approve at spawn only"
        case .everyStateChange: return "Approve every change"
        case .deny: return "Denied"
        }
    }
}

nonisolated struct AgentUsage: Codable, Equatable, Sendable {
    var steps: Int = 0
    var promptTokens: Int = 0
    var completionTokens: Int = 0
    var cachedTokens: Int = 0
    var costUSD: Double = 0
    var wallClockSeconds: Double = 0
    var toolCalls: Int = 0
    var backendRequests: Int = 0

    mutating func merge(_ other: AgentUsage) {
        steps += other.steps
        promptTokens += other.promptTokens
        completionTokens += other.completionTokens
        cachedTokens += other.cachedTokens
        costUSD += other.costUSD
        wallClockSeconds += other.wallClockSeconds
        toolCalls += other.toolCalls
        backendRequests += other.backendRequests
    }

    var totalTokens: Int { promptTokens + completionTokens }
}

nonisolated struct AgentFinding: Codable, Equatable, Sendable {
    var claim: String
    var evidence: String
    var confidence: Double
}

nonisolated struct AgentTestOutcome: Codable, Equatable, Sendable {
    var command: String
    var exitCode: Int32
    var passed: Int?
    var failed: Int?
    var summary: String

    var succeeded: Bool { exitCode == 0 && (failed ?? 0) == 0 }
}

nonisolated struct AgentReport: Codable, Equatable, Sendable {
    var status: AgentRunStatus
    var summary: String
    var findings: [AgentFinding]
    var changedFiles: [String]
    var diff: String?
    var tests: [AgentTestOutcome]
    var risks: [String]
    var openQuestions: [String]
    var usage: AgentUsage
    var backend: AgentBackendDescriptor
}

nonisolated enum AgentTranscriptEntryKind: String, Codable, Equatable, Sendable {
    case task
    case assistantText
    case assistantThinking
    case toolCall
    case toolResult
    case status
    case usage
}

nonisolated struct AgentTranscriptEntry: Identifiable, Codable, Equatable, Sendable {
    var id: UUID
    var index: Int
    var stepIndex: Int
    var timestamp: Date
    var kind: AgentTranscriptEntryKind
    var text: String?
    var toolRecord: ToolCallRecord?

    init(
        id: UUID = UUID(),
        index: Int,
        stepIndex: Int,
        timestamp: Date = Date(),
        kind: AgentTranscriptEntryKind,
        text: String? = nil,
        toolRecord: ToolCallRecord? = nil
    ) {
        self.id = id
        self.index = index
        self.stepIndex = stepIndex
        self.timestamp = timestamp
        self.kind = kind
        self.text = text
        self.toolRecord = toolRecord
    }
}

nonisolated struct AgentMessageRecord: Identifiable, Codable, Equatable, Sendable {
    var id: UUID
    var sender: String
    var content: String
    var timestamp: Date

    init(id: UUID = UUID(), sender: String, content: String, timestamp: Date = Date()) {
        self.id = id
        self.sender = sender
        self.content = content
        self.timestamp = timestamp
    }
}

nonisolated struct AgentRunRecord: Identifiable, Codable, Equatable, Sendable {
    var id: UUID
    var parentSessionId: UUID?
    var parentTurnId: UUID?
    var role: String
    var preset: AgentRolePreset
    var task: String
    var contextSummary: String?
    var backend: AgentBackendDescriptor
    var workingDirectory: String
    var isolation: AgentIsolation
    var allowedTools: [String]
    var budget: AgentBudget
    var status: AgentRunStatus
    var createdAt: Date
    var startedAt: Date?
    var completedAt: Date?
    var transcript: [AgentTranscriptEntry]
    var report: AgentReport?
    var usage: AgentUsage
    var messages: [AgentMessageRecord]
    var error: String?
    var depth: Int
    var summaryRelayed: Bool
    /// When an isolated run's changes were applied to the main tree, if they
    /// were. Optional and additive: records written before Phase 4 decode with
    /// no value, and a newer build's field is never rewritten by an older one.
    var appliedAt: Date?
    var schemaVersion: Int

    static let currentSchemaVersion = 1

    init(
        id: UUID = UUID(),
        parentSessionId: UUID? = nil,
        parentTurnId: UUID? = nil,
        role: String,
        preset: AgentRolePreset = .general,
        task: String,
        contextSummary: String? = nil,
        backend: AgentBackendDescriptor,
        workingDirectory: String,
        isolation: AgentIsolation = .none,
        allowedTools: [String],
        budget: AgentBudget = .standard,
        status: AgentRunStatus = .pending,
        createdAt: Date = Date(),
        startedAt: Date? = nil,
        completedAt: Date? = nil,
        transcript: [AgentTranscriptEntry] = [],
        report: AgentReport? = nil,
        usage: AgentUsage = AgentUsage(),
        messages: [AgentMessageRecord] = [],
        error: String? = nil,
        depth: Int = 0,
        summaryRelayed: Bool = false,
        appliedAt: Date? = nil,
        schemaVersion: Int = AgentRunRecord.currentSchemaVersion
    ) {
        self.id = id
        self.parentSessionId = parentSessionId
        self.parentTurnId = parentTurnId
        self.role = role
        self.preset = preset
        self.task = task
        self.contextSummary = contextSummary
        self.backend = backend
        self.workingDirectory = workingDirectory
        self.isolation = isolation
        self.allowedTools = allowedTools
        self.budget = budget
        self.status = status
        self.createdAt = createdAt
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.transcript = transcript
        self.report = report
        self.usage = usage
        self.messages = messages
        self.error = error
        self.depth = depth
        self.summaryRelayed = summaryRelayed
        self.appliedAt = appliedAt
        self.schemaVersion = schemaVersion
    }
}

nonisolated enum AgentRunEvent: Sendable {
    case status(AgentRunStatus, detail: String)
    case assistantDelta(entryId: UUID, text: String, isThinking: Bool)
    case toolStarted(ToolCallRecord)
    case toolFinished(ToolCallRecord)
    case usage(AgentUsage)
    case completed(AgentReport)
}

nonisolated struct AgentToolCall: @unchecked Sendable {
    let id: String
    let name: String
    let arguments: [String: Any]
    let rawArguments: String
    let rawText: String

    init(
        id: String = UUID().uuidString,
        name: String,
        arguments: [String: Any],
        rawArguments: String = "",
        rawText: String = ""
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.rawArguments = rawArguments
        self.rawText = rawText
    }

    var parsedCall: ParsedToolCall {
        ParsedToolCall(
            name: name,
            arguments: arguments,
            rawArguments: rawArguments,
            rawText: rawText.isEmpty ? name : rawText
        )
    }

    var fingerprint: String {
        let encoded: String
        if JSONSerialization.isValidJSONObject(arguments),
           let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            encoded = text
        } else {
            encoded = arguments.keys.sorted().map { "\($0)=\(arguments[$0] ?? "")" }.joined(separator: "&")
        }
        return "\(name)|\(encoded)"
    }
}

nonisolated struct AgentSampling: Codable, Equatable, Sendable {
    var temperature: Double
    var topP: Double
    var maxTokens: Int

    static let `default` = AgentSampling(temperature: 0.2, topP: 0.95, maxTokens: 2048)
}

nonisolated enum AgentBackendMessage: Sendable {
    case user(String)
    case assistant(text: String, thinking: String, calls: [AgentToolCall])
    case toolResult(callId: String, name: String, content: String, isError: Bool)
}

nonisolated public struct AgentApprovalRequest: Sendable {
    var runId: UUID
    var toolName: String
    var arguments: [String: String]
    var workingDirectory: String?
    var reason: String
}

nonisolated public enum AgentRunScopeError: LocalizedError {
    case notRunScoped
    case unknownTool(String)
    case forbiddenTool(String)

    public var errorDescription: String? {
        switch self {
        case .notRunScoped:
            return "This harness was not created for a subagent run."
        case .unknownTool(let name):
            return "Tool '\(name)' is not installed."
        case .forbiddenTool(let name):
            return "Tool '\(name)' may never be given to a subagent run."
        }
    }
}
