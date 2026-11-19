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

    /// Model-supplied overrides are clamped into these bounds at resolution.
    /// A huge `max_wall_clock_seconds` traps the loop's nanosecond conversion
    /// and takes the app down; a runaway `max_steps` or token budget outlives
    /// every real task. Clamping keeps the spawn's promise bounded without
    /// refusing the spawn over a number the model inflated.
    static let maximumSteps = 1_000
    static let maximumCompletionTokens = 10_000_000
    static let maximumWallClockSeconds = 86_400

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

    var worktreeBranch: String? {
        if case .worktree(_, let branch) = self { return branch }
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

nonisolated enum AgentUnpricedReason: String, Codable, Equatable, Sendable {
    /// The model has no recorded price, so a token count can be named but not
    /// converted to dollars.
    case noPriceRecorded
    /// The provider billed a request but reported no usage block (or an
    /// incomplete one), so neither the tokens nor the dollars can be named.
    /// This is NOT a missing price: sending the operator to record one would
    /// fix nothing.
    case providerOmittedUsage
}

nonisolated struct AgentUsage: Codable, Equatable, Sendable {
    var steps: Int = 0
    var promptTokens: Int = 0
    var completionTokens: Int = 0
    var cachedTokens: Int = 0
    var costUSD: Double = 0
    /// True when a backend reported token usage for a model with no recorded
    /// price, so `costUSD` is a FLOOR rather than the bill. A cost ceiling that
    /// compares against an unpriced run is inert — the tokens cost something
    /// the app cannot name — so the loop refuses to run with a ceiling it
    /// cannot enforce instead of reading this as "$0 spent".
    var costIsUnpriced: Bool = false
    /// WHY the cost could not be priced, so the refusal names something the
    /// operator can act on. See `AgentUnpricedReason`.
    var costUnpricedReason: AgentUnpricedReason? = nil
    /// The completion count of the LAST request, beside `latestPromptTokens`.
    /// The context footprint is the last prompt (which already contains every
    /// earlier completion) plus only this step's generation, not the sum of
    /// every step's completions.
    var latestCompletionTokens: Int = 0
    var wallClockSeconds: Double = 0
    var toolCalls: Int = 0
    var backendRequests: Int = 0
    /// The prompt size of the LAST request this usage folded in. Every backend
    /// reports the FULL prompt each step was sent (correct for billing, and the
    /// spend column prices it), so `promptTokens` summed over steps grows with
    /// the square of the step count while the run's context only grows with the
    /// conversation. This field is what the UI's "tokens" number is made of
    /// (see `contextTokens`); it is never summed by `merge`.
    var latestPromptTokens: Int = 0

    /// Present because `init(from:)` below displaces the memberwise init, and
    /// every call site builds a usage as `AgentUsage()` then sets fields.
    init() {}

    mutating func merge(_ other: AgentUsage) {
        steps += other.steps
        promptTokens += other.promptTokens
        completionTokens += other.completionTokens
        cachedTokens += other.cachedTokens
        costUSD += other.costUSD
        costIsUnpriced = costIsUnpriced || other.costIsUnpriced
        // `noPriceRecorded` is the ACTIONABLE cause (recording a price fixes
        // it) and a provider that omitted usage is a different fact, so it
        // wins over `providerOmittedUsage` regardless of which step reported
        // first. Previously the first cause won for the life of the run.
        if costUnpricedReason == nil
            || costUnpricedReason == .providerOmittedUsage && other.costUnpricedReason == .noPriceRecorded {
            costUnpricedReason = other.costUnpricedReason
        }
        wallClockSeconds += other.wallClockSeconds
        toolCalls += other.toolCalls
        backendRequests += other.backendRequests
        latestPromptTokens = max(latestPromptTokens, other.latestPromptTokens)
        latestCompletionTokens = max(latestCompletionTokens, other.latestCompletionTokens)
    }

    /// The billed token total: every step's full prompt plus everything
    /// generated. This is what the requests cost, not what the conversation
    /// carries.
    var totalTokens: Int { promptTokens + completionTokens }

    /// The context footprint the UI shows — the same doctrine as the chat
    /// header's token chip: "the prompt the model last ingested plus the tokens
    /// it generated", not a running total of what each step billed. A record
    /// that predates `latestPromptTokens` falls back to its recorded sum, the
    /// only number it holds.
    var contextTokens: Int {
        guard latestPromptTokens > 0 else {
            // A record that predates both fields holds only the sums.
            return promptTokens + completionTokens
        }
        // The last prompt already CONTAINS every earlier step's generation,
        // so adding the summed `completionTokens` counted them twice. Only
        // this step's output is still part of the live context.
        return latestPromptTokens + latestCompletionTokens
    }

    /// Defaults-tolerant on purpose: the store's decode-failure path is "the
    /// record is lost", so a stored usage that predates a field must still
    /// decode with that field's default rather than failing the whole record.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        steps = try container.decodeIfPresent(Int.self, forKey: .steps) ?? 0
        promptTokens = try container.decodeIfPresent(Int.self, forKey: .promptTokens) ?? 0
        completionTokens = try container.decodeIfPresent(Int.self, forKey: .completionTokens) ?? 0
        cachedTokens = try container.decodeIfPresent(Int.self, forKey: .cachedTokens) ?? 0
        costUSD = try container.decodeIfPresent(Double.self, forKey: .costUSD) ?? 0
        costIsUnpriced = try container.decodeIfPresent(Bool.self, forKey: .costIsUnpriced) ?? false
        costUnpricedReason = try container.decodeIfPresent(AgentUnpricedReason.self, forKey: .costUnpricedReason)
        wallClockSeconds = try container.decodeIfPresent(Double.self, forKey: .wallClockSeconds) ?? 0
        toolCalls = try container.decodeIfPresent(Int.self, forKey: .toolCalls) ?? 0
        backendRequests = try container.decodeIfPresent(Int.self, forKey: .backendRequests) ?? 0
        latestPromptTokens = try container.decodeIfPresent(Int.self, forKey: .latestPromptTokens) ?? 0
        latestCompletionTokens = try container.decodeIfPresent(Int.self, forKey: .latestCompletionTokens) ?? 0
    }
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
    /// The branch the main tree had CHECKED OUT when this run's work landed.
    /// An apply lands on whatever is checked out, so the record is the only
    /// place that answers "where did it go" after the fact.
    var appliedToBranch: String?
    /// Files a patch could not carry that are still in the run's worktree, so
    /// an APPLIED run can still keep its tree. Optional and additive: a record
    /// written before this field decodes with nil, which means no kept tree.
    /// The launch sweep protects a worktree while this is non-empty, because
    /// deleting it would destroy the only copy of those files.
    var keptIgnoredFiles: [String]?
    /// The main repository an isolated worktree was cut from. Optional and
    /// additive: a record written before this field falls back to deriving the
    /// root from git, which cannot name the main worktree of a submodule or a
    /// `--separate-git-dir` repository from a linked worktree alone.
    var repositoryRoot: String?
    /// The commit the worktree was cut from. Optional and additive. Snapshots
    /// are taken against it, so a reset or rebase on the main branch cannot
    /// change what a restored run appears to have changed.
    var baseCommit: String?
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
        appliedToBranch: String? = nil,
        keptIgnoredFiles: [String]? = nil,
        repositoryRoot: String? = nil,
        baseCommit: String? = nil,
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
        self.appliedToBranch = appliedToBranch
        self.keptIgnoredFiles = keptIgnoredFiles
        self.repositoryRoot = repositoryRoot
        self.baseCommit = baseCommit
        self.schemaVersion = schemaVersion
    }

    /// Whether deleting this record would leave a worktree nothing protects.
    ///
    /// An unapplied isolated run's tree is the only full copy of its work; so
    /// is an APPLIED run's tree kept because a patch could not carry its
    /// git-ignored files. Both are the operator's to clear, never the clock's.
    nonisolated var holdsWorktreeWork: Bool {
        isolation.isIsolated
            && (appliedAt == nil || !(keptIgnoredFiles?.isEmpty ?? true))
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
    /// True when this card exists because the call could NOT be proved confined
    /// to the run's isolated workspace (or it targets the sandbox's own git
    /// metadata, which can redirect the run's commits into the operator's
    /// repository). Turbo Mode is a switch over ROUTINE writes; an escape from
    /// the configuration presented as safe is not routine, and the approval
    /// center never bypasses a card marked here.
    var containmentEscape: Bool = false
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
