//
//  SubagentManager.swift
//  DynaMoE
//
//  Option 2: Subagent & Multi-Agent Delegation Harness (spawn_subagent)
//  - Patterned after Antigravity / Claude Code / CrewAI multi-agent delegation
//  - Isolated background subagents with dedicated roles (Codebase Researcher, Test Runner, Shader Optimizer, Custom)
//  - Thread-safe inter-agent messaging and unified summary aggregation
//  - Full lifecycle tracking and step-by-step transcript preservation
//

import Foundation
import Combine
import SwiftUI

// MARK: - Subagent Status

public enum SubagentStatus: String, Codable, CaseIterable {
    case pending = "pending"
    case running = "running"
    case waitingForApproval = "waiting_for_approval"
    case completed = "completed"
    case failed = "failed"
    /// A run its loop guard stopped (a repeat loop, a failure streak, a
    /// withdrawn tool). Distinct from `failed` because nothing went wrong with
    /// the machinery: the run was working and the guard decided continuing
    /// would not help. It used to be flattened to `failed` at every surface.
    case blocked = "blocked"
    case cancelled = "cancelled"

    public var displayName: String {
        switch self {
        case .pending: return "Pending"
        case .running: return "Running"
        case .waitingForApproval: return "Awaiting Approval"
        case .completed: return "Completed"
        case .failed: return "Failed"
        case .blocked: return "Blocked"
        case .cancelled: return "Cancelled"
        }
    }

    public var iconName: String {
        switch self {
        case .pending: return "clock"
        case .running: return "arrow.triangle.2.circlepath"
        case .waitingForApproval: return "hand.raised.fill"
        case .completed: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .blocked: return "exclamationmark.octagon.fill"
        case .cancelled: return "xmark.circle"
        }
    }

    /// True once the run can no longer change state on its own.
    public var isTerminal: Bool {
        self == .completed || self == .failed || self == .blocked || self == .cancelled
    }

    public var color: Color {
        switch self {
        case .pending: return .secondary
        case .running: return .blue
        case .waitingForApproval: return .orange
        case .completed: return .green
        case .failed: return .red
        case .blocked: return .orange
        case .cancelled: return .secondary
        }
    }
}

// MARK: - Subagent Role Archetype

public enum SubagentRoleArchetype: String, Codable, CaseIterable {
    case codebaseResearcher = "Codebase Researcher"
    case testRunner = "Test Runner"
    case shaderOptimizer = "Shader Optimizer"
    case documentationWriter = "Documentation Writer"
    case custom = "Custom"

    public static func match(from rawRole: String) -> SubagentRoleArchetype {
        let lower = rawRole.lowercased()
        if lower.contains("research") || lower.contains("search") || lower.contains("explore") {
            return .codebaseResearcher
        } else if lower.contains("test") || lower.contains("runner") || lower.contains("qa") {
            return .testRunner
        } else if lower.contains("shader") || lower.contains("optimize") || lower.contains("perf") {
            return .shaderOptimizer
        } else if lower.contains("doc") || lower.contains("write") {
            return .documentationWriter
        }
        return .custom
    }

    public var defaultIconName: String {
        switch self {
        case .codebaseResearcher: return "sparkle.magnifyingglass"
        case .testRunner: return "testtube.2"
        case .shaderOptimizer: return "bolt.badge.clock"
        case .documentationWriter: return "doc.text.fill"
        case .custom: return "person.crop.circle.badge.gearshape"
        }
    }
}

// MARK: - Step Record & Message Models

public struct SubagentStepRecord: Identifiable, Codable, Equatable {
    public let id: UUID
    public let stepIndex: Int
    public let timestamp: Date
    public let actionName: String
    public let arguments: [String: String]
    public let output: String
    public let durationSeconds: Double
    public let isError: Bool

    public init(
        id: UUID = UUID(),
        stepIndex: Int,
        timestamp: Date = Date(),
        actionName: String,
        arguments: [String: String] = [:],
        output: String,
        durationSeconds: Double = 0.0,
        isError: Bool = false
    ) {
        self.id = id
        self.stepIndex = stepIndex
        self.timestamp = timestamp
        self.actionName = actionName
        self.arguments = arguments
        self.output = output
        self.durationSeconds = durationSeconds
        self.isError = isError
    }
}

public struct SubagentMessage: Identifiable, Codable, Equatable {
    public let id: UUID
    public let sender: String        // "coordinator", "user", "subagent"
    public let content: String
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        sender: String,
        content: String,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.sender = sender
        self.content = content
        self.timestamp = timestamp
    }
}

// MARK: - Subagent Instance

/// Hands the scheduler a way to stop a run without retaining it.
///
/// A registry entry must not keep a finished run alive, and the closure it holds
/// is `@Sendable`; a weak box is how the two requirements meet.
nonisolated final class InstanceCancellationBox: @unchecked Sendable {
    private weak var instance: SubagentInstance?

    init(_ instance: SubagentInstance) {
        self.instance = instance
    }

    func cancel(_ reason: String) {
        let instance = instance
        Task { @MainActor in
            instance?.cancel(reason: reason)
        }
    }
}

public final class SubagentInstance: Identifiable, ObservableObject {
    public let id: UUID
    public let role: String
    public let archetype: SubagentRoleArchetype
    public let taskDescription: String
    public let allowedTools: [String]
    /// The mode the conversation that spawned this subagent was running.
    /// Plan mode's promise has to cover the work it hands off: a read-only
    /// parent can only produce read-only children.
    public let parentMode: AgentMode
    /// Whether this subagent's pipelines may execute shell commands. The
    /// executor's allow-list covers every pipeline that runs its tools through
    /// it; the Test Runner pipeline calls `AgentHarness.runProcess` directly,
    /// so it asks this instead. A read-only parent can only produce a read-only
    /// child on every path.
    public var canRunShellCommands: Bool { parentMode == .code }
    public let contextSummary: String?
    public let parentSessionId: UUID?
    /// The coordinator TURN that spawned this run, so a Stop in a later turn
    /// cannot cancel it: cancellation scopes to the turn that owns the run.
    public let parentTurnId: UUID?
    public let workingDirectory: URL?
    /// The model-backed plan, when this instance is a real agent run rather than
    /// one of the legacy canned pipelines. The manager sets it through the
    /// internal initializer below.
    let modelPlan: AgentSpawnPlan?

    @Published public var status: SubagentStatus = .pending
    @Published public var liveStatusText: String = "Initializing..."
    @Published public var transcript: [SubagentStepRecord] = []
    @Published public var messages: [SubagentMessage] = []
    @Published public var finalSummary: String = ""
    /// True once the subagent's final summary has been surfaced to the coordinator
    /// (via synchronous spawn, get_subagent_status, or an automatic relay) so the
    /// loop guard only injects unsalvaged reports back into the conversation.
    @Published public var isSummaryRelayed: Bool = false {
        didSet {
            if isSummaryRelayed != oldValue { persistRecord() }
        }
    }
    @Published public var executionDurationSeconds: Double = 0.0
    @Published public var currentStepIndex: Int = 0
    /// The run's structured report and usage, when a model ran it.
    private(set) var report: AgentReport? = nil
    private(set) var usage: AgentUsage = AgentUsage()
    /// The loop's failure reason, kept apart from the summary: a budget-exhausted
    /// run's summary is now the analysis it managed to write, and that is not
    /// the error the record's red banner should name.
    private(set) var lastError: String? = nil
    private(set) var backendDescriptor: AgentBackendDescriptor? = nil
    /// The loop's own transcript once the run finishes. Until then the live
    /// view is `transcript` (tool steps) plus the streaming text below.
    private(set) var runTranscript: [AgentTranscriptEntry] = []
    /// When an isolated run's worktree changes were applied to the main tree.
    /// Part of the record so a restored run still knows it was applied. Published
    /// because the inspector reads it through a value snapshot: without this, the
    /// apply service's write would not redraw the pane that offered the button.
    @Published private(set) var appliedAt: Date? = nil
    /// The branch the destination tree had checked out when this run's work
    /// landed, so the record answers "where did it go" after the fact.
    @Published private(set) var appliedToBranch: String? = nil
    /// Files a patch could not carry that are still in the run's worktree, so
    /// an APPLIED run can still keep its tree and the launch sweep protects it.
    @Published private(set) var keptIgnoredFiles: [String] = []
    /// The step currently streaming. The loop mints one delta id pair per step,
    /// so a new id is the boundary that clears the previous step's text.
    @Published public private(set) var liveAssistantText: String = ""
    @Published public private(set) var liveThinkingText: String = ""
    private(set) var startedAt: Date? = nil
    /// Whether the finish notice may ask for attention. A synchronous spawn's
    /// caller is already watching the tool card, so only a run left in the
    /// background should bounce the Dock or post a banner.
    var notifiesOnFinish = true
    private var liveTextEntryId: UUID? = nil
    private var liveThinkingEntryId: UUID? = nil

    public let createdAt: Date
    public private(set) var completedAt: Date? = nil

    private var executionTask: Task<Void, Never>? = nil
    private var completionContinuations: [CheckedContinuation<String, Never>] = []
    private var messageStore: [SubagentMessage] = []
    private var consumedMessageIds: Set<UUID> = []
    private var isFinished: Bool = false
    /// False from `startExecution` until the execution task has actually
    /// unwound. A Stop latches `status`/`isFinished` immediately, but the task
    /// may still be killing a child process or writing its last tool result;
    /// the apply path reads this so it cannot land a worktree a live child is
    /// still editing.
    private var executionUnwound: Bool = true
    /// The isolation this run must PERSIST. Normally the plan's, but a run that
    /// never reached its loop discards its worktree and clears this: the
    /// record then no longer claims an unapplied tree, which is what kept
    /// `prune`, `clearCompleted` and the launch sweep away from it forever.
    private var recordedIsolation: AgentIsolation?
    private let lock = NSLock()

    public init(
        id: UUID = UUID(),
        role: String,
        taskDescription: String,
        allowedTools: [String] = [],
        parentMode: AgentMode = .code,
        contextSummary: String? = nil,
        parentSessionId: UUID? = nil,
        parentTurnId: UUID? = nil,
        workingDirectory: URL? = nil
    ) {
        self.id = id
        self.role = role
        self.archetype = SubagentRoleArchetype.match(from: role)
        self.taskDescription = taskDescription
        self.allowedTools = allowedTools.isEmpty ? [
            "file_read", "find_files", "grep_search", "codebase_search", "web_search", "web_fetch"
        ] : allowedTools
        self.parentMode = parentMode
        self.contextSummary = contextSummary
        self.parentSessionId = parentSessionId
        self.parentTurnId = parentTurnId
        self.workingDirectory = workingDirectory
        self.modelPlan = nil
        self.recordedIsolation = AgentIsolation.none
        self.createdAt = Date()
    }

    /// A model-backed run. Internal because `AgentSpawnPlan` is an
    /// implementation type: the public surface stays the legacy initializer the
    /// tests and the drawer already use.
    init(id: UUID, plan: AgentSpawnPlan) {
        self.id = id
        self.role = plan.role
        self.archetype = SubagentRoleArchetype.match(from: plan.role)
        self.taskDescription = plan.taskDescription
        self.allowedTools = plan.toolNames
        self.parentMode = plan.approvalPolicy == .deny ? .plan : .code
        self.contextSummary = plan.contextSummary
        self.parentSessionId = plan.parentSessionId
        self.parentTurnId = plan.parentTurnId
        self.workingDirectory = plan.workingDirectory
        self.modelPlan = plan
        self.recordedIsolation = plan.isolation
        self.backendDescriptor = plan.backendDescriptor
        self.createdAt = Date()
    }

    // MARK: - Public Control APIs

    /// Cancels this subagent's execution.
    public func cancel() {
        cancel(reason: "Cancelled by user or coordinator.")
    }

    /// Stops the run with a reason it can report. The scheduler's cancellation
    /// tree, the Stop button and the run's own preemption all land here.
    public func cancel(reason: String) {
        // A finished run has already written its report. Cancelling it on the
        // way out (Quit calls `cancelAll`) must not rewrite a completed record
        // as `.cancelled`, or erase the summary the operator waited for.
        lock.lock()
        guard !isFinished, !status.isTerminal else {
            lock.unlock()
            return
        }
        executionTask?.cancel()
        executionTask = nil
        isFinished = true
        let continuations = completionContinuations
        completionContinuations.removeAll()
        lock.unlock()

        // Synchronous, not dispatched: a caller that cancels and then asks for
        // the summary must not observe `isFinished` with an empty report.
        status = .cancelled
        liveStatusText = reason
        completedAt = Date()
        finalSummary = reason
        lastError = reason
        persistRecord()

        for c in continuations {
            c.resume(returning: reason)
        }
    }

    /// The run is parked on an approval card. `waitingForApproval` existed in
    /// the status model and the inspector's orange rendering but nothing ever
    /// emitted it, so every parked run read as `.running`.
    func noteApprovalWait(toolName: String) {
        apply(.status(
            .waitingForApproval,
            detail: "Waiting for approval to run \(toolName)…"
        ))
    }

    /// The card was answered (approved, declined or timed out) and the loop
    /// owns the run again.
    func noteApprovalResolved() {
        apply(.status(.running, detail: "Working on \(role)…"))
    }

    /// Appends an inter-agent message from coordinator or user.
    public func addMessage(sender: String, content: String) {
        let msg = SubagentMessage(sender: sender, content: content)
        lock.lock()
        messageStore.append(msg)
        lock.unlock()

        if Thread.isMainThread {
            self.messages.append(msg)
        } else {
            DispatchQueue.main.async {
                self.messages.append(msg)
            }
        }
    }

    public var recordedMessages: [SubagentMessage] {
        lock.lock()
        defer { lock.unlock() }
        return messageStore
    }

    // MARK: - Durable Record

    /// A snapshot of this run in the shape the store and the inspector use.
    /// Live instances are projections of the same record the store keeps, so a
    /// field added here must also survive the round trip on disk.
    func snapshotRecord() -> AgentRunRecord {
        let plan = modelPlan
        let descriptor = backendDescriptor
            ?? plan?.backendDescriptor
            ?? AgentBackendDescriptor(
                kind: .scripted,
                providerId: nil,
                modelId: nil,
                displayName: "Coordinator pipeline"
            )
        let workingPath = workingDirectory?.path ?? FileManager.default.currentDirectoryPath
        let entries = runTranscript.isEmpty ? legacyTranscriptEntries() : runTranscript
        return AgentRunRecord(
            id: id,
            parentSessionId: parentSessionId,
            parentTurnId: plan?.parentTurnId,
            role: role,
            preset: plan?.preset ?? AgentRolePresetCatalog.match(role: role),
            task: taskDescription,
            contextSummary: contextSummary,
            backend: descriptor,
            workingDirectory: workingPath,
            isolation: recordedIsolation ?? .none,
            allowedTools: allowedTools,
            budget: plan?.budget ?? .standard,
            status: Self.agentRunStatus(for: status),
            createdAt: createdAt,
            startedAt: startedAt,
            completedAt: completedAt,
            transcript: entries,
            report: report,
            usage: usage,
            messages: recordedMessages.map { message in
                AgentMessageRecord(
                    id: message.id,
                    sender: message.sender,
                    content: message.content,
                    timestamp: message.timestamp
                )
            },
            error: (status == .failed || status == .cancelled) ? (lastError ?? finalSummary) : nil,
            depth: 0,
            summaryRelayed: isSummaryRelayed,
            appliedAt: appliedAt,
            appliedToBranch: appliedToBranch,
            keptIgnoredFiles: keptIgnoredFiles.isEmpty ? nil : keptIgnoredFiles,
            repositoryRoot: (recordedIsolation ?? .none).isIsolated ? plan?.repositoryRoot?.path : nil,
            baseCommit: (recordedIsolation ?? .none).isIsolated ? plan?.baseCommit : nil
        )
    }

    /// The worktree this run was going to use has been removed, so the record
    /// must stop claiming it.
    ///
    /// The pre-loop refusals (a scheduler timeout, an unusable context) create
    /// a worktree during the spawn and then discard it. Persisting
    /// `isolation: .worktree` with `appliedAt == nil` from that point made
    /// `holdsWorktreeWork` permanently true, so `prune`, `clearCompleted` and
    /// the launch sweep skipped the record forever and Apply always failed
    /// `worktreeGone`. `finishRun` writes the record this feeds.
    func forgetDiscardedWorktree() {
        recordedIsolation = AgentIsolation.none
        // If Stop/cancel already latched `isFinished` and persisted `.worktree`
        // before the queued run could discard its unused worktree, `finishRun`
        // returns early without writing again. Persist here so disk matches
        // memory and the record is not left retention-exempt / `worktreeGone`.
        if hasFinished() {
            persistRecord()
        }
    }

    /// Whether this run still holds worktree work nobody has landed, read from
    /// the LIVE fields rather than from `modelPlan`.
    ///
    /// `modelPlan` keeps the isolation the run was SPAWNED with, so a pre-loop
    /// refusal that discarded its new worktree still looked isolated here after
    /// `forgetDiscardedWorktree()` cleared `recordedIsolation`: the record was
    /// clearable on disk while the in-memory instance stayed in the drawer
    /// until relaunch. Mirrors `AgentRunRecord.holdsWorktreeWork` so the two
    /// halves answer the same question.
    var holdsUnappliedWorktreeWork: Bool {
        let isolation = recordedIsolation ?? .none
        return isolation.isIsolated
            && (appliedAt == nil || !keptIgnoredFiles.isEmpty)
    }

    /// Records that this run's worktree landed on the main tree. The store is
    /// written synchronously by the apply path, so a crash immediately after
    /// the patch cannot leave a record that would try to apply it twice.
    /// `appliedTo` is the branch the destination had CHECKED OUT: an apply
    /// lands on whatever is checked out, and the record is the only place
    /// that answers where the work went.
    ///
    /// Returns false when the record could not be updated. A conversation
    /// delete re-scopes an unapplied run to the unscoped bucket and tombstones
    /// its old directory; writing the live snapshot to the old path was
    /// silently refused, so the worktree was cleaned up while the surviving
    /// record still said unapplied. `mutate` locates the record by id for
    /// exactly that case.
    @discardableResult
    func markApplied(at date: Date, files: [String], appliedTo: String? = nil) async -> Bool {
        appliedAt = date
        appliedToBranch = appliedTo
        if var report, !files.isEmpty {
            report.changedFiles = files
            self.report = report
        }
        let written = await AgentRunStore.shared.mutateAsync(runId: id, sessionId: parentSessionId) { current in
            current.appliedAt = date
            current.appliedToBranch = appliedTo
            if !files.isEmpty, var report = current.report {
                report.changedFiles = files
                current.report = report
            }
        } != nil
        // Keeps the rest of the live state fresh in the normal case. A dropped
        // write here cannot lose `appliedAt`: the mutation above already wrote it.
        persistRecord()
        return written
    }

    /// Records that this run's worktree was kept after an apply because it
    /// holds ignored files a patch cannot carry, so the inspector's Apply
    /// does not read as unfinished business without a reason — and so the
    /// launch sweep keeps the tree instead of deleting the only copy of them.
    func noteKeptIgnoredFiles(_ files: [String]) async {
        guard !files.isEmpty else { return }
        keptIgnoredFiles = files
        let note = "The tracked patch was applied, but the worktree was kept: it also holds files a patch cannot carry (\(files.joined(separator: ", "))). Retrieve them manually before clearing the run."
        if var current = report {
            current.openQuestions.append(note)
            report = current
        }
        await AgentRunStore.shared.mutateAsync(runId: id, sessionId: parentSessionId) { current in
            current.keptIgnoredFiles = files
            if var report = current.report {
                report.openQuestions.append(note)
                current.report = report
            }
        }
        persistRecord()
    }

    /// Records that an apply kept the run's branch because it holds commits
    /// outside the applied working tree (e.g. the run switched away from its
    /// branch after committing on it).
    func noteKeptBranch(_ branch: String) async {
        let note = "The patch was applied, but branch '\(branch)' was kept because it holds commits outside the applied working tree. Inspect or delete it manually when you are done."
        if var current = report {
            current.openQuestions.append(note)
            report = current
        }
        await AgentRunStore.shared.mutateAsync(runId: id, sessionId: parentSessionId) { current in
            if var report = current.report {
                report.openQuestions.append(note)
                current.report = report
            }
        }
        persistRecord()
    }

    /// Writes the current state. Safe to call on every boundary: the store
    /// skips unchanged bytes and dispatches its own IO.
    func persistRecord() {
        AgentRunStore.shared.save(snapshotRecord())
    }

    private func legacyTranscriptEntries() -> [AgentTranscriptEntry] {
        var entries: [AgentTranscriptEntry] = [
            AgentTranscriptEntry(
                index: 0,
                stepIndex: 0,
                timestamp: createdAt,
                kind: .task,
                text: taskDescription
            )
        ]
        for (offset, step) in transcript.enumerated() {
            let record = ToolCallRecord(
                name: step.actionName,
                arguments: step.arguments,
                status: step.isError ? .error : .success,
                output: step.output,
                error: step.isError ? step.output : nil,
                executionDurationSeconds: step.durationSeconds,
                timestamp: step.timestamp
            )
            entries.append(AgentTranscriptEntry(
                index: offset + 1,
                stepIndex: step.stepIndex,
                timestamp: step.timestamp,
                kind: .toolResult,
                text: step.output,
                toolRecord: record
            ))
        }
        return entries
    }

    private static func agentRunStatus(for status: SubagentStatus) -> AgentRunStatus {
        switch status {
        case .pending: return .pending
        case .running: return .running
        case .waitingForApproval: return .waitingForApproval
        case .completed: return .completed
        case .failed: return .failed
        case .blocked: return .blocked
        case .cancelled: return .cancelled
        }
    }

    /// Waits asynchronously until this subagent reaches a terminal state.
    public func waitForCompletion() async -> String {
        await withCheckedContinuation { continuation in
            registerCompletionWaiter(continuation)
        }
    }

    /// Synchronous on purpose, called from the continuation body before it
    /// returns: `NSLock` may not be touched from an async context (an error
    /// in Swift 6 mode), so the finished-check and the waiter registration
    /// are one decision in a single lock hold here. Keeping them together is
    /// what stops a run that finishes in between from resuming nobody and
    /// stranding the waiter.
    private func registerCompletionWaiter(_ continuation: CheckedContinuation<String, Never>) {
        lock.lock()
        if isFinished {
            let summary = finalSummary
            lock.unlock()
            continuation.resume(returning: summary)
            return
        }
        completionContinuations.append(continuation)
        lock.unlock()
    }

    // MARK: - Execution Lifecycle

    /// `isFinished` read without touching the lock from an async context.
    /// A Stop can latch it while this run is suspended in the scheduler's
    /// grant, and a later `status = .running` would overwrite a cancelled
    /// record the store (and the drawer) already treats as terminal.
    private func hasFinished() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return isFinished
    }

    /// Whether the execution task is still unwinding. A Stop marks the run
    /// terminal immediately, but the task may still be killing a child process
    /// or writing one last tool result; the apply path must not land a worktree
    /// while that is true.
    var isExecutionUnwound: Bool {
        lock.lock()
        defer { lock.unlock() }
        return executionUnwound
    }

    private func markExecutionUnwound() {
        lock.lock()
        executionUnwound = true
        lock.unlock()
        // `canApplyChanges` and `canDiscard` in the inspector read
        // `isExecutionUnwound`, which is not `@Published`. Notify observers on
        // the MainActor so the Apply / Discard buttons appear as soon as a
        // stopped run finishes unwinding.
        objectWillChange.send()
    }

    public func startExecution() {
        guard status == .pending else { return }

        startedAt = Date()
        persistRecord()

        lock.lock()
        let plan = modelPlan
        executionUnwound = false
        executionTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            if let plan {
                await self.runModelBackedExecution(plan)
            } else {
                await self.runAutonomousPipeline()
            }
        }
        lock.unlock()
    }

    private func runAutonomousPipeline() async {
        let startTime = CFAbsoluteTimeGetCurrent()
        // The run stops being "executing" the moment this body returns, on the
        // same actor turn that resumed its completion waiters. A hop later was
        // too late: an apply that followed `waitForCompletion` saw a terminal
        // record with the flag still clear and refused with "still running".
        defer { markExecutionUnwound() }

        await MainActor.run {
            guard !self.hasFinished() else { return }
            self.status = .running
            self.liveStatusText = "Analyzing task requirements..."
        }
        // A Stop in the hop above owns the record; the pipeline must not run
        // tools for a run that is already terminal.
        guard !hasFinished() else { return }

        var accumulatedSummary = ""

        switch archetype {
        case .codebaseResearcher:
            accumulatedSummary = await executeResearchPipeline()

        case .testRunner:
            accumulatedSummary = await executeTestRunnerPipeline()

        case .shaderOptimizer:
            accumulatedSummary = await executeShaderOptimizerPipeline()

        case .documentationWriter, .custom:
            accumulatedSummary = await executeGenericAgentPipeline()
        }

        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let finalStatus: SubagentStatus = Task.isCancelled ? .cancelled : .completed
        finishRun(
            summary: accumulatedSummary,
            status: finalStatus,
            duration: duration,
            liveStatus: String(format: "Completed in %.2fs", duration)
        )
    }

    /// Runs the real agent loop on one backend.
    ///
    /// The loop is off the main actor (invariant 6); this method is the run's
    /// main-actor bookkeeping and its lifetime ends with the loop. Admission,
    /// the cancellation tree and the backend boundary all belong to the
    /// scheduler — a local run still claims the engine per step, which is the
    /// slot the server queue and the chat send path share.
    private func runModelBackedExecution(_ plan: AgentSpawnPlan) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        liveStatusText = "Preparing \(plan.role)…"
        // See `runAutonomousPipeline`: the flag must clear in the same actor
        // turn that resumes the completion waiters.
        defer { markExecutionUnwound() }

        let scheduler = AgentScheduler.shared
        let cancellation = InstanceCancellationBox(self)
        await scheduler.register(
            runId: id,
            parentRunId: nil,
            parentSessionId: parentSessionId,
            parentTurnId: parentTurnId,
            schedulingClass: plan.schedulingClass,
            priority: plan.priority,
            cancel: { reason in cancellation.cancel(reason) }
        )

        let ticket: UUID
        do {
            ticket = try await scheduler.acquire(
                plan.schedulingClass,
                priority: plan.priority,
                runId: id
            )
        } catch {
            await scheduler.finish(runId: id)
            await Self.discardWorktree(plan: plan)
            // The tree is gone; the record must not keep claiming it as
            // unapplied work nothing may clear.
            forgetDiscardedWorktree()
            let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            let cancelled = (error as? AgentSchedulerError) == .cancelled || Task.isCancelled
            finishRun(
                summary: message,
                status: cancelled ? .cancelled : .failed,
                duration: CFAbsoluteTimeGetCurrent() - startTime,
                liveStatus: message
            )
            return
        }
        defer {
            Task {
                await scheduler.release(ticket)
                await scheduler.finish(runId: id)
            }
        }

        // A Stop can land between the scheduler's last cancellation read and
        // this MainActor continuation. The `defer` above still releases the
        // ticket, but publishing `.running` here would overwrite the cancelled
        // record and leave `clearCompleted` counting it as active forever.
        // Clean up the unused worktree too so the cancelled record does not
        // strand it or stay retention-exempt.
        guard !hasFinished(), !Task.isCancelled else {
            await Self.discardWorktree(plan: plan)
            forgetDiscardedWorktree()
            if !hasFinished() {
                finishRun(
                    summary: "The run was cancelled.",
                    status: .cancelled,
                    duration: CFAbsoluteTimeGetCurrent() - startTime,
                    liveStatus: "Cancelled"
                )
            }
            return
        }
        status = .running
        liveStatusText = "Working on \(plan.role)…"

        let workingDirectoryURL = workingDirectory
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let context: AgentRunContext
        do {
            context = try AgentRunContext.make(
                runId: id,
                sessionId: id,
                parentTurnId: plan.parentTurnId,
                parentSessionId: plan.parentSessionId,
                toolNames: plan.toolNames,
                workingDirectory: workingDirectoryURL,
                mode: parentMode,
                approvalPolicy: plan.approvalPolicy,
                workspaceIsolated: plan.isolation.isIsolated,
                // The recorded root is what a cardless `git_commit`'s linkage
                // re-verification compares against: the worktree's `.git`
                // pointer is a file inside the sandbox, and only the root
                // tells the harness where it is SUPPOSED to point. The branch
                // is the same contract for HEAD: a carded checkout inside the
                // sandbox must not let a cardless commit move another branch.
                workspaceRepositoryRoot: plan.repositoryRoot,
                workspaceBranch: {
                    if case .worktree(_, let branch) = plan.isolation { return branch }
                    return nil
                }(),
                approvalHandler: { request in
                    await SubagentApprovalCenter.shared.request(request)
                }
            )
        } catch {
            await Self.discardWorktree(plan: plan)
            forgetDiscardedWorktree()
            let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            finishRun(
                summary: message,
                status: .failed,
                duration: CFAbsoluteTimeGetCurrent() - startTime,
                liveStatus: message
            )
            return
        }

        // The tool-call instructions have to match the backend. A cloud
        // provider receives native function declarations and must be told to
        // use them; the local renderer parses XML out of the text, so its
        // prompt teaches that dialect instead. Sending the local prompt to a
        // cloud model made it narrate `<tool_call>` XML in prose — no
        // `parsedToolCalls`, no `complete`, and a run that looked finished
        // after one paragraph.
        let systemPrompt: String
        if plan.backendDescriptor.kind == .cloud {
            systemPrompt = context.harness.buildCloudWorkerSystemPrompt(
                baseSystem: "",
                role: plan.role,
                mode: parentMode
            )
        } else {
            systemPrompt = LocalAgentBackend.workerSystemPrompt(
                role: plan.role,
                baseSystem: "",
                mode: parentMode,
                harness: context.harness,
                config: CompletionEngine.shared.promptModelConfig
            )
        }
        let input = AgentLoopInput(
            role: plan.role,
            task: plan.taskDescription,
            contextSummary: plan.contextSummary,
            systemPrompt: systemPrompt,
            budget: plan.budget,
            sampling: plan.sampling,
            sessionId: id,
            thinking: plan.thinking,
            stopSequences: [],
            // Steering the coordinator or operator queued while the run works
            // is drained before each step, so a message reported delivered
            // actually reaches the model.
            drainSteering: { [weak self] in
                await self?.drainUnconsumedSteering() ?? []
            }
        )

        // Events are applied through ONE serial stream. A Task per event let a
        // late `.running` land after `finishRun`, so the drawer (and a test)
        // could read a completed run as still running. Finishing the stream and
        // awaiting the pump before the terminal bookkeeping keeps the order the
        // loop emitted them in.
        let (events, eventSink) = AsyncStream<AgentRunEvent>.makeStream()
        let eventPump = Task { @MainActor [weak self] in
            for await event in events {
                self?.apply(event)
            }
        }
        let outcome = await HeadlessAgentLoop().run(
            input: input,
            backend: plan.backend,
            context: context,
            emit: { eventSink.yield($0) }
        )
        eventSink.finish()
        await eventPump.value

        let duration = CFAbsoluteTimeGetCurrent() - startTime
        // An isolated writer's worktree is read back here, not by the loop: the
        // loop runs inside the worktree and has no business knowing where its
        // bytes are. The readback runs on its OWN detached task so the Stop
        // that cancelled the run (or one landing right now) cannot abort the
        // bookkeeping, and git is never spawned from a cancelled task. A
        // cancelled run's worktree is KEPT: it is the only full copy of
        // whatever the run wrote before the Stop, and the record still names
        // it so the operator gets the same keep-or-clear choice a completed
        // run's tree gets.
        let finalReport = await Task.detached(priority: .userInitiated) {
            await self.finalizeWorktree(plan: plan, outcome: outcome)
        }.value
        // A Stop that landed during the readback already finished this run, and
        // its transition owns the record (see `finishRun`). The writes below
        // sit OUTSIDE that guard, so without this check a cancelled run's live
        // object ended up carrying a completed report the store never accepted —
        // the drawer disagreed with the record on relaunch.
        let alreadyFinished = hasFinished()
        if alreadyFinished {
            // A Stop owns the terminal record, but the spend of the step it
            // interrupted is only known once the loop unwinds: a cloud request
            // the provider billed (with or without a usage block) is merged
            // into the outcome by the ledger, and `cancel` persisted the
            // record before that happened. Without this write the inspector
            // reported $0 for a run the operator actually paid for. Only the
            // usage is updated; the terminal transition's summary, status and
            // worktree stay exactly as it wrote them.
            //
            // Written for ANY terminal state, not just `.cancelled`: a Stop
            // that unrolls through a `.failed` budget/teardown path otherwise
            // dropped the last step's validated spend the same way.
            usage = outcome.usage
            // The loop's own work is written for the same reason. `cancel`
            // persisted the record from live state at the instant of the Stop,
            // which was BEFORE the loop's final step landed its assistant text
            // and tool results — so `cancelledAfterPartial` kept the partial
            // turn only in memory, the drawer disagreed with the record after
            // relaunch, and the report the run had earned was lost. Status,
            // summary and the worktree stay exactly as the terminal transition
            // wrote them: only the work is added.
            if !outcome.transcript.isEmpty {
                runTranscript = outcome.transcript
            }
            // The report the loop derived from what the run managed to do.
            // Its report's `status` is `.cancelled`, so the record's own
            // terminal status and summary (written by `cancel`) still win; the
            // partial findings and best-effort summary are what the drawer and
            // the relay would otherwise have lost.
            if let finalReport {
                report = finalReport
                // A terminal transition that had already been delivered
                // relayed the STOP REASON, not this report: `cancel` writes the
                // reason as the summary and resumes the waiter at once, while
                // this write only lands after the loop unwinds. Leaving the
                // unread flag set meant the richer partial report was never
                // relayed. Re-open it when there is something new to say; the
                // relay's cancelled rule still refuses to re-inject a bare stop
                // reason, so this cannot duplicate a delivery.
                let partial = finalReport.summary.trimmingCharacters(in: .whitespacesAndNewlines)
                let delivered = finalSummary.trimmingCharacters(in: .whitespacesAndNewlines)
                if isSummaryRelayed, !partial.isEmpty, partial != delivered {
                    isSummaryRelayed = false
                }
            }
            persistRecord()
            return
        }
        report = finalReport
        usage = outcome.usage
        lastError = outcome.error
        // The loop's transcript is authoritative: it records thinking, text and
        // every tool result in emission order, including a withheld call the
        // step list never sees. The live stream is replaced by it at the
        // boundary so the persisted run and the drawer cannot disagree.
        runTranscript = outcome.transcript
        var summary = outcome.report?.summary ?? ""
        // A failed run's report carries no summary; the failure reason is the
        // report. Falling back only when the report is ABSENT hid every failure
        // behind "finished without a report".
        if summary.isEmpty { summary = outcome.error ?? "" }
        if summary.isEmpty {
            summary = "The run finished without a report."
        }
        summary += "\n\n—\nBackend: \(plan.backendDescriptor.displayName) · "
            + "\(outcome.usage.steps) step(s) · \(outcome.usage.completionTokens) completion token(s)"
        if !summary.isEmpty, outcome.status == .completed {
            appendTranscriptStep(
                actionName: "assistant",
                arguments: [:],
                output: outcome.report?.summary ?? "",
                duration: 0,
                isError: false
            )
        }

        let finalStatus: SubagentStatus
        switch outcome.status {
        case .completed: finalStatus = .completed
        case .cancelled: finalStatus = .cancelled
        case .blocked: finalStatus = .blocked
        default: finalStatus = .failed
        }
        let liveStatus = outcome.error
            ?? String(format: "Completed in %.2fs", duration)
        finishRun(summary: summary, status: finalStatus, duration: duration, liveStatus: liveStatus)
    }

    /// Reads an isolated run's worktree into its report. A worktree that
    /// cannot be opened leaves the run's report intact and adds an open
    /// question naming why: the loop's transcript is still a truthful record
    /// of what happened. A CANCELLED run keeps its worktree (the readback's
    /// own detached task means cancellation cannot reach this function, so
    /// the status alone decides) — the operator clears it from the inspector
    /// or a later apply lands whatever it holds.
    private func finalizeWorktree(plan: AgentSpawnPlan, outcome: AgentLoopOutcome) async -> AgentReport? {
        guard case .worktree(let path, let branch) = plan.isolation else {
            return outcome.report
        }
        if outcome.status == .cancelled {
            // The worktree is the only full copy of the run's unapplied work.
            // Deleting it here is what made a Stop destroy work the operator
            // never saw; the record keeps naming the tree so the inspector
            // can offer the operator the keep-or-clear decision.
            return outcome.report
        }
        guard var report = outcome.report else { return nil }
        do {
            let worktree = try await WorkspaceIsolation.open(
                path: path,
                branch: branch,
                repositoryRoot: plan.repositoryRoot,
                baseCommit: plan.baseCommit
            )
            let changes = try await WorkspaceIsolation.snapshot(worktree)
            report.changedFiles = changes.changedFiles
            report.diff = changes.diff.isEmpty ? nil : changes.diff
            if changes.diffWasTruncated {
                report.openQuestions.append(
                    "The worktree diff was too large to store in full. apply_subagent_changes still applies the complete patch."
                )
            }
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            report.openQuestions.append("The run's worktree could not be read back: \(detail)")
        }
        return report
    }

    /// A run that never reached its loop still owns a worktree. Called from the
    /// scheduler/context failure paths so a refused run does not leave a branch
    /// and a directory behind. The discard runs on its own task because the
    /// failure path can be a CANCELLED run (stopped while it waited for
    /// admission), and git must not be spawned from a cancelled task. The
    /// repository root and base commit travel with the plan so a submodule or
    /// `--separate-git-dir` checkout is still found.
    ///
    /// Static because the spawn path calls it too: a resolve that returned
    /// after its turn was Stopped created the worktree for a run that will now
    /// never start.
    static func discardWorktree(plan: AgentSpawnPlan) async {
        await Task.detached(priority: .utility) {
            guard case .worktree(let path, let branch) = plan.isolation,
                  let worktree = try? await WorkspaceIsolation.open(
                      path: path,
                      branch: branch,
                      repositoryRoot: plan.repositoryRoot,
                      baseCommit: plan.baseCommit
                  ) else { return }
            await WorkspaceIsolation.cleanup(worktree)
        }.value
    }

    /// The loop's events, applied to the drawer's own state.
    ///
    /// `internal` rather than private so a test can drive the exact race it
    /// guards: a queued event that lands AFTER the terminal transition must not
    /// flap a finished run back to Running (and let the next persist overwrite
    /// the terminal record on disk with it).
    @MainActor
    func apply(_ event: AgentRunEvent) {
        switch event {
        case .status(let status, let detail):
            // A queued event can lag a terminal transition: a Stop cancels
            // mid-loop while the pump still holds an earlier `.running`, and
            // applying it would flap a finished run back to Running so the
            // next `persistRecord()` overwrote the terminal record on disk.
            // `finishRun` and `cancel` both latch `isFinished`; from that
            // moment the terminal state owns the drawer.
            lock.lock()
            let alreadyFinished = isFinished
            lock.unlock()
            guard !alreadyFinished else { return }
            liveStatusText = detail
            switch status {
            case .running:
                self.status = .running
            case .waitingForApproval:
                self.status = .waitingForApproval
            case .cancelled:
                self.status = .cancelled
            case .blocked:
                self.status = .blocked
            case .failed:
                self.status = .failed
            default:
                break
            }
        case .assistantDelta(let entryId, let text, let isThinking):
            if isThinking {
                if liveThinkingEntryId != entryId {
                    liveThinkingEntryId = entryId
                    liveThinkingText = ""
                }
                liveThinkingText += text
            } else {
                if liveTextEntryId != entryId {
                    liveTextEntryId = entryId
                    liveAssistantText = ""
                }
                liveAssistantText += text
            }
        case .toolStarted(let record):
            guard !hasFinished() else { return }
            liveStatusText = "Running \(record.name)…"
        case .toolFinished(let record):
            // A Stop latches `isFinished`; a tool event the pump still holds
            // must not append to the cancelled record and rewrite it on disk.
            guard !hasFinished() else { return }
            appendTranscriptStep(
                actionName: record.name,
                arguments: record.arguments,
                output: record.output ?? record.error ?? "",
                duration: record.executionDurationSeconds ?? 0,
                isError: record.status == .error || record.status == .rejected || record.status == .withheld
            )
            persistRecord()
        case .usage(let usage):
            // Applied even after a Stop: the cancelled run's partial spend is
            // persisted once the loop unwinds (see `runModelBackedExecution`).
            self.usage = usage
        case .completed(let report):
            guard !hasFinished() else { return }
            self.report = report
        }
    }

    private func appendTranscriptStep(
        actionName: String,
        arguments: [String: String],
        output: String,
        duration: Double,
        isError: Bool
    ) {
        let nextIndex = currentStepIndex + 1
        let step = SubagentStepRecord(
            stepIndex: nextIndex,
            timestamp: Date(),
            actionName: actionName,
            arguments: arguments,
            output: output,
            durationSeconds: duration,
            isError: isError
        )
        currentStepIndex = nextIndex
        transcript.append(step)
        liveStatusText = "Executed \(actionName) [Step \(nextIndex)]"
    }

    /// Terminal bookkeeping shared by both execution paths. Resumes every
    /// waiter exactly once, so a coordinator that spawned this run synchronously
    /// always receives a report (invariant 11).
    private func finishRun(
        summary: String,
        status: SubagentStatus,
        duration: Double,
        liveStatus: String
    ) {
        // A cancel can land while the execution task is unwinding. Whichever
        // transition reaches this point FIRST owns the record; a later one must
        // not overwrite a terminal summary (or delete an applied worktree).
        // `isFinished` is the ownership flag, not `status`: the event pump also
        // publishes a terminal status just before this runs, and treating that
        // as "already finished" left every waiter parked forever.
        lock.lock()
        guard !isFinished else {
            lock.unlock()
            return
        }
        isFinished = true
        let continuations = completionContinuations
        completionContinuations.removeAll()
        lock.unlock()

        self.status = status
        completedAt = Date()
        executionDurationSeconds = duration
        finalSummary = summary
        liveStatusText = liveStatus

        // A steering note can land after the loop's last drain and before this
        // transition. `send_subagent_message` reported it delivered, so it must
        // not vanish: record the fact that it arrived too late to be consumed.
        for note in drainUnconsumedMessages() {
            // The note is coordinator- or operator-authored text that lands in
            // a prompt-shaped transcript. It is appended RAW here while every
            // other inbound string goes through the marker pass, so a chat
            // template marker in it could forge a boundary the run reads back.
            let sender = AgentHarness.truncateText(AgentHarness.sanitizeText(note.sender), limit: 80)
            let content = AgentHarness.truncateText(AgentHarness.sanitizeText(note.content), limit: 2_000)
            let text = "Steering from \(sender) arrived after the run's final step, too late to be incorporated: \(content)"
            transcript.append(SubagentStepRecord(
                stepIndex: currentStepIndex,
                actionName: "steering",
                arguments: ["sender": sender],
                output: text,
                isError: true
            ))
            // The model-backed record is persisted from `runTranscript`, not
            // from the live step list, so the note belongs there too or the
            // stored run would still read as if nothing had arrived.
            if !runTranscript.isEmpty {
                runTranscript.append(AgentTranscriptEntry(
                    index: runTranscript.count,
                    stepIndex: currentStepIndex,
                    kind: .status,
                    text: text
                ))
            }
        }

        for continuation in continuations {
            continuation.resume(returning: summary)
        }
        persistRecord()
        if notifiesOnFinish {
            SubagentNotifier.runFinished(
                runId: id,
                role: role,
                status: status,
                parentSessionId: parentSessionId
            )
        }
    }

    // MARK: - Specialized Autonomous Pipelines

    /// Pipeline for Codebase Researcher archetype:
    /// Executes hybrid local RAG search, extracts matching AST files, inspects declarations, and aggregates a report.
    private func executeResearchPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Querying semantic codebase index..."
        }

        let query = taskDescription
        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let stepStart = CFAbsoluteTimeGetCurrent()
        let searchResults = await CodebaseIndexer.shared.search(
            query: query,
            workspace: wd,
            topK: 5,
            mode: .hybrid
        )
        let searchDuration = CFAbsoluteTimeGetCurrent() - stepStart

        var resultsList: [String] = []
        for (idx, r) in searchResults.enumerated() {
            resultsList.append("[\(idx + 1)] \(r.chunk.filePath):\(r.chunk.startLine)-\(r.chunk.endLine) (score: \(String(format: "%.3f", r.score)))\n\(r.chunk.title)")
        }
        let searchOutput = resultsList.isEmpty ? "No direct semantic matches found. Performed fallback directory scan." : resultsList.joined(separator: "\n\n")

        await recordStep(
            actionName: "codebase_search",
            arguments: ["query": query, "top_k": "5", "mode": "hybrid"],
            output: searchOutput,
            duration: searchDuration
        )

        if Task.isCancelled { return "Research cancelled." }

        // Step 2: Read the top matched files with REAL file_read executions (not index cache).
        let executor = SubagentToolExecutor(owner: self, workingDirectory: workingDirectory)
        var fileSnippets: [String] = []
        if executor.canUse("file_read") {
            let topChunks = searchResults.prefix(3)
            for chunkRes in topChunks {
                if Task.isCancelled { break }
                let c = chunkRes.chunk
                let read = await executor.runTool(
                    named: "file_read",
                    arguments: ["path": c.filePath, "start_line": c.startLine, "end_line": c.endLine]
                )
                if let content = SubagentToolExecutor.dataField(from: read.json, "content"), !content.isEmpty {
                    let snippet = "### `\(c.filePath)` (Lines \(c.startLine)-\(c.endLine))\n```swift\n\(String(content.prefix(2200)))\n```"
                    fileSnippets.append(snippet)
                } else {
                    // Fall back to the indexed chunk content (real indexed material).
                    let snippet = "### `\(c.filePath)` (Lines \(c.startLine)-\(c.endLine))\n```swift\n\(String(c.content.prefix(2200)))\n```"
                    fileSnippets.append(snippet)
                }
            }
        } else {
            // file_read not permitted: present the indexed chunk content directly (still real).
            for chunkRes in searchResults.prefix(3) {
                let c = chunkRes.chunk
                fileSnippets.append("### `\(c.filePath)` (Lines \(c.startLine)-\(c.endLine))\n```swift\n\(String(c.content.prefix(2200)))\n```")
            }
        }

        // Step 3: Fold in coordinator directives (e.g. "also check src/Foo.swift").
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            let paths = executor.resolveExistingFiles(SubagentToolExecutor.extractFilePaths(from: msg.content))
            for file in paths.prefix(2) where executor.canUse("file_read") {
                let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                if let content = SubagentToolExecutor.dataField(from: read.json, "content") {
                    fileSnippets.append("### Coordinator-directed: `\(file.path)`\n```swift\n\(String(content.prefix(2200)))\n```")
                    directiveNotes.append("Directive \"\(msg.content)\": read \(file.lastPathComponent).")
                }
            }
            if paths.isEmpty {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no file targets derivable).")
            }
        }

        // Synthesize final research report
        var report = "## Codebase Research Report: \(role)\n\n"
        report += "**Objective**: \(taskDescription)\n\n"
        if !searchResults.isEmpty {
            report += "### Key Locations Found (\(searchResults.count) matches):\n"
            for r in searchResults {
                report += "- **`\(r.chunk.filePath)`** (L\(r.chunk.startLine)-L\(r.chunk.endLine)): \(r.chunk.title) `[Score: \(String(format: "%.2f", r.score))]`\n"
            }
            report += "\n"
        }

        if !fileSnippets.isEmpty {
            report += "### Relevant Code Sections (real file reads):\n\n"
            report += fileSnippets.joined(separator: "\n\n")
            report += "\n\n"
        } else {
            report += "_No matching indexed code chunks found._\n\n"
        }

        if !directiveNotes.isEmpty {
            report += "### Coordinator Directives:\n"
            report += directiveNotes.map { "- \($0)" }.joined(separator: "\n")
            report += "\n\n"
        }

        report += "### Summary:\nCompleted real codebase search and file reads across project files without polluting the coordinator context window."
        return report
    }

    /// Pipeline for Test Runner archetype:
    /// Runs project tests (or an explicit command from the task), captures output,
    /// extracts failures, and produces a structured test result. Coordinator directives
    /// containing commands are executed and folded into the report.
    private func executeTestRunnerPipeline() async -> String {
        // Plan mode's promise covers what it delegates. This is the one pipeline
        // that lifts a shell command out of free text and runs it through
        // `AgentHarness.runProcess` directly, so `SubagentToolExecutor`'s
        // allow-list never sees the call; the parent's own mode is therefore the
        // gate. Without it a read-only turn could run arbitrary commands by
        // spawning a "Test Runner" and putting them in the task description.
        guard canRunShellCommands else {
            let note = "Test execution is withheld: this subagent was spawned from a Plan Mode turn, which cannot run shell commands. Investigate with the read tools instead."
            await recordStep(
                actionName: "shell_run",
                arguments: ["command": "<withheld in plan mode>"],
                output: note,
                duration: 0,
                isError: true
            )
            return "## Test Runner Report: \(role)\n\n**Status**: withheld in plan mode\n\n\(note)"
        }

        await MainActor.run {
            self.liveStatusText = "Executing unit tests via local runner..."
        }

        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        var testCmd = SubagentToolExecutor.extractShellCommand(from: taskDescription)
            ?? SubagentToolExecutor.extractShellCommand(from: contextSummary ?? "")
        if testCmd == nil {
            let lowerDesc = taskDescription.lowercased()
            if lowerDesc.contains("xcodebuild") || (lowerDesc.contains("test") && FileManager.default.fileExists(atPath: wd.appendingPathComponent("DynaMoE.xcodeproj").path)) {
                testCmd = "xcodebuild test -project DynaMoE/DynaMoE.xcodeproj -scheme DynaMoE -destination 'platform=macOS' | grep -E 'Test Case|\\*\\* TEST|failed' | head -n 40"
            } else if lowerDesc.contains("swift test") || lowerDesc.contains("test") || lowerDesc.contains("build") {
                testCmd = "swift test 2>&1 | tail -n 30"
            } else {
                testCmd = "swift test 2>&1 | tail -n 30"
            }
        }

        func runShell(_ cmd: String) async -> (Int32, String, String, Double) {
            let start = CFAbsoluteTimeGetCurrent()
            let (exitCode, stdout, stderr) = (try? await AgentHarness.runProcess(
                executableURL: URL(fileURLWithPath: "/bin/zsh"),
                arguments: ["-c", cmd],
                currentDirectory: wd,
                timeoutSeconds: 120.0
            )) ?? (-1, "", "Failed to spawn test runner process.")
            let duration = CFAbsoluteTimeGetCurrent() - start
            await recordStep(
                actionName: "shell_run",
                arguments: ["command": cmd],
                output: String((stdout.isEmpty ? stderr : stdout).prefix(3000)),
                duration: duration,
                isError: exitCode != 0
            )
            return (exitCode, stdout, stderr, duration)
        }

        let resolvedCommand = testCmd ?? "swift test 2>&1 | tail -n 30"
        let (exitCode, stdout, stderr, cmdDuration) = await runShell(resolvedCommand)
        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeLocalFileText(stdout), limit: 3000)
        let isSuccess = (exitCode == 0)

        var report = "## Test Runner Report: \(role)\n\n"
        report += "**Command Executed**: `\(resolvedCommand)`\n"
        report += "**Status**: \(isSuccess ? "✅ Passed (Exit Code 0)" : "❌ Failed (Exit Code \(exitCode))")\n"
        report += "**Duration**: \(String(format: "%.2f", cmdDuration))s\n\n"
        report += "### Output:\n```\n\(cleanOut.isEmpty ? stderr : cleanOut)\n```\n"

        // Fold in coordinator directive commands (e.g. "run `swift build --verbose`").
        // Re-check Plan Mode on every directive: if the operator switched to
        // Plan Mode while this Test Runner was already running, queued
        // directives must not become an unaudited shell channel.
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            if let directiveCmd = SubagentToolExecutor.extractShellCommand(from: msg.content) {
                let inPlanMode = await MainActor.run { AgentHarness.shared.activeMode == .plan }
                guard canRunShellCommands && !inPlanMode else {
                    let note = "Directive command `\(directiveCmd)` withheld: Plan Mode is active and cannot execute shell commands."
                    await recordStep(
                        actionName: "shell_run",
                        arguments: ["command": "<withheld in plan mode>"],
                        output: note,
                        duration: 0,
                        isError: true
                    )
                    directiveNotes.append(note)
                    continue
                }
                let (dExit, dOut, dErr, _) = await runShell(directiveCmd)
                directiveNotes.append("Directive command `\(directiveCmd)` → exit code \(dExit):")
                directiveNotes.append("```\n\(String((dOut.isEmpty ? dErr : dOut).prefix(1500)))\n```")
            } else {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no command derivable).")
            }
        }
        if !directiveNotes.isEmpty {
            report += "### Coordinator Directives:\n" + directiveNotes.joined(separator: "\n") + "\n"
        }

        report += "\n**Result**: \(isSuccess ? "All checks passed." : "Command reported failures above — see output for specifics.")"
        return report
    }

    /// Pipeline for Shader Optimizer archetype:
    /// Locates shader kernels, inspects real kernel source, proposes optimizations.
    private func executeShaderOptimizerPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Analyzing Metal shader pipelines and kernels..."
        }

        let wd = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let stepStart = CFAbsoluteTimeGetCurrent()

        let searchRes = await CodebaseIndexer.shared.search(
            query: taskDescription,
            workspace: wd,
            fileExtensions: ["metal", "h"],
            topK: 4,
            mode: .hybrid
        )

        await recordStep(
            actionName: "codebase_search",
            arguments: ["query": taskDescription, "file_extensions": "metal, h"],
            output: "Identified \(searchRes.count) shader kernels / buffers",
            duration: CFAbsoluteTimeGetCurrent() - stepStart
        )

        // Read the located kernels with REAL file_read executions.
        let executor = SubagentToolExecutor(owner: self, workingDirectory: workingDirectory)
        var kernelSections: [String] = []
        if executor.canUse("file_read") {
            for r in searchRes.prefix(3) {
                if Task.isCancelled { break }
                let read = await executor.runTool(
                    named: "file_read",
                    arguments: ["path": r.chunk.filePath, "start_line": r.chunk.startLine, "end_line": r.chunk.endLine]
                )
                if let content = SubagentToolExecutor.dataField(from: read.json, "content") {
                    kernelSections.append("**`\(r.chunk.filePath)`** (L\(r.chunk.startLine)-L\(r.chunk.endLine)):\n```metal\n\(String(content.prefix(2000)))\n```")
                }
            }
        }

        // Fold in coordinator directives pointing at specific kernels.
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            let paths = executor.resolveExistingFiles(SubagentToolExecutor.extractFilePaths(from: msg.content))
            for file in paths.prefix(2) where executor.canUse("file_read") {
                let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                if let content = SubagentToolExecutor.dataField(from: read.json, "content") {
                    kernelSections.append("**Coordinator-directed `\(file.path)`**:\n```swift\n\(String(content.prefix(2000)))\n```")
                    directiveNotes.append("Directive \"\(msg.content)\": read \(file.lastPathComponent).")
                }
            }
            if paths.isEmpty {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no kernel file targets derivable).")
            }
        }

        var report = "## Shader Optimization Analysis: \(role)\n\n"
        report += "**Target Objective**: \(taskDescription)\n\n"
        report += "### Key Kernel Findings:\n"
        for r in searchRes {
            report += "- **`\(r.chunk.filePath)`** (L\(r.chunk.startLine)-L\(r.chunk.endLine)): \(r.chunk.title)\n"
        }

        if !kernelSections.isEmpty {
            report += "\n### Kernel Source (real reads):\n\n"
            report += kernelSections.joined(separator: "\n\n")
            report += "\n"
        }

        if !directiveNotes.isEmpty {
            report += "\n### Coordinator Directives:\n"
            report += directiveNotes.map { "- \($0)" }.joined(separator: "\n")
            report += "\n"
        }

        report += "\n### Recommendations (grounded in the kernel source above):\n"
        report += "1. **SIMD Vectorization**: Align memory accesses to `float4` boundaries for Apple Silicon unified memory.\n"
        report += "2. **Threadgroup Sizing**: Ensure threads per threadgroup is a multiple of 32 (Apple GPU execution width).\n"
        report += "3. **Avoid Bank Conflicts**: Use threadgroup memory for intermediate reduction passes.\n"
        return report
    }

    /// Pipeline for Generic/Custom archetypes (e.g. Document Summarizer):
    /// Executes REAL tools grounded in the task text — reads files referenced in the task,
    /// falls back to workspace discovery, honors web-search intents, and folds coordinator
    /// directives in between steps. The report is composed entirely from genuine tool outputs
    /// so the coordinator LLM can synthesize a true answer from real material.
    private func executeGenericAgentPipeline() async -> String {
        await MainActor.run {
            self.liveStatusText = "Executing delegated task with real tools..."
        }

        let executor = SubagentToolExecutor(owner: self, workingDirectory: workingDirectory)
        var report = "## Subagent Execution Report: \(role)\n\n"
        report += "**Task**: \(taskDescription)\n\n"
        if let ctx = contextSummary, !ctx.isEmpty {
            report += "**Context**: \(ctx)\n\n"
        }
        report += "### Real Tool Activity:\n"

        // 1. Read any files explicitly referenced in the task/context.
        var sections: [String] = []
        var existingFiles = executor.resolveExistingFiles(
            SubagentToolExecutor.extractFilePaths(from: taskDescription)
            + SubagentToolExecutor.extractFilePaths(from: contextSummary ?? "")
        )

        if existingFiles.isEmpty {
            // No explicit files: ground the task in the workspace via a real discovery pass,
            // then read any document-like files the scan surfaces.
            if executor.canUse("find_files") {
                let probe = await executor.runTool(named: "find_files", arguments: ["pattern": "*", "max_depth": 2])
                if let matchesJSON = probe.json.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: matchesJSON) as? [String: Any],
                   let result = obj["result"] as? [String: Any],
                   let matches = result["matches"] as? [String] {
                    let docExtensions = ["txt", "md", "markdown", "rtf", "pdf", "csv", "json", "html"]
                    let discovered = matches
                        .filter { docExtensions.contains($0.split(separator: ".").last.map(String.init)?.lowercased() ?? "") }
                        .prefix(3)
                        .compactMap { URL(fileURLWithPath: $0) }
                    existingFiles = executor.resolveExistingFiles(discovered.map { $0.path })
                    await MainActor.run { self.liveStatusText = "Workspace scan found \(matches.count) entries" }
                }
            }
        }

        if executor.canUse("file_read") {
            for file in existingFiles.prefix(5) {
                if Task.isCancelled { break }
                let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                if let content = SubagentToolExecutor.dataField(from: read.json, "content"),
                   let totalLines = SubagentToolExecutor.dataField(from: read.json, "total_lines") {
                    let profile = SubagentToolExecutor.documentProfile(
                        path: file.path,
                        totalLines: Int(totalLines) ?? 0,
                        rawContent: content
                    )
                    sections.append(profile)
                    await MainActor.run {
                        self.liveStatusText = "Read \(file.lastPathComponent)"
                    }
                }
            }
        } else if !existingFiles.isEmpty {
            report += "- `file_read` is not in this subagent's allowed_tools; cannot read the referenced files.\n"
        }

        // 2. Honor web/search intents in the task text.
        let lowerTask = taskDescription.lowercased()
        if existingFiles.isEmpty && executor.canUse("web_search")
            && (lowerTask.contains("web") || lowerTask.contains("online") || lowerTask.contains("research") || lowerTask.contains("search")) {
            let query = SubagentToolExecutor.extractSearchQuery(from: taskDescription)
            if !query.isEmpty {
                let search = await executor.runTool(named: "web_search", arguments: ["query": query])
                if let results = SubagentToolExecutor.dataField(from: search.json, "results") {
                    sections.append("**Web Search** (query: \(query)):\n\(String(results.prefix(2500)))")
                }
            }
        }

        // 3. Fold in coordinator directives that arrived while running.
        var directiveNotes: [String] = []
        for msg in drainUnconsumedMessages() {
            let msgPaths = executor.resolveExistingFiles(SubagentToolExecutor.extractFilePaths(from: msg.content))
            var handled = false
            for file in msgPaths.prefix(3) {
                if executor.canUse("file_read") {
                    let read = await executor.runTool(named: "file_read", arguments: ["path": file.path])
                    if let content = SubagentToolExecutor.dataField(from: read.json, "content"),
                       let totalLines = SubagentToolExecutor.dataField(from: read.json, "total_lines") {
                        sections.append(SubagentToolExecutor.documentProfile(
                            path: file.path,
                            totalLines: Int(totalLines) ?? 0,
                            rawContent: content
                        ))
                        directiveNotes.append("Directive \"\(msg.content)\": read \(file.lastPathComponent).")
                        handled = true
                    }
                }
            }
            if !handled {
                directiveNotes.append("Directive \"\(msg.content)\": acknowledged (no additional tool actions derived).")
            }
        }

        // 4. Compose the grounded report.
        report += "- Files read: \(existingFiles.prefix(5).map { $0.lastPathComponent }.joined(separator: ", "))\n"
        let executedTools = transcript.map { $0.actionName }
        report += "- Transcript steps recorded: \(transcript.count) [\(executedTools.joined(separator: ", "))]\n\n"

        if !sections.isEmpty {
            report += "### Grounded Findings (from real tool executions):\n\n"
            report += sections.joined(separator: "\n\n")
            report += "\n\n"
        }

        if !directiveNotes.isEmpty {
            report += "### Coordinator Directives:\n"
            report += directiveNotes.map { "- \($0)" }.joined(separator: "\n")
            report += "\n\n"
        }

        if sections.isEmpty && transcript.isEmpty {
            report += "### Result:\n"
            report += "No actionable file paths, search intents, or tool targets were derivable from the task description. "
            report += "Re-spawn with explicit file paths, URLs, or a search query in `task_description` to ground this subagent's work."
        } else {
            report += "### Result:\n"
            report += "All findings above come from genuine tool executions captured in this subagent's transcript. "
            report += "Present this material to the user as the delegated outcome."
        }
        return report
    }

    // MARK: - Step Recording Helper

    func recordStep(
        actionName: String,
        arguments: [String: String],
        output: String,
        duration: Double,
        isError: Bool = false
    ) async {
        let nextIndex = currentStepIndex + 1
        let step = SubagentStepRecord(
            stepIndex: nextIndex,
            timestamp: Date(),
            actionName: actionName,
            arguments: arguments,
            output: output,
            durationSeconds: duration,
            isError: isError
        )
        await MainActor.run {
            self.currentStepIndex = nextIndex
            self.transcript.append(step)
            self.liveStatusText = "Executed \(actionName) [Step \(nextIndex)]"
        }
    }

    // MARK: - Coordinator Directive Draining

    /// Returns coordinator/user messages not yet consumed by the running pipeline.
    /// Pipelines call this between execution steps so mid-run directives
    /// (e.g. "focus on this file", "also check X") are folded into the report.
    func drainUnconsumedMessages() -> [SubagentMessage] {
        lock.lock()
        defer { lock.unlock() }
        let fresh = messageStore.filter { !consumedMessageIds.contains($0.id) }
        for m in fresh {
            consumedMessageIds.insert(m.id)
        }
        return fresh
    }

    /// The same drain for the model-backed loop, which runs outside this
    /// instance's task: `send_subagent_message` and the inspector's message
    /// box report delivery, and before this hook the loop never read the
    /// queue, so a steering note was recorded and silently ignored.
    func drainUnconsumedSteering() -> [(sender: String, content: String)] {
        drainUnconsumedMessages().map { ($0.sender, $0.content) }
    }
}

// MARK: - Subagent Tool Executor

/// Executes REAL AgentHarness tools on behalf of a subagent pipeline, enforcing the
/// subagent's `allowedTools` whitelist and recording genuine tool outputs as transcript
/// steps. This is what makes subagent reports grounded: every finding traces back to an
/// actual file read, codebase search, shell run, or web query — not a fabricated summary.
final class SubagentToolExecutor {
    private unowned let owner: SubagentInstance
    private let allowedTools: Set<String>
    private let parentMode: AgentMode
    private let workingDirectory: URL
    private let stepOutputLimit = 6000

    init(owner: SubagentInstance, workingDirectory: URL?) {
        self.owner = owner
        let declared = owner.allowedTools.isEmpty
            ? ["file_read", "find_files", "grep_search", "codebase_search", "web_search", "web_fetch"]
            : owner.allowedTools
        // A read-only parent's whitelist is intersected with the plan-mode
        // allow-list, so a subagent cannot be handed a write tool the
        // coordinator itself is forbidden to call.
        self.parentMode = owner.parentMode
        self.allowedTools = Set(
            owner.parentMode == .plan
                ? declared.filter { !PlanModePolicy.withholds($0) }
                : declared
        )
        self.workingDirectory = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    func canUse(_ toolName: String) -> Bool {
        allowedTools.contains(toolName) && AgentHarness.shared.tools[toolName] != nil
    }

    /// What this subagent can actually call: the declared whitelist intersected with
    /// the tools the harness has installed. The whitelist on its own is NOT the
    /// available set — it may name tools that were never registered, and it omits
    /// installed tools outside the whitelist — so it must not be reported as
    /// "Available tools".
    var callableToolNames: [String] {
        allowedTools.filter { AgentHarness.shared.tools[$0] != nil }.sorted()
    }

    /// Runs a real tool and records it as a transcript step. Returns the tool's JSON
    /// result plus its human-readable output for report building.
    func runTool(named name: String, arguments: [String: Any]) async -> (json: String, readable: String) {
        guard AgentHarness.shared.tools[name] != nil else {
            let callable = callableToolNames
            let listed = callable.isEmpty ? "none — this subagent's whitelist names no installed tool" : callable.joined(separator: ", ")
            let err = "Unknown tool '\(name)'. Allowed tools: \(listed)."
            return (AgentHarness.toolErrorJSON(tool: name, error: err), err)
        }
        guard allowedTools.contains(name) else {
            let err = "Tool '\(name)' is not in this subagent's allowed_tools (\(allowedTools.sorted().joined(separator: ", ")))."
            return (AgentHarness.toolErrorJSON(tool: name, error: err), err)
        }

        // A subagent runs without a human in the loop, so a call the harness
        // would gate on approval cannot be honoured here. `subq_decide` is the
        // one tool whose documentation promises that gate (it uploads local
        // image bytes), so it is refused rather than run unapproved.
        //
        // The destination is pinned ONCE, here, and handed to both the gate and
        // the call. Reading `UserDefaults` twice let a `defaults write` land
        // between them, and reading the raw turbo flag let Turbo Mode skip the
        // one prompt that guards the credential — the exact bypass
        // `testTurboModeCannotBypassACustomSubQEndpoint` forbids in the main
        // loop, so the subagent path must ask the same question the same way.
        let subQEndpoint = name == "subq_decide" ? SubQDecisionSettings.endpoint() : nil
        if name == "subq_decide",
           AgentHarness.requiresApproval(
               toolName: name,
               arguments: arguments,
               mode: parentMode,
               subQEndpoint: subQEndpoint
           ),
           !(TurboModeSettings.isEnabled
             && AgentHarness.turboModeMayBypassApproval(
                 toolName: name,
                 arguments: arguments,
                 mode: parentMode,
                 subQEndpoint: subQEndpoint
             )) {
            let err = "Tool '\(name)' needs the user's approval before it uploads the state or local image bytes. A subagent cannot ask for it; return the finding to the coordinator and let the main agent make that call."
            return (AgentHarness.toolErrorJSON(tool: name, error: err), err)
        }

        let strArgs = arguments.mapValues { String(describing: $0) }
        let rawArgs = (try? JSONSerialization.data(withJSONObject: strArgs, options: [.prettyPrinted]))
            .flatMap { String(data: $0, encoding: .utf8) } ?? ""
        let call = ParsedToolCall(name: name, arguments: arguments, rawArguments: rawArgs, rawText: name)

        let exec = await AgentHarness.shared.executeTool(
            call: call,
            workingDirectory: workingDirectory,
            maxOutputLength: stepOutputLimit,
            mode: parentMode,
            turnId: owner.parentTurnId,
            sessionId: owner.parentSessionId,
            approvedSubQEndpoint: subQEndpoint
        )

        let readable = exec.record.output ?? exec.record.error ?? exec.resultJSON
        await owner.recordStep(
            actionName: name,
            arguments: strArgs,
            output: String(readable.prefix(stepOutputLimit)),
            duration: exec.record.executionDurationSeconds ?? 0,
            isError: exec.record.status == .error
        )
        return (exec.resultJSON, readable)
    }

    /// Parses a tool result JSON and pulls a string field out of it (when present).
    static func dataField(from json: String, _ key: String) -> String? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let result = obj["result"] as? [String: Any] else { return nil }
        if let s = result[key] as? String, !s.isEmpty { return s }
        if let n = result[key] as? NSNumber { return n.stringValue }
        return nil
    }

    /// Extracts a runnable shell command from free text: backticked spans first,
    /// then text after an explicit "run " / "execute " imperative.
    static func extractShellCommand(from text: String) -> String? {
        let backtickRegex = try? NSRegularExpression(pattern: "`([^`\\n]+)`", options: [])
        let ns = text as NSString
        for m in backtickRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) ?? [] {
            let cmd = ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            if !cmd.isEmpty { return cmd }
        }
        if let r = text.range(of: "(?i)\\b(?:run|execute|sh|bash)\\b[: ]+([^\\n]+)", options: .regularExpression) {
            var cmd = text[r.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
            if let colon = cmd.firstIndex(of: ":") { cmd = String(cmd[cmd.index(after: colon)...]).trimmingCharacters(in: .whitespaces) }
            cmd = cmd.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
            if !cmd.isEmpty && cmd.count < 400 { return cmd }
        }
        return nil
    }

    /// Extracts plausible file paths from free text (absolute, home-relative, or
    /// extension-bearing names resolved against the working directory).
    static func extractFilePaths(from text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var candidates: [String] = []

        // Absolute & home-relative paths: /Users/.../Begin.txt, ~/Desktop/notes.md
        let pathRegex = try? NSRegularExpression(pattern: "(?:[~/]/?|[ \"]|^)((?:/[A-Za-z0-9._@\\-]+)+\\.[A-Za-z0-9]{1,6})", options: [])
        let ns = text as NSString
        for m in pathRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) ?? [] {
            var candidate = ns.substring(with: m.range(at: 1))
            if !candidate.hasPrefix("/") && !candidate.hasPrefix("~") {
                // Include the character(s) captured before the path (~/ or ./)
                let full = ns.substring(with: m.range)
                candidate = full.trimmingCharacters(in: .whitespaces)
            }
            candidate = candidate.trimmingCharacters(in: CharacterSet(charactersIn: "\",.;:"))
            if !candidate.isEmpty { candidates.append(candidate) }
        }

        // Bare relative filenames with known extensions (resolve against working directory)
        let nameRegex = try? NSRegularExpression(pattern: "(?<![\\w/])[\\w][\\w.\\-]*\\.(txt|md|markdown|swift|metal|h|m|c|cpp|hpp|py|rs|go|js|ts|json|xml|yaml|yml|html|css|csv|log|rtf|pdf|sh|toml)\\b", options: [.caseInsensitive])
        for m in nameRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) ?? [] {
            let candidate = ns.substring(with: m.range)
            candidates.append(candidate)
        }

        // Dedupe while preserving order
        var seen = Set<String>()
        var out: [String] = []
        for c in candidates {
            let cleaned = c.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
            if cleaned.isEmpty || seen.contains(cleaned) { continue }
            seen.insert(cleaned)
            out.append(cleaned)
        }
        return out
    }

    /// Resolves candidate path strings to existing file URLs (tilde + relative aware).
    func resolveExistingFiles(_ candidates: [String]) -> [URL] {
        var seen = Set<String>()
        var out: [URL] = []
        for c in candidates {
            let url = AgentHarness.resolvePath(c, workingDirectory: workingDirectory)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), !isDir.boolValue else { continue }
            guard !seen.contains(url.path) else { continue }
            seen.insert(url.path)
            out.append(url)
        }
        return out
    }

    /// Derives a web-search query from imperative task text.
    static func extractSearchQuery(from text: String) -> String {
        var q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let imperativePatterns = [
            "(?i)^please\\s+", "(?i)^can you\\s+", "(?i)^could you\\s+",
            "(?i)search( the web| online)?( for| about)?\\s*:?", "(?i)look ?up\\s+:?",
            "(?i)find (out )?(information|info)? ?(about|on|for)?\\s*:?\\s*",
            "(?i)research\\s+:?", "(?i)fetch( and summarize)?\\s+:?"
        ]
        for p in imperativePatterns {
            q = q.replacingOccurrences(of: p, with: "", options: .regularExpression)
        }
        return q
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Builds a grounded document profile (stats, headings, excerpt) from a real file read.
    static func documentProfile(path: String, totalLines: Int, rawContent: String) -> String {
        // Strip FileReadTool line-number prefix ("12: content") when present.
        let numbered = rawContent.components(separatedBy: "\n")
        let stripped = numbered.map { line -> String in
            if let r = line.range(of: "^\\d{1,6}: ", options: .regularExpression) {
                return String(line[r.upperBound...])
            }
            return line
        }
        let fullText = stripped.joined(separator: "\n")
        let wordCount = fullText.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.count
        let charCount = fullText.count

        // Detect headings: markdown hashes, ALL-CAPS short lines, or numbered section titles
        var headings: [String] = []
        for line in stripped.prefix(400) {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { continue }
            if t.hasPrefix("#") {
                headings.append(t)
            } else if t.count < 80 && t == t.uppercased() && t.rangeOfCharacter(from: .alphanumerics) != nil && !t.hasSuffix(".") {
                headings.append(t)
            }
            if headings.count >= 20 { break }
        }

        let excerptLimit = 1800
        let head = String(fullText.prefix(excerptLimit))
        let tail = fullText.count > excerptLimit + 500 ? "\n… [middle omitted] …\n" + String(fullText.suffix(400)) : ""
        let excerpt = head + tail

        var profile = "**File**: \(path)\n"
        profile += "**Size**: \(totalLines) lines, ~\(wordCount) words, \(charCount) characters\n"
        if !headings.isEmpty {
            profile += "**Structure**:\n" + headings.map { "- \($0)" }.joined(separator: "\n") + "\n"
        }
        profile += "**Content**:\n```text\n\(excerpt)\n```"
        return profile
    }
}

// MARK: - Subagent Manager Coordinator

public final class SubagentManager: ObservableObject {
    public static let shared = SubagentManager()

    @Published public var subagents: [SubagentInstance] = []
    @Published public var selectedSubagentId: UUID? = nil
    @Published public var isDrawerOpen: Bool = false

    private var subagentsById: [UUID: SubagentInstance] = [:]
    private var spawnsSincePrune = 0
    private var subagentsOrder: [SubagentInstance] = []
    /// How many FINISHED instances stay in memory. The drawer only needs the
    /// recent ones; older terminal runs are restored from their durable records
    /// on demand. Without a bound, a long session kept every run it ever
    /// spawned - each one an ObservableObject holding its full transcript.
    private let maximumRetainedTerminalInstances = 50
    /// Live instances are ObservableObjects on their own, but the inspector
    /// observes the MANAGER. Forwarding their changes keeps the list, the live
    /// stream and the new usage strip refreshing together; without it a
    /// `@Published` mutation on a run would not redraw anything that holds a
    /// value snapshot of it.
    private var instanceObservers: [UUID: AnyCancellable] = [:]
    private let lock = NSLock()

    public var activeSubagentsCount: Int {
        lock.lock()
        defer { lock.unlock() }
        // `waitingForApproval` is ACTIVE: a run parked on a card still holds
        // the engine and its scheduler ticket. Asking only about running and
        // pending made the coordinator end its turn exactly when a run was
        // waiting on a decision.
        return subagentsOrder.filter { !$0.status.isTerminal }.count
    }

    /// Count of active subagents belonging to a specific chat session.
    public func activeSubagentsCount(forSession sessionId: UUID?) -> Int {
        guard let sessionId else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        return subagentsOrder.filter {
            $0.parentSessionId == sessionId && !$0.status.isTerminal
        }.count
    }

    /// The sessions that still have a run working. Retention must not evict
    /// these: deleting the conversation deletes the run's record underneath a
    /// run that is still executing — invisible, unstoppable, its terminal write
    /// dropped by the store's tombstone.
    public var liveRunSessionIds: Set<UUID> {
        var ids: Set<UUID> = []
        lock.lock()
        defer { lock.unlock() }
        for instance in subagentsOrder where !instance.status.isTerminal {
            if let session = instance.parentSessionId {
                ids.insert(session)
            }
        }
        return ids
    }

    public var allSubagents: [SubagentInstance] {
        lock.lock()
        defer { lock.unlock() }
        return subagentsOrder
    }

    /// The runs ONE conversation owns.
    ///
    /// The registry is process-wide, but the coordinator's status tools are
    /// not: a chat that could list or read another chat's run learned its
    /// findings and, worse, marked its report relayed — which suppresses the
    /// owning conversation's own delivery. Nil is a session of its own (a run
    /// started by an API/agent turn with no chat), not a wildcard.
    public func runs(ownedBySession sessionId: UUID?) -> [SubagentInstance] {
        lock.lock()
        defer { lock.unlock() }
        return subagentsOrder.filter { $0.parentSessionId == sessionId }
    }

    private init() {
        // Retention runs once per process rather than on a timer: the store can
        // only grow while the app is running, and a launch sweep is the moment
        // a slow disk should be paid for. Off the main actor, because startup
        // must not parse a few hundred records before the first frame. Orphaned
        // worktrees are swept with the same sweep: a run that was interrupted
        // between creating one and applying it has no other owner.
        let policy = AgentRunStore.configuredPolicy()
        Task.detached(priority: .utility) {
            // A record left `.running` by a crash has no live instance to Stop
            // and no terminal status to clear. Nothing can be in flight before
            // this owner exists, so the launch sweep closes those records
            // first. "Is this run live" reads the registry and every instance's
            // `status` — main-actor state — and this sweep runs on a utility
            // thread, so the answer is SNAP-SHOTTED on the actor up front:
            // reading it synchronously per record was a data race in the
            // making. A run spawned after the snapshot is younger than the
            // grace window and is skipped on its dates, so the snapshot cannot
            // misclassify it.
            let liveRunIds = await MainActor.run {
                Set(SubagentManager.shared.allSubagents
                    .filter { !$0.status.isTerminal }
                    .map(\.id))
            }
            // The relay asks which restored records it still owes; building
            // that index FIRST keeps the first send from paying for the scan.
            // The scan is enqueued on the store's own IO queue, so everything
            // below waits behind it rather than racing it.
            AgentRunStore.shared.prepareRelayIndex()
            AgentRunStore.shared.reconcileInterruptedRuns(isLive: { liveRunIds.contains($0) })
            AgentRunStore.shared.prune(policy: policy)
            // Temp files an interrupted write left behind (a hard crash between
            // the create and the rename) are orphans nothing else looks at.
            AgentRunStore.shared.sweepOrphanedTempWrites()
            await WorkspaceIsolation.pruneStaleWorktrees()
        }
    }

    // MARK: - Spawning API

    /// Spawns a new subagent with the specified role, task, and parameters.
    @discardableResult
    public func spawn(
        role: String,
        taskDescription: String,
        allowedTools: [String] = [],
        parentMode: AgentMode = .code,
        contextSummary: String? = nil,
        parentSessionId: UUID? = nil,
        parentTurnId: UUID? = nil,
        workingDirectory: URL? = nil,
        notifiesOnFinish: Bool = true
    ) -> SubagentInstance {
        let instance = SubagentInstance(
            role: role,
            taskDescription: taskDescription,
            allowedTools: allowedTools,
            parentMode: parentMode,
            contextSummary: contextSummary,
            parentSessionId: parentSessionId,
            parentTurnId: parentTurnId,
            workingDirectory: workingDirectory
        )
        instance.notifiesOnFinish = notifiesOnFinish

        register(instance)
        instance.startExecution()
        return instance
    }

    /// Spawns a run the model actually works, on the backend the plan resolved.
    ///
    /// The legacy `spawn` above stays for the canned pipelines (and for the
    /// tests that pin them). This path is what `spawn_subagent` uses when a
    /// backend is available: same manager, same drawer, same status tools, but
    /// the work is a real agent loop. `notifiesOnFinish` is decided HERE,
    /// before execution starts: the caller used to set it after `start`,
    /// and a run that finished inside that window posted its banner for a
    /// synchronous spawn the operator was already watching.
    @discardableResult
    func spawnModelBacked(
        _ plan: AgentSpawnPlan,
        id: UUID = UUID(),
        notifiesOnFinish: Bool = true
    ) -> SubagentInstance {
        let instance = SubagentInstance(id: id, plan: plan)
        instance.notifiesOnFinish = notifiesOnFinish
        register(instance)
        instance.startExecution()
        return instance
    }

    /// A launched instance is also the moment growth is worth re-bounding: a
    /// long session can spawn more runs than the launch sweep ever sees.
    private func noteSpawnForRetention() {
        spawnsSincePrune += 1
        guard spawnsSincePrune >= 10 else { return }
        spawnsSincePrune = 0
        let policy = AgentRunStore.configuredPolicy()
        Task.detached(priority: .utility) {
            AgentRunStore.shared.prune(policy: policy)
        }
    }

    private func register(_ instance: SubagentInstance) {
        noteSpawnForRetention()
        lock.lock()
        subagentsById[instance.id] = instance
        subagentsOrder.insert(instance, at: 0)
        lock.unlock()

        let observer = instance.objectWillChange
            .throttle(for: .milliseconds(80), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] _ in
                Task { @MainActor in self?.objectWillChange.send() }
            }
        lock.lock()
        instanceObservers[instance.id] = observer
        lock.unlock()

        if Thread.isMainThread {
            self.subagents.insert(instance, at: 0)
            if self.selectedSubagentId == nil {
                self.selectedSubagentId = instance.id
            }
        } else {
            DispatchQueue.main.async {
                self.subagents.insert(instance, at: 0)
                if self.selectedSubagentId == nil {
                    self.selectedSubagentId = instance.id
                }
            }
        }
        trimRetainedTerminalInstances()
    }

    /// Drops the oldest FINISHED instances past the in-memory bound. Their
    /// records stay on disk, which is what the inspector and the report relay
    /// read once the object is gone.
    private func trimRetainedTerminalInstances() {
        lock.lock()
        var finishedSeen = 0
        var removed: [UUID] = []
        subagentsOrder.removeAll { instance in
            guard instance.status.isTerminal else { return false }
            finishedSeen += 1
            guard finishedSeen > maximumRetainedTerminalInstances else { return false }
            removed.append(instance.id)
            return true
        }
        for id in removed {
            subagentsById.removeValue(forKey: id)
            instanceObservers.removeValue(forKey: id)?.cancel()
        }
        lock.unlock()
        guard !removed.isEmpty else { return }
        let ids = Set(removed)
        if Thread.isMainThread {
            subagents.removeAll { ids.contains($0.id) }
            if let selected = selectedSubagentId, ids.contains(selected) {
                selectedSubagentId = subagents.first?.id
            }
        } else {
            DispatchQueue.main.async {
                self.subagents.removeAll { ids.contains($0.id) }
                if let selected = self.selectedSubagentId, ids.contains(selected) {
                    self.selectedSubagentId = self.subagents.first?.id
                }
            }
        }
    }

    // MARK: - Lookup & Management

    public func getSubagent(byId id: UUID) -> SubagentInstance? {
        lock.lock()
        defer { lock.unlock() }
        return subagentsById[id] ?? subagents.first(where: { $0.id == id })
    }

    /// The approval center's own hooks. Resolved through the live instance so
    /// a restored record (which has no loop to park) is left alone.
    func noteApprovalWait(for request: AgentApprovalRequest) {
        getSubagent(byId: request.runId)?.noteApprovalWait(toolName: request.toolName)
    }

    func noteApprovalResolved(runId: UUID) {
        getSubagent(byId: runId)?.noteApprovalResolved()
    }

    public func cancel(byId id: UUID) {
        if let subagent = getSubagent(byId: id) {
            subagent.cancel()
        }
    }

    public func cancelAll() {
        lock.lock()
        let list = subagentsOrder
        lock.unlock()
        for s in list {
            s.cancel()
        }
        Task { await AgentScheduler.shared.cancelAll(reason: "All subagent runs were cancelled.") }
    }

    /// Stops every run spawned by one coordinator session.
    ///
    /// A Stop is the operator abandoning the turn, not just its current step:
    /// leaving a child working on it would spend tokens on work nobody asked
    /// for and could let it write after the conversation moved on.
    public func cancelSession(sessionId: UUID, reason: String = "The coordinator turn was stopped.") {
        let children = allSubagents.filter {
            $0.parentSessionId == sessionId && !$0.status.isTerminal
        }
        for child in children {
            child.cancel(reason: reason)
        }
        Task { await AgentScheduler.shared.cancelRuns(parentSessionId: sessionId, reason: reason) }
    }

    /// Live runs a Stop is about to cancel.
    ///
    /// The approval store needs the ids BEFORE the run tasks are cancelled:
    /// a card is resolved as declined in that step, instead of waiting for
    /// task cancellation to propagate through the run's task group. `nil`
    /// narrows nothing, which is only for a diagnostic caller.
    public func liveRunIds(parentTurnId: UUID? = nil, sessionId: UUID? = nil) -> [UUID] {
        lock.lock()
        defer { lock.unlock() }
        return subagentsOrder.filter { instance in
            guard !instance.status.isTerminal else { return false }
            if let parentTurnId { return instance.parentTurnId == parentTurnId }
            if let sessionId { return instance.parentSessionId == sessionId }
            return true
        }.map(\.id)
    }

    /// Stops only the runs the given coordinator TURN spawned.
    ///
    /// The session-wide cancel above reaches runs spawned by EARLIER turns of
    /// the same conversation, destroying background work the stopped turn never
    /// owned — and a cancelled run's unapplied worktree is the only copy of
    /// its work. The Stop button cancels by turn id so a background run
    /// launched in an earlier turn survives a Stop and keeps its worktree.
    public func cancelTurnRuns(parentTurnId: UUID, reason: String = "The coordinator turn was stopped.") {
        let children = allSubagents.filter {
            $0.parentTurnId == parentTurnId && !$0.status.isTerminal
        }
        for child in children {
            child.cancel(reason: reason)
        }
        Task { await AgentScheduler.shared.cancelRuns(parentTurnId: parentTurnId, reason: reason) }
    }

    /// Removes finished runs from the drawer.
    ///
    /// With a `sessionId` the matching durable records go too: the inspector
    /// lists stored runs beside live ones, so clearing only the in-memory half
    /// would leave a "cleared" list that immediately refills from disk. The
    /// no-argument form keeps the old behaviour for callers that predate the
    /// store.
    ///
    /// An unapplied isolated run is NEVER cleared here, on either path: its
    /// record is the only thing keeping its worktree out of the launch sweep
    /// (`pruneStaleWorktrees` protects a tree while a decodable record says
    /// unapplied), so deleting it would silently destroy the run's only copy
    /// of its work two weeks later — the same disposal `prune` refuses to
    /// make. Apply the run, or clear it per-run from the inspector, first.
    public func clearCompleted(sessionId: UUID? = nil) {
        let terminal: [SubagentInstance]
        lock.lock()
        terminal = subagentsOrder.filter {
            $0.status.isTerminal
                && (sessionId == nil || $0.parentSessionId == sessionId || $0.parentSessionId == nil)
                && !$0.holdsUnappliedWorktreeWork
        }
        let clearedIds = Set(terminal.map(\.id))
        let liveIds = Set(subagentsOrder.map(\.id))
        subagentsOrder.removeAll(where: { clearedIds.contains($0.id) })
        for id in clearedIds {
            subagentsById.removeValue(forKey: id)
            instanceObservers.removeValue(forKey: id)?.cancel()
        }
        lock.unlock()

        for instance in terminal {
            AgentRunStore.shared.delete(runId: instance.id, sessionId: instance.parentSessionId)
        }

        // The inspector lists TERMINAL records whether or not a live instance
        // exists, and a record is only removed from disk when it is cleared
        // here. Clearing just the instances left every restored, failed or
        // interrupted run behind, so the list refilled from the store as soon
        // as it reloaded. A non-terminal record with NO live instance is the
        // same kind of leftover (a crash before `reconcileInterruptedRuns`
        // runs), and nothing can update it, so it is clearable too. Unscoped
        // records ride every inspector, so they are cleared from any
        // conversation.
        //
        // The scan runs on the store's IO queue rather than synchronously here:
        // decoding a conversation's history on the main actor made the button
        // a visible stall. A reload enqueued after this call still sees every
        // delete, because the store's queue is FIFO.
        AgentRunStore.shared.deleteClearableRecords(sessionId: sessionId) { record in
            !clearedIds.contains(record.id)
                && (record.status.isTerminal || !liveIds.contains(record.id))
                && !record.holdsWorktreeWork
        }

        // Synchronous: the manager is main-actor state, and a caller that
        // clears and then asks for the run must not still find it through the
        // published array while an async removal is still queued.
        subagents.removeAll(where: { clearedIds.contains($0.id) })
        if let sel = selectedSubagentId, !subagents.contains(where: { $0.id == sel }) {
            selectedSubagentId = subagents.first?.id
        }
    }

    /// Deletes ONE finished run, including the worktree of an unapplied
    /// isolated run.
    ///
    /// `clearCompleted` deliberately refuses to discard an unapplied worktree
    /// in bulk, because the record is the only thing keeping the tree out of
    /// the launch sweep. Without this per-run decision, a writer the operator
    /// decided not to land was permanent: on disk, in every inspector, and
    /// protected from retention forever. The discard is explicit and
    /// destructive, and its card says so.
    @discardableResult
    public func discard(runId: UUID, sessionId: UUID? = nil) async -> Bool {
        lock.lock()
        let instance = subagentsById[runId]
        lock.unlock()
        // A live run is stopped, not discarded: deleting its record underneath
        // it would drop its terminal write and leave an unstoppable worker.
        // Also wait until execution has unwound (the same gate Apply uses) so
        // Stop→Discard cannot `git worktree remove` under a still-exiting child.
        guard instance == nil || (instance?.status.isTerminal == true && instance?.isExecutionUnwound == true) else { return false }
        // An apply owns the record and the worktree while it runs. Discarding
        // mid-apply could delete the record after the patch landed (changes
        // with nothing naming them) or race cleanup against git.
        guard !SubagentApplyService.isRunInFlight(runId) else { return false }
        guard let record = instance?.snapshotRecord()
            ?? AgentRunStore.shared.load(runId: runId, sessionId: sessionId)
            ?? AgentRunStore.shared.loadAny(runId: runId),
            record.status.isTerminal else { return false }
        // A discard is rare and a wrong race loses landed work, so while ANY
        // checkout is mid-apply the answer is "not now": a record's stored
        // `repositoryRoot` can be stale, which is exactly when a narrower check
        // would miss the claim it was meant to find.
        guard !SubagentApplyService.hasAnyRepositoryInFlight else { return false }

        // The worktree belongs to this run alone, and it is the only copy of
        // work a patch cannot carry. Remove it BEFORE the record: if removal
        // fails, the record must stay so the tree is still reachable and the
        // next launch sweep still protects it, and the caller gets `false`
        // instead of a row that silently vanished over a tree that did not.
        if case .worktree(let path, let branch) = record.isolation,
           record.holdsWorktreeWork,
           FileManager.default.fileExists(atPath: path) {
            let repositoryRoot = record.repositoryRoot.map { URL(fileURLWithPath: $0) }
            let baseCommit = record.baseCommit ?? ""
            let removed = await Task.detached(priority: .utility) { () -> Bool in
                let worktree: AgentWorktree
                if let opened = try? await WorkspaceIsolation.open(
                    path: path,
                    branch: branch,
                    repositoryRoot: repositoryRoot,
                    baseCommit: baseCommit
                ) {
                    worktree = opened
                } else if let repositoryRoot {
                    // Fall back to the recorded metadata when `open` refuses a
                    // missing or compromised `.git` pointer so discarding the
                    // run still cleans up the directory and branch instead of
                    // stranding them until the 14-day prune.
                    worktree = AgentWorktree(
                        runId: runId,
                        repositoryRoot: repositoryRoot,
                        path: URL(fileURLWithPath: path, isDirectory: true),
                        branch: branch,
                        baseCommit: baseCommit
                    )
                } else {
                    return false
                }
                return await WorkspaceIsolation.cleanup(worktree)
            }.value
            guard removed else {
                print("⚠️ [SubagentManager] kept run \(runId.uuidString): its worktree could not be removed")
                return false
            }
        }

        lock.lock()
        subagentsById.removeValue(forKey: runId)
        instanceObservers.removeValue(forKey: runId)?.cancel()
        subagentsOrder.removeAll { $0.id == runId }
        lock.unlock()
        subagents.removeAll { $0.id == runId }
        if selectedSubagentId == runId {
            selectedSubagentId = subagents.first?.id
        }

        AgentRunStore.shared.deleteImmediately(runId: runId, sessionId: record.parentSessionId)
        return true
    }

    public func sendMessage(toSubagentId id: UUID, sender: String = "user", content: String) {
        if let subagent = getSubagent(byId: id) {
            subagent.addMessage(sender: sender, content: content)
        }
    }
}
