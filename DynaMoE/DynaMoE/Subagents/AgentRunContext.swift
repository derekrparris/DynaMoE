//
//  AgentRunContext.swift
//  DynaMoE
//
//  Per-run tool scope, mode, workspace and approval policy. See SUBAGENTS.md §7.
//

import Foundation

final class AgentRunContext {
    nonisolated let runId: UUID
    /// The run's own id, used as its backend session. Not the conversation.
    nonisolated let sessionId: UUID
    /// The coordinator turn/session that spawned this run, kept for the record
    /// and for a delegation record that has to name its parent. Deliberately
    /// NOT what tools execute under: a run is its own session, and handing a
    /// tool the coordinator's identity would let it read and write the
    /// coordinator's session-scoped state (todos, notes, the session log)
    /// through a run that is supposed to be isolated from it.
    nonisolated let parentTurnId: UUID?
    nonisolated let parentSessionId: UUID?
    nonisolated let allowedTools: Set<String>
    nonisolated let mode: AgentMode
    nonisolated let workingDirectory: URL
    nonisolated let approvalPolicy: AgentApprovalPolicy
    let harness: AgentHarness

    init(
        runId: UUID,
        sessionId: UUID,
        parentTurnId: UUID? = nil,
        parentSessionId: UUID? = nil,
        allowedTools: Set<String>,
        mode: AgentMode,
        workingDirectory: URL,
        approvalPolicy: AgentApprovalPolicy,
        harness: AgentHarness
    ) {
        self.runId = runId
        self.sessionId = sessionId
        self.parentTurnId = parentTurnId
        self.parentSessionId = parentSessionId
        self.allowedTools = allowedTools
        self.mode = mode
        self.workingDirectory = workingDirectory
        self.approvalPolicy = approvalPolicy
        self.harness = harness
    }

    static func make(
        runId: UUID,
        sessionId: UUID,
        parentTurnId: UUID? = nil,
        parentSessionId: UUID? = nil,
        toolNames: [String],
        workingDirectory: URL,
        mode: AgentMode,
        approvalPolicy: AgentApprovalPolicy,
        workspaceIsolated: Bool,
        workspaceRepositoryRoot: URL? = nil,
        workspaceBranch: String? = nil,
        approvalHandler: (@Sendable (AgentApprovalRequest) async -> Bool)?
    ) throws -> AgentRunContext {
        let harness = AgentHarness.makeRunScoped()
        try harness.configureRunScope(
            runId: runId,
            toolNames: toolNames,
            workingDirectory: workingDirectory,
            mode: mode,
            approvalPolicy: approvalPolicy,
            workspaceIsolated: workspaceIsolated,
            workspaceRepositoryRoot: workspaceRepositoryRoot,
            workspaceBranch: workspaceBranch,
            approvalHandler: approvalHandler
        )
        return AgentRunContext(
            runId: runId,
            sessionId: sessionId,
            parentTurnId: parentTurnId,
            parentSessionId: parentSessionId,
            allowedTools: Set(toolNames),
            mode: mode,
            workingDirectory: workingDirectory,
            approvalPolicy: approvalPolicy,
            harness: harness
        )
    }

    func toolDefinitions() -> [ToolDefinition] {
        harness.availableToolDefinitions(mode: mode)
    }

    func workerSystemPrompt(
        role: String,
        baseSystem: String,
        isLingModel: Bool = false,
        isGemma4Model: Bool = false,
        isDeepSeekV41Model: Bool = false,
        isInkling2Model: Bool = false
    ) -> String {
        harness.buildWorkerSystemPrompt(
            baseSystem: baseSystem,
            role: role,
            mode: mode,
            isLingModel: isLingModel,
            isGemma4Model: isGemma4Model,
            isDeepSeekV41Model: isDeepSeekV41Model,
            isInkling2Model: isInkling2Model
        )
    }

    func execute(_ call: AgentToolCall) async -> (resultJSON: String, record: ToolCallRecord, isCompleted: Bool) {
        // A run stopped between the model's call and this execution must not
        // perform it. The cancellation lands here as a cancelled task, and the
        // honest answer is a refused call the transcript records — not work
        // done for a run the operator has already abandoned.
        if Task.isCancelled {
            let error = "This run was stopped before '\(call.name)' could run."
            let json = AgentHarness.toolErrorJSON(tool: call.name, error: error)
            let record = ToolCallRecord(
                name: call.name,
                arguments: call.arguments.mapValues { String(describing: $0) },
                rawArguments: call.rawArguments,
                // WITHHELD, not `.rejected`: nothing ran and the user declined
                // nothing. `.rejected` counts as a failure AND still runs
                // `recordToolCallFingerprint`, which CLEARS the repeat counts of
                // every mutating tool — so a Stop landing here reset the
                // repeated-call guard on the way out.
                status: .withheld,
                output: AgentHarness.renderToolResultForModel(json),
                error: error
            )
            return (json, record, false)
        }
        // The run's OWN identity, not the coordinator's: `sessionId` is the run
        // id (see the property above), and no turn id is passed because the run
        // has no coordinator turn of its own to stamp. Nested spawn is
        // forbidden, so nothing here needs the parent's identity.
        return await harness.executeTool(
            call: call.parsedCall,
            workingDirectory: workingDirectory,
            maxOutputLength: nil,
            mode: mode,
            sessionId: sessionId
        )
    }

    func noteToolOutcome(_ record: ToolCallRecord, call: AgentToolCall) {
        harness.recordToolCallOutcome(toolName: record.name, outcome: record.status.toolCallOutcome)
        if record.status == .withheld || record.status == .rejected {
            // Neither ran, so neither can have mutated state or invalidated an
            // earlier read; count as a repeat without clearing any fingerprints.
            _ = harness.recordWithheldToolCall(toolName: record.name, arguments: call.arguments)
        } else {
            _ = harness.recordToolCallFingerprint(toolName: record.name, arguments: call.arguments)
        }
    }

    func trippedGuard() -> String? {
        if harness.consecutiveFailedToolCalls >= harness.failedToolCallLimit {
            return "\(harness.consecutiveFailedToolCalls) consecutive tool failures (limit \(harness.failedToolCallLimit))"
        }
        if harness.totalFailedToolCalls >= harness.failedToolCallTotalLimit {
            return "\(harness.totalFailedToolCalls) total tool failures (limit \(harness.failedToolCallTotalLimit))"
        }
        if harness.maxRepeatedToolCallCount >= harness.repeatedToolCallLimit {
            return "the same tool call repeated \(harness.maxRepeatedToolCallCount) times (limit \(harness.repeatedToolCallLimit))"
        }
        return nil
    }
}
