//
//  AgentRunContext.swift
//  DynaMoE
//
//  Per-run tool scope, mode, workspace and approval policy. See SUBAGENTS.md §7.
//

import Foundation

final class AgentRunContext {
    nonisolated let runId: UUID
    nonisolated let sessionId: UUID
    nonisolated let allowedTools: Set<String>
    nonisolated let mode: AgentMode
    nonisolated let workingDirectory: URL
    nonisolated let approvalPolicy: AgentApprovalPolicy
    let harness: AgentHarness

    init(
        runId: UUID,
        sessionId: UUID,
        allowedTools: Set<String>,
        mode: AgentMode,
        workingDirectory: URL,
        approvalPolicy: AgentApprovalPolicy,
        harness: AgentHarness
    ) {
        self.runId = runId
        self.sessionId = sessionId
        self.allowedTools = allowedTools
        self.mode = mode
        self.workingDirectory = workingDirectory
        self.approvalPolicy = approvalPolicy
        self.harness = harness
    }

    static func make(
        runId: UUID,
        sessionId: UUID,
        toolNames: [String],
        workingDirectory: URL,
        mode: AgentMode,
        approvalPolicy: AgentApprovalPolicy,
        workspaceIsolated: Bool,
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
            approvalHandler: approvalHandler
        )
        return AgentRunContext(
            runId: runId,
            sessionId: sessionId,
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
        isDeepSeekV41Model: Bool = false
    ) -> String {
        harness.buildWorkerSystemPrompt(
            baseSystem: baseSystem,
            role: role,
            mode: mode,
            isLingModel: isLingModel,
            isGemma4Model: isGemma4Model,
            isDeepSeekV41Model: isDeepSeekV41Model
        )
    }

    func execute(_ call: AgentToolCall) async -> (resultJSON: String, record: ToolCallRecord, isCompleted: Bool) {
        await harness.executeTool(
            call: call.parsedCall,
            workingDirectory: workingDirectory,
            maxOutputLength: nil,
            mode: mode
        )
    }

    func noteToolOutcome(_ record: ToolCallRecord, call: AgentToolCall) {
        harness.recordToolCallOutcome(toolName: record.name, outcome: record.status.toolCallOutcome)
        if record.status == .withheld {
            // Counts as a repeat (so a refusal loop escalates) without the
            // count-clearing a real mutation would cause.
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
