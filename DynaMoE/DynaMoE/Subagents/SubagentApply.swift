//
//  SubagentApply.swift
//  DynaMoE
//
//  Moves an isolated run's work onto the main tree. See SUBAGENTS.md §7 and §13
//  (Phase 4). The coordinator reaches this through apply_subagent_changes.
//
//  Applying is a separate, approved step rather than the end of a run: a writer
//  working in its own worktree cannot disturb the operator's tree, and the
//  coordinator decides when (and whether) the diff lands. Changes are applied
//  unstaged, so the coordinator's own commit tooling stays in charge of history.
//

import Foundation

nonisolated enum SubagentApplyError: LocalizedError, Equatable {
    case runNotFound(UUID)
    case notIsolated(String)
    case stillRunning(String)
    case worktreeGone(String)

    var errorDescription: String? {
        switch self {
        case .runNotFound(let id):
            return "No subagent run with id \(id.uuidString) was found."
        case .notIsolated(let role):
            return "'\(role)' did not run in an isolated worktree, so its changes are either already in the working tree or there are none. Only a run spawned with isolation \"worktree\" can be applied."
        case .stillRunning(let role):
            return "'\(role)' is still running. Wait for it to finish, then apply its changes."
        case .worktreeGone(let detail):
            return detail
        }
    }
}

nonisolated struct SubagentApplyOutcome: Sendable {
    var runId: UUID
    var role: String
    var branch: String
    var changedFiles: [String]
    var appliedAt: Date
    var alreadyApplied: Bool
}

@MainActor
enum SubagentApplyService {
    static func apply(runId: UUID) async throws -> SubagentApplyOutcome {
        let live = SubagentManager.shared.getSubagent(byId: runId)
        if let live, !live.status.isTerminal {
            throw SubagentApplyError.stillRunning(live.role)
        }
        let record = live?.snapshotRecord() ?? AgentRunStore.shared.loadAny(runId: runId)
        guard let record else { throw SubagentApplyError.runNotFound(runId) }
        guard case .worktree(let path, let branch) = record.isolation else {
            throw SubagentApplyError.notIsolated(record.role)
        }
        if let appliedAt = record.appliedAt {
            return SubagentApplyOutcome(
                runId: runId,
                role: record.role,
                branch: branch,
                changedFiles: record.report?.changedFiles ?? [],
                appliedAt: appliedAt,
                alreadyApplied: true
            )
        }

        let worktree: AgentWorktree
        do {
            worktree = try await WorkspaceIsolation.open(path: path, branch: branch)
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            throw SubagentApplyError.worktreeGone(detail)
        }

        let changes = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
        let files: [String]
        if changes.isEmpty {
            files = []
        } else {
            let result = try await WorkspaceIsolation.apply(worktree, changeSet: changes)
            files = result.changedFiles
        }

        let appliedAt = Date()
        if let live {
            live.markApplied(at: appliedAt, files: files)
        } else {
            var updated = record
            updated.appliedAt = appliedAt
            if var report = updated.report, !files.isEmpty {
                report.changedFiles = files
                updated.report = report
            }
            AgentRunStore.shared.saveImmediately(updated)
        }
        await WorkspaceIsolation.cleanup(worktree)

        return SubagentApplyOutcome(
            runId: runId,
            role: record.role,
            branch: branch,
            changedFiles: files,
            appliedAt: appliedAt,
            alreadyApplied: false
        )
    }
}
