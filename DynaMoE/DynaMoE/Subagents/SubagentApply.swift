//
//  SubagentApply.swift
//  DynaMoE
//
//  Moves an isolated run's work onto the main tree. See SUBAGENTS.md §7 and §13
//  (Phases 4 and 5). The coordinator reaches this through
//  `apply_subagent_changes`; the inspector's Apply Changes button reaches the
//  same service through `SubagentApplyControl`, behind the same approval store.
//
//  Applying is a separate, approved step rather than the end of a run: a writer
//  working in its own worktree cannot disturb the operator's tree, and the
//  coordinator (or the operator) decides when the diff lands. Changes are
//  applied unstaged, so the coordinator's own commit tooling stays in charge of
//  history.
//

import Foundation

nonisolated enum SubagentApplyError: LocalizedError, Equatable {
    case runNotFound(UUID)
    case notIsolated(String)
    case stillRunning(String)
    case alreadyApplying(String)
    case repositoryBusy(String)
    case worktreeGone(String)

    var errorDescription: String? {
        switch self {
        case .runNotFound(let id):
            return "No subagent run with id \(id.uuidString) was found."
        case .notIsolated(let role):
            return "'\(role)' did not run in an isolated worktree, so its changes are either already in the working tree or there are none. Only a run spawned with isolation \"worktree\" can be applied."
        case .stillRunning(let role):
            return "'\(role)' is still running. Wait for it to finish, then apply its changes."
        case .alreadyApplying(let role):
            return "'\(role)' is already being applied. Wait for that apply to finish."
        case .repositoryBusy(let role):
            return "Another run from the same repository is being applied while '\(role)' is ready to land. Applying two patches to one tree at once can leave a mixed result; wait for that apply to finish."
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
    /// One apply per run at a time, and one per REPOSITORY.
    ///
    /// The `appliedAt` check below is the durable guard, but it is written only
    /// after the patch lands; the coordinator's tool and the inspector's button
    /// can otherwise reach `snapshot` together and both try to write the same
    /// patch. Two writers race here, so the guard has to exist before the first
    /// `await`, not after it.
    ///
    /// The repository claim covers two DIFFERENT runs of one checkout: their
    /// `git apply --check` calls can interleave with each other's writes, so one
    /// runs against a tree the other has already started changing. It is taken
    /// after the worktree is resolved and before the snapshot, in the same
    /// MainActor turn that checks it, so no second apply can interleave.
    private static var inFlight: Set<UUID> = []
    private static var repositoriesInFlight: Set<String> = []

    static func apply(runId: UUID) async throws -> SubagentApplyOutcome {
        guard !inFlight.contains(runId) else {
            let role = SubagentManager.shared.getSubagent(byId: runId)?.role ?? "This run"
            throw SubagentApplyError.alreadyApplying(role)
        }
        inFlight.insert(runId)
        defer { inFlight.remove(runId) }

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
            worktree = try await WorkspaceIsolation.open(
                path: path,
                branch: branch,
                repositoryRoot: record.repositoryRoot.map { URL(fileURLWithPath: $0, isDirectory: true) }
            )
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            throw SubagentApplyError.worktreeGone(detail)
        }

        let repositoryKey = worktree.repositoryRoot.standardizedFileURL.path
        guard !repositoriesInFlight.contains(repositoryKey) else {
            throw SubagentApplyError.repositoryBusy(record.role)
        }
        repositoriesInFlight.insert(repositoryKey)
        defer { repositoriesInFlight.remove(repositoryKey) }

        let changes = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
        let files: [String]
        if changes.isEmpty {
            // `snapshot` can only see what git does. A run whose work is all in
            // ignored files looks empty, and cleaning up after "no changes"
            // would destroy the only copy. Name them and keep the tree.
            let ignored = await WorkspaceIsolation.ignoredFiles(worktree)
            if !ignored.isEmpty {
                let named = ignored.prefix(5).joined(separator: ", ")
                throw WorkspaceIsolationError.applyFailed(
                    "the run's only changes are in git-ignored files (\(named)), which a patch cannot carry. Its worktree was kept so nothing is lost."
                )
            }
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

// MARK: - The operator's path

nonisolated enum SubagentApplyControlError: LocalizedError, Equatable {
    case runNotFound(UUID)
    case nothingToApply(String)
    case approvalDenied

    var errorDescription: String? {
        switch self {
        case .runNotFound(let id):
            return "No subagent run with id \(id.uuidString) was found."
        case .nothingToApply(let role):
            return "'\(role)' has no isolated worktree waiting to be applied."
        case .approvalDenied:
            return "The changes were not applied because the request was declined."
        }
    }
}

/// The inspector's Apply Changes button, which is the operator's own way to
/// finish a run the coordinator never landed. It raises the same approval
/// request shape as a tool call, through `SubagentApprovalCenter`, and hands
/// the work to `SubagentApplyService`; there is deliberately no second apply
/// implementation here, only a second caller.
@MainActor
enum SubagentApplyControl {
    typealias ApprovalHandler = (AgentApprovalRequest) async -> Bool

    /// Whether the detail pane should offer Apply at all: an isolated run that
    /// has stopped, has not landed yet, and still has its worktree on disk.
    static func canApply(_ record: AgentRunRecord) -> Bool {
        guard case .worktree(let path, _) = record.isolation else { return false }
        guard record.status.isTerminal, record.appliedAt == nil else { return false }
        return FileManager.default.fileExists(atPath: path)
    }

    /// The card the operator sees. `destination` is the main working tree the
    /// patch lands in, resolved by `apply` when the worktree still names it.
    static func applyRequest(
        for record: AgentRunRecord,
        destination: String? = nil
    ) -> AgentApprovalRequest? {
        guard canApply(record),
              case .worktree(let path, let branch) = record.isolation else { return nil }
        let changed = record.report?.changedFiles.count ?? 0
        let summary: String
        switch changed {
        case 0: summary = "this run's worktree changes"
        case 1: summary = "1 changed file"
        default: summary = "\(changed) changed files"
        }
        return AgentApprovalRequest(
            runId: record.id,
            toolName: "apply_subagent_changes",
            arguments: [
                "subagent_id": record.id.uuidString,
                "branch": branch,
                "worktree": path
            ],
            workingDirectory: destination,
            reason: "Apply \(summary) from \(branch) to the main working tree. Changes land unstaged."
        )
    }

    /// Raises the card, then applies. `requestApproval` is injectable so the
    /// tests can approve or deny without a window; production uses
    /// `SubagentApprovalCenter`, the same store every run-scoped approval goes
    /// through.
    static func apply(
        runId: UUID,
        requestApproval: ApprovalHandler? = nil
    ) async throws -> SubagentApplyOutcome {
        let live = SubagentManager.shared.getSubagent(byId: runId)
        guard let record = live?.snapshotRecord() ?? AgentRunStore.shared.loadAny(runId: runId) else {
            throw SubagentApplyControlError.runNotFound(runId)
        }
        guard canApply(record),
              case .worktree(let path, _) = record.isolation else {
            throw SubagentApplyControlError.nothingToApply(record.role)
        }

        var destination: String? = record.repositoryRoot
        if destination == nil,
           let repository = await WorkspaceIsolation.owningRepository(for: URL(fileURLWithPath: path)) {
            destination = repository.path
        }
        guard let request = applyRequest(for: record, destination: destination) else {
            throw SubagentApplyControlError.nothingToApply(record.role)
        }

        let ask = requestApproval ?? {
            // Landing a patch is never a Turbo bypass: the contract in
            // SUBAGENTS.md requires its own card.
            await SubagentApprovalCenter.shared.request($0, honorsTurbo: false)
        }
        guard await ask(request) else { throw SubagentApplyControlError.approvalDenied }
        return try await SubagentApplyService.apply(runId: runId)
    }
}
