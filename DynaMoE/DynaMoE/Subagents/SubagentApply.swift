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
    /// The branch the destination tree had checked out when the patch landed,
    /// or nil for a detached HEAD.
    var destinationBranch: String? = nil
    /// Ignored files still in the kept worktree after the tracked patch
    /// landed. A patch cannot carry them, so the tree stays until the
    /// operator retrieves or clears it.
    var keptIgnoredFiles: [String] = []
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

    /// Whether an apply currently owns a run or a checkout. `discard` consults
    /// these: deleting the record mid-apply can leave landed changes nothing
    /// names, or race the worktree's cleanup against its own git call.
    static func isRunInFlight(_ runId: UUID) -> Bool {
        inFlight.contains(runId)
    }

    static var hasAnyRepositoryInFlight: Bool {
        !repositoriesInFlight.isEmpty
    }

    /// The key one checkout is claimed under. Symlinks are resolved, so two
    /// spellings of it (a repo reached through /tmp against /private/tmp) claim
    /// the SAME entry instead of letting two applies interleave their writes in
    /// one tree.
    static func repositoryKey(_ repositoryRoot: URL) -> String {
        repositoryRoot.standardizedFileURL.resolvingSymlinksInPath().path
    }

#if DEBUG
    /// Tests only: the single-flight refusal cannot be made deterministic
    /// against a real apply — the in-flight window is the duration of one git
    /// call — so a test puts the run in flight by hand and asserts the refusal.
    static func beginApplyForTesting(_ runId: UUID) {
        inFlight.insert(runId)
    }

    static func endApplyForTesting(_ runId: UUID) {
        inFlight.remove(runId)
    }

    /// Tests only: the repository half of the same claim. Its window covers a
    /// DIFFERENT run's apply of the same checkout, which a single-run test
    /// cannot reach without interleaving two real applies.
    static func beginRepositoryApplyForTesting(_ repositoryRoot: URL) {
        repositoriesInFlight.insert(repositoryKey(repositoryRoot))
    }

    static func endRepositoryApplyForTesting(_ repositoryRoot: URL) {
        repositoriesInFlight.remove(repositoryKey(repositoryRoot))
    }
#endif

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
                repositoryRoot: record.repositoryRoot.map { URL(fileURLWithPath: $0, isDirectory: true) },
                baseCommit: record.baseCommit
            )
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            throw SubagentApplyError.worktreeGone(detail)
        }

        // Two runs of one checkout would otherwise interleave: each can be
        // between its `git apply --check` and its write when the other starts,
        // so one runs against a tree the other has already begun changing.
        let repositoryClaimKey = Self.repositoryKey(worktree.repositoryRoot)
        guard !repositoriesInFlight.contains(repositoryClaimKey) else {
            throw SubagentApplyError.repositoryBusy(record.role)
        }
        repositoriesInFlight.insert(repositoryClaimKey)
        defer { repositoriesInFlight.remove(repositoryClaimKey) }

        let changes = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
        // Files a patch cannot carry (git-ignored, e.g. a `.env.local` the run
        // wrote) are only ever in the worktree. When the run also changed
        // tracked files, the patch lands below and the tree used to be deleted
        // with them still in it — silently. They are named here and the
        // worktree is KEPT so the operator can retrieve them.
        var keptIgnoredFiles: [String] = []
        let files: [String]
        if changes.isEmpty {
            // An empty change set is not proof the tree is empty: a patch whose
            // text could not be decoded comes back empty too. Ask git directly
            // before the "no changes" path is allowed to delete the worktree,
            // because that directory is the only copy of an unapplied run.
            guard await WorkspaceIsolation.hasNoWorkingTreeChanges(worktree) else {
                throw WorkspaceIsolationError.applyFailed(
                    "git reports changes in the worktree that the snapshot could not express as a patch, so nothing was applied and the worktree was kept."
                )
            }
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
            // The tracked patch does not remove the run's ignored files; check
            // AFTER the apply so the names describe what is still in the tree.
            let ignored = await WorkspaceIsolation.ignoredFiles(worktree)
            if !ignored.isEmpty {
                keptIgnoredFiles = Array(ignored.prefix(10))
            }
        }

        // An apply lands on whatever the destination has CHECKED OUT, and the
        // record is the only place that answers "where did it go" afterwards.
        let destinationBranch = await WorkspaceIsolation.currentBranch(of: worktree.repositoryRoot)

        let appliedAt = Date()
        let recordUpdated: Bool
        if let live {
            recordUpdated = live.markApplied(at: appliedAt, files: files, appliedTo: destinationBranch)
        } else {
            // Mutated against the CURRENT record on the store's IO queue, not
            // restored from the copy this apply loaded seconds ago: a
            // `summaryRelayed` write landing mid-apply would otherwise be
            // reverted here and the report injected a second time.
            recordUpdated = AgentRunStore.shared.mutate(runId: runId, sessionId: record.parentSessionId) { current in
                current.appliedAt = appliedAt
                current.appliedToBranch = destinationBranch
                if var report = current.report, !files.isEmpty {
                    report.changedFiles = files
                    current.report = report
                }
            } != nil
        }
        // The patch has landed, but a record that still says "unapplied" with
        // no worktree is the one state the Apply UI can never resolve. Keep the
        // tree and say so instead of reporting a clean apply.
        guard recordUpdated else {
            throw WorkspaceIsolationError.applyFailed(
                "the patch landed, but this run's record could not be updated (its conversation may have been deleted). The worktree was kept so the landed changes still have a copy; clear the run manually."
            )
        }
        if keptIgnoredFiles.isEmpty {
            await WorkspaceIsolation.cleanup(worktree)
        } else {
            let named = keptIgnoredFiles.joined(separator: ", ")
            // The record must say why an applied run still has a worktree, or
            // the inspector's Apply reads as unfinished business.
            if let live {
                live.noteKeptIgnoredFiles(keptIgnoredFiles)
            } else {
                AgentRunStore.shared.mutate(runId: runId, sessionId: record.parentSessionId) { current in
                    current.appliedAt = appliedAt
                    current.appliedToBranch = destinationBranch
                    current.keptIgnoredFiles = keptIgnoredFiles
                    // A record without a report (an interrupted run, an older
                    // schema) still deserves the kept-worktree explanation.
                    if current.report == nil {
                        current.report = AgentReport(
                            status: current.status,
                            summary: "",
                            findings: [],
                            changedFiles: files,
                            diff: nil,
                            tests: [],
                            risks: [],
                            openQuestions: [],
                            usage: AgentUsage(),
                            backend: current.backend
                        )
                    }
                    if var report = current.report {
                        report.openQuestions.append(
                            "The tracked patch was applied, but the worktree was kept: it also holds files a patch cannot carry (\(named)). Retrieve them manually before clearing the run."
                        )
                        current.report = report
                    }
                }
            }
        }

        return SubagentApplyOutcome(
            runId: runId,
            role: record.role,
            branch: branch,
            changedFiles: files,
            appliedAt: appliedAt,
            alreadyApplied: false,
            destinationBranch: destinationBranch,
            keptIgnoredFiles: keptIgnoredFiles
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
    /// `diffStat` is `git diff --stat` for the worktree, so the decision is
    /// made against the files that will change rather than a count.
    static func applyRequest(
        for record: AgentRunRecord,
        destination: String? = nil,
        diffStat: String? = nil
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
        // `repositoryRoot` on a record can be stale (a moved checkout), and the
        // service falls back to git's own answer. Naming the DESTINATION on the
        // card keeps the approval attached to the tree the patch lands in.
        let destinationText = destination ?? "the main working tree (resolved when the apply runs)"
        var reason = "Apply \(summary) from \(branch) to \(destinationText). Changes land unstaged."
        if let diffStat, !diffStat.isEmpty {
            reason += "\n\n" + diffStat
        } else if let files = record.report?.changedFiles, !files.isEmpty {
            let named = files.prefix(10).joined(separator: "\n")
            let more = files.count > 10 ? "\n… and \(files.count - 10) more" : ""
            reason += "\n\n" + named + more
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
            reason: reason
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
              case .worktree(let path, let branch) = record.isolation else {
            throw SubagentApplyControlError.nothingToApply(record.role)
        }

        // Git's own answer about the worktree wins over the record's stored
        // root: a checkout that MOVED leaves `repositoryRoot` stale, and the
        // apply itself resolves the owning repository from the worktree. The
        // card must name the tree the patch will actually land in.
        var destination = await WorkspaceIsolation.owningRepository(
            for: URL(fileURLWithPath: path, isDirectory: true)
        )?.path
        if destination == nil { destination = record.repositoryRoot }
        // Read the stat from the worktree the apply reads, so the card lists
        // the changes rather than a count that can be stale.
        let stat = await WorkspaceIsolation.diffStat(
            AgentWorktree(
                runId: record.id,
                repositoryRoot: destination.map { URL(fileURLWithPath: $0, isDirectory: true) }
                    ?? URL(fileURLWithPath: path, isDirectory: true),
                path: URL(fileURLWithPath: path, isDirectory: true),
                branch: branch,
                baseCommit: record.baseCommit ?? ""
            )
        )
        guard let request = applyRequest(for: record, destination: destination, diffStat: stat) else {
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
