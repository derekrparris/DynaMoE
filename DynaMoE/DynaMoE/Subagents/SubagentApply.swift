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
    /// The run's branch holds commits a patch cannot carry, so cleanup kept
    /// it instead of deleting it. The worktree may still be removed; the
    /// branch is the preserved copy.
    var keptBranch: Bool = false
    /// Whether cleanup proved the worktree directory is gone. False means the
    /// tree survived (it is the only copy of ignored work), and the caller
    /// must say so rather than reporting a removal that did not happen.
    var worktreeRemoved: Bool = false
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
        // A Stop marks the record terminal at once, but the execution task may
        // still be unwinding — killing a `shell_run` child (up to its SIGKILL
        // grace) or writing one last tool result. Applying in that window
        // landed a worktree a live child was still editing, and the cleanup
        // that follows removed it under that child.
        if let live, !live.status.isTerminal || !live.isExecutionUnwound {
            throw SubagentApplyError.stillRunning(live.role)
        }
        let record = live?.snapshotRecord() ?? AgentRunStore.shared.loadAny(runId: runId)
        guard let record else { throw SubagentApplyError.runNotFound(runId) }
        // A RESTORED record has no live instance, so the check above cannot
        // see it. This is a belt-and-braces guard for a record the store could
        // still report non-terminal: at launch `reconcileInterruptedRuns`
        // rewrites an interrupted run as `.failed` (terminal) precisely so the
        // inspector can clear it, so that case reaches the apply as FAILED and
        // the operator's card is the decision. Nothing here can distinguish
        // "half-written by a crash" from "failed after doing real work"; the
        // card and the kept worktree are what make the choice explicit.
        guard record.status.isTerminal else {
            throw SubagentApplyError.stillRunning(record.role)
        }
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
            // A clean tree is not proof the work is applied. A run that
            // COMMITTED and then checked the base branch out has no working
            // changes at all, and cleanup's `branch -D` would leave those
            // commits only in the reflog. The patch cannot carry them, so the
            // tree is kept and the operator decides how to land the branch.
            let ahead = await WorkspaceIsolation.commitsAheadOfBase(worktree)
            if ahead != 0 {
                let detail: String
                if let ahead, ahead > 0 {
                    detail = "its branch '\(branch)' holds \(ahead) commit\(ahead == 1 ? "" : "s") that a patch cannot carry"
                } else {
                    detail = "its branch '\(branch)' cannot be compared with a recorded base commit, so git cannot prove the branch is empty"
                }
                throw WorkspaceIsolationError.applyFailed(
                    "the run's working tree is clean, but \(detail). Nothing was applied and the worktree was kept, so no commit is deleted with the branch."
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
            recordUpdated = await live.markApplied(at: appliedAt, files: files, appliedTo: destinationBranch)
        } else {
            // Mutated against the CURRENT record on the store's IO queue, not
            // restored from the copy this apply loaded seconds ago: a
            // `summaryRelayed` write landing mid-apply would otherwise be
            // reverted here and the report injected a second time.
            recordUpdated = await AgentRunStore.shared.mutateAsync(runId: runId, sessionId: record.parentSessionId) { current in
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
        // `snapshot` diffs `baseCommit` against the working tree, so commits on
        // the worktree's branch that are reachable from `HEAD` are already in
        // the patch we just applied. Keep the branch ONLY if it holds commits
        // outside `HEAD` (e.g. the run committed on its branch and then
        // switched away) or git could not prove it does not (`nil`).
        let keptBranch = await WorkspaceIsolation.commitsOnBranchOutsideHead(worktree) != 0
        let worktreeRemoved: Bool
        if keptIgnoredFiles.isEmpty {
            // The branch is kept only if it still holds uncarried commits;
            // otherwise both worktree and branch are removed cleanly. The
            // result is the tree's own answer, not a promise: a locked file
            // leaves it in place and the tool says so.
            worktreeRemoved = await WorkspaceIsolation.cleanup(worktree, deleteBranch: !keptBranch)
        } else {
            worktreeRemoved = false
            let named = keptIgnoredFiles.joined(separator: ", ")
            // Captured immutable: the mutation closure runs on the store's
            // queue, and the local array must not be read from there while
            // this function could still change it.
            let keptNames = keptIgnoredFiles
            // The record must say why an applied run still has a worktree, or
            // the inspector's Apply reads as unfinished business.
            if let live {
                await live.noteKeptIgnoredFiles(keptNames)
            } else {
                await AgentRunStore.shared.mutateAsync(runId: runId, sessionId: record.parentSessionId) { current in
                    current.appliedAt = appliedAt
                    current.appliedToBranch = destinationBranch
                    current.keptIgnoredFiles = keptNames
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

        if keptBranch {
            let note = "The patch was applied, but branch '\(branch)' was kept because it holds commits outside the applied working tree. Inspect or delete it manually when you are done."
            if let live {
                await live.noteKeptBranch(branch)
            } else {
                await AgentRunStore.shared.mutateAsync(runId: runId, sessionId: record.parentSessionId) { current in
                    if var report = current.report {
                        report.openQuestions.append(note)
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
            keptIgnoredFiles: keptIgnoredFiles,
            keptBranch: keptBranch,
            worktreeRemoved: worktreeRemoved
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
    ///
    /// `executionStillUnwinding` keeps the button dark for the short window
    /// after a Stop in which the run's task is still killing a child process or
    /// writing its last tool result. The apply service refuses the same case;
    /// this only avoids offering a button that would answer with a refusal.
    static func canApply(_ record: AgentRunRecord, executionStillUnwinding: Bool = false) -> Bool {
        guard !executionStillUnwinding else { return false }
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
        diffStat: String? = nil,
        destinationNote: String? = nil
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
        // `destination` is the tree the apply will actually use. A recorded
        // root is validated by `WorkspaceIsolation.open`, which refuses a
        // worktree it no longer owns rather than following its `.git` pointer,
        // so naming it on the card keeps the approval attached to that tree.
        let destinationText = destination ?? "the main working tree (resolved when the apply runs)"
        var reason = "Apply \(summary) from \(branch) to \(destinationText). Changes land unstaged."
        if let destinationNote, !destinationNote.isEmpty {
            reason += "\n\n" + destinationNote
        }
        if let diffStat, !diffStat.isEmpty {
            reason += "\n\n" + diffStat
        } else if let files = record.report?.changedFiles, !files.isEmpty {
            // A changed-file list comes out of the RUN's report: the run chose
            // those names, so each is rendered on one line with this app's
            // markers neutralized before it reaches an operator-facing card.
            let named = files.prefix(10).map { Self.displayPath($0) }.joined(separator: "\n")
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

    /// One record-supplied path, safe to draw on a card: no newline can forge
    /// an extra line, a marker cannot reshape the surrounding text, and the
    /// length is bounded.
    nonisolated static func displayPath(_ path: String, limit: Int = 200) -> String {
        let singleLine = path
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        let clean = AgentHarness.sanitizeText(singleLine)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return clean.count > limit ? String(clean.prefix(limit)) + "…" : clean
    }

    /// A line naming uncommitted destination work, or nil when the destination
    /// is clean or git cannot say (a note that may be wrong is worse than
    /// none). A dirty destination is the usual reason an apply conflicts, and
    /// the card is where the operator can still stash or commit first.
    nonisolated static func destinationNote(for destination: String?) async -> String? {
        guard let destination else { return nil }
        guard let count = await WorkspaceIsolation.uncommittedFileCount(
            in: URL(fileURLWithPath: destination, isDirectory: true)
        ), count > 0 else { return nil }
        return "The destination has \(count) uncommitted file\(count == 1 ? "" : "s"); the patch lands on top of that work."
    }

    /// A destination summary for the CHAT approval card, resolved without
    /// touching git.
    ///
    /// The coordinator's `apply_subagent_changes` card is raised by the generic
    /// tool-approval surface, which otherwise showed only
    /// `subagent_id=…`. Naming the branch, the tree it lands in and the files
    /// is what lets the operator decide; the inspector's own card already gets
    /// a live `git diff --stat`, so this is the same information minus the
    /// stat.
    static func approvalSummary(forRunId runId: UUID) -> String? {
        let record = SubagentManager.shared.getSubagent(byId: runId)?.snapshotRecord()
            ?? AgentRunStore.shared.loadAny(runId: runId)
        guard let record else { return nil }
        return approvalSummary(for: record)
    }

    /// The pure form, so a card can build it from a record it read OFF the
    /// main actor. `loadAny` is a synchronous `ioQueue.sync`, so calling it in
    /// a view body parked the UI behind every queued run write.
    nonisolated static func approvalSummary(for record: AgentRunRecord, destinationNote: String? = nil) -> String? {
        guard case .worktree(_, let branch) = record.isolation else { return nil }
        // A record with no stored root has no destination that can be named
        // without following the sandbox-writable `.git` pointer, which is the
        // one thing the apply path refuses to trust: say so rather than
        // naming the WORKTREE as if it were the tree the patch lands in.
        let destination = record.repositoryRoot ?? "the main working tree (resolved when the apply runs)"
        let files = record.report?.changedFiles ?? []
        let count: String
        switch files.count {
        case 0: count = "the run's worktree changes"
        case 1: count = "1 changed file"
        default: count = "\(files.count) changed files"
        }
        var summary = "Applies \(count) from \(branch) to \(destination). Changes land unstaged."
        if let destinationNote, !destinationNote.isEmpty {
            summary += "\n" + destinationNote
        }
        if !files.isEmpty {
            // The run supplies these names; the card draws them, so each goes
            // through the same one-line sanitization as the inspector's.
            summary += "\n" + files.prefix(8).map { Self.displayPath($0) }.joined(separator: ", ")
            if files.count > 8 { summary += ", …" }
        }
        return summary
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
        guard canApply(
            record,
            executionStillUnwinding: !(live?.isExecutionUnwound ?? true)
        ), case .worktree(let path, let branch) = record.isolation else {
            throw SubagentApplyControlError.nothingToApply(record.role)
        }

        // The RECORDED root outranks the worktree's `.git` pointer, exactly as
        // `WorkspaceIsolation.open` (which the apply itself uses) decides it:
        // the pointer is a writable file inside the sandbox, so a retargeted
        // one would make the card name a different repository than the apply
        // touches. Git's own answer is the fallback for an older record that
        // stored no root.
        var destination = record.repositoryRoot
        if destination == nil {
            destination = await WorkspaceIsolation.owningRepository(
                for: URL(fileURLWithPath: path, isDirectory: true)
            )?.path
        }
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
        // The patch lands on whatever the destination has checked out, and a
        // dirty destination is the usual reason an apply conflicts. Saying so
        // on the card is what lets the operator stash or commit first instead
        // of approving an apply that cannot land.
        let destinationNote = await Self.destinationNote(for: destination)
        guard let request = applyRequest(
            for: record,
            destination: destination,
            diffStat: stat,
            destinationNote: destinationNote
        ) else {
            throw SubagentApplyControlError.nothingToApply(record.role)
        }

        let ask = requestApproval ?? {
            // Landing a patch is never a Turbo bypass: the contract in
            // SUBAGENTS.md requires its own card. Its own scope, because a
            // Stop that owns the run did not raise this operator decision.
            await SubagentApprovalCenter.shared.request(
                $0,
                honorsTurbo: false,
                scope: .operatorApply($0.runId)
            )
        }
        guard await ask(request) else { throw SubagentApplyControlError.approvalDenied }
        return try await SubagentApplyService.apply(runId: runId)
    }
}
