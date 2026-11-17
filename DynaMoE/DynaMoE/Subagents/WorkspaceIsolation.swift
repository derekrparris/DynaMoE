//
//  WorkspaceIsolation.swift
//  DynaMoE
//
//  A writer run's git worktree: where it lives, how its work is read back, and
//  how it is applied to the main tree. See SUBAGENTS.md §7 and §13 (Phase 4).
//
//  The worktree is the run's working directory for its whole life. Nothing here
//  merges branches: the run may commit inside its own worktree (the implementer
//  preset can), so a snapshot is diffed against the commit the worktree was
//  created from, and applying writes that net diff onto the main tree with
//  `git apply`. The operator, or the coordinator, then commits what landed.
//

import Foundation

// MARK: - Errors

nonisolated enum WorkspaceIsolationError: LocalizedError, Equatable {
    case notAGitRepository(String)
    case noCommits(String)
    case worktreeExists(String)
    case worktreeCreationFailed(String)
    case worktreeMissing(String)
    case snapshotFailed(String)
    case applyFailed(String)
    case applyConflicted(String)

    var errorDescription: String? {
        switch self {
        case .notAGitRepository(let path):
            return "'\(path)' is not inside a git repository, so an isolated worktree cannot be created."
        case .noCommits(let path):
            return "The repository at '\(path)' has no commits yet, so there is no base to isolate from."
        case .worktreeExists(let path):
            return "A worktree already exists at '\(path)'."
        case .worktreeCreationFailed(let detail):
            return "Could not create the run's worktree: \(detail)"
        case .worktreeMissing(let path):
            return "The run's worktree at '\(path)' is gone. It may already have been applied and cleaned up."
        case .snapshotFailed(let detail):
            return "Could not read the run's changes: \(detail)"
        case .applyFailed(let detail):
            return "Could not apply the run's changes: \(detail)"
        case .applyConflicted(let detail):
            return "The run's changes conflict with the current working tree: \(detail)"
        }
    }
}

// MARK: - Values

/// One run's worktree, with the commit it was branched from.
nonisolated struct AgentWorktree: Equatable, Sendable {
    var runId: UUID
    var repositoryRoot: URL
    var path: URL
    var branch: String
    var baseCommit: String

    var isolation: AgentIsolation {
        .worktree(path: path.path, branch: branch)
    }
}

/// What a snapshot of a worktree found. `diff` is bounded when a caller asks for
/// a display copy; `changedFiles` is always the complete list.
nonisolated struct WorkspaceChangeSet: Equatable, Sendable {
    var changedFiles: [String]
    var diff: String
    var diffWasTruncated: Bool

    var isEmpty: Bool { changedFiles.isEmpty && diff.isEmpty }
}

/// What applying a run's changes did.
nonisolated struct WorkspaceApplyResult: Equatable, Sendable {
    var changedFiles: [String]
    var branch: String
    var diffWasTruncated: Bool
}

// MARK: - Workspace isolation

nonisolated enum WorkspaceIsolation {
    /// The commit a snapshot is taken against is the one the branch was cut
    /// from, not the worktree's HEAD: the run may commit, and a HEAD-relative
    /// diff would then contain none of the work it committed.
    static let branchPrefix = "dynamoe/agent/"
    /// Matches `AgentRunStore.TranscriptBudget.reportDiff`, so a stored report
    /// is not truncated a second time by the store.
    static let diffDisplayLimit = 65_536
    static let staleWorktreeAge: TimeInterval = 14 * 86_400

    /// App Support keeps worktrees out of the operator's repository. A test run
    /// must not create them in the operator's real root, for the same reason
    /// `AgentRunStore` picks a temporary root under XCTest.
    static var processRootURL: URL {
        if isUnderXCTest {
            return FileManager.default.temporaryDirectory
                .appendingPathComponent("dynamoe-agent-worktrees-tests", isDirectory: true)
        }
        return defaultRootURL
    }

    static var defaultRootURL: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base
            .appendingPathComponent("DynaMoE", isDirectory: true)
            .appendingPathComponent("worktrees", isDirectory: true)
    }

    private static var isUnderXCTest: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
            || ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil
    }

    private static let gitURL = URL(fileURLWithPath: "/usr/bin/git")

    // MARK: Repository

    /// The repository root for a directory inside a git working tree, or nil
    /// when it is not in one. A spawn calls this on the coordinator's directory
    /// before any worktree exists; `owningRepository(for:)` answers the same
    /// question for a linked worktree.
    static func repositoryRoot(for directory: URL) async -> URL? {
        guard let (code, stdout, _) = try? await git(["rev-parse", "--show-toplevel"], in: directory),
              code == 0 else { return nil }
        let top = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !top.isEmpty else { return nil }
        return URL(fileURLWithPath: top, isDirectory: true)
    }

    static func isGitRepository(at directory: URL) async -> Bool {
        await repositoryRoot(for: directory) != nil
    }

    /// The main repository that owns a linked worktree.
    static func owningRepository(for worktree: URL) async -> URL? {
        guard let (code, stdout, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: worktree
        ), code == 0 else { return nil }
        let common = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !common.isEmpty else { return nil }
        let gitDirectory = URL(fileURLWithPath: common, isDirectory: true)
        return gitDirectory.deletingLastPathComponent()
    }

    // MARK: Creating

    /// Creates `worktrees/<runId>` on `dynamoe/agent/<short>`, branched from the
    /// repository's current HEAD.
    static func createWorktree(
        repositoryRoot: URL,
        runId: UUID,
        root: URL = WorkspaceIsolation.processRootURL
    ) async throws -> AgentWorktree {
        let rootPath = repositoryRoot.standardizedFileURL
        guard let (headCode, headOut, _) = try? await git(["rev-parse", "--verify", "HEAD"], in: rootPath),
              headCode == 0 else {
            throw WorkspaceIsolationError.noCommits(rootPath.path)
        }
        let baseCommit = headOut.trimmingCharacters(in: .whitespacesAndNewlines)
        let branch = branchPrefix + String(runId.uuidString.prefix(8)).lowercased()
        let worktreeURL = root.appendingPathComponent(runId.uuidString, isDirectory: true)

        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: worktreeURL.path) {
            throw WorkspaceIsolationError.worktreeExists(worktreeURL.path)
        }
        do {
            try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        } catch {
            throw WorkspaceIsolationError.worktreeCreationFailed(error.localizedDescription)
        }

        let (code, _, stderr) = try await git(
            ["worktree", "add", "-b", branch, worktreeURL.path, "HEAD"],
            in: rootPath
        )
        guard code == 0 else {
            try? fileManager.removeItem(at: worktreeURL)
            // `worktree add -b` can create the branch before it fails on the
            // checkout. A retry with the same run id would then meet a branch
            // that already exists, so the failed attempt gives it back.
            _ = try? await git(["branch", "-D", branch], in: rootPath)
            throw WorkspaceIsolationError.worktreeCreationFailed(
                stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        return AgentWorktree(
            runId: runId,
            repositoryRoot: rootPath,
            path: worktreeURL,
            branch: branch,
            baseCommit: baseCommit
        )
    }

    // MARK: Reopening

    /// Rebuilds a worktree's metadata from the repository itself, which is what
    /// a run restored from disk needs: the record keeps its path and branch, and
    /// the base commit is the merge base of the branch and the main tree's HEAD.
    /// (A run's own commits move the branch, never the merge base.)
    static func open(path: String, branch: String) async throws -> AgentWorktree {
        let worktreeURL = URL(fileURLWithPath: path, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw WorkspaceIsolationError.worktreeMissing(path)
        }
        guard let repositoryRoot = await owningRepository(for: worktreeURL) else {
            throw WorkspaceIsolationError.notAGitRepository(path)
        }
        guard let (branchCode, _, _) = try? await git(["rev-parse", "--verify", branch], in: repositoryRoot),
              branchCode == 0 else {
            throw WorkspaceIsolationError.worktreeMissing("\(path) (\(branch))")
        }
        let baseResult = try? await git(
            ["merge-base", branch, "HEAD"],
            in: repositoryRoot
        )
        guard let baseResult, baseResult.0 == 0 else {
            let detail = baseResult?.2.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            throw WorkspaceIsolationError.snapshotFailed(
                detail.isEmpty ? "git merge-base failed" : detail
            )
        }
        let baseCommit = baseResult.1.trimmingCharacters(in: .whitespacesAndNewlines)
        return AgentWorktree(
            runId: UUID(uuidString: worktreeURL.lastPathComponent) ?? UUID(),
            repositoryRoot: repositoryRoot,
            path: worktreeURL,
            branch: branch,
            baseCommit: baseCommit
        )
    }

    static func open(_ isolation: AgentIsolation) async throws -> AgentWorktree {
        guard case .worktree(let path, let branch) = isolation else {
            throw WorkspaceIsolationError.notAGitRepository("the run did not use a worktree")
        }
        return try await open(path: path, branch: branch)
    }

    // MARK: Reading changes

    /// Stages the worktree's files (so new files are in the diff) and reads the
    /// net change against the base commit. Staging is safe here: the worktree is
    /// disposable and the index is not part of what is applied.
    ///
    /// `diffLimit` bounds only the returned text. Applying re-reads the diff
    /// with no limit, because a truncated patch cannot be applied.
    static func snapshot(
        _ worktree: AgentWorktree,
        diffLimit: Int? = WorkspaceIsolation.diffDisplayLimit
    ) async throws -> WorkspaceChangeSet {
        let directory = worktree.path
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw WorkspaceIsolationError.worktreeMissing(directory.path)
        }

        let (addCode, _, addErr) = try await git(["add", "-A"], in: directory)
        guard addCode == 0 else {
            throw WorkspaceIsolationError.snapshotFailed(
                addErr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        let (nameCode, nameOut, nameErr) = try await git(
            ["diff", worktree.baseCommit, "--name-status", "--find-renames", "-z"],
            in: directory
        )
        guard nameCode == 0 else {
            throw WorkspaceIsolationError.snapshotFailed(
                nameErr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        let changedFiles = parseNameStatus(nameOut)

        let (diffCode, diffOut, diffErr) = try await git(
            ["diff", worktree.baseCommit, "--binary", "--find-renames", "--no-color"],
            in: directory
        )
        guard diffCode == 0 else {
            throw WorkspaceIsolationError.snapshotFailed(
                diffErr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        var diff = diffOut
        var truncated = false
        if let diffLimit, diff.count > diffLimit {
            let remaining = diff.count - diffLimit
            let marker = "\n… [diff truncated: \(remaining) more characters. Applying uses the full patch.]"
            let keep = max(0, diffLimit - marker.count)
            diff = String(diff.prefix(keep)) + marker
            truncated = true
        }
        return WorkspaceChangeSet(
            changedFiles: changedFiles,
            diff: diff,
            diffWasTruncated: truncated
        )
    }

    private static func parseNameStatus(_ raw: String) -> [String] {
        var files: [String] = []
        var seen = Set<String>()

        func append(_ path: String) {
            guard !path.isEmpty, seen.insert(path).inserted else { return }
            files.append(path)
        }

        // `--name-status -z` separates the STATUS from its path with NUL too,
        // not a tab: "A\0path\0" and "R100\0old\0new\0". The tab form is kept
        // for a caller that ever drops `-z`, because the two differ exactly
        // where a filename contains a newline.
        let tokens = raw.split(separator: "\0", omittingEmptySubsequences: true).map(String.init)
        var index = 0
        while index < tokens.count {
            let token = tokens[index]
            if token.contains("\t") {
                let fields = token.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
                if let path = fields.last, fields.count >= 2 { append(path) }
                index += 1
                continue
            }
            let status = token.first.map(String.init) ?? ""
            if ["A", "B", "C", "D", "M", "R", "T", "U", "X"].contains(status) {
                let isRename = status == "R" || status == "C"
                let pathIndex = index + (isRename ? 2 : 1)
                guard pathIndex < tokens.count else { break }
                append(tokens[pathIndex])
                index = pathIndex + 1
            } else {
                index += 1
            }
        }
        return files.sorted()
    }

    // MARK: Applying

    /// Applies the run's net change to the main tree. `git apply` validates the
    /// whole patch before touching a file, so a failure leaves the working tree
    /// as it was; a conflict is reported, never resolved silently.
    static func apply(
        _ worktree: AgentWorktree,
        changeSet: WorkspaceChangeSet? = nil
    ) async throws -> WorkspaceApplyResult {
        let changes: WorkspaceChangeSet
        if let changeSet {
            changes = changeSet
        } else {
            changes = try await snapshot(worktree, diffLimit: nil)
        }
        guard !changes.isEmpty else {
            return WorkspaceApplyResult(
                changedFiles: [],
                branch: worktree.branch,
                diffWasTruncated: false
            )
        }
        guard !changes.diffWasTruncated else {
            throw WorkspaceIsolationError.applyFailed("the patch was truncated before it could be applied")
        }
        try await refuseIfTheMainTreeIsMidOperation(worktree.repositoryRoot)

        let patchURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-\(worktree.branch.replacingOccurrences(of: "/", with: "-")).patch")
        do {
            try changes.diff.data(using: .utf8)?.write(to: patchURL, options: .atomic)
        } catch {
            throw WorkspaceIsolationError.applyFailed(error.localizedDescription)
        }
        defer { try? FileManager.default.removeItem(at: patchURL) }

        let (checkCode, _, checkErr) = try await git(
            ["apply", "--check", "--binary", "--whitespace=nowarn", patchURL.path],
            in: worktree.repositoryRoot
        )
        guard checkCode == 0 else {
            let detail = checkErr.trimmingCharacters(in: .whitespacesAndNewlines)
            if detail.localizedCaseInsensitiveContains("patch does not apply")
                || detail.localizedCaseInsensitiveContains("already exists") {
                throw WorkspaceIsolationError.applyConflicted(detail)
            }
            throw WorkspaceIsolationError.applyFailed(detail)
        }

        let (applyCode, _, applyErr) = try await git(
            ["apply", "--binary", "--whitespace=nowarn", patchURL.path],
            in: worktree.repositoryRoot
        )
        guard applyCode == 0 else {
            throw WorkspaceIsolationError.applyFailed(
                applyErr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        return WorkspaceApplyResult(
            changedFiles: changes.changedFiles,
            branch: worktree.branch,
            diffWasTruncated: false
        )
    }

    /// Applying during a merge or rebase is a different operation than applying
    /// to a settled tree, and the failure would land in the middle of someone
    /// else's conflict resolution. Refuse by name instead.
    private static func refuseIfTheMainTreeIsMidOperation(_ repositoryRoot: URL) async throws {
        if let (mergeCode, _, _) = try? await git(
            ["rev-parse", "-q", "--verify", "MERGE_HEAD"],
            in: repositoryRoot
        ), mergeCode == 0 {
            throw WorkspaceIsolationError.applyFailed(
                "a merge is in progress in the main tree. Finish or abort it, then apply again."
            )
        }
        for operation in ["rebase-merge", "rebase-apply"] {
            guard let (pathCode, pathOut, _) = try? await git(
                ["rev-parse", "--path-format=absolute", "--git-path", operation],
                in: repositoryRoot
            ), pathCode == 0 else { continue }
            let path = pathOut.trimmingCharacters(in: .whitespacesAndNewlines)
            if !path.isEmpty, FileManager.default.fileExists(atPath: path) {
                throw WorkspaceIsolationError.applyFailed(
                    "a rebase is in progress in the main tree. Finish or abort it, then apply again."
                )
            }
        }
    }

    // MARK: Cleaning up

    /// Removes the worktree and its branch. Called on cancel and after a
    /// successful apply; never on a completed run that has not been applied,
    /// because the worktree is then the only copy of work worth applying.
    @discardableResult
    static func cleanup(
        _ worktree: AgentWorktree,
        deleteBranch: Bool = true
    ) async -> Bool {
        let fileManager = FileManager.default
        _ = try? await git(
            ["worktree", "remove", "--force", worktree.path.path],
            in: worktree.repositoryRoot
        )
        _ = try? await git(["worktree", "prune"], in: worktree.repositoryRoot)
        if fileManager.fileExists(atPath: worktree.path.path) {
            try? fileManager.removeItem(at: worktree.path)
        }
        if deleteBranch, worktree.branch != "HEAD" {
            _ = try? await git(["branch", "-D", worktree.branch], in: worktree.repositoryRoot)
        }
        return !fileManager.fileExists(atPath: worktree.path.path)
    }

    /// Removes worktrees older than `olderThan` that no live run can still be
    /// using. A run record keeps its diff, but a worktree is the only place an
    /// unapplied run can be applied from, so the age is deliberately generous.
    @discardableResult
    static func pruneStaleWorktrees(
        root: URL = WorkspaceIsolation.processRootURL,
        olderThan: TimeInterval = WorkspaceIsolation.staleWorktreeAge,
        now: Date = Date()
    ) async -> [String] {
        let fileManager = FileManager.default
        guard let names = try? fileManager.contentsOfDirectory(atPath: root.path) else { return [] }
        var removed: [String] = []
        for name in names {
            let path = root.appendingPathComponent(name, isDirectory: true)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: path.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let values = try? path.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
            let created = values?.creationDate ?? values?.contentModificationDate ?? now
            guard now.timeIntervalSince(created) >= olderThan else { continue }

            if let repositoryRoot = await owningRepository(for: path),
               let (branchCode, branchOut, _) = try? await git(
                   ["rev-parse", "--abbrev-ref", "HEAD"],
                   in: path
               ), branchCode == 0 {
                let branch = branchOut.trimmingCharacters(in: .whitespacesAndNewlines)
                if branch != "HEAD" {
                    let worktree = AgentWorktree(
                        runId: UUID(uuidString: name) ?? UUID(),
                        repositoryRoot: repositoryRoot,
                        path: path,
                        branch: branch,
                        baseCommit: ""
                    )
                    _ = await cleanup(worktree)
                } else {
                    _ = try? await git(["worktree", "remove", "--force", path.path], in: repositoryRoot)
                    _ = try? await git(["worktree", "prune"], in: repositoryRoot)
                    try? fileManager.removeItem(at: path)
                }
            } else {
                try? fileManager.removeItem(at: path)
            }
            removed.append(name)
        }
        return removed
    }

    // MARK: Process

    private static func git(
        _ arguments: [String],
        in directory: URL
    ) async throws -> (Int32, String, String) {
        try await AgentHarness.runProcess(
            executableURL: gitURL,
            arguments: arguments,
            currentDirectory: directory,
            timeoutSeconds: 300
        )
    }
}
