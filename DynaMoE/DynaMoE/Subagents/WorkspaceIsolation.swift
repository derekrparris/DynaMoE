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
    case worktreeLinkageCompromised(String)
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
        case .worktreeLinkageCompromised(let detail):
            return "The run's worktree no longer points at the repository it was created from: \(detail). Nothing was staged, applied or deleted, and the worktree was kept so its files can be inspected."
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
    ///
    /// The parent of `--git-common-dir` is the repository root only for an
    /// ordinary checkout. A submodule's common dir lives at
    /// `<super>/.git/modules/<name>` and records its checkout in
    /// `core.worktree`; a `--separate-git-dir` repository records nothing at
    /// all, which is why a run's own record persists the root and `open`
    /// prefers it. This derivation is the fallback for a record that predates
    /// that field and for the launch sweep, which only has the path.
    static func owningRepository(for worktree: URL) async -> URL? {
        guard let (code, stdout, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: worktree
        ), code == 0 else { return nil }
        let common = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !common.isEmpty else { return nil }
        let gitDirectory = URL(fileURLWithPath: common, isDirectory: true)

        if let (configCode, configOut, _) = try? await git(
            ["--git-dir", common, "config", "--get", "core.worktree"],
            in: worktree
        ), configCode == 0 {
            let recorded = configOut.trimmingCharacters(in: .whitespacesAndNewlines)
            if !recorded.isEmpty {
                let root = URL(fileURLWithPath: recorded, relativeTo: gitDirectory)
                    .standardizedFileURL
                if await isMainWorktree(root, owning: worktree) { return root }
            }
        }

        if let (listCode, listOut, _) = try? await git(
            ["worktree", "list", "--porcelain"],
            in: worktree
        ), listCode == 0 {
            let first = listOut
                .split(separator: "\n")
                .first { $0.hasPrefix("worktree ") }
                .map { String($0.dropFirst("worktree ".count)) }
            if let first, !first.isEmpty {
                let root = URL(fileURLWithPath: first, isDirectory: true)
                if await isMainWorktree(root, owning: worktree) { return root }
            }
        }

        let parent = gitDirectory.deletingLastPathComponent()
        if await isMainWorktree(parent, owning: worktree) { return parent }
        return nil
    }

    /// True when `candidate` is a working tree of the SAME repository as
    /// `worktree`. It rejects both the linked worktree itself and a bare git
    /// directory, either of which would make `apply` write to the wrong tree.
    private static func isMainWorktree(_ candidate: URL, owning worktree: URL) async -> Bool {
        // The worktree itself is a working tree of the same repository; it is
        // not the place a patch may land.
        guard candidate.standardizedFileURL != worktree.standardizedFileURL else { return false }
        guard let (topCode, topOut, _) = try? await git(
            ["rev-parse", "--show-toplevel"],
            in: candidate
        ), topCode == 0 else { return false }
        let top = topOut.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !top.isEmpty else { return false }
        guard URL(fileURLWithPath: top).resolvingSymlinksInPath().standardizedFileURL
            == candidate.resolvingSymlinksInPath().standardizedFileURL else { return false }

        guard let (worktreeCode, worktreeOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: worktree
        ), worktreeCode == 0,
        let (candidateCode, candidateOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: candidate
        ), candidateCode == 0 else { return false }
        let a = URL(fileURLWithPath: worktreeOut.trimmingCharacters(in: .whitespacesAndNewlines))
            .resolvingSymlinksInPath().standardizedFileURL
        let b = URL(fileURLWithPath: candidateOut.trimmingCharacters(in: .whitespacesAndNewlines))
            .resolvingSymlinksInPath().standardizedFileURL
        return a == b
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
            ["worktree", "add", "-b", branch, worktreeURL.path, baseCommit],
            in: rootPath
        )
        guard code == 0 else {
            try? fileManager.removeItem(at: worktreeURL)
            // `worktree add -b` can create the branch before it fails on the
            // checkout. A retry with the same run id would then meet a branch
            // that already exists, so the failed attempt gives it back — but
            // only when the branch points at the base commit we asked git to
            // cut it from: the failure can also BE a pre-existing branch of
            // the same name (a collision with an earlier run's leftover), and
            // `branch -D` would then delete a branch this call never owned.
            if let (verifyCode, verifyOut, _) = try? await git(
                ["rev-parse", "--verify", "\(branch)^{commit}"],
                in: rootPath
            ), verifyCode == 0,
               verifyOut.trimmingCharacters(in: .whitespacesAndNewlines) == baseCommit {
                _ = try? await git(["branch", "-D", branch], in: rootPath)
            }
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
    /// `recordedBase` is the commit the worktree was cut from.
    ///
    /// A recorded base is preferred over `merge-base(branch, HEAD)` because the
    /// two answer different questions once the main branch is reset or rebased:
    /// the branch still starts at the commit the run was given, while the merge
    /// base moves, so the run's apparent change would grow or shrink without its
    /// work changing. The recorded commit must still exist and still be an
    /// ancestor of the branch; otherwise the derivation is the only option left.
    static func open(
        path: String,
        branch: String,
        repositoryRoot recordedRoot: URL? = nil,
        baseCommit recordedBase: String? = nil
    ) async throws -> AgentWorktree {
        let worktreeURL = URL(fileURLWithPath: path, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw WorkspaceIsolationError.worktreeMissing(path)
        }
        var repositoryRoot: URL?
        if let recordedRoot, await isMainWorktree(recordedRoot, owning: worktreeURL) {
            repositoryRoot = recordedRoot
        } else {
            repositoryRoot = await owningRepository(for: worktreeURL)
        }
        guard let repositoryRoot else {
            throw WorkspaceIsolationError.notAGitRepository(path)
        }
        guard let (branchCode, _, _) = try? await git(["rev-parse", "--verify", branch], in: repositoryRoot),
              branchCode == 0 else {
            throw WorkspaceIsolationError.worktreeMissing("\(path) (\(branch))")
        }
        let baseCommit: String
        if let recordedBase,
           let pinned = await verifiedBaseCommit(recordedBase, on: branch, in: repositoryRoot) {
            baseCommit = pinned
        } else {
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
            baseCommit = baseResult.1.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return AgentWorktree(
            runId: UUID(uuidString: worktreeURL.lastPathComponent) ?? UUID(),
            repositoryRoot: repositoryRoot,
            path: worktreeURL,
            branch: branch,
            baseCommit: baseCommit
        )
    }

    /// The recorded commit, resolved to a full id, when it still exists and is
    /// still part of the branch's history. Nil means the derivation in `open`
    /// has to answer instead.
    private static func verifiedBaseCommit(
        _ recorded: String,
        on branch: String,
        in repositoryRoot: URL
    ) async -> String? {
        let trimmed = recorded.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let (verifyCode, verifyOut, _) = try? await git(
                  ["rev-parse", "--verify", "\(trimmed)^{commit}"],
                  in: repositoryRoot
              ), verifyCode == 0 else { return nil }
        let resolved = verifyOut.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !resolved.isEmpty,
              let (ancestorCode, _, _) = try? await git(
                  ["merge-base", "--is-ancestor", resolved, branch],
                  in: repositoryRoot
              ), ancestorCode == 0 else { return nil }
        return resolved
    }

    static func open(_ isolation: AgentIsolation) async throws -> AgentWorktree {
        guard case .worktree(let path, let branch) = isolation else {
            throw WorkspaceIsolationError.notAGitRepository("the run did not use a worktree")
        }
        return try await open(path: path, branch: branch)
    }

    // MARK: Reading changes

    /// What makes a worktree's `.git` linkage untrustworthy, or nil when it
    /// still points where `createWorktree` put it.
    ///
    /// A linked worktree's `.git` is a regular FILE inside the containment
    /// prefix, so a sandboxed run can reach it: a rewritten pointer makes every
    /// later git command in the worktree run somewhere else — `git add -A`
    /// stages the operator's tree into their shared index, `git_commit` lands
    /// on their checked-out branch — and deleting the pointer orphans work
    /// nothing can land. The comparison is against the RECORDED repository
    /// (never against wherever the pointer happens to point now), plus the
    /// per-worktree admin directory's own back-pointer, which lives in the
    /// main repository's `.git` where the run cannot write it.
    static func worktreeLinkageProblem(_ worktree: AgentWorktree) async -> String? {
        guard let (rootCode, rootOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: worktree.repositoryRoot
        ), rootCode == 0 else {
            return "the recorded repository at '\(worktree.repositoryRoot.path)' could not be read"
        }
        let commonPath = URL(fileURLWithPath: rootOut.trimmingCharacters(in: .whitespacesAndNewlines), isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath().path
        guard let (gitDirCode, gitDirOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-dir"],
            in: worktree.path
        ), gitDirCode == 0 else {
            return "its .git linkage cannot be resolved, so git does not treat '\(worktree.path.path)' as a working tree"
        }
        let gitDir = URL(fileURLWithPath: gitDirOut.trimmingCharacters(in: .whitespacesAndNewlines), isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let worktreesRoot = commonPath.hasSuffix("/") ? commonPath + "worktrees/" : commonPath + "/worktrees/"
        guard gitDir.path.hasPrefix(worktreesRoot) else {
            return "its .git file points at '\(gitDir.path)', which is not a linked-worktree git directory of the repository at '\(worktree.repositoryRoot.path)'"
        }
        guard let pointer = try? String(contentsOf: gitDir.appendingPathComponent("gitdir"), encoding: .utf8) else {
            return "the repository's worktree record for '\(worktree.path.lastPathComponent)' could not be read"
        }
        let recordedBack = URL(fileURLWithPath: pointer.trimmingCharacters(in: .whitespacesAndNewlines))
            .standardizedFileURL.resolvingSymlinksInPath().path
        let expectedBack = worktree.path
            .appendingPathComponent(".git")
            .standardizedFileURL.resolvingSymlinksInPath().path
        guard recordedBack == expectedBack else {
            return "the repository's worktree record points at '\(recordedBack)' instead of this worktree"
        }
        return nil
    }

    /// Refuses to trust a worktree whose `.git` linkage moved, after one
    /// repair attempt.
    ///
    /// `git worktree repair` restores the pointer from the repository's own
    /// admin files, which live OUTSIDE the sandbox the run can write, so the
    /// repair can only ever put the ORIGINAL linkage back: a deliberately
    /// retargeted pointer is repaired (when the repo's records are intact) or
    /// refused — never followed.
    static func verifyWorktreeLinkage(_ worktree: AgentWorktree) async throws {
        if await worktreeLinkageProblem(worktree) == nil { return }
        _ = try? await git(["worktree", "repair", worktree.path.path], in: worktree.repositoryRoot)
        if let problem = await worktreeLinkageProblem(worktree) {
            throw WorkspaceIsolationError.worktreeLinkageCompromised(problem)
        }
    }

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
        // `git add -A` below is a WRITE whose destination is decided by the
        // worktree's `.git` pointer — a file inside the sandbox. A retargeted
        // pointer would stage the operator's tree into their shared index, so
        // the linkage is verified before the first git command runs.
        try await verifyWorktreeLinkage(worktree)

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

        // The two halves come from the same tree, so one with entries and the
        // other empty is not "no changes": it means the patch could not be
        // represented. `runProcess` decodes stdout as UTF-8 and a diff of a
        // non-UTF-8 text file fails that decode, leaving an empty string. Left
        // unchecked, that empty patch reached `git apply` and failed with the
        // bare "No valid patches", made every later apply fail the same way,
        // and let an "empty" snapshot authorize deleting the worktree - the only
        // copy of the run's work.
        if changedFiles.isEmpty != diffOut.isEmpty {
            throw WorkspaceIsolationError.snapshotFailed(
                changedFiles.isEmpty
                    ? "git reported no changed files but produced a patch, so the worktree index is inconsistent. The worktree was kept."
                    : "git reported \(changedFiles.count) changed file(s) but produced no usable patch. A file whose contents are not UTF-8 is the usual cause; the worktree was kept so nothing is lost."
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
                if isRename {
                    // BOTH sides: the patch deletes the old path as well, so
                    // the pre-flight and the rollback snapshot have to know it
                    // or a failed rename could leave the file gone.
                    guard index + 2 < tokens.count else { break }
                    append(tokens[index + 1])
                    append(tokens[index + 2])
                    index += 3
                } else {
                    guard index + 1 < tokens.count else { break }
                    append(tokens[index + 1])
                    index += 2
                }
            } else {
                index += 1
            }
        }
        return files.sorted()
    }

    // MARK: Applying

    /// Applies the run's net change to the main tree.
    ///
    /// `git apply --check` validates the whole patch, but the apply that follows
    /// is NOT atomic: a write that fails partway through (a permission, a full
    /// disk, a path replaced between the two calls) leaves the files it already
    /// wrote in place. The main tree is snapshotted first and restored if the
    /// apply fails, so a reported failure really does leave it as it was. A
    /// conflict is reported by name, never resolved silently.
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
        // The patch was read through the worktree's git linkage, and a
        // retargeted `.git` pointer could have made that read describe another
        // repository's state (or the operator's own tree). Verified only when
        // the worktree is still present: an apply that carries its own
        // changeSet against an already-cleaned-up worktree never consults it.
        if FileManager.default.fileExists(atPath: worktree.path.path) {
            try await verifyWorktreeLinkage(worktree)
        }
        try await refuseIfTheMainTreeIsMidOperation(worktree.repositoryRoot)
        try refuseIfTargetsAreNotWritable(changes.changedFiles, in: worktree.repositoryRoot)

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

        let rollback = try stageRollback(changes.changedFiles, in: worktree.repositoryRoot)
        defer { try? FileManager.default.removeItem(at: rollback.directory) }

        let (applyCode, _, applyErr) = try await git(
            ["apply", "--binary", "--whitespace=nowarn", patchURL.path],
            in: worktree.repositoryRoot
        )
        guard applyCode == 0 else {
            let detail = applyErr.trimmingCharacters(in: .whitespacesAndNewlines)
            let restored = rollback.restore(in: worktree.repositoryRoot)
            let suffix = detail.isEmpty ? "" : " " + detail
            if restored {
                throw WorkspaceIsolationError.applyFailed(
                    "the write failed partway through and was rolled back." + suffix
                )
            }
            throw WorkspaceIsolationError.applyFailed(
                "the write failed partway through and the rollback did NOT fully restore the tree; inspect it before retrying." + suffix
            )
        }
        return WorkspaceApplyResult(
            changedFiles: changes.changedFiles,
            branch: worktree.branch,
            diffWasTruncated: false
        )
    }

    /// Refuses a patch whose destination the process could not write BEFORE a
    /// byte is touched. `git apply --check` only reads, so a read-only target
    /// passes it and fails during the real apply.
    private static func refuseIfTargetsAreNotWritable(_ paths: [String], in root: URL) throws {
        let fileManager = FileManager.default
        for path in paths {
            guard let target = safeURL(for: path, in: root) else {
                throw WorkspaceIsolationError.applyFailed("the patch names an unsafe path: \(path)")
            }
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: target.path, isDirectory: &isDirectory) {
                guard !isDirectory.boolValue, fileManager.isWritableFile(atPath: target.path) else {
                    throw WorkspaceIsolationError.applyFailed(
                        "'\(path)' exists and is not a writable file, so the patch cannot land there."
                    )
                }
            }
            guard let ancestor = existingAncestor(of: target, fileManager: fileManager) else {
                throw WorkspaceIsolationError.applyFailed(
                    "no existing directory holds '\(path)', so the patch cannot land there."
                )
            }
            guard fileManager.isWritableFile(atPath: ancestor.path) else {
                throw WorkspaceIsolationError.applyFailed(
                    "the directory holding '\(path)' is not writable."
                )
            }
        }
    }

    private static func existingAncestor(of url: URL, fileManager: FileManager) -> URL? {
        var candidate = url.deletingLastPathComponent()
        while candidate.path != "/" {
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory) {
                return isDirectory.boolValue ? candidate : nil
            }
            candidate = candidate.deletingLastPathComponent()
        }
        return nil
    }

    /// A repository-relative path that cannot escape the main tree.
    static func safeURL(for path: String, in root: URL) -> URL? {
        guard !path.isEmpty, !path.hasPrefix("/") else { return nil }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.contains(".."), !components.contains("") else { return nil }
        let url = root.appendingPathComponent(path)
        // Compare the RAW spellings. `standardizedFileURL` and
        // `resolvingSymlinksInPath` both rewrite `/private/var/...` to
        // `/var/...` (and disagree about a child that does not exist yet), so
        // either one compared two spellings of the same tree and refused every
        // file the patch would create. `..` and empty components are already
        // rejected above, so the prefix holds by construction.
        let rootPath = root.path
        let targetPath = url.path
        guard targetPath.hasPrefix(rootPath.hasSuffix("/") ? rootPath : rootPath + "/") else {
            return nil
        }
        return url
    }

    /// The main tree's current bytes for every path the patch can touch.
    struct ApplyRollback {
        var directory: URL
        var entries: [(relativePath: String, original: URL?)]

        func restore(in root: URL) -> Bool {
            let fileManager = FileManager.default
            var restoredEverything = true
            let rootPath = root.path
            for entry in entries {
                guard let target = WorkspaceIsolation.safeURL(for: entry.relativePath, in: root) else {
                    restoredEverything = false
                    continue
                }
                do {
                    if let original = entry.original {
                        try fileManager.createDirectory(
                            at: target.deletingLastPathComponent(),
                            withIntermediateDirectories: true
                        )
                        if fileManager.fileExists(atPath: target.path) {
                            try fileManager.removeItem(at: target)
                        }
                        try fileManager.copyItem(at: original, to: target)
                    } else if fileManager.fileExists(atPath: target.path) {
                        try fileManager.removeItem(at: target)
                        // A patch that failed partway may also have created
                        // the DIRECTORIES its new files lived in, and removing
                        // the files alone leaves them: the tree is not "as it
                        // was". Walk upward removing directories the patch
                        // made, stopping at the first non-empty one and
                        // always before the repository root itself.
                        var directory = target.deletingLastPathComponent()
                        while directory.path != rootPath, directory.path != "/" {
                            guard let contents = try? fileManager.contentsOfDirectory(atPath: directory.path),
                                  contents.isEmpty else { break }
                            do {
                                try fileManager.removeItem(at: directory)
                            } catch {
                                restoredEverything = false
                                break
                            }
                            directory = directory.deletingLastPathComponent()
                        }
                    }
                } catch {
                    restoredEverything = false
                }
            }
            return restoredEverything
        }
    }

    static func stageRollback(_ paths: [String], in root: URL) throws -> ApplyRollback {
        let fileManager = FileManager.default
        let directory = fileManager.temporaryDirectory
            .appendingPathComponent("dynamoe-rollback-\(UUID().uuidString)", isDirectory: true)
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw WorkspaceIsolationError.applyFailed(
                "could not prepare a rollback snapshot: \(error.localizedDescription)"
            )
        }
        var entries: [(String, URL?)] = []
        for path in paths {
            guard let target = safeURL(for: path, in: root) else {
                throw WorkspaceIsolationError.applyFailed("the patch names an unsafe path: \(path)")
            }
            guard fileManager.fileExists(atPath: target.path) else {
                entries.append((path, nil))
                continue
            }
            let copy = directory.appendingPathComponent(UUID().uuidString)
            do {
                try fileManager.copyItem(at: target, to: copy)
            } catch {
                try? fileManager.removeItem(at: directory)
                throw WorkspaceIsolationError.applyFailed(
                    "could not snapshot '\(path)' before applying: \(error.localizedDescription)"
                )
            }
            entries.append((path, copy))
        }
        return ApplyRollback(directory: directory, entries: entries)
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
    ///
    /// A path and branch restored from disk are treated as untrusted: the branch
    /// is deleted only when it carries this app's prefix, and the directory is
    /// removed by hand only when it is under this process's worktree root. A
    /// `git worktree remove` naming something else fails harmlessly; the manual
    /// fallback would not.
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
        if fileManager.fileExists(atPath: worktree.path.path),
           isInsideProcessWorktreeRoot(worktree.path) {
            try? fileManager.removeItem(at: worktree.path)
        }
        if deleteBranch,
           worktree.branch != "HEAD",
           worktree.branch.hasPrefix(branchPrefix) {
            _ = try? await git(["branch", "-D", worktree.branch], in: worktree.repositoryRoot)
        }
        return !fileManager.fileExists(atPath: worktree.path.path)
    }

    /// Whether a path may be deleted by the launch sweep or an apply. Only this
    /// process's own worktree root qualifies; a record's path is data.
    static func isInsideProcessWorktreeRoot(_ url: URL) -> Bool {
        let root = processRootURL.standardizedFileURL.resolvingSymlinksInPath().path
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        guard !root.isEmpty else { return false }
        return path.hasPrefix(root.hasSuffix("/") ? root : root + "/")
    }

    /// Files the run created that git ignores. `snapshot` stages with `git add
    /// -A`, which cannot carry them, so an ignored-only run looks empty and
    /// would otherwise be reported as "no changes" and deleted with its
    /// worktree. Naming them lets the apply refuse and keep the tree.
    static func ignoredFiles(_ worktree: AgentWorktree) async -> [String] {
        guard let (code, stdout, _) = try? await git(
            ["status", "--porcelain", "--ignored=matching", "-z"],
            in: worktree.path
        ), code == 0 else { return [] }
        return stdout
            .split(separator: "\0", omittingEmptySubsequences: true)
            .compactMap { entry -> String? in
                guard entry.hasPrefix("!! ") else { return nil }
                return String(entry.dropFirst(3))
            }
    }

    /// Whether git itself agrees the worktree has nothing to land.
    ///
    /// `snapshot` stages with `git add -A`, so an unchanged tree reports no
    /// status entries and a changed one still does even when its diff could not
    /// be decoded. Cleanup is destructive, and an empty `WorkspaceChangeSet` is
    /// not proof: it is also what a decode failure produces (see `snapshot`).
    /// Anything other than a successful, empty status is treated as "do not
    /// delete".
    static func hasNoWorkingTreeChanges(_ worktree: AgentWorktree) async -> Bool {
        guard let (code, stdout, _) = try? await git(
            ["status", "--porcelain", "-z"],
            in: worktree.path
        ), code == 0 else { return false }
        return stdout.isEmpty
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

            // A worktree is the ONLY full copy of a run that has not been
            // applied. Age is not evidence that its work is safe to destroy:
            // the record keeps a bounded diff, not the files. Leave it alone
            // until the operator applies or clears the run — including a
            // CANCELLED run's tree, which a Stop must not turn into the one
            // path that destroys work.
            if let runId = UUID(uuidString: name),
               let record = AgentRunStore.shared.loadAny(runId: runId),
               record.appliedAt == nil {
                continue
            }

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

    /// The branch a repository has CHECKED OUT, so an apply can record where
    /// the patch landed. Nil for a detached HEAD (which is itself an answer:
    /// the apply landed on no branch) or when git cannot say.
    static func currentBranch(of repositoryRoot: URL) async -> String? {
        guard let (code, out, _) = try? await git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            in: repositoryRoot
        ), code == 0 else { return nil }
        let branch = out.trimmingCharacters(in: .whitespacesAndNewlines)
        return branch.isEmpty || branch == "HEAD" ? nil : branch
    }

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
