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
    case nestedRepositoryUnsupported(String)

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
        case .nestedRepositoryUnsupported(let path):
            return "The run's worktree contains a nested git repository at '\(path)', which a patch cannot carry: git can only record a gitlink, and applying one reports success while leaving the nested files behind. Nothing was applied and the worktree was kept so nothing is lost."
        }
    }
}

// MARK: - Values

/// One run's worktree, with the commit it was branched from.
nonisolated struct AgentWorktree: Equatable, Sendable {
    /// The run that owns this tree, when the caller knows it. `open` derives it
    /// from the directory name and leaves it nil rather than inventing one: a
    /// fabricated identity is a value something later trusts.
    var runId: UUID?
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

    /// What `<worktree>/.git` is on disk, decided without running git.
    ///
    /// A linked worktree's `.git` is a FILE holding `gitdir: <path>`, but it
    /// sits inside a directory the run can write, so a carded `shell_run` can
    /// replace it. Both replacements are hostile in ways git itself cannot
    /// report: a DANGLING SYMLINK makes `git rev-parse` fail (which reads as
    /// "repairable"), and `git worktree repair` then opens the path with
    /// `O_WRONLY | O_CREAT | O_TRUNC` and clobbers whatever the link names
    /// OUTSIDE the sandbox; a FIFO makes `git rev-parse` block on open for the
    /// whole process timeout. Only a missing entry is repairable — a symlink,
    /// directory, FIFO or anything else is a pointer this app must not follow.
    static func worktreeGitEntryState(_ worktree: URL) async -> WorktreeGitEntryState {
        let path = worktree.appendingPathComponent(".git").path
        var info = stat()
        if lstat(path, &info) != 0 {
            return errno == ENOENT ? .missing : .unreadable(String(cString: strerror(errno)))
        }
        let mode = info.st_mode & S_IFMT
        switch mode {
        case S_IFREG: return .regularFile
        case S_IFLNK: return .hostile("its .git is a symlink, which is not how git writes a linked worktree")
        case S_IFDIR: return .hostile("its .git is a directory, so it is not a linked worktree of this repository")
        case S_IFIFO: return .hostile("its .git is a FIFO, which would block every git command in it")
        default: return .hostile("its .git is not a regular file")
        }
    }

    enum WorktreeGitEntryState: Equatable {
        case regularFile
        case missing
        case unreadable(String)
        case hostile(String)
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
        // Same reason as `worktreeLinkageState`: a `.git` the run can replace
        // (a FIFO blocks git for the whole process timeout, a symlink is
        // followed) must be inspected before any git command runs in here.
        switch await worktreeGitEntryState(worktree) {
        case .regularFile:
            break
        case .missing, .unreadable, .hostile:
            return nil
        }
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
    ///
    /// The worktree's `.git` entry state is deliberately NOT checked here. It
    /// used to be, behind `candidate == worktree`, and that branch could never
    /// run: every caller passes a repository ROOT or a parent, never the
    /// worktree itself — and the first guard below returns for the worktree
    /// anyway. Reading it as protection was worse than having none, so it is
    /// gone rather than reordered: the paths that run git INSIDE the worktree
    /// (`owningRepository`, the linkage checks) inspect the entry themselves,
    /// and this function runs git in the CANDIDATE, which for the recorded-root
    /// path is the operator's own repository — a `.git` a run cannot replace.
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

        guard let (candidateCode, candidateOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: candidate
        ), candidateCode == 0 else { return false }
        let candidateCommon = candidateOut.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !candidateCommon.isEmpty else { return false }
        let b = URL(fileURLWithPath: candidateCommon)
            .resolvingSymlinksInPath().standardizedFileURL

        // The repository's OWN admin directory must name this worktree, and
        // this check runs even when `git` inside the worktree can answer.
        //
        // `.git` inside the sandbox is a POINTER a run can rewrite. A pointer
        // retargeted at another repository makes every git answer inside the
        // worktree agree with it — including `--git-common-dir`, which is what
        // the comparison below reads — so the sandbox-side comparison alone
        // accepted a destination the worktree was never created from. The
        // admin back-pointer lives outside the sandbox and is written by
        // `git worktree add`, so a legitimate worktree always has it.
        guard adminDirectory(b, references: worktree) else { return false }

        if let (worktreeCode, worktreeOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: worktree
        ), worktreeCode == 0 {
            let worktreeCommon = worktreeOut.trimmingCharacters(in: .whitespacesAndNewlines)
            if !worktreeCommon.isEmpty {
                let a = URL(fileURLWithPath: worktreeCommon)
                    .resolvingSymlinksInPath().standardizedFileURL
                return a == b
            }
        }
        // The worktree's `.git` is gone (or unreadable), so git inside it
        // cannot answer; the back-pointer above is the only remaining link and
        // it already matched.
        return true
    }

    /// True when `commonDir` (a repository's git directory) holds a
    /// `worktrees/*/gitdir` back-pointer naming `worktree`.
    ///
    /// Resolves symlinks on the existing worktree directory (not the deleted
    /// `.git` leaf, which `resolvingSymlinksInPath()` may not canonicalize
    /// across `/var` -> `/private/var`).
    private static func adminDirectory(_ commonDir: URL, references worktree: URL) -> Bool {
        let expectedWorktreeDir = worktree
            .resolvingSymlinksInPath().standardizedFileURL.path
        let adminRoot = commonDir.appendingPathComponent("worktrees", isDirectory: true)
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: adminRoot,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return false }
        for entry in entries {
            guard let pointer = try? String(
                contentsOf: entry.appendingPathComponent("gitdir"),
                encoding: .utf8
            ) else { continue }
            let recordedURL = URL(fileURLWithPath: pointer.trimmingCharacters(in: .whitespacesAndNewlines))
            guard recordedURL.lastPathComponent == ".git" else { continue }
            let recordedWorktreeDir = recordedURL
                .deletingLastPathComponent()
                .resolvingSymlinksInPath().standardizedFileURL.path
            if recordedWorktreeDir == expectedWorktreeDir {
                return true
            }
        }
        return false
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
        // 48 bits, not 32: a branch name is the only handle `open` has on a
        // worktree after a crash, and a same-named leftover from an earlier
        // run is exactly the collision an 8-hex prefix hit in practice.
        let branch = branchPrefix + String(runId.uuidString.prefix(12)).lowercased()
        let worktreeURL = root.appendingPathComponent(runId.uuidString, isDirectory: true)

        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: worktreeURL.path) {
            throw WorkspaceIsolationError.worktreeExists(worktreeURL.path)
        }
        do {
            // 0700: a worktree holds the run's whole working copy, including
            // whatever it read, and `createDirectory` otherwise makes it 0755
            // readable by every account on the machine.
            try fileManager.createDirectory(
                at: root,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try? fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
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
        baseCommit recordedBase: String? = nil,
        runId: UUID? = nil
    ) async throws -> AgentWorktree {
        let worktreeURL = URL(fileURLWithPath: path, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw WorkspaceIsolationError.worktreeMissing(path)
        }
        // A run's worktree is always `root/<runId>`, and every check below
        // proves only "some registered worktree of this repo". Without this
        // one, a stale or tampered record naming a SIBLING run's worktree
        // would pass them all: its diff would land under this run's identity,
        // and the cleanup afterwards would delete the sibling's tree.
        if let runId {
            let expected = runId.uuidString
            let actual = worktreeURL.standardizedFileURL.lastPathComponent
            guard actual.caseInsensitiveCompare(expected) == .orderedSame else {
                throw WorkspaceIsolationError.worktreeLinkageCompromised(
                    "the worktree at '\(path)' belongs to a different run (expected '\(expected)', found '\(actual)')"
                )
            }
        }
        var repositoryRoot: URL?
        if let recordedRoot {
            // A RECORDED root outranks the pointer. The worktree's `.git` is a
            // writable file inside the sandbox, so deriving the root from it
            // is what let a retargeted pointer hand the operator's tree to the
            // apply path. The record is the only answer this worktree was
            // created with; when it no longer owns the worktree, refuse rather
            // than follow wherever the pointer says.
            guard await isMainWorktree(recordedRoot, owning: worktreeURL) else {
                throw WorkspaceIsolationError.worktreeLinkageCompromised(
                    "the worktree no longer belongs to the recorded repository at '\(recordedRoot.path)'"
                )
            }
            repositoryRoot = recordedRoot
        } else {
            // An older record with no stored root, or the launch sweep, which
            // has only the path. The linkage/HEAD checks below still gate what
            // it is used for.
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
            runId: UUID(uuidString: worktreeURL.lastPathComponent),
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
    /// later git command in the worktree run somewhere else — `git add -N .`
    /// registers paths in the operator's shared index, `git_commit` lands
    /// on their checked-out branch — and deleting the pointer orphans work
    /// nothing can land. The comparison is against the RECORDED repository
    /// (never against wherever the pointer happens to point now), plus the
    /// per-worktree admin directory's own back-pointer, which lives in the
    /// main repository's `.git` where the run cannot write it.
    static func worktreeLinkageProblem(_ worktree: AgentWorktree) async -> String? {
        switch await worktreeLinkageState(worktree) {
        case .ok: return nil
        case .repairable(let problem), .compromised(let problem): return problem
        }
    }

    /// Whether the linkage can be restored without touching another worktree.
    ///
    /// `repairable` is a MISSING or unreadable pointer: git's own admin record
    /// still describes this worktree, so `worktree repair <path>` can put the
    /// pointer back. `compromised` is a pointer that RESOLVES somewhere else —
    /// another repository, or a sibling worktree's admin directory. Repairing
    /// that case is not scoped by the path argument: git rewrites the admin
    /// file's back-pointer, which belongs to the sibling, so the check must
    /// refuse without calling repair at all.
    private enum WorktreeLinkageState {
        case ok
        case repairable(String)
        case compromised(String)
    }

    private static func worktreeLinkageState(_ worktree: AgentWorktree) async -> WorktreeLinkageState {
        // Before any git command runs in the worktree: its `.git` entry is in a
        // directory the run can write, and git would FOLLOW a symlink or BLOCK
        // on a FIFO there. Only a missing entry is repairable; see
        // `worktreeGitEntryState`.
        switch await worktreeGitEntryState(worktree.path) {
        case .regularFile:
            break
        case .missing:
            return .repairable("it has no .git pointer, so git does not treat '\(worktree.path.path)' as a working tree")
        case .unreadable(let reason):
            return .compromised("its .git entry could not be inspected (\(reason))")
        case .hostile(let reason):
            return .compromised(reason)
        }
        guard let (rootCode, rootOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            in: worktree.repositoryRoot
        ), rootCode == 0 else {
            // Without the recorded repository's own record there is nothing
            // authoritative to repair towards.
            return .compromised("the recorded repository at '\(worktree.repositoryRoot.path)' could not be read")
        }
        // An EMPTY answer is not the current directory. `URL(fileURLWithPath: "")`
        // resolves to "/", which used to turn a silent git failure into a linkage
        // report against the filesystem root.
        let commonOut = rootOut.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !commonOut.isEmpty else {
            return .compromised("the recorded repository at '\(worktree.repositoryRoot.path)' answered with no git directory")
        }
        let commonPath = URL(fileURLWithPath: commonOut, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath().path
        guard let (gitDirCode, gitDirOut, _) = try? await git(
            ["rev-parse", "--path-format=absolute", "--git-dir"],
            in: worktree.path
        ), gitDirCode == 0 else {
            return .repairable("its .git linkage cannot be resolved, so git does not treat '\(worktree.path.path)' as a working tree")
        }
        let gitDirPath = gitDirOut.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !gitDirPath.isEmpty else {
            return .repairable("its .git linkage resolved to no git directory, so git does not treat '\(worktree.path.path)' as a working tree")
        }
        let gitDir = URL(fileURLWithPath: gitDirPath, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let worktreesRoot = commonPath.hasSuffix("/") ? commonPath + "worktrees/" : commonPath + "/worktrees/"
        guard gitDir.path.hasPrefix(worktreesRoot) else {
            return .compromised("its .git file points at '\(gitDir.path)', which is not a linked-worktree git directory of the repository at '\(worktree.repositoryRoot.path)'")
        }
        guard let pointer = try? String(contentsOf: gitDir.appendingPathComponent("gitdir"), encoding: .utf8) else {
            // The admin directory exists but its back-pointer is gone: git's
            // own record of the worktree is incomplete, and repair restores it
            // from the path given.
            return .repairable("the repository's worktree record for '\(worktree.path.lastPathComponent)' could not be read")
        }
        let recordedBack = URL(fileURLWithPath: pointer.trimmingCharacters(in: .whitespacesAndNewlines))
            .standardizedFileURL.resolvingSymlinksInPath().path
        let expectedBack = worktree.path
            .appendingPathComponent(".git")
            .standardizedFileURL.resolvingSymlinksInPath().path
        guard recordedBack == expectedBack else {
            return .compromised("the repository's worktree record points at '\(recordedBack)' instead of this worktree")
        }
        return .ok
    }

    /// Refuses to trust a worktree whose `.git` linkage moved, after one
    /// repair attempt FOR A MISSING POINTER ONLY.
    ///
    /// `git worktree repair` restores the pointer from the repository's own
    /// admin files, which live OUTSIDE the sandbox the run can write. That is
    /// safe while the admin file still describes THIS worktree; a pointer
    /// retargeted at another worktree's admin directory makes repair rewrite
    /// that sibling's record (git resolves the path through the pointer, so
    /// the path argument does not scope it). A retargeted pointer is therefore
    /// refused without repair, and never followed.
    static func verifyWorktreeLinkage(_ worktree: AgentWorktree) async throws {
        switch await worktreeLinkageState(worktree) {
        case .ok:
            try await verifyWorktreeHead(worktree)
            return
        case .compromised(let problem):
            throw WorkspaceIsolationError.worktreeLinkageCompromised(problem)
        case .repairable:
            break
        }
        _ = try? await git(["worktree", "repair", worktree.path.path], in: worktree.repositoryRoot)
        switch await worktreeLinkageState(worktree) {
        case .ok:
            break
        case .repairable(let problem), .compromised(let problem):
            throw WorkspaceIsolationError.worktreeLinkageCompromised(problem)
        }
        try await verifyWorktreeHead(worktree)
    }

    /// What makes a worktree's HEAD untrustworthy, or nil when it is still on
    /// the branch `createWorktree` checked out.
    ///
    /// A worktree is a real checkout, so a carded `shell_run` can `git
    /// checkout` anything in it. The next cardless `git_commit` would then
    /// commit onto — and move — that other branch while the record still
    /// claims the run's own. The linkage check alone cannot see this: the
    /// `.git` pointer is intact and points at the right repository. An empty
    /// recorded branch (a caller that has none to compare against) answers
    /// nil.
    static func worktreeHeadProblem(_ worktree: AgentWorktree) async -> String? {
        guard !worktree.branch.isEmpty else { return nil }
        guard let (code, out, _) = try? await git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            in: worktree.path
        ), code == 0 else {
            return "its HEAD could not be read, so the branch the run was created on cannot be confirmed"
        }
        let checkedOut = out.trimmingCharacters(in: .whitespacesAndNewlines)
        guard checkedOut == worktree.branch else {
            return "its HEAD is on '\(checkedOut)', not the branch it was created on ('\(worktree.branch)'). A checkout inside the worktree would make the run commit onto another branch; the worktree was kept so nothing is lost."
        }
        return nil
    }

    static func verifyWorktreeHead(_ worktree: AgentWorktree) async throws {
        if let problem = await worktreeHeadProblem(worktree) {
            throw WorkspaceIsolationError.worktreeLinkageCompromised(problem)
        }
    }

    /// Registers the worktree's untracked files so they appear in the diff
    /// WITHOUT writing their contents into the repository's object store, and
    /// reads the net change against the base commit. A linked worktree shares
    /// the main repository's objects, so `git add -A` here blobbed every file
    /// the run touched into the OPERATOR's object DB just to read a patch
    /// back; intent-to-add (`-N`) records only the paths, and the diff carries
    /// the same content either way. The index is not part of what is applied.
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
        // A commit id is interpolated into git's argv below, and `AgentWorktree`
        // is constructible from a RECORD, so it is data: `git diff --output=<file>
        // <commit>` writes an arbitrary path if the "commit" starts with `--`.
        // `createWorktree` and `open` always produce a resolved hex id; this
        // refuses anything else rather than trusting every caller.
        guard Self.isResolvedCommitID(worktree.baseCommit) else {
            throw WorkspaceIsolationError.snapshotFailed(
                "the recorded base commit '\(worktree.baseCommit)' is not a resolved commit id"
            )
        }
        // `git add -N .` below is a WRITE whose destination is decided by the
        // worktree's `.git` pointer — a file inside the sandbox. A retargeted
        // pointer would register paths in the operator's shared index, so
        // the linkage is verified before the first git command runs.
        try await verifyWorktreeLinkage(worktree)

        let (addCode, _, addErr) = try await git(["add", "-N", "."], in: directory)
        guard addCode == 0 else {
            throw WorkspaceIsolationError.snapshotFailed(
                addErr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        let (nameCode, nameOut, nameErr) = try await git(
            ["diff", worktree.baseCommit, "--no-ext-diff", "--no-textconv", "--name-status", "--find-renames", "-z"],
            in: directory
        )
        guard nameCode == 0 else {
            throw WorkspaceIsolationError.snapshotFailed(
                nameErr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        let changedFiles = parseNameStatus(nameOut)

        let (diffCode, diffOut, diffErr) = try await git(
            ["diff", worktree.baseCommit, "--no-ext-diff", "--no-textconv", "--binary", "--find-renames", "--no-color"],
            in: directory
        )
        guard diffCode == 0 else {
            throw WorkspaceIsolationError.snapshotFailed(
                diffErr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        // A nested repository is a gitlink, and `git apply` reports success
        // while writing only an empty directory. Refuse it at the first place
        // the patch is read, so the report names it and the worktree is kept.
        if let nested = nestedRepositoryPath(inPatch: diffOut) {
            throw WorkspaceIsolationError.nestedRepositoryUnsupported(nested)
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

    /// The destination path of a `diff --git a/<a> b/<b>` header, or nil.
    ///
    /// Git C-quotes a path containing a quote, a backslash or a non-ASCII byte,
    /// so `diff --git "a/README \"notes\".md" "b/README \"notes\".md"` is a
    /// legal header. Reading only the last `" b/"` then returned
    /// `README \"notes\".md"` — quotes and escapes included — which is not the
    /// path git would write. The b/ side is the LAST token, quoted or not.
    private static func diffDestination(of header: Substring) -> String? {
        let payload = String(header.dropFirst("diff --git ".count))
        if payload.hasSuffix("\"") {
            // Find the last two UNESCAPED quotes: the a/ side may be quoted
            // too, and a `\"` inside either path is not a delimiter.
            var quotes: [String.Index] = []
            var backslashes = 0
            var cursor = payload.startIndex
            while cursor < payload.endIndex {
                let character = payload[cursor]
                if character == "\\" {
                    backslashes += 1
                } else {
                    if character == "\"", backslashes % 2 == 0 {
                        quotes.append(cursor)
                    }
                    backslashes = 0
                }
                cursor = payload.index(after: cursor)
            }
            guard quotes.count >= 2 else { return nil }
            let open = quotes[quotes.count - 2]
            let close = quotes[quotes.count - 1]
            let inner = payload[payload.index(after: open)..<close]
            guard let unquoted = unquoteGitPath(inner), unquoted.hasPrefix("b/") else { return nil }
            let path = String(unquoted.dropFirst(2))
            return path.isEmpty ? nil : path
        }
        guard let separator = payload.range(of: " b/", options: .backwards) else { return nil }
        let path = payload[separator.upperBound...]
        return path.isEmpty ? nil : String(path)
    }

    /// The bytes a git C-quoted path names. Octal escapes are how git spells a
    /// non-UTF-8 byte, so the result is assembled as bytes and decoded once.
    private static func unquoteGitPath(_ quoted: Substring) -> String? {
        var bytes: [UInt8] = []
        var cursor = quoted.startIndex
        while cursor < quoted.endIndex {
            let character = quoted[cursor]
            guard character == "\\" else {
                bytes.append(contentsOf: String(character).utf8)
                cursor = quoted.index(after: cursor)
                continue
            }
            cursor = quoted.index(after: cursor)
            guard cursor < quoted.endIndex else { return nil }
            let escape = quoted[cursor]
            switch escape {
            case "\\": bytes.append(0x5C)
            case "\"": bytes.append(0x22)
            case "t": bytes.append(0x09)
            case "n": bytes.append(0x0A)
            case "r": bytes.append(0x0D)
            case "a": bytes.append(0x07)
            case "b": bytes.append(0x08)
            case "f": bytes.append(0x0C)
            case "v": bytes.append(0x0B)
            default:
                guard let first = escape.wholeNumberValue, first < 8 else {
                    bytes.append(contentsOf: String(escape).utf8)
                    cursor = quoted.index(after: cursor)
                    continue
                }
                var value = first
                var digits = 1
                while digits < 3 {
                    let next = quoted.index(after: cursor)
                    guard next < quoted.endIndex,
                          let digit = quoted[next].wholeNumberValue,
                          digit < 8 else { break }
                    value = value * 8 + digit
                    cursor = next
                    digits += 1
                }
                bytes.append(UInt8(value & 0xFF))
            }
            cursor = quoted.index(after: cursor)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// The first file in a patch that git can only express as a gitlink
    /// (mode 160000, a nested repository/submodule), or nil.
    ///
    /// `git apply` does NOT fail on one: it writes an empty directory, warns
    /// "unable to rmdir", and exits 0 while the nested contents never land.
    /// Treating that exit status as success is what let the apply path report
    /// "applied N files" and then delete the worktree, the only copy of the
    /// nested files.
    static func nestedRepositoryPath(inPatch diff: String) -> String? {
        var current: String?
        for line in diff.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("diff --git ") {
                current = diffDestination(of: line)
                continue
            }
            // A gitlink's mode appears in the index line, which is present for
            // both an added and an updated submodule (the latter has no
            // "new file mode" line).
            if line.hasPrefix("index "),
               line.split(separator: " ").contains("160000") {
                return current ?? "a nested repository"
            }
            if line.hasPrefix("new file mode 160000")
                || line.hasPrefix("deleted file mode 160000")
                || line.hasPrefix("old mode 160000")
                || line.hasPrefix("new mode 160000") {
                return current ?? "a nested repository"
            }
        }
        return nil
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
        // `snapshot` refuses one, but a caller can hand a change set in
        // directly. Check again before a byte is touched: `git apply` exits 0
        // on a gitlink while writing only an empty directory.
        if let nested = nestedRepositoryPath(inPatch: changes.diff) {
            throw WorkspaceIsolationError.nestedRepositoryUnsupported(nested)
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
        // The FILE LIST is caller-supplied data too, and it is what the
        // preflight and the rollback snapshot are taken over: a change set whose
        // diff touches a path the list omits would get neither, so a partial
        // apply could not restore it. A list that disagrees with the patch is
        // refused rather than reconciled — one of the two is not what the
        // operator approved, and guessing which would be worse.
        //
        // A patch whose file list cannot be read is refused too (the helper
        // throws): skipping the cross-check would apply exactly the change set
        // nothing could vouch for.
        let patchFiles = try await Self.filesTouched(byPatch: changes.diff)
        let listed = Set(changes.changedFiles)
        if !patchFiles.isSubset(of: listed) {
            let missing = patchFiles.subtracting(listed).sorted()
            throw WorkspaceIsolationError.applyFailed(
                "the change set's file list omits \(missing.joined(separator: ", ")), which its patch touches."
            )
        }
        try refuseIfTargetsAreNotWritable(changes.changedFiles, in: worktree.repositoryRoot)

        // A PRIVATE directory, not a name derived from the branch. The old path
        // was `dynamoe-<branch>.patch` in the shared temp directory: predictable
        // across runs, so another process (or a later run of this app) could
        // pre-create or replace it between the check and the apply and have
        // `git apply` read bytes nobody reviewed. One UUID directory at 0700,
        // removed with everything in it.
        let patchDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-apply-\(UUID().uuidString)", isDirectory: true)
        let patchURL = patchDirectory.appendingPathComponent("change.patch", isDirectory: false)
        do {
            try FileManager.default.createDirectory(
                at: patchDirectory,
                withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700]
            )
            guard let data = changes.diff.data(using: .utf8) else {
                throw WorkspaceIsolationError.applyFailed("the patch could not be encoded as UTF-8.")
            }
            try data.write(to: patchURL, options: .atomic)
        } catch let error as WorkspaceIsolationError {
            try? FileManager.default.removeItem(at: patchDirectory)
            throw error
        } catch {
            try? FileManager.default.removeItem(at: patchDirectory)
            throw WorkspaceIsolationError.applyFailed(error.localizedDescription)
        }
        defer { try? FileManager.default.removeItem(at: patchDirectory) }

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
        // The snapshot exists for ONE failure: an apply that dies partway
        // through. When the restore itself fails, deleting this directory was
        // deleting the only recovery copy in exactly the state the snapshot
        // was taken for. The error names the path instead.
        var keepRollback = false
        defer {
            if !keepRollback {
                try? FileManager.default.removeItem(at: rollback.directory)
            }
        }

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
            keepRollback = true
            throw WorkspaceIsolationError.applyFailed(
                "the write failed partway through and the rollback did NOT fully restore the tree. The files captured before the apply are kept at '\(rollback.directory.path)'; inspect the tree and that snapshot before retrying." + suffix
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
    /// The entry's own type, never the type of what it points at.
    ///
    /// `FileManager.fileExists` is `stat`, which FOLLOWS symlinks: a tracked
    /// directory symlink (`120000`) read as a directory and was refused as
    /// unwritable, and a dangling symlink read as absent, so the rollback
    /// neither backed it up nor removed it.
    static func entryKind(_ url: URL) -> (exists: Bool, isDirectory: Bool, isSymlink: Bool) {
        var info = stat()
        guard lstat(url.path, &info) == 0 else { return (false, false, false) }
        let mode = info.st_mode & S_IFMT
        return (true, mode == S_IFDIR, mode == S_IFLNK)
    }

    private static func refuseIfTargetsAreNotWritable(_ paths: [String], in root: URL) throws {
        let fileManager = FileManager.default
        for path in paths {
            guard let target = safeURL(for: path, in: root) else {
                throw WorkspaceIsolationError.applyFailed("the patch names an unsafe path: \(path)")
            }
            let kind = entryKind(target)
            if kind.exists {
                // A symlink is replaced at the DIRECTORY level (unlink + create),
                // so its target's writability is irrelevant — and
                // `isWritableFile` follows the link, which refused a tracked
                // DANGLING symlink the patch was about to replace. A directory
                // is still refused: a patch names file paths.
                guard !kind.isDirectory,
                      kind.isSymlink || fileManager.isWritableFile(atPath: target.path) else {
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

    /// The nearest existing ancestor DIRECTORY.
    ///
    /// A path where a same-patch change replaces a file with a directory
    /// (`foo` deleted, `foo/bar` created) has a regular file in its ancestor
    /// chain at preflight time. That is not a reason to refuse: the patch
    /// removes it before the directory exists. Climbing past a non-directory is
    /// safe because `git apply --check` — which runs before this refusal — is
    /// what rejects a path the patch never clears.
    private static func existingAncestor(of url: URL, fileManager: FileManager) -> URL? {
        var candidate = url.deletingLastPathComponent()
        while candidate.path != "/" {
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory) {
                if isDirectory.boolValue { return candidate }
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
        // A git directory is repository METADATA, not content: a change set
        // that names `.git/...` (a hook, a config, an object) would write
        // executable code or repository state through what is supposed to be a
        // file patch. Case-folded because the default macOS filesystem is
        // case-insensitive, so `.GIT/hooks/post-commit` names the same place.
        guard !components.contains(where: { $0.lowercased() == ".git" }) else { return nil }
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
        // The RAW comparison above proves the spelling, not the filesystem.
        // A symlinked directory inside the tree can point outside it, and git
        // would happily write through it; resolve the deepest existing
        // ancestor (and the leaf, when it exists) and refuse when it leaves
        // the resolved root. Both sides are resolved together so `/var` and
        // `/private/var` spellings still compare.
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: root.path) else { return url }
        let resolvedRoot = root.resolvingSymlinksInPath().standardizedFileURL.path
        let resolvedPrefix = resolvedRoot.hasSuffix("/") ? resolvedRoot : resolvedRoot + "/"
        func isInside(_ resolved: String) -> Bool {
            resolved == resolvedRoot || resolved.hasPrefix(resolvedPrefix)
        }
        // The raw component check above only sees the SPELLING. A directory
        // symlink inside the tree can point at `.git` and still resolve inside
        // the root, which made that check unreachable for exactly the path a
        // patch could use to write a hook. Git metadata is refused in the
        // RESOLVED form too.
        func isGitMetadata(_ resolved: String) -> Bool {
            URL(fileURLWithPath: resolved).pathComponents.contains {
                $0.lowercased() == ".git"
            }
        }
        if let ancestor = existingAncestor(of: url, fileManager: fileManager) {
            let resolvedAncestor = ancestor.resolvingSymlinksInPath().standardizedFileURL.path
            guard isInside(resolvedAncestor), !isGitMetadata(resolvedAncestor) else { return nil }
        }
        if fileManager.fileExists(atPath: url.path) {
            let resolvedLeaf = url.resolvingSymlinksInPath().standardizedFileURL.path
            guard isInside(resolvedLeaf), !isGitMetadata(resolvedLeaf) else { return nil }
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
                    // `entryKind`, not `fileExists`: a dangling symlink the
                    // failed patch created reports absent to `stat`, so it was
                    // never removed and the tree was not "as it was".
                    if let original = entry.original {
                        try fileManager.createDirectory(
                            at: target.deletingLastPathComponent(),
                            withIntermediateDirectories: true
                        )
                        if WorkspaceIsolation.entryKind(target).exists {
                            try fileManager.removeItem(at: target)
                        }
                        try fileManager.copyItem(at: original, to: target)
                    } else if WorkspaceIsolation.entryKind(target).exists {
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
            // The snapshot holds the main tree's bytes for the files a patch
            // can touch, so its directory gets the same owner-only floor as a
            // run record. `createDirectory` otherwise makes it 0755.
            try fileManager.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            throw WorkspaceIsolationError.applyFailed(
                "could not prepare a rollback snapshot: \(error.localizedDescription)"
            )
        }
        var entries: [(String, URL?)] = []
        for path in paths {
            guard let target = safeURL(for: path, in: root) else {
                try? fileManager.removeItem(at: directory)
                throw WorkspaceIsolationError.applyFailed("the patch names an unsafe path: \(path)")
            }
            guard entryKind(target).exists else {
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

    /// Applying during a merge, rebase, cherry-pick, revert or bisect is a
    /// different operation than applying to a settled tree, and the failure
    /// would land in the middle of someone else's conflict resolution or
    /// history walk. Refuse by name instead.
    private static func refuseIfTheMainTreeIsMidOperation(_ repositoryRoot: URL) async throws {
        let headOperations: [(ref: String, name: String)] = [
            ("MERGE_HEAD", "merge"),
            ("CHERRY_PICK_HEAD", "cherry-pick"),
            ("REVERT_HEAD", "revert")
        ]
        for operation in headOperations {
            if let (code, _, _) = try? await git(
                ["rev-parse", "-q", "--verify", operation.ref],
                in: repositoryRoot
            ), code == 0 {
                throw WorkspaceIsolationError.applyFailed(
                    "a \(operation.name) is in progress in the main tree. Finish or abort it, then apply again."
                )
            }
        }
        let pathOperations: [(path: String, name: String)] = [
            ("rebase-merge", "rebase"),
            ("rebase-apply", "rebase"),
            ("BISECT_LOG", "bisect")
        ]
        for operation in pathOperations {
            guard let (pathCode, pathOut, _) = try? await git(
                ["rev-parse", "--path-format=absolute", "--git-path", operation.path],
                in: repositoryRoot
            ), pathCode == 0 else { continue }
            let path = pathOut.trimmingCharacters(in: .whitespacesAndNewlines)
            if !path.isEmpty, FileManager.default.fileExists(atPath: path) {
                throw WorkspaceIsolationError.applyFailed(
                    "a \(operation.name) is in progress in the main tree. Finish or abort it, then apply again."
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
    /// removed by hand only when it is under this process's worktree root. The
    /// CONTAINMENT CHECK runs before the first git command, not only before the
    /// manual removal: `git worktree remove --force` names the record's path
    /// too, so a tampered or stale record could otherwise hand git a path
    /// outside this app's root to delete. `root` is a parameter so the launch
    /// sweep, which iterates a root of its own, applies the same gate.
    @discardableResult
    static func cleanup(
        _ worktree: AgentWorktree,
        deleteBranch: Bool = true,
        root: URL = WorkspaceIsolation.processRootURL
    ) async -> Bool {
        let fileManager = FileManager.default
        let contained = isInsideRoot(worktree.path, root: root)
        // A FIFO or symlink `.git` makes every git command in that directory
        // block for the full process timeout (see `worktreeGitEntryState`),
        // so the discard path checks it before running one there — the same
        // gate the launch sweep applies.
        var gitSafe = true
        switch await worktreeGitEntryState(worktree.path) {
        case .regularFile, .missing: break
        case .unreadable, .hostile: gitSafe = false
        }
        // The git commands below run in the RECORD's `repositoryRoot`, which is
        // data. Containment proves the PATH is ours; it says nothing about the
        // repository a forged record could pair it with. Prove the worktree's
        // own `.git` agrees before letting `worktree remove` or `branch -D`
        // touch anything, so a record cannot aim our cleanup at somebody
        // else's checkout.
        var repositoryOwnsWorktree = false
        if contained, gitSafe, let owning = await owningRepository(for: worktree.path) {
            repositoryOwnsWorktree = owning.standardizedFileURL.resolvingSymlinksInPath()
                == worktree.repositoryRoot.standardizedFileURL.resolvingSymlinksInPath()
        }
        if contained && gitSafe && repositoryOwnsWorktree {
            _ = try? await git(
                ["worktree", "remove", "--force", worktree.path.path],
                in: worktree.repositoryRoot
            )
            _ = try? await git(["worktree", "prune"], in: worktree.repositoryRoot)
        }
        if fileManager.fileExists(atPath: worktree.path.path), contained {
            try? fileManager.removeItem(at: worktree.path)
        }
        // The branch is deleted only once the TREE is actually gone. A tree
        // that survived both removals (an immutable file, a corrupt admin
        // entry) is the only copy of the run's work, and `open` requires the
        // branch to re-open it: deleting the branch here would leave commits
        // reachable only through the reflog. The caller's `Bool` reports this,
        // so an apply can say the worktree was kept instead of claiming it was
        // removed.
        let treeGone = !fileManager.fileExists(atPath: worktree.path.path)
        // The branch command runs in the RECORD's repository too, and deletes a
        // ref there, so it takes the same ownership proof as the removals
        // above: a path that is not this process's worktree, or a repository
        // that does not own it, deletes nothing on the record's word.
        if contained,
           repositoryOwnsWorktree,
           treeGone,
           deleteBranch,
           worktree.branch != "HEAD",
           worktree.branch.hasPrefix(branchPrefix) {
            _ = try? await git(["branch", "-D", worktree.branch], in: worktree.repositoryRoot)
        }
        return treeGone
    }

    /// Whether a path may be deleted by the launch sweep or an apply. Only this
    /// process's own worktree root qualifies; a record's path is data.
    static func isInsideProcessWorktreeRoot(_ url: URL) -> Bool {
        isInsideRoot(url, root: processRootURL)
    }

    /// Whether a recorded commit is a resolved object id (hex, non-empty).
    ///
    /// `AgentWorktree.baseCommit` is interpolated into git's argv, and git
    /// accepts option-looking operands there (`--output=<path>` for `diff`), so
    /// a value read from a record must be proven to be an id before it is
    /// passed. `createWorktree` and `open` resolve it with `rev-parse`; this is
    /// the check for every other source.
    static func isResolvedCommitID(_ value: String) -> Bool {
        !value.isEmpty && value.count <= 64 && value.allSatisfy(\.isHexDigit)
    }

    /// Every path a unified diff touches, both sides of a rename.
    ///
    /// Parsed by git itself (`apply --numstat`, which reads the patch and writes
    /// nothing) rather than by hand: the patch format has quoting, binary
    /// hunks and rename headers that a hand parser gets subtly wrong exactly
    /// where it matters — deciding which files a preflight and a rollback have
    /// to cover.
    ///
    /// THROWS rather than answering an empty set when the patch cannot be read:
    /// the caller uses this to cross-check the patch against the change set's
    /// file list, and an empty set there means "nothing to check", so a failure
    /// silently skipped the check on exactly the change set nobody could vouch
    /// for. An empty PATCH is still a legitimate empty answer — git reads it
    /// happily and reports no files.
    static func filesTouched(byPatch patch: String) async throws -> Set<String> {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-numstat-\(UUID().uuidString)", isDirectory: true)
        let url = directory.appendingPathComponent("change.patch", isDirectory: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700]
            )
            try patch.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            throw WorkspaceIsolationError.applyFailed(
                "the patch could not be staged to read its file list: \(error.localizedDescription)"
            )
        }
        let (code, out, err): (Int32, String, String)
        do {
            // `-z` for the same reason the snapshot uses it: without it git
            // C-quotes any path it considers unusual (`"caf\303\251.txt"`,
            // escaped quotes and backslashes), and the cross-check below would
            // reject a valid patch because the two sides spell the name
            // differently. It also drops the rename arrow. NOTE: `apply
            // --numstat -z` emits only the DESTINATION of a rename
            // (`0\t0\tnew.txt\0`, measured), unlike `git diff --numstat -z`,
            // which emits both paths as separate NUL fields. The other side is
            // not lost from the check: the caller compares this set with the
            // change set's own `--name-status -z` list, which carries both.
            (code, out, err) = try await git(["apply", "--numstat", "-z", url.path], in: directory)
        } catch {
            throw WorkspaceIsolationError.applyFailed(
                "the patch's file list could not be read: \(error.localizedDescription)"
            )
        }
        guard code == 0 else {
            throw WorkspaceIsolationError.applyFailed(
                "the patch's file list could not be read: \(err.trimmingCharacters(in: .whitespacesAndNewlines))"
            )
        }
        // One NUL-terminated record per path: "added\tdeleted\tpath\0", with a
        // binary file's counts as dashes.
        var files = Set<String>()
        for record in out.split(separator: "\0", omittingEmptySubsequences: true) {
            let fields = record.split(separator: "\t", maxSplits: 2, omittingEmptySubsequences: false)
            guard fields.count == 3 else { continue }
            let path = String(fields[2])
            if !path.isEmpty { files.insert(path) }
        }
        return files
    }

    static func isInsideRoot(_ url: URL, root: URL) -> Bool {
        let root = root.standardizedFileURL.resolvingSymlinksInPath().path
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        guard !root.isEmpty else { return false }
        return path.hasPrefix(root.hasSuffix("/") ? root : root + "/")
    }

    /// Commits the worktree's branch holds beyond the run's base commit.
    ///
    /// A patch carries the WORKING TREE only, so a run that committed before
    /// it stopped has work no apply will land. The branch is the only copy of
    /// it, and `cleanup` deletes the branch; nil means git could not answer
    /// (a record without a base, a ref that no longer exists), which callers
    /// treat as "do not assume it is empty".
    static func commitsAheadOfBase(_ worktree: AgentWorktree) async -> Int? {
        guard !worktree.baseCommit.isEmpty, !worktree.branch.isEmpty else { return nil }
        // Both operands are read from a record and interpolated into git's argv,
        // and git accepts option-looking operands there (`--output=<path>` for
        // `diff`). The base must be a resolved id; the branch is checked
        // against the same shape the branch commands require.
        guard isResolvedCommitID(worktree.baseCommit),
              worktree.branch.hasPrefix(branchPrefix) else { return nil }
        guard let (code, out, _) = try? await git(
            ["rev-list", "--count", "\(worktree.baseCommit)..\(worktree.branch)"],
            in: worktree.repositoryRoot
        ), code == 0 else { return nil }
        return Int(out.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Commits on the worktree's branch that are NOT reachable from the
    /// worktree's `HEAD`.
    ///
    /// `snapshot` diffs `baseCommit` against the working tree on top of `HEAD`,
    /// so every commit on `branch` that is an ancestor of `HEAD` (the normal
    /// case when a run commits on its own branch) is ALREADY carried by the
    /// patch and does not need the branch kept after apply. Only if the run
    /// switched away from `branch` after committing on it does `HEAD..branch`
    /// report uncarried commits (or `nil` if git could not compare).
    static func commitsOnBranchOutsideHead(_ worktree: AgentWorktree) async -> Int? {
        guard !worktree.branch.isEmpty else { return nil }
        guard let (code, out, _) = try? await git(
            ["rev-list", "--count", "HEAD..refs/heads/\(worktree.branch)"],
            in: worktree.path
        ), code == 0 else { return nil }
        return Int(out.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Files the run created that git ignores. `snapshot` registers intent to
    /// add, which cannot carry them either, so an ignored-only run looks empty
    /// and would otherwise be reported as "no changes" and deleted with its
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
                let path = String(entry.dropFirst(3))
                // A run that built or tested leaves `.build/`, `node_modules/`
                // and friends behind. Naming those as "work a patch cannot
                // carry" kept the record and its multi-GB tree exempt from
                // retention forever, with no clock and no path back; a stray
                // `.DS_Store` was enough. They are re-creatable artifacts, not
                // the run's answer.
                return isDisposableBuildArtifact(path) ? nil : path
            }
    }

    /// The ignored entries a build or test run produces by itself.
    ///
    /// Deliberately a named list of TOOLCHAIN OUTPUT directories: a run-written
    /// `.env.local`, fixture or data file is still reported and still keeps the
    /// worktree. A component match, because `git status` names the directory
    /// (`.build/`) rather than every file under it, and a nested
    /// `Package/.build` is the same artifact.
    static func isDisposableBuildArtifact(_ path: String) -> Bool {
        let disposable: Set<String> = [
            ".build",
            ".swiftpm",
            "DerivedData",
            "xcuserdata",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".gradle",
            ".dart_tool",
            ".DS_Store"
        ]
        return path.split(separator: "/").contains { disposable.contains(String($0)) }
    }

    /// Whether git itself agrees the worktree has nothing to land.
    ///
    /// `snapshot` registers intent to add for every untracked path, so an
    /// unchanged tree reports no status entries and a changed one still does
    /// even when its diff could not be decoded. Cleanup is destructive, and an
    /// empty `WorkspaceChangeSet` is not proof: it is also what a decode
    /// failure produces (see `snapshot`). Anything other than a successful,
    /// empty status is treated as "do not delete".
    /// The worktree's untracked entries that a patch cannot carry, so deleting
    /// the tree after an apply would destroy them.
    ///
    /// `ignoredFiles` is not enough: an untracked NESTED GIT REPOSITORY (a run
    /// that did `git init subrepo`) is not ignored, and `git add -N .` skips it
    /// because it has its own (possibly unborn) HEAD — so it never enters the
    /// patch, and the apply's cleanup deleted it after landing the tracked
    /// diff. Anything left under `?? ` after the snapshot was taken is work
    /// only this directory holds.
    static func untrackedLeftovers(_ worktree: AgentWorktree) async -> [String] {
        guard let (code, stdout, _) = try? await git(
            ["status", "--porcelain", "-z", "--untracked-files=normal"],
            in: worktree.path
        ), code == 0 else { return [] }
        return stdout
            .split(separator: "\0", omittingEmptySubsequences: true)
            .compactMap { entry -> String? in
                guard entry.hasPrefix("?? ") else { return nil }
                let path = String(entry.dropFirst(3))
                return isDisposableBuildArtifact(path) ? nil : path
            }
    }

    /// Whether the worktree holds nothing the operator could lose, checked
    /// without WRITING to it.
    ///
    /// It used to `git reset -q -- .` first, to clear intent-to-add entries an
    /// earlier `snapshot` or `diffStat` left behind (`git add -N .`: a removed
    /// intent-to-add file reports ` D <file>` though the tree matches HEAD).
    /// That reset also unstages real staged work, and since `snapshot` diffs
    /// base against the WORKING TREE, a run whose only work was staged is
    /// invisible to the patch — so the reset erased it, this read clean, and
    /// the apply's "no changes" path deleted the worktree and branch.
    ///
    /// The index is now left alone, which means a leftover intent-to-add entry
    /// can make a genuinely clean tree read as dirty. That is the safe
    /// direction: the caller throws and KEEPS the worktree, costing an
    /// operator one manual clear, where the old false "clean" cost them the
    /// run's only copy of its work.
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
        var uncontained: [String] = []
        for name in names {
            // One entry's discard claim, if it took one. Released at the end of
            // the entry (or on any `continue`), never before the removal.
            var discardClaim: UUID?
            defer {
                if let discardClaim {
                    Task { @MainActor in SubagentApplyService.endDiscard(discardClaim) }
                }
            }
            let path = root.appendingPathComponent(name, isDirectory: true)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: path.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else { continue }
            let values = try? path.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
            let created = values?.creationDate ?? values?.contentModificationDate ?? now
            guard now.timeIntervalSince(created) >= olderThan else { continue }

            // A worktree is the ONLY full copy of a run's work when the run
            // was never applied, or was applied while keeping git-ignored
            // files a patch could not carry. Age is not evidence that the
            // work is safe to destroy: the record keeps a bounded diff, not
            // the files. Leave it alone until the operator clears the run —
            // including a CANCELLED run's tree, which a Stop must not turn
            // into the one path that destroys work.
            //
            // A record this build CANNOT decode is not an absent record: the
            // store deliberately keeps a newer-schema or corrupt file, and
            // that file is the worktree's only index. Probe for the file
            // itself before treating a UUID-named tree as orphaned.
            if let runId = UUID(uuidString: name) {
                if let record = AgentRunStore.shared.loadAny(runId: runId) {
                    if record.holdsWorktreeWork { continue }
                } else if AgentRunStore.shared.hasRecordFile(runId: runId) {
                    continue
                }
                // An apply that is running RIGHT NOW has opened this tree and
                // is reading it; a sweep under it would delete the files out
                // from under the patch. A discard owns it for the length of
                // its removal, and the two must not race either. Both claims
                // are MainActor state, so the question is asked there.
                // Taken as a CLAIM, not just read: the git calls below await,
                // so a discard or an apply starting in that window could delete
                // or read the same tree concurrently. Holding the discard slot
                // for the deletion makes the check and the removal one step,
                // and it is what makes a re-check unnecessary: both sides take
                // their claim synchronously on the MainActor, so an apply is
                // either already in flight (seen above, and this entry skipped)
                // or it sees the discard claim and refuses to start.
                let owned = await MainActor.run { () -> Bool in
                    if SubagentApplyService.isRunInFlight(runId) { return true }
                    return !SubagentApplyService.beginDiscard(runId)
                }
                if owned { continue }
                // Released on EVERY exit below and only AFTER the deletion it
                // guards. A `defer` here fires at the end of this `if let`
                // block, i.e. BEFORE the awaits that remove the tree, and an
                // apply starting in that window would have its worktree
                // deleted underneath it.
                discardClaim = runId
            }

            // Nothing below may touch git in this tree until its `.git` is
            // known to be the regular file git writes: a run can rewrite it
            // (it is inside the sandbox), and a FIFO there blocks every git
            // command for the whole process timeout, per call, at launch.
            // `open` and `worktreeLinkageState` check this too; the sweep is
            // the path that had no check at all.
            switch await worktreeGitEntryState(path) {
            case .regularFile, .missing:
                break
            case .unreadable(let why), .hostile(let why):
                DiagLog.warning("workspace", "leaving the worktree at \(name) alone: \(why)")
                continue
            }

            var removedTree = false
            // Both branches below DELETE by this path, so containment is
            // decided once, up front, and every removal is gated on it. The
            // sweep found `path` through a stat (which follows symlinks), so a
            // link planted in the root can name a checkout elsewhere: without
            // this, `git worktree remove --force` on the detached-HEAD branch
            // destroyed a repository that was never a run's.
            let contained = isInsideRoot(path, root: root)
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
                    removedTree = await cleanup(worktree, root: root)
                } else if contained {
                    _ = try? await git(["worktree", "remove", "--force", path.path], in: repositoryRoot)
                    _ = try? await git(["worktree", "prune"], in: repositoryRoot)
                    // A detached HEAD is still a real checkout with the run's
                    // work in it, so the containment check above is what makes
                    // this removal safe to attempt at all.
                    try? fileManager.removeItem(at: path)
                    removedTree = !fileManager.fileExists(atPath: path.path)
                } else {
                    // Outside the root: leave it, and say so once per sweep —
                    // a silent skip reads as "handled".
                    uncontained.append(path.path)
                }
            } else {
                if contained {
                    try? fileManager.removeItem(at: path)
                }
                removedTree = !fileManager.fileExists(atPath: path.path)
            }
            // Reporting a tree that survived makes the sweep's log say work
            // was destroyed when it is still on disk.
            if removedTree {
                removed.append(name)
            }
            // The removal is done, so the claim can go. `defer` at the top of
            // the entry does this on the `continue` paths too; releasing here
            // keeps the successful path explicit about the ordering.
            if let claim = discardClaim {
                discardClaim = nil
                await MainActor.run { SubagentApplyService.endDiscard(claim) }
            }
        }
        if !uncontained.isEmpty {
            DiagLog.warning(
                "subagent.isolation",
                "\(uncontained.count) stale entr\(uncontained.count == 1 ? "y" : "ies") in the worktree root resolve outside it and were left untouched: \(uncontained.joined(separator: ", "))"
            )
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

    /// A best-effort `git diff --stat` for the approval card, bounded so a
    /// change set with hundreds of files cannot turn the card into a wall of
    /// text. Nil when git cannot say (the card then falls back to the recorded
    /// file list).
    ///
    /// `git add -N .` below is the same WRITE `snapshot` makes, and it is
    /// decided by the worktree's `.git` pointer. Reading the stat for a card
    /// ran it before any linkage check, so merely OPENING the apply card in a
    /// retargeted worktree wrote intent-to-add entries into the operator's
    /// index with no approval. The linkage (and HEAD) is verified first; a
    /// worktree that fails it gets no stat at all, and the card falls back to
    /// the recorded file list.
    static func diffStat(_ worktree: AgentWorktree, lineLimit: Int = 14) async -> String? {
        guard (try? await verifyWorktreeLinkage(worktree)) != nil,
              isResolvedCommitID(worktree.baseCommit) else { return nil }
        _ = try? await git(["add", "-N", "."], in: worktree.path)
        guard let (code, out, _) = try? await git(
            ["diff", worktree.baseCommit, "--no-ext-diff", "--no-textconv", "--stat", "--stat-width=78", "--no-color"],
            in: worktree.path
        ), code == 0 else { return nil }
        let lines = out
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else { return nil }
        if lines.count <= lineLimit { return lines.joined(separator: "\n") }
        let kept = lines.prefix(lineLimit).joined(separator: "\n")
        return kept + "\n… and \(lines.count - lineLimit) more line(s)"
    }

    /// How many entries the destination has uncommitted, or nil when git
    /// cannot say. Read-only, so it is safe to ask before anything is written.
    static func uncommittedFileCount(in root: URL) async -> Int? {
        guard let (code, out, _) = try? await git(["status", "--porcelain", "-z"], in: root),
              code == 0 else { return nil }
        return out.split(separator: "\0", omittingEmptySubsequences: true).count
    }

    private static func git(
        _ arguments: [String],
        in directory: URL
    ) async throws -> (Int32, String, String) {
        // An inherited `diff.noprefix=true` strips the `a/` and `b/` prefixes
        // from every patch this file generates, and `git apply --check` then
        // refuses it with "header lacks filename information". An inherited
        // `GIT_DIR` / `GIT_WORK_TREE` can redirect the apply into another
        // repository entirely, so the child gets neither.
        //
        // The `-c` overrides below close the same hole from the other side: a
        // linked worktree shares the MAIN repository's config file, and a run
        // can write it from inside its own worktree. `core.fsmonitor` and
        // hooks are then EXECUTED as commands by the plain read-only git calls
        // this file makes in the operator's tree (status, diff, add), which is
        // code execution outside the sandbox that needs no shell escape at all.
        // `filter.<n>.clean` is the same class and cannot be enumerated away,
        // so the destructive calls also avoid `git add` where they can.
        var environment = ControlledProcessRunner.toolEnvironment(
            scoped: AgentHarness.executingInRunScope
        )
        for key in environment.keys where key.hasPrefix("GIT_") {
            environment.removeValue(forKey: key)
        }
        var arguments = arguments
        arguments.insert(
            contentsOf: [
                "-c", "diff.noprefix=false",
                // The default prefix is what `diffDestination` parses, and a
                // user's `diff.mnemonicPrefix=true` replaces it with letters
                // keyed to the invocation (`c/`, `w/`, …). Pinning it keeps a
                // personal gitconfig from making an apply look like a
                // different patch.
                "-c", "diff.mnemonicPrefix=false",
                "-c", "core.fsmonitor=false",
                "-c", "core.hooksPath=/dev/null",
                "-c", "core.untrackedCache=false",
            ],
            at: 0
        )
        return try await AgentHarness.runProcess(
            executableURL: gitURL,
            arguments: arguments,
            currentDirectory: directory,
            timeoutSeconds: 300,
            environment: environment
        )
    }
}
