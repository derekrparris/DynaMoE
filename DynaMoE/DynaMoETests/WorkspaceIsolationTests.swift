//
//  WorkspaceIsolationTests.swift
//  DynaMoETests
//
//  Phase 4's worktree lifecycle: create, snapshot, apply, clean up. Every case
//  builds a throwaway git repository under the temporary directory, so no test
//  touches the developer's repository or the operator's real worktree root.
//

import XCTest
@testable import DynaMoE

@MainActor
final class WorkspaceIsolationTests: XCTestCase {
    private var repository: URL!
    private var worktreeRoot: URL!

    override func setUp() async throws {
        repository = try makeTemporaryDirectory("repo")
        // Under the process worktree root: `cleanup` refuses to delete a path
        // outside it (a record's path is data), so a fixture that lived in a
        // bare temp directory would no longer be cleaned up.
        let root = WorkspaceIsolation.processRootURL
            .appendingPathComponent("tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        worktreeRoot = root
        try await git(["init", "-b", "main"], in: repository)
        try "line one\nline two\nline three\n".write(
            to: repository.appendingPathComponent("README.md"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: repository)
        try await git(["commit", "-m", "init"], in: repository)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: repository)
        try? FileManager.default.removeItem(at: worktreeRoot)
    }

    // MARK: - Creating

    func testCreatesAWorktreeOnItsOwnBranch() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))
        XCTAssertEqual(worktree.repositoryRoot.standardizedFileURL, repository.standardizedFileURL)
        XCTAssertEqual(worktree.branch, WorkspaceIsolation.branchPrefix + String(runId.uuidString.prefix(12)).lowercased())

        let head = try await git(["rev-parse", "--abbrev-ref", "HEAD"], in: worktree.path)
        XCTAssertEqual(head.0, 0)
        XCTAssertEqual(head.1.trimmingCharacters(in: .whitespacesAndNewlines), worktree.branch)

        let base = try await git(["rev-parse", "HEAD"], in: repository)
        XCTAssertEqual(worktree.baseCommit, base.1.trimmingCharacters(in: .whitespacesAndNewlines))
        XCTAssertEqual(worktree.isolation.worktreePath, worktree.path.path)

        let removed = await WorkspaceIsolation.cleanup(worktree)
        XCTAssertTrue(removed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: worktree.path.path))
    }

    /// A tree that survives both removals must NOT lose its branch.
    ///
    /// The branch is what `open` re-derives a worktree from; deleting it left a
    /// surviving tree that no apply could read, with its commits reachable only
    /// through the reflog. Cleanup also has to SAY the tree survived, because
    /// the apply tool used to report "worktree removed" unconditionally.
    func testCleanupKeepsTheBranchWhenTheWorktreeSurvives() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        // Deny the parent directory so the final unlink cannot happen, while
        // git is still able to clear the tree's contents.
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o500],
            ofItemAtPath: worktreeRoot.path
        )
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o700],
                ofItemAtPath: worktreeRoot.path
            )
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
        }

        let removed = await WorkspaceIsolation.cleanup(worktree)
        XCTAssertFalse(removed, "a tree that survived both removals must report false")
        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))

        let branch = try await git(["rev-parse", "--verify", worktree.branch], in: repository)
        XCTAssertEqual(
            branch.0,
            0,
            "the branch must outlive a surviving tree so the worktree can be re-opened"
        )
    }

    /// Build and test output is not the run's answer. Naming `.build/`,
    /// `node_modules/` or a `.DS_Store` as "work a patch cannot carry" kept a
    /// run's multi-GB worktree exempt from retention forever, with no clock.
    func testOrdinaryBuildOutputIsNotKeptWork() {
        for path in [
            ".build",
            ".build/debug/foo",
            "Package/.build/repositories",
            "node_modules",
            "node_modules/left-pad/index.js",
            "__pycache__",
            "src/__pycache__/x.pyc",
            ".DS_Store",
            "sub/.DS_Store",
            "DerivedData/Build",
            ".pytest_cache/v/cache"
        ] {
            XCTAssertTrue(
                WorkspaceIsolation.isDisposableBuildArtifact(path),
                "\(path) is re-creatable output, not kept work"
            )
        }
        for path in [
            ".env.local",
            "notes.txt",
            "fixtures/data.json",
            "config/production.yaml",
            "build.log"
        ] {
            XCTAssertFalse(
                WorkspaceIsolation.isDisposableBuildArtifact(path),
                "\(path) may be the run's only copy"
            )
        }
    }

    func testCreatingTwiceAtTheSameRunIsRefused() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        do {
            _ = try await WorkspaceIsolation.createWorktree(
                repositoryRoot: repository,
                runId: runId,
                root: worktreeRoot
            )
            XCTFail("a second worktree at the same path must be refused")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeExists(let path) = error else {
                return XCTFail("expected worktreeExists, got \(error)")
            }
            XCTAssertEqual(path, worktree.path.path)
        }
        _ = await WorkspaceIsolation.cleanup(worktree)
    }

    func testANonRepositoryIsRefused() async throws {
        let directory = try makeTemporaryDirectory("not-a-repo")
        defer { try? FileManager.default.removeItem(at: directory) }
        let isRepository = await WorkspaceIsolation.isGitRepository(at: directory)
        let root = await WorkspaceIsolation.repositoryRoot(for: directory)
        XCTAssertFalse(isRepository)
        XCTAssertNil(root)
    }

    // MARK: - Snapshots

    func testSnapshotReportsEditedAddedAndCommittedWork() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        var readme = try String(contentsOf: worktree.path.appendingPathComponent("README.md"), encoding: .utf8)
        readme += "edited in the worktree\n"
        try readme.write(to: worktree.path.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try "brand new\n".write(
            to: worktree.path.appendingPathComponent("new.txt"),
            atomically: true,
            encoding: .utf8
        )
        // The implementer preset can commit. The snapshot must be relative to
        // the commit the worktree branched from, so a committed file still lands.
        try "committed work\n".write(
            to: worktree.path.appendingPathComponent("committed.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: worktree.path)
        try await git(["commit", "-m", "worktree work"], in: worktree.path)

        let changes = try await WorkspaceIsolation.snapshot(worktree)
        XCTAssertEqual(changes.changedFiles, ["README.md", "committed.txt", "new.txt"])
        XCTAssertTrue(changes.diff.contains("edited in the worktree"))
        XCTAssertTrue(changes.diff.contains("committed work"))
        XCTAssertTrue(changes.diff.contains("brand new"))
        XCTAssertFalse(changes.diffWasTruncated)
    }

    func testAnUnchangedWorktreeSnapshotsEmpty() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let changes = try await WorkspaceIsolation.snapshot(worktree)
        XCTAssertTrue(changes.isEmpty)
        XCTAssertEqual(changes.changedFiles, [])
        XCTAssertTrue(changes.diff.isEmpty)
        let clean = await WorkspaceIsolation.hasNoWorkingTreeChanges(worktree)
        XCTAssertTrue(
            clean,
            "cleanup may only trust an empty change set once git agrees the tree is clean"
        )
    }

    /// A diff of a non-UTF-8 text file fails `runProcess`'s UTF-8 decode and
    /// came back as an empty string. The name list still had the file, so the
    /// "patch" was empty: `git apply` answered "No valid patches" forever, and
    /// the empty change set also authorized deleting the only copy of the run's
    /// work.
    func testASnapshotRefusesAChangedFileItCannotRepresent() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        // Latin-1 "café\n" is text to git (no NUL) and invalid UTF-8 to Swift.
        let latin1 = Data([0x63, 0x61, 0x66, 0xE9, 0x0A])
        try latin1.write(to: worktree.path.appendingPathComponent("latin1.txt"))

        do {
            _ = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
            XCTFail("a changed file with no representable patch must not look empty")
        } catch let error as WorkspaceIsolationError {
            guard case .snapshotFailed(let detail) = error else {
                return XCTFail("expected snapshotFailed, got \(error)")
            }
            XCTAssertTrue(detail.contains("patch"), detail)
        }
        let stillDirty = await WorkspaceIsolation.hasNoWorkingTreeChanges(worktree)
        XCTAssertFalse(
            stillDirty,
            "git still reports the change, so nothing may be deleted as 'no changes'"
        )
    }

    func testARecordedBaseCommitIgnoresARewrittenMainBranch() async throws {
        try "second\n".write(
            to: repository.appendingPathComponent("second.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: repository)
        try await git(["commit", "-m", "second"], in: repository)
        let fork = try await git(["rev-parse", "HEAD"], in: repository)
        let forkCommit = fork.1.trimmingCharacters(in: .whitespacesAndNewlines)

        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        XCTAssertEqual(worktree.baseCommit, forkCommit)
        try "worktree change\n".write(
            to: worktree.path.appendingPathComponent("worktree.txt"),
            atomically: true,
            encoding: .utf8
        )

        // Rewind the main branch below the fork point and move it somewhere
        // else. The branch the worktree is on still starts at `forkCommit`, so
        // only `merge-base` moves.
        let rootResult = try await git(["rev-list", "--max-parents=0", "HEAD"], in: repository)
        let rootCommit = rootResult.1.trimmingCharacters(in: .whitespacesAndNewlines)
        try await git(["reset", "--hard", rootCommit], in: repository)
        try "main moved\n".write(
            to: repository.appendingPathComponent("moved.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: repository)
        try await git(["commit", "-m", "moved"], in: repository)

        let recorded = try await WorkspaceIsolation.open(
            path: worktree.path.path,
            branch: worktree.branch,
            repositoryRoot: repository,
            baseCommit: worktree.baseCommit
        )
        XCTAssertEqual(recorded.baseCommit, forkCommit)
        let changes = try await WorkspaceIsolation.snapshot(recorded)
        XCTAssertEqual(
            changes.changedFiles,
            ["worktree.txt"],
            "the recorded base must keep the main branch's rewrite out of the run's diff"
        )

        let derived = try await WorkspaceIsolation.open(
            path: worktree.path.path,
            branch: worktree.branch,
            repositoryRoot: repository
        )
        XCTAssertNotEqual(
            derived.baseCommit,
            forkCommit,
            "a recomputed merge base is what the record exists to avoid"
        )
        _ = await WorkspaceIsolation.cleanup(worktree)
    }

    /// The apply path snapshots the main tree before writing and restores it on
    /// a partial failure. The rollback is what makes "the write failed" mean
    /// "your tree is unchanged", so it is exercised directly.
    func testARollbackSnapshotRestoresExistingFilesAndRemovesNewOnes() throws {
        let readme = repository.appendingPathComponent("README.md")
        let original = try String(contentsOf: readme, encoding: .utf8)
        let added = repository.appendingPathComponent("added.txt")
        let rollback = try WorkspaceIsolation.stageRollback(["README.md", "added.txt"], in: repository)
        defer { try? FileManager.default.removeItem(at: rollback.directory) }

        try "partially applied\n".write(to: readme, atomically: true, encoding: .utf8)
        try "created by the failed apply\n".write(to: added, atomically: true, encoding: .utf8)

        XCTAssertTrue(rollback.restore(in: repository))
        XCTAssertEqual(try String(contentsOf: readme, encoding: .utf8), original)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: added.path),
            "a file the failed apply created must be removed again"
        )
    }

    /// `safeURL` is the containment preflight for every path a patch touches.
    /// It used to compare raw spellings only, so a symlinked directory inside
    /// the tree could send a write outside it. The traversal and both symlink
    /// forms are checked here; git refusing a symlinked write was the only
    /// thing standing behind this before.
    func testSafeURLRefusesTraversalAndSymlinksThatLeaveTheTree() throws {
        let fileManager = FileManager.default
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-safeurl-\(UUID().uuidString)", isDirectory: true)
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-safeurl-outside-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: outside, withIntermediateDirectories: true)
        defer {
            try? fileManager.removeItem(at: root)
            try? fileManager.removeItem(at: outside)
        }
        try "outside\n".write(to: outside.appendingPathComponent("target.txt"), atomically: true, encoding: .utf8)

        XCTAssertEqual(
            WorkspaceIsolation.safeURL(for: "src/new/file.txt", in: root)?.path,
            root.appendingPathComponent("src/new/file.txt").path
        )
        XCTAssertNil(WorkspaceIsolation.safeURL(for: "../escape.txt", in: root))
        XCTAssertNil(WorkspaceIsolation.safeURL(for: "/etc/passwd", in: root))

        // A directory symlink escaping the tree.
        try fileManager.createSymbolicLink(
            at: root.appendingPathComponent("link"),
            withDestinationURL: outside
        )
        XCTAssertNil(
            WorkspaceIsolation.safeURL(for: "link/target.txt", in: root),
            "a write through a symlinked directory must not leave the tree"
        )
        XCTAssertNil(
            WorkspaceIsolation.safeURL(for: "link/new.txt", in: root),
            "a file the patch would CREATE through the symlink must be refused too"
        )

        // A leaf symlink escaping the tree.
        try fileManager.createSymbolicLink(
            at: root.appendingPathComponent("leaf.txt"),
            withDestinationURL: outside.appendingPathComponent("target.txt")
        )
        XCTAssertNil(
            WorkspaceIsolation.safeURL(for: "leaf.txt", in: root),
            "a write through a leaf symlink must not leave the tree"
        )

        // Repository METADATA is not content. A change set that names `.git/`
        // would write executable hooks or config through a surface that is
        // supposed to carry a file patch, and the default macOS filesystem is
        // case-insensitive, so `.GIT` names the same place.
        XCTAssertNil(WorkspaceIsolation.safeURL(for: ".git/hooks/post-commit", in: root))
        XCTAssertNil(WorkspaceIsolation.safeURL(for: ".GIT/config", in: root))
        XCTAssertNil(WorkspaceIsolation.safeURL(for: "sub/.git/config", in: root))
        XCTAssertNotNil(
            WorkspaceIsolation.safeURL(for: "src/.gitignore", in: root),
            "a name that merely starts with .git is ordinary content"
        )
        XCTAssertNotNil(WorkspaceIsolation.safeURL(for: "nested/.gitattributes", in: root))

        // A symlink is the way the raw `.git` component check could be
        // bypassed: `metadata/hooks/...` names no `.git` component but
        // RESOLVES into the repository's metadata, which is still a hook write.
        try fileManager.createDirectory(
            at: root.appendingPathComponent(".git/hooks", isDirectory: true),
            withIntermediateDirectories: true
        )
        try fileManager.createSymbolicLink(
            at: root.appendingPathComponent("metadata"),
            withDestinationURL: root.appendingPathComponent(".git", isDirectory: true)
        )
        XCTAssertNil(
            WorkspaceIsolation.safeURL(for: "metadata/hooks/post-commit", in: root),
            "a path that RESOLVES into .git is metadata even when its spelling is not"
        )
        XCTAssertNil(
            WorkspaceIsolation.safeURL(for: "metadata", in: root),
            "and the symlink itself must not be writable as content"
        )
    }

    /// A C-quoted `diff --git` header is legal whenever a path holds a quote, a
    /// backslash or a non-ASCII byte. Reading only the last ` b/` returned the
    /// quotes and escapes as part of the path, so the nested-repository refusal
    /// named a file git would never write.
    func testAQuotedDiffHeaderStillResolvesItsDestination() {
        let patch = #"""
        diff --git "a/README \"notes\".md" "b/README \"notes\".md"
        new file mode 160000
        index 0000000..1111111
        --- /dev/null
        +++ "b/README \"notes\".md"
        """#
        XCTAssertEqual(
            WorkspaceIsolation.nestedRepositoryPath(inPatch: patch),
            "README \"notes\".md"
        )
    }

    /// The rollback exists for an apply that dies PARTWAY through, and the
    /// helper test alone cannot prove the real `git apply` path reaches it.
    /// This patch passes `git apply --check` and then fails while writing its
    /// second file (the first file is in the way of the second's directory), so
    /// it is the exact shape the snapshot is taken for: one file created, then
    /// an error. The main tree must come back with nothing new in it.
    func testAPartialGitApplyRollsTheTreeBack() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let changes = WorkspaceChangeSet(
            changedFiles: ["partial-file", "partial-file/child"],
            diff: """
            diff --git a/partial-file b/partial-file
            new file mode 100644
            index 0000000..3e75765
            --- /dev/null
            +++ b/partial-file
            @@ -0,0 +1 @@
            +first
            diff --git a/partial-file/child b/partial-file/child
            new file mode 100644
            index 0000000..3e75765
            --- /dev/null
            +++ b/partial-file/child
            @@ -0,0 +1 @@
            +second
            """ + "\n",
            diffWasTruncated: false
        )

        do {
            _ = try await WorkspaceIsolation.apply(worktree, changeSet: changes)
            XCTFail("the second file cannot be written under the first, so the apply must fail")
        } catch let error as WorkspaceIsolationError {
            guard case .applyFailed = error else {
                return XCTFail("expected applyFailed, got \(error)")
            }
        }

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("partial-file").path),
            "the file the failed apply had already created must be removed again"
        )
        let status = try await git(["status", "--porcelain"], in: repository)
        XCTAssertEqual(status.1, "", "the main tree must be exactly as it was")
    }

    func testSnapshotBoundsOnlyItsDisplayCopy() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let long = String(repeating: "x", count: 4_000) + "\n"
        try long.write(
            to: worktree.path.appendingPathComponent("long.txt"),
            atomically: true,
            encoding: .utf8
        )

        let bounded = try await WorkspaceIsolation.snapshot(worktree, diffLimit: 100)
        XCTAssertTrue(bounded.diffWasTruncated)
        XCTAssertTrue(bounded.diff.count < 500)
        XCTAssertEqual(bounded.changedFiles, ["long.txt"])

        let full = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
        XCTAssertFalse(full.diffWasTruncated)
        XCTAssertTrue(full.diff.contains(long))
    }

    // MARK: - Applying

    func testApplyLandsTheDiffOnTheMainTree() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        var readme = try String(contentsOf: worktree.path.appendingPathComponent("README.md"), encoding: .utf8)
        readme += "applied line\n"
        try readme.write(to: worktree.path.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try "new file\n".write(
            to: worktree.path.appendingPathComponent("new.txt"),
            atomically: true,
            encoding: .utf8
        )

        let opened = try await WorkspaceIsolation.open(worktree.isolation)
        XCTAssertEqual(opened.repositoryRoot.standardizedFileURL, repository.standardizedFileURL)
        XCTAssertEqual(opened.baseCommit, worktree.baseCommit)

        let result = try await WorkspaceIsolation.apply(opened)
        XCTAssertEqual(result.changedFiles, ["README.md", "new.txt"])
        XCTAssertEqual(result.branch, worktree.branch)

        let mainReadme = try String(contentsOf: repository.appendingPathComponent("README.md"), encoding: .utf8)
        XCTAssertTrue(mainReadme.contains("applied line"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: repository.appendingPathComponent("new.txt").path))

        // Applying is not a commit: the coordinator decides what history the
        // changes get.
        let status = try await git(["status", "--porcelain"], in: repository)
        XCTAssertTrue(status.1.contains("README.md"))
        XCTAssertTrue(status.1.contains("new.txt"))
    }

    func testAConflictingApplyLeavesTheMainTreeUntouched() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        var worktreeReadme = try String(contentsOf: worktree.path.appendingPathComponent("README.md"), encoding: .utf8)
        worktreeReadme = worktreeReadme.replacingOccurrences(of: "line two", with: "worktree line two")
        try worktreeReadme.write(
            to: worktree.path.appendingPathComponent("README.md"),
            atomically: true,
            encoding: .utf8
        )

        let mainReadmeURL = repository.appendingPathComponent("README.md")
        var mainReadme = try String(contentsOf: mainReadmeURL, encoding: .utf8)
        mainReadme = mainReadme.replacingOccurrences(of: "line two", with: "main line two")
        try mainReadme.write(to: mainReadmeURL, atomically: true, encoding: .utf8)

        do {
            _ = try await WorkspaceIsolation.apply(worktree)
            XCTFail("a conflicting patch must not be applied")
        } catch let error as WorkspaceIsolationError {
            guard case .applyConflicted = error else {
                return XCTFail("expected applyConflicted, got \(error)")
            }
        }

        let after = try String(contentsOf: mainReadmeURL, encoding: .utf8)
        XCTAssertTrue(after.contains("main line two"))
        XCTAssertFalse(after.contains("worktree line two"))
    }

    /// `git apply --check` only reads, so a read-only destination passed the
    /// check and then failed during the real apply, leaving the files it had
    /// already written in place. The pre-flight must refuse before any write.
    func testAnUnwritableTargetLeavesEveryFileUntouched() async throws {
        let marker = "stable line\n"
        try marker.write(
            to: repository.appendingPathComponent("applied.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: repository)
        try await git(["commit", "-m", "add applied"], in: repository)

        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        var readme = try String(contentsOf: worktree.path.appendingPathComponent("README.md"), encoding: .utf8)
        readme += "should not land\n"
        try readme.write(to: worktree.path.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try "changed\n".write(
            to: worktree.path.appendingPathComponent("applied.txt"),
            atomically: true,
            encoding: .utf8
        )

        let appliedURL = repository.appendingPathComponent("applied.txt")
        try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: appliedURL.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: appliedURL.path) }

        do {
            _ = try await WorkspaceIsolation.apply(worktree)
            XCTFail("an unwritable destination must be refused")
        } catch let error as WorkspaceIsolationError {
            guard case .applyFailed = error else {
                return XCTFail("expected applyFailed, got \(error)")
            }
        }

        let mainReadme = try String(contentsOf: repository.appendingPathComponent("README.md"), encoding: .utf8)
        XCTAssertFalse(mainReadme.contains("should not land"), "no file may be written before the refusal")
        XCTAssertEqual(
            try String(contentsOf: appliedURL, encoding: .utf8),
            marker,
            "the unwritable file must keep its original bytes"
        )
    }

    func testOpenRefusesAPathThatIsGone() async throws {
        do {
            _ = try await WorkspaceIsolation.open(
                path: worktreeRoot.appendingPathComponent(UUID().uuidString).path,
                branch: "dynamoe/agent/deadbeef"
            )
            XCTFail("a missing worktree must be refused")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeMissing = error else {
                return XCTFail("expected worktreeMissing, got \(error)")
            }
        }
    }

    func testTwoWriterWorktreesAreIndependent() async throws {
        let first = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let second = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        XCTAssertNotEqual(first.path, second.path)
        XCTAssertNotEqual(first.branch, second.branch)

        try "first\n".write(
            to: first.path.appendingPathComponent("first.txt"),
            atomically: true,
            encoding: .utf8
        )
        try "second\n".write(
            to: second.path.appendingPathComponent("second.txt"),
            atomically: true,
            encoding: .utf8
        )

        let firstChanges = try await WorkspaceIsolation.snapshot(first)
        let secondChanges = try await WorkspaceIsolation.snapshot(second)
        XCTAssertEqual(firstChanges.changedFiles, ["first.txt"])
        XCTAssertEqual(secondChanges.changedFiles, ["second.txt"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: repository.appendingPathComponent("first.txt").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: repository.appendingPathComponent("second.txt").path))

        _ = await WorkspaceIsolation.cleanup(first)
        _ = await WorkspaceIsolation.cleanup(second)
    }

    // MARK: - Sweeping

    func testPruneRemovesAWorktreePastItsAge() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))

        // Every directory is "older than -1 seconds", which is how the sweep's
        // age rule is exercised without touching file dates.
        let removed = await WorkspaceIsolation.pruneStaleWorktrees(
            root: worktreeRoot,
            olderThan: -1
        )
        XCTAssertEqual(removed, [runId.uuidString])
        XCTAssertFalse(FileManager.default.fileExists(atPath: worktree.path.path))
    }

    func testTheSweepKeepsACancelledUnappliedWorktree() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        // A cancelled run's record with no appliedAt: the worktree is the only
        // full copy of whatever the run wrote before its Stop, and the sweep
        // used to treat "cancelled" as cleanup's licence to delete it.
        let store = AgentRunStore.shared
        let cancelled = AgentRunRecord(
            id: runId,
            parentSessionId: nil,
            role: "Implementer",
            preset: .implementer,
            task: "stopped mid-write",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .cancelled,
            createdAt: Date(),
            startedAt: Date(),
            completedAt: Date()
        )
        store.saveImmediately(cancelled)
        defer {
            store.delete(runId: runId, sessionId: nil)
            store.flush()
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
        }

        let removed = await WorkspaceIsolation.pruneStaleWorktrees(
            root: worktreeRoot,
            olderThan: -1
        )
        XCTAssertTrue(removed.isEmpty, "a cancelled unapplied worktree must not be swept")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: worktree.path.path),
            "the sweep must keep the only copy of a cancelled run's work"
        )
    }

    func testDiscardIsRefusedWhileAnApplyOwnsTheRun() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        let store = AgentRunStore.shared
        let record = AgentRunRecord(
            id: runId,
            parentSessionId: nil,
            role: "Implementer",
            preset: .implementer,
            task: "mid-apply",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed,
            createdAt: Date(),
            startedAt: Date(),
            completedAt: Date()
        )
        XCTAssertTrue(store.saveImmediately(record))
        defer {
            SubagentApplyService.endApplyForTesting(runId)
            store.delete(runId: runId, sessionId: nil)
            store.flush()
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
        }

        SubagentApplyService.beginApplyForTesting(runId)
        _ = await SubagentManager.shared.discard(runId: runId)

        XCTAssertNotNil(
            store.loadAny(runId: runId),
            "a discard must not delete the record while an apply owns it"
        )
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: worktree.path.path),
            "a discard must not remove the worktree while an apply is reading it"
        )
    }

    func testTheSweepKeepsAWorktreeApplyKeptForItsIgnoredFiles() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        let store = AgentRunStore.shared
        let applied = AgentRunRecord(
            id: runId,
            parentSessionId: nil,
            role: "Implementer",
            preset: .implementer,
            task: "landed a tracked change and kept a .env",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed,
            createdAt: Date(),
            startedAt: Date(),
            completedAt: Date(),
            appliedAt: Date(),
            appliedToBranch: "main",
            keptIgnoredFiles: [".env.local"]
        )
        XCTAssertTrue(store.saveImmediately(applied))
        defer {
            store.delete(runId: runId, sessionId: nil)
            store.flush()
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
        }

        let removed = await WorkspaceIsolation.pruneStaleWorktrees(
            root: worktreeRoot,
            olderThan: -1
        )
        XCTAssertTrue(removed.isEmpty, "an applied worktree kept for ignored files must not be swept")
        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))
    }

    func testTheSweepKeepsAWorktreeWhoseRecordThisBuildCannotDecode() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        // A newer-schema (or corrupt) record is KEPT by the store on purpose.
        // Treating "cannot decode" as "no record" handed the tree — the only
        // full copy of its work — to the 14-day clock.
        let store = AgentRunStore.shared
        let recordURL = store.fileURL(runId: runId, sessionId: nil)
        try FileManager.default.createDirectory(
            at: recordURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data("not a run record".utf8).write(to: recordURL)
        defer {
            try? FileManager.default.removeItem(at: recordURL)
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
        }

        let removed = await WorkspaceIsolation.pruneStaleWorktrees(
            root: worktreeRoot,
            olderThan: -1
        )
        XCTAssertTrue(removed.isEmpty, "an undecodable record still names the worktree")
        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))
    }

    func testStoppingAnIsolatedRunKeepsItsWorktreeAndTheWorkInIt() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        // Turn 1 writes real work inside the tree; turn 2 stalls so the Stop
        // lands mid-run. The cancelled run must keep BOTH the worktree and
        // the file, and its record must say cancelled — not crash the process
        // on git spawned from a cancelled task, and not delete the tree.
        let backend = ScriptedBackend(turns: [
            ScriptedBackend.ScriptedTurn(
                text: "",
                calls: [
                    AgentToolCall(
                        name: "file_write",
                        arguments: ["path": "half_done.txt", "content": "written before the Stop\n"]
                    )
                ]
            ),
            ScriptedBackend.ScriptedTurn(text: "still working", delayNanoseconds: 20_000_000_000)
        ])
        let plan = AgentSpawnPlan(
            role: "Implementer",
            taskDescription: "Write half of the work.",
            contextSummary: nil,
            preset: .implementer,
            backendDescriptor: .scripted,
            schedulingClass: .cloud(providerId: "scripted"),
            backend: backend,
            toolNames: ["file_write"],
            budget: AgentBudget(maxSteps: 4, maxCompletionTokens: 4_000, maxWallClockSeconds: 60, maxCostUSD: nil),
            approvalPolicy: .spawnOnly,
            isolation: worktree.isolation,
            thinking: false,
            sampling: .default,
            priority: .userRequested,
            parentSessionId: nil,
            parentTurnId: nil,
            workingDirectory: worktree.path
        )
        let subagent = SubagentManager.shared.spawnModelBacked(plan, id: runId)
        // Registered rather than a `defer` with an unawaited Task: a cleanup
        // still running after the case returned raced the next case's fixture
        // (both live in the same process-wide worktree root) and could delete
        // a directory that case had just created. XCTest awaits this block.
        addTeardownBlock {
            await MainActor.run {
                SubagentManager.shared.clearCompleted(sessionId: nil)
            }
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
            AgentRunStore.shared.flush()
            _ = await WorkspaceIsolation.cleanup(worktree)
        }

        // Wait until the run has actually produced its work before stopping it,
        // so the tree provably holds something worth keeping.
        let written = worktree.path.appendingPathComponent("half_done.txt")
        let readyDeadline = Date().addingTimeInterval(15)
        while Date() < readyDeadline,
              !FileManager.default.fileExists(atPath: written.path) {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: written.path),
            "the run must reach its write before the Stop"
        )

        subagent.cancel(reason: "The coordinator turn was stopped.")
        _ = await subagent.waitForCompletion()

        // The record is written by the run's own detached bookkeeping task; the
        // worktree assertions are only honest once that has landed.
        let settled = Date().addingTimeInterval(10)
        while Date() < settled,
              AgentRunStore.shared.loadAny(runId: runId)?.status != .cancelled {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(AgentRunStore.shared.loadAny(runId: runId)?.status, .cancelled)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: worktree.path.path),
            "a cancelled writer's worktree is the only copy of its work"
        )
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: written.path),
            "the work the run completed before the Stop must survive it"
        )
    }

    // MARK: - The coordinator's tool

    func testAnIsolatedScriptedRunWritesInItsWorktreeAndTheCoordinatorAppliesIt() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        let backend = ScriptedBackend(turns: [
            ScriptedBackend.ScriptedTurn(
                text: "",
                calls: [
                    AgentToolCall(
                        name: "file_write",
                        arguments: ["path": "from_agent.txt", "content": "written by the run\n"]
                    )
                ]
            ),
            ScriptedBackend.ScriptedTurn(text: "Created the file.")
        ])
        let plan = AgentSpawnPlan(
            role: "Implementer",
            taskDescription: "Create from_agent.txt.",
            contextSummary: nil,
            preset: .implementer,
            backendDescriptor: .scripted,
            schedulingClass: .cloud(providerId: "scripted"),
            backend: backend,
            toolNames: ["file_write"],
            budget: AgentBudget(maxSteps: 4, maxCompletionTokens: 4_000, maxWallClockSeconds: 60, maxCostUSD: nil),
            approvalPolicy: .spawnOnly,
            isolation: worktree.isolation,
            thinking: false,
            sampling: .default,
            priority: .userRequested,
            parentSessionId: nil,
            parentTurnId: nil,
            workingDirectory: worktree.path
        )
        let subagent = SubagentManager.shared.spawnModelBacked(plan, id: runId)
        defer {
            SubagentManager.shared.clearCompleted(sessionId: nil)
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
        }

        let summary = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .completed, summary)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: worktree.path.appendingPathComponent("from_agent.txt").path),
            "the run must write inside its own worktree"
        )
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("from_agent.txt").path),
            "the run must not touch the main tree"
        )
        XCTAssertEqual(subagent.report?.changedFiles, ["from_agent.txt"])
        XCTAssertTrue(subagent.report?.diff?.contains("written by the run") ?? false)

        let tool = ApplySubagentChangesTool()
        let applied = try await tool.execute(
            arguments: ["subagent_id": runId.uuidString],
            workingDirectory: repository,
            maxOutputLength: 4_000
        )
        XCTAssertTrue(applied.stdout?.contains("Applied") ?? false, applied.resultJSON)
        XCTAssertTrue(FileManager.default.fileExists(atPath: repository.appendingPathComponent("from_agent.txt").path))
        XCTAssertNotNil(subagent.appliedAt)
        XCTAssertFalse(FileManager.default.fileExists(atPath: worktree.path.path))

        let status = try await GetSubagentStatusTool().execute(
            arguments: ["subagent_id": runId.uuidString],
            workingDirectory: repository,
            maxOutputLength: 4_000
        )
        XCTAssertTrue(
            status.stdout?.contains("already been applied") ?? false,
            "a coordinator that asks after the apply must not be told to apply again: \(status.resultJSON)"
        )
        XCTAssertFalse(status.resultJSON.contains("apply_directive"), status.resultJSON)
    }

    /// An operator who decides a writer's work should not be landed must be
    /// able to discard it. `clearCompleted` deliberately refuses to make that
    /// decision in bulk (the record is what protects the worktree), so the
    /// per-run discard is the only path that can ever remove one. Without it
    /// the record, the tree and the branch were permanent.
    func testAnUnappliedIsolatedRunCanBeDiscardedPerRun() async throws {
        let runId = UUID()
        let session = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        let written = worktree.path.appendingPathComponent("discard_me.txt")
        let backend = ScriptedBackend(turns: [
            ScriptedBackend.ScriptedTurn(
                text: "",
                calls: [
                    AgentToolCall(
                        name: "file_write",
                        arguments: ["path": "discard_me.txt", "content": "work the operator does not want\n"]
                    )
                ]
            ),
            ScriptedBackend.ScriptedTurn(text: "Created the file.")
        ])
        let plan = AgentSpawnPlan(
            role: "Implementer",
            taskDescription: "Create discard_me.txt.",
            contextSummary: nil,
            preset: .implementer,
            backendDescriptor: .scripted,
            schedulingClass: .cloud(providerId: "scripted"),
            backend: backend,
            toolNames: ["file_write"],
            budget: AgentBudget(maxSteps: 4, maxCompletionTokens: 4_000, maxWallClockSeconds: 60, maxCostUSD: nil),
            approvalPolicy: .spawnOnly,
            isolation: worktree.isolation,
            thinking: false,
            sampling: .default,
            priority: .userRequested,
            parentSessionId: session,
            parentTurnId: nil,
            workingDirectory: worktree.path
        )
        let subagent = SubagentManager.shared.spawnModelBacked(plan, id: runId)
        defer {
            SubagentManager.shared.clearCompleted(sessionId: session)
            AgentRunStore.shared.delete(runId: runId, sessionId: session)
            AgentRunStore.shared.flush()
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
        }

        _ = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .completed)
        XCTAssertTrue(FileManager.default.fileExists(atPath: written.path))
        XCTAssertNotNil(
            AgentRunStore.shared.load(runId: runId, sessionId: session),
            "the terminal record is what keeps the worktree from the launch sweep"
        )

        // The commit's safeguard, asserted directly: the bulk Clear button
        // must not dispose of an unapplied isolated run (its record is the
        // worktree's only protection). The per-run discard below is the one
        // path allowed to make that decision.
        SubagentManager.shared.clearCompleted(sessionId: session)
        XCTAssertNotNil(SubagentManager.shared.getSubagent(byId: runId))
        XCTAssertNotNil(
            AgentRunStore.shared.load(runId: runId, sessionId: session),
            "clearCompleted must spare an unapplied isolated run"
        )

        // Discard removes the tree before the record, so both are gone by the
        // time it returns; a failure would have kept the record and returned
        // false, which the assertions below would catch.
        let discarded = await SubagentManager.shared.discard(runId: runId, sessionId: session)
        XCTAssertTrue(discarded, "the tree was removable, so the discard must report success")

        XCTAssertNil(SubagentManager.shared.getSubagent(byId: runId))
        XCTAssertNil(AgentRunStore.shared.load(runId: runId, sessionId: session))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: worktree.path.path),
            "discarding the run must remove its only copy of the work"
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: written.path))
    }

    func testApplyIsStateChangingAndForbiddenToRuns() {
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "apply_subagent_changes"))
        XCTAssertTrue(AgentHarness.runScopedForbiddenToolNames.contains("apply_subagent_changes"))
        XCTAssertEqual(ApplySubagentChangesTool.category(for: "apply_subagent_changes"), "subagents")
    }

    func testTheApplyToolAppliesARecordedRunsWorktreeAndIsIdempotent() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        try "tool applied\n".write(
            to: worktree.path.appendingPathComponent("tool.txt"),
            atomically: true,
            encoding: .utf8
        )
        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Add a file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer { AgentRunStore.shared.delete(runId: runId, sessionId: nil) }

        let tool = ApplySubagentChangesTool()
        let first = try await tool.execute(
            arguments: ["subagent_id": runId.uuidString],
            workingDirectory: repository,
            maxOutputLength: 4_000
        )
        XCTAssertTrue(first.stdout?.contains("Applied") ?? false, first.resultJSON)
        XCTAssertTrue(FileManager.default.fileExists(atPath: repository.appendingPathComponent("tool.txt").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: worktree.path.path))

        let stored = AgentRunStore.shared.loadAny(runId: runId)
        XCTAssertNotNil(stored?.appliedAt, "the record must remember the apply")
        AgentRunStore.shared.flush()

        let second = try await tool.execute(
            arguments: ["subagent_id": runId.uuidString],
            workingDirectory: repository,
            maxOutputLength: 4_000
        )
        XCTAssertTrue(second.stdout?.contains("already applied") ?? false, second.resultJSON)
    }

    func testAnApplyWithIgnoredFilesBesideTrackedChangesKeepsTheWorktree() async throws {
        let runId = UUID()
        // The ignore rule must be IN the tree the worktree is cut from, so
        // the run's `.local` file is invisible to the patch.
        try "*.local\n".write(
            to: repository.appendingPathComponent(".gitignore"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: repository)
        try await git(["commit", "-m", "ignore local files"], in: repository)

        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        try "tracked change\n".write(
            to: worktree.path.appendingPathComponent("from_agent.txt"),
            atomically: true,
            encoding: .utf8
        )
        try "api key\n".write(
            to: worktree.path.appendingPathComponent("secret.local"),
            atomically: true,
            encoding: .utf8
        )
        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Write the file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer {
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
            AgentRunStore.shared.flush()
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
            try? FileManager.default.removeItem(at: repository.appendingPathComponent("from_agent.txt"))
        }

        let tool = ApplySubagentChangesTool()
        let applied = try await tool.execute(
            arguments: ["subagent_id": runId.uuidString],
            workingDirectory: repository,
            maxOutputLength: 4_000
        )
        // The tracked patch lands; the ignored file cannot ride it, so the
        // worktree is KEPT and the apply names what is still in it. The old
        // behaviour deleted the tree with the run's only copy of the file.
        XCTAssertTrue(applied.stdout?.contains("Applied") ?? false, applied.resultJSON)
        XCTAssertTrue(applied.stdout?.contains("KEPT") ?? false, applied.resultJSON)
        XCTAssertTrue(applied.stdout?.contains("secret.local") ?? false, applied.resultJSON)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("from_agent.txt").path),
            "the tracked patch must land"
        )
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: worktree.path.appendingPathComponent("secret.local").path),
            "the ignored file must survive in the kept worktree"
        )
        let stored = AgentRunStore.shared.loadAny(runId: runId)
        XCTAssertNotNil(stored?.appliedAt)
        XCTAssertTrue(
            stored?.report?.openQuestions.contains { $0.contains("secret.local") } == true,
            "the record must say why an applied run still has a worktree"
        )
    }

    func testTheApplyToolRefusesARunThatWasNotIsolated() async throws {
        let runId = UUID()
        let record = AgentRunRecord(
            id: runId,
            role: "Explore",
            preset: .explore,
            task: "Look around.",
            backend: .scripted,
            workingDirectory: repository.path,
            allowedTools: ["file_read"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer { AgentRunStore.shared.delete(runId: runId, sessionId: nil) }

        let tool = ApplySubagentChangesTool()
        let result = try await tool.execute(
            arguments: ["subagent_id": runId.uuidString],
            workingDirectory: repository,
            maxOutputLength: 1_000
        )
        XCTAssertTrue(result.resultJSON.contains("did not run in an isolated worktree"), result.resultJSON)
    }

    // MARK: - The operator's path

    func testTheOperatorPathOnlyOffersApplyForAStoppedUnappliedIsolatedRun() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )

        var record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Add a file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )

        XCTAssertTrue(SubagentApplyControl.canApply(record))
        XCTAssertFalse(
            SubagentApplyControl.canApply(record, executionStillUnwinding: true),
            "a Stop marks the record terminal while its task may still be killing a child; apply must wait for the unwind"
        )
        let request = SubagentApplyControl.applyRequest(for: record)
        XCTAssertEqual(request?.runId, runId)
        XCTAssertEqual(request?.toolName, "apply_subagent_changes")
        XCTAssertEqual(request?.arguments["subagent_id"], runId.uuidString)
        XCTAssertEqual(request?.arguments["branch"], worktree.branch)
        XCTAssertTrue(request?.reason.contains(worktree.branch) ?? false)
        XCTAssertNil(request?.workingDirectory, "the card resolves the main tree only when the caller has it")

        record.status = .running
        XCTAssertFalse(SubagentApplyControl.canApply(record))
        XCTAssertNil(SubagentApplyControl.applyRequest(for: record))

        record.status = .completed
        record.appliedAt = Date()
        XCTAssertFalse(SubagentApplyControl.canApply(record))
        XCTAssertNil(SubagentApplyControl.applyRequest(for: record))

        record.appliedAt = nil
        record.isolation = .none
        XCTAssertFalse(SubagentApplyControl.canApply(record))

        record.isolation = worktree.isolation
        await WorkspaceIsolation.cleanup(worktree)
        XCTAssertFalse(SubagentApplyControl.canApply(record), "a missing worktree offers nothing to apply")
    }

    func testTheOperatorPathDeclinedApprovalLeavesTheWorktreeAndRecordUntouched() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        try "declined\n".write(
            to: worktree.path.appendingPathComponent("declined.txt"),
            atomically: true,
            encoding: .utf8
        )
        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Add a file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer { AgentRunStore.shared.delete(runId: runId, sessionId: nil) }

        do {
            _ = try await SubagentApplyControl.apply(runId: runId, requestApproval: { _ in false })
            XCTFail("a declined card must not apply")
        } catch let error as SubagentApplyControlError {
            XCTAssertEqual(error, .approvalDenied)
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: repository.appendingPathComponent("declined.txt").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))
        XCTAssertNil(AgentRunStore.shared.loadAny(runId: runId)?.appliedAt)
        _ = await WorkspaceIsolation.cleanup(worktree)
    }

    func testTheOperatorPathAppliesAfterApprovalAndThenOffersNothing() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        try "operator applied\n".write(
            to: worktree.path.appendingPathComponent("operator.txt"),
            atomically: true,
            encoding: .utf8
        )
        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Add a file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer { AgentRunStore.shared.delete(runId: runId, sessionId: nil) }

        let asked = expectation(description: "the operator is asked once")
        let directory = repository.resolvingSymlinksInPath().standardizedFileURL.path
        let outcome = try await SubagentApplyControl.apply(runId: runId) { request in
            XCTAssertEqual(request.runId, runId)
            XCTAssertEqual(request.toolName, "apply_subagent_changes")
            XCTAssertEqual(request.arguments["branch"], worktree.branch)
            XCTAssertEqual(
                URL(fileURLWithPath: request.workingDirectory ?? "")
                    .resolvingSymlinksInPath().standardizedFileURL.path,
                directory,
                "the card names the tree the patch lands in"
            )
            asked.fulfill()
            return true
        }
        await fulfillment(of: [asked], timeout: 2)

        XCTAssertFalse(outcome.alreadyApplied)
        XCTAssertEqual(outcome.changedFiles, ["operator.txt"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: repository.appendingPathComponent("operator.txt").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: worktree.path.path))
        XCTAssertNotNil(AgentRunStore.shared.loadAny(runId: runId)?.appliedAt)

        do {
            _ = try await SubagentApplyControl.apply(runId: runId, requestApproval: { _ in true })
            XCTFail("an applied run must not offer a second apply")
        } catch let error as SubagentApplyControlError {
            guard case .nothingToApply = error else {
                return XCTFail("expected nothingToApply, got \(error)")
            }
        }
    }

    // MARK: - Helpers

    /// A submodule's common git dir lives under `<super>/.git/modules`, so
    /// taking its parent named the modules directory as the repository root
    /// and every snapshot/apply then ran in the wrong tree. `core.worktree`
    /// records the checkout, and that is what the derivation reads.
    func testOwningRepositoryResolvesASubmoduleWorktreeToItsCheckout() async throws {
        let superRepository = try makeTemporaryDirectory("super")
        let innerRepository = try makeTemporaryDirectory("inner")
        let linked = try makeTemporaryDirectory("linked")
        defer {
            try? FileManager.default.removeItem(at: superRepository)
            try? FileManager.default.removeItem(at: innerRepository)
            try? FileManager.default.removeItem(at: linked)
        }

        try await git(["init", "-b", "main"], in: innerRepository)
        try "inner\n".write(
            to: innerRepository.appendingPathComponent("inner.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: innerRepository)
        try await git(["commit", "-m", "init"], in: innerRepository)

        try await git(["init", "-b", "main"], in: superRepository)
        try "outer\n".write(
            to: superRepository.appendingPathComponent("outer.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: superRepository)
        try await git(["commit", "-m", "init"], in: superRepository)
        let added = try await git(
            ["-c", "protocol.file.allow=always", "submodule", "add", innerRepository.path, "sub"],
            in: superRepository
        )
        XCTAssertEqual(added.0, 0, added.2)
        try await git(["commit", "-m", "add submodule"], in: superRepository)

        let submodule = superRepository.appendingPathComponent("sub", isDirectory: true)
        let linkedPath = linked.appendingPathComponent("wt", isDirectory: true)
        let worktree = try await git(
            ["worktree", "add", "-b", "agent-wt", linkedPath.path, "HEAD"],
            in: submodule
        )
        XCTAssertEqual(worktree.0, 0, worktree.2)

        let root = await WorkspaceIsolation.owningRepository(for: linkedPath)
        XCTAssertEqual(
            root?.resolvingSymlinksInPath().standardizedFileURL,
            submodule.resolvingSymlinksInPath().standardizedFileURL,
            "a submodule worktree must resolve to the submodule checkout, not <super>/.git/modules"
        )
        _ = try? await git(["worktree", "remove", "--force", linkedPath.path], in: submodule)
    }

    private func makeTemporaryDirectory(_ label: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-worktree-\(label)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @discardableResult
    private func git(_ arguments: [String], in directory: URL) async throws -> (Int32, String, String) {
        // A test commits without the developer's identity, so every commit is
        // explicit rather than depending on a machine-wide git config.
        var args = arguments
        if args.first == "commit" {
            args = ["-c", "user.name=DynaMoE Tests", "-c", "user.email=tests@dynamoe.invalid"] + args
        }
        return try await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: args,
            currentDirectory: directory,
            timeoutSeconds: 60
        )
    }

    // MARK: - Git linkage verification (merge blocker B1)

    /// A linked worktree's `.git` is a regular FILE inside the containment
    /// prefix, so a sandboxed run can rewrite it. `git worktree repair`
    /// restores it from the repository's own records — which live OUTSIDE the
    /// sandbox — and the snapshot then reads the run's work against the base
    /// commit it was created from, with the operator's shared index never
    /// touched by the tampered pointer.
    func testARetargetedGitPointerIsRefusedRatherThanRepaired() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let readme = worktree.path.appendingPathComponent("README.md")
        try "line one\nline two\nline three\nline four\n".write(to: readme, atomically: true, encoding: .utf8)

        // The exploit shape: the pointer now names the repository's own .git,
        // so a plain `git add -A` in the worktree would stage the OPERATOR's
        // tree into their shared index. `git worktree repair <path>` resolves
        // the path THROUGH that pointer, so repairing it here rewrote the
        // main repository's admin file instead of this worktree's — the
        // refusal must come without repair at all.
        let retargeted = "gitdir: \(repository.appendingPathComponent(".git").path)\n"
        try retargeted.write(
            to: worktree.path.appendingPathComponent(".git"),
            atomically: true,
            encoding: .utf8
        )

        do {
            _ = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
            XCTFail("a pointer aimed at another repository's git directory must be refused")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeLinkageCompromised = error else {
                XCTFail("expected a linkage refusal, got \(error)")
                return
            }
        }

        XCTAssertEqual(
            try String(contentsOf: worktree.path.appendingPathComponent(".git"), encoding: .utf8),
            retargeted,
            "the refusal must not call repair, which would follow the pointer and rewrite another worktree's record"
        )
        let status = try await git(["status", "--porcelain"], in: repository)
        XCTAssertEqual(status.1, "", "the operator's tree must never be staged by the snapshot")
    }

    /// Deleting the pointer (the work-loss half of the exploit) orphans work
    /// the app could not land. The repair restores it from the repository's
    /// records and the work stays readable.
    func testADeletedGitPointerIsRepairedAndTheWorkIsStillReadable() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        try "line one\nline two\nline three\nline four\n".write(
            to: worktree.path.appendingPathComponent("README.md"),
            atomically: true,
            encoding: .utf8
        )
        try FileManager.default.removeItem(at: worktree.path.appendingPathComponent(".git"))

        let changes = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
        XCTAssertEqual(changes.changedFiles, ["README.md"])
    }

    /// A carded `shell_run` can `git checkout` another branch inside the
    /// sandbox. The `.git` pointer is intact, so the linkage check alone says
    /// yes; the HEAD check is what stops a later cardless `git_commit` from
    /// moving a branch the run never owned.
    func testASnapshotRefusesAWorktreeWhoseHeadMoved() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        _ = try await git(["checkout", "-b", "operator-other"], in: worktree.path)

        do {
            _ = try await WorkspaceIsolation.snapshot(worktree)
            XCTFail("a worktree on another branch must refuse the snapshot")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeLinkageCompromised(let detail) = error else {
                return XCTFail("expected worktreeLinkageCompromised, got \(error)")
            }
            XCTAssertTrue(detail.contains("HEAD"), detail)
            XCTAssertTrue(detail.contains("operator-other"), detail)
        }
    }

    /// When the repository's own records are gone too, nothing can repair the
    /// linkage, and the snapshot must REFUSE rather than run a single git
    /// command through it — keeping the worktree so the files survive.
    func testAnUnrepairableLinkageRefusesTheSnapshotAndKeepsTheWorktree() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        try "line four\n".write(
            to: worktree.path.appendingPathComponent("Added.txt"),
            atomically: true,
            encoding: .utf8
        )
        // The per-worktree admin directory lives in the MAIN repository's
        // .git, where in-sandbox file tools cannot reach it — but a shell the
        // operator approved could. With it gone, repair has no source of
        // truth.
        let adminDir = repository
            .appendingPathComponent(".git", isDirectory: true)
            .appendingPathComponent("worktrees", isDirectory: true)
            .appendingPathComponent(worktree.path.lastPathComponent, isDirectory: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: adminDir.path))
        try FileManager.default.removeItem(at: adminDir)

        do {
            _ = try await WorkspaceIsolation.snapshot(worktree)
            XCTFail("an unrepairable linkage must refuse the snapshot")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeLinkageCompromised = error else {
                return XCTFail("expected worktreeLinkageCompromised, got \(error)")
            }
        }
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: worktree.path.appendingPathComponent("Added.txt").path),
            "the refused worktree keeps its files"
        )
        let status = try await git(["status", "--porcelain"], in: repository)
        XCTAssertEqual(status.1, "", "nothing was staged anywhere by the refusal")
    }

    /// The worktree linkage check answers the same question for an APPLY: a
    /// patch read through a retargeted pointer must not land on the main tree.
    func testAnApplyRefusesAWorktreeWhoseLinkageCannotBeRepaired() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        try "line four\n".write(
            to: worktree.path.appendingPathComponent("Added.txt"),
            atomically: true,
            encoding: .utf8
        )
        let adminDir = repository
            .appendingPathComponent(".git", isDirectory: true)
            .appendingPathComponent("worktrees", isDirectory: true)
            .appendingPathComponent(worktree.path.lastPathComponent, isDirectory: true)
        try FileManager.default.removeItem(at: adminDir)

        do {
            _ = try await WorkspaceIsolation.apply(worktree)
            XCTFail("an unrepairable linkage must refuse the apply")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeLinkageCompromised = error else {
                return XCTFail("expected worktreeLinkageCompromised, got \(error)")
            }
        }
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("Added.txt").path),
            "nothing landed on the main tree"
        )
    }

    /// A run's own `git_commit` is cardless only while the worktree's pointer
    /// still says what `git worktree add` wrote. The harness re-verifies against
    /// the RECORDED repository root before the call runs, so an approved `.git`
    /// tamper cannot buy a cardless commit into the operator's shared index.
    func testAGitCommitInAnIsolatedRunRefusesWhenTheWorktreeLinkageMoved() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        let adminDir = repository
            .appendingPathComponent(".git", isDirectory: true)
            .appendingPathComponent("worktrees", isDirectory: true)
            .appendingPathComponent(worktree.path.lastPathComponent, isDirectory: true)
        try FileManager.default.removeItem(at: adminDir)

        let context = try AgentRunContext.make(
            runId: runId,
            sessionId: runId,
            toolNames: ["git_commit", "complete"],
            workingDirectory: worktree.path,
            mode: .code,
            approvalPolicy: .spawnOnly,
            workspaceIsolated: true,
            workspaceRepositoryRoot: repository,
            approvalHandler: { _ in
                XCTFail("a git_commit with a moved linkage is refused before the card")
                return false
            }
        )
        let result = await context.execute(AgentToolCall(
            name: "git_commit",
            arguments: ["message": "tampered", "stage_all": true]
        ))
        XCTAssertEqual(result.record.status, .withheld, "the linkage refusal is a policy withholding")
        XCTAssertTrue(
            result.record.error?.contains("no longer the checkout") == true,
            result.record.error ?? ""
        )
        let status = try await git(["status", "--porcelain"], in: repository)
        XCTAssertEqual(status.1, "", "nothing was staged or committed through the moved linkage")
    }

    /// `worktree add -b` can fail BECAUSE the branch name already exists, and
    /// the failure path used to `branch -D` a branch this call never owned.
    /// The branch is only given back when it points at the base commit this
    /// call asked git to cut it from.
    func testAFailedCreateDoesNotDeleteABranchItDidNotOwn() async throws {
        let runId = UUID()
        let branch = WorkspaceIsolation.branchPrefix + String(runId.uuidString.prefix(12)).lowercased()
        // A pre-existing branch of the collision name, pointing at the
        // INITIAL commit while HEAD has moved one commit on: `worktree add -b`
        // fails because the name is taken, and the branch (which does not
        // point at the base commit this call passes) must survive.
        let base = try await git(["rev-parse", "HEAD"], in: repository)
        let initialCommit = base.1.trimmingCharacters(in: .whitespacesAndNewlines)
        try "unrelated\n".write(
            to: repository.appendingPathComponent("second.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: repository)
        try await git(["commit", "-q", "-m", "second"], in: repository)
        try await git(["branch", branch, initialCommit], in: repository)

        do {
            _ = try await WorkspaceIsolation.createWorktree(
                repositoryRoot: repository,
                runId: runId,
                root: worktreeRoot
            )
            XCTFail("a pre-existing branch of the same name must fail the worktree add")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeCreationFailed = error else {
                return XCTFail("expected worktreeCreationFailed, got \(error)")
            }
        }

        let survivors = try await git(["rev-parse", "--verify", branch], in: repository)
        XCTAssertEqual(survivors.0, 0, "the failure path must not delete a branch it did not own")
    }

    // MARK: - Applying into a busy tree

    /// A new-file patch hand-carried through `WorkspaceIsolation.apply`, so
    /// the mid-operation refusals below exercise the guard without first
    /// needing a run, a snapshot, or a real change set.
    private func newFileChangeSet(name: String) -> WorkspaceChangeSet {
        WorkspaceChangeSet(
            changedFiles: [name],
            // git apply refuses a patch whose final line has no newline, so the
            // hand-written diff carries one explicitly.
            diff: """
            diff --git a/\(name) b/\(name)
            new file mode 100644
            index 0000000..b5ba4d5
            --- /dev/null
            +++ b/\(name)
            @@ -0,0 +1 @@
            +landed by the apply
            """ + "\n",
            diffWasTruncated: false
        )
    }

    /// Applying into a merge, cherry-pick, revert or bisect lands in the middle
    /// of someone else's operation; each is refused by name, and the same patch
    /// applies once the tree is settled (the control, run last because it
    /// writes the file).
    func testApplyIsRefusedWhileTheMainTreeIsMidOperation() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }

        // A side branch with a different README, so the merge stays pending
        // under `--no-commit --no-ff`.
        try await git(["checkout", "-q", "-b", "side"], in: repository)
        try "side-version\n".write(
            to: repository.appendingPathComponent("README.md"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["commit", "-aqm", "side change"], in: repository)
        let sideCommit = try await git(["rev-parse", "HEAD"], in: repository)
        try await git(["checkout", "-q", "main"], in: repository)
        try "main-version\n".write(
            to: repository.appendingPathComponent("README.md"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["commit", "-aqm", "main line"], in: repository)
        let baseCommit = worktree.baseCommit

        let refusals: [(label: String, begin: [String], abort: [String])] = [
            (
                "merge",
                ["merge", "--no-commit", "--no-ff", "side"],
                ["merge", "--abort"]
            ),
            (
                "cherry-pick",
                ["cherry-pick", sideCommit.1.trimmingCharacters(in: .whitespacesAndNewlines)],
                ["cherry-pick", "--abort"]
            ),
            (
                "revert",
                ["revert", "--no-edit", baseCommit],
                ["revert", "--abort"]
            ),
            (
                "bisect",
                ["bisect", "start"],
                ["bisect", "reset"]
            )
        ]

        for refusal in refusals {
            _ = try? await git(refusal.begin, in: repository)
            do {
                _ = try await WorkspaceIsolation.apply(
                    worktree,
                    changeSet: newFileChangeSet(name: "\(refusal.label)-probe.txt")
                )
                XCTFail("an apply during a pending \(refusal.label) must be refused")
            } catch let error as WorkspaceIsolationError {
                guard case .applyFailed(let detail) = error else {
                    return XCTFail("expected applyFailed during a \(refusal.label), got \(error)")
                }
                XCTAssertTrue(
                    detail.contains(refusal.label),
                    "the refusal must name the \(refusal.label), got: \(detail)"
                )
            }
            _ = try await git(refusal.abort, in: repository)
        }

        // The control: with no operation pending, the same patch lands.
        let settled = newFileChangeSet(name: "landed-by-control.txt")
        let result = try await WorkspaceIsolation.apply(worktree, changeSet: settled)
        XCTAssertEqual(result.changedFiles, settled.changedFiles)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("landed-by-control.txt").path),
            "the control patch must actually land once the tree is settled"
        )
    }

    /// `snapshot` registers intent to add instead of staging content: a linked
    /// worktree shares the main repository's object DB, and staging blobbed
    /// every file the run touched into the operator's store just to read a
    /// patch back. The content blob must not exist after a snapshot, while the
    /// patch still carries the file.
    func testASnapshotDoesNotBlobTheRunsFilesIntoTheSharedObjectStore() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }

        let big = String(repeating: "blob-bloat guard ", count: 6_400)
        try big.write(
            to: worktree.path.appendingPathComponent("big-new-file.txt"),
            atomically: true,
            encoding: .utf8
        )
        let changeSet = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)

        XCTAssertEqual(changeSet.changedFiles, ["big-new-file.txt"])
        XCTAssertTrue(
            changeSet.diff.contains("blob-bloat guard"),
            "the patch carries the new file's content"
        )

        let hash = try await git(
            ["hash-object", worktree.path.appendingPathComponent("big-new-file.txt").path],
            in: repository
        )
        let blobId = hash.1.trimmingCharacters(in: .whitespacesAndNewlines)
        let lookup = try await git(["cat-file", "-e", blobId], in: repository)
        XCTAssertNotEqual(
            lookup.0, 0,
            "the file's content must not be blobbed into the shared object store by a snapshot"
        )
    }

    /// The apply single-flight: a second apply for the same run is refused
    /// while the first is in flight. The in-flight window is the duration of
    /// one git call, which no test can occupy deterministically, so the window
    /// is opened by the testing seam instead.
    func testASecondApplyForTheSameRunIsRefusedWhileTheFirstIsInFlight() async {
        let runId = UUID()
        SubagentApplyService.beginApplyForTesting(runId)
        defer { SubagentApplyService.endApplyForTesting(runId) }

        do {
            _ = try await SubagentApplyService.apply(runId: runId)
            XCTFail("an apply during an in-flight apply of the same run must be refused")
        } catch let error as SubagentApplyError {
            guard case .alreadyApplying = error else {
                return XCTFail("expected alreadyApplying, got \(error)")
            }
        } catch {
            XCTFail("unexpected error type: \(error)")
        }
    }

    /// The mirror claim: a DISCARD owns the record and its tree for the length
    /// of the tree's removal (an `await`), so an Apply clicking into that
    /// window used to land a patch in a directory being deleted and then stamp
    /// `appliedAt` on a record the discard tombstones.
    func testAnApplyDuringADiscardIsRefused() async {
        let runId = UUID()
        XCTAssertTrue(SubagentApplyService.beginDiscard(runId))
        defer { SubagentApplyService.endDiscard(runId) }

        do {
            _ = try await SubagentApplyService.apply(runId: runId)
            XCTFail("an apply during a discard of the same run must be refused")
        } catch let error as SubagentApplyError {
            guard case .beingDiscarded = error else {
                return XCTFail("expected beingDiscarded, got \(error)")
            }
        } catch {
            XCTFail("unexpected error type: \(error)")
        }

        // One claim at a time: a second discard of the same run gets "not now"
        // rather than removing the tree twice.
        XCTAssertFalse(SubagentApplyService.beginDiscard(runId), "a second discard of one run is refused")
        XCTAssertTrue(SubagentApplyService.isBeingDiscarded(runId))
    }

    /// The repository claim serializes applies of two DIFFERENT runs of one
    /// checkout. Their `git apply --check` calls otherwise interleave with each
    /// other's writes, so one lands against a tree the other has already
    /// started changing. The in-flight window is one apply's duration, which no
    /// test can hold deterministically, so the claim is taken by hand.
    func testASecondRunsApplyInTheSameRepositoryIsRefusedWhileTheFirstIsInFlight() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        try "same repo\n".write(
            to: worktree.path.appendingPathComponent("same_repo.txt"),
            atomically: true,
            encoding: .utf8
        )
        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Add a file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer {
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
            try? FileManager.default.removeItem(at: repository.appendingPathComponent("same_repo.txt"))
        }

        SubagentApplyService.beginRepositoryApplyForTesting(repository)
        do {
            _ = try await SubagentApplyService.apply(runId: runId)
            XCTFail("an apply of the same checkout must wait its turn, not start")
        } catch let error as SubagentApplyError {
            guard case .repositoryBusy = error else {
                SubagentApplyService.endRepositoryApplyForTesting(repository)
                return XCTFail("expected repositoryBusy, got \(error)")
            }
        } catch {
            SubagentApplyService.endRepositoryApplyForTesting(repository)
            return XCTFail("unexpected error type: \(error)")
        }
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("same_repo.txt").path),
            "a refused apply must not have written anything"
        )

        // The claim is the only thing standing in the way: released, the same
        // run lands. This is also the proof the refusal path gave nothing back
        // it was not holding.
        SubagentApplyService.endRepositoryApplyForTesting(repository)
        let applied = try await SubagentApplyService.apply(runId: runId)
        XCTAssertFalse(applied.alreadyApplied)
        XCTAssertEqual(applied.changedFiles, ["same_repo.txt"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: repository.appendingPathComponent("same_repo.txt").path))
        // A REAL apply must give both claims back, not only the refusal path.
        XCTAssertFalse(SubagentApplyService.isRunInFlight(runId))
        XCTAssertFalse(SubagentApplyService.hasAnyRepositoryInFlight)
    }

    /// The claims must be released on a FAILED apply, not only refused while
    /// one is held by the debug seam. A conflict throws after both claims are
    /// taken, and a leaked claim wedges the run and its checkout for the rest
    /// of the process.
    func testAFailedApplyReleasesItsRunAndRepositoryClaims() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        var worktreeReadme = try String(contentsOf: worktree.path.appendingPathComponent("README.md"), encoding: .utf8)
        worktreeReadme = worktreeReadme.replacingOccurrences(of: "line two", with: "worktree line two")
        try worktreeReadme.write(
            to: worktree.path.appendingPathComponent("README.md"),
            atomically: true,
            encoding: .utf8
        )
        let mainReadmeURL = repository.appendingPathComponent("README.md")
        var mainReadme = try String(contentsOf: mainReadmeURL, encoding: .utf8)
        mainReadme = mainReadme.replacingOccurrences(of: "line two", with: "main line two")
        try mainReadme.write(to: mainReadmeURL, atomically: true, encoding: .utf8)

        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Edit the readme.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer {
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
            AgentRunStore.shared.flush()
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
            try? FileManager.default.removeItem(at: repository.appendingPathComponent("README.md"))
            try? mainReadme.write(to: mainReadmeURL, atomically: true, encoding: .utf8)
        }

        do {
            _ = try await SubagentApplyService.apply(runId: runId)
            XCTFail("a conflicting apply must fail")
        } catch let error as WorkspaceIsolationError {
            guard case .applyConflicted = error else {
                return XCTFail("expected applyConflicted, got \(error)")
            }
        }

        XCTAssertFalse(
            SubagentApplyService.isRunInFlight(runId),
            "a failed apply must release its run claim"
        )
        XCTAssertFalse(
            SubagentApplyService.hasAnyRepositoryInFlight,
            "a failed apply must release its repository claim"
        )
    }

    /// A run can COMMIT and then check the base branch out. The tree is clean,
    /// so the old empty-change path wrote `appliedAt` and ran `branch -D`,
    /// leaving the commit reachable only through the reflog. The HEAD check
    /// refuses the snapshot and the apply keeps the worktree.
    func testAWorktreeThatCheckedOutTheBaseIsNotAppliedOrDeleted() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        try "committed work\n".write(
            to: worktree.path.appendingPathComponent("committed.txt"),
            atomically: true,
            encoding: .utf8
        )
        _ = try await git(["add", "-A"], in: worktree.path)
        let commit = try await git(["commit", "-m", "run's own commit"], in: worktree.path)
        XCTAssertEqual(commit.0, 0, commit.2)
        // The run leaves its branch after committing (the base branch itself
        // is checked out in the main tree, so a linked worktree cannot take it;
        // a detached checkout of the base is the reachable "clean tree, HEAD
        // elsewhere" shape). The branch ref still holds the commit.
        let checkout = try await git(["checkout", "--detach", worktree.baseCommit], in: worktree.path)
        XCTAssertEqual(checkout.0, 0, checkout.2)

        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Commit a file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write"],
            status: .completed
        )
        AgentRunStore.shared.saveImmediately(record)
        defer {
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
            AgentRunStore.shared.flush()
            addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }
        }

        do {
            _ = try await SubagentApplyService.apply(runId: runId)
            XCTFail("a worktree that is no longer on its own branch must not be applied")
        } catch let error as WorkspaceIsolationError {
            guard case .worktreeLinkageCompromised(let detail) = error else {
                return XCTFail("expected worktreeLinkageCompromised, got \(error)")
            }
            XCTAssertTrue(detail.contains("HEAD"), detail)
            XCTAssertTrue(detail.contains(worktree.branch), detail)
        }

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: worktree.path.path),
            "the worktree is the operator's only copy of the committed work"
        )
        XCTAssertNil(AgentRunStore.shared.loadAny(runId: runId)?.appliedAt)
        let branch = try await git(["rev-parse", "--verify", worktree.branch], in: repository)
        XCTAssertEqual(branch.0, 0, "the run's branch, and its commit, must still exist")
    }

    // MARK: - Nested repositories (gitlinks)

    /// `git diff` can only express a nested repository as a gitlink, and
    /// `git apply` reports success on one while writing an empty directory and
    /// leaving the nested files behind. Treating that as a landed apply deleted
    /// the worktree, the only copy of those files.
    func testANestedRepositoryRefusesTheSnapshotAndKeepsItsFiles() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let nested = worktree.path.appendingPathComponent("vendor/tool", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try await git(["init", "-b", "main"], in: nested)
        try "nested work\n".write(
            to: nested.appendingPathComponent("nested.txt"),
            atomically: true,
            encoding: .utf8
        )
        try await git(["add", "-A"], in: nested)
        try await git(["commit", "-m", "nested"], in: nested)

        do {
            _ = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)
            XCTFail("a nested repository must be refused, not expressed as a gitlink")
        } catch let error as WorkspaceIsolationError {
            guard case .nestedRepositoryUnsupported = error else {
                return XCTFail("expected nestedRepositoryUnsupported, got \(error)")
            }
        }

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: nested.appendingPathComponent("nested.txt").path),
            "refusing the patch must keep the only copy of the nested files"
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))
    }

    /// A caller may hand `apply` its own change set, so the refusal cannot live
    /// only in `snapshot`.
    func testAnApplyRefusesAGitlinkInASuppliedChangeSet() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let gitlinkPatch = """
        diff --git a/vendor/tool b/vendor/tool
        new file mode 160000
        index 0000000..1111111
        --- /dev/null
        +++ b/vendor/tool
        @@ -0,0 +1 @@
        +Subproject commit 1111111111111111111111111111111111111111
        """
        let changes = WorkspaceChangeSet(
            changedFiles: ["vendor/tool"],
            diff: gitlinkPatch,
            diffWasTruncated: false
        )

        do {
            _ = try await WorkspaceIsolation.apply(worktree, changeSet: changes)
            XCTFail("a gitlink in a supplied change set must be refused before any byte is written")
        } catch let error as WorkspaceIsolationError {
            guard case .nestedRepositoryUnsupported(let path) = error else {
                return XCTFail("expected nestedRepositoryUnsupported, got \(error)")
            }
            XCTAssertEqual(path, "vendor/tool")
        }
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("vendor/tool").path),
            "a refused apply must not have created the empty directory git would write"
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: worktree.path.path))
    }

    /// M2: `snapshot` diffs `baseCommit` against the working tree, so commits
    /// made on the worktree's branch are already carried by the patch. Applying
    /// a committing implementer run must delete the branch cleanly instead of
    /// leaking it with a bogus "holds commits a patch cannot carry" notice.
    func testApplyingACommittingRunRemovesItsBranchWhenHeadIsOnBranch() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        let committedFile = worktree.path.appendingPathComponent("committed.txt")
        try "committed by agent\n".write(to: committedFile, atomically: true, encoding: .utf8)
        _ = try await git(["add", "committed.txt"], in: worktree.path)
        _ = try await git(["commit", "-m", "agent commit"], in: worktree.path)

        let record = AgentRunRecord(
            id: runId,
            role: "Implementer",
            preset: .implementer,
            task: "Commit a file.",
            backend: .scripted,
            workingDirectory: worktree.path.path,
            isolation: worktree.isolation,
            allowedTools: ["file_write", "git_commit"],
            status: .completed,
            repositoryRoot: repository.path,
            baseCommit: worktree.baseCommit
        )
        AgentRunStore.shared.saveImmediately(record)
        defer {
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
            try? FileManager.default.removeItem(at: repository.appendingPathComponent("committed.txt"))
        }

        let outcome = try await SubagentApplyService.apply(runId: runId)
        XCTAssertFalse(outcome.keptBranch, "A normal committing run's changes are in the patch, so its branch must not be leaked")
        XCTAssertEqual(outcome.changedFiles, ["committed.txt"])
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: repository.appendingPathComponent("committed.txt").path),
            "Committed file must land in the main working tree"
        )
        let branchCheck = try await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: ["rev-parse", "--verify", "refs/heads/\(worktree.branch)"],
            currentDirectory: repository
        )
        XCTAssertNotEqual(branchCheck.exitCode, 0, "Worktree branch should be deleted after clean apply")
    }

    /// M3: Calling `forgetDiscardedWorktree()` after `cancel()` has already
    /// latched `isFinished` must persist `.none` to disk so the record does not
    /// stay retention-exempt / `worktreeGone` after relaunch.
    func testForgettingDiscardedWorktreeAfterCancelPersistsClearedIsolation() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }

        let backend = ScriptedBackend(turns: [.init(text: "never runs")])
        let plan = AgentSpawnPlan(
            role: "Implementer",
            taskDescription: "Queued work",
            contextSummary: nil,
            preset: .implementer,
            backendDescriptor: .scripted,
            schedulingClass: .cloud(providerId: "scripted"),
            backend: backend,
            toolNames: ["file_write"],
            budget: AgentBudget(maxSteps: 4, maxCompletionTokens: 4_000, maxWallClockSeconds: 60, maxCostUSD: nil),
            approvalPolicy: .spawnOnly,
            isolation: worktree.isolation,
            thinking: false,
            sampling: .default,
            priority: .userRequested,
            parentSessionId: nil,
            parentTurnId: nil,
            workingDirectory: worktree.path,
            repositoryRoot: repository,
            baseCommit: worktree.baseCommit
        )
        let instance = SubagentInstance(id: runId, plan: plan)
        defer { AgentRunStore.shared.delete(runId: runId, sessionId: nil) }

        instance.cancel(reason: "Stopped while queued.")
        instance.forgetDiscardedWorktree()

        let stored = try XCTUnwrap(AgentRunStore.shared.loadAny(runId: runId))
        XCTAssertEqual(stored.isolation, .none, "Persisted record must clear isolation after queued worktree discard")
        XCTAssertFalse(stored.holdsWorktreeWork)
    }

    func testDanglingSymlinkWithParentTraversalThroughRealSymlinkEscapesContainment() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ContainmentDangling_\(UUID().uuidString)", isDirectory: true)
        let workspace = root.appendingPathComponent("ws", isDirectory: true)
        let outside = root.appendingPathComponent("outside/dir", isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // /ws/realLink -> /outside/dir, and /ws/dangling -> realLink/../pwned.txt
        let realLink = workspace.appendingPathComponent("realLink")
        try FileManager.default.createSymbolicLink(atPath: realLink.path, withDestinationPath: outside.path)
        let dangling = workspace.appendingPathComponent("dangling")
        try FileManager.default.createSymbolicLink(atPath: dangling.path, withDestinationPath: "realLink/../pwned.txt")

        let resolved = AgentHarness.resolvedForContainment(dangling).path
        let expectedOutside = outside.deletingLastPathComponent()
            .resolvingSymlinksInPath()
            .appendingPathComponent("pwned.txt").path
        let resolvedWorkspace = AgentHarness.resolvedForContainment(workspace).path
        XCTAssertEqual(
            resolved,
            expectedOutside,
            "Dangling symlink traversing `realLink/..` must resolve `realLink` before applying `..`"
        )
        XCTAssertFalse(
            resolved.hasPrefix(resolvedWorkspace + "/"),
            "Resolved path must not appear inside the workspace: \(resolved)"
        )
    }

    func testOpenAllowsRepairableMissingGitPointerForRecordedRoot() async throws {
        let runId = UUID()
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: runId,
            root: worktreeRoot
        )
        addTeardownBlock { _ = await WorkspaceIsolation.cleanup(worktree) }

        // Delete the worktree's `.git` pointer file inside the sandbox.
        let dotGit = worktree.path.appendingPathComponent(".git")
        try FileManager.default.removeItem(at: dotGit)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dotGit.path))

        // Production callers call `open` with `recordedRoot` before `verifyWorktreeLinkage`.
        let reopened = try await WorkspaceIsolation.open(
            path: worktree.path.path,
            branch: worktree.branch,
            repositoryRoot: repository,
            baseCommit: worktree.baseCommit
        )
        try await WorkspaceIsolation.verifyWorktreeLinkage(reopened)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: dotGit.path),
            "verifyWorktreeLinkage must restore the missing `.git` pointer after `open`"
        )
    }
}
