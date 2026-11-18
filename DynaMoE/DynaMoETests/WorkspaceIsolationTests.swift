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
        worktreeRoot = try makeTemporaryDirectory("worktrees")
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
        XCTAssertEqual(worktree.branch, WorkspaceIsolation.branchPrefix + String(runId.uuidString.prefix(8)).lowercased())

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
            Task { await WorkspaceIsolation.cleanup(worktree) }
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
        defer {
            SubagentManager.shared.clearCompleted(sessionId: nil)
            AgentRunStore.shared.delete(runId: runId, sessionId: nil)
            Task { await WorkspaceIsolation.cleanup(worktree) }
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
            Task { await WorkspaceIsolation.cleanup(worktree) }
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
    func testARetargetedGitPointerIsRepairedBeforeTheSnapshotReadsAnything() async throws {
        let worktree = try await WorkspaceIsolation.createWorktree(
            repositoryRoot: repository,
            runId: UUID(),
            root: worktreeRoot
        )
        let readme = worktree.path.appendingPathComponent("README.md")
        try "line one\nline two\nline three\nline four\n".write(to: readme, atomically: true, encoding: .utf8)

        // The exploit shape: the pointer now names the repository's own .git,
        // so a plain `git add -A` in the worktree would stage the OPERATOR's
        // tree into their shared index.
        try "gitdir: \(repository.appendingPathComponent(".git").path)\n".write(
            to: worktree.path.appendingPathComponent(".git"),
            atomically: true,
            encoding: .utf8
        )

        let changes = try await WorkspaceIsolation.snapshot(worktree, diffLimit: nil)

        XCTAssertEqual(changes.changedFiles, ["README.md"], "the repaired snapshot reads the run's own work")
        XCTAssertEqual(
            try String(contentsOf: worktree.path.appendingPathComponent(".git"), encoding: .utf8)
                .contains("/worktrees/"),
            true,
            "the repair must restore the per-worktree pointer"
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
            result.record.error?.contains("git metadata") == true,
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
        let branch = WorkspaceIsolation.branchPrefix + String(runId.uuidString.prefix(8)).lowercased()
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
}
