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
}
