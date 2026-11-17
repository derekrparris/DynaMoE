//
//  SubagentLoopTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for the model-backed subagent runtime. No model is
//  required: the ScriptedBackend drives the same loop the real backends use.
//

import XCTest
@testable import DynaMoE

/// Counts approval cards from inside the run's own `@Sendable` handlers.
nonisolated private final class CallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func increment() {
        lock.lock()
        value += 1
        lock.unlock()
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

@MainActor
final class SubagentLoopTests: XCTestCase {

    private func makeWorkspace() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubagentLoopTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeContext(
        workspace: URL,
        tools: [String],
        mode: AgentMode = .code,
        policy: AgentApprovalPolicy = .auto,
        isolated: Bool = false,
        handler: (@Sendable (AgentApprovalRequest) async -> Bool)? = nil
    ) throws -> AgentRunContext {
        try AgentRunContext.make(
            runId: UUID(),
            sessionId: UUID(),
            toolNames: tools,
            workingDirectory: workspace,
            mode: mode,
            approvalPolicy: policy,
            workspaceIsolated: isolated,
            approvalHandler: handler
        )
    }

    private func input(
        task: String = "Do the task",
        budget: AgentBudget = .standard,
        role: String = "Explorer"
    ) -> AgentLoopInput {
        AgentLoopInput(
            role: role,
            task: task,
            systemPrompt: "worker",
            budget: budget,
            sessionId: UUID()
        )
    }

    private func readCall(_ path: String) -> AgentToolCall {
        AgentToolCall(name: "file_read", arguments: ["path": path])
    }

    private func completeCall(_ summary: String) -> AgentToolCall {
        AgentToolCall(name: "complete", arguments: ["summary": summary])
    }

    func testScriptedRunExecutesToolsAndCompletes() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Notes.txt")
        try "QUANTUM_RELAY_SEQUENCE".write(to: file, atomically: true, encoding: .utf8)

        let backend = ScriptedBackend(turns: [
            .init(text: "Reading the file.", calls: [readCall(file.path)]),
            .init(text: "Done.", calls: [completeCall("Found QUANTUM_RELAY_SEQUENCE in Notes.txt.")])
        ])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        XCTAssertEqual(outcome.usage.toolCalls, 2)
        XCTAssertEqual(outcome.usage.backendRequests, 2)
        XCTAssertTrue(outcome.report?.summary.contains("QUANTUM_RELAY_SEQUENCE") == true)
        XCTAssertTrue(outcome.transcript.contains { $0.kind == .toolResult && $0.toolRecord?.name == "file_read" })
        XCTAssertTrue(outcome.transcript.contains { $0.kind == .toolResult && $0.toolRecord?.name == "complete" })
        XCTAssertTrue(backend.requests.first?.messages.isEmpty == false)
    }

    func testRunWithoutToolsReturnsFinalText() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [.init(text: "The answer is 42.")])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        XCTAssertEqual(outcome.report?.summary, "The answer is 42.")
        XCTAssertEqual(outcome.usage.toolCalls, 0)
    }

    /// Every step reports the FULL prompt it was sent, so a multi-step run's
    /// summed prompt tokens grow with the square of the step count while its
    /// context only grows with the conversation. The usage the drawer reads
    /// must be the last request's size, not the billed sum.
    func testTheRunsContextCountIsTheLastRequestsSizeNotTheStepSum() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Tokens.txt")
        try "context".write(to: file, atomically: true, encoding: .utf8)

        var first = AgentUsage()
        first.promptTokens = 100
        first.completionTokens = 5
        var second = AgentUsage()
        second.promptTokens = 120
        second.completionTokens = 7
        let backend = ScriptedBackend(turns: [
            .init(text: "one", calls: [readCall(file.path)], usage: first),
            .init(text: "Done.", calls: [completeCall("done")], usage: second)
        ])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        // Billed: both steps' full prompts, summed.
        XCTAssertEqual(outcome.usage.promptTokens, 220)
        // Context: the LAST request's size plus everything generated.
        XCTAssertEqual(outcome.usage.latestPromptTokens, 120)
        XCTAssertEqual(outcome.usage.contextTokens, 132)
    }

    func testStepBudgetIsEnforced() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Loop.txt")
        try "loop".write(to: file, atomically: true, encoding: .utf8)

        let backend = ScriptedBackend(turns: [
            .init(text: "one", calls: [readCall(file.path)]),
            .init(text: "two", calls: [readCall(file.path)]),
            .init(text: "three", calls: [readCall(file.path)])
        ])
        var budget = AgentBudget.standard
        budget.maxSteps = 2
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: budget),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.error?.contains("step budget") == true, outcome.error ?? "")
        XCTAssertEqual(outcome.usage.backendRequests, 2)
    }

    func testCostBudgetIsEnforced() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Cost.txt")
        try "cost".write(to: file, atomically: true, encoding: .utf8)

        var usage = AgentUsage()
        usage.costUSD = 0.02
        let backend = ScriptedBackend(turns: [
            .init(text: "one", calls: [readCall(file.path)], usage: usage),
            .init(text: "two", calls: [readCall(file.path)])
        ])
        var budget = AgentBudget.standard
        budget.maxCostUSD = 0.01
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: budget),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.error?.lowercased().contains("cost") == true, outcome.error ?? "")
        XCTAssertEqual(outcome.usage.backendRequests, 1)
    }

    func testCancellationStopsRun() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [
            .init(text: "slow", delayNanoseconds: 2_000_000_000)
        ])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let task = Task {
            await HeadlessAgentLoop().run(
                input: input(),
                backend: backend,
                context: context,
                emit: { _ in }
            )
        }
        try await Task.sleep(nanoseconds: 100_000_000)
        task.cancel()
        let outcome = await task.value

        XCTAssertEqual(outcome.status, .cancelled)
    }

    func testDisallowedToolIsRefusedWithoutExecution() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let marker = workspace.appendingPathComponent("should-not-exist.txt")

        let backend = ScriptedBackend(turns: [
            .init(text: "trying", calls: [AgentToolCall(name: "shell_run", arguments: ["command": "touch \(marker.path)"])]),
            .init(text: "done", calls: [completeCall("Could not run the command.")])
        ])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
        XCTAssertTrue(outcome.transcript.contains {
            $0.toolRecord?.name == "shell_run" && $0.toolRecord?.status == .withheld
        })
    }

    func testApprovalDenialPreventsWrite() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let target = workspace.appendingPathComponent("denied.txt")

        final class Counter: @unchecked Sendable {
            private let lock = NSLock()
            private var value = 0
            func increment() { lock.lock(); value += 1; lock.unlock() }
            var count: Int { lock.lock(); defer { lock.unlock() }; return value }
        }
        let counter = Counter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [AgentToolCall(name: "file_write", arguments: ["path": target.path, "content": "nope"])]),
            .init(text: "done", calls: [completeCall("The write was declined.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "file_read", "complete"],
            policy: .everyStateChange,
            handler: { _ in
                counter.increment()
                return false
            }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        XCTAssertEqual(counter.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertTrue(outcome.transcript.contains {
            $0.toolRecord?.name == "file_write" && $0.toolRecord?.status == .rejected
        })
    }

    func testAppPreferenceWriteNeedsApprovalEvenInIsolatedWorkspace() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }

        final class Counter: @unchecked Sendable {
            private let lock = NSLock()
            private var value = 0
            func increment() { lock.lock(); value += 1; lock.unlock() }
            var count: Int { lock.lock(); defer { lock.unlock() }; return value }
        }
        let counter = Counter()

        let backend = ScriptedBackend(turns: [
            .init(text: "changing prefs", calls: [AgentToolCall(name: "shell_run", arguments: ["command": "defaults write DRP.DynaMoE dynamoe_agent_turbo_mode -bool NO"])]),
            .init(text: "done", calls: [completeCall("Declined.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["shell_run", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.reason.lowercased().contains("preferences"))
                return false
            }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(counter.count, 1)
        XCTAssertTrue(outcome.transcript.contains {
            $0.toolRecord?.name == "shell_run" && $0.toolRecord?.status == .rejected
        })
    }

    func testReportCollectsChangedFilesAndTests() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let target = workspace.appendingPathComponent("Changed.txt")

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": target.path, "content": "changed"]),
                AgentToolCall(name: "shell_run", arguments: ["command": "echo 'test suite: 3 passed, 0 failed'"])
            ]),
            .init(text: "done", calls: [completeCall("Wrote Changed.txt and ran the tests.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "shell_run", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { _ in true }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        XCTAssertEqual(outcome.report?.changedFiles, [target.path])
        XCTAssertEqual(outcome.report?.tests.count, 1)
        XCTAssertEqual(outcome.report?.tests.first?.exitCode, 0)
        XCTAssertEqual(outcome.report?.tests.first?.passed, 3)
        XCTAssertEqual(outcome.report?.tests.first?.failed, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: target.path))
    }

    func testPerRunHarnessDoesNotMutateSharedHarness() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }

        let sharedBefore = AgentHarness.shared.loadedTools.keys.sorted()
        let first = try makeContext(workspace: workspace, tools: ["file_read"])
        let second = try makeContext(workspace: workspace, tools: ["grep_search", "find_files"])

        let firstNames = first.toolDefinitions().map(\.function.name).sorted()
        let secondNames = second.toolDefinitions().map(\.function.name).sorted()

        XCTAssertEqual(firstNames, ["file_read"])
        XCTAssertEqual(secondNames, ["find_files", "grep_search"])
        XCTAssertEqual(AgentHarness.shared.loadedTools.keys.sorted(), sharedBefore)
    }

    func testRunScopeRefusesToolsThatWouldReachTheSharedHarness() throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }

        for forbidden in ["tools_load", "tools_discover", "tools_unload", "spawn_subagent"] {
            XCTAssertThrowsError(
                try makeContext(workspace: workspace, tools: ["file_read", forbidden])
            ) { error in
                guard case AgentRunScopeError.forbiddenTool(let name) = error else {
                    return XCTFail("expected forbiddenTool for \(forbidden), got \(error)")
                }
                XCTAssertEqual(name, forbidden)
            }
        }
    }

    func testWorkerPromptOmitsCoordinatorPolicy() throws {
        let harness = AgentHarness.makeRunScoped()
        let prompt = harness.buildWorkerSystemPrompt(baseSystem: "You inspect code.", role: "Reviewer", mode: .code)
        XCTAssertTrue(prompt.contains("Subagent Role"))
        XCTAssertTrue(prompt.contains("Subagent Rules"))
        XCTAssertTrue(prompt.contains("cite"))
        XCTAssertFalse(prompt.contains("Subagent Result Delivery Policy"))
    }

    func testBackendReceivesRunToolSchemasAndWorkerPrompt() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [.init(text: "done")])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])
        let systemPrompt = "worker contract"
        let sessionId = UUID()

        _ = await HeadlessAgentLoop().run(
            input: AgentLoopInput(role: "Explorer", task: "task", systemPrompt: systemPrompt, sessionId: sessionId),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        let request = try XCTUnwrap(backend.requests.first)
        XCTAssertEqual(request.tools.map(\.function.name).sorted(), ["complete", "file_read"])
        XCTAssertEqual(request.systemPrompt, systemPrompt)
        XCTAssertEqual(request.sessionId, sessionId)
    }

    func testRecordRoundTripsThroughCodable() throws {
        let record = AgentRunRecord(
            role: "Explorer",
            preset: .explore,
            task: "Find the relay",
            backend: .scripted,
            workingDirectory: "/tmp",
            allowedTools: ["file_read", "complete"]
        )
        let data = try JSONEncoder().encode(record)
        let decoded = try JSONDecoder().decode(AgentRunRecord.self, from: data)
        XCTAssertEqual(decoded, record)
    }

    // MARK: - Report derivation

    func testTheReportCarriesTheRunsFindingsAndOpenQuestions() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Notes.txt")
        try "CONTENT".write(to: file, atomically: true, encoding: .utf8)

        let claim = "The scheduler lives in AgentScheduler.swift, and every run is admitted through it."
        let backend = ScriptedBackend(turns: [
            .init(text: claim + "\nIs the server queue's claim the same one?", calls: [readCall(file.path)]),
            .init(text: "Done.", calls: [completeCall("Read Notes.txt.")])
        ])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        let findings = try XCTUnwrap(outcome.report?.findings)
        XCTAssertEqual(findings.count, 1, "one claim was made, so the report carries one finding")
        XCTAssertEqual(findings.first?.claim, claim)
        XCTAssertTrue(findings.first?.evidence.contains("step 1") == true, findings.first?.evidence ?? "")
        XCTAssertTrue(findings.first?.evidence.contains("file_read") == true, "evidence names the tool result in front of the claim")

        XCTAssertEqual(outcome.report?.openQuestions.count, 1)
        XCTAssertTrue(outcome.report?.openQuestions.first?.hasPrefix("Is the server queue") == true)
    }

    func testABudgetExhaustedRunKeepsTheAnalysisItAlreadyFinished() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Notes.txt")
        try "CONTENT".write(to: file, atomically: true, encoding: .utf8)

        let analysis = "The review is complete: the FIFO cannot admit two runs because the tail is written before any suspension."
        let backend = ScriptedBackend(turns: [
            .init(text: analysis, calls: [readCall(file.path)]),
            .init(text: "Let me check one more file before I write the final report.", calls: [readCall(file.path)])
        ])
        var budget = AgentBudget.standard
        budget.maxSteps = 2
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: budget),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.error?.contains("step budget") == true, outcome.error ?? "")
        XCTAssertTrue(
            outcome.report?.summary.contains("the FIFO cannot admit two runs") == true,
            "a budget-exhausted run must keep the analysis it wrote, not its last preamble: \(outcome.report?.summary ?? "")"
        )
        XCTAssertTrue(outcome.report?.risks.first?.contains("step budget") == true)
        XCTAssertEqual(outcome.report?.findings.count, 2)
    }

    // MARK: - Isolation is about the call's destination

    func testAnIsolatedWriterMayNotEditOutsideItsWorktreeWithoutACard() async throws {
        let workspace = try makeWorkspace()
        let outside = try makeWorkspace()
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: outside)
        }
        let target = outside.appendingPathComponent("Operator.txt")

        let counter = CallCounter()
        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": target.path, "content": "escaped"])
            ]),
            .init(text: "refused", calls: [completeCall("Could not write outside the worktree.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.reason.lowercased().contains("outside"), request.reason)
                return false
            }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(counter.count, 1, "a path outside the worktree must produce a card")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: target.path),
            "a declined card must leave the operator's tree untouched"
        )
        XCTAssertTrue(outcome.transcript.contains {
            $0.toolRecord?.name == "file_write" && $0.toolRecord?.status == .rejected
        })
    }

    func testAnIsolatedWriterStillEditsInsideItsWorktreeWithoutACard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let target = workspace.appendingPathComponent("Inside.txt")
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": target.path, "content": "inside"])
            ]),
            .init(text: "done", calls: [completeCall("Wrote Inside.txt.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { _ in
                counter.increment()
                return true
            }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        XCTAssertEqual(counter.count, 0, "a write inside the worktree is what spawn-only approval is for")
        XCTAssertTrue(FileManager.default.fileExists(atPath: target.path))
    }

    /// A shell command executes arbitrary code (and takes its own `cwd`), so a
    /// worktree that merely EXISTS cannot make it safe. The policy falls back
    /// to the per-change card rather than treating isolation as proof.
    func testAnIsolatedShellCommandStillGetsACard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "running", calls: [AgentToolCall(name: "shell_run", arguments: ["command": "echo hello"])]),
            .init(text: "done", calls: [completeCall("Ran echo.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["shell_run", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.reason.lowercased().contains("outside"), request.reason)
                return true
            }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed)
        XCTAssertEqual(counter.count, 1)
    }

    // MARK: - Cost ceilings fail closed

    func testACostCeilingTheBackendCannotPriceFailsClosed() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Notes.txt")
        try "CONTENT".write(to: file, atomically: true, encoding: .utf8)

        var firstUsage = AgentUsage()
        firstUsage.promptTokens = 900_000
        firstUsage.completionTokens = 100
        firstUsage.costIsUnpriced = true

        let backend = ScriptedBackend(turns: [
            .init(text: "working", calls: [readCall(file.path)], usage: firstUsage),
            .init(text: "should not run")
        ])
        var budget = AgentBudget.standard
        budget.maxCostUSD = 1
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: budget),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.error?.contains("no price is recorded") == true, outcome.error ?? "")
        XCTAssertEqual(backend.requests.count, 1, "the ceiling stops the run instead of spending on")
    }
}
