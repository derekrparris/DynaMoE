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

/// Yields exactly once, for a drain or a gate that must fire a single time
/// from inside a `@Sendable` closure.
nonisolated private final class OnceFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false

    /// True the first time only.
    func take() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let was = !fired
        fired = true
        return was
    }
}

/// Resumes exactly one waiter, so a test can park a task and release it only
/// after it has been cancelled.
nonisolated private final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var resumed = false

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if resumed {
                lock.unlock()
                continuation.resume()
                return
            }
            self.continuation = continuation
            lock.unlock()
        }
    }

    func resume() {
        lock.lock()
        let waiter = continuation
        continuation = nil
        resumed = true
        lock.unlock()
        waiter?.resume()
    }
}

/// Bills one request into the out-of-band ledger, then parks until the run is
/// cancelled. That is the shape of a cloud turn a Stop interrupts: the
/// provider was paid, but no turn comes back to carry the usage.
nonisolated private final class BilledThenParkedBackend: AgentModelBackend, @unchecked Sendable {
    let descriptor = AgentBackendDescriptor(
        kind: .cloud,
        providerId: "billed-test",
        modelId: "billed-test-model",
        displayName: "Billed test provider"
    )
    private let started: AsyncStream<Void>.Continuation

    init(started: AsyncStream<Void>.Continuation) {
        self.started = started
    }

    func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn {
        var usage = AgentUsage()
        usage.backendRequests = 1
        usage.costIsUnpriced = true
        usage.costUnpricedReason = .providerOmittedUsage
        request.usageLedger?.add(usage)
        started.yield()
        try await Task.sleep(nanoseconds: 60_000_000_000)
        return AgentBackendTurn(text: "", finishReason: "stop")
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

    /// A cancelled run's `shell_run` used to keep writing for up to its 120 s
    /// timeout: `ControlledProcessRunner.runCommand` had no cancellation
    /// handler and no `terminate()`. The tool now spawns through
    /// `AgentHarness.runProcess`, which kills the process tree on cancellation.
    func testAShellCommandStopsWhenItsRunIsCancelled() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let tool = ShellRunTool()
        let start = Date()
        let call = Task {
            try await tool.execute(
                arguments: ["command": "sleep 5"],
                workingDirectory: workspace,
                maxOutputLength: 4_000
            )
        }
        try await Task.sleep(nanoseconds: 200_000_000)
        call.cancel()
        _ = try? await call.value
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(
            elapsed,
            3.0,
            "the child must be terminated at the Stop, not left to its own timeout"
        )
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
        // Context: the LAST request's size plus the tokens THAT step generated.
        // The earlier steps' output is already inside the last prompt, so
        // adding the summed completions would count it twice.
        XCTAssertEqual(outcome.usage.latestPromptTokens, 120)
        XCTAssertEqual(outcome.usage.latestCompletionTokens, 7)
        XCTAssertEqual(outcome.usage.contextTokens, 127)
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

    func testSteeringQueuedMidRunReachesTheNextStep() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let file = workspace.appendingPathComponent("Notes.txt")
        try "QUANTUM_RELAY_SEQUENCE".write(to: file, atomically: true, encoding: .utf8)
        // The first drain (step 0) sees nothing queued; the note arrives while
        // the run is between steps and must ride the NEXT request, exactly once.
        let stepZeroDrain = OnceFlag()
        let note = OnceFlag()
        var loopInput = input()
        loopInput.drainSteering = {
            guard stepZeroDrain.take() else {
                return note.take() ? [("coordinator", "focus on the token budget")] : []
            }
            return []
        }
        let backend = ScriptedBackend(turns: [
            .init(text: "Reading.", calls: [readCall(file.path)]),
            .init(text: "Done.", calls: [completeCall("Steered onto the budget.")])
        ])
        let context = try makeContext(workspace: workspace, tools: ["file_read", "complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: loopInput,
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed, outcome.error ?? "")
        guard backend.requests.count == 2 else {
            return XCTFail("the run must take two turns, took \(backend.requests.count)")
        }
        let secondRequest = backend.requests[1]
        XCTAssertTrue(
            secondRequest.messages.contains { message in
                if case .user(let text) = message {
                    return text.contains("[coordinator]")
                        && text.contains("focus on the token budget")
                }
                return false
            },
            "a steering message reported delivered must reach the model, not just the record"
        )
        XCTAssertTrue(outcome.transcript.contains {
            $0.kind == .status && $0.text?.contains("focus on the token budget") == true
        })
    }

    /// The deadline arm must not drop the work's answer. `withTaskGroup` of
    /// `T?` makes `next()` a `T??`, and unwrapping only the OUTER optional
    /// treated "the deadline won" as the result: a tool that finished at the
    /// boundary was reported as a wall-clock failure and its record disappeared
    /// from the transcript. The deadline here is short enough to win; the work
    /// is cancelled by the collect path and its value must still come back.
    func testADeadlineThatWinsStillCollectsTheCompletedResult() async {
        let value = await HeadlessAgentLoop.firstResult(deadlineNanoseconds: 2_000_000) {
            try? await Task.sleep(nanoseconds: 40_000_000)
            return 42
        }
        XCTAssertEqual(value, 42, "the cancelled tool's own answer is the result")

        let fast = await HeadlessAgentLoop.firstResult(deadlineNanoseconds: 1_000_000_000) {
            return 7
        }
        XCTAssertEqual(fast, 7, "work that finishes first is returned unchanged")
    }

    func testAStopRacingAToolIsCancelledNotAWallClockFailureAndItsApprovalDoesNotExecute() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        // The marker proves the second half: an approval that resolves after
        // the Stop must not execute the write it approved.
        let marker = workspace.appendingPathComponent("post-stop-write.txt")
        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "shell_run", arguments: ["command": "touch \(marker.path)"])
            ]),
            .init(text: "unreachable")
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["shell_run", "complete"],
            policy: .everyStateChange,
            handler: { _ in
                // The approval is decided "after" the Stop: the sleep is cut
                // short by the cancellation and the handler still says yes.
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                return true
            }
        )

        let task = Task {
            await HeadlessAgentLoop().run(
                input: input(budget: AgentBudget(
                    maxSteps: 4,
                    maxCompletionTokens: 2_000,
                    maxWallClockSeconds: 600,
                    maxCostUSD: nil
                )),
                backend: backend,
                context: context,
                emit: { _ in }
            )
        }
        try await Task.sleep(nanoseconds: 400_000_000)
        task.cancel()
        let outcome = await task.value

        XCTAssertEqual(outcome.status, .cancelled, outcome.error ?? "")
        XCTAssertFalse(
            outcome.error?.lowercased().contains("wall-clock") == true,
            "a Stop racing a tool must not be reported as the run's budget running out"
        )
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: marker.path),
            "a write approved after the Stop must not execute"
        )
    }

    func testRunProcessRefusesToSpawnFromACancelledTask() async throws {
        // A cancelled task must not spawn a child process: the cancellation
        // handler fires before `process.run()` and `terminate()` on the
        // never-launched task aborts the process. The gate guarantees the
        // cancellation is delivered BEFORE `runProcess` is entered.
        let gate = Gate()
        let task = Task { () -> Int32 in
            await gate.wait()
            do {
                let result = try await AgentHarness.runProcess(
                    executableURL: URL(fileURLWithPath: "/bin/echo"),
                    arguments: ["cancellation-guard"],
                    currentDirectory: FileManager.default.temporaryDirectory,
                    timeoutSeconds: 10
                )
                return result.exitCode
            } catch {
                return -1
            }
        }
        try await Task.sleep(nanoseconds: 100_000_000)
        task.cancel()
        gate.resume()
        let exit = await task.value
        XCTAssertEqual(
            exit, -1,
            "a task cancelled before the spawn must refuse instead of launching a child"
        )
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
            // The `false &&` keeps the command harmless if an approval-gate
            // regression ever lets it run: a test must not be able to rewrite
            // the developer's real Turbo setting. The classifier still sees
            // the `defaults write` on the app's domain, which is what the card
            // is testing.
            .init(text: "changing prefs", calls: [AgentToolCall(name: "shell_run", arguments: ["command": "false && defaults write DRP.DynaMoE dynamoe_agent_turbo_mode -bool NO"])]),
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

    /// A write the operator refused changed nothing, so the report must not
    /// claim the file as one of the run's changed files.
    func testARejectedWriteIsNotReportedAsAChangedFile() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let target = workspace.appendingPathComponent("Denied.txt")

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": target.path, "content": "no"])
            ]),
            .init(text: "done", calls: [completeCall("The write was denied.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .everyStateChange,
            handler: { _ in false }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertEqual(outcome.report?.changedFiles, [])
    }

    /// A withheld call used to `continue` before either loop guard saw it, so a
    /// model that kept asking for a tool it was never given ran to its step
    /// budget without escalating.
    func testARepeatedWithheldCallTripsTheLoopGuard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let target = workspace.appendingPathComponent("Never.txt")
        let withheld = AgentToolCall(
            name: "file_write",
            arguments: ["path": target.path, "content": "nope"]
        )

        let backend = ScriptedBackend(turns: [
            .init(text: "trying", calls: [withheld, withheld, withheld, withheld])
        ])
        let context = try makeContext(workspace: workspace, tools: ["complete"])

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: AgentBudget(
                maxSteps: 4,
                maxCompletionTokens: 200,
                maxWallClockSeconds: 20,
                maxCostUSD: nil
            )),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .blocked)
        XCTAssertTrue(outcome.error?.contains("loop guard") == true, outcome.error ?? "no error")
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
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
            .init(
                text: "CONTENT was found in Notes.txt, so the reader works.",
                calls: [completeCall("Read Notes.txt.")]
            )
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
        XCTAssertEqual(findings.count, 2, "two claims were made, so the report carries two findings")
        XCTAssertEqual(findings.first?.claim, claim)
        XCTAssertTrue(findings.first?.evidence.contains("step 1") == true, findings.first?.evidence ?? "")
        // A step's own tools run AFTER its text: the scheduler claim was
        // written before the read it asked for, so its evidence must not cite
        // a result it had not seen. (It used to, which is what the review
        // called out: evidence for a claim the claim could not have used.)
        XCTAssertFalse(
            findings.first?.evidence.contains("file_read") == true,
            "a claim cannot cite a tool result that ran after it was written: \(findings.first?.evidence ?? "")"
        )
        // The second claim comes after the read completed, so it may.
        let later = try XCTUnwrap(findings.last)
        XCTAssertTrue(
            later.evidence.contains("file_read") == true,
            "a claim made after a read cites the read: \(later.evidence)"
        )

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

    // MARK: - The sandbox's own git metadata (merge blocker B1)

    /// A linked worktree's `.git` is a regular FILE inside the containment
    /// prefix, so it passes the plain prefix check while being the one file
    /// whose rewrite breaks the sandbox: git would stage and commit wherever
    /// the retargeted pointer says. It must get its own card, marked as a
    /// containment escape so Turbo Mode cannot bypass it.
    func testADotGitWriteInsideTheSandboxNeedsItsOwnCard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        try "gitdir: /elsewhere/.git\n".write(
            to: workspace.appendingPathComponent(".git"),
            atomically: true,
            encoding: .utf8
        )
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "tampering", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": ".git", "content": "gitdir: /tmp/attacker.git"])
            ]),
            .init(text: "done", calls: [completeCall("Could not rewrite the pointer.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.containmentEscape, "a .git write is a containment escape")
                XCTAssertTrue(request.reason.lowercased().contains("git metadata"), request.reason)
                return false
            }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(counter.count, 1, "a .git write inside the sandbox must produce a card")
        let pointer = try String(contentsOf: workspace.appendingPathComponent(".git"), encoding: .utf8)
        XCTAssertTrue(pointer.contains("/elsewhere"), "a declined card must leave the pointer untouched")
        XCTAssertTrue(outcome.transcript.contains {
            $0.toolRecord?.name == "file_write" && $0.toolRecord?.status == .rejected
        })
    }

    /// Any `.git` path COMPONENT counts — a nested repository's metadata can
    /// smuggle the same retargeting — not just the worktree's own pointer.
    func testANestedDotGitPathComponentNeedsTheCard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": "vendor/sub/.git/config", "content": "x"])
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.containmentEscape)
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: workspace.appendingPathComponent("vendor/sub/.git/config").path)
        )
    }

    /// `.gitignore` and friends are ordinary work files: only a component
    /// spelled exactly `.git` is metadata, and a write to one stays cardless
    /// inside an isolated workspace.
    func testADotGitignoreIsNotGitMetadataAndStaysCardless() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": ".gitignore", "content": "build/\n"])
            ]),
            .init(text: "done", calls: [completeCall("Added .gitignore.")])
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

        let outcome = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(outcome.status, .completed)
        XCTAssertEqual(counter.count, 0, "an ordinary work file inside the worktree is not git metadata")
        XCTAssertTrue(FileManager.default.fileExists(atPath: workspace.appendingPathComponent(".gitignore").path))
    }

    /// A symlink inside the worktree pointing outside it is an escape, and the
    /// containment guard resolves symlinks before the prefix comparison — this
    /// is the case that fails if `resolvingSymlinksInPath()` is ever dropped.
    func testASymlinkEscapeFromTheWorkspaceNeedsTheCard() async throws {
        let workspace = try makeWorkspace()
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: workspace.deletingLastPathComponent().appendingPathComponent("escape-target"))
        }
        let outside = workspace.deletingLastPathComponent()
            .appendingPathComponent("escape-target", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: workspace.appendingPathComponent("shortcut"),
            withDestinationURL: outside
        )
        let target = workspace.appendingPathComponent("shortcut/escaped.txt")
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": target.path, "content": "escaped"])
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.containmentEscape, "a symlink escape is a containment failure")
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1, "a write through an escaping symlink must produce a card")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: outside.appendingPathComponent("escaped.txt").path),
            "a declined card must leave the target outside the worktree untouched"
        )
    }

    /// `..` inside a path that still PRINTS inside the prefix is an escape
    /// after standardisation, and must card.
    func testADotDotEscapeNeedsTheCard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": "../\(workspace.lastPathComponent)-escape.txt", "content": "escaped"])
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.containmentEscape)
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: workspace.deletingLastPathComponent().appendingPathComponent("\(workspace.lastPathComponent)-escape.txt").path
            )
        )
    }

    /// A symlink escape and `..` in ONE path: `<worktree>/link/../x`. The
    /// kernel follows `link` first and then applies `..`, which lands outside
    /// the worktree, while a lexical `standardizedFileURL` collapses the pair
    /// to `<worktree>/x` and calls it inside. Resolving components in order is
    /// what keeps the two answers the same; this is the case that fails if
    /// `standardizedFileURL` is ever put back in front of the symlink walk.
    func testASymlinkFollowedByDotDotEscapesTheWorkspace() async throws {
        let workspace = try makeWorkspace()
        let outside = workspace.deletingLastPathComponent()
            .appendingPathComponent("\(workspace.lastPathComponent)-combo-target", isDirectory: true)
        let landed = workspace.deletingLastPathComponent()
            .appendingPathComponent("\(workspace.lastPathComponent)-combo-escaped.txt")
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: outside)
            try? FileManager.default.removeItem(at: landed)
        }
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: workspace.appendingPathComponent("shortcut"),
            withDestinationURL: outside
        )
        // `<worktree>/shortcut/../<file>`: the kernel resolves `shortcut` and
        // then applies `..`, so the write lands beside `outside`, never inside
        // the worktree a lexical standardisation would name.
        let target = workspace.appendingPathComponent("shortcut/../\(landed.lastPathComponent)")
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(name: "file_write", arguments: ["path": target.path, "content": "escaped"])
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.containmentEscape, "a symlink followed by .. is a containment failure")
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1, "the combined escape must produce a card")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: landed.path),
            "a declined card must leave the target outside the worktree untouched"
        )
    }

    /// A symlink whose target does not exist yet is invisible to Foundation:
    /// `fileExists` follows the link and says no, and
    /// `resolvingSymlinksInPath` leaves the link in place. An `O_CREAT` write
    /// still follows it, so leaving the link path "inside" would let the write
    /// land anywhere. The link's own destination is what the guard must read.
    func testADanglingSymlinkEscapeNeedsTheCard() async throws {
        let workspace = try makeWorkspace()
        let outside = workspace.deletingLastPathComponent()
            .appendingPathComponent("\(workspace.lastPathComponent)-dangling-target.txt")
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: outside)
        }
        try FileManager.default.createSymbolicLink(
            atPath: workspace.appendingPathComponent("dangling").path,
            withDestinationPath: outside.path
        )
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(
                    name: "file_write",
                    arguments: ["path": workspace.appendingPathComponent("dangling").path, "content": "escaped"]
                )
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.containmentEscape, "a dangling symlink is still an escape")
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1, "the dangling link's write must produce a card")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: outside.path),
            "a declined card must not create the link's target"
        )
    }

    // MARK: - Worker prompts match the tool surface

    /// Plan mode withholds `complete` and refuses it with "do not retry", so a
    /// worker prompt that says "call `complete`" describes an ending the run
    /// cannot have. The local variant carries the plan directive and the
    /// ready-marker ending; the code variant keeps the `complete` contract.
    func testThePlanModeWorkerPromptMatchesItsToolSurface() throws {
        let planHarness = AgentHarness.makeRunScoped()
        XCTAssertNoThrow(try planHarness.configureRunScope(
            runId: UUID(),
            toolNames: ["file_read", "grep_search", "complete"],
            workingDirectory: FileManager.default.temporaryDirectory,
            mode: .plan,
            approvalPolicy: .deny,
            workspaceIsolated: false,
            approvalHandler: nil
        ))
        let planPrompt = planHarness.buildWorkerSystemPrompt(baseSystem: "", role: "Researcher", mode: .plan)
        XCTAssertTrue(planPrompt.contains("# Plan Mode"), planPrompt)
        XCTAssertTrue(planPrompt.contains(PlanMarkers.ready), "the plan ending must name the ready marker")
        XCTAssertFalse(
            planPrompt.contains("call `complete` with a summary"),
            "plan mode withholds complete; the prompt must not teach an impossible ending"
        )
        XCTAssertTrue(planPrompt.contains("withheld in this run"), planPrompt)

        let codeHarness = AgentHarness.makeRunScoped()
        XCTAssertNoThrow(try codeHarness.configureRunScope(
            runId: UUID(),
            toolNames: ["file_read", "complete"],
            workingDirectory: FileManager.default.temporaryDirectory,
            mode: .code,
            approvalPolicy: .auto,
            workspaceIsolated: false,
            approvalHandler: nil
        ))
        let codePrompt = codeHarness.buildWorkerSystemPrompt(baseSystem: "", role: "Researcher", mode: .code)
        XCTAssertFalse(codePrompt.contains("# Plan Mode"))
        XCTAssertTrue(codePrompt.contains("call `complete` with a summary"), codePrompt)

        // The cloud worker's completion rule is plan-aware too.
        let cloudPlan = codeHarness.buildCloudWorkerSystemPrompt(baseSystem: "", role: "Researcher", mode: .plan)
        XCTAssertTrue(cloudPlan.contains(PlanMarkers.ready), cloudPlan)
        XCTAssertFalse(cloudPlan.contains("call `complete` with a summary"), "plan mode must not teach an impossible ending on the cloud path either")
        let cloudCode = codeHarness.buildCloudWorkerSystemPrompt(baseSystem: "", role: "Researcher", mode: .code)
        XCTAssertTrue(cloudCode.contains("call `complete` with a summary"), cloudCode)
    }

    // MARK: - In-loop context compaction

    private func assistantBlock(text: String, thinking: String = "", toolResults: [String] = []) -> [AgentBackendMessage] {
        var block: [AgentBackendMessage] = [.assistant(text: text, thinking: thinking, calls: [])]
        for (index, content) in toolResults.enumerated() {
            block.append(.toolResult(callId: "c\(index)", name: "file_read", content: content, isError: false))
        }
        return block
    }

    func testCompactionLeavesAShortHistoryAlone() {
        var messages: [AgentBackendMessage] = [.user("Do the task.")]
        messages.append(contentsOf: assistantBlock(text: "one", thinking: "thinking one", toolResults: ["r1"]))
        messages.append(contentsOf: assistantBlock(text: "two", thinking: "thinking two", toolResults: ["r2"]))

        let compacted = HeadlessAgentLoop.compactContextIfNeeded(messages)
        XCTAssertTrue(
            sameMessages(compacted, messages),
            "two recent turns are inside the kept window and must not change"
        )
    }

    /// The budget covers the whole prompt. A history just under it plus a
    /// large system block / tool catalogue used to overflow the engine AFTER
    /// the spend, because only the message texts were counted.
    func testCompactionCountsTheSystemBlockAndTheToolSchemas() {
        var messages: [AgentBackendMessage] = [.user("Do the task.")]
        for step in 0..<6 {
            messages.append(contentsOf: assistantBlock(
                text: "step \(step)",
                thinking: "",
                toolResults: ["result \(step)"]
            ))
        }
        XCTAssertTrue(
            sameMessages(
                HeadlessAgentLoop.compactContextIfNeeded(messages),
                messages
            ),
            "control: under the budget the history is untouched"
        )
        let compacted = HeadlessAgentLoop.compactContextIfNeeded(
            messages,
            overheadCharacters: HeadlessAgentLoop.contextCharacterBudget
        )
        XCTAssertFalse(
            sameMessages(compacted, messages),
            "the overhead alone must be enough to trigger folding"
        )
    }

    /// A `file_write` body is the largest thing a tool call can carry, and it
    /// is part of the prompt. It was not counted, so arguments alone could
    /// cross the budget after the spend.
    func testCompactionCountsToolCallArguments() {
        let hugeArgument = String(repeating: "x", count: HeadlessAgentLoop.contextCharacterBudget)
        var messages: [AgentBackendMessage] = [.user("Do the task.")]
        messages.append(.assistant(
            text: "writing",
            thinking: "",
            calls: [
                AgentToolCall(
                    id: "c1",
                    name: "file_write",
                    arguments: ["content": hugeArgument],
                    rawArguments: hugeArgument,
                    rawText: ""
                )
            ]
        ))
        for step in 0..<3 {
            messages.append(contentsOf: assistantBlock(text: "step \(step)", toolResults: ["r\(step)"]))
        }
        XCTAssertFalse(
            sameMessages(HeadlessAgentLoop.compactContextIfNeeded(messages), messages),
            "the argument alone exceeds the budget, so the history must fold"
        )
    }

    /// The prefix-pin contract: while the history is under the character
    /// budget it comes back UNCHANGED even when it has aged well past the kept
    /// window, because folding rewrites bytes in the middle of the prompt and
    /// a rewritten prompt is not an extension of the previous step's — the
    /// local backend's KV pin only survives steps whose prompts extend the
    /// last one byte for byte. Folding on turn count made every step from the
    /// third on re-render the turn that had just aged, so no multi-step local
    /// run ever reused its pin.
    func testCompactionLeavesAnAgedHistoryAloneUnderTheBudget() {
        var messages: [AgentBackendMessage] = [.user("Do the task.")]
        for step in 0..<6 {
            messages.append(contentsOf: assistantBlock(
                text: "step \(step)",
                thinking: "thinking \(step)",
                toolResults: ["result \(step)"]
            ))
        }

        let compacted = HeadlessAgentLoop.compactContextIfNeeded(messages)
        XCTAssertTrue(
            sameMessages(compacted, messages),
            "an over-window history under the budget must stay byte-identical so each step extends the previous prompt"
        )
    }

    func testCompactionFoldsAgedTurnsAndKeepsTheRecentWindowVerbatim() {
        var messages: [AgentBackendMessage] = [.user("Do the task.")]
        // Sized past the character budget: folding is budget-gated now, so a
        // fixture that only aged past the kept window would come back
        // unchanged. Thirty thousand characters a block over seven blocks puts
        // the raw history at ~210k against the 160k budget.
        let longText = String(repeating: "a", count: 10_000)
        let longResult = String(repeating: "b", count: 10_000)
        for step in 0..<6 {
            messages.append(contentsOf: assistantBlock(
                text: "step \(step) " + longText,
                thinking: "thinking \(step) " + longText,
                toolResults: ["result \(step) " + longResult]
            ))
        }
        messages.append(contentsOf: assistantBlock(text: "recent", thinking: "recent thinking", toolResults: ["recent result"]))

        let compacted = HeadlessAgentLoop.compactContextIfNeeded(messages)

        // The task and the newest TWO turns are verbatim.
        guard case .user(let task) = compacted.first else {
            return XCTFail("the task message must stay first")
        }
        XCTAssertEqual(task, "Do the task.")
        for verbatim in ["step 5 " + longText, "recent"] {
            XCTAssertTrue(compacted.contains { message in
                if case .assistant(let text, _, _) = message { return text == verbatim }
                return false
            }, "the recent window must keep \(verbatim.prefix(12)) verbatim")
        }
        XCTAssertTrue(compacted.contains { message in
            if case .assistant(_, let thinking, _) = message { return thinking == "recent thinking" }
            return false
        }, "the newest turn keeps its thinking")

        // The five aged turns lose their thinking and fold their results.
        for aged in 0..<5 {
            XCTAssertTrue(compacted.contains { message in
                if case .assistant(let text, let thinking, _) = message {
                    return text.hasPrefix("step \(aged)") && thinking.isEmpty
                }
                return false
            }, "aged turn \(aged) must lose its thinking")
            XCTAssertTrue(compacted.contains { message in
                if case .toolResult(_, _, let content, _) = message {
                    return content.hasPrefix("result \(aged)") && content.contains("[compacted:")
                }
                return false
            }, "aged turn \(aged)'s result must carry the fold marker")
        }
    }

    func testCompactionDropsTheOldestBlocksWhenFoldingCannotFit() {
        var messages: [AgentBackendMessage] = [.user("Do the task.")]
        let huge = String(repeating: "x", count: 4_000)
        for step in 0..<90 {
            messages.append(contentsOf: assistantBlock(text: huge, thinking: "", toolResults: [huge, huge]))
            _ = step
        }
        messages.append(contentsOf: assistantBlock(text: "recent", thinking: "", toolResults: ["recent result"]))

        let compacted = HeadlessAgentLoop.compactContextIfNeeded(messages)
        let total = compacted.reduce(0) { $0 + $1.textAndThinkingCount }
        XCTAssertLessThanOrEqual(total, HeadlessAgentLoop.contextCharacterBudget, "the folded history must fit the budget")

        // The task and the recent window survive the drop pass.
        XCTAssertTrue(compacted.contains { message in
            if case .user(let text) = message { return text == "Do the task." }
            return false
        })
        XCTAssertTrue(compacted.contains { message in
            if case .assistant(let text, _, _) = message { return text == "recent" }
            return false
        })
        XCTAssertLessThan(
            compacted.count, messages.count,
            "a history this far over budget must lose whole oldest blocks"
        )
    }

    /// A steering note can arrive BETWEEN two assistant turns. The drop pass
    /// used to end a block only at the next assistant, so the range swallowed
    /// the note in front of it and a mid-run directive vanished once the
    /// history crossed the budget.
    func testTheDropPassNeverDropsASteeringNote() {
        var messages: [AgentBackendMessage] = [.user("Do the task.")]
        let huge = String(repeating: "x", count: 4_000)
        let notes = ["focus on the token budget", "do not touch the network"]
        for step in 0..<90 {
            messages.append(contentsOf: assistantBlock(text: huge, thinking: "", toolResults: [huge, huge]))
            if step == 10 || step == 40 {
                messages.append(.user(notes[step == 10 ? 0 : 1]))
            }
        }
        messages.append(contentsOf: assistantBlock(text: "recent", thinking: "", toolResults: ["recent result"]))

        let compacted = HeadlessAgentLoop.compactContextIfNeeded(messages)
        for note in notes {
            XCTAssertTrue(
                compacted.contains { message in
                    if case .user(let text) = message { return text == note }
                    return false
                },
                "a steering note must survive the whole block-drop pass: \(note)"
            )
        }
        XCTAssertLessThanOrEqual(
            compacted.reduce(0) { $0 + $1.textAndThinkingCount },
            HeadlessAgentLoop.contextCharacterBudget
        )
    }

    /// A turn can emit several calls. When a Stop lands mid-turn, every call
    /// that never ran still needs a result entry: the assistant text names
    /// them, and a transcript with calls and no result read as if they had
    /// silently succeeded.
    func testACancelledTurnRecordsTheCallsItNeverRan() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let gate = Gate()
        let backend = ScriptedBackend(turns: [
            .init(text: "three calls", calls: [
                AgentToolCall(name: "shell_run", arguments: ["command": "touch blocked.txt"]),
                AgentToolCall(name: "file_read", arguments: ["path": "one.txt"]),
                AgentToolCall(name: "file_read", arguments: ["path": "two.txt"])
            ])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["shell_run", "file_read", "complete"],
            policy: .everyStateChange,
            handler: { _ in
                await gate.wait()
                return true
            }
        )

        let task = Task {
            await HeadlessAgentLoop().run(
                input: input(budget: AgentBudget(
                    maxSteps: 4,
                    maxCompletionTokens: 2_000,
                    maxWallClockSeconds: 600,
                    maxCostUSD: nil
                )),
                backend: backend,
                context: context,
                emit: { _ in }
            )
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        task.cancel()
        gate.resume()
        let outcome = await task.value

        XCTAssertEqual(outcome.status, .cancelled, outcome.error ?? "")
        let abandoned = outcome.transcript.compactMap { entry -> ToolCallRecord? in
            guard entry.kind == .toolResult,
                  let record = entry.toolRecord,
                  record.error?.contains("Not completed") == true else { return nil }
            return record
        }
        XCTAssertEqual(
            abandoned.map(\.name), ["file_read", "file_read"],
            "the two calls behind the stopped one must each get a result"
        )
    }

    /// `complete` succeeds without its (schema-required) summary argument, and
    /// a run whose only output is that call must still complete rather than
    /// Calls the model emits after a valid `complete` are recorded as abandoned
    /// instead of vanishing at the break.
    func testCompleteWithSummaryRecordsLaterCallsAsAbandoned() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [
            .init(text: "", calls: [
                AgentToolCall(name: "complete", arguments: ["summary": "All done."]),
                AgentToolCall(name: "file_read", arguments: ["path": "never.txt"])
            ])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["complete", "file_read"],
            policy: .everyStateChange,
            handler: { _ in true }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: AgentBudget(
                maxSteps: 4,
                maxCompletionTokens: 2_000,
                maxWallClockSeconds: 600,
                maxCostUSD: nil
            )),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed, outcome.error ?? "")
        XCTAssertEqual(outcome.report?.summary, "All done.")
        let abandoned = outcome.transcript.compactMap { entry -> ToolCallRecord? in
            guard entry.kind == .toolResult,
                  let record = entry.toolRecord,
                  record.error?.contains("Not completed") == true else { return nil }
            return record
        }
        XCTAssertEqual(abandoned.map(\.name), ["file_read"])
    }

    /// A `complete` call with missing/empty required arguments must be refused
    /// without fabricating a completed run; the model can recover on a later step.
    func testEmptyCompleteIsRefusedByRequiredArgumentGuard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [
            .init(text: "", calls: [
                AgentToolCall(name: "complete", arguments: [:])
            ]),
            .init(text: "", calls: [
                AgentToolCall(name: "complete", arguments: ["summary": "Recovered with summary."])
            ])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["complete"],
            policy: .everyStateChange,
            handler: { _ in true }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: AgentBudget(
                maxSteps: 4,
                maxCompletionTokens: 2_000,
                maxWallClockSeconds: 600,
                maxCostUSD: nil
            )),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed, outcome.error ?? "")
        XCTAssertEqual(outcome.report?.summary, "Recovered with summary.")
        let errors = outcome.transcript.compactMap { entry -> ToolCallRecord? in
            guard entry.kind == .toolResult,
                  let record = entry.toolRecord,
                  record.status == .error else { return nil }
            return record
        }
        XCTAssertEqual(errors.count, 1)
        XCTAssertTrue(errors.first?.error?.contains("required arguments") == true)
    }

    /// Malformed JSON arguments (e.g. a cut-off tool call) must be refused
    /// instead of executing the tool with empty arguments.
    func testMalformedToolCallArgumentsAreRefusedWithoutExecution() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [
            .init(text: "", calls: [
                AgentToolCall(
                    name: "complete",
                    arguments: [:],
                    rawArguments: #"{"summary":"cut"#,
                    argumentsMalformed: true
                )
            ]),
            .init(text: "Done after retry.")
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["complete"],
            policy: .everyStateChange,
            handler: { _ in true }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: AgentBudget(
                maxSteps: 4,
                maxCompletionTokens: 2_000,
                maxWallClockSeconds: 600,
                maxCostUSD: nil
            )),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .completed, outcome.error ?? "")
        XCTAssertEqual(outcome.report?.summary, "Done after retry.")
        let malformed = outcome.transcript.compactMap { entry -> ToolCallRecord? in
            guard entry.kind == .toolResult,
                  let record = entry.toolRecord,
                  record.status == .error,
                  record.error?.contains("Malformed arguments JSON") == true else { return nil }
            return record
        }
        XCTAssertEqual(malformed.count, 1)
    }

    /// A token-capped turn (`finish_reason == "length"`) with a `complete` call
    /// must fail the run rather than executing the truncated call.
    func testTruncatedTurnWithToolCallsFailsWithoutExecutingComplete() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [
            .init(
                text: "",
                calls: [AgentToolCall(name: "complete", arguments: ["summary": "Cut off"])],
                finishReason: "length"
            )
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["complete"],
            policy: .everyStateChange,
            handler: { _ in true }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: AgentBudget(
                maxSteps: 4,
                maxCompletionTokens: 2_000,
                maxWallClockSeconds: 600,
                maxCostUSD: nil
            )),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.error?.contains("token limit") == true, outcome.error ?? "")
    }

    /// A blank turn (no text, no calls, finishReason == "stop") must fail rather
    /// than reporting `.completed`.
    func testBlankTurnWithStopFinishReasonFailsInsteadOfCompleting() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let backend = ScriptedBackend(turns: [
            .init(text: "   \n  ", calls: [], finishReason: "stop")
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_read"],
            policy: .everyStateChange,
            handler: { _ in true }
        )

        let outcome = await HeadlessAgentLoop().run(
            input: input(budget: AgentBudget(
                maxSteps: 4,
                maxCompletionTokens: 2_000,
                maxWallClockSeconds: 600,
                maxCostUSD: nil
            )),
            backend: backend,
            context: context,
            emit: { _ in }
        )

        XCTAssertEqual(outcome.status, .failed)
        XCTAssertTrue(outcome.error?.contains("empty turn") == true, outcome.error ?? "")
    }

    func testAgentUsageArithmeticSaturatesInsteadOfTrappingOnOverflow() {
        var a = AgentUsage()
        a.promptTokens = Int.max - 5
        a.completionTokens = Int.max - 5
        a.latestPromptTokens = Int.max - 5
        a.latestCompletionTokens = 100
        var b = AgentUsage()
        b.promptTokens = 100
        b.completionTokens = 100
        a.merge(b)
        XCTAssertEqual(a.promptTokens, Int.max)
        XCTAssertEqual(a.completionTokens, Int.max)
        XCTAssertEqual(a.totalTokens, Int.max)
        XCTAssertEqual(a.contextTokens, Int.max)
    }

    // MARK: - Turbo cannot bypass the two reserved cards

    /// On case-insensitive APFS a not-yet-existing `.GIT` is the same directory
    /// as `.git`. The component check used to be case-sensitive, so a write
    /// through that spelling got no card and Turbo was not even consulted.
    func testADotGitPathInAnyCaseGetsItsOwnCard() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(
                    name: "file_write",
                    arguments: ["path": "vendor/sub/.GIT/config", "content": "x"]
                )
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .spawnOnly,
            isolated: true,
            handler: { request in
                counter.increment()
                XCTAssertTrue(request.containmentEscape)
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1, "a case-variant .GIT is still git metadata")
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: workspace.appendingPathComponent("vendor/sub/.GIT/config").path
            )
        )
    }

    /// SUBAGENTS.md §7 reserves a shell for a real card in every policy. The
    /// `.everyStateChange` branch used to ask with `containmentEscape: false`,
    /// which Turbo Mode is allowed to skip.
    func testAShellRunInEveryStateChangeIsACardTurboCannotBypass() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "running", calls: [
                AgentToolCall(name: "shell_run", arguments: ["command": "echo hello"])
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["shell_run", "complete"],
            policy: .everyStateChange,
            handler: { request in
                counter.increment()
                XCTAssertTrue(
                    request.containmentEscape,
                    "a shell must never be bypassable by Turbo Mode"
                )
                let bypassed = await MainActor.run {
                    SubagentApprovalCenter.turboBypasses(
                        request: request,
                        honorsTurbo: true,
                        turboEnabled: true
                    )
                }
                XCTAssertFalse(bypassed)
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1)
    }

    /// The same rule for a write whose destination is outside the run's
    /// workspace. `.spawnOnly` already marks it an escape; `.everyStateChange`
    /// asked for a Turbo-bypassable card, so an unisolated run could write
    /// anywhere with no approval at all.
    func testAnOutOfWorkspaceWriteInEveryStateChangeIsACardTurboCannotBypass() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-escape-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: outside) }
        let counter = CallCounter()

        let backend = ScriptedBackend(turns: [
            .init(text: "writing", calls: [
                AgentToolCall(
                    name: "file_write",
                    arguments: ["path": outside.path, "content": "escaped"]
                )
            ]),
            .init(text: "done", calls: [completeCall("Refused.")])
        ])
        let context = try makeContext(
            workspace: workspace,
            tools: ["file_write", "complete"],
            policy: .everyStateChange,
            handler: { request in
                counter.increment()
                XCTAssertTrue(
                    request.containmentEscape,
                    "a write outside the workspace is a containment escape"
                )
                let bypassed = await MainActor.run {
                    SubagentApprovalCenter.turboBypasses(
                        request: request,
                        honorsTurbo: true,
                        turboEnabled: true
                    )
                }
                XCTAssertFalse(bypassed)
                return false
            }
        )

        _ = await HeadlessAgentLoop().run(input: input(), backend: backend, context: context, emit: { _ in })
        XCTAssertEqual(counter.count, 1)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: outside.path),
            "a declined escape must not be written"
        )
    }

    /// A Stop persists the cancelled record BEFORE the loop unwinds, so the
    /// spend of the step it interrupted used to be lost: the provider billed
    /// the attempt, the ledger held it, and the inspector reported $0. The
    /// terminal write now folds the loop's usage back in.
    func testACancelledRunsInFlightSpendReachesItsRecord() async throws {
        let (started, sink) = AsyncStream<Void>.makeStream()
        let runId = UUID()
        let session = UUID()
        let backend = BilledThenParkedBackend(started: sink)
        let plan = AgentSpawnPlan(
            role: "Implementer",
            taskDescription: "Spend one billed request.",
            contextSummary: nil,
            preset: .implementer,
            backendDescriptor: backend.descriptor,
            schedulingClass: .cloud(providerId: "billed-test"),
            backend: backend,
            toolNames: ["complete"],
            budget: AgentBudget(
                maxSteps: 2,
                maxCompletionTokens: 1_000,
                maxWallClockSeconds: 60,
                maxCostUSD: nil
            ),
            approvalPolicy: .auto,
            isolation: .none,
            thinking: false,
            sampling: .default,
            priority: .userRequested,
            parentSessionId: session,
            parentTurnId: nil,
            workingDirectory: FileManager.default.temporaryDirectory
        )
        let subagent = SubagentManager.shared.spawnModelBacked(plan, id: runId)
        defer {
            SubagentManager.shared.clearCompleted(sessionId: session)
            AgentRunStore.shared.delete(runId: runId, sessionId: session)
        }

        // The backend has billed a request and is parked inside it.
        for await _ in started { break }
        subagent.cancel()
        _ = await subagent.waitForCompletion()

        // Cancel's own persist lands first; the loop's usage write follows it.
        let deadline = Date().addingTimeInterval(5)
        var stored = AgentRunStore.shared.loadAny(runId: runId)
        while Date() < deadline, (stored?.usage.backendRequests ?? 0) == 0 {
            try await Task.sleep(nanoseconds: 20_000_000)
            stored = AgentRunStore.shared.loadAny(runId: runId)
        }

        let record = try XCTUnwrap(stored)
        XCTAssertEqual(record.status, .cancelled)
        XCTAssertEqual(
            record.usage.backendRequests, 1,
            "the billed attempt that never produced a turn must reach the record"
        )
        XCTAssertTrue(record.usage.costIsUnpriced)
        XCTAssertEqual(record.usage.costUnpricedReason, .providerOmittedUsage)
    }
}

private extension AgentBackendMessage {
    var textAndThinkingCount: Int {
        switch self {
        case .user(let text): return text.count
        case .assistant(let text, let thinking, _): return text.count + thinking.count
        case .toolResult(_, _, let content, _): return content.count
        }
    }
}

/// Structural equality for the compaction tests: `AgentToolCall` carries
/// `Any` arguments, so the enum cannot synthesize `Equatable`, and the
/// compaction never touches the calls anyway.
private func sameMessages(_ lhs: [AgentBackendMessage], _ rhs: [AgentBackendMessage]) -> Bool {
    guard lhs.count == rhs.count else { return false }
    for (a, b) in zip(lhs, rhs) {
        switch (a, b) {
        case (.user(let x), .user(let y)) where x == y:
            continue
        case (.assistant(let xt, let xh, let xc), .assistant(let yt, let yh, let yc))
        where xt == yt && xh == yh && xc.map(\.id) == yc.map(\.id):
            continue
        case (.toolResult(let xid, let xn, let xc, let xe), .toolResult(let yid, let yn, let yc, let ye))
        where xid == yid && xn == yn && xc == yc && xe == ye:
            continue
        default:
            return false
        }
    }
    return true
}
