//
//  SubagentLoopTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for the model-backed subagent runtime. No model is
//  required: the ScriptedBackend drives the same loop the real backends use.
//

import XCTest
@testable import DynaMoE

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
}
