//
//  AgentRunIntegrationTests.swift
//  DynaMoETests
//
//  A model-backed run end to end through SubagentManager, with a scripted
//  backend so no checkpoint or network is involved. This is the seam Phase 2
//  rewires: manager -> AgentRunContext -> HeadlessAgentLoop -> SubagentInstance.
//

import XCTest
@testable import DynaMoE

@MainActor
final class AgentRunIntegrationTests: XCTestCase {
    private var workingDirectory: URL!

    override func setUp() async throws {
        try await super.setUp()
        workingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubagentRunTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        await MainActor.run { SubagentManager.shared.cancelAll() }
        if let workingDirectory {
            try? FileManager.default.removeItem(at: workingDirectory)
        }
        try await super.tearDown()
    }

    private func plan(
        backend: ScriptedBackend,
        sessionId: UUID,
        toolNames: [String] = ["file_read"]
    ) -> AgentSpawnPlan {
        AgentSpawnPlan(
            role: "Codebase Researcher",
            taskDescription: "Find the scheduler.",
            contextSummary: nil,
            preset: .explore,
            backendDescriptor: backend.descriptor,
            schedulingClass: .local,
            backend: backend,
            toolNames: toolNames,
            budget: AgentBudget(
                maxSteps: 3,
                maxCompletionTokens: 200,
                maxWallClockSeconds: 20,
                maxCostUSD: nil
            ),
            approvalPolicy: .auto,
            isolation: .none,
            thinking: false,
            sampling: .default,
            priority: .userRequested,
            parentSessionId: sessionId,
            parentTurnId: nil,
            workingDirectory: workingDirectory
        )
    }

    func testAModelBackedRunCompletesAndReportsItsBackend() async throws {
        var usage = AgentUsage()
        usage.promptTokens = 40
        usage.completionTokens = 9
        usage.backendRequests = 1
        usage.steps = 1
        let backend = ScriptedBackend(turns: [
            .init(text: "The scheduler lives in AgentScheduler.swift.", usage: usage)
        ])
        let session = UUID()
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(backend: backend, sessionId: session),
            id: UUID()
        )

        let summary = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .completed)
        XCTAssertTrue(summary.contains("AgentScheduler.swift"), "the model's answer must survive")
        XCTAssertTrue(summary.contains("Backend: Scripted"), "a report names its backend (invariant 11)")
        XCTAssertEqual(subagent.report?.status, .completed)
        XCTAssertEqual(subagent.backendDescriptor, .scripted)
        XCTAssertEqual(subagent.usage.completionTokens, 9)
        XCTAssertEqual(subagent.usage.promptTokens, 40)
        XCTAssertTrue(subagent.transcript.contains { $0.actionName == "assistant" })
        XCTAssertEqual(backend.requests.count, 1)
        XCTAssertFalse(subagent.isSummaryRelayed, "the manager never marks a run relayed on its own")
    }

    func testTheRunRendersTheWorkersPromptWithItsOwnTools() async throws {
        let backend = ScriptedBackend(turns: [.init(text: "done")])
        let session = UUID()
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(backend: backend, sessionId: session, toolNames: ["file_read", "grep_search"]),
            id: UUID()
        )
        _ = await subagent.waitForCompletion()

        let request = try XCTUnwrap(backend.requests.first)
        XCTAssertTrue(request.systemPrompt.contains("# Subagent Role"))
        XCTAssertTrue(request.systemPrompt.contains("Codebase Researcher"))
        let names = request.tools.map(\.function.name)
        XCTAssertEqual(names.sorted(), ["file_read", "grep_search"])
        XCTAssertEqual(request.sessionId, subagent.id, "a run's prefix pin is keyed by the run")
        XCTAssertTrue(request.messages.count >= 1)
    }

    func testStoppingASessionCancelsItsRun() async throws {
        let backend = ScriptedBackend(turns: [
            .init(text: "late", delayNanoseconds: 3_000_000_000)
        ])
        let session = UUID()
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(backend: backend, sessionId: session),
            id: UUID()
        )

        for _ in 0..<200 {
            if subagent.status == .running { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(subagent.status, .running)

        SubagentManager.shared.cancelSession(sessionId: session, reason: "the coordinator turn was stopped.")
        let summary = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .cancelled)
        XCTAssertFalse(summary.isEmpty)
    }

    func testASpawnWithAForbiddenToolNeverReachesABackend() async {
        do {
            _ = try await AgentSpawnResolver.resolve(
                AgentSpawnRequest(
                    role: "Custom",
                    taskDescription: "delegate again",
                    explicitToolNames: ["spawn_subagent"],
                    workingDirectory: workingDirectory
                ),
                runId: UUID(),
                sessionId: UUID()
            )
            XCTFail("a forbidden tool must be refused before a backend is resolved")
        } catch {
            XCTAssertEqual(error as? AgentSpawnError, .forbiddenTool("spawn_subagent"))
        }
    }
}
