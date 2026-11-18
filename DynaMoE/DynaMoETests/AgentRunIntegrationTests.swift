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
        await MainActor.run {
            SubagentManager.shared.cancelAll()
            // `cancelAll` deliberately skips runs that already finished, so
            // every instance a test completed stayed in the process-wide
            // registry for the rest of the suite — a later case's
            // `allSubagents` saw earlier cases' runs. The records go too,
            // which is what a test wants: nothing here is operator data (the
            // store redirects to a /tmp root under XCTest).
            SubagentManager.shared.clearCompleted()
        }
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

    /// A completion that stopped at the token limit is a fragment, not an
    /// answer. It used to be reported `.completed` because the loop never read
    /// the backend's finish reason.
    func testATokenCappedTurnIsAFailureRatherThanAnAnswer() async throws {
        let backend = ScriptedBackend(turns: [
            .init(text: "The scheduler lives in", finishReason: "lengthCapped")
        ])
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(backend: backend, sessionId: UUID()),
            id: UUID()
        )
        _ = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .failed)
        XCTAssertTrue(
            subagent.lastError?.contains("token limit") == true,
            subagent.lastError ?? "no error recorded"
        )
    }

    func testADegenerateCycleTurnIsAFailureRatherThanAnAnswer() async throws {
        let backend = ScriptedBackend(turns: [
            .init(text: "again again again", finishReason: "degenerateCycle")
        ])
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(backend: backend, sessionId: UUID()),
            id: UUID()
        )
        _ = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .failed)
        XCTAssertTrue(
            subagent.lastError?.contains("repeating itself") == true,
            subagent.lastError ?? "no error recorded"
        )
    }

    /// The budget checks used to run only BEFORE a step, so the final turn
    /// could overspend a cost ceiling and still be reported as a completion.
    func testAFinalTurnThatOverspendsTheCostCeilingFailsTheRun() async throws {
        var usage = AgentUsage()
        usage.promptTokens = 1_000
        usage.completionTokens = 1_000
        usage.costUSD = 5
        usage.backendRequests = 1
        let backend = ScriptedBackend(turns: [.init(text: "done", usage: usage)])
        var runPlan = plan(backend: backend, sessionId: UUID())
        runPlan.budget = AgentBudget(
            maxSteps: 3,
            maxCompletionTokens: 200,
            maxWallClockSeconds: 20,
            maxCostUSD: 1
        )
        let subagent = SubagentManager.shared.spawnModelBacked(runPlan, id: UUID())
        _ = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .failed)
        XCTAssertTrue(
            subagent.lastError?.contains("cost budget") == true,
            subagent.lastError ?? "no error recorded"
        )
        XCTAssertGreaterThan(subagent.usage.costUSD, 1)
    }

    func testCancellingAFinishedRunCannotRewriteItsRecord() async throws {
        let backend = ScriptedBackend(turns: [.init(text: "the answer")])
        let session = UUID()
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(backend: backend, sessionId: session),
            id: UUID()
        )
        let summary = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .completed)
        let completedAt = subagent.completedAt
        AgentRunStore.shared.flush()
        let stored = try XCTUnwrap(
            AgentRunStore.shared.load(runId: subagent.id, sessionId: session),
            "the finished run must have a durable record"
        )
        XCTAssertEqual(stored.status, .completed)

        subagent.cancel(reason: "quitting")
        AgentRunStore.shared.flush()

        XCTAssertEqual(subagent.status, .completed, "a finished run must keep its terminal status")
        XCTAssertEqual(subagent.finalSummary, summary)
        XCTAssertEqual(subagent.completedAt, completedAt)
        let afterCancel = try XCTUnwrap(
            AgentRunStore.shared.load(runId: subagent.id, sessionId: session),
            "cancelling on the way out must not delete the record"
        )
        XCTAssertEqual(afterCancel.status, .completed)
        XCTAssertEqual(afterCancel.completedAt, stored.completedAt)
        XCTAssertEqual(afterCancel.report?.summary, stored.report?.summary)
    }

    /// "Clear Finished" used to match only completed and cancelled runs, so a
    /// failed run (and its record) could never be cleared from the inspector.
    func testClearFinishedRemovesFailedRunsAndTheirRecords() async throws {
        let backend = ScriptedBackend(turns: [
            .init(text: "truncated", finishReason: "lengthCapped")
        ])
        let session = UUID()
        let runId = UUID()
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(backend: backend, sessionId: session),
            id: runId
        )
        _ = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .failed)
        AgentRunStore.shared.flush()

        SubagentManager.shared.clearCompleted(sessionId: session)

        XCTAssertNil(SubagentManager.shared.getSubagent(byId: runId))
        XCTAssertNil(AgentRunStore.shared.load(runId: runId, sessionId: session))
    }

    func testFinishReasonTruncationIsRecognised() {
        XCTAssertTrue(HeadlessAgentLoop.finishReasonIsTruncated("length"))
        XCTAssertTrue(HeadlessAgentLoop.finishReasonIsTruncated("lengthCapped"))
        XCTAssertTrue(HeadlessAgentLoop.finishReasonIsTruncated("max_tokens"))
        XCTAssertFalse(HeadlessAgentLoop.finishReasonIsTruncated("stop"))
        XCTAssertFalse(HeadlessAgentLoop.finishReasonIsTruncated("toolFreeze"))
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

    /// A cloud provider gets native function declarations, so its worker prompt
    /// must tell it to issue those calls. The local spawn prompt teaches an XML
    /// dialect the local parser understands; sending that to a cloud model made
    /// it narrate the XML in prose instead of calling anything.
    func testACloudRunIsToldToUseNativeFunctionCalling() async throws {
        let backend = ScriptedBackend(turns: [.init(text: "done")])
        let session = UUID()
        var cloudPlan = plan(backend: backend, sessionId: session)
        cloudPlan.backendDescriptor = AgentBackendDescriptor(
            kind: .cloud,
            providerId: "synthetic",
            modelId: "syn:large:text",
            displayName: "Syn Large"
        )
        cloudPlan.schedulingClass = .cloud(providerId: "synthetic")
        let subagent = SubagentManager.shared.spawnModelBacked(cloudPlan, id: UUID())
        _ = await subagent.waitForCompletion()

        let prompt = try XCTUnwrap(backend.requests.first?.systemPrompt)
        XCTAssertTrue(prompt.contains("native function-calling"), prompt)
        XCTAssertFalse(prompt.contains("<tool_call>"), "a cloud model must not be taught the local XML dialect")
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
