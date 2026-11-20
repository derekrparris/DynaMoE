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
        // Turbo Mode is deliberately NOT touched here. It is the operator's
        // process-wide preference and this target is hosted inside the app
        // (same bundle id, same defaults domain), so setting it would turn a
        // running app's Turbo off mid-turn — and leave it off if the run were
        // interrupted. The cards these cases park on come from requests that
        // are never Turbo-bypassable (a containment escape), so the switch
        // cannot change their outcome either way.
        workingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubagentRunTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        // AWAITED, not fire-and-forget. `cancelAll` and `clearCompleted` hand
        // their work to tasks and to the store's IO queue; returning before
        // they land let a later case see this one's runs still finishing (and
        // wrote their records while the next fixture was already reading the
        // process-wide registry).
        SubagentManager.shared.cancelAll()
        await AgentScheduler.shared.cancelAll(reason: "test teardown")
        SubagentManager.shared.clearCompleted()
        AgentRunStore.shared.flush()
        if let workingDirectory {
            try? FileManager.default.removeItem(at: workingDirectory)
        }
        try await super.tearDown()
    }

    private func plan(
        backend: ScriptedBackend,
        sessionId: UUID,
        toolNames: [String] = ["file_read"],
        policy: AgentApprovalPolicy = .auto
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
            approvalPolicy: policy,
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

    /// The status tools read the conversation that is asking. The registry is
    /// process-wide, and a second chat could previously read (and mark relayed)
    /// the first chat's report — silently suppressing its own delivery.
    func testTheStatusToolsOnlySeeTheCallersConversation() async throws {
        let sessionA = UUID()
        let sessionB = UUID()
        let runA = SubagentManager.shared.spawnModelBacked(
            plan(
                backend: ScriptedBackend(turns: [.init(text: "finding from A")]),
                sessionId: sessionA
            ),
            id: UUID()
        )
        _ = await runA.waitForCompletion()
        let runB = SubagentManager.shared.spawnModelBacked(
            plan(
                backend: ScriptedBackend(turns: [.init(text: "finding from B")]),
                sessionId: sessionB
            ),
            id: UUID()
        )
        _ = await runB.waitForCompletion()

        let statusTool = GetSubagentStatusTool()
        let foreign = try await AgentHarness.$executingSessionId.withValue(sessionA) {
            try await statusTool.execute(
                arguments: ["subagent_id": runB.id.uuidString],
                workingDirectory: nil,
                maxOutputLength: 4_000
            )
        }
        XCTAssertTrue(foreign.resultJSON.contains("not found"))
        XCTAssertFalse(
            runB.isSummaryRelayed,
            "a foreign read must not consume the owning conversation's relay"
        )
        XCTAssertFalse(foreign.resultJSON.contains("finding from B"))

        let own = try await AgentHarness.$executingSessionId.withValue(sessionA) {
            try await statusTool.execute(
                arguments: ["subagent_id": runA.id.uuidString],
                workingDirectory: nil,
                maxOutputLength: 4_000
            )
        }
        XCTAssertFalse(own.resultJSON.contains("not found"))
        XCTAssertTrue(own.resultJSON.contains("finding from A"))

        let listTool = ListSubagentsTool()
        let listed = try await AgentHarness.$executingSessionId.withValue(sessionA) {
            try await listTool.execute(
                arguments: [:],
                workingDirectory: nil,
                maxOutputLength: 4_000
            )
        }
        XCTAssertTrue(listed.resultJSON.contains(runA.id.uuidString))
        XCTAssertFalse(
            listed.resultJSON.contains(runB.id.uuidString),
            "another conversation's run must not be listed"
        )

        let sendTool = SendSubagentMessageTool()
        let steering = try await AgentHarness.$executingSessionId.withValue(sessionA) {
            try await sendTool.execute(
                arguments: ["subagent_id": runB.id.uuidString, "message": "keep going"],
                workingDirectory: nil,
                maxOutputLength: 4_000
            )
        }
        XCTAssertTrue(
            steering.resultJSON.contains("not found"),
            "steering another conversation's run is a cross-session write"
        )
    }

    /// A run holds its class's scheduler ticket for its whole life, so a
    /// finished run that forgot to release would shrink the provider's (or the
    /// engine's) capacity by one until relaunch. The release is handed to a
    /// task from the run's `defer`, so poll rather than assume it lands inline.
    func testAFinishedRunReleasesItsSchedulerTicket() async throws {
        let subagent = SubagentManager.shared.spawnModelBacked(
            plan(
                backend: ScriptedBackend(turns: [.init(text: "done")]),
                sessionId: UUID()
            ),
            id: UUID()
        )
        _ = await subagent.waitForCompletion()

        var tickets = -1
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            tickets = await AgentScheduler.shared.snapshot().totalTickets
            if tickets == 0 { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(tickets, 0, "the finished run must give its scheduler ticket back")
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

    // MARK: - Lifecycle with a parked approval

    /// A run parked on an approval card is the one state in which a quit or a
    /// conversation delete has a live task, a durable record AND an
    /// outstanding continuation at the same time. Both transitions used to be
    /// exercised only against running or finished runs.
    private func parkedWriteRun(sessionId: UUID) -> SubagentInstance {
        // A CONTAINMENT ESCAPE, not an ordinary in-workspace write. Turbo Mode
        // is the operator's process-wide preference and this target is hosted
        // inside the app, so an ordinary write's card is bypassable when they
        // left Turbo on — the fixture would then never park, and the cases
        // below would fail for a setting they must not depend on. An escape is
        // never Turbo-bypassable, so the card is guaranteed either way.
        let backend = ScriptedBackend(turns: [
            .init(
                text: "writing outside the workspace",
                calls: [AgentToolCall(name: "file_write", arguments: [
                    "path": "../parked.txt",
                    "content": "parked\n"
                ])]
            )
        ])
        return SubagentManager.shared.spawnModelBacked(
            plan(
                backend: backend,
                sessionId: sessionId,
                toolNames: ["file_read", "file_write"],
                policy: .everyStateChange
            ),
            id: UUID()
        )
    }

    private func parkedCard(for runId: UUID) -> PendingSubagentApproval? {
        SubagentApprovalCenter.shared.pending.first { $0.request.runId == runId }
    }

    private func waitUntil(
        timeout: TimeInterval,
        _ condition: () -> Bool
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 30_000_000)
        }
        return condition()
    }

    func testDeletingAConversationStopsAParkedRunAndRemovesItsRecord() async throws {
        let session = UUID()
        let subagent = parkedWriteRun(sessionId: session)
        defer {
            SubagentManager.shared.clearCompleted()
            AgentRunStore.shared.flush()
        }

        let parked = await waitUntil(timeout: 10) { self.parkedCard(for: subagent.id) != nil }
        XCTAssertTrue(parked, "the run must park on the write's card")
        XCTAssertEqual(subagent.status, .waitingForApproval)

        // What ContentView does for a conversation delete: cancel its runs,
        // then delete their records.
        SubagentManager.shared.cancelSession(sessionId: session)
        AgentRunStore.shared.deleteSession(session)
        // Release the parked continuation so the run's task can unwind; a
        // deleted conversation cannot answer its card.
        if let card = parkedCard(for: subagent.id) {
            SubagentApprovalCenter.shared.reject(card.id)
        }
        AgentRunStore.shared.flush()

        let stopped = await waitUntil(timeout: 10) { subagent.status.isTerminal }
        XCTAssertTrue(stopped)
        XCTAssertEqual(subagent.status, .cancelled)
        let cleared = await waitUntil(timeout: 5) { self.parkedCard(for: subagent.id) == nil }
        XCTAssertTrue(cleared)
        XCTAssertNil(
            AgentRunStore.shared.load(runId: subagent.id, sessionId: session),
            "a deleted conversation must not leave its runs on disk"
        )
        XCTAssertNil(AgentRunStore.shared.loadAny(runId: subagent.id))
    }

    func testAQuitWithAParkedRunLeavesATerminalRecord() async throws {
        let session = UUID()
        let subagent = parkedWriteRun(sessionId: session)
        defer {
            SubagentManager.shared.clearCompleted()
            AgentRunStore.shared.flush()
        }

        let parked = await waitUntil(timeout: 10) { self.parkedCard(for: subagent.id) != nil }
        XCTAssertTrue(parked)

        // `willTerminate` cancels every run and then flushes the store.
        SubagentManager.shared.cancelAll()
        if let card = parkedCard(for: subagent.id) {
            SubagentApprovalCenter.shared.reject(card.id)
        }
        AgentRunStore.shared.flush()

        let stopped = await waitUntil(timeout: 10) { subagent.status.isTerminal }
        XCTAssertTrue(stopped)
        XCTAssertEqual(subagent.status, .cancelled)
        let record = AgentRunStore.shared.load(runId: subagent.id, sessionId: session)
        XCTAssertEqual(record?.status, .cancelled, "the quit path must persist a terminal record")
        XCTAssertTrue(record?.status.isTerminal == true)
        AgentRunStore.shared.deleteImmediately(runId: subagent.id, sessionId: session)
    }
}
