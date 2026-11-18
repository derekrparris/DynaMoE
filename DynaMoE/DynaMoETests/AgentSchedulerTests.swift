//
//  AgentSchedulerTests.swift
//  DynaMoETests
//
//  Capacity, priority, cancellation and queue timeouts for the subagent
//  scheduler (SUBAGENTS.md §8). No model is needed: the scheduler is pure state.
//

import XCTest
@testable import DynaMoE

@MainActor
final class AgentSchedulerTests: XCTestCase {
    private func makeScheduler(_ limits: AgentSchedulerLimits) -> AgentScheduler {
        AgentScheduler(limits: limits, pollIntervalNanoseconds: 2_000_000)
    }

    private func waitForWaiters(_ scheduler: AgentScheduler, _ count: Int) async throws {
        for _ in 0..<200 {
            if await scheduler.snapshot().waiting >= count { return }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTFail("The scheduler never reached \(count) waiting run(s).")
    }

    func testLocalCapacityIsOne() async throws {
        let scheduler = makeScheduler(.default)
        let first = try await scheduler.acquire(.local, priority: .userRequested, runId: UUID())

        let secondRun = UUID()
        let second = Task { try await scheduler.acquire(.local, priority: .userRequested, runId: secondRun, timeout: 5) }
        try await waitForWaiters(scheduler, 1)
        var snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.totalTickets, 1)

        await scheduler.release(first)
        let secondTicket = try await second.value
        snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.totalTickets, 1)
        await scheduler.release(secondTicket)
        snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.totalTickets, 0)
    }

    func testPriorityOrdersTheQueue() async throws {
        var limits = AgentSchedulerLimits.default
        limits.local = 1
        let scheduler = makeScheduler(limits)
        let blocking = try await scheduler.acquire(.local, priority: .interactive, runId: UUID())

        let order = GrantOrder()
        let speculative = UUID()
        let user = UUID()
        let speculativeTask = Task {
            let ticket = try await scheduler.acquire(.local, priority: .speculative, runId: speculative, timeout: 5)
            order.record(speculative)
            return ticket
        }
        try await waitForWaiters(scheduler, 1)
        let userTask = Task {
            let ticket = try await scheduler.acquire(.local, priority: .userRequested, runId: user, timeout: 5)
            order.record(user)
            return ticket
        }
        try await waitForWaiters(scheduler, 2)

        await scheduler.release(blocking)
        let userTicket = try await userTask.value
        XCTAssertEqual(order.values.first, user, "the user-requested run must be granted first")
        await scheduler.release(userTicket)

        let speculativeTicket = try await speculativeTask.value
        XCTAssertEqual(order.values, [user, speculative])
        await scheduler.release(speculativeTicket)
    }

    func testCloudCapacityIsPerProvider() async throws {
        var limits = AgentSchedulerLimits.default
        limits.cloudPerProvider = 1
        limits.cloudGlobal = 6
        let scheduler = makeScheduler(limits)

        let providerA = AgentSchedulingClass.cloud(providerId: "a")
        let firstA = try await scheduler.acquire(providerA, priority: .userRequested, runId: UUID(), timeout: 1)
        // Another provider has its own quota.
        let firstB = try await scheduler.acquire(
            .cloud(providerId: "b"), priority: .userRequested, runId: UUID(), timeout: 1
        )

        let secondA = Task {
            try await scheduler.acquire(providerA, priority: .userRequested, runId: UUID(), timeout: 5)
        }
        try await waitForWaiters(scheduler, 1)

        await scheduler.release(firstA)
        let secondATicket = try await secondA.value
        await scheduler.release(secondATicket)
        await scheduler.release(firstB)
    }

    func testCloudCapacityIsGlobal() async throws {
        var limits = AgentSchedulerLimits.default
        limits.cloudPerProvider = 2
        limits.cloudGlobal = 2
        let scheduler = makeScheduler(limits)

        let providerA = AgentSchedulingClass.cloud(providerId: "a")
        let first = try await scheduler.acquire(providerA, priority: .userRequested, runId: UUID(), timeout: 1)
        let second = try await scheduler.acquire(providerA, priority: .userRequested, runId: UUID(), timeout: 1)

        // A different provider is refused by the GLOBAL cap even though its own
        // quota is untouched.
        let otherProvider = Task {
            try await scheduler.acquire(
                .cloud(providerId: "c"), priority: .userRequested, runId: UUID(), timeout: 5
            )
        }
        try await waitForWaiters(scheduler, 1)

        await scheduler.release(first)
        let otherTicket = try await otherProvider.value
        await scheduler.release(second)
        await scheduler.release(otherTicket)
    }

    func testCancellingARunStopsItsDescendants() async {
        let scheduler = makeScheduler(.default)
        let recorder = CancellationRecorder()
        let parent = UUID()
        let child = UUID()
        let grandchild = UUID()
        await scheduler.register(
            runId: parent, parentRunId: nil, parentSessionId: UUID(),
            schedulingClass: .local, priority: .userRequested,
            cancel: { recorder.record("parent", $0) }
        )
        await scheduler.register(
            runId: child, parentRunId: parent, parentSessionId: UUID(),
            schedulingClass: .local, priority: .userRequested,
            cancel: { recorder.record("child", $0) }
        )
        await scheduler.register(
            runId: grandchild, parentRunId: child, parentSessionId: UUID(),
            schedulingClass: .cloud(providerId: "p"), priority: .userRequested,
            cancel: { recorder.record("grandchild", $0) }
        )

        await scheduler.cancel(runId: parent, reason: "the coordinator stopped")
        XCTAssertEqual(Set(recorder.names), ["parent", "child", "grandchild"])
        XCTAssertEqual(Set(recorder.reasons), ["the coordinator stopped"])
    }

    func testAQueuedRunTimesOut() async throws {
        var limits = AgentSchedulerLimits.default
        limits.local = 1
        let scheduler = makeScheduler(limits)
        let blocking = try await scheduler.acquire(.local, priority: .interactive, runId: UUID())

        do {
            _ = try await scheduler.acquire(.local, priority: .userRequested, runId: UUID(), timeout: 0.05)
            XCTFail("A run behind a full local class must not be admitted.")
        } catch AgentSchedulerError.queueTimeout(let schedulingClass, let seconds) {
            XCTAssertEqual(schedulingClass, .local)
            XCTAssertEqual(seconds, 0.05, accuracy: 0.0001)
        }
        await scheduler.release(blocking)
    }

    func testAQueuedRunCanBeCancelled() async throws {
        var limits = AgentSchedulerLimits.default
        limits.local = 1
        let scheduler = makeScheduler(limits)
        let blocking = try await scheduler.acquire(.local, priority: .interactive, runId: UUID())

        let queuedRun = UUID()
        let queued = Task {
            try await scheduler.acquire(.local, priority: .userRequested, runId: queuedRun, timeout: 5)
        }
        try await waitForWaiters(scheduler, 1)
        await scheduler.cancel(runId: queuedRun, reason: "the operator stopped it")

        do {
            _ = try await queued.value
            XCTFail("A cancelled waiter must not be admitted.")
        } catch AgentSchedulerError.cancelled {
            // expected
        }
        await scheduler.release(blocking)
    }

    func testPreemptionStopsOnlyBackgroundLocalRuns() async {
        let scheduler = makeScheduler(.default)
        let recorder = CancellationRecorder()
        let localRun = UUID()
        let cloudRun = UUID()
        await scheduler.register(
            runId: localRun, parentRunId: nil, parentSessionId: nil,
            schedulingClass: .local, priority: .userRequested,
            cancel: { recorder.record("local", $0) }
        )
        await scheduler.register(
            runId: cloudRun, parentRunId: nil, parentSessionId: nil,
            schedulingClass: .cloud(providerId: "p"), priority: .userRequested,
            cancel: { recorder.record("cloud", $0) }
        )

        let preempted = await scheduler.preemptLocalRuns(reason: "an interactive turn went first")
        XCTAssertEqual(preempted, 1)
        XCTAssertEqual(recorder.names, ["local"])
        XCTAssertEqual(recorder.reasons, ["an interactive turn went first"])
    }

    func testCancellingASessionStopsItsRunsOnly() async {
        let scheduler = makeScheduler(.default)
        let recorder = CancellationRecorder()
        let session = UUID()
        await scheduler.register(
            runId: UUID(), parentRunId: nil, parentSessionId: session,
            schedulingClass: .local, priority: .userRequested,
            cancel: { recorder.record("ours", $0) }
        )
        await scheduler.register(
            runId: UUID(), parentRunId: nil, parentSessionId: UUID(),
            schedulingClass: .local, priority: .userRequested,
            cancel: { recorder.record("theirs", $0) }
        )

        await scheduler.cancelRuns(parentSessionId: session, reason: "turn stopped")
        XCTAssertEqual(recorder.names, ["ours"])
    }

    /// The instance's execution task starts asynchronously, so a Stop can reach
    /// the scheduler before its registration exists. That cancellation used to
    /// be dropped, and the run then started normally.
    func testACancellationBeforeRegistrationIsAppliedWhenTheRunRegisters() async {
        let scheduler = makeScheduler(.default)
        let recorder = CancellationRecorder()
        let runId = UUID()

        await scheduler.cancel(runId: runId, reason: "stopped before start")
        let cancelledEarly = await scheduler.isCancelled(runId: runId)
        XCTAssertTrue(cancelledEarly)

        await scheduler.register(
            runId: runId, parentRunId: nil, parentSessionId: nil,
            schedulingClass: .cloud(providerId: "p"), priority: .userRequested,
            cancel: { recorder.record("run", $0) }
        )
        XCTAssertEqual(recorder.reasons, ["stopped before start"])

        do {
            _ = try await scheduler.acquire(
                .cloud(providerId: "p"),
                priority: .userRequested,
                runId: runId,
                timeout: 1
            )
            XCTFail("a run cancelled before registration must not be admitted")
        } catch AgentSchedulerError.cancelled {
            // expected
        } catch {
            XCTFail("expected a cancellation, got \(error)")
        }
    }
}

// MARK: - Test recorders (lock boxes, not actors: the closures are synchronous)

private nonisolated final class GrantOrder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [UUID] = []

    func record(_ id: UUID) {
        lock.lock()
        stored.append(id)
        lock.unlock()
    }

    var values: [UUID] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

private nonisolated final class CancellationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [(name: String, reason: String)] = []

    func record(_ name: String, _ reason: String) {
        lock.lock()
        entries.append((name, reason))
        lock.unlock()
    }

    var names: [String] {
        lock.lock()
        defer { lock.unlock() }
        return entries.map(\.name)
    }

    var reasons: [String] {
        lock.lock()
        defer { lock.unlock() }
        return entries.map(\.reason)
    }
}
