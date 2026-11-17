//
//  AgentScheduler.swift
//  DynaMoE
//
//  Concurrency, priority and cancellation for model-backed subagent runs.
//  See SUBAGENTS.md §8.
//

import Foundation

/// What a run needs in order to take a step.
///
/// The local engine is one slot wide, so a local run is its own class; cloud
/// providers are independent quotas and are grouped by provider, with a global
/// cap on top.
nonisolated enum AgentSchedulingClass: Hashable, Sendable {
    case local
    case appleFoundationModel
    case cloud(providerId: String)

    var displayName: String {
        switch self {
        case .local: return "Local"
        case .appleFoundationModel: return "Apple Foundation Model"
        case .cloud(let providerId): return "Cloud (\(providerId))"
        }
    }

    var isCloud: Bool {
        if case .cloud = self { return true }
        return false
    }
}

/// FIFO within a class: interactive coordinator work outranks a run the
/// operator asked for, which outranks speculative work.
nonisolated enum AgentSchedulingPriority: Int, Sendable, Comparable, CaseIterable {
    case interactive = 0
    case userRequested = 1
    case speculative = 2

    static func < (lhs: AgentSchedulingPriority, rhs: AgentSchedulingPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

nonisolated struct AgentSchedulerLimits: Equatable, Sendable {
    var local = 1
    var cloudPerProvider = 3
    var cloudGlobal = 6
    var appleFoundationModel = 1
    var queueTimeoutSeconds: TimeInterval = 120

    static let `default` = AgentSchedulerLimits()

    /// Debug/settings seams. Each falls back to the default when absent or not
    /// a positive number, because a capacity of zero would deadlock the queue.
    static func configured(defaults: UserDefaults = .standard) -> AgentSchedulerLimits {
        var limits = AgentSchedulerLimits()
        if let value = positiveInteger(defaults, "dynamoe_subagent_local_capacity") { limits.local = value }
        if let value = positiveInteger(defaults, "dynamoe_subagent_cloud_per_provider") { limits.cloudPerProvider = value }
        if let value = positiveInteger(defaults, "dynamoe_subagent_cloud_global") { limits.cloudGlobal = value }
        if let value = positiveInteger(defaults, "dynamoe_subagent_afm_capacity") { limits.appleFoundationModel = value }
        if let number = defaults.object(forKey: "dynamoe_subagent_queue_timeout") as? NSNumber,
           number.doubleValue > 0 {
            limits.queueTimeoutSeconds = number.doubleValue
        }
        return limits
    }

    private static func positiveInteger(_ defaults: UserDefaults, _ key: String) -> Int? {
        guard let number = defaults.object(forKey: key) as? NSNumber else { return nil }
        let value = number.intValue
        return value > 0 ? value : nil
    }
}

nonisolated enum AgentSchedulerError: LocalizedError, Equatable {
    case queueTimeout(AgentSchedulingClass, TimeInterval)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .queueTimeout(let schedulingClass, let seconds):
            return "The subagent run waited \(Int(seconds)) s for \(schedulingClass.displayName) capacity and was refused."
        case .cancelled:
            return "The subagent run was cancelled while it waited for capacity."
        }
    }
}

/// An observation of the scheduler, for tests and diagnostics.
nonisolated struct AgentSchedulerSnapshot: Equatable, Sendable {
    var tickets: [AgentSchedulingClass: Int] = [:]
    var waiting: Int = 0
    var registeredRuns: Int = 0

    var totalTickets: Int { tickets.values.reduce(0, +) }
    var cloudTickets: Int {
        tickets.filter { $0.key.isCloud }.values.reduce(0, +)
    }
}

/// Admits runs to a class when it has capacity, in priority order.
///
/// It is an actor because the queue is state: a ticket is granted only when the
/// highest-priority waiter fits, so a burst of speculative spawns cannot starve
/// the run the operator actually asked for. It does NOT own the engine lease —
/// a local run still claims the single-flight slot per step, which is the
/// mechanism the server queue and the chat send path share (`EngineTurnSlot.swift`).
/// What this enforces is that only one local run is ever asking for that slot.
actor AgentScheduler {
    static let shared = AgentScheduler()

    private struct Waiter {
        var ticket: UUID
        var schedulingClass: AgentSchedulingClass
        var priority: AgentSchedulingPriority
        var runId: UUID
        var enqueuedAt: Date
        var sequence: UInt64
    }

    private let limits: AgentSchedulerLimits
    private let pollIntervalNanoseconds: UInt64

    private var tickets: [AgentSchedulingClass: Set<UUID>] = [:]
    private var ticketClasses: [UUID: AgentSchedulingClass] = [:]
    private var waiters: [UUID: Waiter] = [:]
    private var nextSequence: UInt64 = 0
    private var cancelledRunIds: Set<UUID> = []
    private var registry: [UUID: Registration] = [:]

    private struct Registration {
        var parentRunId: UUID?
        var parentSessionId: UUID?
        var schedulingClass: AgentSchedulingClass
        var priority: AgentSchedulingPriority
        var cancel: @Sendable (String) -> Void
    }

    init(
        limits: AgentSchedulerLimits = .default,
        pollIntervalNanoseconds: UInt64 = 20_000_000
    ) {
        self.limits = limits
        self.pollIntervalNanoseconds = pollIntervalNanoseconds
    }

    var currentLimits: AgentSchedulerLimits { limits }

    // MARK: - Admission

    /// Waits until `schedulingClass` has capacity and grants a ticket.
    ///
    /// The wait is priority ordered and bounded: a run that cannot start within
    /// the queue timeout is refused by name rather than left pending forever.
    func acquire(
        _ schedulingClass: AgentSchedulingClass,
        priority: AgentSchedulingPriority,
        runId: UUID,
        timeout: TimeInterval? = nil
    ) async throws -> UUID {
        let ticket = UUID()
        let budget = timeout ?? limits.queueTimeoutSeconds
        let deadline = Date().addingTimeInterval(budget)
        nextSequence &+= 1
        waiters[ticket] = Waiter(
            ticket: ticket,
            schedulingClass: schedulingClass,
            priority: priority,
            runId: runId,
            enqueuedAt: Date(),
            sequence: nextSequence
        )
        defer { waiters.removeValue(forKey: ticket) }

        while true {
            if Task.isCancelled || cancelledRunIds.contains(runId) {
                cancelledRunIds.remove(runId)
                throw AgentSchedulerError.cancelled
            }
            if canGrant(schedulingClass), isHeadWaiter(ticket) {
                tickets[schedulingClass, default: []].insert(ticket)
                ticketClasses[ticket] = schedulingClass
                return ticket
            }
            if Date() >= deadline {
                throw AgentSchedulerError.queueTimeout(schedulingClass, budget)
            }
            try? await Task.sleep(nanoseconds: pollIntervalNanoseconds)
        }
    }

    /// Gives a granted ticket back. Idempotent.
    func release(_ ticket: UUID) {
        guard let schedulingClass = ticketClasses.removeValue(forKey: ticket) else { return }
        tickets[schedulingClass]?.remove(ticket)
    }

    // MARK: - Cancellation tree

    /// Records how to stop a run, and who it belongs to.
    func register(
        runId: UUID,
        parentRunId: UUID?,
        parentSessionId: UUID?,
        schedulingClass: AgentSchedulingClass,
        priority: AgentSchedulingPriority,
        cancel: @escaping @Sendable (String) -> Void
    ) {
        cancelledRunIds.remove(runId)
        registry[runId] = Registration(
            parentRunId: parentRunId,
            parentSessionId: parentSessionId,
            schedulingClass: schedulingClass,
            priority: priority,
            cancel: cancel
        )
    }

    /// Stops one run: a queued run fails its wait, a live one is cancelled, and
    /// every descendant is cancelled with it.
    func cancel(runId: UUID, reason: String = "The run was cancelled.") {
        cancelSubtree(runId: runId, reason: reason)
    }

    /// Stops every run whose coordinator turn belongs to `sessionId`.
    ///
    /// Delegation is part of the coordinator's turn: a Stop must not leave a
    /// child working on work the operator has abandoned.
    func cancelRuns(parentSessionId: UUID, reason: String = "The coordinator turn was stopped.") {
        let roots = registry.filter { $0.value.parentSessionId == parentSessionId }.map(\.key)
        for runId in roots {
            cancelSubtree(runId: runId, reason: reason)
        }
    }

    /// Interactive work outranks background work: a local run is cancelled at
    /// its next boundary rather than allowed to hold the engine indefinitely.
    /// Returns how many local runs were asked to stop.
    @discardableResult
    func preemptLocalRuns(reason: String = "An interactive turn needed the engine.") -> Int {
        let localRuns = registry.filter { runId, registration in
            registration.priority != .interactive && isLocalRun(runId)
        }
        for (runId, _) in localRuns {
            cancelSubtree(runId: runId, reason: reason)
        }
        return localRuns.count
    }

    /// Stops every registered run. Application teardown and the Task Manager's
    /// "cancel all" both land here.
    func cancelAll(reason: String) {
        for runId in Array(registry.keys) {
            cancelSubtree(runId: runId, reason: reason)
        }
        for waiter in waiters.values {
            cancelledRunIds.insert(waiter.runId)
        }
    }

    func finish(runId: UUID) {
        registry.removeValue(forKey: runId)
        cancelledRunIds.remove(runId)
    }

    // MARK: - Observation

    func snapshot() -> AgentSchedulerSnapshot {
        AgentSchedulerSnapshot(
            tickets: tickets.mapValues(\.count),
            waiting: waiters.count,
            registeredRuns: registry.count
        )
    }

    func isWaiting(runId: UUID) -> Bool {
        waiters.values.contains { $0.runId == runId }
    }

    /// True once a cancellation has been requested for `runId`.
    func isCancelled(runId: UUID) -> Bool {
        cancelledRunIds.contains(runId)
    }

    // MARK: - Internals

    private func isLocalRun(_ runId: UUID) -> Bool {
        if let registration = registry[runId] {
            return registration.schedulingClass == .local
        }
        return waiters.values.contains { $0.runId == runId && $0.schedulingClass == .local }
    }

    private func cancelSubtree(runId: UUID, reason: String) {
        let children = registry.filter { $0.value.parentRunId == runId }.map(\.key)
        if registry[runId] != nil || waiters.values.contains(where: { $0.runId == runId }) {
            cancelledRunIds.insert(runId)
        }
        registry[runId]?.cancel(reason)
        for child in children {
            cancelSubtree(runId: child, reason: reason)
        }
    }

    private func canGrant(_ schedulingClass: AgentSchedulingClass) -> Bool {
        switch schedulingClass {
        case .local:
            return (tickets[.local]?.count ?? 0) < limits.local
        case .appleFoundationModel:
            return (tickets[.appleFoundationModel]?.count ?? 0) < limits.appleFoundationModel
        case .cloud(let providerId):
            let perProvider = (tickets[.cloud(providerId: providerId)]?.count ?? 0)
            let global = tickets.filter { $0.key.isCloud }.values.reduce(0) { $0 + $1.count }
            return perProvider < limits.cloudPerProvider && global < limits.cloudGlobal
        }
    }

    /// Lowest priority number first, then earliest enqueue — among waiters of
    /// the SAME class. Ordering across classes would let one class's head starve
    /// another class that has capacity, which is exactly what independent cloud
    /// quotas are for.
    private func isHeadWaiter(_ ticket: UUID) -> Bool {
        guard let mine = waiters[ticket] else { return false }
        for other in waiters.values
        where other.ticket != mine.ticket && other.schedulingClass == mine.schedulingClass {
            if other.priority < mine.priority { return false }
            if other.priority == mine.priority, other.sequence < mine.sequence { return false }
        }
        return true
    }
}
