//
//  EngineTurnSlot.swift
//  DynaMoE
//
//  Phase: the engine's single-flight generation slot, as something a caller can
//  HOLD rather than merely look at.
//

import Foundation

/// Why a turn wants the engine.
///
/// There is one slot, so a background turn (a subagent's step) must be the one
/// that gives way: an operator who sends a message outranks work they are
/// waiting on. Giving way is a CANCEL, not a pause. The background turn stops at
/// its next layer boundary, the engine drains, and the run reports what it lost;
/// nothing resumes it, because there is no paused status, no serialized loop
/// state and no step index to restart from. Interactive turns are never
/// cancelled to make room.
nonisolated enum TurnPriority: Sendable, Equatable {
    case interactive
    case background
}

/// A claim on the engine's single-flight generation slot.
///
/// Asking whether the slot is free and then taking it are two steps, and a turn
/// can start between them. That gap used to cost a queued request everything it
/// had waited for: the server queue probed the slot, spent real time preparing the
/// turn (tokenizing, rendering the prompt, looking the prefix up), and launched
/// only to find a chat turn had started in the meantime, which the engine refused
/// and the client saw as a 503 (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #84").
///
/// A claim collapses the pair. It is taken in the same step that decides the slot
/// is free and is passed to `CompletionEngine.launchTurn`, which honours it —
/// and, while one is outstanding, refuses a launch that holds none. So the queue
/// either owns the slot for the whole of its preparation or is told to keep
/// waiting, and the request never has to be thrown away.
///
/// `release` is idempotent, so the queue can release in a `defer` while the launch
/// path releases on its own way out without either knowing about the other.
///
/// `nonisolated` for the same reason as the queue's own boxes: the target builds
/// with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, and an unannotated class here
/// would become a main-actor type crossed on the generation path.
nonisolated final class TurnSlotClaim: @unchecked Sendable {
    private let ledger: TurnSlotLedger
    /// The ledger's ticket for this claim. `launchTurn` CONTINUES it as the
    /// turn's generation id, because a caller that preempted a background turn
    /// holds a token from before the launch and compares it after: the two
    /// must name the same turn.
    let ticket: UInt64
    /// Why the slot was taken. A claim is the ONLY evidence a turn in
    /// PREPARATION leaves: the ledger can say the slot is spoken for, and an
    /// operator's send must be able to tell a background run's preparation (a
    /// subagent step, which may be stopped) from a chat turn, a server request
    /// or a model load (none of which may be).
    let priority: TurnPriority

    fileprivate init(ledger: TurnSlotLedger, ticket: UInt64, priority: TurnPriority) {
        self.ledger = ledger
        self.ticket = ticket
        self.priority = priority
    }

    /// Gives the slot back. Idempotent, and callable from any isolation domain.
    func release() {
        ledger.release(ticket)
    }

    /// True while this claim is still the one the ledger is holding.
    var isHeld: Bool {
        ledger.holds(ticket)
    }
}

/// The generation slot's ownership, as a ledger of one outstanding claim.
///
/// A lock box rather than part of the engine's `stateLock` because the queue
/// releases claims from its own task's `defer`, and a release that had to hop to
/// the main actor could land after the next turn had already looked.
nonisolated final class TurnSlotLedger: @unchecked Sendable {
    private let lock = NSLock()
    /// Ticket of the outstanding claim, or 0 when the slot is unclaimed.
    private var outstanding: UInt64 = 0
    /// Why the outstanding claim was taken, so a prober can tell a background
    /// run's preparation from a chat turn's.
    private var outstandingPriority: TurnPriority = .interactive
    private var nextTicket: UInt64 = 0

    /// Takes the slot if nobody else holds it. Nil means somebody does.
    func tryClaim(priority: TurnPriority = .interactive) -> TurnSlotClaim? {
        lock.lock()
        defer { lock.unlock() }
        guard outstanding == 0 else { return nil }
        nextTicket &+= 1
        outstanding = nextTicket
        outstandingPriority = priority
        return TurnSlotClaim(ledger: self, ticket: nextTicket, priority: priority)
    }

    /// True while any claim is outstanding, i.e. a turn is being prepared.
    var isClaimed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return outstanding != 0
    }

    /// The priority of the outstanding claim, or nil when the slot is
    /// unclaimed. A claim's priority cannot change while it is held, so this is
    /// a stable answer for the length of a preparation.
    var claimedPriority: TurnPriority? {
        lock.lock()
        defer { lock.unlock() }
        return outstanding == 0 ? nil : outstandingPriority
    }

    func release(_ ticket: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        guard ticket != 0, outstanding == ticket else { return }
        outstanding = 0
        outstandingPriority = .interactive
    }

    /// Drops the outstanding claim when it belongs to `priority`.
    ///
    /// A claim is the only handle a turn in PREPARATION leaves, so this is how
    /// an operator's send stops one: the holder still runs its preparation, but
    /// `launchTurn` refuses a claim that is no longer held rather than running a
    /// turn nobody is waiting for. Only a background claim can be dropped here;
    /// an interactive one (chat, the API server, a model load) is never
    /// cancelled to make room.
    @discardableResult
    func cancelClaim(ifPriority priority: TurnPriority) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard outstanding != 0, outstandingPriority == priority else { return false }
        outstanding = 0
        outstandingPriority = .interactive
        return true
    }

    /// Ticket of the outstanding claim, or nil when the slot is unclaimed.
    ///
    /// A token for a caller that has to `await` between deciding to preempt and
    /// doing it: a claim released and re-taken carries a NEW ticket, so a stale
    /// token cannot stop the successor that took its place.
    var claimedTicket: UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return outstanding == 0 ? nil : outstanding
    }

    /// Drops the outstanding claim only when it is still the one named.
    @discardableResult
    func cancelClaim(ticket: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard ticket != 0, outstanding == ticket else { return false }
        outstanding = 0
        outstandingPriority = .interactive
        return true
    }

    func holds(_ ticket: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ticket != 0 && outstanding == ticket
    }
}
