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

    /// Marks this claim committed, i.e. past the point where a preemption may
    /// still drop it. False when it was preempted first, which means the launch
    /// must refuse itself rather than mint a generation id over another turn.
    @discardableResult
    func commit() -> Bool {
        ledger.commit(ticket)
    }

    /// True when the ledger dropped this claim to make room for another turn.
    ///
    /// The holder needs to tell that from "somebody else holds the slot now":
    /// the first means its work was preempted and should be reported as
    /// cancelled, the second is an ordinary busy engine. `cancelClaim` is the
    /// only writer, and it compares tickets, so a later claim's cancellation
    /// can never make this one answer true.
    var wasCancelled: Bool {
        ledger.wasCancelled(ticket)
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
    /// Tickets that were dropped by a preemption rather than released by their
    /// holder. Kept so the holder can tell "I was cancelled" from "I lost a
    /// race": both leave it not holding the slot, and only the first means its
    /// work should be reported as cancelled instead of failed.
    ///
    /// Bounded: an entry matters only until its holder looks, and a holder that
    /// never looks would otherwise grow this for the process's life. The oldest
    /// are dropped past `cancelledTicketLimit`, which can only lose a
    /// distinction for a claim cancelled that long ago.
    private var cancelledTickets: [UInt64] = []
    private static let cancelledTicketLimit = 64
    /// The highest ticket ever dropped by a preemption.
    ///
    /// The ring above is exact but bounded, so a holder that looks LATE (after
    /// 64 more cancellations) would be told its ticket was never cancelled and
    /// report a preemption as a failure. Ticket numbers only grow, so "at or
    /// below the watermark" answers for the ones the ring has aged out. The
    /// ring is still consulted first: within the limit the answer stays exact,
    /// and the watermark only speaks for tickets older than everything the
    /// ring holds. It can over-report for a ticket that was RELEASED normally
    /// before a later one was cancelled, which is why the only caller asks
    /// about a claim it did not release itself.
    private var cancelledWatermark: UInt64 = 0
    /// True once the holder has reached the point where its turn becomes the
    /// live generation. Past it a preemption is refused rather than dropped:
    /// the turn owns the KV cache by then, and dropping the claim would let a
    /// second turn take the slot and overwrite the first one's generation id.
    private var outstandingCommitted = false
    private var nextTicket: UInt64 = 0

    /// Takes the slot if nobody else holds it. Nil means somebody does.
    func tryClaim(priority: TurnPriority = .interactive) -> TurnSlotClaim? {
        lock.lock()
        defer { lock.unlock() }
        guard outstanding == 0 else { return nil }
        nextTicket &+= 1
        // 0 means "unclaimed", so a wrapped counter must never produce it.
        if nextTicket == 0 { nextTicket = 1 }
        outstanding = nextTicket
        outstandingPriority = priority
        outstandingCommitted = false
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
        outstandingCommitted = false
    }

    /// Marks the claim as past its point of no return.
    ///
    /// `launchTurn` calls this in the same locked step that mints the turn's
    /// generation id, so a preemption either lands before the turn is live (and
    /// the launch refuses itself) or not at all. Without it a cancel could land
    /// in the seconds a launch spends preparing — building a runtime, encoding
    /// images, resetting the KV cache — after which the preparing turn would
    /// still mint its id and overwrite the preemptor's.
    ///
    /// Returns false when the ticket is no longer held, i.e. it was preempted
    /// first; the caller must then refuse the launch instead of committing.
    @discardableResult
    func commit(_ ticket: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard ticket != 0, outstanding == ticket else { return false }
        outstandingCommitted = true
        return true
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
        guard outstanding != 0, outstandingPriority == priority, !outstandingCommitted else { return false }
        noteCancelledLocked(outstanding)
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
        guard ticket != 0, outstanding == ticket, !outstandingCommitted else { return false }
        noteCancelledLocked(ticket)
        outstanding = 0
        outstandingPriority = .interactive
        return true
    }

    /// Whether `ticket` was dropped by a preemption rather than released.
    func wasCancelled(_ ticket: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let oldest = cancelledTickets.first, ticket >= oldest {
            // Inside the ring's coverage: answer exactly, so a ticket that was
            // merely released is never reported as cancelled.
            return cancelledTickets.contains(ticket)
        }
        // Aged out (or the ring is empty): the watermark is all that is left,
        // and it is right for a ticket that was cancelled. A released ticket
        // below it would read as cancelled — accepted, because the one caller
        // asks only about a claim it did not release itself.
        return ticket != 0 && ticket <= cancelledWatermark
    }

    /// Records a dropped ticket. Caller holds the lock.
    private func noteCancelledLocked(_ ticket: UInt64) {
        cancelledTickets.append(ticket)
        cancelledWatermark = max(cancelledWatermark, ticket)
        if cancelledTickets.count > Self.cancelledTicketLimit {
            cancelledTickets.removeFirst(cancelledTickets.count - Self.cancelledTicketLimit)
        }
    }

    func holds(_ ticket: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ticket != 0 && outstanding == ticket
    }
}
