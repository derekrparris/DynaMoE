//
//  EngineTurnSlot.swift
//  DynaMoE
//
//  Phase: the engine's single-flight generation slot, as something a caller can
//  HOLD rather than merely look at.
//

import Foundation

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
    private let ticket: UInt64

    fileprivate init(ledger: TurnSlotLedger, ticket: UInt64) {
        self.ledger = ledger
        self.ticket = ticket
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
    private var nextTicket: UInt64 = 0

    /// Takes the slot if nobody else holds it. Nil means somebody does.
    func tryClaim() -> TurnSlotClaim? {
        lock.lock()
        defer { lock.unlock() }
        guard outstanding == 0 else { return nil }
        nextTicket &+= 1
        outstanding = nextTicket
        return TurnSlotClaim(ledger: self, ticket: nextTicket)
    }

    /// True while any claim is outstanding, i.e. a turn is being prepared.
    var isClaimed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return outstanding != 0
    }

    func release(_ ticket: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        guard ticket != 0, outstanding == ticket else { return }
        outstanding = 0
    }

    func holds(_ ticket: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ticket != 0 && outstanding == ticket
    }
}
