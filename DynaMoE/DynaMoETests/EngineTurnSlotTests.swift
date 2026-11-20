//
//  EngineTurnSlotTests.swift
//  DynaMoETests
//
//  The engine's single-flight generation slot as a holdable claim: the ledger
//  itself, and the two launch paths that have to honour it. No model is loaded —
//  a refused launch gets as far as the missing checkpoint, which is what makes
//  the difference between "refused for holding no claim" and "refused because
//  nothing is loaded" observable.
//

import XCTest
@testable import DynaMoE

final class EngineTurnSlotTests: XCTestCase {

    /// The smallest request the engine will look at. Every field is read only
    /// after the slot is taken, so the values never matter: a launch in this file
    /// is refused for its claim or for the missing model.
    private func makeRequest(priority: TurnPriority = .interactive) -> CompletionTurnRequest {
        CompletionTurnRequest(
            promptTokenIds: [1, 2, 3],
            formattedPrompt: "hi",
            sessionId: nil,
            sampling: TurnSampling(
                temperature: 0,
                topP: 1,
                minP: 0,
                topK: 1,
                repetitionPenalty: 1,
                maxTokens: 1
            ),
            memory: TurnMemorySettings(
                executionMode: .autoDetect,
                budgetMode: .balanced16GB,
                kvPrecision: .fp16
            ),
            agentToolsEnabled: false,
            thinkingEnabled: false,
            agentStep: 0,
            forceSynthesis: false,
            isSplicedContinuation: false,
            stopSequences: [],
            asOfDate: nil,
            images: [],
            turnPriority: priority
        )
    }

    private func message(_ error: Error) -> String {
        (error as? CompletionLaunchError)?.errorDescription ?? "\(error)"
    }

    // MARK: - The ledger

    func testTheLedgerGrantsOneClaimAtATime() {
        let ledger = TurnSlotLedger()
        XCTAssertFalse(ledger.isClaimed)

        let first = ledger.tryClaim()
        XCTAssertNotNil(first)
        XCTAssertTrue(ledger.isClaimed)
        XCTAssertNil(ledger.tryClaim(), "a second caller must be told no, not handed the same slot")

        first?.release()
        XCTAssertFalse(ledger.isClaimed)
        XCTAssertNotNil(ledger.tryClaim(), "and the slot is usable again once it is given back")
    }

    func testAClaimKnowsWhenItNoLongerHoldsTheSlot() {
        let ledger = TurnSlotLedger()
        let claim = ledger.tryClaim()
        XCTAssertEqual(claim?.isHeld, true)
        claim?.release()
        XCTAssertEqual(claim?.isHeld, false)
        claim?.release()
        XCTAssertEqual(claim?.isHeld, false, "releasing twice is free")
    }

    /// Tickets are unique, so a claim that was already given back cannot release
    /// the slot out from under whoever holds it now. Both the queue and the launch
    /// path release on their way out, and a stale one must not land on a live turn.
    func testAStaleClaimCannotReleaseSomebodyElsesSlot() throws {
        let ledger = TurnSlotLedger()
        let stale = try XCTUnwrap(ledger.tryClaim())
        stale.release()

        let live = try XCTUnwrap(ledger.tryClaim())
        stale.release()
        XCTAssertTrue(live.isHeld, "the stale release must be a no-op")
        XCTAssertTrue(ledger.isClaimed)

        live.release()
        XCTAssertFalse(ledger.isClaimed)
    }

    /// The three callers that compete for the one slot — chat, the API
    /// server's queue and a subagent run's step — reserving from their own
    /// concurrent tasks. The sequential cases above could not see a ledger that
    /// let two of them in at once.
    func testConcurrentReservationsFromEveryCallerGrantExactlyOneHolder() async {
        let ledger = TurnSlotLedger()
        let counter = HeldClaimCounter()
        let priorities: [TurnPriority] = [.interactive, .interactive, .background]
        await withTaskGroup(of: Void.self) { group in
            for priority in priorities {
                for _ in 0..<32 {
                    group.addTask {
                        for _ in 0..<20 {
                            guard let claim = ledger.tryClaim(priority: priority) else {
                                await Task.yield()
                                continue
                            }
                            counter.enter()
                            // Hold ACROSS a suspension point: that is exactly
                            // the window in which a second caller must be
                            // refused.
                            await Task.yield()
                            counter.leave()
                            claim.release()
                        }
                    }
                }
            }
        }
        XCTAssertGreaterThan(counter.granted, 0, "the race must actually grant claims, or it proves nothing")
        XCTAssertEqual(counter.peak, 1, "two callers must never believe they hold the single slot")
        XCTAssertFalse(ledger.isClaimed)
    }

    // MARK: - The engine

    @MainActor
    func testAServerClaimMakesTheEngineLookBusy() throws {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")

        let claim = try XCTUnwrap(engine.reserveTurnSlot())
        defer { claim.release() }
        XCTAssertTrue(engine.isActive, "a turn being prepared occupies the slot exactly like a running one")
        XCTAssertNil(engine.reserveTurnSlot(), "and nobody else may claim it")
    }

    /// The window this closes: the queue has decided the slot is free and is
    /// preparing a turn. A chat turn arriving now must be told, immediately and
    /// truthfully, rather than launching into a slot the queue is about to take.
    @MainActor
    func testAnInAppLaunchIsRefusedWhileAServerTurnHoldsTheSlot() throws {
        let engine = CompletionEngine.shared
        let claim = try XCTUnwrap(engine.reserveTurnSlot())
        defer { claim.release() }

        XCTAssertThrowsError(try engine.launchTurn(makeRequest())) { error in
            let text = message(error)
            // The ledger does not know WHO holds the slot (a queued server turn, a
            // chat turn still preparing, or a model load all claim the same way),
            // so the honest refusal says that rather than naming the API server
            // and being wrong the rest of the time.
            XCTAssertTrue(
                text.contains("holding the generation slot"),
                "the refusal must say the slot is held: \(text)"
            )
            XCTAssertTrue(
                text.contains("turn being prepared") && text.contains("model load"),
                "the refusal must name what can be holding it: \(text)"
            )
        }
    }

    /// A launch handed the queue's claim is honoured: it must get past the claim
    /// check and fail on the missing model instead, and it must leave the slot
    /// free for the next turn.
    @MainActor
    func testAServerClaimIsHonouredByItsLaunchAndGivenBack() throws {
        let engine = CompletionEngine.shared
        let claim = try XCTUnwrap(engine.reserveTurnSlot())

        XCTAssertThrowsError(try engine.launchTurn(makeRequest(), claim: claim)) { error in
            let text = message(error)
            XCTAssertFalse(
                text.contains("API server"),
                "a launch that holds the claim must not be refused for holding it: \(text)"
            )
            XCTAssertTrue(
                text.contains("Metal or Tokenizer"),
                "it must reach the missing model, which is the real reason it cannot run: \(text)"
            )
        }
        XCTAssertFalse(claim.isHeld, "the launch released the claim when it refused")
        XCTAssertFalse(engine.isActive, "and nothing is left occupying the slot")
    }

    /// Every early throw in `launchTurn` — this case takes the first one that can
    /// fire without a model — must give the slot back, or a refused turn would
    /// wedge the server for the rest of the session.
    @MainActor
    func testARefusedLaunchWithoutAClaimGivesTheSlotBack() {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")
        XCTAssertThrowsError(try engine.launchTurn(makeRequest())) { error in
            // Not just "an error": a refusal for the slot would also throw,
            // and would make this case pass while proving the opposite.
            let text = message(error)
            XCTAssertTrue(
                text.contains("Metal or Tokenizer"),
                "the refusal must come from the missing model, not from a lost slot: \(text)"
            )
        }
        XCTAssertFalse(engine.isActive, "a launch that took the slot and refused must not keep it")
    }

    // MARK: - The in-app send path reserves too

    /// The mirror of `testAnInAppLaunchIsRefusedWhileAServerTurnHoldsTheSlot`, and
    /// the reason `ContentView.startAutoregressiveGeneration` reserves before it
    /// renders anything: a chat turn that holds the slot makes the SERVER's claim
    /// fail, and the queue answers a failed claim by going back to waiting instead
    /// of losing the request it had already prepared. The chat turn is the one that
    /// runs, and it runs because of the order, not because of who asked first.
    @MainActor
    func testAnInAppReservationKeepsAServerTurnFromClaimingMidPreparation() throws {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")

        let inApp = try XCTUnwrap(engine.reserveTurnSlot())
        defer { inApp.release() }

        XCTAssertTrue(engine.isActive, "a chat turn being prepared occupies the slot like a running one")
        XCTAssertNil(
            engine.reserveTurnSlot(),
            "the server's claim must fail, which its queue turns into another wait rather than a refusal"
        )
        XCTAssertTrue(inApp.isHeld, "and the failed attempt must not have taken the slot from the chat turn")
    }

    /// A claim is what authorises a launch, so one that was given back must not
    /// authorise anything. Without this the launch would run a turn while the slot
    /// read free to every other caller.
    @MainActor
    func testAReleasedClaimCannotAuthoriseALaunch() throws {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")

        let spent = try XCTUnwrap(engine.reserveTurnSlot())
        spent.release()

        XCTAssertThrowsError(try engine.launchTurn(makeRequest(), claim: spent)) { error in
            let text = message(error)
            XCTAssertTrue(
                text.contains("reservation"),
                "the refusal must be about the claim, not about a missing model: \(text)"
            )
        }
        XCTAssertFalse(engine.isActive, "and a launch refused for its claim must not have taken the slot")
    }

    /// The claim the in-app path holds across its preparation is the same claim its
    /// launch redeems: past the claim check, on to the missing model, and given back
    /// on the way out. This is the pair `startAutoregressiveGeneration` performs.
    @MainActor
    func testAnInAppClaimIsRedeemedByItsLaunchAndGivenBack() throws {
        let engine = CompletionEngine.shared
        let claim = try XCTUnwrap(engine.reserveTurnSlot())

        XCTAssertThrowsError(try engine.launchTurn(makeRequest(), claim: claim)) { error in
            let text = message(error)
            XCTAssertFalse(
                text.contains("API server"),
                "a launch holding the chat turn's claim must not be refused for the server: \(text)"
            )
            XCTAssertFalse(text.contains("reservation"), "the claim was live: \(text)")
            XCTAssertTrue(
                text.contains("Metal or Tokenizer"),
                "it must reach the missing model, which is the real reason it cannot run: \(text)"
            )
        }
        XCTAssertFalse(claim.isHeld, "the launch released the claim when it refused")
        XCTAssertFalse(engine.isActive, "and nothing is left occupying the slot")
    }

    // MARK: - Ownership-scoped cancellation

    /// `cancelTurn(id:)` names a TURN; `cancelBackgroundTurn()` names a
    /// PRIORITY. A subagent's cancel arrives while its run unwinds, by which
    /// time the slot may already belong to a successor, so a stale id has to be
    /// a no-op. No live turn is needed to prove the reachable half of that: a
    /// claim being PREPARED is exactly the state a stale cancel must not touch.
    @MainActor
    func testAStaleCancelIsANoOpOnAPreparingTurn() throws {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")

        XCTAssertFalse(engine.cancelTurn(id: UInt64.max), "an id the engine never issued cancels nothing")
        XCTAssertFalse(engine.cancelBackgroundTurn(), "and no priority-scoped cancel fires without a turn")
        XCTAssertFalse(engine.hasActiveBackgroundTurn)

        let preparing = try XCTUnwrap(engine.reserveTurnSlot())
        defer { preparing.release() }
        XCTAssertFalse(
            engine.cancelTurn(id: UInt64.max),
            "a stale id must not reach the claim another caller is preparing under"
        )
        XCTAssertTrue(preparing.isHeld, "the preparing turn's claim survives a stale cancel")
        XCTAssertTrue(engine.isActive, "and the engine still reads busy to every prober")
    }

    /// The successor case the old version could not observe: a REGISTERED
    /// turn exists (so `cancelTurn`'s second branch is reachable), and a stale
    /// id must leave it running while its own id stops it. The test bundle has
    /// no model, so the turn is installed through the DEBUG seam.
    @MainActor
    func testAStaleCancelIsANoOpOnARegisteredSuccessorTurn() async throws {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")

        let successorId: UInt64 = 4_242
        let successor = Task<Void, Never> {
            try? await Task.sleep(nanoseconds: 60_000_000_000)
        }
        engine.installTurnForTesting(id: successorId, task: successor)
        defer {
            engine.clearTurnForTesting()
            successor.cancel()
        }

        XCTAssertTrue(engine.isActive, "the installed turn owns the slot")
        XCTAssertFalse(
            engine.cancelTurn(id: successorId &+ 1),
            "a stale id must be a no-op even while a successor turn is registered"
        )
        XCTAssertFalse(successor.isCancelled, "the stale id must not stop the successor")
        XCTAssertTrue(engine.isActive, "and the successor still owns the slot")

        XCTAssertTrue(engine.cancelTurn(id: successorId), "the owning id does stop it")
        XCTAssertTrue(successor.isCancelled)
    }

    /// `hasActiveBackgroundTurn` counts a turn being PREPARED (a claim, no
    /// task), so a cancel that only looked at `currentGenerationTask` was
    /// inert in exactly the window the prober had just called cancellable.
    @MainActor
    func testAPreparationClaimIsStoppableOnlyWhenItIsBackground() throws {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")

        let background = try XCTUnwrap(engine.reserveTurnSlot(priority: .background))
        XCTAssertTrue(engine.hasActiveBackgroundTurn, "a background preparation is a background turn")
        XCTAssertTrue(engine.cancelBackgroundTurn(), "and it can be stopped without a task to cancel")
        XCTAssertFalse(background.isHeld, "its launch is refused because the claim was given back")
        XCTAssertFalse(engine.isActive)
        XCTAssertThrowsError(try engine.launchTurn(makeRequest(priority: .background), claim: background)) { error in
            XCTAssertTrue(
                message(error).contains("reservation"),
                "a turn stopped during preparation must refuse for its claim: \(message(error))"
            )
        }

        let interactive = try XCTUnwrap(engine.reserveTurnSlot(priority: .interactive))
        defer { interactive.release() }
        XCTAssertFalse(engine.hasActiveBackgroundTurn)
        XCTAssertFalse(
            engine.cancelBackgroundTurn(),
            "a chat/server/model-load preparation is never cancelled to make room"
        )
        XCTAssertTrue(interactive.isHeld)
    }

    /// The claim's priority is what every other prober reads to decide whether
    /// the slot may be preempted; a claim taken for one priority and launched
    /// as the other would lie about that for the turn's whole length.
    @MainActor
    func testAClaimTakenForOnePriorityCannotLaunchTheOther() throws {
        let engine = CompletionEngine.shared
        XCTAssertFalse(engine.isActive, "the slot must be free at the start of the case")

        let claim = try XCTUnwrap(engine.reserveTurnSlot(priority: .interactive))
        XCTAssertThrowsError(try engine.launchTurn(makeRequest(priority: .background), claim: claim)) { error in
            XCTAssertTrue(
                message(error).contains("priority"),
                "the refusal must name the priority mismatch: \(message(error))"
            )
        }
        XCTAssertFalse(claim.isHeld, "the launch released the claim it refused")
        XCTAssertFalse(engine.isActive, "and nothing is left occupying the slot")
    }
}

/// Counts how many claims are held at once, so the concurrency case can prove
/// the ledger never looked free to two callers.
private final class HeldClaimCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var current = 0
    private var peakValue = 0
    private var grantedValue = 0

    func enter() {
        lock.lock()
        current += 1
        peakValue = max(peakValue, current)
        grantedValue += 1
        lock.unlock()
    }

    func leave() {
        lock.lock()
        current -= 1
        lock.unlock()
    }

    var peak: Int {
        lock.lock()
        defer { lock.unlock() }
        return peakValue
    }

    var granted: Int {
        lock.lock()
        defer { lock.unlock() }
        return grantedValue
    }
}
