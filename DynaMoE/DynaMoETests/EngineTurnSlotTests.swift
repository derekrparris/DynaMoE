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
    private func makeRequest() -> CompletionTurnRequest {
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
            images: []
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
            XCTAssertTrue(
                text.contains("API server"),
                "the refusal must name the caller holding the slot, not claim a generation is running: \(text)"
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
        XCTAssertThrowsError(try engine.launchTurn(makeRequest()))
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
}
