//
//  TurnFitLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the launch-time KV fit check. The memory budget is
//  the authority over the provision (a turn is never allocated past it), but
//  the session floor is a preference: an agent reserve the budget cannot afford
//  is capped, not turned into a rejection of prompts it could have served. The
//  check must also price the same sliding-ring layout the allocator uses.
//

import XCTest
@testable import DynaMoE

final class TurnFitLogicTests: XCTestCase {

    private func fit(
        isAgentSession: Bool = false,
        configuredFloor: Int? = nil,
        promptTokens: Int,
        requestedMaxTokens: Int,
        modelFootprintGB: Double,
        physicalGB: Double = 16.0,
        budgetMode: MemoryBudgetMode = .balanced16GB,
        totalSlots: Int = 36,
        kStride: Int = 1024,
        vStride: Int = 1024,
        precision: KVCachePrecision = .fp16,
        slidingSlotFlags: [Bool]? = nil,
        slidingRingLen: Int = 0
    ) -> KVCacheProvisioning.TurnFit {
        KVCacheProvisioning.turnFit(
            sessionFloor: KVCacheProvisioning.sessionFloor(isAgentSession: isAgentSession, configured: configuredFloor),
            promptTokens: promptTokens,
            requestedMaxTokens: requestedMaxTokens,
            config: nil,
            actualLayers: totalSlots,
            totalLoops: 1,
            numKvHeads: 4,
            headDim: 256,
            precision: precision,
            slidingSlotFlags: slidingSlotFlags,
            slidingRingLen: slidingRingLen,
            budgetMode: budgetMode,
            modelFootprintGB: modelFootprintGB,
            physicalGB: physicalGB
        )
    }

    /// 30 Gemma-4-shaped slots: the first 25 are sliding rings (window 1,024 +
    /// the 512-row prefill chunk slack), the last 5 are global (linear).
    private func gemmaSlidingFlags() -> [Bool] {
        (0..<30).map { $0 < 25 }
    }

    func testAmpleBudgetAcceptsPromptAsRequested() {
        let result = fit(promptTokens: 500, requestedMaxTokens: 512, modelFootprintGB: 2.0)
        XCTAssertEqual(result.outcome, .ok)
        XCTAssertEqual(result.effectiveMaxTokens, 512)
        XCTAssertEqual(result.provisionedSeqLen, 2048)
        XCTAssertFalse(result.clampedByBudget)
    }

    func testPromptLongerThanClampedBudgetIsRejected() {
        // Streaming hybrid shape like the 16 GB repro: 0.25 GB KV budget floor,
        // 40 uniform slots, 19,352-token prompt, 4k-token request. The clamped
        // provision lands far below the prompt length, so the turn must fail
        // before prefill runs off the allocated rows.
        let result = fit(
            isAgentSession: true,
            promptTokens: 19_352,
            requestedMaxTokens: 4_096,
            modelFootprintGB: 8.75,
            totalSlots: 40
        )
        guard case .promptTooLong(let provisioned) = result.outcome else {
            return XCTFail("expected promptTooLong, got \(result.outcome)")
        }
        XCTAssertEqual(provisioned, result.affordableRows)
        XCTAssertLessThan(provisioned, 19_352)
        XCTAssertLessThanOrEqual(result.budgetGB, 0.25)
    }

    func testPromptLargerThanAffordableProvisionIsRejected() {
        // 0.25 GB over 36 slots affords 910 rows at the config-less fp16 shape.
        // A 1,700-token prompt cannot fit those rows, and the rejection reports
        // the provision the budget actually affords.
        let result = fit(
            isAgentSession: false,
            promptTokens: 1_700,
            requestedMaxTokens: 512,
            modelFootprintGB: 8.75
        )
        guard case .promptTooLong(let provisioned) = result.outcome else {
            return XCTFail("expected promptTooLong, got \(result.outcome) (provisioned \(result.provisionedSeqLen))")
        }
        XCTAssertEqual(provisioned, 910)
        XCTAssertEqual(result.affordableRows, 910)
        XCTAssertTrue(result.clampedByBudget)
        XCTAssertEqual(result.effectiveMaxTokens, 0)
    }

    func testMaxTokensIsRatchetedToHeadroom() {
        // A budget just big enough to hold the 2,048-row non-agent floor:
        // 0.5625 GB over 36 slots affords exactly 2,048 rows while a 1,700
        // token prompt and 512-token request need 2,724, so the provision
        // leaves thin headroom and the request is ratcheted instead of
        // overrunning the cache.
        let result = fit(
            isAgentSession: false,
            promptTokens: 1_700,
            requestedMaxTokens: 512,
            modelFootprintGB: 8.4375
        )
        guard case .maxTokensReduced(let provisioned) = result.outcome else {
            return XCTFail("expected maxTokensReduced, got \(result.outcome) (provisioned \(result.provisionedSeqLen))")
        }
        XCTAssertEqual(provisioned, 2_048)
        XCTAssertTrue(result.clampedByBudget)
        // 348 rows of headroom minus the 512-row margin goes negative, so the
        // ratchet floors at a 64-token generation budget instead of collapsing.
        XCTAssertEqual(result.effectiveMaxTokens, 64)
        XCTAssertLessThanOrEqual(
            result.effectiveMaxTokens + 1_700,
            provisioned,
            "the ratchet must keep the generated row inside the provisioned cache"
        )
    }

    func testNearFullAgentPromptKeepsUsableGenerationBudget() {
        // Opencode's real shape on a budget that exactly holds the 8,192-row
        // agent floor (4 slots x 0.25 GB => 8,192 affordable rows): a 7,746
        // token prompt leaves 446 rows of headroom, still below the 512-row
        // margin. The ratchet must floor at 64 tokens of generation rather
        // than dropping to 1.
        let result = fit(
            isAgentSession: true,
            promptTokens: 7_746,
            requestedMaxTokens: 8_192,
            modelFootprintGB: 14.0,
            totalSlots: 4
        )
        if case .maxTokensReduced(let provisioned) = result.outcome {
            XCTAssertEqual(provisioned, 8192)
            XCTAssertEqual(result.effectiveMaxTokens, 64)
        } else {
            XCTFail("expected maxTokensReduced, got \(result.outcome)")
        }
    }

    func testAgentSessionFloorCannotSwallowPromptWithinBudget() {
        // 1.25 GB budget over 20 slots affords exactly 8,192 rows, so the
        // agent floor holds: a prompt that consumes most of it with a large
        // request gets its max_tokens ratcheted to what remains, never
        // rejected for budget it technically fills.
        let result = fit(
            isAgentSession: true,
            promptTokens: 8_000,
            requestedMaxTokens: 4_096,
            modelFootprintGB: 7.75,
            totalSlots: 20
        )
        if case .promptTooLong = result.outcome {
            XCTFail("a prompt inside an affordable 8192-row floor must not be rejected")
        }
        XCTAssertGreaterThan(result.effectiveMaxTokens, 0)
    }

    func testAgentFloorIsCappedByTheBudgetForShortPrompts() {
        // The field regression: a 0.25 GB budget affords 910 rows, fewer than
        // the 8,192-row agent floor. The floor must not reject a prompt those
        // rows can hold — it is capped to what the budget affords and the turn
        // runs with a ratcheted generation budget.
        let result = fit(
            isAgentSession: true,
            promptTokens: 500,
            requestedMaxTokens: 4_096,
            modelFootprintGB: 8.75
        )
        if case .promptTooLong = result.outcome {
            XCTFail("a 500-token agent prompt must fit the 910 rows the budget affords")
        }
        XCTAssertEqual(result.provisionedSeqLen, 910)
        XCTAssertEqual(result.affordableRows, 910)
        XCTAssertEqual(result.sessionFloor, 8192)
        XCTAssertGreaterThan(result.effectiveMaxTokens, 0)
    }

    func testSessionFloorOverrideReplacesOnlyTheAgentReserve() {
        XCTAssertEqual(KVCacheProvisioning.sessionFloor(isAgentSession: true, configured: nil), 8192)
        XCTAssertEqual(KVCacheProvisioning.sessionFloor(isAgentSession: true, configured: 0), 8192)
        XCTAssertEqual(KVCacheProvisioning.sessionFloor(isAgentSession: true, configured: 4_096), 4_096)
        // The reserve cannot raise the provision past the sequence cap.
        XCTAssertEqual(KVCacheProvisioning.sessionFloor(isAgentSession: true, configured: 40_000), 32_768)
        XCTAssertEqual(KVCacheProvisioning.sessionFloor(isAgentSession: false, configured: 4_096), 2048)
        XCTAssertEqual(KVCacheProvisioning.affordableFloor(sessionFloor: 8_192, affordableRows: 910), 910)
        XCTAssertEqual(KVCacheProvisioning.affordableFloor(sessionFloor: 2_048, affordableRows: 9_000), 2_048)
        XCTAssertEqual(KVCacheProvisioning.affordableFloor(sessionFloor: 8_192, affordableRows: 0), 1)
    }

    func testUniformGeometryOverpricesGemmaSlidingRingTurn() {
        // The exact field error (16 GB Mac, 0.25 GB KV budget at fp8, 3,065
        // prompt tokens from the in-app agent prompt): priced with 30 linear
        // slots the budget "affords" 2,184 rows, which is below the prompt and
        // rejects the turn. This is the shape the pre-fix fit check used.
        let result = fit(
            isAgentSession: true,
            promptTokens: 3_065,
            requestedMaxTokens: 8_192,
            modelFootprintGB: 8.75,
            totalSlots: 30,
            precision: .fp8
        )
        guard case .promptTooLong(let provisioned) = result.outcome else {
            return XCTFail("expected promptTooLong, got \(result.outcome)")
        }
        XCTAssertEqual(provisioned, 2_184)
        XCTAssertEqual(result.affordableRows, 2_184)
    }

    func testRingAwareGeometryServesTheFieldPrompt() {
        // Same machine, model, budget and prompt as the field error, priced with
        // the layout the allocator uses: 25 ring slots hold window+slack rows
        // regardless of the sequence length, so the budget affords 5,427 rows
        // rather than 2,184 and the 3,065-token agent prompt runs with a
        // ratcheted generation budget.
        let result = fit(
            isAgentSession: true,
            promptTokens: 3_065,
            requestedMaxTokens: 8_192,
            modelFootprintGB: 8.75,
            totalSlots: 30,
            precision: .fp8,
            slidingSlotFlags: gemmaSlidingFlags(),
            slidingRingLen: 1_536
        )
        guard case .maxTokensReduced(let provisioned) = result.outcome else {
            return XCTFail("expected maxTokensReduced, got \(result.outcome) (provisioned \(result.provisionedSeqLen))")
        }
        XCTAssertEqual(provisioned, 5_427)
        XCTAssertEqual(result.affordableRows, 5_427)
        XCTAssertEqual(result.effectiveMaxTokens, 1_850)
        XCTAssertEqual(result.effectiveMaxTokens + 3_065, 4_915)
        XCTAssertLessThanOrEqual(result.effectiveMaxTokens + 3_065, provisioned)
    }

    func testPromptBeyondSequenceCapIsRejected() {
        // A 50,000-token prompt exceeds the 32,768-row cap and the affordable
        // provision, so it is rejected for its own length.
        let result = fit(
            isAgentSession: true,
            promptTokens: 50_000,
            requestedMaxTokens: 8_192,
            modelFootprintGB: 2.0
        )
        if case .promptTooLong = result.outcome {} else {
            XCTFail("expected promptTooLong for a 50k prompt in a 32k-capped provision, got \(result.outcome)")
        }
    }

    func testNeededSeqLenMatchesLaunchPathFormula() {
        XCTAssertEqual(
            KVCacheProvisioning.neededSeqLen(sessionFloor: 8192, promptTokens: 100, requestedMaxTokens: 64),
            8192
        )
        XCTAssertEqual(
            KVCacheProvisioning.neededSeqLen(sessionFloor: 2048, promptTokens: 100, requestedMaxTokens: 64),
            2048
        )
        XCTAssertEqual(
            KVCacheProvisioning.neededSeqLen(sessionFloor: 8192, promptTokens: 20_000, requestedMaxTokens: 4_096),
            24_608
        )
        XCTAssertEqual(
            KVCacheProvisioning.neededSeqLen(sessionFloor: 8192, promptTokens: 40_000, requestedMaxTokens: 40_000),
            32_768
        )
    }
}
