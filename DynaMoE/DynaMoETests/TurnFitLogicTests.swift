//
//  TurnFitLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the launch-time KV fit check: the memory budget is
//  the authority over the session floor — a prompt that the machine's KV
//  budget cannot serve must be rejected (or have max_tokens ratcheted) before
//  any multi-GB allocation runs, and a budget that cannot hold even the floor
//  fails fast instead of over-provisioning it.
//

import XCTest
@testable import DynaMoE

final class TurnFitLogicTests: XCTestCase {

    private func fit(
        isAgentSession: Bool = false,
        promptTokens: Int,
        requestedMaxTokens: Int,
        modelFootprintGB: Double,
        physicalGB: Double = 16.0,
        budgetMode: MemoryBudgetMode = .balanced16GB,
        totalSlots: Int = 36,
        kStride: Int = 1024,
        vStride: Int = 1024,
        precision: KVCachePrecision = .fp16
    ) -> KVCacheProvisioning.TurnFit {
        KVCacheProvisioning.turnFit(
            isAgentSession: isAgentSession,
            promptTokens: promptTokens,
            requestedMaxTokens: requestedMaxTokens,
            config: nil,
            actualLayers: totalSlots,
            totalLoops: 1,
            numKvHeads: 4,
            headDim: 256,
            precision: precision,
            slidingSlotFlags: nil,
            slidingRingLen: 0,
            budgetMode: budgetMode,
            modelFootprintGB: modelFootprintGB,
            physicalGB: physicalGB
        )
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
        // 40 uniform slots, 19,352-token prompt, 4-m-token request. The clamped
        // provision lands far below the prompt length, so the turn must fail
        // before prefill runs off the allocated rows.
        let result = fit(
            isAgentSession: true,
            promptTokens: 19_352,
            requestedMaxTokens: 4_096,
            modelFootprintGB: 8.75,
            totalSlots: 40
        )
        if case .promptTooLong(let provisioned) = result.outcome {
            XCTAssertLessThan(provisioned, 19_352)
        } else {
            XCTFail("expected promptTooLong, got \(result.outcome)")
        }
        XCTAssertLessThanOrEqual(result.budgetGB, 0.25)
    }

    func testBudgetBelowFloorRejectsEvenARecentPrompt() {
        // 0.25 GB budget over 36 slots affords 910 rows, fewer than the
        // non-agent 2,048-row floor. (The config-less geometry floors KV heads
        // at 8, so a row costs (2,048+2,048) fp16 elements = 8,192 B here.)
        // The floor must not win over the budget: the turn fails fast
        // (promptTooLong) instead of provisioning 2,048 rows it cannot afford
        // and thrashing mid-generation.
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

    func testBudgetBelowAgentFloorFailsFastEvenForShortPrompts() {
        // The 8,192-row agent floor must not resurrect over-budget
        // provisions: a 0.25 GB budget affords 910 rows, so even a small
        // 1,000-token agent prompt rejects in milliseconds instead of
        // allocating a ~9x-over-budget floor and dying mid-generation.
        let result = fit(
            isAgentSession: true,
            promptTokens: 1_000,
            requestedMaxTokens: 4_096,
            modelFootprintGB: 8.75
        )
        guard case .promptTooLong(let provisioned) = result.outcome else {
            return XCTFail("expected promptTooLong, got \(result.outcome) (provisioned \(result.provisionedSeqLen))")
        }
        XCTAssertEqual(provisioned, 910)
        XCTAssertTrue(result.clampedByBudget)
    }

    func testPromptSlightlyAboveFloorIsRejected() {
        // minSeq 8192 < prompt 8300: the prompt no longer fits the floor'd
        // allocation relative to the budget, and there is no headroom slack.
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
            KVCacheProvisioning.neededSeqLen(isAgentSession: true, promptTokens: 100, requestedMaxTokens: 64),
            8192
        )
        XCTAssertEqual(
            KVCacheProvisioning.neededSeqLen(isAgentSession: false, promptTokens: 100, requestedMaxTokens: 64),
            2048
        )
        XCTAssertEqual(
            KVCacheProvisioning.neededSeqLen(isAgentSession: true, promptTokens: 20_000, requestedMaxTokens: 4_096),
            24_608
        )
        XCTAssertEqual(
            KVCacheProvisioning.neededSeqLen(isAgentSession: true, promptTokens: 40_000, requestedMaxTokens: 40_000),
            32_768
        )
    }
}
