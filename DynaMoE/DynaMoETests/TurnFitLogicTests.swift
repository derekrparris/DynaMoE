//
//  TurnFitLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the launch-time KV fit check: a prompt that the
//  machine's KV budget cannot serve must be rejected (or have max_tokens
//  ratcheted) before any multi-GB allocation runs.
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

    func testMaxTokensIsRatchetedToHeadroom() {
        // 0.25 GB budget floor over 36 slots => 1,820 affordable rows, but the
        // non-agent 2,048-row floor holds the provision. A 1,700-token prompt
        // fits with only 348 rows of headroom left, so the 512-token request is
        // ratcheted down instead of silently overrunning the cache.
        let result = fit(
            isAgentSession: false,
            promptTokens: 1_700,
            requestedMaxTokens: 512,
            modelFootprintGB: 8.75
        )
        guard case .maxTokensReduced(let provisioned) = result.outcome else {
            return XCTFail("expected maxTokensReduced, got \(result.outcome) (provisioned \(result.provisionedSeqLen))")
        }
        XCTAssertEqual(provisioned, 2048)
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
        // Opencode's real shape on a 16 GB machine: a 7,746-token tool-enabled
        // prompt lands inside the 8,192-row agent floor with only 446 rows of
        // headroom, still below the 512-row margin. The ratchet must floor at
        // 64 tokens of generation rather than dropping to 1.
        let result = fit(
            isAgentSession: true,
            promptTokens: 7_746,
            requestedMaxTokens: 8_192,
            modelFootprintGB: 14.0,
            totalSlots: 40
        )
        if case .maxTokensReduced(let provisioned) = result.outcome {
            XCTAssertEqual(provisioned, 8192)
            XCTAssertEqual(result.effectiveMaxTokens, 64)
        } else {
            XCTFail("expected maxTokensReduced, got \(result.outcome)")
        }
    }

    func testAgentSessionFloorCannotSwallowPromptWithinBudget() {
        // The KV floor is at least minSeq (8192 for agent sessions); a prompt
        // that exactly consumes the floor with a large request gets its
        // max_tokens ratcheted to what remains, never rejected for budget it
        // technically fills.
        let result = fit(
            isAgentSession: true,
            promptTokens: 8_000,
            requestedMaxTokens: 4_096,
            modelFootprintGB: 14.0,
            totalSlots: 40
        )
        if case .promptTooLong = result.outcome {
            XCTFail("a prompt inside the 8192-row floor must not be rejected")
        }
        XCTAssertGreaterThan(result.effectiveMaxTokens, 0)
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
