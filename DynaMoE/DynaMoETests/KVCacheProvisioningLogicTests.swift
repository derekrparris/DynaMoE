//
//  KVCacheProvisioningLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the KV-cache provisioning math: layout geometry,
//  provisioned bytes, the budget clamp, and the machine/model/mode budget.
//  No Metal device, no model files, no MainActor required.
//

import XCTest
@testable import DynaMoE

final class KVCacheProvisioningLogicTests: XCTestCase {

    /// Spark X2.5 4B-shaped uniform layout: 36 slots, kvStride 4 × 256 = 1024,
    /// FP16. No ring slots (the sliding-ring layout is Gemma 4 only).
    private func sparkGeometry(precision: KVCachePrecision = .fp16) -> KVCacheProvisioning.Geometry {
        KVCacheProvisioning.Geometry(
            totalSlots: 36,
            slidingSlotFlags: nil,
            slidingRingLen: 0,
            kStride: 1024,
            vStride: 1024,
            elementBytes: precision.bytesPerElement
        )
    }

    func testUniformGeometryBytesPerPosition() {
        let geometry = sparkGeometry()
        // 36 slots × 32768 positions × 1024 stride × 2 bytes × 2 caches (K and V).
        let expected = 36 * 32768 * 1024 * 2 * 2
        XCTAssertEqual(KVCacheProvisioning.requiredBytes(maxSeqLen: 32768, geometry: geometry), expected)
        XCTAssertEqual(
            KVCacheProvisioning.requiredBytes(maxSeqLen: 16384, geometry: geometry) * 2,
            expected,
            "provisioned bytes must scale linearly with sequence length"
        )
    }

    func testPrecisionScalesProvisionedBytes() {
        let fp16 = KVCacheProvisioning.requiredBytes(maxSeqLen: 8192, geometry: sparkGeometry(precision: .fp16))
        let fp8 = KVCacheProvisioning.requiredBytes(maxSeqLen: 8192, geometry: sparkGeometry(precision: .fp8))
        let fp32 = KVCacheProvisioning.requiredBytes(maxSeqLen: 8192, geometry: sparkGeometry(precision: .fp32))
        XCTAssertLessThan(fp8, fp16)
        XCTAssertLessThan(fp16, fp32)
    }

    func testRingSlotsDoNotGrowWithSequenceLength() {
        // 8 slots, half of them ring (window 512 + slack): the ring rows are fixed
        // and only the linear slots scale with maxSeqLen.
        let geometry = KVCacheProvisioning.Geometry(
            totalSlots: 8,
            slidingSlotFlags: [true, false, true, false, true, false, true, false],
            slidingRingLen: 528,
            kStride: 1024,
            vStride: 1024,
            elementBytes: 2
        )
        XCTAssertEqual(KVCacheProvisioning.ringSlotCount(geometry), 4)
        XCTAssertEqual(KVCacheProvisioning.positions(maxSeqLen: 32768, geometry: geometry), 4 * 528 + 4 * 32768)
        XCTAssertEqual(KVCacheProvisioning.positions(maxSeqLen: 1024, geometry: geometry), 4 * 528 + 4 * 1024)
    }

    func testClampFitsBudgetAndRespectsFloor() {
        let geometry = sparkGeometry()
        // 1 GB budget, 36 linear slots × (1024 K + 1024 V) × 2 bytes per position.
        let clamped = KVCacheProvisioning.clampedSeqLen(requested: 32768, floor: 2048, geometry: geometry, budgetGB: 1.0)
        let expected = 1_073_741_824 / (36 * 4096)
        XCTAssertEqual(clamped, expected)
        XCTAssertLessThanOrEqual(
            KVCacheProvisioning.requiredBytes(maxSeqLen: clamped, geometry: geometry),
            1_073_741_824,
            "the clamped length's K+V allocation must fit the budget"
        )
        // A budget too small to hold anything still cannot drop below the floor.
        XCTAssertEqual(KVCacheProvisioning.clampedSeqLen(requested: 32768, floor: 2048, geometry: geometry, budgetGB: 0.001), 2048)
    }

    func testClampPassesThroughWhenBudgetAmpleOrRequestAtFloor() {
        let geometry = sparkGeometry()
        XCTAssertEqual(KVCacheProvisioning.clampedSeqLen(requested: 8192, floor: 8192, geometry: geometry, budgetGB: 0.5), 8192,
                       "a request at the floor is never clamped")
        XCTAssertEqual(KVCacheProvisioning.clampedSeqLen(requested: 4096, floor: 8192, geometry: geometry, budgetGB: 0.5), 4096,
                       "a request below the floor passes through unchanged")
        XCTAssertEqual(KVCacheProvisioning.clampedSeqLen(requested: 32768, floor: 2048, geometry: geometry, budgetGB: .infinity), 32768,
                       "an infinite budget never clamps")
        XCTAssertEqual(KVCacheProvisioning.clampedSeqLen(requested: 32768, floor: 2048, geometry: geometry, budgetGB: 64), 32768,
                       "an ample budget never clamps")
    }

    func testAffordableSeqLenDerivesRowsFromBudget() {
        let geometry = sparkGeometry()
        // Same arithmetic the budget clamp uses: floor(budget / bytes-per-row-slot).
        XCTAssertEqual(KVCacheProvisioning.affordableSeqLen(geometry: geometry, budgetGB: 1.0),
                       Int(1_073_741_824 / (36 * 4096)))
        XCTAssertEqual(KVCacheProvisioning.affordableSeqLen(geometry: geometry, budgetGB: 0), 0)
        // A non-finite budget means unconstrained, and must not trap the Int cast.
        XCTAssertEqual(KVCacheProvisioning.affordableSeqLen(geometry: geometry, budgetGB: .infinity), Int.max)
    }

    func testAffordableSeqLenPricesSlidingRingsAsFixedRows() {
        // Gemma-4 shape: 30 slots, 25 of them sliding rings of 1,536 rows
        // (window 1,024 + 512-row chunk slack), fp8 KV, 0.25 GB budget — the
        // 16 GB field machine. The ring rows are a fixed cost; charging every
        // slot as linear (the old fit check) under-reported the affordable rows
        // by ~2.5x and rejected prompts the allocation could hold.
        let flags: [Bool] = (0..<30).map { $0 < 25 }
        let ringGeometry = KVCacheProvisioning.Geometry(
            totalSlots: 30,
            slidingSlotFlags: flags,
            slidingRingLen: 1_536,
            kStride: 2048,
            vStride: 2048,
            elementBytes: 1
        )
        let uniformGeometry = KVCacheProvisioning.Geometry(
            totalSlots: 30,
            slidingSlotFlags: nil,
            slidingRingLen: 0,
            kStride: 2048,
            vStride: 2048,
            elementBytes: 1
        )
        XCTAssertEqual(KVCacheProvisioning.affordableSeqLen(geometry: ringGeometry, budgetGB: 0.25), 5_427)
        XCTAssertEqual(KVCacheProvisioning.affordableSeqLen(geometry: uniformGeometry, budgetGB: 0.25), 2_184)
        XCTAssertEqual(
            KVCacheProvisioning.requiredBytes(maxSeqLen: 8_192, geometry: ringGeometry),
            (25 * 1_536 + 5 * 8_192) * 4_096,
            "the ring layout must charge sliding slots their fixed rows and linear slots the sequence length"
        )
    }

    // MARK: - Rejection message

    private func message(
        promptTokens: Int = 200,
        provisioned: Int = 910,
        affordableRows: Int = 910,
        precision: KVCachePrecision = .fp16,
        isAgentSession: Bool = true,
        sessionFloor: Int = 8_192,
        budgetGB: Double = 0.25
    ) -> String {
        KVCacheProvisioning.capacityMessage(
            promptTokens: promptTokens,
            provisioned: provisioned,
            affordableRows: affordableRows,
            budgetGB: budgetGB,
            precision: precision,
            isAgentSession: isAgentSession,
            sessionFloor: sessionFloor,
            modelFootprintGB: 8.75,
            physicalGB: 16
        )
    }

    func testCapacityMessageBlamesTheBindNotThePromptLength() {
        // The 200-token prompt fits the rows the budget affords; what it runs out
        // of is generation room. The message must say that, and must report the
        // numbers the gate used.
        let tight = message(promptTokens: 200, provisioned: 910, affordableRows: 910, sessionFloor: 2_048)
        XCTAssertTrue(tight.contains("Context does not fit"), tight)
        XCTAssertFalse(tight.contains("Prompt too long"), tight)
        XCTAssertTrue(tight.contains("200 prompt tokens"), tight)
        XCTAssertTrue(tight.contains("910 provisioned KV rows"), tight)
        XCTAssertTrue(tight.contains("FP16"), tight)
        XCTAssertTrue(tight.contains("0.25 GB"), tight)
        XCTAssertTrue(tight.contains("affords 910 rows"), tight)

        // A prompt past the provision is reported against it, not as a mystery.
        let oversized = message(promptTokens: 5_000, provisioned: 910)
        XCTAssertTrue(oversized.contains("Prompt does not fit"), oversized)
        XCTAssertTrue(oversized.contains("5000 prompt tokens against 910 provisioned KV rows"), oversized)
    }

    func testCapacityMessageNamesACappedReserve() {
        // The agent floor asked for 8,192 rows and the budget capped it to 910;
        // the message must not present 8,192 as what was provisioned.
        let capped = message(affordableRows: 910, sessionFloor: 8_192)
        XCTAssertTrue(capped.contains("agent sessions reserve 8192 rows, capped by the budget to 910"), capped)
        XCTAssertTrue(capped.contains("dynamoe_agent_session_floor"), capped)

        let roomy = message(provisioned: 9_000, affordableRows: 9_000, sessionFloor: 8_192)
        XCTAssertTrue(roomy.contains("agent sessions reserve 8192 rows;"), roomy)
        XCTAssertFalse(roomy.contains("capped by the budget"), roomy)
    }

    func testCapacityMessageRendersUnboundedCapacityAndChatRemedy() {
        // All-ring layouts report Int.max affordability; the message must not
        // print 9223372036854775807 at a user.
        let unbounded = message(affordableRows: Int.max, sessionFloor: 2_048)
        XCTAssertTrue(unbounded.contains("affords unbounded rows"), unbounded)
        XCTAssertFalse(unbounded.contains("\(Int.max)"), unbounded)

        let chat = message(provisioned: 910, affordableRows: 910, isAgentSession: false, sessionFloor: 2_048)
        XCTAssertTrue(chat.contains("chat turns reserve 2048 rows"), chat)
        XCTAssertTrue(chat.contains("Shorten the prompt"), chat)
        XCTAssertFalse(chat.contains("dynamoe_agent_session_floor"), chat)
    }

    // MARK: - Machine/model/mode budget

    func testBudgetReservesWeightsAndWorkingSet() {
        // balanced16GB target 11.5 GB, 7.7 GB model, 2.5 GB reserve.
        let budget = KVCacheProvisioning.budgetGB(physicalGB: 16, modelFootprintGB: 7.7, budgetMode: .balanced16GB)
        XCTAssertEqual(budget, 11.5 - 7.7 - 2.5, accuracy: 0.0001)
    }

    func testUnrestrictedBudgetFallsBackToPhysicalRAMCeiling() {
        // Auto mode resolves a small model on a roomy machine to "Full RAM"
        // (unrestricted), which used to mean no KV ceiling at all.
        let budget = KVCacheProvisioning.budgetGB(physicalGB: 16, modelFootprintGB: 7.7, budgetMode: .unrestricted)
        XCTAssertEqual(budget, 16 * 0.75 - 7.7 - 2.5, accuracy: 0.0001)

        // A big machine with a big model still gets a large but bounded budget.
        let big = KVCacheProvisioning.budgetGB(physicalGB: 64, modelFootprintGB: 37, budgetMode: .unrestricted)
        XCTAssertGreaterThan(big, 8)
    }

    func testBudgetNeverGoesNegative() {
        let budget = KVCacheProvisioning.budgetGB(physicalGB: 16, modelFootprintGB: 30, budgetMode: .balanced16GB)
        XCTAssertEqual(budget, 0.25)
    }

    // MARK: - Geometry derivation

    func testGeometryMirrorsTheAllocationLayout() {
        // A nil config pads the KV head count to 8 (the allocation's capacity pad),
        // so the stride is 8 × headDim rather than the raw 4.
        let uniform = KVCacheProvisioning.geometry(
            config: nil,
            actualLayers: 36,
            totalLoops: 1,
            numKvHeads: 4,
            headDim: 256,
            precision: .fp16,
            slidingSlotFlags: nil,
            slidingRingLen: 0
        )
        XCTAssertEqual(uniform.totalSlots, 36, "a nil config (non-Ling) uses actualLayers × loops")
        XCTAssertEqual(uniform.kStride, max(8 * 256, 1024))
        XCTAssertEqual(uniform.vStride, max(8 * 256, 1024))
        XCTAssertEqual(uniform.elementBytes, 2)
        XCTAssertEqual(uniform.kvHeadCapacity, 8)

        // Loops multiply the slot count; FP8 halves the element size.
        let looped = KVCacheProvisioning.geometry(
            config: nil,
            actualLayers: 36,
            totalLoops: 2,
            numKvHeads: 8,
            headDim: 256,
            precision: .fp8,
            slidingSlotFlags: nil,
            slidingRingLen: 0
        )
        XCTAssertEqual(looped.totalSlots, 72)
        XCTAssertEqual(looped.elementBytes, 1)
    }
}
