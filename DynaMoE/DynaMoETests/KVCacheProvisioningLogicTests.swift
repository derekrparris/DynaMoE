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
