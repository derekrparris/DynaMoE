//
//  MemoryTelemetryLogicTests.swift
//  DynaMoE
//
//  Pins the memory-telemetry contract: the user-facing total is the Darwin
//  footprint plus the cached weight pages, while the eviction-budget input stays
//  on resident size (RSIZE). No Metal, no model files.
//

import XCTest
@testable import DynaMoE

final class MemoryTelemetryLogicTests: XCTestCase {

    override func tearDown() {
        let manager = WorkingSetManager.shared
        manager.footprintProviderOverride = nil
        manager.residentMemoryProviderOverride = nil
        manager.releaseShardMappings()
        super.tearDown()
    }

    func testAttributedMemoryIsFootprintPlusCachedWeights() {
        let manager = WorkingSetManager.shared
        manager.footprintProviderOverride = { 3.5 }
        let cachedWeightsGB = manager.residentShardBytesGB

        XCTAssertEqual(
            manager.attributedMemoryGB,
            3.5 + cachedWeightsGB,
            accuracy: 0.0001,
            "the reported total must be the Darwin footprint plus the cached weight pages"
        )
    }

    func testReportedTotalIsNotTheResidentPageFigure() {
        let manager = WorkingSetManager.shared
        manager.footprintProviderOverride = { 3.5 }
        manager.residentMemoryProviderOverride = { 0.9 }
        let cachedWeightsGB = manager.residentShardBytesGB

        // The display metric deliberately differs from RSIZE: under swap pressure
        // RSIZE collapses (0.4 GB was observed against a 20.6 GB footprint) and is
        // useless as "how much is DynaMoE using".
        XCTAssertNotEqual(manager.attributedMemoryGB, manager.effectiveResidentMemoryGB)
        XCTAssertEqual(manager.attributedMemoryGB, 3.5 + cachedWeightsGB, accuracy: 0.0001)
    }

    func testEvictionBudgetInputStaysOnResidentSize() {
        let manager = WorkingSetManager.shared
        manager.residentMemoryProviderOverride = { 2.5 }
        manager.footprintProviderOverride = { 7.5 }

        XCTAssertEqual(
            manager.effectiveResidentMemoryGB,
            2.5,
            accuracy: 0.0001,
            "trimToBudget compares RSIZE against the mode target; it must not follow the display metric"
        )
    }
}
