//
//  MemorySettingsPersistenceTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the in-app memory settings the local server reads:
//  what the UI persists must round-trip into a turn's TurnMemorySettings, with
//  sane fallbacks for unset or unrecognized values.
//

import XCTest
@testable import DynaMoE

final class MemorySettingsPersistenceTests: XCTestCase {

    private func makeDefaults(_ values: [String: String] = [:]) -> UserDefaults {
        let suite = "dynamoe_memory_settings_test_\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        for (key, value) in values {
            defaults.set(value, forKey: key)
        }
        return defaults
    }

    func testEmptyDefaultsFallBackToUIStartingValues() {
        let memory = TurnMemorySettings.persisted(makeDefaults())
        XCTAssertEqual(memory.executionMode, .autoDetect)
        XCTAssertEqual(memory.budgetMode, .balanced16GB)
        XCTAssertEqual(memory.kvPrecision, .fp16)
    }

    func testPersistedSelectionsRoundTrip() {
        // The "Spark + Unrestricted + FP8 KV in-app" configuration from the
        // 16 GB tuning notes must survive the trip into a server turn.
        let memory = TurnMemorySettings.persisted(makeDefaults([
            "dynamoe_memory_execution_mode": MemoryExecutionMode.residentRAM.rawValue,
            "dynamoe_memory_budget_mode": MemoryBudgetMode.unrestricted.rawValue,
            "dynamoe_kv_cache_precision": KVCachePrecision.fp8.rawValue
        ]))
        XCTAssertEqual(memory.executionMode, .residentRAM)
        XCTAssertEqual(memory.budgetMode, .unrestricted)
        XCTAssertEqual(memory.kvPrecision, .fp8)
    }

    func testUnrecognizedRawValuesFallBackInsteadOfCrashing() {
        let memory = TurnMemorySettings.persisted(makeDefaults([
            "dynamoe_memory_execution_mode": "Turbo Mode",
            "dynamoe_memory_budget_mode": "999 GB",
            "dynamoe_kv_cache_precision": "int4"
        ]))
        XCTAssertEqual(memory.executionMode, .autoDetect)
        XCTAssertEqual(memory.budgetMode, .balanced16GB)
        XCTAssertEqual(memory.kvPrecision, .fp16)
    }
}
