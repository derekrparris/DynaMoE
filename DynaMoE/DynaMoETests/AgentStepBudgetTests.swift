//
//  AgentStepBudgetTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the Max Multi-Step Agent Iterations budget: the
//  stored switch-and-number pair, the finite floor, the step comparison both
//  agent loops share, and the run-end status lines. No model, no provider, no
//  turn.
//

import XCTest
@testable import DynaMoE

final class AgentStepBudgetTests: XCTestCase {

    func testUnlimitedResolvesToNoCap() {
        XCTAssertNil(AgentStepBudget.resolvedCap(unlimited: true, configured: 15))
        XCTAssertNil(AgentStepBudget.resolvedCap(unlimited: true, configured: 100))
    }

    func testFiniteValuePassesThroughUnchanged() {
        XCTAssertEqual(AgentStepBudget.resolvedCap(unlimited: false, configured: 15), 15)
        XCTAssertEqual(AgentStepBudget.resolvedCap(unlimited: false, configured: 1), 1)
    }

    func testANonPositiveStoredValueFloorsAtOneStep() {
        // `defaults write … -int 0` must not produce a run that ends before the
        // model can call a tool.
        XCTAssertEqual(AgentStepBudget.resolvedCap(unlimited: false, configured: 0), 1)
        XCTAssertEqual(AgentStepBudget.resolvedCap(unlimited: false, configured: -3), 1)
    }

    func testAllowsAnotherStepMatchesTheInlineArithmeticItReplaced() {
        // The loops computed `step + 1 < cap`; a cap of 3 allows steps 0 and 1.
        let cap: Int? = 3
        XCTAssertTrue(AgentStepBudget.allowsAnotherStep(step: 0, cap: cap))
        XCTAssertTrue(AgentStepBudget.allowsAnotherStep(step: 1, cap: cap))
        XCTAssertFalse(AgentStepBudget.allowsAnotherStep(step: 2, cap: cap))
    }

    func testAnIntMinCapIsExhaustedInsteadOfTrapping() {
        // The comparison subtracts one from the cap, which would overflow for
        // `Int.min`. A cap that small allows no step anyway.
        XCTAssertFalse(AgentStepBudget.allowsAnotherStep(step: 0, cap: Int.min))
        XCTAssertTrue(AgentStepBudget.isExhausted(step: 0, cap: Int.min))
    }

    func testIsExhaustedIsTheComplementOfAllowsAnotherStep() {
        for cap in [1, 3, 100, nil] as [Int?] {
            for step in 0...5 {
                XCTAssertNotEqual(
                    AgentStepBudget.allowsAnotherStep(step: step, cap: cap),
                    AgentStepBudget.isExhausted(step: step, cap: cap),
                    "cap \(String(describing: cap)) step \(step)"
                )
            }
        }
    }

    func testNoCapNeverExhausts() {
        for step in [0, 1, 99, 10_000, 1_000_000] {
            XCTAssertTrue(AgentStepBudget.allowsAnotherStep(step: step, cap: nil), "step \(step)")
            XCTAssertFalse(AgentStepBudget.isExhausted(step: step, cap: nil), "step \(step)")
        }
    }

    func testCompareCapBridgesNoLimitForTheLegacyIntPolicy() {
        XCTAssertEqual(AgentStepBudget.compareCap(nil), Int.max)
        XCTAssertEqual(AgentStepBudget.compareCap(15), 15)

        // The bridge is only sound if the old policy agrees there is room far
        // past any real iteration count.
        XCTAssertTrue(CloudAgentLoopPolicy.shouldContinue(
            stepCallCount: 1,
            step: 10_000,
            maxSteps: AgentStepBudget.compareCap(nil),
            ranComplete: false,
            failureLoop: false,
            cancelled: false
        ))
    }

    func testALegacyIntCapStillStopsAtItsBudget() {
        XCTAssertFalse(CloudAgentLoopPolicy.shouldContinue(
            stepCallCount: 1,
            step: 2,
            maxSteps: AgentStepBudget.compareCap(3),
            ranComplete: false,
            failureLoop: false,
            cancelled: false
        ))
    }

    /// The app resolves the budget from UserDefaults at each comparison (the
    /// loops run off the main actor), so the stored pair is what actually
    /// decides a run. A temporary suite keeps the developer's own settings out
    /// of it.
    func testResolvedFromDefaultsHonoursTheStoredPair() {
        let suiteName = "AgentStepBudgetTests.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            return XCTFail("could not open a scratch defaults suite")
        }
        defer { defaults.removePersistentDomain(forName: suiteName) }

        // Absent keys resolve to the shipped default, not to nil.
        XCTAssertEqual(AgentStepBudget.resolvedFromDefaults(defaults), AgentStepBudget.defaultFiniteSteps)

        defaults.set(42, forKey: AgentStepBudget.finiteStepsDefaultsKey)
        XCTAssertEqual(AgentStepBudget.resolvedFromDefaults(defaults), 42)

        // The switch wins over whatever number is retained beside it, and the
        // number survives so turning the switch off restores it.
        defaults.set(true, forKey: AgentStepBudget.unlimitedDefaultsKey)
        XCTAssertNil(AgentStepBudget.resolvedFromDefaults(defaults))

        defaults.set(false, forKey: AgentStepBudget.unlimitedDefaultsKey)
        XCTAssertEqual(AgentStepBudget.resolvedFromDefaults(defaults), 42)

        // A zero written by hand still floors at one step rather than
        // producing a run that ends before its first tool call.
        defaults.set(0, forKey: AgentStepBudget.finiteStepsDefaultsKey)
        XCTAssertEqual(AgentStepBudget.resolvedFromDefaults(defaults), 1)
    }

    func testStatusLabelPrintsSteps() {
        XCTAssertEqual(AgentStepBudget.statusLabel(cap: 15), "15 steps")
        XCTAssertEqual(AgentStepBudget.statusLabel(cap: 1), "1 step")
    }

    func testStepBudgetNoticeNamesTheCapThatEndedTheRun() {
        let finite = AgentStepBudget.maxStepsReachedNotice(cap: 15)
        XCTAssertTrue(finite.contains("15"), finite)
        XCTAssertFalse(AgentStepBudget.loopGuardStoppedNotice.isEmpty)
        XCTAssertNotEqual(AgentStepBudget.loopGuardStoppedNotice, finite)
    }

    func testRunEndNoticeNamesOnlyTheReasonThatEndedTheRun() {
        XCTAssertEqual(
            AgentStepBudget.runEndNotice(anyCompleted: true, loopGuardTripped: false, cap: 15),
            AgentStepBudget.completedNotice
        )
        // Completion outranks a guard that also tripped during the run.
        XCTAssertEqual(
            AgentStepBudget.runEndNotice(anyCompleted: true, loopGuardTripped: true, cap: nil),
            AgentStepBudget.completedNotice
        )
        XCTAssertEqual(
            AgentStepBudget.runEndNotice(anyCompleted: false, loopGuardTripped: true, cap: 15),
            AgentStepBudget.loopGuardStoppedNotice
        )
        XCTAssertEqual(
            AgentStepBudget.runEndNotice(anyCompleted: false, loopGuardTripped: false, cap: 15),
            AgentStepBudget.maxStepsReachedNotice(cap: 15)
        )

        // The uncapped, unfinished, guard-free state cannot be reached from the
        // loop, because a nil cap always allows another step. It reports no
        // notice rather than naming a limit that never applied.
        XCTAssertNil(AgentStepBudget.runEndNotice(anyCompleted: false, loopGuardTripped: false, cap: nil))
    }
}
