//
//  SubagentApprovalCenterTests.swift
//  DynaMoETests
//
//  The operator's half of a run-scoped approval policy: the Turbo bypass
//  rules and the timeout resolution, as pure functions, so no test flips
//  the real Turbo preference (the switch has exactly one owner) and no card
//  parks a live continuation.
//

import XCTest
@testable import DynaMoE

@MainActor
final class SubagentApprovalCenterTests: XCTestCase {

    private func request(
        tool: String,
        arguments: [String: String] = [:],
        containmentEscape: Bool = false
    ) -> AgentApprovalRequest {
        AgentApprovalRequest(
            runId: UUID(),
            toolName: tool,
            arguments: arguments,
            workingDirectory: nil,
            reason: "test",
            containmentEscape: containmentEscape
        )
    }

    // MARK: - Turbo bypass rules

    func testTurboBypassesARoutineWriteInsideTheSandbox() {
        XCTAssertTrue(SubagentApprovalCenter.turboBypasses(
            request: request(tool: "file_write"),
            honorsTurbo: true,
            turboEnabled: true
        ))
    }

    func testTurboNeverBypassesLandingARunsWork() {
        XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
            request: request(tool: "apply_subagent_changes"),
            honorsTurbo: true,
            turboEnabled: true
        ), "every apply gets its own card, whatever the switches say")
    }

    func testTurboNeverBypassesAContainmentEscape() {
        // The exact shape of the exploit the card exists for: an out-of-workspace
        // write or a `.git` pointer rewrite in the configuration presented as
        // safe. Turbo is a switch over ROUTINE writes; this one is not routine.
        XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
            request: request(tool: "file_write", containmentEscape: true),
            honorsTurbo: true,
            turboEnabled: true
        ))
        XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
            request: request(tool: "shell_run", containmentEscape: true),
            honorsTurbo: true,
            turboEnabled: true
        ))
    }

    func testTurboNeverBypassesAnAppPreferenceWrite() {
        XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
            request: request(
                tool: "shell_run",
                arguments: ["command": "defaults write DRP.DynaMoE dynamoe_agent_turbo_mode -bool NO"]
            ),
            honorsTurbo: true,
            turboEnabled: true
        ))
    }

    func testTheBypassNeedsBothTheCallerAndTheSwitch() {
        XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
            request: request(tool: "file_write"),
            honorsTurbo: false,
            turboEnabled: true
        ), "a caller that opted out (the cost refusal) never bypasses")
        XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
            request: request(tool: "file_write"),
            honorsTurbo: true,
            turboEnabled: false
        ), "Turbo off means every card is real")
    }

    // MARK: - Timeout resolution

    private func freshDefaults() -> UserDefaults {
        UserDefaults.standard.removePersistentDomain(forName: "dynamoe-tests-approval-timeout")
        let defaults = UserDefaults(suiteName: "dynamoe-tests-approval-timeout")!
        addTeardownBlock {
            defaults.removePersistentDomain(forName: "dynamoe-tests-approval-timeout")
        }
        return defaults
    }

    func testTheApprovalTimeoutDefaultsToTenMinutes() {
        XCTAssertEqual(
            SubagentApprovalCenter.resolvedApprovalTimeout(defaults: freshDefaults()),
            600
        )
    }

    func testAStoredTimeoutIsHonouredAndABrokenOneIsNot() {
        let defaults = freshDefaults()
        defaults.set(30.0, forKey: SubagentApprovalCenter.approvalTimeoutKey)
        XCTAssertEqual(SubagentApprovalCenter.resolvedApprovalTimeout(defaults: defaults), 30)

        defaults.set(0, forKey: SubagentApprovalCenter.approvalTimeoutKey)
        XCTAssertEqual(
            SubagentApprovalCenter.resolvedApprovalTimeout(defaults: defaults),
            SubagentApprovalCenter.defaultApprovalTimeout,
            "a zero timeout must fall back, not resolve instantly"
        )
        defaults.set(-5.0, forKey: SubagentApprovalCenter.approvalTimeoutKey)
        XCTAssertEqual(
            SubagentApprovalCenter.resolvedApprovalTimeout(defaults: defaults),
            SubagentApprovalCenter.defaultApprovalTimeout
        )
    }
}
