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

    /// Both halves of containment are escape cards. The read that TAINTS a run
    /// and the egress that carries the taint out were created with the default
    /// `false`, so Turbo approved them silently — the one path a read of
    /// `~/.ssh/id_ed25519` had out of the sandbox.
    func testTurboNeverBypassesAnOutOfWorkspaceReadOrItsEgress() {
        for tool in ["file_read", "lint_diagnostics", "web_fetch", "subq_decide"] {
            XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
                request: request(tool: tool, containmentEscape: true),
                honorsTurbo: true,
                turboEnabled: true
            ), "\(tool) escaped containment, so its card is not routine")
        }
    }

    /// `subq_decide` uploads local image bytes whatever the endpoint is, and
    /// the approval request only keeps the STRING form of the arguments — where
    /// an `images` array is a description. The wire form is what keeps that
    /// card alive under Turbo.
    func testTurboNeverBypassesTheSubQImageUploadCard() {
        let wire = #"{"state":"is this red?","questions":{},"images":[{"path":"shot.png"}]}"#
        let displayOnly = AgentApprovalRequest(
            runId: UUID(),
            toolName: "subq_decide",
            arguments: ["state": "is this red?", "images": "[(path: shot.png)]"],
            workingDirectory: nil,
            reason: "test",
            rawArguments: wire
        )
        XCTAssertFalse(SubagentApprovalCenter.turboBypasses(
            request: displayOnly,
            honorsTurbo: true,
            turboEnabled: true
        ), "an image upload is not a routine write")

        // The decoded form answers the same question on the turn path.
        XCTAssertFalse(AgentHarness.turboModeMayBypassApproval(
            toolName: "subq_decide",
            arguments: ["state": "x", "images": [["path": "shot.png"]]],
            mode: .code,
            subQEndpoint: URL(string: "https://api.subq.ai/v1/systemone")
        ))
        // A text-only call against SubQ's own origin stays bypassable.
        XCTAssertTrue(AgentHarness.turboModeMayBypassApproval(
            toolName: "subq_decide",
            arguments: ["state": "x"],
            mode: .code,
            subQEndpoint: URL(string: "https://api.subq.ai/v1/systemone")
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

    // MARK: - The round trip

    /// A card nobody answers must resolve as REFUSED after the timeout, and it
    /// must leave the pending list: an unanswered card that parked its run
    /// forever is exactly what the bound exists to prevent, and nothing pinned
    /// that the continuation ever came back.
    func testAnUnansweredCardTimesOutAsRejectedAndLeavesTheQueue() async {
        // A containment-escape card, because Turbo Mode is a real operator
        // preference this test must not depend on: every routine write is
        // bypassed when it is on, and this suite must not flip it.
        let card = request(tool: "file_write", containmentEscape: true)
        let approved = await SubagentApprovalCenter.shared.request(card, timeout: 0.05)
        XCTAssertFalse(approved, "an unanswered card is a refusal")
        XCTAssertFalse(
            SubagentApprovalCenter.shared.pending.contains { $0.request.runId == card.runId },
            "the timed-out card must not stay in the operator's queue"
        )
    }

    /// The other half: an answered card resolves true, and the queue drains.
    func testAnAnsweredCardResolvesApproved() async throws {
        let card = request(tool: "file_write", containmentEscape: true)
        let answer = Task { await SubagentApprovalCenter.shared.request(card, timeout: 5) }
        // Wait for the card to appear rather than sleeping a fixed time.
        var entryId: UUID?
        for _ in 0..<200 {
            if let entry = SubagentApprovalCenter.shared.pending.first(where: { $0.request.runId == card.runId }) {
                entryId = entry.id
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let id = try XCTUnwrap(entryId, "the card must reach the operator")
        SubagentApprovalCenter.shared.approve(id)
        let approved = await answer.value
        XCTAssertTrue(approved)
        XCTAssertTrue(SubagentApprovalCenter.shared.pending.isEmpty, "an answered card is removed")
    }

    /// The queue is bounded: a model that loops on a refused tool must not be
    /// able to grow the operator's list without limit.
    func testThePendingQueueIsBounded() {
        XCTAssertEqual(SubagentApprovalCenter.maximumPendingCards, 32)
        XCTAssertFalse(SubagentApprovalCenter.refusesCard(pendingCount: 0))
        XCTAssertFalse(SubagentApprovalCenter.refusesCard(pendingCount: 31))
        XCTAssertTrue(SubagentApprovalCenter.refusesCard(pendingCount: 32))
        XCTAssertTrue(SubagentApprovalCenter.refusesCard(pendingCount: 33))
    }
}
