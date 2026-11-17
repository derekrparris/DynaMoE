//
//  TurboModeGuardTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for the two halves of the "Turbo switched itself to Safe
//  Mode" fix: the key's single owner, and the rule that a tool call can never
//  silently rewrite the preferences domain that holds the switch.
//
//  No model, no GPU, and no writes to `UserDefaults.standard`: the operator's
//  real preferences domain is shared with these hosted tests.
//

import XCTest
@testable import DynaMoE

final class TurboModeGuardTests: XCTestCase {

    // MARK: - The key

    func testTheKeySpellingIsPinned() {
        // The UI still binds `@AppStorage("dynamoe_agent_turbo_mode")`
        // literally. Renaming the constant without renaming those bindings
        // would read the operator's setting as absent and drop them to Safe
        // Mode on the next launch.
        XCTAssertEqual(TurboModeSettings.key, "dynamoe_agent_turbo_mode")
    }

    // MARK: - Commands that rewrite this app's preferences

    func testWritingThisApplicationsDefaultsIsRecognized() {
        let commands = [
            "defaults write DRP.DynaMoE dynamoe_agent_turbo_mode -bool NO",
            "defaults write \"DRP.DynaMoE\" dynamoe_agent_turbo_mode -bool false",
            "defaults delete DRP.DynaMoE dynamoe_agent_turbo_mode",
            "defaults remove drp.dynamoe dynamoe_agent_turbo_mode",
            "defaults -currentHost write DRP.DynaMoE something 1",
            "bash -lc 'defaults write DRP.DynaMoE dynamoe_agent_turbo_mode -bool NO'"
        ]
        for command in commands {
            XCTAssertTrue(
                AgentHarness.shellCommandWritesApplicationPreferences(command),
                "should be refused without approval: \(command)"
            )
        }
    }

    func testTouchingThePreferencePlistIsRecognized() {
        let commands = [
            "rm ~/Library/Preferences/DRP.DynaMoE.plist",
            "cp /tmp/evil.plist ~/Library/Preferences/DRP.DynaMoE.plist",
            "plutil -replace dynamoe_agent_turbo_mode -bool NO ~/Library/Preferences/DRP.DynaMoE.plist",
            "echo x > ~/Library/Preferences/DRP.DynaMoE.plist"
        ]
        for command in commands {
            XCTAssertTrue(
                AgentHarness.shellCommandWritesApplicationPreferences(command),
                "should be refused without approval: \(command)"
            )
        }
    }

    func testKillingThePreferencesDaemonIsRecognized() {
        XCTAssertTrue(AgentHarness.shellCommandWritesApplicationPreferences("killall cfprefsd"))
        XCTAssertTrue(AgentHarness.shellCommandWritesApplicationPreferences("pkill -9 cfprefsd"))
    }

    func testReadingThisApplicationsDefaultsStaysAllowed() {
        let commands = [
            "defaults read DRP.DynaMoE",
            "defaults read DRP.DynaMoE dynamoe_agent_turbo_mode",
            "defaults read-type DRP.DynaMoE dynamoe_agent_turbo_mode"
        ]
        for command in commands {
            XCTAssertFalse(
                AgentHarness.shellCommandWritesApplicationPreferences(command),
                "a read must not be gated: \(command)"
            )
        }
    }

    func testAnotherApplicationsDefaultsStayAllowed() {
        let commands = [
            "defaults write com.apple.dock autohide -bool true",
            "defaults write com.example.app key -bool NO",
            "defaults delete com.example.app key"
        ]
        for command in commands {
            XCTAssertFalse(
                AgentHarness.shellCommandWritesApplicationPreferences(command),
                "editing another app's preferences is legitimate: \(command)"
            )
        }
    }

    func testAnUnrelatedCommandMentioningTheBundleIsAllowed() {
        XCTAssertFalse(AgentHarness.shellCommandWritesApplicationPreferences("grep -r DRP.DynaMoE ."))
        XCTAssertFalse(AgentHarness.shellCommandWritesApplicationPreferences("ls ~/Library/Preferences"))
    }

    // MARK: - File tools

    func testWritingThePreferencePlistWithAFileToolIsRecognized() {
        XCTAssertTrue(AgentHarness.pathWritesApplicationPreferences(
            "~/Library/Preferences/DRP.DynaMoE.plist"
        ))
        XCTAssertTrue(AgentHarness.pathWritesApplicationPreferences(
            "/Users/someone/Library/Preferences/DRP.DynaMoE.plist"
        ))
        XCTAssertFalse(AgentHarness.pathWritesApplicationPreferences(
            "~/Library/Preferences/com.example.app.plist"
        ))
        XCTAssertFalse(AgentHarness.pathWritesApplicationPreferences("/tmp/DRP.DynaMoE.plist"))
    }

    // MARK: - The gate

    func testTurboCannotBypassAWriteToTheAppDomain() {
        let turboKey = ToolScalar.command("defaults write DRP.DynaMoE dynamoe_agent_turbo_mode -bool NO")
        XCTAssertFalse(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "shell_run",
                arguments: turboKey,
                mode: .code
            ),
            "Turbo must not wave through the command that turns Turbo off"
        )
    }

    func testTurboStillBypassesOrdinaryWrites() throws {
        XCTAssertTrue(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "shell_run",
                arguments: ToolScalar.command("swift build"),
                mode: .code
            )
        )
        XCTAssertTrue(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "file_write",
                arguments: ToolScalar.path("Sources/main.swift"),
                mode: .code
            )
        )
    }

    func testTurboCannotBypassAPlistWriteThroughAFileTool() {
        XCTAssertFalse(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "file_write",
                arguments: ToolScalar.path("~/Library/Preferences/DRP.DynaMoE.plist"),
                mode: .code
            )
        )
        XCTAssertFalse(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "file_edit",
                arguments: ToolScalar.path("~/Library/Preferences/DRP.DynaMoE.plist"),
                mode: .code
            )
        )
    }

    // MARK: - Approval continuation lifecycle

    func testADecisionThatArrivesBeforeItsWaiterStillResolves() async {
        let id = UUID()
        // A click can land before the tool loop has parked its continuation.
        // The manager must remember the answer rather than lose it.
        XCTAssertTrue(ActionApprovalManager.shared.approve(id: id))
        let approved = await ActionApprovalManager.shared.waitForApproval(id: id)
        XCTAssertTrue(approved)
    }

    func testARejectionThatArrivesBeforeItsWaiterStillResolves() async {
        let id = UUID()
        XCTAssertTrue(ActionApprovalManager.shared.reject(id: id))
        let approved = await ActionApprovalManager.shared.waitForApproval(id: id)
        XCTAssertFalse(approved)
    }

    func testACancelledWaitResolvesAsRejectedInsteadOfHanging() async {
        let id = UUID()
        let waiter = Task { await ActionApprovalManager.shared.waitForApproval(id: id) }
        // Let the waiter park before cancelling, so the cancellation handler
        // takes the "resume the parked continuation" branch. The other branch
        // (cancel before registration) is covered by the pre-registration test
        // shape below.
        try? await Task.sleep(nanoseconds: 50_000_000)
        waiter.cancel()
        let approved = await waiter.value
        XCTAssertFalse(approved, "a stopped turn must not leave the call approved")
    }

    func testACancellationBeforeRegistrationIsNotLost() async {
        let id = UUID()
        // Cancel first: the waiter must observe `cancelledWaits` on arrival
        // instead of parking on a continuation nobody will resume.
        let waiter = Task { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            return await ActionApprovalManager.shared.waitForApproval(id: id)
        }
        let approved = await waiter.value
        XCTAssertFalse(approved)
    }
}

/// Argument-dictionary shorthands. The gate takes `[String: Any]`, so a test
/// that spells the key wrong would pass by silently falling through the
/// `guard let ... as? String` and never exercise the rule it claims to.
private enum ToolScalar {
    static func command(_ value: String) -> [String: Any] { ["command": value] }
    static func path(_ value: String) -> [String: Any] { ["path": value] }
}
