//
//  PlanModeTests.swift
//  DynaMoE
//
//  The plan-mode contract, pinned without a model: the markers the UI reads, the
//  read-only tool surface, the prompt sections, the execution backstop, the
//  subagent inheritance and the per-conversation persistence.
//
//  These are pure-logic cases on purpose. Plan mode's promise is that a
//  withholding decision is made in ONE place and read in four, so the tests can
//  prove the guarantee without loading 475 GB of weights.
//

import XCTest
@testable import DynaMoE

final class PlanModeTests: XCTestCase {

    /// The harness is a process-wide singleton and other suites load tools into
    /// it, so each case starts from the coder surface. `tearDown` alone is not
    /// enough: a suite that ran before this one and failed part-way through can
    /// leave `grep_search` loaded, and the "leaving plan mode unloads it"
    /// assertion below would then be measuring that suite, not this one.
    override func setUp() {
        super.setUp()
        AgentHarness.shared.beginAgentSearchGuard()
        AgentHarness.shared.setActiveMode(.code)
        AgentHarness.shared.resetLoadedToolsToCore()
    }

    override func tearDown() {
        // The harness is a singleton: leave it on the coder surface even when a
        // test fails part-way through.
        AgentHarness.shared.beginAgentSearchGuard()
        AgentHarness.shared.setActiveMode(.code)
        AgentHarness.shared.resetLoadedToolsToCore()
        super.tearDown()
    }

    // MARK: - Markers

    func testPlanMarkersAreDetectedOnlyOnTheirOwnLine() {
        let ready = PlanMarkers.ready
        let start = PlanMarkers.start

        XCTAssertTrue(PlanMarkers.readyPresent(in: "# Plan\n\nSteps\n\n\(ready)"))
        XCTAssertTrue(PlanMarkers.readyPresent(in: "  \(ready)  "))
        XCTAssertTrue(PlanMarkers.readyPresent(in: "`\(ready)`"))
        XCTAssertFalse(
            PlanMarkers.readyPresent(in: "I will end with \(ready) once the plan is done."),
            "a model narrating the convention has not finished the plan"
        )
        XCTAssertFalse(PlanMarkers.startPresent(in: "# Plan\n\n\(ready)"))
        XCTAssertTrue(PlanMarkers.startPresent(in: "\(start)\n\n# Plan"))
    }

    func testStrippingPlanMarkersLeavesProseAndRealCodeFencesIntact() {
        let text = """
        \(PlanMarkers.start)

        # Plan

        Read the file.

        ```go
        func main() {}
        ```

        \(PlanMarkers.ready)
        """
        let stripped = PlanMarkers.strip(text)
        XCTAssertFalse(stripped.contains(PlanMarkers.start))
        XCTAssertFalse(stripped.contains(PlanMarkers.ready))
        XCTAssertTrue(stripped.contains("func main() {}"), "a real code block keeps its fences and body")
        XCTAssertTrue(stripped.contains("```go"), "a real opening fence is not consumed")
        XCTAssertTrue(stripped.contains("# Plan"))

        // Prose that mentions a marker is content, not a marker line.
        let prose = "I will end with \(PlanMarkers.ready) when finished."
        XCTAssertEqual(PlanMarkers.strip(prose), prose)

        // An empty fence pair wrapped around a marker is removed with it.
        let fenced = "```\n\(PlanMarkers.ready)\n```\nAfter"
        XCTAssertEqual(PlanMarkers.strip(fenced), "After")
    }

    // MARK: - Read-only surface

    func testPlanModeWithholdsTheWriteExecuteAndCompletionTools() {
        AgentHarness.shared.setActiveMode(.plan)
        let planNames = Set(AgentHarness.shared.availableToolDefinitions(mode: .plan).map { $0.function.name })
        for withheld in ["file_write", "file_edit", "shell_run", "git_commit", "complete"] {
            XCTAssertFalse(planNames.contains(withheld), "\(withheld) must not be offered in plan mode")
        }
        // Entering the mode arms a small read core rather than the whole read
        // surface: the rest of `allowedToolNames` is still callable, but it
        // arrives through `tools_load` instead of riding in every plan prompt.
        for readTool in PlanModePolicy.coreToolNames {
            XCTAssertTrue(planNames.contains(readTool), "\(readTool) must be armed by plan mode")
        }
        XCTAssertFalse(
            planNames.contains("find_symbol_definition"),
            "a tool outside the core is not armed until the turn asks for it"
        )
        _ = try? AgentHarness.shared.loadTool(named: "find_symbol_definition")
        XCTAssertTrue(
            Set(AgentHarness.shared.availableToolDefinitions(mode: .plan).map { $0.function.name })
                .contains("find_symbol_definition"),
            "an allowed read tool stays one tools_load away"
        )
        XCTAssertTrue(PlanModePolicy.withholds("mcp__anything__do"), "MCP is withheld wholesale")
        XCTAssertFalse(PlanModePolicy.withholds("file_read"))

        // Entering the mode arms the read surface it needs; leaving it puts the
        // registry back so the coder prompt does not grow permanent schemas.
        AgentHarness.shared.setActiveMode(.code)
        let coderNames = Set(AgentHarness.shared.availableToolDefinitions(mode: .code).map { $0.function.name })
        XCTAssertFalse(coderNames.contains("grep_search"), "plan mode's armed tools are not left loaded")
    }

    func testCoderModeStillOffersTheWriteSurface() {
        AgentHarness.shared.setActiveMode(.code)
        let names = Set(AgentHarness.shared.availableToolDefinitions(mode: .code).map { $0.function.name })
        XCTAssertTrue(names.contains("shell_run"))
        XCTAssertTrue(names.contains("complete"))
        XCTAssertTrue(PlanModePolicy.withholds("mcp__server__tool"), "MCP is withheld unconditionally")
    }

    func testTheGrammarRegistryFollowsTheActiveMode() {
        let harness = AgentHarness.shared
        harness.setActiveMode(.code)
        XCTAssertTrue(
            GrammarConstrainedSampler.shared.snapshotRegistry().names.contains("shell_run"),
            "the coder surface is what the grammar accepts by default"
        )

        harness.setActiveMode(.plan)
        let planNames = GrammarConstrainedSampler.shared.snapshotRegistry().names
        XCTAssertFalse(planNames.contains("shell_run"))
        XCTAssertFalse(planNames.contains("complete"))
        XCTAssertTrue(planNames.contains("tools_discover"), "the read tools are still callable")

        harness.setActiveMode(.code)
        XCTAssertTrue(
            GrammarConstrainedSampler.shared.snapshotRegistry().names.contains("shell_run"),
            "leaving plan mode restores the coder registry"
        )
    }

    // MARK: - Prompt

    func testPlanPromptLeadsWithItsRoleAndAdvertisesOnlyReadTools() {
        AgentHarness.shared.setActiveMode(.plan)
        let prompt = AgentHarness.shared.buildSystemPrompt(
            baseSystem: "You are an expert software engineer and agentic coding assistant.",
            mode: .plan
        )

        XCTAssertTrue(prompt.contains("# Plan Mode"))
        XCTAssertTrue(prompt.contains("physically absent"))
        XCTAssertTrue(prompt.contains(PlanMarkers.start))
        XCTAssertTrue(prompt.contains(PlanMarkers.ready))
        XCTAssertTrue(prompt.contains("file_read"))
        XCTAssertFalse(prompt.contains("\"name\":\"file_write\""), "the tool block must not carry write schemas")
        XCTAssertFalse(prompt.contains("\"name\":\"shell_run\""))

        // Role before tools: the model reads what it is before what it can call.
        let role = prompt.range(of: "# Plan Mode")
        let tools = prompt.range(of: "# Tools")
        XCTAssertNotNil(role)
        XCTAssertNotNil(tools)
        if let role, let tools {
            XCTAssertLessThan(role.lowerBound, tools.lowerBound)
        }

        // The coder persona lands last, so the directive is restated after it.
        let persona = prompt.range(of: "agentic coding assistant")
        let override = prompt.range(of: "Plan Mode Directive (overrides the persona above)")
        XCTAssertNotNil(persona)
        XCTAssertNotNil(override)
        if let persona, let override {
            XCTAssertLessThan(persona.lowerBound, override.lowerBound)
        }
    }

    func testThePlanDirectiveNamesEveryWithheldToolItsPolicyDeclares() {
        let directive = PlanModePolicy.directive(toolNames: ["file_read"])
        // The bullet list is GENERATED from `withheldTools`, which is the only
        // place those names are written down. This pins the property the dead
        // constant never had: adding a tool to the list cannot leave the prompt
        // describing the old one, and the prompt cannot name a tool the policy
        // does not withhold.
        for tool in PlanModePolicy.withheldTools {
            XCTAssertTrue(
                directive.contains("`\(tool.name)`"),
                "the directive must name \(tool.name), with its purpose: \(tool.purpose)"
            )
        }
        XCTAssertTrue(directive.contains("every `mcp__*` tool"))

        XCTAssertTrue(
            PlanModePolicy.allowedToolNames.isDisjoint(with: PlanModePolicy.withheldToolNames),
            "a tool can be callable or named as absent, never both"
        )
        XCTAssertTrue(
            PlanModePolicy.coreToolNames.isSubset(of: PlanModePolicy.allowedToolNames),
            "the tools entering plan mode arms must be ones it may call"
        )
        for name in PlanModePolicy.withheldToolNames {
            XCTAssertTrue(PlanModePolicy.withholds(name), "\(name) is named as withheld and must be")
        }
    }

    func testRefreshingTheToolBlockKeepsPlanModeWithheldToolsOut() throws {
        let harness = AgentHarness.shared
        // Load a write tool on the coder surface, then rebuild the block as a
        // plan turn — the mid-run refresh is where a plan silently regained its
        // write tools before.
        harness.setActiveMode(.code)
        _ = try harness.loadTool(named: "file_write")

        let planBlock = harness.buildToolsSection(mode: .plan)
        XCTAssertFalse(planBlock.contains("\"name\":\"file_write\""))
        XCTAssertTrue(planBlock.contains("Plan mode withholds"))

        let prompt = harness.buildSystemPrompt(baseSystem: "persona", mode: .plan)
        let refreshed = harness.refreshingLoadedToolsSection(inPrompt: prompt, mode: .plan)
        XCTAssertFalse(
            refreshed.contains("\"name\":\"file_write\""),
            "a refresh must not re-advertise a tool the mode withholds"
        )
    }

    func testCloudGuidanceNamesTheWithheldTools() {
        let guidance = AgentHarness.shared.buildCloudToolGuidance(mode: .plan)
        XCTAssertTrue(guidance.contains("Plan mode withholds"))
        XCTAssertTrue(guidance.contains("file_write"))
    }

    // MARK: - Execution backstop

    func testExecuteToolRefusesAWithheldToolInPlanMode() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-plan-mode-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("written.txt")

        let call = ParsedToolCall(
            name: "file_write",
            arguments: ["path": target.path, "content": "should never be written"],
            rawArguments: "",
            rawText: "file_write"
        )
        let result = await AgentHarness.shared.executeTool(call: call, mode: .plan)

        XCTAssertEqual(result.record.status, .withheld)
        XCTAssertTrue(result.record.error?.contains("PLAN MODE") == true)
        XCTAssertFalse(result.isCompleted)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: target.path),
            "the refusal must happen before the tool runs, not after"
        )
        XCTAssertEqual(
            AgentHarness.shared.consecutiveFailedToolCalls, 0,
            "a policy refusal is the environment's answer, not a malformed call"
        )
    }

    /// Both turn loops used to ask `call.name == "complete"` directly, which made
    /// the withheld `complete` end the run — with no plan, no ready marker and no
    /// handoff, the exact dead end the plan backstop exists to prevent.
    func testAWithheldCompleteEndsNothing() {
        XCTAssertTrue(
            ToolExecutionStatus.success.endsAgentRun(toolName: "complete", isCompleted: true),
            "a complete call ends the run"
        )
        XCTAssertTrue(
            ToolExecutionStatus.success.endsAgentRun(toolName: "shell_run", isCompleted: true),
            "the harness's own loop guards end the run through isCompleted"
        )
        XCTAssertFalse(
            ToolExecutionStatus.withheld.endsAgentRun(toolName: "complete", isCompleted: false),
            "a withheld complete must let the turn finish with its plan"
        )
        XCTAssertFalse(
            ToolExecutionStatus.success.endsAgentRun(toolName: "file_read", isCompleted: false)
        )
    }

    func testAWithheldCallChargesTheTotalBudgetButNotTheStreak() {
        let harness = AgentHarness.shared
        harness.beginAgentSearchGuard()

        harness.recordToolCallOutcome(toolName: "file_write", outcome: .failed)
        harness.recordToolCallOutcome(toolName: "file_write", outcome: .failed)
        XCTAssertEqual(harness.consecutiveFailedToolCalls, 2)

        // A refusal must neither extend the streak (which would end a plan turn
        // for asking once too often) nor reset it (which would let a model
        // alternate real failures with refusals forever).
        harness.recordToolCallOutcome(toolName: "file_write", outcome: .withheld)
        XCTAssertEqual(harness.consecutiveFailedToolCalls, 2)
        XCTAssertEqual(harness.totalFailedToolCalls, 3)

        harness.recordToolCallOutcome(toolName: "file_write", outcome: .withheld)
        harness.recordToolCallOutcome(toolName: "file_write", outcome: .withheld)
        harness.recordToolCallOutcome(toolName: "file_write", outcome: .withheld)
        harness.recordToolCallOutcome(toolName: "file_write", outcome: .withheld)
        XCTAssertGreaterThanOrEqual(
            harness.totalFailedToolCalls, harness.failedToolCallTotalLimit,
            "the total budget is what bounds a loop the mode keeps refusing"
        )
        harness.beginAgentSearchGuard()
    }

    func testAWithheldStatusCountsAsItsOwnOutcome() {
        XCTAssertEqual(ToolExecutionStatus.error.toolCallOutcome, .failed)
        XCTAssertEqual(ToolExecutionStatus.withheld.toolCallOutcome, .withheld)
        XCTAssertEqual(
            ToolExecutionStatus.rejected.toolCallOutcome, .failed,
            "a call the user declined has always been charged as a failure"
        )
        XCTAssertEqual(ToolExecutionStatus.success.toolCallOutcome, .succeeded)
    }

    func testExecuteToolStillRunsAReadToolInPlanMode() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-plan-read-\(UUID().uuidString).txt")
        try "plan mode reads files".write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let call = ParsedToolCall(
            name: "file_read",
            arguments: ["path": url.path],
            rawArguments: "",
            rawText: "file_read"
        )
        let result = await AgentHarness.shared.executeTool(call: call, mode: .plan)
        XCTAssertEqual(result.record.status, .success)
        XCTAssertTrue(result.record.output?.contains("plan mode reads files") == true)
    }

    func testPlanModeCannotLoadAWithheldToolBackIn() throws {
        AgentHarness.shared.setActiveMode(.plan)
        XCTAssertThrowsError(try AgentHarness.shared.loadTool(named: "file_write")) { error in
            XCTAssertTrue(
                error.localizedDescription.contains("withheld in plan mode"),
                "the refusal must say why, got: \(error.localizedDescription)"
            )
        }
        XCTAssertNil(
            AgentHarness.shared.availableToolDefinitions(mode: .plan)
                .first { $0.function.name == "file_write" }
        )
    }

    // MARK: - Delegation

    @MainActor
    func testASubagentOfAReadOnlyParentInheritsTheWithholding() {
        let owner = SubagentInstance(
            role: "Researcher",
            taskDescription: "look around",
            allowedTools: ["file_read", "file_write", "shell_run"],
            parentMode: .plan
        )
        let executor = SubagentToolExecutor(owner: owner, workingDirectory: nil)
        XCTAssertTrue(executor.callableToolNames.contains("file_read"))
        XCTAssertFalse(executor.callableToolNames.contains("file_write"))
        XCTAssertFalse(executor.callableToolNames.contains("shell_run"))

        let coderOwner = SubagentInstance(
            role: "Researcher",
            taskDescription: "look around",
            allowedTools: ["file_read", "file_write", "shell_run"],
            parentMode: .code
        )
        let coderExecutor = SubagentToolExecutor(owner: coderOwner, workingDirectory: nil)
        XCTAssertTrue(coderExecutor.callableToolNames.contains("file_write"))
    }

    // MARK: - Slash command

    func testPlanCommandParsesAndDefaultsToEntering() {
        XCTAssertEqual(ChatCommand.parse("/plan"), .plan(enabled: true))
        XCTAssertEqual(ChatCommand.parse("/PLAN"), .plan(enabled: true))
        XCTAssertEqual(ChatCommand.parse("/plan on"), .plan(enabled: true))
        XCTAssertEqual(ChatCommand.parse("/plan off"), .plan(enabled: false))
        XCTAssertEqual(ChatCommand.parse("/plan exit"), .plan(enabled: false))
        XCTAssertNil(ChatCommand.parse("/planning"))
        // An argument that is neither a known on-spelling nor a known
        // off-spelling is not a command: `/plan off.` used to ENTER the mode,
        // the opposite of what was typed. Falling through to ordinary text
        // leaves the typo visible instead of acting on a guess.
        XCTAssertNil(ChatCommand.parse("/plan off."))
        XCTAssertNil(ChatCommand.parse("/plan please"))
        XCTAssertEqual(ChatCommand.names, ["/compact", "/plan"])
    }

    // MARK: - Persistence

    func testAWithheldRecordStaysReadableByABuildWithoutTheCase() throws {
        // `.withheld` postdates plan mode, and an unknown raw value fails the
        // decode of the WHOLE record — and with it the session file. It is
        // therefore persisted as a status every build knows, with the real
        // meaning in a companion key this build reads back.
        let withheld = ToolCallRecord(name: "file_write", status: .withheld, error: "PLAN MODE")
        let data = try JSONEncoder().encode(withheld)
        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertFalse(
            json.contains("\"status\":\"withheld\""),
            "an older build would lose the conversation rather than one record's nuance"
        )
        XCTAssertTrue(json.contains("\"withheldInPlanMode\":true"))
        XCTAssertEqual(try JSONDecoder().decode(ToolCallRecord.self, from: data).status, .withheld)

        // Every other status round-trips unchanged, and a plain rejection is not
        // read back as a mode refusal.
        for status in [ToolExecutionStatus.running, .success, .error, .awaitingApproval, .rejected] {
            let record = ToolCallRecord(name: "file_write", status: status)
            let roundTripped = try JSONDecoder().decode(
                ToolCallRecord.self,
                from: JSONEncoder().encode(record)
            )
            XCTAssertEqual(roundTripped.status, status)
        }
    }

    func testAgentModeSurvivesAPersistenceRoundTrip() throws {
        let session = ChatSession(title: "Plan a feature", agentMode: .plan)
        let data = try JSONEncoder().encode(session)
        let decoded = try JSONDecoder().decode(ChatSession.self, from: data)
        XCTAssertEqual(decoded.agentMode, .plan)

        let coder = try JSONDecoder().decode(
            ChatSession.self,
            from: JSONEncoder().encode(ChatSession(title: "Code", agentMode: .code))
        )
        XCTAssertEqual(coder.agentMode, .code)
    }

    func testALegacySessionWithoutModeDecodesAsNil() throws {
        // A conversation written before plan mode existed has no `agentMode`
        // key; nil is the coder, so an upgrade cannot trap it read-only.
        let legacy = ChatSession(title: "Old")
        let data = try JSONEncoder().encode(legacy)
        XCTAssertFalse(String(data: data, encoding: .utf8)?.contains("agentMode") == true)
        let decoded = try JSONDecoder().decode(ChatSession.self, from: data)
        XCTAssertNil(decoded.agentMode)
        XCTAssertEqual(decoded.agentMode ?? .code, .code)
    }
}
