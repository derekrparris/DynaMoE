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

    /// A plan may legitimately SHOW the convention — or, planning this feature,
    /// these markers — inside a code sample. Marker-shaped lines are markup only
    /// outside a fence; deleting one from a real block corrupted the plan's own
    /// content.
    func testAMarkerShownInsideACodeFenceIsContentNotMarkup() {
        let sample = """
        \(PlanMarkers.start)

        # Plan

        The convention is:

        ```
        step 1
        \(PlanMarkers.ready)
        step 2
        ```

        \(PlanMarkers.ready)
        """
        let stripped = PlanMarkers.strip(sample)
        XCTAssertTrue(
            stripped.contains("step 1\n\(PlanMarkers.ready)\nstep 2"),
            "a fenced sample keeps the marker it shows"
        )
        XCTAssertFalse(stripped.hasPrefix(PlanMarkers.start), "the real opening marker is still removed")
        XCTAssertFalse(stripped.hasSuffix(PlanMarkers.ready), "the real closing marker is still removed")
        XCTAssertEqual(
            stripped.components(separatedBy: PlanMarkers.ready).count - 1,
            1,
            "only the marker inside the sample survives"
        )
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

    func testCloudGuidanceCarriesThePlanDirectiveAndItsMarkers() {
        // A cloud turn assembles its own prompt, so the directive has to be
        // rendered there too. Without it the turn got a withheld-tools note and
        // no marker contract: the plan card and the approval bar (both keyed on
        // the markers) could never form, and the pill was the only way out.
        let guidance = AgentHarness.shared.buildCloudToolGuidance(mode: .plan)
        XCTAssertTrue(guidance.contains("# Plan Mode"))
        XCTAssertTrue(guidance.contains(PlanMarkers.start))
        XCTAssertTrue(guidance.contains(PlanMarkers.ready))
        XCTAssertTrue(guidance.contains("file_write"), "the withheld set is named")
        XCTAssertTrue(guidance.contains("# Tools"))
        // Role before capabilities, as in the local prompt.
        let role = guidance.range(of: "# Plan Mode")
        let tools = guidance.range(of: "# Tools")
        XCTAssertNotNil(role)
        XCTAssertNotNil(tools)
        if let role, let tools {
            XCTAssertLessThan(role.lowerBound, tools.lowerBound)
        }
        // `complete` is withheld, so the plan turn must not be told to call it.
        XCTAssertFalse(guidance.contains("Call `complete` only"))

        // The coder turn gains no plan section and keeps its complete guidance.
        let coder = AgentHarness.shared.buildCloudToolGuidance(mode: .code)
        XCTAssertFalse(coder.contains("# Plan Mode"))
        XCTAssertTrue(coder.contains("Call `complete` only"))
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

    /// Every path that executes a tool asks this, so a withheld call is never
    /// offered to the user for approval: the approval came first and the write
    /// was then refused by the mode, which read as a broken control rather than
    /// as a promise being kept.
    func testAWithheldCallNeverRaisesAnApprovalPrompt() {
        XCTAssertTrue(AgentHarness.requiresApproval(toolName: "file_write", mode: .code))
        XCTAssertTrue(AgentHarness.requiresApproval(toolName: "shell_run", mode: .code))
        XCTAssertFalse(AgentHarness.requiresApproval(toolName: "file_write", mode: .plan))
        XCTAssertFalse(AgentHarness.requiresApproval(toolName: "shell_run", mode: .plan))
        XCTAssertFalse(
            AgentHarness.requiresApproval(toolName: "mcp__server__write", mode: .plan),
            "MCP is withheld wholesale in plan mode"
        )
        XCTAssertFalse(AgentHarness.requiresApproval(toolName: "file_read", mode: .code))
    }

    /// A name nothing provides is UNKNOWN, not withheld. The withheld branch ran
    /// first, so a hallucinated `web_search_fetch` was answered with "do not
    /// retry it" instead of the available-names hint that exists to stop the
    /// model repeating the bad string.
    @MainActor
    func testAMisspelledToolIsUnknownRatherThanWithheld() async {
        AgentHarness.shared.setActiveMode(.plan)
        let call = ParsedToolCall(
            name: "web_search_fetch",
            arguments: ["query": "anything"],
            rawArguments: "",
            rawText: "web_search_fetch"
        )
        let result = await AgentHarness.shared.executeTool(call: call, mode: .plan)

        XCTAssertEqual(result.record.status, .error)
        XCTAssertTrue(result.record.error?.contains("Unknown tool") == true)
        XCTAssertFalse(
            result.record.error?.contains("PLAN MODE") == true,
            "a typo must not be reported as a policy refusal"
        )
        XCTAssertTrue(
            result.record.error?.contains("file_read") == true,
            "the unknown-tool answer echoes what can be called"
        )
    }

    /// `tools_discover` and `tools_load` take no `mode:` parameter, so they read
    /// the mode the executor published for THIS turn rather than the
    /// process-global one another turn can move.
    @MainActor
    func testToolDiscoveryAnswersForTheTurnsModeNotTheGlobalOne() async {
        let harness = AgentHarness.shared
        harness.setActiveMode(.code)
        XCTAssertFalse(PlanModePolicy.withholds("file_write", in: harness.activeMode))

        let discover = ParsedToolCall(
            name: "tools_discover",
            arguments: [:],
            rawArguments: "",
            rawText: "tools_discover"
        )
        let listing = await harness.executeTool(call: discover, mode: .plan)
        XCTAssertEqual(listing.record.status, .success)
        XCTAssertTrue(
            listing.resultJSON.contains("withheld_in_plan_mode"),
            "the listing must answer for the plan turn, got: \(listing.resultJSON)"
        )

        let load = ParsedToolCall(
            name: "tools_load",
            arguments: ["name": "file_write"],
            rawArguments: "",
            rawText: "tools_load"
        )
        let refused = await harness.executeTool(call: load, mode: .plan)
        XCTAssertEqual(refused.record.status, .error)
        XCTAssertTrue(
            refused.record.error?.contains("withheld in plan mode") == true,
            "a plan turn cannot arm what the mode withholds, got: \(refused.record.error ?? "nil")"
        )

        // The same call in the turn's own coder mode still loads it, so the
        // scoped value is a turn's answer and not a one-way latch.
        let loaded = await harness.executeTool(call: load, mode: .code)
        XCTAssertEqual(loaded.record.status, .success)
        XCTAssertNotNil(harness.loadedTools["file_write"])
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

    /// Plan mode is an agent mode: a stored pair that says "plan" with tools
    /// off has no directive, no read tools and no handoff, so its turns would
    /// run as plain chat under a pill promising an agent. Both construction and
    /// decode normalize the pair rather than trusting the file.
    func testAPlanSessionAlwaysRestoresWithToolsOn() throws {
        let storedApart = ChatSession(
            title: "Plan with tools off",
            isAgentToolsEnabled: false,
            agentMode: .plan
        )
        XCTAssertEqual(storedApart.isAgentToolsEnabled, true, "the pair cannot be represented apart")

        let decoded = try JSONDecoder().decode(
            ChatSession.self,
            from: JSONEncoder().encode(storedApart)
        )
        XCTAssertEqual(decoded.agentMode, .plan)
        XCTAssertEqual(decoded.isAgentToolsEnabled, true)

        // A hand-edited file that bypasses the initializer still normalizes.
        var raw = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(storedApart)) as? [String: Any]
        )
        raw["isAgentToolsEnabled"] = false
        let handEdited = try JSONDecoder().decode(
            ChatSession.self,
            from: JSONSerialization.data(withJSONObject: raw)
        )
        XCTAssertEqual(handEdited.agentMode, .plan)
        XCTAssertEqual(handEdited.isAgentToolsEnabled, true)

        // The coder keeps whatever the tool toggle says.
        XCTAssertEqual(
            ChatSession(title: "Code", isAgentToolsEnabled: false, agentMode: .code).isAgentToolsEnabled,
            false
        )
    }

    // MARK: - Review follow-ups

    /// Detection and stripping read one rule. They used to disagree: the
    /// predicates scanned every line while `strip` kept fenced markers, so a
    /// plan that QUOTED the convention raised an approval bar for a plan that
    /// was never written.
    func testAMarkerInsideACodeSampleNeitherRaisesTheCardNorGetsStripped() {
        let sample = """
        The convention is:

        ```
        \(PlanMarkers.start)
        # Plan
        \(PlanMarkers.ready)
        ```
        """
        XCTAssertFalse(PlanMarkers.startPresent(in: sample))
        XCTAssertFalse(PlanMarkers.readyPresent(in: sample))
        XCTAssertEqual(PlanMarkers.strip(sample), sample, "a fenced sample is content, not markup")

        // A marker the model wrapped in a fence pair of its own IS the
        // delimiter: there is nothing else in the block for the fence to show.
        let wrapped = "```\n\(PlanMarkers.ready)\n```"
        XCTAssertTrue(PlanMarkers.readyPresent(in: wrapped))
        XCTAssertEqual(PlanMarkers.strip(wrapped), "")

        // A whole answer wrapped in one fence is presentation, not a sample, so
        // its markers still count. Models do this routinely.
        let wholeAnswer = "```\n\(PlanMarkers.start)\n# Plan\n\(PlanMarkers.ready)\n```"
        XCTAssertTrue(PlanMarkers.startPresent(in: wholeAnswer))
        XCTAssertTrue(PlanMarkers.readyPresent(in: wholeAnswer))
        let stripped = PlanMarkers.strip(wholeAnswer)
        XCTAssertFalse(stripped.contains(PlanMarkers.start))
        XCTAssertFalse(stripped.contains(PlanMarkers.ready))
        XCTAssertTrue(stripped.contains("# Plan"))
    }

    /// A CRLF transcript leaves `\r` on every line. Trimming `.whitespaces`
    /// alone never matched the marker, so a plan sent with Windows line endings
    /// showed no card and offered no handoff.
    func testCRLFMarkerLinesAreDetectedAndStripped() {
        let text = "\(PlanMarkers.start)\r\n# Plan\r\n\r\n\(PlanMarkers.ready)\r\n"
        XCTAssertTrue(PlanMarkers.startPresent(in: text))
        XCTAssertTrue(PlanMarkers.readyPresent(in: text))
        let stripped = PlanMarkers.strip(text)
        XCTAssertFalse(stripped.contains(PlanMarkers.start))
        XCTAssertFalse(stripped.contains(PlanMarkers.ready))
        XCTAssertTrue(stripped.contains("# Plan"))
    }

    /// The local prompt is built by `buildSystemPrompt`, not by the cloud
    /// guidance, and it instructed the model to call `complete` in every mode.
    /// In plan mode that tool is withheld and the grammar cannot emit it, so the
    /// instruction contradicted the directive a few paragraphs above it.
    func testTheLocalPlanPromptDoesNotTeachComplete() {
        AgentHarness.shared.setActiveMode(.plan)
        let plan = AgentHarness.shared.buildSystemPrompt(baseSystem: "persona", mode: .plan)
        XCTAssertFalse(plan.contains("Call `complete`"), "plan mode must not instruct a withheld tool")
        XCTAssertTrue(plan.contains("`complete` is withheld"))
        XCTAssertTrue(plan.contains("ready marker"))

        AgentHarness.shared.setActiveMode(.code)
        let coder = AgentHarness.shared.buildSystemPrompt(baseSystem: "persona", mode: .code)
        XCTAssertTrue(coder.contains("Call `complete`"), "the coder keeps its completion guidance")
    }

    /// `shell_run` and `complete` are coder CORE, so they stay in the registry
    /// while plan mode filters them out of the prompt. The already-loaded branch
    /// ran first and answered a withheld name with success plus its full schema.
    @MainActor
    func testToolsLoadCannotReServeAWithheldToolThatStaysLoaded() async {
        let harness = AgentHarness.shared
        harness.setActiveMode(.plan)
        XCTAssertNotNil(
            harness.loadedTools["shell_run"],
            "the coder core stays loaded; the mode is what withholds it"
        )

        let call = ParsedToolCall(
            name: "tools_load",
            arguments: ["name": "shell_run"],
            rawArguments: "",
            rawText: "tools_load"
        )
        let result = await harness.executeTool(call: call, mode: .plan)
        XCTAssertEqual(result.record.status, .error)
        XCTAssertTrue(
            result.record.error?.contains("withheld in plan mode") == true,
            "got: \(result.record.error ?? "nil")"
        )
        XCTAssertFalse(
            result.resultJSON.contains("\"name\":\"shell_run\""),
            "the refused tool's schema must not be handed back"
        )
    }

    /// Settings is not a turn. Loading an MCP tool there used to be judged
    /// against whichever conversation was selected, so a merely-selected plan
    /// conversation refused a control the operator had clicked.
    func testSettingsCanLoadAToolWhileAConversationIsLeftInPlanMode() throws {
        let harness = AgentHarness.shared
        harness.setActiveMode(.plan)
        XCTAssertThrowsError(try harness.loadTool(named: "file_write"))
        XCTAssertNoThrow(
            try harness.loadTool(named: "file_write", mode: .code),
            "an explicit caller outside a turn is not bound by the turn's mode"
        )
        XCTAssertNotNil(harness.loadedTools["file_write"])
    }

    /// The Test Runner pipeline lifts a shell command out of free text and runs
    /// it through `AgentHarness.runProcess`, bypassing `SubagentToolExecutor`'s
    /// allow-list entirely. The parent's mode is the only gate that sees it.
    func testATestRunnerSubagentOfAPlanParentCannotRunShell() {
        let planOwner = SubagentInstance(
            role: "Test Runner",
            taskDescription: "run `rm -rf /tmp/x`",
            parentMode: .plan
        )
        XCTAssertEqual(planOwner.archetype, .testRunner)
        XCTAssertFalse(planOwner.canRunShellCommands)

        let coderOwner = SubagentInstance(
            role: "Test Runner",
            taskDescription: "run `swift test`",
            parentMode: .code
        )
        XCTAssertTrue(coderOwner.canRunShellCommands)
    }
}
