//
//  AgentSpawnPolicyTests.swift
//  DynaMoETests
//
//  Presets, spawn approval and the spawn tool's schema. No model is needed:
//  every case that would need a backend stops before it is resolved.
//

import XCTest
@testable import DynaMoE

@MainActor
final class AgentSpawnPolicyTests: XCTestCase {
    private let workingDirectory = URL(fileURLWithPath: NSTemporaryDirectory())

    private func request(
        role: String = "Codebase Researcher",
        task: String = "Find where the scheduler lives.",
        tools: [String]? = nil,
        preset: String? = nil,
        isolation: AgentIsolation = .none,
        mode: AgentMode = .code
    ) -> AgentSpawnRequest {
        AgentSpawnRequest(
            role: role,
            taskDescription: task,
            presetName: preset,
            explicitToolNames: tools,
            isolation: isolation,
            workingDirectory: workingDirectory,
            parentMode: mode
        )
    }

    // MARK: - Presets

    func testPresetSurfacesMatchTheDesign() {
        let explore = AgentRolePresetCatalog.definition(for: .explore)
        XCTAssertTrue(explore.toolNames.contains("file_read"))
        XCTAssertTrue(explore.toolNames.contains("web_search"))
        XCTAssertFalse(explore.canMutate)
        XCTAssertEqual(explore.maxSteps, 16)

        let implementer = AgentRolePresetCatalog.definition(for: .implementer)
        XCTAssertTrue(implementer.toolNames.contains("file_edit"))
        XCTAssertTrue(implementer.toolNames.contains("shell_run"))
        XCTAssertTrue(implementer.canMutate)
        XCTAssertEqual(implementer.maxSteps, 40)

        let verifier = AgentRolePresetCatalog.definition(for: .verifier)
        XCTAssertTrue(verifier.toolNames.contains("lint_diagnostics"))
        XCTAssertFalse(verifier.toolNames.contains("file_write"))
        // `shell_run` counts as mutating: a test command can change the tree, so
        // the verifier still goes through per-call approval.
        XCTAssertTrue(verifier.canMutate)

        let reviewer = AgentRolePresetCatalog.definition(for: .reviewer)
        XCTAssertTrue(reviewer.toolNames.contains("git_diff"))
        XCTAssertFalse(reviewer.canMutate)

        let general = AgentRolePresetCatalog.definition(for: .general)
        XCTAssertFalse(general.canMutate, "general runs read by default")
    }

    func testRoleTitlesMapToPresets() {
        XCTAssertEqual(AgentRolePresetCatalog.match(role: "Codebase Researcher"), .explore)
        XCTAssertEqual(AgentRolePresetCatalog.match(role: "Test Runner"), .verifier)
        XCTAssertEqual(AgentRolePresetCatalog.match(role: "Shader Optimizer"), .implementer)
        XCTAssertEqual(AgentRolePresetCatalog.match(role: "Documentation Writer"), .implementer)
        XCTAssertEqual(AgentRolePresetCatalog.match(role: "Security Reviewer"), .reviewer)
        XCTAssertEqual(AgentRolePresetCatalog.match(role: "Helpmate"), .general)
    }

    func testExplicitPresetWinsOverTheRoleTitle() {
        XCTAssertEqual(
            AgentRolePresetCatalog.preset(named: "reviewer", role: "Test Runner"),
            .reviewer
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.preset(named: nil, role: "Test Runner"),
            .verifier
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.preset(named: "not-a-preset", role: "Test Runner"),
            .verifier,
            "an unknown spelling falls back to the role's own inference"
        )
    }

    func testBudgetOverridesKeepTheCallersNumbers() {
        let definition = AgentRolePresetCatalog.definition(for: .explore)
        let overridden = AgentRolePresetCatalog.budget(
            for: definition,
            overrides: AgentBudgetOverrides(maxSteps: 3, maxCompletionTokens: 500, maxWallClockSeconds: 30)
        )
        XCTAssertEqual(overridden.maxSteps, 3)
        XCTAssertEqual(overridden.maxCompletionTokens, 500)
        XCTAssertEqual(overridden.maxWallClockSeconds, 30)
        XCTAssertEqual(
            AgentRolePresetCatalog.budget(for: definition, overrides: nil).maxSteps,
            definition.maxSteps
        )
    }

    func testModelSuppliedBudgetsAreClampedToBoundsTheAppCanSurvive() {
        // The spawn's budget fields arrive from a MODEL's arguments. A huge
        // `max_wall_clock_seconds` used to reach the loop's nanosecond
        // conversion and trap, taking the whole app down; every field is now
        // clamped to `AgentBudget`'s own bounds rather than merely floored.
        let definition = AgentRolePresetCatalog.definition(for: .general)
        let inflated = AgentRolePresetCatalog.budget(
            for: definition,
            overrides: AgentBudgetOverrides(
                maxSteps: Int.max,
                maxCompletionTokens: Int.max,
                maxWallClockSeconds: Int.max
            )
        )
        XCTAssertEqual(inflated.maxSteps, AgentBudget.maximumSteps)
        XCTAssertEqual(inflated.maxCompletionTokens, AgentBudget.maximumCompletionTokens)
        XCTAssertEqual(inflated.maxWallClockSeconds, AgentBudget.maximumWallClockSeconds)
        // Within the bounds, the caller's number still wins exactly.
        let honest = AgentRolePresetCatalog.budget(
            for: definition,
            overrides: AgentBudgetOverrides(maxSteps: 5, maxCompletionTokens: 900, maxWallClockSeconds: 90)
        )
        XCTAssertEqual(honest.maxSteps, 5)
        XCTAssertEqual(honest.maxCompletionTokens, 900)
        XCTAssertEqual(honest.maxWallClockSeconds, 90)
        // A non-finite ceiling never reaches the loop as a number at all: the
        // resolver refuses it at the card (see the resolve test below), and
        // the budget builder folds it away defensively so no OTHER caller can
        // turn `.nan` into a "$0 ceiling" (`max(0, .nan)` returns 0).
        let malformed = AgentRolePresetCatalog.budget(
            for: definition,
            overrides: AgentBudgetOverrides(maxCostUSD: .nan)
        )
        XCTAssertNil(malformed.maxCostUSD)
    }

    /// `max(0, .nan)` returns 0 in Swift, so a malformed ceiling used to become
    /// a "$0 ceiling" that failed the run at step 0 with a cost failure the
    /// operator never asked for. The spawn refuses it by name instead.
    func testANonFiniteCostCeilingIsRefusedAtTheSpawn() async {
        for malformed in [Double.nan, .infinity, -.infinity] {
            do {
                _ = try await AgentSpawnResolver.resolve(
                    AgentSpawnRequest(
                        role: "Codebase Researcher",
                        taskDescription: "Find the scheduler.",
                        explicitToolNames: ["file_read"],
                        budgetOverrides: AgentBudgetOverrides(maxCostUSD: malformed),
                        workingDirectory: workingDirectory
                    ),
                    runId: UUID(),
                    sessionId: UUID(),
                    defaults: UserDefaults(suiteName: "dynamoe-tests-spawn-malformed-\(UUID().uuidString)")!
                )
                XCTFail("resolve must refuse a non-finite ceiling")
            } catch {
                guard let spawnError = error as? AgentSpawnError,
                      case .invalidBudgetOverride = spawnError else {
                    return XCTFail("expected invalidBudgetOverride, got \(error)")
                }
            }
        }
    }

    func testApprovalPolicyIsDenyInPlanMode() {
        let implementer = AgentRolePresetCatalog.definition(for: .implementer)
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(preset: implementer, mode: .plan),
            .deny
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(preset: implementer, mode: .code),
            .everyStateChange
        )
        let explore = AgentRolePresetCatalog.definition(for: .explore)
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(preset: explore, mode: .code),
            .auto
        )
    }

    /// `allowed_tools` can hand a read-only preset a writer, and the approval
    /// policy used to read the PRESET: explore + file_write classified as
    /// automatic, so the operator saw no card, and the run scope then denied
    /// every write it had been given — inside a worktree, an empty one.
    func testApprovalPolicyReadsTheToolsTheRunWasGiven() {
        let explore = AgentRolePresetCatalog.definition(for: .explore)
        let withWriter = explore.toolNames + ["file_write"]
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(toolNames: withWriter, mode: .code),
            .everyStateChange
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(toolNames: withWriter, mode: .code, isolated: true),
            .spawnOnly
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(toolNames: withWriter, mode: .plan),
            .deny
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(
                toolNames: explore.toolNames,
                mode: .code
            ),
            .auto,
            "a read-only tool set still runs without cards"
        )
    }

    // MARK: - Spawn approval

    func testAWriteCapableSpawnNeedsApproval() {
        XCTAssertTrue(AgentHarness.spawnSubagentRequiresApproval(
            arguments: ["role": "Shader Optimizer", "task_description": "opt"],
            mode: .code
        ))
        XCTAssertTrue(AgentHarness.spawnSubagentRequiresApproval(
            arguments: [
                "role": "Custom",
                "task_description": "edit",
                "allowed_tools": ["file_read", "file_edit"]
            ],
            mode: .code
        ))
    }

    func testAReadOnlySpawnNeedsNoApproval() {
        XCTAssertFalse(AgentHarness.spawnSubagentRequiresApproval(
            arguments: ["role": "Codebase Researcher", "task_description": "look"],
            mode: .code
        ))
        XCTAssertFalse(AgentHarness.spawnSubagentRequiresApproval(
            arguments: [
                "role": "Shader Optimizer",
                "task_description": "opt",
                "allowed_tools": ["file_read", "grep_search"]
            ],
            mode: .code
        ))
    }

    func testAReadOnlySpawnThatForcesCloudNeedsAnOffDeviceCard() {
        let suiteName = "AgentSpawnPolicyTests-cloud-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let arguments: [String: Any] = [
            "role": "Codebase Researcher",
            "task_description": "read only",
            "allowed_tools": ["file_read", "grep_search"],
            "backend": "cloud"
        ]

        XCTAssertTrue(
            AgentHarness.spawnSubagentRequiresApproval(
                arguments: arguments,
                mode: .code,
                defaults: defaults
            ),
            "a read-only run still uploads repository content when it runs in the cloud"
        )

        defaults.set("cloud", forKey: SubagentBackendSettings.backendKey)
        XCTAssertFalse(
            AgentHarness.spawnSubagentRequiresApproval(
                arguments: arguments,
                mode: .code,
                defaults: defaults
            ),
            "the Settings choice already names the off-device backend, so this read-only spawn needs no second card"
        )
    }

    func testLandingAWorktreeIsNeverABypass() {
        XCTAssertFalse(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "apply_subagent_changes",
                arguments: ["subagent_id": UUID().uuidString],
                mode: .code
            ),
            "SUBAGENTS.md gives an apply its own card even in Turbo Mode"
        )
    }

    func testATurboSpawnThatForcesCloudKeepsItsOffDeviceCard() {
        let suiteName = "AgentSpawnPolicyTests-turbo-cloud-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set("local", forKey: SubagentBackendSettings.backendKey)

        XCTAssertFalse(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "spawn_subagent",
                arguments: [
                    "role": "Codebase Researcher",
                    "task_description": "read only",
                    "allowed_tools": ["file_read", "grep_search"],
                    "backend": "cloud"
                ],
                mode: .code,
                defaults: defaults
            ),
            "Turbo must not skip the one card that says repository content leaves the machine"
        )
        XCTAssertTrue(
            AgentHarness.turboModeMayBypassApproval(
                toolName: "spawn_subagent",
                arguments: [
                    "role": "Codebase Researcher",
                    "task_description": "read only",
                    "allowed_tools": ["file_read", "grep_search"]
                ],
                mode: .code,
                defaults: defaults
            ),
            "a read-only LOCAL spawn follows the stored Settings choice and stays routine"
        )
    }

    func testARunScopedCatalogIsTheBuiltinInstallAndStillForbidsDelegation() {
        let names = AgentHarness.runScopedToolNames()
        XCTAssertTrue(names.contains("file_read"))
        XCTAssertTrue(names.contains("complete"))
        // The installer registers these so the shared catalog can name them,
        // but a run may never be GIVEN one. The spawn path checks the forbidden
        // set before the catalog, which is what keeps delegation one level.
        XCTAssertTrue(AgentHarness.runScopedForbiddenToolNames.contains("spawn_subagent"))
        XCTAssertTrue(AgentHarness.runScopedForbiddenToolNames.contains("apply_subagent_changes"))
    }

    func testTheSubagentStatusFamilyIsNeverGivenToARun() {
        // `get_subagent_status`, `list_subagents` and `send_subagent_message`
        // look read-only but operate PROCESS-WIDE: the status tool answers any
        // run's summary and marks it relayed, which suppresses that run's own
        // report — and none of the three is write-capable, so without this
        // entry a read-only spawn needed no card to reach them.
        for name in ["get_subagent_status", "send_subagent_message", "list_subagents"] {
            XCTAssertTrue(
                AgentHarness.runScopedForbiddenToolNames.contains(name),
                "'\(name)' must be in the run-scoped forbidden set"
            )
        }
        do {
            _ = try AgentRunContext.make(
                runId: UUID(),
                sessionId: UUID(),
                toolNames: ["file_read", "get_subagent_status"],
                workingDirectory: FileManager.default.temporaryDirectory,
                mode: .code,
                approvalPolicy: .auto,
                workspaceIsolated: false,
                approvalHandler: nil
            )
            XCTFail("the status family must be refused at run-scope configuration")
        } catch let error as AgentRunScopeError {
            guard case .forbiddenTool("get_subagent_status") = error else {
                return XCTFail("expected a get_subagent_status refusal, got \(error)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        do {
            let definition = AgentRolePresetCatalog.definition(for: .explore)
            _ = try AgentSpawnResolver.resolveTools(
                request(tools: ["file_read", "list_subagents", "send_subagent_message"]),
                definition: definition
            )
            XCTFail("the status family must be refused at spawn resolution")
        } catch let error as AgentSpawnError {
            guard case .forbiddenTool = error else {
                return XCTFail("expected a forbiddenTool refusal, got \(error)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    /// Plan mode withholds writes, so a local or legacy spawn cannot change
    /// anything and needs no card. A spawn that works OFF-DEVICE still does:
    /// its read tools (`file_read`, `grep_search`) upload repository content on
    /// every step, which is the decision the spawn card exists for.
    func testPlanModeOnlyAsksForASpawnThatWorksOffDevice() {
        let suiteName = "AgentSpawnPolicyTests.plan.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.removeObject(forKey: SubagentBackendSettings.backendKey)

        let base: [String: Any] = ["role": "Shader Optimizer", "task_description": "opt"]

        // No stored backend: `auto` keeps the legacy/local pipelines.
        XCTAssertFalse(AgentHarness.spawnSubagentRequiresApproval(arguments: base, mode: .plan, defaults: defaults))
        XCTAssertFalse(AgentHarness.spawnSubagentRequiresApproval(
            arguments: base.merging(["backend": "local"]) { $1 },
            mode: .plan,
            defaults: defaults
        ))
        XCTAssertFalse(AgentHarness.spawnSubagentRequiresApproval(
            arguments: base.merging(["backend": "legacy"]) { $1 },
            mode: .plan,
            defaults: defaults
        ))

        // An explicit Cloud backend asks even in plan mode.
        XCTAssertTrue(AgentHarness.spawnSubagentRequiresApproval(
            arguments: base.merging(["backend": "cloud"]) { $1 },
            mode: .plan,
            defaults: defaults
        ))

        // Settings=Cloud makes an omitted/`auto` backend resolve off-device,
        // so it asks too; an explicit local backend still does not.
        defaults.set("cloud", forKey: SubagentBackendSettings.backendKey)
        XCTAssertTrue(AgentHarness.spawnSubagentRequiresApproval(arguments: base, mode: .plan, defaults: defaults))
        XCTAssertTrue(AgentHarness.spawnSubagentRequiresApproval(
            arguments: base.merging(["backend": "auto"]) { $1 },
            mode: .plan,
            defaults: defaults
        ))
        XCTAssertFalse(AgentHarness.spawnSubagentRequiresApproval(
            arguments: base.merging(["backend": "local"]) { $1 },
            mode: .plan,
            defaults: defaults
        ))
    }

    // MARK: - Resolver refusals

    func testEmptyTaskIsRefused() async {
        do {
            _ = try await AgentSpawnResolver.resolve(
                request(task: "   "),
                runId: UUID(),
                sessionId: UUID()
            )
            XCTFail("an empty task must be refused")
        } catch {
            XCTAssertEqual(error as? AgentSpawnError, .emptyTask)
        }
    }

    func testWorktreeIsolationIsRefusedForAReadOnlyRun() async {
        do {
            _ = try await AgentSpawnResolver.resolve(
                request(isolation: .worktree(path: "", branch: "")),
                runId: UUID(),
                sessionId: UUID()
            )
            XCTFail("a read-only run must not get a worktree")
        } catch {
            guard case .unsupportedIsolation(let detail)? = error as? AgentSpawnError else {
                return XCTFail("expected an isolation refusal, got \(error)")
            }
            XCTAssertTrue(detail.contains("worktree"))
        }
    }

    func testForbiddenToolsAreRefused() async {
        do {
            _ = try await AgentSpawnResolver.resolve(
                request(tools: ["file_read", "spawn_subagent"]),
                runId: UUID(),
                sessionId: UUID()
            )
            XCTFail("a forbidden tool must be refused")
        } catch {
            XCTAssertEqual(error as? AgentSpawnError, .forbiddenTool("spawn_subagent"))
        }
    }

    func testWorktreeIsolationRefusesANonGitWriterWorkspace() async {
        do {
            _ = try await AgentSpawnResolver.resolve(
                request(tools: ["file_write"], isolation: .worktree(path: "", branch: "")),
                runId: UUID(),
                sessionId: UUID()
            )
            XCTFail("a writer on a non-git workspace must not get a worktree")
        } catch {
            guard case .unsupportedIsolation(let detail)? = error as? AgentSpawnError else {
                return XCTFail("expected an isolation refusal, got \(error)")
            }
            XCTAssertTrue(detail.lowercased().contains("git"), detail)
        }
    }

    func testIsolatedWritersApproveAtSpawnOnly() {
        let implementer = AgentRolePresetCatalog.definition(for: .implementer)
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(preset: implementer, mode: .code, isolated: true),
            .spawnOnly
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(preset: implementer, mode: .code, isolated: false),
            .everyStateChange
        )
        let explore = AgentRolePresetCatalog.definition(for: .explore)
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(preset: explore, mode: .code, isolated: false),
            .auto
        )
        XCTAssertEqual(
            AgentRolePresetCatalog.approvalPolicy(preset: implementer, mode: .plan, isolated: true),
            .deny
        )
    }

    func testUnknownToolsAreRefusedByName() async {
        do {
            _ = try await AgentSpawnResolver.resolve(
                request(tools: ["file_read", "definitely_not_a_tool"]),
                runId: UUID(),
                sessionId: UUID()
            )
            XCTFail("an unknown tool must be refused")
        } catch {
            XCTAssertEqual(error as? AgentSpawnError, .unknownTool("definitely_not_a_tool"))
        }
    }

    func testPlanModeDropsWithheldToolsInsteadOfAdvertisingThem() throws {
        let tools = try AgentSpawnResolver.resolveTools(
            request(tools: ["file_read", "file_edit", "shell_run"], mode: .plan),
            definition: AgentRolePresetCatalog.definition(for: .implementer)
        )
        XCTAssertTrue(tools.contains("file_read"))
        XCTAssertFalse(tools.contains("file_edit"))
        XCTAssertFalse(tools.contains("shell_run"))
    }

    // MARK: - Ending a run

    /// The worker prompt tells every run to finish with `complete`, and
    /// `endsAgentRun` keys the completion on it. A resolved tool set that omits
    /// it leaves a run with no way to declare it finished: it either stops
    /// calling tools (reported as a completion it never made) or dies on its
    /// step budget.
    func testEveryPresetCanEndItsRunWithComplete() throws {
        for preset in AgentRolePreset.allCases {
            let definition = AgentRolePresetCatalog.definition(for: preset)
            let tools = try AgentSpawnResolver.resolveTools(
                request(preset: preset.rawValue),
                definition: definition
            )
            XCTAssertTrue(tools.contains("complete"), "\(preset) must be able to call complete")
        }
    }

    func testAnExplicitToolListStillGetsComplete() throws {
        let tools = try AgentSpawnResolver.resolveTools(
            request(tools: ["file_read"]),
            definition: AgentRolePresetCatalog.definition(for: .implementer)
        )
        XCTAssertEqual(tools, ["complete", "file_read"])
    }

    /// An EXPLICIT empty list is not an omitted one. It used to fall back to
    /// the preset, so a caller that asked for no tools got a writer.
    func testAnExplicitlyEmptyToolListIsRefusedRatherThanUsingThePreset() {
        XCTAssertThrowsError(
            try AgentSpawnResolver.resolveTools(
                request(tools: []),
                definition: AgentRolePresetCatalog.definition(for: .implementer)
            )
        ) { error in
            guard case .noBackend(let message)? = error as? AgentSpawnError else {
                return XCTFail("expected noBackend, got \(error)")
            }
            XCTAssertTrue(message.contains("empty list"), message)
        }
    }

    func testPlanModeStillWithholdsComplete() throws {
        // In plan mode the plain-text plan IS the deliverable, so `complete` is
        // withheld by policy and must not be appended back.
        let tools = try AgentSpawnResolver.resolveTools(
            request(tools: ["file_read"], mode: .plan),
            definition: AgentRolePresetCatalog.definition(for: .explore)
        )
        XCTAssertEqual(tools, ["file_read"])
    }

    // MARK: - Cost ceilings

    /// A ceiling compared against an unpriced run can never trip: the backend
    /// reports `costUSD` 0 for every token because it has no rate to apply.
    /// The spawn refuses instead of pretending the limit binds.
    func testACostCeilingWithoutAPriceIsRefused() {
        let suiteName = "AgentSpawnPolicyTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let descriptor = AgentBackendDescriptor(
            kind: .cloud,
            providerId: "synthetic",
            modelId: "syn:large:text",
            displayName: "Syn Large"
        )
        let budget = AgentBudget(
            maxSteps: 4,
            maxCompletionTokens: 400,
            maxWallClockSeconds: 60,
            maxCostUSD: 0.25
        )
        XCTAssertEqual(
            AgentSpawnResolver.costCeilingProblem(budget: budget, descriptor: descriptor, defaults: defaults),
            .costCeilingWithoutPricing("Syn Large")
        )

        CloudPricingStore.setPricing(
            ["syn:large:text": CloudPricing(input: 1, output: 2)],
            for: "synthetic",
            defaults: defaults
        )
        XCTAssertNil(
            AgentSpawnResolver.costCeilingProblem(budget: budget, descriptor: descriptor, defaults: defaults)
        )
    }

    func testACostCeilingOnALocalRunIsRefusedToo() {
        let descriptor = AgentBackendDescriptor(
            kind: .local,
            providerId: nil,
            modelId: nil,
            displayName: "Local checkpoint"
        )
        let budget = AgentBudget(
            maxSteps: 4,
            maxCompletionTokens: 400,
            maxWallClockSeconds: 60,
            maxCostUSD: 1
        )
        XCTAssertEqual(
            AgentSpawnResolver.costCeilingProblem(budget: budget, descriptor: descriptor),
            .costCeilingOnUnpricedBackend("Local checkpoint"),
            "a local run's cost is engine time, not dollars, so a dollar ceiling cannot bind — and the refusal must not send the operator to record a cloud price"
        )
        XCTAssertTrue(
            (AgentSpawnError.costCeilingOnUnpricedBackend("Local checkpoint").errorDescription ?? "")
                .contains("not a cloud model"),
            "the refusal names the real reason for a local backend"
        )
    }

    func testNoCeilingNeedsNoPrice() {
        let descriptor = AgentBackendDescriptor(
            kind: .cloud,
            providerId: "synthetic",
            modelId: "syn:large:text",
            displayName: "Syn Large"
        )
        XCTAssertNil(
            AgentSpawnResolver.costCeilingProblem(budget: .standard, descriptor: descriptor)
        )
    }

    // MARK: - Tool schema

    func testSpawnSchemaCarriesBackendPresetBudgetAndIsolation() throws {
        let parameters = SpawnSubagentTool().definition.function.parameters
        let properties = try XCTUnwrap(parameters["properties"]?.value as? [String: Any])
        for key in ["backend", "preset", "budget", "isolation", "thinking"] {
            XCTAssertNotNil(properties[key], "spawn_subagent must advertise '\(key)'")
        }
        let backend = try XCTUnwrap(properties["backend"] as? [String: Any])
        XCTAssertEqual(backend["type"] as? String, "string")
        let budget = try XCTUnwrap(properties["budget"] as? [String: Any])
        let budgetProperties = try XCTUnwrap(budget["properties"] as? [String: Any])
        XCTAssertNotNil(budgetProperties["max_steps"])
        XCTAssertNotNil(budgetProperties["max_cost_usd"])

        let required = try XCTUnwrap(parameters["required"]?.value as? [String])
        XCTAssertEqual(required, ["role", "task_description"])
    }
}
