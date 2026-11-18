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

    func testPlanModeNeverAsksForASpawn() {
        XCTAssertFalse(AgentHarness.spawnSubagentRequiresApproval(
            arguments: ["role": "Shader Optimizer", "task_description": "opt"],
            mode: .plan
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
            .costCeilingWithoutPricing("Local checkpoint"),
            "a local run's cost is engine time, not dollars, so a dollar ceiling cannot bind"
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
