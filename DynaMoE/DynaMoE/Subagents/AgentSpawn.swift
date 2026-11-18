//
//  AgentSpawn.swift
//  DynaMoE
//
//  What a `spawn_subagent` call turns into: a preset, a backend, a tool set and
//  a budget. See SUBAGENTS.md §7, §8 and §9.
//

import Foundation

nonisolated enum AgentBackendPreference: String, Sendable, CaseIterable {
    case auto
    case local
    case cloud
}

nonisolated enum AgentSpawnError: LocalizedError, Equatable {
    case emptyTask
    case unsupportedIsolation(String)
    case noBackend(String)
    case forbiddenTool(String)
    case unknownTool(String)
    case notAvailableToRuns(String)
    case costCeilingWithoutPricing(String)
    case costCeilingOnUnpricedBackend(String)
    case invalidBudgetOverride(String)

    var errorDescription: String? {
        switch self {
        case .emptyTask:
            return "A subagent needs a task to work on."
        case .unsupportedIsolation(let detail):
            return "Isolated subagent workspaces are not available yet: \(detail) The run was not started. Use isolation \"none\" and approve each change."
        case .noBackend(let message):
            return message
        case .forbiddenTool(let name):
            return "Tool '\(name)' may never be given to a subagent run."
        case .unknownTool(let name):
            return "Tool '\(name)' is not installed, so it cannot be given to a subagent run."
        case .notAvailableToRuns(let name):
            return "Tool '\(name)' exists only in the coordinator's catalog (a dynamically loaded or MCP tool), and a subagent run installs the built-in tools alone."
        case .costCeilingWithoutPricing(let model):
            return "This spawn sets a max_cost_usd ceiling, but no price is recorded for '\(model)', so the ceiling could never bind. Record the model's input and output prices in Settings under Cloud Providers, or clear the ceiling."
        case .costCeilingOnUnpricedBackend(let backend):
            return "This spawn sets a max_cost_usd ceiling, but '\(backend)' is not a cloud model: it bills no dollars, so the run would always report $0 spent and the ceiling could never bind. Clear the ceiling, or run this work on a cloud model with recorded prices."
        case .invalidBudgetOverride(let detail):
            return "The spawn's budget was refused: \(detail) The run was not started."
        }
    }
}

/// Everything one spawn needs. The tool fills this from its arguments; the
/// manager resolves it into a run.
nonisolated struct AgentSpawnRequest: Sendable {
    var role: String
    var taskDescription: String
    var contextSummary: String?
    var presetName: String?
    var backendPreference: AgentBackendPreference
    var explicitToolNames: [String]?
    var budgetOverrides: AgentBudgetOverrides?
    var isolation: AgentIsolation
    var workingDirectory: URL
    var parentSessionId: UUID?
    var parentTurnId: UUID?
    var parentMode: AgentMode
    var thinking: Bool
    var operatorOverride: Bool
    /// The estimate the spawn card was priced with. Resolved once by the
    /// caller so the number shown and the number admission compares are the
    /// same inputs.
    var localCostEstimate: LocalSubagentCostEstimate? = nil
    /// The estimate the operator actually approved. Admission refuses when a
    /// recomputed estimate no longer matches it, because a click approves a
    /// number rather than the idea of a local run.
    var approvedLocalCost: LocalSubagentCostEstimate? = nil
    var priority: AgentSchedulingPriority
    var sampling: AgentSampling

    init(
        role: String,
        taskDescription: String,
        contextSummary: String? = nil,
        presetName: String? = nil,
        backendPreference: AgentBackendPreference = .auto,
        explicitToolNames: [String]? = nil,
        budgetOverrides: AgentBudgetOverrides? = nil,
        isolation: AgentIsolation = .none,
        workingDirectory: URL,
        parentSessionId: UUID? = nil,
        parentTurnId: UUID? = nil,
        parentMode: AgentMode = .code,
        thinking: Bool = false,
        operatorOverride: Bool = false,
        localCostEstimate: LocalSubagentCostEstimate? = nil,
        approvedLocalCost: LocalSubagentCostEstimate? = nil,
        priority: AgentSchedulingPriority = .userRequested,
        sampling: AgentSampling = .default
    ) {
        self.role = role
        self.taskDescription = taskDescription
        self.contextSummary = contextSummary
        self.presetName = presetName
        self.backendPreference = backendPreference
        self.explicitToolNames = explicitToolNames
        self.budgetOverrides = budgetOverrides
        self.isolation = isolation
        self.workingDirectory = workingDirectory
        self.parentSessionId = parentSessionId
        self.parentTurnId = parentTurnId
        self.parentMode = parentMode
        self.thinking = thinking
        self.operatorOverride = operatorOverride
        self.localCostEstimate = localCostEstimate
        self.approvedLocalCost = approvedLocalCost
        self.priority = priority
        self.sampling = sampling
    }
}

/// A resolved spawn: the run's identity, scope and model boundary.
nonisolated struct AgentSpawnPlan: @unchecked Sendable {
    var role: String
    var taskDescription: String
    var contextSummary: String?
    var preset: AgentRolePreset
    var backendDescriptor: AgentBackendDescriptor
    var schedulingClass: AgentSchedulingClass
    var backend: any AgentModelBackend
    var toolNames: [String]
    var budget: AgentBudget
    var approvalPolicy: AgentApprovalPolicy
    var isolation: AgentIsolation
    var thinking: Bool
    var sampling: AgentSampling
    var priority: AgentSchedulingPriority
    var parentSessionId: UUID?
    var parentTurnId: UUID?
    var workingDirectory: URL
    /// The repository an isolated worktree was cut from. Persisted with the run
    /// so a restored record can apply to the operator's tree even when the
    /// worktree's own git metadata cannot name it (a submodule, or a repo using
    /// `--separate-git-dir`).
    var repositoryRoot: URL? = nil
    /// The commit the worktree was cut from. Persisted because recomputing it
    /// as `merge-base(branch, HEAD)` answers a different question: a reset or a
    /// rebase on the main branch below the fork point moves that merge base, so
    /// a run's apparent change would grow or shrink without its work changing.
    var baseCommit: String? = nil
}

/// Turns a request into a plan, or refuses by name.
///
/// Backend selection is the only part that reads app state: a local run needs a
/// loaded checkpoint and its cost admitted, a cloud run needs a configured
/// provider and a stored key. Everything else is pure, so a spawn card can price
/// a run before it starts.
@MainActor
enum AgentSpawnResolver {
    static func resolve(
        _ request: AgentSpawnRequest,
        runId: UUID,
        sessionId: UUID,
        defaults: UserDefaults = .standard
    ) async throws -> AgentSpawnPlan {
        let task = request.taskDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !task.isEmpty else { throw AgentSpawnError.emptyTask }

        let preset = AgentRolePresetCatalog.preset(named: request.presetName, role: request.role)
        let definition = AgentRolePresetCatalog.definition(for: preset)
        let toolNames = try resolveTools(request, definition: definition)
        // A non-finite ceiling is not a budget: `max(0, .nan)` folds to 0 in
        // the budget builder, and a "$0 ceiling" fails the run at step 0 with
        // a cost failure the operator never asked for. Refuse both at the
        // card, before any backend or worktree exists to clean up. Zero is
        // refused too: no step can ever cost less than $0, so the loop would
        // report "exhausted its cost budget" before doing any work.
        if let ceiling = request.budgetOverrides?.maxCostUSD {
            guard ceiling.isFinite, ceiling > 0 else {
                throw AgentSpawnError.invalidBudgetOverride(
                    "max_cost_usd must be a finite dollar amount greater than 0; '\(ceiling)' cannot be enforced as a ceiling."
                )
            }
        }
        let budget = AgentRolePresetCatalog.budget(for: definition, overrides: request.budgetOverrides)

        // Isolation is validated BEFORE the backend is resolved, so a spawn that
        // cannot have a worktree is refused for that reason rather than for
        // "no model loaded". The worktree itself is created after the backend
        // resolves: a local cost refusal re-resolves the request, and a worktree
        // created before it would be left behind.
        var worktreeRepository: URL?
        var worktreeBaseCommit: String?
        if request.isolation.isIsolated {
            guard !Set(toolNames).isDisjoint(with: AgentToolSurface.writeCapable) else {
                throw AgentSpawnError.unsupportedIsolation(
                    "this run's tools cannot change the workspace, so a worktree would be an empty copy. Use isolation \"none\" for a read-only run."
                )
            }
            guard let repositoryRoot = await WorkspaceIsolation.repositoryRoot(for: request.workingDirectory) else {
                throw AgentSpawnError.unsupportedIsolation(
                    "'\(request.workingDirectory.path)' is not inside a git repository. A writer on a non-git workspace must use isolation \"none\" and approve each change."
                )
            }
            worktreeRepository = repositoryRoot
        }

        let (backend, schedulingClass, descriptor) = try resolveBackend(
            request,
            task: task,
            runId: runId,
            sessionId: sessionId,
            defaults: defaults
        )

        // A cost ceiling is only a ceiling when the model has a recorded price.
        // An unpriced cloud model reports `costUSD` 0 for every token it bills,
        // so the ceiling would never trip and the run would spend an amount the
        // spawn card could not name. Refuse before anything runs.
        if let problem = Self.costCeilingProblem(budget: budget, descriptor: descriptor, defaults: defaults) {
            throw problem
        }

        var isolation = AgentIsolation.none
        var workingDirectory = request.workingDirectory
        if let repositoryRoot = worktreeRepository {
            do {
                let worktree = try await WorkspaceIsolation.createWorktree(
                    repositoryRoot: repositoryRoot,
                    runId: runId
                )
                isolation = worktree.isolation
                workingDirectory = worktree.path
                worktreeBaseCommit = worktree.baseCommit
            } catch {
                let detail = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                throw AgentSpawnError.unsupportedIsolation(detail)
            }
        }

        let approvalPolicy = AgentRolePresetCatalog.approvalPolicy(
            toolNames: toolNames,
            mode: request.parentMode,
            isolated: isolation.isIsolated
        )

        return AgentSpawnPlan(
            role: request.role,
            taskDescription: task,
            contextSummary: request.contextSummary,
            preset: preset,
            backendDescriptor: descriptor,
            schedulingClass: schedulingClass,
            backend: backend,
            toolNames: toolNames,
            budget: budget,
            approvalPolicy: approvalPolicy,
            isolation: isolation,
            thinking: request.thinking,
            sampling: request.sampling,
            priority: request.priority,
            parentSessionId: request.parentSessionId,
            parentTurnId: request.parentTurnId,
            workingDirectory: workingDirectory,
            repositoryRoot: worktreeRepository,
            baseCommit: worktreeBaseCommit
        )
    }

    /// Tools every model-backed run carries on top of whatever its preset or
    /// the caller asked for.
    ///
    /// `complete` is how a run ENDS: the worker prompt tells the model to call
    /// it, and `endsAgentRun` keys the completion on it. A run whose tool set
    /// omitted it could only stop calling tools (reported as a completion it
    /// never declared) or die on its step budget, which is exactly what
    /// happened while the preset surfaces alone decided the run's tools. Plan
    /// mode still withholds it, because there a plain-text plan is the
    /// deliverable.
    static let alwaysAvailableToolNames: [String] = ["complete"]

    static func resolveTools(
        _ request: AgentSpawnRequest,
        definition: AgentPresetDefinition
    ) throws -> [String] {
        // A run's harness is `makeRunScoped()`, which installs the built-ins
        // only. Validating here against the coordinator's catalog accepted an
        // `mcp__…` name the run could never configure, so the approved spawn
        // failed at start instead of at the card.
        let catalog = AgentHarness.runScopedToolNames()
        // An EXPLICIT empty list is not the same as an omitted one. It used to
        // silently fall back to the preset, so a caller that asked for no
        // tools got a writer. Refuse it by name instead: a model-backed run
        // with no tools cannot even call `complete`, so there is no run to
        // start.
        if let explicit = request.explicitToolNames, explicit.isEmpty {
            throw AgentSpawnError.noBackend(
                "This spawn set allowed_tools to an empty list. A model-backed run with no tools cannot finish or change anything; omit allowed_tools to use the preset's tool set."
            )
        }
        let requested = request.explicitToolNames ?? definition.toolNames
        var seen = Set<String>()
        var resolved: [String] = []
        for raw in requested {
            let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { continue }
            if AgentHarness.runScopedForbiddenToolNames.contains(name) {
                throw AgentSpawnError.forbiddenTool(name)
            }
            guard catalog.contains(name) else {
                if AgentHarness.shared.tools[name] != nil {
                    throw AgentSpawnError.notAvailableToRuns(name)
                }
                throw AgentSpawnError.unknownTool(name)
            }
            // Plan mode's promise covers the work it delegates: advertise only
            // what the run's execution path would actually allow.
            if PlanModePolicy.withholds(name, in: request.parentMode) { continue }
            if seen.insert(name).inserted {
                resolved.append(name)
            }
        }
        guard !resolved.isEmpty else {
            throw AgentSpawnError.noBackend("Every tool this run asked for is withheld in the current mode, so there is nothing it could do.")
        }
        for name in alwaysAvailableToolNames where !seen.contains(name) {
            guard catalog.contains(name) else { continue }
            if PlanModePolicy.withholds(name, in: request.parentMode) { continue }
            resolved.append(name)
        }
        return resolved.sorted()
    }

    /// Why this spawn's cost ceiling cannot bind, or nil when it can.
    ///
    /// Pure so a test can price a run without a backend: a ceiling needs a cloud
    /// model AND a recorded price for it. A local, scripted or unpriced backend
    /// reports `costUSD` 0 no matter how many tokens it bills, which makes the
    /// ceiling inert rather than strict.
    static func costCeilingProblem(
        budget: AgentBudget,
        descriptor: AgentBackendDescriptor,
        defaults: UserDefaults = .standard
    ) -> AgentSpawnError? {
        guard let ceiling = budget.maxCostUSD, ceiling > 0 else { return nil }
        guard descriptor.kind == .cloud else {
            // A local or pipeline backend reports no dollar cost at all, so
            // the ceiling is inert for a reason recording a price cannot
            // fix. The old message sent the operator to Cloud Providers.
            return .costCeilingOnUnpricedBackend(descriptor.displayName)
        }
        guard let providerId = descriptor.providerId,
              let modelId = descriptor.modelId,
              CloudPricingStore.pricing(forModel: modelId, providerId: providerId, defaults: defaults) != nil else {
            return .costCeilingWithoutPricing(descriptor.displayName)
        }
        return nil
    }

    static func resolveBackend(
        _ request: AgentSpawnRequest,
        task: String,
        runId: UUID,
        sessionId: UUID,
        defaults: UserDefaults
    ) throws -> (any AgentModelBackend, AgentSchedulingClass, AgentBackendDescriptor) {
        let engine = CompletionEngine.shared
        let hasLocalModel = engine.loadedModelInfo != nil && engine.promptModelConfig != nil
        let cloudPreference = SubagentBackendSettings.preferredCloudModel(defaults: defaults)

        func makeLocal() throws -> (any AgentModelBackend, AgentSchedulingClass, AgentBackendDescriptor) {
            let estimate = LocalAgentBackend.spawnPromptTokenEstimate(
                task: task,
                contextSummary: request.contextSummary
            )
            let backend = try LocalAgentBackend.make(
                runId: runId,
                sessionId: sessionId,
                thinking: request.thinking,
                promptTokenEstimate: estimate,
                // The first pass shows the number the card was priced with.
                // An OVERRIDE must be judged against a CURRENT estimate, or a
                // card left open while the pin or the measured rate moved
                // would approve a number that is no longer true; `admission`
                // compares the recomputed estimate against `approvedEstimate`.
                estimatedCost: request.operatorOverride ? nil : request.localCostEstimate,
                operatorOverride: request.operatorOverride,
                approvedEstimate: request.approvedLocalCost,
                defaults: defaults
            )
            return (backend, .local, backend.descriptor)
        }

        func makeCloud() throws -> (any AgentModelBackend, AgentSchedulingClass, AgentBackendDescriptor) {
            switch cloudPreference {
            case .configured(let cloudAddress):
                switch CloudAgentBackend.make(address: cloudAddress) {
                case .success(let backend):
                    let providerId = backend.descriptor.providerId ?? "cloud"
                    return (backend, .cloud(providerId: providerId), backend.descriptor)
                case .failure(let failure):
                    // A stored preference that parses but cannot run (for
                    // example a provider with no stored key) refuses with
                    // the provider named; it is never swapped for the default.
                    throw failure
                }
            case .unresolvable(let stored):
                throw AgentSpawnError.noBackend(
                    "The subagent Cloud model setting names '\(stored)', which is not a configured cloud model. "
                        + "Fix or clear it in Settings under Agent & Tools, or load a local checkpoint."
                )
            case .noneConfigured:
                throw AgentSpawnError.noBackend(
                    "No cloud model is configured for subagents. Add a provider and a model in Settings, or load a local checkpoint."
                )
            }
        }

        switch request.backendPreference {
        case .local:
            guard hasLocalModel else {
                throw AgentSpawnError.noBackend("This spawn asked for the local model, but no checkpoint is loaded.")
            }
            return try makeLocal()
        case .cloud:
            return try makeCloud()
        case .auto:
            if hasLocalModel {
                return try makeLocal()
            }
            return try makeCloud()
        }
    }
}
