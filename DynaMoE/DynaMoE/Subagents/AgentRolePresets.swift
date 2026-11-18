//
//  AgentRolePresets.swift
//  DynaMoE
//
//  Roles as data: what a preset may touch, how it is isolated, and how much it
//  may spend. See SUBAGENTS.md §9.
//

import Foundation

/// The tools a run can be given, grouped so a preset reads as a capability.
nonisolated enum AgentToolSurface {
    static let read: [String] = [
        "file_read", "find_files", "grep_search", "codebase_search",
        "find_symbol_definition", "find_references"
    ]
    static let web: [String] = ["web_search", "web_fetch"]
    static let skills: [String] = ["skill_load"]
    static let write: [String] = ["file_write", "file_edit"]
    static let shell: [String] = ["shell_run"]
    static let gitRead: [String] = ["git_status", "git_diff"]
    static let gitWrite: [String] = ["git_commit"]
    static let lint: [String] = ["lint_diagnostics"]

    static let writeCapable: Set<String> = Set(write + shell + gitWrite)
}

nonisolated struct AgentPresetDefinition: Equatable, Sendable {
    var preset: AgentRolePreset
    var toolNames: [String]
    var maxSteps: Int
    var maxCompletionTokens: Int
    var maxWallClockSeconds: Int

    /// True when this preset can change the tree, the repository or the system.
    var canMutate: Bool {
        !Set(toolNames).isDisjoint(with: AgentToolSurface.writeCapable)
    }
}

nonisolated enum AgentRolePresetCatalog {
    /// The doc's table, as data. A preset never carries a pipeline: the loop and
    /// the tools are the same for every role; the role changes what it may use
    /// and how long it may work.
    static func definition(for preset: AgentRolePreset) -> AgentPresetDefinition {
        switch preset {
        case .explore:
            return AgentPresetDefinition(
                preset: .explore,
                toolNames: AgentToolSurface.read + AgentToolSurface.web + AgentToolSurface.skills,
                maxSteps: 16,
                maxCompletionTokens: 16_000,
                maxWallClockSeconds: 420
            )
        case .implementer:
            return AgentPresetDefinition(
                preset: .implementer,
                toolNames: AgentToolSurface.read + AgentToolSurface.write + AgentToolSurface.shell
                    + AgentToolSurface.gitRead + AgentToolSurface.gitWrite + AgentToolSurface.lint,
                maxSteps: 40,
                maxCompletionTokens: 32_000,
                maxWallClockSeconds: 900
            )
        case .verifier:
            return AgentPresetDefinition(
                preset: .verifier,
                toolNames: AgentToolSurface.read + AgentToolSurface.shell + AgentToolSurface.lint,
                maxSteps: 24,
                maxCompletionTokens: 24_000,
                maxWallClockSeconds: 600
            )
        case .reviewer:
            return AgentPresetDefinition(
                preset: .reviewer,
                toolNames: AgentToolSurface.read + AgentToolSurface.gitRead,
                maxSteps: 16,
                maxCompletionTokens: 16_000,
                maxWallClockSeconds: 420
            )
        case .general, .custom:
            return AgentPresetDefinition(
                preset: preset,
                toolNames: AgentToolSurface.read,
                maxSteps: 20,
                maxCompletionTokens: 20_000,
                maxWallClockSeconds: 540
            )
        }
    }

    /// A role title from the model, mapped onto a preset.
    ///
    /// Matches the old archetype vocabulary so a prompt that asks for a
    /// "Codebase Researcher" or a "Test Runner" keeps meaning what it meant,
    /// but the run behind it is a real agent loop rather than a canned pipeline.
    static func match(role: String) -> AgentRolePreset {
        let lowered = role.lowercased()
        if lowered.contains("test") || lowered.contains("runner") || lowered.contains("verif") || lowered.contains("qa") {
            return .verifier
        }
        if lowered.contains("review") || lowered.contains("audit") {
            return .reviewer
        }
        if lowered.contains("implement") || lowered.contains("shader") || lowered.contains("optimi")
            || lowered.contains("refactor") || lowered.contains("document") {
            return .implementer
        }
        if lowered.contains("research") || lowered.contains("search") || lowered.contains("explore")
            || lowered.contains("investigat") {
            return .explore
        }
        return .general
    }

    static func preset(named raw: String?, role: String) -> AgentRolePreset {
        guard let raw, !raw.isEmpty else { return match(role: role) }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "explore": return .explore
        case "implementer": return .implementer
        case "verifier": return .verifier
        case "reviewer": return .reviewer
        case "general": return .general
        case "custom": return .custom
        default: return match(role: role)
        }
    }

    /// The run's budget: the preset's defaults, with the caller's explicit
    /// numbers winning. A budget is enforcement, not a suggestion, so a value
    /// the caller gave is never silently clamped to the preset — but the
    /// values arrive from a MODEL's spawn arguments, so each is clamped to
    /// `AgentBudget`'s own bounds: an inflated `max_wall_clock_seconds` would
    /// trap the loop's nanosecond arithmetic and take the app down, and a
    /// runaway `max_steps` would keep a run alive for days.
    static func budget(for definition: AgentPresetDefinition, overrides: AgentBudgetOverrides?) -> AgentBudget {
        var budget = AgentBudget(
            maxSteps: definition.maxSteps,
            maxCompletionTokens: definition.maxCompletionTokens,
            maxWallClockSeconds: definition.maxWallClockSeconds,
            maxCostUSD: nil
        )
        guard let overrides else { return budget }
        if let value = overrides.maxSteps {
            budget.maxSteps = min(max(1, value), AgentBudget.maximumSteps)
        }
        if let value = overrides.maxCompletionTokens {
            budget.maxCompletionTokens = min(max(1, value), AgentBudget.maximumCompletionTokens)
        }
        if let value = overrides.maxWallClockSeconds {
            budget.maxWallClockSeconds = min(max(1, value), AgentBudget.maximumWallClockSeconds)
        }
        if let value = overrides.maxCostUSD {
            // A non-finite override is refused by the spawn resolver before it
            // reaches here; folded anyway so no OTHER caller can turn `.nan`
            // into a "$0 ceiling" (Swift's `max(0, .nan)` returns 0) or let an
            // infinite ceiling silently disable the guard.
            budget.maxCostUSD = value.isFinite ? max(0, value) : nil
        }
        return budget
    }

    /// Read-only runs may be automatic; a run that can write on the main tree
    /// asks before every change, one that writes inside its own worktree asks
    /// once at spawn, and plan mode denies writes outright.
    ///
    /// `definition`'s own tool names are only a DEFAULT: a spawn may replace
    /// them with `allowed_tools`, so the decision reads the tools the run was
    /// actually given. Deriving it from the preset meant `preset:"explore"` +
    /// `allowed_tools:["file_write"]` was classified read-only, approved with
    /// no card, and then denied every write by its own run scope — with a
    /// worktree, an empty one after the operator had approved a writer.
    static func approvalPolicy(
        toolNames: [String],
        mode: AgentMode,
        isolated: Bool = false
    ) -> AgentApprovalPolicy {
        if mode == .plan { return .deny }
        let canMutate = !Set(toolNames).isDisjoint(with: AgentToolSurface.writeCapable)
        guard canMutate else { return .auto }
        return isolated ? .spawnOnly : .everyStateChange
    }

    /// The preset-only form, for callers that have no resolved tool list.
    static func approvalPolicy(
        preset definition: AgentPresetDefinition,
        mode: AgentMode,
        isolated: Bool = false
    ) -> AgentApprovalPolicy {
        approvalPolicy(toolNames: definition.toolNames, mode: mode, isolated: isolated)
    }
}

/// The budget fields a spawn may override, all optional.
nonisolated struct AgentBudgetOverrides: Equatable, Sendable {
    var maxSteps: Int?
    var maxCompletionTokens: Int?
    var maxWallClockSeconds: Int?
    var maxCostUSD: Double?

    init(
        maxSteps: Int? = nil,
        maxCompletionTokens: Int? = nil,
        maxWallClockSeconds: Int? = nil,
        maxCostUSD: Double? = nil
    ) {
        self.maxSteps = maxSteps
        self.maxCompletionTokens = maxCompletionTokens
        self.maxWallClockSeconds = maxWallClockSeconds
        self.maxCostUSD = maxCostUSD
    }
}
