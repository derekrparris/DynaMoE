//
//  AgentStepBudget.swift
//  DynaMoE
//
//  The Max Multi-Step Agent Iterations budget, resolved once for both agent
//  loops (the weights path in ContentView and the cloud path's
//  `CloudAgentLoopPolicy`). Pure functions so the "no limit" case is testable
//  without a model, a provider, or a running turn.
//
//  Two stored values, not one sentinel: `dynamoe_agent_max_steps` keeps the
//  user's last finite number and `dynamoe_agent_max_steps_unlimited` says
//  whether it applies. A `0` in the number would have to mean "no limit",
//  which loses the previous value the moment the switch goes back off — the
//  same reason Auto-Delete pairs a toggle with its retention count.
//

import Foundation

// MARK: - Step Budget

/// Resolves and applies the agent step cap. `nil` means no cap: the run ends
/// on the model's own `complete`, a tool-loop guard, a KV capacity error, or
/// Stop, but never on a step count.
public nonisolated enum AgentStepBudget {

    /// Persisted switch for "No limit". Written by the Settings toggle.
    public static let unlimitedDefaultsKey = "dynamoe_agent_max_steps_unlimited"

    /// Persisted finite step count. Written by the Settings slider and kept
    /// while `unlimitedDefaultsKey` is on, so the switch can be turned back off
    /// without the user re-picking a number.
    public static let finiteStepsDefaultsKey = "dynamoe_max_agent_steps"

    /// The slider's starting value, and the fallback when the key is absent.
    public static let defaultFiniteSteps = 15

    /// The cap as the app currently stores it.
    ///
    /// Read straight from `UserDefaults` rather than through the settings view's
    /// `@AppStorage` mirrors: the generation loops call this between steps from
    /// inside `Task.detached`, and `@AppStorage`'s wrapper is main-actor
    /// isolated, so going through it would need an actor hop per comparison.
    /// Reading defaults at call time also keeps a settings change live — the
    /// same behaviour `CompletionEngine` gets from the `dynamoe_*` keys it
    /// resolves per turn.
    public static func resolvedFromDefaults(_ defaults: UserDefaults = .standard) -> Int? {
        resolvedCap(
            unlimited: defaults.bool(forKey: unlimitedDefaultsKey),
            configured: (defaults.object(forKey: finiteStepsDefaultsKey) as? Int) ?? defaultFiniteSteps
        )
    }

    /// The cap a turn should enforce.
    ///
    /// A finite configured value floors at 1: `1` is the smallest budget that
    /// still lets the model answer, and a hand-written `defaults` value of `0`
    /// or a negative would otherwise be indistinguishable from a switch that
    /// ends the run before it can call a tool.
    public static func resolvedCap(unlimited: Bool, configured: Int) -> Int? {
        unlimited ? nil : max(1, configured)
    }

    /// Whether `step` may be followed by another turn.
    ///
    /// `step` is zero-based, so a cap of N allows steps 0…N-1 — the same
    /// `step + 1 < cap` arithmetic both loops used inline before this existed.
    /// Written as a subtraction so a cap of `Int.max` (see `compareCap`) can
    /// never overflow the comparison. `Int.min` is the other edge, where the
    /// subtraction would trap, and no cap that small allows a step.
    public static func allowsAnotherStep(step: Int, cap: Int?) -> Bool {
        guard let cap else { return true }
        guard cap != Int.min else { return false }
        return step < cap - 1
    }

    /// The complement of `allowsAnotherStep`, for call sites that phrase the
    /// question as "has this run run out of steps".
    public static func isExhausted(step: Int, cap: Int?) -> Bool {
        !allowsAnotherStep(step: step, cap: cap)
    }

    /// Bridges the optional cap into an API that still takes an `Int` —
    /// `CloudAgentLoopPolicy.shouldContinue(maxSteps:)`. `Int.max` is a safe
    /// stand-in: `step` is a live iteration count, so it can never approach it.
    public static func compareCap(_ cap: Int?) -> Int {
        cap ?? Int.max
    }

    /// The value shown beside the settings slider: `"15 steps"` or `"1 step"`.
    /// Only a finite cap reaches here: the settings row's own "No limit"
    /// switch is what says a run is uncapped, so this label never repeats it.
    public static func statusLabel(cap: Int) -> String {
        cap == 1 ? "1 step" : "\(cap) steps"
    }

    /// The run-end status line for a run that spent a finite step budget.
    ///
    /// A run with no cap cannot reach this: `allowsAnotherStep` is true at
    /// every step, so only `complete` or a tool-loop guard ends it, and each
    /// of those has its own status line. Naming the cap here therefore always
    /// names a limit that applied.
    public static func maxStepsReachedNotice(cap: Int) -> String {
        "✨ Agent completed maximum allowed steps (\(cap))."
    }

    /// The run-end status line for a run one of the harness guards stopped
    /// (failed-call streak, repeat guard, or the search budget). Named apart
    /// from `maxStepsReachedNotice` because the step budget is not the reason
    /// it ended — which matters once "no limit" makes that the only way a
    /// tool round can end without `complete`.
    public static let loopGuardStoppedNotice = "🛡️ Agent stopped by the tool loop guard."
}
