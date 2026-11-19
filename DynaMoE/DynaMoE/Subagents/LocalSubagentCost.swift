//
//  LocalSubagentCost.swift
//  DynaMoE
//
//  Local prefill cost measurement and the spawn refusal it feeds.
//  See SUBAGENTS.md §6 and invariant 10.
//

import Foundation

/// The prefill rate the app last observed, in prompt tokens per second.
///
/// The local backend has to price a run BEFORE it starts, and the only honest
/// input is what this machine measured. The engine already computes the rate for
/// its prefill progress line, so that number is recorded here rather than
/// estimated from a constant: a 16 GB reference machine and a 128 GB one differ
/// by more than an order of magnitude on the same prompt.
nonisolated enum PrefillThroughput {
    /// Used only when this process has not prefilled anything yet. Deliberately
    /// pessimistic (the reference M1 Pro / 16 GB machine measures ~5 tokens/s on
    /// a long prompt): an unmeasured estimate must over-price the re-prefill, so
    /// the refusal errs toward asking the operator rather than starting a run
    /// that would spend minutes re-prefilling afterwards.
    static let fallbackTokensPerSecond: Double = 5.0

    private static let lock = NSLock()
    private static var measured: Double?

    static func note(tokensPerSecond: Double) {
        guard tokensPerSecond.isFinite, tokensPerSecond > 0 else { return }
        lock.lock()
        measured = tokensPerSecond
        lock.unlock()
    }

    static var recentTokensPerSecond: Double? {
        lock.lock()
        defer { lock.unlock() }
        return measured
    }

    static func resolvedTokensPerSecond() -> Double {
        recentTokensPerSecond ?? fallbackTokensPerSecond
    }

    /// Forgets the measured rate. A model switch changes what a prompt costs
    /// by orders of magnitude: keeping the previous checkpoint's rate would
    /// price a 510 GB model's spawn from a small model's speed.
    static func reset() {
        lock.lock()
        measured = nil
        lock.unlock()
    }

    static func resetForTesting() {
        reset()
    }
}

/// What a local subagent run is expected to cost the coordinator.
///
/// A run holds the engine (one turn at a time) and records its own session on the
/// single global prefix pin, so the coordinator's next turn cannot reuse the
/// prompt it had pinned: it re-prefills `coordinatorPrefixTokens` before it can
/// answer. That is the number invariant 10 gates on.
nonisolated struct LocalSubagentCostEstimate: Equatable, Sendable {
    /// The coordinator's pinned prefix, which the run's session displaces.
    var coordinatorPrefixTokens: Int
    /// The run's own first prompt, priced for the spawn card. It never decides
    /// the refusal: the run's cost is paid by the run, the re-prefill by the
    /// coordinator.
    var runPromptTokens: Int
    var tokensPerSecond: Double
    var limitSeconds: Double

    static let defaultLimitSeconds: Double = 120
    static let limitDefaultsKey = "dynamoe_subagent_local_cost_limit"

    static func configuredLimitSeconds(defaults: UserDefaults = .standard) -> Double {
        let stored = defaults.object(forKey: limitDefaultsKey) as? Double
        guard let stored, stored > 0 else { return defaultLimitSeconds }
        return stored
    }

    /// Prices a spawn. `tokensPerSecond` must already be resolved (measured, or
    /// the fallback); a non-positive rate is treated as one token per second so
    /// the estimate is large rather than infinite.
    static func make(
        coordinatorPrefixTokens: Int,
        runPromptTokens: Int,
        tokensPerSecond: Double,
        limitSeconds: Double = LocalSubagentCostEstimate.defaultLimitSeconds
    ) -> LocalSubagentCostEstimate {
        let rate = tokensPerSecond > 0 ? tokensPerSecond : 1
        return LocalSubagentCostEstimate(
            coordinatorPrefixTokens: max(0, coordinatorPrefixTokens),
            runPromptTokens: max(0, runPromptTokens),
            tokensPerSecond: rate,
            limitSeconds: limitSeconds > 0 ? limitSeconds : defaultLimitSeconds
        )
    }

    var coordinatorReprefillSeconds: Double {
        Double(coordinatorPrefixTokens) / tokensPerSecond
    }

    var runPrefillSeconds: Double {
        Double(runPromptTokens) / tokensPerSecond
    }

    var exceedsLimit: Bool {
        coordinatorReprefillSeconds > limitSeconds
    }

    /// The message a refused spawn shows. It names the numbers, because the
    /// operator's next action is either to raise the limit or to send the same
    /// work to a cloud model.
    var refusalMessage: String? {
        guard exceedsLimit else { return nil }
        let prefix = "The coordinator would re-prefill \(coordinatorPrefixTokens) tokens after this run "
        let cost = String(format: "about %.0f s at %.1f tok/s", coordinatorReprefillSeconds, tokensPerSecond)
        let limit = String(format: "over the %.0f s local limit", limitSeconds)
        return prefix + cost + " — " + limit
            + ". Run it on a cloud backend, raise the limit, or approve the cost for this spawn."
    }

    /// A one-line summary for the spawn card, regardless of refusal.
    var summary: String {
        String(
            format: "coordinator re-prefill %.0f s (%d tokens at %.1f tok/s), run prefill %.0f s",
            coordinatorReprefillSeconds,
            coordinatorPrefixTokens,
            tokensPerSecond,
            runPrefillSeconds
        )
    }

    /// The admission decision. The backend asks this in `make`, so there is no
    /// path that starts a local run without consulting it; the override is the
    /// operator's permission, carried in from the spawn card (invariant 10).
    ///
    /// An override approves THE NUMBER THE CARD SHOWED, not the idea of a
    /// local run. The prefix pin and the measured prefill rate can both move
    /// while the card is open, and admitting a larger estimate because the old
    /// card was clicked is consent the operator never gave. `approvedEstimate`
    /// is that number; nil keeps the older direct-call meaning (a caller with
    /// no card, e.g. a test or the opt-in published-turn harness).
    /// Whether an operator approval of `approved` covers this estimate.
    /// Re-measurements that are equal or cheaper (within floating-point noise)
    /// are still covered; only a cost increase refuses the spawn.
    func isCoveredBy(_ approved: LocalSubagentCostEstimate) -> Bool {
        let tolerance = 1e-6
        return coordinatorReprefillSeconds <= approved.coordinatorReprefillSeconds + tolerance
            && runPrefillSeconds <= approved.runPrefillSeconds + tolerance
    }

    func admits(
        operatorOverride: Bool,
        approvedEstimate: LocalSubagentCostEstimate? = nil
    ) -> Bool {
        guard exceedsLimit else { return true }
        guard operatorOverride else { return false }
        guard let approvedEstimate else { return true }
        return isCoveredBy(approvedEstimate)
    }

    /// Why an approved estimate no longer covers this spawn, or nil when it
    /// does.
    func approvalDriftMessage(approved: LocalSubagentCostEstimate) -> String? {
        guard !isCoveredBy(approved) else { return nil }
        return "The estimated cost changed after the card was shown (was \(approved.summary); now \(summary)). "
            + "The approval does not cover the new estimate; the run was not started. Try the spawn again to approve the current number."
    }
}
