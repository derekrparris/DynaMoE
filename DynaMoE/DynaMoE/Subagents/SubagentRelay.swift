//
//  SubagentRelay.swift
//  DynaMoE
//
//  The local-backend start-of-turn relay. See SUBAGENTS.md §11.
//

import Foundation

/// The directive that delivers unread subagent reports to a coordinator whose
/// prompt is rendered locally.
///
/// The cloud path has its own (`CloudSubagentRelay`): it rides the trailing
/// position of every request so the provider's cached prefix stays
/// byte-identical. A local render has no cached prefix to protect, and every
/// family renders a `.user` turn, so one hidden message ahead of the user's
/// message reaches the model on every template without a per-family branch.
nonisolated enum LocalSubagentRelay {
    /// Total characters of report text one directive carries. Each report is
    /// already bounded on its own, but the sum across a burst of completions
    /// goes into every later prompt, so it needs a bound of its own.
    static let maximumContextCharacters = 24_000

    static func directive(reports: [String]) -> String {
        let context = boundedContext(
            reports.map { PromptMarkers.sanitizeControlMarkers($0) }
        )
        return """
        <system>
        A delegated subagent finished while you were idle. You are the coordinator: the user only talks to you, and you are responsible for presenting subagent results. Answer the user's message below normally, folding this report in where it is relevant; if it is unrelated, mention the finished run briefly at the end. Do not claim you did the work yourself.

        \(context)
        </system>
        """
    }

    /// Joins reports up to the budget, then says how many were elided rather
    /// than truncating mid-report: a coordinator that reads a cut-off report as
    /// whole is worse off than one told to fetch the rest.
    static func boundedContext(_ reports: [String]) -> String {
        var kept: [String] = []
        var total = 0
        for report in reports {
            let cost = report.count + (kept.isEmpty ? 0 : 2)
            guard total + cost <= maximumContextCharacters else { break }
            kept.append(report)
            total += cost
        }
        let omitted = reports.count - kept.count
        var context = kept.joined(separator: "\n\n")
        if omitted > 0 {
            let which = omitted == 1 ? "it" : "them"
            context += "\n\n[\(omitted) more report\(omitted == 1 ? "" : "s") omitted here for length; read \(which) with get_subagent_status.]"
        }
        return context
    }
}
