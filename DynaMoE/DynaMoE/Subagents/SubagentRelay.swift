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
    static func directive(reports: [String]) -> String {
        let context = reports.joined(separator: "\n\n")
        return """
        <system>
        A delegated subagent finished while you were idle. You are the coordinator: the user only talks to you, and you are responsible for presenting subagent results. Answer the user's message below normally, folding this report in where it is relevant; if it is unrelated, mention the finished run briefly at the end. Do not claim you did the work yourself.

        \(context)
        </system>
        """
    }
}
