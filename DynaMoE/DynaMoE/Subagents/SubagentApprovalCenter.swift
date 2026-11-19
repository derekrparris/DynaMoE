//
//  SubagentApprovalCenter.swift
//  DynaMoE
//
//  The operator's half of a run-scoped approval policy. See SUBAGENTS.md §7.
//

import Foundation
import Combine
import SwiftUI

nonisolated struct PendingSubagentApproval: Identifiable, Sendable {
    let id: UUID
    let request: AgentApprovalRequest
}

/// Collects approval requests from subagent runs and resolves them with the
/// same continuation machinery the coordinator's tool cards use.
///
/// A run is a background object, so its approval cannot ride the coordinator's
/// tool timeline: the coordinator may already have finished the step that
/// spawned it. The Subagent Inspector renders these instead, which keeps every write
/// behind a card without inventing a second approval store.
@MainActor
final class SubagentApprovalCenter: ObservableObject {
    static let shared = SubagentApprovalCenter()

    @Published private(set) var pending: [PendingSubagentApproval] = []

    private init() {}

    static let approvalTimeoutKey = "dynamoe_subagent_approval_timeout"
    static let defaultApprovalTimeout: TimeInterval = 600

    /// Bounded so a card nobody answers cannot hold a scheduler ticket (and a
    /// run's engine work) for the rest of the process's life.
    static func resolvedApprovalTimeout(defaults: UserDefaults = .standard) -> TimeInterval {
        guard let stored = defaults.object(forKey: approvalTimeoutKey) as? Double,
              stored > 0 else { return defaultApprovalTimeout }
        return stored
    }

    /// Asks the operator. `honorsTurbo` is false for decisions Turbo Mode must
    /// not make on the operator's behalf — the local cost refusal is the one
    /// case: bypassing routine writes is the point of Turbo, spending more than
    /// the configured re-prefill budget is not. Landing a run's work is the
    /// other: the SUBAGENTS.md contract gives every apply its own card, and a
    /// blanket switch must not insert a patch with no human in the loop.
    ///
    /// A containment-escape card is the third case, enforced here rather than
    /// at the call site so no future caller can forget it: Turbo Mode's
    /// contract is "stop confirming routine writes inside the sandbox", and an
    /// escape from the configuration presented as safe — an out-of-workspace
    /// write, a shell with its own `cwd`, a `.git` pointer rewrite — is
    /// exactly the write the operator would most want to see.
    func request(_ request: AgentApprovalRequest, honorsTurbo: Bool = true) async -> Bool {
        if Self.turboBypasses(
            request: request,
            honorsTurbo: honorsTurbo,
            turboEnabled: TurboModeSettings.isEnabled
        ) {
            return true
        }
        let entry = PendingSubagentApproval(id: UUID(), request: request)
        pending.append(entry)
        SubagentManager.shared.isDrawerOpen = true
        // The card is visible and the run is blocked on it, so the drawer must
        // say so. Without this a run waiting on the operator read as `.running`
        // for as long as the card sat unanswered.
        SubagentManager.shared.noteApprovalWait(for: request)
        defer {
            pending.removeAll { $0.id == entry.id }
            SubagentManager.shared.noteApprovalResolved(runId: request.runId)
        }

        let timeout = min(Self.resolvedApprovalTimeout(), 86_400)
        let approved = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await ActionApprovalManager.shared.waitForApproval(id: entry.id)
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        if !approved {
            // Resolve the card even when the timeout won, so a late click
            // cannot leave a decision for an id nobody is waiting on.
            ActionApprovalManager.shared.reject(id: entry.id)
        }
        return approved
    }

    func approve(_ id: UUID) {
        _ = ActionApprovalManager.shared.approve(id: id)
    }

    func reject(_ id: UUID) {
        _ = ActionApprovalManager.shared.reject(id: id)
    }

    /// Whether Turbo Mode may skip the card for this request, as a pure
    /// function of the inputs so a test can pin every rule without flipping
    /// the real Turbo preference (the switch has exactly one owner).
    ///
    /// Landing a run's work, a containment escape, a caller that opts out via
    /// `honorsTurbo`, and Turbo being off are each sufficient to keep the
    /// card; `turboModeMayBypassApproval` owns the per-tool rules (app
    /// preferences, the SubQ credential) on top.
    static func turboBypasses(
        request: AgentApprovalRequest,
        honorsTurbo: Bool,
        turboEnabled: Bool
    ) -> Bool {
        guard honorsTurbo,
              request.toolName != "apply_subagent_changes",
              !request.containmentEscape,
              turboEnabled else { return false }
        return AgentHarness.turboModeMayBypassApproval(
            toolName: request.toolName,
            arguments: request.arguments,
            mode: .code
        )
    }

    /// The card's own description of what is at stake.
    static func confirmation(for request: AgentApprovalRequest) -> PendingActionConfirmation {
        PendingActionConfirmation(
            toolName: request.toolName,
            arguments: request.arguments,
            targetSummary: request.reason,
            destinationSummary: AgentHarness.approvalDestination(
                toolName: request.toolName,
                arguments: request.arguments,
                workingDirectory: request.workingDirectory
            ),
            commandPreview: request.arguments["command"]
        )
    }
}
