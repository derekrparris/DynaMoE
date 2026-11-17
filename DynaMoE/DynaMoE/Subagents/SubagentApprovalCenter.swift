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

    /// Asks the operator. `honorsTurbo` is false for decisions Turbo Mode must
    /// not make on the operator's behalf — the local cost refusal is the one
    /// case: bypassing routine writes is the point of Turbo, spending more than
    /// the configured re-prefill budget is not.
    func request(_ request: AgentApprovalRequest, honorsTurbo: Bool = true) async -> Bool {
        if honorsTurbo,
           TurboModeSettings.isEnabled,
           AgentHarness.turboModeMayBypassApproval(
               toolName: request.toolName,
               arguments: request.arguments,
               mode: .code
           ) {
            return true
        }
        let entry = PendingSubagentApproval(id: UUID(), request: request)
        pending.append(entry)
        SubagentManager.shared.isDrawerOpen = true
        defer { pending.removeAll { $0.id == entry.id } }
        return await ActionApprovalManager.shared.waitForApproval(id: entry.id)
    }

    func approve(_ id: UUID) {
        _ = ActionApprovalManager.shared.approve(id: id)
    }

    func reject(_ id: UUID) {
        _ = ActionApprovalManager.shared.reject(id: id)
    }

    /// The card's own description of what is at stake.
    static func confirmation(for request: AgentApprovalRequest) -> PendingActionConfirmation {
        PendingActionConfirmation(
            toolName: request.toolName,
            arguments: request.arguments,
            targetSummary: request.reason,
            destinationSummary: request.workingDirectory,
            commandPreview: request.arguments["command"]
        )
    }
}
