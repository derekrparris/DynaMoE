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
    /// The most cards that may wait on the operator at once.
    static let maximumPendingCards = 32

    /// Whether a new card would exceed the queue. Pure so the bound is pinned
    /// without filling the operator's real list.
    static func refusesCard(pendingCount: Int) -> Bool {
        pendingCount >= maximumPendingCards
    }

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
    func request(
        _ request: AgentApprovalRequest,
        honorsTurbo: Bool = true,
        timeout overrideTimeout: TimeInterval? = nil,
        scope overrideScope: ActionApprovalScope? = nil
    ) async -> Bool {
        if Self.turboBypasses(
            request: request,
            honorsTurbo: honorsTurbo,
            turboEnabled: TurboModeSettings.isEnabled
        ) {
            return true
        }
        let entry = PendingSubagentApproval(id: UUID(), request: request)
        // The card is visible before the waiter registers, so its scope is
        // recorded here: a Stop that owns the RUN declines it, and a Stop in
        // another conversation leaves it alone. An operator-initiated apply
        // supplies its own scope because a Stop did not raise that card.
        let scope = overrideScope ?? ActionApprovalScope.subagentRun(request.runId)
        ActionApprovalManager.shared.register(id: entry.id, scope: scope)
        // Bounded, because nothing else bounds it. A model that loops on a
        // refused tool used to append a card per call for a run nobody is
        // reading; the list is the operator's queue and it is not infinite.
        // Past the cap the answer is an honest refusal rather than a card the
        // pane cannot show.
        guard !Self.refusesCard(pendingCount: pending.count) else {
            DiagLog.warning("subagent.approval", "refusing a card from \(request.toolName): \(pending.count) already pending")
            ActionApprovalManager.shared.unregister(id: entry.id)
            return false
        }
        pending.append(entry)
        SubagentManager.shared.isDrawerOpen = true
        // The card is visible and the run is blocked on it, so the drawer must
        // say so. Without this a run waiting on the operator read as `.running`
        // for as long as the card sat unanswered.
        SubagentManager.shared.noteApprovalWait(for: request)
        defer {
            pending.removeAll { $0.id == entry.id }
            ActionApprovalManager.shared.unregister(id: entry.id)
            // Only when no OTHER card of this run is still pending: a run can
            // have several in flight (a looping model, a tool call plus a
            // local-cost card), and resolving one used to flip the run back to
            // "running" while it was still parked on the next.
            if !pending.contains(where: { $0.request.runId == request.runId }) {
                SubagentManager.shared.noteApprovalResolved(runId: request.runId)
            }
        }

        let timeout = min(overrideTimeout ?? Self.resolvedApprovalTimeout(), 86_400)
        let approved = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await ActionApprovalManager.shared.waitForApproval(id: entry.id, scope: scope)
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            // A click that lands as the timeout fires is not lost: the group
            // returns the FIRST result, so if the sleep won by a hair, the
            // operator's decision is still recorded in the manager and is
            // consumed here instead of being reported as a decline.
            if !first, let decided = ActionApprovalManager.shared.rememberedDecision(id: entry.id) {
                return decided
            }
            return first
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
            mode: .code,
            // `request.arguments` is the card's STRING copy, where an
            // `images` array is a description; the wire form is what lets the
            // upload card stay up.
            rawArguments: request.rawArguments
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
