//
//  ActionConfirmationView.swift
//  DynaMoE
//
//  Phase 5: macOS UI & Human-In-The-Loop (HITL) Safety
//  Interactive confirmation drawer & card in SwiftUI for state-changing operations
//  (file_write, file_edit, shell_run).
//  Includes "Turbo Mode" bypass (when enabled, auto-approves mutations).
//

import SwiftUI

// MARK: - Action Approval Continuation Manager

/// Who a parked decision belongs to, for a cancellation that must not reach
/// another conversation's card.
///
/// A chat's Stop cancels the cards ITS turn raised, including the ones a
/// background subagent it spawned is parked on. `cancelAll` is for a process
/// shutdown, where every card really is going away.
/// `nonisolated` because Stop and cancellation reach it from the tool loop and
/// background run tasks, neither of which shares the main actor.
public nonisolated struct ActionApprovalScope: Hashable, Sendable {
    let rawValue: String

    public static func session(_ id: UUID) -> ActionApprovalScope {
        ActionApprovalScope(rawValue: "s:\(id.uuidString)")
    }

    public static func subagentRun(_ id: UUID) -> ActionApprovalScope {
        ActionApprovalScope(rawValue: "r:\(id.uuidString)")
    }

    /// The operator's own Apply card. It names a run, but a Stop that owns that
    /// run did not raise this card and must not answer it.
    public static func operatorApply(_ id: UUID) -> ActionApprovalScope {
        ActionApprovalScope(rawValue: "a:\(id.uuidString)")
    }
}

/// The one place a human-in-the-loop decision is parked.
///
/// `nonisolated` because a waiter is resumed from the tool loop, from a Stop,
/// and from task cancellation, none of which share an executor. All state is
/// behind `lock`; nothing resumes a continuation while holding it.
public nonisolated final class ActionApprovalManager: @unchecked Sendable {
    public static let shared = ActionApprovalManager()

    private let lock = NSLock()
    private var pendingContinuations: [UUID: CheckedContinuation<Bool, Never>] = [:]
    /// Ids cancelled BEFORE their waiter registered. Without this the cancel
    /// races registration: `onCancel` finds nothing to resume, the waiter then
    /// parks on a continuation nobody will ever resume, and the call stalls for
    /// the rest of the process's life.
    private var cancelledWaits: Set<UUID> = []
    /// Decisions that arrived BEFORE their waiter registered (a fast click on a
    /// freshly rendered card). The waiter consumes its entry on arrival, so a
    /// click can never be lost to that window. Bounded: ids are single-use, so
    /// the oldest entries are dead weight rather than a leak that grows.
    private var pendingDecisions: [UUID: Bool] = [:]
    private var pendingDecisionOrder: [UUID] = []
    /// The scope each id belongs to, recorded when the CARD is rendered (before
    /// the waiter registers) so a scoped cancel can find the click-before-wait
    /// window too. Removed with the decision or by the waiter's own exit.
    private var scopes: [UUID: ActionApprovalScope] = [:]
    /// Bounded because ids are single-use, but comfortably above the number of
    /// cards that can be pending at once (the subagent center caps at 32): a
    /// decision evicted before its waiter registers parks that run until its
    /// timeout, which is the exact failure the memory exists to prevent.
    private static let rememberedDecisionLimit = 64

    private init() {}

    /// Records who an id belongs to. Call before showing its card: the card is
    /// visible (and clickable, and stoppable) before `waitForApproval` has
    /// registered a continuation.
    public func register(id: UUID, scope: ActionApprovalScope) {
        lock.lock()
        scopes[id] = scope
        lock.unlock()
    }

    /// Drops a scope record for an id whose card went away without a waiter
    /// ever registering — and any decision remembered for it, which nothing
    /// could consume once the card is gone (ids are single-use, so it would
    /// otherwise sit until eviction).
    public func unregister(id: UUID) {
        lock.lock()
        scopes.removeValue(forKey: id)
        if pendingDecisions.removeValue(forKey: id) != nil {
            pendingDecisionOrder.removeAll { $0 == id }
        }
        lock.unlock()
    }

    /// A decision remembered for an id whose waiter has not registered yet,
    /// or nil. A caller racing a timeout reads this so a click that landed
    /// first is honoured rather than reported as a decline.
    public func rememberedDecision(id: UUID) -> Bool? {
        lock.lock()
        defer { lock.unlock() }
        return pendingDecisions[id]
    }

    /// Drops an id whose waiter has left. Synchronous on purpose: calling
    /// `lock.lock()` from an `async` frame's `defer` is a Swift 6 error, so the
    /// deferred cleanup goes through this helper.
    private func finishWaiting(id: UUID) {
        lock.lock()
        cancelledWaits.remove(id)
        scopes.removeValue(forKey: id)
        lock.unlock()
    }

    public func waitForApproval(
        id: UUID,
        scope: ActionApprovalScope? = nil
    ) async -> Bool {
        // A pending prompt is part of a running agent operation even though
        // the engine is idle, so a config reload must wait for the decision.
        AgentActivity.shared.begin(.approval)
        defer {
            AgentActivity.shared.end(.approval)
            // A cancellation that lands AFTER `decide` resumed this id inserts
            // it below with nobody left to consume it. The handler cannot run
            // once `withTaskCancellationHandler` returns, so dropping the id
            // here is what keeps this set bounded.
            finishWaiting(id: id)
        }
        if let scope {
            register(id: id, scope: scope)
        }
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.lock()
                if cancelledWaits.remove(id) != nil {
                    lock.unlock()
                    // A stopped turn must not leave a call awaiting a decision
                    // the transcript no longer offers.
                    continuation.resume(returning: false)
                } else if let decided = pendingDecisions.removeValue(forKey: id) {
                    lock.unlock()
                    continuation.resume(returning: decided)
                } else {
                    pendingContinuations[id] = continuation
                    lock.unlock()
                }
            }
        } onCancel: {
            lock.lock()
            if let continuation = pendingContinuations.removeValue(forKey: id) {
                lock.unlock()
                continuation.resume(returning: false)
            } else {
                cancelledWaits.insert(id)
                lock.unlock()
            }
        }
    }

    /// Records a decision. True when it resolved a waiter or was remembered for
    /// one that has not registered yet.
    @discardableResult
    public func approve(id: UUID) -> Bool {
        decide(id: id, approved: true)
    }

    @discardableResult
    public func reject(id: UUID) -> Bool {
        decide(id: id, approved: false)
    }

    /// Declines every card in one scope and nothing else. A Stop in chat A
    /// must not answer chat B's parked card, and it must still win over a
    /// click that arrived before the waiter registered.
    public func cancel(scope: ActionApprovalScope) {
        lock.lock()
        let ids = scopes.filter { $0.value == scope }.map(\.key)
        var resumed: [CheckedContinuation<Bool, Never>] = []
        for id in ids {
            // Only an id with no waiter left needs the cancelled mark: one
            // that was resumed here has nobody to consume it, and ids are
            // single-use, so inserting anyway grew the set for the process's
            // lifetime with every scoped Stop.
            if let continuation = pendingContinuations.removeValue(forKey: id) {
                resumed.append(continuation)
            } else {
                // A card whose waiter has not registered yet must still resolve
                // as declined; the waiter consumes this on arrival.
                cancelledWaits.insert(id)
            }
            pendingDecisions.removeValue(forKey: id)
            pendingDecisionOrder.removeAll { $0 == id }
            scopes.removeValue(forKey: id)
        }
        lock.unlock()
        for continuation in resumed {
            continuation.resume(returning: false)
        }
    }

    public func cancelAll() {
        lock.lock()
        let all = pendingContinuations
        pendingContinuations.removeAll()
        // A decision recorded for a waiter that never registered belongs to the
        // turn being stopped: cancelling wins over a click that arrived first,
        // or a stopped turn would execute the call it was stopped for.
        cancelledWaits.formUnion(pendingDecisions.keys)
        // A scoped card whose waiter has not registered yet is in the same
        // window: it is about to be answered by a turn that is going away.
        cancelledWaits.formUnion(scopes.keys)
        pendingDecisions.removeAll()
        pendingDecisionOrder.removeAll()
        scopes.removeAll()
        lock.unlock()
        for (_, cont) in all {
            cont.resume(returning: false)
        }
    }

    private func decide(id: UUID, approved: Bool) -> Bool {
        lock.lock()
        if let continuation = pendingContinuations.removeValue(forKey: id) {
            lock.unlock()
            continuation.resume(returning: approved)
            return true
        }
        // No waiter yet. Remember the answer so a late registration resolves
        // instead of parking forever; a cancel for this id is now moot.
        cancelledWaits.remove(id)
        if pendingDecisions.updateValue(approved, forKey: id) == nil {
            // A second click on one id replaces the answer rather than adding
            // a second order entry that would evict an unrelated decision.
            pendingDecisionOrder.append(id)
            if pendingDecisionOrder.count > Self.rememberedDecisionLimit {
                let oldest = pendingDecisionOrder.removeFirst()
                pendingDecisions.removeValue(forKey: oldest)
                scopes.removeValue(forKey: oldest)
            }
        }
        lock.unlock()
        return true
    }
}

public struct PendingActionConfirmation: Identifiable, Equatable {
    public let id: UUID
    public let toolName: String
    public let arguments: [String: String]
    public let targetSummary: String
    /// Where the call actually sends data (origin and, for an upload, the file
    /// names). The card used to name only the tool, so a call redirected to a
    /// custom host looked identical to the official one.
    public let destinationSummary: String?
    public let diffPreview: String?
    public let commandPreview: String?

    public init(
        id: UUID = UUID(),
        toolName: String,
        arguments: [String: String],
        targetSummary: String,
        destinationSummary: String? = nil,
        diffPreview: String? = nil,
        commandPreview: String? = nil
    ) {
        self.id = id
        self.toolName = toolName
        self.arguments = arguments
        self.targetSummary = targetSummary
        self.destinationSummary = destinationSummary
        self.diffPreview = diffPreview
        self.commandPreview = commandPreview
    }
}

public struct ActionConfirmationView: View {
    public let confirmation: PendingActionConfirmation
    public let onApprove: () -> Void
    public let onReject: () -> Void
    /// False while the card is still gathering the context the decision needs
    /// (an apply card waits for its run's destination before it can be
    /// approved). Reject stays available: the operator can always decline.
    public var approveEnabled: Bool = true

    @AppStorage("dynamoe_agent_turbo_mode") private var isTurboModeEnabled: Bool = false

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header: Warning & Tool Badge
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.shield.fill")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.orange)

                Text("Action Requires Confirmation")
                    .font(.system(size: 12.5, weight: .bold))
                    .foregroundColor(.primary)

                Spacer()

                // Turbo Mode indicator. It may ENABLE Turbo and never disable it.
                // This used to be a blind toggle drawn as a status badge, and it
                // is the first control in the header: clicking it to turn Turbo
                // on while Turbo was already on turned it OFF. Disabling belongs
                // to the composer pill and Settings, where the state being
                // changed is the thing under the pointer.
                if isTurboModeEnabled {
                    HStack(spacing: 4) {
                        Image(systemName: "bolt.fill")
                            .font(.system(size: 10))
                        Text("Turbo ON")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.18))
                    .foregroundColor(.orange)
                    .cornerRadius(5)
                    .help("Turbo Mode is on. Turn it off from the composer's Turbo pill or Settings.")
                } else {
                    Button(action: {
                        TurboModeSettings.set(true, from: "ActionConfirmationView")
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "bolt.slash")
                                .font(.system(size: 10))
                            Text("Enable Turbo")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.secondary.opacity(0.1))
                        .foregroundColor(.secondary)
                        .cornerRadius(5)
                    }
                    .buttonStyle(.plain)
                    .help("Turn Turbo Mode on so later file writes and commands run without a per-call confirmation. This pending action still needs its own approval, and this control cannot turn Turbo off.")
                }
            }

            // Summary of Target / Action
            HStack(spacing: 6) {
                Text(confirmation.toolName)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.indigo.opacity(0.15))
                    .foregroundColor(.indigo)
                    .cornerRadius(4)

                Text(confirmation.targetSummary)
                    .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            // Destination (a third-party upload, a custom endpoint)
            if let destination = confirmation.destinationSummary, !destination.isEmpty {
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "arrow.up.forward.square")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.orange)
                    Text(destination)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(7)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.08))
                .cornerRadius(5)
            }

            // Diff or Command Preview
            if let cmd = confirmation.commandPreview {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Command:")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(cmd)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.primary)
                            .padding(6)
                            .background(Color(NSColor.textBackgroundColor).opacity(0.5))
                            .cornerRadius(4)
                    }
                }
            } else if let diff = confirmation.diffPreview {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Proposed Changes:")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    ScrollView(.vertical, showsIndicators: true) {
                        Text(diff)
                            .font(.system(size: 10.5, design: .monospaced))
                            .foregroundColor(.primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(6)
                            .background(Color(NSColor.textBackgroundColor).opacity(0.5))
                            .cornerRadius(4)
                    }
                    .frame(maxHeight: 140)
                }
            }

            // Action Buttons: Approve vs Reject
            HStack(spacing: 8) {
                Button(action: onApprove) {
                    HStack(spacing: 5) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 11))
                        Text("Approve & Execute")
                            .font(.system(size: 11.5, weight: .semibold))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.green.opacity(approveEnabled ? 0.85 : 0.3))
                    .foregroundColor(.white)
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .disabled(!approveEnabled)
                .help(approveEnabled ? "Run this action" : "Gathering the details this decision needs…")

                Button(action: onReject) {
                    HStack(spacing: 5) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 11))
                        Text("Reject Action")
                            .font(.system(size: 11.5, weight: .medium))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.red.opacity(0.15))
                    .foregroundColor(.red)
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)

                Spacer()
            }
        }
        .padding(12)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.orange.opacity(0.35), lineWidth: 1.5)
        )
    }
}
