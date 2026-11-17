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
    private static let rememberedDecisionLimit = 8

    private init() {}

    public func waitForApproval(id: UUID) async -> Bool {
        // A pending prompt is part of a running agent operation even though
        // the engine is idle, so a config reload must wait for the decision.
        AgentActivity.shared.begin(.approval)
        defer { AgentActivity.shared.end(.approval) }
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

    /// Whether a call is still waiting on this id, for the card to render.
    public func isWaiting(id: UUID) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return pendingContinuations[id] != nil
    }

    public func cancelAll() {
        lock.lock()
        let all = pendingContinuations
        pendingContinuations.removeAll()
        // A decision recorded for a waiter that never registered belongs to the
        // turn being stopped: cancelling wins over a click that arrived first,
        // or a stopped turn would execute the call it was stopped for.
        cancelledWaits.formUnion(pendingDecisions.keys)
        pendingDecisions.removeAll()
        pendingDecisionOrder.removeAll()
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
        pendingDecisions[id] = approved
        pendingDecisionOrder.append(id)
        if pendingDecisionOrder.count > Self.rememberedDecisionLimit {
            let oldest = pendingDecisionOrder.removeFirst()
            pendingDecisions.removeValue(forKey: oldest)
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
                    .help("Turn Turbo Mode on so this and later file writes and commands run without a per-call confirmation. This control cannot turn Turbo off.")
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
                    .background(Color.green.opacity(0.85))
                    .foregroundColor(.white)
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)

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
