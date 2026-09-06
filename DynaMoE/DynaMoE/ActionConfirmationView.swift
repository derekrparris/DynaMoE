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

public final class ActionApprovalManager: @unchecked Sendable {
    public static let shared = ActionApprovalManager()

    private let lock = NSLock()
    private var pendingContinuations: [UUID: CheckedContinuation<Bool, Never>] = [:]

    private init() {}

    public func waitForApproval(id: UUID) async -> Bool {
        await withCheckedContinuation { continuation in
            lock.lock()
            pendingContinuations[id] = continuation
            lock.unlock()
        }
    }

    public func approve(id: UUID) {
        lock.lock()
        let continuation = pendingContinuations.removeValue(forKey: id)
        lock.unlock()
        continuation?.resume(returning: true)
    }

    public func reject(id: UUID) {
        lock.lock()
        let continuation = pendingContinuations.removeValue(forKey: id)
        lock.unlock()
        continuation?.resume(returning: false)
    }

    public func cancelAll() {
        lock.lock()
        let all = pendingContinuations
        pendingContinuations.removeAll()
        lock.unlock()
        for (_, cont) in all {
            cont.resume(returning: false)
        }
    }
}

public struct PendingActionConfirmation: Identifiable, Equatable {
    public let id: UUID
    public let toolName: String
    public let arguments: [String: String]
    public let targetSummary: String
    public let diffPreview: String?
    public let commandPreview: String?

    public init(
        id: UUID = UUID(),
        toolName: String,
        arguments: [String: String],
        targetSummary: String,
        diffPreview: String? = nil,
        commandPreview: String? = nil
    ) {
        self.id = id
        self.toolName = toolName
        self.arguments = arguments
        self.targetSummary = targetSummary
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

                // Turbo Mode indicator
                Button(action: {
                    isTurboModeEnabled.toggle()
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: isTurboModeEnabled ? "bolt.fill" : "bolt.slash")
                            .font(.system(size: 10))
                        Text(isTurboModeEnabled ? "Turbo ON" : "Turbo OFF")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(isTurboModeEnabled ? Color.orange.opacity(0.18) : Color.secondary.opacity(0.1))
                    .foregroundColor(isTurboModeEnabled ? .orange : .secondary)
                    .cornerRadius(5)
                }
                .buttonStyle(.plain)
                .help("When Turbo Mode is ON, file writes and commands execute automatically without individual confirmation.")
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
