//
//  ToolExecutionTimelineView.swift
//  DynaMoE
//
//  Phase 5: Real-Time Tool Execution Timeline
//  Renders multi-step agent actions in an interactive sequential timeline
//  (e.g., [Scout Files] -> [Read Document] -> [Apply Changes])
//  with live durations, status indicators, and expandable inspection views.
//

import SwiftUI

public struct ToolExecutionTimelineView: View {
    public let toolCalls: [ToolCallRecord]
    public var onApproveCall: ((UUID) -> Void)? = nil
    public var onRejectCall: ((UUID) -> Void)? = nil

    public init(
        toolCalls: [ToolCallRecord],
        onApproveCall: ((UUID) -> Void)? = nil,
        onRejectCall: ((UUID) -> Void)? = nil
    ) {
        self.toolCalls = toolCalls
        self.onApproveCall = onApproveCall
        self.onRejectCall = onRejectCall
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Horizontal Step Nodes Flow
            if toolCalls.count > 1 {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(Array(toolCalls.enumerated()), id: \.element.id) { index, call in
                            HStack(spacing: 6) {
                                StepNodeBadge(stepIndex: index + 1, call: call)
                                if index < toolCalls.count - 1 {
                                    Image(systemName: "arrow.right")
                                        .font(.system(size: 8, weight: .bold))
                                        .foregroundColor(.secondary.opacity(0.4))
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 2)
                    .padding(.vertical, 4)
                }
            }

            // Detailed Cards for Each Step
            VStack(alignment: .leading, spacing: 8) {
                ForEach(toolCalls) { call in
                    TimelineStepCardView(
                        call: call,
                        onApprove: { onApproveCall?(call.id) },
                        onReject: { onRejectCall?(call.id) }
                    )
                }
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Mini Step Node Badge

struct StepNodeBadge: View {
    let stepIndex: Int
    let call: ToolCallRecord

    private var nodeColor: Color {
        switch call.status {
        case .running: return .indigo
        case .success: return .green
        case .error: return .red
        case .awaitingApproval: return .orange
        case .rejected: return .secondary
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(nodeColor)
                .frame(width: 6, height: 6)

            Text("\(stepIndex). \(call.name)")
                .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                .foregroundColor(.primary)

            if let dur = call.executionDurationSeconds {
                Text(String(format: "%.2fs", dur))
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(nodeColor.opacity(0.08))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(nodeColor.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - Timeline Step Card View

struct TimelineStepCardView: View {
    let call: ToolCallRecord
    var onApprove: (() -> Void)? = nil
    var onReject: (() -> Void)? = nil

    @State private var isExpanded: Bool = false
    @State private var isCopied: Bool = false

    private var iconName: String {
        switch call.name {
        case "shell_run": return "terminal.fill"
        case "file_read": return "doc.text.fill"
        case "file_write": return "doc.badge.plus"
        case "file_edit": return "square.and.pencil"
        case "find_files": return "folder.badge.gearshape"
        case "grep_search": return "magnifyingglass"
        case "web_search": return "globe"
        case "web_fetch": return "arrow.down.doc.fill"
        case "complete": return "checkmark.seal.fill"
        default: return "wrench.and.screwdriver.fill"
        }
    }

    private var statusColor: Color {
        switch call.status {
        case .running: return .indigo
        case .success: return .green
        case .error: return .red
        case .awaitingApproval: return .orange
        case .rejected: return .secondary
        }
    }

    private var primarySummary: String {
        switch call.name {
        case "shell_run":
            return call.arguments["command"] ?? call.rawArguments
        case "file_read", "file_write", "file_edit":
            if let path = call.arguments["path"] {
                return (path as NSString).lastPathComponent
            }
            return call.rawArguments
        case "find_files":
            return call.arguments["pattern"] ?? call.rawArguments
        case "grep_search", "web_search":
            return "\"\(call.arguments["query"] ?? "")\""
        case "web_fetch":
            if let url = call.arguments["url"] {
                return (URL(string: url)?.host ?? url)
            }
            return call.rawArguments
        case "complete":
            return call.arguments["summary"] ?? "Completed"
        default:
            return call.rawArguments
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.15)) {
                    isExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Image(systemName: iconName)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(statusColor)

                    Text(call.name)
                        .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                        .foregroundColor(.primary)

                    Text(primarySummary)
                        .font(.system(size: 11.5, weight: .regular, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)

                    Spacer()

                    // Status Indicator
                    switch call.status {
                    case .running:
                        HStack(spacing: 4) {
                            ProgressView()
                                .controlSize(.mini)
                            Text("Running")
                                .font(.system(size: 10.5, weight: .medium))
                                .foregroundColor(.indigo)
                        }
                    case .success:
                        HStack(spacing: 4) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 11))
                                .foregroundColor(.green)
                            if let dur = call.executionDurationSeconds {
                                Text(String(format: "%.2fs", dur))
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                        }
                    case .error:
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 11))
                                .foregroundColor(.red)
                            Text("Failed")
                                .font(.system(size: 10.5, weight: .medium))
                                .foregroundColor(.red)
                        }
                    case .awaitingApproval:
                        HStack(spacing: 4) {
                            Image(systemName: "shield.lefthalf.filled")
                                .font(.system(size: 11))
                                .foregroundColor(.orange)
                            Text("Requires Approval")
                                .font(.system(size: 10.5, weight: .bold))
                                .foregroundColor(.orange)
                        }
                    case .rejected:
                        HStack(spacing: 4) {
                            Image(systemName: "xmark.circle")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                            Text("Rejected")
                                .font(.system(size: 10.5, weight: .medium))
                                .foregroundColor(.secondary)
                        }
                    }

                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9.5, weight: .semibold))
                        .foregroundColor(.secondary.opacity(0.7))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.85))
                .cornerRadius(7)
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .stroke(call.status == .running ? Color.indigo.opacity(0.3) : (call.status == .awaitingApproval ? Color.orange.opacity(0.4) : Color.primary.opacity(0.06)), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            // HITL Action Confirmation Panel if awaiting approval
            if call.status == .awaitingApproval {
                ActionConfirmationView(
                    confirmation: PendingActionConfirmation(
                        id: call.id,
                        toolName: call.name,
                        arguments: call.arguments,
                        targetSummary: primarySummary,
                        diffPreview: call.arguments["diff"] ?? call.arguments["content"],
                        commandPreview: call.arguments["command"]
                    ),
                    onApprove: {
                        ActionApprovalManager.shared.approve(id: call.id)
                        onApprove?()
                    },
                    onReject: {
                        ActionApprovalManager.shared.reject(id: call.id)
                        onReject?()
                    }
                )
                .padding(.top, 2)
            }

            // Expanded Details
            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    if !call.arguments.isEmpty {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Arguments:")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(.secondary)
                            ForEach(call.arguments.sorted(by: { $0.key < $1.key }), id: \.key) { k, v in
                                HStack(alignment: .top, spacing: 4) {
                                    Text("\(k):")
                                        .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                                        .foregroundColor(.secondary)
                                    Text(v)
                                        .font(.system(size: 10.5, design: .monospaced))
                                        .foregroundColor(.primary)
                                        .textSelection(.enabled)
                                }
                            }
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.secondary.opacity(0.04))
                        .cornerRadius(6)
                    }

                    if let out = call.output, !out.isEmpty {
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text("Output:")
                                    .font(.system(size: 10, weight: .semibold))
                                    .foregroundColor(.secondary)
                                Spacer()
                                Button(action: {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(out, forType: .string)
                                    isCopied = true
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                        isCopied = false
                                    }
                                }) {
                                    HStack(spacing: 3) {
                                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                                            .font(.system(size: 9))
                                        Text(isCopied ? "Copied" : "Copy")
                                            .font(.system(size: 9))
                                    }
                                    .foregroundColor(.secondary)
                                }
                                .buttonStyle(.plain)
                            }

                            ScrollView(.horizontal, showsIndicators: false) {
                                Text(out)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(.primary)
                                    .textSelection(.enabled)
                            }
                            .padding(8)
                            .background(Color.secondary.opacity(0.04))
                            .cornerRadius(6)
                        }
                    }

                    if let err = call.error, !err.isEmpty {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Error:")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(.red)
                            Text(err)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.red)
                                .textSelection(.enabled)
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.red.opacity(0.06))
                                .cornerRadius(6)
                        }
                    }
                }
                .padding(.horizontal, 4)
            }
        }
    }
}
