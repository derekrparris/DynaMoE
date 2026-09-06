//
//  SubagentDrawerView.swift
//  DynaMoE
//
//  Background Task Manager UI Drawer showing child agent lifecycles in parallel.
//  - Patterned after Antigravity / Claude Code / CrewAI
//  - Live subagent status badges, duration meters, and animated indicators
//  - Expandable step-by-step transcript inspector
//  - Direct inter-agent message sender and cancellation controls
//

import SwiftUI

public struct SubagentDrawerView: View {
    @ObservedObject var subagentManager = SubagentManager.shared
    public var onClose: () -> Void

    @State private var expandedSubagentIds: Set<UUID> = []
    @State private var messageDrafts: [UUID: String] = [:]

    public init(onClose: @escaping () -> Void) {
        self.onClose = onClose
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: "square.2.layers.3d")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(.purple)

                Text("Task Manager")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.primary)

                if subagentManager.activeSubagentsCount > 0 {
                    Text("\(subagentManager.activeSubagentsCount) active")
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.blue.opacity(0.15))
                        .foregroundColor(.blue)
                        .clipShape(Capsule())
                }

                Spacer()

                if !subagentManager.subagents.isEmpty {
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            subagentManager.clearCompleted()
                        }
                    }) {
                        Text("Clear Done")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear completed and cancelled subagents")
                }

                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)
                        .frame(width: 20, height: 20)
                        .background(Color.secondary.opacity(0.1))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Close Drawer")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(Color(NSColor.controlBackgroundColor))

            Divider()

            // Subagent List
            if subagentManager.subagents.isEmpty {
                VStack(spacing: 12) {
                    Spacer()
                    Image(systemName: "person.crop.circle.badge.plus")
                        .font(.system(size: 36))
                        .foregroundColor(.secondary.opacity(0.4))
                    Text("No Active Subagents")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.secondary)
                    Text("The coordinator agent will spawn isolated background subagents here for complex tasks (e.g. Codebase Researcher, Test Runner, Shader Optimizer).")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 20)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(subagentManager.subagents) { subagent in
                            SubagentCardView(
                                subagent: subagent,
                                isExpanded: Binding(
                                    get: { expandedSubagentIds.contains(subagent.id) },
                                    set: { expanded in
                                        if expanded {
                                            expandedSubagentIds.insert(subagent.id)
                                        } else {
                                            expandedSubagentIds.remove(subagent.id)
                                        }
                                    }
                                ),
                                messageDraft: Binding(
                                    get: { messageDrafts[subagent.id] ?? "" },
                                    set: { messageDrafts[subagent.id] = $0 }
                                ),
                                onSendMessage: { text in
                                    subagentManager.sendMessage(toSubagentId: subagent.id, sender: "user", content: text)
                                    messageDrafts[subagent.id] = ""
                                },
                                onCancel: {
                                    subagent.cancel()
                                }
                            )
                        }
                    }
                    .padding(12)
                }
            }
        }
        .frame(minWidth: 320, idealWidth: 360, maxWidth: 420)
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - Subagent Card View

private struct SubagentCardView: View {
    @ObservedObject var subagent: SubagentInstance
    @Binding var isExpanded: Bool
    @Binding var messageDraft: String
    var onSendMessage: (String) -> Void
    var onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Top Row: Role + Status Badge + Duration
            HStack(spacing: 6) {
                Image(systemName: subagent.archetype.defaultIconName)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.purple)

                Text(subagent.role)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.primary)
                    .lineLimit(1)

                Spacer()

                // Status Badge
                HStack(spacing: 4) {
                    if subagent.status == .running {
                        ProgressView()
                            .scaleEffect(0.55)
                            .frame(width: 12, height: 12)
                    } else {
                        Image(systemName: subagent.status.iconName)
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(subagent.status.color)
                    }

                    Text(subagent.status.displayName)
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(subagent.status.color)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(subagent.status.color.opacity(0.12))
                .clipShape(Capsule())

                // Expand/Collapse
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isExpanded.toggle()
                    }
                }) {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.secondary)
                        .padding(4)
                }
                .buttonStyle(.plain)
            }

            // Task Description Preview
            Text(subagent.taskDescription)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .lineLimit(isExpanded ? nil : 2)

            // Live status line
            HStack(spacing: 4) {
                Circle()
                    .fill(subagent.status == .running ? Color.blue : Color.secondary.opacity(0.4))
                    .frame(width: 6, height: 6)

                Text(subagent.liveStatusText)
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundColor(subagent.status == .running ? .blue : .secondary)
                    .lineLimit(1)

                Spacer()

                if subagent.executionDurationSeconds > 0 || subagent.status == .completed {
                    Text(String(format: "%.2fs", subagent.executionDurationSeconds))
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }

            // Expanded Details: Transcript, Summary, Inter-Agent Messaging
            if isExpanded {
                Divider()
                    .padding(.vertical, 2)

                // Transcript Steps
                if !subagent.transcript.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("TRANSCRIPT (\(subagent.transcript.count) STEPS)")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary.opacity(0.7))

                        ForEach(subagent.transcript) { step in
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(spacing: 4) {
                                    Image(systemName: step.isError ? "exclamationmark.circle.fill" : "checkmark.circle.fill")
                                        .font(.system(size: 9))
                                        .foregroundColor(step.isError ? .red : .green)

                                    Text("Step \(step.stepIndex): \(step.actionName)")
                                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                        .foregroundColor(.primary)

                                    Spacer()

                                    Text(String(format: "%.2fs", step.durationSeconds))
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundColor(.secondary)
                                }

                                if !step.output.isEmpty {
                                    Text(step.output.prefix(200) + (step.output.count > 200 ? "..." : ""))
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundColor(.secondary)
                                        .padding(4)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(Color.secondary.opacity(0.06))
                                        .cornerRadius(4)
                                }
                            }
                            .padding(.vertical, 2)
                        }
                    }
                }

                // Final Summary Preview
                if !subagent.finalSummary.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("SYNTHESIZED SUMMARY")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.secondary.opacity(0.7))

                            Spacer()

                            Button(action: {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(subagent.finalSummary, forType: .string)
                            }) {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 9))
                                    .foregroundColor(.purple)
                            }
                            .buttonStyle(.plain)
                            .help("Copy Summary")
                        }

                        Text(subagent.finalSummary)
                            .font(.system(size: 10))
                            .foregroundColor(.primary)
                            .padding(6)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.purple.opacity(0.05))
                            .cornerRadius(6)
                    }
                }

                // Inter-Agent Message Log
                if !subagent.messages.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("INTER-AGENT MESSAGES (\(subagent.messages.count))")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary.opacity(0.7))

                        ForEach(subagent.messages) { msg in
                            HStack(alignment: .top, spacing: 4) {
                                Text("\(msg.sender):")
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundColor(.purple)
                                Text(msg.content)
                                    .font(.system(size: 9))
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }

                // Inline Inter-Agent Directive Input
                HStack(spacing: 6) {
                    TextField("Send directive to subagent...", text: $messageDraft)
                        .textFieldStyle(.plain)
                        .font(.system(size: 10))
                        .padding(5)
                        .background(Color.secondary.opacity(0.08))
                        .cornerRadius(6)
                        .onSubmit {
                            guard !messageDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                            onSendMessage(messageDraft)
                        }

                    Button(action: {
                        guard !messageDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                        onSendMessage(messageDraft)
                    }) {
                        Image(systemName: "paperplane.fill")
                            .font(.system(size: 10))
                            .foregroundColor(.purple)
                            .padding(5)
                    }
                    .buttonStyle(.plain)
                    .disabled(messageDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                // Bottom Action Row
                HStack {
                    if subagent.status == .running || subagent.status == .pending {
                        Button(action: onCancel) {
                            HStack(spacing: 4) {
                                Image(systemName: "xmark.circle")
                                    .font(.system(size: 10))
                                Text("Cancel Subagent")
                                    .font(.system(size: 10, weight: .medium))
                            }
                            .foregroundColor(.red)
                        }
                        .buttonStyle(.plain)
                    }

                    Spacer()

                    Text("ID: \(subagent.id.uuidString.prefix(8))...")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.secondary.opacity(0.5))
                }
                .padding(.top, 4)
            }
        }
        .padding(10)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.8))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(subagent.status == .running ? Color.blue.opacity(0.3) : Color.secondary.opacity(0.12), lineWidth: 1)
        )
    }
}
