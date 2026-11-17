//
//  SubagentInspectorView.swift
//  DynaMoE
//
//  The runs inspector: every delegated run of the current conversation, live or
//  restored from disk, with its stream, timeline, report and controls.
//  See SUBAGENTS.md §11.
//

import SwiftUI

// MARK: - Presentation helpers

extension AgentRunStatus {
    var tint: Color {
        switch self {
        case .pending, .preparing: return .secondary
        case .running: return .blue
        case .waitingForApproval: return .orange
        case .completed: return .green
        case .failed, .blocked: return .red
        case .cancelled: return .secondary
        }
    }

    var isActive: Bool { !isTerminal }
}

extension AgentUsage {
    var compactDescription: String {
        var parts: [String] = []
        if steps > 0 { parts.append("\(steps) step\(steps == 1 ? "" : "s")") }
        if contextTokens > 0 { parts.append("\(contextTokens) tok") }
        if toolCalls > 0 { parts.append("\(toolCalls) tool\(toolCalls == 1 ? "" : "s")") }
        if costUSD > 0 { parts.append(String(format: "$%.4f", costUSD)) }
        return parts.isEmpty ? "No usage recorded" : parts.joined(separator: " · ")
    }
}

// MARK: - Inspector

struct SubagentInspectorView: View {
    @ObservedObject private var manager = SubagentManager.shared
    @ObservedObject private var approvalCenter = SubagentApprovalCenter.shared

    let sessionId: UUID?
    var onClose: () -> Void

    @State private var selectedRunId: UUID?
    @State private var stored: [AgentRunRecord] = []

    init(sessionId: UUID?, onClose: @escaping () -> Void) {
        self.sessionId = sessionId
        self.onClose = onClose
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            if !allRecords.isEmpty {
                usageStrip
            }
            Divider()
            if selectedRunId == nil, !approvalCenter.pending.isEmpty {
                approvalStack
                Divider()
            }
            if let selectedRunId, let detail = detailRecord(for: selectedRunId) {
                SubagentRunDetailView(
                    record: detail.record,
                    live: detail.live,
                    approvals: approvalCenter.pending.filter { $0.request.runId == detail.record.id },
                    onApprove: { approvalCenter.approve($0) },
                    onReject: { approvalCenter.reject($0) },
                    onBack: { withAnimation(.easeInOut(duration: 0.15)) { self.selectedRunId = nil } },
                    onClose: onClose,
                    onRecordChanged: reloadStored
                )
                .transition(.opacity)
            } else {
                runList
            }
        }
        .frame(minWidth: 340, idealWidth: 420, maxWidth: 560)
        .background(Color(NSColor.windowBackgroundColor))
        .onAppear(perform: reloadStored)
        .onChange(of: manager.subagents.count) { _ in reloadStored() }
        .onChange(of: manager.activeSubagentsCount) { _ in reloadStored() }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "square.2.layers.3d")
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(.purple)

            Text("Subagent Runs")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.primary)

            if manager.activeSubagentsCount > 0 {
                Text("\(manager.activeSubagentsCount) active")
                    .font(.system(size: 10, weight: .bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.15))
                    .foregroundColor(.blue)
                    .clipShape(Capsule())
            }

            Spacer()

            if !terminalRuns.isEmpty {
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        manager.clearCompleted(sessionId: sessionId)
                        reloadStored()
                    }
                }) {
                    Text("Clear Finished")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Remove finished runs from this conversation")
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
            .help("Close Inspector")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(Color(NSColor.controlBackgroundColor))
    }

    /// The conversation's cost and usage at a glance. A run's cost is only
    /// meaningful in aggregate (one web search costs less than a reviewer's
    /// forty steps), so this is the dashboard the design doc asks for without a
    /// second window to keep in sync with the list.
    private var usageStrip: some View {
        let totals = sessionTotals
        return HStack(spacing: 16) {
            if totals.active > 0 {
                usageStat("\(totals.active)", "active", tint: .blue)
            }
            usageStat("\(totals.runs)", totals.runs == 1 ? "run" : "runs")
            if totals.tokens > 0 {
                usageStat(compactTokens(totals.tokens), "tokens")
                    .help("Context this conversation's runs are carrying: each run's last request size plus the tokens it generated, summed over runs. Every backend counts the full prompt a request carried, so this is not a running total of what each step billed; spend prices those.")
            }
            if totals.cost > 0 {
                usageStat(String(format: "$%.4f", totals.cost), "spend")
            }
            if totals.seconds > 0 {
                usageStat(compactDuration(totals.seconds), "time")
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.55))
    }

    private func usageStat(_ value: String, _ label: String, tint: Color = .secondary) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(tint == .secondary ? .primary : tint)
            Text(label)
                .font(.system(size: 8.5, weight: .medium))
                .foregroundColor(.secondary)
        }
    }

    private var sessionTotals: (runs: Int, active: Int, tokens: Int, cost: Double, seconds: Double) {
        var runs = 0
        var active = 0
        var tokens = 0
        var cost = 0.0
        var seconds = 0.0
        for record in allRecords {
            runs += 1
            if !record.status.isTerminal { active += 1 }
            tokens += record.usage.contextTokens
            cost += record.usage.costUSD
            if record.usage.wallClockSeconds > 0 {
                seconds += record.usage.wallClockSeconds
            } else if let completed = record.completedAt, let started = record.startedAt {
                seconds += completed.timeIntervalSince(started)
            }
        }
        return (runs, active, tokens, cost, seconds)
    }

    private func compactTokens(_ count: Int) -> String {
        if count >= 1_000_000 { return String(format: "%.1fM", Double(count) / 1_000_000) }
        if count >= 1_000 { return String(format: "%.1fk", Double(count) / 1_000) }
        return "\(count)"
    }

    private func compactDuration(_ seconds: Double) -> String {
        if seconds >= 3_600 { return String(format: "%.1fh", seconds / 3_600) }
        if seconds >= 60 { return String(format: "%.1fm", seconds / 60) }
        return String(format: "%.1fs", seconds)
    }

    private var approvalStack: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("WAITING FOR YOUR DECISION")
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.orange)
            ForEach(approvalCenter.pending) { entry in
                ActionConfirmationView(
                    confirmation: SubagentApprovalCenter.confirmation(for: entry.request),
                    onApprove: { approvalCenter.approve(entry.id) },
                    onReject: { approvalCenter.reject(entry.id) }
                )
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.05))
    }

    // MARK: List

    @ViewBuilder
    private var runList: some View {
        if activeRuns.isEmpty && terminalRuns.isEmpty {
            emptyState
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if !activeRuns.isEmpty {
                        sectionLabel("ACTIVE")
                        ForEach(activeRuns, id: \.id) { record in
                            row(for: record)
                        }
                    }
                    if !terminalRuns.isEmpty {
                        sectionLabel("FINISHED")
                        ForEach(terminalRuns, id: \.id) { record in
                            row(for: record)
                        }
                    }
                }
                .padding(12)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer(minLength: 40)
            Image(systemName: "person.crop.circle.badge.plus")
                .font(.system(size: 36))
                .foregroundColor(.secondary.opacity(0.4))
            Text("No Subagent Runs")
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.secondary)
            Text("When the coordinator delegates work with spawn_subagent, the run appears here with its live stream, tool timeline and final report. Runs survive relaunch.")
                .font(.system(size: 11))
                .foregroundColor(.secondary.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)
            Spacer(minLength: 40)
        }
        .frame(maxWidth: .infinity)
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .bold))
            .foregroundColor(.secondary.opacity(0.7))
            .padding(.top, 2)
    }

    private func row(for record: AgentRunRecord) -> some View {
        Button(action: {
            withAnimation(.easeInOut(duration: 0.15)) { selectedRunId = record.id }
        }) {
            SubagentRunRow(record: record, isLive: liveInstance(record.id) != nil)
        }
        .buttonStyle(.plain)
    }

    // MARK: Merge

    /// Stored records are the history; a live instance wins for its own id, so
    /// a run that is still working is never displayed from its last snapshot.
    private var allRecords: [AgentRunRecord] {
        var byId: [UUID: AgentRunRecord] = [:]
        for record in stored { byId[record.id] = record }
        for instance in liveInstances { byId[instance.id] = instance.snapshotRecord() }
        return byId.values.sorted { lhs, rhs in
            let l = lhs.completedAt ?? lhs.startedAt ?? lhs.createdAt
            let r = rhs.completedAt ?? rhs.startedAt ?? rhs.createdAt
            return l > r
        }
    }

    private var activeRuns: [AgentRunRecord] {
        allRecords.filter { !$0.status.isTerminal }
    }

    private var terminalRuns: [AgentRunRecord] {
        allRecords.filter(\.status.isTerminal)
    }

    /// Runs started outside a conversation (an API/agent turn with no chat
    /// session) have no bucket of their own, so they ride every inspector
    /// rather than becoming invisible in the one that happens to be open.
    private var liveInstances: [SubagentInstance] {
        manager.allSubagents.filter {
            sessionId == nil || $0.parentSessionId == sessionId || $0.parentSessionId == nil
        }
    }

    private func liveInstance(_ runId: UUID) -> SubagentInstance? {
        liveInstances.first { $0.id == runId }
    }

    private func detailRecord(for runId: UUID) -> (record: AgentRunRecord, live: SubagentInstance?)? {
        if let live = liveInstance(runId) {
            return (live.snapshotRecord(), live)
        }
        if let record = allRecords.first(where: { $0.id == runId }) {
            return (record, nil)
        }
        return nil
    }

    private func reloadStored() {
        let sessionId = sessionId
        Task {
            let records = await Task.detached(priority: .utility) {
                var records = AgentRunStore.shared.load(sessionId: sessionId)
                if sessionId != nil {
                    records += AgentRunStore.shared.load(sessionId: nil)
                }
                return records
            }.value
            self.stored = records
        }
    }
}

// MARK: - Row

private struct SubagentRunRow: View {
    let record: AgentRunRecord
    let isLive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: iconName)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.purple)

                Text(record.role)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.primary)
                    .lineLimit(1)

                Spacer()

                if record.status == .completed && !record.summaryRelayed {
                    Text("UNREAD")
                        .font(.system(size: 8, weight: .bold))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color.purple.opacity(0.14))
                        .foregroundColor(.purple)
                        .clipShape(Capsule())
                        .help("The coordinator has not received this report yet")
                }

                SubagentStatusPill(status: record.status)
            }

            Text(record.task)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .lineLimit(2)
                .multilineTextAlignment(.leading)

            HStack(spacing: 5) {
                Text(record.backend.displayName)
                    .lineLimit(1)
                if let model = record.backend.modelId, !model.isEmpty, model != record.backend.displayName {
                    Text("·")
                    Text(model)
                        .lineLimit(1)
                }
                Spacer()
                Text(durationText)
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundColor(.secondary.opacity(0.6))
            }
            .font(.system(size: 10, weight: .medium))
            .foregroundColor(.secondary)
        }
        .padding(10)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.85))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    record.status.isActive ? Color.blue.opacity(0.3) : Color.secondary.opacity(0.12),
                    lineWidth: 1
                )
        )
        .contentShape(Rectangle())
    }

    private var iconName: String {
        if isLive && record.status.isActive { return "arrow.triangle.2.circlepath" }
        switch record.status {
        case .completed: return "checkmark.circle.fill"
        case .failed, .blocked: return "exclamationmark.triangle.fill"
        case .cancelled: return "xmark.circle"
        default: return "clock"
        }
    }

    private var durationText: String {
        if let completed = record.completedAt, let started = record.startedAt {
            return String(format: "%.1fs", completed.timeIntervalSince(started))
        }
        if let started = record.startedAt {
            return String(format: "%.0fs…", Date().timeIntervalSince(started))
        }
        return record.usage.compactDescription
    }
}

// MARK: - Status pill

private struct SubagentStatusPill: View {
    let status: AgentRunStatus

    var body: some View {
        HStack(spacing: 4) {
            if status.isActive {
                ProgressView()
                    .scaleEffect(0.5)
                    .frame(width: 11, height: 11)
            } else {
                Circle()
                    .fill(status.tint)
                    .frame(width: 6, height: 6)
            }
            Text(status.displayName)
                .font(.system(size: 9.5, weight: .bold))
                .foregroundColor(status.tint)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(status.tint.opacity(0.12))
        .clipShape(Capsule())
    }
}

// MARK: - Detail

private struct SubagentRunDetailView: View {
    let record: AgentRunRecord
    let live: SubagentInstance?
    let approvals: [PendingSubagentApproval]
    var onApprove: (UUID) -> Void
    var onReject: (UUID) -> Void
    var onBack: () -> Void
    var onClose: () -> Void
    /// Reloading the stored list is what makes a non-live record show the
    /// apply it just completed; a live run publishes its own `appliedAt`.
    var onRecordChanged: () -> Void

    @State private var messageDraft = ""
    @State private var showThinking = true
    @State private var isApplying = false
    @State private var applyError: String?
    @State private var appliedHereAt: Date?

    /// The record is a value snapshot, so an apply performed from this pane is
    /// shown from local state until the parent reloads; the two are the same
    /// date once it does.
    private var appliedAt: Date? { appliedHereAt ?? record.appliedAt }

    private var canApplyChanges: Bool {
        appliedHereAt == nil && SubagentApplyControl.canApply(record)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                breadcrumb
                title
                metaGrid
                usageLine
                if !approvals.isEmpty {
                    approvalSection
                }
                if let live, !record.status.isTerminal {
                    liveStream(live)
                }
                taskSection
                timelineSection
                if let report = record.report {
                    reportSection(report)
                }
                if let live, !record.status.isTerminal {
                    steeringSection(live)
                }
                controls
            }
            .padding(12)
        }
    }

    // MARK: Pieces

    private var breadcrumb: some View {
        HStack(spacing: 6) {
            Button(action: onBack) {
                HStack(spacing: 3) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 9, weight: .bold))
                    Text("Runs")
                        .font(.system(size: 10, weight: .medium))
                }
                .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)

            Spacer()

            Text(record.id.uuidString.prefix(8) + "…")
                .font(.system(size: 9, design: .monospaced))
                .foregroundColor(.secondary.opacity(0.6))
                .textSelection(.enabled)

            Button(action: {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(record.id.uuidString, forType: .string)
            }) {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .help("Copy the run's identifier")
        }
    }

    private var title: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(record.role)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(.primary)
                Spacer()
                SubagentStatusPill(status: record.status)
            }
            HStack(spacing: 6) {
                Text(record.preset.displayName + " preset")
                Text("·")
                Text(record.backend.displayName)
                if let model = record.backend.modelId, !model.isEmpty, model != record.backend.displayName {
                    Text("·")
                    Text(model)
                }
                if record.isolation.isIsolated, let path = record.isolation.worktreePath {
                    Text("·")
                    Text("worktree \((path as NSString).lastPathComponent)")
                }
            }
            .font(.system(size: 10.5))
            .foregroundColor(.secondary)
        }
    }

    private var metaGrid: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("Working directory")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.secondary)
                    .frame(width: 96, alignment: .leading)
                Text(record.workingDirectory)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Button(action: {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: record.workingDirectory)])
                }) {
                    Image(systemName: "folder")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Reveal in Finder")
                Spacer(minLength: 0)
            }
            if case .worktree(let worktreePath, let branch) = record.isolation {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("Worktree")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.secondary)
                        .frame(width: 96, alignment: .leading)
                    Text(worktreePath)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                    if FileManager.default.fileExists(atPath: worktreePath) {
                        Button(action: {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: worktreePath)])
                        }) {
                            Image(systemName: "folder")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Reveal the run's worktree in Finder")
                    }
                    Spacer(minLength: 0)
                }
                metaRow("Branch", branch)
                if let appliedAt {
                    metaRow("Applied", Self.dateFormatter.string(from: appliedAt))
                } else if record.status.isTerminal, FileManager.default.fileExists(atPath: worktreePath) {
                    Text("Isolated changes are waiting. Apply them below, or let the coordinator land them with apply_subagent_changes.")
                        .font(.system(size: 9.5))
                        .foregroundColor(.secondary)
                        .padding(.leading, 102)
                }
            }
            if let started = record.startedAt {
                metaRow("Started", Self.dateFormatter.string(from: started))
            }
            if let completed = record.completedAt {
                metaRow("Finished", Self.dateFormatter.string(from: completed))
            }
            metaRow(
                "Budget",
                "\(record.budget.maxSteps) steps · \(record.budget.maxCompletionTokens) tok · \(record.budget.maxWallClockSeconds)s"
            )
            if let error = record.error, !error.isEmpty {
                Text(error)
                    .font(.system(size: 10.5))
                    .foregroundColor(.red)
                    .padding(6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.red.opacity(0.06))
                    .cornerRadius(5)
            }
        }
    }

    private func metaRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(.secondary)
                .frame(width: 96, alignment: .leading)
            Text(value)
                .font(.system(size: 10))
                .foregroundColor(.primary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
    }

    private var usageLine: some View {
        HStack(spacing: 5) {
            Image(systemName: "chart.bar")
                .font(.system(size: 9))
            Text(record.usage.compactDescription)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
            if record.usage.wallClockSeconds > 0 {
                Text(String(format: "· %.1fs", record.usage.wallClockSeconds))
                    .font(.system(size: 10, design: .monospaced))
            }
        }
        .foregroundColor(.secondary)
    }

    private var approvalSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("WAITING FOR YOUR DECISION")
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.orange)
            ForEach(approvals) { entry in
                ActionConfirmationView(
                    confirmation: SubagentApprovalCenter.confirmation(for: entry.request),
                    onApprove: { onApprove(entry.id) },
                    onReject: { onReject(entry.id) }
                )
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.05))
        .cornerRadius(6)
    }

    private func liveStream(_ live: SubagentInstance) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text("LIVE")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(.blue)
                Text(live.liveStatusText)
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                Spacer()
                if !live.liveThinkingText.isEmpty {
                    Button(action: { showThinking.toggle() }) {
                        Text(showThinking ? "Hide reasoning" : "Show reasoning")
                            .font(.system(size: 9, weight: .medium))
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }

            if showThinking, !live.liveThinkingText.isEmpty {
                Text(live.liveThinkingText)
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.secondary.opacity(0.06))
                    .cornerRadius(6)
                    .textSelection(.enabled)
            }

            if !live.liveAssistantText.isEmpty {
                Text(live.liveAssistantText)
                    .font(.system(size: 11))
                    .foregroundColor(.primary)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.blue.opacity(0.05))
                    .cornerRadius(6)
                    .textSelection(.enabled)
            }
        }
    }

    private var taskSection: some View {
        section("TASK") {
            Text(record.task)
                .font(.system(size: 11))
                .foregroundColor(.primary)
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.secondary.opacity(0.05))
                .cornerRadius(6)
            if let context = record.contextSummary, !context.isEmpty {
                Text(context)
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private var timelineSection: some View {
        section("TIMELINE") {
            let entries = record.transcript.filter { $0.kind == .toolResult || $0.kind == .status }
            if entries.isEmpty {
                Text(record.status.isActive ? "No tool calls yet." : "No tool calls were made.")
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
            } else {
                ForEach(entries) { entry in
                    SubagentTimelineRow(entry: entry)
                }
            }
        }
    }

    private func reportSection(_ report: AgentReport) -> some View {
        section("REPORT") {
            if !report.summary.isEmpty {
                Text(report.summary)
                    .font(.system(size: 11))
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.purple.opacity(0.05))
                    .cornerRadius(6)
            }

            if !report.findings.isEmpty {
                reportHeading("Findings")
                ForEach(Array(report.findings.enumerated()), id: \.offset) { _, finding in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(finding.claim)
                            .font(.system(size: 10.5, weight: .medium))
                        Text(finding.evidence)
                            .font(.system(size: 9.5, design: .monospaced))
                            .foregroundColor(.secondary)
                            .textSelection(.enabled)
                    }
                    .padding(.vertical, 2)
                }
            }

            if !report.changedFiles.isEmpty {
                reportHeading("Changed files")
                ForEach(report.changedFiles, id: \.self) { file in
                    HStack(spacing: 4) {
                        Image(systemName: "doc.text")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                        Text(file)
                            .font(.system(size: 9.5, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                }
            }

            if !report.tests.isEmpty {
                reportHeading("Tests")
                ForEach(Array(report.tests.enumerated()), id: \.offset) { _, outcome in
                    HStack(alignment: .top, spacing: 5) {
                        Image(systemName: outcome.succeeded ? "checkmark.circle.fill" : "xmark.octagon.fill")
                            .font(.system(size: 9))
                            .foregroundColor(outcome.succeeded ? .green : .red)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(outcome.command)
                                .font(.system(size: 9.5, design: .monospaced))
                                .lineLimit(1)
                            if !outcome.summary.isEmpty {
                                Text(outcome.summary)
                                    .font(.system(size: 9))
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
            }

            if !report.risks.isEmpty {
                reportHeading("Risks")
                ForEach(report.risks, id: \.self) { risk in
                    Text("• " + risk)
                        .font(.system(size: 10))
                        .foregroundColor(.orange)
                }
            }

            if !report.openQuestions.isEmpty {
                reportHeading("Open questions")
                ForEach(report.openQuestions, id: \.self) { question in
                    Text("• " + question)
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                }
            }

            if let diff = report.diff, !diff.isEmpty {
                DisclosureGroup("Diff") {
                    Text(diff)
                        .font(.system(size: 9, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(5)
                }
                .font(.system(size: 10, weight: .medium))
            }
        }
    }

    private func steeringSection(_ live: SubagentInstance) -> some View {
        section("STEER") {
            if !live.messages.isEmpty {
                ForEach(live.messages) { message in
                    HStack(alignment: .top, spacing: 4) {
                        Text(message.sender + ":")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.purple)
                        Text(message.content)
                            .font(.system(size: 9.5))
                            .foregroundColor(.secondary)
                    }
                }
            }
            HStack(spacing: 6) {
                TextField("Send a directive…", text: $messageDraft)
                    .textFieldStyle(.plain)
                    .font(.system(size: 10))
                    .padding(5)
                    .background(Color.secondary.opacity(0.08))
                    .cornerRadius(6)
                    .onSubmit { send(live) }
                Button(action: { send(live) }) {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 10))
                        .foregroundColor(.purple)
                        .padding(5)
                }
                .buttonStyle(.plain)
                .disabled(messageDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private func send(_ live: SubagentInstance) {
        let text = messageDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        SubagentManager.shared.sendMessage(toSubagentId: live.id, sender: "user", content: text)
        messageDraft = ""
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                if let live, !record.status.isTerminal {
                    Button(role: .destructive) {
                        live.cancel()
                    } label: {
                        Label("Stop Run", systemImage: "stop.circle")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }

                if canApplyChanges {
                    Button(action: applyChanges) {
                        Label(isApplying ? "Applying…" : "Apply Changes", systemImage: "arrow.down.doc")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .tint(.purple)
                    .disabled(isApplying)
                    .help("Land this run's worktree changes on the main working tree, unstaged. You will be asked to confirm first.")
                }

                if let report = record.report, !report.summary.isEmpty {
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(report.summary, forType: .string)
                    } label: {
                        Label("Copy Report", systemImage: "doc.on.doc")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()

                if !record.status.isTerminal {
                    Button {
                        // The coordinator keeps the conversation; closing the panel
                        // while a run works is a view change, not a cancellation.
                        onClose()
                    } label: {
                        Text("Hide")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.secondary)
                }
            }

            if let applyError {
                Text(applyError)
                    .font(.system(size: 10))
                    .foregroundColor(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func applyChanges() {
        guard !isApplying else { return }
        isApplying = true
        applyError = nil
        Task {
            do {
                let outcome = try await SubagentApplyControl.apply(runId: record.id)
                appliedHereAt = outcome.appliedAt
                onRecordChanged()
            } catch {
                applyError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            }
            isApplying = false
        }
    }

    private func section<Content: View>(_ label: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.secondary.opacity(0.7))
            content()
        }
    }

    private func reportHeading(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9.5, weight: .semibold))
            .foregroundColor(.secondary)
            .padding(.top, 2)
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        return formatter
    }()
}

// MARK: - Timeline row

private struct SubagentTimelineRow: View {
    let entry: AgentTranscriptEntry

    @State private var isExpanded = false

    var body: some View {
        let record = entry.toolRecord
        let isError = record.map {
            $0.status == .error || $0.status == .rejected || $0.status == .withheld
        } ?? false
        VStack(alignment: .leading, spacing: 4) {
            Button(action: { withAnimation(.easeInOut(duration: 0.15)) { isExpanded.toggle() } }) {
                HStack(spacing: 5) {
                    Image(systemName: isError ? "exclamationmark.circle.fill" : "checkmark.circle.fill")
                        .font(.system(size: 9))
                        .foregroundColor(isError ? .red : .green)
                    Text(record?.name ?? "Step \(entry.stepIndex + 1)")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundColor(.primary)
                    Spacer()
                    if let duration = record?.executionDurationSeconds {
                        Text(String(format: "%.2fs", duration))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 8, weight: .semibold))
                        .foregroundColor(.secondary.opacity(0.6))
                }
            }
            .buttonStyle(.plain)

            if isExpanded {
                if let arguments = record?.arguments, !arguments.isEmpty {
                    ForEach(arguments.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                        Text("\(key): \(value)")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.secondary)
                            .lineLimit(3)
                            .textSelection(.enabled)
                    }
                }
                if let text = entry.text ?? record?.output ?? record?.error, !text.isEmpty {
                    Text(String(text.prefix(2_000)) + (text.count > 2_000 ? "\n…" : ""))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.secondary)
                        .textSelection(.enabled)
                        .padding(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(5)
                }
            }
        }
        .padding(.vertical, 1)
    }
}

// MARK: - Inline spawn card

/// The transcript card for one `spawn_subagent` call. It links a live run (or a
/// restored record) to the inspector instead of showing a bare button, so the
/// user can see what the delegated work is doing without leaving the chat.
struct SubagentInlineRunCard: View {
    @ObservedObject private var manager = SubagentManager.shared

    let runId: UUID
    var onOpenInspector: () -> Void

    @State private var storedRecord: AgentRunRecord?

    var body: some View {
        if let instance = manager.allSubagents.first(where: { $0.id == runId }) {
            card(instance.snapshotRecord(), live: instance)
        } else if let record = storedRecord {
            card(record, live: nil)
        } else {
            EmptyView()
        }
    }

    private func card(_ record: AgentRunRecord, live: SubagentInstance?) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "person.2.badge.gearshape")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.purple)
                Text(record.role)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
                Spacer()
                SubagentStatusPill(status: record.status)
            }
            HStack(spacing: 5) {
                Text(record.backend.displayName)
                Text("·")
                Text(record.usage.compactDescription)
            }
            .font(.system(size: 9.5))
            .foregroundColor(.secondary)

            if let live, !record.status.isTerminal, !live.liveStatusText.isEmpty {
                Text(live.liveStatusText)
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundColor(.blue)
                    .lineLimit(1)
            } else if let summary = record.report?.summary,
                      !summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(summary)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            Button(action: onOpenInspector) {
                HStack(spacing: 4) {
                    Image(systemName: "sidebar.trailing")
                        .font(.system(size: 9))
                    Text("Open in Inspector")
                        .font(.system(size: 9.5, weight: .medium))
                }
                .foregroundColor(.purple)
            }
            .buttonStyle(.plain)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.purple.opacity(0.05))
        .cornerRadius(6)
        .task(id: runId) {
            if storedRecord == nil {
                let record = await Task.detached(priority: .utility) {
                    AgentRunStore.shared.loadAny(runId: runId)
                }.value
                storedRecord = record
            }
        }
    }
}

// MARK: - Linking a transcript card to its run

/// Resolves which run a `spawn_subagent` record created.
///
/// The id is not an argument (the model never knows it): it is minted by the
/// harness and returned in the tool's JSON, which the record stores as the
/// model-facing rendering. Reading it back from there keeps the card tied to
/// the exact run even when two spawns used the same role and task.
nonisolated enum SubagentRunLink {
    static func runId(from record: ToolCallRecord) -> UUID? {
        runId(fromOutput: record.output ?? "")
            ?? runId(fromOutput: record.error ?? "")
    }

    static func runId(fromOutput output: String) -> UUID? {
        guard let key = output.range(of: "subagent_id") else { return nil }
        let tail = output[key.upperBound...].prefix(160)
        guard let match = tail.range(
            of: #"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"#,
            options: .regularExpression
        ) else { return nil }
        return UUID(uuidString: String(tail[match]))
    }
}
