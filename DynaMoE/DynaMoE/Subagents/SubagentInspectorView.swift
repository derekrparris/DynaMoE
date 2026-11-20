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
        if costUSD > 0 {
            // A priced figure that is also flagged unpriced is a PARTIAL one
            // (some of the turn could not be priced), so it reads as a floor —
            // without the marker it looks like the exact spend.
            parts.append(String(format: costIsUnpriced ? "$%.4f+" : "$%.4f", costUSD))
        } else if costIsUnpriced {
            parts.append("cost unpriced")
        }
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
    /// Guards the async reload: a slow read for the previous conversation must
    /// not overwrite the new one's rows when it lands later.
    @State private var reloadGeneration: UInt64 = 0
    @State private var confirmClear = false
    /// A discard that could not remove the worktree (a locked file, a git
    /// failure) must say so. The old control closed the alert and left the row
    /// in place with no message, so a confirmed Discard looked like a no-op.
    @State private var discardFailure: String?
    /// The pane defaults to THIS conversation. Runs of the others used to be
    /// reachable through the process-wide drawer list, and deleting that list
    /// left a run working in another chat with no surface that could show,
    /// stop or clear it. This is the way back in, without re-mixing the
    /// default list.
    @State private var showsAllSessions = false
    /// Coalesces store-change notifications. One apply or a bulk delete can
    /// land several writes, and each one would otherwise start its own read.
    @State private var pendingStoreReload: Task<Void, Never>?

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
            // Approvals for a run other than the one being read stay visible:
            // an unanswered card was hidden by merely selecting a run.
            if !orphanApprovals.isEmpty {
                approvalStack(orphanApprovals)
                Divider()
            }
            if let selectedRunId, let detail = detailRecord(for: selectedRunId) {
                SubagentRunDetailView(
                    record: detail.record,
                    live: detail.live,
                    approvals: approvalCenter.pending.filter { $0.request.runId == detail.record.id },
                    onApprove: { approvalCenter.approve($0) },
                    onReject: { approvalCenter.reject($0) },
                    onBack: {
                        manager.selectedSubagentId = nil
                        withAnimation(.easeInOut(duration: 0.15)) { self.selectedRunId = nil }
                    },
                    onClose: onClose,
                    onRecordChanged: reloadStored,
                    onDiscard: {
                        let runId = detail.record.id
                        Task {
                            if await SubagentManager.shared.discard(runId: runId, sessionId: sessionId) {
                                withAnimation(.easeInOut(duration: 0.15)) { self.selectedRunId = nil }
                                reloadStored()
                            } else {
                                discardFailure = "The run was not discarded. An apply may still own it, or its worktree could not be removed (a file in it may be locked, or git refused). Nothing was deleted."
                            }
                        }
                    }
                )
                // A run's pane is reused when the selection moves, and every
                // scratch field below (apply notice, discard confirmation,
                // steering draft) belongs to ONE run. Without this identity an
                // applied run left Apply hidden on the next selection.
                .id(detail.record.id)
                .transition(.opacity)
                // The second alert lives HERE, not beside "Clear finished
                // runs?" on the container: SwiftUI keeps one presentation
                // modifier per view, so chaining two `.alert`s left the one
                // that lost silently dead — and the loser was the only feedback
                // a failed Discard had. It also only applies to the detail pane,
                // which is the only place a discard can be started.
                .alert(
                    "Discard failed",
                    isPresented: Binding(
                        get: { discardFailure != nil },
                        set: { if !$0 { discardFailure = nil } }
                    )
                ) {
                    Button("OK", role: .cancel) { discardFailure = nil }
                } message: {
                    Text(discardFailure ?? "")
                }
            } else {
                runList
            }
        }
        .frame(minWidth: 340, idealWidth: 420, maxWidth: 560)
        .background(Color(NSColor.windowBackgroundColor))
        .onAppear {
            reloadStored()
            consumeManagerSelection()
        }
        .onChange(of: manager.subagents.count) { _, _ in scheduleStoreReload() }
        .onChange(of: manager.activeSubagentsCount) { _, _ in scheduleStoreReload() }
        // A count is not a status. A run moving into an approval wait, or a
        // stored record whose apply completed, changes no count at all; the
        // manager's publisher redraws the pane but does not reload the list.
        // Watching the statuses reloads on exactly those transitions.
        .onChange(of: manager.subagents.map(\.status)) { _, _ in scheduleStoreReload() }
        // The pane is not keyed by conversation, so switching chats left the
        // previous conversation's rows (and any detail selection) on screen.
        // Clear both before loading the new session's records.
        .onChange(of: sessionId) { _, _ in
            selectedRunId = nil
            stored = []
            reloadStored()
            consumeManagerSelection()
        }
        // A notification click sets the manager's selection; the pane keeps its
        // own selection state, so it has to take the value and clear it.
        .onChange(of: manager.selectedSubagentId) { _, _ in consumeManagerSelection() }
        // A store-only mutation changes no manager count or status: an apply
        // that landed on a RESTORED run, a relayed flag, a conversation
        // delete. Without this the pane kept drawing the pre-apply record
        // (stale APPLY / UNREAD markers) until something unrelated published.
        .onReceive(NotificationCenter.default.publisher(for: AgentRunStore.recordsDidChangeNotification)) { _ in
            scheduleStoreReload()
        }
        // "Clear Finished" always asks: the row count is a destructive action
        // even when every report is already read, and only an explicit
        // confirmation should delete durable records.
        .alert("Clear finished runs?", isPresented: $confirmClear) {
            Button("Clear", role: .destructive) { clearFinished() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(clearFinishedMessage)
        }
    }

    /// Pending approvals whose run is not the one the detail pane is showing.
    private var orphanApprovals: [PendingSubagentApproval] {
        guard let selectedRunId else { return approvalCenter.pending }
        return approvalCenter.pending.filter { $0.request.runId != selectedRunId }
    }

    private func consumeManagerSelection() {
        guard let id = manager.selectedSubagentId else { return }
        // Only consume and clear `manager.selectedSubagentId` once the target
        // run is actually visible in the active session. When a notification
        // triggers a cross-session switch (or `reloadStored` is still in
        // flight), clearing immediately lets `onChange(of: sessionId)` wipe
        // `selectedRunId` with the manager's selection already gone.
        guard detailRecord(for: id) != nil else { return }
        selectedRunId = id
        manager.selectedSubagentId = nil
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

            // Scoped to the rows this pane shows. The manager's count is
            // process-wide, so a subagent working in another conversation (or
            // one the API spawned) read as "2 active" over a list with no
            // running run in it.
            if !activeRuns.isEmpty {
                Text("\(activeRuns.count) active")
                    .font(.system(size: 10, weight: .bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.15))
                    .foregroundColor(.blue)
                    .clipShape(Capsule())
            }

            Spacer()

            // The default list is this conversation's. Other chats' runs are
            // reachable here rather than through a second, process-wide list:
            // the rows are read-only except for Stop/Clear, and the border of
            // the pane does not move when another run starts.
            Button(action: {
                showsAllSessions.toggle()
                reloadStored()
            }) {
                HStack(spacing: 3) {
                    Image(systemName: showsAllSessions ? "eye" : "eye.slash")
                        .font(.system(size: 9))
                    Text(showsAllSessions ? "All Runs" : "This Chat")
                        .font(.system(size: 10, weight: .medium))
                }
                .foregroundColor(showsAllSessions ? .purple : .secondary)
            }
            .buttonStyle(.plain)
            .help(showsAllSessions
                ? "Showing runs from every conversation"
                : "Show runs from every conversation (including API turns)")

            if !terminalRuns.isEmpty {
                Button(action: {
                    confirmClear = true
                }) {
                    Text("Clear Finished")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Delete finished runs and their reports from this conversation")
                .accessibilityLabel("Clear finished runs")
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
            .accessibilityLabel("Close inspector")
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
            if totals.unpriced {
                // A total printed beside unpriced runs reads as the whole
                // conversation's spend, which it is not. The plus makes it a
                // floor, and the help names which runs are missing.
                usageStat(
                    totals.cost > 0 ? String(format: "$%.4f+", totals.cost) : "unpriced",
                    "spend"
                )
                .help("One or more runs billed tokens for a model with no recorded price, so their spend is unknown rather than zero. A figure shown with '+' is a floor: it excludes those runs. Record input and output prices under Cloud Providers to see them priced.")
            } else if totals.cost > 0 {
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

    private var sessionTotals: (runs: Int, active: Int, tokens: Int, cost: Double, unpriced: Bool, seconds: Double) {
        var runs = 0
        var active = 0
        var tokens = 0
        var cost = 0.0
        var unpriced = false
        var seconds = 0.0
        for record in allRecords {
            runs += 1
            if !record.status.isTerminal { active += 1 }
            tokens += record.usage.contextTokens
            cost += record.usage.costUSD
            unpriced = unpriced || record.usage.costIsUnpriced
            if record.usage.wallClockSeconds > 0 {
                seconds += record.usage.wallClockSeconds
            } else if let completed = record.completedAt, let started = record.startedAt {
                seconds += completed.timeIntervalSince(started)
            }
        }
        return (runs, active, tokens, cost, unpriced, seconds)
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

    private func approvalStack(_ entries: [PendingSubagentApproval]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("WAITING FOR YOUR DECISION")
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.orange)
            ForEach(entries) { entry in
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
        if activeRuns.isEmpty && terminalRuns.isEmpty && orphanedRuns.isEmpty {
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
                    // A non-terminal record with no live instance (a process
                    // that crashed or was killed before reconcile reached it)
                    // used to be dropped here while `usageStrip` still counted
                    // it: the list said "No Subagent Runs" over "1 run", and
                    // there was no Stop, Clear or detail for it at all.
                    if !orphanedRuns.isEmpty {
                        sectionLabel("ORPHANED")
                        ForEach(orphanedRuns, id: \.id) { record in
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
            manager.selectedSubagentId = nil
            withAnimation(.easeInOut(duration: 0.15)) { selectedRunId = record.id }
        }) {
            SubagentRunRow(
                record: record,
                isLive: liveInstance(record.id) != nil,
                sessionLabel: showsAllSessions ? record.parentSessionId : nil
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: Merge

    /// Stored records are the history; a live instance wins for its own id, so
    /// a run that is still working is never displayed from its last snapshot.
    ///
    /// Computed ONCE per render pass rather than per reader: the header's
    /// counts, the active/restored split and each row all want it, and a
    /// `var` re-merged and re-sorted the whole list for every one of them, on
    /// every body pass (the live list changes with every streamed token).
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

    /// Runs that are actually working. Non-terminal is not enough on its own:
    /// a record from a process that is gone stays non-terminal until
    /// `reconcileInterruptedRuns` reaches it, and counting it as active makes
    /// the header promise work that is not happening.
    private var activeRuns: [AgentRunRecord] {
        allRecords.filter { !$0.status.isTerminal && liveInstance($0.id) != nil }
    }

    /// Non-terminal records with no live instance behind them: shown, but as
    /// what they are — a run whose process is gone.
    private var orphanedRuns: [AgentRunRecord] {
        allRecords.filter { !$0.status.isTerminal && liveInstance($0.id) == nil }
    }

    private var terminalRuns: [AgentRunRecord] {
        allRecords.filter(\.status.isTerminal)
    }

    /// The rows "Clear Finished" will actually delete.
    ///
    /// `allRecords` is what the LIST shows, which with `showsAllSessions` on
    /// includes other conversations' runs. `clearCompleted(sessionId:)` only
    /// removes this conversation's rows plus the UNSCOPED ones (a run started
    /// by an API turn belongs to no chat and rides every inspector), so the
    /// confirmation used to promise to delete rows the button leaves alone.
    private var clearableRuns: [AgentRunRecord] {
        terminalRuns.filter { record in
            let ownedByThisConversation = sessionId == nil
                || record.parentSessionId == sessionId
                || record.parentSessionId == nil
            return ownedByThisConversation && !record.holdsWorktreeWork
        }
    }

    /// Finished runs whose report the coordinator has never been shown. The
    /// bulk clear still removes them (the operator asked), but it says how
    /// many unread reports go with it first.
    private var unreadTerminalCount: Int {
        clearableRuns.filter { record in
            guard !record.summaryRelayed else { return false }
            // The relay's own fallback: a run with no report can still carry
            // the failure text the coordinator has never been shown.
            let summary = record.report?.summary ?? record.error ?? ""
            return !summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }.count
    }

    private func clearFinished() {
        withAnimation(.easeInOut(duration: 0.2)) {
            manager.clearCompleted(sessionId: sessionId)
            reloadStored()
        }
    }

    /// The confirmation says exactly what goes and what is remembered by the
    /// coordinator. `Clear Finished` no longer runs unconfirmed: a terminal
    /// record and its report are durable data, not a view filter.
    private var clearFinishedMessage: String {
        let count = clearableRuns.count
        var text = "This deletes \(count) finished run\(count == 1 ? "" : "s") and their reports from this conversation. It cannot be undone."
        let keptForWorktree = terminalRuns.count - count
        if keptForWorktree > 0 {
            text += " \(keptForWorktree) more hold an unapplied worktree and stay until you keep or clear each one."
        }
        if unreadTerminalCount > 0 {
            text = "\(unreadTerminalCount) of them still \(unreadTerminalCount == 1 ? "has" : "have") a report the coordinator has not seen. " + text
        }
        if showsAllSessions {
            text += " Rows from other conversations are left alone."
        }
        return text
    }

    /// Runs started outside a conversation (an API/agent turn with no chat
    /// session) have no bucket of their own, so they ride every inspector
    /// rather than becoming invisible in the one that happens to be open.
    private var liveInstances: [SubagentInstance] {
        if showsAllSessions { return manager.allSubagents }
        return manager.allSubagents.filter {
            sessionId == nil || $0.parentSessionId == sessionId || $0.parentSessionId == nil
        }
    }

    func liveInstance(_ runId: UUID) -> SubagentInstance? {
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
        reloadGeneration &+= 1
        let generation = reloadGeneration
        let sessionId = sessionId
        let allSessions = showsAllSessions
        Task {
            let records = await Task.detached(priority: .utility) {
                if allSessions {
                    return AgentRunStore.shared.loadAll()
                }
                var records = AgentRunStore.shared.load(sessionId: sessionId)
                if sessionId != nil {
                    records += AgentRunStore.shared.load(sessionId: nil)
                }
                return records
            }.value
            guard generation == self.reloadGeneration else { return }
            self.stored = records
            self.consumeManagerSelection()
        }
    }

    /// Debounces a burst of store writes into one reload. A run finishing
    /// writes several records within a few milliseconds, and an apply or a
    /// bulk delete more.
    private func scheduleStoreReload() {
        pendingStoreReload?.cancel()
        pendingStoreReload = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 150_000_000)
            guard !Task.isCancelled else { return }
            reloadStored()
        }
    }
}

// MARK: - Row

private struct SubagentRunRow: View {
    let record: AgentRunRecord
    let isLive: Bool
    /// The owning conversation, shown only while the pane lists every one.
    var sessionLabel: UUID? = nil

    private var hasUnreadReport: Bool {
        guard record.status.isTerminal, !record.summaryRelayed else { return false }
        let summary = record.report?.summary ?? record.error ?? ""
        return !summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

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

                if hasUnreadReport {
                    Text("UNREAD")
                        .font(.system(size: 8, weight: .bold))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color.purple.opacity(0.14))
                        .foregroundColor(.purple)
                        .clipShape(Capsule())
                        .help("The coordinator has not received this report yet")
                }

                SubagentStatusPill(status: record.status, isLive: isLive)
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
                if let sessionLabel {
                    Text("·")
                    Text(String(sessionLabel.uuidString.prefix(8)))
                        .help("Conversation \(sessionLabel.uuidString)")
                }
                Spacer()
                // A live run's duration is Date() minus its start, and a
                // computed property alone only re-reads when something ELSE
                // redraws the row -- so a run that streams nothing (waiting on
                // an approval, a slow tool) showed a frozen clock. A timeline
                // ticks it once a second while it is live, and costs nothing
                // for a finished row, whose text is fixed.
                if isLive && !record.status.isTerminal {
                    TimelineView(.periodic(from: .now, by: 1)) { _ in
                        Text(durationText)
                    }
                } else {
                    Text(durationText)
                }
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
        // A live clock only while something is live: a non-terminal record
        // whose process is gone would otherwise tick forever against a start
        // time nothing is advancing.
        if isLive, let started = record.startedAt {
            return String(format: "%.0fs…", Date().timeIntervalSince(started))
        }
        if let started = record.startedAt {
            return String(format: "%.0fs (not running)", Date().timeIntervalSince(started))
        }
        return record.usage.compactDescription
    }
}

// MARK: - Status pill

private struct SubagentStatusPill: View {
    let status: AgentRunStatus
    /// Whether a live instance exists for this run. A stored record left
    /// non-terminal (a crashed or still-reconciling process) reads as active
    /// from its status alone, and drawing it as live shows an indeterminate
    /// spinner for a run that is gone.
    var isLive: Bool = true

    var body: some View {
        HStack(spacing: 4) {
            if status.isActive && isLive {
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
    /// Deletes this finished run and, for an unapplied isolated one, its
    /// worktree. The bulk "Clear Finished" deliberately skips those, so this
    /// is the operator's only way to decide a writer's work is not wanted.
    var onDiscard: () -> Void

    @State private var messageDraft = ""
    @State private var showThinking = true
    @State private var isApplying = false
    @State private var applyError: String?
    @State private var applyNotice: String?
    @State private var appliedHereAt: Date?
    @State private var keptIgnoredHere: [String] = []
    @State private var confirmDiscard = false

    /// The record is a value snapshot, so an apply performed from this pane is
    /// shown from local state until the parent reloads; the two are the same
    /// date once it does.
    private var appliedAt: Date? { appliedHereAt ?? record.appliedAt }

    private var hasKeptIgnoredFiles: Bool {
        !(record.keptIgnoredFiles?.isEmpty ?? true) || !keptIgnoredHere.isEmpty
    }

    private var canApplyChanges: Bool {
        appliedHereAt == nil && SubagentApplyControl.canApply(
            record,
            executionStillUnwinding: !(SubagentManager.shared.getSubagent(byId: record.id)?.isExecutionUnwound ?? true),
            // `apply` refuses while any checkout is mid-apply, so the button
            // has to read the same rule instead of answering with a refusal.
            otherApplyInFlight: SubagentApplyService.hasAnyRepositoryInFlight
                || SubagentApplyService.isRunInFlight(record.id)
        )
    }

    /// Any finished isolated run that still holds a worktree (either unapplied,
    /// or applied with git-ignored files kept in the tree) needs a per-run
    /// discard: "Clear Finished" deliberately skips both so work is never lost
    /// in bulk. Wait for execution to unwind so Stop→Discard cannot remove the
    /// tree under a still-exiting child.
    private var canDiscard: Bool {
        record.status.isTerminal
            && record.isolation.isIsolated
            && (appliedAt == nil || hasKeptIgnoredFiles)
            && !isApplying
            && !SubagentApplyService.isRunInFlight(record.id)
            // `discard` refuses while ANY checkout is mid-apply, because a
            // record's recorded root can be stale. The button has to read the
            // same rule or a confirmed discard silently does nothing.
            && !SubagentApplyService.hasAnyRepositoryInFlight
            && (SubagentManager.shared.getSubagent(byId: record.id)?.isExecutionUnwound ?? true)
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
        .alert("Discard this run?", isPresented: $confirmDiscard) {
            Button("Discard", role: .destructive) { onDiscard() }
                .disabled(
                    isApplying
                        || SubagentApplyService.isRunInFlight(record.id)
                        || SubagentApplyService.hasAnyRepositoryInFlight
                )
            Button("Cancel", role: .cancel) {}
        } message: {
            if appliedAt != nil {
                Text("This run's tracked changes are already applied. Discarding deletes its record, its kept worktree (including any git-ignored files), and its branch. This cannot be undone.")
            } else {
                Text("The run's record and its unapplied worktree are deleted. Its branch is removed too. This cannot be undone; apply the run first if you want to keep the work.")
            }
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
                SubagentStatusPill(status: record.status, isLive: live != nil)
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
                .accessibilityLabel("Reveal the working directory in Finder")
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
                        .accessibilityLabel("Reveal the run's worktree in Finder")
                    }
                    Spacer(minLength: 0)
                }
                metaRow("Branch", branch)
                if let appliedAt {
                    metaRow("Applied", Self.dateFormatter.string(from: appliedAt))
                    // The record answers "where did it go": an apply lands on
                    // whatever was checked out, and the branch is recorded for
                    // exactly that reason.
                    if let destination = record.appliedToBranch {
                        metaRow("Applied to", destination)
                    }
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
                // Indexed, not `id: \.self`: a model-authored list can repeat a
                // path, and duplicate ids make SwiftUI drop rows.
                ForEach(Array(report.changedFiles.enumerated()), id: \.offset) { _, file in
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
                ForEach(Array(report.risks.enumerated()), id: \.offset) { _, risk in
                    Text("• " + risk)
                        .font(.system(size: 10))
                        .foregroundColor(.orange)
                }
            }

            if !report.openQuestions.isEmpty {
                reportHeading("Open questions")
                ForEach(Array(report.openQuestions.enumerated()), id: \.offset) { _, question in
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

                if canDiscard {
                    Button(role: .destructive) {
                        confirmDiscard = true
                    } label: {
                        Label("Discard Run", systemImage: "trash")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .disabled(isApplying || SubagentApplyService.isRunInFlight(record.id))
                    .help("Delete this run's record and its unapplied worktree. Apply it first to keep the work.")
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
            if let applyNotice {
                Text(applyNotice)
                    .font(.system(size: 10))
                    .foregroundColor(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func applyChanges() {
        guard !isApplying else { return }
        isApplying = true
        applyError = nil
        applyNotice = nil
        Task {
            do {
                let outcome = try await SubagentApplyControl.apply(runId: record.id)
                appliedHereAt = outcome.appliedAt
                keptIgnoredHere = outcome.keptIgnoredFiles
                if !outcome.keptIgnoredFiles.isEmpty {
                    applyNotice = "Applied the tracked patch, but kept the worktree: it also holds git-ignored files a patch cannot carry (\(outcome.keptIgnoredFiles.joined(separator: ", "))). Retrieve them manually before clearing the run."
                }
                if outcome.keptBranch {
                    let branchNote = "Kept branch '\(outcome.branch)' because it holds commits outside the applied working tree. Inspect or delete it manually when you are done."
                    applyNotice = [applyNotice, branchNote].compactMap { $0 }.joined(separator: " ")
                }
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
    /// Bumped by a store change so the load re-runs. `task(id: runId)` alone
    /// never re-fired: an applied or deleted record kept whatever the first
    /// read returned (a stale summary, or a row the operator had deleted).
    @State private var storeRevision = 0
    @State private var storeReload: Task<Void, Never>?

    var body: some View {
        // The task is attached HERE, not to the card: when neither a live
        // instance nor a stored record is present the card branch renders the
        // "not stored" row below, and a task inside the record branch never ran
        // for a run whose record loaded after the first frame.
        Group {
            if let instance = manager.allSubagents.first(where: { $0.id == runId }) {
                card(instance.snapshotRecord(), live: instance)
            } else if let record = storedRecord {
                card(record, live: nil)
            } else {
                // The transcript keeps the card even after the run aged out of
                // retention (or the manager pruned its instance). An empty view
                // made a spawn call look like it produced nothing.
                missingCard
            }
        }
        .task(id: storeRevision) {
            let record = await Task.detached(priority: .utility) {
                AgentRunStore.shared.loadAny(runId: runId)
            }.value
            guard !Task.isCancelled else { return }
            storedRecord = record
        }
        .onReceive(NotificationCenter.default.publisher(for: AgentRunStore.recordsDidChangeNotification)) { _ in
            scheduleStoreReload()
        }
    }

    /// Coalesces store notifications.
    ///
    /// A single apply, a conversation delete or a bulk clear lands several
    /// writes, and bumping the revision on each one started a detached
    /// `loadAny` per write — the card is in the transcript, so its cost is paid
    /// for every visible spawn row. The inspector's pane debounces the same
    /// notifications the same way.
    private func scheduleStoreReload() {
        storeReload?.cancel()
        storeReload = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 150_000_000)
            guard !Task.isCancelled else { return }
            storeRevision &+= 1
        }
    }

    /// What is left of a spawn whose run has no live instance and no stored
    /// record: the transcript keeps a row, and it says why there is nothing
    /// more to show.
    private var missingCard: some View {
        HStack(spacing: 6) {
            Image(systemName: "person.2.badge.gearshape")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)
            Text("Subagent run \(runId.uuidString.prefix(8))")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)
            Text("· details are no longer stored")
                .font(.system(size: 9.5))
                .foregroundColor(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.05))
        .cornerRadius(6)
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
                // A restored record has no live instance: its status alone would
                // draw an indeterminate spinner for a run that is gone.
                SubagentStatusPill(status: record.status, isLive: live != nil)
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
