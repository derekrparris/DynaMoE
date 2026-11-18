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

    /// Calls the summary strip shows before it collapses to a `+N more` chip.
    static let badgeLimit = 8

    /// Cards drawn inline before the list becomes a scroll area. Every card is
    /// another view inside the same transcript row, and an agent loop can issue
    /// dozens of calls in one turn, so an unbounded list is a row that grows
    /// without limit, which is what the compositor refuses to paint.
    ///
    /// When the list does scroll it is the ONLY vertical scroll area in the
    /// list: the cards then draw their output inline, so a transcript row never
    /// contains nested vertical scrollers competing for the wheel.
    static let inlineCardLimit = 5

    /// Height cap for the card list once it scrolls.
    static var cardListHeightCeiling: CGFloat { TranscriptSegmenter.inspectionHeightCeiling }

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
            // Step nodes. The strip is a SUMMARY, not the record — the cards
            // below it are the record — so it shows the first few calls and
            // counts the rest. It also used to be a horizontal `ScrollView`,
            // which meant every multi-call turn allocated another layer inside
            // the transcript row; a long agent turn accumulated dozens of them.
            if toolCalls.count > 1 {
                HStack(spacing: 6) {
                    ForEach(Array(toolCalls.prefix(Self.badgeLimit).enumerated()), id: \.element.id) { index, call in
                        HStack(spacing: 6) {
                            StepNodeBadge(stepIndex: index + 1, call: call)
                            if index < min(toolCalls.count, Self.badgeLimit) - 1 {
                                Image(systemName: "arrow.right")
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.secondary.opacity(0.4))
                            }
                        }
                    }
                    if toolCalls.count > Self.badgeLimit {
                        Text("+\(toolCalls.count - Self.badgeLimit) more")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.08))
                            .cornerRadius(6)
                            .help("\(toolCalls.count) tool calls in this turn; the cards below list them all")
                    }
                }
                .padding(.horizontal, 2)
                .padding(.vertical, 4)
            }

            // Detailed Cards for Each Step. Bounded past a handful of calls: a
            // long agent turn can issue dozens, and the list rides one
            // transcript row. Scrolling keeps every card reachable, which is why
            // the whole list is not truncated the way the badge strip is.
            if isCardListScrollable {
                ScrollView(.vertical, showsIndicators: true) {
                    cardList(toolCalls)
                }
                .frame(maxHeight: Self.cardListHeightCeiling)
            } else {
                cardList(toolCalls)
            }
        }
        .padding(.vertical, 2)
    }

    /// The per-call detail cards, in call order.
    @ViewBuilder
    private func cardList(_ calls: [ToolCallRecord]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(calls) { call in
                TimelineStepCardView(
                    call: call,
                    // One vertical scroll area per transcript row, never two
                    // nested: when this list scrolls, the cards inside it do not.
                    scrollsOutput: !isCardListScrollable,
                    onApprove: { onApproveCall?(call.id) },
                    onReject: { onRejectCall?(call.id) }
                )
            }
        }
    }

    /// True when the card list itself is a scroll area, i.e. when the cards
    /// inside it may not add one of their own.
    private var isCardListScrollable: Bool { toolCalls.count > Self.inlineCardLimit }
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
        case .withheld: return .purple
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
                // Names like `mcp__<server>__<tool>` are long, and up to
                // `badgeLimit` of them share one line of a transcript row that
                // must not grow: without a line limit the strip wraps and makes
                // the row taller at narrow widths. Middle truncation keeps both
                // ends readable, which is the half that identifies the tool.
                .lineLimit(1)
                .truncationMode(.middle)

            if let dur = call.executionDurationSeconds {
                Text(String(format: "%.2fs", dur))
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundColor(.secondary)
                    // The shortest text in the badge: a truncated name must not
                    // push the duration out.
                    .layoutPriority(1)
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
    /// Whether this card owns the vertical scroll area that bounds its output.
    ///
    /// True for a short list, where the card is the only thing that can bound a
    /// `file_read` of 200 KB. False inside a scrollable list, where the list
    /// already bounds the row and a second nested vertical scroll would capture
    /// the wheel and make the transcript feel stuck over a tool-heavy turn.
    var scrollsOutput: Bool = true
    var onApprove: (() -> Void)? = nil
    var onReject: (() -> Void)? = nil

    /// Height cap for the expanded output. A tool result can be a whole file;
    /// the card is an inspection view, not the transcript.
    static let maxOutputHeight: CGFloat = 240

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
        case "skill_load": return "book.closed.fill"
        case "subq_decide": return "arrow.triangle.branch"
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
        case .withheld: return .purple
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
        case "skill_load":
            if let name = call.arguments["name"] {
                let file = call.arguments["file"].map { " · \($0)" } ?? ""
                return name + file
            }
            return "Installed skills"
        case "subq_decide":
            let names = SubQDecideTool.questionNames(from: call.arguments["questions"])
            guard let first = names.first else { return "Decision" }
            return names.count == 1 ? first : "\(first) +\(names.count - 1)"
        case "complete":
            return call.arguments["summary"] ?? "Completed"
        case "spawn_subagent":
            let role = call.arguments["role"] ?? "Subagent"
            let backend = (call.arguments["backend"] ?? "").lowercased()
            return backend == "cloud" ? "\(role) · off-device" : "\(role)"
        case "apply_subagent_changes":
            let raw = call.arguments["subagent_id"] ?? call.arguments["run_id"]
            if let raw,
               let id = UUID(uuidString: raw),
               let live = SubagentManager.shared.getSubagent(byId: id) {
                if let branch = live.modelPlan?.isolation.worktreeBranch {
                    return "\(live.role) · \(branch)"
                }
                return live.role
            }
            if let branch = call.arguments["branch"] { return branch }
            return "Apply isolated changes"
        default:
            return call.rawArguments
        }
    }

    /// This card's output, drawn as bounded chunks so no single text view is
    /// taller than a couple of thousand points. Whether it scrolls is the
    /// caller's decision (see `scrollsOutput`).
    private func outputText(_ out: String) -> some View {
        ChunkedText(text: out, lineSpacing: 2)
            .font(.system(size: 11, design: .monospaced))
            .foregroundColor(.primary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// One expanded field's text, bounded the way `output` is: chunked so no
    /// single text view is oversized, and scroll-capped when this card owns the
    /// scroll area. An argument or an error is as unbounded as output (a
    /// whole-file write payload, a wall of stderr), so it takes the same rule.
    @ViewBuilder
    private func boundedField(_ text: String, fontSize: CGFloat, color: Color) -> some View {
        let chunked = ChunkedText(text: text, lineSpacing: 2)
            .font(.system(size: fontSize, design: .monospaced))
            .foregroundColor(color)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        if scrollsOutput {
            ScrollView(.vertical, showsIndicators: true) { chunked }
                .frame(maxHeight: Self.maxOutputHeight)
        } else {
            chunked
        }
    }

    /// The expanded arguments. Each value is chunked rather than drawn as one
    /// `Text`: a write payload is one argument and can be a whole file.
    private var argumentRows: some View {
        ForEach(call.arguments.sorted(by: { $0.key < $1.key }), id: \.key) { k, v in
            HStack(alignment: .top, spacing: 4) {
                Text("\(k):")
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
                ChunkedText(text: v, lineSpacing: 2)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
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
                    case .withheld:
                        HStack(spacing: 4) {
                            Image(systemName: "shield.lefthalf.filled")
                                .font(.system(size: 11))
                                .foregroundColor(.purple)
                            Text("Withheld in Plan Mode")
                                .font(.system(size: 10.5, weight: .medium))
                                .foregroundColor(.purple)
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
                        destinationSummary: call.name == "subq_decide"
                            ? SubQDecideTool.approvalDetail(
                                rawArguments: call.rawArguments,
                                endpointString: call.pinnedEndpoint
                            )
                            : nil,
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
                            // Same bound as output, and the same single scroll:
                            // an argument can be a whole file, and exactly one of
                            // this card or the list around it scrolls.
                            Group {
                                if scrollsOutput {
                                    ScrollView(.vertical, showsIndicators: true) {
                                        argumentRows
                                    }
                                    .frame(maxHeight: Self.maxOutputHeight)
                                } else {
                                    argumentRows
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

                            // Bounded either way, but bounded ONCE: `ChunkedText`
                            // keeps any one text view small enough to paint, and
                            // exactly one of this card or the list around it
                            // scrolls. When the list is a scroll area it already
                            // bounds the row and keeps every output reachable, so
                            // the card adds no second vertical scroll for the
                            // wheel to get captured by.
                            Group {
                                if scrollsOutput {
                                    ScrollView(.vertical, showsIndicators: true) {
                                        outputText(out)
                                    }
                                    .frame(maxHeight: Self.maxOutputHeight)
                                } else {
                                    outputText(out)
                                }
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
                            boundedField(err, fontSize: 11, color: .red)
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
