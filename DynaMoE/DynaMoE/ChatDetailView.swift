//
//  ChatDetailView.swift
//  DynaMoE
//

import SwiftUI
import AppKit
import ImageIO
import UniformTypeIdentifiers

// Memoization note: several views below keep parse, cleanup, or highlight
// results in a `@State`-held reference type and refresh it in place from
// `body`. That is deliberate: mutating a class property does not write
// `@State`, so it cannot trigger "modifying state during view update", while a
// struct would force a copy or an update per refresh. Do not "simplify" these
// memo types into value types.

struct ChatDetailView: View {
    @ObservedObject private var zoomManager = AppZoomManager.shared
    @Binding var session: ChatSession?
    @Binding var promptText: String
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared
    
    var isGenerating: Bool
    /// True while any turn is live anywhere in the process, including a
    /// local-API one. The mode pill is a promise to the turn already running, so
    /// it is disabled for the duration instead of offering a change the app will
    /// refuse.
    var isModeChangeLocked: Bool = false
    /// The progress bubble /compact is streaming its summary into, or nil.
    /// Rendering flags are scoped to this message id, never a bare
    /// "compacting" boolean, so historical bubbles keep their settled
    /// Markdown treatment while the pass runs.
    var compactionProgressMessageId: UUID? = nil
    var isStreamingOffDisk: Bool = false
    var generationSpeed: Double
    var generationTokens: Int
    var jetSpecEnabled: Bool = false
    var jetSpecMeanTau: Double = 1.0
    var jetSpecDraftAccepted: Int = 0
    var modelName: String?
    /// True when the active model emits the native Gemma tool-call dialect. Gates the
    /// Gemma-specific display cleanup so a non-Gemma answer that merely documents the
    /// `<|tool_call>` token is not truncated.
    var isGemmaDialect: Bool = false
    var tokenizer: DynaMoeTokenizer? = nil
    var activeProfile: ModelProfileType = .coder
    var onSelectProfile: ((ModelProfileType) -> Void)? = nil
    var supportsThinking: Bool = false
    var isThinkingEnabled: Bool = true
    /// Resolved graded thinking level shown as selected when
    /// `supportsThinkingLevels`; nil renders the model default (High).
    var thinkingEffort: ThinkingEffort? = nil
    /// Whether the model's chat template exposes graded thinking levels
    /// (Spark X2.6's `reasoning_effort`). When false, the plain On/Off menu shows.
    var supportsThinkingLevels: Bool = false
    /// Whether the active model can be sent images. Cloud models opt in per model
    /// in Settings; local weights report true only for a checkpoint that ships a
    /// vision tower, and the composer refuses an attachment for every other
    /// family rather than sending pixels that would be silently dropped.
    var supportsVision: Bool = false
    /// What the chat window is momentarily saying: a refused send, a turn
    /// outcome, a model that failed to load. The window hands it down so it is
    /// drawn in the same layer as the composer's notices, above the composer
    /// rather than on top of it.
    var windowNotice: StatusNotice? = nil
    /// The chat store cannot read or save. That is a standing condition rather
    /// than an event, so it is not a notice: it stays until the store recovers,
    /// and it is drawn at the top of the transcript rather than over the whole
    /// column, whose header controls must stay reachable for the whole session.
    var isHistoryUnavailable: Bool = false
    /// What the active model's vision tower will cost for a staged image, phrased
    /// for the chip ("about 5 s"). `nil` when the backend prices nothing, which
    /// keeps every non-local and non-vision model's chip exactly as it was.
    var visionCostLabel: ((PreparedImage) -> String?)? = nil
    var isAgentToolsEnabled: Bool = true
    var isModelLoaded: Bool = true
    var isAppleFoundationModelActive: Bool = false
    /// True when the active backend is a connected cloud provider. Cloud turns
    /// support tools through the provider's native function calling, so the
    /// tool controls stay enabled.
    var isCloudModelActive: Bool = false
    /// The system on-device model is the only backend still restricted to
    /// plain chat; cloud providers call tools through the API's function
    /// interface and share the local agent controls.
    private var isPlainChatOnlyBackend: Bool {
        isAppleFoundationModelActive
    }

    /// The conversation's agent surface. `nil` (every conversation that predates
    /// plan mode) is the coder.
    private var isPlanMode: Bool {
        session?.agentMode == .plan
    }

    /// The mode pill's tooltip. When a turn is live it says WHY the control is
    /// unavailable: a disabled pill still explaining what plan mode is reads as
    /// a broken toggle rather than a promise being kept.
    private var planPillHelp: String {
        if isModeChangeLocked {
            return "The agent's mode cannot change while a turn is running (⇧⌘P)."
        }
        return isPlanMode
            ? "Plan Mode is ON: DynaMoE reads and searches only, then writes an implementation plan for your approval. No file writes, shell commands, git commits or MCP tools (⇧⌘P)."
            : "Click to plan before implementing: DynaMoE reads and searches only, then proposes a plan you approve (⇧⌘P)."
    }

    /// The assistant message carrying a finished plan, when the conversation is
    /// in plan mode, the plan ended, and the user has not dismissed that plan's
    /// handoff. Derived rather than recorded so it cannot desync from whichever
    /// asynchronous path ended the turn.
    ///
    /// Gated on `!isGenerating`: the marker streams in as the LAST line of the
    /// answer, so for the rest of that same turn the bar would be on screen
    /// offering a send the app would refuse. Approving then flipped the mode and
    /// sent nothing.
    ///
    /// Searches for the last READY message rather than taking the last assistant
    /// message: the harness appends a forced-synthesis or subagent-relay reply
    /// after a plan, and keying the bar to the final message hid it while the
    /// plan card still said the plan was ready.
    ///
    /// A plan is only current until the next user turn. Once a follow-up has
    /// been sent, the older plan's handoff must not come back when that
    /// follow-up is answered without markers — "Implement the plan." would then
    /// point at a plan the conversation has already moved past. A follow-up that
    /// itself produces a plan is a NEWER ready message, so this search finds it.
    private var readyPlanMessage: ChatMessage? {
        guard isPlanMode, !isGenerating, let session else { return nil }
        guard let planIndex = session.messages.lastIndex(where: {
            $0.role == .assistant
                && PlanMarkers.anyMarkerMentioned(in: $0.content)
                && PlanMarkers.readyPresent(in: $0.content)
        }) else { return nil }
        if let lastUserIndex = session.messages.lastIndex(where: { $0.role == .user }),
           lastUserIndex > planIndex {
            return nil
        }
        let message = session.messages[planIndex]
        guard dismissedPlanMessageId != message.id else { return nil }
        return message
    }
    var onSendMessage: (String, [PreparedImage]) -> Bool
    var onStopGeneration: () -> Void
    var onQueuePrompt: ((String, [PreparedImage]) -> Bool)? = nil
    var onSendImmediate: ((String, [PreparedImage]) async -> Bool)? = nil
    var onRemoveQueuedPrompt: ((UUID) -> Void)? = nil
    var onSelectDiscoveredModel: ((DiscoveredModel) -> Void)? = nil
    var onOpenSettings: (() -> Void)? = nil
    /// Sets the conversation's thinking mode: enabled flag plus the graded
    /// effort (nil for plain on/off models or when disabling).
    var onSetThinking: ((Bool, ThinkingEffort?) -> Void)? = nil
    var onToggleAgentTools: ((Bool) -> Void)? = nil
    /// Enters or leaves plan mode for this conversation. The composer's mode
    /// pill and `/plan` both land here.
    var onSetPlanMode: ((Bool) -> Void)? = nil
    /// Confirms a ready plan. The Bool asks for Turbo Mode on the implement
    /// turn — the analogue of Crush's "Code with YOLO".
    var onApprovePlan: ((Bool) -> Void)? = nil
    var onRenameSession: ((String) -> Void)? = nil

    @FocusState private var isInputFocused: Bool
    @FocusState private var isSessionTitleFocused: Bool
    @State private var isRenamingSession = false
    @State private var sessionTitleDraft = ""
    @AppStorage("dynamoe_reasoning_expanded_by_default") private var reasoningExpandedByDefault: Bool = false
    @State private var isReasoningExpanded: [UUID: Bool] = [:]
    /// Expansion state of a `/compact` digest card, keyed by message id. The
    /// card defaults to expanded: keeping the summary readable is the point of
    /// the card, and collapsing is the opt-out for reclaiming the space.
    @State private var isCompactionSummaryExpanded: [UUID: Bool] = [:]
    /// The plan whose handoff bar the user dismissed. The plan itself is derived
    /// from message content, so this is the only piece of handoff state that has
    /// to be remembered; being view state, it resets on relaunch and a dismissed
    /// plan comes back rather than disappearing for good.
    @State private var dismissedPlanMessageId: UUID? = nil
    @State private var promptTokenCount: Int = 0
    @State private var tokenCountTask: Task<Void, Never>? = nil
    @State private var isPinnedToBottom: Bool = true
    /// Identity of the invisible marker at the end of the transcript. It must
    /// stay constant: bumping it per streamed chunk re-lays out the whole
    /// transcript several times a frame, which SwiftUI flags as an invalid
    /// configuration and which can leave a long conversation blank until a
    /// manual scroll realizes its rows again.
    private static let scrollBottomAnchorID = "chat-scroll-bottom-anchor"

    /// The draft currently in flight through the async immediate-send path.
    /// The send suspends while it waits out a generation teardown, so without
    /// this a second Cmd-Return/bolt press would submit the same text again.
    /// `nil` when no immediate send is pending.
    @State private var pendingImmediateSend: String? = nil
    /// Bumped on every edit to the draft. The immediate-send completion clears the
    /// composer only when this still equals the revision it submitted, so a newer
    /// draft is never clobbered even if the user edited away and back to the same
    /// text.
    @State private var draftRevision: UInt64 = 0
    /// Queued prompts with an in-flight "Send Now", so re-activating the same
    /// row cannot launch two tasks for it.
    @State private var sendingQueuedIds: Set<UUID> = []
    /// Images staged in the composer, not yet persisted. They become
    /// `ImageAttachmentRef`s only when a send is accepted, so a draft the user
    /// abandons leaves nothing on disk.
    @State private var pendingImages: [PreparedImage] = []
    /// What the composer is momentarily saying — a refused attachment, a decode
    /// in flight. It is drawn above the composer rather than added to its rows,
    /// because the composer's height is what sizes the transcript above it.
    @State private var composerNotice: StatusNotice? = nil
    /// Mints notice identities, so a repeat of the same text still restarts the
    /// dismissal timer.
    @State private var noticeSerial: UInt64 = 0
    /// The in-flight auto-dismissal, cancelled whenever the notice changes.
    @State private var noticeDismissTask: Task<Void, Never>? = nil
    @State private var isAttachmentDropTargeted = false
    /// How much of the transcript the floating layer is covering. The layer is an
    /// overlay so it cannot resize anything, which also means the last message
    /// could sit under it forever; the scroll content reserves this much instead.
    @State private var floatingLayerHeight: CGFloat = 0
    /// The same reservation at the top, for the standing history warning.
    @State private var historyWarningHeight: CGFloat = 0

    /// Distance-from-bottom bookkeeping for the pin/detach logic.
    private struct ScrollSnapshot: Equatable {
        var offset: CGFloat
        var contentHeight: CGFloat
        var containerHeight: CGFloat

        var distanceFromBottom: CGFloat {
            contentHeight - (offset + containerHeight)
        }
    }
    @AppStorage("dynamoe_agent_turbo_mode") private var isTurboModeEnabled: Bool = false
    @ObservedObject private var subagentManager = SubagentManager.shared

    /// Characters of streamed output that make up one follow-scroll bucket,
    /// roughly one line. This is the knob that controls how often the
    /// transcript scrolls while a turn streams: smaller follows more often.
    private static let streamedFollowBucketCharacters = 120

    /// Coarse progress marker for the newest message (content or thinking
    /// grows — a chat turn, or the /compact progress bubble typing out a
    /// summary). The follow handler keys off buckets of roughly a line of
    /// output instead of the exact length: a per-token value made the handler
    /// run several times per frame, and every run had to invalidate layout to
    /// scroll, which a long transcript cannot absorb.
    private var streamedContentBucket: Int {
        guard let last = session?.messages.last else { return 0 }
        return (last.content.count + (last.thinkingContent?.count ?? 0))
            / Self.streamedFollowBucketCharacters
    }

    /// Scrolls the transcript to its newest line. Callers are throttled by
    /// buckets, message counts, and turn boundaries, never by per-token
    /// updates, and the jump is deferred so the frame carrying the growth is
    /// laid out before the scroll target is resolved.
    ///
    /// Streaming follows jump, because an animation per bucket would queue
    /// faster than it completes. Discrete transitions (a message landing, a
    /// turn starting or ending) animate, as they did before the throttle.
    private func followBottom(_ proxy: ScrollViewProxy, animated: Bool = false) {
        DispatchQueue.main.async {
            if animated {
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo(Self.scrollBottomAnchorID, anchor: .bottom)
                }
            } else {
                proxy.scrollTo(Self.scrollBottomAnchorID, anchor: .bottom)
            }
        }
    }

    /// Lands a freshly compacted transcript on its digest card. That state is
    /// exactly two rows, the card followed by its marker, and the card is what
    /// compaction just produced: following the bottom instead can leave it far
    /// above the viewport when the summary is long. Returns true when it
    /// handled the scroll, so callers fall through to their normal follow.
    ///
    /// The pin is left to the geometry handler on purpose. Landing at the top of
    /// a long card scrolls away from the bottom, and that offset change is what
    /// unpins. Clearing it here instead would strand a summary that fits the
    /// viewport as unpinned: the scroll is a no-op there, so no geometry change
    /// arrives to correct it, and only the next turn would re-pin.
    @discardableResult
    private func landOnCompactedSummary(_ proxy: ScrollViewProxy) -> Bool {
        guard let first = session?.messages.first,
              first.isCompactionSummary == true,
              session?.messages.count == 2 else { return false }
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(first.id, anchor: .top)
            }
        }
        return true
    }

    /// True while a /compact pass is streaming into its progress bubble.
    /// Only the scroll handlers need the conversation-level bit; everything
    /// rendered is keyed to `compactionProgressMessageId` instead.
    private var isCompactingConversation: Bool {
        compactionProgressMessageId != nil
    }

    /// Minimum height of the composer's text view that still fits exactly one
    /// line: the editor font's line height plus its top and bottom insets.
    /// Deriving this from the real font metrics keeps the first line and the
    /// placeholder from being clipped when the composer is collapsed, while the
    /// surrounding spacing is trimmed instead to keep the box compact.
    private var composerMinHeight: CGFloat {
        let font = NSFont.systemFont(ofSize: max(10, 13.5 * zoomManager.zoomScale))
        let lineHeight = ceil(font.ascender - font.descender + font.leading)
        return lineHeight + MacTextEditor.textContainerInsetHeight * 2
    }

    /// The context window recorded for this conversation's cloud model, or nil.
    ///
    /// Nil for a weights model (whose capacity is the KV budget's business, not
    /// a provider's) and for any cloud model nobody recorded a window for — in
    /// which case the header shows the plain token total instead of inventing a
    /// ceiling.
    ///
    /// Resolved from the recorded window facts directly rather than through
    /// `CloudModelRegistry.registeredSpecs`: this runs on every body
    /// evaluation, and that call rebuilds every spec on the provider (its model
    /// list, its no-tools list, its vision marks) to answer one lookup. A
    /// Settings edit still lands on the next evaluation, because nothing is
    /// cached.
    private func cloudContextWindow(for session: ChatSession?) -> Int? {
        guard let id = session?.selectedModelId,
              let parsed = CloudModelAddress.parseModelId(id) else { return nil }
        return CloudContextPolicy.window(
            forModelId: parsed.providerModelId,
            providerId: parsed.provider.id
        )
    }

    private func modelIconName(for name: String?, isCloud: Bool = false) -> String {
        if isCloud {
            return "cloud.fill"
        }
        let lower = (name ?? "").lowercased()
        if lower.contains("apple foundation") {
            return "apple.logo"
        } else if lower.contains("ornith") || lower.contains("bird") {
            return "bird.fill"
        } else if lower.contains("nanbeige") {
            return "building.columns.fill"
        } else if lower.contains("deepseek") {
            return "sparkles"
        } else if lower.contains("qwen") {
            return "cpu.fill"
        } else if lower.contains("llama") {
            return "flame.fill"
        }
        return "cube.fill"
    }

    private func beginSessionRename() {
        sessionTitleDraft = session?.title ?? ""
        isRenamingSession = true
    }

    private func commitSessionRename() {
        let trimmed = sessionTitleDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            onRenameSession?(trimmed)
        }
        isRenamingSession = false
    }

    /// One row of the model selector menu. Shared by the system-models and
    /// Hugging Face sections so the AFM entry renders identically to a real
    /// snapshot model (checkmark, size, capability badges).
    private func discoveredModelMenuButton(_ dm: DiscoveredModel) -> some View {
        let isCurrent = (session?.selectedModelId == dm.id) ||
                        (session?.selectedModelPath == dm.snapshotPath) ||
                        (modelName == dm.displayName) ||
                        (session?.selectedModelName == dm.displayName)
        return Button(action: {
            onSelectDiscoveredModel?(dm)
        }) {
            HStack {
                if isCurrent {
                    Image(systemName: "checkmark")
                }
                Text(dm.displayName)
                Text("(\(dm.formattedSize))")
                if dm.isMoE {
                    Text("• MoE")
                }
                if dm.supportsThinking {
                    Text("• 🧠 Thinking")
                }
                if dm.isAppleFoundationModel {
                    // Decorative branding glyph; the row title already says
                    // Apple, so keep it out of the VoiceOver reading.
                    Text("• 🍎").accessibilityHidden(true)
                }
                if dm.isCloudModel {
                    // The display name already carries the provider in
                    // parentheses; this is a decorative marker only.
                    Text("• ☁️").accessibilityHidden(true)
                }
            }
        }
    }

    /// Shared chrome for the Thinking menu label, so the graded-level and
    /// plain on/off menus cannot drift apart.
    @ViewBuilder
    private func thinkingMenuLabel(_ title: String, isOn: Bool) -> some View {
        HStack(spacing: 5) {
            Image(systemName: isOn ? "brain.head.profile" : "bolt.slash")
                .font(.system(size: max(8.5, 10 * zoomManager.zoomScale)))
                .foregroundColor(isOn ? .purple : .secondary)
            Text(title)
                .font(.system(size: max(9.5, 11.5 * zoomManager.zoomScale), weight: .medium))
                .foregroundColor(isOn ? .primary : .secondary)
            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: max(6.5, 7.5 * zoomManager.zoomScale), weight: .semibold))
                .foregroundColor(.secondary.opacity(0.7))
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 4.5)
        .background(isOn ? Color.purple.opacity(0.12) : Color.secondary.opacity(0.08))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isOn ? Color.purple.opacity(0.3) : Color.clear, lineWidth: 1)
        )
    }

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                // Header Bar (Antigravity breadcrumb style)
                HStack(spacing: 8) {
                    if isRenamingSession {
                        TextField("Session name", text: $sessionTitleDraft)
                            .textFieldStyle(.plain)
                            .font(.system(size: max(10, 13 * zoomManager.zoomScale), weight: .semibold))
                            .foregroundColor(.primary)
                            .frame(minWidth: max(120, 180 * zoomManager.zoomScale), maxWidth: max(200, 360 * zoomManager.zoomScale))
                            .focused($isSessionTitleFocused)
                            .onSubmit { commitSessionRename() }
                            .onExitCommand { isRenamingSession = false }
                            .onAppear { isSessionTitleFocused = true }
                    } else {
                        Text(session?.displayTitle ?? "App")
                            .font(.system(size: max(10, 13 * zoomManager.zoomScale), weight: .semibold))
                            .foregroundColor(.primary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .onTapGesture(count: 2) { beginSessionRename() }
                            .help("Double-click to rename")

                        Button(action: beginSessionRename) {
                            Image(systemName: "pencil")
                                .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .medium))
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Rename Conversation")
                    }
                    
                    Spacer()
                    
                    if let model = modelName {
                        HStack(spacing: 6) {
                            Image(systemName: modelIconName(for: model, isCloud: isCloudModelActive))
                                .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .semibold))
                                .foregroundColor(isGenerating ? (isStreamingOffDisk ? Color.orange : Color.green) : Color.purple)
                            Text(model)
                                .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                                .foregroundColor(.secondary)
                            if isGenerating {
                                Text("•")
                                    .foregroundColor(.secondary)
                                if let activePrefill = session?.messages.last?.prefillStatus {
                                    HStack(spacing: 4) {
                                        Image(systemName: "arrow.down.circle.fill")
                                            .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .bold))
                                            .foregroundColor(.purple)
                                        Text(activePrefill)
                                            .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .semibold, design: .monospaced))
                                            .foregroundColor(.purple)
                                    }
                                } else {
                                    HStack(spacing: 3) {
                                        Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")
                                            .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .bold))
                                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                        Text(String(format: "%.1f tok/s", generationSpeed))
                                            .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                                            .fontWeight(.bold)
                                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                    }
                                    if jetSpecEnabled && (jetSpecMeanTau > 1.0 || jetSpecDraftAccepted > 0) {
                                        Text("•")
                                            .foregroundColor(.secondary.opacity(0.6))
                                        HStack(spacing: 3) {
                                            Image(systemName: "bolt.badge.sparkle")
                                                .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .bold))
                                            .foregroundColor(.cyan)
                                            Text(String(format: "🚀 JetSpec τ=%.1f", jetSpecMeanTau))
                                                .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .bold, design: .monospaced))
                                                .foregroundColor(.cyan)
                                        }
                                    }
                                }
                            }
                            if let cumulativeTokens = session?.cumulativeTokenCount, cumulativeTokens > 0 {
                                let contextWindow = cloudContextWindow(for: session)
                                Text("•")
                                    .foregroundColor(.secondary)
                                HStack(spacing: 3) {
                                    Image(systemName: "number.circle")
                                        .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .semibold))
                                        .foregroundColor(.secondary)
                                    if let window = contextWindow,
                                       let percent = CloudContextPolicy.usagePercent(
                                           promptTokens: cumulativeTokens,
                                           window: window
                                       ) {
                                        // A cloud conversation can see its own
                                        // ceiling: the window is a recorded
                                        // registration fact, and the percentage
                                        // is what turns "12,345 tokens" into
                                        // "how much room is left".
                                        Text("\(cumulativeTokens.formatted()) · \(percent)% of \(window.formatted())")
                                            .font(.system(size: max(9, 11 * zoomManager.zoomScale), design: .monospaced))
                                            .foregroundColor(percent >= 80 ? .orange : .secondary)
                                    } else {
                                        Text(cumulativeTokens.formatted())
                                            .font(.system(size: max(9, 11 * zoomManager.zoomScale), design: .monospaced))
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .help("Context this conversation is carrying: the prompt the model last ingested plus the tokens it generated. Local turns count only prompt tokens the pinned prefix had not already supplied; cloud turns report the last request's full prompt, because a stateless provider is re-sent the whole conversation and only the newest step describes its size.")
                            }
                            if let spend = session?.cumulativeCost, spend > 0 {
                                Text("•")
                                    .foregroundColor(.secondary)
                                HStack(spacing: 3) {
                                    Image(systemName: "dollarsign.circle")
                                        .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .semibold))
                                        .foregroundColor(.secondary)
                                    Text(CloudPricingStore.formatted(spend))
                                        .font(.system(size: max(9, 11 * zoomManager.zoomScale), design: .monospaced))
                                        .foregroundColor(.secondary)
                                }
                                .help("Reported spend for this conversation, summed from provider usage and the price list in Settings → Cloud Providers")
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(8)
                    }
                    
                    if isGenerating {
                        Button(action: onStopGeneration) {
                            HStack(spacing: 4) {
                                Image(systemName: "stop.fill")
                                    .font(.system(size: max(8, 10 * zoomManager.zoomScale)))
                                Text("Stop")
                                    .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                                    .fontWeight(.semibold)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color.red.opacity(0.12))
                            .foregroundColor(.red)
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                    }

                    // Background Subagents & Task Manager Toggle Button
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            subagentManager.isDrawerOpen.toggle()
                        }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "square.2.layers.3d")
                                .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .medium))
                            Text("Tasks")
                                .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .medium))
                            if subagentManager.activeSubagentsCount(forSession: session?.id) > 0 {
                                Text("\(subagentManager.activeSubagentsCount(forSession: session?.id))")
                                    .font(.system(size: max(8, 9 * zoomManager.zoomScale), weight: .bold))
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .clipShape(Capsule())
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(subagentManager.isDrawerOpen ? Color.purple.opacity(0.15) : Color.secondary.opacity(0.08))
                        .foregroundColor(subagentManager.isDrawerOpen ? .purple : .secondary)
                        .cornerRadius(6)
                    }
                    .buttonStyle(.plain)
                    .help("Toggle Background Task Manager Drawer (⌘D)")
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 12)
                .background(Color(NSColor.windowBackgroundColor))
                .onChange(of: session?.id) { _ in
                    isRenamingSession = false
                }
                
                Divider()

            // Main Message Content or Empty State (Centered & Constrained to Max Width for Readability)
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 0) {
                        if let session = session, !session.messages.isEmpty {
                            LazyVStack(spacing: 24) {
                                ForEach(session.messages) { message in
                                    Group {
                                        if message.isCompactionSummary == true {
                                            CompactionSummaryCard(
                                                message: message,
                                                isExpanded: Binding(
                                                    get: { isCompactionSummaryExpanded[message.id] ?? true },
                                                    set: { isCompactionSummaryExpanded[message.id] = $0 }
                                                )
                                            )
                                        } else if message.role == .assistant,
                                                  PlanMarkers.anyMarkerMentioned(in: message.content),
                                                  PlanMarkers.startPresent(in: message.content) {
                                            PlanCard(
                                                content: PlanMarkers.strip(message.content),
                                                thinkingContent: message.thinkingContent,
                                                isReady: PlanMarkers.readyPresent(in: message.content),
                                                isStreaming: isGenerating && message.id == session.messages.last?.id,
                                                isReasoningExpanded: Binding(
                                                    get: { isReasoningExpanded[message.id] ?? reasoningExpandedByDefault },
                                                    set: { isReasoningExpanded[message.id] = $0 }
                                                )
                                            )
                                        } else {
                                            ChatMessageView(
                                                message: message,
                                                isGenerating: isGenerating && message.id == session.messages.last?.id,
                                                isStreamingOffDisk: isStreamingOffDisk,
                                                isCompactingConversation: message.id == compactionProgressMessageId,
                                                isGemmaDialect: isGemmaDialect,
                                                isExpanded: Binding(
                                                    get: { isReasoningExpanded[message.id] ?? reasoningExpandedByDefault },
                                                    set: { isReasoningExpanded[message.id] = $0 }
                                                )
                                            )
                                        }
                                    }
                                    .id(message.id)
                                }
                            }
                            .padding(.top, 24)
                            .padding(.bottom, 24)
                        } else {
                            EmptyWelcomeView(
                                modelName: modelName,
                                isStreamingOffDisk: isStreamingOffDisk
                            )
                            .padding(.top, 48)
                            .padding(.bottom, 32)
                        }
                        Color.clear
                            .frame(height: 1)
                            .id(Self.scrollBottomAnchorID)
                    }
                    .frame(maxWidth: max(600, 800 * zoomManager.zoomScale))
                    .padding(.horizontal, 24)
                    .frame(maxWidth: .infinity)
                    // The floating layer covers the bottom of this view, so the
                    // content reserves its height: an opaque queue card must never
                    // hold the newest message's last lines out of reach.
                    .padding(.bottom, floatingLayerHeight)
                    // The standing history warning covers the top the same way,
                    // and a conversation scrolled back to its first message must
                    // still be reachable under it.
                    .padding(.top, historyWarningHeight)
                }
                .onChange(of: session?.id) { _ in
                    isPinnedToBottom = true
                    guard !landOnCompactedSummary(proxy) else { return }
                    followBottom(proxy)
                }
                .onChange(of: session?.messages.count) { _ in
                    guard !landOnCompactedSummary(proxy) else { return }
                    guard isPinnedToBottom else { return }
                    followBottom(proxy, animated: true)
                }
                .onChange(of: streamedContentBucket) { _ in
                    guard isPinnedToBottom, isGenerating || isCompactingConversation else { return }
                    followBottom(proxy)
                }
                .onChange(of: isGenerating) { generating in
                    if generating {
                        // A new turn is starting (user sent or a queued prompt fired):
                        // re-pin and jump to the newest message.
                        isPinnedToBottom = true
                        followBottom(proxy, animated: true)
                    } else if isPinnedToBottom {
                        followBottom(proxy, animated: true)
                    }
                }
                .onScrollGeometryChange(for: ScrollSnapshot.self) { geometry in
                    ScrollSnapshot(
                        offset: geometry.contentOffset.y,
                        contentHeight: geometry.contentSize.height,
                        containerHeight: geometry.containerSize.height
                    )
                } action: { old, new in
                    if new.contentHeight != old.contentHeight && new.offset == old.offset {
                        // Content grew under us (streaming output) — not a user
                        // scroll. Keep the current pin state; the bucket handler
                        // will follow if pinned. Otherwise growing content would
                        // read as "scrolled away from bottom" and unpin itself.
                        return
                    }
                    // Only write on a real change: this action runs on every
                    // geometry update, and an unconditional write invalidated
                    // the view several times per frame while a turn streamed.
                    let pinned = new.distanceFromBottom <= 48
                    if pinned != isPinnedToBottom {
                        isPinnedToBottom = pinned
                    }
                }
                // Everything the chat column says without moving anything: the
                // queued-prompt card and the composer's notices. Overlay content
                // cannot change this view's size, and sitting inside the
                // transcript's own bounds means nothing overflows or is clipped.
                .overlay(alignment: .bottom) {
                    chatFloatingLayer
                }
                // The standing history warning sits under the header rather than
                // over it: it lasts as long as the store is broken, so covering
                // Stop, the token counts or Tasks would hide them all session.
                .overlay(alignment: .top) {
                    if isHistoryUnavailable {
                        historyWarning
                            .transition(.opacity)
                    }
                }
                .animation(.easeInOut(duration: 0.2), value: isHistoryUnavailable)
            }

            // Floating Bottom Composer (Antigravity Centered Constrained Width)
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 6) {
                    if !pendingImages.isEmpty {
                        attachmentChipsRow()
                    }
                    // Multi-line Expanding Input
                    MacTextEditor(
                        text: $promptText,
                        placeholder: isGenerating
                            ? "Ask a follow-up to queue (↵) or send now (⌘↵)..."
                            : (isPlanMode
                               ? "Describe what you want built — DynaMoE will plan it first..."
                               : "Ask DynaMoE anything... (Enter to send, Shift+Enter for newline)"),
                        zoomScale: zoomManager.zoomScale,
                        onCommit: {
                            let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty || !pendingImages.isEmpty else {
                                // Return in an empty composer brings a dismissed
                                // plan handoff back, the way Crush reopens it.
                                dismissedPlanMessageId = nil
                                return
                            }
                            let accepted: Bool
                            if isGenerating {
                                accepted = onQueuePrompt?(trimmed, pendingImages) ?? onSendMessage(trimmed, pendingImages)
                            } else {
                                accepted = onSendMessage(trimmed, pendingImages)
                            }
                            if accepted {
                                promptText = ""
                                clearAttachments()
                            }
                        },
                        onCommitImmediate: {
                            submitImmediateDraft()
                        },
                        onPasteImage: { handlePasteboardImage() }
                    )
                    .frame(minHeight: composerMinHeight, maxHeight: max(100, 140 * zoomManager.zoomScale))
                    .padding(.horizontal, 4)
                    .padding(.top, 1)
                    
                    // Bottom Controls Bar inside the floating card
                    HStack(spacing: 8) {
                        // Attachment menu
                        Menu {
                            // Disabled for a model without a tower: the refusal
                            // is the fallback for a drop, not something to walk
                            // the user into from a menu. A disabled row cannot be
                            // activated, so the reason rides on it as a tooltip.
                            Button(action: presentAttachmentPicker) {
                                Label("Attach Image…", systemImage: "photo.badge.plus")
                            }
                            .disabled(!supportsVision)
                            .help(attachImageHelp)
                            Button(action: { _ = handlePasteboardImage() }) {
                                Label("Attach from Clipboard", systemImage: "doc.on.clipboard")
                            }
                            .disabled(!supportsVision)
                            .help(attachImageHelp)
                            if !pendingImages.isEmpty {
                                Divider()
                                Button(role: .destructive, action: clearAttachments) {
                                    Label("Remove All Attachments", systemImage: "trash")
                                }
                            }
                        } label: {
                            Image(systemName: pendingImages.isEmpty ? "plus" : "plus.circle.fill")
                                .font(.system(size: max(10, 12 * zoomManager.zoomScale), weight: .medium))
                                .foregroundColor(supportsVision ? .secondary : .secondary.opacity(0.5))
                                .frame(width: max(20, 24 * zoomManager.zoomScale), height: max(20, 24 * zoomManager.zoomScale))
                                .background(Color.secondary.opacity(0.08))
                                .clipShape(Circle())
                        }
                        .menuStyle(.borderlessButton)
                        .menuIndicator(.hidden)
                        .fixedSize()
                        .help(attachImageHelp)

                        // Antigravity-Style Model Selector Menu
                        Menu {
                            let systemModels = localModelManager.discoveredModels.filter { $0.isAppleFoundationModel }
                            let cloudModels = localModelManager.discoveredModels.filter { $0.isCloudModel }
                            let weightsModels = localModelManager.discoveredModels.filter {
                                !$0.isAppleFoundationModel && !$0.isCloudModel
                            }
                            if !systemModels.isEmpty {
                                Section("System Models") {
                                    ForEach(systemModels) { dm in
                                        discoveredModelMenuButton(dm)
                                    }
                                }
                            }
                            if !cloudModels.isEmpty {
                                Section("Cloud Models") {
                                    ForEach(cloudModels) { dm in
                                        discoveredModelMenuButton(dm)
                                    }
                                }
                            }
                            if weightsModels.isEmpty && systemModels.isEmpty && cloudModels.isEmpty {
                                Text("No models found in ~/.cache/huggingface/hub")
                            }
                            if !weightsModels.isEmpty {
                                Section("Discovered Hugging Face Models") {
                                    ForEach(weightsModels) { dm in
                                        discoveredModelMenuButton(dm)
                                    }
                                }
                            }

                            Divider()

                            if let onOpenSettings = onOpenSettings {
                                Button(action: onOpenSettings) {
                                    Label("Manage Models in Settings...", systemImage: "gearshape")
                                }
                            }
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: modelIconName(for: session?.selectedModelName ?? modelName, isCloud: isCloudModelActive))
                                    .font(.system(size: max(8.5, 10 * zoomManager.zoomScale)))
                                    .foregroundColor(.purple)
                                Text(session?.selectedModelName ?? modelName ?? "Select Model")
                                    .font(.system(size: max(9.5, 11.5 * zoomManager.zoomScale), weight: .medium))
                                    .foregroundColor(.primary)
                                    .lineLimit(1)
                                Image(systemName: "chevron.up.chevron.down")
                                    .font(.system(size: max(6.5, 7.5 * zoomManager.zoomScale), weight: .semibold))
                                    .foregroundColor(.secondary.opacity(0.7))
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4.5)
                            .background(Color.secondary.opacity(0.08))
                            .cornerRadius(10)
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()

                        let currentProfileModel = session?.selectedModelName ?? modelName
                        // Profile Selector Menu (Coder / Assistant)
                        Menu {
                            Section("Generation Profile") {
                                ForEach(ModelProfileType.allCases) { profile in
                                    Button(action: {
                                        onSelectProfile?(profile)
                                    }) {
                                        HStack {
                                            if activeProfile == profile {
                                                Image(systemName: "checkmark")
                                            }
                                            Label("\(profile.rawValue) (\(profile.profileDisplayName(for: currentProfileModel)))", systemImage: profile.icon)
                                        }
                                    }
                                }
                            }

                            Divider()

                            if let onOpenSettings = onOpenSettings {
                                Button(action: onOpenSettings) {
                                    Label("Configure Profiles in Settings...", systemImage: "slider.horizontal.3")
                                }
                            }
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: activeProfile.icon)
                                    .font(.system(size: max(8.5, 10 * zoomManager.zoomScale)))
                                    .foregroundColor(.purple)
                                Text(activeProfile.rawValue)
                                    .font(.system(size: max(9.5, 11.5 * zoomManager.zoomScale), weight: .medium))
                                    .foregroundColor(.primary)
                                Image(systemName: "chevron.up.chevron.down")
                                    .font(.system(size: max(6.5, 7.5 * zoomManager.zoomScale), weight: .semibold))
                                    .foregroundColor(.secondary.opacity(0.7))
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4.5)
                            .background(Color.purple.opacity(0.12))
                            .cornerRadius(10)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.purple.opacity(0.25), lineWidth: 1)
                            )
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()
                        .help("Active generation profile: \(activeProfile.profileDisplayName(for: currentProfileModel)) (\(activeProfile.description(for: currentProfileModel)))")

                        // Thinking dropdown beside Model Selector (shown only if the loaded
                        // model supports thinking). Level-capable models (Spark X2.6's
                        // `reasoning_effort`) get High/Low/Off instead of plain on/off.
                        // Cloud reasoning follows the provider's own default:
                        // the OpenAI-compatible surface has no portable
                        // on/off switch, so the control is hidden rather than
                        // pretending to change anything.
                        if supportsThinking && !isCloudModelActive && supportsThinkingLevels {
                            let activeEffort: ThinkingEffort? = isThinkingEnabled ? (thinkingEffort ?? .high) : nil
                            Menu {
                                Button(action: {
                                    onSetThinking?(true, .high)
                                }) {
                                    HStack {
                                        if activeEffort == .high {
                                            Image(systemName: "checkmark")
                                        }
                                        Label("Thinking High (Full Reasoning)", systemImage: "brain.head.profile")
                                    }
                                }

                                Button(action: {
                                    onSetThinking?(true, .low)
                                }) {
                                    HStack {
                                        if activeEffort == .low {
                                            Image(systemName: "checkmark")
                                        }
                                        Label("Thinking Low (Brief Reasoning)", systemImage: "brain")
                                    }
                                }

                                Button(action: {
                                    onSetThinking?(false, nil)
                                }) {
                                    HStack {
                                        if activeEffort == nil {
                                            Image(systemName: "checkmark")
                                        }
                                        Label("Thinking Off (Direct Response)", systemImage: "bolt.slash")
                                    }
                                }
                            } label: {
                                thinkingMenuLabel(
                                    isThinkingEnabled
                                        ? "Thinking \(activeEffort?.rawValue.capitalized ?? "High")"
                                        : "Thinking Off",
                                    isOn: isThinkingEnabled
                                )
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()
                            .help(isThinkingEnabled ? "Graded thinking level: \(activeEffort?.rawValue ?? "high")" : "Direct response without thinking is enabled")
                            .transition(.opacity.combined(with: .scale))
                        } else if supportsThinking && !isCloudModelActive {
                            Menu {
                                Button(action: {
                                    onSetThinking?(true, nil)
                                }) {
                                    HStack {
                                        if isThinkingEnabled {
                                            Image(systemName: "checkmark")
                                        }
                                        Label("Thinking On (Reasoning Process)", systemImage: "brain.head.profile")
                                    }
                                }

                                Button(action: {
                                    onSetThinking?(false, nil)
                                }) {
                                    HStack {
                                        if !isThinkingEnabled {
                                            Image(systemName: "checkmark")
                                        }
                                        Label("Thinking Off (Direct Response)", systemImage: "bolt.slash")
                                    }
                                }
                            } label: {
                                thinkingMenuLabel(
                                    isThinkingEnabled ? "Thinking On" : "Thinking Off",
                                    isOn: isThinkingEnabled
                                )
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()
                            .help(isThinkingEnabled ? "Thinking / Reasoning is enabled for this model" : "Direct response without thinking is enabled")
                            .transition(.opacity.combined(with: .scale))
                        }

                        // Agent Tools On/Off Dropdown
                        Menu {
                            Button(action: {
                                onToggleAgentTools?(true)
                            }) {
                                HStack {
                                    if isAgentToolsEnabled {
                                        Image(systemName: "checkmark")
                                    }
                                    Label("Agent Tools On (Shell & File Execution)", systemImage: "wrench.and.screwdriver.fill")
                                }
                            }

                            Button(action: {
                                onToggleAgentTools?(false)
                            }) {
                                HStack {
                                    if !isAgentToolsEnabled {
                                        Image(systemName: "checkmark")
                                    }
                                    Label("Agent Tools Off", systemImage: "wrench.and.screwdriver")
                                }
                            }
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: isAgentToolsEnabled ? "wrench.and.screwdriver.fill" : "wrench.and.screwdriver")
                                    .font(.system(size: max(8.5, 10 * zoomManager.zoomScale)))
                                    .foregroundColor(isAgentToolsEnabled ? .indigo : .secondary)
                                Text(isAgentToolsEnabled ? "Tools On" : "Tools Off")
                                    .font(.system(size: max(9.5, 11.5 * zoomManager.zoomScale), weight: .medium))
                                    .foregroundColor(isAgentToolsEnabled ? .primary : .secondary)
                                Image(systemName: "chevron.up.chevron.down")
                                    .font(.system(size: max(6.5, 7.5 * zoomManager.zoomScale), weight: .semibold))
                                    .foregroundColor(.secondary.opacity(0.7))
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4.5)
                            .background(isAgentToolsEnabled ? Color.indigo.opacity(0.12) : Color.secondary.opacity(0.08))
                            .cornerRadius(10)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(isAgentToolsEnabled ? Color.indigo.opacity(0.3) : Color.clear, lineWidth: 1)
                            )
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()
                        .disabled(isPlainChatOnlyBackend)
                        .help(isAppleFoundationModelActive
                              ? "Agent tools require a model with function calling — Apple Foundation Model runs in plain chat mode"
                              : (isCloudModelActive
                                 ? (isAgentToolsEnabled
                                    ? "Agent mode is enabled: the provider's native function calling runs shell, file, and search tools"
                                    : "Agent mode is disabled")
                                 : (isAgentToolsEnabled ? "Agent mode is enabled: Model can run shell commands, inspect, and edit files" : "Agent mode is disabled")))
                        .transition(.opacity.combined(with: .scale))

                        // Turbo Mode (Auto-Approve) Toggle when tools are active
                        if isAgentToolsEnabled && !isPlainChatOnlyBackend {
                            Button(action: {
                                isTurboModeEnabled.toggle()
                            }) {
                                HStack(spacing: 4) {
                                    Image(systemName: isTurboModeEnabled ? "bolt.fill" : "shield.lefthalf.filled")
                                        .font(.system(size: max(8.5, 10 * zoomManager.zoomScale)))
                                        .foregroundColor(isTurboModeEnabled ? .orange : .blue)
                                    Text(isTurboModeEnabled ? "Turbo On" : "Safe Mode")
                                        .font(.system(size: max(9.5, 11.5 * zoomManager.zoomScale), weight: .medium))
                                        .foregroundColor(isTurboModeEnabled ? .orange : .primary)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4.5)
                                .background(isTurboModeEnabled ? Color.orange.opacity(0.12) : Color.blue.opacity(0.08))
                                .cornerRadius(10)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(isTurboModeEnabled ? Color.orange.opacity(0.35) : Color.blue.opacity(0.25), lineWidth: 1)
                                )
                            }
                            .buttonStyle(.plain)
                            .help(isTurboModeEnabled ? "Turbo Mode is ON: Model executes file writes and shell commands automatically without individual approval." : "Safe Mode: Requires confirmation before executing state-changing actions.")
                            .transition(.opacity.combined(with: .scale))
                        }

                        // Plan Mode pill. Next to Turbo because both change what a
                        // turn is allowed to DO, but unlike Turbo (app-wide) this is
                        // per conversation, like the Tools toggle beside it. Shown
                        // even while tools are OFF: entering plan mode arms them,
                        // and hiding the only way in would hide that.
                        //
                        // Labelled by what a click DOES when plan mode is off and
                        // by what is ACTIVE when it is on, because "Code Mode" on a
                        // button read as an offer to switch back to a mode the
                        // conversation was already in.
                        if !isPlainChatOnlyBackend {
                            Button(action: {
                                onSetPlanMode?(!isPlanMode)
                            }) {
                                HStack(spacing: 4) {
                                    Image(systemName: isPlanMode ? "checklist" : "chevron.left.forwardslash.chevron.right")
                                        .font(.system(size: max(8.5, 10 * zoomManager.zoomScale)))
                                        .foregroundColor(isPlanMode ? .purple : .secondary)
                                    Text(isPlanMode ? "Plan Mode" : "Plan")
                                        .font(.system(size: max(9.5, 11.5 * zoomManager.zoomScale), weight: .medium))
                                        .foregroundColor(isPlanMode ? .purple : .secondary)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4.5)
                                .background(isPlanMode ? Color.purple.opacity(0.12) : Color.secondary.opacity(0.08))
                                .cornerRadius(10)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(isPlanMode ? Color.purple.opacity(0.4) : Color.clear, lineWidth: 1)
                                )
                            }
                            .buttonStyle(.plain)
                            .keyboardShortcut("p", modifiers: [.command, .shift])
                            .disabled(isModeChangeLocked)
                            .opacity(isModeChangeLocked ? 0.45 : 1)
                            .accessibilityLabel(isPlanMode ? "Plan Mode is on" : "Enter Plan Mode")
                            .help(planPillHelp)
                            .transition(.opacity.combined(with: .scale))
                        }

                        // Real-time Prompt Token Counter (to the right of Thinking toggle or Model selector)
                        if promptTokenCount > 0 {
                            HStack(spacing: 3.5) {
                                Image(systemName: "number.circle.fill")
                                    .font(.system(size: max(8, 9.5 * zoomManager.zoomScale), weight: .semibold))
                                    .foregroundColor(.secondary)
                                Text("\(promptTokenCount.formatted()) \(promptTokenCount == 1 ? "token" : "tokens")")
                                    .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .medium, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal, 7)
                            .padding(.vertical, 4.5)
                            .background(Color.secondary.opacity(0.06))
                            .cornerRadius(8)
                            .help("Real-time token count of current prompt input")
                            .transition(.opacity.combined(with: .scale(scale: 0.95)))
                        }

                        Spacer()

                        // Mic Button
                        Button(action: {}) {
                            Image(systemName: "mic")
                                .font(.system(size: max(10, 13 * zoomManager.zoomScale)))
                                .foregroundColor(.secondary)
                                .frame(width: max(20, 24 * zoomManager.zoomScale), height: max(20, 24 * zoomManager.zoomScale))
                        }
                        .buttonStyle(.plain)

                        // Send / Stop / Queue Buttons
                        if isGenerating {
                            HStack(spacing: 8) {
                                if !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pendingImages.isEmpty {
                                    Button(action: {
                                        let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
                                        guard !trimmed.isEmpty || !pendingImages.isEmpty else { return }
                                        let accepted: Bool
                                        if let queueAction = onQueuePrompt {
                                            accepted = queueAction(trimmed, pendingImages)
                                        } else {
                                            accepted = onSendMessage(trimmed, pendingImages)
                                        }
                                        if accepted {
                                            promptText = ""
                                            clearAttachments()
                                        }
                                    }) {
                                        Image(systemName: "tray.and.arrow.down.fill")
                                            .font(.system(size: max(11, 14 * zoomManager.zoomScale), weight: .semibold))
                                            .foregroundColor(.purple)
                                            .frame(width: max(20, 26 * zoomManager.zoomScale), height: max(20, 26 * zoomManager.zoomScale))
                                            .background(Color.purple.opacity(0.12))
                                            .clipShape(Circle())
                                    }
                                    .buttonStyle(.plain)
                                    .help("Queue message for next turn (↵)")

                                    Button(action: {
                                        submitImmediateDraft()
                                    }) {
                                        Image(systemName: "bolt.circle.fill")
                                            .font(.system(size: max(20, 26 * zoomManager.zoomScale)))
                                            .foregroundColor(.orange)
                                    }
                                    .buttonStyle(.plain)
                                    .disabled(pendingImmediateSend != nil)
                                    .help("Interrupt and send immediately (⌘↵)")
                                }

                                Button(action: onStopGeneration) {
                                    Image(systemName: "stop.circle.fill")
                                        .font(.system(size: max(20, 26 * zoomManager.zoomScale)))
                                        .foregroundColor(.red)
                                }
                                .buttonStyle(.plain)
                                .help("Stop generating")
                            }
                        } else {
                            // Slash commands are backend-independent app actions
                            // (/compact runs on the system model), so the button
                            // must dispatch them even with no weights model
                            // loaded instead of dying silently like a chat send.
                            let draftIsCommand = ChatCommand.parse(promptText) != nil
                            Button(action: {
                                let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
                                if (!trimmed.isEmpty || !pendingImages.isEmpty) && (isModelLoaded || ChatCommand.parse(trimmed) != nil) {
                                    if onSendMessage(trimmed, pendingImages) {
                                        promptText = ""
                                        clearAttachments()
                                    }
                                }
                            }) {
                                Image(systemName: "arrow.right.circle.fill")
                                    .font(.system(size: max(20, 26 * zoomManager.zoomScale)))
                                    .foregroundColor(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (!isModelLoaded && !draftIsCommand) ? .secondary.opacity(0.3) : .purple)
                            }
                            .buttonStyle(.plain)
                            .disabled((promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && pendingImages.isEmpty) || (!isModelLoaded && !draftIsCommand))
                            .help(isModelLoaded || draftIsCommand ? "Send Message" : "Load a model first")
                        }
                    }
                }
                .padding(12)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(18)
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(
                            isAttachmentDropTargeted && supportsVision
                                ? Color.purple.opacity(0.65)
                                : Color.primary.opacity(0.09),
                            lineWidth: isAttachmentDropTargeted && supportsVision ? 2 : 1
                        )
                )
                .onDrop(
                    of: [UTType.fileURL, UTType.image],
                    isTargeted: $isAttachmentDropTargeted,
                    perform: handleAttachmentDrop
                )
                .shadow(color: Color.black.opacity(0.06), radius: 10, x: 0, y: 4)
                .frame(maxWidth: max(600, 800 * zoomManager.zoomScale))
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
                .frame(maxWidth: .infinity)
            }
            .background(Color(NSColor.windowBackgroundColor))
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if subagentManager.isDrawerOpen {
                Divider()
                SubagentDrawerView(onClose: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        subagentManager.isDrawerOpen = false
                    }
                })
                .transition(.move(edge: .trailing))
            }
        }
        .onAppear {
            updateTokenCount(for: promptText)
        }
        .onChange(of: promptText) { newText in
            draftRevision &+= 1
            updateTokenCount(for: newText)
            detectTypedImagePaths(in: newText)
        }
        .onChange(of: tokenizer != nil) { _ in
            updateTokenCount(for: promptText)
        }
        // A refusal is about the model that was selected when it was raised, so a
        // switch has to retract it. Staging survives a switch, so a model that
        // cannot see what is already staged has to say so.
        .onChange(of: supportsVision) { canSeeImages in
            if canSeeImages {
                // The refusal is retracted; a decode that is still running is not
                // a refusal, and clearing it would drop "Preparing image…" while
                // the image it names is still being prepared.
                if composerNotice?.kind != .progress { clearNotice() }
            } else if !pendingImages.isEmpty {
                showNotice(
                    ImageAttachmentMessages.cannotSeeImagesSend(modelName: modelName),
                    kind: .warning
                )
            }
        }
        .onPreferenceChange(FloatingLayerHeight.self) { height in
            floatingLayerHeight = height
        }
        .onPreferenceChange(HistoryWarningHeight.self) { height in
            historyWarningHeight = height
        }
    }

    /// Submits the current draft through the async immediate-send path
    /// (Cmd-Return / bolt button). Rejects a second submission while one is in
    /// flight: the send suspends until any generation teardown finishes, so
    /// without this guard the same draft would be queued twice.
    private func submitImmediateDraft() {
        let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (!trimmed.isEmpty || !pendingImages.isEmpty), pendingImmediateSend == nil else { return }
        let imagesToSend = pendingImages
        guard let immediate = onSendImmediate else {
            if onSendMessage(trimmed, imagesToSend) {
                promptText = ""
                clearAttachments()
            }
            return
        }
        // Capture the submitted draft so a newer edit made while the send is in
        // flight survives. Compare the revision, not just the text: the user could
        // edit away and back to the same string, and a text match would then clear
        // a draft they intentionally re-typed.
        let submitted = promptText
        let submittedRevision = draftRevision
        pendingImmediateSend = submitted
        Task {
            if await immediate(trimmed, imagesToSend), draftRevision == submittedRevision {
                promptText = ""
                clearAttachments()
            }
            if pendingImmediateSend == submitted {
                pendingImmediateSend = nil
            }
        }
    }

    // MARK: - Floating Chat Layer

    /// Height the floating layer covers at the bottom of the transcript, reported
    /// up so the scroll content can reserve exactly that much.
    private struct FloatingLayerHeight: PreferenceKey {
        static let defaultValue: CGFloat = 0
        static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
            value = max(value, nextValue())
        }
    }

    /// Everything the chat column says without moving anything. The queued-prompt
    /// card and the notices are drawn over the bottom of the transcript instead of
    /// inset into the composer's rows: the composer's height is what sizes the
    /// transcript above it, so a queue entry or a refused drop must never reach it.
    /// Overlay content cannot change its host's size, and because this sits inside
    /// the transcript's own bounds nothing overflows or is clipped.
    @ViewBuilder
    private var chatFloatingLayer: some View {
        VStack(spacing: 6) {
            if let plan = readyPlanMessage {
                planHandoffBar(plan)
                    .transition(.opacity)
            }
            if let queued = session?.queuedPrompts, !queued.isEmpty {
                queuedPromptsCard(queued)
                    .transition(.opacity)
            }
            ForEach(layerNotices) { notice in
                // Closest to the composer is what the notice is about. None of
                // them takes a click: the transcript under it is live.
                StatusNoticeBubble(notice: notice, zoomScale: zoomManager.zoomScale)
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }
        }
        .frame(maxWidth: max(600, 800 * zoomManager.zoomScale))
        .padding(.horizontal, 24)
        .padding(.bottom, 10)
        .background(
            GeometryReader { proxy in
                Color.clear.preference(key: FloatingLayerHeight.self, value: proxy.size.height)
            }
        )
        .animation(.easeInOut(duration: 0.15), value: session?.queuedPrompts.count ?? 0)
        .animation(.easeInOut(duration: 0.15), value: readyPlanMessage?.id)
        .animation(.easeInOut(duration: 0.15), value: layerNotices)
    }

    /// The plan handoff: Crush's "ready to start coding?" prompt, placed where the
    /// user is already looking when a turn ends. It asks the one question plan
    /// mode exists to ask — approve (optionally with Turbo) or revise — and it
    /// deliberately offers no way to start coding without leaving plan mode, so
    /// approving is what re-arms the write tools.
    @ViewBuilder
    private func planHandoffBar(_ plan: ChatMessage) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "checklist")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.purple)
                Text("Plan ready")
                    .font(.system(size: max(10, 12 * zoomManager.zoomScale), weight: .bold))
                Spacer(minLength: 0)
                Button(action: {
                    dismissedPlanMessageId = plan.id
                }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(.secondary)
                        .padding(4)
                        .background(Color.secondary.opacity(0.1))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Dismiss the plan prompt")
                .help("Dismiss this prompt. Press Return in an empty composer to bring it back.")
            }

            Text("Approve to switch this conversation to Code Mode and start implementing.")
                .font(.system(size: max(9.5, 11 * zoomManager.zoomScale)))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                Button(action: { onApprovePlan?(false) }) {
                    HStack(spacing: 5) {
                        Image(systemName: "play.fill")
                            .font(.system(size: 9))
                        Text("Start coding")
                            .font(.system(size: max(9.5, 11 * zoomManager.zoomScale), weight: .semibold))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.purple.opacity(0.9))
                    .foregroundColor(.white)
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .disabled(isModeChangeLocked)
                .opacity(isModeChangeLocked ? 0.45 : 1)
                .help(isModeChangeLocked
                    ? "Wait for the running turn to finish before starting the implementation"
                    : "Switch to Code Mode and start implementing the plan")

                Button(action: { onApprovePlan?(true) }) {
                    HStack(spacing: 5) {
                        Image(systemName: "bolt.fill")
                            .font(.system(size: 9))
                        Text("Code with Turbo")
                            .font(.system(size: max(9.5, 11 * zoomManager.zoomScale), weight: .medium))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.orange.opacity(0.14))
                    .foregroundColor(.orange)
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .disabled(isModeChangeLocked)
                .opacity(isModeChangeLocked ? 0.45 : 1)
                .help(isModeChangeLocked
                    ? "Wait for the running turn to finish before starting the implementation"
                    : "Enable Turbo Mode, switch to Code Mode, and start implementing without per-action approval")

                Button(action: {
                    dismissedPlanMessageId = plan.id
                    if !promptText.isEmpty { promptText += "\n" }
                    promptText += "Revise the plan: "
                    isInputFocused = true
                }) {
                    Text("Revise plan")
                        .font(.system(size: max(9.5, 11 * zoomManager.zoomScale), weight: .medium))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.secondary.opacity(0.09))
                        .foregroundColor(.primary)
                        .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .help("Stay in plan mode and send feedback in the composer")

                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.purple.opacity(0.35), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.14), radius: 10, x: 0, y: 4)
    }

    /// The notices this layer shows, in the order they stack. The window and the
    /// composer can hold the same sentence at once, and a fact repeated on two
    /// pills reads as two problems, so `stacked` keeps one copy.
    private var layerNotices: [StatusNotice] {
        StatusNoticePolicy.stacked(window: windowNotice, composer: composerNotice)
    }

    /// Height the standing history warning covers at the top of the transcript,
    /// reported up so the scroll content can reserve exactly that much.
    private struct HistoryWarningHeight: PreferenceKey {
        static let defaultValue: CGFloat = 0
        static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
            value = max(value, nextValue())
        }
    }

    /// Shown when the store reports persistence as unavailable. The specific
    /// cause varies (unopenable folder, a failed read or write, encoding failure,
    /// failed cleanup), and a listing failure means existing conversations may not
    /// have loaded, so the wording covers both reading and saving rather than
    /// blaming the folder or promising the failure only affects future saves.
    ///
    /// It floats over the transcript rather than insetting it: a banner that
    /// pushes the conversation down moves the very thing it is warning about. The
    /// solid background is what keeps it readable over the messages beneath it.
    @ViewBuilder
    private var historyWarning: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
                .font(.system(size: 13))
            VStack(alignment: .leading, spacing: 1) {
                Text("Chat history is unavailable")
                    .font(.system(size: 12, weight: .semibold))
                Text("DynaMoE couldn't read or save chat history. Existing conversations may not appear, and recent ones may not persist between launches.")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.orange.opacity(0.45), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.14), radius: 10, x: 0, y: 4)
        .padding(.horizontal, 12)
        .padding(.top, 6)
        .allowsHitTesting(false)
        .background(
            GeometryReader { proxy in
                Color.clear.preference(key: HistoryWarningHeight.self, value: proxy.size.height)
            }
        )
    }

    /// The queued-prompt rows. Capped and internally scrollable because the card
    /// floats: a long queue must not swallow the conversation it queues into.
    private func queuedPromptsCard(_ queued: [QueuedPrompt]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "tray.full.fill")
                    .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .semibold))
                    .foregroundColor(.purple)
                Text("QUEUED")
                    .font(.system(size: max(9, 10.5 * zoomManager.zoomScale), weight: .bold))
                    .foregroundColor(.purple)
                Text("(\(queued.count))")
                    .font(.system(size: max(9, 10.5 * zoomManager.zoomScale), weight: .semibold, design: .monospaced))
                    .foregroundColor(.secondary)

                Spacer()

                Text(isGenerating ? "Will run automatically next turn" : "Waiting to send")
                    .font(.system(size: max(8.5, 10 * zoomManager.zoomScale), weight: .medium))
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 4)

            ScrollView(.vertical, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(queued) { item in
                        HStack(spacing: 8) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.text)
                                    .font(.system(size: max(10, 12 * zoomManager.zoomScale)))
                                    .foregroundColor(.primary)
                                    .lineLimit(2)
                                    .truncationMode(.tail)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                if let count = item.attachments?.count, count > 0 {
                                    Label(
                                        "\(count) image\(count == 1 ? "" : "s")",
                                        systemImage: "photo"
                                    )
                                    .font(.system(size: max(8.5, 9.5 * zoomManager.zoomScale), weight: .medium))
                                    .foregroundColor(.secondary)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)

                            HStack(spacing: 6) {
                                Button(action: {
                                    let textToSend = item.text
                                    // The row stays visible while the send waits
                                    // out a generation teardown, so a second
                                    // activation must be ignored until this one
                                    // finishes or the prompt would be sent twice.
                                    guard !sendingQueuedIds.contains(item.id) else { return }
                                    sendingQueuedIds.insert(item.id)
                                    let queuedRefs = item.attachments ?? []
                                    if let immediate = onSendImmediate {
                                        // Only drop the queued prompt once the
                                        // send actually lands, so a rejected
                                        // send (no model, still loading) does
                                        // not lose it.
                                        Task {
                                            let queuedImages = await Task.detached(priority: .userInitiated) {
                                                ImageAttachmentStore.shared.preparedImages(for: queuedRefs)
                                            }.value
                                            if await immediate(textToSend, queuedImages) {
                                                onRemoveQueuedPrompt?(item.id)
                                            }
                                            sendingQueuedIds.remove(item.id)
                                        }
                                    } else {
                                        Task {
                                            let queuedImages = await Task.detached(priority: .userInitiated) {
                                                ImageAttachmentStore.shared.preparedImages(for: queuedRefs)
                                            }.value
                                            if onSendMessage(textToSend, queuedImages) {
                                                onRemoveQueuedPrompt?(item.id)
                                            }
                                            sendingQueuedIds.remove(item.id)
                                        }
                                    }
                                }) {
                                    HStack(spacing: 3) {
                                        Image(systemName: "bolt.fill")
                                            .font(.system(size: max(8.5, 9.5 * zoomManager.zoomScale)))
                                        Text("Send Now")
                                            .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .medium))
                                    }
                                    .padding(.horizontal, 7)
                                    .padding(.vertical, 3.5)
                                    .background(Color.orange.opacity(0.15))
                                    .foregroundColor(.orange)
                                    .cornerRadius(6)
                                }
                                .buttonStyle(.plain)
                                .disabled(sendingQueuedIds.contains(item.id))
                                .help("Send this message now (interrupts current turn)")

                                Button(action: {
                                    promptText = item.text
                                    clearNotice()
                                    // Read the bytes before dropping the queue
                                    // entry: removal now deletes the assets, so
                                    // pulling the prompt back must copy first.
                                    let queuedRefs = item.attachments ?? []
                                    Task {
                                        let images = await Task.detached(priority: .userInitiated) {
                                            ImageAttachmentStore.shared.preparedImages(for: queuedRefs)
                                        }.value
                                        pendingImages = images
                                        onRemoveQueuedPrompt?(item.id)
                                    }
                                }) {
                                    Image(systemName: "pencil")
                                        .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                                        .foregroundColor(.secondary)
                                        .frame(width: max(18, 22 * zoomManager.zoomScale), height: max(18, 22 * zoomManager.zoomScale))
                                        .background(Color.secondary.opacity(0.08))
                                        .clipShape(Circle())
                                }
                                .buttonStyle(.plain)
                                .disabled(sendingQueuedIds.contains(item.id))
                                .help("Edit prompt in composer")

                                Button(action: {
                                    onRemoveQueuedPrompt?(item.id)
                                }) {
                                    Image(systemName: "xmark")
                                        .font(.system(size: max(8, 10 * zoomManager.zoomScale), weight: .bold))
                                        .foregroundColor(.secondary)
                                        .frame(width: max(18, 22 * zoomManager.zoomScale), height: max(18, 22 * zoomManager.zoomScale))
                                        .background(Color.secondary.opacity(0.08))
                                        .clipShape(Circle())
                                }
                                .buttonStyle(.plain)
                                .disabled(sendingQueuedIds.contains(item.id))
                                .help("Remove from queue")
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Color.primary.opacity(0.04))
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                        )
                    }
                }
            }
            .frame(maxHeight: max(120, 200 * zoomManager.zoomScale))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(14)
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.purple.opacity(0.2), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.08), radius: 12, x: 0, y: 4)
    }

    // MARK: - Image Attachments

    /// Stages an image in the composer. Only a send persists it, so an abandoned
    /// draft leaves nothing on disk.
    private func stageAttachment(_ image: PreparedImage) {
        guard supportsVision else {
            showNotice(ImageAttachmentMessages.cannotSeeImagesAttach(modelName: modelName), kind: .warning)
            return
        }
        guard pendingImages.count < ImageAttachmentPreparer.maxImagesPerMessage else {
            showNotice(
                ImageAttachmentError.tooManyImages(
                    limit: ImageAttachmentPreparer.maxImagesPerMessage
                ).errorDescription ?? "Too many images.",
                kind: .warning
            )
            return
        }
        // The same picture dropped twice is one attachment, not two billed ones.
        guard !pendingImages.contains(where: { $0.sha256 == image.sha256 }) else { return }
        pendingImages.append(image)
        clearNotice()
    }

    private func attachFile(at url: URL) async {
        guard supportsVision else {
            showNotice(ImageAttachmentMessages.cannotSeeImagesAttach(modelName: modelName), kind: .warning)
            return
        }
        showNotice(StatusNoticePolicy.preparingText, kind: .progress)
        defer { clearProgressNotice() }
        do {
            let prepared = try await Task.detached(priority: .userInitiated) {
                try ImageAttachmentPreparer.prepare(fileURL: url)
            }.value
            stageAttachment(prepared)
        } catch {
            showNotice(error.localizedDescription, kind: .warning)
        }
    }

    private func attachImageData(_ data: Data, fileName: String) async {
        guard supportsVision else {
            showNotice(ImageAttachmentMessages.cannotSeeImagesAttach(modelName: modelName), kind: .warning)
            return
        }
        showNotice(StatusNoticePolicy.preparingText, kind: .progress)
        defer { clearProgressNotice() }
        do {
            let prepared = try await Task.detached(priority: .userInitiated) {
                try ImageAttachmentPreparer.prepare(data: data, fileName: fileName)
            }.value
            stageAttachment(prepared)
        } catch {
            showNotice(error.localizedDescription, kind: .warning)
        }
    }

    /// Intercepts an image paste so raw TIFF/PNG bytes never land in the text
    /// view. Returns true when the pasteboard held an image and was consumed.
    private func handlePasteboardImage() -> Bool {
        let pasteboard = NSPasteboard.general
        if let data = pasteboard.data(forType: .png) ?? pasteboard.data(forType: .tiff) {
            Task { await attachImageData(data, fileName: "Pasted image") }
            return true
        }
        let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: nil) as? [URL] ?? []
        if let url = urls.first(where: { ImageAttachmentPreparer.isAttachableImageFile(atPath: $0.path) }) {
            Task { await attachFile(at: url) }
            return true
        }
        return false
    }

    private func handleAttachmentDrop(_ providers: [NSItemProvider]) -> Bool {
        guard supportsVision else {
            showNotice(ImageAttachmentMessages.cannotSeeImagesAttach(modelName: modelName), kind: .warning)
            return true
        }
        var claimed = false
        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                claimed = true
                provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                    var url: URL?
                    if let data = item as? Data {
                        url = URL(dataRepresentation: data, relativeTo: nil)
                    } else if let itemURL = item as? URL {
                        url = itemURL
                    }
                    guard let url else { return }
                    Task { @MainActor in
                        await attachFile(at: url)
                    }
                }
            } else if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
                claimed = true
                provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                    guard let data else { return }
                    Task { @MainActor in
                        await attachImageData(data, fileName: "Dropped image")
                    }
                }
            }
        }
        return claimed
    }

    private func presentAttachmentPicker() {
        let panel = NSOpenPanel()
        panel.title = "Attach Images"
        panel.message = "Choose one or more images to send with this message."
        panel.allowedContentTypes = [.png, .jpeg, .heic, .tiff, .gif, .webP, .bmp]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK else { return }
        for url in panel.urls {
            Task { await attachFile(at: url) }
        }
    }

    /// Converts a typed absolute path into an attachment. Typing the path is the
    /// "provide a filepath" affordance: the app reads the file itself because the
    /// model can never open one (the agent's file tool refuses binary).
    private func detectTypedImagePaths(in text: String) {
        guard supportsVision else { return }
        let candidates = ImageAttachmentPreparer.imageFilePathCandidates(in: text)
        guard let path = candidates.first(where: { ImageAttachmentPreparer.isAttachableImageFile(atPath: $0) }) else {
            return
        }
        if let range = text.range(of: path) {
            promptText = text.replacingCharacters(in: range, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        Task { await attachFile(at: URL(fileURLWithPath: path)) }
    }

    private func clearAttachments() {
        pendingImages.removeAll()
        clearNotice()
    }

    // MARK: - Composer Notices

    /// Shows a transient composer message. It is drawn above the composer, never
    /// added to its rows, so it cannot resize the transcript. A warning clears
    /// itself; progress stays until the work it describes finishes.
    private func showNotice(_ text: String, kind: StatusNotice.Kind) {
        assert(
            !StatusNoticePolicy.statesContradictorySeverity(text: text, kind: kind),
            "\(text) leads with a glyph of another severity: drop the explicit kind, or the glyph"
        )
        noticeDismissTask?.cancel()
        noticeDismissTask = nil
        // A repeat of what is already on screen keeps its identity, so it does
        // not re-run the fade, but it does restart the dismissal timer.
        let isRepeat = StatusNoticePolicy.isRepeat(
            text: text, kind: kind, source: .composer, current: composerNotice
        )
        let identity: StatusNotice.Identity
        if isRepeat, let current = composerNotice {
            identity = current.id
        } else {
            noticeSerial &+= 1
            identity = StatusNotice.Identity(source: .composer, serial: noticeSerial)
        }
        composerNotice = StatusNotice(id: identity, kind: kind, text: text)
        // The old row was reachable by tabbing; an overlay is not, so announce it
        // — once per identity, because a repeat is the same sentence continued.
        if !isRepeat {
            AccessibilityNotification.Announcement(text).post()
        }
        guard kind.autoDismisses else { return }
        noticeDismissTask = Task {
            try? await Task.sleep(nanoseconds: StatusNoticePolicy.autoDismissNanoseconds)
            guard !Task.isCancelled, composerNotice?.id == identity else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                composerNotice = nil
            }
            noticeDismissTask = nil
        }
    }

    private func clearNotice() {
        noticeDismissTask?.cancel()
        noticeDismissTask = nil
        composerNotice = nil
    }

    /// Clears the notice only if it is the in-flight "preparing" one, so finishing
    /// an attachment cannot wipe a refusal that landed while it decoded.
    private func clearProgressNotice() {
        guard composerNotice?.kind == .progress else { return }
        clearNotice()
    }

    /// The attach menu's tooltip. A disabled row cannot be activated, so this is
    /// the only place the reason is discoverable before a drop or a paste fails.
    private var attachImageHelp: String {
        supportsVision
            ? "Attach an image (or paste / drop one)"
            : ImageAttachmentMessages.cannotSeeImagesAttach(modelName: modelName)
    }

    private func attachmentDetailLine(_ image: PreparedImage) -> String {
        let size = "\(image.pixelWidth)×\(image.pixelHeight) · "
            + ByteCountFormatter.string(fromByteCount: Int64(image.byteCount), countStyle: .file)
        // Only a backend that can price an encode adds anything here, and the cost
        // is measured per machine rather than fixed, so it reads as an estimate.
        guard let cost = visionCostLabel?(image), !cost.isEmpty else { return size }
        return size + " · " + cost
    }

    private func attachmentChipsRow() -> some View {
        VStack(alignment: .leading, spacing: 4) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(pendingImages.indices, id: \.self) { index in
                        let image = pendingImages[index]
                        HStack(spacing: 6) {
                            AttachmentChipThumbnail(image: image)
                                // A staged image the current model cannot see is
                                // marked on the chip itself: durable state at a
                                // constant size, where an extra row would resize
                                // the composer and jump the transcript.
                                .overlay(alignment: .topTrailing) {
                                    if !supportsVision {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .font(.system(size: max(8, 9.5 * zoomManager.zoomScale)))
                                            .foregroundColor(.orange)
                                            .padding(1)
                                            .background(Color(NSColor.controlBackgroundColor))
                                            .clipShape(Circle())
                                            .offset(x: 4, y: -4)
                                            // The refusal copy has one owner: a
                                            // second hand-written sentence here
                                            // could explain the same model
                                            // differently from the notice a drop
                                            // or a send raises.
                                            .help(ImageAttachmentMessages.cannotSeeImagesSend(modelName: modelName))
                                    }
                                }
                            VStack(alignment: .leading, spacing: 1) {
                                Text(image.fileName)
                                    .font(.system(size: max(9, 10.5 * zoomManager.zoomScale), weight: .semibold))
                                    .lineLimit(1)
                                Text(attachmentDetailLine(image))
                                    .font(.system(size: max(8, 9.5 * zoomManager.zoomScale), design: .monospaced))
                                    .foregroundColor(.secondary)
                                    .lineLimit(1)
                            }
                            Button {
                                guard pendingImages.indices.contains(index) else { return }
                                pendingImages.remove(at: index)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                                    .foregroundColor(.secondary)
                            }
                            .buttonStyle(.plain)
                            .help("Remove attachment")
                        }
                        .padding(.horizontal, 7)
                        .padding(.vertical, 5)
                        .background(Color.secondary.opacity(0.08))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(supportsVision ? Color.purple.opacity(0.25) : Color.orange.opacity(0.5), lineWidth: 1)
                        )
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func updateTokenCount(for text: String) {
        tokenCountTask?.cancel()
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            withAnimation(.easeInOut(duration: 0.15)) {
                self.promptTokenCount = 0
            }
            return
        }

        let currentTokenizer = self.tokenizer
        tokenCountTask = Task { @MainActor in
            // 40ms debounce for 120Hz smooth typing
            try? await Task.sleep(nanoseconds: 40_000_000)
            if Task.isCancelled { return }

            let count: Int
            if let tok = currentTokenizer {
                count = (try? tok.encode(text: trimmed).count) ?? max(1, trimmed.count / 4)
            } else {
                count = max(1, trimmed.count / 4)
            }

            if Task.isCancelled { return }
            withAnimation(.easeInOut(duration: 0.15)) {
                self.promptTokenCount = count
            }
        }
    }
}

// MARK: - Custom Mac NSTextView Subclass
final class ComposerTextView: NSTextView {
    var onCommit: (() -> Void)?
    var onCommitImmediate: (() -> Void)?
    /// Consumes an image paste before the text view inserts anything. Returning
    /// true means the pasteboard held an image and was handled.
    var onPasteImage: (() -> Bool)?

    override func paste(_ sender: Any?) {
        if let onPasteImage, onPasteImage() { return }
        super.paste(sender)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 36 && !hasMarkedText() { // Return key
            if event.modifierFlags.contains(.shift) {
                super.keyDown(with: event)
                return
            } else if event.modifierFlags.contains(.command) {
                if let immediate = onCommitImmediate {
                    immediate()
                } else {
                    onCommit?()
                }
                return
            } else {
                onCommit?()
                return
            }
        }
        super.keyDown(with: event)
    }
}

// MARK: - Custom Mac NSTextView Wrapper
struct MacTextEditor: NSViewRepresentable {
    /// Vertical padding above and below the text inside the editor. Shared with
    /// the composer's minimum-height math so a single line of text plus these
    /// insets always fits without vertical clipping.
    static let textContainerInsetHeight: CGFloat = 2

    @Binding var text: String
    var placeholder: String
    var zoomScale: CGFloat = 1.0
    var onCommit: () -> Void
    var onCommitImmediate: (() -> Void)? = nil
    var onPasteImage: (() -> Bool)? = nil

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true

        let textStorage = NSTextStorage()
        let layoutManager = NSLayoutManager()
        textStorage.addLayoutManager(layoutManager)
        let textContainer = NSTextContainer()
        textContainer.widthTracksTextView = true
        layoutManager.addTextContainer(textContainer)

        let textView = ComposerTextView(frame: .zero, textContainer: textContainer)
        textView.delegate = context.coordinator
        textView.onCommit = onCommit
        textView.onCommitImmediate = onCommitImmediate
        textView.onPasteImage = onPasteImage
        textView.font = .systemFont(ofSize: max(10, 13.5 * zoomScale))
        textView.isRichText = false
        textView.drawsBackground = false
        textView.allowsUndo = true
        textView.isEditable = true
        textView.isSelectable = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = NSSize(width: 0, height: Self.textContainerInsetHeight)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        guard let textView = nsView.documentView as? ComposerTextView else { return }
        textView.onCommit = onCommit
        textView.onCommitImmediate = onCommitImmediate
        textView.onPasteImage = onPasteImage
        if textView.string != text {
            textView.string = text
        }
        let targetFont = NSFont.systemFont(ofSize: max(10, 13.5 * zoomScale))
        if abs((textView.font?.pointSize ?? 13.5) - targetFont.pointSize) > 0.1 {
            textView.font = targetFont
        }
    }

    class Coordinator: NSObject, NSTextViewDelegate {
        var parent: MacTextEditor

        init(_ parent: MacTextEditor) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            self.parent.text = textView.string
        }
    }
}

// MARK: - Bounded-Height Text
/// Splits long text into pieces small enough to draw.
///
/// An expanded reasoning block of ~13k tokens left the transcript blank until a
/// manual scroll forced a repaint (the "whiteout"), and output can now run to
/// tens of thousands of tokens. Long text is therefore drawn as a stack of
/// bounded pieces, so no single text view is taller than a couple of thousand
/// points and the transcript never depends on one oversized view painting
/// correctly.
enum LongTextChunker {
    /// Groups lines so no chunk exceeds `maxLines` or `maxCharacters`. A single
    /// line longer than `maxCharacters` (minified JSON, one long diff line) is
    /// split on its own so it cannot produce an enormous text view either.
    nonisolated static func chunks(
        of text: String,
        maxLines: Int = 120,
        maxCharacters: Int = 8000
    ) -> [String] {
        guard !text.isEmpty else { return [] }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        guard lines.count > maxLines || text.count > maxCharacters else { return [text] }

        var chunks: [String] = []
        var current: [Substring] = []
        var currentCharacters = 0

        func flush() {
            guard !current.isEmpty else { return }
            chunks.append(current.joined(separator: "\n"))
            current.removeAll(keepingCapacity: true)
            currentCharacters = 0
        }

        for line in lines {
            if line.count > maxCharacters {
                flush()
                chunks.append(contentsOf: hardSplit(String(line), maxCharacters: maxCharacters))
                continue
            }
            if current.count >= maxLines || currentCharacters + line.count > maxCharacters {
                flush()
            }
            current.append(line)
            currentCharacters += line.count + 1
        }
        flush()
        return chunks.isEmpty ? [text] : chunks
    }

    /// True when `chunks(of:)` would return more than one piece, i.e. when a
    /// single text view would be too tall to paint reliably. Renderers use it
    /// to choose `ChunkedText` without splitting the block itself.
    nonisolated static func isOversized(
        _ text: String,
        maxLines: Int = 120,
        maxCharacters: Int = 8000
    ) -> Bool {
        if text.count > maxCharacters { return true }
        var newlines = 0
        for byte in text.utf8 where byte == 0x0A {
            newlines += 1
            if newlines >= maxLines { return true }
        }
        return false
    }

    private nonisolated static func hardSplit(_ text: String, maxCharacters: Int) -> [String] {
        var pieces: [String] = []
        var start = text.startIndex
        while start < text.endIndex {
            let end = text.index(start, offsetBy: maxCharacters, limitedBy: text.endIndex) ?? text.endIndex
            pieces.append(String(text[start..<end]))
            start = end
        }
        return pieces.isEmpty ? [text] : pieces
    }
}

/// A long string drawn as several `Text` views of bounded height. Inherits
/// font, color, and selection from its parent like a single `Text` would.
struct ChunkedText: View {
    let text: String
    /// Extra leading applied inside each chunk and between chunks, so a chunk
    /// boundary is indistinguishable from an ordinary line break.
    var lineSpacing: CGFloat = 0

    /// Parse each chunk as inline markdown, matching the `Text(LocalizedStringKey:)`
    /// the unbounded path uses. Only markdown blocks set this: a plain user
    /// message or reasoning block must stay verbatim.
    var parsesMarkdown = false

    private final class ChunksMemo {
        var text: String?
        var chunks: [String] = []
    }

    @State private var memo = ChunksMemo()

    private var chunks: [String] {
        if memo.text == text { return memo.chunks }
        let split = LongTextChunker.chunks(of: text)
        memo.text = text
        memo.chunks = split
        return split
    }

    /// One chunk's text. Without markdown parsing an oversized paragraph would
    /// show literal `**bold**`, backticks, and raw link targets — exactly the
    /// case this view exists for. A split inside an inline span can still leave
    /// a marker unbalanced, but that beats dropping the styling entirely.
    @ViewBuilder
    private func chunkText(_ chunk: String) -> some View {
        if parsesMarkdown {
            Text(LocalizedStringKey(chunk))
        } else {
            Text(chunk)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: lineSpacing) {
            if chunks.isEmpty {
                // An empty string still drew one line box as a plain `Text`;
                // a `VStack` with no children would collapse instead.
                chunkText(text)
                    .lineSpacing(lineSpacing)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ForEach(Array(chunks.enumerated()), id: \.offset) { _, chunk in
                    chunkText(chunk)
                        .lineSpacing(lineSpacing)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

// MARK: - Single Message Bubble View
// MARK: - Attachment Thumbnails

/// A composer chip's preview, decoded once per image rather than on every frame:
/// the composer re-renders on every streamed chunk of the transcript.
struct AttachmentChipThumbnail: View {
    let image: PreparedImage
    @State private var thumbnail: NSImage?

    var body: some View {
        Group {
            if let thumbnail {
                Image(nsImage: thumbnail)
                    .resizable()
                    .scaledToFill()
            } else {
                RoundedRectangle(cornerRadius: 5)
                    .fill(Color.secondary.opacity(0.15))
                    .overlay(
                        Image(systemName: "photo")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                    )
            }
        }
        .frame(width: 30, height: 30)
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .task(id: image.sha256) {
            if thumbnail == nil {
                thumbnail = NSImage(data: image.data)
            }
        }
    }
}

/// Thumbnails for a sent message's attachments. Bytes are read from the asset
/// store once per message and downscaled for display; a ref whose file was
/// pruned (retention, a removed asset folder) shows a photo placeholder instead
/// of a broken view.
struct TranscriptAttachmentStrip: View {
    let attachments: [ImageAttachmentRef]
    @State private var thumbnails: [UUID: NSImage] = [:]

    private static func displayImage(from data: Data, maxPixelSize: CGFloat) -> NSImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            ForEach(attachments) { ref in
                Button {
                    if let url = ImageAttachmentStore.shared.absoluteURL(for: ref) {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    Group {
                        if let image = thumbnails[ref.id] {
                            Image(nsImage: image)
                                .resizable()
                                .scaledToFill()
                        } else {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.secondary.opacity(0.12))
                                .overlay(
                                    Image(systemName: "photo")
                                        .font(.system(size: 12))
                                        .foregroundColor(.secondary)
                                )
                        }
                    }
                    .frame(width: 104, height: 78)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
                .help("\(ref.fileName) — \(ref.pixelWidth)×\(ref.pixelHeight), click to open")
            }
        }
        .task(id: attachments.map(\.sha256).joined(separator: "|")) {
            var loaded: [UUID: NSImage] = [:]
            for ref in attachments {
                let data = await Task.detached(priority: .utility) {
                    ImageAttachmentStore.shared.data(for: ref)
                }.value
                if let data, let image = Self.displayImage(from: data, maxPixelSize: 208) {
                    loaded[ref.id] = image
                }
            }
            thumbnails = loaded
        }
    }
}

/// The digest card `/compact` leaves at the head of a compacted conversation.
///
/// Deliberately not a `ChatMessageView`: the card is UI state rather than a
/// turn, and rendering it as chrome (bordered, titled, collapsible) is what
/// keeps a summary of several thousand characters from reading as something the
/// assistant just said. The message is system-role, so no prompt builder ever
/// replays it; `ChatSession.compactionContextBlock` is what the model receives.
struct CompactionSummaryCard: View {
    @EnvironmentObject private var zoomManager: AppZoomManager
    let message: ChatMessage
    @Binding var isExpanded: Bool
    @State private var isCopied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: "archivebox.fill")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [.purple, .indigo],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )

                        Text("Compacted conversation summary")
                            .font(.system(size: max(10, 12 * zoomManager.zoomScale), weight: .semibold))
                            .foregroundColor(.primary)

                        Spacer(minLength: 0)

                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(.secondary.opacity(0.8))
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(isExpanded ? "Hide the compacted summary" : "Show the compacted summary")
                .accessibilityLabel("Compacted conversation summary")
                .accessibilityHint(isExpanded ? "Collapse the summary" : "Expand the summary")

                // The summary is the only copy of the digests on screen, so it
                // is offered as one run instead of requiring a selection across
                // rendered markdown blocks.
                Button(action: {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(message.content, forType: .string)
                    isCopied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                        isCopied = false
                    }
                }) {
                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                        .foregroundColor(isCopied ? .green : .secondary)
                        .padding(6)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .help("Copy the compacted summary")
                // Icon-only, so VoiceOver would otherwise read the SF Symbol's
                // own name. The label carries the copied state, which is how the
                // confirmation that used to be a bare checkmark reaches someone
                // who cannot see it.
                .accessibilityLabel(isCopied ? "Compacted summary copied" : "Copy the compacted summary")
            }

            if isExpanded {
                MarkdownMessageView(content: message.content)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.purple.opacity(0.16), lineWidth: 1)
        )
    }
}

/// The plan-mode deliverable: an assistant turn whose content opens with
/// `PlanMarkers.start`. Rendered as chrome rather than as an ordinary reply so a
/// several-hundred-line plan reads as a document to approve, not as chat.
///
/// The border is dashed while the plan streams and solid once
/// `PlanMarkers.ready` arrives — the visual counterpart of Crush's open and
/// closed plan card. The markers themselves are stripped by the caller.
struct PlanCard: View {
    @EnvironmentObject private var zoomManager: AppZoomManager
    let content: String
    /// The plan turn's chain of thought. The card replaces `ChatMessageView`
    /// entirely, so without this the one message a user reads most carefully
    /// would be the only one whose reasoning could not be inspected.
    let thinkingContent: String?
    let isReady: Bool
    let isStreaming: Bool
    @Binding var isReasoningExpanded: Bool
    @State private var isCopied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "checklist")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.purple, .indigo],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Text("Implementation Plan")
                    .font(.system(size: max(10, 12 * zoomManager.zoomScale), weight: .semibold))
                    .foregroundColor(.primary)

                Text(isReady ? "Plan ready" : (isStreaming ? "Planning…" : "Plan"))
                    .font(.system(size: max(8.5, 10 * zoomManager.zoomScale), weight: .bold))
                    .foregroundColor(isReady ? .purple : .secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background((isReady ? Color.purple : Color.secondary).opacity(0.12))
                    .cornerRadius(5)

                Spacer(minLength: 0)

                Button(action: {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(content, forType: .string)
                    isCopied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                        isCopied = false
                    }
                }) {
                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                        .foregroundColor(isCopied ? .green : .secondary)
                        .padding(6)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .help("Copy the plan")
                .accessibilityLabel(isCopied ? "Plan copied" : "Copy the plan")
            }

            if let thinking = thinkingContent, !thinking.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            isReasoningExpanded.toggle()
                        }
                    }) {
                        HStack(spacing: 5) {
                            Image(systemName: "brain.head.profile")
                                .font(.system(size: max(8.5, 10 * zoomManager.zoomScale)))
                                .foregroundColor(.secondary)
                            Text("Reasoning")
                                .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .medium))
                                .foregroundColor(.secondary)
                            Image(systemName: isReasoningExpanded ? "chevron.down" : "chevron.right")
                                .font(.system(size: 8, weight: .bold))
                                .foregroundColor(.secondary.opacity(0.8))
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(isReasoningExpanded ? "Hide the plan's reasoning" : "Show the plan's reasoning")

                    if isReasoningExpanded {
                        Text(thinking)
                            .font(.system(size: max(9.5, 11.5 * zoomManager.zoomScale)))
                            .foregroundColor(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.secondary.opacity(0.05))
                            .cornerRadius(6)
                    }
                }
            }

            MarkdownMessageView(content: content, isStreaming: isStreaming && !isReady)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(
                    Color.purple.opacity(isReady ? 0.45 : 0.22),
                    style: StrokeStyle(lineWidth: 1, dash: isReady ? [] : [4, 3])
                )
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Implementation plan")
        .accessibilityValue(isReady ? "Ready to implement" : "In progress")
    }
}

struct ChatMessageView: View {
    @EnvironmentObject private var zoomManager: AppZoomManager
    let message: ChatMessage
    var isGenerating: Bool = false
    var isStreamingOffDisk: Bool = false
    var isCompactingConversation: Bool = false
    /// True when the active model emits the native Gemma dialect; gates the Gemma
    /// display cleanup so other models' answers are never truncated at the token.
    var isGemmaDialect: Bool = false
    @Binding var isExpanded: Bool
    @State private var isCopied = false
    @State private var isThinkingCopied = false

    /// Memoizes the display-cleanup pass for this message. It strips tool-call
    /// markup with two regex scans plus three opener searches, and the
    /// transcript re-renders on every streamed chunk; without the memo every
    /// visible message repeated that work per frame. Reference type so the body
    /// can refresh it in place without scheduling another update.
    private final class DisplayMemo {
        var content: String?
        var gemmaDialect = false
        var display = ""
    }

    @State private var displayMemo = DisplayMemo()

    private var displayContent: String {
        if displayMemo.content == message.content, displayMemo.gemmaDialect == isGemmaDialect {
            return displayMemo.display
        }
        let cleaned = Self.responseDisplayContent(message.content, isGemmaDialect: isGemmaDialect)
        displayMemo.content = message.content
        displayMemo.gemmaDialect = isGemmaDialect
        displayMemo.display = cleaned
        return cleaned
    }

    /// True when the footer has any telemetry to show. Kept separate from the
    /// display-content check so tool-only assistant turns (which clean down to
    /// empty text) still surface TTFT, throughput, and token counts while the
    /// agent runs their calls.
    private var hasTelemetry: Bool {
        message.timeToFirstTokenSeconds != nil
            || message.tokensPerSec > 0
            || message.tokenCount > 0
            || (message.jetSpecTau ?? 0) > 1.0
    }

    /// True when the raw content holds a tool-call opener that has not yet parsed
    /// into a concrete call, i.e. the model is mid-stream on the call's arguments.
    private var isStreamingToolCall: Bool {
        if let calls = message.toolCalls, !calls.isEmpty { return false }
        return Self.isToolOnlyStreamingContent(message.content, isGemmaDialect: isGemmaDialect)
    }

    /// One marker a model opens a tool call with.
    struct ToolCallMarker {
        let text: String
        /// Gemma permits no suffix after a call, and its marker is a documented
        /// vocabulary token an answer may quote, so only its own dialect may read it
        /// as an opener. Every other marker is call machinery under any dialect.
        let onlyInGemmaDialect: Bool
    }

    /// Every shape a call can open with, in ONE place. Two consumers read it — the
    /// display truncation (`cleanedMarkdown`) and the streaming-pill predicate
    /// (`firstToolCallOpenerRange`) — and both ask the same question, "from here on the
    /// text is call machinery rather than answer". Keeping two lists let a new dialect
    /// be recognised by one and not the other, which is exactly what the DeepSeek-V4.1
    /// DSML spelling did: its call streamed as an empty bubble (markup stripped) with
    /// no working pill, because nothing said the turn was mid-call. Adding a marker
    /// here is all a new dialect needs.
    static let toolCallMarkers: [ToolCallMarker] = [
        ToolCallMarker(text: StreamingToolParser.qwenToolCallOpen, onlyInGemmaDialect: false),
        ToolCallMarker(text: StreamingToolParser.qwenFunctionOpen, onlyInGemmaDialect: false),
        ToolCallMarker(text: StreamingToolParser.llamaTagOpen, onlyInGemmaDialect: false),
        // DeepSeek-V4.1 writes neither of those shapes: its agent prompt teaches the
        // DSML container (`#27`), one vocabulary tag token wrapping `invoke` elements,
        // so the tag itself is the opener. `callNameOpen` is that family's other
        // spelling, a bare `<tool_call_name>` tag around a JSON argument object.
        ToolCallMarker(
            text: "<" + DeepSeekV41ToolDialect.dsmlTag,
            onlyInGemmaDialect: false
        ),
        ToolCallMarker(text: DeepSeekV41ToolDialect.callNameOpen, onlyInGemmaDialect: false),
        ToolCallMarker(text: StreamingToolParser.gemmaToolCallOpen, onlyInGemmaDialect: true),
    ]

    /// The markers that count as openers under the active dialect.
    private static func activeToolCallMarkers(isGemmaDialect: Bool) -> [String] {
        toolCallMarkers
            .filter { isGemmaDialect || !$0.onlyInGemmaDialect }
            .map(\.text)
    }

    /// The earliest of `markers` in `raw`, or nil when `raw` carries none.
    private static func earliestMarker(
        of markers: [String],
        in raw: String
    ) -> Range<String.Index>? {
        var earliest: Range<String.Index>?
        for marker in markers {
            guard let found = raw.range(of: marker) else { continue }
            if earliest == nil || found.lowerBound < earliest!.lowerBound {
                earliest = found
            }
        }
        return earliest
    }

    /// The range of the earliest tool-call opener of the active dialect, or nil when
    /// the text carries none. The Gemma marker is only an opener under the Gemma
    /// dialect, so a non-Gemma answer that documents `<|tool_call>` keeps its text.
    static func firstToolCallOpenerRange(
        in raw: String,
        isGemmaDialect: Bool
    ) -> Range<String.Index>? {
        earliestMarker(of: activeToolCallMarkers(isGemmaDialect: isGemmaDialect), in: raw)
    }

    /// True when `raw` holds a tool-call opener with no visible text before it, i.e.
    /// the model is mid-stream on a call and has produced nothing readable yet. This
    /// is what suppresses raw call markup (bare `<function=...>` / `<|python_tag|>`)
    /// from rendering and drives the inline "Working..." pill. Documented markers in
    /// prose keep their preceding text, so they never qualify.
    static func isToolOnlyStreamingContent(_ raw: String, isGemmaDialect: Bool) -> Bool {
        guard let range = firstToolCallOpenerRange(in: raw, isGemmaDialect: isGemmaDialect) else {
            return false
        }
        let prefix = String(raw[..<range.lowerBound])
        return cleanedMarkdown(prefix, isGemmaDialect: isGemmaDialect).isEmpty
    }

    /// The response text to render for a turn. Empty when the raw text is only
    /// tool-call markup, so bare `<function=...>` / `<|python_tag|>` calls never
    /// reappear as raw response text (nor expose the copy/speaker actions) once a
    /// `ToolCallRecord` has parsed — the parsed-state check lives only in the
    /// working-indicator predicate, not here.
    static func responseDisplayContent(_ raw: String, isGemmaDialect: Bool) -> String {
        if isToolOnlyStreamingContent(raw, isGemmaDialect: isGemmaDialect) { return "" }
        return cleanedMarkdown(raw, isGemmaDialect: isGemmaDialect)
    }

    /// True while the agent is actively working on a tool call, either streaming
    /// the call's tokens or executing it. Drives the inline "Working..." indicator.
    private var isAgentWorkingOnToolCall: Bool {
        guard message.role == .assistant, isGenerating else { return false }
        if let calls = message.toolCalls, calls.contains(where: { $0.status == .running }) {
            return true
        }
        return isStreamingToolCall
    }

    /// Strips tool-call markup from assistant text for display, truncating at the
    /// earliest marker that survived the block strip. The marker table is shared with
    /// the working-pill predicate, so a dialect cannot be readable in one and invisible
    /// in the other; only the Gemma marker is dialect-gated, because Gemma permits no
    /// suffix after a call while a non-Gemma answer that merely documents
    /// `<|tool_call>` must keep the rest of its text.
    static func cleanedMarkdown(_ content: String, isGemmaDialect: Bool) -> String {
        var clean = content
        // Complete blocks go first, so a closed call is removed without discarding an
        // answer that resumes after it. What is left of a block the regexes cannot
        // match (a dialect's own spelling, or a call still streaming) is caught by the
        // truncation below, which is also why a payload carrying a quoted terminator
        // can never be shown as an answer.
        clean = clean
            .replacingOccurrences(of: "<tool_call>[\\s\\S]*?</tool_call>", with: "", options: .regularExpression)
            .replacingOccurrences(of: "<tool_response>[\\s\\S]*?</tool_response>", with: "", options: .regularExpression)
            // DeepSeek-V4.1's own call tags (`#27`): the call is executed from the raw
            // text, so the bubble must not show it as prose.
            .replacingOccurrences(of: "<tool_call_name>[\\s\\S]*?</tool_call_arguments>", with: "", options: .regularExpression)
        if let opener = firstToolCallOpenerRange(in: clean, isGemmaDialect: isGemmaDialect) {
            clean = String(clean[..<opener.lowerBound])
        }
        return clean.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            if message.role == .assistant {
                // Assistant Avatar
                Image(systemName: "circle.hexagongrid.fill")
                    .font(.system(size: 15))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.purple, .indigo],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 28, height: 28)
                    .background(Color.purple.opacity(0.12))
                    .clipShape(Circle())
                    .padding(.top, 2)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 8) {
                // User Message Bubble (Constrained max width, elegant purple card)
                if message.role == .user {
                    HStack {
                        Spacer(minLength: 40)
                        VStack(alignment: .trailing, spacing: 6) {
                            if let attachments = message.attachments, !attachments.isEmpty {
                                TranscriptAttachmentStrip(attachments: attachments)
                            }
                            // An image-only turn has no text bubble at all; an
                            // empty padded bubble under the pictures reads as a
                            // rendering bug.
                            if !message.content.isEmpty {
                                ChunkedText(text: message.content, lineSpacing: 3)
                                    .font(.system(size: max(9, 14 * zoomManager.zoomScale)))
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 10)
                                    .background(Color.purple.opacity(0.18))
                                    .foregroundColor(.primary)
                                    .cornerRadius(14)
                                    .textSelection(.enabled)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                } else {
                    // Assistant Thinking / Reasoning Accordion (Osaurus Style)
                    let hasThinkingContent = message.thinkingContent != nil && !message.thinkingContent!.isEmpty
                    if (message.isThinking && isGenerating) || hasThinkingContent {
                        let thinking = message.thinkingContent ?? ""
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 6) {
                                Button(action: {
                                    withAnimation(.easeInOut(duration: 0.2)) {
                                        isExpanded.toggle()
                                    }
                                }) {
                                    HStack(spacing: 8) {
                                        Image(systemName: "brain.head.profile")
                                            .font(.system(size: 12, weight: .semibold))
                                            .foregroundStyle(
                                                LinearGradient(
                                                    colors: [.purple, .indigo],
                                                    startPoint: .topLeading,
                                                    endPoint: .bottomTrailing
                                                )
                                            )

                                        if message.isThinking && isGenerating {
                                            if message.prefillStatus != nil {
                                                Text("Ingesting Prompt...")
                                                    .font(.system(size: 12, weight: .medium))
                                                    .foregroundColor(.primary)
                                            } else {
                                                Text("Thinking...")
                                                    .font(.system(size: 12, weight: .medium))
                                                    .foregroundColor(.primary)
                                            }
                                            StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                                        } else if let tTime = message.thinkingTimeSeconds {
                                            Text(String(format: "Thought for %.1fs", tTime))
                                                .font(.system(size: 12, weight: .medium))
                                                .foregroundColor(.secondary)
                                        } else {
                                            Text("Thought Process")
                                                .font(.system(size: 12, weight: .medium))
                                                .foregroundColor(.secondary)
                                        }

                                        Spacer()

                                        let charCount = thinking.count
                                        let charCountStr = charCount >= 1000 ? String(format: "%.1fk chars", Double(charCount) / 1000.0) : "\(charCount) chars"
                                        Text(charCountStr)
                                            .font(.system(size: 11, design: .monospaced))
                                            .foregroundColor(.secondary.opacity(0.7))

                                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                                            .font(.system(size: 10, weight: .bold))
                                            .foregroundColor(.secondary.opacity(0.8))
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(Color(NSColor.controlBackgroundColor).opacity(0.7))
                                    .cornerRadius(8)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                                    )
                                }
                                .buttonStyle(.plain)

                                // Chunked reasoning cannot be selected as one run
                                // across sibling Text views, so the whole thought
                                // process is offered in one gesture here instead.
                                if !thinking.isEmpty {
                                    Button(action: {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(thinking, forType: .string)
                                        isThinkingCopied = true
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                            isThinkingCopied = false
                                        }
                                    }) {
                                        Image(systemName: isThinkingCopied ? "checkmark" : "doc.on.doc")
                                            .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                                            .foregroundColor(isThinkingCopied ? .green : .secondary)
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 8)
                                            .background(Color(NSColor.controlBackgroundColor).opacity(0.7))
                                            .cornerRadius(8)
                                            .overlay(
                                                RoundedRectangle(cornerRadius: 8)
                                                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                                            )
                                    }
                                    .buttonStyle(.plain)
                                    .help("Copy thought process")
                                }
                            }

                            if isExpanded {
                                VStack(alignment: .leading, spacing: 6) {
                                    if let prefill = message.prefillStatus, isGenerating {
                                        HStack(spacing: 8) {
                                            ProgressView()
                                                .controlSize(.small)
                                            Text(prefill)
                                                .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                                                .foregroundColor(.purple)
                                        }
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 7)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(Color.purple.opacity(0.08))
                                        .cornerRadius(6)
                                    }

                                    if !thinking.isEmpty {
                                        // Reasoning can run to tens of thousands of
                                        // tokens; one Text of that height is
                                        // never painted (the whiteout), so it is
                                        // drawn as bounded chunks instead.
                                        ChunkedText(text: thinking, lineSpacing: 3)
                                            .font(.system(size: max(8, 12 * zoomManager.zoomScale), design: .monospaced))
                                            .foregroundColor(.secondary)
                                            .padding(.horizontal, 14)
                                            .padding(.vertical, 10)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .fixedSize(horizontal: false, vertical: true)
                                            .background(Color.secondary.opacity(0.035))
                                            .cornerRadius(8)
                                            .overlay(
                                                RoundedRectangle(cornerRadius: 8)
                                                    .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                                            )
                                            .textSelection(.enabled)
                                    } else if message.isThinking && isGenerating && message.prefillStatus == nil {
                                        HStack(spacing: 8) {
                                            StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                                            Text(isStreamingOffDisk ? "Streaming MoE experts off SSD disk..." : "Generating thought process...")
                                                .font(.system(size: 12, design: .monospaced))
                                                .foregroundColor(.secondary.opacity(0.8))
                                        }
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 10)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(Color.secondary.opacity(0.035))
                                        .cornerRadius(8)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 8)
                                                .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                                        )
                                    }
                                }
                            }
                        }
                    }

                    // Tool Calls Execution Cards & Sequential Timeline (if any tool calls were issued)
                    // Gesture calls keep a record for history reconstruction but are hidden here.
                    if let toolCalls = message.toolCalls {
                        let visibleToolCalls = toolCalls.filter { !$0.isGesture }
                        if !visibleToolCalls.isEmpty {
                            ToolExecutionTimelineView(toolCalls: visibleToolCalls)
                        }
                    }

                    // Inline agent-work indicator: shown while the model streams a
                    // tool call's arguments or the harness executes a running call,
                    // so slow calls (e.g. writing a script) still read as active work.
                    if isAgentWorkingOnToolCall {
                        AgentWorkingIndicatorView()
                            .padding(.top, 2)
                    }

                    // Main Response Text rendered via Rich Markdown Engine.
                    // Raw call markup is suppressed for tool-only turns, both while
                    // streaming and after the call has parsed.
                    if !displayContent.isEmpty {
                        MarkdownMessageView(
                            content: displayContent,
                            isStreaming: (isGenerating || isCompactingConversation) && message.isThinking == false,
                            isStreamingOffDisk: isStreamingOffDisk
                        )
                    } else if (isGenerating || isCompactingConversation) && !message.isThinking && !hasThinkingContent && (message.toolCalls == nil || message.toolCalls!.isEmpty) && !isStreamingToolCall {
                        // Only for non-thinking models during initial prefill / generation,
                        // or the /compact progress bubble before its first summary text
                        if let prefill = message.prefillStatus {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .controlSize(.small)
                                Text(prefill)
                                    .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                                    .foregroundColor(.purple)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(Color.purple.opacity(0.08))
                            .cornerRadius(8)
                        } else {
                            HStack(spacing: 8) {
                                StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                                Text(isCompactingConversation ? "Summarizing the conversation..." : (isStreamingOffDisk ? "Streaming MoE experts off SSD disk..." : "Generating response..."))
                                    .font(.system(size: 13))
                                    .foregroundColor(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }

                    // Action & Metrics Footer (Osaurus Style Telemetry & Actions)
                    if (!message.isThinking || !isGenerating) && (hasTelemetry || !displayContent.isEmpty) {
                        HStack(spacing: 12) {
                            // Telemetry
                            HStack(spacing: 6) {
                                if let ttft = message.timeToFirstTokenSeconds {
                                    Text(String(format: "TTFT %.2fs", ttft))
                                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                                        .foregroundColor(.secondary.opacity(0.8))

                                    Text("•")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary.opacity(0.4))
                                }

                                if message.tokensPerSec > 0 {
                                    HStack(spacing: 3) {
                                        Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")
                                            .font(.system(size: 9, weight: .bold))
                                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                        Text(String(format: "%.1f tok/s", message.tokensPerSec))
                                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                    }

                                    Text("•")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary.opacity(0.4))
                                }

                                if message.tokenCount > 0 {
                                    Text("\(message.tokenCount.formatted()) tokens")
                                        .font(.system(size: max(8.5, 11 * zoomManager.zoomScale), design: .monospaced))
                                        .foregroundColor(.secondary.opacity(0.8))
                                }

                                if let tau = message.jetSpecTau, tau > 1.0 {
                                    Text("•")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary.opacity(0.4))
                                    HStack(spacing: 3) {
                                        Image(systemName: "bolt.badge.sparkle")
                                            .font(.system(size: max(7.5, 9.5 * zoomManager.zoomScale), weight: .bold))
                                            .foregroundColor(.cyan)
                                        Text(String(format: "JetSpec τ=%.1fx", tau))
                                            .font(.system(size: max(8.5, 11 * zoomManager.zoomScale), weight: .semibold, design: .monospaced))
                                            .foregroundColor(.cyan)
                                    }
                                }
                            }

                            Spacer()

                            // Action Buttons (only when there is visible text to act on)
                            if !displayContent.isEmpty {
                                HStack(spacing: 8) {
                                    // Copy button
                                    Button(action: {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(message.content, forType: .string)
                                        isCopied = true
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                            isCopied = false
                                        }
                                    }) {
                                        HStack(spacing: 4) {
                                            Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                                                .font(.system(size: max(8.5, 11 * zoomManager.zoomScale)))
                                            if isCopied {
                                                Text("Copied")
                                                    .font(.system(size: max(8.5, 11 * zoomManager.zoomScale)))
                                            }
                                        }
                                        .foregroundColor(isCopied ? .green : .secondary)
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 3)
                                        .background(Color.secondary.opacity(0.06))
                                        .cornerRadius(4)
                                    }
                                    .buttonStyle(.plain)
                                    .help("Copy full response")

                                    // Audio speak (model replies only)
                                    if message.role == .assistant {
                                        Button(action: {
                                            NSSpeechSynthesizer().startSpeaking(displayContent)
                                        }) {
                                            Image(systemName: "speaker.wave.2")
                                                .font(.system(size: max(8.5, 11 * zoomManager.zoomScale)))
                                                .foregroundColor(.secondary)
                                                .padding(4)
                                                .background(Color.secondary.opacity(0.06))
                                                .cornerRadius(4)
                                        }
                                        .buttonStyle(.plain)
                                        .help("Read response aloud")
                                    }
                                }
                            }
                        }
                        .padding(.top, 4)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
        }
    }
}

// MARK: - Welcome / Empty State View
struct EmptyWelcomeView: View {
    @ObservedObject private var zoomManager = AppZoomManager.shared
    var modelName: String?
    var isStreamingOffDisk: Bool = false

    var body: some View {
        VStack(spacing: max(20, 28 * zoomManager.zoomScale)) {
            VStack(spacing: max(8, 12 * zoomManager.zoomScale)) {
                Image(systemName: "circle.hexagongrid.circle.fill")
                    .font(.system(size: max(36, 52 * zoomManager.zoomScale)))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.purple, .indigo, .blue],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Text("How can I help you today?")
                    .font(.system(size: max(16, 22 * zoomManager.zoomScale), weight: .bold))

                if let model = modelName {
                    HStack(spacing: 6) {
                        Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")
                            .font(.system(size: max(9, 12 * zoomManager.zoomScale)))
                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                        Text("Powered by \(model) • \(isStreamingOffDisk ? "Dynamic SSD Streaming" : "Fast Unified RAM Engine")")
                            .font(.system(size: max(10, 13 * zoomManager.zoomScale)))
                            .foregroundColor(.secondary)
                    }
                } else {
                    Text("Load model weights in Settings (bottom left) to start generating.")
                        .font(.system(size: max(10, 13 * zoomManager.zoomScale)))
                        .foregroundColor(.secondary)
                }
            }

        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Markdown Table & Typography Models

public enum TextAlignmentType {
    case leading
    case center
    case trailing

    public var alignment: Alignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    public var textAlignment: TextAlignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }
}

public struct MarkdownTableData: Identifiable, Equatable {
    public let id: String
    public let headers: [String]
    public let alignments: [TextAlignmentType]
    public let rows: [[String]]

    public init(headers: [String], alignments: [TextAlignmentType], rows: [[String]], id: String? = nil) {
        self.headers = headers
        self.alignments = alignments
        self.rows = rows
        self.id = id ?? "tbl-\(headers.joined(separator: "_"))-\(rows.count)"
    }
}

// MARK: - Markdown Block AST

public enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(text: String)
    case table(MarkdownTableData)
    case codeBlock(language: String, code: String)
    case blockquote(text: String)
    case listItem(number: Int?, text: String)
    case horizontalRule
}

public struct IdentifiableMarkdownBlock: Identifiable, Equatable {
    public let id: String
    public let block: MarkdownBlock
    public let isLast: Bool
}

// MARK: - Markdown Message View

struct MarkdownMessageView: View {
    @EnvironmentObject private var zoomManager: AppZoomManager
    let content: String
    var isStreaming: Bool = false
    var isStreamingOffDisk: Bool = false

    /// Memoizes this message's block parse. Parsing runs ~20 regex passes over
    /// the whole text (measured at ~30 ms per 40 KB and ~85 ms per 120 KB), and
    /// a streaming turn re-renders the transcript on every chunk; re-parsing
    /// every message each frame saturates the main thread once a conversation
    /// gets long, which leaves the transcript unpainted until a manual scroll.
    /// Only the message whose text actually changed re-parses. A reference type
    /// lets the body refresh the memo in place without scheduling an update.
    private final class BlockMemo {
        var content: String?
        var repairs = true
        var blocks: [MarkdownBlock] = []
    }

    @State private var memo = BlockMemo()

    private var parsedBlocks: [MarkdownBlock] {
        // While the answer streams, skip the malformed-markdown repair passes:
        // they cost more than the rest of the parse together and the growing
        // text is re-parsed on every UI update. The settled message re-parses
        // once with them enabled, so its rendering is unaffected.
        let wantsRepairs = !isStreaming
        if memo.content == content, memo.repairs == wantsRepairs { return memo.blocks }
        let parsed = parseMarkdownBlocks(content, repairs: wantsRepairs)
        memo.content = content
        memo.repairs = wantsRepairs
        memo.blocks = parsed
        return parsed
    }

    private var identifiableBlocks: [IdentifiableMarkdownBlock] {
        let rawBlocks = parsedBlocks
        return rawBlocks.enumerated().map { index, block in
            let isLast = isStreaming && (index == rawBlocks.count - 1)
            let blockId: String
            switch block {
            case .heading(let lvl, _): blockId = "h-\(index)-\(lvl)"
            case .paragraph: blockId = "p-\(index)"
            case .table(let tbl): blockId = "tbl-\(index)-\(tbl.headers.count)"
            case .codeBlock(let lang, _): blockId = "code-\(index)-\(lang)"
            case .blockquote: blockId = "q-\(index)"
            case .listItem(let num, _): blockId = "li-\(index)-\(num ?? 0)"
            case .horizontalRule: blockId = "hr-\(index)"
            }
            return IdentifiableMarkdownBlock(id: blockId, block: block, isLast: isLast)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(identifiableBlocks) { item in
                switch item.block {
                case .heading(let level, let text):
                    renderHeading(level: level, text: text, isLast: item.isLast)
                case .paragraph(let text):
                    renderParagraph(text: text, isLast: item.isLast)
                case .table(let table):
                    MarkdownTableView(table: table, isLast: item.isLast, isStreamingOffDisk: isStreamingOffDisk)
                case .codeBlock(let language, let code):
                    CodeBlockCard(
                        language: language,
                        code: code,
                        isStreaming: item.isLast,
                        isStreamingOffDisk: isStreamingOffDisk
                    )
                case .blockquote(let text):
                    renderBlockquote(text: text, isLast: item.isLast)
                case .listItem(let number, let text):
                    renderListItem(number: number, text: text, isLast: item.isLast)
                case .horizontalRule:
                    Divider()
                        .padding(.vertical, 6)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Block Renderers

    @ViewBuilder
    private func renderHeading(level: Int, text: String, isLast: Bool) -> some View {
        let baseSize: CGFloat = (level == 1 ? 19 : (level == 2 ? 16.5 : (level == 3 ? 15 : 14)))
        let size: CGFloat = baseSize * zoomManager.zoomScale
        let weight: Font.Weight = level <= 2 ? .bold : .semibold
        let topPad: CGFloat = level == 1 ? 12 : (level == 2 ? 10 : 6)
        let botPad: CGFloat = level == 1 ? 4 : 2

        if isLast {
            (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: max(8, 11 * zoomManager.zoomScale), weight: .bold)))
                .font(.system(size: max(9, size), weight: weight))
                .foregroundColor(.primary)
                .padding(.top, topPad)
                .padding(.bottom, botPad)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            Text(LocalizedStringKey(text))
                .font(.system(size: max(9, size), weight: weight))
                .foregroundColor(.primary)
                .padding(.top, topPad)
                .padding(.bottom, botPad)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Paragraph text, drawn as bounded chunks only once a single text view
    /// would be too tall to paint. The split happens inside the block rather
    /// than by emitting sibling blocks, so a paragraph keeps its own spacing
    /// and does not gain an inter-block gap at each boundary.
    @ViewBuilder
    private func paragraphText(_ text: String) -> some View {
        if LongTextChunker.isOversized(text) {
            ChunkedText(text: text, lineSpacing: 4, parsesMarkdown: true)
        } else {
            Text(LocalizedStringKey(text))
        }
    }

    @ViewBuilder
    private func renderParagraph(text: String, isLast: Bool) -> some View {
        let fontSize = max(9, 14 * zoomManager.zoomScale)
        let oversized = LongTextChunker.isOversized(text)
        if isLast && !oversized {
            (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: max(8, 11 * zoomManager.zoomScale), weight: .bold)))
                .font(.system(size: fontSize))
                .lineSpacing(4)
                .foregroundColor(.primary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            VStack(alignment: .leading, spacing: 2) {
                paragraphText(text)
                if isLast {
                    StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                }
            }
            .font(.system(size: fontSize))
            .lineSpacing(4)
            .foregroundColor(.primary)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// Blockquote text, bounded the same way as a paragraph: splitting inside
    /// the block keeps one bar and one background instead of a stack of quote
    /// boxes.
    @ViewBuilder
    private func blockquoteText(_ text: String) -> some View {
        if LongTextChunker.isOversized(text) {
            ChunkedText(text: text, lineSpacing: 3.5, parsesMarkdown: true)
        } else {
            Text(LocalizedStringKey(text))
        }
    }

    @ViewBuilder
    private func renderBlockquote(text: String, isLast: Bool) -> some View {
        let fontSize = max(8.5, 13.5 * zoomManager.zoomScale)
        let oversized = LongTextChunker.isOversized(text)
        HStack(alignment: .top, spacing: 10) {
            RoundedRectangle(cornerRadius: 1.5)
                .fill(
                    LinearGradient(
                        colors: [.purple, .indigo],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 3.5)

            if isLast && !oversized {
                (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: max(8, 11 * zoomManager.zoomScale), weight: .bold)))
                    .font(.system(size: fontSize))
                    .italic()
                    .foregroundColor(.secondary)
                    .lineSpacing(3.5)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    blockquoteText(text)
                    if isLast {
                        StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                    }
                }
                .font(.system(size: fontSize))
                .italic()
                .foregroundColor(.secondary)
                .lineSpacing(3.5)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(Color.purple.opacity(0.04))
        .cornerRadius(6)
    }

    @ViewBuilder
    private func listItemText(_ text: String) -> some View {
        if LongTextChunker.isOversized(text) {
            ChunkedText(text: text, lineSpacing: 3, parsesMarkdown: true)
        } else {
            Text(LocalizedStringKey(text))
        }
    }

    @ViewBuilder
    private func renderListItem(number: Int?, text: String, isLast: Bool) -> some View {
        let fontSize = max(9, 14 * zoomManager.zoomScale)
        let oversized = LongTextChunker.isOversized(text)
        HStack(alignment: .top, spacing: 8) {
            if let num = number {
                Text("\(num).")
                    .font(.system(size: max(8.5, 13 * zoomManager.zoomScale), weight: .semibold, design: .monospaced))
                    .foregroundColor(.secondary)
                    .frame(minWidth: 20, alignment: .trailing)
            } else {
                Text("•")
                    .font(.system(size: max(10, 15 * zoomManager.zoomScale), weight: .bold))
                    .foregroundColor(.purple)
                    .frame(width: 14, alignment: .center)
            }

            if isLast && !oversized {
                (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: max(8, 11 * zoomManager.zoomScale), weight: .bold)))
                    .font(.system(size: fontSize))
                    .lineSpacing(3)
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    listItemText(text)
                    if isLast {
                        StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                    }
                }
                .font(.system(size: fontSize))
                .lineSpacing(3)
                .foregroundColor(.primary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 1.5)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Native Markdown Table Card View (SwiftUI Grid)

struct MarkdownTableView: View {
    @EnvironmentObject private var zoomManager: AppZoomManager
    let table: MarkdownTableData
    var isLast: Bool = false
    var isStreamingOffDisk: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView(.horizontal, showsIndicators: true) {
                Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                    // Header Row
                    GridRow {
                        ForEach(0..<table.headers.count, id: \.self) { colIdx in
                            let header = table.headers[colIdx]
                            let align = table.alignments.indices.contains(colIdx) ? table.alignments[colIdx] : .leading
                            HStack(spacing: 0) {
                                Text(LocalizedStringKey(header))
                                    .font(.system(size: max(8.5, 13 * zoomManager.zoomScale), weight: .bold))
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(align.textAlignment)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 9)
                                    .frame(minWidth: 90, alignment: align.alignment)
                                    .textSelection(.enabled)

                                if colIdx < table.headers.count - 1 {
                                    Divider()
                                        .opacity(0.35)
                                }
                            }
                        }
                    }
                    .background(Color.secondary.opacity(0.08))

                    // Header Divider
                    GridRow {
                        Divider()
                            .gridCellColumns(max(1, table.headers.count))
                    }

                    // Data Rows
                    ForEach(0..<table.rows.count, id: \.self) { rowIdx in
                        let row = table.rows[rowIdx]
                        GridRow {
                            ForEach(0..<table.headers.count, id: \.self) { colIdx in
                                let cell = colIdx < row.count ? row[colIdx] : ""
                                let align = table.alignments.indices.contains(colIdx) ? table.alignments[colIdx] : .leading
                                HStack(spacing: 0) {
                                    Text(LocalizedStringKey(cell))
                                        .font(.system(size: max(8.5, 13 * zoomManager.zoomScale)))
                                        .foregroundColor(.primary)
                                        .multilineTextAlignment(align.textAlignment)
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 8)
                                        .frame(minWidth: 90, alignment: align.alignment)
                                        .textSelection(.enabled)

                                    if colIdx < table.headers.count - 1 {
                                        Divider()
                                            .opacity(0.25)
                                    }
                                }
                            }
                        }
                        .background(rowIdx % 2 == 1 ? Color.secondary.opacity(0.03) : Color.clear)

                        if rowIdx < table.rows.count - 1 {
                            GridRow {
                                Divider()
                                    .opacity(0.25)
                                    .gridCellColumns(max(1, table.headers.count))
                            }
                        }
                    }
                }
                .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                )
            }

            if isLast {
                HStack(spacing: 6) {
                    StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                    if isStreamingOffDisk {
                        Text("Streaming MoE experts off NVMe disk...")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.orange)
                    }
                }
                .padding(.top, 2)
            }
        }
        .padding(.vertical, 6)
    }
}

// MARK: - Native Multi-Language Syntax Highlighter (Pure Swift AST / Token Engine)

public struct NativeSyntaxHighlighter {
    public static func highlight(code: String, language: String) -> AttributedString {
        let lang = language.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        
        let kwList: Set<String>
        let isPythonLike = (lang == "python" || lang == "py" || lang == "sh" || lang == "bash" || lang == "yaml" || lang == "yml")
        
        switch lang {
        case "swift":
            kwList = ["func", "var", "let", "struct", "class", "enum", "protocol", "extension", "guard", "if", "else", "for", "in", "while", "return", "import", "public", "private", "fileprivate", "internal", "open", "static", "final", "override", "mutating", "async", "await", "throws", "try", "catch", "throw", "switch", "case", "default", "break", "continue", "where", "init", "deinit", "some", "any", "typealias", "nil", "true", "false", "self", "Self"]
        case "python", "py":
            kwList = ["def", "class", "import", "from", "as", "return", "if", "elif", "else", "for", "while", "in", "try", "except", "finally", "with", "raise", "pass", "break", "continue", "lambda", "yield", "async", "await", "assert", "global", "nonlocal", "True", "False", "None", "self"]
        case "rust", "rs":
            kwList = ["fn", "let", "mut", "struct", "enum", "impl", "trait", "pub", "use", "mod", "crate", "return", "if", "else", "match", "for", "in", "while", "loop", "break", "continue", "async", "await", "move", "ref", "type", "const", "static", "where", "true", "false", "unsafe"]
        case "javascript", "js", "typescript", "ts":
            kwList = ["function", "const", "let", "var", "class", "interface", "type", "import", "export", "from", "return", "if", "else", "for", "while", "switch", "case", "default", "break", "continue", "async", "await", "try", "catch", "throw", "new", "this", "typeof", "instanceof", "true", "false", "null", "undefined"]
        case "c", "cpp", "c++", "metal":
            kwList = ["kernel", "device", "constant", "threadgroup", "thread", "void", "int", "float", "double", "char", "bool", "struct", "class", "return", "if", "else", "for", "while", "switch", "case", "break", "continue", "template", "typename", "namespace", "include", "typedef"]
        default:
            kwList = ["func", "def", "fn", "function", "var", "let", "const", "struct", "class", "return", "import", "if", "else", "for", "while", "true", "false", "null", "nil", "None"]
        }
        
        let typeList: Set<String> = ["Int", "String", "Bool", "Float", "Double", "UInt32", "UInt64", "Int32", "Int64", "Data", "URL", "View", "Text", "Color", "Array", "Dictionary", "Set", "Optional", "Result", "Task", "MainActor", "MTLDevice", "MTLBuffer", "MTLCommandQueue"]
        
        var result = AttributedString()
        let lines = code.components(separatedBy: "\n")
        
        for (lineIdx, line) in lines.enumerated() {
            var lineAttr = AttributedString()
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let isCommentLine = isPythonLike ? trimmed.hasPrefix("#") : (trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*"))
            
            if isCommentLine {
                var commentAttr = AttributedString(line)
                commentAttr.foregroundColor = Color.secondary.opacity(0.8)
                lineAttr.append(commentAttr)
            } else {
                var idx = line.startIndex
                while idx < line.endIndex {
                    let remaining = String(line[idx...])
                    
                    // 1. Comment till end of line
                    if (!isPythonLike && remaining.hasPrefix("//")) || (isPythonLike && remaining.hasPrefix("#")) {
                        var cAttr = AttributedString(remaining)
                        cAttr.foregroundColor = Color.secondary.opacity(0.8)
                        lineAttr.append(cAttr)
                        break
                    }
                    
                    // 2. String literal ("..." or '...' or `...`)
                    if remaining.hasPrefix("\"") || remaining.hasPrefix("'") || remaining.hasPrefix("`") {
                        let quote = remaining.first!
                        var endIdx = line.index(after: idx)
                        var escaped = false
                        while endIdx < line.endIndex {
                            let ch = line[endIdx]
                            if escaped {
                                escaped = false
                            } else if ch == "\\" {
                                escaped = true
                            } else if ch == quote {
                                endIdx = line.index(after: endIdx)
                                break
                            }
                            endIdx = line.index(after: endIdx)
                        }
                        let strContent = String(line[idx..<endIdx])
                        var strAttr = AttributedString(strContent)
                        strAttr.foregroundColor = Color.green
                        lineAttr.append(strAttr)
                        idx = endIdx
                        continue
                    }
                    
                    // 3. Word / Identifier
                    if let wordMatch = remaining.range(of: #"^[a-zA-Z_][a-zA-Z0-9_]*"#, options: .regularExpression) {
                        let word = String(remaining[wordMatch])
                        var wAttr = AttributedString(word)
                        if kwList.contains(word) {
                            wAttr.foregroundColor = Color.purple
                            wAttr.font = .system(size: 12.5, weight: .bold, design: .monospaced)
                        } else if typeList.contains(word) {
                            wAttr.foregroundColor = Color.teal
                            wAttr.font = .system(size: 12.5, weight: .semibold, design: .monospaced)
                        }
                        lineAttr.append(wAttr)
                        idx = line.index(idx, offsetBy: word.count)
                        continue
                    }
                    
                    // 4. Number
                    if let numMatch = remaining.range(of: #"^(0x[0-9a-fA-F]+|\d+(\.\d+)?)"#, options: .regularExpression) {
                        let num = String(remaining[numMatch])
                        var nAttr = AttributedString(num)
                        nAttr.foregroundColor = Color.orange
                        lineAttr.append(nAttr)
                        idx = line.index(idx, offsetBy: num.count)
                        continue
                    }
                    
                    // 5. Plain character
                    let char = remaining.first!
                    let charAttr = AttributedString(String(char))
                    lineAttr.append(charAttr)
                    idx = line.index(after: idx)
                }
            }
            
            result.append(lineAttr)
            if lineIdx < lines.count - 1 {
                result.append(AttributedString("\n"))
            }
        }
        
        return result
    }
}

// MARK: - Dedicated Code Block Card with Header, Syntax Highlighting & Copy Button

struct CodeBlockCard: View {
    @EnvironmentObject private var zoomManager: AppZoomManager
    let language: String
    let code: String
    var isStreaming: Bool = false
    var isStreamingOffDisk: Bool = false
    @State private var isCopied: Bool = false

    /// Highlights at most this many characters. The highlighter walks the code
    /// line by line (measured ~23 ms per 11 KB), so a 90k-character block would
    /// cost most of a second on the main thread; past this size plain
    /// monospaced text is the better trade.
    private static let highlightCharacterLimit = 40_000

    /// Memoizes the chunk split and each chunk's highlighted form. Highlighting
    /// ~11 KB costs ~23 ms, and the old cache was filled in `onAppear` after the
    /// first body pass had already highlighted synchronously, so a block paid
    /// that cost twice the first time it was realized. Reference type so the
    /// body can refresh it in place without scheduling an update.
    private final class HighlightMemo {
        var language: String?
        var code: String?
        var chunks: [String] = []
        var highlighted: [AttributedString?] = []
    }

    @State private var highlightMemo = HighlightMemo()

    /// The block is drawn as bounded chunks: one giant text view is never
    /// painted on macOS once it outgrows the compositor's maximum layer.
    private var chunks: [String] {
        if highlightMemo.language == language, highlightMemo.code == code {
            return highlightMemo.chunks
        }
        let split = LongTextChunker.chunks(of: code)
        highlightMemo.language = language
        highlightMemo.code = code
        highlightMemo.chunks = split
        highlightMemo.highlighted = Array(repeating: nil, count: split.count)
        return split
    }

    private func text(forChunk index: Int) -> AttributedString {
        let split = chunks
        // A stale index can only mean the split shrank under this body pass.
        // Render nothing rather than the whole block, which would put an
        // unbounded text view back in the slot this chunking exists to avoid.
        guard split.indices.contains(index) else { return AttributedString() }
        if isStreaming || code.count > Self.highlightCharacterLimit {
            return AttributedString(split[index])
        }
        if let cached = highlightMemo.highlighted[index] { return cached }
        let value = NativeSyntaxHighlighter.highlight(code: split[index], language: language)
        highlightMemo.highlighted[index] = value
        return value
    }

    private var displayLanguage: String {
        let clean = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if clean.isEmpty { return "CODE" }
        return clean.uppercased()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header Bar
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.left.forwardslash.chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.purple)

                    Text(displayLanguage)
                        .font(.system(size: max(8.5, 11 * zoomManager.zoomScale), weight: .bold, design: .monospaced))
                        .foregroundColor(.secondary)

                    if isStreaming {
                        StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                    }
                }

                Spacer()

                Button(action: copyCode) {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 10))
                        Text(isCopied ? "Copied!" : "Copy")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .foregroundColor(isCopied ? .green : .secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.secondary.opacity(0.08))
                    .cornerRadius(4)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Color(NSColor.controlBackgroundColor))

            Divider()

            // Code Text Canvas with Native Syntax Highlighting (Cached)
            // A block that is still growing renders as plain monospaced text:
            // highlighting ~11 KB costs ~23 ms and used to rerun on every
            // streamed chunk, which is most of a frame budget spent on a block
            // whose tail is still changing. It highlights once it settles.
            ScrollView(.horizontal, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(chunks.indices, id: \.self) { index in
                        Text(text(forChunk: index))
                            .lineSpacing(3)
                    }
                }
                .font(.system(size: max(8.5, 12.5 * zoomManager.zoomScale), design: .monospaced))
                .textSelection(.enabled)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(NSColor.textBackgroundColor))
        }
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .padding(.vertical, 4)
    }

    private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        isCopied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            isCopied = false
        }
    }
}

// MARK: - Streaming Pace Indicator (Bunny vs. Tortoise)

struct StreamingPaceIndicatorView: View {
    let isOffDisk: Bool
    @State private var isFading: Bool = false

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: isOffDisk ? "tortoise.fill" : "hare.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(
                    isOffDisk
                        ? LinearGradient(colors: [.orange, .yellow], startPoint: .topLeading, endPoint: .bottomTrailing)
                        : LinearGradient(colors: [.purple, .indigo, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
                )
                .opacity(isFading ? 1.0 : 0.25)
                .scaleEffect(isFading ? 1.08 : 0.92)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 1)
        .background(
            (isOffDisk ? Color.orange : Color.purple)
                .opacity(isFading ? 0.12 : 0.03)
        )
        .cornerRadius(6)
        .onAppear {
            withAnimation(
                Animation.easeInOut(duration: 0.85)
                    .repeatForever(autoreverses: true)
            ) {
                isFading = true
            }
        }
        .help(isOffDisk ? "Streaming MoE experts off NVMe SSD (Paging)" : "Fast Streaming in Unified RAM (Dense / Resident)")
    }
}

// MARK: - Agent Working Indicator (Inline Tool-Call Activity)

/// A pulsing "Working..." pill shown inline in the chat while the agent streams
/// or executes a tool call, so long-running calls read as active work rather than
/// a stalled conversation.
struct AgentWorkingIndicatorView: View {
    var label: String = "Working..."
    @ObservedObject private var zoomManager = AppZoomManager.shared
    @State private var isFading: Bool = false

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "gearshape.2.fill")
                .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .semibold))
                .foregroundStyle(
                    LinearGradient(colors: [.purple, .indigo, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
                )
                .opacity(isFading ? 1.0 : 0.3)
                .scaleEffect(isFading ? 1.08 : 0.92)

            Text(label)
                .font(.system(size: max(9.5, 12 * zoomManager.zoomScale), weight: .medium))
                .foregroundColor(.secondary)
                .opacity(isFading ? 1.0 : 0.5)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.purple.opacity(isFading ? 0.12 : 0.03))
        .cornerRadius(6)
        .onAppear {
            withAnimation(
                Animation.easeInOut(duration: 0.85)
                    .repeatForever(autoreverses: true)
            ) {
                isFading = true
            }
        }
        .help("The agent is working on a tool call")
    }
}

// MARK: - Markdown Pre-processing Normalizer (Pre-compiled Regex Cache)

private enum MarkdownRegexCache {
    static let br = try? NSRegularExpression(pattern: #"<br\s*/?>"# , options: [.caseInsensitive])
    static let p = try? NSRegularExpression(pattern: #"</?p\b[^>]*>"#, options: [.caseInsensitive])
    static let div = try? NSRegularExpression(pattern: #"</?div\b[^>]*>"#, options: [.caseInsensitive])
    static let b = try? NSRegularExpression(pattern: #"<(?:b|strong)\b[^>]*>(.*?)</(?:b|strong)>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let i = try? NSRegularExpression(pattern: #"<(?:i|em)\b[^>]*>(.*?)</(?:i|em)>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let code = try? NSRegularExpression(pattern: #"<code\b[^>]*>(.*?)</code>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let li = try? NSRegularExpression(pattern: #"<li\b[^>]*>(.*?)(?:</li>|$)"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let ul = try? NSRegularExpression(pattern: #"</?(?:ul|ol)\b[^>]*>"#, options: [.caseInsensitive])
    static let span = try? NSRegularExpression(pattern: #"<span\b[^>]*>(.*?)</span>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let sup = try? NSRegularExpression(pattern: #"<sup>(.*?)</sup>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let sub = try? NSRegularExpression(pattern: #"<sub>(.*?)</sub>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let strayHtml = try? NSRegularExpression(pattern: #"</?[a-zA-Z][^>]*>"#, options: [])
    static let stuckHeading = try? NSRegularExpression(pattern: #"([^\n#])\s*(#{1,6}\s+)"#, options: [])
    static let headingGlue = try? NSRegularExpression(pattern: #"(#{1,6}\s+[^\n]+?)([a-z])([A-Z]{2,}|\b(?:BF16|FP8|FP16|FP32|INT8|GPU|CPU|NPU|TPU)\b)"#, options: [])
    static let inlineList = try? NSRegularExpression(pattern: #"([^\n])\s*(?:-\s+|•\s+|\*\s+)([A-Z0-9])"#, options: [])
    static let inlineNumList = try? NSRegularExpression(pattern: #"([.:;?!])\s*(\d{1,2}\.\s+[A-Z])"#, options: [])
    static let sentenceGlue = try? NSRegularExpression(pattern: #"([a-z0-9])\.([A-Z])"#, options: [])
    static let titleTable = try? NSRegularExpression(pattern: #"(?:^|\n)([^|\n#]+?)\s*\|\s*([A-Za-z0-9].*\|[^\n]*)"#, options: [])
    static let fixSepPipes = try? NSRegularExpression(pattern: #"(?:^|\n)(\|[|:\- ]+[-:])(?=\n|$)"#, options: [])
    static let multiSep = try? NSRegularExpression(pattern: #"(\|[|:\- ]+\|)\s*\n\s*(\|[|:\- ]+\|)"#, options: [])
    static let sectionHeader = try? NSRegularExpression(pattern: #"(?:^|\n)\s*--\s*([^-\n]+?)\s*--\s*\|?"#, options: [.anchorsMatchLines])
    static let hrHeader = try? NSRegularExpression(pattern: #"---+[\t ]*(#{1,6}\s*)"#, options: [])
    static let stuckBullet = try? NSRegularExpression(pattern: #"([^\n])(•\s+[A-Za-z0-9])"#, options: [])
    static let multiNl = try? NSRegularExpression(pattern: #"\n{3,}"#, options: [])
}

/// Normalizes model-emitted markdown for block parsing.
///
/// `repairs` controls the "repair" passes that fix malformed formatting
/// (a heading glued to a sentence, a bullet stuck to preceding text, and so
/// on). They are six more whole-text regex scans, and while a turn streams the
/// growing answer is re-normalized on every UI update: at 40 KB one parse
/// measured ~25 ms, of which the repairs are the bulk. A streaming render
/// passes `false` and still honours headings, lists, code fences, and tables
/// that are already well formed; the settled message is parsed once with
/// `true`, so the final rendering is identical.
public func normalizeMarkdownText(_ raw: String, repairs: Bool = true) -> String {
    normalizeMarkdownText(raw, repairs: repairs, triggerGuards: true)
}

/// `triggerGuards` exists for the guard-equivalence tests: `false` runs every
/// regex pass unconditionally, so a test can assert that skipping a pass on a
/// missing trigger character cannot change the output. Production only ever
/// takes the guarded path.
func normalizeMarkdownText(_ raw: String, repairs: Bool, triggerGuards: Bool) -> String {
    let runAll = !triggerGuards
    var text = raw
    text = text.replacingOccurrences(of: "\r\n", with: "\n")
    text = text.replacingOccurrences(of: "\r", with: "\n")
    text = text.replacingOccurrences(of: "\\n", with: "\n")

    // Every repair below is a whole-text regex pass, and a streaming turn
    // re-normalizes the growing answer on every UI update, so a long answer
    // used to pay for ~24 full scans per frame (measured ~29 ms per 40 KB).
    // Each pass is skipped when its trigger character cannot occur, which is
    // behavior-preserving — the pattern cannot match without it — and drops
    // typical model output to a handful of scans.
    let mightContainHTML = runAll || text.contains("<")
    if mightContainHTML {
        // 1. Convert HTML Line breaks & Paragraphs to newlines
        if let regexBr = MarkdownRegexCache.br {
            text = regexBr.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n")
        }
        if let regexP = MarkdownRegexCache.p {
            text = regexP.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n")
        }
        if let regexDiv = MarkdownRegexCache.div {
            text = regexDiv.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n")
        }

        // 2. Convert HTML formatting tags to Markdown
        if let regexB = MarkdownRegexCache.b {
            text = regexB.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "**$1**")
        }
        if let regexI = MarkdownRegexCache.i {
            text = regexI.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "*$1*")
        }
        if let regexCode = MarkdownRegexCache.code {
            text = regexCode.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "`$1`")
        }

        // 3. Convert HTML Lists
        if let regexLi = MarkdownRegexCache.li {
            text = regexLi.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n- $1\n")
        }
        if let regexUl = MarkdownRegexCache.ul {
            text = regexUl.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n")
        }

        // 4. Strip span and convert sup/sub tags
        if let regexSpan = MarkdownRegexCache.span {
            text = regexSpan.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1")
        }
        if let regexSup = MarkdownRegexCache.sup {
            text = regexSup.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "^($1)")
        }
        if let regexSub = MarkdownRegexCache.sub {
            text = regexSub.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "_($1)")
        }
        if let regexStrayHtml = MarkdownRegexCache.strayHtml {
            text = regexStrayHtml.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "")
        }
    }

    // 5. Clean double pipes || -> |
    while text.contains("||") {
        text = text.replacingOccurrences(of: "||", with: "|")
    }

    if repairs {
        // 6. Fix heading embedded in paragraph or missing space: "text. #### 1. Heading" -> "text.\n\n#### 1. Heading"
        if (runAll || text.contains("#")), let regexStuckHeading = MarkdownRegexCache.stuckHeading {
            text = regexStuckHeading.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n\n$2")
        }

        // 7. Fix heading title glued to paragraph text
        if (runAll || text.contains("#")), let regexHeadingGlue = MarkdownRegexCache.headingGlue {
            text = regexHeadingGlue.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1$2\n\n$3")
        }

        // 8. Fix inline list glued inside paragraph: "formats: - Total 16 bits" -> "formats:\n\n- Total 16 bits"
        if runAll || text.contains("-") || text.contains("•") || text.contains("*"),
           let regexInlineList = MarkdownRegexCache.inlineList {
            text = regexInlineList.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n\n- $2")
        }

        // 9. Fix inline numbered list glued inside paragraph: "computing. 1. Technical" -> "computing.\n\n1. Technical"
        if (runAll || text.contains(".")), let regexInlineNumList = MarkdownRegexCache.inlineNumList {
            text = regexInlineNumList.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n\n$2")
        }

        // 10. Fix sentence glued to next sentence without space: "quantization.This" -> "quantization. This"
        if (runAll || text.contains(".")), let regexSentenceGlue = MarkdownRegexCache.sentenceGlue {
            text = regexSentenceGlue.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1. $2")
        }

        // 11. Fix title stuck before table header on same line
        if (runAll || text.contains("|")), let regexTitleTable = MarkdownRegexCache.titleTable {
            text = regexTitleTable.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n### $1\n| $2")
        }

        // 12. Fix unclosed table separator line (missing trailing pipe)
        if (runAll || text.contains("|")), let regexFixSepPipes = MarkdownRegexCache.fixSepPipes {
            text = regexFixSepPipes.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n$1|")
        }

        // 13. Fix multiple table separator lines
        if (runAll || text.contains("|")), let regexMultiSep = MarkdownRegexCache.multiSep {
            text = regexMultiSep.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1")
        }

        // 14. Fix table section headers like "-- Section Name -- |"
        if runAll || text.contains("|") || text.contains("--"), let regexSectionHeader = MarkdownRegexCache.sectionHeader {
            text = regexSectionHeader.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n#### $1\n")
        }

        // 15. Fix concatenated horizontal rules + headers: "---##" -> "\n\n---\n\n## "
        if (runAll || text.contains("---")), let regexHrHeader = MarkdownRegexCache.hrHeader {
            text = regexHrHeader.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n---\n\n$1")
        }

        // 16. Fix bullet points stuck to preceding text: "text• Bullet" -> "text\n• Bullet"
        if (runAll || text.contains("•")), let regexStuckBullet = MarkdownRegexCache.stuckBullet {
            text = regexStuckBullet.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n$2")
        }
    }

    // 17. Collapse 3+ newlines to 2
    if (runAll || text.contains("\n\n\n")), let regexMultiNl = MarkdownRegexCache.multiNl {
        text = regexMultiNl.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n")
    }

    return text
}

// MARK: - Table Line Parser

private func parseMarkdownTable(lines: [String], startIndex: Int) -> (table: MarkdownTableData, nextIndex: Int)? {
    guard startIndex + 1 < lines.count else { return nil }
    let headerLine = lines[startIndex].trimmingCharacters(in: .whitespaces)
    let separatorLine = lines[startIndex + 1].trimmingCharacters(in: .whitespaces)

    guard headerLine.contains("|") else { return nil }

    let separatorChars = CharacterSet(charactersIn: "|- :—–â")
    guard separatorLine.contains("|") && separatorLine.contains("-") && separatorLine.unicodeScalars.allSatisfy({ separatorChars.contains($0) }) else {
        return nil
    }

    func splitCells(_ line: String) -> [String] {
        var trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("|") { trimmed.removeFirst() }
        if trimmed.hasSuffix("|") { trimmed.removeLast() }
        return trimmed.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    let headers = splitCells(headerLine)
    guard headers.count >= 2 else { return nil }
    let separatorCells = splitCells(separatorLine)

    var alignments: [TextAlignmentType] = []
    for colIdx in 0..<headers.count {
        if colIdx < separatorCells.count {
            let cell = separatorCells[colIdx]
            let hasLeadingColon = cell.hasPrefix(":")
            let hasTrailingColon = cell.hasSuffix(":")
            if hasLeadingColon && hasTrailingColon {
                alignments.append(.center)
            } else if hasTrailingColon {
                alignments.append(.trailing)
            } else {
                alignments.append(.leading)
            }
        } else {
            alignments.append(.leading)
        }
    }

    var rows: [[String]] = []
    var curIdx = startIndex + 2
    while curIdx < lines.count {
        let line = lines[curIdx].trimmingCharacters(in: .whitespaces)
        if line.isEmpty || !line.contains("|") {
            break
        }
        var cells = splitCells(line)
        if cells.count < headers.count {
            cells.append(contentsOf: Array(repeating: "", count: headers.count - cells.count))
        } else if cells.count > headers.count {
            cells = Array(cells.prefix(headers.count))
        }
        rows.append(cells)
        curIdx += 1
    }

    return (MarkdownTableData(
        headers: headers.map { EmojiShortcodeResolver.resolve($0) },
        alignments: alignments,
        rows: rows.map { $0.map { EmojiShortcodeResolver.resolve($0) } }
    ), curIdx)
}

// MARK: - Full Markdown Block Parsing Engine

// MARK: - Emoji Shortcode Resolution

/// Resolves GitHub-style emoji shortcodes (`:tada:`) to native Unicode emoji (🎉).
/// Native emoji in message text already render via Text; this covers the shortcode
/// form that models frequently emit. Never applied inside code blocks.
public enum EmojiShortcodeResolver {
    /// Precomputed regex matching `:x:` / `:word:` / `:word_word2:` / `:+1:` style tokens.
    private static let shortcodeRegex = try? NSRegularExpression(
        pattern: ":[a-zA-Z0-9_+\\-]{1,40}:",
        options: []
    )

    /// Common shortcodes models actually emit. Kept deliberately focused; unknown
    /// shortcodes are left as literal text rather than guessed.
    private static let map: [String: String] = [
        ":+1:": "👍", ":-1:": "👎", ":100:": "💯", ":tada:": "🎉",
        ":rocket:": "🚀", ":fire:": "🔥", ":sparkles:": "✨", ":star:": "⭐",
        ":star2:": "🌟", ":warning:": "⚠️", ":bulb:": "💡", ":zap:": "⚡",
        ":bug:": "🐛", ":wrench:": "🔧", ":hammer:": "🔨", ":gear:": "⚙️",
        ":mag:": "🔍", ":mag_right:": "🔎", ":eyes:": "👀", ":lock:": "🔒",
        ":unlock:": "🔓", ":key:": "🔑", ":shield:": "🛡️", ":link:": "🔗",
        ":package:": "📦", ":books:": "📚", ":book:": "📖", ":memo:": "📝",
        ":pencil:": "✏️", ":pencil2:": "✏️", ":page_facing_up:": "📄",
        ":clipboard:": "📋", ":pushpin:": "📌", ":paperclip:": "📎",
        ":calendar:": "📅", ":clock:": "🕐", ":hourglass:": "⏳",
        ":dart:": "🎯", ":goal_net:": "🥅", ":trophy:": "🏆", ":medal:": "🏅",
        ":checkered_flag:": "🏁", ":white_check_mark:": "✅", ":check:": "☑️",
        ":x:": "❌", ":no_entry:": "⛔", ":prohibited:": "🚫", ":stop_sign:": "🛑",
        ":heavy_plus_sign:": "➕", ":heavy_minus_sign:": "➖",
        ":heavy_division_sign:": "➗", ":heavy_multiplication_x:": "✖️",
        ":question:": "❓", ":exclamation:": "❗", ":grey_question:": "❔",
        ":information_source:": "ℹ️", ":soon:": "🔜", ":ok:": "🆗",
        ":new:": "🆕", ":up:": "🆙", ":top:": "🔝", ":cool:": "🆒",
        ":arrow_up:": "⬆️", ":arrow_down:": "⬇️", ":arrow_left:": "⬅️",
        ":arrow_right:": "➡️", ":arrows_counterclockwise:": "🔄",
        ":repeat:": "🔁", ":recycle:": "♻️", ":fast_forward:": "⏩",
        ":rewind:": "⏪", ":play_or_pause_button:": "⏯️",
        ":phone:": "📱", ":computer:": "💻", ":desktop_computer:": "🖥️",
        ":keyboard:": "⌨️", ":floppy_disk:": "💾", ":cd:": "💿",
        ":envelope:": "✉️", ":inbox_tray:": "📥", ":outbox_tray:": "📤",
        ":thought_balloon:": "💭", ":speech_balloon:": "💬", ":smile:": "😄",
        ":grin:": "😁", ":joy:": "😂", ":wink:": "😉", ":thinking:": "🤔",
        ":sunglasses:": "😎", ":slightly_smiling_face:": "🙂",
        ":neutral_face:": "😐", ":confused:": "😕", ":cry:": "😢",
        ":scream:": "😱", ":heart:": "❤️", ":blue_heart:": "💙",
        ":green_heart:": "💚", ":purple_heart:": "💜", ":yellow_heart:": "💛",
        ":broken_heart:": "💔", ":seedling:": "🌱", ":herb:": "🌿",
        ":tree:": "🌳", ":cactus:": "🌵", ":sun:": "☀️", ":moon:": "🌙",
        ":rainbow:": "🌈", ":snowflake:": "❄️", ":umbrella:": "☔",
        ":coffee:": "☕", ":tea:": "🍵", ":pizza:": "🍕", ":cake:": "🍰",
        ":beer:": "🍺", ":beers:": "🍻", ":clap:": "👏", ":wave:": "👋",
        ":pray:": "🙏", ":muscle:": "💪", ":point_up:": "☝️",
        ":point_right:": "👉", ":point_left:": "👈", ":point_down:": "👇",
        ":raised_hands:": "🙌", ":handshake:": "🤝", ":ok_hand:": "👌",
        ":v:": "✌️", ":crossed_fingers:": "🤞", ":brain:": "🧠",
        ":robot:": "🤖", ":ghost:": "👻", ":skull:": "💀", ":alien:": "👽",
        ":crown:": "👑", ":gem:": "💎", ":moneybag:": "💰", ":chart_with_upwards_trend:": "📈",
        ":chart_with_downwards_trend:": "📉", ":bar_chart:": "📊",
        ":clipboard_copy:": "📋", ":test_tube:": "🧪", ":microscope:": "🔬",
        ":telescope:": "🔭", ":satellite:": "📡", ":battery:": "🔋",
        ":plug:": "🔌", ":bell:": "🔔", ":mega:": "📣", ":loudspeaker:": "📢",
        ":mute:": "🔇", ":speaker:": "🔊", ":musical_note:": "🎵",
        ":art:": "🎨", ":clapper:": "🎬", ":camera:": "📷", ":video_camera:": "📹",
        ":film_strip:": "🎞️", ":balloon:": "🎈", ":gift:": "🎁",
        ":cupid:": "💘", ":sparkling_heart:": "💖", ":snowman:": "⛄",
        ":zap_face:": "⚡", ":dizzy:": "💫", ":boom:": "💥", ":collision:": "💥",
        ":anger:": "💢", ":bomb:": "💣", ":poop:": "💩", ":hankey:": "💩",
        ":sunny:": "☀️", ":tornado:": "🌪️", ":fog:": "🌫️",
        ":airplane:": "✈️", ":car:": "🚗", ":ship:": "🚢", ":house:": "🏠",
        ":office:": "🏢", ":hospital:": "🏥", ":school:": "🏫",
        ":world_map:": "🗺️", ":globe_with_meridians:": "🌐", ":earth_americas:": "🌎",
        ":earth_asia:": "🌏", ":earth_africa:": "🌍", ":compass:": "🧭",
        ":apple:": "🍎", ":green_apple:": "🍏", ":lemon:": "🍋", ":banana:": "🍌",
        ":watermelon:": "🍉", ":grapes:": "🍇", ":strawberry:": "🍓",
        ":peach:": "🍑", ":cherry_blossom:": "🌸", ":rose:": "🌹",
        ":hibiscus:": "🌺", ":sunflower:": "🌻", ":tulip:": "🌷",
        ":four_leaf_clover:": "🍀", ":maple_leaf:": "🍁", ":leaves:": "🍃"
    ]

    /// Replaces `:shortcode:` tokens with their emoji equivalents. Text inside
    /// backticks is left untouched so inline code is never mangled.
    public static func resolve(_ text: String) -> String {
        guard text.contains(":") else { return text }

        // Split on inline code spans so shortcodes inside `...` are preserved.
        let segments = splitPreservingInlineCode(text)
        guard segments.count > 1 || !segments[0].isCode else {
            return replaceOutside(text)
        }
        var result = ""
        for segment in segments {
            result += segment.isCode ? segment.text : replaceOutside(segment.text)
        }
        return result
    }

    private static func replaceOutside(_ text: String) -> String {
        guard let regex = shortcodeRegex else { return text }
        let ns = text as NSString
        let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return text }
        var rebuilt = ""
        var lastEnd = 0
        for match in matches {
            if match.range.location > lastEnd {
                rebuilt += ns.substring(with: NSRange(location: lastEnd, length: match.range.location - lastEnd))
            }
            let raw = ns.substring(with: match.range).lowercased()
            rebuilt += map[raw] ?? ns.substring(with: match.range)
            lastEnd = match.range.location + match.range.length
        }
        if lastEnd < ns.length {
            rebuilt += ns.substring(from: lastEnd)
        }
        return rebuilt
    }

    private struct CodeSegment {
        let text: String
        let isCode: Bool
    }

    private static func splitPreservingInlineCode(_ text: String) -> [CodeSegment] {
        var segments: [CodeSegment] = []
        var current = ""
        var inCode = false
        var i = text.startIndex
        while i < text.endIndex {
            let ch = text[i]
            if ch == "`" {
                if !current.isEmpty { segments.append(CodeSegment(text: current, isCode: inCode)); current = "" }
                inCode.toggle()
                current.append(ch)
            } else {
                current.append(ch)
            }
            i = text.index(after: i)
        }
        if !current.isEmpty { segments.append(CodeSegment(text: current, isCode: inCode)) }
        return segments
    }
}

/// Parses an `N. text` numbered-list line. Equivalent to the
/// `^\d+\.\s+` regular expression it replaced, without compiling a
/// pattern for every line of the message.
private func numberedListItem(_ line: String) -> (number: Int?, text: String)? {
    var index = line.startIndex
    while index < line.endIndex, line[index].isASCII, line[index].isNumber {
        index = line.index(after: index)
    }
    guard index > line.startIndex, index < line.endIndex, line[index] == "." else { return nil }
    var rest = line.index(after: index)
    guard rest < line.endIndex, line[rest].isWhitespace else { return nil }
    while rest < line.endIndex, line[rest].isWhitespace {
        rest = line.index(after: rest)
    }
    return (
        number: Int(line[line.startIndex..<index]),
        text: String(line[rest...]).trimmingCharacters(in: .whitespaces)
    )
}

public func parseMarkdownBlocks(_ raw: String, repairs: Bool = true) -> [MarkdownBlock] {
    let normalized = normalizeMarkdownText(raw, repairs: repairs)
    var blocks: [MarkdownBlock] = []
    let lines = normalized.components(separatedBy: "\n")

    var inCodeBlock = false
    var codeLang = ""
    var codeLines: [String] = []

    var paragraphLines: [String] = []

    func flushParagraph() {
        if !paragraphLines.isEmpty {
            let combined = EmojiShortcodeResolver.resolve(paragraphLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
            if !combined.isEmpty {
                blocks.append(.paragraph(text: combined))
            }
            paragraphLines.removeAll()
        }
    }

    var i = 0
    while i < lines.count {
        let line = lines[i]
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        // 1. Code Block Fence
        if trimmed.hasPrefix("```") {
            if inCodeBlock {
                blocks.append(.codeBlock(language: codeLang, code: codeLines.joined(separator: "\n")))
                inCodeBlock = false
                codeLang = ""
                codeLines.removeAll()
            } else {
                flushParagraph()
                inCodeBlock = true
                codeLang = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespacesAndNewlines)
                codeLines.removeAll()
            }
            i += 1
            continue
        }

        if inCodeBlock {
            codeLines.append(line)
            i += 1
            continue
        }

        // 2. Table detection
        if !inCodeBlock && trimmed.contains("|") {
            if let tableResult = parseMarkdownTable(lines: lines, startIndex: i) {
                flushParagraph()
                blocks.append(.table(tableResult.table))
                i = tableResult.nextIndex
                continue
            }
        }

        // 3. Headings (#, ##, ###, etc.)
        if trimmed.hasPrefix("#") {
            flushParagraph()
            let hashCount = trimmed.prefix(while: { $0 == "#" }).count
            let headerText = EmojiShortcodeResolver.resolve(String(trimmed.dropFirst(hashCount)).trimmingCharacters(in: .whitespaces))
            blocks.append(.heading(level: hashCount, text: headerText))
            i += 1
            continue
        }

        // 4. Blockquotes
        if trimmed.hasPrefix("> ") || trimmed == ">" {
            flushParagraph()
            var quoteLines: [String] = []
            while i < lines.count {
                let qLine = lines[i].trimmingCharacters(in: .whitespaces)
                if qLine.hasPrefix("> ") {
                    quoteLines.append(String(qLine.dropFirst(2)))
                    i += 1
                } else if qLine == ">" {
                    quoteLines.append("")
                    i += 1
                } else {
                    break
                }
            }
            blocks.append(.blockquote(text: EmojiShortcodeResolver.resolve(quoteLines.joined(separator: "\n"))))
            continue
        }

        // 5. Horizontal Rule
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            flushParagraph()
            blocks.append(.horizontalRule)
            i += 1
            continue
        }

        // 6. Bullet list item
        if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("+ ") || trimmed.hasPrefix("• ") {
            flushParagraph()
            let itemText: String
            if trimmed.hasPrefix("• ") {
                itemText = EmojiShortcodeResolver.resolve(String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces))
            } else {
                itemText = EmojiShortcodeResolver.resolve(String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces))
            }
            blocks.append(.listItem(number: nil, text: itemText))
            i += 1
            continue
        }

        // 7. Numbered list item. Parsed by hand: a regular expression compiled
        // per line dominated this parser's cost on long messages, which mattered
        // once a streamed answer could run to tens of thousands of tokens.
        if let numbered = numberedListItem(trimmed) {
            flushParagraph()
            blocks.append(.listItem(
                number: numbered.number,
                text: EmojiShortcodeResolver.resolve(numbered.text)
            ))
            i += 1
            continue
        }

        // 8. Empty line
        if trimmed.isEmpty {
            flushParagraph()
            i += 1
            continue
        }

        paragraphLines.append(line)
        i += 1
    }

    if inCodeBlock {
        blocks.append(.codeBlock(language: codeLang, code: codeLines.joined(separator: "\n")))
    } else {
        flushParagraph()
    }

    return blocks
}

// MARK: - Tool Calls Visual Cards (Agent Mode)
struct ToolCallListView: View {
    let toolCalls: [ToolCallRecord]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(toolCalls) { call in
                ToolCallCardView(call: call)
            }
        }
        .padding(.vertical, 2)
    }
}

struct ToolCallCardView: View {
    let call: ToolCallRecord
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
        case "codebase_search": return "sparkle.magnifyingglass"
        case "spawn_subagent": return "person.2.badge.gearshape"
        case "get_subagent_status": return "clock.arrow.2.circlepath"
        case "send_subagent_message": return "bubble.left.and.bubble.right.fill"
        case "list_subagents": return "list.bullet.rectangle"
        case "git_status": return "point.topleft.down.curvedto.point.bottomright.up"
        case "git_diff": return "plus.forwardslash.minus"
        case "git_commit": return "arrow.triangle.branch"
        case "find_symbol_definition": return "character.textbox"
        case "find_references": return "arrow.triangle.swap"
        case "lint_diagnostics": return "stethoscope"
        case "complete": return "checkmark.seal.fill"
        default: return "wrench.and.screwdriver.fill"
        }
    }

    private var primarySummary: String {
        switch call.name {
        case "shell_run":
            return call.arguments["command"] ?? call.rawArguments
        case "file_read", "file_write", "file_edit", "lint_diagnostics":
            if let path = call.arguments["path"] {
                return (path as NSString).lastPathComponent
            }
            return call.rawArguments
        case "find_files":
            return call.arguments["pattern"] ?? call.rawArguments
        case "grep_search":
            return "\"\(call.arguments["query"] ?? "")\""
        case "web_search":
            return "\"\(call.arguments["query"] ?? "")\""
        case "web_fetch":
            if let url = call.arguments["url"] {
                return (URL(string: url)?.host ?? url)
            }
            return call.rawArguments
        case "codebase_search":
            return "\"\(call.arguments["query"] ?? "")\""
        case "spawn_subagent":
            let role = call.arguments["role"] ?? "Subagent"
            let task = call.arguments["task_description"] ?? ""
            return "\(role): \(task.prefix(35))\(task.count > 35 ? "..." : "")"
        case "get_subagent_status", "send_subagent_message":
            return call.arguments["subagent_id"]?.prefix(8).description ?? call.rawArguments
        case "list_subagents":
            return "Filter: \(call.arguments["status_filter"] ?? "all")"
        case "git_status":
            return call.arguments["path"] ?? "working tree"
        case "git_diff":
            if let path = call.arguments["path"] {
                return (path as NSString).lastPathComponent
            }
            return (call.arguments["staged"] == "true") ? "staged changes" : "working tree"
        case "git_commit":
            let msg = call.arguments["message"] ?? "Commit"
            return "\"\(msg.prefix(30))\(msg.count > 30 ? "..." : "")\""
        case "find_symbol_definition", "find_references":
            return "\"\(call.arguments["symbol_name"] ?? call.rawArguments)\""
        case "complete":
            return call.arguments["summary"] ?? "Completed"
        default:
            return call.rawArguments
        }
    }

    private var hasSyntaxDiagnostics: Bool {
        guard let out = call.output else { return false }
        return out.contains("syntax_errors_detected") || out.contains("written_with_syntax_errors") || (call.name == "lint_diagnostics" && out.contains("\"has_errors\": true"))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.15)) {
                    isExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    // Tool Icon
                    Image(systemName: iconName)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(statusColor)

                    // Tool Name Badge
                    Text(call.name)
                        .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                        .foregroundColor(.primary)

                    // Primary Argument / Target
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
                        if hasSyntaxDiagnostics {
                            HStack(spacing: 4) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.system(size: 11))
                                    .foregroundColor(.yellow)
                                Text("Needs Healing")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundColor(.yellow)
                            }
                        } else {
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
                        .stroke(call.status == .running ? Color.indigo.opacity(0.3) : Color.primary.opacity(0.06), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    // Full Arguments
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

                    if call.name == "spawn_subagent" {
                        Button(action: {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                SubagentManager.shared.isDrawerOpen = true
                            }
                        }) {
                            HStack(spacing: 5) {
                                Image(systemName: "square.2.layers.3d")
                                    .font(.system(size: 10, weight: .semibold))
                                Text("Inspect Subagent in Task Manager Drawer")
                                    .font(.system(size: 10.5, weight: .medium))
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4)
                            .background(Color.purple.opacity(0.12))
                            .foregroundColor(.purple)
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                    }

                    // Stdout / Output
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
                                    .foregroundColor(.primary.opacity(0.9))
                                    .lineSpacing(2)
                                    .textSelection(.enabled)
                            }
                            .frame(maxHeight: 200)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.black.opacity(0.2))
                            .cornerRadius(6)
                        }
                    }

                    // Stderr / Error
                    if let err = call.error, !err.isEmpty {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Error Details:")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(.red)
                            Text(err)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.red.opacity(0.9))
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.red.opacity(0.08))
                                .cornerRadius(6)
                                .textSelection(.enabled)
                        }
                    }
                }
                .padding(.leading, 8)
            }
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
}

