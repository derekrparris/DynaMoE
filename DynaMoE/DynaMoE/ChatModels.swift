//
//  ChatModels.swift
//  DynaMoE
//

import Foundation

/// Graded reasoning effort for chat templates that expose a level (Spark
/// X2.6's `reasoning_effort`). Thinking off is represented by `nil` at the
/// call sites; models with only an on/off switch ignore the value entirely.
nonisolated public enum ThinkingEffort: String, Codable, CaseIterable, Equatable, Sendable {
    case low
    case high

    /// Parses arbitrary stored text case-insensitively. Returns nil when the
    /// text is absent or not a supported level.
    public static func parse(_ raw: String?) -> ThinkingEffort? {
        guard let raw else { return nil }
        return ThinkingEffort(rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    /// Clamps stale or hand-edited stored text to a supported level, falling
    /// back to `.high` (the value Spark X2.6's template defaults to).
    public static func clamped(_ raw: String?) -> ThinkingEffort {
        parse(raw) ?? .high
    }
}

nonisolated public enum MessageRole: String, Codable, Equatable, Sendable {
    case user
    case assistant
    case system
}

nonisolated public enum ToolExecutionStatus: String, Codable, Equatable, Sendable {
    case running
    case success
    case error
    case awaitingApproval
    case rejected
    /// The agent mode refused the call before it ran (plan mode withholding a
    /// write tool). Distinct from `.rejected`, which is the USER declining a
    /// confirmation: a withheld call is neither a failure the model caused nor a
    /// success. The loop guards read it through `ToolCallOutcome.withheld`, and
    /// the timeline renders it as a mode refusal rather than an error.
    case withheld
}

nonisolated public struct ToolCallRecord: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var name: String
    public var arguments: [String: String]
    public var rawArguments: String
    public var status: ToolExecutionStatus
    public var output: String?
    public var error: String?
    public var executionDurationSeconds: Double?
    public var timestamp: Date
    /// True for a no-op "gesture" call the harness skipped instead of running. The
    /// record is kept so reconstructed history has a matching result for the call
    /// the assistant emitted, but it is hidden from the tool timeline UI.
    public var isGesture: Bool

    public init(
        id: UUID = UUID(),
        name: String,
        arguments: [String: String] = [:],
        rawArguments: String = "",
        status: ToolExecutionStatus = .running,
        output: String? = nil,
        error: String? = nil,
        executionDurationSeconds: Double? = nil,
        timestamp: Date = Date(),
        isGesture: Bool = false
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.rawArguments = rawArguments
        self.status = status
        self.output = output
        self.error = error
        self.executionDurationSeconds = executionDurationSeconds
        self.timestamp = timestamp
        self.isGesture = isGesture
    }

    // MARK: - Codable

    private enum CodingKeys: String, CodingKey {
        case id, name, arguments, rawArguments, status, output, error
        case executionDurationSeconds, timestamp, isGesture
        /// Written beside a withheld `status` and read back as the real meaning.
        /// See `encode(to:)`.
        case withheldInPlanMode
    }

    /// Decodes a record written by this build or any earlier one.
    ///
    /// `withheldInPlanMode` is what restores `.withheld`: the status itself is
    /// written as `.rejected` so that a build predating plan mode (which has no
    /// `.withheld` case and would fail on it) can still decode the conversation
    /// instead of losing it. A record written directly with `.withheld` — an
    /// intermediate build, or a hand-edited file — is accepted too.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        arguments = try container.decode([String: String].self, forKey: .arguments)
        rawArguments = try container.decode(String.self, forKey: .rawArguments)
        output = try container.decodeIfPresent(String.self, forKey: .output)
        error = try container.decodeIfPresent(String.self, forKey: .error)
        executionDurationSeconds = try container.decodeIfPresent(
            Double.self,
            forKey: .executionDurationSeconds
        )
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        isGesture = try container.decodeIfPresent(Bool.self, forKey: .isGesture) ?? false
        let recorded = try container.decode(ToolExecutionStatus.self, forKey: .status)
        let flagged = try container.decodeIfPresent(Bool.self, forKey: .withheldInPlanMode) ?? false
        status = (flagged || recorded == .withheld) ? .withheld : recorded
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(arguments, forKey: .arguments)
        try container.encode(rawArguments, forKey: .rawArguments)
        try container.encodeIfPresent(output, forKey: .output)
        try container.encodeIfPresent(error, forKey: .error)
        try container.encodeIfPresent(executionDurationSeconds, forKey: .executionDurationSeconds)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(isGesture, forKey: .isGesture)
        // `.withheld` did not exist before plan mode, and an unknown raw value
        // fails the decode of the WHOLE record and with it the session file. It
        // is therefore persisted as the closest status an older build knows, with
        // the real meaning beside it for this build to read back.
        try container.encode(status == .withheld ? .rejected : status, forKey: .status)
        if status == .withheld {
            try container.encode(true, forKey: .withheldInPlanMode)
        }
    }
}

/// One image a user attached to a message. The pixels live in the asset store
/// (`~/Library/Application Support/DynaMoE/sessions/assets/`) and only this
/// metadata rides in the session JSON: base64 in the conversation file would
/// rewrite megabytes on every streaming save and defeat the store's digest
/// write-skip. `relativePath` is self-describing (`<sessionId>/<id>.<ext>`) so
/// resolving an attachment never needs the session it came from.
nonisolated public struct ImageAttachmentRef: Identifiable, Codable, Equatable, Hashable, Sendable {
    public var id: UUID
    public var fileName: String
    public var pixelWidth: Int
    public var pixelHeight: Int
    public var byteCount: Int
    public var sha256: String
    public var relativePath: String
    /// Wire MIME type of the stored bytes ("image/png" or "image/jpeg").
    public var mimeType: String

    public init(
        id: UUID = UUID(),
        fileName: String,
        pixelWidth: Int,
        pixelHeight: Int,
        byteCount: Int,
        sha256: String,
        relativePath: String,
        mimeType: String
    ) {
        self.id = id
        self.fileName = fileName
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
        self.byteCount = byteCount
        self.sha256 = sha256
        self.relativePath = relativePath
        self.mimeType = mimeType
    }

    /// Metadata-only rendering for prompts, request logs, and Console lines.
    /// Image bytes must never reach a log row or a unified-log line.
    public var logDescription: String {
        "[image: \(fileName) \(pixelWidth)×\(pixelHeight) \(formattedByteCount)]"
    }

    public var formattedByteCount: String {
        if byteCount >= 1_048_576 {
            return String(format: "%.1f MB", Double(byteCount) / 1_048_576)
        }
        if byteCount >= 1_024 {
            return String(format: "%.0f KB", Double(byteCount) / 1_024)
        }
        return "\(byteCount) B"
    }
}

nonisolated public struct ChatMessage: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var role: MessageRole
    public var content: String
    public var thinkingContent: String?
    public var isThinking: Bool
    public var timestamp: Date
    public var tokenCount: Int
    public var tokensPerSec: Double
    public var timeToFirstTokenSeconds: Double?
    public var thinkingTimeSeconds: Double?
    public var prefillStatus: String?
    /// Runtime-only marker for messages that must never survive a relaunch:
    /// the compaction progress bubble streams an uncommitted summary into the
    /// persisted `messages` array, and a save that caught it mid-pass would
    /// otherwise restore a partial summary as an ordinary settled system
    /// message. `ChatSessionStore.normalizedForRestore` drops these on load.
    public var isTransient: Bool?
    /// True on the one visible digest card `/compact` leaves at the head of a
    /// compacted conversation. The row is a `.system` message, so every prompt
    /// and transcript builder skips it (the same digests already ride in
    /// `compactionContextBlock`), but it is not a notice: ChatDetailView renders
    /// it as the collapsible summary card rather than as plain text.
    public var isCompactionSummary: Bool?
    public var toolCalls: [ToolCallRecord]?
    /// Images the user attached to this message, in display order. Only user
    /// turns carry them; nil for every turn that predates image input.
    public var attachments: [ImageAttachmentRef]?
    public var jetSpecTau: Double?
    public var jetSpecDraftAccepted: Int?

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        thinkingContent: String? = nil,
        isThinking: Bool = false,
        timestamp: Date = Date(),
        tokenCount: Int = 0,
        tokensPerSec: Double = 0.0,
        timeToFirstTokenSeconds: Double? = nil,
        thinkingTimeSeconds: Double? = nil,
        prefillStatus: String? = nil,
        isTransient: Bool? = nil,
        isCompactionSummary: Bool? = nil,
        toolCalls: [ToolCallRecord]? = nil,
        attachments: [ImageAttachmentRef]? = nil,
        jetSpecTau: Double? = nil,
        jetSpecDraftAccepted: Int? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.thinkingContent = thinkingContent
        self.isThinking = isThinking
        self.timestamp = timestamp
        self.tokenCount = tokenCount
        self.tokensPerSec = tokensPerSec
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.thinkingTimeSeconds = thinkingTimeSeconds
        self.prefillStatus = prefillStatus
        self.isTransient = isTransient
        self.isCompactionSummary = isCompactionSummary
        self.toolCalls = toolCalls
        self.attachments = attachments
        self.jetSpecTau = jetSpecTau
        self.jetSpecDraftAccepted = jetSpecDraftAccepted
    }
}

nonisolated public struct QueuedPrompt: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var text: String
    public var timestamp: Date
    /// Attachments already persisted to the asset store and waiting with the
    /// prompt. A queued prompt that drops these would silently send text only.
    public var attachments: [ImageAttachmentRef]?

    public init(
        id: UUID = UUID(),
        text: String,
        timestamp: Date = Date(),
        attachments: [ImageAttachmentRef]? = nil
    ) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.attachments = attachments
    }
}

/// How a conversation's `title` came to be, so the UI can mark a shortened
/// auto-generated name and the first-message auto-titler can leave a
/// user-chosen name alone.
nonisolated public enum SessionTitleOrigin: String, Codable, Equatable, Sendable {
    /// Generated from the first user message, which fit inside the title.
    case autoComplete
    /// Generated from the first user message, which continued past the cut-off.
    case autoTruncated
    /// Chosen by the user.
    case user
}

nonisolated public struct ChatSession: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var title: String
    public var messages: [ChatMessage]
    public var createdAt: Date
    public var updatedAt: Date
    public var selectedModelId: String?
    public var selectedModelName: String?
    public var selectedModelPath: String?
    public var isThinkingEnabled: Bool?
    /// Graded thinking level for level-capable models (Spark X2.6's
    /// `reasoning_effort`); meaningful only while `isThinkingEnabled` is true
    /// and the selected model supports levels. nil = unset, so the app-wide
    /// default applies. Back-compatible: older session files decode without
    /// it, and an unsupported stored value clamps to `.high` on read.
    public var thinkingEffort: ThinkingEffort?
    public var isAgentToolsEnabled: Bool?
    /// The conversation's agent surface: the coder (nil or `.code`) or plan
    /// mode, which reads and searches but cannot write. Per conversation, like
    /// `isAgentToolsEnabled`, so a plan survives a relaunch and switching chats
    /// cannot leave one conversation's read-only promise over another's writes.
    public var agentMode: AgentMode?
    public var queuedPrompts: [QueuedPrompt]
    /// Rolling general summary of everything a manual /compact removed from
    /// `messages`, written by the Apple Foundation Model and merged forward at
    /// every later compaction. Optional and back-compatible: older files
    /// decode without it, and conversations never compacted carry nil. The
    /// current pair is mirrored on screen by `compactedHistory`, whose card is
    /// display-only.
    public var rollingSummary: String?
    /// Detailed recap of the most recent evicted work, replaced wholesale by
    /// each /compact so recent context survives summarization at full fidelity
    /// until the next compaction folds it into `rollingSummary`.
    public var recentWorkDigest: String?
    /// Where `title` came from. `nil` for a fresh conversation still showing its
    /// placeholder name, so the first user message is free to auto-title it.
    /// Persisted so the truncation marker survives compaction (which replaces
    /// `messages` and would otherwise erase the evidence) and renames.
    public var titleOrigin: SessionTitleOrigin?
    /// How much context this conversation is carrying: the prompt the model
    /// last ingested plus the tokens generated after it. One meaning for every
    /// backend, because the header renders it as a share of a cloud model's
    /// window — a cloud step re-sends the whole history, a local turn
    /// re-renders it, and a cache read is still context the model was given,
    /// so none of those subtract. Each turn RESEEDS it from its own prompt
    /// (see `contextTokenBase`); it is never a running sum, which counted the
    /// same conversation once per turn and inflated by a whole history when a
    /// chat switched backends. `nil` when unmetered — a history from before
    /// the counter existed, or a conversation freshly compacted by /compact,
    /// which resets it — and the next send then initializes it from that
    /// turn's prompt.
    public var cumulativeTokenCount: Int?
    /// Running spend for this conversation, in whatever currency the recorded
    /// price list uses (USD by convention). Only a cloud turn contributes: a
    /// weights model has no invoice. `nil` when the conversation has never had
    /// a priced provider turn, and never reset by compaction — the money was
    /// already spent.
    public var cumulativeCost: Double?

    public init(
        id: UUID = UUID(),
        title: String = "New Conversation",
        messages: [ChatMessage] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        selectedModelId: String? = nil,
        selectedModelName: String? = nil,
        selectedModelPath: String? = nil,
        isThinkingEnabled: Bool? = nil,
        thinkingEffort: ThinkingEffort? = nil,
        isAgentToolsEnabled: Bool? = nil,
        agentMode: AgentMode? = nil,
        queuedPrompts: [QueuedPrompt] = [],
        rollingSummary: String? = nil,
        recentWorkDigest: String? = nil,
        titleOrigin: SessionTitleOrigin? = nil,
        cumulativeTokenCount: Int? = nil,
        cumulativeCost: Double? = nil
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.selectedModelId = selectedModelId
        self.selectedModelName = selectedModelName
        self.selectedModelPath = selectedModelPath
        self.isThinkingEnabled = isThinkingEnabled
        self.thinkingEffort = thinkingEffort
        // Plan mode is an agent MODE, not a prompt preference: with the tool
        // surface off it has no directive, no read tools and no handoff, so a
        // turn runs as plain chat under a pill that claims otherwise. Callers
        // that enable the mode go through `applyPlanMode`, which turns tools on
        // too; this keeps the pair from ever being representable apart.
        self.isAgentToolsEnabled = agentMode == .plan ? true : isAgentToolsEnabled
        self.agentMode = agentMode
        self.queuedPrompts = queuedPrompts
        self.rollingSummary = rollingSummary
        self.recentWorkDigest = recentWorkDigest
        self.titleOrigin = titleOrigin
        self.cumulativeTokenCount = cumulativeTokenCount
        self.cumulativeCost = cumulativeCost
    }

    enum CodingKeys: String, CodingKey {
        case id, title, messages, createdAt, updatedAt
        case selectedModelId, selectedModelName, selectedModelPath
        case isThinkingEnabled, thinkingEffort, isAgentToolsEnabled
        case agentMode
        case queuedPrompts
        case rollingSummary, recentWorkDigest
        case titleOrigin
        case cumulativeTokenCount
        case cumulativeCost
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(UUID.self, forKey: .id)
        self.title = try container.decode(String.self, forKey: .title)
        self.messages = try container.decode([ChatMessage].self, forKey: .messages)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.updatedAt = try container.decode(Date.self, forKey: .updatedAt)
        self.selectedModelId = try container.decodeIfPresent(String.self, forKey: .selectedModelId)
        self.selectedModelName = try container.decodeIfPresent(String.self, forKey: .selectedModelName)
        self.selectedModelPath = try container.decodeIfPresent(String.self, forKey: .selectedModelPath)
        self.isThinkingEnabled = try container.decodeIfPresent(Bool.self, forKey: .isThinkingEnabled)
        if let storedEffort = try container.decodeIfPresent(String.self, forKey: .thinkingEffort) {
            self.thinkingEffort = ThinkingEffort.parse(storedEffort) ?? .high
        } else {
            self.thinkingEffort = nil
        }
        self.isAgentToolsEnabled = try container.decodeIfPresent(Bool.self, forKey: .isAgentToolsEnabled)
        // An unknown stored value (a hand-edited file, a newer build's mode)
        // falls back to the coder rather than trapping the conversation in a
        // read-only surface nobody asked for.
        self.agentMode = (try container.decodeIfPresent(String.self, forKey: .agentMode))
            .flatMap(AgentMode.init(rawValue:))
        // A session written before the pairing was forced (a hand edit, an older
        // build, a file saved across the moment the tools were toggled) can
        // carry plan mode with tools off. Restoring that verbatim leaves turns
        // running as plain chat while the pill promises an agent, and the plan
        // handoff can never appear because nothing renders the directive. The
        // mode implies its surface, so normalize on the way in.
        if self.agentMode == .plan {
            self.isAgentToolsEnabled = true
        }
        self.queuedPrompts = try container.decodeIfPresent([QueuedPrompt].self, forKey: .queuedPrompts) ?? []
        self.rollingSummary = try container.decodeIfPresent(String.self, forKey: .rollingSummary)
        self.recentWorkDigest = try container.decodeIfPresent(String.self, forKey: .recentWorkDigest)
        self.titleOrigin = try container.decodeIfPresent(SessionTitleOrigin.self, forKey: .titleOrigin)
        self.cumulativeTokenCount = try container.decodeIfPresent(Int.self, forKey: .cumulativeTokenCount)
        self.cumulativeCost = try container.decodeIfPresent(Double.self, forKey: .cumulativeCost)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(title, forKey: .title)
        try container.encode(messages, forKey: .messages)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encodeIfPresent(selectedModelId, forKey: .selectedModelId)
        try container.encodeIfPresent(selectedModelName, forKey: .selectedModelName)
        try container.encodeIfPresent(selectedModelPath, forKey: .selectedModelPath)
        try container.encodeIfPresent(isThinkingEnabled, forKey: .isThinkingEnabled)
        try container.encodeIfPresent(thinkingEffort?.rawValue, forKey: .thinkingEffort)
        try container.encodeIfPresent(isAgentToolsEnabled, forKey: .isAgentToolsEnabled)
        try container.encodeIfPresent(agentMode?.rawValue, forKey: .agentMode)
        try container.encode(queuedPrompts, forKey: .queuedPrompts)
        try container.encodeIfPresent(rollingSummary, forKey: .rollingSummary)
        try container.encodeIfPresent(recentWorkDigest, forKey: .recentWorkDigest)
        try container.encodeIfPresent(titleOrigin, forKey: .titleOrigin)
        try container.encodeIfPresent(cumulativeTokenCount, forKey: .cumulativeTokenCount)
        try container.encodeIfPresent(cumulativeCost, forKey: .cumulativeCost)
    }
}

public extension ChatSession {
    /// Number of leading characters the first user message contributes to an
    /// auto-generated conversation title.
    nonisolated static let autoTitlePrefixLength = 28

    /// The auto-generated title for the first user message: its leading
    /// characters, trimmed, or "Chat" when there is nothing usable.
    nonisolated static func autoTitle(forFirstMessage text: String) -> String {
        let cleanTitle = text.prefix(autoTitlePrefixLength).trimmingCharacters(in: .whitespacesAndNewlines)
        return cleanTitle.isEmpty ? "Chat" : String(cleanTitle)
    }

    /// The origin to record alongside `autoTitle(forFirstMessage:)`: whether the
    /// message continued past the characters the title kept.
    nonisolated static func autoTitleOrigin(forFirstMessage text: String) -> SessionTitleOrigin {
        let cleanTitle = text.prefix(autoTitlePrefixLength).trimmingCharacters(in: .whitespacesAndNewlines)
        let fullLength = text.trimmingCharacters(in: .whitespacesAndNewlines).count
        return fullLength > cleanTitle.count ? .autoTruncated : .autoComplete
    }

    /// The title to render in the UI, with a trailing ellipsis when the stored
    /// name is a shortened auto-title so it does not read as complete.
    nonisolated var displayTitle: String {
        titleOrigin == .autoTruncated ? title + "…" : title
    }

    /// True when a conversation still holds nothing the user chose: no messages
    /// and a title that was never set by hand. The startup history merge uses
    /// this to decide whether the launch placeholder can be replaced by loaded
    /// history, so a rename to that placeholder is not discarded.
    nonisolated var isUnusedPlaceholder: Bool {
        messages.isEmpty && titleOrigin != .user
    }

    /// The most recent timestamp we can actually trust for a conversation.
    ///
    /// `updatedAt` is only set at creation/decode; the UI mutation paths that
    /// append messages, stream tokens, edit titles, and change queued prompts do
    /// not touch it, so it cannot mean "last used". Message and queued-prompt
    /// timestamps are maintained on those paths, so prefer the newest of them and
    /// only fall back to `updatedAt`/`createdAt`.
    nonisolated var lastActivityAt: Date {
        var latest = max(updatedAt, createdAt)
        for message in messages {
            latest = max(latest, message.timestamp)
            // Tool calls are appended after the message is created, so a long
            // agent run's freshest activity lives on them, not on the message.
            for call in message.toolCalls ?? [] {
                latest = max(latest, call.timestamp)
            }
        }
        for prompt in queuedPrompts {
            latest = max(latest, prompt.timestamp)
        }
        return latest
    }

    /// The two digests a compaction carries, in the form every consumer uses:
    /// scrubbed of template control tokens, trimmed, and reported as nil when
    /// nothing is left. Both the prompt block and the on-screen card are framed
    /// from this one resolution, so the text a reader sees and the text the
    /// model is handed cannot drift apart; only the framing around them
    /// differs. Scrubbing happens here rather than at storage time because the
    /// digests are derived from arbitrary user content and are kept verbatim.
    nonisolated static func compactionDigests(
        summary: String?,
        recap: String?
    ) -> (general: String?, detail: String?) {
        func resolve(_ text: String?) -> String? {
            let scrubbed = sanitizeCompactionDigest(text ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return scrubbed.isEmpty ? nil : scrubbed
        }
        return (resolve(summary), resolve(recap))
    }

    /// The compacted-context block injected into every post-compaction prompt,
    /// for either backend: the standing general summary first, then the
    /// detailed recap of the most recently evicted work. Nil before the first
    /// compaction, and trimmed to nil if both digests are empty.
    nonisolated var compactionContextBlock: String? {
        let digests = ChatSession.compactionDigests(
            summary: rollingSummary,
            recap: recentWorkDigest
        )
        var blocks: [String] = []
        if let general = digests.general {
            blocks.append("Summary of this conversation's earlier history (compacted, so reply with that knowledge in mind):\n\(general)")
        }
        if let detail = digests.detail {
            blocks.append("The most recently compacted work, kept in detail:\n\(detail)")
        }
        return blocks.isEmpty ? nil : blocks.joined(separator: "\n\n")
    }

    /// The rendered body of the `/compact` digest card: the standing general
    /// summary, then the detailed recap of the most recent work. This is DISPLAY
    /// text only. It mirrors what `compactionContextBlock` injects into the
    /// system prompt, from the same `compactionDigests` resolution, but it is
    /// never itself sent, because the card is a system-role message and every
    /// prompt builder skips those.
    nonisolated static func compactionSummaryCardContent(summary: String, recap: String?) -> String {
        let digests = compactionDigests(summary: summary, recap: recap)
        var blocks: [String] = []
        if let general = digests.general {
            blocks.append(general)
        }
        if let detail = digests.detail {
            blocks.append("---")
            blocks.append("**Recent work, in detail**")
            blocks.append(detail)
        }
        return blocks.joined(separator: "\n\n")
    }

    /// The transcript `/compact` commits: the visible digest card at the head of
    /// the conversation, then the outcome marker. Both rows are system-role, so
    /// neither reaches a later prompt (the digests travel in
    /// `compactionContextBlock` instead), and both survive a relaunch, unlike
    /// the transient progress bubble they replace. Rebuilding the pair from the
    /// fresh digests is what guarantees at most one card per conversation.
    nonisolated static func compactedHistory(
        summary: String,
        recap: String?,
        evictedMessageCount: Int,
        modelLabel: String = "Apple Foundation Model"
    ) -> [ChatMessage] {
        let card = ChatMessage(
            role: .system,
            content: compactionSummaryCardContent(summary: summary, recap: recap),
            isCompactionSummary: true
        )
        let marker = ChatMessage(
            role: .system,
            content: "🗂 \(evictedMessageCount) messages compacted into a rolling summary by \(modelLabel). The conversation continues from that summary context."
        )
        return [card, marker]
    }

    /// How many messages a compaction pass is reported as having folded away.
    /// The chunker is handed the whole array, but only conversation rows are
    /// turns: standing notices, and the card plus marker an earlier compaction
    /// left behind, are UI state, so counting them would inflate the figure the
    /// progress pill, the committed marker and the status notice all print.
    nonisolated static func compactableMessageCount(in messages: [ChatMessage]) -> Int {
        messages.filter { $0.role != .system }.count
    }

    /// Rows appended to a conversation while a long pass held an older snapshot
    /// of it.
    ///
    /// Compaction rewrites `messages` from the snapshot it summarized, and any
    /// row the app appended in the meantime (a `/plan` notice, a command's own
    /// report) is not in that snapshot — replacing the array would drop it
    /// after the user watched it appear. Identity decides rather than an index:
    /// the snapshot may have been reordered or pruned while the pass ran.
    nonisolated static func messagesAdded(
        to live: [ChatMessage],
        since snapshot: [ChatMessage],
        excluding: Set<UUID> = []
    ) -> [ChatMessage] {
        let known = Set(snapshot.map(\.id))
        return live.filter { !known.contains($0.id) && !excluding.contains($0.id) }
    }

    /// Odometer seeding for one generation step's prompt.
    ///
    /// The number every backend stores is the conversation's CONTEXT SIZE — the
    /// prompt the last request carried plus the tokens generated after it —
    /// because that is the quantity the header renders and the only one a
    /// context-window percentage can be taken from. A cloud turn re-sends its
    /// whole history and a local turn re-renders it, so this turn's prompt
    /// alone IS that size.
    ///
    /// It deliberately does not build on the stored total. Every backend used
    /// to add its own delta here, which counted the same conversation once per
    /// turn; once a cloud turn stored its context size (that path reseeds),
    /// the next local turn added the whole history to it and the pill jumped by
    /// a conversation's worth of tokens on a backend switch. Prefix reuse does
    /// not subtract either: a cached prefix is still context the model was
    /// given, and the pinned-vs-sent distinction is the cloud path's cache
    /// columns, not this figure. Generated tokens are added on top as they
    /// stream.
    nonisolated static func contextTokenBase(promptTokenCount: Int) -> Int {
        max(0, promptTokenCount)
    }

    /// Reseeds the odometer from this turn's prompt.
    ///
    /// Takes no previous total on purpose: the stored figure is REPLACED, never
    /// added to (see `contextTokenBase`). Written as one method so every seed
    /// point shares the rule, and so the rule is testable — a test can set a
    /// large stored total and prove it is ignored, which is exactly the
    /// regression the old code shipped (each backend added its own delta, so a
    /// cloud total plus the next prompt counted the same history twice).
    nonisolated mutating func seedContextTotal(promptTokenCount: Int) {
        cumulativeTokenCount = Self.contextTokenBase(promptTokenCount: promptTokenCount)
    }

    /// Neutralizes chat-template control tokens in a compaction digest before it
    /// rides inside the system block. The digests are derived from arbitrary
    /// user messages and tool output, and every weights dialect wraps the system
    /// block verbatim, so a digest carrying a delimiter such as `</s>`,
    /// `<|im_end|>`, the DeepSeek sentence marker, or `<|role_end|>` could close
    /// the system span and inject or malform the roles that follow. Matching the
    /// general token shapes covers every supported dialect without enumerating
    /// them by hand.
    nonisolated static func sanitizeCompactionDigest(_ text: String) -> String {
        var s = text
        for rule in compactionDigestScrubRules {
            guard let regex = try? NSRegularExpression(pattern: rule.pattern) else { continue }
            let range = NSRange(s.startIndex..<s.endIndex, in: s)
            s = regex.stringByReplacingMatches(in: s, range: range, withTemplate: rule.replacement)
        }
        return s
    }

    private static let compactionDigestScrubRules: [(pattern: String, replacement: String)] = [
        // <|im_start|>, <|im_end|>, <|endoftext|>, <|role_end|>, <|System|>, <|User|>, <|Bot|>, <|Tool|>
        (#"<\|[^|<>]{1,40}\|>"#, "[template-token]"),
        // DeepSeek/Spark fullwidth-bar markers: <｜start▁of▁sentence｜> and its peers.
        ("<\u{FF5C}[^|<>]{1,60}\u{FF5C}>", "[template-token]"),
        // Ling/Hermes role spans.
        (#"</?role>"#, "[role-token]"),
        // Llama end/start markers.
        (#"</?s>"#, "[end-token]")
    ]
}
