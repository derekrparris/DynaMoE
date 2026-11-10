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
    /// Running total of every token this conversation has consumed: the prompt
    /// tokens each turn actually ingested (the portion the pinned prefix did
    /// not already cover) plus every generated token, across all agent steps.
    /// `nil` when unmetered — a history from before the counter existed, or a
    /// conversation freshly compacted by /compact, which resets the total —
    /// and the next send then initializes it from that turn's full prompt,
    /// which is the conversation's context size.
    public var cumulativeTokenCount: Int?

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
        queuedPrompts: [QueuedPrompt] = [],
        rollingSummary: String? = nil,
        recentWorkDigest: String? = nil,
        titleOrigin: SessionTitleOrigin? = nil,
        cumulativeTokenCount: Int? = nil
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
        self.isAgentToolsEnabled = isAgentToolsEnabled
        self.queuedPrompts = queuedPrompts
        self.rollingSummary = rollingSummary
        self.recentWorkDigest = recentWorkDigest
        self.titleOrigin = titleOrigin
        self.cumulativeTokenCount = cumulativeTokenCount
    }

    enum CodingKeys: String, CodingKey {
        case id, title, messages, createdAt, updatedAt
        case selectedModelId, selectedModelName, selectedModelPath
        case isThinkingEnabled, thinkingEffort, isAgentToolsEnabled
        case queuedPrompts
        case rollingSummary, recentWorkDigest
        case titleOrigin
        case cumulativeTokenCount
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
        self.queuedPrompts = try container.decodeIfPresent([QueuedPrompt].self, forKey: .queuedPrompts) ?? []
        self.rollingSummary = try container.decodeIfPresent(String.self, forKey: .rollingSummary)
        self.recentWorkDigest = try container.decodeIfPresent(String.self, forKey: .recentWorkDigest)
        self.titleOrigin = try container.decodeIfPresent(SessionTitleOrigin.self, forKey: .titleOrigin)
        self.cumulativeTokenCount = try container.decodeIfPresent(Int.self, forKey: .cumulativeTokenCount)
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
        try container.encode(queuedPrompts, forKey: .queuedPrompts)
        try container.encodeIfPresent(rollingSummary, forKey: .rollingSummary)
        try container.encodeIfPresent(recentWorkDigest, forKey: .recentWorkDigest)
        try container.encodeIfPresent(titleOrigin, forKey: .titleOrigin)
        try container.encodeIfPresent(cumulativeTokenCount, forKey: .cumulativeTokenCount)
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
        evictedMessageCount: Int
    ) -> [ChatMessage] {
        let card = ChatMessage(
            role: .system,
            content: compactionSummaryCardContent(summary: summary, recap: recap),
            isCompactionSummary: true
        )
        let marker = ChatMessage(
            role: .system,
            content: "🗂 \(evictedMessageCount) messages compacted into a rolling summary by Apple Foundation Model. The conversation continues from that summary context."
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

    /// Odometer arithmetic for one generation step's prompt ingestion: the
    /// conversation total after a turn's prompt tokens land on it. An unmetered
    /// conversation (`nil` — a pre-feature history, or one just compacted, whose
    /// pin is also invalid) seeds from the entire prompt, because that prompt
    /// IS its context size; a metered one adds only the tokens the pinned prefix
    /// did not already supply, so prefix reuse never double counts. Generated
    /// tokens are added on top of the returned base as they stream.
    nonisolated static func cumulativeTokenBase(
        previousTotal: Int?,
        promptTokenCount: Int,
        prefixTokensReused: Int
    ) -> Int {
        guard let previousTotal else { return promptTokenCount }
        return previousTotal + max(0, promptTokenCount - prefixTokensReused)
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
