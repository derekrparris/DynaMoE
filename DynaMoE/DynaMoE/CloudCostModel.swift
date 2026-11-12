//
//  CloudCostModel.swift
//  DynaMoE
//
//  What a cloud turn costs and how much context it may carry.
//
//  Both are registration facts rather than provider truth: no chat-completions
//  catalog publishes a context window or a price list, and inventing either
//  would misprice a turn in the direction that looks authoritative. A model
//  with neither recorded simply reports no cost and never auto-compacts,
//  which is exactly the behaviour before this file existed.
//

import Foundation

// MARK: - Pricing

/// Per-million-token prices for one model, in US dollars. The price list is a
/// registration fact the operator writes by hand, so there is nothing here to
/// convert: a provider that bills in another currency is entered in USD by
/// whoever registers it, and the display renders the same `$` it was typed
/// with.
public nonisolated struct CloudPricing: Codable, Equatable, Sendable {
    /// USD per 1,000,000 fresh input tokens.
    public let input: Double
    /// USD per 1,000,000 cache-read input tokens. Nil means the provider has no
    /// discounted cache rate, so a cached token costs what a fresh one does.
    public let cachedInput: Double?
    /// USD per 1,000,000 output tokens.
    public let output: Double

    public init(input: Double, cachedInput: Double? = nil, output: Double) {
        self.input = input
        self.cachedInput = cachedInput
        self.output = output
    }

    enum CodingKeys: String, CodingKey {
        case input
        case cachedInput = "cached_input"
        case output
    }

    /// What one reported usage block cost. Nil when the provider reported no
    /// prompt count, because a cost built from half the numbers is worse than
    /// no cost at all.
    ///
    /// Cache-write tokens are charged at the fresh-input rate: Anthropic bills
    /// them at a premium, but this record has no fourth price, and overstating
    /// an unknown premium is the honest direction for a display figure.
    public func cost(for usage: CloudUsage) -> Double? {
        guard let promptTokens = usage.promptTokens else { return nil }
        let cached = max(0, min(usage.cachedTokens ?? 0, promptTokens))
        let fresh = Double(promptTokens - cached)
        let cachedRate = cachedInput ?? input
        let inputCost = (fresh * input + Double(cached) * cachedRate) / 1_000_000
        let outputCost = Double(max(0, usage.completionTokens ?? 0)) * output / 1_000_000
        return inputCost + outputCost
    }
}

/// Per-provider price registrations, stored as one JSON object under a
/// provider-scoped key:
///
///     defaults write DRP.DynaMoE dynamoe_cloud_synthetic_pricing \
///       '{"hf:org/model":{"input":0.27,"cached_input":0.07,"output":1.1}}'
///
/// A malformed value costs the price list and never a turn.
public nonisolated enum CloudPricingStore {
    public static func defaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_pricing"
    }

    public static func pricing(
        for providerId: String,
        defaults: UserDefaults = .standard
    ) -> [String: CloudPricing] {
        guard let raw = defaults.string(forKey: defaultsKey(for: providerId)),
              let data = raw.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([String: CloudPricing].self, from: data) else {
            return [:]
        }
        return decoded
    }

    public static func pricing(
        forModel providerModelId: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) -> CloudPricing? {
        pricing(for: providerId, defaults: defaults)[providerModelId]
    }

    public static func setPricing(
        _ pricing: [String: CloudPricing],
        for providerId: String,
        defaults: UserDefaults = .standard
    ) {
        guard !pricing.isEmpty else {
            defaults.removeObject(forKey: defaultsKey(for: providerId))
            return
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(pricing),
              let raw = String(data: data, encoding: .utf8) else { return }
        defaults.set(raw, forKey: defaultsKey(for: providerId))
    }

    /// A cost rendered for a status line. Small amounts keep enough precision
    /// to be non-zero, because "$0.00" on a turn that did cost something reads
    /// as "free". The symbol is fixed because the price list is entered in USD
    /// (see `CloudPricing`); nothing is converted.
    ///
    /// Below a ten-thousandth of a dollar there is no rendering that is both
    /// short and non-zero (`$%.4f` prints "$0.0000"), so the figure says it is
    /// smaller than the smallest one it can show rather than claiming zero.
    public static func formatted(_ cost: Double) -> String {
        guard cost.isFinite, cost > 0 else { return "$0" }
        if cost < 0.0001 { return "<$0.0001" }
        if cost < 0.01 { return String(format: "$%.4f", cost) }
        return String(format: "$%.2f", cost)
    }
}

// MARK: - Context budget

/// The advice appended to a context-overflow error.
///
/// Three different histories reach that error and they call for different next
/// steps, so the sentence has to say which one it is: a compaction that ran and
/// the request still did not fit, a compaction that ran and failed, or no
/// attempt at all. The old copy claimed "could not be compacted automatically"
/// for all three, including the run that had already compacted once.
public nonisolated enum CloudOverflowAdvice {
    public static func text(
        alreadyCompacted: Bool,
        failure: CloudCompactionFailure?,
        providerName: String
    ) -> String {
        if alreadyCompacted {
            return " This turn already compacted older history once and the request still exceeded the window — start a new chat or run /compact."
        }
        if let failure {
            var detail = failure.clause(providerName: providerName)
            if !detail.hasSuffix(".") { detail += "." }
            return " Auto-compaction left the history unchanged: \(detail)"
        }
        return " The conversation exceeded this model's context window — older history could not be compacted automatically, so start a new chat or run /compact."
    }
}

/// How much of a cloud model's context window may be used before the next send
/// compacts first, and how large a summarizer chunk may be.
///
/// The window is a registration fact per model (`dynamoe_cloud_<provider>_context`,
/// `model=window` entries) with a single global fallback. Zero or absent means
/// unknown, and an unknown window never triggers a compaction: guessing one
/// would summarise a 1M-token conversation at 100k tokens and spend money to
/// lose context.
public nonisolated enum CloudContextPolicy {
    /// The fallback window for every model on a provider that records no
    /// per-model value. 0 disables auto-compaction.
    public static let globalWindowKey = "dynamoe_cloud_context_window_tokens"

    /// Compact when the assembled prompt would use this much of the window.
    /// 0.8 leaves room for the reply and for the estimate's own error.
    public static let autoCompactFraction = 0.8

    /// Below this a "window" is a units mistake, not a context size.
    public static let minimumPlausibleWindow = 1_024

    public static func defaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_context"
    }

    /// One provider's per-model windows, parsed from `model=window` entries
    /// separated by `;`, `,` or newline. Tolerates `model@window` and bare
    /// integers, because those are the two spellings a person types.
    public static func overrides(
        for providerId: String,
        defaults: UserDefaults = .standard
    ) -> [String: Int] {
        let raw = defaults.string(forKey: defaultsKey(for: providerId)) ?? ""
        var result: [String: Int] = [:]
        for chunk in raw.split(whereSeparator: { $0 == ";" || $0 == "," || $0 == "\n" }) {
            let entry = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !entry.isEmpty else { continue }
            let parts = entry.split(maxSplits: 1, whereSeparator: { $0 == "=" || $0 == "@" })
            guard parts.count == 2 else { continue }
            let model = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
            let window = Int(parts[1].trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
            guard !model.isEmpty, window >= minimumPlausibleWindow else { continue }
            result[model] = window
        }
        return result
    }

    public static func setOverrides(
        _ overrides: [String: Int],
        for providerId: String,
        defaults: UserDefaults = .standard
    ) {
        let cleaned = overrides
            .filter { !$0.key.isEmpty && $0.value >= minimumPlausibleWindow }
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "; ")
        if cleaned.isEmpty {
            defaults.removeObject(forKey: defaultsKey(for: providerId))
        } else {
            defaults.set(cleaned, forKey: defaultsKey(for: providerId))
        }
    }

    /// The window this model is believed to have, or nil when nobody recorded
    /// one. The per-model value wins over the provider-wide fallback.
    ///
    /// The fallback is a SINGLE global key, so it answers for every provider.
    /// A default recorded for one provider's large-window model therefore also
    /// describes a smaller-window model elsewhere, and under-compacting that
    /// one is the failure mode: the turn runs until the provider refuses it,
    /// and the overflow path compacts then. Record a per-model window wherever
    /// the difference matters.
    public static func window(
        for spec: CloudModelSpec?,
        defaults: UserDefaults = .standard
    ) -> Int? {
        resolvedWindow(recorded: spec?.contextWindowTokens, defaults: defaults)
    }

    /// The window recorded for one model id, resolved without building the
    /// provider's whole registry.
    ///
    /// `CloudModelRegistry.registeredSpecs` rebuilds every spec on the provider
    /// to answer one lookup, and the chat header asks this on every body
    /// evaluation (including 60 fps streaming updates). The recorded fact it
    /// would report is the same per-model override this reads, so the two
    /// answers cannot disagree.
    public static func window(
        forModelId modelId: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) -> Int? {
        resolvedWindow(
            recorded: overrides(for: providerId, defaults: defaults)[modelId],
            defaults: defaults
        )
    }

    private static func resolvedWindow(recorded: Int?, defaults: UserDefaults) -> Int? {
        if let recorded, recorded >= minimumPlausibleWindow {
            return recorded
        }
        let fallback = defaults.integer(forKey: globalWindowKey)
        return fallback >= minimumPlausibleWindow ? fallback : nil
    }

    /// Whether a send of `promptTokens` should compact first.
    public static func shouldCompact(promptTokens: Int, window: Int?) -> Bool {
        guard let window, window >= minimumPlausibleWindow else { return false }
        return Double(promptTokens) >= Double(window) * autoCompactFraction
    }

    /// The share of the window a prompt occupies, for a status line.
    public static func usagePercent(promptTokens: Int, window: Int?) -> Int? {
        guard let window, window > 0 else { return nil }
        return Int((Double(promptTokens) / Double(window) * 100).rounded())
    }

    /// Characters of transcript one summarizer pass may carry.
    ///
    /// Four characters per token is the same estimate the rest of the cloud
    /// path uses. A third of the window leaves room for the digest being folded
    /// in, the pass instructions, and the summary itself.
    public static func chunkCharacterBudget(window: Int?) -> Int {
        guard let window, window >= minimumPlausibleWindow else { return 24_000 }
        return max(4_000, Int(Double(window) * 0.33) * 4)
    }
}

// MARK: - Compaction boundary

/// Where a compaction is allowed to cut a conversation.
///
/// The cut is always a message boundary. That is the whole safety rule in this
/// data model: a tool exchange is emitted from ONE assistant message
/// (`CloudMessageAssembler.toolExchange` derives the `tool_calls` message and
/// every `tool` result from it), so no boundary can leave a result whose call
/// was evicted. The user-turn preference below is therefore about where a
/// conversation reads naturally, not about validity.
public nonisolated enum CloudCompactionBoundary {
    /// Index of the first message to KEEP, or nil when the history offers no
    /// cut that would evict any of the conversation.
    ///
    /// `keepRecentMessages <= 0` keeps nothing (manual `/compact`); every
    /// returned index is a message boundary, so the caller can split without
    /// validating the tail again.
    public static func tailStart(messages: [ChatMessage], keepRecentMessages: Int) -> Int? {
        guard !messages.isEmpty else { return nil }
        guard keepRecentMessages > 0 else { return messages.count }
        let target = max(0, messages.count - keepRecentMessages)
        // A cut is only useful when it folds away something the summary can
        // describe. System rows are UI state (standing notices, and the digest
        // card plus marker a previous pass left behind), so a cut that only
        // evicts those summarises nothing and the pass that receives it
        // reports a failure.
        func evictsConversation(at cut: Int) -> Bool {
            messages[..<cut].contains { $0.role != .system }
        }
        // Preferred: the newest user turn at or before the requested cut.
        var userBoundary = target
        while userBoundary > 0, messages[userBoundary].role != .user { userBoundary -= 1 }
        if messages[userBoundary].role == .user, userBoundary > 0,
           evictsConversation(at: userBoundary) {
            return userBoundary
        }
        // Either the only user turn in range is the head of the conversation,
        // or the user turn in range sits behind a compacted head. One long
        // agent run is one question followed by assistant and tool turns, so
        // the newest user turn before the cut is that question and the rows
        // before it are the digest card and marker: cutting there evicted the
        // two system rows and nothing else, which is what made auto-compaction
        // fail for every pass after its first. Take the requested cut instead;
        // the recap digest is what carries the evicted work forward.
        if target > 0, evictsConversation(at: target) { return target }
        // Nothing to evict at the target, and no usable user turn at or before
        // it. Keep the NEXT user turn when that cut evicts conversation rows;
        // otherwise this history has nothing worth compacting, and saying so is
        // better than returning a cut that folds away only the card.
        guard let nextUser = messages[target...].firstIndex(where: { $0.role == .user }),
              evictsConversation(at: nextUser) else { return nil }
        return nextUser
    }

    /// The messages a compaction evicts and the ones it keeps.
    ///
    /// The cut is `tailStart`'s, with one adjustment: a tail that would open
    /// with an assistant message gets the newest user turn before the cut
    /// moved into it. Two reasons, and both are real. A request whose first
    /// message is an assistant turn is refused by some OpenAI-compatible
    /// endpoints, and a mid-run cut leaves exactly that shape; and that user
    /// turn is the question the evicted work answered, so keeping it verbatim
    /// beats relying on the digest to restate it.
    ///
    /// Nil when the plan would fold away no conversation at all: the anchor
    /// move can put back the only non-system row the cut had, and a pass that
    /// rewrote the history for nothing would report a success it did not have.
    public static func plan(
        messages: [ChatMessage],
        keepRecentMessages: Int
    ) -> CloudCompactionPlan? {
        guard let keptStart = tailStart(
            messages: messages,
            keepRecentMessages: keepRecentMessages
        ) else { return nil }
        var evicted = Array(messages[..<keptStart])
        var kept = Array(messages[keptStart...])
        // A manual compaction (`keepRecentMessages <= 0`) keeps nothing on
        // purpose, so the anchor rule must not pull a message back into it.
        if let first = kept.first, first.role != .user,
           let anchor = evicted.lastIndex(where: { $0.role == .user }) {
            kept.insert(evicted.remove(at: anchor), at: 0)
        }
        guard evicted.contains(where: { $0.role != .system }) else { return nil }
        return CloudCompactionPlan(evicted: evicted, kept: kept)
    }
}

/// What one compaction pass folds away and what it leaves in the transcript.
public nonisolated struct CloudCompactionPlan: Equatable, Sendable {
    /// Messages summarized and removed from the conversation.
    public let evicted: [ChatMessage]
    /// Messages kept verbatim, in their original order.
    public let kept: [ChatMessage]

    public init(evicted: [ChatMessage], kept: [ChatMessage]) {
        self.evicted = evicted
        self.kept = kept
    }
}

/// Why a cloud compaction left the conversation untouched.
///
/// Every refusal used to be reported as "<provider> could not summarize this
/// conversation", which blamed the provider for outcomes that never sent a
/// request: no stored key, a chat with nothing evictable, an empty transcript.
/// Each cause has its own fix, so each gets its own sentence.
public nonisolated enum CloudCompactionFailure: Equatable, Error, Sendable {
    /// Too few messages for a summary to replace anything.
    case conversationTooShort
    /// The history offers no cut that folds away conversation rows.
    case nothingToEvict
    /// The conversation was closed or pruned while the summary was running.
    case conversationUnavailable
    /// No summarizer request could be made; the payload is the credential
    /// failure's own description, which already says where the key lives.
    case noCredential(String)
    /// The evicted messages carry no content the transcript builder renders.
    case noTranscript
    /// A pass came back with no text at all.
    case emptySummary
    /// A pass stopped at its output limit, so its text never covered the chunk
    /// it was given. Committing it would delete work the summary omits.
    case truncatedSummary
    /// Stop was pressed while the summary was being written.
    case cancelled
    /// The provider request failed; the payload is the failure's description.
    case provider(String)

    /// What the chat surface prints for this outcome. A cause that sent no
    /// request is reported as a precondition, never as a provider fault.
    public func notice(providerName: String) -> String {
        switch self {
        case .conversationTooShort:
            return "ℹ️ Nothing to compact — this conversation has too few messages."
        case .nothingToEvict:
            return "ℹ️ Nothing to compact — there is no older history to summarize yet."
        case .conversationUnavailable:
            return "⚠️ This conversation is no longer available — nothing was compacted."
        case .noCredential(let description):
            return "⚠️ \(description)"
        case .noTranscript:
            return "ℹ️ Nothing to compact — no user or assistant content to summarize yet."
        case .emptySummary:
            return "⚠️ \(providerName) returned an empty summary — the conversation was left unchanged."
        case .truncatedSummary:
            return "⚠️ \(providerName)'s summary stopped at its output limit before it covered the whole conversation, so the conversation was left unchanged."
        case .cancelled:
            return "⏹ Compaction stopped — the conversation was left unchanged."
        case .provider(let description):
            return "⚠️ \(providerName) could not summarize this conversation: \(description) — it was left unchanged."
        }
    }

    /// The same cause as a clause with no glyph, for embedding in a longer
    /// sentence (the context-overflow error names why compaction did not run).
    public func clause(providerName: String) -> String {
        switch self {
        case .conversationTooShort:
            return "there are too few messages to summarize"
        case .nothingToEvict:
            return "there is no older history to fold away"
        case .conversationUnavailable:
            return "the conversation is no longer available"
        case .noCredential:
            return "the summarizer's API key could not be used"
        case .noTranscript:
            return "no user or assistant content could be summarized"
        case .emptySummary:
            return "\(providerName) returned an empty summary"
        case .truncatedSummary:
            return "\(providerName)'s summary hit its output limit before it covered the conversation"
        case .cancelled:
            return "the summary was stopped"
        case .provider(let description):
            return "summarizing failed: \(description)"
        }
    }
}

// MARK: - Context odometer

/// The context figure the chat header renders.
///
/// This is a CONTEXT size, not an invoice. A cloud turn re-sends the whole
/// conversation on every step, so what the model has seen is the last
/// request's prompt plus the tokens generated after it. Summing what each step
/// billed would only agree while the provider reports a cache split; a
/// provider that reports none bills the whole prompt every step, and a
/// twenty-step agent run would then read as twenty conversations' worth.
public nonisolated enum CloudContextOdometer {
    public static func total(promptTokens: Int, generatedTokens: Int) -> Int {
        max(0, promptTokens) + max(0, generatedTokens)
    }
}

// MARK: - Turn report

/// The numbers one cloud turn actually produced, formatted for the status
/// line. Built from the provider's own usage where it reported any, and from
/// the app's estimate where it did not, so the figure never silently reads as
/// a provider measurement it is not.
public nonisolated struct CloudTurnReport: Equatable, Sendable {
    public let promptTokens: Int?
    public let cachedTokens: Int
    public let outputTokens: Int
    public let cost: Double?
    public let wasEstimated: Bool

    public init(
        usage: CloudUsage?,
        estimatedPromptTokens: Int,
        outputTokens: Int,
        cost: Double?
    ) {
        self.wasEstimated = usage?.promptTokens == nil
        self.promptTokens = usage?.promptTokens ?? estimatedPromptTokens
        self.cachedTokens = usage?.cachedTokens ?? 0
        self.outputTokens = max(0, usage?.completionTokens ?? outputTokens)
        self.cost = cost
    }

    /// A whole run's totals rather than one request's usage block.
    ///
    /// A cloud agent turn is several billed requests, so the summary line at
    /// the end of the run reports what every step cost, matching the session's
    /// cumulative spend. `wasEstimated` is true when ANY step fell back to the
    /// app's own token estimate.
    public init(
        promptTokens: Int,
        cachedTokens: Int,
        outputTokens: Int,
        cost: Double?,
        wasEstimated: Bool
    ) {
        self.promptTokens = max(0, promptTokens)
        self.cachedTokens = max(0, cachedTokens)
        self.outputTokens = max(0, outputTokens)
        self.cost = cost
        self.wasEstimated = wasEstimated
    }

    /// `1,204 in (1,000 cached) / 56 out · $0.0031`, or the estimate-marked
    /// form when the provider reported no usage.
    public var statusSuffix: String {
        var parts: [String] = []
        if let promptTokens {
            let number = Self.format(promptTokens)
            if cachedTokens > 0 {
                parts.append("\(number) in (\(Self.format(cachedTokens)) cached)")
            } else {
                parts.append("\(number) in")
            }
        }
        parts.append("\(Self.format(outputTokens)) out")
        var text = parts.joined(separator: " / ")
        if wasEstimated { text += " (est.)" }
        if let cost { text += " · " + CloudPricingStore.formatted(cost) }
        return text
    }

    static func format(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = ","
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }
}
