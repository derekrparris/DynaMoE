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

/// Per-million-token prices for one model, in the currency the user keeps
/// their provider bill in (USD by convention; nothing here assumes it).
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
    /// as "free".
    public static func formatted(_ cost: Double) -> String {
        if cost <= 0 { return "$0" }
        if cost < 0.01 { return String(format: "$%.4f", cost) }
        return String(format: "$%.2f", cost)
    }
}

// MARK: - Context budget

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
    public static func window(
        for spec: CloudModelSpec?,
        defaults: UserDefaults = .standard
    ) -> Int? {
        if let recorded = spec?.contextWindowTokens, recorded >= minimumPlausibleWindow {
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
/// The kept tail must begin at a user turn. A tool result whose assistant call
/// was evicted is a hard API error on the next request, and a keep-by-count
/// rule would happily cut an exchange in half — this is the one rule that makes
/// mid-run compaction safe.
public nonisolated enum CloudCompactionBoundary {
    /// Index of the first message to KEEP, or nil when no safe cut exists.
    ///
    /// `keepRecentMessages <= 0` keeps nothing (manual `/compact`); every
    /// returned index is either `messages.count` or the position of a user
    /// turn, so the caller can split without validating the tail again.
    public static func tailStart(messages: [ChatMessage], keepRecentMessages: Int) -> Int? {
        guard !messages.isEmpty else { return nil }
        guard keepRecentMessages > 0 else { return messages.count }
        var index = max(0, messages.count - keepRecentMessages)
        while index > 0, messages[index].role != .user { index -= 1 }
        if messages[index].role == .user { return index }
        // No user turn at or before the target index: a short conversation, or
        // one whose head is the system-role card and marker a previous
        // compaction left behind. Keep the NEXT user turn instead, so a
        // compacted history can compact again rather than silently refusing
        // forever.
        return messages[index...].firstIndex(where: { $0.role == .user })
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
