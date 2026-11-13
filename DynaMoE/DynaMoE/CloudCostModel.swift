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
    /// USD per 1,000,000 cache-WRITE input tokens. Nil means the provider
    /// charges a write at the fresh-input rate, which is what OpenAI-shaped
    /// endpoints do. Record it for a provider that bills a write premium
    /// (Anthropic charges 1.25x) or every cached write is understated.
    public let cacheWriteInput: Double?

    public init(
        input: Double,
        cachedInput: Double? = nil,
        output: Double,
        cacheWriteInput: Double? = nil
    ) {
        self.input = input
        self.cachedInput = cachedInput
        self.output = output
        self.cacheWriteInput = cacheWriteInput
    }

    enum CodingKeys: String, CodingKey {
        case input
        case cachedInput = "cached_input"
        case output
        case cacheWriteInput = "cache_write_input"
    }

    /// Whether every recorded rate is a usable price. Prices are hand-typed
    /// JSON, so a negative, non-finite, or absurd number is a typo; a cost
    /// built from it would be negative or absurd and the display's `> 0` guard
    /// would hide it as "$0", which is worse than reporting no cost at all.
    ///
    /// The ceiling matters as much as the floor. A finite rate of `1e300` per
    /// million tokens overflows the arithmetic once it meets a real token
    /// count: the product is `+inf`, `formatted` prints "$0" for it, and the
    /// infinite cost is then added to `cumulativeCost`, which stays infinite
    /// for the rest of the conversation. A million dollars per million tokens
    /// is already past every list price by three orders of magnitude, so
    /// anything above it is a units or decimal-point mistake.
    public var isValid: Bool {
        func usable(_ value: Double) -> Bool {
            value.isFinite && value >= 0 && value <= Self.maximumUsableRate
        }
        return usable(input) && usable(output)
            && (cachedInput.map(usable) ?? true)
            && (cacheWriteInput.map(usable) ?? true)
    }

    /// The largest rate a price entry may record, in USD per million tokens.
    public static let maximumUsableRate: Double = 1_000_000

    /// What one reported usage block cost. Nil when the provider reported no
    /// prompt count, because a cost built from half the numbers is worse than
    /// no cost at all.
    ///
    /// A block with a prompt count and no COMPLETION count is refused for the
    /// same reason: it would price the input and silently charge nothing for
    /// the reply, so the status line shows a definite, understated figure
    /// beside `(est.)`. Both halves are required or neither is priced.
    ///
    /// Cache-write tokens are charged at `cacheWriteInput` when the operator
    /// recorded it, and at the fresh-input rate otherwise. Every rate is
    /// clamped non-negative here as well as at decode, because this type is
    /// also constructed in code where a negative rate would otherwise turn a
    /// turn's spend negative.
    public func cost(for usage: CloudUsage) -> Double? {
        guard let reportedPrompt = usage.promptTokens,
              let reportedCompletion = usage.completionTokens else { return nil }
        let promptTokens = max(0, reportedPrompt)
        let cached = max(0, min(usage.cachedTokens ?? 0, promptTokens))
        let written = max(0, min(usage.cacheWriteTokens ?? 0, promptTokens - cached))
        let fresh = Double(promptTokens - cached - written)
        let inputCost = (fresh * Self.rate(input)
            + Double(cached) * Self.rate(cachedInput ?? input)
            + Double(written) * Self.rate(cacheWriteInput ?? input)) / 1_000_000
        let outputCost = Double(max(0, reportedCompletion)) * Self.rate(output) / 1_000_000
        let total = inputCost + outputCost
        // Rates are bounded at decode, but this type is constructible in code:
        // an infinite total would be laid into `cumulativeCost` once and never
        // recover, so it is reported as "no price" instead.
        return total.isFinite ? total : nil
    }

    private static func rate(_ value: Double) -> Double {
        value.isFinite ? max(0, value) : 0
    }
}

/// Per-provider price registrations, stored as one JSON object under a
/// provider-scoped key:
///
///     defaults write DRP.DynaMoE dynamoe_cloud_synthetic_pricing -string \
///       '{"hf:org/model":{"input":0.27,"cached_input":0.07,"output":1.1}}'
///
/// A malformed entry costs that model's price and never the turn. Decoding is
/// per key on purpose: the whole-list decoder used to drop every price on the
/// provider when one hand-typed entry had a typo, which read as "this provider
/// reports no cost" and hid the mistake.
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
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        let decoder = JSONDecoder()
        var prices: [String: CloudPricing] = [:]
        for (model, value) in object {
            guard let entry = try? JSONSerialization.data(withJSONObject: value),
                  let price = try? decoder.decode(CloudPricing.self, from: entry),
                  price.isValid else { continue }
            prices[model] = price
        }
        return prices
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

    /// Removes ONE model's price without rewriting the rest of the list.
    ///
    /// Pairing the reader above with `setPricing` loses data: `pricing(for:)`
    /// returns only the entries that DECODE, so writing that dictionary back
    /// deletes every other model's hand-typed price that happened to have a
    /// typo. That is exactly what unregistering a different model did. This
    /// edits the stored object key by key, so an entry with a typo stays where
    /// the operator can see and fix it.
    public static func removePrice(
        forModel model: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) {
        let key = defaultsKey(for: providerId)
        guard let raw = defaults.string(forKey: key),
              let data = raw.data(using: .utf8),
              var object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              object.removeValue(forKey: model) != nil else { return }
        if object.isEmpty {
            defaults.removeObject(forKey: key)
            return
        }
        guard let updated = try? JSONSerialization.data(
                  withJSONObject: object,
                  options: [.sortedKeys]
              ),
              let text = String(data: updated, encoding: .utf8) else { return }
        defaults.set(text, forKey: key)
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
        if cost < 0.01 {
            let text = String(format: "$%.4f", cost)
            // 0.009999 renders as "$0.0100", which claims a cent it is under.
            // Say it is below the smallest figure at this scale instead.
            return text == "$0.0100" ? "<$0.0100" : text
        }
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
        // The recorded failure describes the MOST RECENT attempt, so it is the
        // more actionable sentence even when an earlier pass did rewrite the
        // history: "it compacted once and then the summary came back empty" is
        // a different situation from "it compacted once and the result still
        // did not fit". Reaching the wall with no failure means the last pass
        // succeeded, which is the only case the already-compacted sentence
        // describes.
        if let failure {
            var detail = failure.clause(providerName: providerName)
            if !detail.hasSuffix(".") { detail += "." }
            let prefix = alreadyCompacted
                ? " This turn compacted older history and a later attempt failed"
                : " Auto-compaction left the history unchanged"
            return prefix + ": \(detail)"
        }
        if alreadyCompacted {
            // No count: a run may spend up to `CloudCompactionAttempts`'
            // budget, so "once" was wrong after the second pass. What the
            // reader needs is that compaction has been tried and did not
            // clear the wall.
            return " This turn already compacted older history and the request still exceeded the window — start a new chat or run /compact."
        }
        return " The conversation exceeded this model's context window — older history could not be compacted automatically, so start a new chat or run /compact."
    }
}

/// How many compaction passes one cloud run may spend.
///
/// A cloud agent run re-sends the whole conversation every step, so a gate
/// that only closed on a SUCCESSFUL pass retried a failed one (an empty or
/// truncated summary, a provider error) at the top of every remaining step.
/// Each retry re-resolved the keychain, re-chunked the history and issued a
/// billed summarizer request with no new chance of success.
///
/// The budget is up to `maximumRewritesPerRun` PRE-FLIGHT passes plus one
/// overflow-recovery pass:
///
/// - a pre-flight pass is marked when it STARTS, because the request is billed
///   before its outcome is known; an attempt that turns out to have sent
///   nothing is given back, because it spent nothing (`abandonPreflight`), and
///   a pass that SUCCEEDED releases the gate for a later one;
/// - a later pre-flight is allowed only after the conversation has grown past
///   the size the previous rewrite left behind, and only until the rewrite cap
///   is reached, so a run pays for summaries only when they can cover new
///   work;
/// - overflow recovery keeps its own attempt, because the provider rejecting
///   the request is evidence the pre-flight estimate did not have.
///
/// `historyWasRewritten` rides beside the two flags so the overflow advice can
/// still tell "already compacted and still too long" from "compaction left the
/// history alone". It used to close the pre-flight gate for the rest of the
/// run, which starved an uncapped agent run: after one rewrite the third wall
/// ended the turn with no further attempt. A rewrite now records the prompt
/// size it left behind, and a later pre-flight is allowed once the
/// conversation has GROWN past what was already compacted — the pass would
/// summarize new work rather than the digest it just wrote, which is the only
/// reason the gate was closed. `rewriteCount` bounds the spend at one pass per
/// `maximumRewritesPerRun` collapses.
public nonisolated struct CloudCompactionAttempts: Equatable, Sendable {
    public private(set) var preflightAttempted = false
    public private(set) var overflowAttempted = false
    public private(set) var historyWasRewritten = false
    /// The prompt size a rewrite left behind, when the caller recorded one.
    public private(set) var promptTokensAtLastRewrite: Int?
    /// Passes that actually replaced the history, across the whole run.
    public private(set) var rewriteCount = 0

    /// A run that keeps refilling its window is not paying for summaries that
    /// can help; each one costs a full conversation's prompt. Three is enough
    /// for a long agent run to cross two windows and stop.
    public static let maximumRewritesPerRun = 3

    public init() {}

    /// Whether a pre-flight pass is worth running for a prompt of this size.
    ///
    /// Before any rewrite this is the old one-attempt gate. After one, it
    /// reopens only when the new projection exceeds the size the last rewrite
    /// left behind: below that there is nothing the last summary did not
    /// already cover, and above it the conversation has genuinely refilled.
    public func shouldAttemptPreflight(projectedTokens: Int) -> Bool {
        guard !preflightAttempted else { return false }
        guard historyWasRewritten else { return true }
        guard rewriteCount < Self.maximumRewritesPerRun else { return false }
        guard let baseline = promptTokensAtLastRewrite else { return false }
        return projectedTokens > baseline
    }

    /// The cheap half of the same question, for the caller that has to get the
    /// projected size before it can ask: false once the budget is spent or
    /// once a rewrite left no baseline, so an exhausted run does not assemble
    /// and tokenise its history on every step just to decline.
    public var allowsPreflightCheck: Bool {
        guard !preflightAttempted, rewriteCount < Self.maximumRewritesPerRun else { return false }
        return !historyWasRewritten || promptTokensAtLastRewrite != nil
    }

    public var shouldAttemptOverflowRecovery: Bool {
        !overflowAttempted
    }

    public mutating func beginPreflight() {
        preflightAttempted = true
    }

    /// Gives the pre-flight attempt back when the pass never reached the
    /// provider. A refusal that sent no request costs nothing and says nothing
    /// about the next step, so holding the gate shut would make a one-attempt
    /// budget cost a compaction that became possible later.
    public mutating func abandonPreflight() {
        preflightAttempted = false
    }

    public mutating func beginOverflowRecovery() {
        overflowAttempted = true
    }

    /// Records a pass that replaced the history. `promptTokens` is the size the
    /// rewritten prompt LEFT behind, not the size the pass was deciding about:
    /// it becomes the floor a later pre-flight has to exceed, and the surviving
    /// prompt is what following steps actually send.
    ///
    /// The in-flight attempt is released here rather than left spent: a pass
    /// that SUCCEEDED is not a pending question, and a later pre-flight is a new
    /// attempt whose worth `shouldAttemptPreflight(projectedTokens:)` and the
    /// rewrite cap decide.
    public mutating func recordRewrite(promptTokens: Int? = nil) {
        historyWasRewritten = true
        rewriteCount += 1
        preflightAttempted = false
        if let promptTokens {
            promptTokensAtLastRewrite = max(0, promptTokens)
        }
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

    /// The reply budget a cloud model is held to when it records NO window.
    ///
    /// An unrecorded window used to mean "no cap": `replyBudget` passed the
    /// caller's request straight through, so whatever Max Output Tokens the
    /// settings slider allowed was sent verbatim as `max_tokens`. At the slider's
    /// old 32,768 that was already generous; once the slider became honest about
    /// a 128k-window model it became a budget no provider asked for and several
    /// refuse outright. With no window there is no capacity to justify a larger
    /// number, so this is the number instead.
    ///
    /// It is deliberately the value the generic (cloud) profile's coder default
    /// now carries, so an unconfigured provider gets a coherent budget rather
    /// than a value that is then clamped by a different constant.
    public static let unknownWindowReplyCeiling = 16_384

    public static func defaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_context"
    }

    /// One provider's per-model windows, parsed from `model=window` entries
    /// separated by `;`, `,` or newline. Tolerates `model@window`, grouped
    /// integers (`1,048,576`), and bare integers.
    public static func overrides(
        for providerId: String,
        defaults: UserDefaults = .standard
    ) -> [String: Int] {
        let raw = defaults.string(forKey: defaultsKey(for: providerId)) ?? ""
        var result: [String: Int] = [:]
        for entry in splitOverrideEntries(raw) {
            guard let parts = splitEntry(entry) else { continue }
            let window = parseWindowValue(parts.value) ?? 0
            guard window >= minimumPlausibleWindow else { continue }
            result[parts.model] = window
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

    /// Removes ONE model's window from the stored list, leaving every other
    /// entry — including one this build could not parse — exactly as typed.
    ///
    /// The `overrides(for:)` + `setOverrides` pair cannot do that: the reader
    /// drops unparseable entries, so writing its result back erased an
    /// operator's typo AND their valid neighbours in the same pass. Unregistering
    /// an unrelated model used to take that path.
    public static func removeOverride(
        model: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) {
        let key = defaultsKey(for: providerId)
        guard let raw = defaults.string(forKey: key) else { return }
        var kept: [String] = []
        var removed = false
        for entry in splitOverrideEntries(raw) {
            if let parts = splitEntry(entry), parts.model == model {
                removed = true
                continue
            }
            kept.append(entry)
        }
        guard removed else { return }
        if kept.isEmpty {
            defaults.removeObject(forKey: key)
        } else {
            defaults.set(kept.joined(separator: "; "), forKey: key)
        }
    }

    /// Splits `;`, `,`, or newline-separated entries without tearing apart a
    /// comma-grouped integer such as `m=1,048,576`.
    private static func splitOverrideEntries(_ raw: String) -> [String] {
        var entries: [String] = []
        for lineOrSemicolon in raw.split(whereSeparator: { $0 == ";" || $0 == "\n" }) {
            var current: String?
            for piece in lineOrSemicolon.split(separator: ",", omittingEmptySubsequences: false) {
                let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                if let existing = current,
                   trimmed.count == 3,
                   trimmed.allSatisfy({ $0 >= "0" && $0 <= "9" }),
                   let lastChar = existing.last,
                   lastChar >= "0" && lastChar <= "9" {
                    current = existing + "," + trimmed
                } else {
                    if let existing = current, !existing.isEmpty {
                        entries.append(existing)
                    }
                    current = trimmed
                }
            }
            if let existing = current, !existing.isEmpty {
                entries.append(existing)
            }
        }
        return entries
    }

    /// Parses a positive integer window, tolerating ASCII thousands separators
    /// (`1,048,576` or `1_048_576`).
    private static func parseWindowValue(_ raw: String) -> Int? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let direct = Int(trimmed) {
            return direct
        }
        if trimmed.contains(",") {
            let groups = trimmed.split(separator: ",", omittingEmptySubsequences: false)
            if groups.count >= 2,
               let first = groups.first,
               (1...3).contains(first.count),
               first.allSatisfy({ $0 >= "0" && $0 <= "9" }),
               groups.dropFirst().allSatisfy({ $0.count == 3 && $0.allSatisfy({ $0 >= "0" && $0 <= "9" }) }) {
                return Int(groups.joined())
            }
        }
        if trimmed.contains("_") {
            let stripped = trimmed.replacingOccurrences(of: "_", with: "")
            if !stripped.isEmpty, stripped.allSatisfy({ $0 >= "0" && $0 <= "9" }) {
                return Int(stripped)
            }
        }
        return nil
    }

    /// Splits one `model=window` (or `model@window`) entry at its LAST
    /// separator.
    ///
    /// Splitting at the first one made a revision-pinned id unreadable and
    /// un-removable: `model@rev=8192` parted at the `@`, so the model became
    /// `model`, the window became `rev=8192` (which is not a number, so the
    /// entry was dropped), and unregistering `model@rev` could never match it.
    /// The window is the trailing value, so the last separator is the boundary.
    private static func splitEntry(_ entry: String) -> (model: String, value: String)? {
        guard let separator = entry.lastIndex(where: { $0 == "=" || $0 == "@" }) else {
            return nil
        }
        let model = entry[..<separator].trimmingCharacters(in: .whitespacesAndNewlines)
        let value = entry[entry.index(after: separator)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty, !value.isEmpty else { return nil }
        return (model, value)
    }

    /// The window this model is believed to have, or nil when nobody recorded
    /// one. The per-model value wins over the global fallback.
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
        let intFallback = defaults.integer(forKey: globalWindowKey)
        if intFallback >= minimumPlausibleWindow {
            return intFallback
        }
        if let raw = defaults.string(forKey: globalWindowKey),
           let parsed = parseWindowValue(raw),
           parsed >= minimumPlausibleWindow {
            return parsed
        }
        return nil
    }

    /// Whether a send of `promptTokens` should compact first.
    ///
    /// Two things can make a request not fit, and only one of them used to be
    /// checked. The fraction is the "large context" trigger. `reservedTokens` is
    /// the reply budget the same request carries, and a prompt under the
    /// fraction can still overflow the window beside it: 0.8 x a 16k window is
    /// 12.8k tokens, and an 8,192-token reply makes that request 21k against
    /// 16k. Compacting at the fraction alone therefore spent a pre-flight
    /// attempt on a wall it could not clear and left the overflow path to
    /// recover — which is the path that refuses when its own attempt is already
    /// spent.
    ///
    /// A reply budget that is itself as large as the window makes every send
    /// pre-flight and still overflow, however much history is summarized: no
    /// compaction can leave room for a reply that alone fills the model's
    /// context. That is not a compaction problem, so this answers false and the
    /// caller sends the budget `replyBudget(requested:window:)` clamps to the
    /// room the threshold leaves. Without the guard, a 1,024-token model with
    /// the default 8,192-token reply paid for up to three summarizer passes on
    /// every single send.
    public static func shouldCompact(
        promptTokens: Int,
        window: Int?,
        reservedTokens: Int
    ) -> Bool {
        guard let window, window >= minimumPlausibleWindow else { return false }
        let prompt = Double(max(0, promptTokens))
        let reserve = Double(max(0, reservedTokens))
        // Compaction cannot make room for a reply larger than the window.
        guard reserve < Double(window) else { return false }
        if prompt >= Double(window) * autoCompactFraction { return true }
        return prompt + reserve > Double(window)
    }

    /// The reply budget a request may carry and still share the window with a
    /// prompt this app would send.
    ///
    /// `max_tokens` is a ceiling, so lowering it never breaks a turn that would
    /// otherwise have fitted; leaving it above the window guarantees the
    /// request is refused (or truncated) no matter how short the prompt is.
    /// The cap is the room the auto-compaction threshold leaves — 20% of the
    /// window — so the two rules agree: once the prompt is under 80%, the
    /// prompt plus this reply fits, and the pre-flight is only triggered by a
    /// prompt compaction can actually shrink.
    ///
    /// A model with NO recorded window is capped at `unknownWindowReplyCeiling`
    /// rather than passed through. The pass-through was safe only while the
    /// settings slider topped out at a budget most providers accept; it let the
    /// app send an arbitrary `max_tokens` to a provider whose capacity nobody had
    /// written down, which is a 400 (or an expensive success) with no local
    /// evidence behind it. The ceiling is the same number
    /// `maximumOutputTokens(window:)` advertises, so the slider and this clamp
    /// cannot disagree. A caller that asked for no budget (`nil`/`0`) is still
    /// left alone.
    public static func replyBudget(requested: Int, window: Int?) -> Int {
        guard requested > 0 else { return max(0, requested) }
        guard let window, window >= minimumPlausibleWindow else {
            return min(requested, unknownWindowReplyCeiling)
        }
        let room = (Double(window) * (1 - autoCompactFraction)).rounded()
        guard room.isFinite, room >= 1, room < Double(Int.max) else {
            return min(requested, unknownWindowReplyCeiling)
        }
        return min(requested, Int(room))
    }

    /// The Max Output Tokens ceiling Settings should advertise for a cloud
    /// model — the same number `replyBudget` enforces for it.
    ///
    /// Written against `replyBudget`'s own arithmetic on purpose: a slider that
    /// computed its range independently would eventually advertise a budget the
    /// send path silently lowers, which reads as the app ignoring a setting.
    /// An unrecorded (or sub-plausible) window answers with
    /// `unknownWindowReplyCeiling`, exactly as the clamp does.
    public static func maximumOutputTokens(window: Int?) -> Int {
        replyBudget(requested: Int.max, window: window)
    }

    /// The share of the window a prompt occupies, for a status line.
    ///
    /// The window is hand-typed, so it can be as small as 1; `Int.max` tokens
    /// over such a window overflows a Double multiplication into `+inf`, and
    /// the `Int(...)` conversion of either that or a value above `Int.max`
    /// traps. Both are clamped: a 100%-plus figure is the same answer either
    /// way, and the header must never crash over a unit mistake.
    ///
    /// The result is also capped at 100. A conversation carried over from a
    /// larger-window model can genuinely hold more context than the new
    /// model's window, but "150% of 16,000" reads as a broken meter rather than
    /// a fact about the conversation; the token count beside it still shows the
    /// true size. This mirrors `CloudUsage.cacheReadFraction`, which clamps the
    /// same way for the same reason.
    public static func usagePercent(promptTokens: Int, window: Int?) -> Int? {
        guard let window, window > 0 else { return nil }
        let percent = (Double(max(0, promptTokens)) / Double(window) * 100).rounded()
        // `Double(Int.max)` rounds UP to 2^63, so comparing against it and then
        // converting is still a trap; the comparison must be the conversion's
        // guard, not a clamp applied to a value that is already out of range.
        guard percent.isFinite, percent < Double(Int.max) else { return 100 }
        return min(100, Int(percent))
    }

    /// Upper bound on a derived chunk budget. A third of a 128k-token window is
    /// ~170k characters; anything past this is a typo, and the transcript
    /// builder truncates to the budget either way.
    public static let maximumChunkCharacters = 8_000_000

    /// Characters of transcript one summarizer pass may carry.
    ///
    /// Four characters per token is the same estimate the rest of the cloud
    /// path uses. A third of the window leaves room for the digest being folded
    /// in, the pass instructions, and the summary itself.
    ///
    /// The window is a hand-typed value, so the arithmetic is done in `Double`
    /// and clamped before the conversion: `Int(Double(window) * 0.33) * 4`
    /// traps on a window near `Int.max`, and a budget that large is a units
    /// mistake rather than a chunk anybody can send.
    public static func chunkCharacterBudget(window: Int?) -> Int {
        guard let window, window >= minimumPlausibleWindow else { return 24_000 }
        let scaled = (Double(window) * 0.33 * 4).rounded(.down)
        let ceiling = Double(maximumChunkCharacters)
        return max(4_000, Int(min(scaled, ceiling)))
    }

    /// Derives the chunk character budget for one cloud compaction pass after
    /// reserving the fixed occupants of that request's window: the prior
    /// standing summary, the prior recap, the pass instructions (including any
    /// `/compact` focus), the output budget, and prompt framing.
    ///
    /// Returns `0` when those fixed occupants already fill the window so the
    /// caller refuses before sending a guaranteed 400.
    public static func chunkCharacterBudget(
        window: Int?,
        previousSummary: String?,
        previousRecap: String?,
        instructions: String,
        responseTokens: Int
    ) -> Int {
        let base = chunkCharacterBudget(window: window)
        guard let window, window >= minimumPlausibleWindow else { return base }
        let summaryChars = (previousSummary ?? "").trimmingCharacters(in: .whitespacesAndNewlines).count
        let recapChars = (previousRecap ?? "").trimmingCharacters(in: .whitespacesAndNewlines).count
        let reserved = Double(max(0, responseTokens)) * 4.0
            + Double(summaryChars)
            + Double(recapChars)
            + Double(instructions.count)
            + 256.0
        let remaining = (Double(window) * 4.0 - reserved).rounded(.down)
        guard remaining.isFinite, remaining > 0 else { return 0 }
        let cappedRemaining = remaining < Double(Int.max) ? Int(remaining) : Int.max
        return min(base, cappedRemaining)
    }

    /// Output ceiling for a single summarizer pass. Most gateways cap
    /// `max_tokens` at 8–32k and answer 400 above it; the cap's job is avoiding
    /// summary truncation without triggering an impossible-length refusal on
    /// 1M-window models.
    public static let maximumSummaryResponseTokens = 8_000

    /// Tokens one summarizer pass may write.
    ///
    /// Scales from 2,000 up to `maximumSummaryResponseTokens` (10% of the
    /// chunk's estimated tokens, bounded by a quarter of the window for small
    /// models and by the output ceiling for large ones).
    public static func summaryResponseTokens(window: Int?) -> Int {
        let chunkTokens = chunkCharacterBudget(window: window) / 4
        let proportion = min(maximumSummaryResponseTokens, max(2_000, chunkTokens / 10))
        guard let window, window >= minimumPlausibleWindow else { return proportion }
        // The window is hand-typed, so the conversion is guarded the way
        // `usagePercent`'s is: a value at or above `Int.max` is a units mistake,
        // and the proportion is the smaller answer either way.
        let room = Double(window) * 0.25
        guard room.isFinite, room < Double(Int.max) else { return proportion }
        return min(proportion, Int(room))
    }
}

/// The output budget one summarizer request may carry.
///
/// A raw `Int` at the call site is indistinguishable from a constant, and the
/// cap has to scale with the model's window or a long conversation's summary
/// stops at the limit and is refused after being billed. This type exists so
/// the only way to build a summarizer request's budget is from the window the
/// compaction was priced against (`CloudContextPolicy.summaryResponseTokens`),
/// and a caller cannot silently pass a fixed number instead.
public nonisolated struct CloudSummarizerBudget: Equatable, Sendable {
    public let maxTokens: Int

    public init(window: Int?) {
        self.maxTokens = CloudContextPolicy.summaryResponseTokens(window: window)
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
    /// How far before the requested cut a user turn may pull that cut.
    ///
    /// The preference exists so the kept tail opens with a question instead of
    /// mid-answer. It is bounded because an unbounded look-back finds the head
    /// of a long agent run: one user turn followed by two hundred assistant
    /// turns is a single question, and moving the cut all the way back there
    /// evicts a handful of rows, bills a summary, and shrinks nothing. Eight
    /// messages is wider than the alternating shape ever needs.
    public static let maximumUserTurnLookback = 8

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
        // Preferred: the newest user turn within a short look-back at or
        // before the requested cut.
        var userBoundary = target
        var stepsBack = 0
        while userBoundary > 0,
              messages[userBoundary].role != .user,
              stepsBack < Self.maximumUserTurnLookback {
            userBoundary -= 1
            stepsBack += 1
        }
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
    /// Existing digests, instructions, and reply budget already fill this model's
    /// context window, leaving no room for any transcript chunk.
    case digestExceedsWindow
    /// A pass came back with no text at all.
    case emptySummary
    /// A pass did not report a clean stop, so its text never proved it covered
    /// the chunk it was given. That is the output limit, a provider's own
    /// filter, or a reason this app does not recognise — all of which look the
    /// same from here. Committing it would delete work the summary omits.
    case truncatedSummary
    /// Stop was pressed while the summary was being written.
    case cancelled
    /// The provider request failed; the payload is the failure's description,
    /// and `spent` says whether that request was billed. A 4xx refusal (an
    /// expired key, an unknown model, a rate limit) never reached inference and
    /// is not invoiced; a 5xx or a stream that died mid-answer was.
    case provider(String, spent: Bool)

    /// Whether this outcome sent a summarizer request, and was therefore
    /// billed. A refusal that never reached the provider is a precondition and
    /// spends nothing; the pre-flight gate is reopened for those whose reason
    /// the next step could change (see `isRetryableOnNextStep`).
    public var spentRequest: Bool {
        switch self {
        case .conversationTooShort, .nothingToEvict, .conversationUnavailable,
             .noCredential, .noTranscript, .digestExceedsWindow:
            return false
        case .emptySummary, .truncatedSummary, .cancelled:
            return true
        case .provider(_, let spent):
            return spent
        }
    }

    /// The same rule applied to an arbitrary error the summarizer path caught,
    /// so the outcome the chat surface prints and the spend it records cannot
    /// disagree about whether the pass was billed.
    ///
    /// This lives here rather than inline in the catch because the mapping is
    /// the rule: a `.provider(..., spent: false)` built by hand in a test
    /// proves only that the enum stores its own flag. Classifying a real 4xx
    /// through this function is what proves the 4xx-to-unbilled rule.
    public static func fromProviderError(_ error: Error) -> CloudCompactionFailure {
        .provider(error.localizedDescription, spent: CloudTurnFailure.spentRequest(for: error))
    }

    /// Whether the same pre-flight attempt is worth making again on the next
    /// step. A refusal that sent no request and depends only on the history's
    /// SHAPE can change as the run appends rows (an empty tail, too few rows,
    /// nothing renderable), so it gives the attempt back. A key that could not
    /// be read is not re-read at every step, and a vanished conversation ends
    /// the run.
    public var isRetryableOnNextStep: Bool {
        switch self {
        case .conversationTooShort, .nothingToEvict, .noTranscript:
            return true
        case .conversationUnavailable, .noCredential, .digestExceedsWindow,
             .emptySummary, .truncatedSummary, .cancelled, .provider:
            return false
        }
    }

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
        case .digestExceedsWindow:
            return "⚠️ Cannot compact — the existing summary, instructions, and reply budget already fill this model's context window."
        case .emptySummary:
            return "⚠️ \(providerName) returned an empty summary — the conversation was left unchanged."
        case .truncatedSummary:
            return "⚠️ \(providerName)'s summary did not finish cleanly (it hit its output limit or was cut off), so the conversation was left unchanged."
        case .cancelled:
            return "⏹ Compaction stopped — the conversation was left unchanged."
        case .provider(let description, _):
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
        case .digestExceedsWindow:
            return "the existing summary, instructions, and reply budget already fill this model's context window"
        case .emptySummary:
            return "\(providerName) returned an empty summary"
        case .truncatedSummary:
            return "\(providerName)'s summary did not finish cleanly before it covered the conversation"
        case .cancelled:
            return "the summary was stopped"
        case .provider(let description, _):
            return "summarizing failed: \(description)"
        }
    }
}

// MARK: - Compaction gate

/// The refusal a summarizer pass's completion earns, if any.
///
/// This is the one branch where being wrong is DESTRUCTIVE: an accepted summary
/// replaces the messages it was written from, so a pass that came back empty,
/// or without a clean finish it can be trusted over, must be refused rather
/// than committed. It lives here, as a pure function of the completion, because
/// the call site is a closure inside the chat view's turn loop and that branch
/// cannot be reached from a test.
public nonisolated enum CloudCompactionGate {
    public static func refusal(for completion: CloudTextCompletion) -> CloudCompactionFailure? {
        if completion.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .emptySummary
        }
        // The ALLOWLIST (`finishedCleanly`), not "not `length`": an unknown
        // reason is exactly what a dialect spelling `MAX_TOKENS` or a
        // provider-side `content_filter` stop looks like, and either one
        // committed here would delete messages the summary never covered.
        guard completion.finishedCleanly else { return .truncatedSummary }
        return nil
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
        let (sum, overflow) = max(0, promptTokens).addingReportingOverflow(max(0, generatedTokens))
        return overflow ? Int.max : sum
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
    /// Whether `promptTokens` sums the billed prompts of several requests
    /// rather than describing one request's context. A run total has to say
    /// so: the same figure beside the header's context pill otherwise reads
    /// as the size of the conversation.
    public let countsBilledInput: Bool

    public init(
        usage: CloudUsage?,
        estimatedPromptTokens: Int,
        outputTokens: Int,
        cost: Double?
    ) {
        // A missing completion count is as much an estimate as a missing prompt
        // count: the output figure then comes from the app's character
        // estimate, and the suffix has to say so.
        self.wasEstimated = usage?.promptTokens == nil || usage?.completionTokens == nil
        let prompt = max(0, usage?.promptTokens ?? estimatedPromptTokens)
        self.promptTokens = prompt
        // Cached input is a SUBSET of the prompt, so the two columns have to
        // agree: pricing clamps the cached share to what was billed, and a
        // provider that reports more cached than prompt tokens (some gateways
        // count a cache write in both) otherwise printed "1,000 in (5,000
        // cached)" beside a correctly discounted price.
        self.cachedTokens = min(max(0, usage?.cachedTokens ?? 0), prompt)
        self.outputTokens = max(0, usage?.completionTokens ?? outputTokens)
        self.cost = cost
        self.countsBilledInput = false
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
        let prompt = max(0, promptTokens)
        self.promptTokens = prompt
        // Same subset rule as the single-request form: the cached column can
        // never say more than the prompt it is part of.
        self.cachedTokens = min(max(0, cachedTokens), prompt)
        self.outputTokens = max(0, outputTokens)
        self.cost = cost
        self.wasEstimated = wasEstimated
        self.countsBilledInput = true
    }

    /// `1,204 in (1,000 cached) / 56 out · $0.0031`, or the estimate-marked
    /// form when the provider reported no usage. A run total says `billed in`
    /// instead: it sums several requests' prompts and is not the conversation's
    /// context size. The cache figure is always phrased as part of that prompt
    /// (`1,000 of it cached`) rather than beside it, so "billed" and "cached"
    /// cannot read as two different totals.
    public var statusSuffix: String {
        var parts: [String] = []
        if let promptTokens {
            let number = Self.format(promptTokens)
            let inputLabel = countsBilledInput ? " billed in" : " in"
            if cachedTokens > 0 {
                let cachedLabel = countsBilledInput ? " of it cached" : " cached"
                parts.append("\(number)\(inputLabel) (\(Self.format(cachedTokens))\(cachedLabel))")
            } else {
                parts.append("\(number)\(inputLabel)")
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

/// One cloud run's billed totals, accumulated request by request.
///
/// The alternative is five `+=` lines at every call site, which is how a rule
/// about what counts (a retry, a summarizer pass, a request that was sent and
/// never answered) reaches the status line of one path and not another. The
/// rules live here instead: every request the provider processed contributes
/// its prompt, its cached share and its output; a request whose usage never
/// arrived contributes the app's own estimate and marks the whole total
/// `(est.)`; and a cost joins the total only when there is one, so an estimate
/// alone never invents a price.
public nonisolated struct CloudRunTotals: Equatable, Sendable {
    public private(set) var promptTokens: Int = 0
    public private(set) var cachedTokens: Int = 0
    public private(set) var outputTokens: Int = 0
    public private(set) var cost: Double?
    public private(set) var wasEstimated = false

    public init() {}

    /// Whether any request contributed anything. The failure path uses this to
    /// decide whether a "run total so far" line would say anything at all.
    public var isEmpty: Bool {
        promptTokens == 0 && outputTokens == 0 && cost == nil
    }

    /// Adds one request's already-resolved columns.
    ///
    /// `estimated` is the caller's answer to "did any column here come from the
    /// app's estimate", because only the caller knows which fallbacks it used.
    /// A non-positive or non-finite cost is ignored rather than credited: no
    /// provider pays the user.
    public mutating func add(
        promptTokens: Int,
        cachedTokens: Int = 0,
        outputTokens: Int,
        cost: Double?,
        estimated: Bool
    ) {
        let prompt = max(0, promptTokens)
        self.promptTokens = Self.clampedSum(self.promptTokens, prompt)
        // The cached share is part of the prompt, so it is clamped to it here
        // too: a total that summed an over-reported column would print a cache
        // figure larger than the input it belongs to, and an `Int.max` column
        // would trap the display's subtraction.
        self.cachedTokens = Self.clampedSum(
            self.cachedTokens,
            min(max(0, cachedTokens), prompt)
        )
        self.outputTokens = Self.clampedSum(self.outputTokens, max(0, outputTokens))
        if estimated { wasEstimated = true }
        guard let cost, cost.isFinite, cost > 0 else { return }
        // A run total is sums of sums; a gateway reporting absurd columns must
        // saturate rather than trap the status line.
        let total = (self.cost ?? 0) + cost
        self.cost = total.isFinite ? total : Double.greatestFiniteMagnitude
    }

    /// Adds two token counts, saturating at `Int.max`.
    ///
    /// Every column here comes from a provider's usage block, and nothing
    /// validates an integer beyond its sign. `a += b` traps on overflow, so a
    /// hostile or broken gateway could crash the run's summary line.
    private static func clampedSum(_ left: Int, _ right: Int) -> Int {
        let (sum, overflow) = left.addingReportingOverflow(right)
        return overflow ? Int.max : sum
    }

    /// Adds one request from its provider usage block, with the app's estimates
    /// standing in for the columns it omitted.
    ///
    /// The mapping is `CloudTurnReport(usage:...)`'s, so a run total and that
    /// request's own report cannot disagree about which figure was measured.
    /// `cost` is passed in because pricing belongs to the caller: a request with
    /// no usable usage block has no cost, and one that was sent but never
    /// answered is priced from its estimate deliberately.
    public mutating func add(
        usage: CloudUsage?,
        estimatedPromptTokens: Int,
        estimatedOutputTokens: Int,
        cost: Double?
    ) {
        let one = CloudTurnReport(
            usage: usage,
            estimatedPromptTokens: estimatedPromptTokens,
            outputTokens: estimatedOutputTokens,
            cost: cost
        )
        add(
            promptTokens: one.promptTokens ?? 0,
            cachedTokens: one.cachedTokens,
            outputTokens: one.outputTokens,
            cost: cost,
            estimated: one.wasEstimated
        )
    }

    /// The run's summary line: several requests' billed prompts, so it says
    /// `billed in` rather than reading as one conversation's context size.
    public var report: CloudTurnReport {
        CloudTurnReport(
            promptTokens: promptTokens,
            cachedTokens: cachedTokens,
            outputTokens: outputTokens,
            cost: cost,
            wasEstimated: wasEstimated
        )
    }
}
