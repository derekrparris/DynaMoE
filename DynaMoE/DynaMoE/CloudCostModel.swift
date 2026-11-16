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

    /// Below this a published `max output` is a units mistake or a mis-keyed
    /// field, not a reply limit. The window has a floor for the same reason: a
    /// catalog is machine-written, and one bad number must not silently become
    /// the app's policy. Nothing real publishes a cap this small — the smallest
    /// chat limits in the wild are in the hundreds — while a `limit.output` of
    /// `1` would truncate every answer to a single token.
    public static let minimumPlausibleOutputLimit = 64

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
    /// It is what the generic (cloud) profile's coder default carries —
    /// `GenerationProfileSettings.defaultFor` reads this constant rather than
    /// restating the number, so an unconfigured provider gets a coherent budget
    /// rather than a value that is then clamped by a different constant, and the
    /// two cannot drift apart unnoticed.
    public static let unknownWindowReplyCeiling = 16_384

    public static func defaultsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_context"
    }

    /// Every entry for one provider EXACTLY as typed, keyed by model id —
    /// including values this build cannot read as a window.
    ///
    /// `overrides(for:)` is the PARSE view: it drops what it cannot read, which
    /// is right for the send path and wrong for the editor. A row that read only
    /// the parse view rendered a stored `zen-model=12k` as an empty field, so the
    /// typo was invisible AND the next Save — resolving "no stored value" from
    /// that empty field — took the clear branch and deleted it. The editor needs
    /// the text to show, and the save path needs it to refuse.
    ///
    /// Last entry wins for a duplicated model id, and that is deliberately the
    /// LAST ENTRY rather than the last readable one: this is the view the editor
    /// shows and re-submits, so it has to be the text the operator's file ends
    /// with. `overrides(for:)` picks differently on purpose — see its note.
    public static func rawOverrides(
        for providerId: String,
        defaults: UserDefaults = .standard
    ) -> [String: String] {
        let raw = defaults.string(forKey: defaultsKey(for: providerId)) ?? ""
        var result: [String: String] = [:]
        for entry in splitOverrideEntries(raw) {
            guard let parts = splitEntry(entry) else { continue }
            result[parts.model] = parts.value
        }
        return result
    }

    /// One provider's per-model windows, parsed from `model=window` entries
    /// separated by `;`, `,` or newline. Tolerates `model@window`, grouped
    /// integers (`1,048,576`), and bare integers.
    ///
    /// Scans EVERY entry and keeps the last one this build can READ, which is
    /// not the same entry `rawOverrides(for:)` shows when a duplicated model id
    /// ends in junk (`a=200000; a=12k`): the raw view reports `12k` because that
    /// is what the file ends with and the editor has to show it, while this view
    /// reports 200,000 because that is the only capacity anyone recorded. Parsing
    /// the raw view's pick instead would have dropped the model entirely and, on
    /// the send path, silently swapped a known window for the unrecorded-window
    /// ceiling — the under-compaction failure `window(for:providerId:)` warns
    /// about. The two views therefore differ on which DUPLICATE they pick and on
    /// nothing else.
    ///
    /// Only `setOverride` writes this key in the app (it dedupes), so a duplicate
    /// only reaches here from an out-of-band `defaults write` or a hand-edited
    /// sync. It is cheap to read either way.
    public static func overrides(
        for providerId: String,
        defaults: UserDefaults = .standard
    ) -> [String: Int] {
        let raw = defaults.string(forKey: defaultsKey(for: providerId)) ?? ""
        var result: [String: Int] = [:]
        for entry in splitOverrideEntries(raw) {
            guard let parts = splitEntry(entry),
                  let window = parseWindowValue(parts.value),
                  window >= minimumPlausibleWindow else { continue }
            result[parts.model] = window
        }
        return result
    }

    /// Writes a provider's entire window map in one call.
    ///
    /// TEST-ONLY. It is the bulk form of `setOverride`, so it rewrites the key
    /// from a parsed map and therefore cannot preserve an entry this build
    /// cannot read — the exact data loss `setOverride` and `removeOverride`
    /// exist to avoid. Production has no caller and must not gain one: a save
    /// goes through `setOverride`, a removal through `removeOverride`. It stays
    /// so a test can set up a known map in one line.
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
    ///
    /// Returns whether anything changed, so a caller that pairs this with a write
    /// can stay a true no-op instead of rewriting identical bytes.
    @discardableResult
    public static func removeOverride(
        model: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) -> Bool {
        let key = defaultsKey(for: providerId)
        guard let raw = defaults.string(forKey: key) else { return false }
        var kept: [String] = []
        var removed = false
        for entry in splitOverrideEntries(raw) {
            if let parts = splitEntry(entry), parts.model == model {
                removed = true
                continue
            }
            kept.append(entry)
        }
        guard removed else { return false }
        if kept.isEmpty {
            defaults.removeObject(forKey: key)
        } else {
            defaults.set(kept.joined(separator: "; "), forKey: key)
        }
        return true
    }

    /// Stores ONE model's window, leaving every other entry — including one this
    /// build could not parse — exactly as typed. `window == nil` removes it.
    ///
    /// The `overrides(for:)` + `setOverrides` pair is the wrong tool for a
    /// single-model save for the same reason it was wrong for unregister: the
    /// reader drops unparseable entries, so writing its result back deletes an
    /// operator's typo AND every valid neighbour the same pass could not
    /// represent. Saving one row must not erase another.
    ///
    /// The return answers ONE question — does the store now hold the requested
    /// value — and both directions read it the same way. A save of the value
    /// already stored is `true` (nothing to write, but the request is satisfied),
    /// and so is a clear of a model that had no entry (the requested state,
    /// absent, is what the store holds). `false` therefore means only that the
    /// request could not be honoured at all: an empty model id, or a
    /// sub-plausible window the reader would drop and so read back as absent.
    ///
    /// `removeOverride` keeps its own narrower meaning — did the bytes change —
    /// because it is also used to prune, where "nothing changed" is the useful
    /// answer. Callers that ask "did my save land" want this one.
    @discardableResult
    public static func setOverride(
        model: String,
        window: Int?,
        providerId: String,
        defaults: UserDefaults = .standard
    ) -> Bool {
        // The model id is trimmed the way `splitEntry` trims a stored entry's
        // model, and then used in its trimmed form. An untrimmed `" a "` passed
        // the emptiness check, then compared unequal to the `a` the split had
        // trimmed, so a save appended a duplicate instead of replacing in place —
        // the very duplicate this writer exists to collapse.
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else { return false }
        guard let window else {
            // A clear is satisfied by the entry already being absent. The write
            // is still skipped when there is nothing to remove, which is what
            // `removeOverride` reports and what keeps an empty key from being
            // created.
            removeOverride(model: model, providerId: providerId, defaults: defaults)
            return true
        }
        guard window >= minimumPlausibleWindow else { return false }

        let key = defaultsKey(for: providerId)
        let raw = defaults.string(forKey: key) ?? ""
        let replacement = "\(model)=\(window)"
        var kept: [String] = []
        var replaced = false
        for entry in splitOverrideEntries(raw) {
            guard let parts = splitEntry(entry), parts.model == model else {
                kept.append(entry)
                continue
            }
            // First match takes the new value in place; a later duplicate is
            // dropped rather than left to win the reader's last-one-wins merge.
            if !replaced {
                kept.append(replacement)
                replaced = true
            }
        }
        if !replaced { kept.append(replacement) }

        // A save that stores what is already stored is a success with nothing
        // to do: rewriting the key would churn defaults (and every observer of
        // the domain) for a value nobody changed. `true` because the store now
        // holds the requested value, which is what the return answers.
        let updated = kept.joined(separator: "; ")
        guard updated != raw else { return true }
        defaults.set(updated, forKey: key)
        return true
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
    ///
    /// Public so the Cloud tab's Save button validates with the READER's parser
    /// instead of a second one. A view that worked the number out for itself
    /// accepted values this reader drops, which stored a window that then read
    /// back as absent — the save looked like it landed and the ceiling never
    /// moved.
    public static func parseWindowValue(_ raw: String) -> Int? {
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

    /// Whether `raw` is a window this build can READ — the exact rule
    /// `overrides(for:)` applies, exposed so a UI that reports on stored text uses
    /// the reader's own test instead of a second copy of it.
    ///
    /// The Save button already validated with `parseWindowValue` + a comparison
    /// against `minimumPlausibleWindow`, and the Cloud tab's row was about to
    /// grow a third copy to decide whether the field's text was usable. Three
    /// statements of "readable" is how the validator and the reader drift: a
    /// button that accepts a value the reader drops stores a window that reads
    /// back as absent, while the ceiling never moves.
    public static func isUsableWindow(_ raw: String?) -> Bool {
        guard let raw, let window = parseWindowValue(raw) else { return false }
        return window >= minimumPlausibleWindow
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
    /// or published one. The spec's hand-recorded value wins over the
    /// catalog-observed one, which wins over the global fallback.
    ///
    /// An entry point to `windowSource(recorded:observed:fallback:)`, not a
    /// second rule: a spec already carries the recorded value, so that read is
    /// skipped, and the resolution under it is the one every other reader
    /// shares.
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
        guard let spec else {
            return windowSource(recorded: nil, defaults: defaults).window
        }
        return windowSource(
            recorded: spec.contextWindowTokens,
            observed: observedLimit(
                forModelId: spec.providerModelId,
                providerId: spec.providerId,
                defaults: defaults
            )?.contextWindow,
            defaults: defaults
        ).window
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
        windowSource(forModelId: modelId, providerId: providerId, defaults: defaults).window
    }

    /// What one model's entry in a provider's `/models` catalog said about its
    /// capacity.
    ///
    /// Kept apart from the hand-recorded window on purpose. A recorded window is
    /// an operator's fact about the model and outranks this; these values are
    /// what the provider publishes and a sync may refresh them. Merging the two
    /// would make a re-sync either overwrite an operator's correction or leave a
    /// stale number indistinguishable from a hand-typed one (`#95` review's
    /// "a re-sync must not delete a hand-typed window").
    public struct ModelLimits: Codable, Equatable, Sendable {
        /// The model's total context window, when the catalog publishes one.
        public let contextWindow: Int?
        /// The model's own maximum output, when the catalog publishes one. It
        /// binds before the window fraction on models whose output cap is a
        /// smaller share of the window than this app reserves.
        public let maxOutput: Int?

        public init(contextWindow: Int? = nil, maxOutput: Int? = nil) {
            self.contextWindow = contextWindow
            self.maxOutput = maxOutput
        }

        public var isEmpty: Bool { contextWindow == nil && maxOutput == nil }
    }

    /// The UserDefaults key holding one provider's observed catalog limits,
    /// keyed by upstream model id.
    ///
    /// JSON rather than the `model=window` text the hand-recorded store uses:
    /// model ids may contain `=`, `;` and `,`, and a separator format would have
    /// to invent escaping for them.
    public static func observedLimitsKey(for providerId: String) -> String {
        "dynamoe_cloud_\(providerId)_model_limits"
    }

    /// Every observed limit one provider's last sync recorded. A corrupt or
    /// foreign payload answers an empty map rather than failing a turn.
    public static func observedLimits(
        forProviderId providerId: String,
        defaults: UserDefaults = .standard
    ) -> [String: ModelLimits] {
        guard let data = defaults.data(forKey: observedLimitsKey(for: providerId)) else { return [:] }
        return (try? JSONDecoder().decode([String: ModelLimits].self, from: data)) ?? [:]
    }

    /// Replaces one provider's observed limits. A sync passes only the models it
    /// registered, so a model the catalog stopped serving loses its observed
    /// facts with the registration. Never touches the hand-recorded windows.
    public static func setObservedLimits(
        _ limits: [String: ModelLimits],
        forProviderId providerId: String,
        defaults: UserDefaults = .standard
    ) {
        let cleaned = limits.filter { !$0.key.isEmpty && !$0.value.isEmpty }
        guard !cleaned.isEmpty else {
            defaults.removeObject(forKey: observedLimitsKey(for: providerId))
            return
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(cleaned) else { return }
        guard data != defaults.data(forKey: observedLimitsKey(for: providerId)) else { return }
        defaults.set(data, forKey: observedLimitsKey(for: providerId))
    }

    /// Records an output cap a provider's own refusal named, merging it into
    /// the provider's observed limits.
    ///
    /// A catalog may publish a window without an output cap, so the first time
    /// the app learns that cap can be a 400 on a budget derived from that
    /// window. That is capacity evidence about this model from the provider
    /// itself, and keeping it only for the retry would send the same rejected
    /// number on the next turn. An existing smaller cap is kept: the point is
    /// to stop asking for too much, not to talk the provider up.
    ///
    /// A sub-plausible value is ignored rather than recorded, because that is
    /// the same units-mistake rule the catalog path applies. Returns whether
    /// the stored map changed.
    @discardableResult
    public static func recordObservedOutputLimit(
        _ limit: Int,
        forModelId modelId: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) -> Bool {
        guard limit >= minimumPlausibleOutputLimit else { return false }
        var all = observedLimits(forProviderId: providerId, defaults: defaults)
        let existing = all[modelId]
        let merged = min(existing?.maxOutput ?? limit, limit)
        guard existing?.maxOutput != merged else { return false }
        all[modelId] = ModelLimits(
            contextWindow: existing?.contextWindow,
            maxOutput: merged
        )
        setObservedLimits(all, forProviderId: providerId, defaults: defaults)
        return true
    }

    /// One model's observed limits, or nil when the provider's catalog has
    /// published none. The upstream model id is matched exactly.
    public static func observedLimit(
        forModelId modelId: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) -> ModelLimits? {
        observedLimits(forProviderId: providerId, defaults: defaults)[modelId]
    }

    /// Removes one model's observed limits, leaving every other entry. Used
    /// when a model is unregistered so a later re-add cannot inherit facts the
    /// endpoint no longer published.
    public static func removeObservedLimit(
        model: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) {
        var all = observedLimits(forProviderId: providerId, defaults: defaults)
        guard all.removeValue(forKey: model) != nil else { return }
        setObservedLimits(all, forProviderId: providerId, defaults: defaults)
    }

    /// Where a model's resolved window came from.
    ///
    /// The number alone cannot say it, and the difference is not cosmetic: a
    /// window the model RECORDED is a fact about that model, a window its
    /// provider PUBLISHED is a fact about that provider's catalog, and a window
    /// the single global fallback supplied belongs to whichever model the
    /// operator had in mind when they set it. `maximumOutputTokens` prices all
    /// three the same way, but a row that explains the ceiling has to name the
    /// right one or it asserts a fact the store does not hold (`#95` review).
    public enum WindowSource: Equatable, Sendable {
        /// The model's own recorded window.
        case model(Int)
        /// The provider's catalog published this window. A sync refreshes it; a
        /// hand-recorded value outranks it.
        case observed(Int)
        /// No per-model value; the global fallback supplied this one.
        case global(Int)
        /// Nobody recorded or published anything, so there is no window to price
        /// against.
        case none

        /// The window the send path will actually use, if any.
        public var window: Int? {
            switch self {
            case .model(let window), .observed(let window), .global(let window): return window
            case .none: return nil
            }
        }
    }

    /// One model's window together with where it came from.
    ///
    /// Written against the same two reads `window(forModelId:providerId:)` makes,
    /// and the number it carries is that function's answer because
    /// `window(forModelId:)` delegates here: the two cannot disagree about which
    /// window applies, only about how to describe it.
    public static func windowSource(
        forModelId modelId: String,
        providerId: String,
        defaults: UserDefaults = .standard
    ) -> WindowSource {
        windowSource(
            recorded: overrides(for: providerId, defaults: defaults)[modelId],
            observed: observedLimit(forModelId: modelId, providerId: providerId, defaults: defaults)?.contextWindow,
            defaults: defaults
        )
    }

    /// The ONE rule every window reader shares: a recorded per-model value wins
    /// when this build can read it, else the provider's own published window,
    /// else the single global fallback, else there is no window.
    ///
    /// `window(for:)` (a spec already in hand) and
    /// `windowSource(forModelId:providerId:)` (a model id read from storage) are
    /// two ways IN to this function, not two resolutions expected to agree. They
    /// used to be two copies of this rule, which is how a display reader drifts
    /// from the send path it is meant to report.
    public static func windowSource(
        recorded: Int?,
        observed: Int? = nil,
        defaults: UserDefaults = .standard
    ) -> WindowSource {
        windowSource(
            recorded: recorded,
            observed: observed,
            fallback: globalWindow(defaults: defaults)
        )
    }

    /// The same rule with the global fallback already read, for a caller that
    /// resolves many models in one pass (the settings list) and must not
    /// re-parse the same default per row. `fallback` is taken as the caller
    /// read it, which `globalWindow` already filters for plausibility.
    public static func windowSource(
        recorded: Int?,
        observed: Int? = nil,
        fallback: Int?
    ) -> WindowSource {
        if let recorded, recorded >= minimumPlausibleWindow {
            return .model(recorded)
        }
        if let observed, observed >= minimumPlausibleWindow {
            return .observed(observed)
        }
        if let fallback {
            return .global(fallback)
        }
        return .none
    }

    /// The single global fallback value, or nil when none is set. Exposed so a
    /// UI row can say WHICH window it is reporting against instead of implying
    /// the model has none.
    public static func globalWindow(defaults: UserDefaults = .standard) -> Int? {
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

    /// The raw window facts for one cloud address, as one comparable string.
    ///
    /// This is the identity a SwiftUI `task(id:)` needs to re-resolve a ceiling
    /// when a window is edited in a different pane of the same window: the
    /// provider's stored entries EXACTLY as typed plus the global fallback.
    ///
    /// It carries the raw text rather than the parsed values on purpose, so a
    /// change the reader cannot parse still moves the identity. Correcting a typo
    /// therefore re-runs a resolution that lands on the same number — a wasted
    /// lookup, not a wrong answer — where the parse view would have made the
    /// correction invisible to a ceiling the next send already prices
    /// differently. Nil for a non-cloud address, which has no such facts.
    ///
    /// `defaults` is injectable for the same reason `setOverride` and
    /// `globalWindow` take one: this is the only part of the ceiling logic that
    /// otherwise reads the process domain directly, and an untestable branch is
    /// the one that rots.
    public static func windowFacts(
        forAddress address: String?,
        defaults: UserDefaults = .standard
    ) -> String? {
        guard let address,
              let parsed = CloudModelAddress.parseModelId(address)
                ?? CloudModelAddress.parsePath(address) else { return nil }
        // The observed limits are stored as JSON DATA, not as a string: reading
        // them through `string(forKey:)` answered nil and left this identity
        // blind to a sync that had just moved the ceiling.
        let observed = defaults.data(forKey: observedLimitsKey(for: parsed.provider.id))
            .map { String(decoding: $0, as: UTF8.self) } ?? ""
        return [
            defaults.string(forKey: defaultsKey(for: parsed.provider.id)) ?? "",
            observed,
            globalWindow(defaults: defaults).map(String.init) ?? ""
        ].joined(separator: "|")
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

    /// Why a reply budget is the number it is.
    ///
    /// The number alone cannot explain a truncation: `16,384` can be the user's
    /// setting, the unrecorded-window clamp, 20% of a small window, or a model's
    /// published output cap, and only one of those can be raised from Settings.
    /// The old notice named that one control unconditionally, so a model whose
    /// cap came from an unrecorded window was told to raise a setting already at
    /// its maximum. Carrying the binding lets the notice name the control that
    /// can actually move the number.
    public struct ReplyBudget: Equatable, Sendable {
        public enum Binding: Equatable, Sendable {
            /// The caller's own Max Output Tokens setting is the smallest cap.
            case setting
            /// No context window is recorded, so the conservative clamp applies.
            case unrecordedWindow
            /// 20% of the model's context window is the smallest cap.
            case windowFraction
            /// The catalog publishes a max output that binds: it is either
            /// smaller than the window room, or (with no usable window) the
            /// only capacity fact the app has.
            case outputLimit
            /// The provider refused a larger reply budget without naming a
            /// number. This turn was lowered; the saved setting was not, so
            /// the Settings control cannot move the number the provider
            /// already rejected.
            case providerRefused
        }

        public let requested: Int
        public let window: Int?
        public let outputLimit: Int?
        public let effective: Int
        public let binding: Binding
    }

    /// The reply budget a request may carry and still share the window with a
    /// prompt this app would send.
    ///
    /// `max_tokens` is a ceiling, so lowering it never breaks a turn that would
    /// otherwise have fitted; leaving it above the window guarantees the
    /// request is refused (or truncated) no matter how short the prompt is.
    /// The cap is the room the auto-compaction threshold leaves, 20% of the
    /// window, so the two rules agree: once the prompt is under 80%, the
    /// prompt plus this reply fits, and the pre-flight is only triggered by a
    /// prompt compaction can actually shrink.
    ///
    /// A model with NO recorded or published window is capped at
    /// `unknownWindowReplyCeiling` rather than passed through. The pass-through
    /// was safe only while the settings slider topped out at a budget most
    /// providers accept; it let the app send an arbitrary `max_tokens` to a
    /// provider whose capacity nobody had written down, which is a 400 (or an
    /// expensive success) with no local evidence behind it. The ceiling is the
    /// same number `maximumOutputTokens(window:)` advertises, so the slider and
    /// this clamp cannot disagree.
    ///
    /// A model whose catalog publishes its own `max output` is capped by that
    /// too: the window fraction is this app's reserve, not a promise the model
    /// will accept a reply longer than its own published limit.
    ///
    /// With no usable window, a published output limit stands IN for the
    /// missing window rather than being clamped by `unknownWindowReplyCeiling`.
    /// That ceiling exists for the ABSENCE of evidence — it stops the app
    /// sending an arbitrary `max_tokens` to a provider whose capacity nobody
    /// recorded — and a published output limit is evidence about this model.
    /// Clamping it to the ceiling made the catalog's one useful number change
    /// nothing for an endpoint that publishes output caps without context
    /// windows. A sub-plausible published value is treated as absent, because
    /// that is a units mistake and not capacity.
    ///
    /// A caller that asked for no budget (`nil`/`0`) is still left alone.
    public static func replyBudgetResolution(
        requested: Int,
        window: Int?,
        outputLimit: Int? = nil
    ) -> ReplyBudget {
        guard requested > 0 else {
            return ReplyBudget(
                requested: requested,
                window: window,
                outputLimit: outputLimit,
                effective: max(0, requested),
                binding: .setting
            )
        }

        // Constraints are evaluated before the setting, and a TIE keeps the
        // constraint: when the setting equals the ceiling, raising the setting
        // cannot move the effective number, so naming it would repeat the dead
        // end this type exists to remove.
        var effective = unknownWindowReplyCeiling
        var binding = ReplyBudget.Binding.unrecordedWindow
        var windowBinds = false
        if let window, window >= minimumPlausibleWindow {
            let room = (Double(window) * (1 - autoCompactFraction)).rounded()
            if room.isFinite, room >= 1, room < Double(Int.max) {
                effective = Int(room)
                binding = .windowFraction
                windowBinds = true
            }
        }
        if let outputLimit, outputLimit >= minimumPlausibleOutputLimit {
            if !windowBinds || outputLimit < effective {
                effective = outputLimit
                binding = .outputLimit
            }
        }
        if requested < effective {
            effective = requested
            binding = .setting
        }
        return ReplyBudget(
            requested: requested,
            window: window,
            outputLimit: outputLimit,
            effective: effective,
            binding: binding
        )
    }

    /// The plain budget, for callers that do not need the explanation.
    public static func replyBudget(
        requested: Int,
        window: Int?,
        outputLimit: Int? = nil
    ) -> Int {
        replyBudgetResolution(
            requested: requested,
            window: window,
            outputLimit: outputLimit
        ).effective
    }

    /// A lower reply budget a provider's own rejection just made necessary.
    public struct ReplyBudgetRecovery: Equatable, Sendable {
        /// The budget the retry should carry, with its binding recomputed.
        public let budget: ReplyBudget
        /// The output cap the provider named, when the body stated one. This
        /// is capacity evidence about the model and may be recorded.
        public let publishedLimit: Int?

        public init(budget: ReplyBudget, publishedLimit: Int?) {
            self.budget = budget
            self.publishedLimit = publishedLimit
        }
    }

    /// The smaller budget one provider rejection justifies, or nil when the
    /// body is not about the reply budget or no smaller number is available.
    ///
    /// One retry only: this never walks a provider's limit down in steps. A
    /// body that names the limit is used exactly; one that only says "too
    /// large" halves the budget, floored by the unrecorded-window ceiling, so
    /// a rejected 200,000 falls back to the conservative number the app used
    /// before capacity was known rather than to an arbitrary fraction of it.
    ///
    /// The returned `ReplyBudget` is rebuilt through
    /// `replyBudgetResolution` rather than constructed by hand, so the window
    /// and output-limit arithmetic is the send path's own. A rejection that
    /// names no number is marked `.providerRefused`: the turn was lowered, not
    /// the saved setting, and the notice must not send the reader to a control
    /// that already holds the rejected number.
    public static func replyBudgetRecovery(
        afterRejectionOf current: ReplyBudget,
        body: String
    ) -> ReplyBudgetRecovery? {
        guard CloudTurnFailure.indicatesOutputBudgetRejection(body),
              current.effective > 1 else { return nil }
        let named = publishedOutputLimit(in: body).flatMap {
            $0 < current.effective ? $0 : nil
        }
        let fallback = min(current.effective / 2, unknownWindowReplyCeiling)
        let requested = max(1, min(named ?? fallback, current.effective - 1))
        guard requested < current.effective else { return nil }
        let limits = [current.outputLimit, named].compactMap { $0 }
        let resolved = replyBudgetResolution(
            requested: requested,
            window: current.window,
            outputLimit: limits.min()
        )
        guard resolved.effective < current.effective else { return nil }
        let budget = named == nil
            ? ReplyBudget(
                requested: resolved.requested,
                window: resolved.window,
                outputLimit: resolved.outputLimit,
                effective: resolved.effective,
                binding: .providerRefused
            )
            : resolved
        return ReplyBudgetRecovery(budget: budget, publishedLimit: named)
    }

    /// The output-token cap a provider's rejection names, if it names one.
    ///
    /// Only called after `CloudTurnFailure.indicatesOutputBudgetRejection`
    /// says the body is about the reply budget, so each pattern can assume
    /// that context. The spellings covered are the ones the app has seen and
    /// the ones the OpenAI/Anthropic/OpenRouter/Groq families produce:
    /// `max_tokens: 200000 > 64000`, `at most 4096`, `must be less than or
    /// equal to 32768`, `valid range … [1, 16384]`, and `cannot exceed 8192`.
    ///
    /// A number from a model name is never taken: every pattern requires its
    /// own cue word or comparison immediately before the digits, and the
    /// plausibility floor rejects the small numbers that a model name leaks
    /// (`claude-sonnet-4-6`, `gpt-4o`). Returning nil is not a failure — the
    /// caller halves the budget instead.
    public static func publishedOutputLimit(in summary: String) -> Int? {
        let patterns = [
            #"(?:max_tokens|max_completion_tokens)[^\n]{0,80}>\s*(\d[\d,]*)"#,
            #"(?:at most|up to|cannot exceed|not exceed|exceed(?:s|ing)?|less than(?: or equal to)?|<=|≤)\s*(\d[\d,]*)"#,
            #"(?:valid range[^0-9]{0,40}?\[\s*\d[\d,]*\s*,\s*)(\d[\d,]*)"#,
            #"(?:maximum|max)\s+(?:allowed\s+)?(?:number\s+of\s+)?(?:output\s+)?tokens?[^0-9]{0,40}?(\d[\d,]*)"#,
            #"(?:output\s+token\s+limit|output\s+limit|max(?:imum)?\s+output(?:\s+tokens?)?)[^0-9]{0,40}?(\d[\d,]*)"#
        ]
        let range = NSRange(summary.startIndex..., in: summary)
        var found: [Int] = []
        for pattern in patterns {
            guard let expression = try? NSRegularExpression(
                pattern: pattern,
                options: [.caseInsensitive]
            ) else { continue }
            for match in expression.matches(in: summary, options: [], range: range) {
                guard match.numberOfRanges > 1,
                      let captured = Range(match.range(at: 1), in: summary) else { continue }
                let digits = String(summary[captured])
                    .replacingOccurrences(of: ",", with: "")
                guard let value = Int(digits),
                      (minimumPlausibleOutputLimit...10_000_000).contains(value) else { continue }
                found.append(value)
            }
        }
        return found.min()
    }

    /// The Max Output Tokens ceiling Settings should advertise for a cloud
    /// model — the same number `replyBudget` enforces for it.
    ///
    /// Written against `replyBudget`'s own arithmetic on purpose: a slider that
    /// computed its range independently would eventually advertise a budget the
    /// send path silently lowers, which reads as the app ignoring a setting.
    /// An unrecorded (or sub-plausible) window answers with
    /// `unknownWindowReplyCeiling`, exactly as the clamp does.
    public static func maximumOutputTokens(window: Int?, outputLimit: Int? = nil) -> Int {
        replyBudget(requested: Int.max, window: window, outputLimit: outputLimit)
    }

    /// The transcript notice for a turn the provider stopped at `length`.
    ///
    /// Each binding gets the sentence that can actually change it, and the
    /// reasoning note names why a long thinking block can spend the budget
    /// before the visible answer starts. Plan Mode is the mode that made this
    /// concrete: its deliverable is one long message and its reasoning shares
    /// the same output budget, so a 16k cap cut the plan mid-sentence.
    public static func outputLimitNotice(_ budget: ReplyBudget) -> String {
        let head = "⚠️ This answer stopped at the Max Output Tokens limit (\(budget.effective.formatted()) tokens) and may be truncated. "
        let tail = " Reasoning counts toward the same budget, so a long thinking block can use it before the answer starts."
        let ceiling = maximumOutputTokens(
            window: budget.window,
            outputLimit: budget.outputLimit
        )
        switch budget.binding {
        case .setting:
            return head
                + "This model allows up to \(ceiling.formatted()). Raise Max Output Tokens in Settings ▸ Sampling for a longer reply."
                + tail
        case .unrecordedWindow:
            return head
                + "No context window is recorded for this model, so its reply budget is held to \(budget.effective.formatted()). Run Test & Sync Models on Settings ▸ Cloud (or record its window) to price it against the provider's own capacity, then raise Max Output Tokens."
                + tail
        case .windowFraction:
            let windowText = budget.window.map { "this model's \($0.formatted())-token context window" } ?? "this model's context window"
            return head
                + "That is the room \(windowText) leaves for a reply, so raising Max Output Tokens cannot go higher. Record a larger window on Settings ▸ Cloud if the provider publishes one."
                + tail
        case .outputLimit:
            let limitText = budget.outputLimit.map { "this model's published \($0.formatted())-token output limit" } ?? "this model's published output limit"
            return head
                + "That is \(limitText), so raising Max Output Tokens cannot go higher. Choose a model with a larger output limit for a longer reply."
                + tail
        case .providerRefused:
            return head
                + "The provider refused a larger reply budget for this model, so this turn continued at \(budget.effective.formatted()). Your Max Output Tokens setting was not lowered; record the model's real output limit on Settings ▸ Cloud so a later turn is not refused again."
                + tail
        }
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

// MARK: - Max Output Tokens ceiling

/// Which backend's limit governs the Max Output Tokens slider for one model.
///
/// A single flat number was wrong for every backend: a local checkpoint's
/// ceiling is its KV policy, a cloud model's is a fraction of the context window
/// its provider records, and the on-device model has a fixed response limit of
/// its own. The slider has to show whichever one binds, or it either advertises
/// a budget the launch path lowers or invites one no provider will accept.
///
/// This lives beside `maximumOutputTokens` rather than in the Settings view
/// because it is pure policy over recorded facts: it reads `UserDefaults` and a
/// `DiscoveredModel`, never a view's state, and it is what the tests exercise.
enum MaxOutputCeilingBackend {
    /// A cloud turn, carrying where its window came from: the model's own
    /// recorded value, the provider's published window, the single global
    /// fallback, or neither; plus the catalog's own output cap when it
    /// published one. The ceiling is 20% of the window either way, but an
    /// explanation that called a fallback value "this model's context window"
    /// would name a fact the store does not hold.
    case cloud(window: CloudContextPolicy.WindowSource, outputLimit: Int?)
    case appleFoundationModel
    /// A local checkpoint, carrying what actually binds its ceiling. The number
    /// is `max(32_768, min(declared, 262_144))`, so it is NOT always the
    /// checkpoint's own limit: an 8k checkpoint is capped by the policy floor,
    /// 24,576 rows below what it declares. An explanation has to name the case
    /// it is in for the same reason the cloud branch carries its `WindowSource`.
    case local(basis: KVCacheProvisioning.LocalCeilingBasis)

    /// `declaredMaxPositionEmbeddings` is asked about only for the LOCAL branch,
    /// which is why it is optional and defaulted: the cloud and on-device
    /// branches have no checkpoint to read it from, and their callers do not
    /// have one either. The Settings pane passes the DECLARED limit as a value
    /// so this stays pure — the ceiling itself was resolved in an action, and a
    /// second `config.json` read here would be both a disk hit in `body` and a
    /// chance for the sentence and the number to describe different checkpoints.
    static func classify(
        model: DiscoveredModel?,
        fallbackAddress: String?,
        declaredMaxPositionEmbeddings: Int? = nil,
        defaults: UserDefaults = .standard
    ) -> MaxOutputCeilingBackend {
        let local = MaxOutputCeilingBackend.local(
            basis: KVCacheProvisioning.localCeilingBasis(
                declaredMaxPositionEmbeddings: declaredMaxPositionEmbeddings
            )
        )
        if let model {
            if let parsed = CloudModelAddress.parseModelId(model.id)
                ?? CloudModelAddress.parsePath(model.snapshotPath) {
                let limits = CloudContextPolicy.observedLimit(
                    forModelId: parsed.providerModelId,
                    providerId: parsed.provider.id,
                    defaults: defaults
                )
                return .cloud(
                    window: CloudContextPolicy.windowSource(
                        forModelId: parsed.providerModelId,
                        providerId: parsed.provider.id,
                        defaults: defaults
                    ),
                    outputLimit: limits?.maxOutput
                )
            }
            if CloudModelAddress.isCloudModelId(model.id) || CloudModelAddress.isCloudPath(model.snapshotPath) {
                return .cloud(window: globalWindowSource(defaults: defaults), outputLimit: nil)
            }
            if model.isAppleFoundationModel {
                return .appleFoundationModel
            }
            return local
        }
        guard let address = fallbackAddress, !address.isEmpty else {
            return local
        }
        if address == AppleFoundationModelService.modelPath
            || address == AppleFoundationModelService.modelId {
            return .appleFoundationModel
        }
        if let parsed = CloudModelAddress.parseModelId(address)
            ?? CloudModelAddress.parsePath(address) {
            let limits = CloudContextPolicy.observedLimit(
                forModelId: parsed.providerModelId,
                providerId: parsed.provider.id,
                defaults: defaults
            )
            return .cloud(
                window: CloudContextPolicy.windowSource(
                    forModelId: parsed.providerModelId,
                    providerId: parsed.provider.id,
                    defaults: defaults
                ),
                outputLimit: limits?.maxOutput
            )
        }
        if CloudModelAddress.isCloudModelId(address) || CloudModelAddress.isCloudPath(address) {
            return .cloud(window: globalWindowSource(defaults: defaults), outputLimit: nil)
        }
        return local
    }

    /// A cloud id this build cannot parse into (provider, model) still has a
    /// window only if the global fallback supplies one, and it is exactly that:
    /// a fallback, never the model's own record.
    private static func globalWindowSource(
        defaults: UserDefaults
    ) -> CloudContextPolicy.WindowSource {
        guard let global = CloudContextPolicy.globalWindow(defaults: defaults) else {
            return CloudContextPolicy.WindowSource.none
        }
        return .global(global)
    }
}

/// Resolves the Max Output Tokens ceiling for a non-local backend (cloud or
/// Apple Foundation Model), returning nil when the backend is a local checkpoint
/// whose `config.json` position limit governs instead.
func resolveNonLocalMaxOutputCeiling(
    model: DiscoveredModel?,
    fallbackAddress: String?,
    defaults: UserDefaults = .standard
) -> Int? {
    switch MaxOutputCeilingBackend.classify(model: model, fallbackAddress: fallbackAddress, defaults: defaults) {
    case .cloud(let source, let outputLimit):
        return CloudContextPolicy.maximumOutputTokens(
            window: source.window,
            outputLimit: outputLimit
        )
    case .appleFoundationModel:
        return AppleFoundationModelService.maxResponseTokenCeiling
    case .local:
        return nil
    }
}

/// Explains why the Max Output Tokens ceiling sits where it does, handling both
/// discovered entries and fallback addresses (e.g., an ad-hoc cloud model or
/// the on-device model before a scan completes).
///
/// `declaredMaxPositionEmbeddings` is the DECLARED limit of whatever checkpoint
/// the row is about, for the same reason `classify` takes it: the local ceiling
/// is `max(32_768, min(declared, 262_144))`, so a checkpoint under the floor is
/// capped by the POLICY, not by its own positions, and the two cases need
/// different sentences. Saying "capped by this checkpoint's own position limit"
/// beside a 32,768 slider told every 8k model the opposite of what was binding
/// it (`#95` review).
func maxOutputCeilingExplanation(
    for model: DiscoveredModel?,
    fallbackAddress: String?,
    declaredMaxPositionEmbeddings: Int? = nil,
    defaults: UserDefaults = .standard
) -> String {
    switch MaxOutputCeilingBackend.classify(
        model: model,
        fallbackAddress: fallbackAddress,
        declaredMaxPositionEmbeddings: declaredMaxPositionEmbeddings,
        defaults: defaults
    ) {
    case .cloud(let source, let outputLimit):
        return cloudCeilingExplanation(source: source, outputLimit: outputLimit)
    case .appleFoundationModel:
        return "Capped by the on-device model's own response limit (\(AppleFoundationModelService.maxResponseTokenCeiling.formatted()) tokens); a larger value is clamped before the framework sees it."
    case .local(let basis):
        return localCeilingExplanation(basis)
    }
}

/// The cloud branch's sentence, naming whichever fact actually binds.
///
/// The output cap is checked first because it can be smaller than the window
/// fraction: a model with a 200k window and a 32k published output limit is
/// capped at 32k, and a sentence that said `20% of the window` beside a 32k
/// slider would name a limit the model does not have.
private func cloudCeilingExplanation(
    source: CloudContextPolicy.WindowSource,
    outputLimit: Int?
) -> String {
    let ceiling = CloudContextPolicy.maximumOutputTokens(
        window: source.window,
        outputLimit: outputLimit
    )
    if let outputLimit, ceiling == outputLimit {
        return "Capped by this model's published \(outputLimit.formatted())-token output limit. That limit binds before the 20% window reserve, so raising the window alone cannot move it."
    }
    switch source {
    case .model(let recorded):
        return "Capped by 20% of this model's \(recorded.formatted())-token context window. A reply budget larger than that cannot share the window with any prompt. Record a larger window on the Cloud tab to raise it."
    case .observed(let recorded):
        return "Detected from the provider's model catalog: a \(recorded.formatted())-token context window, which caps a reply at \(ceiling.formatted()). Record a window on the Cloud tab to override it."
    case .global(let fallback):
        return "No window of its own is recorded for this model, so the global \(fallback.formatted())-token fallback applies and caps the reply at 20% of it. Record a window for this model on the Cloud tab to price it against its own context."
    case .none:
        return "No context window is recorded for this model, so the reply budget is held to \(CloudContextPolicy.unknownWindowReplyCeiling.formatted()) tokens. Run Test & Sync Models on the Cloud tab (or record its window) to raise the ceiling to 20% of its context."
    }
}

/// The local branch's sentence, split out so each case is written once and the
/// text cannot drift from `KVCacheProvisioning.localCeilingBasis`'s switch.
private func localCeilingExplanation(_ basis: KVCacheProvisioning.LocalCeilingBasis) -> String {
    let budget = "The KV cache budget also lowers the effective ceiling for a long prompt."
    switch basis {
    case .policyFloor(let declared):
        // The number shown is the floor, which is LARGER than the checkpoint's
        // own limit. Both facts are stated, because "held to 32,768" reads as a
        // cap the checkpoint asked for otherwise.
        return "This checkpoint declares \(declared.formatted()) positions, below the \(KVCacheProvisioning.maxProvisionedRows.formatted())-token policy floor every local model is provisioned at, so the floor is what sets the ceiling. \(budget)"
    case .checkpointPositions(let declared):
        return "Capped by this checkpoint's own \(declared.formatted())-position limit. \(budget)"
    case .policyMaximum(let declared):
        return "This checkpoint declares \(declared.formatted()) positions, above the \(KVCacheProvisioning.absoluteMaxProvisionedRows.formatted())-row maximum this app will provision for a local model, so that maximum is what sets the ceiling. \(budget)"
    case .unknown:
        // No declared limit was read, which is two situations in one sentence:
        // a local model whose `config.json` has not landed yet, and a fallback
        // address with no checkpoint behind it at all. Both are provisioned at
        // the floor, so the sentence states that rather than claiming a
        // position limit either of them does not have.
        return "This checkpoint's position limit has not been read, so the \(KVCacheProvisioning.maxProvisionedRows.formatted())-token local policy floor is what a turn is provisioned at until it is. \(budget)"
    }
}

/// Explains a STORED Max Output Tokens value that sits above the ceiling shown
/// for its model, and what a turn will actually do with it.
///
/// Every backend keeps the saved number and lowers the turn, so the note must
/// not promise that the value is sent: a cloud turn's `max_tokens` goes through
/// `CloudContextPolicy.replyBudget`, a local turn's allocation is lowered by
/// `KVCacheProvisioning.turnFit` (which logs `.maxTokensReduced`), and the
/// on-device turn is clamped before the framework sees it. Only the stored value
/// survives intact, which is why the note explains the number instead of the
/// view silently rewriting it — a profile is per model, so a value saved against
/// a wider model (or a cloud window since lowered) can legitimately sit here.
///
/// The sentence is shared by the Generation tab and the profile editor on
/// purpose: unpinned, the two panes said opposite things about the same model.
func maxOutputOverCeilingExplanation(
    savedTokens: Int,
    ceiling: Int,
    for model: DiscoveredModel?,
    fallbackAddress: String?,
    defaults: UserDefaults = .standard
) -> String {
    let saved = "Saved value \(savedTokens.formatted()) exceeds this model's \(ceiling.formatted())-token ceiling. It is kept; "
    switch MaxOutputCeilingBackend.classify(model: model, fallbackAddress: fallbackAddress, defaults: defaults) {
    case .cloud:
        return saved + "a cloud turn lowers the reply budget to the ceiling rather than sending more."
    case .appleFoundationModel:
        return saved + "an on-device turn clamps it before the framework sees it."
    case .local:
        return saved + "a local turn lowers the allocation to what the KV budget affords rather than generating the full number."
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
