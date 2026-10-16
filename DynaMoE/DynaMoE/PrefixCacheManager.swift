//
//  PrefixCacheManager.swift
//  DynaMoE
//
//  Manages KV-Cache Prefix Pinning and Prompt Reuse for high-speed multi-turn
//  inference and agent tool calling loops on Apple Silicon Metal.
//

import Foundation
import Metal

/// Telemetry metrics for KV-cache prefix pinning performance
nonisolated public struct PrefixCacheMetrics: Sendable, Equatable {
    public var totalTurns: Int = 0
    public var cacheHits: Int = 0
    public var totalTokensRequested: Int = 0
    public var totalTokensReused: Int = 0
    public var estimatedPrefillMsSaved: Double = 0.0

    public var hitRatePercent: Double {
        guard totalTokensRequested > 0 else { return 0.0 }
        return (Double(totalTokensReused) / Double(totalTokensRequested)) * 100.0
    }

    public var averageSpeedupFactor: Double {
        guard totalTokensRequested > 0, (totalTokensRequested - totalTokensReused) > 0 else { return 1.0 }
        return Double(totalTokensRequested) / Double(totalTokensRequested - totalTokensReused)
    }
}

/// Coordinates prefix pinning and prompt reuse across conversation turns and tool executions.
nonisolated public final class PrefixCacheManager: @unchecked Sendable {
    public static let shared = PrefixCacheManager()

    private let lock = NSRecursiveLock()

    /// The active session ID that owns the currently cached KV states
    private var activeSessionId: UUID?

    /// The full sequence of tokens whose KV states are currently pinned in KVCacheManager
    private var pinnedTokenIds: [UInt32] = []

    /// The date whose rendering the active pin's prompt carries, owned by the
    /// session that will record the next pin. `# Current Date & Time` sits at
    /// the head of the system prompt with minute+second precision, so a fresh
    /// `Date()` per turn kills the common token prefix a sentence in — and a
    /// hybrid (GDN) session, which cannot rewind linear state, then takes a
    /// full ~30 s re-prefill on every agent step. Turns must render the same
    /// date the pinned context carries; a new one is minted (owned by that
    /// session) when the next full prefill is going to pin it anyway.
    private var pinnedAsOfDate: (sessionId: UUID?, date: Date)? = nil

    /// Dates minted for sessions that have rendered a turn but not yet recorded
    /// (pinned) one. A request is planned (and mints its date) long before it is
    /// admitted, so a rejected or timed-out turn from another session must not
    /// clobber the live pin's date; its date waits here until its own turn
    /// actually pins. Bounded so a long-lived server cannot accumulate one entry
    /// per conversation ever seen.
    private var pendingAsOfDates: [(sessionId: UUID, date: Date)] = []
    private let maxPendingAsOfDates = 64

    /// Performance metrics tracking
    public private(set) var metrics = PrefixCacheMetrics()

    /// Flag enabling or disabling prefix pinning (defaults to true)
    public var isEnabled: Bool = true

    private init() {}

    /// Finds the length of the longest matching token prefix between the proposed prompt and the cached state.
    /// - Parameters:
    ///   - promptTokenIds: The full tokenized prompt for the current turn.
    ///   - sessionId: The conversation session ID making the inference request.
    /// - Returns: The number of prefix tokens that match and can be skipped during prefill.
    public func findCommonPrefix(promptTokenIds: [UInt32], sessionId: UUID?) -> Int {
        lock.lock()
        defer { lock.unlock() }

        guard isEnabled else { return 0 }
        guard let sId = sessionId, sId == activeSessionId else {
            return 0
        }

        let maxPossible = min(promptTokenIds.count, pinnedTokenIds.count)
        guard maxPossible > 0 else { return 0 }

        var commonLen = 0
        while commonLen < maxPossible {
            if promptTokenIds[commonLen] != pinnedTokenIds[commonLen] {
                break
            }
            commonLen += 1
        }

        // We only reuse up to promptTokenIds.count - 1 because the very last prompt token
        // must be passed through the model to compute next-token logits.
        let usablePrefix = min(commonLen, max(0, promptTokenIds.count - 1))

        // Record metrics
        metrics.totalTurns += 1
        metrics.totalTokensRequested += promptTokenIds.count
        if usablePrefix > 0 {
            metrics.cacheHits += 1
            metrics.totalTokensReused += usablePrefix
            // Estimate ~0.15ms saved per prefilled token on Apple Silicon Metal
            metrics.estimatedPrefillMsSaved += Double(usablePrefix) * 0.15
        }

        return usablePrefix
    }

    /// The date to render into a turn's system prompt for `sessionId`.
    /// Returns the date the current pin was built with whenever that session
    /// owns it (byte-stable across its turns, so tokenized prompts share the
    /// full pinned prefix). A session that does not own the pin gets a date
    /// cached in `pendingAsOfDates` that stays stable across its own planning
    /// calls, but does NOT take over the live pin's slot: only a turn that
    /// actually pins (see `promoteAsOfDateLocked`) replaces it, so a rejected or
    /// timed-out request cannot invalidate another session's pinned prefix.
    /// `invalidate` clears it so a new prefill carries fresh temporal context
    /// instead of a growing-stale one.
    public func asOfDate(sessionId: UUID?) -> Date {
        lock.lock()
        defer { lock.unlock() }
        guard isEnabled else { return Date() }
        if let pinnedAsOfDate, pinnedAsOfDate.sessionId == sessionId {
            return pinnedAsOfDate.date
        }
        if let sessionId, let pending = pendingAsOfDates.first(where: { $0.sessionId == sessionId }) {
            return pending.date
        }
        let minted = Date()
        if let sessionId {
            pendingAsOfDates.append((sessionId, minted))
            if pendingAsOfDates.count > maxPendingAsOfDates {
                pendingAsOfDates.removeFirst(pendingAsOfDates.count - maxPendingAsOfDates)
            }
        }
        return minted
    }

    /// Makes `sessionId` the owner of the live pin's rendered date. Called only
    /// when a turn actually records a pin, so a session that never runs leaves
    /// the existing pin's date untouched.
    private func promoteAsOfDateLocked(for sessionId: UUID?) {
        guard let sessionId else { return }
        if let idx = pendingAsOfDates.firstIndex(where: { $0.sessionId == sessionId }) {
            pinnedAsOfDate = (sessionId, pendingAsOfDates.remove(at: idx).date)
        } else if pinnedAsOfDate?.sessionId != sessionId {
            pinnedAsOfDate = (sessionId, Date())
        }
    }

    /// Records the full token sequence after a turn completes (prompt + generated tokens).
    /// - Parameters:
    ///   - promptTokenIds: The prompt tokens used.
    ///   - generatedTokenIds: The tokens generated during this turn.
    ///   - sessionId: The active session ID.
    public func recordTurn(promptTokenIds: [UInt32], generatedTokenIds: [UInt32], sessionId: UUID?) {
        lock.lock()
        defer { lock.unlock() }

        guard isEnabled else { return }
        self.activeSessionId = sessionId
        var full = promptTokenIds
        full.append(contentsOf: generatedTokenIds)
        self.pinnedTokenIds = full
        promoteAsOfDateLocked(for: sessionId)
    }

    /// Explicitly updates the pinned token list (e.g. after prefill or partial generation).
    public func updatePinnedTokens(tokens: [UInt32], sessionId: UUID?) {
        lock.lock()
        defer { lock.unlock() }

        self.activeSessionId = sessionId
        self.pinnedTokenIds = tokens
    }

    /// Returns the currently pinned token count.
    public var currentPinnedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pinnedTokenIds.count
    }

    /// Resets/invalidates the pinned prefix (e.g. on new chat or context clearance).
    public func invalidate(sessionId: UUID? = nil) {
        lock.lock()
        defer { lock.unlock() }

        if let sId = sessionId {
            if self.activeSessionId == sId {
                self.pinnedTokenIds.removeAll(keepingCapacity: true)
                self.activeSessionId = nil
            }
            if self.pinnedAsOfDate?.sessionId == sId {
                self.pinnedAsOfDate = nil
            }
            pendingAsOfDates.removeAll { $0.sessionId == sId }
        } else {
            self.pinnedTokenIds.removeAll(keepingCapacity: true)
            self.activeSessionId = nil
            self.pinnedAsOfDate = nil
            pendingAsOfDates.removeAll(keepingCapacity: true)
        }
    }

    /// Resets telemetry metrics.
    public func resetMetrics() {
        lock.lock()
        defer { lock.unlock() }
        metrics = PrefixCacheMetrics()
    }
}
