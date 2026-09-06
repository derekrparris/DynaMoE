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
public struct PrefixCacheMetrics: Sendable, Equatable {
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
public final class PrefixCacheManager: @unchecked Sendable {
    public static let shared = PrefixCacheManager()

    private let lock = NSLock()

    /// The active session ID that owns the currently cached KV states
    private var activeSessionId: UUID?

    /// The full sequence of tokens whose KV states are currently pinned in KVCacheManager
    private var pinnedTokenIds: [UInt32] = []

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
        } else {
            self.pinnedTokenIds.removeAll(keepingCapacity: true)
            self.activeSessionId = nil
        }
    }

    /// Resets telemetry metrics.
    public func resetMetrics() {
        lock.lock()
        defer { lock.unlock() }
        metrics = PrefixCacheMetrics()
    }
}
