//
//  Inkling2KVCache.swift
//  DynaMoE
//
//  Executor state for Inkling 2 Mini: what one visit keeps, and what a sequence
//  keeps across visits.
//
//  Two facts drive this file:
//
//    1. State is per EXECUTED VISIT, not per physical layer. Weights are shared
//       across the loop passes; KV and the four convolution histories are not.
//       At the published schedule that is 48 state bundles, eight of which own
//       a growing full-attention cache and forty of which are bounded rings.
//    2. A local visit is a CHRONOLOGICAL ring. Relative attention indexes a
//       key by backward distance, so a ring read newest-first would silently
//       misread every bias. The ring therefore drops from the front and keeps
//       its rows in position order, and `windowStart` is what turns a row index
//       back into a position.
//
//  The KV rows are stored as FP32 here because the host reference computes in
//  FP32; the INT8/FP16 storage DynaMoE's FP8 setting uses is a later phase, and
//  it reuses the same addressing rule (a ring read is chronological either way).
//

import Foundation

// MARK: - Per-visit state

/// One visit's KV rows and four convolution histories.
///
/// Four states per visit, FP32, `kernel - 1` history values each:
///
///   0: post-K convolution   (width `kvHeads * headDim`)
///   1: post-V convolution   (same)
///   2: post-attention       (width `hidden`)
///   3: post-MLP             (width `hidden`)
nonisolated public final class Inkling2VisitState {
    public let isFullAttention: Bool
    public let window: Int
    public let historyDepth: Int
    /// Keys and values, one row per retained position, chronologically.
    public private(set) var keys: [[Float]] = []
    public private(set) var values: [[Float]] = []
    /// Absolute position of the next row to be appended.
    public private(set) var committedPositions = 0
    /// Absolute position of `keys[0]`; a ring drops from the front.
    public private(set) var windowStart = 0
    public private(set) var convolutionHistory: [[[Float]]]

    /// Rows this visit may hold, or nil for no cap. A full-attention visit
    /// keeps every position, so this is the KV budget's cap on it; a ring visit
    /// drops from the front and never reaches it.
    public let maxRows: Int?

    public init(isFullAttention: Bool, window: Int, stateWidths: [Int], historyDepth: Int, maxRows: Int? = nil) {
        self.isFullAttention = isFullAttention
        self.window = max(1, window)
        self.historyDepth = max(0, historyDepth)
        self.maxRows = maxRows
        self.convolutionHistory = stateWidths.map { _ in [] }
    }

    /// Whether one more row would exceed the cap, which the forward turns into
    /// a refusal by name rather than a process growing until it is killed.
    public var isAtCapacity: Bool {
        guard let maxRows else { return false }
        return committedPositions >= maxRows
    }

    /// Appends one position's K/V and returns the absolute range of rows the
    /// position may attend over.
    public func append(key: [Float], value: [Float]) -> Range<Int> {
        keys.append(key)
        values.append(value)
        committedPositions += 1
        if !isFullAttention {
            let excess = max(0, keys.count - window)
            if excess > 0 {
                keys.removeFirst(excess)
                values.removeFirst(excess)
                windowStart += excess
            }
        }
        return windowStart..<(windowStart + keys.count)
    }

    /// Drops every row at or after `positions`: the rollback a rejected
    /// speculative suffix needs. Returns the number of rows removed.
    @discardableResult
    public func rollback(to positions: Int) -> Int {
        let removed = max(0, committedPositions - positions)
        committedPositions = positions
        if removed > 0 {
            let drop = min(removed, keys.count)
            keys.removeLast(drop)
            values.removeLast(drop)
        }
        return removed
    }

    /// Drops convolution history past `positions` rows, matching a rollback.
    public func rollbackConvolutions(to positions: Int) {
        for index in convolutionHistory.indices where convolutionHistory[index].count > positions {
            convolutionHistory[index].removeLast(convolutionHistory[index].count - positions)
        }
    }

    public func pushConvolution(_ index: Int, value: [Float]) {
        guard index >= 0, index < convolutionHistory.count else { return }
        convolutionHistory[index].append(value)
        let excess = convolutionHistory[index].count - historyDepth
        if excess > 0 { convolutionHistory[index].removeFirst(excess) }
    }

    /// The history for a convolution, oldest first, newest last.
    public func history(_ index: Int) -> [[Float]] {
        guard index >= 0, index < convolutionHistory.count else { return [] }
        return convolutionHistory[index]
    }

    public func reset() {
        keys.removeAll()
        values.removeAll()
        committedPositions = 0
        windowStart = 0
        convolutionHistory = convolutionHistory.map { _ in [] }
    }
}

// MARK: - Sequence state

/// Every visit's state for one sequence, plus the token history that produced
/// it. The history is what makes a prefix reuse provable rather than guessed: a
/// session key or a matching id prefix is not evidence that this cache holds
/// those tokens, because two conversations can share a prefix.
nonisolated public final class Inkling2SequenceState {
    public let visits: [Inkling2VisitState]
    /// The committed token ids, one per forwarded position.
    public private(set) var tokenHistory: [Int] = []
    /// One leading block stack per committed position's final pass, only when
    /// a caller asks to keep them (a message read needs the previous pass's).
    public let plan: [Inkling2VisitPlan]

    /// Positions the sequence may hold. A full-attention visit keeps every
    /// position's K/V, so this is the KV budget's per-visit row cap; a ring
    /// visit never reaches it because it holds at most its window.
    public let maxPositions: Int

    public init(
        plan: [Inkling2VisitPlan],
        window: Int,
        kvWidth: Int,
        hidden: Int,
        historyDepth: Int,
        maxPositions: Int
    ) {
        self.maxPositions = max(1, maxPositions)
        self.plan = plan
        self.visits = plan.map { visit in
            Inkling2VisitState(
                isFullAttention: visit.isFullAttention,
                window: window,
                stateWidths: [kvWidth, kvWidth, hidden, hidden],
                historyDepth: historyDepth,
                // Only a full-attention visit grows without bound; a ring is
                // capped at its window and must not be refused at the budget.
                maxRows: visit.isFullAttention ? maxPositions : nil
            )
        }
    }

    /// Positions every visit agrees on. They advance together or not at all:
    /// a visit that committed more than another is a bug, and reporting the
    /// minimum keeps a resume from trusting the one that ran ahead.
    public var committedPositions: Int {
        visits.map(\.committedPositions).min() ?? 0
    }

    /// Whether every visit is at the same count, which a resume requires.
    public var isConsistent: Bool {
        Set(visits.map(\.committedPositions)).count <= 1 && tokenHistory.count == committedPositions
    }

    public func record(token: Int) {
        tokenHistory.append(token)
    }

    /// How much of `promptTokenIds` is provably resident: the longest common
    /// prefix of the history and the prompt, capped at what the caller
    /// proposed and by the committed count. Leaving the last position out is
    /// deliberate -- its logits are what the next turn needs to re-derive.
    public func reusablePrefix(proposedPrefix: Int, promptTokenIds: [Int]) -> Int {
        guard isConsistent else { return 0 }
        let cap = max(0, min(proposedPrefix, committedPositions, promptTokenIds.count - 1))
        var count = 0
        while count < cap, tokenHistory[count] == promptTokenIds[count] { count += 1 }
        return count
    }

    /// Drops everything past `positions`, in every visit and the history.
    public func rollback(to positions: Int) {
        for visit in visits {
            visit.rollback(to: positions)
            visit.rollbackConvolutions(to: positions)
        }
        if tokenHistory.count > positions {
            tokenHistory.removeLast(tokenHistory.count - positions)
        }
    }

    public func reset() {
        for visit in visits { visit.reset() }
        tokenHistory.removeAll()
    }
}
