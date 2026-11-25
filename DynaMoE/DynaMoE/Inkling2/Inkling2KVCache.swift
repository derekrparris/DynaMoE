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
//  The KV rows are FP32 by default because the host reference computes in
//  FP32; INT8 storage is opt-in per sequence (`Inkling2KVStorage.int8`), using
//  DynaMoE's existing KV contract, and it reuses the same addressing rule (a
//  ring read is chronological either way).
//

import Foundation

// MARK: - Storage

/// How a visit's KV rows are held.
///
/// `float32` is what the host reference computes in and what every existing
/// path gets. `int8` is Phase 7's DynaMoE KV contract: per (token, KV head) for
/// K and V independently, `scale = max(abs(x), 1e-7) / 127`, the stored value
/// rounded and clamped to +/-127, the scale kept as FP16, and a read back is
/// `stored * scale`. It is NOT E4M3 and is independent of expert-activation
/// quantization.
///
/// The stored rows are exposed through the same `keys`/`values` accessors, so
/// every reader -- attention, the profile dump, the device staging copy -- sees
/// dequantized FP32 and none of them needs to know which storage is in use.
nonisolated public enum Inkling2KVStorage: String, Sendable, Equatable {
    case float32
    case int8

    /// Bytes one stored element occupies, before the FP16 scale.
    public var bytesPerElement: Int {
        switch self {
        case .float32: return 4
        case .int8: return 1
        }
    }

    /// The FP16 scale one (token, head) carries in `int8`, zero otherwise.
    public var scaleBytesPerHead: Int {
        self == .int8 ? MemoryLayout<UInt16>.stride : 0
    }
}

/// The INT8 store/dequantize rule, in one place.
///
/// Written on the host exactly as `store_kv_cache_fp8` writes it on the device,
/// because the two are the same contract: a divergence here is a cache whose
/// bytes depend on which path wrote them, and a resumed prefix would then
/// dequantize differently than the turn that produced it.
nonisolated public enum Inkling2KVQuantization {
    /// `max(abs(x), 1e-7) / 127`: the smallest scale that keeps the row in range.
    public static func scale(_ row: ArraySlice<Float>) -> Float {
        var maximum: Float = 0
        for value in row { maximum = max(maximum, abs(value)) }
        return max(maximum, 1e-7) / 127
    }

    public static func scale(_ row: [Float]) -> Float { scale(row[...]) }

    /// Rounds half away from zero and clamps, then stores as a signed byte.
    /// `round` in MSL is half away from zero, and Swift's `.rounded()` is the
    /// same, so the two agree on the tie cases rather than only on the easy ones.
    public static func store(_ value: Float, inverseScale: Float) -> Int8 {
        let scaled = (value * inverseScale).rounded()
        return Int8(max(-127, min(127, scaled)))
    }

    /// A stored byte and its FP16 scale, read back as the attention kernels read it.
    public static func load(_ stored: Int8, scale: UInt16) -> Float {
        Float(stored) * Float(Float16(bitPattern: scale))
    }

    /// One row through the whole round trip, for a golden that pins the rule.
    public static func stored(_ row: [Float]) -> (values: [Int8], scale: UInt16) {
        let factor = scale(row)
        let inverse = 1 / factor
        return (row.map { store($0, inverseScale: inverse) }, Float16(factor).bitPattern)
    }
}

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
    /// How these rows are held. FP32 unless the sequence asked otherwise.
    public let storage: Inkling2KVStorage
    /// The stored rows in `storage`'s own form: one row per retained position,
    /// chronologically, exactly as `keys`/`values` were handed in.
    private var storedKeys: [[Int8]] = []
    private var storedValues: [[Int8]] = []
    /// Per (row, KV head) FP16 scales, K and V separately. Empty in FP32.
    private var keyScales: [[UInt16]] = []
    private var valueScales: [[UInt16]] = []
    /// The dequantized rows, kept so a reader gets FP32 whatever the storage is.
    /// They are the cache's own copy, not the caller's array: a row is read many
    /// times (every attention over it) and dequantizing per read would put the
    /// conversion in the inner loop.
    private var decodedKeys: [[Float]] = []
    private var decodedValues: [[Float]] = []

    /// Keys and values, dequantized to FP32, one row per retained position,
    /// chronologically. In FP32 storage these ARE the stored rows.
    public var keys: [[Float]] { decodedKeys }
    public var values: [[Float]] { decodedValues }
    /// Absolute position of the next row to be appended.
    public private(set) var committedPositions = 0
    /// Absolute position of `keys[0]`; a ring drops from the front.
    public private(set) var windowStart = 0
    public private(set) var convolutionHistory: [[[Float]]]

    /// Rows this visit may hold, or nil for no cap. A full-attention visit
    /// keeps every position, so this is the KV budget's cap on it; a ring visit
    /// drops from the front and never reaches it.
    public let maxRows: Int?

    public init(
        isFullAttention: Bool, window: Int, stateWidths: [Int], historyDepth: Int,
        maxRows: Int? = nil, storage: Inkling2KVStorage = .float32, heads: Int = 1
    ) {
        self.isFullAttention = isFullAttention
        self.window = max(1, window)
        self.historyDepth = max(0, historyDepth)
        self.maxRows = maxRows
        self.storage = storage
        self.heads = max(1, heads)
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
        switch storage {
        case .float32:
            decodedKeys.append(key)
            decodedValues.append(value)
        case .int8:
            // Per KV head, both directions, as the device kernel does it: one
            // scale per (token, head), the row read back as stored * scale.
            func quantize(_ row: [Float], into stored: inout [[Int8]], scales: inout [[UInt16]]) {
                var bytes = [Int8](repeating: 0, count: row.count)
                var rowScales = [UInt16](repeating: 0, count: heads)
                for head in 0..<heads {
                    let range = (head * headWidth)..<min(row.count, (head + 1) * headWidth)
                    guard !range.isEmpty else { continue }
                    let factor = Inkling2KVQuantization.scale(row[range])
                    let inverse = 1 / factor
                    for index in range {
                        bytes[index] = Inkling2KVQuantization.store(row[index], inverseScale: inverse)
                    }
                    rowScales[head] = Float16(factor).bitPattern
                }
                stored.append(bytes)
                scales.append(rowScales)
            }
            quantize(key, into: &storedKeys, scales: &keyScales)
            quantize(value, into: &storedValues, scales: &valueScales)
            decodedKeys.append(Self.dequantize(
                storedKeys[storedKeys.count - 1], keyScales[keyScales.count - 1], headWidth: headWidth
            ))
            decodedValues.append(Self.dequantize(
                storedValues[storedValues.count - 1], valueScales[valueScales.count - 1], headWidth: headWidth
            ))
        }
        committedPositions += 1
        if !isFullAttention {
            let excess = max(0, decodedKeys.count - window)
            if excess > 0 {
                dropFirst(excess)
                windowStart += excess
            }
        }
        return windowStart..<(windowStart + decodedKeys.count)
    }

    /// KV heads, so a scale is per head and not per row.
    public var heads: Int = 1

    /// The width of one KV head's slice, which is what a scale covers: a row is
    /// `heads` equal slices.
    private var headWidth: Int {
        guard let first = decodedKeys.first, !first.isEmpty else { return 1 }
        return max(1, first.count / max(1, heads))
    }

    private func dropFirst(_ count: Int) {
        decodedKeys.removeFirst(count)
        decodedValues.removeFirst(count)
        if !storedKeys.isEmpty {
            storedKeys.removeFirst(count)
            storedValues.removeFirst(count)
            keyScales.removeFirst(count)
            valueScales.removeFirst(count)
        }
    }

    /// `stored * scale`, per head, as attention reads it.
    static func dequantize(_ stored: [Int8], _ scales: [UInt16], headWidth: Int) -> [Float] {
        guard !scales.isEmpty, headWidth > 0 else { return stored.map(Float.init) }
        return stored.enumerated().map { index, byte in
            let head = min(index / headWidth, scales.count - 1)
            return Inkling2KVQuantization.load(byte, scale: scales[head])
        }
    }

    /// Drops every row at or after `positions`: the rollback a rejected
    /// speculative suffix needs. Returns the number of rows removed.
    @discardableResult
    public func rollback(to positions: Int) -> Int {
        let removed = max(0, committedPositions - positions)
        committedPositions = positions
        if removed > 0 {
            let drop = min(removed, decodedKeys.count)
            decodedKeys.removeLast(drop)
            decodedValues.removeLast(drop)
            if !storedKeys.isEmpty {
                storedKeys.removeLast(min(drop, storedKeys.count))
                storedValues.removeLast(min(drop, storedValues.count))
                keyScales.removeLast(min(drop, keyScales.count))
                valueScales.removeLast(min(drop, valueScales.count))
            }
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
        decodedKeys.removeAll()
        decodedValues.removeAll()
        storedKeys.removeAll()
        storedValues.removeAll()
        keyScales.removeAll()
        valueScales.removeAll()
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

    /// `storage` is how every visit holds its KV rows, and `kvHeads` is what a
    /// per-head INT8 scale covers. Both default to the FP32 reference behavior,
    /// so every existing caller is unchanged and INT8 is an explicit choice.
    public init(
        plan: [Inkling2VisitPlan],
        window: Int,
        kvWidth: Int,
        hidden: Int,
        historyDepth: Int,
        maxPositions: Int,
        storage: Inkling2KVStorage = .float32,
        kvHeads: Int = 1
    ) {
        self.maxPositions = max(1, maxPositions)
        self.storage = storage
        self.plan = plan
        self.visits = plan.map { visit in
            Inkling2VisitState(
                isFullAttention: visit.isFullAttention,
                window: window,
                stateWidths: [kvWidth, kvWidth, hidden, hidden],
                historyDepth: historyDepth,
                // Only a full-attention visit grows without bound; a ring is
                // capped at its window and must not be refused at the budget.
                maxRows: visit.isFullAttention ? maxPositions : nil,
                storage: storage,
                heads: kvHeads
            )
        }
    }

    /// How this sequence's KV rows are held.
    public let storage: Inkling2KVStorage

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

    /// Refuses a position the KV budget cannot hold, before anything is
    /// written. A full-attention visit keeps every position, so the tightest
    /// one decides; ring visits are capped at their window and never refuse.
    public func checkCapacity(position: Int) throws {
        for (slot, visit) in visits.enumerated() where visit.isFullAttention {
            if visit.isAtCapacity {
                throw Inkling2ForwardError.unsupported(
                    "position \(position) is past the KV budget: this sequence holds "
                        + "\(visit.committedPositions) positions and the plan provisions "
                        + "\(visit.maxRows ?? 0) (visit \(slot)). Start a new conversation or "
                        + "raise the context budget."
                )
            }
        }
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
