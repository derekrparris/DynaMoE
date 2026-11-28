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
//  An FP32 visit keeps its rows in ONE flat ring rather than one array per row,
//  because the device attention reads a copy of the whole window every visit and
//  that copy is the stage's entire cost: 8.4 MB per visit at the published
//  1,032-row context, measured at 1.28-1.54 ms of host time a visit (slice 38).
//  A flat ring leaves in at most two contiguous moves instead of one call per
//  row, which the isolation probe priced at 1.6x (slice 38's 57.3 against
//  34.5 GB/s). `copyKeyRows`/`copyValueRows` are that path; `keys`/`values`
//  remain the row-at-a-time reader's form and are what the host reference uses.
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
    ///
    /// Only `int8` uses these: an FP32 visit keeps its rows in the flat ring
    /// below instead.
    private var storedKeys: [[Int8]] = []
    private var storedValues: [[Int8]] = []
    /// Per (row, KV head) FP16 scales, K and V separately. Empty in FP32.
    private var keyScales: [[UInt16]] = []
    private var valueScales: [[UInt16]] = []
    /// The FP32 rows, in `float32` storage, as ONE flat ring: `ringCapacity`
    /// rows of `ringWidth` floats, row 0 of the window at `ringHead`, row `i`
    /// at `(ringHead + i) % ringCapacity`.
    ///
    /// A flat ring rather than `[[Float]]` because the device attention reads a
    /// copy of the whole window and the copy is what this stage costs: at the
    /// published 1,032-row context one visit stages 1,024 rows x 1,024 floats
    /// x 2 regions = 8.4 MB, and a host-row-at-a-time copy of it was measured at
    /// 1.28-1.54 ms of a 1.28-1.90 ms stage (slice 38). Out of a flat ring the
    /// same bytes leave in at most TWO contiguous moves (a wrapped ring is two
    /// spans), which the isolation probe priced at 1.6x the per-row form
    /// (57.3 vs 34.5 GB/s).
    ///
    /// `int8` keeps its `[[Int8]]` rows: its staging is a quarter of the bytes
    /// and the kernel reads a scale per (row, head), so there is nothing here
    /// for a flat copy to win.
    private var ringKeys: [Float] = []
    private var ringValues: [Float] = []
    /// Floats per row in the ring. Zero until the first append.
    private var ringWidth = 0
    /// Rows the ring has room for. The ring is reserved, not compacted: a
    /// windowed visit reserves its window on the first append and never grows,
    /// a full visit doubles as its positions arrive. `residentBytes` reports
    /// the rows HELD, which is the number a KV budget prices.
    private var ringCapacity = 0
    /// Ring slot holding the oldest retained row.
    private var ringHead = 0
    /// Rows currently held.
    private var ringCount = 0

    /// Keys and values as FP32, one row per retained position, chronologically.
    ///
    /// A row-at-a-time reader's form, not the staging form: in `float32` each
    /// row is copied out of the flat ring, so this allocates a row per call,
    /// and a caller that wants the whole window moved should use
    /// `copyKeyRows(into:)` (one or two contiguous moves) rather than ask for
    /// this and copy again. In `int8` the row is dequantized as it is asked for
    /// (`stored * scale`, per head): that costs a multiply per element per
    /// read, and it is the price of the cache actually being a quarter of the
    /// size.
    public var keys: [[Float]] {
        storage == .float32 ? ringRows(ringKeys)
            : zip(storedKeys, keyScales).map { Self.dequantize($0, $1, headWidth: headWidth) }
    }
    public var values: [[Float]] {
        storage == .float32 ? ringRows(ringValues)
            : zip(storedValues, valueScales).map { Self.dequantize($0, $1, headWidth: headWidth) }
    }

    /// Moves every retained key row into `destination`, in position order, and
    /// returns the row count. The staging form: at most two `memcpy` spans
    /// (a wrapped ring is contiguous from the head to the end, then from the
    /// start to the head again), where a row-at-a-time copy is one call per
    /// row. Zero rows in `int8`, whose staging walks stored bytes per row.
    @discardableResult
    public func copyKeyRows(into destination: UnsafeMutablePointer<Float>) -> Int {
        copyRingRows(ringKeys, into: destination)
    }

    @discardableResult
    public func copyValueRows(into destination: UnsafeMutablePointer<Float>) -> Int {
        copyRingRows(ringValues, into: destination)
    }

    /// The stored bytes of a row and its per-head scales, for a reader that
    /// wants the compact form (a device staging copy can hand these straight to
    /// a dequantizing kernel). Nil in `float32` storage.
    public func quantizedKey(_ row: Int) -> (stored: [Int8], scales: [UInt16])? {
        guard storage == .int8, row >= 0, row < storedKeys.count else { return nil }
        return (storedKeys[row], keyScales[row])
    }

    public func quantizedValue(_ row: Int) -> (stored: [Int8], scales: [UInt16])? {
        guard storage == .int8, row >= 0, row < storedValues.count else { return nil }
        return (storedValues[row], valueScales[row])
    }

    /// Bytes the rows this cache HOLDS occupy, which is the number an INT8
    /// setting exists to move: the stored rows plus their FP16 scales.
    ///
    /// Held rows, not reserved: the FP32 flat ring reserves up to twice the
    /// rows it holds while a full-attention visit is growing, but a KV budget
    /// prices the positions it can answer for, not the allocator's slack.
    public var residentBytes: Int {
        if storage == .float32 {
            return ringCount * ringWidth * 2 * MemoryLayout<Float>.stride
        }
        let rows = storedKeys.count
        let keyWidth = storedKeys.first?.count ?? 0
        let valueWidth = storedValues.first?.count ?? 0
        let scales = (keyScales.first?.count ?? 0) + (valueScales.first?.count ?? 0)
        return rows * (keyWidth + valueWidth) * MemoryLayout<Int8>.stride
            + rows * scales * MemoryLayout<UInt16>.stride
    }
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
            if key.count != ringWidth { resizeRing(toWidth: key.count) }
            guard ringWidth > 0 else {
                // A zero-width row is not a position: without this the row
                // count would advance while nothing became addressable.
                return windowStart..<(windowStart + rowCount)
            }
            ensureRingCapacity(ringCount + 1)
            writeRingRow(key, into: &ringKeys, at: ringCount)
            writeRingRow(value, into: &ringValues, at: ringCount)
            ringCount += 1
        case .int8:
            // Per KV head, both directions, as the device kernel does it: one
            // scale per (token, head), the row read back as stored * scale.
            //
            // The head width is taken BEFORE the closure: it reads the stored
            // rows, and reading them while `&storedKeys` is in flight is an
            // overlapping access (measured: Swift traps it as "Fatal access
            // conflict", not as a wrong number).
            let headWidth = max(1, key.count / max(1, heads))
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
        }
        committedPositions += 1
        if !isFullAttention {
            let excess = max(0, rowCount - window)
            if excess > 0 {
                dropFirst(excess)
                windowStart += excess
            }
        }
        return windowStart..<(windowStart + rowCount)
    }

    /// Rows the cache holds, in either storage.
    public var rowCount: Int {
        storage == .int8 ? storedKeys.count : ringCount
    }

    /// The width of one row (K and V have the same width here: both are
    /// `kvHeads * headDim`), without dequantizing anything.
    public var rowWidth: Int {
        storage == .int8 ? (storedKeys.first?.count ?? 0) : ringWidth
    }

    /// KV heads, so a scale is per head and not per row.
    public var heads: Int = 1

    /// The width of one KV head's slice, which is what a scale covers.
    ///
    /// Exactly `heads` equal slices by construction -- `kvWidth` is
    /// `kvHeads * headDim` -- and the division is of the WIDTH, not of a padded
    /// stride: the device kernel takes a `simd_max` over one head's `headDim`
    /// values, so a host scale that covered a different span would be a
    /// different contract with the same name.
    public var headWidth: Int {
        let width = rowWidth
        guard width > 0 else { return 1 }
        return max(1, width / max(1, heads))
    }

    private func dropFirst(_ count: Int) {
        if storage == .int8 {
            storedKeys.removeFirst(count)
            storedValues.removeFirst(count)
            keyScales.removeFirst(count)
            valueScales.removeFirst(count)
        } else {
            let drop = min(count, ringCount)
            guard drop > 0 else { return }
            ringHead = (ringHead + drop) % max(1, ringCapacity)
            ringCount -= drop
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
            if storage == .int8 {
                let drop = min(removed, storedKeys.count)
                storedKeys.removeLast(drop)
                storedValues.removeLast(drop)
                keyScales.removeLast(min(drop, keyScales.count))
                valueScales.removeLast(min(drop, valueScales.count))
            } else {
                ringCount -= min(removed, ringCount)
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
        ringKeys.removeAll()
        ringValues.removeAll()
        ringWidth = 0
        ringCapacity = 0
        ringHead = 0
        ringCount = 0
        storedKeys.removeAll()
        storedValues.removeAll()
        keyScales.removeAll()
        valueScales.removeAll()
        committedPositions = 0
        windowStart = 0
        convolutionHistory = convolutionHistory.map { _ in [] }
    }

    // MARK: - The flat FP32 ring

    /// One row as `[[Float]]`, chronologically: the reader's form.
    private func ringRows(_ storage: [Float]) -> [[Float]] {
        guard ringWidth > 0, ringCount > 0, ringCapacity > 0 else { return [] }
        var rows = [[Float]]()
        rows.reserveCapacity(ringCount)
        storage.withUnsafeBufferPointer { source in
            let base = source.baseAddress!
            for row in 0..<ringCount {
                let start = ((ringHead + row) % ringCapacity) * ringWidth
                rows.append(Array(UnsafeBufferPointer(start: base + start, count: ringWidth)))
            }
        }
        return rows
    }

    /// Every retained row into `destination`, in position order: one `memcpy`
    /// from the head to the end of the ring, then one more for the rows that
    /// wrapped to its start. Returns the row count.
    private func copyRingRows(_ storage: [Float], into destination: UnsafeMutablePointer<Float>) -> Int {
        guard ringWidth > 0, ringCount > 0, ringCapacity > 0 else { return 0 }
        let rowBytes = ringWidth * MemoryLayout<Float>.stride
        let leading = min(ringCount, ringCapacity - ringHead)
        storage.withUnsafeBufferPointer { source in
            let base = source.baseAddress!
            if leading > 0 {
                memcpy(destination, base + ringHead * ringWidth, leading * rowBytes)
            }
            let wrapped = ringCount - leading
            if wrapped > 0 {
                memcpy(destination + leading * ringWidth, base, wrapped * rowBytes)
            }
        }
        return ringCount
    }

    /// Writes one row at ring position `row` (0 is the oldest held row).
    /// A short row is copied as far as it goes, which is what the row-at-a-time
    /// staging copy did: a ragged row is a caller's bug, not a fault.
    private func writeRingRow(_ row: [Float], into storage: inout [Float], at index: Int) {
        guard ringWidth > 0, ringCapacity > 0 else { return }
        let start = ((ringHead + index) % ringCapacity) * ringWidth
        let count = min(ringWidth, row.count)
        _ = storage.withUnsafeMutableBufferPointer { destination in
            row.withUnsafeBufferPointer { source in
                memcpy(destination.baseAddress! + start, source.baseAddress!, count * MemoryLayout<Float>.stride)
            }
        }
    }

    /// Grows the ring so it can hold `needed` rows.
    ///
    /// Geometric growth (1.5x) rather than a reserve of the whole window: a
    /// reserve would cost every visit 8 MB the moment it took its first row,
    /// and the growth copies are amortised to a few KB per position. It also
    /// keeps a half-full ring's slack bounded, which is what `residentBytes`
    /// reports against.
    private func ensureRingCapacity(_ needed: Int) {
        guard needed > ringCapacity else { return }
        relayoutRing(to: max(needed, ringCapacity + max(ringCapacity / 2, 16)))
    }

    /// Re-lays the ring out chronologically into a new buffer of `capacity`
    /// rows. Rare: once per growth of a full-attention visit's cache.
    private func relayoutRing(to capacity: Int) {
        guard ringWidth > 0 else { return }
        resizeRing(toWidth: ringWidth, capacity: capacity, rows: ringCount)
    }

    /// Rebuilds the ring with a new row width and/or capacity, keeping the
    /// held rows in position order from slot 0.
    private func resizeRing(toWidth width: Int, capacity: Int? = nil, rows: Int? = nil) {
        guard width > 0 else { return }
        let held = rows ?? ringCount
        let target = capacity ?? max(held, 16)
        var keys = [Float](repeating: 0, count: target * width)
        var values = [Float](repeating: 0, count: target * width)
        if ringWidth > 0, ringCapacity > 0, held > 0 {
            let copyWidth = min(ringWidth, width)
            ringKeys.withUnsafeBufferPointer { source in
                keys.withUnsafeMutableBufferPointer { destination in
                    for row in 0..<held {
                        let from = ((ringHead + row) % ringCapacity) * ringWidth
                        memcpy(
                            destination.baseAddress! + row * width, source.baseAddress! + from,
                            copyWidth * MemoryLayout<Float>.stride
                        )
                    }
                }
            }
            ringValues.withUnsafeBufferPointer { source in
                values.withUnsafeMutableBufferPointer { destination in
                    for row in 0..<held {
                        let from = ((ringHead + row) % ringCapacity) * ringWidth
                        memcpy(
                            destination.baseAddress! + row * width, source.baseAddress! + from,
                            copyWidth * MemoryLayout<Float>.stride
                        )
                    }
                }
            }
        }
        ringKeys = keys
        ringValues = values
        ringWidth = width
        ringCapacity = target
        ringHead = 0
        ringCount = held
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
    ///
    /// `headsByVisit` overrides `kvHeads` per visit when the two attention
    /// geometries disagree about how many KV heads a layer has: a scale is
    /// stored per (row, KV head), so a visit quantized under one count cannot be
    /// read under the other. Empty uses `kvHeads` for every visit.
    public init(
        plan: [Inkling2VisitPlan],
        window: Int,
        kvWidth: Int,
        hidden: Int,
        historyDepth: Int,
        maxPositions: Int,
        storage: Inkling2KVStorage = .float32,
        kvHeads: Int = 1,
        headsByVisit: [Int] = []
    ) {
        self.maxPositions = max(1, maxPositions)
        self.storage = storage
        self.plan = plan
        self.visits = plan.enumerated().map { index, visit in
            Inkling2VisitState(
                isFullAttention: visit.isFullAttention,
                window: window,
                stateWidths: [kvWidth, kvWidth, hidden, hidden],
                historyDepth: historyDepth,
                // Only a full-attention visit grows without bound; a ring is
                // capped at its window and must not be refused at the budget.
                maxRows: visit.isFullAttention ? maxPositions : nil,
                storage: storage,
                heads: index < headsByVisit.count ? headsByVisit[index] : kvHeads
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
