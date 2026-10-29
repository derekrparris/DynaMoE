//
//  DeepSeekV41KVCache.swift
//  DynaMoE
//
//  The CSA2 cache: what DeepSeek-V4.1 keeps per token, and where it lives.
//
//  This is not the engine's `KVCacheManager` layout. That one holds one K and one
//  V tensor per slot with a per-head stride, because every other family in the
//  repo has per-head keys and values. DeepSeek V4.1 keeps a single LATENT record
//  per position (`kv_lora_rank` values plus the decoupled rope payload), and it
//  keeps two kinds of it:
//
//    * a bounded sliding-window ring per layer, and
//    * a COMPRESSED cache per cache-owning layer, where every `compress_ratio`
//      consecutive positions pool into one row. Only the owners allocate one, and
//      every other compressed layer READS its owner's rows rather than keeping
//      its own, which is what makes the decoder half of the CED stack cheap.
//
//  On top of that sit two derived caches: the indexer's keys (one short key per
//  compressed row, used to pick the candidate blocks) and the candidate block
//  scores themselves.
//
//  Sizing rules that matter:
//    * window rings do not grow with context (bounded by `window_size`),
//    * compressed rows grow as context / ratio, so a 1M-token context at ratio 2
//      on two owner layers is a few hundred MB, not gigabytes,
//    * every offset below is derived once, at load, and never recomputed per
//      token.
//
//  The precision is the checkpoint's designed FP8 form: E4M3 values with one
//  E8M0 scale per 32. `DeepSeekV41CachePrecision.fp16` exists so the same layout
//  arithmetic can be exercised (and used as the CPU reference) without a GPU
//  FP8 unpack in the test harness.
//

import Foundation
import Metal

// MARK: - Precision

nonisolated public enum DeepSeekV41CachePrecision: String, Sendable, Equatable, Codable {
    /// One IEEE half per value.
    case fp16
    /// E4M3 values with one E8M0 scale per 32 values (the checkpoint's design).
    case fp8Block32

    /// Bytes one record of `width` values occupies.
    public func recordBytes(width: Int) -> Int {
        switch self {
        case .fp16:
            return width * MemoryLayout<UInt16>.stride
        case .fp8Block32:
            return width + (width + 31) / 32
        }
    }

    /// Bytes of the value payload, before the scale footer.
    public var valueBytesPerElement: Int {
        switch self {
        case .fp16: return 2
        case .fp8Block32: return 1
        }
    }
}

// MARK: - Geometry

/// Sizes and strides for one cache generation. Everything the layout and the
/// kernels need, derived from the config once.
nonisolated public struct DeepSeekV41CacheGeometry: Sendable, Equatable {
    /// Values in one latent record: the compressed latent plus any decoupled
    /// rope payload.
    public let width: Int
    public let precision: DeepSeekV41CachePrecision
    /// Rows in one layer's sliding-window ring.
    public let windowRows: Int
    /// Rows in the widest owner layer's compressed cache. Per-owner row counts
    /// live in `DeepSeekV41CacheLayout.compressedRowsByOwner`.
    public let compressedRows: Int
    /// Sequence length the caches were provisioned for; per-owner rows are
    /// derived from it and the owner's own compress ratio.
    public let contextLength: Int
    /// Values in one indexer key.
    public let indexerKeyWidth: Int
    /// Candidate blocks scored per token position.
    public let candidateBlocks: Int

    public var recordBytes: Int { precision.recordBytes(width: width) }
    public var windowBytesPerLayer: Int { windowRows * recordBytes }
    public var compressedBytesPerOwner: Int { compressedRows * recordBytes }
    public var indexerKeyBytes: Int { indexerKeyWidth * MemoryLayout<UInt16>.stride }

    public init(
        width: Int,
        precision: DeepSeekV41CachePrecision,
        windowRows: Int,
        compressedRows: Int,
        indexerKeyWidth: Int,
        candidateBlocks: Int,
        contextLength: Int = 0
    ) {
        self.width = width
        self.precision = precision
        self.windowRows = windowRows
        self.compressedRows = compressedRows
        self.indexerKeyWidth = indexerKeyWidth
        self.candidateBlocks = candidateBlocks
        self.contextLength = max(0, contextLength)
    }

    /// Rows one owner with `ratio` produces over the provisioned context.
    public func compressedRows(forRatio ratio: Int) -> Int {
        guard contextLength > 0, ratio > 0 else { return 0 }
        return contextLength / ratio + 1
    }

    /// Geometry for a config. `contextLength` is the sequence length the turn may
    /// reach; compressed rows are provisioned for it in full so a turn never has
    /// to grow the allocation mid-generation.
    public static func make(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan],
        contextLength: Int,
        precision: DeepSeekV41CachePrecision = .fp8Block32
    ) -> DeepSeekV41CacheGeometry {
        // Only cache OWNERS allocate compressed rows; a compressed layer that
        // reads its owner's rows must not widen every owner's allocation. Each
        // owner's row count comes from its own ratio.
        let ownerRatios = plan.filter(\.ownsCompressedKV).map { max(1, $0.compressRatio) }
        let widestRatio = max(1, ownerRatios.max() ?? 1)
        let rows = max(0, contextLength) / widestRatio + 1
        return DeepSeekV41CacheGeometry(
            width: config.cacheRecordWidth,
            precision: precision,
            windowRows: max(1, config.windowSize),
            compressedRows: rows,
            indexerKeyWidth: max(0, config.indexHeadDim),
            candidateBlocks: config.candidateBlockSize > 0
                ? max(0, contextLength) / config.candidateBlockSize + 1
                : 0,
            contextLength: contextLength
        )
    }
}

// MARK: - Layout

/// Byte offsets of every cache region. `window` and `compressed` are per layer;
/// `indexer` exists only for index-owning layers and `candidate` only for the
/// candidate source layer.
nonisolated public struct DeepSeekV41CacheLayout: Sendable, Equatable {
    public let geometry: DeepSeekV41CacheGeometry
    /// Layer index -> byte offset inside the window buffer.
    public let windowBase: [Int: Int]
    /// Owner layer index -> byte offset inside the compressed buffer.
    public let compressedBase: [Int: Int]
    /// Owner layer index -> rows allocated to it. Owners with different compress
    /// ratios do not share a row count.
    public let compressedRowsByOwner: [Int: Int]
    /// Index-owning layer -> byte offset inside the indexer key buffer.
    public let indexerBase: [Int: Int]
    /// Index-owning layer -> rows allocated to it.
    public let indexerRowsByOwner: [Int: Int]
    /// Candidate-source layer -> byte offset inside the candidate score buffer.
    public let candidateBase: [Int: Int]

    public let windowBytes: Int
    public let compressedBytes: Int
    public let indexerBytes: Int
    public let candidateBytes: Int

    public var totalBytes: Int { windowBytes + compressedBytes + indexerBytes + candidateBytes }

    public init(geometry: DeepSeekV41CacheGeometry, plan: [DeepSeekV41LayerPlan]) {
        self.geometry = geometry

        var windowBase: [Int: Int] = [:]
        var compressedBase: [Int: Int] = [:]
        var compressedRowsByOwner: [Int: Int] = [:]
        var indexerBase: [Int: Int] = [:]
        var indexerRowsByOwner: [Int: Int] = [:]
        var candidateBase: [Int: Int] = [:]

        var windowCursor = 0
        for layer in plan {
            windowBase[layer.index] = windowCursor
            windowCursor += geometry.windowBytesPerLayer
        }
        var compressedCursor = 0
        var indexerCursor = 0
        var candidateCursor = 0
        for layer in plan {
            if layer.ownsCompressedKV {
                let rows = geometry.compressedRows(forRatio: layer.compressRatio)
                compressedBase[layer.index] = compressedCursor
                compressedRowsByOwner[layer.index] = rows
                compressedCursor += rows * geometry.recordBytes
            }
            if layer.ownsCompressedIndex {
                let rows = geometry.compressedRows(forRatio: layer.compressRatio)
                indexerBase[layer.index] = indexerCursor
                indexerRowsByOwner[layer.index] = rows
                indexerCursor += rows * geometry.indexerKeyBytes
            }
            if layer.isCandidateSource {
                candidateBase[layer.index] = candidateCursor
                candidateCursor += geometry.candidateBlocks * MemoryLayout<Float>.stride
            }
        }

        self.windowBase = windowBase
        self.compressedBase = compressedBase
        self.compressedRowsByOwner = compressedRowsByOwner
        self.indexerBase = indexerBase
        self.indexerRowsByOwner = indexerRowsByOwner
        self.candidateBase = candidateBase
        self.windowBytes = windowCursor
        self.compressedBytes = compressedCursor
        self.indexerBytes = indexerCursor
        self.candidateBytes = candidateCursor
    }

    /// One-line summary for the load log.
    public func describe() -> String {
        func mb(_ bytes: Int) -> String {
            String(format: "%.1f MB", Double(bytes) / 1_048_576.0)
        }
        return "window \(mb(windowBytes)), compressed \(mb(compressedBytes)), "
            + "indexer \(mb(indexerBytes)), candidates \(mb(candidateBytes)), "
            + "total \(mb(totalBytes))"
    }
}

// MARK: - Addressing

nonisolated public enum DeepSeekV41CacheAddressing {

    /// Physical row of a logical position inside a sliding-window ring.
    ///
    /// The ring length is `window_size`, so the row for position `p` is
    /// `p % windowRows`. The mapping is a pure function of the position, which is
    /// what lets a reused prefix still address its own rows: nothing has to be
    /// replayed as long as no position beyond the ring length was written.
    public static func windowRow(for position: Int, geometry: DeepSeekV41CacheGeometry) -> Int {
        guard geometry.windowRows > 0 else { return 0 }
        let row = position % geometry.windowRows
        return row < 0 ? row + geometry.windowRows : row
    }

    /// Byte offset of one window row of one layer.
    public static func windowByteOffset(
        layer: Int,
        position: Int,
        layout: DeepSeekV41CacheLayout
    ) -> Int? {
        guard let base = layout.windowBase[layer] else { return nil }
        let row = windowRow(for: position, geometry: layout.geometry)
        return base + row * layout.geometry.recordBytes
    }

    /// Number of compressed rows a position's history has produced. Position `p`
    /// has been seen `p + 1` times, and every full block of `ratio` positions
    /// pools into one row; a partial trailing block has not been pooled yet.
    public static func compressedRowCount(afterPosition position: Int, ratio: Int) -> Int {
        guard position >= 0, ratio > 0 else { return 0 }
        return (position + 1) / ratio
    }

    /// The `ratio` positions that pool into compressed row `row`.
    public static func poolRange(row: Int, ratio: Int) -> Range<Int> {
        guard ratio > 0, row >= 0 else { return 0..<0 }
        return (row * ratio)..<((row + 1) * ratio)
    }

    /// Byte offset of one compressed row of one owner layer.
    public static func compressedByteOffset(
        ownerLayer: Int,
        row: Int,
        layout: DeepSeekV41CacheLayout
    ) -> Int? {
        guard let base = layout.compressedBase[ownerLayer] else { return nil }
        let rows = layout.compressedRowsByOwner[ownerLayer] ?? layout.geometry.compressedRows
        guard row >= 0, row < rows else { return nil }
        return base + row * layout.geometry.recordBytes
    }

    /// Byte offset of one indexer key.
    public static func indexerByteOffset(
        layer: Int,
        row: Int,
        layout: DeepSeekV41CacheLayout
    ) -> Int? {
        guard let base = layout.indexerBase[layer] else { return nil }
        let rows = layout.indexerRowsByOwner[layer] ?? layout.geometry.compressedRows
        guard row >= 0, row < rows else { return nil }
        return base + row * layout.geometry.indexerKeyBytes
    }

    /// Byte offset of one candidate block score.
    public static func candidateByteOffset(
        layer: Int,
        block: Int,
        layout: DeepSeekV41CacheLayout
    ) -> Int? {
        guard let base = layout.candidateBase[layer] else { return nil }
        guard block >= 0, block < layout.geometry.candidateBlocks else { return nil }
        return base + block * MemoryLayout<Float>.stride
    }

    // MARK: Compressor group

    /// Which block of compressor slots each compressed-KV owner fills.
    ///
    /// `Compressor` keeps `kv_state` and `score_state` per module instance, so the
    /// slots one layer fills are never the slots another layer fills. Addressing a
    /// shared block by `position % ratio` alone does not give that: owners run in
    /// index order at every position, so the earlier slots of a block were last
    /// written by whichever owner ran latest at the previous position (layer 20,
    /// ratio 1, writes slot 0 every position), and a ratio-2 owner then pooled its
    /// own latent together with that other layer's. One block per owner is what
    /// the reference has, and it also makes a group survive a turn boundary
    /// without a separate state machine.
    public static func compressorOwnerSlots(plan: [DeepSeekV41LayerPlan]) -> [Int: Int] {
        var out: [Int: Int] = [:]
        var ordinal = 0
        for layer in plan.sorted(by: { $0.index < $1.index }) where layer.ownsCompressedKV {
            out[layer.index] = ordinal
            ordinal += 1
        }
        return out
    }

    /// Byte offset of one slot in one owner's compressor group.
    ///
    /// `groupWidth` is one owner's block in slots (the widest ratio the checkpoint
    /// declares), so a narrow owner leaves its spare slots unused instead of
    /// reaching into the next owner's block.
    public static func compressorSlotByteOffset(
        owner: Int,
        slot: Int,
        groupWidth: Int,
        width: Int
    ) -> Int? {
        guard owner >= 0, slot >= 0, width > 0, groupWidth > 0, slot < groupWidth else {
            return nil
        }
        return (owner * groupWidth + slot) * width * MemoryLayout<Float>.stride
    }

    /// The compressed rows a position's compressed attentions may read: every
    /// row produced so far, capped to the fewest the token budget allows.
    ///
    /// `topKRows` is the indexer's selection when the layer runs an indexer, and
    /// 0 (meaning "all rows") when it does not. The caller resolves the selection
    /// into an explicit row list; this only bounds it.
    ///
    /// These rows can overlap the sliding window. A block of `ratio` positions is
    /// pooled once complete, so when `ratio <= windowSize` the newest block covers
    /// positions the window also reads, and the same latent is attended over twice
    /// (once as a window row, once as part of a pooled row). That is the reference
    /// behavior for this architecture, not an accounting bug: the scheduler keeps
    /// the two kinds as separate rows and the softmax sees both.
    public static func compressedRowsForAttention(
        position: Int,
        ratio: Int,
        ownerRowCount: Int,
        topKRows: Int
    ) -> Range<Int> {
        let produced = min(compressedRowCount(afterPosition: position, ratio: ratio), ownerRowCount)
        guard produced > 0 else { return 0..<0 }
        if topKRows <= 0 || topKRows >= produced { return 0..<produced }
        return (produced - topKRows)..<produced
    }

    /// The live window positions for a query at `position`: the trailing
    /// `windowSize` positions, never before the start of the sequence.
    public static func windowRange(position: Int, windowSize: Int) -> Range<Int> {
        guard position >= 0, windowSize > 0 else { return 0..<0 }
        let start = max(0, position - windowSize + 1)
        return start..<(position + 1)
    }

    /// Whether a cache holding `storedPositions` positions can serve a prefix of
    /// `prefixCount` positions.
    ///
    /// Two bounds apply. The count must be stored at all, and it must fit inside the
    /// window ring: past `windowRows` positions the oldest window rows have been
    /// overwritten, so a prefix that "exists" by count would still attend stale
    /// rows even though its compressed rows are intact.
    public static func prefixIsReusable(
        prefixCount: Int,
        storedPositions: Int,
        windowRows: Int
    ) -> Bool {
        guard prefixCount > 0 else { return true }
        guard storedPositions <= max(0, windowRows) else { return false }
        guard prefixCount <= storedPositions else { return false }
        return prefixCount <= max(0, windowRows)
    }
}

// MARK: - Row plan

/// The cache rows one attention stage reads for one position: the live window,
/// then the compressed rows, both resolved by the same rule on both backends.
///
/// This exists so the host never has to read the cache back to know what to
/// attend over. Selection is a function of the position and the ratio alone, so
/// the row count is known before any dispatch and the CPU reference and the
/// Metal executor cannot drift. The indexer's block scores are recorded (they
/// are what `Tools/dsv41_probe.py` compares against), but they do not choose
/// rows: `DeepSeekV41Conformance.candidateSelection` is still unverified, and a
/// value-dependent selection would need a per-layer device-to-host readback of
/// the count, which is exactly the kind of silent divergence this file is
/// trying to avoid.
nonisolated public struct DeepSeekV41AttentionRowPlan: Sendable, Equatable {
    /// Logical positions in the sliding window, oldest first.
    public let windowPositions: [Int]
    /// Those same positions as physical rows inside the layer's window ring.
    public let windowRows: [Int]
    /// Compressed row indices, ascending, read from the layer's owner cache.
    public let compressedRows: [Int]

    public var totalRows: Int { windowPositions.count + compressedRows.count }

    public init(windowPositions: [Int], windowRows: [Int], compressedRows: [Int]) {
        self.windowPositions = windowPositions
        self.windowRows = windowRows
        self.compressedRows = compressedRows
    }
}

public extension DeepSeekV41CacheAddressing {

    /// Resolves the rows one attention stage reads.
    ///
    /// - Parameters:
    ///   - ownerRowLimit: how many rows the owner's compressed cache was
    ///     provisioned for. The reference passes `.max` (it keeps every row it
    ///     has ever pooled); the device path passes the allocation, because a
    ///     row beyond it has no buffer to read.
    nonisolated static func attentionRowPlan(
        layer: DeepSeekV41LayerPlan,
        position: Int,
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan],
        geometry: DeepSeekV41CacheGeometry,
        ownerRowLimit: Int = .max
    ) -> DeepSeekV41AttentionRowPlan {
        let windowPositions = Array(windowRange(position: position, windowSize: config.windowSize))
        let windowRows = windowPositions.map { windowRow(for: $0, geometry: geometry) }
        guard layer.flavor == .compressed, layer.compressedKVSource >= 0 else {
            return DeepSeekV41AttentionRowPlan(
                windowPositions: windowPositions,
                windowRows: windowRows,
                compressedRows: []
            )
        }
        // The owner pools with ITS ratio, so that is the row count that exists;
        // the reading layer's ratio only bounds what it may look at.
        let ownerRatio = plan.first(where: { $0.index == layer.compressedKVSource })?.compressRatio
            ?? layer.compressRatio
        let produced = min(
            compressedRowCount(afterPosition: position, ratio: max(1, ownerRatio)),
            max(0, ownerRowLimit)
        )
        let rows = compressedRowsForAttention(
            position: position,
            ratio: max(1, layer.compressRatio),
            ownerRowCount: produced,
            topKRows: config.indexTopK
        )
        return DeepSeekV41AttentionRowPlan(
            windowPositions: windowPositions,
            windowRows: windowRows,
            compressedRows: Array(rows)
        )
    }
}

// MARK: - Handle

/// The resident CSA2 cache. One buffer per region (`MTLBuffer`s are allocated in
/// one call each, so the hot path only ever adds an offset to a base pointer).
nonisolated public final class DeepSeekV41KVCache: @unchecked Sendable {

    public let layout: DeepSeekV41CacheLayout
    public let windowBuffer: MTLBuffer?
    public let compressedBuffer: MTLBuffer?
    public let indexerBuffer: MTLBuffer?
    public let candidateBuffer: MTLBuffer?

    /// Number of positions stored so far, or 0 when the cache is empty. Prefix
    /// reuse reads it: a turn whose prompt matches the pinned prefix up to
    /// `storedPositions` can start decoding at that position. Guarded because the
    /// class is `@unchecked Sendable` and the reset path can run off the turn's
    /// thread.
    public var storedPositions: Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        return storedValue
    }

    private let stateLock = NSLock()
    private var storedValue: Int = 0

    public init?(device: MTLDevice, layout: DeepSeekV41CacheLayout) {
        self.layout = layout
        func allocate(_ bytes: Int) -> MTLBuffer? {
            guard bytes > 0 else { return nil }
            return device.makeBuffer(length: bytes, options: .storageModePrivate)
                ?? device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard layout.windowBytes > 0, let window = allocate(layout.windowBytes) else {
            return nil
        }
        self.windowBuffer = window
        self.compressedBuffer = allocate(layout.compressedBytes)
        self.indexerBuffer = allocate(layout.indexerBytes)
        self.candidateBuffer = allocate(layout.candidateBytes)
    }

    /// Records that `positions` have been appended. A smaller value than what is
    /// already stored means the caller is starting a fresh sequence.
    public func noteStored(_ count: Int) {
        stateLock.lock()
        storedValue = max(0, count)
        stateLock.unlock()
    }

    public func reset() {
        stateLock.lock()
        storedValue = 0
        stateLock.unlock()
    }

    /// Whether the cache still holds every position up to `prefixCount` intact.
    ///
    /// Two bounds apply, not one. The compressed rows survive for the whole
    /// context, but the window rings keep only the newest `windowRows` positions:
    /// once the sequence is longer than the ring, the oldest window rows have been
    /// overwritten by later positions, so a "hit" there would attend stale rows
    /// even though the position count says otherwise.
    public func canReuse(prefixCount: Int) -> Bool {
        DeepSeekV41CacheAddressing.prefixIsReusable(
            prefixCount: prefixCount,
            storedPositions: storedPositions,
            windowRows: layout.geometry.windowRows
        )
    }
}
