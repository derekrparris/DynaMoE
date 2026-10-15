//
//  EngineRuntime.swift
//  DynaMoE
//
//  Engine-runtime state moved verbatim out of ContentView.swift so the
//  inference machinery (KV cache, working-set budgeting, prefill pools,
//  expert transition tracking) lives headless and shareable with the
//  API server. Behavior is unchanged. Extraction contract documented at
//  the top of CompletionEngine.swift.
//

import Foundation
import Metal

// MARK: - KV Precision

enum KVCachePrecision: String, CaseIterable, Identifiable {
    case fp16 = "FP16 (Half - 50% Memory)"
    case fp8 = "FP8 (8-Bit Quantized - 75% Memory)"
    case fp32 = "FP32 (Full Precision)"

    var id: String { rawValue }

    var bytesPerElement: Int {
        switch self {
        case .fp16: return 2
        case .fp8: return 1
        case .fp32: return 4
        }
    }
}

/// Single source of truth for JetSpec eligibility (used by startAutoregressiveGeneration
/// and covered by DynaMoETests). FP8 KV is excluded: the tree store/verify kernels are
/// F16/FP32-only, and the FP32 fallback would write 4-byte elements into a
/// 1-byte-per-element cache.
nonisolated func isJetSpecEligible(jetSpecEnabled: Bool, hasLinearRecurrence: Bool, isSparkModel: Bool, kvPrecision: KVCachePrecision) -> Bool {
    jetSpecEnabled && !hasLinearRecurrence && !isSparkModel && kvPrecision != .fp8
}

/// Batched-prefill chunk cap for Gemma 4: tokens per `runLayerWisePrefill` call.
/// Bounds the sliding-window ring's slack (window + chunk, so an in-flight
/// batch never overwrites a row attention still needs) and the transient
/// per-call prefill buffers (~250 KB/token) that previously scaled with the
/// entire delta prompt.
let gemmaPrefillChunkCap = 512

/// Reusable transient buffers for the batched prefill (`runLayerWisePrefill`).
/// The chunked Gemma prefill re-enters that function once per 512-token chunk;
/// re-allocating the ~3 GB of double-buffered expert staging (plus the per-chunk
/// scratch) on every chunk — while the weight residency grows toward the memory
/// budget — can exhaust the allocator mid-prefill: `makeBuffer` returns nil and
/// the chunk silently fails (observed deterministically at the third chunk of a
/// fresh 3k-token prompt at FP8 on a 16 GB machine). The pool allocates each
/// named buffer once per prefill (on the first chunk, while residency is still
/// low) and reuses it across chunks; the generation loop drains it when prefill
/// ends so the decode phase never holds the staging.
final class PrefillTransientPool: @unchecked Sendable {
    static let shared = PrefillTransientPool()
    private var buffers: [String: MTLBuffer] = [:]
    private init() {}

    /// Returns a pooled buffer for `name` with at least `byteCount` bytes,
    /// re-allocating (growing) only when the pooled one is too small.
    func buffer(device: MTLDevice, name: String, byteCount: Int) -> MTLBuffer? {
        if let existing = buffers[name], existing.length >= byteCount { return existing }
        guard let made = device.makeBuffer(length: byteCount, options: .storageModeShared) else { return nil }
        buffers[name] = made
        return made
    }

    /// Releases every pooled buffer (call when prefill completes or fails).
    func drain() { buffers.removeAll() }

    var heldBytes: Int { buffers.values.reduce(0) { $0 + $1.length } }
}


// MARK: - KV Cache Manager

final class KVCacheManager {
    static let shared = KVCacheManager()

    var kCacheBuffer: MTLBuffer?
    var vCacheBuffer: MTLBuffer?
    var kScaleBuffer: MTLBuffer?
    var vScaleBuffer: MTLBuffer?
    var linearStateBuffer: MTLBuffer?
    var convStateBuffer: MTLBuffer?
    var allocatedSeqLen: Int = 0
    var allocatedKvBytes: Int = 0
    /// Geometry of the resident KV allocation, so a buffer laid out for a
    /// different model / stride / precision is never reused (it also pinned a
    /// stale multi-GB allocation when it was).
    private var allocatedSlotCount: Int = 0
    private var allocatedKStride: Int = 0
    private var allocatedVStride: Int = 0
    private var allocatedElementBytes: Int = 0
    var activePrecision: KVCachePrecision = .fp16
    private var device: MTLDevice?

    // Gemma 4 sliding-window ring KV cache. The 25 sliding layers only ever
    // attend within effectiveSlidingWindow of the current position, so their KV
    // slots hold window + slack ring positions instead of maxSeqLen linear
    // positions: physical row = logical position % slidingRingLen. The slack
    // covers the largest batched prefill chunk and the JetSpec tree scratch
    // (node slots step..step+N-1), so in-flight batches never overwrite a row
    // an attention pass still needs. Empty arrays = uniform maxSeqLen layout
    // (every non-Gemma model, and Gemma's global layers).
    var slidingRingLen: Int = 0
    var slotIsRing: [Bool] = []
    /// Per-slot position base (prefix sum over slots) into the K/V buffers.
    var slotPosBase: [Int] = []
    /// Per-slot position capacity (ringLen for ring slots, maxSeqLen otherwise).
    var slotPosCount: [Int] = []
    /// Per ring slot, the logical position whose K/V the row currently holds
    /// (-1 = never written). Exact bookkeeping for prefix-splice validation:
    /// a pinned prefix is only spliced when every reused row still holds its
    /// own position, so a turn that ran past the pin boundary (overwriting the
    /// window) forces a full re-prefill instead of attending stale rows.
    private var ringSlotLastPos: [Int] = []

    /// Position base of a slot's region (uniform layout falls back to slot * allocatedSeqLen).
    func kvSlotPositionBase(_ slot: Int) -> Int {
        guard slotPosBase.indices.contains(slot) else { return slot * allocatedSeqLen }
        return slotPosBase[slot]
    }

    /// Position capacity of a slot's region.
    func kvSlotPositionCount(_ slot: Int) -> Int {
        guard slotPosCount.indices.contains(slot) else { return allocatedSeqLen }
        return slotPosCount[slot]
    }

    /// Byte offset of a slot's region within the shared K/V cache buffers.
    func kvSlotByteOffset(slot: Int, stride: Int, elementBytes: Int) -> Int {
        kvSlotPositionBase(slot) * stride * elementBytes
    }

    /// Byte offset of a slot's region within the FP8 per-head scale buffers
    /// (layout: [positions, logicalHeads] halves per slot).
    func kvSlotScaleByteOffset(slot: Int, layoutHeads: Int) -> Int {
        kvSlotPositionBase(slot) * layoutHeads * MemoryLayout<UInt16>.stride
    }

    /// Whether the slot is a sliding-window ring slot (Gemma 4 sliding layers).
    func isRingSlot(_ slot: Int) -> Bool {
        slidingRingLen > 0 && slotIsRing.indices.contains(slot) && slotIsRing[slot]
    }

    /// Physical row for a logical position in a slot (linear slots pass through).
    func ringSlotForRow(_ pos: Int, slot: Int) -> Int {
        isRingSlot(slot) ? pos % slidingRingLen : pos
    }

    /// Records that [range.lowerBound, range.upperBound) were dispatched into
    /// the ring. No-op unless the ring layout is active.
    func noteRingStores(range: Range<Int>) {
        guard slidingRingLen > 0, ringSlotLastPos.count == slidingRingLen, !range.isEmpty else { return }
        for pos in range where pos >= 0 {
            ringSlotLastPos[pos % slidingRingLen] = pos
        }
    }

    /// True when every ring row the pinned prefix still needs holds exactly
    /// its own position (nothing stored past the pin overwrote the window).
    func ringPrefixRowsIntact(prefixCount: Int, window: Int) -> Bool {
        guard slidingRingLen > 0, window > 0 else { return true }
        return KVCacheManager.ringRowsIntact(
            lastPos: ringSlotLastPos, ringLen: slidingRingLen,
            prefixCount: prefixCount, window: window
        )
    }

    private static func ringRowsIntact(lastPos: [Int], ringLen: Int, prefixCount: Int, window: Int) -> Bool {
        guard ringLen > 0, window > 0, lastPos.count == ringLen, prefixCount > 0 else { return false }
        let lo = max(0, prefixCount - window)
        for pos in lo..<prefixCount {
            if pos < 0 || lastPos[pos % ringLen] != pos { return false }
        }
        return true
    }

    // FIX #10: snapshots of the GDN recurrent + conv states taken at the moment
    // the pinned prefix token list is recorded (turn end). When the next turn's
    // prompt matches the ENTIRE pinned prefix, these restore the O(1) recurrence
    // state so only the delta tokens need prefilling.
    private var prefixLinearStateBuffer: MTLBuffer?
    private var prefixConvStateBuffer: MTLBuffer?
    private var prefixLinearStateBytes: Int = 0
    private var prefixConvStateBytes: Int = 0
    public private(set) var linearStatesPinned: Bool = false

    /// Captures the live GDN recurrent + conv states into prefix snapshots.
    /// Call at the exact moment the pinned token list is recorded so the state
    /// corresponds to the pinned token count.
    func captureLinearStates() {
        if let live = linearStateBuffer {
            if prefixLinearStateBuffer == nil || prefixLinearStateBytes < live.length {
                prefixLinearStateBuffer = device?.makeBuffer(length: live.length, options: .storageModeShared)
                prefixLinearStateBytes = live.length
            }
            if let snap = prefixLinearStateBuffer, snap.length >= live.length {
                memcpy(snap.contents(), live.contents(), live.length)
                linearStatesPinned = true
            }
        }
        if let live = convStateBuffer {
            if prefixConvStateBuffer == nil || prefixConvStateBytes < live.length {
                prefixConvStateBuffer = device?.makeBuffer(length: live.length, options: .storageModeShared)
                prefixConvStateBytes = live.length
            }
            if let snap = prefixConvStateBuffer, snap.length >= live.length {
                memcpy(snap.contents(), live.contents(), live.length)
            }
        }
    }

    func reset(
        device: MTLDevice,
        config: ModelConfig? = nil,
        actualLayers: Int = 40,
        totalLoops: Int = 1,
        numKvHeads: Int = 8,
        headDim: Int = 128,
        maxSeqLen: Int = 2048,
        precision: KVCachePrecision = .fp16,
        preservePrefixCount: Int = 0,
        slidingSlotFlags: [Bool]? = nil,
        slidingRingLen ringLenParam: Int = 0
    ) {
        let oldMaxSeq = self.allocatedSeqLen
        let wasAllocated = (kCacheBuffer != nil)
        let oldSlotIsRing = self.slotIsRing
        let oldSlotPosBase = self.slotPosBase
        let oldSlotPosCount = self.slotPosCount
        let oldSlidingRingLen = self.slidingRingLen
        let oldRingSlotLastPos = self.ringSlotLastPos
        self.allocatedSeqLen = maxSeqLen
        self.activePrecision = precision
        self.device = device

        let isLing = (config?.isLingModel == true)
        // Layout derived by the shared, pure provisioning math so the budget
        // clamp in the caller sees exactly the bytes this allocation commits.
        let geometry = KVCacheProvisioning.geometry(
            config: config,
            actualLayers: actualLayers,
            totalLoops: totalLoops,
            numKvHeads: numKvHeads,
            headDim: headDim,
            precision: precision,
            slidingSlotFlags: slidingSlotFlags,
            slidingRingLen: ringLenParam
        )
        let totalSlots = geometry.totalSlots
        let kStride = geometry.kStride
        let vStride = geometry.vStride
        let elementBytes = geometry.elementBytes
        let kvHeads = geometry.kvHeadCapacity

        // Sliding-window ring layout (Gemma 4): per-slot position capacity is
        // ringLen for flagged (sliding) slots and maxSeqLen for the rest. The
        // arrays are only populated when at least one slot rings; otherwise the
        // helpers below fall back to the uniform slot * maxSeqLen layout.
        let ringLen = max(0, ringLenParam)
        let ringFlags = slidingSlotFlags ?? []
        var newSlotIsRing = [Bool](repeating: false, count: totalSlots)
        var newSlotPosBase = [Int](repeating: 0, count: totalSlots)
        var newSlotPosCount = [Int](repeating: maxSeqLen, count: totalSlots)
        var totalPositions = 0
        for slot in 0..<totalSlots {
            let rings = (ringLen > 0 && slot < ringFlags.count && ringFlags[slot])
            newSlotIsRing[slot] = rings
            newSlotPosBase[slot] = totalPositions
            newSlotPosCount[slot] = rings ? ringLen : maxSeqLen
            totalPositions += newSlotPosCount[slot]
        }
        let ringLayoutActive = newSlotIsRing.contains(true)
        self.slotIsRing = ringLayoutActive ? newSlotIsRing : []
        self.slotPosBase = ringLayoutActive ? newSlotPosBase : []
        self.slotPosCount = ringLayoutActive ? newSlotPosCount : []
        self.slidingRingLen = ringLayoutActive ? ringLen : 0
        // Ring bookkeeping: keep the per-slot last-position table only when the
        // layout and ring length are unchanged AND the preserved rows validated
        // (the caller validates before reset and drops the pin otherwise); any
        // other transition restarts from "never written" so the next turn's
        // splice check conservatively fails until rows are re-stored.
        if ringLayoutActive {
            if preservePrefixCount > 0, oldSlidingRingLen == ringLen, oldRingSlotLastPos.count == ringLen,
               KVCacheManager.ringRowsIntact(lastPos: oldRingSlotLastPos, ringLen: ringLen, prefixCount: preservePrefixCount, window: config?.effectiveSlidingWindow ?? 0) {
                self.ringSlotLastPos = oldRingSlotLastPos
            } else {
                self.ringSlotLastPos = [Int](repeating: -1, count: ringLen)
            }
        } else {
            self.ringSlotLastPos = []
        }

        // kStride/vStride come from the shared geometry (Ling MLA keeps its
        // 3072/2048 halfs there); kBytes/vBytes mirror KVCacheProvisioning.requiredBytes.
        let kBytes = KVCacheProvisioning.positions(maxSeqLen: maxSeqLen, geometry: geometry) * kStride * elementBytes
        let vBytes = KVCacheProvisioning.positions(maxSeqLen: maxSeqLen, geometry: geometry) * vStride * elementBytes
        let requiredKvBytes = max(kBytes, vBytes)
        assert(
            totalPositions == KVCacheProvisioning.positions(maxSeqLen: maxSeqLen, geometry: geometry),
            "KV slot bookkeeping drifted from KVCacheProvisioning.positions — the budget clamp would mis-size"
        )

        let oldK = self.kCacheBuffer
        let oldV = self.vCacheBuffer
        // Lazy commit: a fresh MTLBuffer is zero-filled by the OS on demand, so pages the
        // model never touches are never committed to RAM. Releasing and reallocating is
        // far cheaper than memset-ing every page, which commits them all up front and
        // invites eviction + KV-cache paging thrash at long context.
        let forceFreshAlloc = (preservePrefixCount == 0)
        // A preserved reset whose maxSeqLen differs from the last allocation must
        // relayout every slot region even when the retained buffers still have spare
        // capacity: neededSeqLen can SHRINK turn-to-turn (a tools_unload shortening the
        // prompt above the min-sequence floor, or a lower max-tokens setting), and the
        // generation paths index slot regions by allocatedSeqLen — leaving the pinned
        // prefix at the old stride while scales move to the new one would misalign every
        // later layer. Same condition as the FP8 scale-buffer rebuild below.
        let spliceStrideChanged = (preservePrefixCount > 0 && maxSeqLen != oldMaxSeq)
        // A ring-layout change (first ring allocation, or a different ring length)
        // moves every slot region; a retained buffer would keep the old layout.
        let spliceLayoutChanged = (preservePrefixCount > 0 && (oldSlotIsRing != self.slotIsRing || oldSlotPosCount != self.slotPosCount || oldSlidingRingLen != self.slidingRingLen))
        // A different slot count / stride / element size means the retained buffer
        // describes another model's cache layout entirely. It can never be reused
        // safely, and keeping it also pinned a stale multi-GB allocation: a 48-slot
        // 32k cache (3 GiB, exact) left over from a previous model survived a switch
        // to a 36-slot model because the reuse check only compared *capacity*.
        let geometryChanged = wasAllocated && (allocatedSlotCount != totalSlots
            || allocatedKStride != kStride
            || allocatedVStride != vStride
            || allocatedElementBytes != elementBytes)
        let needsRealloc = forceFreshAlloc || geometryChanged
            || (kCacheBuffer == nil || vCacheBuffer == nil || allocatedKvBytes < requiredKvBytes || spliceStrideChanged || spliceLayoutChanged)

        if needsRealloc {
            self.kCacheBuffer = nil
            self.vCacheBuffer = nil
            self.kCacheBuffer = device.makeBuffer(length: kBytes, options: .storageModeShared)
            self.vCacheBuffer = device.makeBuffer(length: vBytes, options: .storageModeShared)
            self.allocatedKvBytes = requiredKvBytes
            self.allocatedSlotCount = totalSlots
            self.allocatedKStride = kStride
            self.allocatedVStride = vStride
            self.allocatedElementBytes = elementBytes
        }

        if preservePrefixCount > 0 && wasAllocated {
            // Per-slot region bases: the ring layout when active, uniform otherwise.
            // Ring slots preserve only the last `window` rows (all a future turn
            // can ever read, and exactly the rows the caller validated); linear
            // slots preserve the whole [0, pin) prefix as before.
            let ringWindow = max(0, config?.effectiveSlidingWindow ?? 0)
            let rowBytesK = kStride * elementBytes
            let rowBytesV = vStride * elementBytes
            for slot in 0..<totalSlots {
                if ringLayoutActive && newSlotIsRing[slot] {
                    let live = min(preservePrefixCount, ringWindow)
                    if needsRealloc, let oldKBuf = oldK, let oldVBuf = oldV, let newKBuf = kCacheBuffer, let newVBuf = vCacheBuffer {
                        // Slot region bases are position counts; scale them to bytes
                        // before adding row offsets.
                        let slotBaseOldBytes = ((slot < oldSlotPosBase.count) ? oldSlotPosBase[slot] : slot * oldMaxSeq) * kStride * elementBytes
                        let slotBaseNewBytes = newSlotPosBase[slot] * kStride * elementBytes
                        let oldRings = (oldSlidingRingLen > 0 && slot < oldSlotIsRing.count && oldSlotIsRing[slot])
                        for pos in (preservePrefixCount - live)..<preservePrefixCount {
                            let oldRow = oldRings ? (pos % oldSlidingRingLen) : pos
                            let newRow = pos % ringLen
                            let kSrc = slotBaseOldBytes + oldRow * rowBytesK
                            let kDst = slotBaseNewBytes + newRow * rowBytesK
                            let vSrc = slotBaseOldBytes + oldRow * rowBytesV
                            let vDst = slotBaseNewBytes + newRow * rowBytesV
                            if kSrc + rowBytesK <= oldKBuf.length && kDst + rowBytesK <= newKBuf.length {
                                memcpy(newKBuf.contents().advanced(by: kDst), oldKBuf.contents().advanced(by: kSrc), rowBytesK)
                            }
                            if vSrc + rowBytesV <= oldVBuf.length && vDst + rowBytesV <= newVBuf.length {
                                memcpy(newVBuf.contents().advanced(by: vDst), oldVBuf.contents().advanced(by: vSrc), rowBytesV)
                            }
                        }
                    }
                    // Retained buffer (no realloc): the ring mapping is unchanged,
                    // so the live rows are already exactly where the next turn
                    // reads them — nothing to copy.
                    continue
                }
                let newKLayerByteOffset = (ringLayoutActive ? newSlotPosBase[slot] : slot * maxSeqLen) * kStride * elementBytes
                let newVLayerByteOffset = (ringLayoutActive ? newSlotPosBase[slot] : slot * maxSeqLen) * vStride * elementBytes
                let prefixKBytes = min(preservePrefixCount, maxSeqLen) * kStride * elementBytes
                let prefixVBytes = min(preservePrefixCount, maxSeqLen) * vStride * elementBytes

                if needsRealloc, let oldKBuf = oldK, let oldVBuf = oldV, let newKBuf = kCacheBuffer, let newVBuf = vCacheBuffer {
                    let oldSlotBase = (slot < oldSlotPosBase.count) ? oldSlotPosBase[slot] : slot * oldMaxSeq
                    let oldKLayerByteOffset = oldSlotBase * kStride * elementBytes
                    let oldVLayerByteOffset = oldSlotBase * vStride * elementBytes
                    let copyKBytes = min(prefixKBytes, max(0, oldKBuf.length - oldKLayerByteOffset), max(0, newKBuf.length - newKLayerByteOffset))
                    let copyVBytes = min(prefixVBytes, max(0, oldVBuf.length - oldVLayerByteOffset), max(0, newVBuf.length - newVLayerByteOffset))
                    if copyKBytes > 0 {
                        memcpy(newKBuf.contents().advanced(by: newKLayerByteOffset), oldKBuf.contents().advanced(by: oldKLayerByteOffset), copyKBytes)
                    }
                    if copyVBytes > 0 {
                        memcpy(newVBuf.contents().advanced(by: newVLayerByteOffset), oldVBuf.contents().advanced(by: oldVLayerByteOffset), copyVBytes)
                    }
                }

                // The tail [preservePrefixCount, maxSeqLen) is NOT cleared. Every KV
                // position is written before it is ever read (prefill writes the delta
                // above the pinned prefix, decode writes each step before attending),
                // so stale bytes are never observed. Zeroing the tail here touched every
                // page of the whole cache — on a 30-layer hybrid model that committed
                // multi-GB of anonymous memory the VM would otherwise keep lazy.
            }
        } else {
            // Freshly allocated shared MTLBuffers are zero-filled by the VM on demand;
            // no memset needed. Only memset when reusing a buffer that held prior data.
            if !forceFreshAlloc, !needsRealloc {
                if let kBuf = kCacheBuffer { memset(kBuf.contents(), 0, min(kBytes, kBuf.length)) }
                if let vBuf = vCacheBuffer { memset(vBuf.contents(), 0, min(vBytes, vBuf.length)) }
            }
        }

        if precision == .fp8 {
            let scaleCount = totalPositions * kvHeads
            let scaleBytes = scaleCount * MemoryLayout<UInt16>.stride // half precision per-head scales
            let scaleElemBytes = MemoryLayout<UInt16>.stride
            let oldKScale = self.kScaleBuffer
            let oldVScale = self.vScaleBuffer
            // The FP8 store/attention kernels address scales with the LOGICAL head count
            // the generation path uses (config?.effectiveNumKeyValueHeads ?? numKvHeads —
            // e.g. 2 when a nil-config run defaults to 2 KV heads), laying each slot out
            // as [maxSeq x logicalHeads] halves. `kvHeads` above is only a capacity pad
            // (>= 8 when config is nil); using it for the restore's offsets/sizes would
            // copy the wrong regions after a stride change and leave later slots reading
            // zero or foreign scales. Copy with the logical count; keep the padded count
            // for sizing/zeroing so over-provisioned tails stay clean.
            //
            // The allocation condition below is capacity-based, so an earlier, longer
            // conversation can leave scale buffers big enough for BOTH the old and new
            // strides. A spliced reset that changes the stride must then relayout
            // slot-by-slot between the oldMaxSeq and maxSeqLen layouts inside the SAME
            // buffer — and a forward slot loop would overwrite a later slot's source
            // data before it is copied (slot 1's 2048-stride destination overlaps slot
            // 2's 1024-stride source). memmove would not help: the overlap is between
            // separate slot regions, not within one copy. Rebuild both buffers on
            // every preserve-with-stride-change so the restore is always
            // buffer-to-buffer; fresh resets keep the capacity-retention fast path.
            let scaleLayoutHeads = config?.effectiveNumKeyValueHeads ?? numKvHeads
            if kScaleBuffer == nil || kScaleBuffer!.length < scaleBytes || spliceStrideChanged || spliceLayoutChanged || geometryChanged {
                self.kScaleBuffer = device.makeBuffer(length: scaleBytes, options: .storageModeShared)
                self.vScaleBuffer = device.makeBuffer(length: scaleBytes, options: .storageModeShared)
            }
            if preservePrefixCount == 0 {
                if let ksBuf = kScaleBuffer { memset(ksBuf.contents(), 0, min(scaleBytes, ksBuf.length)) }
                if let vsBuf = vScaleBuffer { memset(vsBuf.contents(), 0, min(scaleBytes, vsBuf.length)) }
            } else {
                // The preserved INT8 K/V prefix is meaningless without its per-(token,
                // head) dequant scales: with zeroed scales every pinned position's score
                // and V read collapses to zero and decode degenerates into word salad.
                // Rebuilding the scale buffers on a grown maxSeqLen throws the pinned
                // prefix's scales away — restore them slot-by-slot exactly like the
                // K/V memcpy above, using the LOGICAL head-count layout the store
                // kernels write (scaleLayoutHeads), not the padded capacity count.
                // spliceStrideChanged above guarantees src and dst are distinct buffers
                // whenever the layouts differ; when the stride is unchanged a retained
                // buffer means every newOff equals oldOff and the loop is a no-op.
                // Ring slots (Gemma 4 sliding layers) restore only the last `window`
                // rows, mirroring the K/V ring preserve.
                let prefixScaleBytes = min(min(preservePrefixCount, oldMaxSeq), maxSeqLen) * scaleLayoutHeads * scaleElemBytes
                let scaleWasRebuilt = (kScaleBuffer !== oldKScale)
                let scaleRowBytes = scaleLayoutHeads * scaleElemBytes
                for slot in 0..<totalSlots {
                    let slotBaseOld = (slot < oldSlotPosBase.count) ? oldSlotPosBase[slot] : slot * oldMaxSeq
                    let slotBaseNew = ringLayoutActive ? newSlotPosBase[slot] : slot * maxSeqLen
                    let oldOff = slotBaseOld * scaleLayoutHeads * scaleElemBytes
                    let newOff = slotBaseNew * scaleLayoutHeads * scaleElemBytes
                    if ringLayoutActive && newSlotIsRing[slot] {
                        guard scaleWasRebuilt else { continue }
                        if let oldKS = oldKScale, let newKS = kScaleBuffer, let oldVS = oldVScale, let newVS = vScaleBuffer {
                            let live = min(preservePrefixCount, max(0, config?.effectiveSlidingWindow ?? 0))
                            let oldRings = (oldSlidingRingLen > 0 && slot < oldSlotIsRing.count && oldSlotIsRing[slot])
                            for pos in (preservePrefixCount - live)..<preservePrefixCount {
                                let oldRow = oldRings ? (pos % oldSlidingRingLen) : pos
                                let newRow = pos % ringLen
                                let srcK = oldOff + oldRow * scaleRowBytes
                                let dstK = newOff + newRow * scaleRowBytes
                                if srcK + scaleRowBytes <= oldKS.length && dstK + scaleRowBytes <= newKS.length {
                                    memcpy(newKS.contents().advanced(by: dstK), oldKS.contents().advanced(by: srcK), scaleRowBytes)
                                }
                                let srcV = oldOff + oldRow * scaleRowBytes
                                let dstV = newOff + newRow * scaleRowBytes
                                if srcV + scaleRowBytes <= oldVS.length && dstV + scaleRowBytes <= newVS.length {
                                    memcpy(newVS.contents().advanced(by: dstV), oldVS.contents().advanced(by: srcV), scaleRowBytes)
                                }
                            }
                        }
                        continue
                    }
                    guard scaleWasRebuilt || newOff != oldOff else { continue }
                    if let oldKS = oldKScale, let newKS = kScaleBuffer {
                        let copy = min(prefixScaleBytes, max(0, oldKS.length - oldOff), max(0, newKS.length - newOff))
                        if copy > 0 {
                            memcpy(newKS.contents().advanced(by: newOff), oldKS.contents().advanced(by: oldOff), copy)
                        }
                    }
                    if let oldVS = oldVScale, let newVS = vScaleBuffer {
                        let copy = min(prefixScaleBytes, max(0, oldVS.length - oldOff), max(0, newVS.length - newOff))
                        if copy > 0 {
                            memcpy(newVS.contents().advanced(by: newOff), oldVS.contents().advanced(by: oldOff), copy)
                        }
                    }
                }
                // Splice-turn telemetry: reallocation + scale-rebuild decisions decide
                // whether the prefix survives intact. Without this line, a run's binary
                // vintage and cache-lifecycle history are indistinguishable after the fact.
                print("🔍 [KVCACHE] splice reset: pin=\(preservePrefixCount) maxSeq \(oldMaxSeq)->\(maxSeqLen) kvRealloc=\(needsRealloc) scaleRebuilt=\(kScaleBuffer !== oldKScale) slots=\(totalSlots)")
            }
        }

        let linLayers = isLing ? 18 : actualLayers
        let linValHeads = isLing ? 16 : max(48, config?.effectiveLinearNumValueHeads ?? 32)
        let linHeadDim = max(128, config?.effectiveLinearValueHeadDim ?? 128)
        let linStateBytes = (isLing ? linLayers : max(linLayers, 48)) * linValHeads * linHeadDim * linHeadDim * MemoryLayout<Float>.stride
        if linearStateBuffer == nil || linearStateBuffer!.length < linStateBytes {
            self.linearStateBuffer = device.makeBuffer(length: linStateBytes, options: .storageModeShared)
        }
        // FIX #10: with a fully-matched prefix we restore the snapshot instead of
        // zeroing, so the delta prefill continues from the correct recurrence state.
        var canRestoreLinear = false
        if preservePrefixCount > 0 && linearStatesPinned,
           let snapS = prefixLinearStateBuffer,
           let liveS = linearStateBuffer,
           snapS.length >= min(linStateBytes, snapS.length),
           liveS.length >= min(linStateBytes, snapS.length) {
            let copyBytes = min(linStateBytes, liveS.length, snapS.length)
            // Self-check: at reset time nothing should have advanced the live state
            // since capture, so the restore should be a byte-identical no-op. A
            // mismatch means the state ran ahead of the pinned token list between
            // turns (e.g. an unpinned prefill-only pass) — surfacing it loudly.
            if memcmp(liveS.contents(), snapS.contents(), copyBytes) != 0 {
                var diffs = 0
                let a = liveS.contents().bindMemory(to: Float.self, capacity: copyBytes / 4)
                let b = snapS.contents().bindMemory(to: Float.self, capacity: copyBytes / 4)
                for i in 0..<(copyBytes / 4) where a[i] != b[i] { diffs += 1 }
                print("⚠️ [PREFIX] GDN state diverged from snapshot at restore: \(diffs)/\(copyBytes / 4) floats differ (pin=\(preservePrefixCount))")
            }
            memcpy(liveS.contents(), snapS.contents(), copyBytes)
            canRestoreLinear = true
        } else if isLing || config?.hasLinearRecurrence == true || preservePrefixCount == 0 {
            if let sBuf = linearStateBuffer { memset(sBuf.contents(), 0, min(linStateBytes, sBuf.length)) }
        }
        if preservePrefixCount == 0 {
            linearStatesPinned = false
        }

        let convChannels = isLing ? 6144 : max(10240, (config?.effectiveLinearNumValueHeads ?? 32) > 32 ? 10240 : 8192)
        let convBytes = (isLing ? linLayers : max(linLayers, 48)) * convChannels * 4 * MemoryLayout<Float>.stride
        if convStateBuffer == nil || convStateBuffer!.length < convBytes {
            self.convStateBuffer = device.makeBuffer(length: convBytes, options: .storageModeShared)
        }
        if canRestoreLinear, let snapC = prefixConvStateBuffer, let liveC = convStateBuffer, snapC.length >= min(convBytes, snapC.length), liveC.length >= min(convBytes, snapC.length) {
            let copyBytes = min(convBytes, liveC.length, snapC.length)
            memcpy(liveC.contents(), snapC.contents(), copyBytes)
        } else if isLing || config?.hasLinearRecurrence == true || preservePrefixCount == 0 {
            if let cBuf = convStateBuffer { memset(cBuf.contents(), 0, min(convBytes, cBuf.length)) }
        }
    }
}


// MARK: - Working Set & Expert Tracking

enum MemoryBudgetMode: String, CaseIterable, Identifiable {
    case lowMemory8GB = "8 GB"
    case balanced16GB = "16 GB"
    case unrestricted = "Unrestricted"

    var id: String { rawValue }

    var maxResidentExperts: Int {
        switch self {
        case .lowMemory8GB: return 480       // Up to 480 active resident experts (~2.36 GB FP8)
        case .balanced16GB: return 1280      // Up to 1,280 active resident experts (~6.29 GB FP8)
        case .unrestricted: return 65536     // All experts resident in RAM
        }
    }

    var targetMaxRssGB: Double {
        switch self {
        case .lowMemory8GB: return 6.5
        case .balanced16GB: return 11.5
        case .unrestricted: return 36.6
        }
    }
}

/// Pure KV-provisioning math. The allocation (`KVCacheManager.reset`) and the
/// budget clamp both derive their layout from `geometry(...)` here, so the
/// bytes a turn provisions and the bytes the clamp believes it provisions can
/// never drift apart.
enum KVCacheProvisioning {
    struct Geometry {
        var totalSlots: Int
        /// Per-slot ring flags (nil/empty = uniform layout).
        var slidingSlotFlags: [Bool]?
        var slidingRingLen: Int
        var kStride: Int
        var vStride: Int
        var elementBytes: Int
        /// Capacity-padded KV head count (the FP8 scale layout sizes against it).
        var kvHeadCapacity: Int = 8
    }

    /// Derives the KV layout exactly as `KVCacheManager.reset` did inline.
    static func geometry(
        config: ModelConfig?,
        actualLayers: Int,
        totalLoops: Int,
        numKvHeads: Int,
        headDim: Int,
        precision: KVCachePrecision,
        slidingSlotFlags: [Bool]?,
        slidingRingLen: Int
    ) -> Geometry {
        let isLing = (config?.isLingModel == true)
        let loops = max(totalLoops, config?.effectiveNumLoops ?? 1)
        let kvHeads = max(numKvHeads, config?.effectiveNumKeyValueHeads ?? 8)
        let hDim = max(headDim, config?.effectiveHeadDim ?? 128)
        // One KV slot per full-attention layer (indexed by fullAttnIndex, which is
        // < actualLayers). No fixed floor: padding a 30-layer hybrid model (Gemma 4)
        // to 44 slots wasted ~32% of the KV cache for nothing.
        let totalSlots = isLing ? (6 * loops) : (actualLayers * loops)
        let kvStride = kvHeads * hDim
        let isMla = (isLing || config?.kvLoraRank != nil)
        let effectiveKvStride = isMla ? 4096 : max(kvStride, 1024)
        return Geometry(
            totalSlots: totalSlots,
            slidingSlotFlags: slidingSlotFlags,
            slidingRingLen: max(0, slidingRingLen),
            kStride: isLing ? 3072 : effectiveKvStride,
            vStride: isLing ? 2048 : effectiveKvStride,
            elementBytes: precision.bytesPerElement,
            kvHeadCapacity: kvHeads
        )
    }

    /// Number of ring (sliding-window) slots in the layout.
    static func ringSlotCount(_ geometry: Geometry) -> Int {
        guard geometry.slidingRingLen > 0, let flags = geometry.slidingSlotFlags else { return 0 }
        let slots = max(0, geometry.totalSlots)
        return (0..<slots).reduce(0) { $0 + (($1 < flags.count && flags[$1]) ? 1 : 0) }
    }

    /// Positions (rows) provisioned across every slot for `maxSeqLen`. Ring
    /// slots are capped at the ring length and never grow with `maxSeqLen`.
    static func positions(maxSeqLen: Int, geometry: Geometry) -> Int {
        let flags = geometry.slidingSlotFlags ?? []
        var total = 0
        for slot in 0..<max(0, geometry.totalSlots) {
            let rings = geometry.slidingRingLen > 0 && slot < flags.count && flags[slot]
            total += rings ? geometry.slidingRingLen : max(0, maxSeqLen)
        }
        return total
    }

    /// Bytes the K and V caches together allocate for `maxSeqLen` — the amount
    /// the KV budget is actually spent on.
    static func requiredBytes(maxSeqLen: Int, geometry: Geometry) -> Int {
        let rows = positions(maxSeqLen: maxSeqLen, geometry: geometry)
        return rows * (geometry.kStride + geometry.vStride) * geometry.elementBytes
    }

    /// Largest sequence length whose KV allocation fits `budgetGB`, never below
    /// `floor`. An infinite budget (or a request already at/below the floor)
    /// passes through unchanged.
    static func clampedSeqLen(requested: Int, floor: Int, geometry: Geometry, budgetGB: Double) -> Int {
        guard requested > floor, budgetGB.isFinite else { return requested }
        let ringSlots = ringSlotCount(geometry)
        let linearSlots = max(0, geometry.totalSlots - ringSlots)
        let bytesPerLinearPosition = Double((geometry.kStride + geometry.vStride) * geometry.elementBytes)
        guard linearSlots > 0, bytesPerLinearPosition > 0 else { return requested }
        let fixedRingBytes = Double(ringSlots * geometry.slidingRingLen) * bytesPerLinearPosition
        let budgetBytes = max(0.0, budgetGB * 1_073_741_824)
        let affordable = Int((max(0.0, budgetBytes - fixedRingBytes)) / (Double(linearSlots) * bytesPerLinearPosition))
        return max(floor, min(requested, affordable))
    }

    /// KV byte budget for this machine, model, and budget mode. Weights and a
    /// fixed working reserve (scratch, prefill transients, UI, Metal overhead)
    /// come off the top of the process target; unrestricted modes fall back to
    /// 75% of physical RAM instead of having no ceiling at all — the mode that
    /// let a 16 GB Mac provision a 32k-token KV for a 4B model and thrash.
    static func budgetGB(physicalGB: Double, modelFootprintGB: Double, budgetMode: MemoryBudgetMode) -> Double {
        let processTargetGB = (budgetMode == .unrestricted) ? physicalGB * 0.75 : budgetMode.targetMaxRssGB
        let workingReserveGB = 2.5
        return max(0.25, processTargetGB - modelFootprintGB - workingReserveGB)
    }
}

struct ExpertSlice {
    let shardIndex: UInt32
    let offset: UInt64
    let length: UInt64
}

struct ExpertKey: Hashable {
    let layer: Int
    let expertId: Int
}

final class ExpertTransitionTracker {
    // [sourceLayer: [sourceExpertId: [targetExpertId: count]]]
    private var transitions: [Int: [Int: [Int: Int]]] = [:]
    private let lock = NSRecursiveLock()

    func recordTransition(fromLayer: Int, fromExperts: [Int], toLayer: Int, toExperts: [Int]) {
        guard !fromExperts.isEmpty && !toExperts.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        for src in fromExperts {
            for dst in toExperts {
                transitions[fromLayer, default: [:]][src, default: [:]][dst, default: 0] += 1
            }
        }
    }

    func predictNextExperts(currentLayer: Int, currentExperts: [Int], topN: Int) -> [Int] {
        guard !currentExperts.isEmpty else { return [] }
        lock.lock()
        defer { lock.unlock() }
        guard let layerMap = transitions[currentLayer] else { return [] }
        var scoreMap: [Int: Int] = [:]
        for exp in currentExperts {
            if let dstCounts = layerMap[exp] {
                for (dst, count) in dstCounts {
                    scoreMap[dst, default: 0] += count
                }
            }
        }
        return scoreMap.sorted(by: { $0.value > $1.value }).prefix(topN).map { $0.key }
    }

    func reset() {
        lock.lock()
        transitions.removeAll()
        lock.unlock()
    }
}

final class WorkingSetManager {
    static let shared = WorkingSetManager()

    private let lock = NSRecursiveLock()
    private var expertSlices: [ExpertKey: [ExpertSlice]] = [:]
    private var denseSlices: [ExpertSlice] = []
    private var denseBytes: UInt64 = 0
    private var residentExpertBytes: UInt64 = 0
    private var residentExperts: Set<ExpertKey> = []
    private var prefetchedKeys: Set<ExpertKey> = []
    private var accessOrder: [ExpertKey: UInt64] = [:]
    private var accessCounter: UInt64 = 0
    private let prefetchQueue = DispatchQueue(label: "com.dynamoe.prefetch", qos: .userInitiated, attributes: .concurrent)
    private let evictionQueue = DispatchQueue(label: "com.dynamoe.eviction", qos: .utility)
    private var prefetchedBackboneLayers: Set<UInt32> = []
    public let transitionTracker = ExpertTransitionTracker()

    private var totalAccesses: Int = 0
    private var cacheHits: Int = 0
    private var cacheMisses: Int = 0
    private var prefetchCount: Int = 0
    private var prefetchHits: Int = 0
    private var lastPagingLatencyMs: Double = 0.0
    private var expertOnlyShardIndices: Set<UInt32> = []
    private var shardMappings: [(base: UnsafeRawPointer, length: Int)] = []
    var residentMemoryProviderOverride: (() -> Double)?

    var totalExpertKeysCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return expertSlices.count
    }

    var residentExpertsCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return residentExperts.count
    }

    var cacheHitRatePercent: Double {
        lock.lock()
        defer { lock.unlock() }
        guard totalAccesses > 0 else { return 100.0 }
        return (Double(cacheHits) / Double(totalAccesses)) * 100.0
    }

    var prefetchEfficiencyPercent: Double {
        lock.lock()
        defer { lock.unlock() }
        guard prefetchCount > 0 else { return 100.0 }
        return (Double(prefetchHits) / Double(prefetchCount)) * 100.0
    }

    var currentPagingLatencyMs: Double {
        lock.lock()
        defer { lock.unlock() }
        return lastPagingLatencyMs
    }

    var effectiveResidentMemoryGB: Double {
        // mach resident_size (RSIZE) already includes every resident weight page
        // — file-backed mmap pages and anonymous pinned/staging buffers alike.
        // Adding the catalog's tracked dense/expert bytes on top double-counted
        // them, inflating the reported working set by up to the full tracked size.
        // residentMemoryProviderOverride is a test seam pinning that contract.
        //
        // This is the *eviction budget* input (trimToBudget compares it against
        // the mode's targetMaxRssGB). It stays on RSIZE deliberately: it counts
        // resident pages only, so under memory pressure it shrinks as the kernel
        // swaps the app's allocations out. The user-facing "how much is DynaMoE
        // using" figure is `attributedMemoryGB` below, not this.
        return residentMemoryProviderOverride?() ?? getProcessResidentMemoryGB()
    }

    /// Test seam for the Darwin footprint reading (see `attributedMemoryGB`).
    var footprintProviderOverride: (() -> Double)?

    /// Total memory attributable to this process: Darwin `phys_footprint` (dirty
    /// heap + compressed memory + IOKit/GPU, which already includes the FlashMoE
    /// pinned backbone) plus the clean file-backed weight pages the unified
    /// buffer cache holds for the model.
    ///
    /// The two components are disjoint by construction — `phys_footprint`
    /// excludes clean file-backed pages, and the shard registry excludes the
    /// pinned backbone (it is anonymous, and the footprint already counts it) —
    /// so summing them cannot double count. This is the honest answer to "how
    /// much is DynaMoE using"; RSIZE (`effectiveResidentMemoryGB`) is not, because
    /// it collapses whenever the kernel swaps the app's allocations out (measured
    /// live: 0.4 GB resident against a 20.6 GB footprint).
    ///
    /// The cached-weight share is reclaimable under pressure; the footprint is not.
    var attributedMemoryGB: Double {
        let footprint = footprintProviderOverride?() ?? getActivityMonitorFootprintGB()
        return footprint + residentShardBytesGB
    }

    // Unified Memory buffer-cache residency measured directly over the shard
    // mappings via mincore(): the clean file-backed weight pages that Activity
    // Monitor's footprint excludes. RSIZE - phys_footprint is NOT used — that is
    // only an accounting delta (it also contains clean framework/library
    // mappings and is offset by compressed/IOKit footprint charges).
    // The walk is O(#pages) — a 37 GB model walks ~2.4M entries — and tooltip
    // strings evaluate this getter eagerly during every SwiftUI body build
    // (currentRssGB ticks ~10x/s during prefill), so the walk runs on a bounded
    // cadence off the main thread and the view-facing getter is O(1).
    private var cachedResidentShardBytesGB: Double = 0.0
    private var shardWalkGeneration: UInt64 = 0
    private var scheduledWalkGeneration: UInt64 = 0
    private static let shardWalkInterval: TimeInterval = 2.0
    private let shardWalkQueue = DispatchQueue(label: "com.dynamoe.shardresidency", qos: .utility)

    var residentShardBytesGB: Double {
        lock.lock()
        defer { lock.unlock() }
        return cachedResidentShardBytesGB
    }

    /// Synchronously walks the registered shard mappings with mincore() and
    /// publishes the result into the cached, view-facing value. A utility-queue
    /// loop calls this at a bounded cadence; tests call it for determinism.
    /// The generation is snapshotted WITH the mappings and checked again before
    /// publishing, so a walk that was in flight across a registry replacement
    /// (`registerShardMappings`) or `releaseShardMappings` cannot republish the
    /// previous model's residency into the new model's cache.
    @discardableResult
    func refreshShardResidencyNow() -> Double {
        let generation: UInt64
        let mappings: [(base: UnsafeRawPointer, length: Int)]
        lock.lock()
        mappings = shardMappings
        generation = shardWalkGeneration
        lock.unlock()

        let pageSize = Int(vm_page_size) // 16384 bytes on Apple Silicon
        var residentBytes: UInt64 = 0
        for mapping in mappings {
            let alignedLen = mapping.length & ~(pageSize - 1)
            if alignedLen > 0 {
                var residencyMap = [UInt8](repeating: 0, count: alignedLen / pageSize)
                if mincore(mapping.base, alignedLen, &residencyMap) == 0 {
                    for flag in residencyMap where flag & 1 != 0 {
                        residentBytes &+= UInt64(pageSize)
                    }
                }
            }
            // An unaligned tail lives in the page spanning the mapping's end; probe
            // it on its own so partial pages are not omitted (mincore fails cleanly
            // when the page is no longer mapped, e.g. after the model unmaps).
            if mapping.length > alignedLen {
                var tailFlag: UInt8 = 0
                if mincore(mapping.base + alignedLen, pageSize, &tailFlag) == 0, tailFlag & 1 != 0 {
                    residentBytes &+= UInt64(pageSize)
                }
            }
        }
        let value = Double(residentBytes) / (1024.0 * 1024.0 * 1024.0)
        lock.lock()
        if generation == shardWalkGeneration {
            cachedResidentShardBytesGB = value
        }
        lock.unlock()
        return value
    }

    /// Registers the loaded model's shard mappings for the cached residency walk,
    /// invalidating any walk still running against a previous model's mappings.
    /// Independent of `initialize`'s paging-catalog setup so every load path can
    /// call it — FlashMoE loads never go through `initialize`.
    func registerShardMappings(_ shardBuffers: [UInt32: MTLBuffer]) {
        lock.lock()
        shardWalkGeneration &+= 1
        shardMappings = shardBuffers.values.map { (base: UnsafeRawPointer($0.contents()), length: $0.length) }
        cachedResidentShardBytesGB = 0.0
        lock.unlock()
        scheduleShardResidencyWalks()
    }

    private func scheduleShardResidencyWalks() {
        lock.lock()
        let generation = shardWalkGeneration
        if generation == scheduledWalkGeneration {
            lock.unlock()
            return // a loop for this registry generation is already scheduled
        }
        scheduledWalkGeneration = generation
        lock.unlock()
        shardWalkQueue.async { [weak self] in
            guard let self = self else { return }
            while true {
                self.lock.lock()
                let alive = (generation == self.shardWalkGeneration) && !self.shardMappings.isEmpty
                self.lock.unlock()
                guard alive else { return }
                self.refreshShardResidencyNow()
                Thread.sleep(forTimeInterval: WorkingSetManager.shardWalkInterval)
            }
        }
    }

    private static var pageFaultSink: Int64 = 0

    func releaseShardMappings() {
        lock.lock()
        defer { lock.unlock() }
        shardWalkGeneration &+= 1 // stops any in-flight refresh loop
        shardMappings.removeAll()
        cachedResidentShardBytesGB = 0.0
    }

    /// Faults expert/backbone slices into the Unified Memory Buffer Cache by
    /// touching one byte per page through the read-only shard mmap, in parallel
    /// across CPU cores (measured 13-20 GB/s warm, PERF_FINDINGS T8).
    func primeSlices(_ slices: [ExpertSlice], shardBuffers: [UInt32: MTLBuffer]) {
        guard !slices.isEmpty else { return }

        // Bring slices resident by faulting pages through the read-only shard mmap.
        // pread() into a PROT_READ mapping returns EFAULT on the first chunk, so the
        // old fd-based "fast path" silently primed nothing; the stride touch below
        // is the only mechanism that actually pages the slices in (PERF_FINDINGS T8).
        let pageSize = Int(vm_page_size) // 16384 bytes on Apple Silicon
        DispatchQueue.concurrentPerform(iterations: slices.count) { i in
            let slice = slices[i]
            guard let buf = shardBuffers[slice.shardIndex] else { return }
            let len = Int(slice.length)
            let offset = Int(slice.offset)
            guard len > 0, offset + len <= buf.length else { return }
            let rawPtr = buf.contents().advanced(by: offset)
            posix_madvise(rawPtr, len, POSIX_MADV_WILLNEED)
            let bytePtr = rawPtr.assumingMemoryBound(to: UInt8.self)
            var dummy: UInt64 = 0
            for off in stride(from: 0, to: len, by: pageSize) {
                dummy &+= UInt64(bytePtr[off])
            }
            dummy &+= UInt64(bytePtr[len - 1])
            OSAtomicAdd64(Int64(bitPattern: dummy), &WorkingSetManager.pageFaultSink)
        }
    }

    func initialize(summary: ModelSummary, shardBuffers: [UInt32: MTLBuffer], mode: MemoryBudgetMode) {
        lock.lock()
        expertSlices.removeAll()
        denseSlices.removeAll()
        denseBytes = 0
        residentExpertBytes = 0
        residentExperts.removeAll()
        prefetchedKeys.removeAll()
        prefetchedBackboneLayers.removeAll()
        accessOrder.removeAll()
        accessCounter = 0
        transitionTracker.reset()
        totalAccesses = 0
        cacheHits = 0
        cacheMisses = 0
        prefetchCount = 0
        prefetchHits = 0
        lastPagingLatencyMs = 0.0

        var shardHasDense: [UInt32: Bool] = [:]
        for tensor in summary.tensors {
            let name = tensor.name.lowercased()
            if name.contains("ple.") || name.contains("ngram_embedding") || name.contains("mtp.") || name.contains("visual.") {
                continue
            }
            let length = tensor.offsetEnd - tensor.offsetStart
            let slice = ExpertSlice(
                shardIndex: tensor.shardIndex,
                offset: tensor.offsetStart,
                length: length
            )

            if let l = tensor.layerIndex, let exp = tensor.expertId {
                let key = ExpertKey(layer: Int(l), expertId: Int(exp))
                expertSlices[key, default: []].append(slice)
            } else {
                denseSlices.append(slice)
                denseBytes += length
                shardHasDense[tensor.shardIndex] = true
            }
        }
        let allShardIndices = Set(summary.shards.map { $0.index })
        expertOnlyShardIndices = allShardIndices.filter { shardHasDense[$0] != true }
        lock.unlock()

        if mode == .unrestricted {
            preFaultAll(shardBuffers: shardBuffers, summary: summary)
        }
    }

    func preFaultAll(shardBuffers: [UInt32: MTLBuffer], summary: ModelSummary) {
        lock.lock()
        for key in expertSlices.keys {
            residentExperts.insert(key)
        }
        residentExpertBytes = expertSlices.reduce(0) { sum, pair in
            sum + pair.value.reduce(0) { $0 + $1.length }
        }
        lock.unlock()

        let shards = summary.shards
        let pageSize = Int(vm_page_size) // 16384 bytes on Apple Silicon

        // Parallel priming across CPU cores. Stride-touch only: pread() into the
        // read-only mmap fails with EFAULT and never primed anything.
        DispatchQueue.concurrentPerform(iterations: shards.count) { i in
            let shard = shards[i]
            let len = Int(shard.length)
            guard len > 0 else { return }

            // Stride touch through mmap MTLBuffer to map hardware page tables directly
            if let buf = shardBuffers[shard.index] {
                let rawPtr = buf.contents()
                posix_madvise(rawPtr, len, POSIX_MADV_WILLNEED)
                let bytePtr = rawPtr.assumingMemoryBound(to: UInt8.self)
                var dummy: UInt64 = 0
                for off in stride(from: 0, to: len, by: pageSize) {
                    dummy &+= UInt64(bytePtr[off])
                }
                dummy &+= UInt64(bytePtr[len - 1])
                OSAtomicAdd64(Int64(bitPattern: dummy), &WorkingSetManager.pageFaultSink)
            }
        }
    }

    func flushAllExperts(shardBuffers: [UInt32: MTLBuffer]) {
        var evictSlices: [ExpertSlice] = []
        lock.lock()
        for (_, slices) in expertSlices {
            for slice in slices {
                if expertOnlyShardIndices.contains(slice.shardIndex) {
                    evictSlices.append(slice)
                }
            }
        }
        residentExperts.removeAll()
        prefetchedKeys.removeAll()
        accessOrder.removeAll()
        accessCounter = 0
        residentExpertBytes = 0
        lock.unlock()

        if !evictSlices.isEmpty {
            prefetchQueue.async {
                for slice in evictSlices {
                    if let buf = shardBuffers[slice.shardIndex] {
                        let ptr = buf.contents().advanced(by: Int(slice.offset))
                        posix_madvise(ptr, Int(slice.length), POSIX_MADV_DONTNEED)
                    }
                }
            }
        }
    }

    func trimToBudget(mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer]) {
        if mode == .unrestricted { return }

        var evictSlices: [ExpertSlice] = []
        lock.lock()
        let maxAllowed = mode.maxResidentExperts
        var excess = residentExperts.count - maxAllowed
        let currentRss = getProcessResidentMemoryGB()
        if currentRss > mode.targetMaxRssGB {
            let overRssGB = currentRss - mode.targetMaxRssGB
            let extraExpertsToTrim = Int(ceil(overRssGB * 1024.0 / 4.9))
            excess = max(excess, extraExpertsToTrim)
        }

        if excess > 0 {
            let sortedOldest = accessOrder.sorted(by: { $0.value < $1.value }).prefix(excess)
            for item in sortedOldest {
                let evictKey = item.key
                residentExperts.remove(evictKey)
                prefetchedKeys.remove(evictKey)
                accessOrder.removeValue(forKey: evictKey)
                if let slices = expertSlices[evictKey] {
                    let expertSize = slices.reduce(0) { $0 + $1.length }
                    residentExpertBytes = residentExpertBytes >= expertSize ? (residentExpertBytes - expertSize) : 0
                    for slice in slices {
                        if expertOnlyShardIndices.contains(slice.shardIndex) {
                            evictSlices.append(slice)
                        }
                    }
                }
            }
        }
        lock.unlock()

        if !evictSlices.isEmpty {
            evictionQueue.async {
                for slice in evictSlices {
                    if let buf = shardBuffers[slice.shardIndex] {
                        let ptr = buf.contents().advanced(by: Int(slice.offset))
                        posix_madvise(ptr, Int(slice.length), POSIX_MADV_DONTNEED)
                    }
                }
            }
        }
    }

    func trimAfterPrefill(shardBuffers: [UInt32: MTLBuffer], mode: MemoryBudgetMode) {
        trimToBudget(mode: mode, shardBuffers: shardBuffers)
    }

    func prefetchLayerBackbone(layer: CachedLayer, shardBuffers: [UInt32: MTLBuffer]) {
        lock.lock()
        if prefetchedBackboneLayers.contains(layer.layerIndex) {
            lock.unlock()
            return
        }
        prefetchedBackboneLayers.insert(layer.layerIndex)
        lock.unlock()

        let tensors = layer.backboneTensors
        guard !tensors.isEmpty else { return }
        let slices = tensors.map { t in
            ExpertSlice(shardIndex: t.shardIndex, offset: t.offsetStart, length: t.offsetEnd - t.offsetStart)
        }
        prefetchQueue.async { [weak self] in
            guard let self = self else { return }
            self.primeSlices(slices, shardBuffers: shardBuffers)
        }
    }

    func prefetchLayerExperts(layer: Int, expertIds: [Int], shardBuffers: [UInt32: MTLBuffer]) {
        guard !expertIds.isEmpty else { return }
        var slicesToPrefetch: [ExpertSlice] = []

        lock.lock()
        for expId in expertIds {
            let key = ExpertKey(layer: layer, expertId: expId)
            if !residentExperts.contains(key) && !prefetchedKeys.contains(key) {
                prefetchedKeys.insert(key)
                prefetchCount += 1
                if let slices = expertSlices[key] {
                    slicesToPrefetch.append(contentsOf: slices)
                }
            }
        }
        lock.unlock()

        guard !slicesToPrefetch.isEmpty else { return }
        prefetchQueue.async { [weak self] in
            guard let self = self else { return }
            self.primeSlices(slicesToPrefetch, shardBuffers: shardBuffers)
        }
    }

    func predictNextLayerExperts(currentLayer: Int, currentActiveExperts: [Int], topN: Int = 4) -> [Int] {
        return transitionTracker.predictNextExperts(currentLayer: currentLayer, currentExperts: currentActiveExperts, topN: topN)
    }

    func touchAndEvict(layer: Int, activeExpertIds: [Int], mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer], isPrefill: Bool = false) {
        // In unrestricted full-RAM mode, bypass eviction overhead completely
        if mode == .unrestricted {
            return
        }

        var demandSlices: [ExpertSlice] = []

        lock.lock()
        for expId in activeExpertIds {
            let key = ExpertKey(layer: layer, expertId: expId)
            totalAccesses += 1
            accessCounter &+= 1
            accessOrder[key] = accessCounter

            if residentExperts.contains(key) {
                cacheHits += 1
            } else {
                cacheMisses += 1
                if prefetchedKeys.contains(key) {
                    prefetchHits += 1
                }
                residentExperts.insert(key)
                if let slices = expertSlices[key] {
                    let expertSize = slices.reduce(0) { $0 + $1.length }
                    residentExpertBytes += expertSize
                    demandSlices.append(contentsOf: slices)
                }
            }
        }
        lock.unlock()

        // primeSlices intentionally bypassed: ExpertIOThreadPool.dispatchSync in the
        // forward pass handles the same pread directly into staging buffers,
        // making touchAndEvict's duplicate read redundant.
        // Cache tracking and LRU state are still updated above.
    }

    func setBudgetMode(mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer], summary: ModelSummary) {
        if mode == .unrestricted {
            preFaultAll(shardBuffers: shardBuffers, summary: summary)
        } else {
            var evictSlices: [ExpertSlice] = []
            lock.lock()
            let maxAllowed = mode.maxResidentExperts
            let excess = residentExperts.count - maxAllowed
            if excess > 0 {
                let sortedOldest = accessOrder.sorted(by: { $0.value < $1.value }).prefix(excess)
                for item in sortedOldest {
                    let evictKey = item.key
                    residentExperts.remove(evictKey)
                    prefetchedKeys.remove(evictKey)
                    accessOrder.removeValue(forKey: evictKey)
                    if let slices = expertSlices[evictKey] {
                        let expertSize = slices.reduce(0) { $0 + $1.length }
                        residentExpertBytes = residentExpertBytes >= expertSize ? (residentExpertBytes - expertSize) : 0
                        for slice in slices {
                            if expertOnlyShardIndices.contains(slice.shardIndex) {
                                evictSlices.append(slice)
                            }
                        }
                    }
                }
            }
            lock.unlock()

            if !evictSlices.isEmpty {
                evictionQueue.async {
                    for slice in evictSlices {
                        if let buf = shardBuffers[slice.shardIndex] {
                            let ptr = buf.contents().advanced(by: Int(slice.offset))
                            posix_madvise(ptr, Int(slice.length), POSIX_MADV_DONTNEED)
                        }
                    }
                }
            }
        }
    }
}
