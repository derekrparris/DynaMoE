//
//  Inkling2ExpertArena.swift
//  DynaMoE
//
//  Phase 6, item 6: an expert ARENA for Inkling 2, plus the instrument item 1
//  was missing because of it.
//
//  This family's routed experts are FUSED banks read straight out of the shard
//  mmap: expert (layer, e) is a contiguous run of `w13` and of `w2`, and the
//  GPU faults those pages in as it reads them. That is what made the profile
//  unable to report misses, hits, evictions or pread time -- there was no
//  residency policy to report on. An arena makes each slot a resident buffer
//  filled by explicit pread, so the same bytes are counted instead of faulted.
//
//  What is deliberately NOT here: a policy that beats the mmap. The DSV4.1
//  journal's #58/#59 measured a frequency rule beating LRU on warm passes, and
//  #21 measured a short run where every miss was a first touch -- both readings
//  are about a working set that does or does not fit. This family's working set
//  at the published schedule is 48 visits x top-k experts of a 38 GB
//  checkpoint, far past any 16 GB machine's arena, so the policy question has
//  to be asked again HERE with the arena's own counters rather than copied.
//
//  The geometry is pure arithmetic and is pinned without a model; the arena's
//  residency bookkeeping is a deterministic sequence over a small fixture.
//

import Foundation

// MARK: - Geometry

/// Where one expert's bytes live inside a fused bank, and what a resident slot
/// must hold.
nonisolated public struct Inkling2ExpertSlotGeometry: Sendable, Equatable {
    /// Bytes of one expert's `w13` rows (interleaved gate/up).
    public let gateUpWeightBytes: Int
    /// Bytes of one expert's `w13` scale rows.
    public let gateUpScaleBytes: Int
    /// Bytes of one expert's `w2` rows.
    public let downWeightBytes: Int
    /// Bytes of one expert's `w2` scale rows.
    public let downScaleBytes: Int
    /// The packed row stride the kernel reads (weight and scale), which is the
    /// bank's own column count, not the logical width.
    public let weightRowStride: Int
    public let scaleRowStride: Int

    /// One slot holds all four ranges, in this order.
    public var gateUpWeightOffset: Int { 0 }
    public var gateUpScaleOffset: Int { gateUpWeightBytes }
    public var downWeightOffset: Int { gateUpScaleOffset + gateUpScaleBytes }
    public var downScaleOffset: Int { downWeightOffset + downWeightBytes }
    public var slotBytes: Int { downScaleOffset + downScaleBytes }

    /// The geometry of the published bank shapes.
    ///
    /// `w13` is `[experts, 2 * intermediate, packedHidden]` and `w2` is
    /// `[experts, hidden, packedIntermediate]`; the scale banks carry the same
    /// expert count with the block-scaled column counts. Every stride is taken
    /// from the stored shape, so a layout change moves this with it.
    public static func make(
        gateUpRows: Int,
        gateUpRowStride: Int,
        gateUpScaleRowStride: Int,
        downRows: Int,
        downRowStride: Int,
        downScaleRowStride: Int
    ) -> Inkling2ExpertSlotGeometry? {
        guard gateUpRows > 0, gateUpRowStride > 0, gateUpScaleRowStride > 0,
              downRows > 0, downRowStride > 0, downScaleRowStride > 0 else { return nil }
        return Inkling2ExpertSlotGeometry(
            gateUpWeightBytes: gateUpRows * gateUpRowStride,
            gateUpScaleBytes: gateUpRows * gateUpScaleRowStride,
            downWeightBytes: downRows * downRowStride,
            downScaleBytes: downRows * downScaleRowStride,
            weightRowStride: gateUpRowStride,
            scaleRowStride: gateUpScaleRowStride
        )
    }
}

// MARK: - The arena

/// What one slot is holding.
nonisolated public struct Inkling2ResidentExpert: Sendable, Equatable {
    public let layer: Int
    public let expert: Int
    /// Slot index in the arena's buffer.
    public let slot: Int
    /// The byte offset of this expert's first row inside its fused bank, so a
    /// hit can be checked against the source it claims to mirror.
    public let gateUpSourceOffset: UInt64
    public let downSourceOffset: UInt64
}

/// The arena's counters, which are item 1's missing fields.
nonisolated public struct Inkling2ArenaCounters: Sendable, Equatable {
    public var hits = 0
    public var misses = 0
    public var evictions = 0
    /// Bytes read from the shard (a miss only).
    public var bytesRead: UInt64 = 0
    /// Wall time spent in pread, in nanoseconds.
    public var preadNanoseconds: UInt64 = 0
    /// Slots allocated so far.
    public var slotsAllocated = 0

    public var hitRate: Double {
        let total = hits + misses
        return total == 0 ? 0 : Double(hits) / Double(total)
    }
}

/// A least-recently-used set of resident (layer, expert) pairs backed by
/// caller-provided slot buffers.
///
/// It owns the BOOKKEEPING only -- which pair occupies which slot, in what
/// order they were last used, and the counters -- and takes the fill as a
/// closure, so the same policy is exercised by a test with an in-memory fill and
/// by the executor with pread. That split is what makes the policy testable
/// without a checkpoint.
nonisolated public final class Inkling2ExpertArena: @unchecked Sendable {
    public let capacity: Int
    public let geometry: Inkling2ExpertSlotGeometry
    private var residents: [Inkling2ResidentExpert?]
    /// Most recently used first.
    private var order: [(layer: Int, expert: Int)] = []
    private var counters = Inkling2ArenaCounters()

    public init(capacity: Int, geometry: Inkling2ExpertSlotGeometry) {
        self.capacity = max(1, capacity)
        self.geometry = geometry
        self.residents = Array(repeating: nil, count: max(1, capacity))
    }

    public var statistics: Inkling2ArenaCounters {
        var snapshot = counters
        snapshot.slotsAllocated = residents.compactMap { $0 }.count
        return snapshot
    }

    /// The slot holding (layer, expert), if any, marking it most recently used.
    public func resident(layer: Int, expert: Int) -> Int? {
        guard let index = order.firstIndex(where: { $0.layer == layer && $0.expert == expert })
        else { return nil }
        let entry = order.remove(at: index)
        order.insert(entry, at: 0)
        counters.hits += 1
        return residents.firstIndex { $0?.layer == layer && $0?.expert == expert }
    }

    /// The slot to fill for (layer, expert): an empty one, or the least recently
    /// used one (counted as an eviction). The caller fills it and then calls
    /// `didFill`.
    public func slotForFill(layer: Int, expert: Int) -> Int {
        counters.misses += 1
        if let empty = residents.firstIndex(where: { $0 == nil }) { return empty }
        counters.evictions += 1
        let victim = order.removeLast()
        return residents.firstIndex { $0?.layer == victim.layer && $0?.expert == victim.expert } ?? 0
    }

    /// Records a completed fill (and what it cost).
    public func didFill(
        slot: Int, layer: Int, expert: Int,
        gateUpSourceOffset: UInt64, downSourceOffset: UInt64,
        bytes: UInt64, preadNanoseconds: UInt64
    ) {
        residents[slot] = Inkling2ResidentExpert(
            layer: layer, expert: expert, slot: slot,
            gateUpSourceOffset: gateUpSourceOffset, downSourceOffset: downSourceOffset
        )
        order.removeAll { $0.layer == layer && $0.expert == expert }
        order.insert((layer, expert), at: 0)
        counters.bytesRead += bytes
        counters.preadNanoseconds += preadNanoseconds
    }

    public func evictAll() {
        residents = Array(repeating: nil, count: capacity)
        order.removeAll()
    }

    /// One line for the load log / the profile.
    public func describe() -> String {
        let stats = statistics
        let gigabytes = Double(stats.bytesRead) / 1_073_741_824
        return "expert arena \(capacity) slot(s) of \(geometry.slotBytes) B: "
            + "\(stats.hits) hit(s), \(stats.misses) miss(es), \(stats.evictions) eviction(s), "
            + String(format: "%.2f GB read in %.1f ms (%.1f GB/s), hit rate %.1f%%",
                     gigabytes, Double(stats.preadNanoseconds) / 1_000_000,
                     stats.preadNanoseconds > 0
                        ? gigabytes / (Double(stats.preadNanoseconds) / 1_000_000_000) : 0,
                     100 * stats.hitRate)
    }
}
