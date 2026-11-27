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
import Metal

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
    /// The down bank's own strides. They are NOT the gate/up ones: `w13` rows are
    /// `packedHidden` wide and `w2` rows are `packedIntermediate`, so one shared
    /// stride would read the down projection's scales off by the width ratio.
    public let downWeightRowStride: Int
    public let downScaleRowStride: Int

    /// One slot holds all four ranges, in this order.
    public var gateUpWeightOffset: Int { 0 }
    public var gateUpScaleOffset: Int { gateUpWeightBytes }
    public var downWeightOffset: Int { gateUpScaleOffset + gateUpScaleBytes }
    public var downScaleOffset: Int { downWeightOffset + downWeightBytes }
    public var slotBytes: Int { downScaleOffset + downScaleBytes }

    /// How the mixture kernel addresses a RESIDENT slot.
    ///
    /// A slot is one packed buffer, not four separate tensors, so every region
    /// is a byte offset from the slot's base and every row is a stride inside
    /// it -- there is no per-tensor bind offset to carry. This is the whole
    /// reason the mixture could not read a resident slot: its kernel takes a
    /// weight pointer and a scale pointer with independent strides, which a
    /// packed slot does not have.
    public struct SlotAddress: Sendable, Equatable {
        /// Byte offset of the expert's first `w13` weight row from the slot base.
        public let gateUpWeight: Int
        public let gateUpScale: Int
        public let downWeight: Int
        public let downScale: Int
        /// Rows of each region, so a caller projects exactly the range the
        /// geometry priced.
        public let gateUpRows: Int
        public let downRows: Int
        public let gateUpWeightRowStride: Int
        public let gateUpScaleRowStride: Int
        public let downWeightRowStride: Int
        public let downScaleRowStride: Int

        /// What one projection needs from the packed slot: where its weight and
        /// scale rows start, and the stride of each. They are per region on
        /// purpose -- `w13` and `w2` do not share a row width.
        public func projection(isGateUp: Bool) -> (weightOffset: Int, scaleOffset: Int, weightRowStride: Int, scaleRowStride: Int) {
            isGateUp
                ? (gateUpWeight, gateUpScale, gateUpWeightRowStride, gateUpScaleRowStride)
                : (downWeight, downScale, downWeightRowStride, downScaleRowStride)
        }
    }

    /// The published shape of the gate/up bank's rows: two rows per intermediate
    /// index, because the bank interleaves gate and up.
    public var gateUpLogicalRows: Int { gateUpWeightBytes / weightRowStride }
    public var downLogicalRows: Int { downWeightBytes / downWeightRowStride }

    /// Whether slots laid out this way can hold a bank read at these strides.
    ///
    /// A slot is a byte copy of four regions at FIXED strides, so an expert
    /// whose bank packs differently cannot be copied into it: the ranges come
    /// out the wrong length and no read reports an error. The published
    /// checkpoint is exactly this case -- NVFP4 for the first twenty layers
    /// (`w13` rows 1,152 B with 144 scale columns) and MXFP8 for the last four
    /// (2,304 B with 72) -- so the discriminating numbers are the strides; the
    /// ROW counts are the config's and agree across layouts.
    public func holds(
        weightRowStride: Int, scaleRowStride: Int,
        downWeightRowStride: Int, downScaleRowStride: Int
    ) -> Bool {
        weightRowStride == self.weightRowStride
            && scaleRowStride == self.scaleRowStride
            && downWeightRowStride == self.downWeightRowStride
            && downScaleRowStride == self.downScaleRowStride
    }

    /// The slot as the mixture kernel reads it.
    public var slotAddress: SlotAddress {
        SlotAddress(
            gateUpWeight: gateUpWeightOffset, gateUpScale: gateUpScaleOffset,
            downWeight: downWeightOffset, downScale: downScaleOffset,
            gateUpRows: gateUpLogicalRows, downRows: downLogicalRows,
            gateUpWeightRowStride: weightRowStride, gateUpScaleRowStride: scaleRowStride,
            downWeightRowStride: downWeightRowStride, downScaleRowStride: downScaleRowStride
        )
    }

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
            scaleRowStride: gateUpScaleRowStride,
            downWeightRowStride: downRowStride,
            downScaleRowStride: downScaleRowStride
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
    /// Experts this arena could not hold at all: their bank's row strides are
    /// not the ones its slots were laid out with.
    ///
    /// A checkpoint can mix quantized layouts between layers (the published one
    /// is NVFP4 for the first twenty layers and MXFP8 for the last four, whose
    /// packed rows are twice as wide with half the scale rows), and a slot is a
    /// byte copy at FIXED strides. Counting the refusals keeps that visible:
    /// without it a mixed checkpoint fills slots with the wrong byte ranges and
    /// reports a hit rate over garbage.
    public var incompatible = 0

    public var hitRate: Double {
        let total = hits + misses
        return total == 0 ? 0 : Double(hits) / Double(total)
    }

    /// One set of counters over several arenas, which is how a checkpoint that
    /// mixes two packed layouts reaches the profile as ONE line: the NVFP4 slots
    /// and the MXFP8 ones are separate caches, but a hit rate per layout would
    /// be two numbers nobody wants to add up by hand mid-turn.
    public static func merged(_ counters: [Inkling2ArenaCounters]) -> Inkling2ArenaCounters {
        counters.reduce(into: Inkling2ArenaCounters()) { total, next in
            total.hits += next.hits
            total.misses += next.misses
            total.evictions += next.evictions
            total.bytesRead += next.bytesRead
            total.preadNanoseconds += next.preadNanoseconds
            total.slotsAllocated += next.slotsAllocated
            total.incompatible += next.incompatible
        }
    }
}

// MARK: - Reading a shard

/// Errors an arena read can fail with, named rather than left as an errno.
nonisolated public enum Inkling2ArenaError: Error, Equatable, CustomStringConvertible {
    case cannotOpen(String)
    case shortRead(path: String, offset: Int, wanted: Int, got: Int)
    case rangeOutsideTheShard(path: String, offset: Int, length: Int, fileSize: Int)

    public var description: String {
        switch self {
        case .cannotOpen(let detail):
            return "cannot open an expert shard: \(detail)"
        case let .shortRead(path, offset, wanted, got):
            return "expert read (\(path)) at \(offset) returned \(got) of \(wanted) bytes"
        case let .rangeOutsideTheShard(path, offset, length, fileSize):
            return "expert read (\(path)) at \(offset) for \(length) bytes runs past the \(fileSize)-byte shard"
        }
    }
}

/// POSIX `pread` into caller-chosen destinations, fanned out across a concurrent
/// queue when there is more than one range.
///
/// The same shape the DeepSeek V4.1 port ships, for the same reason: one read per
/// range is a round trip each, and a fill that misses one expert asks for a
/// handful of ranges, so the queue DEPTH rather than the range size is what turns
/// a round trip into throughput. Every range writes a disjoint destination, so
/// the fan-out needs no coordination. Deliberately no `F_NOCACHE`: the DSV4.1
/// journal measured that as a 2.2x LOSS through Metal's shared memory.
nonisolated public final class Inkling2ArenaShardReader: @unchecked Sendable {
    public struct Range {
        public let fileOffset: Int
        public let destination: UnsafeMutableRawPointer
        public let length: Int
        public init(fileOffset: Int, destination: UnsafeMutableRawPointer, length: Int) {
            self.fileOffset = fileOffset
            self.destination = destination
            self.length = length
        }
    }

    public let path: String
    public let fileSize: Int
    private let descriptor: Int32
    private let lock = NSLock()
    private var isClosed = false

    public init(url: URL) throws {
        let fd = open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw Inkling2ArenaError.cannotOpen("\(url.lastPathComponent): errno \(errno)")
        }
        var status = stat()
        guard fstat(fd, &status) == 0 else {
            let code = errno
            close(fd)
            throw Inkling2ArenaError.cannotOpen("\(url.lastPathComponent): fstat failed, errno \(code)")
        }
        self.descriptor = fd
        self.path = url.lastPathComponent
        self.fileSize = Int(status.st_size)
    }

    deinit { closeNow() }

    public func closeNow() {
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else { return }
        isClosed = true
        close(descriptor)
    }

    /// Reads every range. Returns the bytes read, and throws the first failure
    /// with the range that caused it.
    @discardableResult
    public func read(_ ranges: [Range]) throws -> Int {
        guard !ranges.isEmpty else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else { throw Inkling2ArenaError.cannotOpen("\(path) was closed") }
        // Checked before any read: a short read in the middle of a matrix is a
        // silently wrong weight row.
        for range in ranges {
            guard range.fileOffset >= 0, range.length >= 0,
                  range.fileOffset + range.length <= fileSize else {
                throw Inkling2ArenaError.rangeOutsideTheShard(
                    path: path, offset: range.fileOffset,
                    length: range.length, fileSize: fileSize
                )
            }
        }
        let fd = descriptor
        let path = self.path
        let failure = Inkling2ReadFailure()
        let readOne: @Sendable (Int) -> Void = { index in
            let range = ranges[index]
            var total = 0
            while total < range.length {
                let got = pread(
                    fd, range.destination.advanced(by: total),
                    range.length - total, off_t(range.fileOffset + total)
                )
                if got < 0 {
                    if errno == EINTR { continue }
                    failure.record(.shortRead(
                        path: path, offset: range.fileOffset + total,
                        wanted: range.length - total, got: Int(got)
                    ))
                    return
                }
                if got == 0 {
                    failure.record(.shortRead(
                        path: path, offset: range.fileOffset + total,
                        wanted: range.length - total, got: 0
                    ))
                    return
                }
                total += got
            }
        }
        if ranges.count >= 2 {
            DispatchQueue.concurrentPerform(iterations: ranges.count, execute: readOne)
        } else {
            for index in ranges.indices { readOne(index) }
        }
        if let first = failure.first { throw first }
        return ranges.reduce(0) { $0 + $1.length }
    }
}

/// The first failure of a fan-out, shared across its reads.
nonisolated private final class Inkling2ReadFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Inkling2ArenaError?
    func record(_ error: Inkling2ArenaError) {
        lock.lock()
        if stored == nil { stored = error }
        lock.unlock()
    }
    var first: Inkling2ArenaError? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

/// Which expert-row kernel the mixture dispatches, and how wide.
///
/// One warp per output row, a BLOCK of sixteen values per lane iteration. Slice
/// 35 priced it over the published slot shape against the row kernel it
/// replaced, on the same eight-slot mixture the forward pass encodes: 1.00 ms
/// against 3.30 ms (3.3x, 35.8 GB/s against 10.8), with every decode identical
/// and the output 2.1e-7 from the row kernel's. The width matters less than the
/// shape -- 64 lanes measured 1.07 ms and 128 measured 1.16 -- but a warp is
/// the reduction's own unit, so it needs no threadgroup memory at all.
nonisolated public enum Inkling2ExpertRowDispatch {
    public static let kernelName = "inkling_expert_row_block_f32"
    public static let threadgroup = 32
}

/// One expert projection straight out of a RESIDENT slot: the arena's packed
/// buffer, with each region addressed from the slot's own base.
///
/// This is the same dispatch the mmap path makes, with two differences and no
/// third: the weight and scale are the SAME buffer (a slot is packed), and each
/// region's offset is the slot's base plus the geometry's region offset rather
/// than a fused bank's expert offset. The row strides are per region -- `w13`
/// rows are `packedHidden` wide, `w2` rows `packedIntermediate` -- so a caller
/// passing the gate/up stride for both would read the down projection's scales
/// off by the width ratio, which is a plausible number rather than a crash.
///
/// It lives here rather than beside the other dispatch helpers because the
/// packed-slot layout is the arena's own, and this way the binding can be run
/// against the host rule with no checkpoint at all.
nonisolated public enum Inkling2SlotProjection {
    public static func encode(
        pipeline: MTLComputePipelineState,
        encoder: MTLComputeCommandEncoder,
        slot: MTLBuffer, slotBaseOffset: Int,
        address: Inkling2ExpertSlotGeometry.SlotAddress,
        isGateUp: Bool,
        input: MTLBuffer, output: MTLBuffer,
        rows: Int, logicalColumns: Int,
        layout: Inkling2ExpertQuantLayout, outerScale: Float,
        threadgroup: Int = Inkling2ExpertRowDispatch.threadgroup,
        arena: Inkling2ScalarArena
    ) {
        let region = address.projection(isGateUp: isGateUp)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(slot, offset: 0, index: 0)
        encoder.setBuffer(slot, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(arena.make(UInt32(logicalColumns)), offset: 0, index: 4)
        encoder.setBuffer(arena.make(UInt32(region.weightRowStride)), offset: 0, index: 5)
        encoder.setBuffer(arena.make(UInt32(region.scaleRowStride)), offset: 0, index: 6)
        encoder.setBuffer(arena.make(UInt32(layout.blockSize)), offset: 0, index: 7)
        encoder.setBuffer(arena.make(layout == .nvfp4), offset: 0, index: 8)
        encoder.setBuffer(arena.make(outerScale), offset: 0, index: 9)
        encoder.setBuffer(
            arena.make(UInt64(slotBaseOffset + region.weightOffset)), offset: 0, index: 10
        )
        encoder.setBuffer(
            arena.make(UInt64(slotBaseOffset + region.scaleOffset)), offset: 0, index: 11
        )
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroup, height: 1, depth: 1)
        )
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

    /// Takes a slot for (layer, expert) and records the residency NOW, before its
    /// bytes are read.
    ///
    /// This is what a batch needs and what a single fill does not: with nothing
    /// written into `residents` between two reservations, every miss of one batch
    /// would pick the same first empty slot. Recording the residency at
    /// reservation time also makes a pair asked for twice in one batch behave as
    /// the serial path would -- the second ask is a hit on the first one's slot
    /// -- and keeps the eviction order identical, because the recency insert
    /// happens in request order either way.
    ///
    /// The source offsets are the caller's to record with `recordFill` once the
    /// bytes are in; until then they are zero, which is distinguishable from a
    /// real offset only in that it can never match a bank.
    private func reserve(layer: Int, expert: Int) -> Int {
        counters.misses += 1
        let slot: Int
        if let empty = residents.firstIndex(where: { $0 == nil }) {
            slot = empty
        } else {
            counters.evictions += 1
            let victim = order.removeLast()
            slot = residents.firstIndex {
                $0?.layer == victim.layer && $0?.expert == victim.expert
            } ?? 0
        }
        residents[slot] = Inkling2ResidentExpert(
            layer: layer, expert: expert, slot: slot,
            gateUpSourceOffset: 0, downSourceOffset: 0
        )
        order.insert((layer, expert), at: 0)
        return slot
    }

    /// Records a reserved slot's true source offsets and what its read cost.
    private func recordFill(
        slot: Int, gateUpSourceOffset: UInt64, downSourceOffset: UInt64,
        bytes: UInt64, preadNanoseconds: UInt64
    ) {
        if let entry = residents[slot] {
            residents[slot] = Inkling2ResidentExpert(
                layer: entry.layer, expert: entry.expert, slot: slot,
                gateUpSourceOffset: gateUpSourceOffset, downSourceOffset: downSourceOffset
            )
        }
        counters.bytesRead += bytes
        counters.preadNanoseconds += preadNanoseconds
    }

    /// The slot buffers, when the arena was given memory to fill. One buffer
    /// holds every slot, `geometry.slotBytes` apart.
    public private(set) var slotBuffer: MTLBuffer?

    /// The slot holding (layer, expert) as the mixture kernel binds it: the
    /// arena's buffer plus the byte offset of that slot's four regions. Nil on a
    /// miss or before a slot buffer was attached -- a caller must fill first, so
    /// a kernel is never handed a slot that holds another expert's bytes.
    public func residentSlot(layer: Int, expert: Int) -> (buffer: MTLBuffer, address: Inkling2ExpertSlotGeometry.SlotAddress, slot: Int)? {
        guard let buffer = slotBuffer,
              let slot = residents.firstIndex(where: { $0?.layer == layer && $0?.expert == expert })
        else { return nil }
        return (buffer, geometry.slotAddress, slot)
    }

    /// Gives the arena a device buffer of `capacity * geometry.slotBytes`.
    public func attachSlotBuffer(_ buffer: MTLBuffer) {
        slotBuffer = buffer
    }

    /// The slot holding (layer, expert), filled from `reader` on a miss.
    ///
    /// A hit costs the bookkeeping above. A miss reads the expert's four ranges
    /// -- `w13` weight and scale, `w2` weight and scale -- out of the fused banks
    /// at this expert's offset into its slot, timed, and records what it cost.
    /// Returns the slot index; the caller binds `slotBuffer` at
    /// `slot * geometry.slotBytes`.
    @discardableResult
    public func slot(
        layer: Int, expert: Int, reader: Inkling2ArenaShardReader,
        gateUpSourceOffset: UInt64, gateUpScaleSourceOffset: UInt64,
        downSourceOffset: UInt64, downScaleSourceOffset: UInt64
    ) throws -> Int {
        if let hit = resident(layer: layer, expert: expert) { return hit }
        let slot = slotForFill(layer: layer, expert: expert)
        guard let buffer = slotBuffer else {
            // Bookkeeping only: the caller fills the slot itself.
            didFill(
                slot: slot, layer: layer, expert: expert,
                gateUpSourceOffset: gateUpSourceOffset, downSourceOffset: downSourceOffset,
                bytes: 0, preadNanoseconds: 0
            )
            return slot
        }
        let base = buffer.contents().advanced(by: slot * geometry.slotBytes)
        let ranges = [
            Inkling2ArenaShardReader.Range(
                fileOffset: Int(gateUpSourceOffset),
                destination: base.advanced(by: geometry.gateUpWeightOffset),
                length: geometry.gateUpWeightBytes
            ),
            Inkling2ArenaShardReader.Range(
                fileOffset: Int(gateUpScaleSourceOffset),
                destination: base.advanced(by: geometry.gateUpScaleOffset),
                length: geometry.gateUpScaleBytes
            ),
            Inkling2ArenaShardReader.Range(
                fileOffset: Int(downSourceOffset),
                destination: base.advanced(by: geometry.downWeightOffset),
                length: geometry.downWeightBytes
            ),
            Inkling2ArenaShardReader.Range(
                fileOffset: Int(downScaleSourceOffset),
                destination: base.advanced(by: geometry.downScaleOffset),
                length: geometry.downScaleBytes
            )
        ]
        let started = DispatchTime.now().uptimeNanoseconds
        let read = try reader.read(ranges)
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        didFill(
            slot: slot, layer: layer, expert: expert,
            gateUpSourceOffset: gateUpSourceOffset, downSourceOffset: downSourceOffset,
            bytes: UInt64(read), preadNanoseconds: elapsed
        )
        return slot
    }

    /// One expert whose slot this arena is asked to hold, with the four source
    /// offsets its banks put it at.
    public struct FillRequest: Sendable, Equatable {
        public let layer: Int
        public let expert: Int
        public let gateUpSourceOffset: UInt64
        public let gateUpScaleSourceOffset: UInt64
        public let downSourceOffset: UInt64
        public let downScaleSourceOffset: UInt64

        public init(
            layer: Int, expert: Int,
            gateUpSourceOffset: UInt64, gateUpScaleSourceOffset: UInt64,
            downSourceOffset: UInt64, downScaleSourceOffset: UInt64
        ) {
            self.layer = layer
            self.expert = expert
            self.gateUpSourceOffset = gateUpSourceOffset
            self.gateUpScaleSourceOffset = gateUpScaleSourceOffset
            self.downSourceOffset = downSourceOffset
            self.downScaleSourceOffset = downScaleSourceOffset
        }
    }

    /// Fills every miss among `requests` in ONE fan-out, and returns the slots
    /// in request order.
    ///
    /// The mixture knows all of a visit's routed experts before it reads any of
    /// them, so filling them one call at a time asks this SSD for four
    /// concurrent reads when thirty-two are available. Slice 34 measured the
    /// difference at 1.6x on this machine (5.2 GB/s at the shipped depth of 4,
    /// 8.2 GB/s at 32), which is why the mixture calls this rather than `slot`
    /// in a loop.
    ///
    /// The residency bookkeeping is the serial path's, step for step: the same
    /// requests in the same order mark hits, reserve slots and evict the same
    /// victims, so the batch is a shape change and not a policy change. Time in
    /// `pread` is split evenly across the misses, because every miss reads the
    /// same `slotBytes`.
    @discardableResult
    public func fill(
        _ requests: [FillRequest], reader: Inkling2ArenaShardReader
    ) throws -> [Int] {
        guard !requests.isEmpty else { return [] }
        var slots: [Int] = []
        slots.reserveCapacity(requests.count)
        var misses: [(request: FillRequest, slot: Int)] = []
        for request in requests {
            if let hit = resident(layer: request.layer, expert: request.expert) {
                slots.append(hit)
            } else {
                let slot = reserve(layer: request.layer, expert: request.expert)
                slots.append(slot)
                misses.append((request, slot))
            }
        }
        guard !misses.isEmpty else { return slots }
        guard let buffer = slotBuffer else {
            // Bookkeeping only: the caller fills the slots itself, so there is no
            // read to race and every reservation is recorded.
            for miss in misses {
                recordFill(
                    slot: miss.slot,
                    gateUpSourceOffset: miss.request.gateUpSourceOffset,
                    downSourceOffset: miss.request.downSourceOffset,
                    bytes: 0, preadNanoseconds: 0
                )
            }
            return slots
        }
        // A batch larger than the arena -- which the mixture cannot produce,
        // because it asks for one visit's routing and the arena's floor is that
        // width -- can reserve the same slot twice. Keep the later request, which
        // is the state the serial sequence ends in, and skip the bytes it evicted:
        // reading both into one slot would be a race whose winner is the order the
        // fan-out happened to run in.
        var lastReservation: [Int: Int] = [:]
        for (index, miss) in misses.enumerated() { lastReservation[miss.slot] = index }
        misses = misses.enumerated()
            .filter { lastReservation[$0.element.slot] == $0.offset }
            .map(\.element)
        var ranges: [Inkling2ArenaShardReader.Range] = []
        ranges.reserveCapacity(misses.count * 4)
        for miss in misses {
            let base = buffer.contents().advanced(by: miss.slot * geometry.slotBytes)
            let request = miss.request
            ranges.append(Inkling2ArenaShardReader.Range(
                fileOffset: Int(request.gateUpSourceOffset),
                destination: base.advanced(by: geometry.gateUpWeightOffset),
                length: geometry.gateUpWeightBytes
            ))
            ranges.append(Inkling2ArenaShardReader.Range(
                fileOffset: Int(request.gateUpScaleSourceOffset),
                destination: base.advanced(by: geometry.gateUpScaleOffset),
                length: geometry.gateUpScaleBytes
            ))
            ranges.append(Inkling2ArenaShardReader.Range(
                fileOffset: Int(request.downSourceOffset),
                destination: base.advanced(by: geometry.downWeightOffset),
                length: geometry.downWeightBytes
            ))
            ranges.append(Inkling2ArenaShardReader.Range(
                fileOffset: Int(request.downScaleSourceOffset),
                destination: base.advanced(by: geometry.downScaleOffset),
                length: geometry.downScaleBytes
            ))
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let read: Int
        do {
            read = try reader.read(ranges)
        } catch {
            // A reservation whose bytes never arrived must not read as a
            // resident expert: the next call would hit a slot holding another
            // expert's weights, which is a plausible number rather than a crash.
            // The counters keep the misses; only the residency is dropped.
            evictAll()
            throw error
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        let share = elapsed / UInt64(misses.count)
        let remainder = elapsed - share * UInt64(misses.count)
        for (index, miss) in misses.enumerated() {
            recordFill(
                slot: miss.slot,
                gateUpSourceOffset: miss.request.gateUpSourceOffset,
                downSourceOffset: miss.request.downSourceOffset,
                bytes: UInt64(read) / UInt64(misses.count),
                preadNanoseconds: index == 0 ? share + remainder : share
            )
        }
        return slots
    }

    public func evictAll() {
        residents = Array(repeating: nil, count: capacity)
        order.removeAll()
    }

    /// Records an expert whose bank this arena cannot hold (see
    /// `Inkling2ArenaCounters.incompatible`). The caller falls back to the mmap.
    public func recordIncompatible() {
        counters.incompatible += 1
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
            + (stats.incompatible > 0 ? ", \(stats.incompatible) refused (a bank this slot cannot hold)" : "")
    }
}
