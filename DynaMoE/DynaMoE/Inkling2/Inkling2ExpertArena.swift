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
    /// Misses the arena READ but did not keep: the admission rule found the
    /// candidate less requested than the resident it would displace, so its
    /// bytes went through a scratch slot instead of taking a residency. Slice
    /// 36 measured this at 25% fewer fill bytes on the real routing.
    public var bypassed = 0
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
            total.bypassed += next.bypassed
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
///
/// `chunkBytes` is that depth lever at the level BELOW the expert: a visit only
/// misses about two and a half experts, which is ten ranges, so splitting a wide
/// range into pieces asks the SSD for more reads in flight without moving a
/// different byte. Slice 40 PRICED that and shipped it OFF: on a temporary file
/// the pieces bought 0 to 1.4 GB/s at two and three experts and nothing at eight
/// (7.4-10.6 GB/s against 7.4-9.2 with one read per range, every arm inside the
/// run-to-run noise), and on the REAL shard the whole shape measures 4.71-6.34
/// GB/s cold against 5.49-5.69 for one contiguous read of the same bytes -- so
/// the scatter is nearly free and the depth is not what limits the fill. The
/// drive is. `shippedChunkBytes` is therefore zero and the seam stays as the way
/// a future machine's answer would be taken.
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

    /// The largest contiguous `pread` the shipped reader issues: zero, one read
    /// per range. Slice 40 priced splitting a range into pieces and measured no
    /// gain, so the shipped shape is the unsplit one; see the class comment.
    nonisolated public static let shippedChunkBytes = 0

    public let path: String
    public let fileSize: Int
    /// Zero reads one range per `pread`; anything else caps a piece at that many
    /// bytes. A piece is never wider than the range it came from.
    public let chunkBytes: Int
    private let descriptor: Int32
    private let lock = NSLock()
    private var isClosed = false

    public init(url: URL, chunkBytes: Int = Inkling2ArenaShardReader.shippedChunkBytes) throws {
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
        self.chunkBytes = chunkBytes
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
        // A range wider than `chunkBytes` becomes several reads at the SAME
        // destination offset the range would have written, so the bytes are
        // identical to the unsplit read by construction.
        var pieces: [Inkling2ReadPiece] = []
        pieces.reserveCapacity(ranges.count * 2)
        for range in ranges {
            guard chunkBytes > 0, range.length > chunkBytes else {
                pieces.append(Inkling2ReadPiece(
                    fileOffset: range.fileOffset,
                    destination: range.destination,
                    length: range.length
                ))
                continue
            }
            var done = 0
            while done < range.length {
                let length = min(chunkBytes, range.length - done)
                pieces.append(Inkling2ReadPiece(
                    fileOffset: range.fileOffset + done,
                    destination: range.destination.advanced(by: done),
                    length: length
                ))
                done += length
            }
        }
        let failure = Inkling2ReadFailure()
        let readOne: @Sendable (Int) -> Void = { index in
            let piece = pieces[index]
            var total = 0
            while total < piece.length {
                let got = pread(
                    fd, piece.destination.advanced(by: total),
                    piece.length - total, off_t(piece.fileOffset + total)
                )
                if got < 0 {
                    if errno == EINTR { continue }
                    failure.record(.shortRead(
                        path: path, offset: piece.fileOffset + total,
                        wanted: piece.length - total, got: Int(got)
                    ))
                    return
                }
                if got == 0 {
                    failure.record(.shortRead(
                        path: path, offset: piece.fileOffset + total,
                        wanted: piece.length - total, got: 0
                    ))
                    return
                }
                total += got
            }
        }
        if pieces.count >= 2 {
            DispatchQueue.concurrentPerform(iterations: pieces.count, execute: readOne)
        } else if !pieces.isEmpty {
            readOne(0)
        }
        if let first = failure.first { throw first }
        return ranges.reduce(0) { $0 + $1.length }
    }
}

/// One contiguous `pread` the reader will issue, after chunking.
///
/// `@unchecked` because the destination is a raw pointer into a caller-owned
/// buffer: every piece writes a disjoint span, which is the property the fan-out
/// needs and the caller's reason for passing raw memory at all.
nonisolated private struct Inkling2ReadPiece: @unchecked Sendable {
    let fileOffset: Int
    let destination: UnsafeMutableRawPointer
    let length: Int

    init(fileOffset: Int, destination: UnsafeMutableRawPointer, length: Int) {
        self.fileOffset = fileOffset
        self.destination = destination
        self.length = length
    }
}

/// The series of `(arena, layer, expert)` requests a walk made, in order.
///
/// Slice 36 prices the arena's eviction policy on the REAL routing sequence
/// rather than on a synthetic one: a walk records what the mixture asked for,
/// and a replay feeds the same requests through the shipped arena, so an
/// alternative policy is compared on the workload the model actually generates
/// -- and the shipped policy's replay has to reproduce the walk's own hit,
/// miss and eviction counts, which is what makes the comparison trustworthy.
///
/// A request with `arena == -1` never reached an arena (no installed arena held
/// its bank), so it is not a caching event and the replay only counts it.
nonisolated public final class Inkling2RoutingTrace: @unchecked Sendable {
    public struct Request: Sendable, Equatable {
        public let arena: Int
        public let layer: Int
        public let expert: Int
    }

    /// What a replay needs to check itself: each installed arena's capacity and
    /// slot size, and the counters its arena ended the walk with.
    public struct Header: Sendable, Equatable {
        public var capacities: [Int] = []
        public var slotBytes: [Int] = []
        public var counters: [Inkling2ArenaCounters] = []
        /// The policy each arena ran, and its scratch slots, because a replay
        /// has to reproduce the arm that produced the counters. Comparing a
        /// replay under one rule with a walk under another is the one mistake
        /// this file cannot afford: it reads exactly like a policy result.
        public var admission: [Inkling2ArenaAdmission] = []
        public var scratchSlots: [Int] = []
    }

    public private(set) var requests: [Request] = []
    public init() {}

    public func record(arena: Int, layer: Int, experts: [Int]) {
        for expert in experts {
            requests.append(Request(arena: arena, layer: layer, expert: expert))
        }
    }

    /// Requests that never reached an arena (no installed arena held the bank),
    /// so no policy can change them. Named apart from `Inkling2ArenaCounters
    /// .bypassed`, which is the admission rule refusing to KEEP an expert.
    public var unheld: Int { requests.lazy.filter { $0.arena < 0 }.count }

    // MARK: The file

    /// A header line, then one `arena layer expert` line per request: readable
    /// with `grep`, and small enough (one line per routed expert) that a whole
    /// walk's routing is a few megabytes.
    public func write(to url: URL, header: Header) throws {
        var text = "# inkling2 routing trace\n"
        for index in header.capacities.indices {
            text += "arena \(index) capacity \(header.capacities[index])"
                + " slotBytes \(header.slotBytes[index])"
                + " hits \(header.counters[index].hits)"
                + " misses \(header.counters[index].misses)"
                + " evictions \(header.counters[index].evictions)"
                + " incompatible \(header.counters[index].incompatible)"
                + " scratch \(index < header.scratchSlots.count ? header.scratchSlots[index] : 0)"
                + " policy \(index < header.admission.count && header.admission[index] == .byFrequency ? "byFrequency" : "always")\n"
        }
        text += "# arena layer expert\n"
        for request in requests {
            text += "\(request.arena) \(request.layer) \(request.expert)\n"
        }
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    public static func read(from url: URL) throws -> (header: Header, requests: [Request]) {
        let text = try String(contentsOf: url, encoding: .utf8)
        var header = Header()
        var requests: [Request] = []
        requests.reserveCapacity(1 << 19)
        var inRequests = false
        for line in text.split(separator: "\n") {
            if line.hasPrefix("# arena layer expert") { inRequests = true; continue }
            if line.hasPrefix("#") { continue }
            if !inRequests {
                // arena N capacity C slotBytes B hits H misses M evictions E incompatible X
                let fields = line.split(separator: " ")
                guard fields.count >= 14, fields[0] == "arena" else { continue }
                header.capacities.append(Int(fields[3]) ?? 0)
                header.slotBytes.append(Int(fields[5]) ?? 0)
                var counters = Inkling2ArenaCounters()
                counters.hits = Int(fields[7]) ?? 0
                counters.misses = Int(fields[9]) ?? 0
                counters.evictions = Int(fields[11]) ?? 0
                counters.incompatible = Int(fields[13]) ?? 0
                header.counters.append(counters)
                let scratch = line.firstRange(of: " scratch ")
                if let scratch {
                    let tail = line[scratch.upperBound...].split(separator: " ")
                    header.scratchSlots.append(Int(tail.first ?? "0") ?? 0)
                    header.admission.append(tail.count >= 3 && tail[2] == "byFrequency"
                        ? .byFrequency : .always)
                } else {
                    header.scratchSlots.append(0)
                    header.admission.append(.always)
                }
                continue
            }
            let fields = line.split(separator: " ")
            guard fields.count == 3,
                  let arena = Int(fields[0]), let layer = Int(fields[1]),
                  let expert = Int(fields[2]) else { continue }
            requests.append(Request(arena: arena, layer: layer, expert: expert))
        }
        return (header, requests)
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
/// One warp per output row, a BLOCK of sixteen values per lane iteration, and a
/// warp-scoped reduction. Slice 35 priced the shape over the published slot
/// shape against the row kernel it replaced: 1.00 ms against 3.30 ms (3.3x,
/// 35.8 GB/s against 10.8), with every decode identical.
///
/// Slice 37 priced the last lever the shape left -- one threadgroup per row
/// re-reads the row's whole activation vector per row -- by giving a warp
/// several rows and hoisting the activation block above the row loop. It LOSES,
/// monotonically: width 1 measured 1.005 ms, two rows 1.034, four 1.064, eight
/// 1.077, all bit-identical. The activation stream is therefore NOT what limits
/// this kernel, which is what the shape table says on its own; the same probe's
/// control, the width-1 shape with `inkling_round_bf16` dropped, measures
/// 0.728 ms, so that one required per-value step is a quarter of the leg.
///
/// The same probe found that the width-1 GROUP body, which computes the same
/// sums with the same lane map and reduces with a bare `simd_sum` instead of
/// `threadgroup_reduce_sum`'s three barriers, measured 0.967 ms against the
/// block kernel's 1.005 -- 4% for a bit-identical output (worst relative
/// 0.00e+00 over the mixture's 2,304 outputs). That is what ships.
nonisolated public enum Inkling2ExpertRowDispatch {
    public static let kernelName = "inkling_expert_row_group1_f32"
    public static let threadgroup = 32
    /// Output rows one threadgroup projects. More than 1 is the priced shape
    /// that lost; it stays available so the table can be re-measured.
    public static let rowsPerThreadgroup = 1
    /// The group-family kernel of a given width.
    public static func groupKernelName(rowsPerGroup: Int) -> String? {
        switch rowsPerGroup {
        case 1: return "inkling_expert_row_group1_f32"
        case 2: return "inkling_expert_row_group2_f32"
        case 4: return "inkling_expert_row_group4_f32"
        case 8: return "inkling_expert_row_group8_f32"
        default: return nil
        }
    }
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
        rowsPerGroup: Int = 1,
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
        // The group kernels take the row count so their last group can be
        // partial. It is bound whatever the pipeline is: an index a pipeline
        // does not declare is ignored, which is what lets a caller pass a
        // width-1 group kernel here without having to say so.
        encoder.setBuffer(arena.make(UInt32(rows)), offset: 0, index: 12)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + rowsPerGroup - 1) / rowsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroup, height: 1, depth: 1)
        )
    }
}

/// Whether a miss is allowed to take a residency.
///
/// `always` is the rule every slice before 36 shipped: the least recently used
/// resident is evicted for any miss. `byFrequency` is an ADMISSION filter -- a
/// miss replaces a resident only when it has been requested more often than the
/// resident would be -- and its misses land in scratch slots instead.
///
/// Measured on the walk's own routing (slice 36): `always` 147,505 + 21,636
/// misses, `byFrequency` 110,764 + 16,102 -- 25% fewer bytes read, on both
/// halves of the walk and on both arenas, with no decay horizon to tune. What it
/// must NOT be confused with is frequency EVICTION, which measured far worse
/// than recency (45.4% and 11.9% hit rates): the win is refusing to install, not
/// choosing the victim by count.
nonisolated public enum Inkling2ArenaAdmission: Sendable, Equatable {
    case always
    case byFrequency
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
    /// How many misses the admission rule may read without keeping them. A
    /// visit's fill carries up to `numExpertsPerToken` reroutes, and each needs a
    /// slot that survives until the mixture encodes it, so a caller that wants
    /// `byFrequency` must give it that many. Zero -- the default -- means every
    /// miss keeps its residency, which is what a caller with its own small slot
    /// buffer wants.
    public let scratchSlots: Int
    public let admission: Inkling2ArenaAdmission
    private var residents: [Inkling2ResidentExpert?]
    /// Most recently used first.
    private var order: [(layer: Int, expert: Int)] = []
    /// Requests per (layer, expert) over the arena's whole life. This is what an
    /// admission filter needs and what frequency EVICTION would also need -- the
    /// difference is that this only decides whether a miss may install, so a
    /// resident is never displaced for a less-wanted expert. Keys are
    /// `layer * 128 + expert`, bounded by the checkpoint's own 3,072 pairs.
    private var requests: [Int: Int] = [:]
    private var scratchCursor = 0
    private var counters = Inkling2ArenaCounters()

    public init(
        capacity: Int,
        geometry: Inkling2ExpertSlotGeometry,
        scratchSlots: Int = 0,
        admission: Inkling2ArenaAdmission = .always
    ) {
        self.capacity = max(1, capacity)
        self.geometry = geometry
        self.scratchSlots = max(0, scratchSlots)
        self.admission = scratchSlots > 0 ? admission : .always
        self.residents = Array(repeating: nil, count: max(1, capacity))
    }

    /// The slots the arena's buffer must hold: one per resident plus the scratch
    /// ones a bypassed miss is read through.
    public var slotCount: Int { capacity + scratchSlots }

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
        if admission == .byFrequency { noteRequest(layer: layer, expert: expert) }
        return residents.firstIndex { $0?.layer == layer && $0?.expert == expert }
    }

    /// One request for (layer, expert), for the admission filter. Its `false`
    /// verdict on a cold expert is only correct if a HIT counts as a request
    /// too, which is why `resident` calls this as well -- a filter that only
    /// counted misses would never learn that a resident is wanted.
    @discardableResult
    private func noteRequest(layer: Int, expert: Int) -> Int {
        let key = layer * 128 + expert
        let next = (requests[key] ?? 0) + 1
        requests[key] = next
        return next
    }

    /// Whether `slot` is a scratch slot: a bypassed miss's landing place, which
    /// never holds a residency.
    public func isScratch(slot: Int) -> Bool { slot >= capacity }

    /// Whether (layer, expert) is resident, without asking the arena for it:
    /// `resident` is a REQUEST (it counts a hit and, under admission, a request),
    /// so an inspection that used it would change what it inspected.
    public func isResident(layer: Int, expert: Int) -> Bool {
        order.contains { $0.layer == layer && $0.expert == expert }
    }

    /// The slot to fill for (layer, expert): an empty one, a scratch one when
    /// the admission rule refuses the residency, or the least recently used one
    /// (counted as an eviction). The caller fills it and then calls `didFill`
    /// only when the second element says the slot keeps the expert.
    public func slotForFill(layer: Int, expert: Int) -> Int {
        take(layer: layer, expert: expert).slot
    }

    /// One miss's slot, and whether that slot KEEPS the expert.
    ///
    /// The residency is recorded HERE, before the bytes arrive, because a batch
    /// needs it: a fill resolves every request of one visit before reading any of
    /// them, and with nothing written into `residents` between two reservations
    /// every miss of the batch would pick the same free slot (slice 34).
    private func take(layer: Int, expert: Int) -> (slot: Int, kept: Bool) {
        counters.misses += 1
        let wanted = admission == .byFrequency ? noteRequest(layer: layer, expert: expert) : 0
        if let empty = residents.firstIndex(where: { $0 == nil }) {
            install(empty, layer: layer, expert: expert)
            return (empty, true)
        }
        // Which resident leaves. Recency is the shipped rule; an admission
        // filter has to look at the COLDEST resident rather than the oldest,
        // because the resident it wants to compare against is the one it would
        // actually displace -- evicting a frequent expert to refuse a cold one
        // would be the worse of both rules. The tie-break is recency, so the
        // scan starts at the oldest and keeps a strictly colder resident.
        var victim = order[order.count - 1]
        if admission == .byFrequency {
            var coldest = requests[victim.layer * 128 + victim.expert] ?? 0
            for entry in order.reversed() {
                let count = requests[entry.layer * 128 + entry.expert] ?? 0
                if count < coldest {
                    coldest = count
                    victim = entry
                }
            }
            if scratchSlots > 0, wanted <= coldest {
                // Read it, but do not keep it: the resident it would displace
                // has been asked for at least as often. The slot is scratch, so
                // no residency and no eviction -- the next visit misses it
                // again, which is the point.
                counters.bypassed += 1
                let slot = capacity + scratchCursor
                scratchCursor = (scratchCursor + 1) % scratchSlots
                return (slot, false)
            }
        }
        counters.evictions += 1
        order.removeAll { $0.layer == victim.layer && $0.expert == victim.expert }
        let slot = residents.firstIndex {
            $0?.layer == victim.layer && $0?.expert == victim.expert
        } ?? 0
        install(slot, layer: layer, expert: expert)
        return (slot, true)
    }

    private func install(_ slot: Int, layer: Int, expert: Int) {
        residents[slot] = Inkling2ResidentExpert(
            layer: layer, expert: expert, slot: slot,
            gateUpSourceOffset: 0, downSourceOffset: 0
        )
        order.removeAll { $0.layer == layer && $0.expert == expert }
        order.insert((layer, expert), at: 0)
    }

    /// Records a completed fill (and what it cost).
    ///
    /// `kept` false is a bypassed miss: its bytes were read and are billed, but
    /// it took a scratch slot, so it must not read as a resident expert.
    public func didFill(
        slot: Int, layer: Int, expert: Int,
        gateUpSourceOffset: UInt64, downSourceOffset: UInt64,
        bytes: UInt64, preadNanoseconds: UInt64,
        kept: Bool = true
    ) {
        if kept, slot < capacity {
            residents[slot] = Inkling2ResidentExpert(
                layer: layer, expert: expert, slot: slot,
                gateUpSourceOffset: gateUpSourceOffset, downSourceOffset: downSourceOffset
            )
            order.removeAll { $0.layer == layer && $0.expert == expert }
            order.insert((layer, expert), at: 0)
        }
        counters.bytesRead += bytes
        counters.preadNanoseconds += preadNanoseconds
    }

    /// Records a reserved slot's true source offsets and what its read cost.
    ///
    /// A bypassed miss still READ its bytes, so the byte and pread counters are
    /// billed either way; only the residency is skipped, and a scratch slot is
    /// never written into `residents` (`slot >= capacity` would be out of
    /// bounds, and a scratch slot that claimed an expert is exactly the false
    /// hit the arena's own contract forbids).
    private func recordFill(
        slot: Int, kept: Bool, gateUpSourceOffset: UInt64, downSourceOffset: UInt64,
        bytes: UInt64, preadNanoseconds: UInt64
    ) {
        if kept, slot < capacity, let entry = residents[slot] {
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
        let taken = take(layer: layer, expert: expert)
        let slot = taken.slot
        guard let buffer = slotBuffer else {
            // Bookkeeping only: the caller fills the slot itself.
            didFill(
                slot: slot, layer: layer, expert: expert,
                gateUpSourceOffset: gateUpSourceOffset, downSourceOffset: downSourceOffset,
                bytes: 0, preadNanoseconds: 0, kept: taken.kept
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
            bytes: UInt64(read), preadNanoseconds: elapsed, kept: taken.kept
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
        var misses: [(request: FillRequest, slot: Int, kept: Bool)] = []
        for request in requests {
            if let hit = resident(layer: request.layer, expert: request.expert) {
                slots.append(hit)
            } else {
                let taken = take(layer: request.layer, expert: request.expert)
                slots.append(taken.slot)
                misses.append((request, taken.slot, taken.kept))
            }
        }
        guard !misses.isEmpty else { return slots }
        guard let buffer = slotBuffer else {
            // Bookkeeping only: the caller fills the slots itself, so there is no
            // read to race and every reservation is recorded.
            for miss in misses {
                recordFill(
                    slot: miss.slot, kept: miss.kept,
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
                slot: miss.slot, kept: miss.kept,
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
            + "\(stats.bypassed) bypassed (" + (admission == .byFrequency ? "admission" : "always")
            + (scratchSlots > 0 ? ", \(scratchSlots) scratch" : "") + "), "
            + String(format: "%.2f GB read in %.1f ms (%.1f GB/s), hit rate %.1f%%",
                     gigabytes, Double(stats.preadNanoseconds) / 1_000_000,
                     stats.preadNanoseconds > 0
                        ? gigabytes / (Double(stats.preadNanoseconds) / 1_000_000_000) : 0,
                     100 * stats.hitRate)
            + (stats.incompatible > 0 ? ", \(stats.incompatible) refused (a bank this slot cannot hold)" : "")
    }
}
