//
//  DeepSeekV41ExpertStreamer.swift
//  DynaMoE
//
//  Streams routed MXFP4 experts from the shards into resident arena slots.
//
//  The routed mixture is 384 experts per layer over 40 layers, i.e. 15,360
//  `(layer, expert)` pairs of ~19 MB each, so the checkpoint's experts are not
//  weights this machine holds: they are a working set it reads. Two ways to read
//  them exist and they are not equivalent:
//
//    * through the zero-copy shard `mmap`, which is what the loader already gave
//      the engine. A page fault inside a Metal read is served one page at a time
//      and the measured throughput collapses to ~0.065 GB/s when it interleaves
//      with other GPU work (`Benchmarks/PERF_FINDINGS.md`), which is ~4.5 GB of
//      expert traffic per position cold.
//    * into an anonymous slot with POSIX `pread`, which is 3.7-4.0 GB/s cold and
//      19-23 GB/s warm, and whose destination the GPU can already be reading.
//
//  This file is the second. The arena keeps deciding *which* pairs are resident
//  (whole pairs, LRU, never half an expert); this only moves bytes for the pairs
//  the arena hands out, six contiguous ranges per expert: gate, up and down,
//  each with its E8M0 scale companion.
//
//  Correctness rests on one invariant rather than on luck: a slot is only ever
//  overwritten when no committed command buffer can still read it. The layer loop
//  honours that by settling every pending buffer before it encodes a layer's front
//  half, and every layer's expert reads happen after the routing readback, so the
//  slot writes for layer N always follow a full drain of layer N-1's work. If that
//  ordering ever changes, the pread can land under a running kernel and the result
//  is wrong logits with no error anywhere.
//

import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

// MARK: - Errors

nonisolated public enum DeepSeekV41ExpertStreamError: Error, Equatable, CustomStringConvertible {
    case cannotOpen(String)
    case shortRead(path: String, offset: Int, wanted: Int, got: Int)
    case geometryMismatch(String)
    case missingTensor(String)
    case allocationFailed(slot: Int, bytes: Int)
    case tooManyExperts(requested: Int, capacity: Int)

    public var description: String {
        switch self {
        case .cannotOpen(let detail):
            return "cannot open an expert shard: \(detail)"
        case let .shortRead(path, offset, wanted, got):
            return "expert read (\(path)) at \(offset) returned \(got) of \(wanted) bytes"
        case .geometryMismatch(let detail):
            return "DeepSeek V4.1 expert layout \(detail)"
        case .missingTensor(let name):
            return "DeepSeek V4.1 checkpoint is missing the routed expert '\(name)'"
        case let .allocationFailed(slot, bytes):
            return "DeepSeek V4.1 could not allocate the \(bytes)-byte arena slot \(slot)"
        case let .tooManyExperts(requested, capacity):
            return "DeepSeek V4.1 routed \(requested) experts into an arena of \(capacity) slots"
        }
    }
}

// MARK: - Slot geometry

/// Byte layout of one `(layer, expert)` pair in an arena slot.
///
/// The six ranges are exactly the six safetensors entries the checkpoint publishes
/// per expert, in the order the executor binds them. Nothing is repacked and
/// nothing is padded: the kernels index a row as `row * inDim / 2` bytes and a
/// scale row as `row * inDim / 32` bytes from the tensor base, so a contiguous
/// copy of the published bytes is what they already expect.
nonisolated public struct DeepSeekV41ExpertSlotGeometry: Sendable, Equatable {
    public let format: DeepSeekV41LinearFormat
    public let gateWeightBytes: Int
    public let gateScaleBytes: Int
    public let upWeightBytes: Int
    public let upScaleBytes: Int
    public let downWeightBytes: Int
    public let downScaleBytes: Int

    /// Byte offset of each range inside the slot, in the order they are read.
    public var gateWeightOffset: Int { 0 }
    public var gateScaleOffset: Int { gateWeightBytes }
    public var upWeightOffset: Int { gateScaleOffset + gateScaleBytes }
    public var upScaleOffset: Int { upWeightOffset + upWeightBytes }
    public var downWeightOffset: Int { upScaleOffset + upScaleBytes }
    public var downScaleOffset: Int { downWeightOffset + downWeightBytes }

    /// Bytes one whole expert occupies, weights plus scales.
    public var slotBytes: Int { downScaleOffset + downScaleBytes }

    /// Bytes one matrix occupies on disk: packed two values per byte for MXFP4,
    /// one per byte for the block-scaled FP8 grid, two per value for bf16.
    public static func weightBytes(
        values: Int,
        format: DeepSeekV41LinearFormat
    ) -> Int {
        switch format {
        case .mxfp4: return (max(0, values) + 1) / 2
        case .fp8Block32: return max(0, values)
        case .bf16: return 2 * max(0, values)
        case .unsupported: return 0
        }
    }

    public static func make(
        hidden: Int,
        intermediate: Int,
        format: DeepSeekV41LinearFormat
    ) -> DeepSeekV41ExpertSlotGeometry? {
        guard hidden > 0, intermediate > 0, format.isSupported else { return nil }
        let block = format.blockWidth
        // A block-scaled expert splits its input width into quantisation blocks, so
        // an input width that does not divide is a shape no kernel can read.
        if block > 0, (hidden % block != 0 || intermediate % block != 0) { return nil }

        let gateValues = intermediate * hidden
        let downValues = hidden * intermediate
        func scales(_ rows: Int, _ columns: Int) -> Int {
            block > 0 ? rows * (columns / block) : 0
        }
        return DeepSeekV41ExpertSlotGeometry(
            format: format,
            gateWeightBytes: weightBytes(values: gateValues, format: format),
            gateScaleBytes: scales(intermediate, hidden),
            upWeightBytes: weightBytes(values: gateValues, format: format),
            upScaleBytes: scales(intermediate, hidden),
            downWeightBytes: weightBytes(values: downValues, format: format),
            downScaleBytes: scales(hidden, intermediate)
        )
    }
}

// MARK: - Resident binding

/// Where one resident expert's six ranges live, ready to bind.
///
/// One buffer holds the whole pair, so every kernel binds the same `MTLBuffer` and
/// differs only in the byte offset it passes as a kernel argument. That keeps the
/// offsets off the `setBuffer` path, where Metal's alignment rules would apply.
nonisolated public struct DeepSeekV41ResidentExpert: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let gateWeightOffset: UInt64
    public let gateScaleOffset: UInt64
    public let upWeightOffset: UInt64
    public let upScaleOffset: UInt64
    public let downWeightOffset: UInt64
    public let downScaleOffset: UInt64

    public init(
        buffer: MTLBuffer,
        gateWeightOffset: UInt64,
        gateScaleOffset: UInt64,
        upWeightOffset: UInt64,
        upScaleOffset: UInt64,
        downWeightOffset: UInt64,
        downScaleOffset: UInt64
    ) {
        self.buffer = buffer
        self.gateWeightOffset = gateWeightOffset
        self.gateScaleOffset = gateScaleOffset
        self.upWeightOffset = upWeightOffset
        self.upScaleOffset = upScaleOffset
        self.downWeightOffset = downWeightOffset
        self.downScaleOffset = downScaleOffset
    }

    public init(buffer: MTLBuffer, geometry: DeepSeekV41ExpertSlotGeometry) {
        self.init(
            buffer: buffer,
            gateWeightOffset: UInt64(geometry.gateWeightOffset),
            gateScaleOffset: UInt64(geometry.gateScaleOffset),
            upWeightOffset: UInt64(geometry.upWeightOffset),
            upScaleOffset: UInt64(geometry.upScaleOffset),
            downWeightOffset: UInt64(geometry.downWeightOffset),
            downScaleOffset: UInt64(geometry.downScaleOffset)
        )
    }
}

// MARK: - Shard reader

/// One shard's descriptor, plus the batch read the streamer issues.
///
/// `pread` on a descriptor is positional and needs no seek, so disjoint ranges can
/// be read concurrently with no shared state beyond the descriptor itself. The lock
/// is held for a whole batch so `closeNow()` cannot recycle the number under an
/// in-flight read: a recycled descriptor would silently read another file's bytes.
nonisolated public final class DeepSeekV41ExpertShardReader: @unchecked Sendable {

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
            throw DeepSeekV41ExpertStreamError.cannotOpen("\(url.lastPathComponent): errno \(errno)")
        }
        var status = stat()
        guard fstat(fd, &status) == 0 else {
            let code = errno
            close(fd)
            throw DeepSeekV41ExpertStreamError.cannotOpen(
                "\(url.lastPathComponent): fstat failed, errno \(code)"
            )
        }
        self.descriptor = fd
        self.path = url.lastPathComponent
        self.fileSize = Int(status.st_size)
    }

    deinit {
        closeNow()
    }

    public func closeNow() {
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else { return }
        isClosed = true
        close(descriptor)
    }

    /// Reads every range, in parallel once there are enough of them for queue depth
    /// to matter. Returns the bytes read. Throws the first failure, naming the range.
    @discardableResult
    public func read(_ ranges: [Range]) throws -> Int {
        guard !ranges.isEmpty else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else {
            throw DeepSeekV41ExpertStreamError.cannotOpen("\(path) was closed")
        }
        // Checked before any read: `pread` past the end returns a short count, and a
        // short count in the middle of a matrix is a silently wrong weight row.
        for range in ranges {
            guard range.fileOffset >= 0, range.length >= 0,
                  range.fileOffset + range.length <= fileSize else {
                throw DeepSeekV41ExpertStreamError.shortRead(
                    path: path,
                    offset: range.fileOffset,
                    wanted: range.length,
                    got: 0
                )
            }
        }

        let fd = descriptor
        let failures = DeepSeekV41ReadFailureBox()
        let context = DeepSeekV41RangeReadContext(
            fd: fd,
            path: path,
            ranges: ranges,
            failures: failures
        )

        let readOne: @Sendable (Int) -> Void = { index in context.read(index) }
        if ranges.count >= 2 {
            DispatchQueue.concurrentPerform(iterations: ranges.count, execute: readOne)
        } else {
            for index in ranges.indices { readOne(index) }
        }
        if let first = failures.first { throw first }
        return ranges.reduce(0) { $0 + $1.length }
    }
}

/// Values the parallel reads need. Every range writes a disjoint destination, so
/// sharing the raw pointers across the concurrent reads is safe.
nonisolated private struct DeepSeekV41RangeReadContext: @unchecked Sendable {
    let fd: Int32
    let path: String
    let ranges: [DeepSeekV41ExpertShardReader.Range]
    let failures: DeepSeekV41ReadFailureBox

    func read(_ index: Int) {
        let range = ranges[index]
        var total = 0
        while total < range.length {
            let got = pread(
                fd,
                range.destination.advanced(by: total),
                range.length - total,
                off_t(range.fileOffset + total)
            )
            if got < 0 {
                if errno == EINTR { continue }
                failures.record(.shortRead(
                    path: path,
                    offset: range.fileOffset + total,
                    wanted: range.length - total,
                    got: Int(got)
                ))
                return
            }
            if got == 0 {
                failures.record(.shortRead(
                    path: path,
                    offset: range.fileOffset + total,
                    wanted: range.length - total,
                    got: 0
                ))
                return
            }
            total += got
        }
    }
}

nonisolated private final class DeepSeekV41ReadFailureBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: DeepSeekV41ExpertStreamError?

    func record(_ error: DeepSeekV41ExpertStreamError) {
        lock.lock()
        if stored == nil { stored = error }
        lock.unlock()
    }

    var first: DeepSeekV41ExpertStreamError? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

/// One shard's worth of ranges, plus what the concurrent reads need to report a
/// failure. Ranges in different shards write disjoint destinations, so the batches
/// can run at the same time.
nonisolated private struct DeepSeekV41ShardBatchContext: @unchecked Sendable {
    let batches: [(shard: UInt32, ranges: [DeepSeekV41ExpertShardReader.Range])]
    let readers: [UInt32: DeepSeekV41ExpertShardReader]
    let failures: DeepSeekV41ReadFailureBox

    func run(_ index: Int) {
        let batch = batches[index]
        guard let reader = readers[batch.shard] else {
            failures.record(.cannotOpen("no descriptor for shard \(batch.shard)"))
            return
        }
        do {
            try reader.read(batch.ranges)
        } catch let error as DeepSeekV41ExpertStreamError {
            failures.record(error)
        } catch {
            failures.record(.cannotOpen("\(error)"))
        }
    }
}

// MARK: - Streamer

/// Owns the arena slots, the shard descriptors, and the read that fills them.
///
/// Slots are allocated on first use rather than all at once: the arena's capacity
/// is what the memory plan says this machine can hold, but a turn that routes over
/// a handful of layers has no reason to commit the rest up front.
nonisolated public final class DeepSeekV41ExpertStreamer: @unchecked Sendable {

    public struct Reservation: Sendable, Equatable {
        public let expert: Int
        public let slot: Int
        /// True when the slot already held this pair, i.e. no bytes were read.
        public let isHit: Bool

        public init(expert: Int, slot: Int, isHit: Bool) {
            self.expert = expert
            self.slot = slot
            self.isHit = isHit
        }
    }

    public let geometry: DeepSeekV41ExpertSlotGeometry
    public let capacity: Int

    private let device: MTLDevice
    private let arena: DeepSeekV41ExpertArena
    private let weights: DeepSeekV41DeviceWeights
    private let readers: [UInt32: DeepSeekV41ExpertShardReader]
    /// One buffer per slot, allocated when the slot is first written.
    private var slotBuffers: [MTLBuffer?]
    private let lock = NSLock()

    /// Diagnostic counters, so the load log and a dogfood run can show whether the
    /// arena is sized sanely instead of guessing from wall time.
    public private(set) var expertsLoaded: Int = 0
    public private(set) var bytesRead: Int = 0

    public init(
        device: MTLDevice,
        weights: DeepSeekV41DeviceWeights,
        summary: ModelSummary,
        snapshotDir: URL,
        arena: DeepSeekV41ExpertArena,
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan]
    ) throws {
        // The geometry is one layout for the whole arena, so every layer's expert
        // matrices have to agree with it. Sampling one expert per layer is cheap and
        // turns a heterogeneous checkpoint into a load-time failure instead of a
        // slot that is read at the wrong stride.
        var sample: DeepSeekV41DeviceTensor?
        var sampleScale: DeepSeekV41DeviceTensor?
        var sampleLayer = 0
        for layer in plan {
            guard let gate = weights.tensor(.expertW1, layer: layer.index, expert: 0) else { continue }
            let gateScale = weights.scaleTensor(.expertW1, layer: layer.index, expert: 0)
            if let sample, let sampleScale {
                guard gate.rows == sample.rows, gate.columns == sample.columns,
                      gate.dtype == sample.dtype,
                      gateScale?.rows == sampleScale.rows,
                      gateScale?.columns == sampleScale.columns,
                      gateScale?.dtype == sampleScale.dtype else {
                    throw DeepSeekV41ExpertStreamError.geometryMismatch(
                        "differs between layers: layer \(layer.index) publishes "
                            + "\(gate.rows)x\(gate.columns) \(gate.dtype)"
                    )
                }
            } else {
                sample = gate
                sampleScale = gateScale
                sampleLayer = layer.index
            }
        }
        guard let sample else {
            throw DeepSeekV41ExpertStreamError.missingTensor("ffn.experts.0.w1")
        }
        // One layer has to fit in the arena for streaming to be usable at all: a
        // capacity below the top-k would have to overwrite a slot inside the layer
        // that is still reading it. The caller falls back to the shard mappings.
        guard arena.capacity >= max(1, config.numActivatedExperts) else {
            throw DeepSeekV41ExpertStreamError.tooManyExperts(
                requested: max(1, config.numActivatedExperts), capacity: arena.capacity
            )
        }
        let format = DeepSeekV41DevicePlan.linearFormat(tensor: sample, scale: sampleScale)
        guard let geometry = DeepSeekV41ExpertSlotGeometry.make(
            hidden: config.hiddenSize,
            intermediate: config.moeIntermediateSize,
            format: format
        ) else {
            throw DeepSeekV41ExpertStreamError.geometryMismatch(
                "resolves to \(format.rawValue), which no routed-expert kernel reads at "
                    + "\(config.hiddenSize)x\(config.moeIntermediateSize)"
            )
        }
        // The published row widths are the ground truth for how a slot is laid out:
        // a packed MXFP4 weight row is half its input width, an E4M3 row is the full
        // width, and the scale row is one byte per 32 values. Disagreement here means
        // the kernel would index past the range that was read.
        let expectedWeightColumns = geometry.format == .mxfp4
            ? config.hiddenSize / 2
            : config.hiddenSize
        guard sample.columns == expectedWeightColumns else {
            throw DeepSeekV41ExpertStreamError.geometryMismatch(
                "publishes \(sample.columns) weight columns for a \(config.hiddenSize)-wide input "
                    + "at \(format.rawValue); expected \(expectedWeightColumns)"
            )
        }
        let expectedDownColumns = geometry.format == .mxfp4
            ? config.moeIntermediateSize / 2
            : config.moeIntermediateSize
        if let down = weights.tensor(.expertW2, layer: sampleLayer, expert: 0) {
            guard down.columns == expectedDownColumns, down.rows == config.hiddenSize else {
                throw DeepSeekV41ExpertStreamError.geometryMismatch(
                    "publishes a \(down.rows)x\(down.columns) down projection for a "
                        + "\(config.hiddenSize)-wide output over \(config.moeIntermediateSize) "
                        + "intermediate values; expected \(config.hiddenSize)x\(expectedDownColumns)"
                )
            }
        }
        if geometry.gateScaleBytes > 0 {
            guard let sampleScale,
                  sampleScale.columns == config.hiddenSize / 32 else {
                throw DeepSeekV41ExpertStreamError.geometryMismatch(
                    "publishes \(sampleScale?.columns ?? 0) scale columns for a "
                        + "\(config.hiddenSize)-wide input; expected \(config.hiddenSize / 32)"
                )
            }
        }

        // Every expert of a layer is published in one shard, so one probe per layer
        // finds every shard that holds routed experts. The count is bounded by the
        // shard count (48 here), and a descriptor per shard is what makes the
        // per-layer reads positional instead of opening a file per layer.
        var shardIndexes = Set<UInt32>()
        for layer in plan {
            guard let tensor = weights.tensor(.expertW1, layer: layer.index, expert: 0) else {
                continue
            }
            shardIndexes.insert(tensor.shardIndex)
        }
        var readers: [UInt32: DeepSeekV41ExpertShardReader] = [:]
        for shard in shardIndexes.sorted() {
            let name = summary.shards.first(where: { $0.index == shard })?.filename
                ?? "model-\(shard).safetensors"
            let url = snapshotDir.appendingPathComponent(name)
            readers[shard] = try DeepSeekV41ExpertShardReader(url: url)
        }
        guard !readers.isEmpty else {
            throw DeepSeekV41ExpertStreamError.missingTensor("ffn.experts.*")
        }

        self.device = device
        self.weights = weights
        self.arena = arena
        self.geometry = geometry
        self.capacity = arena.capacity
        self.readers = readers
        self.slotBuffers = Array(repeating: nil, count: arena.capacity)
    }

    deinit {
        for reader in readers.values { reader.closeNow() }
    }

    public func close() {
        for reader in readers.values { reader.closeNow() }
    }

    /// One-line residency report for the load log.
    public func describe() -> String {
        String(
            format: "experts: streamed into %d arena slots of %.1f MB (%@, %d loaded, %.2f GB read, %.0f%% hit rate)",
            capacity,
            Double(geometry.slotBytes) / 1_048_576,
            geometry.format.rawValue,
            expertsLoaded,
            Double(bytesRead) / 1_073_741_824,
            arena.hitRate * 100
        )
    }

    /// Makes every requested expert resident and returns its bindings, in the order
    /// requested.
    ///
    /// The arena decides placement (whole pairs, LRU) and this fills the misses.
    /// A failed read vacates the slots it touched, so the arena never claims a pair
    /// is resident when its bytes are only half there.
    public func ensureResident(
        layer: Int,
        experts: [Int]
    ) throws -> [DeepSeekV41ResidentExpert] {
        guard !experts.isEmpty else { return [] }
        guard experts.count <= capacity else {
            throw DeepSeekV41ExpertStreamError.tooManyExperts(
                requested: experts.count, capacity: capacity
            )
        }
        let reservations = arena.acquireDetailed(
            experts.map { (layer: layer, expert: $0) }
        )

        // Group the misses by shard so each file takes one batch of ranges, which is
        // the only way the reads reach queue depth.
        var batches: [UInt32: [DeepSeekV41ExpertShardReader.Range]] = [:]
        var misses: [(slot: Int, expert: Int)] = []
        for (index, reservation) in reservations.enumerated() where !reservation.isHit {
            let expert = experts[index]
            guard let gate = weights.tensor(.expertW1, layer: layer, expert: expert),
                  let up = weights.tensor(.expertW3, layer: layer, expert: expert),
                  let down = weights.tensor(.expertW2, layer: layer, expert: expert) else {
                vacate(misses.map(\.slot) + [reservation.slot])
                throw DeepSeekV41ExpertStreamError.missingTensor(
                    "layers.\(layer).ffn.experts.\(expert)"
                )
            }
            let gateScale = weights.scaleTensor(.expertW1, layer: layer, expert: expert)
            let upScale = weights.scaleTensor(.expertW3, layer: layer, expert: expert)
            let downScale = weights.scaleTensor(.expertW2, layer: layer, expert: expert)
            if geometry.gateScaleBytes > 0, gateScale == nil || upScale == nil || downScale == nil {
                vacate(misses.map(\.slot) + [reservation.slot])
                throw DeepSeekV41ExpertStreamError.missingTensor(
                    "layers.\(layer).ffn.experts.\(expert).w*.scale"
                )
            }
            let buffer: MTLBuffer
            do {
                buffer = try slotBuffer(reservation.slot)
            } catch {
                vacate(misses.map(\.slot) + [reservation.slot])
                throw error
            }
            let base = buffer.contents()
            func range(
                _ tensor: DeepSeekV41DeviceTensor?,
                offset: Int,
                length: Int
            ) -> DeepSeekV41ExpertShardReader.Range? {
                guard length > 0, let tensor else { return nil }
                return DeepSeekV41ExpertShardReader.Range(
                    fileOffset: tensor.byteOffset,
                    destination: base.advanced(by: offset),
                    length: length
                )
            }
            let ranges = [
                range(gate, offset: geometry.gateWeightOffset, length: geometry.gateWeightBytes),
                range(gateScale, offset: geometry.gateScaleOffset, length: geometry.gateScaleBytes),
                range(up, offset: geometry.upWeightOffset, length: geometry.upWeightBytes),
                range(upScale, offset: geometry.upScaleOffset, length: geometry.upScaleBytes),
                range(down, offset: geometry.downWeightOffset, length: geometry.downWeightBytes),
                range(downScale, offset: geometry.downScaleOffset, length: geometry.downScaleBytes)
            ].compactMap { $0 }
            for range in ranges {
                batches[gate.shardIndex, default: []].append(range)
            }
            misses.append((slot: reservation.slot, expert: expert))
        }

        if !misses.isEmpty {
            let failures = DeepSeekV41ReadFailureBox()
            let context = DeepSeekV41ShardBatchContext(
                batches: batches.map { (shard: $0.key, ranges: $0.value) },
                readers: readers,
                failures: failures
            )
            let runOne: @Sendable (Int) -> Void = { index in context.run(index) }
            DispatchQueue.concurrentPerform(iterations: context.batches.count, execute: runOne)
            if let failure = failures.first {
                vacate(misses.map(\.slot))
                throw failure
            }
            lock.lock()
            expertsLoaded += misses.count
            bytesRead += misses.count * geometry.slotBytes
            lock.unlock()
        }

        return reservations.map { reservation in
            let buffer = slotBuffers[reservation.slot]
            guard let buffer else {
                // Unreachable: every miss allocated its buffer above and every hit
                // necessarily has one, but a nil here must not become a silent read
                // of an empty buffer.
                assertionFailure("DeepSeek V4.1 arena slot \(reservation.slot) has no buffer")
                return DeepSeekV41ResidentExpert(
                    buffer: device.makeBuffer(
                        length: max(geometry.slotBytes, 16),
                        options: .storageModeShared
                    )!,
                    geometry: geometry
                )
            }
            return DeepSeekV41ResidentExpert(buffer: buffer, geometry: geometry)
        }
    }

    private func vacate(_ slots: [Int]) {
        for slot in Set(slots) { arena.vacate(slot: slot) }
    }

    private func slotBuffer(_ slot: Int) throws -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
        guard slot >= 0, slot < slotBuffers.count else {
            throw DeepSeekV41ExpertStreamError.allocationFailed(slot: slot, bytes: geometry.slotBytes)
        }
        if let existing = slotBuffers[slot] { return existing }
        guard let buffer = device.makeBuffer(
            length: max(geometry.slotBytes, 16),
            options: .storageModeShared
        ) else {
            throw DeepSeekV41ExpertStreamError.allocationFailed(
                slot: slot, bytes: geometry.slotBytes
            )
        }
        buffer.label = "dsv41.expert.slot.\(slot)"
        slotBuffers[slot] = buffer
        return buffer
    }
}
