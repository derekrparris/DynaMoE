//
//  DeepSeekV41DensePin.swift
//  DynaMoE
//
//  Pins the dense backbone into resident buffers instead of reading it through
//  the shard mappings.
//
//  Every token reads ~170 MB of dense weights per layer plus the whole 1.3 GB
//  output head. Through a `MTLBuffer(bytesNoCopy:)` over a shard's mmap those
//  reads are GPU page faults: correct, and two orders of magnitude slower than
//  DRAM once they interleave with the expert stream, because the pages were
//  evicted by the routed experts read since the last token. The same bytes pread
//  into an anonymous `.storageModeShared` buffer are resident DRAM for the life
//  of the runtime, which is what `BackbonePinPool` already does for the
//  packed-format families.
//
//  This model cannot use that pool as-is: its backbone is scattered across 48
//  shards, each of which is mostly routed experts that must NOT be pinned. So the
//  plan packs the selected tensors into a handful of buffers, rewriting each
//  tensor's `(shardIndex, byteOffset)` to address its new home, and everything
//  downstream keeps binding tensors exactly the way it did before.
//
//  What is deliberately left on the mapping:
//
//    * routed experts (296 GB) — the arena preads whole pairs into slots,
//    * the Engram tables (203 GB) — read one row at a time,
//    * the embedding (1.3 GB) — read as single rows, so pinning it would buy
//      nothing for a token and cost a gigabyte,
//    * the vision tower and the draft block — the text graph never reads them.
//
//  One case the plan refuses: a tensor in a shard that `segmentShardsForMetal`
//  split, when the whole file collapsed to a single segment. The segment's file
//  start is not in the summary, and only the file's safetensors header carries it,
//  so those tensors stay on the mapping and the load log names them. In this
//  checkpoint that is the two Engram shards' dense half (`engram.wkv` and the two
//  hash vectors, ~314 MB of the backbone), because each 94.56 GiB file is almost
//  entirely a streamed ~100 GB table.
//

import Foundation
import Metal

// MARK: - Candidate

/// One tensor the pin plan may move: what it is called, where it lives in the
/// segmented summary, and how many bytes it occupies.
nonisolated public struct DeepSeekV41DensePinCandidate: Sendable, Equatable {
    public let name: String
    public let shardIndex: UInt32
    /// Relative to the segment its shard names, which is what the executor binds.
    public let byteOffset: Int
    public let byteLength: Int

    public init(name: String, shardIndex: UInt32, byteOffset: Int, byteLength: Int) {
        self.name = name
        self.shardIndex = shardIndex
        self.byteOffset = byteOffset
        self.byteLength = byteLength
    }
}

// MARK: - Plan

/// Where one tensor goes: the file range to read and the slot to read it into.
nonisolated public struct DeepSeekV41DensePinRange: Sendable, Equatable {
    public let tensorName: String
    /// The shard FILE, with any segmentation suffix stripped.
    public let filename: String
    /// Absolute offset in that file.
    public let fileOffset: Int
    public let length: Int
    /// Index into the plan's chunks, i.e. into the buffers the loader allocates.
    public let chunk: Int
    public let chunkOffset: Int

    public init(
        tensorName: String,
        filename: String,
        fileOffset: Int,
        length: Int,
        chunk: Int,
        chunkOffset: Int
    ) {
        self.tensorName = tensorName
        self.filename = filename
        self.fileOffset = fileOffset
        self.length = length
        self.chunk = chunk
        self.chunkOffset = chunkOffset
    }
}

/// The packing decision: which tensors are pinned, and into which buffers.
///
/// Pure arithmetic on the summary, so it is testable without a GPU or a
/// checkpoint, and so a load can report exactly what it is about to do before it
/// allocates anything.
nonisolated public struct DeepSeekV41DensePinPlan: Sendable, Equatable {

    /// Synthetic shard indices for the pinned chunks.
    ///
    /// `segmentShardsForMetal` numbers segments from 0, so a real index is always
    /// small; the pin lives far above it and the merged buffer registry stays
    /// collision-free.
    public static let pinnedShardBase: UInt32 = 0xD000_0000

    public let ranges: [DeepSeekV41DensePinRange]
    /// Exact byte length of each chunk, in chunk order. The loader allocates
    /// these rather than a round gigabyte each, so the tail of the last chunk is
    /// not wasted.
    public let chunkLengths: [Int]
    /// Tensors the plan refused, each with the reason, so the load log can say
    /// why a tensor is still paying page faults.
    public let skipped: [String]

    public var isEmpty: Bool { ranges.isEmpty }
    public var totalBytes: Int { ranges.reduce(0) { $0 + $1.length } }

    public func pinnedShardIndex(chunk: Int) -> UInt32 {
        Self.pinnedShardBase &+ UInt32(chunk)
    }

    public init(
        ranges: [DeepSeekV41DensePinRange],
        chunkLengths: [Int],
        skipped: [String]
    ) {
        self.ranges = ranges
        self.chunkLengths = chunkLengths
        self.skipped = skipped
    }

    /// Packs `candidates` into buffers of at most `chunkLimit` bytes, in a
    /// deterministic order, until `budgetBytes` is spent.
    ///
    /// `alignment` is the byte alignment of every slot. Metal needs a buffer
    /// binding offset to be at least word-aligned, and a page-aligned slot also
    /// keeps a tensor's rows from sharing a cache line with its neighbour.
    ///
    /// A tensor wider than `chunkLimit` gets a buffer of its own rather than being
    /// refused, up to `bufferLimit`: the output head is 1.3 GB and would otherwise
    /// be the one tensor every token reads in full and never gets pinned. Only a
    /// tensor past `bufferLimit`, i.e. past what the device can allocate at all,
    /// is skipped.
    public static func make(
        summary: ModelSummary,
        candidates: [DeepSeekV41DensePinCandidate],
        budgetBytes: Int,
        alignment: Int = 4096,
        chunkLimit: Int = 1 << 30,
        bufferLimit: Int? = nil
    ) -> DeepSeekV41DensePinPlan {
        let align = max(1, alignment)
        let limit = max(align, chunkLimit)
        let hardLimit = max(limit, bufferLimit ?? limit)
        // A segment's `filename` carries a `#n` suffix that names no file, and
        // its base address is the file's plus the segment's start. Resolving the
        // file once per name keeps both cases addressable: an unsegmented shard
        // resolves to start 0, which is the only case this checkpoint has.
        var fileBase: [String: UInt64] = [:]
        for shard in summary.shards where !shard.filename.contains("#") {
            fileBase[shard.filename] = shard.baseAddress
        }

        var located: [(candidate: DeepSeekV41DensePinCandidate, filename: String, fileOffset: Int)] = []
        var skipped: [String] = []
        for candidate in candidates.sorted(by: { priority($0.name) < priority($1.name) }) {
            guard candidate.byteLength > 0 else {
                skipped.append("\(candidate.name): the summary gives it no bytes")
                continue
            }
            guard candidate.byteLength <= hardLimit else {
                skipped.append(
                    "\(candidate.name): \(megabytes(candidate.byteLength)) MB is past what a "
                        + "single buffer may hold (\(megabytes(hardLimit)) MB)"
                )
                continue
            }
            guard let shard = summary.shards.first(where: { $0.index == candidate.shardIndex }) else {
                skipped.append("\(candidate.name): shard \(candidate.shardIndex) is not in the summary")
                continue
            }
            let filename = shard.filename.split(separator: "#", maxSplits: 1)
                .first.map(String.init) ?? shard.filename
            guard let base = fileBase[filename] else {
                skipped.append("\(candidate.name): no shard names \(filename) at file offset 0")
                continue
            }
            guard shard.baseAddress >= base else {
                skipped.append("\(candidate.name): \(filename) starts past its own segment")
                continue
            }
            // A segment-relative range has to fit the segment; the segment itself
            // is inside the file, so this also bounds the read.
            guard candidate.byteOffset >= 0,
                  candidate.byteOffset + candidate.byteLength <= Int(shard.length) else {
                skipped.append(
                    "\(candidate.name): \(candidate.byteOffset)+\(candidate.byteLength) is past "
                        + "\(filename)'s \(shard.length)-byte segment"
                )
                continue
            }
            located.append((
                candidate,
                filename,
                Int(shard.baseAddress - base) + candidate.byteOffset
            ))
        }

        var ranges: [DeepSeekV41DensePinRange] = []
        var chunkLengths: [Int] = []
        var chunk = 0
        var cursor = 0
        var spent = 0
        for entry in located {
            if spent + entry.candidate.byteLength > budgetBytes {
                skipped.append(
                    "\(entry.candidate.name): the \(megabytes(budgetBytes)) MB pin budget is spent"
                )
                continue
            }
            // A tensor wider than a packing chunk gets one of its own, sized to it.
            if entry.candidate.byteLength > limit {
                if cursor > 0 {
                    chunkLengths.append(max(align, cursor))
                    chunk += 1
                    cursor = 0
                }
                ranges.append(
                    DeepSeekV41DensePinRange(
                        tensorName: entry.candidate.name,
                        filename: entry.filename,
                        fileOffset: entry.fileOffset,
                        length: entry.candidate.byteLength,
                        chunk: chunk,
                        chunkOffset: 0
                    )
                )
                chunkLengths.append(entry.candidate.byteLength)
                chunk += 1
                spent += entry.candidate.byteLength
                continue
            }
            var offset = alignUp(cursor, align)
            if offset + entry.candidate.byteLength > limit {
                chunkLengths.append(max(align, cursor))
                chunk += 1
                cursor = 0
                offset = 0
            }
            ranges.append(
                DeepSeekV41DensePinRange(
                    tensorName: entry.candidate.name,
                    filename: entry.filename,
                    fileOffset: entry.fileOffset,
                    length: entry.candidate.byteLength,
                    chunk: chunk,
                    chunkOffset: offset
                )
            )
            cursor = offset + entry.candidate.byteLength
            spent += entry.candidate.byteLength
        }
        if cursor > 0 { chunkLengths.append(cursor) }

        return DeepSeekV41DensePinPlan(
            ranges: ranges,
            chunkLengths: chunkLengths,
            skipped: skipped
        )
    }

    /// The order tensors are pinned in, which is what a budget smaller than the
    /// backbone is spent on first.
    ///
    /// The output head goes first: it is model-level, it is read in full once per
    /// token, and it is the single largest tensor the graph touches. Everything
    /// else follows by name, which groups a layer's tensors together.
    private static func priority(_ name: String) -> String {
        name == "head.weight" ? "0|\(name)" : "1|\(name)"
    }

    private static func alignUp(_ value: Int, _ alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }

    private static func megabytes(_ bytes: Int) -> String {
        String(format: "%.1f", Double(bytes) / 1_048_576)
    }
}

// MARK: - Loader

nonisolated public enum DeepSeekV41DensePinError: Error, CustomStringConvertible {
    case cannotAllocate(bytes: Int)
    case cannotOpen(String)
    case shortRead(String)

    public var description: String {
        switch self {
        case .cannotAllocate(let bytes):
            return "a \(bytes)-byte pin buffer could not be allocated"
        case .cannotOpen(let file):
            return "\(file) could not be opened for the dense pin"
        case .shortRead(let detail):
            return "the dense pin could not read \(detail)"
        }
    }
}

/// Reads a plan's file ranges into fresh resident buffers.
///
/// One descriptor per shard file, one parallel pread per range, which is the same
/// shape the expert streamer uses and for the same reason: a single-threaded
/// pread of nine gigabytes is minutes of load time.
nonisolated public enum DeepSeekV41DensePinLoader {

    public struct Result: @unchecked Sendable {
        /// Keyed by the synthetic shard index a pinned tensor now addresses.
        public let buffers: [UInt32: MTLBuffer]
        public let bytesRead: Int
        public let seconds: Double
    }

    public static func load(
        plan: DeepSeekV41DensePinPlan,
        snapshotDir: URL,
        device: MTLDevice
    ) throws -> Result {
        guard !plan.isEmpty else {
            return Result(buffers: [:], bytesRead: 0, seconds: 0)
        }
        var chunks: [MTLBuffer] = []
        chunks.reserveCapacity(plan.chunkLengths.count)
        for length in plan.chunkLengths {
            // A zero-length Metal buffer is an error, and a chunk is never empty
            // once a range was placed in it; the floor is for a pathological
            // one-byte tensor.
            guard let buffer = device.makeBuffer(
                length: max(16, length),
                options: .storageModeShared
            ) else {
                throw DeepSeekV41DensePinError.cannotAllocate(bytes: length)
            }
            chunks.append(buffer)
        }

        var byFile: [String: [DeepSeekV41ExpertShardReader.Range]] = [:]
        for range in plan.ranges {
            guard range.chunk < chunks.count else {
                throw DeepSeekV41DensePinError.cannotAllocate(bytes: range.chunk)
            }
            byFile[range.filename, default: []].append(
                DeepSeekV41ExpertShardReader.Range(
                    fileOffset: range.fileOffset,
                    destination: chunks[range.chunk].contents().advanced(by: range.chunkOffset),
                    length: range.length
                )
            )
        }

        var readers: [String: DeepSeekV41ExpertShardReader] = [:]
        do {
            for filename in byFile.keys {
                readers[filename] = try DeepSeekV41ExpertShardReader(
                    url: snapshotDir.appendingPathComponent(filename)
                )
            }
        } catch {
            for reader in readers.values { reader.closeNow() }
            throw DeepSeekV41DensePinError.cannotOpen("\(error)")
        }

        let failures = DeepSeekV41DensePinFailureBox()
        let files = byFile.keys.sorted()
        let context = DeepSeekV41PinReadContext(
            files: files,
            ranges: byFile,
            readers: readers,
            failures: failures
        )
        let start = CFAbsoluteTimeGetCurrent()
        if files.count >= 2 {
            DispatchQueue.concurrentPerform(iterations: files.count) { context.run($0) }
        } else {
            for index in files.indices { context.run(index) }
        }
        let seconds = CFAbsoluteTimeGetCurrent() - start
        for reader in readers.values { reader.closeNow() }
        if let first = failures.first { throw first }

        var buffers: [UInt32: MTLBuffer] = [:]
        for (index, buffer) in chunks.enumerated() {
            buffers[plan.pinnedShardIndex(chunk: index)] = buffer
        }
        return Result(buffers: buffers, bytesRead: plan.totalBytes, seconds: seconds)
    }
}

/// Every file's ranges plus what the concurrent reads need to report a failure.
/// Ranges in different files write disjoint destinations, so the files can run at
/// the same time.
nonisolated private struct DeepSeekV41PinReadContext: @unchecked Sendable {
    let files: [String]
    let ranges: [String: [DeepSeekV41ExpertShardReader.Range]]
    let readers: [String: DeepSeekV41ExpertShardReader]
    let failures: DeepSeekV41DensePinFailureBox

    func run(_ index: Int) {
        let file = files[index]
        guard let reader = readers[file] else {
            failures.record(.cannotOpen(file))
            return
        }
        do {
            try reader.read(ranges[file] ?? [])
        } catch {
            failures.record(.shortRead("\(file): \(error)"))
        }
    }
}

nonisolated private final class DeepSeekV41DensePinFailureBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: DeepSeekV41DensePinError?

    func record(_ error: DeepSeekV41DensePinError) {
        lock.lock()
        if stored == nil { stored = error }
        lock.unlock()
    }

    var first: DeepSeekV41DensePinError? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}
