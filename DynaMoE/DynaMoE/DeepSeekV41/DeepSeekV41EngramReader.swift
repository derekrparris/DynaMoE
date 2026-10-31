//
//  DeepSeekV41EngramReader.swift
//  DynaMoE
//
//  Row-granular streaming for the Engram tables.
//
//  Each table is ~100 GB in ONE safetensors tensor, which is far past any Apple
//  GPU's `maxBufferLength` (the segmentation step in `InferenceEngine` refuses
//  such a tensor). So the table is never mapped and never becomes an
//  `MTLBuffer`: a token needs 24 rows of 264 bytes per Engram layer, and those
//  rows are pread straight into a staging buffer that the GPU dequantizes.
//
//  The read pattern is the reason the API takes a batch: the hash ids for a
//  position are known before the layer runs, so all of a layer's rows are
//  issued together and can be serviced with real queue depth. A serial loop of
//  24 small random reads per token would otherwise spend the whole decode step
//  waiting on seek latency.
//

import Foundation
#if canImport(Darwin)
import Darwin
#endif

nonisolated public enum DeepSeekV41EngramReadError: Error, Equatable, CustomStringConvertible {
    case cannotOpen(String)
    case tableTooSmall(expected: Int, actual: Int)
    case rowOutOfRange(offset: Int, length: Int, fileSize: Int)
    case shortRead(offset: Int, wanted: Int, got: Int)
    case layerMismatch(request: Int, table: Int)

    public var description: String {
        switch self {
        case .cannotOpen(let detail):
            return "cannot open Engram table: \(detail)"
        case .tableTooSmall(let expected, let actual):
            return "Engram table is \(actual) bytes, needs at least \(expected) for its row count"
        case .rowOutOfRange(let offset, let length, let fileSize):
            return "Engram row read [\(offset), \(offset + length)) is outside the \(fileSize)-byte table"
        case .shortRead(let offset, let wanted, let got):
            return "Engram row read at \(offset) returned \(got) of \(wanted) bytes"
        case .layerMismatch(let request, let table):
            return "Engram row request for layer \(request) was issued to layer \(table)'s table"
        }
    }
}

/// Values the parallel row reads need. The pointer is only written at
/// disjoint offsets, so sharing it across the concurrent reads is safe.
nonisolated private struct ReadContext: @unchecked Sendable {
    let fd: Int32
    /// Byte offset of the table's VALUES block inside the shard file. Requests
    /// carry interleaved table-relative offsets, so a split table turns one into a
    /// row index before addressing `row * head_dimension`.
    let valuesOffset: Int
    let rowBytes: Int
    /// Value bytes per row (`head_dimension`). The staged row is values first.
    let valueBytes: Int
    /// File offset of the scale tensor's first byte, or nil when the table keeps
    /// each row's scales inline.
    let scalesOffset: Int?
    let scaleBytes: Int
    let requests: [DeepSeekV41EngramRowRequest]
    let destination: UnsafeMutableRawPointer
    let failures: FailureBox

    func read(_ index: Int) {
        let request = requests[index]
        let slice = destination.advanced(by: index * rowBytes)
        let row = request.byteOffset / rowBytes
        // The checkpoint publishes a row's values and its E8M0 scales as TWO
        // tensors, so a row is two spans that happen to be `rowCount * valueBytes`
        // apart. The staged row is still values-then-scales, which is what the
        // dequant kernel reads.
        guard let scalesOffset else {
            readSpan(into: slice, from: valuesOffset + request.byteOffset, length: valueBytes)
            return
        }
        readSpan(into: slice, from: valuesOffset + row * valueBytes, length: valueBytes)
        readSpan(
            into: slice.advanced(by: valueBytes),
            from: scalesOffset + row * scaleBytes,
            length: scaleBytes
        )
    }

    private func readSpan(into destination: UnsafeMutableRawPointer, from: Int, length: Int) {
        guard length > 0 else { return }
        var total = 0
        while total < length {
            let got = pread(
                fd,
                destination.advanced(by: total),
                length - total,
                off_t(from + total)
            )
            if got < 0 {
                if errno == EINTR { continue }
                failures.record(.shortRead(offset: from + total, wanted: length - total, got: Int(got)))
                return
            }
            if got == 0 {
                failures.record(.shortRead(offset: from + total, wanted: length - total, got: 0))
                return
            }
            total += got
        }
    }
}

/// First-error slot for the parallel row reads.
nonisolated private final class FailureBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: DeepSeekV41EngramReadError?

    func record(_ error: DeepSeekV41EngramReadError) {
        lock.lock()
        if stored == nil { stored = error }
        lock.unlock()
    }

    var first: DeepSeekV41EngramReadError? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

/// One table's reader. Synchronous and thread-safe: one batch at a time holds the
/// descriptor, so `closeNow()` cannot recycle the fd under an in-flight `pread`.
nonisolated public final class DeepSeekV41EngramReader: @unchecked Sendable {
    public let spec: DeepSeekV41EngramSpec
    public let fileSize: Int
    /// Byte offset of the table tensor inside the shard file.
    public let valuesOffset: Int
    /// File offset of the table's scale tensor, or nil when each row's E8M0
    /// scales follow its values in place.
    ///
    /// The published checkpoint splits the two: `engram.embed.weight` is
    /// `[rows, head_dim]` of E4M3 and `engram.embed.scale` is `[rows, head_dim/32]`
    /// of E8M0, and the two blocks sit megabytes to gigabytes apart in the shard.
    /// Reading `head_dim + head_dim/32` bytes at `row * rowBytes` therefore takes
    /// the NEXT row's values as this row's exponents, which produces finite,
    /// plausible-looking garbage rather than a failure.
    public let scalesOffset: Int?

    private let descriptor: Int32
    private let lock = NSLock()
    private var isClosed = false

    /// Value bytes in a staged row: values first, then the block scales.
    public var valueBytes: Int { spec.headDimension }
    /// E8M0 scales in a staged row.
    public var scaleBytes: Int { spec.rowBytes - spec.headDimension }

    /// Opens the table file and validates it against the spec, so a truncated
    /// or half-downloaded table fails at load rather than mid-generation.
    public init(
        fileURL: URL,
        spec: DeepSeekV41EngramSpec,
        valuesOffset: Int = 0,
        scalesOffset: Int? = nil
    ) throws {
        guard valuesOffset >= 0, (scalesOffset ?? 0) >= 0 else {
            throw DeepSeekV41EngramReadError.cannotOpen(
                "\(fileURL.lastPathComponent): negative table offset "
                    + "\(valuesOffset)/\(scalesOffset.map(String.init) ?? "-")"
            )
        }
        let fd = open(fileURL.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw DeepSeekV41EngramReadError.cannotOpen(
                "\(fileURL.lastPathComponent): errno \(errno)"
            )
        }
        var status = stat()
        guard fstat(fd, &status) == 0 else {
            close(fd)
            throw DeepSeekV41EngramReadError.cannotOpen(
                "\(fileURL.lastPathComponent): fstat failed, errno \(errno)"
            )
        }
        let size = Int(status.st_size)
        // The table starts at the tensor's own offset inside the shard; only the
        // bytes after that offset belong to it. A split table is checked as two
        // spans, because each one is addressed on its own.
        let valueSpan = spec.rowCount * spec.headDimension
        let scaleSpan = spec.rowCount * (spec.rowBytes - spec.headDimension)
        let valuesFit = valuesOffset <= size && size - valuesOffset >= valueSpan
        let scalesFit: Bool
        if let scalesOffset {
            scalesFit = scalesOffset <= size && size - scalesOffset >= scaleSpan
        } else {
            scalesFit = valuesOffset <= size && size - valuesOffset >= spec.totalBytes
        }
        guard valuesFit, scalesFit else {
            close(fd)
            throw DeepSeekV41EngramReadError.tableTooSmall(
                expected: valuesOffset + spec.totalBytes, actual: size
            )
        }
        self.descriptor = fd
        self.spec = spec
        self.fileSize = size
        self.valuesOffset = valuesOffset
        self.scalesOffset = scalesOffset
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

    /// Reads one batch of rows into `destination`, laid out densely in request
    /// order (`request[i]` at `i * spec.rowBytes`). Returns the bytes read.
    ///
    /// Reads are issued in parallel once there are enough of them for queue
    /// depth to matter; each request writes a disjoint destination slice, so
    /// the concurrent writes are safe.
    ///
    /// The lock is held for the whole batch rather than just around the
    /// `isClosed` check: `closeNow()` closes the descriptor, and a descriptor
    /// number recycled by an unrelated `open()` would turn a concurrent `pread`
    /// into a silent read of another file's bytes. Holding the lock makes close
    /// wait for the in-flight rows instead.
    @discardableResult
    public func readRows(
        _ requests: [DeepSeekV41EngramRowRequest],
        into destination: UnsafeMutableRawPointer
    ) throws -> Int {
        guard !requests.isEmpty else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else {
            throw DeepSeekV41EngramReadError.cannotOpen("table reader was closed")
        }
        let fd = descriptor

        let rowBytes = spec.rowBytes
        for request in requests {
            guard request.layer == spec.layer else {
                throw DeepSeekV41EngramReadError.layerMismatch(
                    request: request.layer, table: spec.layer
                )
            }
            guard request.byteCount == rowBytes else {
                throw DeepSeekV41EngramReadError.rowOutOfRange(
                    offset: request.byteOffset,
                    length: request.byteCount,
                    fileSize: fileSize
                )
            }
            guard request.byteOffset >= 0,
                  valuesOffset + request.byteOffset + valueBytes <= fileSize else {
                throw DeepSeekV41EngramReadError.rowOutOfRange(
                    offset: valuesOffset + request.byteOffset,
                    length: valueBytes,
                    fileSize: fileSize
                )
            }
            if let scalesOffset {
                let scaleStart = scalesOffset + (request.byteOffset / rowBytes) * scaleBytes
                guard scaleStart + scaleBytes <= fileSize else {
                    throw DeepSeekV41EngramReadError.rowOutOfRange(
                        offset: scaleStart,
                        length: scaleBytes,
                        fileSize: fileSize
                    )
                }
            }
        }

        // Concurrent rows must not race on the first-error slot, and the raw
        // destination pointer has to cross the concurrency boundary explicitly.
        let failures = FailureBox()
        let shared = ReadContext(
            fd: fd,
            valuesOffset: valuesOffset,
            rowBytes: rowBytes,
            valueBytes: valueBytes,
            scalesOffset: scalesOffset,
            scaleBytes: scaleBytes,
            requests: requests,
            destination: destination,
            failures: failures
        )

        let readOne: @Sendable (Int) -> Void = { index in
            shared.read(index)
        }

        if requests.count >= 8 {
            DispatchQueue.concurrentPerform(iterations: requests.count, execute: readOne)
        } else {
            for index in requests.indices { readOne(index) }
        }

        if let firstError = failures.first { throw firstError }
        return requests.count * rowBytes
    }
}

// MARK: - Locating a table's file

/// Finds the shard FILE that holds one Engram table.
///
/// The segmented `ModelSummary` cannot answer this. A table is streamed, so
/// `segmentShardsForMetal` leaves the table's `shardIndex` pointing into the
/// original whole-file numbering while the summary's own shard list has been
/// renumbered, and a shard that held nothing but streamed tensors produces no
/// segment and is dropped from that list entirely. The name is resolved from the
/// checkpoint's own manifest instead, and falls back to reading the shards'
/// headers, so opening a table never depends on how the shards were mapped.
nonisolated public enum DeepSeekV41EngramTableLocator {

    /// The shard filename, relative to `snapshotDir`, that publishes `tensor`.
    public static func shardFilename(
        forTensorNamed tensor: String,
        in snapshotDir: URL
    ) -> String? {
        if let listed = indexWeightMap(in: snapshotDir)[tensor], !listed.isEmpty {
            return listed
        }
        let shards = shardFiles(in: snapshotDir)
        if shards.count == 1 { return shards[0] }
        for shard in shards where headerNames(
            in: snapshotDir.appendingPathComponent(shard)
        ).contains(tensor) {
            return shard
        }
        return nil
    }

    /// `tensor name -> shard filename` from the directory's `*.index.json`.
    static func indexWeightMap(in snapshotDir: URL) -> [String: String] {
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: snapshotDir.path)) ?? []
        for name in contents.sorted() where name.hasSuffix(".index.json") {
            let url = snapshotDir.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url),
                  let object = try? JSONSerialization.jsonObject(with: data),
                  let root = object as? [String: Any],
                  let map = root["weight_map"] as? [String: String] else { continue }
            return map
        }
        return [:]
    }

    static func shardFiles(in snapshotDir: URL) -> [String] {
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: snapshotDir.path)) ?? []
        return contents.filter { $0.hasSuffix(".safetensors") }.sorted()
    }

    /// The tensor names a safetensors file's header declares. Only the length
    /// prefix and the JSON header are read, never the tensor data.
    static func headerNames(in url: URL) -> Set<String> {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8),
              lengthData.count == 8 else { return [] }
        let length = lengthData.withUnsafeBytes { $0.load(as: UInt64.self) }
        // Far past any real header (the published checkpoint's is under 1 KB).
        guard length > 0, length < 64 << 20 else { return [] }
        var header = Data()
        while header.count < Int(length) {
            let remaining = Int(length) - header.count
            let piece: Data?
            do {
                piece = try handle.read(upToCount: remaining)
            } catch {
                return []
            }
            guard let piece, !piece.isEmpty else { return [] }
            header.append(piece)
        }
        guard let object = try? JSONSerialization.jsonObject(with: header),
              let root = object as? [String: Any] else { return [] }
        return Set(root.keys.filter { $0 != "__metadata__" })
    }
}

// MARK: - Lookup

/// Builds the compressed token map both backends hash with.
nonisolated public extension DeepSeekV41EngramTokenMap {

    /// Reads the checkpoint's tokenizer and compiles the compressed vocabulary the
    /// n-grams hash over.
    ///
    /// This is load work rather than per-turn work: the map is one entry per
    /// vocabulary token and it decides every row the Engram layers read.
    static func make(
        snapshotDir: URL,
        config: DeepSeekV41Config
    ) throws -> DeepSeekV41EngramTokenMap {
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshotDir.appendingPathComponent("tokenizer.json").path
        )
        let compiled = try tokenizer.buildEngramTokenMap(
            expectedCompressedVocabSize: UInt32(clamping: config.engramCompressedVocabSize),
            padTokenId: UInt32(clamping: config.engramPadTokenID)
        )
        return try DeepSeekV41EngramTokenMap(
            lookup: compiled.lookup,
            compressedVocabSize: Int(compiled.compressedVocabSize),
            padID: compiled.padId
        )
    }
}

/// Where the forward gets one position's Engram rows.
///
/// The forward owns the token history and therefore the hash inputs; a lookup only
/// turns "this layer, this position's history" into rows. The streaming
/// implementation below reads them off disk, and the in-memory weight store answers
/// from rows a test pinned, which is why the parameter is a protocol rather than a
/// reader.
nonisolated public protocol DeepSeekV41EngramLookup: AnyObject {
    /// One row of `engram_head_dim` values per hash column, in column order, or nil
    /// when `layer` has no table. The array MAY be shorter than the layer's column
    /// count, and a missing column is zero-filled by the caller: the projection
    /// reads a positional concatenation, so a gap has to occupy its slot.
    func engramRows(layer: Int, history: [Int32], dead: [Bool]) throws -> [[Float]]?
}

/// Resolves Engram rows for the reference forward by preading the rows the hashes
/// address and dequantizing them.
///
/// 24 rows of 264 bytes per Engram layer per token is ~6 KB, which is why the read
/// is a batch: the whole layer's rows go out at once and the reader services them
/// with queue depth instead of one seek at a time. A reference forward is not the
/// hot path, so the batch buffer is allocated per call rather than pooled.
nonisolated public final class DeepSeekV41EngramStreamingLookup: DeepSeekV41EngramLookup {

    public let readers: [Int: DeepSeekV41EngramReader]
    public let layouts: [Int: DeepSeekV41EngramLayout]
    public let tokenMap: DeepSeekV41EngramTokenMap

    public init(
        readers: [Int: DeepSeekV41EngramReader],
        layouts: [Int: DeepSeekV41EngramLayout],
        tokenMap: DeepSeekV41EngramTokenMap
    ) {
        self.readers = readers
        self.layouts = layouts
        self.tokenMap = tokenMap
    }

    /// Builds the compressed token map from the checkpoint's own tokenizer, so the
    /// n-grams hash the vocabulary the model trained on.
    ///
    /// Throws rather than falling back to a stand-in map: the compressed vocabulary
    /// size feeds the hash multipliers, so a wrong map reads the wrong rows and
    /// still produces finite output.
    public static func make(
        snapshotDir: URL,
        config: DeepSeekV41Config,
        readers: [Int: DeepSeekV41EngramReader],
        layouts: [Int: DeepSeekV41EngramLayout]
    ) throws -> DeepSeekV41EngramStreamingLookup {
        let tokenMap = try DeepSeekV41EngramTokenMap.make(
            snapshotDir: snapshotDir, config: config
        )
        return DeepSeekV41EngramStreamingLookup(
            readers: readers, layouts: layouts, tokenMap: tokenMap
        )
    }

    public func engramRows(
        layer: Int,
        history: [Int32],
        dead: [Bool]
    ) throws -> [[Float]]? {
        guard let reader = readers[layer], let layout = layouts[layer] else { return nil }
        let requests = DeepSeekV41Engram.rowRequests(
            layer: layer,
            history: history,
            dead: dead,
            tokenMap: tokenMap,
            layout: layout
        )
        guard !requests.isEmpty else { return nil }
        let rowBytes = layout.spec.rowBytes
        var raw = [UInt8](repeating: 0, count: requests.count * rowBytes)
        try raw.withUnsafeMutableBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            try reader.readRows(requests, into: base)
        }
        let headDimension = layout.spec.headDimension
        return requests.indices.map { index in
            let start = index * rowBytes
            return DeepSeekV41EngramRowDequantizer.row(
                Array(raw[start..<(start + rowBytes)]),
                headDimension: headDimension
            )
        }
    }
}
