//
//  Inkling2SnapshotWeights.swift
//  DynaMoE
//
//  The host forward's weight source over a real snapshot.
//
//  Every tensor is read from the shard mappings the engine already has, by
//  name, through the catalog's roles. Nothing here allocates a second copy of
//  the model: a dense vector is decoded on demand, and an expert bank is handed
//  back as raw stored bytes so the packed matrix does the dequantization per
//  row (a whole NVFP4 layer dequantized eagerly would be 2.7 GB of floats).
//
//  Two layouts are decoded here, both from the STORED shapes:
//
//    * BF16 / F32: widened value by value. BF16 is the top half of a Float32,
//      so the widen is a bit shift and not a conversion.
//    * E4M3 (the MXFP8 experts' values, and the NVFP4 experts' block scales):
//      decoded with `Inkling2Element.decodeE4M3`.
//
//  The expert banks are published as one fused tensor per layer and the Rust
//  summary splits them per expert; `Inkling2WeightCatalog` coalesces them back,
//  so a bank's byte range here is the whole `[128, rows, columns]` tensor and
//  expert `e` starts at `e * rows * columns * elementWidth`.
//

import Foundation
import Metal

nonisolated public final class Inkling2SnapshotWeights: Inkling2WeightSource {
    public let config: Inkling2Config
    public let catalog: Inkling2WeightCatalog
    /// One mapped shard per index, the zero-copy buffers the engine already
    /// holds. `MTLBuffer` is only used for its `contents`/`length` view, so
    /// this file needs no device of its own.
    private let shardBuffers: [UInt32: MTLBuffer]
    /// Decoded control matrices, keyed by role and layer.
    ///
    /// A visit asks for the same message-gate and convolution-tap matrices on
    /// every position, and decoding one copies the tensor out of the shard and
    /// widens it -- work that was being redone dozens of times per token. Only
    /// tensors under `matrixCacheByteLimit` are kept: this path also serves the
    /// fused expert banks and the unembedding, and caching those would hold
    /// gigabytes for a single read.
    private let matrixCacheLock = NSLock()
    private var matrixCache: [String: [Float]] = [:]
    /// What one cached matrix may cost. The largest control matrix is an
    /// `attn_res` query block (hidden x 3 * hidden, ~64 MB at the published
    /// width), so this admits every control tensor and excludes the banks.
    public static let matrixCacheByteLimit = 96 * 1_024 * 1_024

    public init(
        config: Inkling2Config,
        catalog: Inkling2WeightCatalog,
        shardBuffers: [UInt32: MTLBuffer]
    ) {
        self.config = config
        self.catalog = catalog
        self.shardBuffers = shardBuffers
    }

    // MARK: Locating

    private func tensor(_ role: Inkling2TensorRole, layer: Int?) throws -> Inkling2TensorRef {
        guard let ref = catalog.tensor(role, layer: layer) else {
            throw Inkling2ForwardError.missingWeight(
                "\(role.rawValue)\(layer.map { " layer \($0)" } ?? "")"
            )
        }
        return ref
    }

    /// The bytes of one tensor, as a pointer into its shard mapping.
    private func bytes(of ref: Inkling2TensorRef) throws -> UnsafeRawPointer {
        guard let shard = ref.shardIndex, let buffer = shardBuffers[shard] else {
            throw Inkling2ForwardError.missingWeight("\(ref.name): its shard is not mapped")
        }
        guard let offset = ref.byteOffset else {
            throw Inkling2ForwardError.missingWeight("\(ref.name): no byte offset")
        }
        let width = Self.elementWidth(ref.dtype)
        let length = ref.elementCount * width
        guard Int(offset) + length <= buffer.length else {
            throw Inkling2ForwardError.shapeMismatch(
                "\(ref.name) needs \(length) bytes at \(offset) of a \(buffer.length)-byte shard"
            )
        }
        return UnsafeRawPointer(buffer.contents().advanced(by: Int(offset)))
    }

    static func elementWidth(_ dtype: String) -> Int {
        let upper = dtype.uppercased()
        if upper == "U8" || upper.contains("F8") { return 1 }
        if upper.contains("BF16") || upper.contains("F16") { return 2 }
        if upper.contains("F32") { return 4 }
        return 2
    }

    private static func rawValues(_ pointer: UnsafeRawPointer, count: Int, dtype: String) -> [UInt8] {
        let width = elementWidth(dtype)
        let buffer = UnsafeRawBufferPointer(start: pointer, count: count * width)
        return Array(buffer)
    }

    /// Decodes stored elements to Float. BF16 is a bit shift, F32 a load, E4M3
    /// the format table; U8 is returned as its raw value (scales).
    static func decode(_ raw: [UInt8], dtype: String, count: Int) -> [Float] {
        let upper = dtype.uppercased()
        if upper.contains("BF16") {
            var output = [Float](repeating: 0, count: count)
            raw.withUnsafeBytes { source in
                for index in 0..<count {
                    let bits = source.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    output[index] = Float(bitPattern: UInt32(bits) << 16)
                }
            }
            return output
        }
        if upper.contains("F32") {
            var output = [Float](repeating: 0, count: count)
            raw.withUnsafeBytes { source in
                for index in 0..<count {
                    output[index] = source.loadUnaligned(fromByteOffset: index * 4, as: Float.self)
                }
            }
            return output
        }
        if upper.contains("F8_E4M3") {
            return raw.prefix(count).map { Inkling2Element.decodeE4M3($0) }
        }
        if upper.contains("F16") {
            var output = [Float](repeating: 0, count: count)
            raw.withUnsafeBytes { source in
                for index in 0..<count {
                    let bits = source.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    output[index] = Float(Float16(bitPattern: bits))
                }
            }
            return output
        }
        // U8 scales and packed values stay as their code.
        return raw.prefix(count).map { Float($0) }
    }

    /// The same decode, straight from the mapping into a caller's buffer.
    ///
    /// Same values, same order; what it does not do is build an intermediate
    /// `[UInt8]` copy of the tensor, which is what made the whole-table rule
    /// 2.8 GB of traffic for a table that is the largest in the checkpoint.
    static func decode(
        _ pointer: UnsafeRawPointer, dtype: String, count: Int, into output: inout [Float]
    ) {
        guard count > 0, output.count >= count else { return }
        let upper = dtype.uppercased()
        output.withUnsafeMutableBufferPointer { destination in
            if upper.contains("BF16") {
                for index in 0..<count {
                    let bits = pointer.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    destination[index] = Float(bitPattern: UInt32(bits) << 16)
                }
            } else if upper.contains("F32") {
                for index in 0..<count {
                    destination[index] = pointer.loadUnaligned(fromByteOffset: index * 4, as: Float.self)
                }
            } else if upper.contains("F8_E4M3") {
                for index in 0..<count {
                    destination[index] = Inkling2Element.decodeE4M3(
                        pointer.loadUnaligned(fromByteOffset: index, as: UInt8.self)
                    )
                }
            } else if upper.contains("F16") {
                for index in 0..<count {
                    let bits = pointer.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                    destination[index] = Float(Float16(bitPattern: bits))
                }
            } else {
                for index in 0..<count {
                    destination[index] = Float(pointer.loadUnaligned(fromByteOffset: index, as: UInt8.self))
                }
            }
        }
    }

    // MARK: Inkling2WeightSource

    public func shape(_ role: Inkling2TensorRole, layer: Int?) throws -> [Int] {
        try tensor(role, layer: layer).shape
    }

    public func matrix(_ role: Inkling2TensorRole, layer: Int?) throws -> [Float] {
        let ref = try tensor(role, layer: layer)
        let key = "\(role.rawValue)|\(layer.map(String.init) ?? "-")"
        // A tensor too large to keep is decoded and handed back uncached.
        let cacheable = ref.elementCount * MemoryLayout<Float>.stride <= Self.matrixCacheByteLimit
        if cacheable {
            matrixCacheLock.lock()
            let cached = matrixCache[key]
            matrixCacheLock.unlock()
            if let cached { return cached }
        }
        let pointer = try bytes(of: ref)
        let raw = Self.rawValues(pointer, count: ref.elementCount, dtype: ref.dtype)
        let decoded = Self.decode(raw, dtype: ref.dtype, count: ref.elementCount)
        if cacheable {
            matrixCacheLock.lock()
            matrixCache[key] = decoded
            matrixCacheLock.unlock()
        }
        return decoded
    }

    /// How many matrices the cache holds, and what they cost. For the tests
    /// that have to see a repeat read served from memory rather than a shard.
    public var cachedMatrixCount: Int {
        matrixCacheLock.lock()
        defer { matrixCacheLock.unlock() }
        return matrixCache.count
    }

    /// The unembedding's logits, decoded a CHUNK of rows at a time.
    ///
    /// The base rule decodes the whole tensor first, and this checkpoint's
    /// unembedding is 200,058 x 2,304 BF16: 922 MB stored, 1.84 GB as `[Float]`,
    /// re-decoded on every token. A turn therefore held a transient allocation
    /// larger than the model's dense text weights, on a 16 GB machine. Here the
    /// same arithmetic runs over `headChunkRows` rows at a time, so the peak is
    /// a chunk rather than the table while the dot itself stays the reference's
    /// column-ordered FP32 sum (chunking must not move the pinned host numbers).
    ///
    /// `headChunkRows` is 1,024: 9.4 MB of scratch, 196 passes over the table.
    public static let headChunkRows = 1024

    public func logits(_ hidden: [Float], rows: Int, hiddenSize: Int) throws -> [Float] {
        guard hidden.count == hiddenSize else {
            throw Inkling2ForwardError.shapeMismatch(
                "the head wants a \(hiddenSize)-wide hidden row, got \(hidden.count)"
            )
        }
        let ref = try tensor(.lmHead, layer: nil)
        guard ref.shape.count == 2, ref.shape[1] == hiddenSize else {
            throw Inkling2ForwardError.shapeMismatch(
                "the unembedding is \(ref.shape), not [rows][\(hiddenSize)]"
            )
        }
        guard rows >= 0, rows <= ref.shape[0] else {
            throw Inkling2ForwardError.shapeMismatch(
                "\(rows) logits do not fit the unembedding \(ref.shape)"
            )
        }
        return try Self.headLogits(
            from: try bytes(of: ref), dtype: ref.dtype, rows: rows,
            hiddenSize: hiddenSize, hidden: hidden
        )
    }

    /// The chunked rule itself, as a function of the mapped bytes.
    ///
    /// Split out so the chunk arithmetic is testable on a byte buffer rather
    /// than only through a checkpoint: `chunkRows` is what a test varies, and
    /// the whole point is that it changes the peak allocation and NOT the sums.
    static func headLogits(
        from pointer: UnsafeRawPointer,
        dtype: String,
        rows: Int,
        hiddenSize: Int,
        hidden: [Float],
        chunkRows: Int = Inkling2SnapshotWeights.headChunkRows
    ) throws -> [Float] {
        guard hidden.count == hiddenSize else {
            throw Inkling2ForwardError.shapeMismatch(
                "the head wants a \(hiddenSize)-wide hidden row, got \(hidden.count)"
            )
        }
        guard rows > 0 else { return [] }
        guard chunkRows > 0 else {
            throw Inkling2ForwardError.unsupported("a head chunk of \(chunkRows) rows")
        }
        let width = elementWidth(dtype)
        var logits = [Float](repeating: 0, count: rows)
        let chunk = min(rows, chunkRows)
        var scratch = [Float](repeating: 0, count: chunk * hiddenSize)
        var start = 0
        while start < rows {
            let count = min(chunk, rows - start)
            let elements = count * hiddenSize
            decode(
                pointer.advanced(by: start * hiddenSize * width),
                dtype: dtype, count: elements, into: &scratch
            )
            for row in 0..<count {
                var total: Float = 0
                let offset = row * hiddenSize
                for column in 0..<hiddenSize {
                    total += scratch[offset + column] * hidden[column]
                }
                logits[start + row] = total
            }
            start += count
        }
        return logits
    }


    /// One row, read without materializing the table: the embedding is 201,024
    /// rows of 2,304 and a token needs one.
    public func row(_ role: Inkling2TensorRole, layer: Int?, index: Int) throws -> [Float] {
        let ref = try tensor(role, layer: layer)
        guard ref.shape.count == 2, index >= 0, index < ref.shape[0] else {
            throw Inkling2ForwardError.shapeMismatch("row \(index) of \(ref.shape) is out of range")
        }
        let width = ref.shape[1]
        let elementWidth = Self.elementWidth(ref.dtype)
        let pointer = try bytes(of: ref).advanced(by: index * width * elementWidth)
        let raw = Self.rawValues(pointer, count: width, dtype: ref.dtype)
        return Self.decode(raw, dtype: ref.dtype, count: width)
    }

    /// One expert's bank from a fused tensor.
    ///
    /// `w13` is `[experts][2*intermediate][packedInput]` for NVFP4 and
    /// `[experts][2*intermediate][input]` for MXFP8; `w2` is
    /// `[experts][hidden][packedIntermediate]`. The expert's slice is one
    /// contiguous run, so it is addressed, not copied.
    /// One expert's slice of a fused bank: the catalog holds the bank whole and
    /// the forward reads one expert at a time.
    public func expertMatrix(_ role: Inkling2TensorRole, layer: Int, expert: Int) throws -> Inkling2PackedExpertMatrix {
        let ref = try tensor(role, layer: layer)
        guard ref.shape.count == 3, expert >= 0, expert < ref.shape[0] else {
            throw Inkling2ForwardError.shapeMismatch("expert \(expert) of \(ref.shape)")
        }
        let rows = ref.shape[1]
        let columns = ref.shape[2]
        let width = Self.elementWidth(ref.dtype)
        let expertStride = rows * columns * width
        let pointer = try bytes(of: ref).advanced(by: expert * expertStride)

        let scaleRef = try tensor(
            role == .expertW13 ? .expertW13Scale : .expertW2Scale, layer: layer
        )
        let scaleWidth = Self.elementWidth(scaleRef.dtype)
        let scaleRows = scaleRef.shape[1]
        let scaleColumns = scaleRef.shape[2]
        let scaleStride = scaleRows * scaleColumns * scaleWidth
        let scalePointer = try bytes(of: scaleRef).advanced(by: expert * scaleStride)

        let layout = try Inkling2ExpertQuantLayout.infer(
            w13: role == .expertW13 ? ref : try tensor(.expertW13, layer: layer),
            w13Scale: role == .expertW13 ? scaleRef : try tensor(.expertW13Scale, layer: layer),
            w2: role == .expertW2 ? ref : try tensor(.expertW2, layer: layer),
            w2Scale: role == .expertW2 ? scaleRef : try tensor(.expertW2Scale, layer: layer),
            config: config
        ).get()

        var amax: Float = 1
        var denominator: Float = 1
        if layout == .nvfp4 {
            let amaxRef = try tensor(role == .expertW13 ? .expertW13Amax : .expertW2Amax, layer: layer)
            let amaxValues = Self.decode(
                Self.rawValues(try bytes(of: amaxRef), count: amaxRef.elementCount, dtype: amaxRef.dtype),
                dtype: amaxRef.dtype, count: amaxRef.elementCount
            )
            amax = amaxValues[expert]
            let denominatorRef = try tensor(
                role == .expertW13 ? .expertW13OuterScaleDenominator : .expertW2OuterScaleDenominator,
                layer: layer
            )
            denominator = Self.decode(
                Self.rawValues(try bytes(of: denominatorRef), count: 1, dtype: denominatorRef.dtype),
                dtype: denominatorRef.dtype, count: 1
            )[0]
        }

        // The logical input width: w13 is 2 * intermediate over the hidden, w2 is
        // hidden over the intermediate. The stored stride is the packed one.
        let logicalColumns: Int
        if role == .expertW13 {
            logicalColumns = config.hiddenSize
        } else {
            logicalColumns = config.moeIntermediateSize
        }
        return Inkling2PackedExpertMatrix(
            layout: layout,
            weight: Self.rawValues(pointer, count: rows * columns, dtype: ref.dtype),
            weightRowStride: columns,
            scales: Self.rawValues(scalePointer, count: scaleRows * scaleColumns, dtype: scaleRef.dtype),
            scaleRowStride: scaleColumns,
            rows: rows,
            logicalColumns: logicalColumns,
            expertAmax: amax,
            outerScaleDenominator: denominator
        )
    }
}
