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

    // MARK: Inkling2WeightSource

    public func shape(_ role: Inkling2TensorRole, layer: Int?) throws -> [Int] {
        try tensor(role, layer: layer).shape
    }

    public func matrix(_ role: Inkling2TensorRole, layer: Int?) throws -> [Float] {
        let ref = try tensor(role, layer: layer)
        let pointer = try bytes(of: ref)
        let raw = Self.rawValues(pointer, count: ref.elementCount, dtype: ref.dtype)
        return Self.decode(raw, dtype: ref.dtype, count: ref.elementCount)
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
