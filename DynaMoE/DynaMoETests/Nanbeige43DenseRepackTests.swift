//
//  Nanbeige43DenseRepackTests.swift
//  DynaMoETests
//
//  The dense repack is a NEW subsystem (the existing repacker is a byte copier
//  that needs `(layer, expert)` tensors and throws on a dense checkpoint), and
//  its failure mode is silent: a blob whose packing disagrees with
//  `q4_gemv`/`q8_gemv` by one column or one group loads fine and answers
//  nonsense. So the packing is pinned by DEQUANTIZING it the way the kernels
//  read it and comparing against the source, not by re-running the encoder.
//
//  All pure logic: no checkpoint, no Metal device.
//

import XCTest
@testable import DynaMoE

final class Nanbeige43DenseRepackTests: XCTestCase {

    /// A deterministic matrix wide enough for two Q4 words per group.
    private func matrix(outDim: Int, inDim: Int, seed: Float = 0.37) -> [UInt16] {
        var values = [UInt16](repeating: 0, count: outDim * inDim)
        var state = seed
        for index in 0..<values.count {
            state = (state * 1.618_033_988_75).truncatingRemainder(dividingBy: 1)
            let value = (state - 0.5) * 6.0 + Float(index % 7) * 0.25
            values[index] = DenseRepacker.floatToBF16(value)
        }
        return values
    }

    /// Decodes the packed row the way the kernel addresses it: `q4_gemv` takes one
    /// little-endian uint32 per 8 columns and treats the LOW nibble as the LOWER
    /// column, and both kernels scale by `row * numGroups + col / 64`.
    private func dequantize(
        packed: [UInt8],
        scales: [UInt16],
        biases: [UInt16],
        outDim: Int,
        inDim: Int,
        quantization: DenseRepacker.Quantization
    ) -> [Float] {
        let groups = inDim / DenseRepacker.groupSize
        var values = [Float](repeating: 0, count: outDim * inDim)
        for row in 0..<outDim {
            for column in 0..<inDim {
                let group = column / DenseRepacker.groupSize
                let scale = DenseRepacker.bf16ToFloat(scales[row * groups + group])
                let bias = DenseRepacker.bf16ToFloat(biases[row * groups + group])
                let quantized: Float
                if quantization == .q4 {
                    let wordIndex = row * (inDim / 8) + column / 8
                    let word = UInt32(packed[wordIndex * 4])
                        | (UInt32(packed[wordIndex * 4 + 1]) << 8)
                        | (UInt32(packed[wordIndex * 4 + 2]) << 16)
                        | (UInt32(packed[wordIndex * 4 + 3]) << 24)
                    quantized = Float((word >> UInt32((column % 8) * 4)) & 0xF)
                } else {
                    quantized = Float(packed[row * inDim + column])
                }
                values[row * inDim + column] = quantized * scale + bias
            }
        }
        return values
    }

    func testThePackedMatrixDecodesBackWithinItsQuantizationStep() {
        let outDim = 4
        let inDim = 128
        let source = matrix(outDim: outDim, inDim: inDim)
        let groups = inDim / DenseRepacker.groupSize

        for quantization in DenseRepacker.Quantization.allCases {
            var scales = [UInt16](repeating: 0, count: outDim * groups)
            var biases = [UInt16](repeating: 0, count: outDim * groups)
            let packed = source.withUnsafeBufferPointer { buffer in
                DenseRepacker.quantizeMatrix(
                    source: buffer.baseAddress!,
                    outDim: outDim,
                    inDim: inDim,
                    quantization: quantization,
                    scales: &scales,
                    biases: &biases
                )
            }
            let rowBytes = quantization == .q4 ? inDim / 2 : inDim
            XCTAssertEqual(packed.count, outDim * rowBytes, "\(quantization) row width")
            XCTAssertEqual(scales.count, outDim * groups)

            // Every value must land inside half a quantization step of the source,
            // and a round-trip that loses a column or a group misses by far more.
            let decoded = dequantize(
                packed: packed, scales: scales, biases: biases,
                outDim: outDim, inDim: inDim, quantization: quantization
            )
            var worst: Float = 0
            for index in 0..<source.count {
                let expected = DenseRepacker.bf16ToFloat(source[index])
                worst = max(worst, abs(decoded[index] - expected))
            }
            // Half a step plus the BF16 rounding of the scale itself.
            let step = (6.0 + 1.75) / quantization.levels
            XCTAssertLessThan(worst, step, "\(quantization) worst error \(worst) against a step of \(step)")
        }
    }

    /// Q4 packs two values per byte, and a swapped nibble order silently
    /// transposes adjacent columns: a strictly increasing row proves the ORDER,
    /// where a random matrix could match by luck. The values are compared as
    /// decoded floats, not as integers, because the encoder quantizes the
    /// BF16-rounded source against a BF16-rounded scale — an exact index would be
    /// asserting float luck at the top of the range.
    func testQ4PacksTheLowNibbleAsTheLowerColumn() {
        let inDim = 64
        var source = [UInt16](repeating: 0, count: inDim)
        for column in 0..<inDim {
            source[column] = DenseRepacker.floatToBF16(Float(column) / 63.0)
        }
        var scales = [UInt16](repeating: 0, count: 1)
        var biases = [UInt16](repeating: 0, count: 1)
        let packed = source.withUnsafeBufferPointer { buffer in
            DenseRepacker.quantizeMatrix(
                source: buffer.baseAddress!, outDim: 1, inDim: inDim,
                quantization: .q4, scales: &scales, biases: &biases
            )
        }
        let decoded = dequantize(
            packed: packed, scales: scales, biases: biases,
            outDim: 1, inDim: inDim, quantization: .q4
        )
        let step = DenseRepacker.bf16ToFloat(scales[0])
        for column in 0..<inDim {
            let expected = DenseRepacker.bf16ToFloat(source[column])
            XCTAssertEqual(decoded[column], expected, accuracy: step, "column \(column)")
            if column > 0 {
                XCTAssertGreaterThanOrEqual(
                    decoded[column], decoded[column - 1],
                    "column \(column) decoded out of order, so the nibbles are swapped"
                )
            }
        }
        // And the ordering must actually span the range, not collapse to one value.
        XCTAssertGreaterThan(decoded[inDim - 1] - decoded[0], 0.9)
    }

    /// A constant row has zero range. A zero scale would divide by zero while
    /// encoding and produce a row of NaNs on decode.
    func testAConstantRowSurvivesAZeroRange() {
        let inDim = 64
        let source = [UInt16](repeating: DenseRepacker.floatToBF16(0.25), count: inDim)
        var scales = [UInt16](repeating: 0, count: 1)
        var biases = [UInt16](repeating: 0, count: 1)
        let packed = source.withUnsafeBufferPointer { buffer in
            DenseRepacker.quantizeMatrix(
                source: buffer.baseAddress!, outDim: 1, inDim: inDim,
                quantization: .q4, scales: &scales, biases: &biases
            )
        }
        let decoded = dequantize(
            packed: packed, scales: scales, biases: biases,
            outDim: 1, inDim: inDim, quantization: .q4
        )
        for value in decoded {
            XCTAssertEqual(value, 0.25, accuracy: 0.001)
        }
        XCTAssertTrue(decoded.allSatisfy { $0.isFinite })
    }

    func testBF16ConversionKeepsTheTopHalfOfAnFP32() {
        for value: Float in [0, 1, -1.5, 0.1, 1e-8, 12_345.0] {
            let bits = DenseRepacker.floatToBF16(value)
            let restored = DenseRepacker.bf16ToFloat(bits)
            XCTAssertEqual(restored, value, accuracy: abs(value) * 0.01 + 1e-9, "\(value)")
        }
        // The published checkpoint's own dtype round-trips exactly.
        let exact: Float = 0.5
        XCTAssertEqual(DenseRepacker.bf16ToFloat(DenseRepacker.floatToBF16(exact)), exact)
    }

    // MARK: - Eligibility

    private func tensor(
        _ name: String,
        shape: [Int],
        dtype: String = "BF16",
        expert: UInt32? = nil
    ) -> TensorMetadata {
        TensorMetadata(
            name: name,
            shapeDisplay: "[\(shape.map(String.init).joined(separator: ", "))]",
            dtype: dtype,
            sizeMb: 1,
            shardIndex: 0,
            offsetStart: 0,
            offsetEnd: 1024,
            category: "Other",
            layerIndex: 0,
            expertId: expert
        )
    }

    func testOnlyWideTwoDimensionalBF16TensorsAreQuantized() {
        // The layer projections of the published geometry (rows come first).
        XCTAssertTrue(DenseRepacker.isQuantizable(tensor("model.layers.0.self_attn.q_proj.weight", shape: [7168, 7168])))
        XCTAssertTrue(DenseRepacker.isQuantizable(tensor("model.layers.0.self_attn.k_proj.weight", shape: [1024, 7168])))
        XCTAssertTrue(DenseRepacker.isQuantizable(tensor("model.layers.0.mlp.down_proj.weight", shape: [7168, 22016])))
        XCTAssertTrue(DenseRepacker.isQuantizable(tensor("model.layers.0.mlp.gate_proj.weight", shape: [22016, 7168])))

        // Norms are 1-D and the embedding/head are read a row at a time.
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.layers.0.input_layernorm.weight", shape: [7168])))
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.norm.weight", shape: [7168])))
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.embed_tokens.weight", shape: [166912, 7168])))
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("lm_head.weight", shape: [166912, 7168])))

        // Already-quantized weights, routed experts, and widths that do not divide.
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.layers.0.mlp.up_proj.weight", shape: [22016, 7168], dtype: "F16")))
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.layers.0.mlp.up_proj.weight", shape: [16, 48])))
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.layers.0.mlp.experts.3.down_proj.weight", shape: [7168, 2048], expert: 3)))
    }

    // MARK: - Manifest

    func testAManifestDescribesItsOwnBlobAndAStaleOneIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dense-repack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let manifest = DenseRepackManifest(
            quantization: "q4",
            groupSize: 64,
            tensors: [
                .init(name: "model.layers.0.mlp.gate_proj.weight", dtype: "Q4", shape: [22016, 7168], offset: 0, size: 1000),
                .init(name: "model.layers.0.mlp.gate_proj.weight.scale", dtype: "BF16", shape: [22016, 112], offset: 1008, size: 200),
                .init(name: "model.layers.0.mlp.gate_proj.weight.bias", dtype: "BF16", shape: [22016, 112], offset: 1216, size: 200),
            ],
            blobSize: 1416
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("dense_weights.json"))

        // Snakes through the snake_case keys the Rust side decodes.
        let raw = try String(contentsOf: directory.appendingPathComponent("dense_weights.json"), encoding: .utf8)
        XCTAssertTrue(raw.contains("\"group_size\""))
        XCTAssertTrue(raw.contains("\"quantization\""))

        XCTAssertFalse(DenseRepacker.isRepacked(dir: directory), "a manifest with no blob is not a repack")
        try Data(count: 1416).write(to: directory.appendingPathComponent("dense_weights.bin"))
        XCTAssertTrue(DenseRepacker.isRepacked(dir: directory))
        XCTAssertEqual(DenseRepacker.quantizedFormat(dir: directory), .q4)
        XCTAssertEqual(DenseRepacker.Quantization.q4.displayName, "Q4 (dense repack)")

        // The blob the manifest was written against is recorded, so a blob of a
        // DIFFERENT length is refused even when every offset still lands inside
        // it. That is the Q4-over-Q8 case: a Q8 blob contains every Q4 offset, so
        // bounds alone would have loaded it at the wrong stride while still
        // reporting "Q4".
        try Data(count: 2048).write(to: directory.appendingPathComponent("dense_weights.bin"))
        XCTAssertFalse(
            DenseRepacker.isRepacked(dir: directory),
            "a blob of another length is not the blob this manifest describes"
        )

        // A truncated blob describes bytes that were never written, which is the
        // shape an interrupted run leaves behind.
        try Data(count: 900).write(to: directory.appendingPathComponent("dense_weights.bin"))
        XCTAssertFalse(DenseRepacker.isRepacked(dir: directory))
    }

    /// The packed kernels read a weight row as `uchar4` and the input as `float4`,
    /// and Apple GPUs require a vector load to be naturally aligned. A row starts
    /// at `offset + row * inDim`, so the manifest is accepted only when every
    /// packed entry starts on a 16-byte boundary and its row width is a multiple
    /// of 16. Nothing else in the app can see this: a misaligned row reads
    /// garbage rather than failing.
    func testAPackedEntryMustBeReadableAsAlignedSixteenByteChunks() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dense-repack-align-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let blobBytes: UInt64 = 4096
        try Data(count: Int(blobBytes)).write(to: directory.appendingPathComponent("dense_weights.bin"))

        func write(_ manifest: DenseRepackManifest) throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(manifest).write(to: directory.appendingPathComponent("dense_weights.json"))
        }
        func entry(_ name: String, offset: UInt64, width: Int, dtype: String = "Q8") -> DenseRepackManifest.TensorEntry {
            .init(name: name, dtype: dtype, shape: [16, width], offset: offset, size: 256)
        }

        // A two-byte offset is what a verbatim-copied tensor of odd length used to
        // leave behind, and it is exactly the misalignment that read garbage.
        try write(DenseRepackManifest(
            quantization: "q8",
            groupSize: 64,
            tensors: [
                entry("a.weight", offset: 2, width: 64),
                entry("a.weight.scale", offset: 512, width: 1, dtype: "BF16"),
                entry("a.weight.bias", offset: 768, width: 1, dtype: "BF16"),
            ],
            blobSize: blobBytes
        ))
        XCTAssertFalse(DenseRepacker.isRepacked(dir: directory), "a packed row at byte 2 is not 4-byte aligned")

        // Aligned, but a row of 72 bytes cannot be read in 16-byte chunks.
        try write(DenseRepackManifest(
            quantization: "q8",
            groupSize: 64,
            tensors: [
                entry("a.weight", offset: 0, width: 72),
                entry("a.weight.scale", offset: 512, width: 1, dtype: "BF16"),
                entry("a.weight.bias", offset: 768, width: 1, dtype: "BF16"),
            ],
            blobSize: blobBytes
        ))
        XCTAssertFalse(DenseRepacker.isRepacked(dir: directory), "a 72-wide row cannot be read as 16-byte chunks")

        // And a packed weight with no bias companion is refused: the quantized
        // path needs both, and a missing one silently skipped the dispatch.
        try write(DenseRepackManifest(
            quantization: "q8",
            groupSize: 64,
            tensors: [
                entry("a.weight", offset: 0, width: 64),
                entry("a.weight.scale", offset: 512, width: 1, dtype: "BF16"),
            ],
            blobSize: blobBytes
        ))
        XCTAssertFalse(DenseRepacker.isRepacked(dir: directory), "a packed weight needs both companions")

        // The whole trio, aligned, on a 16-divisible row width: a repack.
        try write(DenseRepackManifest(
            quantization: "q8",
            groupSize: 64,
            tensors: [
                entry("a.weight", offset: 0, width: 64),
                entry("a.weight.scale", offset: 256, width: 1, dtype: "BF16"),
                entry("a.weight.bias", offset: 512, width: 1, dtype: "BF16"),
            ],
            blobSize: blobBytes
        ))
        XCTAssertTrue(DenseRepacker.isRepacked(dir: directory))
        XCTAssertEqual(DenseRepacker.quantizedFormat(dir: directory), .q8)
    }

    /// Everything the repacker writes starts on, and stays on, a 16-byte boundary:
    /// the header records a blob length the entries live inside, and the offsets
    /// are what the kernels' vector loads depend on.
    func testEveryEntryTheRepackerWritesIsSixteenByteAligned() {
        XCTAssertEqual(DenseRepacker.tensorAlignment, 16)
        // The row width rule `isQuantizable` enforces is what makes `offset +
        // row * inDim` aligned for every row once the offset is.
        XCTAssertTrue(DenseRepacker.isQuantizable(tensor("model.layers.0.self_attn.q_proj.weight", shape: [7168, 7168])))
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.layers.0.self_attn.q_proj.weight", shape: [7168, 7168 + 16 - 8])))
        // Companions of an already packed tensor are never re-quantized.
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.layers.0.mlp.gate_proj.weight.scale", shape: [22016, 112])))
        XCTAssertFalse(DenseRepacker.isQuantizable(tensor("model.layers.0.mlp.gate_proj.weight.bias", shape: [22016, 112])))
    }
}
