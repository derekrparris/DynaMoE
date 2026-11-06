//
//  Q8AffineKernelTests.swift
//  DynaMoETests
//
//  The Q8 affine kernels, executed on the GPU against a CPU reference.
//
//  These are the kernels a dense repack depends on, and until this file nothing
//  in the suite ran one: the encoder that packs a blob and the kernels that read
//  it were each checked in isolation, so a disagreement between them (a row
//  stride, a group index, a scale read from the wrong row) could only be found by
//  loading a 30B model and noticing the answer was wrong.
//
//  The reference here is built the way `DenseRepacker.quantizeMatrix` writes and
//  the way `q8_gemv` states it reads: `scale[group] * q + bias[group]`, per
//  64-value group, with the scale and bias being the BF16 values that were
//  STORED rather than the float32 ones that were computed. Every case therefore
//  checks the pair, not one of them.
//

import XCTest
import Metal
@testable import DynaMoE

final class Q8AffineKernelTests: XCTestCase {

    /// The one group size the kernels implement.
    private let groupSize = 64

    // MARK: - Fixtures

    private struct Q8Matrix {
        let outDim: Int
        let inDim: Int
        /// `outDim * inDim` bytes, row-major.
        let packed: [UInt8]
        /// `outDim * groups` BF16 scale values, row-major.
        let scales: [UInt16]
        /// `outDim * groups` BF16 bias values, row-major.
        let biases: [UInt16]

        var groups: Int { inDim / 64 }

        /// The dequantized weight at a coordinate, exactly as the kernels decode it.
        func weight(row: Int, column: Int) -> Float {
            let group = column / 64
            let scale = DenseRepacker.bf16ToFloat(scales[row * groups + group])
            let bias = DenseRepacker.bf16ToFloat(biases[row * groups + group])
            return scale * Float(packed[row * inDim + column]) + bias
        }
    }

    /// Quantizes a deterministic pseudo-random matrix through the REPACKER's own
    /// encoder, so the kernels are checked against the bytes the app writes.
    private func quantize(outDim: Int, inDim: Int, seed: UInt32) throws -> Q8Matrix {
        var values = [UInt16](repeating: 0, count: outDim * inDim)
        var state = seed &* 2_654_435_761
        for index in 0..<values.count {
            state = state &* 1_664_525 &+ 1_013_904_223
            // A magnitude range wide enough that a clamped or mis-scaled group is
            // visible in the comparison rather than lost in the noise.
            let unit = Float(state % 10_000) / 10_000
            values[index] = DenseRepacker.floatToBF16((unit - 0.5) * 4.0)
        }
        let groups = inDim / groupSize
        var scales = [UInt16](repeating: 0, count: outDim * groups)
        var biases = [UInt16](repeating: 0, count: outDim * groups)
        let packed = values.withUnsafeBufferPointer { buffer in
            DenseRepacker.quantizeMatrix(
                source: buffer.baseAddress!,
                outDim: outDim,
                inDim: inDim,
                quantization: .q8,
                scales: &scales,
                biases: &biases
            )
        }
        return Q8Matrix(outDim: outDim, inDim: inDim, packed: packed, scales: scales, biases: biases)
    }

    private func bytes(_ values: [UInt16]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: values.count * 2)
        for (index, value) in values.enumerated() {
            out[index * 2] = UInt8(value & 0xFF)
            out[index * 2 + 1] = UInt8(value >> 8)
        }
        return out
    }

    /// A blob laid out the way the repacker writes one: each tensor starts on a
    /// 16-byte boundary, and a row is `inDim` bytes from the tensor's start.
    ///
    /// `leadingPadding` exists so a case can put the tensor somewhere other than
    /// the start of the buffer, which is what the alignment rule is about.
    private struct Blob {
        let bytes: [UInt8]
        let weightOffset: UInt64
        let scaleOffset: UInt64
        let biasOffset: UInt64

        static func build(
            matrices: [(weights: [UInt8], scales: [UInt16], biases: [UInt16])],
            leadingPadding: UInt64 = 16
        ) -> (blob: Blob, offsets: [(UInt64, UInt64, UInt64)]) {
            var data = [UInt8](repeating: 0, count: Int(leadingPadding))
            var offsets: [(UInt64, UInt64, UInt64)] = []
            for matrix in matrices {
                let weightOffset = UInt64(data.count)
                data += matrix.weights
                data += [UInt8](repeating: 0, count: pad(UInt64(data.count)))
                let scaleOffset = UInt64(data.count)
                data += bf16Bytes(matrix.scales)
                data += [UInt8](repeating: 0, count: pad(UInt64(data.count)))
                let biasOffset = UInt64(data.count)
                data += bf16Bytes(matrix.biases)
                data += [UInt8](repeating: 0, count: pad(UInt64(data.count)))
                offsets.append((weightOffset, scaleOffset, biasOffset))
            }
            // The layout rule the kernels depend on, asserted on the FIXTURE so a
            // case that reads garbage can never be blamed on the test's own blob.
            for entry in offsets {
                XCTAssertEqual(entry.0 % 16, 0)
                XCTAssertEqual(entry.1 % 16, 0)
                XCTAssertEqual(entry.2 % 16, 0)
            }
            let blob = Blob(
                bytes: data,
                weightOffset: offsets.first?.0 ?? 0,
                scaleOffset: offsets.first?.1 ?? 0,
                biasOffset: offsets.first?.2 ?? 0
            )
            return (blob, offsets)
        }

        private static func pad(_ length: UInt64) -> Int {
            Int((16 - length % 16) % 16)
        }

        private static func bf16Bytes(_ values: [UInt16]) -> [UInt8] {
            var out = [UInt8](repeating: 0, count: values.count * 2)
            for (index, value) in values.enumerated() {
                out[index * 2] = UInt8(value & 0xFF)
                out[index * 2 + 1] = UInt8(value >> 8)
            }
            return out
        }
    }

    private func makeDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        return device
    }

    private func floatBuffer(_ device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        try XCTUnwrap(device.makeBuffer(
            bytes: values,
            length: max(values.count, 1) * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
    }

    private func run(
        _ pipeline: MTLComputePipelineState,
        queue: MTLCommandQueue,
        encode: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encode(encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error, "the dispatch failed")
    }

    // MARK: - q8_gemv

    /// One row of the affine product, per position, against the dequantized
    /// definition. The matrix is deliberately NOT square and its row width is a
    /// multiple of 64 but not of 512, so a chunk stride that assumed a round
    /// number of 16-byte chunks per row would drop columns.
    func testTheQ8GemvMatchesTheDequantizedDefinition() throws {
        let device = try makeDevice()
        let library = try XCTUnwrap(device.makeDefaultLibrary())
        let pipeline = try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "q8_gemv"))
        )
        let queue = try XCTUnwrap(device.makeCommandQueue())

        let outDim = 5
        let inDim = 192          // three 64-value groups, twelve 16-byte chunks
        let batch = 2
        let matrix = try quantize(outDim: outDim, inDim: inDim, seed: 7)
        let (blob, _) = Blob.build(matrices: [(matrix.packed, matrix.scales, matrix.biases)])
        let weightBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: blob.bytes, length: blob.bytes.count, options: .storageModeShared
        ))

        var inputs = [Float](repeating: 0, count: batch * inDim)
        for index in inputs.indices {
            inputs[index] = Float((index % 17) - 8) / 8
        }
        let inputBuffer = try floatBuffer(device, inputs)
        let outputBuffer = try XCTUnwrap(device.makeBuffer(
            length: batch * outDim * MemoryLayout<Float>.stride, options: .storageModeShared
        ))

        var weightOffset = blob.weightOffset
        var scaleOffset = blob.scaleOffset
        var biasOffset = blob.biasOffset
        var inDimValue = UInt32(inDim)
        var outDimValue = UInt32(outDim)
        var groupValue = UInt32(groupSize)

        try run(pipeline, queue: queue) { encoder in
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputBuffer, offset: 0, index: 4)
            encoder.setBytes(&weightOffset, length: 8, index: 5)
            encoder.setBytes(&scaleOffset, length: 8, index: 6)
            encoder.setBytes(&biasOffset, length: 8, index: 7)
            encoder.setBytes(&inDimValue, length: 4, index: 8)
            encoder.setBytes(&outDimValue, length: 4, index: 9)
            encoder.setBytes(&groupValue, length: 4, index: 10)
            encoder.dispatchThreadgroups(
                MTLSize(width: outDim, height: batch, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        let output = outputBuffer.contents().bindMemory(to: Float.self, capacity: batch * outDim)
        var worst: Float = 0
        for token in 0..<batch {
            for row in 0..<outDim {
                var expected: Float = 0
                for column in 0..<inDim {
                    expected += matrix.weight(row: row, column: column) * inputs[token * inDim + column]
                }
                let actual = output[token * outDim + row]
                worst = max(worst, abs(actual - expected))
                XCTAssertEqual(
                    actual, expected, accuracy: 2e-4,
                    "token \(token) row \(row): the kernel must read the row the repacker wrote"
                )
            }
        }
        XCTAssertLessThan(worst, 2e-4)
    }

    /// The same weight bytes at a 16-byte-aligned offset that is NOT zero, which
    /// is what every tensor after the first one in a blob is. A kernel that
    /// ignored `weightOffset` would read the padding and still return finite
    /// numbers.
    func testTheQ8GemvHonoursAPaddedWeightOffset() throws {
        let device = try makeDevice()
        let library = try XCTUnwrap(device.makeDefaultLibrary())
        let pipeline = try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "q8_gemv"))
        )
        let queue = try XCTUnwrap(device.makeCommandQueue())

        let outDim = 3
        let inDim = 64
        let matrix = try quantize(outDim: outDim, inDim: inDim, seed: 11)
        // A 16-byte dummy tensor ahead of it, so the weight starts at 16.
        let blobPrefix = [UInt8](repeating: 0xAB, count: 16)
        var data = blobPrefix
        let weightOffset = UInt64(data.count)
        data += matrix.packed
        let scaleOffset = UInt64(data.count)
        data += bytes(matrix.scales)
        let biasOffset = UInt64(data.count)
        data += bytes(matrix.biases)
        XCTAssertEqual(weightOffset, 16)

        let weightBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: data, length: data.count, options: .storageModeShared
        ))
        let inputs = (0..<inDim).map { Float($0 % 7) * 0.25 }
        let inputBuffer = try floatBuffer(device, inputs)
        let outputBuffer = try XCTUnwrap(device.makeBuffer(
            length: outDim * MemoryLayout<Float>.stride, options: .storageModeShared
        ))

        var wOff = weightOffset
        var sOff = scaleOffset
        var bOff = biasOffset
        var inDimValue = UInt32(inDim)
        var outDimValue = UInt32(outDim)
        var groupValue = UInt32(groupSize)

        try run(pipeline, queue: queue) { encoder in
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputBuffer, offset: 0, index: 4)
            encoder.setBytes(&wOff, length: 8, index: 5)
            encoder.setBytes(&sOff, length: 8, index: 6)
            encoder.setBytes(&bOff, length: 8, index: 7)
            encoder.setBytes(&inDimValue, length: 4, index: 8)
            encoder.setBytes(&outDimValue, length: 4, index: 9)
            encoder.setBytes(&groupValue, length: 4, index: 10)
            encoder.dispatchThreadgroups(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        let output = outputBuffer.contents().bindMemory(to: Float.self, capacity: outDim)
        for row in 0..<outDim {
            var expected: Float = 0
            for column in 0..<inDim {
                expected += matrix.weight(row: row, column: column) * inputs[column]
            }
            XCTAssertEqual(output[row], expected, accuracy: 2e-4, "row \(row) at a padded offset")
        }
    }

    // MARK: - q8_gemv_batched

    /// The prefill form against the per-position one. 17 positions is deliberate:
    /// it needs a second 16-token block whose LAST row is partial, which is the
    /// only shape that exercises the `t0 + t < batch` guard and the phantom reads
    /// its clamp allows.
    func testTheBatchedQ8GemvServesEveryPositionOfAPartialBlock() throws {
        let device = try makeDevice()
        let library = try XCTUnwrap(device.makeDefaultLibrary())
        let single = try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "q8_gemv"))
        )
        let batched = try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "q8_gemv_batched"))
        )
        let queue = try XCTUnwrap(device.makeCommandQueue())

        let outDim = 4
        let inDim = 128
        let batch = 17
        let matrix = try quantize(outDim: outDim, inDim: inDim, seed: 23)
        let (blob, _) = Blob.build(matrices: [(matrix.packed, matrix.scales, matrix.biases)])
        let weightBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: blob.bytes, length: blob.bytes.count, options: .storageModeShared
        ))
        var inputs = [Float](repeating: 0, count: batch * inDim)
        for token in 0..<batch {
            for column in 0..<inDim {
                inputs[token * inDim + column] = Float((token * 31 + column * 7) % 19) / 19
            }
        }
        let inputBuffer = try floatBuffer(device, inputs)
        var reference = [Float](repeating: 0, count: batch * outDim)
        for token in 0..<batch {
            for row in 0..<outDim {
                var sum: Float = 0
                for column in 0..<inDim {
                    sum += matrix.weight(row: row, column: column) * inputs[token * inDim + column]
                }
                reference[token * outDim + row] = sum
            }
        }

        var wOff = blob.weightOffset
        var sOff = blob.scaleOffset
        var bOff = blob.biasOffset
        var inDimValue = UInt32(inDim)
        var outDimValue = UInt32(outDim)
        var groupValue = UInt32(groupSize)
        var batchValue = UInt32(batch)

        func bind(_ encoder: MTLComputeCommandEncoder, _ output: MTLBuffer) {
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            encoder.setBytes(&wOff, length: 8, index: 5)
            encoder.setBytes(&sOff, length: 8, index: 6)
            encoder.setBytes(&bOff, length: 8, index: 7)
            encoder.setBytes(&inDimValue, length: 4, index: 8)
            encoder.setBytes(&outDimValue, length: 4, index: 9)
            encoder.setBytes(&groupValue, length: 4, index: 10)
        }

        let perTokenBuffer = try XCTUnwrap(device.makeBuffer(
            length: batch * outDim * MemoryLayout<Float>.stride, options: .storageModeShared
        ))
        try run(single, queue: queue) { encoder in
            bind(encoder, perTokenBuffer)
            encoder.dispatchThreadgroups(
                MTLSize(width: outDim, height: batch, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        let batchedBuffer = try XCTUnwrap(device.makeBuffer(
            length: batch * outDim * MemoryLayout<Float>.stride, options: .storageModeShared
        ))
        try run(batched, queue: queue) { encoder in
            bind(encoder, batchedBuffer)
            encoder.setBytes(&batchValue, length: 4, index: 11)
            encoder.dispatchThreadgroups(
                MTLSize(width: outDim, height: (batch + 15) / 16, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        let perToken = perTokenBuffer.contents().bindMemory(to: Float.self, capacity: batch * outDim)
        let block = batchedBuffer.contents().bindMemory(to: Float.self, capacity: batch * outDim)
        for index in 0..<(batch * outDim) {
            XCTAssertEqual(
                block[index], perToken[index], accuracy: 1e-4,
                "position \(index / outDim) row \(index % outDim): the batched form must serve every position"
            )
            XCTAssertEqual(block[index], reference[index], accuracy: 5e-4)
        }
    }

    // MARK: - q8_swiglu_gate_up + q8_down_proj_accumulate

    /// The fused gate/up pair and the down projection over the same weights,
    /// against the dequantized definition including the SiLU the kernel folds in.
    func testTheQ8FusedMlpMatchesTheDequantizedDefinition() throws {
        let device = try makeDevice()
        let library = try XCTUnwrap(device.makeDefaultLibrary())
        let gateUp = try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "q8_swiglu_gate_up"))
        )
        let down = try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "q8_down_proj_accumulate"))
        )
        let queue = try XCTUnwrap(device.makeCommandQueue())

        let hidden = 128
        let intermediate = 192
        let outDim = hidden
        let gateMatrix = try quantize(outDim: intermediate, inDim: hidden, seed: 31)
        let upMatrix = try quantize(outDim: intermediate, inDim: hidden, seed: 37)
        let downMatrix = try quantize(outDim: outDim, inDim: intermediate, seed: 41)

        let (blob, offsets) = Blob.build(matrices: [
            (gateMatrix.packed, gateMatrix.scales, gateMatrix.biases),
            (upMatrix.packed, upMatrix.scales, upMatrix.biases),
            (downMatrix.packed, downMatrix.scales, downMatrix.biases),
        ])
        let blobBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: blob.bytes, length: blob.bytes.count, options: .storageModeShared
        ))
        let inputs = (0..<hidden).map { Float(($0 % 13) - 6) / 6 }
        let inputBuffer = try floatBuffer(device, inputs)
        let interBuffer = try XCTUnwrap(device.makeBuffer(
            length: intermediate * MemoryLayout<Float>.stride, options: .storageModeShared
        ))
        // `q8_down_proj_accumulate` ADDS into its destination (a mixture layers
        // several projections into one buffer), so the buffer has to start
        // zeroed or the case measures whatever the allocation happened to hold.
        let outBuffer = try XCTUnwrap(device.makeBuffer(
            length: outDim * MemoryLayout<Float>.stride, options: .storageModeShared
        ))
        memset(outBuffer.contents(), 0, outDim * MemoryLayout<Float>.stride)

        var gateW = offsets[0].0, gateS = offsets[0].1, gateB = offsets[0].2
        var upW = offsets[1].0, upS = offsets[1].1, upB = offsets[1].2
        var downW = offsets[2].0, downS = offsets[2].1, downB = offsets[2].2
        var hiddenValue = UInt32(hidden)
        var intermediateValue = UInt32(intermediate)
        var groupValue = UInt32(groupSize)
        var routing: Float = 1.0

        try run(gateUp, queue: queue) { encoder in
            encoder.setBuffer(blobBuffer, offset: 0, index: 0)
            encoder.setBuffer(blobBuffer, offset: 0, index: 1)
            encoder.setBuffer(blobBuffer, offset: 0, index: 2)
            encoder.setBuffer(blobBuffer, offset: 0, index: 3)
            encoder.setBuffer(blobBuffer, offset: 0, index: 4)
            encoder.setBuffer(blobBuffer, offset: 0, index: 5)
            encoder.setBuffer(inputBuffer, offset: 0, index: 6)
            encoder.setBuffer(interBuffer, offset: 0, index: 7)
            encoder.setBytes(&gateW, length: 8, index: 8)
            encoder.setBytes(&gateS, length: 8, index: 9)
            encoder.setBytes(&gateB, length: 8, index: 10)
            encoder.setBytes(&upW, length: 8, index: 11)
            encoder.setBytes(&upS, length: 8, index: 12)
            encoder.setBytes(&upB, length: 8, index: 13)
            encoder.setBytes(&hiddenValue, length: 4, index: 14)
            encoder.setBytes(&intermediateValue, length: 4, index: 15)
            encoder.setBytes(&groupValue, length: 4, index: 16)
            encoder.dispatchThreadgroups(
                MTLSize(width: intermediate, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        try run(down, queue: queue) { encoder in
            encoder.setBuffer(blobBuffer, offset: 0, index: 0)
            encoder.setBuffer(blobBuffer, offset: 0, index: 1)
            encoder.setBuffer(blobBuffer, offset: 0, index: 2)
            encoder.setBuffer(interBuffer, offset: 0, index: 3)
            encoder.setBuffer(outBuffer, offset: 0, index: 4)
            encoder.setBytes(&downW, length: 8, index: 5)
            encoder.setBytes(&downS, length: 8, index: 6)
            encoder.setBytes(&downB, length: 8, index: 7)
            encoder.setBytes(&intermediateValue, length: 4, index: 8)
            encoder.setBytes(&hiddenValue, length: 4, index: 9)
            encoder.setBytes(&groupValue, length: 4, index: 10)
            encoder.setBytes(&routing, length: 4, index: 11)
            encoder.dispatchThreadgroups(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        // Reference: silu(gate) * up per intermediate row, then the down product.
        var expectedIntermediate = [Float](repeating: 0, count: intermediate)
        for row in 0..<intermediate {
            var gate: Float = 0
            var up: Float = 0
            for column in 0..<hidden {
                gate += gateMatrix.weight(row: row, column: column) * inputs[column]
                up += upMatrix.weight(row: row, column: column) * inputs[column]
            }
            expectedIntermediate[row] = (gate / (1 + exp(-gate))) * up
        }
        let actualIntermediate = interBuffer.contents().bindMemory(to: Float.self, capacity: intermediate)
        for row in 0..<intermediate {
            XCTAssertEqual(
                actualIntermediate[row], expectedIntermediate[row], accuracy: 1e-3,
                "intermediate row \(row)"
            )
        }

        let actual = outBuffer.contents().bindMemory(to: Float.self, capacity: outDim)
        for row in 0..<outDim {
            var expected: Float = 0
            for column in 0..<intermediate {
                expected += downMatrix.weight(row: row, column: column) * expectedIntermediate[column]
            }
            XCTAssertEqual(actual[row], expected, accuracy: 2e-3, "down row \(row)")
        }
    }
}
