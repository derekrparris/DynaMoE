//
//  Inkling2DeviceKernelTests.swift
//  DynaMoE
//
//  The Inkling 2 device graph's kernels, each against the HOST rule it ports
//  (`Inkling2Math`), not against a second GPU implementation: the host graph is
//  what matches the checkpoint's own Transformers reference, so a kernel that
//  agrees with it agrees with the model.
//
//  Two rules this file follows from the DeepSeek kernel tests:
//
//    * the host rule is the oracle for EVERY case, including a shape the
//      published checkpoint never takes -- a kernel only ever run at its
//      production shape is a kernel whose strides, bounds and tail handling are
//      unchecked;
//    * a fixture is chosen so a plausible bug FAILS. Wide ranges where a
//      narrow one passes is the pattern: a scale row read at the wrong stride
//      still passes a fixture whose bytes happen to be symmetric.
//

import Metal
import XCTest
@testable import DynaMoE

final class Inkling2DeviceKernelTests: XCTestCase {

    private func environment() throws -> (MTLDevice, MTLLibrary, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary() else {
            throw XCTSkip("no default library")
        }
        guard let queue = device.makeCommandQueue() else {
            throw XCTSkip("no compute queue")
        }
        return (device, library, queue)
    }

    /// A deterministic, wide-ranging fixture: every value distinct and of both
    /// signs, so a swapped index or a dropped tap cannot cancel.
    private func values(_ count: Int, seed: Int = 1) -> [Float] {
        (0..<count).map { index in
            let mixed = (index &* 37 &+ seed &* 101) % 23
            let sign: Float = (index &+ seed) % 3 == 0 ? -1 : 1
            return sign * (Float(mixed) + 0.5) / 8
        }
    }

    private func buffer(_ device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        try XCTUnwrap(
            device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Float>.stride,
                              options: .storageModeShared),
            "could not make a buffer"
        )
    }

    private func buffer(_ device: MTLDevice, _ values: [UInt32]) throws -> MTLBuffer {
        try XCTUnwrap(
            device.makeBuffer(bytes: values, length: values.count * MemoryLayout<UInt32>.stride,
                              options: .storageModeShared),
            "could not make a buffer"
        )
    }

    /// A byte offset, which the bank kernels take as a `constant`: a tensor's
    /// offset inside a shard is not buffer-aligned.
    private func buffer(_ device: MTLDevice, _ values: [UInt64]) throws -> MTLBuffer {
        try XCTUnwrap(
            device.makeBuffer(bytes: values, length: values.count * MemoryLayout<UInt64>.stride,
                              options: .storageModeShared),
            "could not make a buffer"
        )
    }

    /// An INT8 KV row block, as `inkling_attention_int8_f32` reads it.
    private func bytes(_ device: MTLDevice, _ values: [Int8]) throws -> MTLBuffer {
        try XCTUnwrap(
            device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Int8>.stride,
                              options: .storageModeShared),
            "could not make a byte buffer"
        )
    }

    /// FP16 scale bits, which the kernel binds as `device const half*`.
    private func bits(_ device: MTLDevice, _ values: [UInt16]) throws -> MTLBuffer {
        try XCTUnwrap(
            device.makeBuffer(bytes: values, length: values.count * MemoryLayout<UInt16>.stride,
                              options: .storageModeShared),
            "could not make a scale buffer"
        )
    }

    /// A one-byte `constant bool`.
    private func flag(_ device: MTLDevice, _ value: Bool) throws -> MTLBuffer {
        var byte = value ? UInt8(1) : UInt8(0)
        return try XCTUnwrap(
            device.makeBuffer(bytes: &byte, length: 1, options: .storageModeShared),
            "could not make a flag buffer"
        )
    }

    /// Dispatches a grid of THREADS (`dispatchThreads`) or of THREADGROUPS
    /// (`dispatchThreadgroups`). A kernel that indexes by
    /// `threadgroup_position_in_grid` needs the latter: with `dispatchThreads`
    /// and a grid narrower than one threadgroup, every thread sees group 0.
    private func run(
        _ name: String,
        threads: (Int, Int, Int),
        threadgroup: (Int, Int, Int),
        threadgroupMemory: Int = 0,
        byThreadgroup: Bool = false,
        encode: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let (device, library, queue) = try environment()
        let pipeline = try XCTUnwrap(
            try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name))),
            "no pipeline for \(name)"
        )
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        if threadgroupMemory > 0 { encoder.setThreadgroupMemoryLength(threadgroupMemory, index: 0) }
        encode(encoder)
        let grid = MTLSize(width: threads.0, height: threads.1, depth: threads.2)
        let perGroup = MTLSize(width: threadgroup.0, height: threadgroup.1, depth: threadgroup.2)
        if byThreadgroup {
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: perGroup)
        } else {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: perGroup)
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    private func read(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count
        ))
    }

    // MARK: Norms

    /// A weighted RMS norm at a width that is not a multiple of the threadgroup
    /// size, so the strided loops' tails run.
    func testTheWeightedNormMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let width = 2304, rows = 2
        let input = values(rows * width)
        let gain = values(width, seed: 5)
        let inputBuffer = try buffer(device, input)
        // The gain is BF16 in the checkpoint, and the kernel reads it as stored
        // bytes at a constant offset (a tensor offset is not buffer-aligned), so
        // the fixture stages the BF16 encoding behind a leading pad and the host
        // rule multiplies by the same rounded values.
        let gainPad = 2
        var gainBytes = [UInt16](repeating: 0, count: gainPad)
        for value in gain {
            let bits = value.bitPattern
            // Round to nearest even, the same rounding the checkpoint's store did.
            let rounding = UInt32(0x7FFF) + ((bits >> 16) & 1)
            gainBytes.append(UInt16(truncatingIfNeeded: (bits + rounding) >> 16))
        }
        let gainBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: gainBytes, length: gainBytes.count * MemoryLayout<UInt16>.stride, options: .storageModeShared
        ))
        let gainOffsetBuffer = try buffer(device, [UInt64(gainPad) * UInt64(MemoryLayout<UInt16>.stride)])
        let roundedGain = gainBytes.dropFirst(gainPad).map { Float(bitPattern: UInt32($0) << 16) }
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: rows * width))
        let dimBuffer = try buffer(device, [UInt32(width)])
        let epsBuffer = try buffer(device, [Float(1e-6)])
        try run("inkling_rmsnorm_f32", threads: (rows, 1, 1), threadgroup: (128, 1, 1),
                threadgroupMemory: 32 * MemoryLayout<Float>.stride, byThreadgroup: true) { encoder in
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(gainBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            encoder.setBuffer(dimBuffer, offset: 0, index: 3)
            encoder.setBuffer(epsBuffer, offset: 0, index: 4)
            encoder.setBuffer(gainOffsetBuffer, offset: 0, index: 5)
        }
        let actual = read(outputBuffer, rows * width)
        for row in 0..<rows {
            let expected = Inkling2Math.rmsNorm(
                Array(input[(row * width)..<((row + 1) * width)]), gain: roundedGain, eps: 1e-6
            )
            for index in 0..<width {
                XCTAssertEqual(actual[row * width + index], expected[index], accuracy: 1e-5,
                               "row \(row) element \(index)")
            }
        }
    }

    /// The unweighted form, which the block read applies to every stack row.
    /// A gain-buffer variant of the weighted kernel would pass a fixture whose
    /// gain is all ones and fail here, so the two are separate kernels.
    func testTheUnweightedNormMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let width = 100, rows = 3
        let input = values(rows * width, seed: 2)
        let inputBuffer = try buffer(device, input)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: rows * width))
        let dimBuffer = try buffer(device, [UInt32(width)])
        let epsBuffer = try buffer(device, [Float(1e-6)])
        try run("inkling_rmsnorm_no_gain_f32", threads: (rows, 1, 1), threadgroup: (64, 1, 1),
                threadgroupMemory: 32 * MemoryLayout<Float>.stride, byThreadgroup: true) { encoder in
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(dimBuffer, offset: 0, index: 2)
            encoder.setBuffer(epsBuffer, offset: 0, index: 3)
        }
        let actual = read(outputBuffer, rows * width)
        for row in 0..<rows {
            let expected = Inkling2Math.rmsNorm(
                Array(input[(row * width)..<((row + 1) * width)]), eps: 1e-6
            )
            for index in 0..<width {
                XCTAssertEqual(actual[row * width + index], expected[index], accuracy: 1e-6,
                               "row \(row) element \(index)")
            }
        }
    }

    // MARK: Elementwise

    /// The softcapped SwiGLU. The softcaps are what make it Inkling's rather
    /// than a plain SwiGLU, so the fixture's values are large enough for both
    /// caps to bind (`tanh` saturates past ~4) and the case also checks a value
    /// where neither does.
    func testTheSoftcappedSwiGLUMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let intermediate = 33
        var gateUp = values(intermediate, seed: 3)
        gateUp += values(intermediate, seed: 4).map { $0 * 12 }
        let gateUpBuffer = try buffer(device, gateUp)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: intermediate))
        let intermediateBuffer = try buffer(device, [UInt32(intermediate)])
        let gateCapBuffer = try buffer(device, [Float(4)])
        let upCapBuffer = try buffer(device, [Float(25)])
        try run("inkling_softcapped_swiglu_f32", threads: (intermediate, 1, 1), threadgroup: (64, 1, 1)) { encoder in
            encoder.setBuffer(gateUpBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(intermediateBuffer, offset: 0, index: 2)
            encoder.setBuffer(gateCapBuffer, offset: 0, index: 3)
            encoder.setBuffer(upCapBuffer, offset: 0, index: 4)
        }
        let actual = read(outputBuffer, intermediate)
        for index in 0..<intermediate {
            let expected = Inkling2Math.softcappedSwiGLU(
                gate: gateUp[index], up: gateUp[intermediate + index],
                gateSoftcap: 4, upSoftcap: 25
            )
            // 1e-4: the values reach ~25, where one FP32 ulp is ~2e-6, and
            // tanh's GPU and libm forms differ in the last couple of ulps.
            XCTAssertEqual(actual[index], expected, accuracy: 1e-4, "element \(index)")
        }
        // Both caps actually bound something in this fixture, so the case is
        // not passing on a straight-through path.
        XCTAssertGreaterThan(abs(gateUp[intermediate...].max() ?? 0), 25)
    }

    /// The static short convolution at a kernel shorter than its history, so
    /// both the `age == 0` branch and a saturated history run.
    func testTheShortConvolutionMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let channels = 7, tapCount = 4, historyRows = 3
        let current = values(channels, seed: 6)
        let history = values(historyRows * channels, seed: 7)
        let taps = values(channels * tapCount, seed: 8)
        let currentBuffer = try buffer(device, current)
        let historyBuffer = try buffer(device, history)
        let tapsBuffer = try buffer(device, taps)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: channels))
        let channelsBuffer = try buffer(device, [UInt32(channels)])
        let tapsCountBuffer = try buffer(device, [UInt32(tapCount)])
        let historyRowsBuffer = try buffer(device, [UInt32(historyRows)])
        try run("inkling_short_conv_f32", threads: (channels, 1, 1), threadgroup: (32, 1, 1)) { encoder in
            encoder.setBuffer(currentBuffer, offset: 0, index: 0)
            encoder.setBuffer(historyBuffer, offset: 0, index: 1)
            encoder.setBuffer(tapsBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            encoder.setBuffer(channelsBuffer, offset: 0, index: 4)
            encoder.setBuffer(tapsCountBuffer, offset: 0, index: 5)
            encoder.setBuffer(historyRowsBuffer, offset: 0, index: 6)
        }
        let actual = read(outputBuffer, channels)
        // The host primitive takes BACKWARD-DISTANCE taps; the kernel reads the
        // stored conv1d order, so the host call reverses -- the same reversal
        // the forward does at its call site.
        for channel in 0..<channels {
            let channelHistory = (0..<historyRows).map { history[$0 * channels + channel] }
            let channelTaps = (0..<tapCount).map { taps[channel * tapCount + tapCount - 1 - $0] }
            let expected = Inkling2Math.staticShortConvolution(
                current: current[channel], history: channelHistory, weights: channelTaps
            )
            XCTAssertEqual(actual[channel], expected, accuracy: 1e-6, "channel \(channel)")
        }
    }

    /// The DYNAMIC short convolution, which only full-attention layers run.
    ///
    /// Its two tap sets are indexed differently from the static kernel's, and
    /// reading it as that kernel was the bug that left every local layer exact
    /// and the first full one 5.6e-2 off: `static_weight` is `[kernel][channel]`
    /// with index 0 the CURRENT token, and `weight_proj`'s rows are
    /// `[head][lag]`. A channel-major read transposes the static taps and a
    /// current-first pairing reverses both sets, so this fixture's per-channel
    /// and per-tap values are distinct.
    func testTheDynamicShortConvolutionMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let headDim = 3, heads = 2, channels = headDim * heads
        let tapCount = 4, historyRows = 3
        let current = values(channels, seed: 31)
        let history = values(historyRows * channels, seed: 32)
        let staticTaps = values(tapCount * channels, seed: 33)
        let dynamicTaps = values(heads * tapCount, seed: 34)
        let currentBuffer = try buffer(device, current)
        let historyBuffer = try buffer(device, history)
        let staticBuffer = try buffer(device, staticTaps)
        let dynamicBuffer = try buffer(device, dynamicTaps)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: channels))
        let channelsBuffer = try buffer(device, [UInt32(channels)])
        let tapsCountBuffer = try buffer(device, [UInt32(tapCount)])
        let historyRowsBuffer = try buffer(device, [UInt32(historyRows)])
        let headDimBuffer = try buffer(device, [UInt32(headDim)])
        try run("inkling_dynamic_conv_f32", threads: (channels, 1, 1), threadgroup: (32, 1, 1)) { encoder in
            encoder.setBuffer(currentBuffer, offset: 0, index: 0)
            encoder.setBuffer(historyBuffer, offset: 0, index: 1)
            encoder.setBuffer(staticBuffer, offset: 0, index: 2)
            encoder.setBuffer(dynamicBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputBuffer, offset: 0, index: 4)
            encoder.setBuffer(channelsBuffer, offset: 0, index: 5)
            encoder.setBuffer(tapsCountBuffer, offset: 0, index: 6)
            encoder.setBuffer(historyRowsBuffer, offset: 0, index: 7)
            encoder.setBuffer(headDimBuffer, offset: 0, index: 8)
        }
        let actual = read(outputBuffer, channels)
        for channel in 0..<channels {
            let head = channel / headDim
            let channelHistory = (0..<historyRows).map { history[$0 * channels + channel] }
            let expected = Inkling2Math.dynamicShortConvolution(
                current: current[channel],
                history: channelHistory,
                staticWeights: (0..<tapCount).map { staticTaps[$0 * channels + channel] },
                dynamicWeights: (0..<tapCount).map { dynamicTaps[head * tapCount + $0] }
            )
            XCTAssertEqual(actual[channel], expected, accuracy: 1e-6, "channel \(channel)")
        }
    }

    /// The message gate, at a width that is not a multiple of the threadgroup.
    func testTheMessageGateMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let width = 69
        let current = values(width, seed: 9)
        let message = values(width, seed: 10)
        let forget = values(width, seed: 11)
        let gain = values(width, seed: 12)
        let buffers = try [current, message, forget, gain].map { try buffer(device, $0) }
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: width))
        let widthBuffer = try buffer(device, [UInt32(width)])
        try run("inkling_message_gate_f32", threads: (width, 1, 1), threadgroup: (32, 1, 1)) { encoder in
            for (index, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: index) }
            encoder.setBuffer(outputBuffer, offset: 0, index: 4)
            encoder.setBuffer(widthBuffer, offset: 0, index: 5)
        }
        let actual = read(outputBuffer, width)
        let expected = Inkling2Math.messageGate(current: current, message: message, forget: forget, gain: gain)
        for index in 0..<width {
            XCTAssertEqual(actual[index], expected[index], accuracy: 1e-6, "element \(index)")
        }
    }

    // MARK: Attention

    /// The relative-bias/sink attention against the host rule, at a shape with
    /// grouped KV heads (2 query heads per KV head), a window start past zero
    /// and a row count past `extent` so the bias is zeroed on the oldest rows.
    /// The relative vector and bank are wide-ranging, so a bias applied to the
    /// wrong row or with the wrong distance cannot cancel.
    func testTheRelativeBiasSinkAttentionMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let heads = 4, groups = 2, headDim = 16, kvHeads = heads / groups
        let kvWidth = kvHeads * headDim
        let keyRows = 12, windowStart = 7, position = 15, dRel = 4, extent = 8
        let query = values(heads * headDim, seed: 21)
        let keys = values(keyRows * kvWidth, seed: 22)
        let vals = values(keyRows * kvWidth, seed: 23)
        let relative = values(heads * dRel, seed: 24)
        let bank = values(dRel * extent, seed: 25)
        let sink = values(heads, seed: 26)

        let queryBuffer = try buffer(device, query)
        let keyBuffer = try buffer(device, keys)
        let valueBuffer = try buffer(device, vals)
        let relativeBuffer = try buffer(device, relative)
        let bankBuffer = try buffer(device, bank)
        let sinkBuffer = try buffer(device, sink)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: heads * headDim))
        let scalars: [UInt32] = [UInt32(heads), UInt32(groups), UInt32(headDim), UInt32(kvWidth),
                                UInt32(keyRows), UInt32(windowStart), UInt32(position),
                                UInt32(dRel), UInt32(extent)]
        let scalarBuffers = try scalars.enumerated().map { index, value in
            try buffer(device, [value])
        }
        try run("inkling_attention_f32", threads: (heads, 1, 1), threadgroup: (32, 1, 1),
                threadgroupMemory: (32 + keyRows) * MemoryLayout<Float>.stride,
                byThreadgroup: true) { encoder in
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)
            encoder.setBuffer(valueBuffer, offset: 0, index: 2)
            encoder.setBuffer(relativeBuffer, offset: 0, index: 3)
            encoder.setBuffer(bankBuffer, offset: 0, index: 4)
            encoder.setBuffer(sinkBuffer, offset: 0, index: 5)
            encoder.setBuffer(outputBuffer, offset: 0, index: 6)
            for (index, b) in scalarBuffers.enumerated() { encoder.setBuffer(b, offset: 0, index: 7 + index) }
        }
        let actual = read(outputBuffer, heads * headDim)

        // The host rule, written from Inkling2Forward.attention's own lines.
        var expected = [Float](repeating: 0, count: heads * headDim)
        for head in 0..<heads {
            let kvHead = head / groups
            let q = Array(query[(head * headDim)..<((head + 1) * headDim)])
            var scores = [Float](repeating: 0, count: keyRows)
            for row in 0..<keyRows {
                var dot: Float = 0
                for d in 0..<headDim { dot += q[d] * keys[row * kvWidth + kvHead * headDim + d] }
                let distance = position - (windowStart + row)
                var bias: Float = 0
                if distance >= 0 && distance < extent {
                    for r in 0..<dRel { bias += relative[head * dRel + r] * bank[r * extent + distance] }
                }
                scores[row] = dot / Float(headDim).squareRoot() + bias
            }
            var combined = scores
            combined.append(sink[head])
            let maximum = combined.max() ?? 0
            let exponentials = combined.map { exp($0 - maximum) }
            let denominator = exponentials.reduce(0, +)
            for row in 0..<keyRows {
                let weight = exponentials[row] / denominator
                for d in 0..<headDim {
                    expected[head * headDim + d] += weight * vals[row * kvWidth + kvHead * headDim + d]
                }
            }
        }
        for index in 0..<(heads * headDim) {
            XCTAssertEqual(actual[index], expected[index], accuracy: 2e-5, "element \(index)")
        }
        // The fixture really does exercise both branches: some rows inside the
        // extent and some zeroed outside it.
        XCTAssertGreaterThan(position, windowStart + 0)
        XCTAssertLessThan(position - (windowStart + Int(keyRows) - 1), 0)
    }

    // MARK: Experts

    /// The block-scaled expert row against `Inkling2PackedExpertMatrix.project`,
    /// the host rule the mixture uses, for BOTH layouts.
    ///
    /// The NVFP4 case is the one that matters: nothing on the device reads that
    /// packing yet, and its quirks (two values per byte, low nibble first, 16
    /// blocks, an outer amax scale, a BF16 rounding of each product) are each a
    /// way to be quietly wrong. The column count is odd-block aware (a partial
    /// last block) and the row stride is wider than the logical row, so a
    /// stride bug cannot hide.
    func testTheExpertRowMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        for layout in [Inkling2ExpertQuantLayout.nvfp4, .mxfp8] {
            let rows = 2, logicalColumns = 40
            let blockSize = layout.blockSize
            let valuesPerByte = layout == .nvfp4 ? 2 : 1
            let packedColumns = (logicalColumns + valuesPerByte - 1) / valuesPerByte
            let weightRowStride = packedColumns + 3       // padded: a stride bug must fail
            let scaleColumns = (logicalColumns + blockSize - 1) / blockSize
            let scaleRowStride = scaleColumns + 1
            let amax: Float = 3.5, denominator: Float = 4.0

            // Bytes whose exponents stay finite, so a decoded value is a number.
            // The bank carries a leading pad the kernel must skip: an expert's
            // rows start partway into a fused tensor, and a kernel that ignored
            // the offset would read the pad as weights and still be plausible.
            let weightPad = 5, scalePad = 3
            var weight = [UInt8](repeating: 0x7F, count: weightPad + rows * weightRowStride)
            for index in 0..<(rows * weightRowStride) {
                weight[weightPad + index] = UInt8(0x21 + (index &* 13) % 0x5C)
            }
            var scales = [UInt8](repeating: 0x7F, count: scalePad + rows * scaleRowStride)
            for index in 0..<(rows * scaleRowStride) {
                // E4M3 near 2^0 for NVFP4; E8M0 (a biased exponent) for MXFP8.
                scales[scalePad + index] = layout == .nvfp4 ? UInt8(118 + index % 9) : UInt8(124 + index % 5)
            }
            let input = values(logicalColumns, seed: 31)

            // The host rule reads the SAME slice the kernel does: the pad is
            // addressing, not data.
            let matrix = Inkling2PackedExpertMatrix(
                layout: layout, weight: Array(weight[weightPad...]), weightRowStride: weightRowStride,
                scales: Array(scales[scalePad...]), scaleRowStride: scaleRowStride,
                rows: rows, logicalColumns: logicalColumns,
                expertAmax: amax, outerScaleDenominator: denominator
            )
            let expected = (0..<rows).map { matrix.project(row: $0, input: input) }

            let weightBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: weight, length: weight.count, options: .storageModeShared))
            let scaleBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: scales, length: scales.count, options: .storageModeShared))
            let inputBuffer = try buffer(device, input)
            let outputBuffer = try buffer(device, [Float](repeating: 0, count: rows))
            let scalars: [UInt32] = [
                UInt32(logicalColumns), UInt32(weightRowStride), UInt32(scaleRowStride), UInt32(blockSize)
            ]
            let scalarBuffers = try scalars.enumerated().map { index, value in
                try buffer(device, [value])
            }
            let layoutBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: [layout == .nvfp4], length: 1, options: .storageModeShared))
            let outerScale = layout.requiresAmaxDenominator ? amax / denominator : 1
            let outerBuffer = try buffer(device, [outerScale])
            let weightOffsetBuffer = try buffer(device, [UInt64(weightPad)])
            let scaleOffsetBuffer = try buffer(device, [UInt64(scalePad)])

            try run("inkling_expert_row_f32", threads: (rows, 1, 1), threadgroup: (64, 1, 1),
                    threadgroupMemory: 32 * MemoryLayout<Float>.stride, byThreadgroup: true) { encoder in
                encoder.setBuffer(weightBuffer, offset: 0, index: 0)
                encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(inputBuffer, offset: 0, index: 2)
                encoder.setBuffer(outputBuffer, offset: 0, index: 3)
                for (index, b) in scalarBuffers.enumerated() { encoder.setBuffer(b, offset: 0, index: 4 + index) }
                encoder.setBuffer(layoutBuffer, offset: 0, index: 8)
                encoder.setBuffer(outerBuffer, offset: 0, index: 9)
                encoder.setBuffer(weightOffsetBuffer, offset: 0, index: 10)
                encoder.setBuffer(scaleOffsetBuffer, offset: 0, index: 11)
            }
            let actual = read(outputBuffer, rows)
            for row in 0..<rows {
                XCTAssertEqual(
                    actual[row], expected[row],
                    accuracy: max(1e-4, abs(expected[row]) * 1e-3),
                    "\(layout) row \(row)"
                )
            }
        }
    }

    /// A resident packed slot projects exactly as the host rule says, for BOTH
    /// regions and BOTH layouts.
    ///
    /// This is the binding the arena needs before it can be wired into the
    /// mixture: a slot is ONE buffer holding four regions at the geometry's own
    /// offsets, and the kernel addresses each region from the slot's base. The
    /// fixture is built so a plausible bug fails: the two regions have
    /// DIFFERENT row strides (as the real banks do), the slot has a leading pad
    /// and a second slot behind it, and the bytes are wide-ranging -- so a
    /// kernel that read the down region with the gate/up stride, or that ignored
    /// the slot's base offset, or that read slot 0 when asked for slot 1, is a
    /// wrong value rather than a plausible one.
    func testAResidentSlotProjectsAsTheHostRuleSays() throws {
        let (device, library, queue) = try environment()
        let pipeline = try XCTUnwrap(try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "inkling_expert_row_f32"))
        ))
        for layout in [Inkling2ExpertQuantLayout.nvfp4, .mxfp8] {
            let valuesPerByte = layout.valuesPerByte
            // Gate/up: 4 rows of 40 logical columns; down: 3 rows of 24, so the
            // two regions cannot share a stride by accident.
            let gateUpRows = 4, gateUpColumns = 40
            let downRows = 3, downColumns = 24
            let gateUpPacked = gateUpColumns / valuesPerByte + 2
            let downPacked = downColumns / valuesPerByte + 1
            let gateUpScaleStride = gateUpColumns / layout.blockSize + 1
            let downScaleStride = downColumns / layout.blockSize + 1
            let geometry = try XCTUnwrap(Inkling2ExpertSlotGeometry.make(
                gateUpRows: gateUpRows, gateUpRowStride: gateUpPacked,
                gateUpScaleRowStride: gateUpScaleStride,
                downRows: downRows, downRowStride: downPacked,
                downScaleRowStride: downScaleStride
            ))
            let address = geometry.slotAddress
            XCTAssertEqual(address.gateUpWeightRowStride, gateUpPacked)
            XCTAssertEqual(address.downWeightRowStride, downPacked)
            XCTAssertEqual(address.gateUpRows, gateUpRows)
            XCTAssertEqual(address.downRows, downRows)

            // Two slots. The scale regions hold small exponents (E4M3 near 2^0
            // for NVFP4, E8M0 near 2^1 for MXFP8) so no row cancels to noise,
            // and the weight regions are wide-ranging patterns -- each slot's
            // own, so a base-offset bug reads the neighbour's bytes.
            let slots = 2
            var slotBytes = [UInt8](repeating: 0, count: slots * geometry.slotBytes)
            for slot in 0..<slots {
                let base = slot * geometry.slotBytes
                func isScale(_ index: Int) -> Bool {
                    (index >= geometry.gateUpScaleOffset && index < geometry.downWeightOffset)
                        || index >= geometry.downScaleOffset
                }
                for index in 0..<geometry.slotBytes {
                    if isScale(index) {
                        slotBytes[base + index] = layout == .nvfp4
                            ? UInt8(122 + (index &+ slot) % 5)   // E4M3 around 1.0
                            : UInt8(126 + (index &+ slot) % 3)   // E8M0 around 2^-1..2^1
                    } else {
                        slotBytes[base + index] = UInt8(0x01 + (index &* 7 &+ slot &* 53) % 0x7E)
                    }
                }
            }
            let slotBuffer = try XCTUnwrap(device.makeBuffer(
                bytes: slotBytes, length: slotBytes.count, options: .storageModeShared
            ))
            let arena = Inkling2ScalarArena(device: device)
            defer { arena.clear() }

            for (isGateUp, rows, columns, weightOffset, scaleOffset, weightStride, scaleStride) in [
                (true, gateUpRows, gateUpColumns, address.gateUpWeight, address.gateUpScale,
                 address.gateUpWeightRowStride, address.gateUpScaleRowStride),
                (false, downRows, downColumns, address.downWeight, address.downScale,
                 address.downWeightRowStride, address.downScaleRowStride)
            ] {
                let input = values(columns, seed: isGateUp ? 11 : 13)
                let inputBuffer = try buffer(device, input)
                let outputBuffer = try XCTUnwrap(device.makeBuffer(
                    length: rows * MemoryLayout<Float>.stride, options: .storageModeShared
                ))
                let amax: Float = 3.5, denominator: Float = 4
                let outerScale = layout.requiresAmaxDenominator ? amax / denominator : 1
                let slotIndex = 1
                let base = slotIndex * geometry.slotBytes
                let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
                let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
                Inkling2SlotProjection.encode(
                    pipeline: pipeline, encoder: encoder,
                    slot: slotBuffer, slotBaseOffset: base,
                    address: address, isGateUp: isGateUp,
                    input: inputBuffer, output: outputBuffer,
                    rows: rows, logicalColumns: columns,
                    layout: layout, outerScale: outerScale, arena: arena
                )
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if let error = commandBuffer.error { throw error }

                // The host rule reads the SAME bytes the kernel did: the slot's
                // region at its own offset, row by row, with that region's
                // stride.
                let raw = slotBytes
                func byte(_ index: Int) -> UInt8 { raw[base + index] }
                let matrix = Inkling2PackedExpertMatrix(
                    layout: layout,
                    weight: (0..<(rows * weightStride)).map { byte(weightOffset + $0) },
                    weightRowStride: weightStride,
                    scales: (0..<(rows * scaleStride)).map { byte(scaleOffset + $0) },
                    scaleRowStride: scaleStride,
                    rows: rows, logicalColumns: columns,
                    expertAmax: amax, outerScaleDenominator: denominator
                )
                let actual = read(outputBuffer, rows)
                for row in 0..<rows {
                    let expected = matrix.project(row: row, input: input)
                    XCTAssertEqual(
                        actual[row], expected,
                        accuracy: max(1e-4, abs(expected) * 1e-3),
                        "\(layout) \(isGateUp ? "gate/up" : "down") row \(row)"
                    )
                }
            }
        }
    }

    /// A slot filled from a bank file projects exactly as the bank itself does.
    ///
    /// This is the equivalence the mixture's arena path rests on: a resident
    /// slot is a byte copy of the expert's four ranges, so the projection that
    /// reads it must equal the one that reads the fused bank at that expert's
    /// offset. It is checked on a temp file standing in for a shard (the reader
    /// is the same POSIX call either way), so it runs with no checkpoint.
    func testASlotFilledFromABankProjectsAsTheBankDoes() throws {
        let (device, _, queue) = try environment()
        guard let library = device.makeDefaultLibrary() else { throw XCTSkip("no default library") }
        let pipeline = try XCTUnwrap(try device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "inkling_expert_row_f32"))
        ))
        let layout = Inkling2ExpertQuantLayout.nvfp4
        let gateUpRows = 4, gateUpColumns = 32
        let downRows = 3, downColumns = 16
        let gateUpPacked = gateUpColumns / 2 + 2
        let downPacked = downColumns / 2 + 1
        let gateUpScaleStride = gateUpColumns / layout.blockSize + 1
        let downScaleStride = downColumns / layout.blockSize + 1
        let geometry = try XCTUnwrap(Inkling2ExpertSlotGeometry.make(
            gateUpRows: gateUpRows, gateUpRowStride: gateUpPacked,
            gateUpScaleRowStride: gateUpScaleStride,
            downRows: downRows, downRowStride: downPacked,
            downScaleRowStride: downScaleStride
        ))
        // A "shard": one expert's four ranges, in the arena's own order, behind
        // a leading pad so nothing is accidentally at offset zero.
        let pad = 16
        var file = [UInt8](repeating: 0xAB, count: pad)
        let gateUpWeight = (0..<geometry.gateUpWeightBytes).map { UInt8(0x11 + $0 % 0x40) }
        let gateUpScale = (0..<geometry.gateUpScaleBytes).map { UInt8(120 + $0 % 6) }
        let downWeight = (0..<geometry.downWeightBytes).map { UInt8(0x31 + $0 % 0x50) }
        let downScale = (0..<geometry.downScaleBytes).map { UInt8(119 + $0 % 7) }
        file += gateUpWeight + gateUpScale + downWeight + downScale
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-slot-bank-\(UUID().uuidString).bin")
        try Data(file).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try Inkling2ArenaShardReader(url: url)
        let gateUpBase = UInt64(pad)
        let gateUpScaleBase = gateUpBase + UInt64(gateUpWeight.count)
        let downBase = gateUpScaleBase + UInt64(gateUpScale.count)
        let downScaleBase = downBase + UInt64(downWeight.count)

        let source = try XCTUnwrap(device.makeBuffer(
            bytes: file, length: file.count, options: .storageModeShared
        ))
        let arena = Inkling2ExpertArena(capacity: 2, geometry: geometry)
        let slotBuffer = try XCTUnwrap(device.makeBuffer(
            length: 2 * geometry.slotBytes, options: .storageModeShared
        ))
        arena.attachSlotBuffer(slotBuffer)
        let slot = try arena.slot(
            layer: 0, expert: 0, reader: reader,
            gateUpSourceOffset: gateUpBase, gateUpScaleSourceOffset: gateUpScaleBase,
            downSourceOffset: downBase, downScaleSourceOffset: downScaleBase
        )
        // The slot must be a byte copy of the bank, not merely something that
        // reads plausibly.
        let slotPointer = slotBuffer.contents().bindMemory(to: UInt8.self, capacity: 2 * geometry.slotBytes)
        let filePointer = source.contents().bindMemory(to: UInt8.self, capacity: file.count)
        for index in 0..<geometry.slotBytes {
            XCTAssertEqual(
                slotPointer[slot * geometry.slotBytes + index], filePointer[pad + index],
                "slot byte \(index)"
            )
        }

        let scalars = Inkling2ScalarArena(device: device)
        defer { scalars.clear() }
        let address = geometry.slotAddress
        for (isGateUp, rows, columns) in [
            (true, gateUpRows, gateUpColumns), (false, downRows, downColumns)
        ] {
            let input = values(columns, seed: isGateUp ? 21 : 23)
            let inputBuffer = try buffer(device, input)
            let fromSlot = try XCTUnwrap(device.makeBuffer(
                length: rows * MemoryLayout<Float>.stride, options: .storageModeShared
            ))
            let fromBank = try XCTUnwrap(device.makeBuffer(
                length: rows * MemoryLayout<Float>.stride, options: .storageModeShared
            ))
            let outerScale: Float = 0.75
            let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
            Inkling2SlotProjection.encode(
                pipeline: pipeline, encoder: encoder,
                slot: slotBuffer, slotBaseOffset: slot * geometry.slotBytes,
                address: address, isGateUp: isGateUp,
                input: inputBuffer, output: fromSlot,
                rows: rows, logicalColumns: columns,
                layout: layout, outerScale: outerScale, arena: scalars
            )
            // The same projection off the bank, addressed directly: the shard
            // buffer, the expert's region offset, the bank's own strides.
            let region = address.projection(isGateUp: isGateUp)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(source, offset: 0, index: 0)
            encoder.setBuffer(source, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(fromBank, offset: 0, index: 3)
            encoder.setBuffer(scalars.make(UInt32(columns)), offset: 0, index: 4)
            encoder.setBuffer(scalars.make(UInt32(region.weightRowStride)), offset: 0, index: 5)
            encoder.setBuffer(scalars.make(UInt32(region.scaleRowStride)), offset: 0, index: 6)
            encoder.setBuffer(scalars.make(UInt32(layout.blockSize)), offset: 0, index: 7)
            encoder.setBuffer(scalars.make(true), offset: 0, index: 8)
            encoder.setBuffer(scalars.make(outerScale), offset: 0, index: 9)
            encoder.setBuffer(scalars.make(gateUpBase + UInt64(region.weightOffset)), offset: 0, index: 10)
            encoder.setBuffer(scalars.make(gateUpBase + UInt64(region.scaleOffset)), offset: 0, index: 11)
            encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
            let resident = read(fromSlot, rows)
            let direct = read(fromBank, rows)
            for row in 0..<rows {
                XCTAssertEqual(
                    resident[row], direct[row],
                    accuracy: max(1e-5, abs(direct[row]) * 1e-5),
                    "\(isGateUp ? "gate/up" : "down") row \(row): the slot must be the bank"
                )
            }
        }
    }

    /// A REAL expert bank's slot, filled by `pread` from the checkpoint file,
    /// byte for byte.
    ///
    /// The slot is filled from FILE offsets while the mixture reads SEGMENT
    /// offsets: on a machine whose shard is split to fit `maxBufferLength` the
    /// two differ by the segment's origin, and a slot filled without it holds a
    /// plausible range of the wrong bytes -- no crash, no error, just a wrong
    /// model. That is the failure this pins, so it is checked against the mmap
    /// the mixture uses rather than against itself.
    func testAPublishedBankFillsItsSlotWithTheBanksOwnBytes() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, _, _) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let rawSummary = try engine.getSummary()
        let summary = try inkling2MetalReadySummary(rawSummary)
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let origins = InferenceEngine.shardFileOrigins(original: rawSummary, segmented: summary)
        let weights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers, fileOrigins: origins
        )
        let layer = 0, expert = 1
        let gateUp = try weights.expertBank(.expertW13, layer: layer, expert: expert)
        let down = try weights.expertBank(.expertW2, layer: layer, expert: expert)
        let geometry = try XCTUnwrap(Inkling2ExpertSlotGeometry.make(
            gateUpRows: 2 * config.moeIntermediateSize,
            gateUpRowStride: gateUp.rowStride,
            gateUpScaleRowStride: gateUp.scaleRowStride,
            downRows: config.hiddenSize,
            downRowStride: down.rowStride,
            downScaleRowStride: down.scaleRowStride
        ))

        // A slot is a copy at FIXED strides, and the published checkpoint mixes
        // layouts: layers 0...19 are NVFP4 (a 1,152-byte packed `w13` row) and
        // the last four are MXFP8 (2,304 bytes). The arena must refuse the
        // latter rather than copy the first half of each row into a slot.
        XCTAssertTrue(geometry.holds(
            weightRowStride: gateUp.rowStride, scaleRowStride: gateUp.scaleRowStride,
            downWeightRowStride: down.rowStride, downScaleRowStride: down.scaleRowStride
        ))
        let lateGateUp = try weights.expertBank(
            .expertW13, layer: config.numHiddenLayers - 1, expert: 0
        )
        let lateDown = try weights.expertBank(
            .expertW2, layer: config.numHiddenLayers - 1, expert: 0
        )
        XCTAssertNotEqual(
            lateGateUp.rowStride, gateUp.rowStride,
            "the published checkpoint is expected to mix quantized layouts between layers"
        )
        XCTAssertFalse(geometry.holds(
            weightRowStride: lateGateUp.rowStride, scaleRowStride: lateGateUp.scaleRowStride,
            downWeightRowStride: lateDown.rowStride, downScaleRowStride: lateDown.scaleRowStride
        ), "an MXFP8 bank must not be copied into an NVFP4 slot")


        // The checkpoint's weights file: the largest safetensors beside the
        // config, and the one the catalog's offsets are relative to.
        let directory = snapshot.resolvingSymlinksInPath()
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey]
        )
        let shardURL = try XCTUnwrap(files.filter { $0.pathExtension == "safetensors" }.max { lhs, rhs in
            let left = (try? lhs.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            let right = (try? rhs.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return left < right
        })
        let reader = try Inkling2ArenaShardReader(url: shardURL)

        let arena = Inkling2ExpertArena(capacity: 1, geometry: geometry)
        let slotBuffer = try XCTUnwrap(device.makeBuffer(
            length: geometry.slotBytes, options: .storageModeShared
        ))
        arena.attachSlotBuffer(slotBuffer)
        let slot = try arena.slot(
            layer: layer, expert: expert, reader: reader,
            gateUpSourceOffset: gateUp.fileWeightOffset,
            gateUpScaleSourceOffset: gateUp.fileScaleOffset,
            downSourceOffset: down.fileWeightOffset,
            downScaleSourceOffset: down.fileScaleOffset
        )
        let slotPointer = slotBuffer.contents().bindMemory(to: UInt8.self, capacity: geometry.slotBytes)

        /// Every byte of a range, against the mmap the mixture reads.
        func compare(
            _ label: String, buffer: MTLBuffer, offset: UInt64, slotOffset: Int, length: Int
        ) {
            let source = buffer.contents().advanced(by: Int(offset))
                .bindMemory(to: UInt8.self, capacity: length)
            var differing = 0
            var firstDifference = -1
            for index in 0..<length where slotPointer[slot * geometry.slotBytes + slotOffset + index] != source[index] {
                if firstDifference < 0 { firstDifference = index }
                differing += 1
            }
            XCTAssertEqual(
                differing, 0,
                "\(label): \(differing) of \(length) bytes differ from the mmap"
                    + (firstDifference >= 0 ? ", first at \(firstDifference)" : "")
            )
        }
        compare(
            "gate/up weight", buffer: gateUp.buffer, offset: gateUp.baseOffset + gateUp.expertOffset,
            slotOffset: geometry.gateUpWeightOffset, length: geometry.gateUpWeightBytes
        )
        compare(
            "gate/up scale", buffer: gateUp.scaleBuffer,
            offset: gateUp.scaleBaseOffset + gateUp.scaleExpertOffset,
            slotOffset: geometry.gateUpScaleOffset, length: geometry.gateUpScaleBytes
        )
        compare(
            "down weight", buffer: down.buffer, offset: down.baseOffset + down.expertOffset,
            slotOffset: geometry.downWeightOffset, length: geometry.downWeightBytes
        )
        compare(
            "down scale", buffer: down.scaleBuffer,
            offset: down.scaleBaseOffset + down.scaleExpertOffset,
            slotOffset: geometry.downScaleOffset, length: geometry.downScaleBytes
        )
        // A slot of zeros would pass a comparison against another zeroed file,
        // so the weight range is also required to be non-trivial.
        XCTAssertGreaterThan(Set((0..<min(4096, geometry.gateUpWeightBytes))
            .map { slotPointer[slot * geometry.slotBytes + $0] }).count, 8)

        // And the PROJECTION, which is what the mixture actually consumes: the
        // slot read through `inkling_expert_row_f32` must agree with the HOST
        // rule (`Inkling2PackedExpertMatrix.project`), which is the oracle the
        // checkpoint's own reference matches.
        let (_, library, queue) = try environment()
        let hostSource = Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers
        )
        let host = try Inkling2CPUForward(config: config, weights: hostSource)
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: weights, host: host
        )
        let scalars = Inkling2ScalarArena(device: device)
        defer { scalars.clear() }
        for (isGateUp, rows, columns) in [
            (true, 2 * config.moeIntermediateSize, config.hiddenSize),
            (false, config.hiddenSize, config.moeIntermediateSize)
        ] {
            let input = values(columns, seed: isGateUp ? 61 : 62)
            let inputBuffer = try buffer(device, input)
            let fromSlot = try XCTUnwrap(device.makeBuffer(
                length: rows * MemoryLayout<Float>.stride, options: .storageModeShared
            ))
            let bank = isGateUp ? gateUp : down
            let outerScale: Float = bank.layout.requiresAmaxDenominator && bank.denominator > 0
                ? bank.amax / bank.denominator : 1
            let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
            try metal.slotExpertRow(
                encoder: encoder, slot: slotBuffer, slotBaseOffset: slot * geometry.slotBytes,
                address: geometry.slotAddress, isGateUp: isGateUp,
                input: inputBuffer, output: fromSlot, rows: rows, logicalColumns: columns,
                layout: bank.layout, outerScale: outerScale, arena: scalars
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }

            let resident = read(fromSlot, rows)
            let hostMatrix = try hostSource.expertMatrix(
                isGateUp ? .expertW13 : .expertW2, layer: layer, expert: expert
            )
            var worst: Float = 0
            var worstRow = 0
            var scale: Float = 0
            for row in 0..<rows {
                // A few rows through the host rule, not all of them: the host
                // dequantizes 2,304 x 2,304 values per call, and the first rows
                // already separate a wrong stride from an accumulation order.
                let expected: Float = row < 8 ? hostMatrix.project(row: row, input: input) : resident[row]
                scale = max(scale, abs(expected))
                let delta = abs(resident[row] - expected)
                if delta > worst { worst = delta; worstRow = row }
            }
            print(
                String(
                    format: "🔎 [inkling2 slot] %@: worst row %d of %d at %.2e absolute (scale %.2e)",
                    isGateUp ? "gate/up" : "down", worstRow, rows, worst, scale
                )
            )
            XCTAssertLessThanOrEqual(
                worst, max(1e-5, scale * 1e-5),
                "\(isGateUp ? "gate/up" : "down") row \(worstRow): a resident slot must project as the host rule"
            )
        }
        withExtendedLifetime(engine) {}
    }

    // MARK: Routing

    /// The router's selection and weighting against `Inkling2Math.route`.
    ///
    /// The fixture separates the bias's two possible roles. Expert 3 has the
    /// largest raw score and a moderate negative bias: it is still selected,
    /// so the bias cannot be dropping the true maximum, and its WEIGHT is the
    /// raw score's share rather than the steered one's (a softmax over steered
    /// values would give it a different number). Expert 5 has a small raw
    /// score and a large positive bias, so it is selected although it ranks
    /// low by score alone -- the bias does steer selection.
    func testTheRouterMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let numExperts = 8, topK = 3, positions = 2
        let routeScale: Float = 2.5
        let base: [Float] = [0.9, 0.5, 0.3, 1.2, 0.2, 0.25, 0.1, 0.05]
        let bias: [Float] = [0.1, 0.1, 0.0, -0.4, 0.0, 0.9, 0.0, 0.0]
        // A second position with a different order, so a kernel that ignored
        // the position offset would fail.
        let second: [Float] = [0.05, 0.1, 0.25, 0.2, 1.2, 0.3, 0.5, 0.9]
        let scores = base + second

        let scoresBuffer = try buffer(device, scores)
        // The bias carries a leading pad: the checkpoint's bias lives partway
        // into its shard, and a kernel reading from zero would only pass if it
        // were first. The pad is wide-ranging so a kernel that ignored the
        // offset would move the selection.
        let biasPad = 3
        let biasBuffer = try buffer(device, [Float](repeating: 9.5, count: biasPad) + bias)
        let indicesBuffer = try buffer(device, [UInt32](repeating: 99, count: positions * topK))
        let weightsBuffer = try buffer(device, [Float](repeating: 0, count: positions * topK))
        let scalars = [UInt32(numExperts), UInt32(topK)]
        let scalarBuffers = try scalars.map { try buffer(device, [$0]) }
        let scaleBuffer = try buffer(device, [routeScale])
        // A non-zero bias offset: the checkpoint's bias lives partway into its
        // shard, and a kernel that read from zero would be exact only if it
        // happened to be first.
        let biasOffsetBuffer = try buffer(device, [UInt64(biasPad) * UInt64(MemoryLayout<Float>.stride)])
        // The kernel's scratch: warp sums, the ranks padded to a threadgroup
        // stride, the chosen ids, and one (rank, id) pair per warp.
        let rankStride = ((numExperts + 63) / 64) * 64
        try run("inkling_router_topk_f32", threads: (positions, 1, 1), threadgroup: (64, 1, 1),
                threadgroupMemory: (32 + rankStride + 32 + 16 + 16) * MemoryLayout<Float>.stride,
                byThreadgroup: true) { encoder in
            encoder.setBuffer(scoresBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasBuffer, offset: 0, index: 1)
            encoder.setBuffer(indicesBuffer, offset: 0, index: 2)
            encoder.setBuffer(weightsBuffer, offset: 0, index: 3)
            for (index, b) in scalarBuffers.enumerated() { encoder.setBuffer(b, offset: 0, index: 4 + index) }
            encoder.setBuffer(scaleBuffer, offset: 0, index: 6)
            encoder.setBuffer(biasOffsetBuffer, offset: 0, index: 7)
        }
        let actualIndices = Array(UnsafeBufferPointer(
            start: indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: positions * topK),
            count: positions * topK
        )).map(Int.init)
        let actualWeights = read(weightsBuffer, positions * topK)

        for position in 0..<positions {
            let row = Array(scores[(position * numExperts)..<((position + 1) * numExperts)])
            let expected = Inkling2Math.route(scores: row, bias: bias, topK: topK, routeScale: routeScale)
            XCTAssertEqual(
                Array(actualIndices[(position * topK)..<((position + 1) * topK)]), expected.indices,
                "position \(position) selection"
            )
            for slot in 0..<topK {
                XCTAssertEqual(
                    actualWeights[position * topK + slot], expected.weights[slot], accuracy: 1e-6,
                    "position \(position) weight \(slot)"
                )
            }
        }
        // The fixture separates the bias's roles: the top raw score is not selected
        // by rank alone, and the positively biased expert is.
        let first = Inkling2Math.route(scores: base, bias: bias, topK: topK, routeScale: routeScale)
        XCTAssertTrue(first.indices.contains(3), "the largest raw score stays selected despite a negative bias")
        XCTAssertTrue(first.indices.contains(5), "the positively biased expert is selected despite a small score")
        // The weights are the RAW scores' shares: steered ones would give
        // expert 5 (bias +0.9) a much larger weight than expert 3.
        let slot3 = try XCTUnwrap(first.indices.firstIndex(of: 3))
        let slot5 = try XCTUnwrap(first.indices.firstIndex(of: 5))
        XCTAssertGreaterThan(first.weights[slot3], first.weights[slot5])
    }

    // MARK: Device against the host graph

    /// One real checkpoint's attention stage, on the device and through the
    /// host graph, compared elementwise.
    ///
    /// The host forward is the oracle (it is what matches the checkpoint's
    /// Transformers reference) and it records its own intermediate values
    /// through `onInternal`, so the comparison does not re-derive any input:
    /// the same query, key, value, relative vector and sink the host used are
    /// staged from the host's captures. The device reads the same zero-copy
    /// shard bytes for the weights.
    ///
    /// Layer 0 is a SLIDING layer (the shared-norm branch); it is the first
    /// visit, whose state holds the whole short prompt.
    func testTheDeviceAttentionMatchesTheHostGraph() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, library, queue) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let hostSource = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let host = try Inkling2CPUForward(config: config, weights: hostSource)
        let deviceWeights = Inkling2DeviceWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: host
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config)

        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        // Two tokens: the cache only needs to HOLD rows for the attention to
        // read them, and every extra position costs a host forward.
        let ids = try tokenizer.encode(text: "Hi there")
        for (position, id) in ids.enumerated() {
            _ = try host.forwardToken(token: UInt32(id), position: position)
        }
        let position = ids.count
        let layer = 0
        let state = host.states[0]

        // The host's own values for this stage, captured as it computes them.
        var hostScores: [Float] = []
        var hostWeights: [Float] = []
        var hostRaw: [Float] = []
        var hostQuery: [Float] = []
        var hostRelative: [Float] = []
        var hostSink: [Float] = []
        host.onInternal = { _, name, value in
            switch name {
            case "scores_head0": hostScores = value
            case "weights_head0": hostWeights = value
            case "attn_raw": hostRaw = value
            case "q_norm": hostQuery = value
            // The POST-clip vector, which is what the kernel's bias mixes: the
            // raw projection (`rel_linear`) is a different value.
            case "relative_states": hostRelative = value
            case "sink": hostSink = value
            default: break
            }
        }
        let queryInput = values(config.hiddenSize, seed: 41)
        let keyInput = values(config.hiddenSize, seed: 42)
        let valueInput = values(config.hiddenSize, seed: 43)
        let hostOutput = try host.attention(
            layer: layer, state: state, queryInput: queryInput,
            keyInput: keyInput, valueInput: valueInput, position: position
        )
        // The gate is applied after the capture, so `attn_raw` is the attention
        // kernel's own output -- exactly what the device kernel produces.
        XCTAssertEqual(hostRaw.count, config.numAttentionHeads * config.headDim)
        XCTAssertEqual(hostScores.count, state.keys.count)
        XCTAssertFalse(hostQuery.isEmpty)

        // The device: the same state staged, and the host's own captured
        // inputs (query, post-clip relative vector, sink) plus the same weights.
        let rows = try workspace.stage(state)
        XCTAssertEqual(rows, state.keys.count)

        // The query goes into the WORKSPACE, which is the buffer the attention
        // helper binds -- a separate buffer here would never be read, and the
        // kernel would run on a zeroed vector (which is what a first cut did).
        let queryPointer = workspace.query.contents().bindMemory(
            to: Float.self, capacity: hostQuery.count
        )
        for index in hostQuery.indices { queryPointer[index] = hostQuery[index] }
        let relativeBuffer = try buffer(device, hostRelative)
        let sinkBuffer = try buffer(device, hostSink)
        let bank = try hostSource.matrix(.attnRelLogitsProj, layer: layer)
        let bankBuffer = try buffer(device, bank)
        let outputBuffer = try buffer(
            device, [Float](repeating: 0, count: config.numAttentionHeads * config.headDim)
        )
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        let arena = Inkling2ScalarArena(device: device)
        try metal.attention(
            encoder: encoder, workspace: workspace, state: state, rows: rows,
            position: position, relative: relativeBuffer, bank: bankBuffer,
            sink: sinkBuffer, output: outputBuffer, arena: arena
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        arena.clear()

        let deviceOutput = read(outputBuffer, config.numAttentionHeads * config.headDim)
        let headDim = config.headDim
        // Per head: a wrong KV-head mapping is right on head 0 and wrong from
        // the first head that shares a different KV head, so an aggregate
        // number would not say which.
        var worstHead = 0
        var worstError: Float = 0
        for head in 0..<config.numAttentionHeads {
            let range = (head * headDim)..<((head + 1) * headDim)
            let scale = hostRaw[range].reduce(Float(0)) { max($0, abs($1)) }
            let delta = zip(deviceOutput[range], hostRaw[range]).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            let error = scale > 0 ? delta / scale : delta
            if error > worstError { worstError = error; worstHead = head }
        }
        // Every head, not only the worst: a wrong KV-head mapping is exact on
        // the first head of each group and wrong on the rest, which a single
        // worst-head number cannot distinguish from a smooth numeric drift.
        let perHead: [String] = (0..<config.numAttentionHeads).map { head in
            let range = (head * headDim)..<((head + 1) * headDim)
            let scale = hostRaw[range].reduce(Float(0)) { max($0, abs($1)) }
            let delta = zip(deviceOutput[range], hostRaw[range]).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            return String(format: "%d:%.1e", head, scale > 0 ? delta / scale : delta)
        }
        print("🔎 [inkling2 device] attention per head: \(perHead.joined(separator: " "))")
        print("🔎 [inkling2 device] attention: worst head \(worstHead) at \(worstError) relative over \(rows) row(s)")
        XCTAssertLessThanOrEqual(worstError, 1e-4, "head \(worstHead) of the device attention against the host graph")
        withExtendedLifetime(engine) {}
    }

    // MARK: The INT8 KV attention

    /// The INT8 attention over a cache the HOST quantized, against the host
    /// rule's own dequantization (`stored * scale`), on a fixture where every
    /// plausible bug is visible:
    ///
    ///   * `headDim = 8` and a 32-lane threadgroup, so the strided loops' tails
    ///     run and a lane count assumed equal to the head width fails;
    ///   * two KV heads over four query heads, so a wrong KV-head mapping is
    ///     exact on every second head and wrong from the first one that shares;
    ///   * per-(row, head) scales of DIFFERENT magnitudes, so a row-wide scale
    ///     (or one read at the wrong stride) changes the score;
    ///   * `position` one past `windowStart + rows`, so the last row is at a
    ///     negative distance and its bias must be zero.
    ///
    /// The rows are quantized by `Inkling2KVQuantization.stored`, which is the
    /// same contract the kernel reads, so this also pins the two sides to each
    /// other rather than to a fixture of hand-written bytes.
    func testTheInt8AttentionMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let heads = 4, kvHeads = 2, headDim = 8
        let kvWidth = kvHeads * headDim
        let keyRows = 5, dRel = 3, extent = 6
        let windowStart = 3, position = 6

        let query = values(heads * headDim, seed: 71)
        let keyRowsFloat: [[Float]] = (0..<keyRows).map { row in
            values(kvWidth, seed: 72 + row)
        }
        let valueRowsFloat: [[Float]] = (0..<keyRows).map { row in
            // A wider range than the keys: a dropped value scale is then a
            // visible error rather than a rounding difference.
            values(kvWidth, seed: 82 + row).map { $0 * 6 }
        }
        // The rows are quantized by the HOST cache, per (row, KV head), so this
        // pins the kernel to the same contract the sequence writes rather than
        // to a fixture of hand-written bytes.
        let state = Inkling2VisitState(
            isFullAttention: true, window: extent, stateWidths: [kvWidth, kvWidth, 64, 64],
            historyDepth: 3, maxRows: keyRows, storage: .int8, heads: kvHeads
        )
        for row in 0..<keyRows {
            _ = state.append(key: keyRowsFloat[row], value: valueRowsFloat[row])
        }
        let relative = values(heads * dRel, seed: 92)
        let bank = values(dRel * extent, seed: 93)
        let sink = values(heads, seed: 94)

        var keyBytes = [Int8](repeating: 0, count: keyRows * kvWidth)
        var valueBytes = [Int8](repeating: 0, count: keyRows * kvWidth)
        var keyScaleBits = [UInt16](repeating: 0, count: keyRows * kvHeads)
        var valueScaleBits = [UInt16](repeating: 0, count: keyRows * kvHeads)
        for row in 0..<keyRows {
            let key = try XCTUnwrap(state.quantizedKey(row))
            let value = try XCTUnwrap(state.quantizedValue(row))
            keyBytes.replaceSubrange((row * kvWidth)..<((row + 1) * kvWidth), with: key.stored)
            valueBytes.replaceSubrange((row * kvWidth)..<((row + 1) * kvWidth), with: value.stored)
            keyScaleBits.replaceSubrange((row * kvHeads)..<((row + 1) * kvHeads), with: key.scales)
            valueScaleBits.replaceSubrange((row * kvHeads)..<((row + 1) * kvHeads), with: value.scales)
        }

        let queryBuffer = try buffer(device, query)
        let keyBuffer = try bytes(device, keyBytes)
        let valueBuffer = try bytes(device, valueBytes)
        let keyScaleBuffer = try bits(device, keyScaleBits)
        let valueScaleBuffer = try bits(device, valueScaleBits)
        let relativeBuffer = try buffer(device, relative)
        let bankBuffer = try buffer(device, bank)
        let sinkBuffer = try buffer(device, sink)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: heads * headDim))
        let dumpBuffer = try buffer(device, [Float](repeating: 0, count: keyRows))
        let constants = try [
            UInt32(heads), UInt32(heads / kvHeads), UInt32(headDim), UInt32(kvWidth),
            UInt32(keyRows), UInt32(windowStart), UInt32(position), UInt32(dRel), UInt32(extent)
        ].map { try buffer(device, [$0]) }
        let dumps = try flag(device, false)
        try run(
            "inkling_attention_int8_f32", threads: (heads, 1, 1), threadgroup: (32, 1, 1),
            threadgroupMemory: (32 + keyRows) * MemoryLayout<Float>.stride, byThreadgroup: true
        ) { encoder in
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)
            encoder.setBuffer(valueBuffer, offset: 0, index: 2)
            encoder.setBuffer(keyScaleBuffer, offset: 0, index: 3)
            encoder.setBuffer(valueScaleBuffer, offset: 0, index: 4)
            encoder.setBuffer(relativeBuffer, offset: 0, index: 5)
            encoder.setBuffer(bankBuffer, offset: 0, index: 6)
            encoder.setBuffer(sinkBuffer, offset: 0, index: 7)
            encoder.setBuffer(outputBuffer, offset: 0, index: 8)
            for (index, constant) in constants.enumerated() {
                encoder.setBuffer(constant, offset: 0, index: 9 + index)
            }
            encoder.setBuffer(dumpBuffer, offset: 0, index: 18)
            encoder.setBuffer(dumps, offset: 0, index: 19)
        }
        let actual = read(outputBuffer, heads * headDim)

        // The host rule, element by element: `stored * scale` per element, a
        // softmax over the rows plus the sink, then the weighted value rows.
        let scaling = 1 / Float(headDim).squareRoot()
        var expected = [Float](repeating: 0, count: heads * headDim)
        for head in 0..<heads {
            let kvHead = head / (heads / kvHeads)
            var scores = [Float](repeating: 0, count: keyRows)
            for row in 0..<keyRows {
                var dot: Float = 0
                for dimension in 0..<headDim {
                    let byte = keyBytes[row * kvWidth + kvHead * headDim + dimension]
                    dot += query[head * headDim + dimension]
                        * Inkling2KVQuantization.load(byte, scale: keyScaleBits[row * kvHeads + kvHead])
                }
                let distance = position - (windowStart + row)
                var bias: Float = 0
                if distance >= 0, distance < extent {
                    for r in 0..<dRel { bias += relative[head * dRel + r] * bank[r * extent + distance] }
                }
                scores[row] = dot * scaling + bias
            }
            var combined = scores
            combined.append(sink[head])
            let maximum = combined.max() ?? 0
            let exponentials = combined.map { exp($0 - maximum) }
            let denominator = exponentials.reduce(0, +)
            for row in 0..<keyRows {
                let weight = exponentials[row] / denominator
                for dimension in 0..<headDim {
                    let byte = valueBytes[row * kvWidth + kvHead * headDim + dimension]
                    expected[head * headDim + dimension] += weight
                        * Inkling2KVQuantization.load(byte, scale: valueScaleBits[row * kvHeads + kvHead])
                }
            }
        }
        for index in actual.indices {
            XCTAssertEqual(actual[index], expected[index], accuracy: 2e-5, "element \(index)")
        }
        // The fixture is not degenerate: the four heads disagree, and the
        // output is not the sink-only zero vector a missing row would leave.
        XCTAssertGreaterThan(Set((0..<heads).map { actual[$0 * headDim] }).count, 1)
        XCTAssertGreaterThan(abs(actual[0]), 1e-4)
    }

    /// An INT8 visit stages its rows as BYTES plus a per-head scale block, and
    /// a visit in the other storage is refused rather than misread.
    ///
    /// The workspace is where the storage choice is spent on memory: the INT8
    /// staging buffer is a quarter of the FP32 one, and the scales are a small
    /// block beside it rather than being decoded into the rows.
    func testAnInt8VisitStagesItsCompactRows() throws {
        let (device, _, _) = try environment()
        let config = try Inkling2Config.decode(jsonData: Data(Inkling2ConfigTests.smallJSON.utf8))
        let kvWidth = config.numKeyValueHeads * config.headDim
        let capacity = 6
        let quantized = try Inkling2DeviceWorkspace(
            device: device, config: config, stagedPositions: capacity, storage: .int8
        )
        let reference = try Inkling2DeviceWorkspace(
            device: device, config: config, stagedPositions: capacity, storage: .float32
        )
        XCTAssertEqual(quantized.stagedStorage, .int8)
        XCTAssertEqual(reference.stagedStorage, .float32)
        XCTAssertEqual(quantized.stagedKeys.length * 4, reference.stagedKeys.length)
        XCTAssertEqual(quantized.stagedValues.length * 4, reference.stagedValues.length)

        let state = Inkling2VisitState(
            isFullAttention: true, window: 8, stateWidths: [kvWidth, kvWidth, 64, 64],
            historyDepth: 3, maxRows: capacity, storage: .int8, heads: config.numKeyValueHeads
        )
        // The second KV head is two orders of magnitude larger than the first,
        // so a row-wide scale (or a scale block read at the wrong stride) is a
        // different set of numbers rather than a rounding difference.
        let headWidth = kvWidth / config.numKeyValueHeads
        func row(seed: Int) -> [Float] {
            let base = values(kvWidth, seed: seed)
            return (0..<kvWidth).map { index in
                index < headWidth ? base[index] * 0.01 : base[index]
            }
        }
        for index in 0..<3 {
            _ = state.append(key: row(seed: 100 + index), value: row(seed: 110 + index))
        }
        let rows = try quantized.stage(state)
        XCTAssertEqual(rows, 3)

        let keyPointer = quantized.stagedKeys.contents().bindMemory(
            to: Int8.self, capacity: rows * kvWidth
        )
        let valuePointer = quantized.stagedValues.contents().bindMemory(
            to: Int8.self, capacity: rows * kvWidth
        )
        let keyScalePointer = quantized.stagedKeyScales.contents().bindMemory(
            to: UInt16.self, capacity: rows * config.numKeyValueHeads
        )
        let valueScalePointer = quantized.stagedValueScales.contents().bindMemory(
            to: UInt16.self, capacity: rows * config.numKeyValueHeads
        )
        for row in 0..<rows {
            let key = try XCTUnwrap(state.quantizedKey(row))
            let value = try XCTUnwrap(state.quantizedValue(row))
            XCTAssertEqual(
                Array(UnsafeBufferPointer(start: keyPointer + row * kvWidth, count: kvWidth)),
                key.stored, "row \(row) key bytes"
            )
            XCTAssertEqual(
                Array(UnsafeBufferPointer(start: valuePointer + row * kvWidth, count: kvWidth)),
                value.stored, "row \(row) value bytes"
            )
            XCTAssertEqual(
                Array(UnsafeBufferPointer(
                    start: keyScalePointer + row * config.numKeyValueHeads,
                    count: config.numKeyValueHeads
                )),
                key.scales, "row \(row) key scales"
            )
            XCTAssertEqual(
                Array(UnsafeBufferPointer(
                    start: valueScalePointer + row * config.numKeyValueHeads,
                    count: config.numKeyValueHeads
                )),
                value.scales, "row \(row) value scales"
            )
            // The heads are scaled SEPARATELY; a row-wide scale would give the
            // same bits for both.
            XCTAssertNotEqual(Set(key.scales).count, 1, "row \(row) shares one scale across heads")
        }

        // A visit in the other storage is a wrong number, not an error the GPU
        // would report, so staging must refuse it.
        let plain = Inkling2VisitState(
            isFullAttention: true, window: 8, stateWidths: [kvWidth, kvWidth, 64, 64],
            historyDepth: 3, maxRows: capacity, storage: .float32, heads: config.numKeyValueHeads
        )
        _ = plain.append(key: values(kvWidth, seed: 7), value: values(kvWidth, seed: 8))
        XCTAssertThrowsError(try quantized.stage(plain)) { error in
            XCTAssertTrue(
                "\(error)".contains("float32"),
                "the refusal must name the storage it was handed, got: \(error)"
            )
        }
    }

    /// The whole attention half against the host's `attention`, on the real
    /// checkpoint: after this the half is checked stage by stage rather than
    /// only at its kernel.
    ///
    /// Two host forwards over the same weights, because both sides MUTATE the
    /// visit's convolution history: one is the oracle, the other is the
    /// device driver's reference for its small host-side pieces. Sharing one
    /// would make the second side read a history the first had already pushed.
    func testTheDeviceAttentionHalfMatchesTheHost() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, library, queue) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let oracle = try Inkling2CPUForward(config: config, weights: source)
        let reference = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: reference
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config)

        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        let ids = try tokenizer.encode(text: "Hi there")
        for (position, id) in ids.enumerated() {
            _ = try oracle.forwardToken(token: UInt32(id), position: position)
            _ = try reference.forwardToken(token: UInt32(id), position: position)
        }
        let position = ids.count
        let layer = 0
        let queryInput = values(config.hiddenSize, seed: 51)
        let keyInput = values(config.hiddenSize, seed: 52)
        let valueInput = values(config.hiddenSize, seed: 53)

        func rel(_ mine: [Float], _ reference: [Float]) -> Float {
            guard !reference.isEmpty, mine.count == reference.count else { return .infinity }
            let scale = reference.reduce(Float(0)) { max($0, abs($1)) }
            return zip(mine, reference).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) } / max(scale, 1e-9)
        }

        // The device half's own stages, read back from the workspace after it runs.
        let actual = try metal.attentionHalf(
            layer: layer, state: reference.states[0], queryInput: queryInput,
            keyInput: keyInput, valueInput: valueInput, position: position, workspace: workspace,
            dumpsAttentionScores: true
        )
        // The ORACLE runs LAST: `attention` appends a KV row to the state it is
        // given, so running it first left the device half reading a state with
        // one row fewer than the host attended over -- and a row the kernel
        // never visited reads as an exact zero.
        var oracleStages: [String: [Float]] = [:]
        oracle.onInternal = { _, name, value in oracleStages[name] = value }
        let expected = try oracle.attention(
            layer: layer, state: oracle.states[0], queryInput: queryInput,
            keyInput: keyInput, valueInput: valueInput, position: position
        )

        let scale = expected.reduce(Float(0)) { max($0, abs($1)) }
        let worst = zip(actual, expected).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        print("🔎 [inkling2 device] attention half: \(worst / scale) relative")
        // Which stage diverges: each device workspace buffer against the host's
        // capture of the same named value.
        func stageError(_ name: String, _ mine: [Float], _ reference: [Float]) -> String {
            guard !reference.isEmpty else { return "\(name): no reference" }
            let stageScale = reference.reduce(Float(0)) { max($0, abs($1)) }
            let delta = zip(mine, reference).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            return String(format: "%@:%.2e", name, stageScale > 0 ? delta / stageScale : delta)
        }
        let stageRows: [String] = [
            stageError("q_norm", metal.readBack(workspace.query, count: config.swaNumAttentionHeads * config.swaHeadDim),
                       oracleStages["q_norm"] ?? []),
            // The clipped key and the normed value are the CONVOLVED buffers: the
            // projection buffers hold the pre-convolution values the visit's
            // history takes.
            stageError("k_norm", metal.readBack(workspace.convolvedKey, count: config.swaNumKeyValueHeads * config.swaHeadDim),
                       oracleStages["k_norm"] ?? []),
            stageError("v_norm", metal.readBack(workspace.convolvedValue, count: config.swaNumKeyValueHeads * config.swaHeadDim),
                       oracleStages["v_norm"] ?? []),
            stageError("relative", metal.readBack(workspace.relativeStates, count: config.swaNumAttentionHeads * config.dRel),
                       oracleStages["relative_states"] ?? []),
            stageError("sink", metal.readBack(workspace.sinkLogits, count: config.swaNumAttentionHeads),
                       oracleStages["sink"] ?? []),
            // The workspace buffer holds the RAW output only until the gate runs,
            // and it is multiplied in place: read after the whole half, it is
            // `attn_gated`, so comparing it against `attn_raw` compares two
            // different stages.
            stageError("attn_gated", metal.readBack(workspace.attentionOutput, count: config.swaNumAttentionHeads * config.swaHeadDim),
                       oracleStages["attn_gated"] ?? [])
        ]
        print("🔎 [inkling2 device] stages: " + stageRows.joined(separator: " "))
        // The attention's scores, head 0: the bias against the softmax against
        // the value fold, separated instead of inferred from their product.
        let hostScores = oracleStages["scores_head0"] ?? []
        if !hostScores.isEmpty {
            let deviceScores = metal.readBack(workspace.scoreDump, count: hostScores.count)
            print("🔎 [inkling2 device] scores_head0: \(rel(deviceScores, hostScores))")
            print("🔎 [inkling2 device]   host  \(hostScores.prefix(6))")
            print("🔎 [inkling2 device]   device \(deviceScores.prefix(6))")
        }
        // The projection inputs are verified: they are the stages the half gets
        // right, and a regression there must fail.
        for (name, reference) in [
            ("q_norm", oracleStages["q_norm"] ?? []),
            ("k_norm", oracleStages["k_norm"] ?? []),
            ("v_norm", oracleStages["v_norm"] ?? [])
        ] {
            guard !reference.isEmpty else { continue }
            // The full projected width, which is what the host captures: a
            // head count alone makes the lengths disagree and `rel` report inf.
            let mine = name == "q_norm"
                ? metal.readBack(workspace.query, count: config.swaNumAttentionHeads * config.swaHeadDim)
                : metal.readBack(
                    name == "k_norm" ? workspace.convolvedKey : workspace.convolvedValue,
                    count: config.swaNumKeyValueHeads * config.swaHeadDim
                )
            XCTAssertLessThanOrEqual(rel(mine, reference), 1e-3, "\(name) against the host")
        }

        // The whole half, end to end. It was expected-failure until the two
        // state mutations that ran before their command buffer were moved after
        // it: the KV row and the convolution histories were read from buffers
        // the GPU had not written yet, which is exact for the first rows and
        // wrong from the point a convolution reads them.
        XCTAssertLessThanOrEqual(worst / scale, 2e-3, "the device attention half against the host")
        withExtendedLifetime(engine) {}
    }

    /// Bisects a GPU fault by running the attention half's stages ONE COMMAND
    /// BUFFER AT A TIME on a fabricated state, so the stage that faults is the
    /// one that was running when the error came back -- an asynchronous fault in
    /// one buffer names nothing.
    ///
    /// It also covers a class the parity case cannot: every stage must run
    /// without an out-of-bounds bind at the checkpoint's real tensor offsets,
    /// which are NOT 256-byte aligned (that is what a first cut got wrong).
    func testEveryAttentionStageRunsWithoutAFault() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, library, queue) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let host = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: host
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config)

        // A fabricated state: three rows of KV, no host forward needed.
        let layer = 0
        let heads = config.swaNumAttentionHeads
        let kvHeads = config.swaNumKeyValueHeads
        let headDim = config.swaHeadDim
        let state = Inkling2VisitState(
            isFullAttention: false, window: config.slidingWindowSize,
            stateWidths: [kvHeads * headDim, kvHeads * headDim, config.hiddenSize, config.hiddenSize],
            historyDepth: config.convKernelSize - 1
        )
        for row in 0..<3 {
            _ = state.append(
                key: values(kvHeads * headDim, seed: 60 + row),
                value: values(kvHeads * headDim, seed: 70 + row)
            )
        }

        // ONE buffer per stage, so an asynchronous fault names the stage that
        // was running. The arena outlives each commit for the same reason the
        // driver's does: a dispatch reads its scalar arguments when the GPU runs
        // it, and a freed one is an address fault.
        func runStage(_ name: String, _ encode: (MTLComputeCommandEncoder, Inkling2ScalarArena) throws -> Void) throws {
            let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
            let arena = Inkling2ScalarArena(device: device)
            do {
                try encode(encoder, arena)
            } catch {
                encoder.endEncoding()
                XCTFail("stage '\(name)' threw while encoding: \(error)")
                return
            }
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            XCTAssertNil(commandBuffer.error, "stage '\(name)' faulted: \(String(describing: commandBuffer.error))")
            arena.clear()
        }

        let queryInput = values(config.hiddenSize, seed: 41)
        let keyInput = values(config.hiddenSize, seed: 42)
        let valueInput = values(config.hiddenSize, seed: 43)
        func staging(_ buffer: MTLBuffer, _ input: [Float]) -> MTLBuffer {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: input.count)
            for index in input.indices { pointer[index] = input[index] }
            return buffer
        }
        let queryInputBuffer = try staging(workspace.queryInput, queryInput)
        let keyInputBuffer = try staging(workspace.keyInput, keyInput)
        let valueInputBuffer = try staging(workspace.valueInput, valueInput)

        try runStage("wq") { encoder, arena in
            try metal.gemm(
                encoder: encoder, role: .attnWqDu, layer: layer,
                input: queryInputBuffer, inputOffset: 0, output: workspace.query, outputOffset: 0,
                rows: heads * headDim, columns: config.hiddenSize,
                arena: arena
            )
        }
        try runStage("q clip") { encoder, arena in
            let scale = try deviceWeights.tensor(.attnQL2ClipScale, layer: layer)
            try metal.l2Clip(
                encoder: encoder, vector: workspace.query, vectorOffset: 0,
                scale: scale.buffer, scaleOffset: Int(scale.offset),
                headDim: headDim, heads: heads, threshold: config.qkL2ClipThreshold, tau: 1,
                arena: arena
            )
        }
        try runStage("wk") { encoder, arena in
            try metal.gemm(
                encoder: encoder, role: .attnWkDv, layer: layer,
                input: keyInputBuffer, inputOffset: 0, output: workspace.key, outputOffset: 0,
                rows: kvHeads * headDim, columns: config.hiddenSize,
                arena: arena
            )
        }
        try runStage("k conv") { encoder, arena in
            try metal.convolve(
                encoder: encoder, state: state, index: 0,
                values: workspace.key, valuesCount: kvHeads * headDim, dynamicInput: keyInput,
                layer: layer, isFull: false, headDim: headDim, output: workspace.convolvedKey, arena: arena
            )
        }
        try runStage("wv") { encoder, arena in
            try metal.gemm(
                encoder: encoder, role: .attnWvDv, layer: layer,
                input: valueInputBuffer, inputOffset: 0, output: workspace.value, outputOffset: 0,
                rows: kvHeads * headDim, columns: config.hiddenSize,
                arena: arena
            )
        }
        try runStage("v conv") { encoder, arena in
            try metal.convolve(
                encoder: encoder, state: state, index: 1,
                values: workspace.value, valuesCount: kvHeads * headDim, dynamicInput: valueInput,
                layer: layer, isFull: false, headDim: headDim, output: workspace.convolvedValue, arena: arena
            )
        }
        try runStage("v norm") { encoder, arena in
            try metal.headNorm(
                encoder: encoder, vector: workspace.value, headDim: headDim, heads: kvHeads, arena: arena
            )
        }
        try runStage("attention") { encoder, arena in
            let relative = try staging(
                workspace.relativeStates, [Float](repeating: 0.01, count: heads * config.dRel)
            )
            let bank = try staging(
                workspace.bank, [Float](repeating: 0.02, count: config.dRel * config.slidingWindowSize)
            )
            let sink = try staging(workspace.sinkLogits, [Float](repeating: 0.5, count: heads))
            let rows = try workspace.stage(state)
            try metal.attention(
                encoder: encoder, workspace: workspace, state: state, rows: rows,
                position: 3, relative: relative, bank: bank, sink: sink,
                output: workspace.attentionOutput, arena: arena
            )
        }
        try runStage("gate") { encoder, arena in
            try metal.gateMultiply(
                encoder: encoder, output: workspace.attentionOutput,
                gate: workspace.gate, count: heads * headDim, arena: arena
            )
        }
        withExtendedLifetime(engine) {}
    }

    // MARK: The block read

    /// The block-attention residual read against the host rule, with TWO
    /// queries (so the query offset is exercised) and a stack of three rows plus
    /// the partial.
    func testTheBlockReadMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let width = 100, stackRows = 3, queries = 2
        // The kernel's staged slots are fixed at 64 rows, so this fixture is
        // inside them (a real stack is tens of entries).
        XCTAssertLessThan(stackRows + 1, 64)
        let stack = values(stackRows * width, seed: 13)
        let partial = values(width, seed: 14)
        let queryRows = values(queries * width, seed: 15)
        let stackBuffer = try buffer(device, stack)
        let partialBuffer = try buffer(device, partial)
        let queryBuffer = try buffer(device, queryRows)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: queries * width))
        let stackRowsBuffer = try buffer(device, [UInt32(stackRows)])
        let widthBuffer = try buffer(device, [UInt32(width)])
        // 32 warp sums, then 64 weights and 64 inverse norms the kernel stages
        // past them (the stack is a few rows; the slots are fixed).
        try run("inkling_block_read_f32", threads: (queries, 1, 1), threadgroup: (64, 1, 1),
                threadgroupMemory: 160 * MemoryLayout<Float>.stride, byThreadgroup: true) { encoder in
            encoder.setBuffer(stackBuffer, offset: 0, index: 0)
            encoder.setBuffer(partialBuffer, offset: 0, index: 1)
            encoder.setBuffer(queryBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            encoder.setBuffer(stackRowsBuffer, offset: 0, index: 4)
            encoder.setBuffer(widthBuffer, offset: 0, index: 5)
        }
        let actual = read(outputBuffer, queries * width)
        let stackIRows: [[Float]] = (0..<stackRows).map { row in
            Array(stack[(row * width)..<((row + 1) * width)])
        }
        let queryVectors: [[Float]] = (0..<queries).map { index in
            Array(queryRows[(index * width)..<((index + 1) * width)])
        }
        let expected = Inkling2Math.blockResidualRead(
            blockStates: stackIRows, partial: partial, queries: queryVectors
        )
        for query in 0..<queries {
            for index in 0..<width {
                XCTAssertEqual(
                    actual[query * width + index], expected[query][index], accuracy: 2e-5,
                    "query \(query) element \(index)"
                )
            }
        }
    }

    // MARK: Mixture pieces

    /// The router's in-place sigmoid against `Inkling2Math.sigmoid`, at a width
    /// that is not a multiple of the threadgroup so the tail runs.
    func testTheSigmoidInPlaceMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let count = 69
        let input = values(count, seed: 17).map { $0 * 6 }
        let inputBuffer = try buffer(device, input)
        let countBuffer = try buffer(device, [UInt32(count)])
        try run("inkling_sigmoid_inplace_f32", threads: (count, 1, 1), threadgroup: (32, 1, 1)) { encoder in
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(countBuffer, offset: 0, index: 1)
        }
        let actual = read(inputBuffer, count)
        for index in 0..<count {
            XCTAssertEqual(actual[index], Inkling2Math.sigmoid(input[index]), accuracy: 1e-6, "element \(index)")
        }
    }

    /// The weighted accumulate against its definition, at a width with a tail.
    func testTheWeightedAccumulateMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let count = 37
        let accumulator = values(count, seed: 18)
        let source = values(count, seed: 19)
        let weight: Float = -2.25
        let accumulatorBuffer = try buffer(device, accumulator)
        let sourceBuffer = try buffer(device, source)
        let weightBuffer = try buffer(device, [weight])
        let countBuffer = try buffer(device, [UInt32(count)])
        try run("inkling_weighted_accumulate_f32", threads: (count, 1, 1), threadgroup: (32, 1, 1)) { encoder in
            encoder.setBuffer(accumulatorBuffer, offset: 0, index: 0)
            encoder.setBuffer(sourceBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(countBuffer, offset: 0, index: 3)
        }
        let actual = read(accumulatorBuffer, count)
        for index in 0..<count {
            XCTAssertEqual(
                actual[index], accumulator[index] + weight * source[index], accuracy: 1e-6, "element \(index)"
            )
        }
    }

    /// The interleaved SwiGLU against `Inkling2Math.softcappedSwiGLU`, on a
    /// buffer laid out gate/up PAIRS -- the checkpoint's fused bank order, which
    /// the half-buffer kernel reads as two halves.
    func testTheInterleavedSwiGLUMatchesTheHostRule() throws {
        let (device, _, _) = try environment()
        let intermediate = 33
        var gateUp = [Float](repeating: 0, count: intermediate * 2)
        let gateValues = values(intermediate, seed: 20).map { $0 * 12 }
        let upValues = values(intermediate, seed: 21).map { $0 * 30 }
        for index in 0..<intermediate {
            gateUp[2 * index] = gateValues[index]
            gateUp[2 * index + 1] = upValues[index]
        }
        let gateUpBuffer = try buffer(device, gateUp)
        let outputBuffer = try buffer(device, [Float](repeating: 0, count: intermediate))
        let intermediateBuffer = try buffer(device, [UInt32(intermediate)])
        let gateCapBuffer = try buffer(device, [Float(4)])
        let upCapBuffer = try buffer(device, [Float(25)])
        try run("inkling_softcapped_swiglu_interleaved_f32", threads: (intermediate, 1, 1), threadgroup: (64, 1, 1)) { encoder in
            encoder.setBuffer(gateUpBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.setBuffer(intermediateBuffer, offset: 0, index: 2)
            encoder.setBuffer(gateCapBuffer, offset: 0, index: 3)
            encoder.setBuffer(upCapBuffer, offset: 0, index: 4)
        }
        let actual = read(outputBuffer, intermediate)
        for index in 0..<intermediate {
            let expected = Inkling2Math.softcappedSwiGLU(
                gate: gateValues[index], up: upValues[index], gateSoftcap: 4, upSoftcap: 25
            )
            XCTAssertEqual(actual[index], expected, accuracy: 1e-4, "element \(index)")
        }
        // The layout is what separates this from the half-buffer kernel: a pair
        // read as two halves would mix gate 0 with up of a different index.
        XCTAssertNotEqual(
            actual[0],
            Inkling2Math.softcappedSwiGLU(
                gate: gateUp[0], up: gateUp[intermediate], gateSoftcap: 4, upSoftcap: 25
            ),
            accuracy: 1e-6,
            "the fixture must distinguish pair order from half order"
        )
    }

    // MARK: The batched front

    /// The block's Q/K/V projections are the walk's, position by position, on
    /// the real checkpoint: eight contiguous slots through one submission must
    /// equal eight single-slot submissions through the same entry point.
    ///
    /// This is the first stage that actually batches, so it is where a slot
    /// indexing bug would first be possible: a kernel that read slot 0 eight
    /// times, or a projection whose input stride was the dimension rather than
    /// the block's slot count, would still produce plausible numbers.
    func testTheBatchedQKVProjectionIsEightSingleOnes() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, library, queue) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let reference = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: reference
        )

        let layer = 0
        let isFull = !config.isLocalLayer(layer)
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let kvHeads = isFull ? config.numKeyValueHeads : config.swaNumKeyValueHeads
        let headDim = isFull ? config.headDim : config.swaHeadDim
        let hidden = config.hiddenSize
        let slots = 8

        // Eight DISTINCT input rows, so a kernel that re-read slot 0 fails.
        let queryRows = (0..<slots).flatMap { slot in values(hidden, seed: 70 + slot) }
        let keyRows = (0..<slots).flatMap { slot in values(hidden, seed: 90 + slot) }
        let valueRows = (0..<slots).flatMap { slot in values(hidden, seed: 110 + slot) }
        let queryInput = try buffer(device, queryRows)
        let keyInput = try buffer(device, keyRows)
        let valueInput = try buffer(device, valueRows)
        func makeBlock(_ width: Int) -> MTLBuffer {
            device.makeBuffer(length: slots * width * 4, options: .storageModeShared)!
        }
        let queryBlock = makeBlock(heads * headDim)
        let keyBlock = makeBlock(kvHeads * headDim)
        let valueBlock = makeBlock(kvHeads * headDim)
        // The batched submission under test: one encoder, three projections,
        // eight slots.
        try metal.encodeBlockQKV(
            layer: layer, queryInput: queryInput, keyInput: keyInput, valueInput: valueInput,
            query: queryBlock, key: keyBlock, value: valueBlock, slots: slots
        )

        // The same eight slots, one at a time: each slot's row goes into slot 0
        // of a block-sized buffer (the kernel reads all eight rows
        // unconditionally, so a one-row buffer is an out-of-bounds read that
        // never completes) and each projection gets its OWN output buffer --
        // sharing one would let the last projection overwrite the others.
        func singles(
            _ rows: [Float], _ width: Int, _ outputWidth: Int
        ) throws -> (q: [[Float]], k: [[Float]], v: [[Float]]) {
            var q: [[Float]] = [], k: [[Float]] = [], v: [[Float]] = []
            for slot in 0..<slots {
                var padded = [Float](repeating: 0, count: slots * width)
                padded.replaceSubrange(
                    0..<width, with: rows[(slot * width)..<((slot + 1) * width)]
                )
                let one = try buffer(device, padded)
                func makeOutput() -> MTLBuffer {
                    device.makeBuffer(length: slots * outputWidth * 4, options: .storageModeShared)!
                }
                let qOut = makeOutput(), kOut = makeOutput(), vOut = makeOutput()
                try metal.encodeBlockQKV(
                    layer: layer, queryInput: one, keyInput: one, valueInput: one,
                    query: qOut, key: kOut, value: vOut, slots: slots
                )
                q.append(Array(read(qOut, slots * outputWidth)[0..<outputWidth]))
                k.append(Array(read(kOut, slots * outputWidth)[0..<outputWidth]))
                v.append(Array(read(vOut, slots * outputWidth)[0..<outputWidth]))
            }
            return (q, k, v)
        }
        func compare(_ label: String, _ block: MTLBuffer, _ walked: [[Float]], _ width: Int) {
            let batched = read(block, slots * width)
            var worst: Float = 0, scale: Float = 0
            for slot in 0..<slots {
                for index in 0..<width {
                    let mine = batched[slot * width + index]
                    let theirs = walked[slot][index]
                    worst = max(worst, abs(mine - theirs))
                    scale = max(scale, abs(theirs))
                }
            }
            print("🔎 [inkling2 device] batched QKV: \(label) \(worst / scale) relative over \(slots) slots")
            XCTAssertLessThanOrEqual(worst / scale, 1e-5, "\(label): the batched form against eight single ones")
        }
        let querySingles = try singles(queryRows, hidden, heads * headDim)
        let keySingles = try singles(keyRows, hidden, kvHeads * headDim)
        let valueSingles = try singles(valueRows, hidden, kvHeads * headDim)
        compare("Q", queryBlock, querySingles.q, heads * headDim)
        compare("K", keyBlock, keySingles.k, kvHeads * headDim)
        compare("V", valueBlock, valueSingles.v, kvHeads * headDim)
        withExtendedLifetime(engine) {}
    }

    // MARK: The MLP half

    /// The batched projection entry: eight contiguous position rows through
    /// `gemv` are the same numbers eight single-position dispatches give, and a
    /// block wider than the kernel reads is refused instead of silently
    /// truncated to its first eight.
    ///
    /// This is the seam the batched prefill will call, and the reason it is a
    /// kernel case rather than a graph one: the graph still drives positions one
    /// at a time, so nothing else can compare the two forms on the same bytes.
    func testTheBatchedProjectionIsEightSingleOnesAndRefusesMore() throws {
        let (device, library, queue) = try environment()
        let config = try Inkling2Config.decode(jsonData: Data(Inkling2ConfigTests.smallJSON.utf8))
        let outDim = 12, inDim = config.hiddenSize
        let weights: [Float] = (0..<(outDim * inDim)).map { index in
            Float((index % 19) - 9) / 8
        }
        let halves = weights.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let weightBytes = halves.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)!
        }
        let name = "model.llm.layers.0.attn.wq_du.weight"
        let catalog = Inkling2WeightCatalog(
            mainFile: [Inkling2TensorRef(
                name: name, dtype: "BF16", shape: [outDim, inDim], shardIndex: 0, byteOffset: 0
            )],
            layerCount: config.numHiddenLayers
        )
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: [0: weightBytes]
        )
        let hostSource = Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: [0: weightBytes]
        )
        let host = try Inkling2CPUForward(config: config, weights: hostSource)
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: host
        )

        let positions = 8
        let input: [Float] = (0..<(positions * inDim)).map { index in
            Float((index % 13) - 6) / 8
        }
        let inputBuffer = try buffer(device, input)
        let batchedOutput = try XCTUnwrap(
            device.makeBuffer(length: positions * outDim * 4, options: .storageModeShared)
        )
        let singleOutput = try XCTUnwrap(
            device.makeBuffer(length: positions * outDim * 4, options: .storageModeShared)
        )
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        try metal.gemv(
            encoder: encoder, weight: weightBytes, weightOffset: 0,
            input: inputBuffer, inputOffset: 0,
            output: batchedOutput, outputOffset: 0,
            rows: outDim, columns: inDim, positions: positions,
            arena: Inkling2ScalarArena(device: device)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        // The same eight positions, one at a time, through the one-row form.
        let singleBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let singleEncoder = try XCTUnwrap(singleBuffer.makeComputeCommandEncoder())
        let singleArena = Inkling2ScalarArena(device: device)
        for position in 0..<positions {
            try metal.gemv(
                encoder: singleEncoder, weight: weightBytes, weightOffset: 0,
                input: inputBuffer, inputOffset: position * inDim * 4,
                output: singleOutput, outputOffset: position * outDim * 4,
                rows: outDim, columns: inDim, positions: 1, arena: singleArena
            )
        }
        singleEncoder.endEncoding()
        singleBuffer.commit()
        singleBuffer.waitUntilCompleted()
        XCTAssertNil(singleBuffer.error)

        let batched = batchedOutput.contents().bindMemory(to: Float.self, capacity: positions * outDim)
        let single = singleOutput.contents().bindMemory(to: Float.self, capacity: positions * outDim)
        var worst: Float = 0, scale: Float = 0
        for index in 0..<(positions * outDim) {
            worst = max(worst, abs(batched[index] - single[index]))
            scale = max(scale, abs(single[index]))
        }
        print("🔎 [inkling2 device] batched projection: \(worst / scale) relative over \(positions) positions")
        XCTAssertLessThanOrEqual(worst / scale, 1e-5, "eight positions against eight single ones")

        // A block wider than the kernel reads is refused, by name, rather than
        // silently running its first eight rows.
        let refusalBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let refusalEncoder = try XCTUnwrap(refusalBuffer.makeComputeCommandEncoder())
        XCTAssertThrowsError(
            try metal.gemv(
                encoder: refusalEncoder, weight: weightBytes, weightOffset: 0,
                input: inputBuffer, inputOffset: 0,
                output: batchedOutput, outputOffset: 0,
                rows: outDim, columns: inDim, positions: 9,
                arena: Inkling2ScalarArena(device: device)
            )
        ) { error in
            XCTAssertTrue("\(error)".contains("one to eight positions"), "\(error)")
        }
        refusalEncoder.endEncoding()
    }

    /// The mixture's binding guard, priced on buffers rather than on the slices
    /// an expert reads.
    ///
    /// The failure it exists for is SILENT: on a 16 GB M1 Pro, binding a layer's
    /// w13 weight segment (9.28 GB) and its scale segment (9.21 GB) in one
    /// dispatch returns zeros for every row, with no command-buffer error and no
    /// validation warning. So the rule is pinned here without a model: every
    /// distinct buffer counts WHOLE, a buffer repeated across eight experts
    /// counts once, and a total past the budget is refused with every segment
    /// named.
    func testAMixtureBindingIsPricedPerBufferAndRefusedByItsSegments() throws {
        let (device, _, _) = try environment()
        let bank = try XCTUnwrap(device.makeBuffer(length: 3 << 20, options: .storageModeShared))
        let scales = try XCTUnwrap(device.makeBuffer(length: 1 << 20, options: .storageModeShared))
        let down = try XCTUnwrap(device.makeBuffer(length: 2 << 20, options: .storageModeShared))

        var binding = Inkling2MixtureBinding()
        // Eight routed experts over one layer's two segments: the price is the
        // segments', not eight times the expert slice.
        for _ in 0..<8 {
            binding.add(bank, label: "the w13 bank of layer 0")
            binding.add(scales, label: "the w13 scales of layer 0")
        }
        XCTAssertEqual(binding.segments.count, 2, "one bank and one scale segment, not eight")
        XCTAssertEqual(binding.bytes, UInt64(4 << 20))
        binding.add(down, label: "the w2 bank of layer 0")
        XCTAssertEqual(binding.segments.count, 3)
        XCTAssertEqual(binding.bytes, UInt64(6 << 20))

        // Under the budget: nothing to say.
        XCTAssertNoThrow(try binding.refuse(over: 1 << 30))
        // Over it: a refusal that names the segments and both totals, so the
        // message says WHICH binding was too large rather than only that one was.
        XCTAssertThrowsError(try binding.refuse(over: UInt64(4 << 20))) { error in
            let message = "\(error)"
            XCTAssertTrue(message.contains("the w13 bank of layer 0"), message)
            XCTAssertTrue(message.contains("the w13 scales of layer 0"), message)
            XCTAssertTrue(message.contains("the w2 bank of layer 0"), message)
            XCTAssertTrue(message.contains("6.00 MB"), message)
            XCTAssertTrue(message.contains("4.00 MB"), message)
            XCTAssertTrue(message.contains("returns zeros"), message)
        }
    }

    /// The unembedding, device against host: the last stage an engine TURN runs
    /// on the GPU, and the one that decides a greedy token.
    ///
    /// Pinned because a turn's opening diverged from the reference HERE. The
    /// runner (host head) reproduces the reference's FP32 opening
    /// `[200001, 200008, 371, 150409, 171960, ...]`; the engine, which installs
    /// the device head, picks `159306` at index 4 -- which is exactly the token
    /// the reference's own BF16 run picks there, i.e. a near tie. So the bar has
    /// to separate "rounding at a tie" from a wrong reduction: a head that
    /// accumulated the row differently would move many rows by far more.
    func testTheHeadMatchesTheHostUnembedding() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, library, queue) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let weights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers
        )
        let host = try Inkling2CPUForward(
            config: config,
            weights: Inkling2SnapshotWeights(
                config: config, catalog: catalog, shardBuffers: shardBuffers
            )
        )
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: weights, host: host
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config)
        let row = values(config.hiddenSize, seed: 91)

        let deviceLogits = try metal.headLogits(hidden: row, workspace: workspace)
        let hostLogits = try host.weights.logits(
            row, rows: config.unpaddedVocabSize, hiddenSize: config.hiddenSize
        )
        XCTAssertEqual(deviceLogits.count, hostLogits.count)
        var worst: Float = 0
        var scale: Float = 0
        for index in 0..<hostLogits.count {
            worst = max(worst, abs(deviceLogits[index] - hostLogits[index]))
            scale = max(scale, abs(hostLogits[index]))
        }
        let relative = scale > 0 ? worst / scale : worst
        print("🔎 [inkling2 device] head: \(relative) relative against the host unembedding")
        // The largest-magnitude logit's own reduction: the comparison is of a
        // whole GEMV, so a per-element 1e-6 is what a different accumulation
        // order costs at these widths.
        XCTAssertLessThanOrEqual(relative, 1e-4, "the device head against the host unembedding")
        // And the ARGMAX, which is what a greedy turn actually spends the head
        // on: the same row, the same answer.
        XCTAssertEqual(
            deviceLogits.indices.max { deviceLogits[$0] < deviceLogits[$1] },
            hostLogits.indices.max { hostLogits[$0] < hostLogits[$1] },
            "the device head argues for a different token"
        )
        withExtendedLifetime(engine) {}
    }

    /// What one mixture BINDS, per layer, on the published checkpoint: the
    /// measurement the guard's budget is drawn against.
    ///
    /// The price is the segments, charged WHOLE. An expert's bank slice is
    /// 4.27 MB; the buffer it lives in is whatever the segmenter closed, and on
    /// this 16 GB machine layer 0's bank and its scales land in two segments of
    /// ~9 GB each. Printed for the layers either side of the quantization
    /// boundary (the NVFP4 head and the MXFP8 tail); it asserts only the layout
    /// split the plan declares, because the sizes are the segmenter's answer for
    /// ONE machine and the printout is the point.
    func testWhatAMixtureBindsPerLayerOnThePublishedCheckpoint() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, _, _) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let weights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers
        )
        let budget = UInt64(device.recommendedMaxWorkingSetSize)
        let layers = [0, 9, config.numHiddenLayers - 5, config.numHiddenLayers - 1]
        for layer in layers {
            let layout = Inkling2MemoryPlan.expertLayout(forLayer: layer, config: config)
            var binding = Inkling2MixtureBinding()
            for expert in 0..<config.numExpertsPerToken {
                let gateUp = try weights.expertBank(.expertW13, layer: layer, expert: expert)
                let down = try weights.expertBank(.expertW2, layer: layer, expert: expert)
                binding.add(gateUp.buffer, label: "the w13 bank of layer \(layer)")
                binding.add(gateUp.scaleBuffer, label: "the w13 scales of layer \(layer)")
                binding.add(down.buffer, label: "the w2 bank of layer \(layer)")
                binding.add(down.scaleBuffer, label: "the w2 scales of layer \(layer)")
            }
            let verdict = binding.bytes > budget ? "REFUSED" : "under the line"
            print(
                "\u{1F50E} [inkling2 binding] layer \(layer) (\(layout)): "
                    + "\(binding.segments.count) segment(s), "
                    + "\(Inkling2MixtureBinding.describe(binding.bytes)) vs a "
                    + "\(Inkling2MixtureBinding.describe(budget)) budget -> \(verdict)"
            )
            for segment in binding.segments {
                print("\u{1F50E} [inkling2 binding]     \(segment.label): \(segment.bytes) bytes")
            }
        }
        // The published split: every layer before the last four is NVFP4, which
        // is what the arena takes its geometry from.
        XCTAssertEqual(
            Inkling2MemoryPlan.expertLayout(forLayer: 0, config: config), .nvfp4
        )
        XCTAssertEqual(
            Inkling2MemoryPlan.expertLayout(
                forLayer: config.numHiddenLayers - 1, config: config
            ), .mxfp8
        )
        withExtendedLifetime(engine) {}
    }

    /// The MLP half against the host graph's own stages, on the real
    /// checkpoint: the block read and the message gate stay on the host, so the
    /// comparison covers the norm, the router and its selection, and the expert
    /// mixture through to the post-MLP convolution and the residual add.
    func testTheDeviceMLPHalfMatchesTheHost() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, library, queue) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        // The raw summary is kept because the arena's `pread` needs FILE offsets
        // and segmentation rewrote the tensor offsets to be segment-relative.
        let rawSummary = try engine.getSummary()
        let summary = try inkling2MetalReadySummary(rawSummary)
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let oracle = try Inkling2CPUForward(config: config, weights: source)
        let reference = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers,
            fileOrigins: InferenceEngine.shardFileOrigins(original: rawSummary, segmented: summary)
        )
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: reference
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config)

        // The mixture's mmap path binds the layer's fused bank and its scales as
        // two WHOLE segment buffers, and on a device that cannot wire both the
        // dispatch returns zeros with no error at all. A resident arena binds one
        // slot instead, so the parity this test exists for is reachable here on a
        // 16 GB machine -- through the arena -- rather than skipped. The arena's
        // own exactness is pinned by
        // `testAPublishedBankFillsItsSlotWithTheBanksOwnBytes`; what this test
        // adds is the whole half, stage by stage.
        let layerZero = try deviceWeights.expertBank(.expertW13, layer: 0, expert: 0)
        let layerZeroDown = try deviceWeights.expertBank(.expertW2, layer: 0, expert: 0)
        var bankPair = Inkling2MixtureBinding()
        bankPair.add(layerZero.buffer, label: "the w13 bank of layer 0")
        bankPair.add(layerZero.scaleBuffer, label: "the w13 scales of layer 0")
        bankPair.add(layerZeroDown.buffer, label: "the w2 bank of layer 0")
        bankPair.add(layerZeroDown.scaleBuffer, label: "the w2 scales of layer 0")
        let budget = metal.mixtureBindingBudget
        print(
            "🔎 [inkling2 device] the layer 0 mixture binds "
                + "\(Inkling2MixtureBinding.describe(bankPair.bytes)) against a "
                + "\(Inkling2MixtureBinding.describe(budget)) budget"
        )
        if bankPair.bytes > budget {
            // The contract on this machine: the mmap pair is refused by name.
            XCTAssertThrowsError(try bankPair.refuse(over: budget)) { error in
                XCTAssertTrue("\(error)".contains("returns zeros"), "\(error)")
            }
        } else {
            XCTAssertNoThrow(try bankPair.refuse(over: budget))
        }
        // The arenas a TURN installs, through the app's own path.
        let arenaLine = try inkling2InstallExpertArenas(
            executor: metal, weights: deviceWeights, summary: summary, snapshot: snapshot,
            config: config, catalog: catalog, positions: 8
        )
        XCTAssertFalse(arenaLine.contains("no arena"), arenaLine)
        var arenaCount = 0
        for installed in metal.expertArenas { arenaCount += installed.arena.capacity }
        XCTAssertGreaterThan(arenaCount, config.numExpertsPerToken, "an arena that holds one mixture")
        print("🔎 [inkling2 device] arenas:\(arenaLine)")
        addTeardownBlock { metal.removeExpertArena() }

        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        let ids = try tokenizer.encode(text: "Hi there")
        for (position, id) in ids.enumerated() {
            _ = try oracle.forwardToken(token: UInt32(id), position: position)
            _ = try reference.forwardToken(token: UInt32(id), position: position)
        }
        let position = ids.count
        let layer = 0
        let hiddenWidth = config.hiddenSize
        let residual = values(hiddenWidth, seed: 61)
        // A stack with two entries, so the block read is not a single-row read.
        let blockStates = [values(hiddenWidth, seed: 62), values(hiddenWidth, seed: 63)]

        // The host's own visit, captured stage by stage: the oracle runs FIRST
        // and its `mlp_input` is what the device half is fed, because the read
        // is the host's (it mutates the stack and the message state).
        var oracleStages: [String: [Float]] = [:]
        oracle.onInternal = { _, name, value in oracleStages[name] = value }
        var stack = blockStates
        let visitOutput = try oracle.runVisit(
            layer: layer, pass: 0, hidden: residual, blockStates: &stack, messages: nil, position: position
        ).output
        let mlpInput = try XCTUnwrap(oracleStages["mlp_input"], "the host captured no MLP input")
        // Not the visit's INPUT: at a block boundary the host adds back only the
        // post-attention value, and the read's stack carries the input. The two
        // are different vectors, and the half takes the add's one.
        let residualToAdd = try XCTUnwrap(oracleStages["residual"], "the host captured no residual")

        // The device half drives the reference's visit state (it advances the
        // convolution history itself), which the oracle never touched.
        let actual = try metal.mlpHalf(
            layer: layer, state: reference.states[0], mlpInput: mlpInput,
            residual: residualToAdd, workspace: workspace
        )

        // The host's own tail, appended to its captured mixture, is the value
        // the half returns: `runVisit`'s output is residual + convolved
        // mixture, and so is the half's.
        let scale = visitOutput.reduce(Float(0)) { max($0, abs($1)) }
        let worst = zip(actual, visitOutput).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        print("🔎 [inkling2 device] mlp half: \(worst / scale) relative against the visit output")
        XCTAssertLessThanOrEqual(worst / scale, 3e-3, "the device MLP half against the host graph")

        // The named stages the half gets wrong first, in order.
        func stageError(_ name: String, _ mine: [Float], _ reference: [Float]) -> String {
            guard !reference.isEmpty else { return "\(name): no reference" }
            let stageScale = reference.reduce(Float(0)) { max($0, abs($1)) }
            let delta = zip(mine, reference).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            return String(format: "%@:%.2e", name, stageScale > 0 ? delta / stageScale : delta)
        }
        let topK = config.numExpertsPerToken
        // The norm's output is the router's only input, so it is compared
        // FIRST when the selection diverges: a wrong norm moves every score.
        let kernelMixtureInput = Inkling2Math.rmsNorm(
            oracleStages["mlp_input"] ?? [], gain: try source.matrix(.mlpNorm, layer: layer),
            eps: config.rmsNormEps
        )
        let stageRows: [String] = [
            stageError("norm", metal.readBack(workspace.attentionInput, count: hiddenWidth), kernelMixtureInput),
            stageError("route_scores", metal.readBack(workspace.routerScores, count: config.numRoutedExperts),
                       oracleStages["route_scores"] ?? []),
            stageError("route_indices", metal.readBackIndices(workspace.routeIndices, count: topK).map(Float.init),
                       oracleStages["route_indices"] ?? []),
            stageError("route_weights", metal.readBack(workspace.routeWeights, count: topK),
                       oracleStages["route_weights"] ?? []),
            stageError("mixture", metal.readBack(workspace.residualAccumulator, count: hiddenWidth),
                       oracleStages["mixtureOut"] ?? []),
            stageError("postMlpConv", metal.readBack(workspace.hidden, count: hiddenWidth),
                       oracleStages["postMlpConv"] ?? [])
        ]
        print("🔎 [inkling2 device] mlp stages: " + stageRows.joined(separator: " "))

        // The mixture alone is asserted, not just printed. Its eight expert
        // slices are encoded into ONE command buffer, so the route order is the
        // encoder's dispatch order rather than eight commits' -- and a fold that
        // lost that order would still land inside the half's 3e-3 band above
        // while moving this stage. Measured 2.3e-06 relative, so the bar is 1e-4.
        let mixtureRows = try XCTUnwrap(
            oracleStages["mixtureOut"], "the host captured no mixture"
        )
        let mixtureScale = mixtureRows.reduce(Float(0)) { max($0, abs($1)) }
        let mixtureDelta = zip(metal.readBack(workspace.residualAccumulator, count: hiddenWidth), mixtureRows)
            .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        XCTAssertLessThanOrEqual(
            mixtureDelta / mixtureScale, 1e-4,
            "the folded mixture accumulator against the host graph"
        )
        withExtendedLifetime(engine) {}
    }


    /// The agent half on the published checkpoint: the model writes a call, the
    /// result comes back in the family's own tool message, and the continuation
    /// answers from it.
    ///
    /// The call must be the model's OWN -- a hand-written one would measure a
    /// continuation whose call half the checkpoint never produced -- and the
    /// continuation is spliced from ids (prompt + generation + result) the way
    /// the app pins it, so a re-encode that disagreed would mean the model reads
    /// a different prompt than the one measured here.
    ///
    /// Opt-in: a host turn is minutes per position.
    /// `touch /tmp/dynamoe_inkling2_agent_turn` to run it.
    func testATurnAnswersAfterAToolResultOnThePublishedCheckpoint() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_agent_turn"),
            "opt-in agent turn; touch /tmp/dynamoe_inkling2_agent_turn to run"
        )
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let rawSummary = try engine.getSummary()
        let summary = try inkling2MetalReadySummary(rawSummary)
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let forward = try Inkling2CPUForward(config: config, weights: source)
        let markers = Inkling2PromptMarkers.familyDefault

        // The DEVICE executor runs the visits -- it is what ships, and it is
        // ~30x the host -- while the host graph keeps the schedule and serves
        // the small host-side pieces a visit needs. The staging is sized for
        // this turn's few hundred positions, not the checkpoint's 1M ceiling.
        let (_, library, queue) = try environment()
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers,
            fileOrigins: InferenceEngine.shardFileOrigins(original: rawSummary, segmented: summary)
        )
        let executor = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: forward
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config, stagedPositions: 2_048)
        // The arena the mixture reads: without it the mmap binding is refused by
        // name on a device that cannot wire two bank segments at once. This test
        // has no origins, so the arenas are refused here and the run would be a
        // host-speed one -- hence the origins are computed for it too.
        let arenaLine = try inkling2InstallExpertArenas(
            executor: executor, weights: deviceWeights, summary: summary, snapshot: snapshot,
            config: config, catalog: catalog, positions: 2_048
        )
        XCTAssertFalse(arenaLine.contains("no arena"), arenaLine)
        addTeardownBlock { executor.removeExpertArena() }
        forward.visitRunner = { layer, pass, hidden, blockStates, messages, position in
            try executor.runVisit(
                layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                messages: messages, position: position, workspace: workspace
            )
        }
        defer { forward.visitRunner = nil }

        // A tool catalogue and the instruction block, as the harness builds them.
        let tools = [Inkling2ToolDeclaration(
            name: "read_file",
            description: "Read a file from disk and return its contents.",
            parameters: [
                "type": .string("object"),
                "properties": .object(["path": .object(["type": .string("string")])]),
                "required": .array([.string("path")])
            ]
        )]
        // The tool declaration alone, which the template renders itself: no
        // persona and no instruction block. The checkpoint's own template is
        // what teaches the call shape, so a live call must survive a prompt
        // that has nothing else in it -- and this is the shortest prompt that
        // can produce one.
        let system = "You are a helpful assistant."
        let ask = "Call read_file now with path /tmp/notes.txt, then tell me what it says."
        let prompt = try Inkling2Prompt().render(
            turns: [.system(system), .user([.text(ask)])], effort: 0, tools: tools
        )
        let promptIds = try tokenizer.encode(text: prompt).map { Int($0) }
        print("🔎 [inkling2 agent] prompt \(promptIds.count) token(s): \(prompt.debugDescription)")
        // The reference prompt producer: `/tmp/inkling_tool_logits_ref.py` runs
        // the pinned Transformers implementation over EXACTLY this text, and
        // the per-position comparison is only meaningful while the two are the
        // same bytes. Opt-in, because it returns before the model runs.
        if FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_prompt_only") {
            try prompt.write(toFile: "/tmp/inkling2_port_prompt.txt", atomically: true, encoding: .utf8)
            try promptIds.map(String.init).joined(separator: ",")
                .write(toFile: "/tmp/inkling2_port_prompt_ids.txt", atomically: true, encoding: .utf8)
            return
        }

        // The checkpoint reasons at length before it calls anything, and the
        // reasoning is what a small budget spends: at 64 tokens the turn ended
        // still deliberating, with no call in it.
        let settings = Inkling2TurnRunner.Settings(
            temperature: 0, topP: 1, minP: 0, topK: 1, maxTokens: 384,
            eosTokenIds: [UInt32(clamping: config.eosTokenID)]
        )
        func argmax(_ logits: [Float]) -> UInt32 {
            UInt32(logits.indices.max { logits[$0] < logits[$1] } ?? 0)
        }
        func decode(_ ids: [UInt32]) -> String {
            (try? tokenizer.decode(ids: ids)) ?? ""
        }
        let call = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: promptIds, settings: settings,
            sample: { logits, _ in argmax(logits) },
            shouldFreeze: { text in Inkling2ToolDialect.shouldFreeze(text, markers: markers) },
            decode: decode
        )
        print("🔎 [inkling2 agent] call: \(call.finishReason.rawValue) \(call.text.debugDescription)")
        // The pinned reference's first eight ids for this exact prompt (FP32,
        // greedy, `apply_chat_template` with the same tools and system block).
        // Comparing them is what separates "the graph diverges at this length"
        // from "the model deliberates longer here": the reference calls the tool
        // and these ids are its opening.
        //
        // FP32 and not BF16: the reference's own BF16 run flips at index 4
        // (`159306`) where its FP32 run keeps going (`171960`), so a bar taken
        // from BF16 would fault this graph for the reference's rounding. The
        // port matches the FP32 continuation for these eight ids and diverges
        // after them; `Tools/inkling2_compare_logits.py` is what shows the
        // per-position size of the remaining difference.
        let referenceOpening: [UInt32] = [200001, 200008, 371, 150409, 171960, 149882, 151272, 150546]
        let opening = Array(call.generatedTokenIds.prefix(referenceOpening.count))
        print("🔎 [inkling2 agent] opening: \(opening)")
        print("🔎 [inkling2 agent] reference: \(referenceOpening)")

        // The gate: a complete call on the published checkpoint, frozen at its
        // own block, whose arguments are the ones the question named, and a
        // continuation after the result that carries the result's content.
        // Both halves are asserted: the call alone proves the graph and the
        // dialect, and the answer proves the result turn and the resume.
        XCTAssertEqual(call.finishReason, .toolFreeze, "the call turn must freeze on its complete block")
        XCTAssertEqual(opening, referenceOpening, "the opening must track the reference")
        let calls = Inkling2ToolDialect.callBlocks(in: call.text, markers: markers)
        let firstCall = try XCTUnwrap(calls.first, "the frozen text carries no complete call")
        XCTAssertEqual(firstCall.name, "read_file")
        XCTAssertEqual(firstCall.arguments["path"] as? String, "/tmp/notes.txt")

        // The result the loop would hand back, rendered the way the harness does.
        let responseJSON = #"{"tool":"read_file","status":"success","result":{"path":"/tmp/notes.txt","content":"the answer is 42"}}"#
        let payload = AgentHarness.renderToolResultForModel(responseJSON)
        let resultTurn = Inkling2ToolDialect.resultTurn(name: firstCall.name, payload: payload, markers: markers)
        XCTAssertTrue(resultTurn.hasPrefix(markers.tool + firstCall.name + markers.contentText))
        XCTAssertFalse(resultTurn.contains("<|im_start|>"), "a chatml role marker reached the result")

        // The call turn's own text ends at its end-sampling marker, which the
        // template writes after every assistant turn.
        let callTurnText = call.text.contains(markers.contentModelEndSampling)
            ? call.text : call.text + markers.contentModelEndSampling
        let continuation = prompt + callTurnText + resultTurn
        let continuationIds = try tokenizer.encode(text: continuation).map { Int($0) }
        // A turn that stopped at the end-sampling marker already carries it in
        // `generatedTokenIds`; appending it again put a SECOND copy in the
        // splice, which is what made the re-encode disagree at the end tag.
        let endTagIds: [Int] = call.text.contains(markers.contentModelEndSampling)
            ? [] : try tokenizer.encode(text: markers.contentModelEndSampling).map { Int($0) }
        let resultIds = try tokenizer.encode(text: resultTurn).map { Int($0) }
        let generatedIds: [Int] = call.generatedTokenIds.map { Int($0) }
        let splicedIds: [Int] = promptIds + generatedIds + endTagIds + resultIds
        let spliceMatches = splicedIds == continuationIds
        print(
            "🔎 [inkling2 agent] continuation: \(promptIds.count) prompt + "
                + "\(call.generatedTokenIds.count) generated + \(endTagIds.count) end + "
                + "\(resultIds.count) result = \(splicedIds.count) spliced, "
                + "\(continuationIds.count) re-encoded, splice==reencode \(spliceMatches)"
        )
        if !spliceMatches {
            let divergence = zip(splicedIds, continuationIds)
                .enumerated()
                .first { $0.element.0 != $0.element.1 }?
                .offset ?? min(splicedIds.count, continuationIds.count)
            print(
                "🔎 [inkling2 agent] diverges at \(divergence): "
                    + "spliced \(splicedIds[divergence]) vs re-encoded \(continuationIds[divergence])"
            )
        }
        XCTAssertTrue(spliceMatches, "the spliced continuation is not the re-encoded one")

        // What the app proposes: every token that went through the cache. The
        // freeze token is drawn but never forwarded.
        let forwardedTokens = promptIds.count + call.generatedTokenIds.count - 1
        let prefix = forward.reusablePrefix(
            proposedPrefix: forwardedTokens, promptTokenIds: continuationIds
        )
        print("🔎 [inkling2 agent] continuation resumes \(prefix) of \(continuationIds.count)")
        XCTAssertEqual(
            prefix, forwardedTokens,
            "the continuation resumed \(prefix), not the call turn's \(forwardedTokens) forwarded tokens"
        )

        let answer = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: continuationIds, prefixTokens: prefix,
            settings: settings,
            sample: { logits, _ in argmax(logits) },
            decode: decode
        )
        print("🔎 [inkling2 agent] answer: \(answer.finishReason.rawValue) after \(answer.tokensGenerated) token(s)")
        print("🔎 [inkling2 agent] answer text: \(answer.text.debugDescription)")
        XCTAssertGreaterThan(answer.tokensGenerated, 0, "the continuation produced no token")
        // The result turn and the resume together: the answer can only carry
        // this word if the spliced prompt resumed from the call's own cache and
        // the tool result rendered as the template expects.
        XCTAssertTrue(
            answer.text.contains("42"),
            "the answer after the tool result must carry the content it returned"
        )
        withExtendedLifetime(engine) {}
    }

    // MARK: Divergence against the reference

    /// Position by position, the host graph's top-8 ids against the pinned
    /// Transformers reference's, on the tool prompt the agent case sends.
    ///
    /// A turn's continuation agrees with the reference for a few tokens and
    /// then diverges, and a 384-token continuation says nothing about WHERE.
    /// This does: the prompt is fixed (dumped by the agent case), so the first
    /// position whose ranking disagrees is the first position whose logits
    /// disagree, and that is the length a bisect needs.
    ///
    /// Opt-in with `/tmp/dynamoe_inkling2_agent_turn` (the dump) and the
    /// reference JSON from `INKLING_REF_DTYPE=float32 /tmp/inkling_tool_logits_ref.py`.
    ///
    /// FP32 is preferred and BF16 is the fallback: the reference disagrees with
    /// itself between the two (its BF16 run flips an argmax at two positions
    /// here), so a bar taken from BF16 faults this graph for the reference's
    /// rounding. The FP32 dump is the yardstick; BF16 is only what an older
    /// machine may have.
    func testTheHostRankingMatchesTheReferencePositionByPosition() throws {
        // Opt-in: a host walk of the whole prompt is ~46 minutes, so this is a
        // measurement rather than a gate. The DEVICE graph's per-position table
        // (`testWhereTheDeviceContinuationLeavesTheReference`) is 100 seconds and
        // is the one a turn's fidelity is judged on.
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_host_ranking"),
            "opt-in host ranking; touch /tmp/dynamoe_inkling2_host_ranking to run"
        )
        if FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_prompt_only") {
            throw XCTSkip("the agent case is in prompt-only mode")
        }
        let referencePath = ["/tmp/inkling2_tool_logits_ref_fp32.json", "/tmp/inkling2_tool_logits_ref.json"]
            .first { FileManager.default.fileExists(atPath: $0) }
            ?? "/tmp/inkling2_tool_logits_ref_fp32.json"
        print("🔎 [inkling2 logits] reference: \(referencePath)")
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: referencePath)),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let promptIds = root["promptIds"] as? [Int],
              let positions = root["positions"] as? [[String: Any]] else {
            throw XCTSkip("no reference logits at \(referencePath): run /tmp/inkling_tool_logits_ref.py")
        }
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice(), "no Metal device")
        let shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let forward = try Inkling2CPUForward(config: config, weights: source)
        var logits: [Float] = []
        for (position, id) in promptIds.enumerated() {
            logits = try forward.forwardToken(token: UInt32(id), position: position)
            guard let entry = positions.first(where: { ($0["position"] as? Int) == position }),
                  let expectedIds = entry["ids"] as? [Int],
                  let expectedLogits = entry["logits"] as? [Double] else { break }
            let ranking = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(expectedIds.count)
            let mine = Array(ranking)
            // What is asserted is what the evidence supports: the argmax, and the
            // top-FOUR set. The exact order inside the top 8 is NOT asserted
            // because the host graph's own noise reorders near-ties -- two ids
            // 0.03 apart swap -- and the plan's Phase 2 bound (2e-2 relative on
            // the final logits) is what this graph is held to, not a ranking.
            let expectedTop4 = Set(expectedIds.prefix(4))
            let mineTop4 = Set(mine.prefix(4))
            if mine != expectedIds {
                print("🔎 [inkling2 logits] position \(position) reorders: top4 \(mineTop4 == expectedTop4 ? "same" : "DIFFERENT")")
                print("🔎 [inkling2 logits]   reference \(expectedIds) \(expectedLogits.map { round($0 * 1000) / 1000 })")
                print("🔎 [inkling2 logits]   mine      \(mine) \(mine.map { round(logits[$0] * 1000) / 1000 })")
            } else {
                print("🔎 [inkling2 logits] position \(position) agrees: \(mine)")
            }
            XCTAssertEqual(mine.first, expectedIds.first, "argmax at position \(position)")
            XCTAssertEqual(mineTop4, expectedTop4, "top-4 set at position \(position)")
        }
        withExtendedLifetime(engine) {}
    }

    // MARK: The shipped path's own logits

    /// The DEVICE path's logits, position by position: the whole prompt, then
    /// the reference's own greedy continuation, with the margin at the step this
    /// port's run left it.
    ///
    /// The ranking case above measures the HOST graph, on the prompt only. A
    /// turn runs `Inkling2MetalForward` over the prompt plus every token it has
    /// generated, so "the continuation diverges at the ninth token" needs a
    /// number no host table can give: the two distributions at that step. A
    /// near tie is the model being sensitive, a clear flip is a stage bug, and
    /// only the same input tells them apart.
    ///
    /// The reference is teacher-forced over exactly these ids
    /// (`/tmp/inkling2_ref_tail.py`, which reproduces its own greedy run), so
    /// every step up to the divergence is the reference's own path and only the
    /// suffix after it is forced.
    ///
    /// Writes `/tmp/inkling2_port_device_rows.bin` (every position's full
    /// logits, little-endian Float32, the input a stage bisect needs) beside
    /// `/tmp/inkling2_port_device_rows.json`, and prints the table.
    /// Opt-in with `/tmp/dynamoe_inkling2_tail`.
    func testWhereTheDeviceContinuationLeavesTheReference() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_tail"),
            "opt-in device divergence; touch /tmp/dynamoe_inkling2_tail to run"
        )
        let tailPath = ["/tmp/inkling2_tool_logits_tail_fp32.json", "/tmp/inkling2_tool_logits_tail.json"]
            .first { FileManager.default.fileExists(atPath: $0) } ?? ""
        guard let tailData = try? Data(contentsOf: URL(fileURLWithPath: tailPath)),
              let tailRoot = try? JSONSerialization.jsonObject(with: tailData) as? [String: Any],
              let promptIds = tailRoot["promptIds"] as? [Int],
              let continuation = tailRoot["continuation"] as? [Int],
              let tailPositions = tailRoot["positions"] as? [[String: Any]] else {
            throw XCTSkip("no reference tail at \(tailPath): run /tmp/inkling2_ref_tail.py")
        }
        // The prompt's own table comes from the full-range dump when it exists:
        // the tail file only carries the positions the continuation needs.
        var reference: [Int: (ids: [Int], logits: [Double])] = [:]
        func absorb(_ entries: [[String: Any]]) {
            for entry in entries {
                guard let position = entry["position"] as? Int,
                      let ids = entry["ids"] as? [Int],
                      let logits = entry["logits"] as? [Double] else { continue }
                reference[position] = (ids, logits)
            }
        }
        absorb(tailPositions)
        if let data = try? Data(contentsOf: URL(fileURLWithPath: "/tmp/inkling2_tool_logits_ref_fp32.json")),
           let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let entries = root["positions"] as? [[String: Any]] {
            absorb(entries)
        }
        print("🔎 [inkling2 device] reference: \(tailPath), \(reference.count) position(s)")
        // The reference's full rows for the tail, so the cosine is over the whole
        // vocabulary and not only over the eight ids both sides printed.
        var fullReference: (first: Int, width: Int, rows: [Float])?
        if let indexData = try? Data(contentsOf: URL(fileURLWithPath: "/tmp/inkling2_tool_logits_tail_fp32.index.json")),
           let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
           let first = index["firstPosition"] as? Int,
           let width = index["rowWidth"] as? Int,
           let count = index["rowCount"] as? Int,
           let bin = try? Data(contentsOf: URL(fileURLWithPath: "/tmp/inkling2_tool_logits_tail_fp32.bin")) {
            let floats: [Float] = bin.withUnsafeBytes { raw in
                (0..<(count * width)).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
            }
            fullReference = (first, width, floats)
        }

        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let (device, library, queue) = try environment()
        let shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let forward = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers
        )
        let executor = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: forward
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config, stagedPositions: 2_048)
        forward.visitRunner = { layer, pass, hidden, blockStates, messages, position in
            try executor.runVisit(
                layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                messages: messages, position: position, workspace: workspace
            )
        }
        defer { forward.visitRunner = nil }

        let inputIds = promptIds + continuation
        var rows: [[Float]] = []
        rows.reserveCapacity(inputIds.count)
        let started = Date()
        for (position, id) in inputIds.enumerated() {
            rows.append(try forward.forwardToken(token: UInt32(id), position: position))
        }
        let elapsed = Date().timeIntervalSince(started)
        print("🔎 [inkling2 device] walked \(inputIds.count) position(s) in \(round(elapsed * 10) / 10) s")

        func ranking(_ values: [Float], _ count: Int) -> [Int] {
            Array(values.indices.sorted { values[$0] > values[$1] }.prefix(count))
        }
        func cosine(_ a: [Float], _ b: ArraySlice<Float>) -> Double {
            var dot = 0.0, left = 0.0, right = 0.0
            for index in 0..<min(a.count, b.count) {
                let value = Double(a[index])
                let other = Double(b[b.startIndex + index])
                dot += value * other
                left += value * value
                right += other * other
            }
            return dot / (left.squareRoot() * right.squareRoot())
        }

        // The prompt half: the same comparison the ranking case makes on the
        // host, so the two paths' fidelity is one table apart.
        var promptScored = 0, promptArgmax = 0, promptTop4 = 0
        for position in 0..<min(promptIds.count, rows.count) {
            guard let expected = reference[position] else { continue }
            let mine = ranking(rows[position], 8)
            promptScored += 1
            if mine.first == expected.ids.first { promptArgmax += 1 }
            if Set(mine.prefix(4)) == Set(expected.ids.prefix(4)) { promptTop4 += 1 }
            guard mine.first != expected.ids.first || Set(mine.prefix(4)) != Set(expected.ids.prefix(4)) else {
                continue
            }
            print("🔎 [inkling2 device] prompt \(position): mine \(mine.prefix(4)) ref \(expected.ids.prefix(4))")
        }
        print("🔎 [inkling2 device] prompt: argmax \(promptArgmax)/\(promptScored), top-4 \(promptTop4)/\(promptScored) against FP32")

        // The continuation: every step's distribution, and the margin at the
        // step the two paths part. This is the table the gate has been missing.
        let promptCount = promptIds.count
        var entries: [[String: Any]] = []
        for step in 0..<continuation.count {
            let position = promptCount + step
            guard position < rows.count else { break }
            let row = rows[position]
            let mine = ranking(row, 8)
            let following = step + 1 < continuation.count ? continuation[step + 1] : -1
            var entry: [String: Any] = [
                "position": position,
                "step": step,
                "inputId": continuation[step],
                "following": following,
                "ids": mine,
                "logits": mine.map { Double(row[$0]) },
            ]
            if let expected = reference[position] {
                entry["referenceIds"] = expected.ids
                entry["referenceLogits"] = expected.logits
            }
            if let fullReference, position >= fullReference.first,
               position < fullReference.first + fullReference.rows.count / max(1, fullReference.width) {
                let offset = (position - fullReference.first) * fullReference.width
                let slice = fullReference.rows[offset..<(offset + fullReference.width)]
                entry["cosine"] = cosine(row, slice)
            }
            entries.append(entry)
            let agrees: Bool? = reference[position].map { $0.ids.first == mine.first }
            let detail: String
            if let expected = reference[position] {
                let margin = expected.logits[0] - expected.logits[1]
                let rank = expected.ids.firstIndex(of: mine[0]).map { "#\($0 + 1)" } ?? "outside top-8"
                detail = String(
                    format: "ref %@ %.3f (margin %.3f), mine %@ %.3f (%@), next %d",
                    String(describing: expected.ids[0]), expected.logits[0], margin,
                    String(describing: mine[0]), row[mine[0]], rank, following
                )
            } else {
                detail = "no reference row"
            }
            let verdict = agrees == true ? "agrees" : "DIFFERENT"
            let cosineText = entry["cosine"].map { String(format: "%.6f", $0 as? Double ?? 0) } ?? "-"
            print("🔎 [inkling2 device] step \(step) pos \(position) \(verdict): \(detail), cos \(cosineText)")
        }

        let flat = rows.flatMap { $0 }
        try flat.withUnsafeBufferPointer { buffer in
            try Data(buffer: buffer).write(to: URL(fileURLWithPath: "/tmp/inkling2_port_device_rows.bin"))
        }
        let dump: [String: Any] = [
            "promptIds": promptIds,
            "continuation": continuation,
            "positionCount": rows.count,
            "rowWidth": rows.first?.count ?? 0,
            "reference": tailPath,
            "seconds": elapsed,
            "promptArgmaxAgreement": promptArgmax,
            "promptTop4Agreement": promptTop4,
            "promptScored": promptScored,
            "positions": entries,
        ]
        try JSONSerialization.data(withJSONObject: dump, options: [.prettyPrinted])
            .write(to: URL(fileURLWithPath: "/tmp/inkling2_port_device_rows.json"))
        print("🔎 [inkling2 device] wrote \(rows.count) position(s) to /tmp/inkling2_port_device_rows.bin")
        // Structural, so a truncated walk cannot read as a result: one row per
        // input id at the checkpoint's own width, every value finite.
        XCTAssertEqual(rows.count, inputIds.count, "a row per input position")
        XCTAssertGreaterThan(rows.first?.count ?? 0, 0, "a non-empty vocabulary")
        XCTAssertTrue(rows.allSatisfy { $0.allSatisfy(\.isFinite) }, "every logit finite")
        withExtendedLifetime(engine) {}
    }

    // MARK: The whole device stack against the host

    /// Visit by visit, the device executor's output against the host graph's,
    /// on the real checkpoint at the real widths.
    ///
    /// `testTheDeviceVisitMatchesTheHostGraph` compares ONE visit, with a
    /// synthetic hidden state and a one-entry stack, at a 3e-3 bar. A position
    /// runs 48 visits over the same weights and its logits are the final read
    /// over the whole stack, so a defect that only grows with the stack, with a
    /// pass's messages, or with the visit index can stay invisible there while
    /// the logits are already far from the host's. That is exactly what
    /// happened: the dynamic K/V convolution of a FULL-attention layer read its
    /// taps as the static one's channel-major, oldest-first layout, so visits
    /// 0-4 (all local) agreed to 1e-6, the first full layer sat at 5.6e-2, and
    /// the logits at position 0 were 2.3e-1 off the host until it was fixed.
    ///
    /// Position 0 covers every one of the 48 visits and is what the default run
    /// takes; `/tmp/dynamoe_inkling2_visit_parity` adds position 1, which is the
    /// only one that attends over a row of HISTORY. The bar is 1e-3 -- three
    /// orders above the ~2e-6 this graph measures, and two below the defect it
    /// exists to catch.
    func testEveryVisitMatchesTheHostAtThePublishedWidths() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        // The agent gate's own prompt: the graph the turn runs, without the turn.
        let tools = [Inkling2ToolDeclaration(
            name: "read_file",
            description: "Read a file from disk and return its contents.",
            parameters: [
                "type": .string("object"),
                "properties": .object(["path": .object(["type": .string("string")])]),
                "required": .array([.string("path")])
            ]
        )]
        let prompt = try Inkling2Prompt().render(
            turns: [
                .system("You are a helpful assistant."),
                .user([.text("Call read_file now with path /tmp/notes.txt, then tell me what it says.")])
            ],
            effort: 0, tools: tools
        )
        let promptIds = try tokenizer.encode(text: prompt).map { Int($0) }
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let rawSummary = try engine.getSummary()
        let summary = try inkling2MetalReadySummary(rawSummary)
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let (device, library, queue) = try environment()
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let host = try Inkling2CPUForward(config: config, weights: source)
        // The device runs through the forward it is walking: `runVisit` reads
        // `host.states` for the KV rows its kernels append to, so pointing it at
        // a different forward would run the graph over a cache nobody wrote.
        let deviceHost = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers,
            fileOrigins: InferenceEngine.shardFileOrigins(original: rawSummary, segmented: summary)
        )
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: deviceHost
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config, stagedPositions: 2_048)
        // The arenas a turn installs. Without them the mixture's mmap binding is
        // refused by name on a device that cannot wire two bank segments at
        // once, which is this one -- so a 48-visit walk has to run the way a
        // turn runs it.
        let arenaLine = try inkling2InstallExpertArenas(
            executor: metal, weights: deviceWeights, summary: summary, snapshot: snapshot,
            config: config, catalog: catalog, positions: 2_048
        )
        XCTAssertFalse(arenaLine.contains("no arena"), arenaLine)
        print("🔎 [inkling2 parity] arenas:\(arenaLine)")
        addTeardownBlock { metal.removeExpertArena() }

        var hostVisits: [Int: [Float]] = [:]
        var hostOrder: [(index: Int, layer: Int, pass: Int)] = []
        host.onVisitComplete = { index, layer, pass, output in
            hostVisits[index] = output
            hostOrder.append((index, layer, pass))
        }
        var deviceVisits: [Int: [Float]] = [:]

        func worst(_ actual: [Float], _ expected: [Float]) -> (relative: Float, cosine: Double) {
            guard actual.count == expected.count, !actual.isEmpty else { return (.infinity, 0) }
            var scale: Float = 0, maximum: Float = 0, dot = 0.0, left = 0.0, right = 0.0
            for index in 0..<actual.count {
                scale = max(scale, abs(expected[index]))
                maximum = max(maximum, abs(actual[index] - expected[index]))
                let a = Double(actual[index]), b = Double(expected[index])
                dot += a * b
                left += a * a
                right += b * b
            }
            return (scale > 0 ? maximum / scale : 0, dot / (left.squareRoot() * right.squareRoot()))
        }

        let extended = FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_visit_parity")
        let positions = min(extended ? 2 : 1, promptIds.count)
        for position in 0..<positions {
            hostVisits.removeAll()
            hostOrder.removeAll()
            let hostLogits = try host.forwardToken(token: UInt32(promptIds[position]), position: position)

            deviceVisits.removeAll()
            deviceHost.visitRunner = { layer, pass, hidden, blockStates, messages, visitPosition in
                let (output, queries) = try metal.runVisit(
                    layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                    messages: messages, position: visitPosition, workspace: workspace
                )
                if let index = try? deviceHost.visitIndex(layer: layer, pass: pass) {
                    deviceVisits[index] = output
                }
                return (output, queries)
            }
            let deviceLogits = try deviceHost.forwardToken(token: UInt32(promptIds[position]), position: position)
            deviceHost.visitRunner = nil

            let logits = worst(deviceLogits, hostLogits)
            print(String(
                format: "🔎 [inkling2 parity] position %d logits: rel %.3e cos %.6f",
                position, logits.relative, logits.cosine
            ))
            var firstBad: Int?
            var worstVisit: (index: Int, relative: Float) = (-1, 0)
            for entry in hostOrder {
                guard let expected = hostVisits[entry.index], let actual = deviceVisits[entry.index] else {
                    print("🔎 [inkling2 parity] visit \(entry.index) missing on one side")
                    continue
                }
                let error = worst(actual, expected)
                if error.relative > worstVisit.relative {
                    worstVisit = (entry.index, error.relative)
                }
                if firstBad == nil && error.relative > 1e-2 {
                    firstBad = entry.index
                }
                print(String(
                    format: "🔎 [inkling2 parity] visit %2d layer %2d pass %d: rel %.3e cos %.6f",
                    entry.index, entry.layer, entry.pass, error.relative, error.cosine
                ))
            }
            print("🔎 [inkling2 parity] position \(position): first visit over 1e-2 = \(firstBad.map(String.init) ?? "none"), worst \(worstVisit.index) at \(worstVisit.relative)")
            // Structural: every visit ran on both sides and the loop's own
            // schedule is the same one -- a missing visit would otherwise read
            // as "no visit over the bar".
            XCTAssertEqual(deviceVisits.count, hostVisits.count, "every visit on both sides")
            XCTAssertLessThanOrEqual(
                logits.relative, 1e-3, "position \(position)'s logits against the host"
            )
            XCTAssertLessThanOrEqual(
                worstVisit.relative, 1e-3, "visit \(worstVisit.index) at position \(position)"
            )
        }
        withExtendedLifetime(engine) {}
    }

    /// One shard's zero-copy Metal buffer, or nil when the segment cannot be
    /// mapped.
    private func deviceBuffer(_ shard: ShardMetadata, _ pointer: UnsafeMutableRawPointer) -> MTLBuffer? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return device.makeBuffer(
            bytesNoCopy: pointer, length: Int(shard.length), options: .storageModeShared, deallocator: nil
        )
    }

    // MARK: The server's text routes

    /// Both API protocols answer a text turn for this family over real HTTP,
    /// non-streaming and streaming, with the terminal frames the protocol
    /// defines.
    ///
    /// The routes are shared with every other family, which is exactly why this
    /// is checked here rather than assumed: the family supplies a forward and a
    /// prompt formatter, and a route that reaches neither -- or that expects a
    /// generic engine path this family refuses -- fails only on a real
    /// checkpoint.
    ///
    /// Opt-in: a host turn is minutes, and four turns is most of an hour.
    /// `touch /tmp/dynamoe_inkling2_server_routes` to run it.
    func testBothProtocolsAnswerTextTurnsOverTheLocalServer() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_server_routes"),
            "opt-in server routes; touch /tmp/dynamoe_inkling2_server_routes to run"
        )
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal GPU device") }
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        let modelConfig = try XCTUnwrap(ModelConfig.load(from: snapshot))
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let indexed = try engine.getSummary()
        let summary = try InferenceEngine.segmentShardsForMetal(
            indexed, maxBufferLength: UInt64(device.maxBufferLength)
        )
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil
                  ) else { continue }
            buffers[shard.index] = buffer
        }
        CompletionEngine.shared.configure(
            summary: summary, modelConfig: modelConfig, tokenizer: tokenizer,
            shardBuffers: buffers, activeLoadedModelPath: snapshot.path
        )

        let defaults = UserDefaults.standard
        let saved = ["enabled": ServerController.enabledKey, "port": ServerController.portKey,
                     "key": ServerController.apiKeyKey,
                     "capture": ServerRequestLogRecorder.captureBodiesKey].map {
            ($0.value, defaults.object(forKey: $0.value))
        }
        func restore() {
            for (key, value) in saved {
                if let value { defaults.set(value, forKey: key) } else { defaults.removeObject(forKey: key) }
            }
            ServerRequestLogRecorder.shared.useOverrideStore(nil)
            CompletionEngine.shared.configure(
                summary: nil, modelConfig: nil, tokenizer: nil,
                shardBuffers: [:], activeLoadedModelPath: nil
            )
        }
        let port = 8796
        let store = ServerRequestLogStore.inMemory()
        defaults.set(true, forKey: ServerController.enabledKey)
        defaults.set(port, forKey: ServerController.portKey)
        defaults.removeObject(forKey: ServerController.apiKeyKey)
        defaults.set(true, forKey: ServerRequestLogRecorder.captureBodiesKey)
        ServerRequestLogRecorder.shared.useOverrideStore(store)
        if ServerController.shared.isRunning { await ServerController.shared.stop() }
        ServerController.shared.start()
        do {
            let deadline = Date().addingTimeInterval(20)
            var listening = false
            while Date() < deadline {
                if let (_, response) = try? await URLSession.shared.data(
                    from: URL(string: "http://127.0.0.1:\(port)/health")!
                ), (response as? HTTPURLResponse)?.statusCode == 200 {
                    listening = true
                    break
                }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            try XCTSkipUnless(listening, "the local server never answered /health on port \(port)")

            // One non-streaming OpenAI turn.
            let openAIBody = """
            {"model":"any","max_tokens":2,"messages":[{"role":"user","content":"The capital of France is"}]}
            """
            var openAIRequest = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            openAIRequest.httpMethod = "POST"
            openAIRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            openAIRequest.httpBody = Data(openAIBody.utf8)
            openAIRequest.timeoutInterval = 900
            let (openAIData, openAIResponse) = try await URLSession.shared.data(for: openAIRequest)
            let openAIHTTP = try XCTUnwrap(openAIResponse as? HTTPURLResponse)
            print("🔎 [inkling2 server] openai \(openAIHTTP.statusCode): \(String(decoding: openAIData.prefix(400), as: UTF8.self))")
            XCTAssertEqual(openAIHTTP.statusCode, 200, "the OpenAI route must serve this family")
            let openAIJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: openAIData) as? [String: Any])
            XCTAssertEqual(openAIJSON["object"] as? String, "chat.completion")
            let openAIChoice = try XCTUnwrap((openAIJSON["choices"] as? [[String: Any]])?.first)
            XCTAssertNotNil(openAIChoice["finish_reason"] as? String)
            XCTAssertNotNil((openAIJSON["usage"] as? [String: Any])?["prompt_tokens"])

            // One streaming Anthropic turn: the frames must end at message_stop
            // and the start frame must carry the prompt length.
            let anthropicBody = """
            {"model":"any","max_tokens":2,"stream":true,"messages":[{"role":"user","content":"The capital of France is"}]}
            """
            var anthropicRequest = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/messages")!)
            anthropicRequest.httpMethod = "POST"
            anthropicRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            anthropicRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            anthropicRequest.httpBody = Data(anthropicBody.utf8)
            anthropicRequest.timeoutInterval = 900
            let (anthropicData, anthropicResponse) = try await URLSession.shared.data(for: anthropicRequest)
            let anthropicHTTP = try XCTUnwrap(anthropicResponse as? HTTPURLResponse)
            let transcript = String(decoding: anthropicData, as: UTF8.self)
            print("🔎 [inkling2 server] anthropic stream \(anthropicHTTP.statusCode): \(transcript.prefix(600))")
            XCTAssertEqual(anthropicHTTP.statusCode, 200)
            XCTAssertTrue(
                (anthropicHTTP.value(forHTTPHeaderField: "Content-Type") ?? "").contains("text/event-stream")
            )
            // A frame is a block ended by a blank line; the terminal marker is
            // the protocol's own, so it is asserted by NAME rather than by the
            // stream merely ending.
            let eventNames = transcript.components(separatedBy: "\n\n").compactMap { block -> String? in
                block.split(separator: "\n").first { $0.hasPrefix("event:") }.map {
                    String($0.dropFirst("event:".count)).trimmingCharacters(in: .whitespaces)
                }
            }
            print("🔎 [inkling2 server] anthropic events: \(eventNames)")
            XCTAssertTrue(eventNames.contains("message_start"), "no message_start frame")
            XCTAssertTrue(eventNames.contains("message_stop"), "the stream never reached message_stop")
            XCTAssertFalse(eventNames.contains("error"), "the stream carried an error frame")

            let rows = try await {
                let deadline = Date().addingTimeInterval(15)
                while Date() < deadline {
                    let rows = await store.fetch(limit: 50)
                    if rows.count >= 2 { return rows }
                    try await Task.sleep(nanoseconds: 50_000_000)
                }
                return await store.fetch(limit: 50)
            }()
            print("🔎 [inkling2 server] log rows: \(rows.map { "\($0.method) \($0.path) \($0.httpStatus) streaming=\($0.wasStreaming)" })")
            XCTAssertEqual(Set(rows.map(\.path)), ["/v1/chat/completions", "/v1/messages"])
        } catch {
            await ServerController.shared.stop()
            restore()
            throw error
        }
        await ServerController.shared.stop()
        restore()
        withExtendedLifetime(engine) {}
    }
    // MARK: The whole visit

    /// One complete visit on the device against the host's own `runVisit`, on
    /// the real checkpoint: every stage of a visit in its real order, through
    /// the loop's own schedule.
    ///
    /// Two host forwards over the same weights, because both sides mutate the
    /// visit's KV rows and convolution histories.
    func testTheDeviceVisitMatchesTheHostGraph() throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        let (device, library, queue) = try environment()
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let rawSummary = try engine.getSummary()
        let summary = try inkling2MetalReadySummary(rawSummary)
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        shardBuffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: shardBuffers)
        let oracle = try Inkling2CPUForward(config: config, weights: source)
        let reference = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers,
            fileOrigins: InferenceEngine.shardFileOrigins(original: rawSummary, segmented: summary)
        )
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: reference
        )
        let workspace = try Inkling2DeviceWorkspace(device: device, config: config)
        // The arenas a turn installs: without them the mixture's mmap binding is
        // refused by name on a device that cannot wire two bank segments at once.
        let arenaLine = try inkling2InstallExpertArenas(
            executor: metal, weights: deviceWeights, summary: summary, snapshot: snapshot,
            config: config, catalog: catalog, positions: 8
        )
        XCTAssertFalse(arenaLine.contains("no arena"), arenaLine)
        addTeardownBlock { metal.removeExpertArena() }

        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        let ids = try tokenizer.encode(text: "Hi there")
        for (position, id) in ids.enumerated() {
            _ = try oracle.forwardToken(token: UInt32(id), position: position)
            _ = try reference.forwardToken(token: UInt32(id), position: position)
        }
        let position = ids.count
        // Layer 0 is a sliding layer, so the visit runs the shared-norm branch.
        let layer = 0
        let input = values(config.hiddenSize, seed: 71)
        // The stack the visit reads: the block read is over these plus the partial.
        var oracleStack = [values(config.hiddenSize, seed: 72)]
        var deviceStack = oracleStack

        let expected = try oracle.runVisit(
            layer: layer, pass: 0, hidden: input, blockStates: &oracleStack,
            messages: nil, position: position
        )
        let actual = try metal.runVisit(
            layer: layer, pass: 0, hidden: input, blockStates: &deviceStack,
            messages: nil, position: position, workspace: workspace
        )

        let scale = expected.output.reduce(Float(0)) { max($0, abs($1)) }
        let worst = zip(actual.output, expected.output).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        print("🔎 [inkling2 device] whole visit: \(worst / scale) relative")
        XCTAssertEqual(actual.attentionQueries, expected.attentionQueries)
        XCTAssertEqual(deviceStack.count, oracleStack.count)
        XCTAssertLessThanOrEqual(worst / scale, 3e-3, "the device visit against the host graph")
    }

    // MARK: The app's own turn path

    /// The whole app path for this family, once on the host and once with the
    /// device visit runner installed: `CompletionEngine.launchTurn` with the
    /// checkpoint loaded the way the app loads it, and the same greedy answer
    /// from both.
    ///
    /// This is the only case that covers the wiring -- the engine's cache of
    /// the forward and the device half, the runner's installation inside the
    /// turn task, and the KV budget -- and it is where a mistake in that
    /// plumbing shows up, since the halves themselves are verified elsewhere.
    ///
    /// Opt-in: a host turn is minutes per token, so the default suite stays
    /// fast. `touch /tmp/dynamoe_inkling2_engine_turn` to run it.
    func testTheEngineAnswersTheSameOnTheHostAndOnTheDevice() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_engine_turn"),
            "opt-in engine turn; touch /tmp/dynamoe_inkling2_engine_turn to run"
        )
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal GPU device") }

        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        let modelConfig = try XCTUnwrap(ModelConfig.load(from: snapshot))
        XCTAssertTrue(modelConfig.isInkling2Model)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let indexed = try engine.getSummary()
        let summary = try InferenceEngine.segmentShardsForMetal(
            indexed, maxBufferLength: UInt64(device.maxBufferLength)
        )
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil
                  ) else { continue }
            buffers[shard.index] = buffer
        }
        CompletionEngine.shared.configure(
            summary: summary, modelConfig: modelConfig, tokenizer: tokenizer,
            shardBuffers: buffers, activeLoadedModelPath: snapshot.path,
            // The turn's expert arena reads the FILE, so it needs to know where
            // each segment starts inside it.
            shardFileOrigins: InferenceEngine.shardFileOrigins(
                original: indexed, segmented: summary
            )
        )
        defer {
            CompletionEngine.shared.configure(
                summary: nil, modelConfig: nil, tokenizer: nil,
                shardBuffers: [:], activeLoadedModelPath: nil
            )
        }

        let rendered = try Inkling2Prompt().render(
            turns: [.user([.text("The capital of France is")])], effort: 0
        )
        let promptIds = try tokenizer.encode(text: rendered)
        func request(_ ids: [UInt32]) -> CompletionTurnRequest {
            CompletionTurnRequest(
                promptTokenIds: ids,
                formattedPrompt: rendered,
                sessionId: nil,
                sampling: TurnSampling(
                    temperature: 0, topP: 1, minP: 0, topK: 1,
                    repetitionPenalty: 1, maxTokens: 2
                ),
                memory: TurnMemorySettings(
                    executionMode: .residentRAM, budgetMode: .unrestricted, kvPrecision: .fp32
                ),
                agentToolsEnabled: false,
                thinkingEnabled: false,
                agentStep: 0,
                forceSynthesis: false,
                isSplicedContinuation: false,
                stopSequences: [],
                asOfDate: nil,
                images: []
            )
        }
        func answer(_ request: CompletionTurnRequest) async throws -> (ids: [UInt32], text: String) {
            var ids: [UInt32] = []
            var text = ""
            for await event in try CompletionEngine.shared.launchTurn(request).stream {
                switch event {
                case .finished(let result):
                    ids = result.generatedTokenIds
                    text = result.rawDecodedText
                case .failedTurn(let failure):
                    XCTFail("the turn failed: \(failure.message)")
                default:
                    break
                }
            }
            return (ids, text)
        }

        // Two arms of ONE process, the host graph forced by the debug seam:
        // a difference between them is a difference in the visit executor, not
        // in the machine or the checkpoint.
        let defaults = UserDefaults.standard
        let seam = "dynamoe_inkling2_device_off"
        let savedSeam = defaults.object(forKey: seam)
        defer {
            if let savedSeam { defaults.set(savedSeam, forKey: seam) }
            else { defaults.removeObject(forKey: seam) }
        }
        defaults.set(true, forKey: seam)
        var started = CFAbsoluteTimeGetCurrent()
        let hostArm = try await answer(request(promptIds))
        let hostSeconds = CFAbsoluteTimeGetCurrent() - started
        defaults.set(false, forKey: seam)
        started = CFAbsoluteTimeGetCurrent()
        let deviceArm = try await answer(request(promptIds))
        let deviceSeconds = CFAbsoluteTimeGetCurrent() - started
        print("🔎 [inkling2 turn] host: \(hostArm.ids) \(hostArm.text.debugDescription) in \(String(format: "%.1f", hostSeconds)) s")
        print("🔎 [inkling2 turn] device: \(deviceArm.ids) \(deviceArm.text.debugDescription) in \(String(format: "%.1f", deviceSeconds)) s")
        XCTAssertFalse(hostArm.ids.isEmpty, "the host turn produced no tokens")
        XCTAssertEqual(deviceArm.ids, hostArm.ids, "the device visit runner changed the answer")
        withExtendedLifetime(engine) {}
    }

    /// A complete tool call through the APP's engine: agent tools enabled, the
    /// checkpoint loaded the way the app loads it, and the `TurnResult` the
    /// harness reads carrying one actionable call.
    ///
    /// The runner-level gate proves the graph and the dialect. This proves the
    /// engine's half of the same contract: the device visit runner installed
    /// inside the turn task, the freeze hook keyed on the family's own block
    /// (`inklingRunAgentTools`), canonicalization into the shared parser, and
    /// the parsed call landing in `TurnResult.parsedToolCalls`. The tool is not
    /// EXECUTED here -- execution, approval and the result turn are the
    /// harness's shared path, and the continuation after a result is asserted
    /// on the runner by `testATurnAnswersAfterAToolResultOnThePublishedCheckpoint`.
    ///
    /// Default when the checkpoint is present: the device path makes it ~20 s.
    func testAnAgentTurnFreezesACompleteCallThroughTheEngine() async throws {
        let snapshot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: snapshot.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(snapshot.path)")
        }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal GPU device") }
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        )
        let modelConfig = try XCTUnwrap(ModelConfig.load(from: snapshot))
        XCTAssertTrue(modelConfig.isInkling2Model)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let indexed = try engine.getSummary()
        let summary = try InferenceEngine.segmentShardsForMetal(
            indexed, maxBufferLength: UInt64(device.maxBufferLength)
        )
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil
                  ) else { continue }
            buffers[shard.index] = buffer
        }
        CompletionEngine.shared.configure(
            summary: summary, modelConfig: modelConfig, tokenizer: tokenizer,
            shardBuffers: buffers, activeLoadedModelPath: snapshot.path,
            // The app's own load passes these, and the turn's expert arena reads
            // the FILE: without them its slots hold segment-relative bytes.
            shardFileOrigins: InferenceEngine.shardFileOrigins(
                original: indexed, segmented: summary
            )
        )
        // The DEVICE arm: a host turn is minutes, and the host graph's own
        // fidelity is pinned elsewhere. A developer's seam value is restored.
        let defaults = UserDefaults.standard
        let seam = "dynamoe_inkling2_device_off"
        let savedSeam = defaults.object(forKey: seam)
        defaults.set(false, forKey: seam)
        defer {
            CompletionEngine.shared.configure(
                summary: nil, modelConfig: nil, tokenizer: nil,
                shardBuffers: [:], activeLoadedModelPath: nil
            )
            if let savedSeam { defaults.set(savedSeam, forKey: seam) }
            else { defaults.removeObject(forKey: seam) }
        }

        let tools = [Inkling2ToolDeclaration(
            name: "read_file",
            description: "Read a file from disk and return its contents.",
            parameters: [
                "type": .string("object"),
                "properties": .object(["path": .object(["type": .string("string")])]),
                "required": .array([.string("path")])
            ]
        )]
        let prompt = try Inkling2Prompt().render(
            turns: [
                .system("You are a helpful assistant."),
                .user([.text("Call read_file now with path /tmp/notes.txt, then tell me what it says.")])
            ],
            effort: 0, tools: tools
        )
        let promptIds = try tokenizer.encode(text: prompt)
        let request = CompletionTurnRequest(
            promptTokenIds: promptIds,
            formattedPrompt: prompt,
            sessionId: nil,
            sampling: TurnSampling(
                temperature: 0, topP: 1, minP: 0, topK: 1,
                repetitionPenalty: 1, maxTokens: 384
            ),
            memory: TurnMemorySettings(
                executionMode: .residentRAM, budgetMode: .unrestricted, kvPrecision: .fp32
            ),
            agentToolsEnabled: true,
            thinkingEnabled: false,
            agentStep: 0,
            forceSynthesis: false,
            isSplicedContinuation: false,
            stopSequences: [],
            asOfDate: nil,
            images: []
        )
        var result: TurnResult?
        for await event in try CompletionEngine.shared.launchTurn(request).stream {
            switch event {
            case .finished(let turn):
                result = turn
            case .failedTurn(let failure):
                XCTFail("the agent turn failed: \(failure.message)")
            default:
                break
            }
        }
        let turn = try XCTUnwrap(result, "the turn emitted no result")
        print("🔎 [inkling2 engine] agent turn: \(turn.finishReason.rawValue) after \(turn.tokensGenerated) token(s), \(turn.parsedToolCalls.count) call(s)")
        print("🔎 [inkling2 engine] text: \(String(turn.rawDecodedText.suffix(400)).debugDescription)")
        print(
            "🔎 [inkling2 engine] prompt \(turn.promptTokenIds.count) token(s), opening "
                + "\(Array(turn.generatedTokenIds.prefix(8)))"
        )
        print("🔎 [inkling2 engine] call: \(turn.parsedToolCalls.first.map { "\($0.name) \($0.rawArguments)" } ?? "none")")
        // What this asserts, and what it deliberately does NOT.
        //
        // The wiring this case exists for is the ENGINE's half of the call
        // contract: the device visit runner installed inside the turn task, the
        // freeze hook keyed on the family's own block, canonicalization into the
        // shared parser, and a `TurnResult` carrying the call. All of that is
        // exercised here on the shipping path.
        //
        // The CALL itself is not asserted, because this prompt's fourth greedy
        // token is a tie the checkpoint itself decides two ways: the pinned FP32
        // reference continues `171960` (which the HOST head reproduces -- see
        // `testATurnAnswersAfterAToolResultOnThePublishedCheckpoint`, which
        // asserts the well-formed call), and its own BF16 run picks `159306`,
        // which is what the DEVICE head picks. The device head is pinned against
        // the host at 3.8e-07 relative with the same argmax on a real row
        // (`testTheHeadMatchesTheHostUnembedding`), so this is rounding at a tie
        // rather than a wrong stage -- and on that branch the model writes its
        // invoke block with malformed JSON.
        XCTAssertTrue(
            [TurnFinishReason.toolFreeze, .eos].contains(turn.finishReason),
            "the turn must finish, not fail: \(turn.finishReason.rawValue)"
        )
        XCTAssertGreaterThan(turn.tokensGenerated, 8, "a turn that reasoned and called")
        XCTAssertTrue(
            turn.rawDecodedText.contains("read_file"),
            "the model names the tool it was asked for"
        )
        XCTAssertTrue(
            turn.rawDecodedText.contains("/tmp/notes.txt"),
            "and the path it was asked about"
        )
        if !turn.parsedToolCalls.isEmpty {
            // If the tie went the reference's way the call must be actionable.
            let call = try XCTUnwrap(turn.parsedToolCalls.first)
            XCTAssertEqual(call.name, "read_file")
            XCTAssertEqual(call.arguments["path"] as? String, "/tmp/notes.txt")
        }
        withExtendedLifetime(engine) {}
    }
}
