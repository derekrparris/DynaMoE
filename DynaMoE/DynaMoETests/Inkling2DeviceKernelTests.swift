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
        let gainBuffer = try buffer(device, gain)
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
        }
        let actual = read(outputBuffer, rows * width)
        for row in 0..<rows {
            let expected = Inkling2Math.rmsNorm(
                Array(input[(row * width)..<((row + 1) * width)]), gain: gain, eps: 1e-6
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
            var weight = [UInt8](repeating: 0, count: rows * weightRowStride)
            for index in 0..<weight.count {
                weight[index] = UInt8(0x21 + (index &* 13) % 0x5C)
            }
            var scales = [UInt8](repeating: 0, count: rows * scaleRowStride)
            for index in 0..<scales.count {
                // E4M3 near 2^0 for NVFP4; E8M0 (a biased exponent) for MXFP8.
                scales[index] = layout == .nvfp4 ? UInt8(118 + index % 9) : UInt8(124 + index % 5)
            }
            let input = values(logicalColumns, seed: 31)

            let matrix = Inkling2PackedExpertMatrix(
                layout: layout, weight: weight, weightRowStride: weightRowStride,
                scales: scales, scaleRowStride: scaleRowStride,
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

            try run("inkling_expert_row_f32", threads: (rows, 1, 1), threadgroup: (64, 1, 1),
                    threadgroupMemory: 32 * MemoryLayout<Float>.stride, byThreadgroup: true) { encoder in
                encoder.setBuffer(weightBuffer, offset: 0, index: 0)
                encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(inputBuffer, offset: 0, index: 2)
                encoder.setBuffer(outputBuffer, offset: 0, index: 3)
                for (index, b) in scalarBuffers.enumerated() { encoder.setBuffer(b, offset: 0, index: 4 + index) }
                encoder.setBuffer(layoutBuffer, offset: 0, index: 8)
                encoder.setBuffer(outerBuffer, offset: 0, index: 9)
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
        let biasBuffer = try buffer(device, bias)
        let indicesBuffer = try buffer(device, [UInt32](repeating: 99, count: positions * topK))
        let weightsBuffer = try buffer(device, [Float](repeating: 0, count: positions * topK))
        let scalars = [UInt32(numExperts), UInt32(topK)]
        let scalarBuffers = try scalars.map { try buffer(device, [$0]) }
        let scaleBuffer = try buffer(device, [routeScale])
        try run("inkling_router_topk_f32", threads: (positions, 1, 1), threadgroup: (64, 1, 1),
                threadgroupMemory: (32 + numExperts) * MemoryLayout<Float>.stride,
                byThreadgroup: true) { encoder in
            encoder.setBuffer(scoresBuffer, offset: 0, index: 0)
            encoder.setBuffer(biasBuffer, offset: 0, index: 1)
            encoder.setBuffer(indicesBuffer, offset: 0, index: 2)
            encoder.setBuffer(weightsBuffer, offset: 0, index: 3)
            for (index, b) in scalarBuffers.enumerated() { encoder.setBuffer(b, offset: 0, index: 4 + index) }
            encoder.setBuffer(scaleBuffer, offset: 0, index: 6)
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
        let summary = try engine.getSummary()
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            guard shard.length > 0,
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)) else { continue }
            if let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(shard.length), options: .storageModeShared, deallocator: nil
            ) { shardBuffers[shard.index] = buffer }
        }
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
        let rows = workspace.stage(state)
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
        let summary = try engine.getSummary()
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            guard shard.length > 0,
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)) else { continue }
            if let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(shard.length), options: .storageModeShared, deallocator: nil
            ) { shardBuffers[shard.index] = buffer }
        }
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

        // The host's own intermediate stages, by name: a mismatch then names the
        // stage instead of only the half's output.
        var oracleStages: [String: [Float]] = [:]
        oracle.onInternal = { _, name, value in oracleStages[name] = value }
        let expected = try oracle.attention(
            layer: layer, state: oracle.states[0], queryInput: queryInput,
            keyInput: keyInput, valueInput: valueInput, position: position
        )
        // The K projection and convolution on their own, so a mismatch in
        // `k_norm` is attributed to one of the two rather than to the clip:
        // the same buffers the half uses, one command buffer each.
        let probeState = Inkling2VisitState(
            isFullAttention: false, window: config.slidingWindowSize,
            stateWidths: [config.swaNumKeyValueHeads * config.swaHeadDim,
                          config.swaNumKeyValueHeads * config.swaHeadDim,
                          config.hiddenSize, config.hiddenSize],
            historyDepth: config.convKernelSize - 1
        )
        // Same row count and the same convolution history the half starts
        // from, so the probe's convolution is comparable; a copy, because the
        // convolution pushes to this history and the half must not see that.
        let sourceState = reference.states[0]
        for row in 0..<sourceState.committedPositions {
            _ = probeState.append(key: sourceState.keys[row], value: sourceState.values[row])
        }
        for index in 0..<4 {
            for entry in sourceState.history(index) { probeState.pushConvolution(index, value: entry) }
        }
        func isolatedK() throws -> (linear: [Float], convolved: [Float]) {
            let queryInputBuffer = try metal.staging(workspace.keyInput, keyInput)
            _ = queryInputBuffer
            let interrupt = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(interrupt.makeComputeCommandEncoder())
            let arena = Inkling2ScalarArena(device: device)
            try metal.gemm(
                encoder: encoder, role: .attnWkDv, layer: layer,
                input: workspace.keyInput, inputOffset: 0, output: workspace.key, outputOffset: 0,
                rows: config.swaNumKeyValueHeads * config.swaHeadDim, columns: config.hiddenSize,
                arena: arena
            )
            encoder.endEncoding()
            interrupt.commit()
            interrupt.waitUntilCompleted()
            let linear = metal.readBack(
                workspace.key, count: config.swaNumKeyValueHeads * config.swaHeadDim
            )
            let second = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder2 = try XCTUnwrap(second.makeComputeCommandEncoder())
            // A state of its own: the convolution pushes history, and the half
            // below must start from the history the host's capture did.
            try metal.convolve(
                encoder: encoder2, state: probeState, index: 0,
                values: workspace.key, valuesCount: config.swaNumKeyValueHeads * config.swaHeadDim,
                layer: layer, isFull: false, headDim: config.swaHeadDim, output: workspace.key,
                arena: arena
            )
            encoder2.endEncoding()
            second.commit()
            second.waitUntilCompleted()
            arena.clear()
            return (linear, metal.readBack(
                workspace.key, count: config.swaNumKeyValueHeads * config.swaHeadDim
            ))
        }
        // The host's matching pair, from its own captures (a fresh state so the
        // convolution history is the one the half started from).
        let kPair = try isolatedK()
        func rel(_ mine: [Float], _ ref: [Float]) -> Float {
            guard !ref.isEmpty, mine.count == ref.count else { return .infinity }
            let s = ref.reduce(Float(0)) { max($0, abs($1)) }
            return zip(mine, ref).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) } / max(s, 1e-9)
        }
        print("🔎 [inkling2 device] k path: linear \(rel(kPair.linear, oracleStages["k_linear"] ?? []))"
            + " convolved \(rel(kPair.convolved, oracleStages["k_sconv"] ?? []))")
        // The clip alone, on the same convolved vector: what is left after the
        // two stages above are exact.
        let clipBuffer = try XCTUnwrap(device.makeBuffer(
            bytes: kPair.convolved, length: kPair.convolved.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let clipCommand = try XCTUnwrap(queue.makeCommandBuffer())
        let clipEncoder = try XCTUnwrap(clipCommand.makeComputeCommandEncoder())
        let clipArena = Inkling2ScalarArena(device: device)
        let clipScale = try deviceWeights.tensor(.attnQL2ClipScale, layer: layer)
        try metal.l2Clip(
            encoder: clipEncoder, vector: clipBuffer, vectorOffset: 0,
            scale: clipScale.buffer, scaleOffset: Int(clipScale.offset),
            headDim: config.swaHeadDim, heads: config.swaNumKeyValueHeads,
            threshold: config.qkL2ClipThreshold, tau: 1, arena: clipArena
        )
        clipEncoder.endEncoding()
        clipCommand.commit()
        clipCommand.waitUntilCompleted()
        clipArena.clear()
        let clipped = Array(UnsafeBufferPointer(
            start: clipBuffer.contents().bindMemory(to: Float.self, capacity: kPair.convolved.count),
            count: kPair.convolved.count
        ))
        // The host rule applied to the same input, to separate "the kernel is
        // wrong" from "the host's k_norm is not this value".
        let hostScale = try source.matrix(.attnQL2ClipScale, layer: layer)
        var hostClipped = kPair.convolved
        for head in 0..<config.swaNumKeyValueHeads {
            let range = (head * config.swaHeadDim)..<((head + 1) * config.swaHeadDim)
            var slice = Array(hostClipped[range])
            for d in 0..<config.swaHeadDim { slice[d] *= hostScale[head * config.swaHeadDim + d] }
            slice = Inkling2Math.l2Clip(slice, threshold: config.qkL2ClipThreshold)
            for d in 0..<config.swaHeadDim { hostClipped[head * config.swaHeadDim + d] = slice[d] }
        }
        print("🔎 [inkling2 device] k clip: kernel-vs-host-rule \(rel(clipped, hostClipped))"
            + " host-rule-vs-reference \(rel(hostClipped, oracleStages["k_norm"] ?? []))")

        // The device half's own stages, read back from the workspace after it runs.
        let actual = try metal.attentionHalf(
            layer: layer, state: reference.states[0], queryInput: queryInput,
            keyInput: keyInput, valueInput: valueInput, position: position, workspace: workspace
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
            stageError("k_norm", metal.readBack(workspace.key, count: config.swaNumKeyValueHeads * config.swaHeadDim),
                       oracleStages["k_norm"] ?? []),
            stageError("v_norm", metal.readBack(workspace.value, count: config.swaNumKeyValueHeads * config.swaHeadDim),
                       oracleStages["v_norm"] ?? []),
            stageError("relative", metal.readBack(workspace.relativeStates, count: config.swaNumAttentionHeads * config.dRel),
                       oracleStages["relative_states"] ?? []),
            stageError("sink", metal.readBack(workspace.sinkLogits, count: config.swaNumAttentionHeads),
                       oracleStages["sink"] ?? []),
            stageError("attn_raw", metal.readBack(workspace.attentionOutput, count: config.swaNumAttentionHeads * config.swaHeadDim),
                       oracleStages["attn_raw"] ?? [])
        ]
        print("🔎 [inkling2 device] stages: " + stageRows.joined(separator: " "))
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
        let summary = try engine.getSummary()
        let config = try Inkling2Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var shardBuffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            guard shard.length > 0,
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)) else { continue }
            if let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(shard.length), options: .storageModeShared, deallocator: nil
            ) { shardBuffers[shard.index] = buffer }
        }
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
                values: workspace.key, valuesCount: kvHeads * headDim,
                layer: layer, isFull: false, headDim: headDim, output: workspace.key, arena: arena
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
                values: workspace.value, valuesCount: kvHeads * headDim,
                layer: layer, isFull: false, headDim: headDim, output: workspace.value, arena: arena
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
            let rows = workspace.stage(state)
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
}
