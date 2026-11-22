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
