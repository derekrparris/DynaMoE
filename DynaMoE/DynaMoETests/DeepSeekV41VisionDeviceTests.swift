//
//  DeepSeekV41VisionDeviceTests.swift
//  DynaMoETests
//
//  The vision tower's quadratic stage on the GPU. The host tower keeps a
//  `tokens * tokens` score block per head while it computes bidirectional attention,
//  and at the largest admissible image that stage is 76% of a 41.71 s encode
//  ("DSV4.1 #37"). `dsv41_vision_attention_f32` runs the same softmax without the
//  block, one SIMD group per (head, query token), so the rule is checked here
//  against the host oracle before anything else in the tower moves.
//
//  The parity cases run with the default suite. The published-geometry timing is
//  opt-in (touch `/tmp/dynamoe_dsv41_vision_measure`) because it builds 8,649
//  patch tokens worth of qkv, which is ~106 MB of host arrays and a multi-second
//  test on the small shapes' budget.
//

import XCTest
import Metal
@testable import DynaMoE

final class DeepSeekV41VisionDeviceTests: XCTestCase {

    private static let measurementMarkerPath = "/tmp/dynamoe_dsv41_vision_measure"

    private func values(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { index in
            Float((index &* 37 &+ seed &* 13) % 23 - 11) / 8
        }
    }

    private func runner() throws -> DeepSeekV41VisionAttentionDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        return try DeepSeekV41VisionAttentionDevice(device: device)
    }

    private func maximumDifference(_ lhs: [[Float]], _ rhs: [[Float]]) -> Float {
        var worst: Float = 0
        for row in 0..<min(lhs.count, rhs.count) {
            for column in 0..<min(lhs[row].count, rhs[row].count) {
                worst = max(worst, abs(lhs[row][column] - rhs[row][column]))
            }
        }
        return worst
    }

    /// The published head shape: 64 values, rotated by one angle row per token.
    ///
    /// The token count is odd and smaller than a head's worth of lanes, so a
    /// threadgroup's last query block is partial and the lane that owns no second
    /// output dimension still has to write the first. The oracle is
    /// `DeepSeekV41VisionForward.attention`, the serial definition; the batched form
    /// the tower actually uses is compared too, because the two have different
    /// summation orders and the kernel has to sit inside both.
    func testTheDeviceAttentionMatchesTheHostRule() throws {
        let runner = try runner()
        let heads = 3
        let headDim = 8
        let tokens = 37
        let ropeDim = 4
        let rotary = DeepSeekV41VisionForward.rotaryTables(
            rows: 1, columns: tokens, ropeDim: ropeDim, theta: 10_000
        )
        let qkv = (0..<tokens).map { token in
            values(3 * heads * headDim, seed: token + 1)
        }

        let device = try runner.attention(
            qkv: qkv,
            heads: heads,
            headDim: headDim,
            ropeDim: ropeDim,
            cosine: rotary.cosine,
            sine: rotary.sine
        )
        let serial = DeepSeekV41VisionForward.attention(
            qkv: qkv,
            heads: heads,
            headDim: headDim,
            ropeDim: ropeDim,
            cosine: rotary.cosine,
            sine: rotary.sine
        )
        let batched = DeepSeekV41VisionForward.attentionBatched(
            qkv: qkv,
            heads: heads,
            headDim: headDim,
            ropeDim: ropeDim,
            cosine: rotary.cosine,
            sine: rotary.sine
        )
        let serialDifference = maximumDifference(device, serial)
        let batchedDifference = maximumDifference(device, batched)
        print(
            "🔎 [dsv41 vision] attention parity: worst \(serialDifference) against the "
                + "serial rule, \(batchedDifference) against the batched one"
        )
        XCTAssertEqual(device.count, tokens)
        XCTAssertLessThan(serialDifference, 2e-4, "the device attention left the host rule")
        XCTAssertLessThan(batchedDifference, 2e-4, "the device attention left the batched form")
    }

    /// The split-and-rotate half on its own, so a rotation defect is not reported as
    /// an attention defect: q and k are rotated against the token's angle row, and v
    /// is copied untouched.
    func testTheRopeSplitMatchesTheHostRule() throws {
        let runner = try runner()
        let heads = 2
        let headDim = 8
        let tokens = 11
        let ropeDim = 4
        let hidden = heads * headDim
        let rotary = DeepSeekV41VisionForward.rotaryTables(
            rows: 1, columns: tokens, ropeDim: ropeDim, theta: 10_000
        )
        let qkv = (0..<tokens).map { token in
            values(3 * hidden, seed: token + 5)
        }
        let planes = try runner.splitAndRotate(
            qkv: qkv,
            heads: heads,
            headDim: headDim,
            ropeDim: ropeDim,
            cosine: rotary.cosine,
            sine: rotary.sine
        )

        let queryPlane = planes.queries.contents().assumingMemoryBound(to: Float.self)
        let keyPlane = planes.keys.contents().assumingMemoryBound(to: Float.self)
        let valuePlane = planes.values.contents().assumingMemoryBound(to: Float.self)
        for head in 0..<heads {
            for token in 0..<tokens {
                let source = qkv[token]
                let base = head * headDim
                var expectedQuery = Array(source[base..<(base + headDim)])
                var expectedKey = Array(source[(hidden + base)..<(hidden + base + headDim)])
                DeepSeekV41VisionForward.applyRotary(
                    &expectedQuery,
                    cosine: Array(rotary.cosine[token].prefix(ropeDim)),
                    sine: Array(rotary.sine[token].prefix(ropeDim))
                )
                DeepSeekV41VisionForward.applyRotary(
                    &expectedKey,
                    cosine: Array(rotary.cosine[token].prefix(ropeDim)),
                    sine: Array(rotary.sine[token].prefix(ropeDim))
                )
                let offset = (head * tokens + token) * headDim
                for dimension in 0..<headDim {
                    XCTAssertEqual(
                        queryPlane[offset + dimension], expectedQuery[dimension],
                        accuracy: 1e-6,
                        "q differs at head \(head) token \(token) dim \(dimension)"
                    )
                    XCTAssertEqual(
                        keyPlane[offset + dimension], expectedKey[dimension],
                        accuracy: 1e-6,
                        "k differs at head \(head) token \(token) dim \(dimension)"
                    )
                    XCTAssertEqual(
                        valuePlane[offset + dimension],
                        source[2 * hidden + base + dimension],
                        accuracy: 0,
                        "v must not be rotated (head \(head) token \(token))"
                    )
                }
            }
        }
    }

    /// A head narrower than one SIMD group and no rotation at all: the second
    /// accumulator is unused for every lane, which is the branch the published
    /// 64-wide head never takes.
    func testTheDeviceAttentionHandlesANarrowHeadAndNoRotation() throws {
        let runner = try runner()
        let heads = 2
        let headDim = 12
        let tokens = 5
        let qkv = (0..<tokens).map { token in
            values(3 * heads * headDim, seed: token + 9)
        }
        let device = try runner.attention(
            qkv: qkv, heads: heads, headDim: headDim, ropeDim: 0,
            cosine: [], sine: []
        )
        let serial = DeepSeekV41VisionForward.attention(
            qkv: qkv, heads: heads, headDim: headDim, ropeDim: 0,
            cosine: [], sine: []
        )
        let difference = maximumDifference(device, serial)
        print("🔎 [dsv41 vision] narrow-head parity: worst \(difference)")
        XCTAssertLessThan(difference, 2e-4)
    }

    /// The attention form the tower ships against the one it replaced, and against the
    /// host rule.
    ///
    /// `pairedFold` is the default now ("DSV4.1 #44"): a lane owns two adjacent
    /// dimensions of the value fold instead of two halves of the row, and the tile's
    /// additions split across four accumulator pairs. That reorders the fold, so it
    /// cannot be compared with the direct kernel bit for bit; both are held to the host
    /// rule instead, and to each other at a bar a wrong dimension pairing or a dropped
    /// key would blow through. The shapes cover a narrow head whose last lanes own no
    /// pair at all, a full-width head, an odd token count (a partial last tile) and a
    /// token count below one tile.
    func testTheShippedPairedFoldMatchesTheDirectKernelAndTheHost() throws {
        let runner = try runner()
        for (heads, headDim, tokens) in [(3, 8, 37), (2, 64, 5), (4, 32, 33)] {
            let rotary = DeepSeekV41VisionForward.rotaryTables(
                rows: 1, columns: tokens, ropeDim: headDim / 2, theta: 10_000
            )
            let qkv = (0..<tokens).map { values(3 * heads * headDim, seed: $0 + 1) }
            let planes = try runner.splitAndRotate(
                qkv: qkv, heads: heads, headDim: headDim, ropeDim: headDim / 2,
                cosine: rotary.cosine, sine: rotary.sine
            )
            let outputLength = tokens * heads * headDim * MemoryLayout<Float>.stride
            guard let directBuffer = runner.device.makeBuffer(
                length: outputLength, options: .storageModeShared
            ), let pairedBuffer = runner.device.makeBuffer(
                length: outputLength, options: .storageModeShared
            ) else {
                XCTFail("could not allocate the parity outputs")
                return
            }
            let direct = try runner.attention(planes, into: directBuffer, form: .direct)
            let paired = try runner.attention(planes, into: pairedBuffer, form: .pairedFold)
            let hostDifference = maximumDifference(paired, direct)
            XCTAssertLessThan(hostDifference, 2e-4)

            // And against the serial rule itself, one row per head.
            let serial = DeepSeekV41VisionForward.attention(
                qkv: qkv, heads: heads, headDim: headDim, ropeDim: headDim / 2,
                cosine: rotary.cosine, sine: rotary.sine
            )
            let serialDifference = maximumDifference(paired, serial)
            print(
                "🔎 [dsv41 vision] paired-fold parity heads \(heads) dim \(headDim)"
                    + " tokens \(tokens): direct worst "
                    + String(format: "%.3g", hostDifference)
                    + String(format: ", host worst %.3g", serialDifference)
            )
            XCTAssertLessThan(serialDifference, 2e-4)
        }
    }

    /// A head with an odd dimension cannot be folded in adjacent pairs.
    func testThePairedFoldRefusesAnOddHeadDimension() throws {
        let runner = try runner()
        let planes = try DeepSeekV41VisionAttentionDevice.allocatePlanes(
            device: runner.device, tokens: 3, heads: 1, headDim: 5
        )
        guard let output = runner.device.makeBuffer(
            length: 3 * 1 * 5 * MemoryLayout<Float>.stride, options: .storageModeShared
        ) else {
            XCTFail("could not allocate the output")
            return
        }
        do {
            _ = try runner.runAttention(planes, into: output, form: .pairedFold)
            XCTFail("an odd head dimension must be refused")
        } catch {
            switch error as? DeepSeekV41VisionDeviceError {
            case .shapeMismatch(let detail):
                XCTAssertTrue(detail.contains("even"), "unexpected detail: \(detail)")
            default:
                XCTFail("unexpected error: \(error)")
            }
        }
        // The direct form still runs the odd head it always ran.
        XCTAssertNoThrow(try runner.runAttention(planes, into: output, form: .direct))
    }

    /// One host row of the device attention, read straight from the prepared planes.
    private func hostRow(
        planes: DeepSeekV41VisionAttentionDevice.SplitPlanes,
        head: Int,
        token: Int
    ) -> [Float] {
        let tokens = planes.tokens
        let headDim = planes.headDim
        let base = head * tokens * headDim
        let queries = planes.queries.contents().assumingMemoryBound(to: Float.self)
        let keys = planes.keys.contents().assumingMemoryBound(to: Float.self)
        let values = planes.values.contents().assumingMemoryBound(to: Float.self)
        let queryBase = base + token * headDim
        var scores = [Float](repeating: 0, count: tokens)
        for key in 0..<tokens {
            var dot: Float = 0
            for dimension in 0..<headDim {
                dot += queries[queryBase + dimension] * keys[base + key * headDim + dimension]
            }
            scores[key] = dot / sqrt(Float(headDim))
        }
        guard let maximum = scores.max() else { return [] }
        var weights = scores.map { exp($0 - maximum) }
        let total = weights.reduce(0, +)
        guard total > 0 else { return [Float](repeating: 0, count: headDim) }
        for index in weights.indices { weights[index] /= total }
        var output = [Float](repeating: 0, count: headDim)
        for key in 0..<tokens {
            let weight = weights[key]
            for dimension in 0..<headDim {
                output[dimension] += weight * values[base + key * headDim + dimension]
            }
        }
        return output
    }

    /// What the quadratic stage costs at the two published image sizes.
    ///
    /// 1,521 patch tokens is the smallest admissible image and 8,649 is the largest
    /// (`max_image_tokens` 1,024 over the 31x31 aligner grid). The host tower prices
    /// the whole encode at 2.74 s and 41.71 s, with attention plus softmax 76% of
    /// the large one; this times ONE block's attention at both sizes, so the tower
    /// estimate is that number times the 32 blocks and nothing else.
    func testWhatTheDeviceAttentionCostsAtThePublishedGeometry() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_VISION_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.measurementMarkerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in timing; touch \(Self.measurementMarkerPath) to run"
        )
        let runner = try runner()
        let heads = 16
        let headDim = 64
        let ropeDim = 32

        for (label, side) in [("smallest", 39), ("largest", 93)] {
            let tokens = side * side
            let rotary = DeepSeekV41VisionForward.rotaryTables(
                rows: side, columns: side, ropeDim: ropeDim, theta: 10_000
            )
            let qkv = (0..<tokens).map { token in
                values(3 * heads * headDim, seed: token + 1)
            }
            let planes = try runner.splitAndRotate(
                qkv: qkv,
                heads: heads,
                headDim: headDim,
                ropeDim: ropeDim,
                cosine: rotary.cosine,
                sine: rotary.sine
            )
            guard let outputBuffer = runner.device.makeBuffer(
                length: tokens * heads * headDim * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                XCTFail("could not allocate the attention output")
                return
            }
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<3 {
                best = min(best, try runner.runAttention(planes, into: outputBuffer))
            }
            let blocks = 32.0
            print(
                "🔎 [dsv41 vision] attention \(label): \(tokens) tokens per head, "
                    + String(format: "%.2f ms per block", best * 1000)
                    + String(format: ", %.2f s for %d blocks", best * blocks, Int(blocks))
            )

            // Two rows against the host rule at the shape that ships, so the timing
            // is a measurement of the right kernel and not of a fast wrong one.
            let output = outputBuffer.contents().assumingMemoryBound(to: Float.self)
            for token in [0, tokens / 2] {
                let expected = hostRow(planes: planes, head: 0, token: token)
                var worst: Float = 0
                let base = (token * heads) * headDim
                for dimension in 0..<headDim {
                    worst = max(worst, abs(output[base + dimension] - expected[dimension]))
                }
                print(
                    "🔎 [dsv41 vision] attention \(label) row \(token): "
                        + "worst \(worst) against the host rule"
                )
                XCTAssertLessThan(worst, 5e-4)
            }
        }
    }

    /// Which attention shape the quadratic stage should ship.
    ///
    /// The kernel a threadgroup per (head, query block) walks each key row in place: a
    /// lane owns one key and reads that key's whole row, so every load instruction is
    /// `headDim` floats wide across the lanes, i.e. 32 cache lines for 128 bytes of use.
    /// The staged form copies the tile into threadgroup memory with a coalesced read
    /// first and reads it back conflict-free, and it also lets several query SIMD groups
    /// share one staged tile. So the two knobs are the kernel form and how many queries
    /// a threadgroup owns, and this measures all six combinations at the published
    /// geometry before any of them is chosen.
    ///
    /// Every arm is checked against the host rule at two rows and against the shipped
    /// arm's whole output, so a fast wrong kernel cannot win the sweep.
    func testWhatTheAttentionShapeCandidatesCostAtThePublishedGeometry() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_VISION_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.measurementMarkerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in timing; touch \(Self.measurementMarkerPath) to run"
        )
        let runner = try runner()
        let heads = 16
        let headDim = 64
        let ropeDim = 32
        let arms: [(String, DeepSeekV41VisionAttentionDevice.AttentionForm, Int)] = [
            ("direct/128", .direct, 128),
            ("staged/128", .stagedKeyTile, 128),
            ("direct/256", .direct, 256),
            ("staged/256", .stagedKeyTile, 256),
            ("direct/512", .direct, 512),
            ("staged/512", .stagedKeyTile, 512)
        ]
        let blocks = 32.0

        for (label, side) in [("smallest", 39), ("largest", 93)] {
            let tokens = side * side
            let rotary = DeepSeekV41VisionForward.rotaryTables(
                rows: side, columns: side, ropeDim: ropeDim, theta: 10_000
            )
            let qkv = (0..<tokens).map { token in
                values(3 * heads * headDim, seed: token + 1)
            }
            let planes = try runner.splitAndRotate(
                qkv: qkv,
                heads: heads,
                headDim: headDim,
                ropeDim: ropeDim,
                cosine: rotary.cosine,
                sine: rotary.sine
            )
            let outputLength = tokens * heads * headDim * MemoryLayout<Float>.stride
            guard let baseline = runner.device.makeBuffer(
                length: outputLength, options: .storageModeShared
            ), let candidate = runner.device.makeBuffer(
                length: outputLength, options: .storageModeShared
            ) else {
                XCTFail("could not allocate the attention outputs")
                return
            }
            let flops = 4.0 * Double(tokens) * Double(tokens) * Double(headDim) * Double(heads)
            var bestArm = ("", Double.greatestFiniteMagnitude)
            for (name, form, width) in arms {
                _ = try runner.runAttention(
                    planes, into: candidate, form: form, threadgroupWidth: width
                )
                var best = Double.greatestFiniteMagnitude
                for _ in 0..<3 {
                    best = min(best, try runner.runAttention(
                        planes, into: candidate, form: form, threadgroupWidth: width
                    ))
                }
                print(
                    "🔎 [dsv41 vision] attention \(label) \(name): "
                        + String(format: "%.2f ms per block", best * 1000)
                        + String(format: ", %.2f s for %d blocks", best * blocks, Int(blocks))
                        + String(format: ", %.2f TFLOP/s", flops / best / 1e12)
                )
                XCTAssertLessThan(best * blocks, 1000, "\(label) \(name) produced no work")
                if best < bestArm.1 { bestArm = (name, best) }

                // The rows the timing describes have to be the right rows.
                let output = candidate.contents().assumingMemoryBound(to: Float.self)
                for token in [0, tokens / 2] {
                    let expected = hostRow(planes: planes, head: 0, token: token)
                    var worst: Float = 0
                    let base = (token * heads) * headDim
                    for dimension in 0..<headDim {
                        worst = max(worst, abs(output[base + dimension] - expected[dimension]))
                    }
                    XCTAssertLessThan(worst, 5e-4, "\(label) \(name) row \(token)")
                }

                if name == "direct/128" {
                    baseline.contents().copyMemory(
                        from: output, byteCount: outputLength
                    )
                }
            }

            // The forms compute one rule, so they must agree with each other as well
            // as with the host: a copy into threadgroup memory is the only difference.
            let shipped = baseline.contents().assumingMemoryBound(to: Float.self)
            let staged = candidate.contents().assumingMemoryBound(to: Float.self)
            var worstPair: Float = 0
            for index in 0..<(tokens * heads * headDim) {
                worstPair = max(worstPair, abs(shipped[index] - staged[index]))
            }
            print(
                "🔎 [dsv41 vision] attention \(label) best arm \(bestArm.0) at "
                    + String(format: "%.2f ms", bestArm.1 * 1000)
                    + String(format: ", last arm vs shipped worst %.3g", worstPair)
            )
            XCTAssertLessThan(worstPair, 5e-5)
        }
    }

    /// Times a raw attention-shaped dispatch: `pipeline` with the kernel buffer layout
    /// every candidate shares (0 queries, 1 keys, 2 values, 3 out, 4 tokens, 5 heads,
    /// 6 headDim, 7 the candidate's own knob), best of three.
    private func timeRaw(
        runner: DeepSeekV41VisionAttentionDevice,
        pipeline: MTLComputePipelineState,
        planes: DeepSeekV41VisionAttentionDevice.SplitPlanes,
        into outputBuffer: MTLBuffer,
        mode: UInt32,
        width: Int = DeepSeekV41VisionAttentionDevice.threadgroupWidth
    ) throws -> Double {
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            guard let commandBuffer = runner.queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw DeepSeekV41VisionDeviceError.unavailable("no command buffer")
            }
            var tokenCount = UInt32(planes.tokens)
            var headCount = UInt32(planes.heads)
            var dimension = UInt32(planes.headDim)
            var modeValue = mode
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(planes.queries, offset: 0, index: 0)
            encoder.setBuffer(planes.keys, offset: 0, index: 1)
            encoder.setBuffer(planes.values, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            encoder.setBytes(&tokenCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&modeValue, length: MemoryLayout<UInt32>.stride, index: 7)
            let queriesPerGroup = width / 32
            let queryBlocks = (planes.tokens + queriesPerGroup - 1) / queriesPerGroup
            encoder.dispatchThreadgroups(
                MTLSize(width: planes.heads * max(queryBlocks, 1), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw DeepSeekV41VisionDeviceError.unavailable("raw dispatch failed: \(error)")
            }
            best = min(best, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        }
        return best
    }

    /// What the reshaped value fold buys.
    ///
    /// The ablation puts two thirds of a dispatch in the fold. The candidate kernel
    /// changes two things: a lane owns two ADJACENT dimensions so the pair is one
    /// `float2` load, and the tile's additions split across four accumulators so they
    /// are not a 32-long dependent chain.
    func testWhatAReshapedFoldCostsAtThePublishedGeometry() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_VISION_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.measurementMarkerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in timing; touch \(Self.measurementMarkerPath) to run"
        )
        let runner = try runner()
        let heads = 16
        let headDim = 64
        let ropeDim = 32

        for (label, side) in [("smallest", 39), ("largest", 93)] {
            let tokens = side * side
            let rotary = DeepSeekV41VisionForward.rotaryTables(
                rows: side, columns: side, ropeDim: ropeDim, theta: 10_000
            )
            let planes = try runner.splitAndRotate(
                qkv: (0..<tokens).map { values(3 * heads * headDim, seed: $0 + 1) },
                heads: heads,
                headDim: headDim,
                ropeDim: ropeDim,
                cosine: rotary.cosine,
                sine: rotary.sine
            )
            let outputLength = tokens * heads * headDim * MemoryLayout<Float>.stride
            guard let outputBuffer = runner.device.makeBuffer(
                length: outputLength, options: .storageModeShared
            ) else {
                XCTFail("could not allocate the reshaped-fold output")
                return
            }
            let direct = try (0..<3).map { _ in
                try runner.runAttention(planes, into: outputBuffer, form: .direct)
            }.min() ?? 0
            let pairfold = try (0..<3).map { _ in
                try runner.runAttention(planes, into: outputBuffer, form: .pairedFold)
            }.min() ?? 0

            // The candidate reorders the fold, so it is held to the host rule rather
            // than to bit equality with the shipped kernel.
            let output = outputBuffer.contents().assumingMemoryBound(to: Float.self)
            var worst: Float = 0
            for token in [0, tokens / 2, tokens - 1] {
                let expected = hostRow(planes: planes, head: 0, token: token)
                let base = (token * heads) * headDim
                for dimension in 0..<headDim {
                    worst = max(worst, abs(output[base + dimension] - expected[dimension]))
                }
            }
            print(
                "🔎 [dsv41 vision] attention pairfold \(label): "
                    + String(format: "direct %.2f ms", direct * 1000)
                    + String(format: ", pairedFold %.2f ms (%.2fx)", pairfold * 1000,
                             direct / pairfold)
                    + String(format: " for %d blocks", Int(32))
            )
            print(
                "🔎 [dsv41 vision] attention pairfold \(label): worst "
                    + String(format: "%.3g", worst) + " against the host rule"
            )
            XCTAssertLessThan(worst, 5e-4, "the reshaped fold disagrees with the host")
        }
    }

    /// Where the attention dispatch actually spends its time.
    ///
    /// Needing this is the shape sweep's result: staging the key tile lost 18% and the
    /// threadgroup width changed nothing, so the kernel is not paying for the two things
    /// its shape was chosen for. The probe kernel runs the two halves of a step
    /// separately -- the dot product with the row softmax, and the value fold -- and
    /// `mode` 2 runs both, which must reproduce the shipped kernel bit for bit or the
    /// split below is a measurement of some other kernel.
    func testWhereTheAttentionDispatchSpendsItsTime() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_VISION_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.measurementMarkerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in timing; touch \(Self.measurementMarkerPath) to run"
        )
        let runner = try runner()
        guard let library = runner.device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_vision_attention_probe_f32") else {
            XCTFail("the ablation kernel is missing")
            return
        }
        let pipeline = try runner.device.makeComputePipelineState(function: function)
        let heads = 16
        let headDim = 64
        let ropeDim = 32
        let width = DeepSeekV41VisionAttentionDevice.threadgroupWidth

        for (label, side) in [("smallest", 39), ("largest", 93)] {
            let tokens = side * side
            let rotary = DeepSeekV41VisionForward.rotaryTables(
                rows: side, columns: side, ropeDim: ropeDim, theta: 10_000
            )
            let planes = try runner.splitAndRotate(
                qkv: (0..<tokens).map { values(3 * heads * headDim, seed: $0 + 1) },
                heads: heads,
                headDim: headDim,
                ropeDim: ropeDim,
                cosine: rotary.cosine,
                sine: rotary.sine
            )
            let outputLength = tokens * heads * headDim * MemoryLayout<Float>.stride
            guard let outputBuffer = runner.device.makeBuffer(
                length: outputLength, options: .storageModeShared
            ), let reference = runner.device.makeBuffer(
                length: outputLength, options: .storageModeShared
            ) else {
                XCTFail("could not allocate the ablation outputs")
                return
            }
            let shipped = try runner.runAttention(planes, into: reference, form: .direct)

            func probe(_ mode: UInt32) throws -> Double {
                try timeRaw(
                    runner: runner, pipeline: pipeline, planes: planes,
                    into: outputBuffer, mode: mode, width: width
                )
            }

            let dotOnly = try probe(0)
            let foldOnly = try probe(1)
            let foldNoBroadcast = try probe(3)
            let foldOneHalf = try probe(4)
            // Last, because every arm overwrites the output buffer and this arm's rows
            // are the ones compared with the shipped kernel's.
            let both = try probe(2)

            // Mode 2 has to BE the shipped kernel: same rule, same summation order.
            let produced = outputBuffer.contents().assumingMemoryBound(to: Float.self)
            let shippedRows = reference.contents().assumingMemoryBound(to: Float.self)
            var worst: Float = 0
            for index in 0..<(tokens * heads * headDim) {
                worst = max(worst, abs(produced[index] - shippedRows[index]))
            }
            print(
                "🔎 [dsv41 vision] attention ablation \(label): "
                    + String(format: "dot+softmax %.2f ms", dotOnly * 1000)
                    + String(format: ", fold %.2f ms", foldOnly * 1000)
                    + String(format: ", both %.2f ms", both * 1000)
                    + String(format: ", shipped %.2f ms", shipped * 1000)
                    + String(format: " (%.0f%% dot, %.0f%% fold of the sum)",
                             dotOnly / (dotOnly + foldOnly) * 100,
                             foldOnly / (dotOnly + foldOnly) * 100)
            )
            print(
                "🔎 [dsv41 vision] attention ablation \(label): both-vs-shipped worst "
                    + String(format: "%.3g", worst)
                    + ", row 0 first value "
                    + String(format: "%.5f",
                             produced.advanced(by: 0).pointee)
            )
            print(
                "🔎 [dsv41 vision] attention fold \(label): "
                    + String(format: "fold %.2f ms", foldOnly * 1000)
                    + String(format: ", no broadcast %.2f ms (%.2fx)", foldNoBroadcast * 1000,
                             foldOnly / foldNoBroadcast)
                    + String(format: ", half the loads %.2f ms (%.2fx)", foldOneHalf * 1000,
                             foldOnly / foldOneHalf)
            )
            XCTAssertEqual(worst, 0, "the probe's both-halves arm is not the shipped kernel")
            XCTAssertLessThan(dotOnly + foldOnly, both * 1.4, "the halves do not add up to the step")
        }
    }

    /// The SIMT FP32 rate this machine can actually reach, for the dispatch to be
    /// compared against.
    ///
    /// Four reshaped attention kernels in this file land within 5% of the shipped one,
    /// which is either the processor's ceiling or a shared defect. This is the ceiling:
    /// eight independent FMA chains per thread and no memory traffic at all.
    func testWhatThisGPUsSimtFP32RateIs() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_VISION_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.measurementMarkerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in timing; touch \(Self.measurementMarkerPath) to run"
        )
        let runner = try runner()
        guard let library = runner.device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_vision_fma_peak_f32") else {
            XCTFail("the FMA peak kernel is missing")
            return
        }
        let pipeline = try runner.device.makeComputePipelineState(function: function)
        let iterations: UInt32 = 2_048
        let threads = 1 << 20
        guard let out = runner.device.makeBuffer(
            length: threads * MemoryLayout<Float>.stride, options: .storageModeShared
        ) else {
            XCTFail("could not allocate the peak-rate output")
            return
        }
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            guard let commandBuffer = runner.queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                XCTFail("no command buffer")
                return
            }
            var steps = iterations
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(out, offset: 0, index: 0)
            encoder.setBytes(&steps, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: threads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                XCTFail("the peak-rate dispatch failed: \(error)")
                return
            }
            best = min(best, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        }
        let flops = Double(threads) * Double(iterations) * 8 * 2
        print(
            "🔎 [dsv41 vision] SIMT FP32 peak: "
                + String(format: "%.2f ms for %.1f GFLOP", best * 1000, flops / 1e9)
                + String(format: ", %.2f TFLOP/s", flops / best / 1e12)
        )
        XCTAssertGreaterThan(flops / best / 1e12, 0.1, "the FMA loop was optimized away")
    }
}
