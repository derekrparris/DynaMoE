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
}
