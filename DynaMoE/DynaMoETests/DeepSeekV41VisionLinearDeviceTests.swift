//
//  DeepSeekV41VisionLinearDeviceTests.swift
//  DynaMoETests
//
//  The vision tower's block GEMM on the GPU. The host tower already runs one
//  Accelerate `sgemm` per projection rather than a dot product per output row, so
//  the kernel's competition is a near-peak CPU GEMM: at the published widths the
//  per-token term is ~744 GFLOP/s ("DSV4.1 #37", "DSV4.1 #41"). The parity cases
//  here check the device rule against BOTH host forms -- the GEMM the tower calls
//  and the per-row definition kept as its oracle -- because a transposed operand
//  would produce a plausible-looking wrong answer that only the definition catches.
//
//  The published-geometry timing is opt-in (touch
//  `/tmp/dynamoe_dsv41_linear_measure`) because one of its shapes is a 23 MB weight
//  matrix over 8,649 rows.
//

import XCTest
import Metal
@testable import DynaMoE

final class DeepSeekV41VisionLinearDeviceTests: XCTestCase {

    private static let measurementMarkerPath = "/tmp/dynamoe_dsv41_linear_measure"

    /// Deterministic values in `[-1, 1)` that are not a repeating pattern: a dot
    /// product of a pattern with itself can hide a transposed operand.
    private func randomValues(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let unit = Float((state >> 40) & 0xFFFFFF) / Float(1 << 24)
            return unit * 2 - 1
        }
    }

    private func rows(_ count: Int, width: Int, seed: UInt64) -> [[Float]] {
        let flat = randomValues(count * width, seed: seed)
        return (0..<count).map { Array(flat[($0 * width)..<(($0 + 1) * width)]) }
    }

    private func runner() throws -> DeepSeekV41VisionLinearDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        return try DeepSeekV41VisionLinearDevice(device: device)
    }

    /// Relative difference, because these dot products are hundreds of terms long
    /// and an absolute bar would be loose on small sums and tight on large ones.
    private func worstRelativeDifference(_ lhs: [[Float]], _ rhs: [[Float]]) -> Float {
        var worst: Float = 0
        for row in 0..<min(lhs.count, rhs.count) {
            for column in 0..<min(lhs[row].count, rhs[row].count) {
                let expected = abs(rhs[row][column])
                let scale = max(expected, 1)
                worst = max(worst, abs(lhs[row][column] - rhs[row][column]) / scale)
            }
        }
        return worst
    }

    /// Every dimension ragged: 37 rows against a 64-row tile, 21 columns against a
    /// 64-column tile, 47 of depth against a 16-deep staging chunk.
    func testTheDeviceGEMMMatchesTheHostGEMMOnARaggedShape() throws {
        let runner = try runner()
        let tokenRows = 37
        let outputs = 21
        let depth = 47
        let inputs = rows(tokenRows, width: depth, seed: 11)
        let weight = DeepSeekV41Matrix(
            rows: outputs, columns: depth, values: randomValues(outputs * depth, seed: 29)
        )
        let bias = randomValues(outputs, seed: 41)

        let device = try runner.linear(inputs, weight: weight, bias: bias)
        let host = DeepSeekV41VisionForward.linearBatched(inputs, weight: weight, bias: bias)
        let difference = worstRelativeDifference(device, host)
        print("🔎 [dsv41 vision] linear parity (ragged): worst relative \(difference)")
        XCTAssertEqual(device.count, tokenRows)
        XCTAssertEqual(device.first?.count, outputs)
        XCTAssertLessThan(difference, 1e-5, "the device GEMM left the host GEMM")
    }

    /// The per-row definition is the oracle the host GEMM is pinned to, so the
    /// device has to agree with it too: if both GEMMs shared a transposed operand
    /// this case is the only one that fails.
    func testTheDeviceGEMMMatchesThePerRowDefinition() throws {
        let runner = try runner()
        let tokenRows = 5
        let outputs = 9
        let depth = 13
        let inputs = rows(tokenRows, width: depth, seed: 3)
        let weight = DeepSeekV41Matrix(
            rows: outputs, columns: depth, values: randomValues(outputs * depth, seed: 7)
        )

        let device = try runner.linear(inputs, weight: weight, bias: nil)
        let definition = DeepSeekV41VisionForward.linearRowByRow(
            inputs, weight: weight, bias: nil
        )
        let difference = worstRelativeDifference(device, definition)
        print("🔎 [dsv41 vision] linear parity (definition): worst relative \(difference)")
        XCTAssertLessThan(difference, 1e-5, "the device GEMM left the per-row definition")
    }

    /// A one-row, one-output, one-deep projection: every tile in the kernel is
    /// padding except one value, which is the case a guarded write-out exists for.
    func testTheDeviceGEMMHandlesTheSmallestPossibleProjection() throws {
        let runner = try runner()
        let weight = DeepSeekV41Matrix(rows: 1, columns: 1, values: [3])
        let device = try runner.linear([[5]], weight: weight, bias: [-2])
        XCTAssertEqual(device.count, 1)
        XCTAssertEqual(device.first?.count, 1)
        XCTAssertEqual(device[0][0], 13, accuracy: 1e-5)
    }

    /// The tiles are BOUND buffers, so the kernel declares no threadgroup memory of
    /// its own and the bytes it does use fit the device.
    ///
    /// This is a real defect's regression test, not a formality: with the tiles
    /// declared in the shader the offline compiler allocated 49,152 bytes for a
    /// kernel that named 24,576, the driver ran it anyway, and the validation layer
    /// killed the whole test process against the device's 32,768 limit.
    func testTheGEMMBindsItsThreadgroupMemoryInsteadOfDeclaringIt() throws {
        let runner = try runner()
        guard let function = runner.device.makeDefaultLibrary()?
            .makeFunction(name: "dsv41_vision_gemm_f32") else {
            XCTFail("the default library is missing dsv41_vision_gemm_f32")
            return
        }
        let pipeline = try runner.device.makeComputePipelineState(function: function)
        print(
            "🔎 [dsv41 vision] linear threadgroup memory: declared "
                + "\(pipeline.staticThreadgroupMemoryLength), bound "
                + "\(DeepSeekV41VisionLinearDevice.threadgroupBytes), device limit "
                + "\(runner.device.maxThreadgroupMemoryLength)"
        )
        XCTAssertEqual(pipeline.staticThreadgroupMemoryLength, 0)
        XCTAssertLessThanOrEqual(
            DeepSeekV41VisionLinearDevice.threadgroupBytes,
            runner.device.maxThreadgroupMemoryLength
        )
    }

    /// A short input row is refused rather than read past: the kernel would take
    /// the next row's values for the missing tail and produce a plausible number.
    func testTheDeviceGEMMRefusesARaggedInputRow() throws {
        let runner = try runner()
        let weight = DeepSeekV41Matrix(rows: 2, columns: 4, values: randomValues(8, seed: 13))
        XCTAssertThrowsError(
            try runner.linear([[1, 2, 3], [1, 2, 3, 4]], weight: weight, bias: nil)
        )
    }

    /// One block projection at the published width: 169 aligner cells is the
    /// smallest admissible image's grid, and the 2,816-deep second MLP is the
    /// tower's widest reduction.
    func testTheDeviceGEMMMatchesTheHostAtAPublishedWidth() throws {
        let runner = try runner()
        let tokenRows = 169
        let outputs = 1_024
        let depth = 2_816
        let inputs = rows(tokenRows, width: depth, seed: 17)
        let weight = DeepSeekV41Matrix(
            rows: outputs, columns: depth, values: randomValues(outputs * depth, seed: 19)
        )
        let bias = randomValues(outputs, seed: 23)

        let device = try runner.linear(inputs, weight: weight, bias: bias)
        let host = DeepSeekV41VisionForward.linearBatched(inputs, weight: weight, bias: bias)
        let difference = worstRelativeDifference(device, host)
        print("🔎 [dsv41 vision] linear parity (published width): worst relative \(difference)")
        XCTAssertLessThan(difference, 2e-5, "the device GEMM left the host GEMM")
    }

    /// What the tower's four block projections cost on the device against the host
    /// GEMM they would replace, at the two published token counts.
    ///
    /// The device arm times `runLinear`, i.e. the dispatch alone: a device tower
    /// would keep the token matrix on the GPU between stages, so the readback is not
    /// part of that stage. The host arm times `linearBatched` as the tower calls it,
    /// which includes flattening the token matrix and slicing the product back into
    /// rows. Both arms are given the same bytes and both are checked against the
    /// other before the number is printed.
    func testWhatTheDeviceGEMMCostsAtThePublishedWidths() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_VISION_LINEAR_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.measurementMarkerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in timing; touch \(Self.measurementMarkerPath) to run"
        )
        let runner = try runner()
        let hidden = 1_024
        let shapes: [(String, Int, Int)] = [
            ("qkv", 3_072, hidden),
            ("wo", hidden, hidden),
            ("w1", 5_632, hidden),
            ("w2", hidden, 2_816),
        ]
        let sizes: [(String, Int)] = [("smallest", 39 * 39), ("largest", 93 * 93)]

        for (sizeLabel, tokenRows) in sizes {
            var sizeDevice = 0.0
            var sizeHost = 0.0
            for (name, outputs, depth) in shapes {
                let inputs = rows(tokenRows, width: depth, seed: 5)
                let weight = DeepSeekV41Matrix(
                    rows: outputs, columns: depth,
                    values: randomValues(outputs * depth, seed: 31)
                )
                let bias = randomValues(outputs, seed: 37)
                guard let inputBuffer = try? runner.makeBuffer(inputs.flatMap { $0 }),
                      let weightBuffer = try? runner.makeBuffer(weight.values),
                      let biasBuffer = try? runner.makeBuffer(bias),
                      let outputBuffer = runner.device.makeBuffer(
                          length: tokenRows * outputs * MemoryLayout<Float>.stride,
                          options: .storageModeShared
                      ) else {
                    XCTFail("could not allocate the \(name) operands")
                    return
                }

                var deviceBest = Double.greatestFiniteMagnitude
                for _ in 0..<3 {
                    deviceBest = min(
                        deviceBest,
                        try runner.runLinear(
                            inputs: inputBuffer, weights: weightBuffer, bias: biasBuffer,
                            output: outputBuffer, rows: tokenRows, outputs: outputs, depth: depth
                        )
                    )
                }
                var hostBest = Double.greatestFiniteMagnitude
                for _ in 0..<3 {
                    let start = Date()
                    let product = DeepSeekV41VisionForward.linearBatched(
                        inputs, weight: weight, bias: bias
                    )
                    hostBest = min(hostBest, Date().timeIntervalSince(start))
                    // One row of the device result against the host's, so the timing
                    // is a measurement of the right kernel and not of a fast wrong one.
                    if name == "wo", sizeLabel == "smallest" || sizeLabel == "largest" {
                        let row = tokenRows / 2
                        let deviceRows = runner.readRows(
                            outputBuffer, rows: tokenRows, columns: outputs
                        )
                        let difference = worstRelativeDifference([deviceRows[row]], [product[row]])
                        print(
                            "🔎 [dsv41 vision] linear \(sizeLabel) \(name) row \(row): "
                                + "worst relative \(difference)"
                        )
                        XCTAssertLessThan(difference, 2e-5)
                    }
                }
                let flops = 2.0 * Double(tokenRows) * Double(outputs) * Double(depth)
                print(
                    "🔎 [dsv41 vision] linear \(sizeLabel) \(name): "
                        + String(format: "device %.2f ms", deviceBest * 1000)
                        + String(format: ", host %.2f ms", hostBest * 1000)
                        + String(format: ", %.2fx", hostBest / deviceBest)
                        + String(format: ", device %.2f GFLOP/s", flops / deviceBest / 1e9)
                )
                sizeDevice += deviceBest
                sizeHost += hostBest
            }
            print(
                "🔎 [dsv41 vision] linear \(sizeLabel) 32 blocks: "
                    + String(format: "device %.2f s", sizeDevice * 32)
                    + String(format: ", host %.2f s", sizeHost * 32)
                    + String(format: ", %.2fx", sizeHost / sizeDevice)
            )
        }
    }
}
