//
//  DeepSeekV41PrefillMeasurementTests.swift
//  DynaMoETests
//
//  What the block prefill is worth, measured rather than assumed.
//
//  `prefillChunk` runs a prompt as blocks of positions: the dense stages dispatch
//  once over the whole block, and only the cache write, the attention and the
//  routed mixture stay per position. That is a claim about cost, and this repo's
//  own journal records optimizations that were shipped on the strength of the
//  claim and reverted once somebody measured them. So the claim gets a number
//  before anything is built on top of it.
//
//  The measurement runs on `DeepSeekV41SyntheticCheckpoint` because the local
//  checkpoint shards are header-only stubs: there is no real weight to prefill,
//  and the published widths (7,168 hidden, 256 experts) would not run on a
//  laptop's memory budget at test speed anyway. What the synthetic model can say
//  is therefore bounded, and worth being exact about:
//
//  - It measures DISPATCH AND WAIT amortization, which is what batching removes.
//    Per-position arithmetic is identical in both paths, so a model this small
//    over-states the win relative to a model whose arithmetic dominates.
//  - The walk reads the logits back at every position (`forwardToken` finishes a
//    block of one each time); `prefillChunk` reads them only once, at the end.
//    That is part of what the block saves and it is included rather than hidden.
//  - Block width is capped by the ring and by a ceiling of eight positions
//    (`DeepSeekV41DevicePlan.chunkSlots`), so the widths below are the widths the
//    checkpoint could actually use.
//
//  The case is opt-in: it prints a table instead of asserting a threshold, so it
//  must not slow or flake the default suite. `xcodebuild` does not forward the
//  invoking shell's environment into an app-hosted test, so the switch is a marker
//  file as well as an environment variable (`DeepSeekV41PrefillMeasurementTests
//  .markerPath`). Run it with
//
//      touch /tmp/dynamoe_dsv41_prefill_measure
//      xcodebuild test-without-building ... \
//          -only-testing:DynaMoETests/DeepSeekV41PrefillMeasurementTests
//
//  and record the numbers in `Benchmarks/PERF_FINDINGS.md`, naming the machine.
//

import XCTest
import Metal
@testable import DynaMoE

final class DeepSeekV41PrefillMeasurementTests: XCTestCase {

    /// Positions per timed prompt. Longer than one block at every width below, so
    /// the block path really does run several blocks, and several ring wraps.
    private let positions = 32
    private let warmups = 1
    private let repetitions = 3

    /// Touching this file opts the measurement in, for the runs where `xcodebuild`
    /// has not forwarded `DYNAMOE_DSV41_PREFILL_MEASURE`.
    static let markerPath = "/tmp/dynamoe_dsv41_prefill_measure"

    override func setUpWithError() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_PREFILL_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.markerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in measurement; touch \(Self.markerPath) (or set "
                + "DYNAMOE_DSV41_PREFILL_MEASURE=1) to run"
        )
    }

    // MARK: Harness

    private func makeRuntime(
        _ checkpoint: DeepSeekV41SyntheticCheckpoint
    ) throws -> DeepSeekV41Runtime {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        guard let buffer = checkpoint.shardBuffer(device: device) else {
            XCTFail("the synthetic shard could not be mapped")
            throw XCTSkip("no shard buffer")
        }
        guard let runtime = DeepSeekV41Runtime(
            snapshotDir: nil,
            summary: checkpoint.summary(),
            config: checkpoint.config,
            shardBuffers: [0: buffer],
            contextLength: checkpoint.shape.contextLength,
            cachePrecision: .fp16
        ) else {
            XCTFail(
                "the synthetic runtime did not build: "
                    + (DeepSeekV41Runtime.lastInitializationError ?? "no reason recorded")
            )
            throw XCTSkip("no runtime")
        }
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        return runtime
    }

    /// Milliseconds for one run, measured around the whole call.
    ///
    /// Both paths wait inside themselves (the routing readback, the expert slots),
    /// so wall time covers the GPU work rather than only the encoding.
    private func milliseconds(_ body: () throws -> Void) rethrows -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        try body()
        let end = DispatchTime.now().uptimeNanoseconds
        return Double(end - start) / 1_000_000
    }

    /// The fastest of `repetitions` runs, after `warmups`. A minimum rather than a
    /// mean: the floor is the number a scheduler and a thermal state cannot inflate.
    private func best(
        _ body: () throws -> Void
    ) rethrows -> Double {
        for _ in 0..<warmups { try body() }
        var fastest = Double.infinity
        for _ in 0..<repetitions {
            fastest = min(fastest, try milliseconds(body))
        }
        return fastest
    }

    private func worstDelta(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var worst: Float = 0
        for index in 0..<lhs.count {
            worst = max(worst, abs(lhs[index] - rhs[index]))
        }
        return worst
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    // MARK: Measurement

    /// The walk against the block, at every block width the ring allows.
    ///
    /// The prompt is the same tokens at every width; only the ring changes, which
    /// is what sets the block width. A row's walk and block therefore run the same
    /// model, and the ratio inside a row is the honest comparison. Widths are
    /// compared with each other only as a trend, because a wider ring is also a
    /// wider attention.
    func testTheBlockPrefillAgainstTheWalk() throws {
        print("[dsv41 prefill] positions=\(positions) reps=\(repetitions) (best of)")
        print("[dsv41 prefill] window  slots   walk ms   chunk ms   speedup   walk ms/pos   chunk ms/pos")

        for window in [2, 4, 8, 16] {
            var shape = DeepSeekV41SyntheticShape()
            shape.window = window
            shape.contextLength = max(48, window * 4)
            let checkpoint = DeepSeekV41SyntheticCheckpoint(shape: shape)
            let runtime = try makeRuntime(checkpoint)
            let device = runtime.forward()
            let slots = DeepSeekV41DevicePlan.chunkSlots(
                config: checkpoint.config,
                contextLength: checkpoint.shape.contextLength
            )
            let tokens = (0..<positions).map { $0 % checkpoint.config.vocabSize }

            let walked = try best {
                device.reset()
                for (position, token) in tokens.enumerated() {
                    _ = try device.forwardToken(token, position: position)
                }
            }
            let walkLogits = floats(
                runtime.workspace.logits,
                count: checkpoint.config.vocabSize
            )

            let chunked = try best {
                device.reset()
                _ = try device.prefillChunk(tokens)
            }
            device.reset()
            let chunk = try device.prefillChunk(tokens).last

            // The two paths must be timing the same answer, or the comparison is
            // between two different computations.
            let scale = walkLogits.map(abs).max() ?? 1
            XCTAssertLessThan(
                worstDelta(chunk, walkLogits) / max(scale, 1e-6),
                2e-3,
                "window \(window): the block does not return the walk's logits"
            )

            let perWalk = walked / Double(positions)
            let perChunk = chunked / Double(positions)
            print(
                String(
                    format: "[dsv41 prefill] %6d  %5d  %8.2f  %8.2f  %7.2fx  %12.3f  %12.3f",
                    window,
                    slots,
                    walked,
                    chunked,
                    walked / chunked,
                    perWalk,
                    perChunk
                )
            )
        }
    }

    /// What one layer of a block costs, so the per-position floor can be attributed.
    ///
    /// If a block's per-position cost scales with the layer count, the cost is work
    /// the block still does per position (the cache write, the attention, the
    /// mixture's own command buffer and wait). If it did not scale, it would be the
    /// block's fixed cost and the batching would already have removed it.
    func testThePerLayerCostOfABlock() throws {
        var perLayer: [Double] = []
        for layers in [1, 2] {
            var shape = DeepSeekV41SyntheticShape()
            shape.layers = layers
            let checkpoint = DeepSeekV41SyntheticCheckpoint(shape: shape)
            let runtime = try makeRuntime(checkpoint)
            let device = runtime.forward()
            let tokens = (0..<positions).map { $0 % checkpoint.config.vocabSize }
            let block = try best {
                device.reset()
                _ = try device.prefillChunk(tokens)
            }
            perLayer.append(block / Double(positions))
            print(
                String(
                    format: "[dsv41 prefill] layers=%d block of %d: %.3f ms/position",
                    layers,
                    positions,
                    block / Double(positions)
                )
            )
        }
        print(
            String(
                format: "[dsv41 prefill] one layer costs %.3f ms/position of a block",
                perLayer[1] - perLayer[0]
            )
        )
    }

    /// The same block width, run once as a prompt and once as a decode step, to
    /// separate the two things the block buys: fewer dispatches, and no per-position
    /// logits readback.
    ///
    /// `prefillChunk` of one token is still a block: it only reads the logits once.
    /// A walk of the same token is a decode step. Comparing them with an N-token
    /// block's per-position cost says how much of the win is the block and how much
    /// is the readback the block skips.
    func testThePerPositionCostOfABlockAndADecodeStep() throws {
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let runtime = try makeRuntime(checkpoint)
        let device = runtime.forward()
        let slots = DeepSeekV41DevicePlan.chunkSlots(
            config: checkpoint.config,
            contextLength: checkpoint.shape.contextLength
        )
        XCTAssertGreaterThan(slots, 1, "a block of one measures nothing")

        let tokens = (0..<positions).map { $0 % checkpoint.config.vocabSize }

        let blockOfOne = try best {
            device.reset()
            _ = try device.prefillChunk([7])
        }
        let block = try best {
            device.reset()
            _ = try device.prefillChunk(tokens)
        }

        print(
            String(
                format: "[dsv41 prefill] block of one %.3f ms, block of %d %.2f ms, "
                    + "%.3f ms/position (slots=%d)",
                blockOfOne,
                positions,
                block,
                block / Double(positions),
                slots
            )
        )
        XCTAssertLessThan(block, blockOfOne * Double(positions))
    }
}
