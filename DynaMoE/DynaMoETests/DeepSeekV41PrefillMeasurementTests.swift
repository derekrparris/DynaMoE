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

    /// Where a block's remaining per-position time goes.
    ///
    /// After the mixture was batched (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #2") a
    /// block still costs ~0.14 ms/position/layer, and that number has two possible
    /// authors that want opposite fixes: host time spent ENCODING dispatches, which
    /// fewer-and-wider would reduce, or device time spent in the dense half's wait,
    /// which no amount of encoding would. `recordsBlockTiming` splits the two (and
    /// the mixture's own encode), so the next change is chosen by this number rather
    /// than by counting dispatches.
    func testWhereABlocksRemainingTimeGoes() throws {
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let runtime = try makeRuntime(checkpoint)
        let device = runtime.forward()
        let tokens = (0..<positions).map { $0 % checkpoint.config.vocabSize }
        let blockWidth = DeepSeekV41DevicePlan.chunkSlots(
            config: checkpoint.config,
            contextLength: checkpoint.shape.contextLength
        )

        // The same prompt down the walk, so the buckets are known to be timing the
        // same computation the other cases compare against.
        let walked = try makeRuntime(checkpoint).forward()
        var walkLogits: [Float] = []
        for (position, token) in tokens.enumerated() {
            walkLogits = try walked.forwardToken(token, position: position)
        }

        // The fastest run by accounted layer time, so the buckets and the wall time
        // come from the same run rather than from different ones.
        var fastest: (wall: Double, timing: DeepSeekV41BlockTiming)?
        for _ in 0..<(warmups + repetitions) {
            device.reset()
            device.resetBlockTiming()
            device.recordsBlockTiming = true
            let wall = try milliseconds {
                _ = try device.prefillChunk(tokens)
            }
            device.recordsBlockTiming = false
            let timing = device.blockTiming
            if fastest == nil || accounted(timing) < accounted(fastest!.timing) {
                fastest = (wall, timing)
            }
        }
        let measured = try XCTUnwrap(fastest)
        XCTAssertGreaterThan(measured.timing.layers, 0, "the switch recorded nothing")

        device.reset()
        let chunk = try device.prefillChunk(tokens).last
        let scale = walkLogits.map(abs).max() ?? 1
        XCTAssertLessThan(
            worstDelta(chunk, walkLogits) / max(scale, 1e-6),
            2e-3,
            "the bucket stopwatch is timing a different computation"
        )

        let per = measured.timing.millisecondsPerPositionLayer()
        let layers = Double(accounted(measured.timing)) / 1_000_000
        // Every bucket is a sub-interval of the same run, so their sum cannot exceed
        // it. What is left over is the embedding block and the output head, which no
        // bucket covers on purpose.
        XCTAssertLessThanOrEqual(layers, measured.wall, "the layers cost more than the run")
        let encoded = per.denseEncode + per.mixtureEncode
        let encodeShare = encoded + per.wait > 0 ? encoded / (encoded + per.wait) : 0
        let deviceTime = per.denseDevice + per.mixtureDevice
        let deviceShare = deviceTime > 0 ? per.denseDevice / deviceTime : 0

        print(
            String(
                format: "[dsv41 prefill] block of %d (tokens %d), slots=%d, layers=%d "
                    + "recorded, positions=%d, %.2f ms/position",
                positions,
                tokens.count,
                blockWidth,
                measured.timing.layers,
                measured.timing.positions,
                measured.wall / Double(positions)
            )
        )
        print(
            String(
                format: "[dsv41 prefill] host per position/layer: dense encode %.4f  "
                    + "front wait %.4f  mixture encode %.4f ms",
                per.denseEncode,
                per.wait,
                per.mixtureEncode
            )
        )
        print(
            String(
                format: "[dsv41 prefill] DEVICE per position/layer: dense %.4f  mixture %.4f ms "
                    + "(dense holds %.0f%% of the device time)",
                per.denseDevice,
                per.mixtureDevice,
                deviceShare * 100
            )
        )
        print(
            String(
                format: "[dsv41 prefill] encode share of the wait+encode %.0f%%; layers "
                    + "accounted %.2f of a %.2f ms run",
                encodeShare * 100,
                layers,
                measured.wall
            )
        )
    }

    private func accounted(_ timing: DeepSeekV41BlockTiming) -> UInt64 {
        timing.denseEncodeNanoseconds + timing.frontWaitNanoseconds
            + timing.mixtureEncodeNanoseconds
    }

    /// Which stage of the dense half holds its device time.
    ///
    /// `testWhereABlocksRemainingTimeGoes` says the dense half is ~85% of a block's
    /// device time; this says where inside it. `recordsStageTiming` puts each stage
    /// in its own command buffer so every stage reports a `gpuStartTime`/
    /// `gpuEndTime` window, which costs a little per-stage overhead and is therefore
    /// read as a ranking rather than as absolute milliseconds.
    func testWhichDenseStageHoldsTheDeviceTime() throws {
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let runtime = try makeRuntime(checkpoint)
        let device = runtime.forward()
        let tokens = (0..<positions).map { $0 % checkpoint.config.vocabSize }

        var fastest: (wall: Double, stages: [String: UInt64], layers: Int, device: Double)?
        for _ in 0..<(warmups + repetitions) {
            device.reset()
            device.resetStageTiming()
            device.resetBlockTiming()
            device.recordsStageTiming = true
            device.recordsBlockTiming = true
            let wall = try milliseconds {
                _ = try device.prefillChunk(tokens)
            }
            device.recordsStageTiming = false
            device.recordsBlockTiming = false
            let stages = device.stageTiming
            XCTAssertFalse(stages.isEmpty, "no stage reported a device window")
            if fastest == nil || wall < fastest!.wall {
                fastest = (
                    wall,
                    stages,
                    device.blockTiming.layers,
                    Double(device.blockTiming.denseDeviceNanoseconds) / 1_000_000
                )
            }
        }
        let measured = try XCTUnwrap(fastest)

        // The layout change must not change the answer: same prompt, same logits.
        device.reset()
        let chunk = try device.prefillChunk(tokens).last
        let walked = try makeRuntime(checkpoint).forward()
        var walkLogits: [Float] = []
        for (position, token) in tokens.enumerated() {
            walkLogits = try walked.forwardToken(token, position: position)
        }
        let scale = walkLogits.map(abs).max() ?? 1
        XCTAssertLessThan(
            worstDelta(chunk, walkLogits) / max(scale, 1e-6),
            2e-3,
            "the per-stage layout changed the answer"
        )

        let total = measured.stages.values.reduce(0, +)
        print(
            String(
                format: "[dsv41 prefill] dense device time by stage, %.2f ms/position over "
                    + "%d layers (whole-block wall %.2f, dense device in one buffer %.2f "
                    + "ms/position)",
                Double(total) / Double(positions) / 1_000_000,
                measured.layers,
                measured.wall / Double(positions),
                measured.device / Double(positions)
            )
        )
        for (stage, nanoseconds) in measured.stages.sorted(by: { $0.value > $1.value }) {
            print(
                String(
                    format: "[dsv41 prefill] %-24@ %8.3f ms/pos  %5.1f%%",
                    stage as NSString,
                    Double(nanoseconds) / Double(positions) / 1_000_000,
                    Double(nanoseconds) / Double(max(1, total)) * 100
                )
            )
        }
    }

    /// What ONE attention dispatch costs at the widths the checkpoint publishes.
    ///
    /// Every figure above runs on the synthetic model, and that model's attention
    /// is two heads over at most a handful of rows: a shape whose cost is the
    /// dispatch, not the arithmetic. The shipped shape is the opposite. A compressed
    /// layer reads `window + index_topk` rows — 128 + 512 — and each head walks them
    /// three times over a 512-wide latent record, so a head is a serial chain of
    /// about 1.3 million multiply-adds and the whole kernel is as wide as the head
    /// COUNT. This case times the kernel at that shape, on real Metal, so a change
    /// to it can be read against the shape that ships rather than against the shape
    /// a test can afford to build a whole checkpoint around.
    ///
    /// Two honesty notes. The dispatch is issued from one command buffer, so this is
    /// device time and not submission cost: in a block it sits in a buffer with the
    /// rest of the layer, and the per-dispatch latency there is a separate term the
    /// block figures above already carry. And the row counts are the row BUDGET, not
    /// a measured row count for a prompt: a window-only layer reads 128, a compressed
    /// layer reads up to 640 once its cache has filled, and both are below.
    func testWhatOneAttentionDispatchCostsAtThePublishedWidths() throws {
        guard let metal = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        guard let library = metal.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_attention_csa2_f16"),
              let pipeline = try? metal.makeComputePipelineState(function: function),
              let queue = metal.makeCommandQueue() else {
            XCTFail("the attention kernel is not in the default library")
            return
        }

        // The published text_config: 128 heads, head_dim 512, v_head_dim 512, and a
        // cache record 512 wide (`kv_lora_rank` is absent, so the record is the head).
        let heads = 128
        let qkHeadDim = 512
        let valueDim = 512
        let width = 512
        let shape = DeepSeekV41AttentionShape(
            numHeads: heads,
            headsPerGroup: 16,
            qkHeadDim: qkHeadDim,
            vHeadDim: valueDim,
            latentWidth: width,
            qLoraRank: 1536,
            oLoraRank: 1024,
            isCompressed: true,
            compressRatio: 2,
            ropeTheta: 160000,
            rotaryDim: 64
        )

        func sample(_ count: Int, seed: UInt64) -> [Float] {
            var state = seed
            return (0..<count).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 30)
            }
        }

        let query = sample(heads * qkHeadDim, seed: 1)
        let sink = sample(heads, seed: 3).map { $0 * 4 }

        func buffer(_ values: [Float]) -> MTLBuffer? {
            values.withUnsafeBytes { bytes in
                metal.makeBuffer(
                    bytes: bytes.baseAddress!,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
        }

        guard let queryBuffer = buffer(query),
              let sinkBuffer = buffer(sink),
              let outputBuffer = metal.makeBuffer(
                  length: heads * valueDim * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            XCTFail("the attention buffers did not allocate")
            return
        }

        print("[dsv41 attention] heads=\(heads) qkHeadDim=\(qkHeadDim) width=\(width) sink=on")

        for rows in [128, 136, 640] {
            // The records arrive at the cache's width; the host rule is given them
            // widened, which is lossless.
            let keys = sample(rows * width, seed: UInt64(rows) &+ 7).map { Float16($0) }
            let widenedKeys = keys.map { Float($0) }
            guard let keyBuffer = keys.withUnsafeBytes({ bytes in
                metal.makeBuffer(
                    bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared
                )
            }) else {
                XCTFail("the \(rows)-row key block did not allocate")
                return
            }

            func encode(_ encoder: MTLComputeCommandEncoder) {
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(queryBuffer, offset: 0, index: 0)
                encoder.setBuffer(keyBuffer, offset: 0, index: 1)
                encoder.setBuffer(sinkBuffer, offset: 0, index: 2)
                encoder.setBuffer(outputBuffer, offset: 0, index: 3)
                var rowCount = UInt32(rows)
                var headCount = UInt32(heads)
                var headWidth = UInt32(qkHeadDim)
                var headValue = UInt32(valueDim)
                var recordWidth = UInt32(width)
                encoder.setBytes(&rowCount, length: 4, index: 4)
                encoder.setBytes(&headCount, length: 4, index: 5)
                encoder.setBytes(&headWidth, length: 4, index: 6)
                encoder.setBytes(&headValue, length: 4, index: 7)
                encoder.setBytes(&recordWidth, length: 4, index: 8)
            }

            // The device window of `repetitions` dispatches, so submission cost is
            // divided out and what is left is the kernel.
            let repetitions = rows >= 640 ? 5 : 20
            var perDispatch = Double.infinity
            for attempt in 0..<(warmups + repetitions) {
                guard attempt >= warmups else {
                    guard let command = queue.makeCommandBuffer(),
                          let encoder = command.makeComputeCommandEncoder() else { return }
                    encode(encoder)
                    encoder.setThreadgroupMemoryLength(
                        (rows + 32) * MemoryLayout<Float>.stride, index: 0
                    )
                    encoder.dispatchThreadgroups(
                        MTLSize(width: heads, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                    )
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    continue
                }
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return }
                encode(encoder)
                encoder.setThreadgroupMemoryLength((rows + 32) * MemoryLayout<Float>.stride, index: 0)
                for _ in 0..<repetitions {
                    encoder.dispatchThreadgroups(
                        MTLSize(width: heads, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                    )
                }
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                let window = command.gpuEndTime - command.gpuStartTime
                perDispatch = min(perDispatch, window / Double(repetitions) * 1000)
            }

            // The timing above is only worth reading if it is the same computation
            // the host performs. Two heads are enough: the reduction, the sink and
            // the output accumulation are the same code for every head.
            let output = floats(outputBuffer, count: heads * valueDim)
            let reference = DeepSeekV41AttentionReference.attend(
                query: query,
                keys: (0..<rows).map {
                    Array(widenedKeys[($0 * width)..<(($0 + 1) * width)])
                },
                shape: shape,
                sink: sink
            )
            for head in [0, 37] {
                let range = (head * valueDim)..<((head + 1) * valueDim)
                let scale = reference[range].map(abs).max() ?? 1
                XCTAssertLessThan(
                    worstDelta(Array(output[range]), Array(reference[range])) / max(scale, 1e-6),
                    1e-4,
                    "\(rows) rows: head \(head) does not match the host rule"
                )
            }

            print(
                String(
                    format: "[dsv41 attention] %4d rows  %8.3f ms/dispatch  %8.3f us/head",
                    rows,
                    perDispatch,
                    perDispatch * 1000 / Double(heads)
                )
            )
        }
    }

    /// An FP16 gathered block against an FP32 one, at the published geometry.
    ///
    /// `#16` left one fact behind it: on a compressed layer `writeLatent` is 45% of
    /// the dense half and essentially all of that IS the attention, whose cost is
    /// the bytes it reads rather than the arithmetic it does. Every head reads the
    /// whole gathered block twice — once for the scores, once for the values — and
    /// the block is the SAME rows for all 128 of them, so the stage moves
    /// `2 * heads * rows * recordBytes` per position. The record is 512 wide,
    /// arrives from an FP16 cache, and was widened to FP32 by the gather, so half
    /// of that traffic is padding that no kernel asked for.
    ///
    /// The two arms see the SAME key values: the block is sampled as `half` and the
    /// FP32 arm reads those values widened, which is exactly what the gather does
    /// to a cache row today. So this is an A/B of storage width alone, and the arms
    /// must agree to the bit — which is also the assertion that makes wiring it
    /// safe.
    func testWhatAnFP16KeyBlockSavesOverAnFP32One() throws {
        guard let metal = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        guard let library = metal.makeDefaultLibrary(),
              let fp32Function = library.makeFunction(name: "dsv41_attention_csa2_f32"),
              let fp16Function = library.makeFunction(name: "dsv41_attention_csa2_f16"),
              let fp32Pipeline = try? metal.makeComputePipelineState(function: fp32Function),
              let fp16Pipeline = try? metal.makeComputePipelineState(function: fp16Function),
              let queue = metal.makeCommandQueue() else {
            XCTFail("both attention kernels must be in the default library")
            return
        }

        let heads = 128
        let qkHeadDim = 512
        let valueDim = 512
        let width = 512
        let shape = DeepSeekV41AttentionShape(
            numHeads: heads,
            headsPerGroup: 16,
            qkHeadDim: qkHeadDim,
            vHeadDim: valueDim,
            latentWidth: width,
            qLoraRank: 1536,
            oLoraRank: 1024,
            isCompressed: true,
            compressRatio: 2,
            ropeTheta: 160000,
            rotaryDim: 64
        )

        func sample(_ count: Int, seed: UInt64) -> [Float] {
            var state = seed
            return (0..<count).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 30)
            }
        }

        let query = sample(heads * qkHeadDim, seed: 1)
        let sink = sample(heads, seed: 3).map { $0 * 4 }

        func floatBuffer(_ values: [Float]) -> MTLBuffer? {
            values.withUnsafeBytes { bytes in
                metal.makeBuffer(
                    bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared
                )
            }
        }
        func halfBuffer(_ values: [Float16]) -> MTLBuffer? {
            values.withUnsafeBytes { bytes in
                metal.makeBuffer(
                    bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared
                )
            }
        }

        guard let queryBuffer = floatBuffer(query),
              let sinkBuffer = floatBuffer(sink),
              let fp32Output = metal.makeBuffer(
                  length: heads * valueDim * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let fp16Output = metal.makeBuffer(
                  length: heads * valueDim * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            XCTFail("the attention buffers did not allocate")
            return
        }

        // Ramp the clocks before the sweep, so the first row count is not the
        // slowest one by construction.
        do {
            let warmKeys = halfBuffer([Float16](repeating: 0.5, count: 1152 * width))
            let warmOutput = metal.makeBuffer(
                length: heads * valueDim * MemoryLayout<Float>.stride, options: .storageModeShared
            )
            if let warmKeys, let warmOutput, let command = queue.makeCommandBuffer(),
               let encoder = command.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(fp16Pipeline)
                encoder.setBuffer(queryBuffer, offset: 0, index: 0)
                encoder.setBuffer(warmKeys, offset: 0, index: 1)
                encoder.setBuffer(sinkBuffer, offset: 0, index: 2)
                encoder.setBuffer(warmOutput, offset: 0, index: 3)
                var warmRows = UInt32(1152)
                var warmHeads = UInt32(heads)
                var warmWidth = UInt32(qkHeadDim)
                var warmValue = UInt32(valueDim)
                var warmRecord = UInt32(width)
                encoder.setBytes(&warmRows, length: 4, index: 4)
                encoder.setBytes(&warmHeads, length: 4, index: 5)
                encoder.setBytes(&warmWidth, length: 4, index: 6)
                encoder.setBytes(&warmValue, length: 4, index: 7)
                encoder.setBytes(&warmRecord, length: 4, index: 8)
                encoder.setThreadgroupMemoryLength(
                    (1152 + 32) * MemoryLayout<Float>.stride, index: 0
                )
                for _ in 0..<40 {
                    encoder.dispatchThreadgroups(
                        MTLSize(width: heads, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                    )
                }
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
            }
        }

        print("[dsv41 key width] heads=\(heads) record=\(width) sink=on")

        // 136 is one window plus the first compressed row, 640 is the published
        // compressed layer's `window + index_topk`, and 1,152 is what a block's
        // late positions read once the ring is full and the pool has filled.
        for rows in [136, 640, 1152] {
            let halfKeys: [Float16] = sample(rows * width, seed: UInt64(rows) &+ 11)
                .map { Float16($0) }
            let floatKeys = halfKeys.map { Float($0) }
            guard let fp32Keys = floatBuffer(floatKeys),
                  let fp16Keys = halfBuffer(halfKeys) else {
                XCTFail("the \(rows)-row key blocks did not allocate")
                return
            }

            func encode(
                _ encoder: MTLComputeCommandEncoder,
                pipeline: MTLComputePipelineState,
                keys: MTLBuffer,
                output: MTLBuffer
            ) {
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(queryBuffer, offset: 0, index: 0)
                encoder.setBuffer(keys, offset: 0, index: 1)
                encoder.setBuffer(sinkBuffer, offset: 0, index: 2)
                encoder.setBuffer(output, offset: 0, index: 3)
                var rowCount = UInt32(rows)
                var headCount = UInt32(heads)
                var headWidth = UInt32(qkHeadDim)
                var headValue = UInt32(valueDim)
                var recordWidth = UInt32(width)
                encoder.setBytes(&rowCount, length: 4, index: 4)
                encoder.setBytes(&headCount, length: 4, index: 5)
                encoder.setBytes(&headWidth, length: 4, index: 6)
                encoder.setBytes(&headValue, length: 4, index: 7)
                encoder.setBytes(&recordWidth, length: 4, index: 8)
            }

            // One dispatch's GPU window, `repeats` of them per command buffer so
            // submission cost divides out.
            func window(
                pipeline: MTLComputePipelineState,
                keys: MTLBuffer,
                output: MTLBuffer,
                repeats: Int
            ) -> Double? {
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return nil }
                encode(encoder, pipeline: pipeline, keys: keys, output: output)
                encoder.setThreadgroupMemoryLength(
                    (rows + 32) * MemoryLayout<Float>.stride, index: 0
                )
                for _ in 0..<repeats {
                    encoder.dispatchThreadgroups(
                        MTLSize(width: heads, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                    )
                }
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                let elapsed = command.gpuEndTime - command.gpuStartTime
                return elapsed / Double(repeats) * 1000
            }

            // The arms are INTERLEAVED rather than run one after the other: GPU
            // clocks ramp across a sweep, and a first arm timed before the ramp
            // reads slower than the same arm timed after it. Both arms get the same
            // number of attempts, in the same clock state.
            let repeats = rows >= 640 ? 8 : 24
            let attempts = 6
            _ = window(pipeline: fp32Pipeline, keys: fp32Keys, output: fp32Output, repeats: 1)
            _ = window(pipeline: fp16Pipeline, keys: fp16Keys, output: fp16Output, repeats: 1)
            var fp32Milliseconds = Double.infinity
            var fp16Milliseconds = Double.infinity
            for _ in 0..<attempts {
                if let value = window(
                    pipeline: fp32Pipeline, keys: fp32Keys, output: fp32Output, repeats: repeats
                ) {
                    fp32Milliseconds = min(fp32Milliseconds, value)
                }
                if let value = window(
                    pipeline: fp16Pipeline, keys: fp16Keys, output: fp16Output, repeats: repeats
                ) {
                    fp16Milliseconds = min(fp16Milliseconds, value)
                }
            }
            guard fp32Milliseconds.isFinite, fp16Milliseconds.isFinite else {
                XCTFail("the \(rows)-row A/B did not run")
                return
            }

            // Both arms are the same computation over the same values, so compare
            // them to each other before comparing either to the host rule. The bar
            // is a float-rounding one rather than exactness because the two kernels
            // may sum their accumulators in different orders; when they sum them the
            // same way it holds to the bit.
            let fp32Values = floats(fp32Output, count: heads * valueDim)
            let fp16Values = floats(fp16Output, count: heads * valueDim)
            let outputScale = fp32Values.map(abs).max() ?? 1
            XCTAssertLessThan(
                worstDelta(fp32Values, fp16Values) / max(outputScale, 1e-6), 1e-5,
                "\(rows) rows: halving the record changed the answer"
            )

            let reference = DeepSeekV41AttentionReference.attend(
                query: query,
                keys: (0..<rows).map { Array(floatKeys[($0 * width)..<(($0 + 1) * width)]) },
                shape: shape,
                sink: sink
            )
            for head in [0, 37] {
                let range = (head * valueDim)..<((head + 1) * valueDim)
                let scale = reference[range].map(abs).max() ?? 1
                for (label, values) in [("fp32", fp32Values), ("fp16", fp16Values)] {
                    XCTAssertLessThan(
                        worstDelta(Array(values[range]), Array(reference[range]))
                            / max(scale, 1e-6),
                        1e-4,
                        "\(rows) rows (\(label)): head \(head) does not match the host rule"
                    )
                }
            }

            // Two passes over the block per head, which is what the kernel reads:
            // `qkHeadDim` wide for the scores and `valueDim` wide for the output.
            let passBytes = Double(rows * (qkHeadDim + valueDim))
            let fp32Traffic = 2 * Double(heads) * passBytes
            let fp16Traffic = Double(heads) * passBytes
            print(
                String(
                    format: "[dsv41 key width] %4d rows  fp32 %7.3f ms (%6.0f GB/s)  "
                        + "fp16 %7.3f ms (%6.0f GB/s)  %5.2fx",
                    rows,
                    fp32Milliseconds,
                    fp32Traffic / (fp32Milliseconds / 1000) / 1e9,
                    fp16Milliseconds,
                    fp16Traffic / (fp16Milliseconds / 1000) / 1e9,
                    fp32Milliseconds / fp16Milliseconds
                )
            )
        }
    }

    /// One batched attention launch against eight single-position ones.
    ///
    /// `#14` established that `writeLatent`'s time IS the attention and that the
    /// attention runs one dispatch PER POSITION of a block: 128 threadgroups, which
    /// the journal describes as a fifth of the machine, eight times over. The kernel
    /// now carries a slot dimension, so a block can be one launch of 1,024
    /// threadgroups instead. This case is the A/B that decides whether wiring that is
    /// worth anything: same kernel, same per-threadgroup work, same total arithmetic,
    /// same total gathered bytes, and eight DISTINCT gathered blocks in both arms, so
    /// the only variables are the launch count and how much of the die one launch has
    /// to fill.
    ///
    /// The row counts ramp because a block's positions do not read the same number of
    /// rows: the window grows with the position until it fills the ring, so early
    /// positions of a prompt read less than late ones.
    func testWhetherOneBatchedAttentionLaunchBeatsEight() throws {
        guard let metal = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        guard let library = metal.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_attention_csa2_f16"),
              let pipeline = try? metal.makeComputePipelineState(function: function),
              let queue = metal.makeCommandQueue() else {
            XCTFail("the attention kernel is not in the default library")
            return
        }

        // The published text_config, at the block width the published ring allows.
        let slots = 8
        let heads = 128
        let qkHeadDim = 512
        let valueDim = 512
        let width = 512
        let shape = DeepSeekV41AttentionShape(
            numHeads: heads,
            headsPerGroup: 16,
            qkHeadDim: qkHeadDim,
            vHeadDim: valueDim,
            latentWidth: width,
            qLoraRank: 1536,
            oLoraRank: 1024,
            isCompressed: true,
            compressRatio: 2,
            ropeTheta: 160000,
            rotaryDim: 64
        )
        let rowsBySlot = (0..<slots).map { 128 + 64 * $0 }
        let rowStride = rowsBySlot.max() ?? 0

        func sampled(_ count: Int, seed: UInt64) -> [Float] {
            [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                var state = seed
                for index in 0..<count {
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    buffer[index] = Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 30)
                }
                initialized = count
            }
        }

        func buffer(_ values: [Float]) -> MTLBuffer? {
            values.withUnsafeBytes { bytes in
                metal.makeBuffer(
                    bytes: bytes.baseAddress!,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
        }

        let query = sampled(slots * heads * qkHeadDim, seed: 11)
        // Records at the cache's width, widened for the host rule. Both arms read
        // the same block, so the A/B is still launch count and nothing else.
        let keys = sampled(slots * rowStride * width, seed: 13).map { Float16($0) }
        let widenedKeys = keys.map { Float($0) }
        let sink = sampled(heads, seed: 17).map { $0 * 4 }
        let halfKeys = keys.withUnsafeBytes { bytes in
            metal.makeBuffer(
                bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared
            )
        }
        guard let queryBuffer = buffer(query),
              let keyBuffer = halfKeys,
              let sinkBuffer = buffer(sink),
              let outputBuffer = metal.makeBuffer(
                  length: slots * heads * valueDim * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let slotCounts = metal.makeBuffer(
                  length: slots * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared
              ) else {
            XCTFail("the attention buffers did not allocate")
            return
        }
        let slotCountsPointer = slotCounts.contents().assumingMemoryBound(to: UInt32.self)
        for (slot, rows) in rowsBySlot.enumerated() {
            slotCountsPointer[slot] = UInt32(rows)
        }

        let querySlotBytes = heads * qkHeadDim * MemoryLayout<Float>.stride
        let keySlotBytes = rowStride * width * MemoryLayout<Float16>.stride
        let outputSlotBytes = heads * valueDim * MemoryLayout<Float>.stride
        let scoreBytes = (rowStride + 32) * MemoryLayout<Float>.stride

        func bindScalars(_ encoder: MTLComputeCommandEncoder, rows: Int) {
            var rowCount = UInt32(rows)
            var headCount = UInt32(heads)
            var headWidth = UInt32(qkHeadDim)
            var headValue = UInt32(valueDim)
            var recordWidth = UInt32(width)
            encoder.setBytes(&rowCount, length: 4, index: 4)
            encoder.setBytes(&headCount, length: 4, index: 5)
            encoder.setBytes(&headWidth, length: 4, index: 6)
            encoder.setBytes(&headValue, length: 4, index: 7)
            encoder.setBytes(&recordWidth, length: 4, index: 8)
        }

        // ARM A: the shipped form, one dispatch per position, one gathered block each.
        func encodeSinglePerPosition(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(pipeline)
            for slot in 0..<slots {
                encoder.setBuffer(
                    queryBuffer, offset: slot * querySlotBytes, index: 0
                )
                encoder.setBuffer(keyBuffer, offset: slot * keySlotBytes, index: 1)
                encoder.setBuffer(sinkBuffer, offset: 0, index: 2)
                encoder.setBuffer(
                    outputBuffer, offset: slot * outputSlotBytes, index: 3
                )
                bindScalars(encoder, rows: rowsBySlot[slot])
                encoder.dispatchThreadgroups(
                    MTLSize(width: heads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
            }
        }

        // ARM B: one launch, the slot dimension carrying the positions.
        func encodeBatched(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)
            encoder.setBuffer(sinkBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            // The scalar row count is unused once a slot plan is bound; the kernel
            // reads `slotRowCounts` instead.
            bindScalars(encoder, rows: 0)
            var slotPlan = DeepSeekV41AttentionSlotPlan(
                slotCount: UInt32(slots), keyRowStride: UInt32(rowStride)
            )
            encoder.setBytes(
                &slotPlan, length: MemoryLayout<DeepSeekV41AttentionSlotPlan>.stride, index: 9
            )
            encoder.setBuffer(slotCounts, offset: 0, index: 10)
            encoder.dispatchThreadgroups(
                MTLSize(width: slots * heads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
        }

        let blocks = 10
        func time(_ encode: (MTLComputeCommandEncoder) -> Void) -> Double {
            var best = Double.infinity
            for attempt in 0..<(warmups + blocks) {
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return .infinity }
                encoder.setThreadgroupMemoryLength(scoreBytes, index: 0)
                for _ in 0..<(attempt >= warmups ? blocks : 1) {
                    encode(encoder)
                }
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                guard attempt >= warmups else { continue }
                let window = command.gpuEndTime - command.gpuStartTime
                best = min(best, window / Double(blocks) * 1000)
            }
            return best
        }

        // Encode once for correctness, then read the two arms back and compare each
        // against the host rule. The timing is only worth reading if it is the same
        // computation; two heads of two different slots is enough, because the
        // reduction, the sink and the accumulation are the same code everywhere.
        func run(_ encode: (MTLComputeCommandEncoder) -> Void) -> [Float] {
            guard let command = queue.makeCommandBuffer(),
                  let encoder = command.makeComputeCommandEncoder() else { return [] }
            // The scratch length has to be set BEFORE the dispatch is encoded; a
            // dispatch keeps whatever was set when it was written.
            encoder.setThreadgroupMemoryLength(scoreBytes, index: 0)
            encode(encoder)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            return floats(outputBuffer, count: slots * heads * valueDim)
        }

        let shipped = run(encodeSinglePerPosition)
        let batched = run(encodeBatched)
        for (slot, head) in [(2, 5), (7, 100)] {
            // The host rule runs a whole position at a time, so the reference is one
            // slot's worth of heads and the comparison picks the head out of it.
            let queryBase = slot * heads * qkHeadDim
            let reference = DeepSeekV41AttentionReference.attend(
                query: Array(query[queryBase..<(queryBase + heads * qkHeadDim)]),
                keys: (0..<rowsBySlot[slot]).map {
                    Array(
                        widenedKeys[
                            ((slot * rowStride + $0) * width)..<((slot * rowStride + $0 + 1) * width)
                        ]
                    )
                },
                shape: shape,
                sink: sink
            )
            let range = (head * valueDim)..<((head + 1) * valueDim)
            let scale = reference[range].map(abs).max() ?? 1
            let outputBase = (slot * heads + head) * valueDim
            let outputRange = outputBase..<(outputBase + valueDim)
            XCTAssertLessThan(
                worstDelta(Array(shipped[outputRange]), Array(reference[range])) / max(scale, 1e-6),
                1e-4,
                "slot \(slot) head \(head): the per-position launch does not match the host rule"
            )
            XCTAssertLessThan(
                worstDelta(Array(batched[outputRange]), Array(reference[range])) / max(scale, 1e-6),
                1e-4,
                "slot \(slot) head \(head): the batched launch does not match the host rule"
            )
        }

        let perPosition = time(encodeSinglePerPosition)
        let oneLaunch = time(encodeBatched)
        let totalRows = rowsBySlot.reduce(0, +)

        print(
            String(
                format: "[dsv41 attention batch] slots=%d heads=%d rows=%d..%d total=%d",
                slots, heads, rowsBySlot.first ?? 0, rowStride, totalRows
            )
        )
        print(
            String(
                format: "[dsv41 attention batch] %d launches %8.3f ms/block  %8.3f ms/position",
                slots, perPosition, perPosition / Double(slots)
            )
        )
        print(
            String(
                format: "[dsv41 attention batch] %d launch  %8.3f ms/block  %8.3f ms/position  %.2fx",
                1, oneLaunch, oneLaunch / Double(slots), perPosition / max(oneLaunch, 1e-9)
            )
        )
    }

    /// The stage table at the widths the checkpoint publishes.
    ///
    /// Every case above runs on the two-head default shape, and a ranking taken from
    /// it is a ranking of a DIFFERENT model. `#5` and `#6` both ended by naming
    /// `writeLatent` and `outputDown` as the next levers, and both numbers came from
    /// hidden 64 over a four-row window. This case runs the same stage accounting
    /// against `DeepSeekV41SyntheticShape.publishedWidths`: 7,168 hidden, 128 heads
    /// of 512 over a 640-row ring, eight output groups, four hyper-connection copies,
    /// one layer and a deliberately cheap mixture, so the dense half is the shipped
    /// half and what it costs per position is the checkpoint's answer rather than a
    /// toy's.
    ///
    /// Self-validation is not repeated here, on purpose and because it cannot be: a
    /// host reference over 190M weights and a 640-position prompt is not something a
    /// test can run. The GRAPH is pinned at the default shape by
    /// `DeepSeekV41DeviceGraphParityTests` — stage by stage, and the chunk seam
    /// against the walk — and the attention kernel at these widths by
    /// `testWhatOneAttentionDispatchCostsAtThePublishedWidths`. This case only
    /// answers where a block's device time goes when the widths are the checkpoint's.
    /// Times one prefill of `positions` tokens and prints the per-stage table.
    ///
    /// Two runs, fastest kept: the point is a ranking, and a ranking only needs the run
    /// the machine was not busy during. Stage rows are divided by positions times
    /// layers, so a table taken on a one-layer fixture and one taken on a two-layer
    /// fixture are both "what one position pays through one layer", and the stage keys
    /// are shared across layers (a compressed layer's pool lands inside `writeLatent`).
    private func stageTable(
        label: String,
        checkpoint: DeepSeekV41SyntheticCheckpoint,
        positions: Int
    ) throws {
        let runtime = try makeRuntime(checkpoint)
        let device = runtime.forward()
        let slots = DeepSeekV41DevicePlan.chunkSlots(
            config: checkpoint.config,
            contextLength: checkpoint.shape.contextLength
        )
        let shape = checkpoint.shape
        let tokens = (0..<positions).map { $0 % checkpoint.config.vocabSize }
        print(
            label + String(
                format: " hidden=%d heads=%d head_dim=%d record=%d window=%d slots=%d "
                    + "layers=%d hcMult=%d gatherRows=%d context=%d",
                shape.hidden,
                shape.heads,
                shape.headDim,
                shape.latentWidth,
                shape.window,
                slots,
                shape.layers,
                shape.hcMult,
                DeepSeekV41DevicePlan.gatherRowBudget(config: checkpoint.config),
                shape.contextLength
            )
        )

        var fastest: (wall: Double, stages: [String: UInt64], timing: DeepSeekV41BlockTiming)?
        for _ in 0..<2 {
            device.reset()
            device.resetStageTiming()
            device.resetBlockTiming()
            device.recordsStageTiming = true
            device.recordsBlockTiming = true
            let wall = try milliseconds {
                _ = try device.prefillChunk(tokens)
            }
            device.recordsStageTiming = false
            device.recordsBlockTiming = false
            let stages = device.stageTiming
            XCTAssertFalse(stages.isEmpty, "no stage reported a device window")
            if fastest == nil || wall < fastest!.wall {
                fastest = (wall, stages, device.blockTiming)
            }
        }
        let measured = try XCTUnwrap(fastest)
        XCTAssertGreaterThan(measured.timing.layers, 0, "the switch recorded nothing")

        let per = measured.timing.millisecondsPerPositionLayer()
        let encoded = per.denseEncode + per.mixtureEncode
        let deviceTime = per.denseDevice + per.mixtureDevice
        let denseShare = deviceTime > 0 ? per.denseDevice / deviceTime : 0
        let total = measured.stages.values.reduce(0, +)

        print(
            label + String(
                format: " positions=%d layers=%d, %.2f ms/position whole block "
                    + "(%.2f per position/layer)",
                positions,
                shape.layers,
                measured.wall / Double(positions),
                measured.wall / Double(positions * max(1, shape.layers))
            )
        )
        print(
            label + String(
                format: " host per position/layer: dense encode %.4f  front wait %.4f  "
                    + "mixture encode %.4f ms",
                per.denseEncode,
                per.wait,
                per.mixtureEncode
            )
        )
        print(
            label + String(
                format: " DEVICE per position/layer: dense %.4f  mixture %.4f ms "
                    + "(dense holds %.0f%%); encode is %.0f%% of accounted time",
                per.denseDevice,
                per.mixtureDevice,
                denseShare * 100,
                encoded + per.wait > 0 ? encoded / (encoded + per.wait) * 100 : 0
            )
        )
        for (stage, nanoseconds) in measured.stages.sorted(by: { $0.value > $1.value }) {
            print(
                label + String(
                    format: " %-26@ %8.3f ms/pos/layer  %5.1f%%",
                    stage as NSString,
                    Double(nanoseconds) / Double(positions * max(1, shape.layers)) / 1_000_000,
                    Double(nanoseconds) / Double(max(1, total)) * 100
                )
            )
        }
    }

    func testWhereABlocksRemainingTimeGoesAtThePublishedWidths() throws {
        try stageTable(
            label: "[dsv41 published]",
            checkpoint: DeepSeekV41SyntheticCheckpoint(shape: .publishedWidths()),
            positions: 640
        )
    }

    /// The same table with the compressed half switched ON.
    ///
    /// `#15` closed with one load-bearing caveat: the fixture declares no
    /// `compress_ratios > 0`, so attention reads the window half of its row budget
    /// only, the compressor's two per-position projections never run, and
    /// `writeLatent` carries no pool term. Every figure in the published table is
    /// therefore a lower bound for a compressed layer, and the share that table puts on
    /// `writeLatent` is a share of a cheaper stage than the checkpoint's.
    ///
    /// This runs the identical accounting on
    /// `DeepSeekV41SyntheticShape.compressedPublishedWidths`: one window-only layer and
    /// one compressed layer at the checkpoint's own 50/50 split, a 640-row window plus
    /// `index_topk` 512 compressed rows, and a context long enough that the last
    /// positions keep the full 512. The stage keys are shared across layers, so a stage
    /// that only a compressed layer runs (`compressPool`'s two projections land inside
    /// `writeLatent`) reads as the mix the model actually pays.
    func testWhereABlocksRemainingTimeGoesWithCompressedLayers() throws {
        try stageTable(
            label: "[dsv41 compressed]",
            checkpoint: DeepSeekV41SyntheticCheckpoint(shape: .compressedPublishedWidths()),
            positions: 1_024
        )
    }

    /// The compressed half's own cost, separated from the row budget it drags in.
    ///
    /// A compressed layer changes two things at once: it reads `window + index_topk`
    /// rows where a window-only layer reads only the window, and it runs the
    /// compressor's two projections plus the pool that emits a row. The mixed table
    /// above cannot tell those apart, and they call for different work — one means the
    /// gather or the attention kernel, the other means the compressor's GEMVs.
    ///
    /// Both arms here run ONE layer over the same 1,024 positions with the same window,
    /// the same context and the same weights everywhere but `compress_ratios`, so the
    /// per-stage difference is the compressed half alone.
    func testWhatTheCompressedHalfCostsOverTheWindowOnlyHalf() throws {
        try stageTable(
            label: "[dsv41 window-only]",
            checkpoint: DeepSeekV41SyntheticCheckpoint(
                shape: .publishedWidths(contextLength: 1_024)
            ),
            positions: 1_024
        )
        try stageTable(
            label: "[dsv41 compressed-1]",
            checkpoint: DeepSeekV41SyntheticCheckpoint(
                shape: .compressedPublishedWidths(layers: 1)
            ),
            positions: 1_024
        )
    }

    /// What one mHC coefficient projection costs at the published widths.
    ///
    /// `testWhereABlocksRemainingTimeGoesAtThePublishedWidths` leaves the two
    /// hyper-connection projections as the largest term of a block's dense half, at
    /// 0.74 ms/position between them. The cause is visible in the dispatch rather than
    /// the traffic: `(2 + mult) * mult` is 24 at `hc_mult` 4, so the kernel runs 24
    /// threads per position and each one walks a `mult * hidden` = 28,672-wide
    /// reduction twice, once for the RMS statistic and once for the dot product. That
    /// is one warp-ish per position against 2.75 MB of weight rows.
    ///
    /// This times it alone, at that geometry, and checks a few logits against the host
    /// rule before printing, so the number is never reported for a different
    /// computation.
    func testWhatOneHyperConnectionProjectionCostsAtThePublishedWidths() throws {
        guard let metal = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        guard let library = metal.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_hc_project_f32"),
              let pipeline = try? metal.makeComputePipelineState(function: function),
              let queue = metal.makeCommandQueue() else {
            XCTFail("the mHC projection kernel is not in the default library")
            return
        }

        // The published geometry: hidden 7,168 with four hyper-connection copies, so
        // the flattened stack is 28,672 wide and the projection has 24 output rows.
        let mult = 4
        let hidden = 7_168
        let width = mult * hidden
        let outDim = (2 + mult) * mult
        // The ceiling `chunkSlots` applies: eight positions per block.
        let slots = 8
        let epsilon: Float = 1e-6

        func sample(_ count: Int, seed: UInt64) -> [Float] {
            var state = seed
            return (0..<count).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 30)
            }
        }
        func buffer(_ values: [Float]) -> MTLBuffer? {
            values.withUnsafeBytes { bytes in
                metal.makeBuffer(
                    bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared
                )
            }
        }

        let weights = sample(outDim * width, seed: 11)
        let stream = sample(slots * width, seed: 12)
        guard let weightBuffer = buffer(weights),
              let streamBuffer = buffer(stream),
              let logitsBuffer = metal.makeBuffer(
                  length: slots * outDim * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            XCTFail("the mHC projection buffers did not allocate")
            return
        }
        print(
            "[dsv41 hc] width=\(width) outDim=\(outDim) slots=\(slots) "
                + "weights=\(weights.count * 4 / 1_048_576) MB"
        )

        func encode(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(streamBuffer, offset: 0, index: 1)
            encoder.setBuffer(logitsBuffer, offset: 0, index: 2)
            var weightOffset = UInt64(0)
            var normEpsilon = epsilon
            var w = UInt32(width)
            var out = UInt32(outDim)
            var slotCount = UInt32(slots)
            encoder.setBytes(&weightOffset, length: 8, index: 3)
            encoder.setBytes(&normEpsilon, length: 4, index: 4)
            encoder.setBytes(&w, length: 4, index: 5)
            encoder.setBytes(&out, length: 4, index: 6)
            encoder.setBytes(&slotCount, length: 4, index: 7)
        }
        func dispatch(_ encoder: MTLComputeCommandEncoder, threads: Int) {
            encoder.setThreadgroupMemoryLength(64 * MemoryLayout<Float>.stride, index: 0)
            encoder.dispatchThreadgroups(
                MTLSize(width: outDim * slots, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
            )
        }

        let repetitions = 20
        var timings: [(threads: Int, perDispatch: Double)] = []
        for threads in [32, 64, 128, 256, 512] {
            var perDispatch = Double.infinity
            for attempt in 0..<(warmups + repetitions) {
                guard attempt >= warmups else {
                    guard let command = queue.makeCommandBuffer(),
                          let encoder = command.makeComputeCommandEncoder() else { return }
                    encode(encoder)
                    dispatch(encoder, threads: threads)
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    continue
                }
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return }
                encode(encoder)
                for _ in 0..<repetitions {
                    dispatch(encoder, threads: threads)
                }
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                let window = command.gpuEndTime - command.gpuStartTime
                perDispatch = min(perDispatch, window / Double(repetitions) * 1000)
            }
            timings.append((threads, perDispatch))
        }

        // The host rule, for one slot: `rstd` of the flattened stack times the dot of
        // one weight row with it. Two rows are enough to catch a broken reduction.
        let logits = floats(logitsBuffer, count: slots * outDim)
        let slot = 0
        let stack = Array(stream[(slot * width)..<((slot + 1) * width)])
        let sumSquares = stack.reduce(0) { $0 + $1 * $1 }
        let rstd = 1 / (sumSquares / Float(width) + epsilon).squareRoot()
        for row in [0, 7, 23] {
            let weightRow = Array(weights[(row * width)..<((row + 1) * width)])
            var dot: Float = 0
            for index in 0..<width { dot += weightRow[index] * stack[index] }
            let expected = dot * rstd
            let actual = logits[slot * outDim + row]
            let scale = max(abs(expected), 1e-3)
            XCTAssertLessThan(
                abs(actual - expected) / scale,
                2e-3,
                "row \(row) disagrees with the host rule"
            )
        }

        for (threads, perDispatch) in timings {
            print(
                String(
                    format: "[dsv41 hc] %4d threads  %8.3f ms/dispatch  %8.3f ms/position "
                        + "over %d slots",
                    threads,
                    perDispatch,
                    perDispatch / Double(slots),
                    slots
                )
            )
        }
    }

    /// What one dense projection costs, at the shapes the checkpoint runs.
    ///
    /// `bf16_gemv` gives one thread an output row and the reduction is a serial
    /// loop, so its grid is `outDim * rows` threads and nothing else. `attn.wq_b`
    /// is 73,728 rows, which covers a GPU; `attn.wo_a` splits into eight groups of
    /// 128 rows over 8,192 columns, which is 1,024 threads for a block of eight
    /// positions. This case prices that difference on the five projections rather
    /// than on a toy shape, prices the threadgroup-per-row kernel that
    /// `bf16_gemv_rows` offers against it, and then sweeps the row count to find
    /// where one form overtakes the other.
    ///
    /// Every configuration self-checks before it is timed: both kernels' first and
    /// last output rows are compared with a host dot product over the same bytes, so
    /// a sweep cannot report a fast kernel that computes something else.
    func testWhatOneProjectionCostsAtThePublishedWidths() throws {
        guard let metal = MTLCreateSystemDefaultDevice(),
              let library = metal.makeDefaultLibrary(),
              let perRowFunction = library.makeFunction(name: "bf16_gemv"),
              let rowsFunction = library.makeFunction(name: "bf16_gemv_rows"),
              let blockFunction = library.makeFunction(name: "bf16_gemv_rows8"),
              let perRow = try? metal.makeComputePipelineState(function: perRowFunction),
              let rows = try? metal.makeComputePipelineState(function: rowsFunction),
              let block = try? metal.makeComputePipelineState(function: blockFunction),
              let queue = metal.makeCommandQueue() else {
            XCTFail("the GEMV kernels are not in the default library")
            return
        }

        let repetitions = 20
        let laneWidth = 64

        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func nextUnit() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 30)
        }
        // A tile copied over and over: a 226 MB weight does not need 113M distinct
        // values to be priced, and generating them one at a time on the CPU costs
        // more than the measurement.
        let weightTile = (0..<4_096).map { _ in
            UInt16(
                truncatingIfNeeded:
                    DeepSeekV41SyntheticCheckpoint.truncateToBF16(nextUnit() * 0.125).bitPattern >> 16
            )
        }
        let inputTile = (0..<4_096).map { _ in nextUnit() }

        func fill<T>(_ buffer: MTLBuffer, bytes: Int, tile: [T]) {
            let valueStride = MemoryLayout<T>.stride
            let destination = buffer.contents()
            tile.withUnsafeBytes { source in
                guard let base = source.baseAddress else { return }
                var written = 0
                while written < bytes {
                    let chunk = min(tile.count * valueStride, bytes - written)
                    memcpy(destination + written, base, chunk)
                    written += chunk
                }
            }
        }

        /// One shape at one batch height: validate both kernels, then time both.
        ///
        /// `silent` skips the print, which is what the crossover sweep wants — it
        /// only needs the two numbers.
        func measure(
            name: String,
            outDim: Int,
            inDim: Int,
            dispatches: Int,
            slots: Int,
            laneSweep: Bool
        ) throws {
            let weightBytes = outDim * inDim * MemoryLayout<UInt16>.stride
            guard let weight = metal.makeBuffer(length: weightBytes, options: .storageModeShared),
                  let input = metal.makeBuffer(
                      length: slots * inDim * MemoryLayout<Float>.stride, options: .storageModeShared
                  ),
                  let perRowOutput = metal.makeBuffer(
                      length: slots * outDim * MemoryLayout<Float>.stride, options: .storageModeShared
                  ),
                  let rowsOutput = metal.makeBuffer(
                      length: slots * outDim * MemoryLayout<Float>.stride, options: .storageModeShared
                  ),
                  let blockOutput = metal.makeBuffer(
                      length: 8 * outDim * MemoryLayout<Float>.stride, options: .storageModeShared
                  ) else {
                XCTFail("\(name): buffers did not allocate")
                return
            }
            fill(weight, bytes: weightBytes, tile: weightTile)
            fill(input, bytes: slots * inDim * MemoryLayout<Float>.stride, tile: inputTile)
            memset(perRowOutput.contents(), 0, perRowOutput.length)
            memset(rowsOutput.contents(), 0, rowsOutput.length)
            memset(blockOutput.contents(), 0, blockOutput.length)

            func encode(
                _ encoder: MTLComputeCommandEncoder,
                pipeline: MTLComputePipelineState,
                output: MTLBuffer,
                lanes: Int
            ) {
                encoder.setComputePipelineState(pipeline)
                var weightOffset = UInt64(0)
                var inDimension = UInt32(inDim)
                var outDimension = UInt32(outDim)
                var inStride = UInt32(inDim)
                var outStride = UInt32(outDim)
                for _ in 0..<dispatches {
                    encoder.setBuffer(weight, offset: 0, index: 0)
                    encoder.setBuffer(input, offset: 0, index: 1)
                    encoder.setBuffer(output, offset: 0, index: 2)
                    encoder.setBytes(&weightOffset, length: 8, index: 3)
                    encoder.setBytes(&inDimension, length: 4, index: 4)
                    encoder.setBytes(&outDimension, length: 4, index: 5)
                    encoder.setBytes(&inStride, length: 4, index: 6)
                    encoder.setBytes(&outStride, length: 4, index: 7)
                    if pipeline === rows {
                        encoder.setThreadgroupMemoryLength(
                            32 * MemoryLayout<Float>.stride, index: 0
                        )
                        encoder.dispatchThreadgroups(
                            MTLSize(width: outDim * slots, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
                        )
                    } else if pipeline === block {
                        encoder.setThreadgroupMemoryLength(
                            32 * 8 * MemoryLayout<Float>.stride, index: 0
                        )
                        encoder.dispatchThreadgroups(
                            MTLSize(width: outDim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
                        )
                    } else {
                        // The executor's `threadgroup(_:)` caps the width at 64, so
                        // this mirrors the launch the stage actually uses.
                        encoder.dispatchThreads(
                            MTLSize(width: outDim, height: slots, depth: 1),
                            threadsPerThreadgroup: MTLSize(
                                width: max(1, min(outDim, 64)), height: 1, depth: 1
                            )
                        )
                    }
                }
            }

            func run(_ pipeline: MTLComputePipelineState, _ output: MTLBuffer, _ lanes: Int) {
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return }
                encode(encoder, pipeline: pipeline, output: output, lanes: lanes)
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                XCTAssertNil(command.error, "\(name): the dispatch failed")
            }
            run(perRow, perRowOutput, 0)
            run(rows, rowsOutput, laneWidth)
            // The block form only covers a full block of eight positions.
            if slots == 8 { run(block, blockOutput, laneWidth) }

            // The host rule over the same bytes, at the first and last row, for both
            // kernels. Cheap enough at every shape here.
            let weights = weight.contents().assumingMemoryBound(to: UInt16.self)
            let inputs = input.contents().assumingMemoryBound(to: Float.self)
            var validated: [(MTLBuffer, String)] = [
                (perRowOutput, "perRow"), (rowsOutput, "rows"),
            ]
            if slots == 8 { validated.append((blockOutput, "block")) }
            for (output, label) in validated {
                let values = output.contents().assumingMemoryBound(to: Float.self)
                for (row, token) in [(0, 0), (outDim - 1, slots - 1)] {
                    var expected: Float = 0
                    for index in 0..<inDim {
                        let value = Float(bitPattern: UInt32(weights[row * inDim + index]) << 16)
                        expected += value * inputs[token * inDim + index]
                    }
                    let actual = values[token * outDim + row]
                    XCTAssertLessThan(
                        abs(actual - expected) / max(abs(expected), 1e-3),
                        1e-4,
                        "\(name) \(label) row \(row) token \(token) disagrees with the host rule"
                    )
                }
            }

            func best(_ pipeline: MTLComputePipelineState, _ output: MTLBuffer, _ lanes: Int)
                -> (milliseconds: Double, gigabytesPerSecond: Double) {
                var fastest = Double.infinity
                for attempt in 0..<3 {
                    guard let command = queue.makeCommandBuffer(),
                          let encoder = command.makeComputeCommandEncoder() else {
                        return (0, 0)
                    }
                    // `repetitions` identical dispatches per window: the first window
                    // is a single warm-up, and the minimum over the rest is the floor.
                    let count = attempt == 0 ? 1 : repetitions
                    for _ in 0..<count {
                        encode(encoder, pipeline: pipeline, output: output, lanes: lanes)
                    }
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    let window = command.gpuEndTime - command.gpuStartTime
                    fastest = min(fastest, window / Double(count) * 1000)
                }
                let bytes = Double(dispatches * outDim * inDim * 2)
                return (fastest, bytes / (fastest / 1000) / 1e9)
            }

            let shipped = best(perRow, perRowOutput, 0)
            let widened = best(rows, rowsOutput, laneWidth)
            var line = String(
                format: "[dsv41 gemv] %-14@ slots=%-2d out=%-6d in=%-5d "
                    + "perRow %8.4f ms/pos   rows %8.4f ms/pos (%+.2fx)",
                name as NSString,
                slots,
                outDim,
                inDim,
                shipped.milliseconds / Double(slots),
                widened.milliseconds / Double(slots),
                shipped.milliseconds / widened.milliseconds
            )
            if slots == 8 {
                let shared = best(block, blockOutput, laneWidth)
                line += String(
                    format: "   block %8.4f ms/pos (%+.2fx)",
                    shared.milliseconds / 8,
                    shipped.milliseconds / shared.milliseconds
                )
            }
            print(line)
            guard laneSweep else { return }
            for lanes in [32, 64, 128, 256] {
                let swept = best(rows, rowsOutput, lanes)
                print(
                    String(
                        format: "[dsv41 gemv] %-14@ slots=%-2d %11@ lanes=%-4d           "
                            + "%8.4f ms/pos %5.0f GB/s",
                        name as NSString,
                        slots,
                        "" as NSString,
                        lanes,
                        swept.milliseconds / Double(slots),
                        swept.gigabytesPerSecond
                    )
                )
            }
        }

        // The five bf16 projections of a block, at the published widths. The grouped
        // output projection is eight band dispatches of the same geometry, so it is
        // priced as one stage of eight. Both batch heights matter: a decode step is
        // the same kernel with a grid one row tall.
        let shapes: [(name: String, outDim: Int, inDim: Int, dispatches: Int)] = [
            ("latentProject", 448, 7_168, 1),
            ("queryDown", 1_536, 7_168, 1),
            ("queryUp", 73_728, 1_536, 1),
            ("outputUp", 7_168, 1_024, 1),
            ("outputDown(8)", 128, 8_192, 8),
        ]
        for slots in [8, 1] {
            for shape in shapes {
                try measure(
                    name: shape.name,
                    outDim: shape.outDim,
                    inDim: shape.inDim,
                    dispatches: shape.dispatches,
                    slots: slots,
                    laneSweep: slots == 8
                )
            }
        }

        // Where the two forms cross over: one wide reduction (8,192 columns, the
        // output projection's band width) against a growing row count. The rule the
        // executor applies comes from this table, not from intuition.
        for outDim in [96, 192, 384, 768, 1_536, 3_072, 6_144] {
            try measure(
                name: "crossover",
                outDim: outDim,
                inDim: 8_192,
                dispatches: 1,
                slots: 8,
                laneSweep: false
            )
        }
    }

    /// Whether a stage's cost scales with how much DISTINCT weight it reads.
    ///
    /// `testWhatOneProjectionCostsAtThePublishedWidths` prices one band of the
    /// attention output projection over and over — one 2.1 MB weight, eight
    /// dispatches — and reports 0.0133 ms/position. The same projection inside a
    /// block reads eight DIFFERENT bands, 16.8 MB, and the published-widths table
    /// says 0.145. Both forms of the kernel are in the two numbers, so neither
    /// explains the difference; the amount of distinct weight does.
    ///
    /// This holds the geometry fixed and grows only the number of distinct bands
    /// dispatched, so the two throughputs separate: requests per second, which a warm
    /// working set can serve from cache, against distinct bytes per second, which is
    /// what the memory system actually has to deliver. If the second one falls off a
    /// cliff as the footprint grows, a stage's in-place cost is a property of the
    /// footprint and not of its kernel.
    func testHowAStagesCostScalesWithTheWeightItReads() throws {
        guard let metal = MTLCreateSystemDefaultDevice(),
              let library = metal.makeDefaultLibrary(),
              let function = library.makeFunction(name: "bf16_gemv_rows"),
              let pipeline = try? metal.makeComputePipelineState(function: function),
              let queue = metal.makeCommandQueue() else {
            throw XCTSkip("No Metal device or kernel")
        }
        // One band of `attn.wo_a` at the published widths, and a block of positions.
        let outDim = 128
        let inDim = 8_192
        let slots = 8
        let laneWidth = 64
        // Enough headroom for the widest footprint below (64 bands = 134 MB).
        let maxBands = 64
        let weightBytes = maxBands * outDim * inDim * MemoryLayout<UInt16>.stride
        let inputBytes = slots * inDim * MemoryLayout<Float>.stride
        let outputBytes = slots * outDim * MemoryLayout<Float>.stride
        guard let weight = metal.makeBuffer(length: weightBytes, options: .storageModeShared),
              let input = metal.makeBuffer(length: inputBytes, options: .storageModeShared),
              let output = metal.makeBuffer(length: outputBytes, options: .storageModeShared) else {
            throw XCTSkip("could not allocate the footprint sweep's buffers")
        }

        var state: UInt64 = 0x243F_6A88_85A3_08D3
        func nextUnit() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 30)
        }
        let tile = (0..<4_096).map { _ in
            UInt16(
                truncatingIfNeeded:
                    DeepSeekV41SyntheticCheckpoint.truncateToBF16(nextUnit() * 0.125).bitPattern >> 16
            )
        }
        let inputTile = (0..<4_096).map { _ in nextUnit() }
        func fill<T>(_ buffer: MTLBuffer, bytes: Int, values: [T]) {
            let stride = MemoryLayout<T>.stride
            let destination = buffer.contents()
            values.withUnsafeBytes { source in
                guard let base = source.baseAddress else { return }
                var written = 0
                while written < bytes {
                    let chunk = min(values.count * stride, bytes - written)
                    memcpy(destination + written, base, chunk)
                    written += chunk
                }
            }
        }
        fill(weight, bytes: weightBytes, values: tile)
        fill(input, bytes: inputBytes, values: inputTile)
        memset(output.contents(), 0, output.length)

        func encode(_ encoder: MTLComputeCommandEncoder, bands: Int) {
            encoder.setComputePipelineState(pipeline)
            var weightOffset = UInt64(0)
            var inDimension = UInt32(inDim)
            var outDimension = UInt32(outDim)
            var inStride = UInt32(inDim)
            var outStride = UInt32(outDim)
            var noBand = UInt32(0)
            // One dispatch per band, each reading a DIFFERENT 2.1 MB block, which is
            // what the block-diagonal projection does.
            for band in 0..<bands {
                var bandOffset = UInt64(band * outDim * inDim * 2)
                encoder.setBuffer(weight, offset: 0, index: 0)
                encoder.setBuffer(input, offset: 0, index: 1)
                encoder.setBuffer(output, offset: 0, index: 2)
                encoder.setBytes(&bandOffset, length: 8, index: 3)
                encoder.setBytes(&inDimension, length: 4, index: 4)
                encoder.setBytes(&outDimension, length: 4, index: 5)
                encoder.setBytes(&inStride, length: 4, index: 6)
                encoder.setBytes(&outStride, length: 4, index: 7)
                encoder.setBytes(&noBand, length: 4, index: 8)
                encoder.setBytes(&noBand, length: 4, index: 9)
                encoder.setBytes(&noBand, length: 4, index: 10)
                encoder.setThreadgroupMemoryLength(
                    32 * MemoryLayout<Float>.stride, index: 0
                )
                encoder.dispatchThreadgroups(
                    MTLSize(width: outDim * slots, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: laneWidth, height: 1, depth: 1)
                )
                weightOffset = bandOffset
            }
            _ = weightOffset
        }

        // The host rule once, so a fast row is never a different computation: the last
        // band's first output row for the first and the last token.
        if let command = queue.makeCommandBuffer(),
           let encoder = command.makeComputeCommandEncoder() {
            encode(encoder, bands: maxBands)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
        }
        let weights = weight.contents().assumingMemoryBound(to: UInt16.self)
        let inputs = input.contents().assumingMemoryBound(to: Float.self)
        let values = output.contents().assumingMemoryBound(to: Float.self)
        for (row, token) in [(0, 0), (outDim - 1, slots - 1)] {
            var expected: Float = 0
            for index in 0..<inDim {
                let base = ((maxBands - 1) * outDim + row) * inDim + index
                let value = Float(bitPattern: UInt32(weights[base]) << 16)
                expected += value * inputs[token * inDim + index]
            }
            let actual = values[token * outDim + row]
            XCTAssertLessThan(
                abs(actual - expected) / max(abs(expected), 1e-3),
                1e-4,
                "row \(row) token \(token) disagrees with the host rule"
            )
        }

        for bands in [1, 2, 4, 8, 16, 32, 64] {
            var fastest = Double.infinity
            for attempt in 0..<3 {
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return }
                let count = attempt == 0 ? 1 : 5
                for _ in 0..<count {
                    encode(encoder, bands: bands)
                }
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                let window = command.gpuEndTime - command.gpuStartTime
                fastest = min(fastest, window / Double(count) * 1000)
            }
            let distinctBytes = Double(bands * outDim * inDim * 2)
            let requestedBytes = distinctBytes * Double(slots)
            print(
                String(
                    format: "[dsv41 footprint] %2d bands %5.1f MB distinct  %8.4f ms/pos  "
                        + "%5.0f GB/s unique  %5.0f GB/s requested",
                    bands,
                    distinctBytes / 1_048_576,
                    fastest / Double(slots),
                    distinctBytes / (fastest / 1000) / 1e9,
                    requestedBytes / (fastest / 1000) / 1e9
                )
            )
        }
    }

    /// What a dense projection's shape costs, swept three ways: block width, rows per
    /// threadgroup, and input width.
    ///
    /// `#17` left the dense half's three big projections level with each other —
    /// `queryUp` 0.046, `outputDown` 0.045, `outputUp` 0.044 ms/pos/layer — despite
    /// reading 201, 134 and 117 MB of weights apiece. Reading bytes at 580, 408 and
    /// 342 GB/s looked like a streaming band that only a wider block could divide.
    ///
    /// It is not. A sixteen-position block measured 0.97x, 0.62x and 0.54x per
    /// position — WORSE, and worst where the row is longest — because the weight is
    /// already read once per block and what doubles with the block is the INPUT: a
    /// threadgroup reading one output row loads every position's vector for the length
    /// of that row, once per row, over tens of thousands of rows. So the shapes that
    /// matter are rows-per-threadgroup and input width, and this case sweeps both.
    ///
    /// Same matrices, same band structure, same caller layout as the executor uses
    /// (`attn.wo_a` as eight bands of 8,192 -> 1,024 reading `inDim`-wide rows).
    ///
    /// Measured elsewhere and left out here rather than shipped, in the same session
    /// and on the same fixtures: a four-row threadgroup lost everywhere (0.82x, 0.82x,
    /// 1.30x) because thirty-two accumulators cost more occupancy than the input they
    /// save, and a sixteen-position block lost as above.
    func testWhatAWiderBlockSavesOnAWeightRead() throws {
        guard let metal = MTLCreateSystemDefaultDevice(),
              let library = metal.makeDefaultLibrary(),
              let narrowFunction = library.makeFunction(name: "bf16_gemv_rows8"),
              let twoRowFunction = library.makeFunction(name: "bf16_gemv_rows8x2"),
              let narrow = try? metal.makeComputePipelineState(function: narrowFunction),
              let twoRow = try? metal.makeComputePipelineState(function: twoRowFunction),
              let queue = metal.makeCommandQueue() else {
            throw XCTSkip("No Metal device or block GEMV kernels")
        }

        struct Projection {
            let name: String
            let inDim: Int
            let bandOutDim: Int
            let bands: Int
            /// The caller's row width, which is the module's full input width even
            /// when one band reads only a slice of it.
            let inStride: Int
            let outStride: Int
        }
        // The published text_config: hidden 7,168, q_lora_rank 1,536, 128 heads of
        // 512, o_lora_rank 1,024 over 8 output groups.
        let projections = [
            Projection(
                name: "queryUp", inDim: 1_536, bandOutDim: 65_536, bands: 1,
                inStride: 1_536, outStride: 65_536
            ),
            Projection(
                name: "outputDown", inDim: 8_192, bandOutDim: 1_024, bands: 8,
                inStride: 65_536, outStride: 8_192
            ),
            Projection(
                name: "outputUp", inDim: 8_192, bandOutDim: 7_168, bands: 1,
                inStride: 8_192, outStride: 7_168
            )
        ]
        // The pair form is a win where a row is long and neutral where it is short
        // (1,536), so the rule needs the crossover rather than a guess: the same
        // output shape swept across input widths, which is the only thing that moves.
        let crossover = [1_024, 1_536, 2_048, 3_072, 4_096, 6_144].map { width in
            Projection(
                name: "inDim \(width)", inDim: width, bandOutDim: 7_168, bands: 1,
                inStride: width, outStride: 7_168
            )
        }
        let widestOut = projections.map(\.outStride).max() ?? 1
        let widestIn = projections.map(\.inStride).max() ?? 1
        let maxWeightBytes = projections
            .map { $0.bands * $0.bandOutDim * $0.inDim * MemoryLayout<UInt16>.stride }
            .max() ?? 1
        let slots = 16
        let laneWidth = 64

        guard let weight = metal.makeBuffer(length: maxWeightBytes, options: .storageModeShared),
              let input = metal.makeBuffer(
                  length: slots * widestIn * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let output = metal.makeBuffer(
                  length: slots * widestOut * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw XCTSkip("could not allocate the block GEMV buffers")
        }

        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func nextUnit() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: state >> 33)) / Float(1 << 30)
        }
        // BF16 values truncated exactly as the checkpoint's own reader widens them,
        // so the host rule below reads the same numbers the GPU does.
        let weightTile: [UInt16] = (0..<4_096).map { _ in
            UInt16(
                truncatingIfNeeded:
                    DeepSeekV41SyntheticCheckpoint.truncateToBF16(nextUnit() * 0.125).bitPattern >> 16
            )
        }
        let inputTile: [Float] = (0..<4_096).map { _ in nextUnit() }
        func fill<T>(_ buffer: MTLBuffer, bytes: Int, values: [T]) {
            let stride = MemoryLayout<T>.stride
            let destination = buffer.contents()
            values.withUnsafeBytes { source in
                guard let base = source.baseAddress else { return }
                var written = 0
                while written < bytes {
                    let chunk = min(values.count * stride, bytes - written)
                    memcpy(destination + written, base, chunk)
                    written += chunk
                }
            }
        }

        func encode(
            _ encoder: MTLComputeCommandEncoder,
            pipeline: MTLComputePipelineState,
            projection: Projection,
            block: Int,
            rowsPerGroup: Int = 1
        ) {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(weight, offset: 0, index: 0)
            encoder.setBuffer(input, offset: 0, index: 1)
            var inDimension = UInt32(projection.inDim)
            var bandOut = UInt32(projection.bandOutDim)
            var inStride = UInt32(projection.inStride)
            var outStride = UInt32(projection.outStride)
            var noBand = UInt32(0)
            for band in 0..<projection.bands {
                var weightOffset = UInt64(band)
                    * UInt64(projection.bandOutDim * projection.inDim)
                    * UInt64(MemoryLayout<UInt16>.stride)
                encoder.setBuffer(
                    output,
                    offset: band * projection.bandOutDim * MemoryLayout<Float>.stride,
                    index: 2
                )
                encoder.setBytes(&weightOffset, length: 8, index: 3)
                encoder.setBytes(&inDimension, length: 4, index: 4)
                encoder.setBytes(&bandOut, length: 4, index: 5)
                encoder.setBytes(&inStride, length: 4, index: 6)
                encoder.setBytes(&outStride, length: 4, index: 7)
                encoder.setBytes(&noBand, length: 4, index: 8)
                encoder.setBytes(&noBand, length: 4, index: 9)
                encoder.setBytes(&noBand, length: 4, index: 10)
                encoder.setThreadgroupMemoryLength(
                    32 * block * rowsPerGroup * MemoryLayout<Float>.stride, index: 0
                )
                encoder.dispatchThreadgroups(
                    MTLSize(
                        width: (projection.bandOutDim + rowsPerGroup - 1) / rowsPerGroup,
                        height: 1,
                        depth: 1
                    ),
                    threadsPerThreadgroup: MTLSize(width: laneWidth, height: 1, depth: 1)
                )
            }
        }

        func measure(
            pipeline: MTLComputePipelineState,
            projection: Projection,
            block: Int,
            repeats: Int,
            rowsPerGroup: Int = 1
        ) -> Double? {
            guard let command = queue.makeCommandBuffer(),
                  let encoder = command.makeComputeCommandEncoder() else { return nil }
            encoder.setThreadgroupMemoryLength(
                32 * block * rowsPerGroup * MemoryLayout<Float>.stride, index: 0
            )
            for _ in 0..<repeats {
                encode(
                    encoder, pipeline: pipeline, projection: projection, block: block,
                    rowsPerGroup: rowsPerGroup
                )
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            return (command.gpuEndTime - command.gpuStartTime) / Double(repeats) * 1000
        }

        print("[dsv41 block width] laneWidth=\(laneWidth) arms: rows8 (1 row/group) vs rows8x2 (2)")

        for projection in projections + crossover {
            let weightBytes = projection.bands * projection.bandOutDim * projection.inDim * 2
            fill(weight, bytes: weightBytes, values: weightTile)
            fill(input, bytes: slots * widestIn * MemoryLayout<Float>.stride, values: inputTile)
            memset(output.contents(), 0, output.length)

            // Correctness before timing: the last band's row 3 for a position at
            // each end of the block, against the host dot product.
            let arms: [(name: String, pipeline: MTLComputePipelineState, block: Int, rows: Int)] = [
                ("rows8", narrow, 8, 1),
                ("rows8x2", twoRow, 8, 2)
            ]
            for arm in arms {
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return }
                encode(
                    encoder, pipeline: arm.pipeline, projection: projection, block: arm.block,
                    rowsPerGroup: arm.rows
                )
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                XCTAssertNil(command.error)

                let band = projection.bands - 1
                let row = 3
                let weights = weight.contents().assumingMemoryBound(to: UInt16.self)
                let inputs = input.contents().assumingMemoryBound(to: Float.self)
                let values = output.contents().assumingMemoryBound(to: Float.self)
                for position in [0, arm.block - 1] {
                    var expected: Float = 0
                    for index in 0..<projection.inDim {
                        let base = (band * projection.bandOutDim + row) * projection.inDim + index
                        expected += Float(bitPattern: UInt32(weights[base]) << 16)
                            * inputs[position * projection.inStride + index]
                    }
                    let actual = values[
                        band * projection.bandOutDim + row + position * projection.outStride
                    ]
                    XCTAssertLessThan(
                        abs(actual - expected) / max(abs(expected), 1e-3),
                        1e-4,
                        "\(projection.name) \(arm.name): position \(position) disagrees"
                            + " with the host rule"
                    )
                }
            }
            // The two-row form also has to fill its SECOND row, which is a different
            // threadgroup's worth of output in every other form.
            do {
                guard let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else { return }
                encode(
                    encoder, pipeline: twoRow, projection: projection, block: 8, rowsPerGroup: 2
                )
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                let band = projection.bands - 1
                let weights = weight.contents().assumingMemoryBound(to: UInt16.self)
                let inputs = input.contents().assumingMemoryBound(to: Float.self)
                let values = output.contents().assumingMemoryBound(to: Float.self)
                for row in [2, 4] {
                    var expected: Float = 0
                    for index in 0..<projection.inDim {
                        let base = (band * projection.bandOutDim + row) * projection.inDim + index
                        expected += Float(bitPattern: UInt32(weights[base]) << 16)
                            * inputs[5 * projection.inStride + index]
                    }
                    let actual = values[band * projection.bandOutDim + row + 5 * projection.outStride]
                    XCTAssertLessThan(
                        abs(actual - expected) / max(abs(expected), 1e-3),
                        1e-4,
                        "\(projection.name) rows8x2: row \(row) disagrees with the host rule"
                    )
                }
            }

            // Interleaved, so a clock ramp cannot favour whichever arm went first.
            let repeats = 3
            for arm in arms {
                _ = measure(
                    pipeline: arm.pipeline, projection: projection, block: arm.block,
                    repeats: 1, rowsPerGroup: arm.rows
                )
            }
            var best = [Double](repeating: .infinity, count: arms.count)
            for _ in 0..<5 {
                for (index, arm) in arms.enumerated() {
                    if let value = measure(
                        pipeline: arm.pipeline, projection: projection, block: arm.block,
                        repeats: repeats, rowsPerGroup: arm.rows
                    ) {
                        best[index] = min(best[index], value)
                    }
                }
            }
            guard best.allSatisfy({ $0.isFinite }) else {
                XCTFail("\(projection.name) did not run")
                return
            }
            let megabytes = Double(weightBytes) / 1_048_576
            var perPosition: [Double] = []
            var columns: [String] = []
            for (index, arm) in arms.enumerated() {
                let milliseconds = best[index]
                let position = milliseconds / Double(arm.block)
                perPosition.append(position)
                let gigabytes = Double(weightBytes) / (milliseconds / 1000) / 1e9
                columns.append(
                    String(
                        format: "%@ %6.3f ms (%5.3f/pos, %4.0f GB/s)",
                        arm.name as NSString,
                        milliseconds,
                        position,
                        gigabytes
                    )
                )
            }
            var ratios = String(format: "   rows8/pos against:")
            for index in 1..<perPosition.count {
                ratios += String(
                    format: " %@ %.2fx", arms[index].name as NSString,
                    perPosition[0] / perPosition[index]
                )
            }
            let head = String(
                format: "[dsv41 block width] %-11@ %5.0f MB  ",
                projection.name as NSString,
                megabytes
            )
            print(head + columns.joined(separator: "  ") + ratios)
        }
    }
}
