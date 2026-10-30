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
              let function = library.makeFunction(name: "dsv41_attention_csa2_f32"),
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
            let keys = sample(rows * width, seed: UInt64(rows) &+ 7)
            guard let keyBuffer = buffer(keys) else {
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
                    Array(keys[($0 * width)..<(($0 + 1) * width)])
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
    func testWhereABlocksRemainingTimeGoesAtThePublishedWidths() throws {
        let positions = 640
        let checkpoint = DeepSeekV41SyntheticCheckpoint(shape: .publishedWidths())
        let runtime = try makeRuntime(checkpoint)
        let device = runtime.forward()
        let slots = DeepSeekV41DevicePlan.chunkSlots(
            config: checkpoint.config,
            contextLength: checkpoint.shape.contextLength
        )
        let shape = checkpoint.shape
        let tokens = (0..<positions).map { $0 % checkpoint.config.vocabSize }
        print(
            String(
                format: "[dsv41 published] hidden=%d heads=%d head_dim=%d record=%d "
                    + "window=%d slots=%d layers=%d hcMult=%d",
                shape.hidden,
                shape.heads,
                shape.headDim,
                shape.latentWidth,
                shape.window,
                slots,
                shape.layers,
                shape.hcMult
            )
        )

        // Two runs, fastest kept: the point is a ranking, and a ranking only needs the
        // run the machine was not busy during.
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
            String(
                format: "[dsv41 published] positions=%d, %.2f ms/position whole block",
                positions,
                measured.wall / Double(positions)
            )
        )
        print(
            String(
                format: "[dsv41 published] host per position/layer: dense encode %.4f  "
                    + "front wait %.4f  mixture encode %.4f ms",
                per.denseEncode,
                per.wait,
                per.mixtureEncode
            )
        )
        print(
            String(
                format: "[dsv41 published] DEVICE per position/layer: dense %.4f  mixture "
                    + "%.4f ms (dense holds %.0f%%); encode is %.0f%% of accounted time",
                per.denseDevice,
                per.mixtureDevice,
                denseShare * 100,
                encoded + per.wait > 0 ? encoded / (encoded + per.wait) * 100 : 0
            )
        )
        for (stage, nanoseconds) in measured.stages.sorted(by: { $0.value > $1.value }) {
            print(
                String(
                    format: "[dsv41 published] %-26@ %8.3f ms/pos  %5.1f%%",
                    stage as NSString,
                    Double(nanoseconds) / Double(positions) / 1_000_000,
                    Double(nanoseconds) / Double(max(1, total)) * 100
                )
            )
        }
    }
}
