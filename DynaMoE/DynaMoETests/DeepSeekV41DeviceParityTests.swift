//
//  DeepSeekV41DeviceParityTests.swift
//  DynaMoETests
//
//  The device graph against the host reference, on a model small enough to run.
//
//  Every other test in this family feeds ONE kernel and compares it with its own
//  CPU definition. That cannot see the wiring between kernels, which is where this
//  port's real bugs have been: an output projection reading the wrong group's
//  columns, `hc_post` walking a row of the mixing matrix instead of a column, a
//  projection advancing a float pointer by a byte offset. Each of those runs at
//  full speed and produces plausible text.
//
//  `DeepSeekV41SyntheticCheckpoint` builds the same tensor names the checkpoint
//  publishes at widths the config derives, and both backends read the identical
//  values, so a disagreement here is a disagreement about the graph.
//

import XCTest
import Metal
@testable import DynaMoE

final class DeepSeekV41SyntheticCheckpointTests: XCTestCase {

    func testTheCatalogAcceptsTheSyntheticCheckpoint() {
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        XCTAssertEqual(checkpoint.plan.count, checkpoint.shape.layers)
        XCTAssertEqual(checkpoint.catalogProblems(), [])
    }

    /// The formats the executor dispatches on, resolved the way a real load does.
    func testEveryProjectionResolvesToTheFormatItsDtypeSays() {
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let summary = checkpoint.summary()
        for tensor in checkpoint.tensors {
            let deviceTensor = DeepSeekV41DeviceTensor(
                shardIndex: 0,
                byteOffset: 0,
                rows: tensor.shape[0],
                columns: tensor.shape.count == 2 ? tensor.shape[1] : 1,
                dtype: tensor.dtype
            )
            let format = DeepSeekV41DevicePlan.linearFormat(
                tensor: deviceTensor,
                scale: nil
            )
            if tensor.dtype == "F32" {
                // The hyper-connection projections are read as raw FP32 bytes, not
                // through a GEMV: a bf16 one is refused at load.
                XCTAssertEqual(format, .unsupported, tensor.name)
            } else {
                XCTAssertEqual(format, .bf16, tensor.name)
            }
        }
        XCTAssertEqual(summary.tensors.count, checkpoint.tensors.count)
    }

    /// The blob is the tensor table, at the offsets the summary claims.
    func testTheBlobHoldsEveryTensorAtItsSummaryOffset() {
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let blob = checkpoint.blob
        let summary = checkpoint.summary()
        XCTAssertGreaterThan(blob.count, 0)
        for (index, metadata) in summary.tensors.enumerated() {
            let tensor = checkpoint.tensors[index]
            XCTAssertEqual(metadata.name, tensor.name)
            XCTAssertEqual(
                Int(metadata.offsetStart),
                checkpoint.layout[index].offset,
                tensor.name
            )
            XCTAssertEqual(
                metadata.offsetStart % 256,
                0,
                "\(tensor.name) is not aligned for a setBuffer offset"
            )
            let width = DeepSeekV41SyntheticCheckpoint.bytesPerValue(tensor.dtype)
            XCTAssertEqual(
                Int(metadata.offsetEnd - metadata.offsetStart),
                tensor.elementCount * width,
                tensor.name
            )
            // Round-trip the first value through the blob the way a kernel would.
            let at = Int(metadata.offsetStart)
            let read: Float
            if width == 2 {
                let bits = UInt16(blob[at]) | (UInt16(blob[at + 1]) << 8)
                read = Float(bitPattern: UInt32(bits) << 16)
            } else {
                let bits = UInt32(blob[at])
                    | (UInt32(blob[at + 1]) << 8)
                    | (UInt32(blob[at + 2]) << 16)
                    | (UInt32(blob[at + 3]) << 24)
                read = Float(bitPattern: bits)
            }
            XCTAssertEqual(read, tensor.values[0], "\(tensor.name) first value")
        }
    }

    /// A BF16 tensor must be written as the value the host reads back, or the
    /// parity comparison has a floor under it that nothing about the graph put
    /// there.
    func testTheHostReadsTheValuesTheDeviceWillReadBack() {
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let source = checkpoint.weightSource()
        for tensor in checkpoint.tensors where tensor.dtype == "BF16" {
            guard let matrix = source.matrix(
                tensor.role, scope: .backbone, layer: tensor.layer, expert: tensor.expert
            ) else { continue }
            for (index, value) in matrix.values.enumerated() {
                XCTAssertEqual(
                    value,
                    DeepSeekV41SyntheticCheckpoint.truncateToBF16(tensor.values[index]),
                    tensor.name
                )
            }
        }
    }
}

// MARK: - Device graph

final class DeepSeekV41DeviceGraphParityTests: XCTestCase {

    /// One position's decode relative to the largest logit. The device cache is
    /// FP16 and the host's is FP32, so a couple of parts in a thousand is the
    /// cache's own rounding; a stage bound to the wrong buffer moves the logits by
    /// O(1).
    private let decodeTolerance: Float = 2e-3

    private let checkpoint = DeepSeekV41SyntheticCheckpoint()

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

    private func makeHost(
        _ checkpoint: DeepSeekV41SyntheticCheckpoint
    ) -> DeepSeekV41CPUForward {
        DeepSeekV41CPUForward(
            config: checkpoint.config,
            plan: checkpoint.plan,
            weights: checkpoint.weightSource()
        )
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func worstDelta(_ device: [Float], _ expected: [Float]) -> Float {
        guard device.count == expected.count else { return .infinity }
        var worst: Float = 0
        for index in 0..<expected.count {
            worst = max(worst, abs(device[index] - expected[index]))
        }
        return worst
    }

    private func buffer(_ device: MTLDevice, _ count: Int) throws -> MTLBuffer {
        try XCTUnwrap(
            device.makeBuffer(
                length: max(16, count * MemoryLayout<Float>.stride),
                options: .storageModeShared
            )
        )
    }

    func testTheSyntheticRuntimeBuilds() throws {
        let runtime = try makeRuntime(checkpoint)
        print(runtime.describe())
        XCTAssertEqual(runtime.weights.problems, [])
    }

    // MARK: Whole model

    /// A walk over several positions, both backends.
    ///
    /// More positions than one ring holds (`window` is four), so the gather reads
    /// rows the ring has since rotated instead of re-reading one row.
    func testTheDeviceAndTheHostAgreeOverAWalk() throws {
        let runtime = try makeRuntime(checkpoint)
        let device = runtime.forward()
        let host = makeHost(checkpoint)
        let tokens = (0..<(checkpoint.shape.window + 3)).map { 5 + $0 }

        for (position, token) in tokens.enumerated() {
            _ = try device.forwardToken(token, position: position)
            let expected = host.forwardToken(token, position: position)
            let logits = floats(runtime.workspace.logits, count: checkpoint.config.vocabSize)
            XCTAssertEqual(logits.count, expected.count)
            let scale = expected.map(abs).max() ?? 1
            let delta = worstDelta(logits, expected)
            XCTAssertLessThan(
                delta / max(scale, 1e-6),
                decodeTolerance,
                "position \(position) is off by \(delta) against a \(scale) scale"
            )
        }
    }

    /// A block of one is a decode step: the same tokens, the same graph, and the
    /// same logits, wherever the block machinery differs from the walk's.
    func testABlockOfOneMatchesTheWalk() throws {
        let runtime = try makeRuntime(checkpoint)
        let chunk = try runtime.forward().prefillChunk([6])
        let walked = try makeRuntime(checkpoint).forward().forwardToken(6, position: 0)
        XCTAssertEqual(chunk.last.count, walked.count)
        XCTAssertLessThan(worstDelta(chunk.last, walked), 1e-5)
    }

    /// A prefilled chunk returns exactly the logits a walk returns.
    ///
    /// The prompt is longer than one block (the block is capped by the ring), so
    /// this crosses a block boundary and wraps the ring: the second block's
    /// addressing reads rows the first block's writes have since rotated.
    func testTheChunkSeamReturnsTheWalksLogits() throws {
        let runtime = try makeRuntime(checkpoint)
        let tokens = (0..<(checkpoint.shape.window * 2 - 1)).map { 3 + $0 }
        XCTAssertGreaterThan(tokens.count, checkpoint.shape.window)
        let chunk = try runtime.forward().prefillChunk(tokens)

        let walked = try makeRuntime(checkpoint).forward()
        var last: [Float] = []
        for (position, token) in tokens.enumerated() {
            last = try walked.forwardToken(token, position: position)
        }
        XCTAssertEqual(chunk.last.count, last.count)
        XCTAssertEqual(worstDelta(chunk.last, last), 0, accuracy: 1e-6)
    }

    // MARK: One layer, stage by stage

    /// One layer and one position, replayed here from the reference's own helpers,
    /// so a disagreement names the stage that produced it.
    ///
    /// The device's workspace reuses buffers between stages, so the intermediates
    /// are captured through `parityProbes` as the walk writes them.
    func testTheDeviceAndTheHostAgreeStageByStage() throws {
        var shape = DeepSeekV41SyntheticShape()
        shape.layers = 1
        let single = DeepSeekV41SyntheticCheckpoint(shape: shape)
        let runtime = try makeRuntime(single)
        let device = runtime.forward()
        let host = makeHost(single)
        let metal = try XCTUnwrap(MTLCreateSystemDefaultDevice())

        let mult = max(1, single.config.hcMult)
        let hidden = single.shape.hidden
        let token = 3
        let afterAttention = try buffer(metal, mult * hidden)
        device.parityProbes = DeepSeekV41ParityProbes(afterAttention: afterAttention)

        _ = try device.forwardToken(token, position: 0)
        let hostLogits = host.forwardToken(token, position: 0)
        let workspace = runtime.workspace
        let layer = single.plan[0]
        let attention = DeepSeekV41AttentionShape.make(config: single.config, layer: layer)

        // The host's replay of position 0, from the reference's own pieces.
        let embedding = host.embed(token)
        var streams = [[Float]](repeating: embedding, count: mult)
        let mix = DeepSeekV41HyperConnections.identityMix(copies: mult)
        let attentionCoefficients = host.hyperConnectionCoefficients(
            layer: layer.index, entry: .attention, streams: streams
        )
        let attentionInput = DeepSeekV41HyperConnections.pre(streams: streams, mix: mix)
        let normed = host.rmsNorm(
            attentionInput, role: .attnNorm, layer: layer.index, epsilon: single.config.rmsNormEps
        )

        var query = host.queryPath(normed, layer: layer.index, shape: attention)
        DeepSeekV41RoPE.rotateHeads(
            &query,
            heads: attention.numHeads,
            headDim: attention.qkHeadDim,
            sliceStart: attention.queryRopeStart,
            sliceLength: attention.queryRopeWidth,
            position: 0,
            frequencies: DeepSeekV41RoPE.queryFrequencies(for: attention)
        )
        var latent = host.latentPath(normed, layer: layer.index)
        DeepSeekV41RoPE.rotateSlice(
            &latent,
            start: attention.latentRopeStart,
            length: attention.latentRopeWidth,
            position: 0,
            frequencies: DeepSeekV41RoPE.latentFrequencies(for: attention)
        )

        let keys = host.attentionKeys(layer: layer, position: 0)
        var attended = DeepSeekV41AttentionReference.attend(
            query: query,
            keys: keys,
            shape: attention,
            sink: nil
        )
        if attention.outputCarriesRope {
            DeepSeekV41RoPE.rotateHeads(
                &attended,
                heads: attention.numHeads,
                headDim: attention.vHeadDim,
                sliceStart: attention.outputRopeStart,
                sliceLength: attention.latentRopeWidth,
                position: 0,
                frequencies: DeepSeekV41RoPE.latentFrequencies(for: attention),
                inverse: true
            )
        }
        let attnOut = host.outputPath(attended, layer: layer.index, shape: attention)
        streams = DeepSeekV41HyperConnections.post(
            output: attnOut,
            residual: streams,
            gains: attentionCoefficients.post,
            comb: attentionCoefficients.comb
        )
        let postAttention = streams
        let ffnCoefficients = host.hyperConnectionCoefficients(
            layer: layer.index, entry: .feedForward, streams: streams
        )
        let ffnInput = DeepSeekV41HyperConnections.pre(
            streams: streams, mix: attentionCoefficients.pre
        )
        let ffnNormed = host.rmsNorm(
            ffnInput, role: .ffnNorm, layer: layer.index, epsilon: single.config.rmsNormEps
        )
        let ffnOut = host.feedForward(ffnNormed, layer: layer.index)
        let afterFFN = DeepSeekV41HyperConnections.post(
            output: ffnOut,
            residual: streams,
            gains: ffnCoefficients.post,
            comb: ffnCoefficients.comb
        )
        let collapsed = DeepSeekV41HyperConnections.pre(
            streams: afterFFN, mix: ffnCoefficients.pre
        )

        func assertClose(
            _ label: String,
            _ observed: [Float],
            _ expected: [Float],
            _ tolerance: Float = 1e-3
        ) {
            XCTAssertEqual(observed.count, expected.count, label)
            let delta = worstDelta(observed, expected)
            let scale = expected.map(abs).max() ?? 1
            XCTAssertLessThan(
                delta, tolerance * Swift.max(scale, 1),
                "\(label) is off by \(delta) on a \(scale) scale"
            )
        }

        assertClose("latent", floats(workspace.latent, count: single.shape.latentWidth), latent, 1e-5)
        assertClose("query", floats(workspace.query, count: attention.queryWidth), query, 1e-5)
        assertClose(
            "attended",
            floats(workspace.attentionOut, count: attention.outputWidth),
            attended,
            1e-3
        )
        assertClose(
            "mHC attention coefficients (pre)",
            floats(workspace.hcAttentionPre, count: mult),
            attentionCoefficients.pre
        )
        assertClose(
            "mHC attention coefficients (post)",
            floats(workspace.hcAttentionPost, count: mult),
            attentionCoefficients.post
        )
        assertClose(
            "mHC attention coefficients (comb)",
            floats(workspace.hcAttentionComb, count: mult * mult),
            attentionCoefficients.comb
        )
        assertClose(
            "stack after attention",
            floats(afterAttention, count: mult * hidden),
            postAttention.flatMap { $0 }
        )
        assertClose(
            "mHC feed-forward coefficients (pre)",
            floats(workspace.hcFFNPre, count: mult),
            ffnCoefficients.pre
        )
        assertClose(
            "mHC feed-forward coefficients (post)",
            floats(workspace.hcFFNPost, count: mult),
            ffnCoefficients.post
        )
        assertClose(
            "mHC feed-forward coefficients (comb)",
            floats(workspace.hcFFNComb, count: mult * mult),
            ffnCoefficients.comb
        )
        assertClose(
            "feed-forward input (post ffn_norm)",
            floats(workspace.normed, count: hidden),
            ffnNormed
        )
        assertClose("feed-forward output", floats(workspace.moeOut, count: hidden), ffnOut)
        assertClose("collapsed stack", floats(workspace.hidden, count: hidden), collapsed)
        assertClose(
            "logits",
            floats(workspace.logits, count: single.config.vocabSize),
            hostLogits
        )
    }

    /// The mixture's own pieces, which the stage comparison above only covers in
    /// aggregate: the gate's logits and the experts it selected.
    func testTheDeviceRoutesLikeTheHost() throws {
        var shape = DeepSeekV41SyntheticShape()
        shape.layers = 1
        let single = DeepSeekV41SyntheticCheckpoint(shape: shape)
        let runtime = try makeRuntime(single)
        let device = runtime.forward()
        let host = makeHost(single)

        _ = try device.forwardToken(3, position: 0)
        _ = host.forwardToken(3, position: 0)
        let workspace = runtime.workspace
        let layer = single.plan[0]

        // The router reads the post-`ffn_norm` vector, which `finishBlock` leaves
        // in `normed` because it writes the final normalization into `output`.
        let ffnNormed = floats(workspace.normed, count: single.shape.hidden)
        guard let gate = host.weights.matrix(.routerWeight, layer: layer.index, expert: nil),
              let expected = host.route(ffnNormed, layer: layer.index) else {
            throw XCTSkip("the synthetic router is missing")
        }
        let logits = floats(workspace.routerLogits, count: single.config.numRoutedExperts)
        XCTAssertLessThan(worstDelta(logits, gate.apply(ffnNormed)), 1e-4)

        let indexPointer = workspace.expertIndices.contents().assumingMemoryBound(to: UInt32.self)
        let weightPointer = workspace.expertWeights.contents().assumingMemoryBound(to: Float.self)
        let count = single.config.numActivatedExperts
        let deviceExperts = (0..<count).map { Int(indexPointer[$0]) }
        let deviceWeights = (0..<count).map { weightPointer[$0] }
        XCTAssertEqual(deviceExperts, expected.expertIDs)
        for (rank, weight) in expected.weights.enumerated() {
            XCTAssertEqual(deviceWeights[rank], weight, accuracy: 1e-4, "weight \(rank)")
        }
    }

    // MARK: Coefficients

    /// The mHC coefficients at two scales, plus the stack they were projected from.
    ///
    /// The pinned-scale case is the one that matters for the scale's three
    /// sections: a zero-weight fixture leaves every logit at `sigmoid(0)` territory,
    /// where any reading of the sections agrees.
    func testTheHyperConnectionCoefficientsMatchAtEveryScale() throws {
        for (label, pinned) in [("seeded", false), ("pinned", true)] {
            var shape = DeepSeekV41SyntheticShape()
            shape.layers = 1
            if pinned {
                shape.constantRoles = [
                    .hcAttnScale: 1, .hcAttnBase: 0, .hcFfnScale: 1, .hcFfnBase: 0
                ]
            }
            let single = DeepSeekV41SyntheticCheckpoint(shape: shape)
            let runtime = try makeRuntime(single)
            let device = runtime.forward()
            let host = makeHost(single)
            let mult = max(1, single.config.hcMult)
            let metal = try XCTUnwrap(MTLCreateSystemDefaultDevice())

            let logitsProbe = try buffer(metal, (2 + mult) * mult)
            let streamProbe = try buffer(metal, mult * single.shape.hidden)
            device.parityProbes = DeepSeekV41ParityProbes(
                beforeAttention: streamProbe,
                attentionLogits: logitsProbe
            )

            let token = 3
            _ = try device.forwardToken(token, position: 0)
            _ = host.forwardToken(token, position: 0)

            let stack = [[Float]](repeating: host.embed(token), count: mult)
            let expected = host.hyperConnectionCoefficients(
                layer: 0, entry: .attention, streams: stack
            )
            let workspace = runtime.workspace
            let flatten = stack.flatMap { $0 }
            XCTAssertLessThan(
                worstDelta(floats(streamProbe, count: flatten.count), flatten),
                1e-6,
                "\(label): the projected stack"
            )
            XCTAssertLessThan(
                worstDelta(floats(logitsProbe, count: (2 + mult) * mult), expected.logits),
                1e-4,
                "\(label): the projection's logits"
            )
            XCTAssertLessThan(
                worstDelta(floats(workspace.hcAttentionPre, count: mult), expected.pre),
                1e-6,
                "\(label): pre"
            )
            XCTAssertLessThan(
                worstDelta(floats(workspace.hcAttentionPost, count: mult), expected.post),
                1e-6,
                "\(label): post"
            )
            XCTAssertLessThan(
                worstDelta(floats(workspace.hcAttentionComb, count: mult * mult), expected.comb),
                1e-6,
                "\(label): comb"
            )
        }
    }

    // MARK: Kernel contracts

    /// The projection kernel, dispatched exactly as the executor dispatches it.
    ///
    /// The executor binds the whole shard and passes the tensor's BYTE offset,
    /// which is a contract a kernel can get wrong without any test noticing: a
    /// `float*` advanced by a byte count reads four times too far, and binding the
    /// weight at offset 0 — all a dedicated weight buffer can do — hides it.
    func testTheHyperConnectionProjectionReadsItsWeightsFromAByteOffset() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "dsv41_hc_project_f32"),
              let project = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device or kernel")
        }
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let mult = max(1, checkpoint.config.hcMult)
        let mixCount = (2 + mult) * mult
        let hidden = checkpoint.shape.hidden
        let width = mult * hidden

        guard let index = checkpoint.tensors.firstIndex(where: { $0.name == "layers.0.hc_attn_fn" }),
              let shard = checkpoint.shardBuffer(device: device),
              let logits = device.makeBuffer(
                  length: mixCount * MemoryLayout<Float>.stride, options: .storageModeShared
              ) else {
            throw XCTSkip("the synthetic shard could not be mapped")
        }
        let byteOffset = checkpoint.layout[index].offset
        XCTAssertGreaterThan(byteOffset, 0)

        let table = checkpoint.tensors.first { $0.name == "embed.weight" }
        XCTAssertEqual(table?.shape.first, checkpoint.config.vocabSize)
        let embedding = table?.values ?? []
        let rowStart = min(7, checkpoint.config.vocabSize - 1) * hidden
        let row = Array(embedding[rowStart..<(rowStart + hidden)])
        let stack = [[Float]](repeating: row, count: mult).flatMap { $0 }
        guard let streamBuffer = device.makeBuffer(
                  bytes: stack,
                  length: width * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw XCTSkip("no stream buffer")
        }

        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(project)
        encoder.setBuffer(shard, offset: 0, index: 0)
        encoder.setBuffer(streamBuffer, offset: 0, index: 1)
        encoder.setBuffer(logits, offset: 0, index: 2)
        var weightOffset = UInt64(byteOffset)
        var epsilon: Float = checkpoint.config.rmsNormEps
        var widthDimension = UInt32(width)
        var outDimension = UInt32(mixCount)
        // One slot: this case pins the BYTE OFFSET, not the slot grid.
        var slotCount: UInt32 = 1
        encoder.setBytes(&weightOffset, length: 8, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.setBytes(&widthDimension, length: 4, index: 5)
        encoder.setBytes(&outDimension, length: 4, index: 6)
        encoder.setBytes(&slotCount, length: 4, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: mixCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let tensor = checkpoint.tensors[index]
        let expected = DeepSeekV41HyperConnections.coefficients(
            streams: [[Float]](repeating: row, count: mult),
            projection: DeepSeekV41Matrix(
                rows: mixCount, columns: width, values: tensor.values
            ),
            scale: [0, 0, 0],
            base: [Float](repeating: 0, count: mixCount),
            normalizationEpsilon: checkpoint.config.rmsNormEps,
            epsilon: 1e-6,
            iterations: 0
        ).logits
        let pointer = logits.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<mixCount {
            XCTAssertEqual(
                pointer[row], expected[row], accuracy: 1e-5,
                "row \(row) read from byte offset \(byteOffset)"
            )
        }
    }

    /// The mHC coefficient pair dispatched over a BLOCK of slots.
    ///
    /// `dsv41_hc_project_f32` takes its slot from `thread_position_in_grid.y` and
    /// `dsv41_hc_split_f32` from `.x`, so a block of positions shares one dispatch.
    /// The failure mode is the one any batched kernel has: a slot that reads its
    /// neighbour's stack looks like a plausible model for one position and is wrong
    /// everywhere else. Two slots with different stacks pin it.
    func testTheHyperConnectionCoefficientsDispatchOneSlotPerPosition() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary(),
              let projectFunction = library.makeFunction(name: "dsv41_hc_project_f32"),
              let splitFunction = library.makeFunction(name: "dsv41_hc_split_f32"),
              let project = try? device.makeComputePipelineState(function: projectFunction),
              let split = try? device.makeComputePipelineState(function: splitFunction),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device or kernel")
        }
        let checkpoint = DeepSeekV41SyntheticCheckpoint()
        let mult = max(1, checkpoint.config.hcMult)
        let mixCount = (2 + mult) * mult
        let hidden = checkpoint.shape.hidden
        let width = mult * hidden
        let slots = 2

        guard let index = checkpoint.tensors.firstIndex(where: { $0.name == "layers.0.hc_attn_fn" }),
              let shard = checkpoint.shardBuffer(device: device),
              let logits = device.makeBuffer(
                  length: slots * mixCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let pre = device.makeBuffer(
                  length: slots * mult * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let post = device.makeBuffer(
                  length: slots * mult * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let comb = device.makeBuffer(
                  length: slots * mult * mult * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw XCTSkip("the synthetic shard could not be mapped")
        }
        let tensor = checkpoint.tensors[index]
        let byteOffset = checkpoint.layout[index].offset

        // Two stacks that differ in every element, so a slot reading its neighbour
        // cannot pass.
        let table = checkpoint.tensors.first { $0.name == "embed.weight" }
        let embedding = table?.values ?? []
        var stacks: [[Float]] = []
        for slot in 0..<slots {
            let start = min(3 + slot, checkpoint.config.vocabSize - 1) * hidden
            let row = Array(embedding[start..<(start + hidden)])
            stacks.append([[Float]](repeating: row, count: mult).flatMap { $0 })
        }
        guard let streamBuffer = device.makeBuffer(
                  bytes: stacks.flatMap { $0 },
                  length: slots * width * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw XCTSkip("no stream buffer")
        }

        let scale: [Float] = [0.5, -0.25, 1.5]
        let base = (0..<mixCount).map { Float($0) * 0.125 - 0.5 }
        guard let scaleBuffer = device.makeBuffer(
                  bytes: scale, length: 3 * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let baseBuffer = device.makeBuffer(
                  bytes: base, length: mixCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw XCTSkip("no coefficient buffers")
        }

        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(project)
        encoder.setBuffer(shard, offset: 0, index: 0)
        encoder.setBuffer(streamBuffer, offset: 0, index: 1)
        encoder.setBuffer(logits, offset: 0, index: 2)
        var weightOffset = UInt64(byteOffset)
        var epsilon: Float = checkpoint.config.rmsNormEps
        var widthDimension = UInt32(width)
        var outDimension = UInt32(mixCount)
        var slotCount = UInt32(slots)
        encoder.setBytes(&weightOffset, length: 8, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.setBytes(&widthDimension, length: 4, index: 5)
        encoder.setBytes(&outDimension, length: 4, index: 6)
        encoder.setBytes(&slotCount, length: 4, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: mixCount, height: slots, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)

        encoder.setComputePipelineState(split)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(baseBuffer, offset: 0, index: 2)
        encoder.setBuffer(pre, offset: 0, index: 3)
        encoder.setBuffer(post, offset: 0, index: 4)
        encoder.setBuffer(comb, offset: 0, index: 5)
        var copies = UInt32(mult)
        var iterations = UInt32(3)
        var sinkhornEpsilon: Float = 1e-6
        encoder.setBytes(&copies, length: 4, index: 6)
        encoder.setBytes(&iterations, length: 4, index: 7)
        encoder.setBytes(&sinkhornEpsilon, length: 4, index: 8)
        encoder.setBytes(&slotCount, length: 4, index: 9)
        encoder.dispatchThreads(
            MTLSize(width: slots, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: slots, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let matrix = DeepSeekV41Matrix(rows: mixCount, columns: width, values: tensor.values)
        let logitsPointer = logits.contents().assumingMemoryBound(to: Float.self)
        let prePointer = pre.contents().assumingMemoryBound(to: Float.self)
        let postPointer = post.contents().assumingMemoryBound(to: Float.self)
        let combPointer = comb.contents().assumingMemoryBound(to: Float.self)
        for slot in 0..<slots {
            let stack = [[Float]](
                (0..<mult).map { Array(stacks[slot][($0 * hidden)..<(($0 + 1) * hidden)]) }
            )
            let expected = DeepSeekV41HyperConnections.coefficients(
                streams: stack,
                projection: matrix,
                scale: scale,
                base: base,
                normalizationEpsilon: checkpoint.config.rmsNormEps,
                epsilon: sinkhornEpsilon,
                iterations: 3
            )
            for row in 0..<mixCount {
                XCTAssertEqual(
                    logitsPointer[slot * mixCount + row], expected.logits[row],
                    accuracy: 1e-5,
                    "slot \(slot): logit \(row) came from another slot's stack"
                )
            }
            for copy in 0..<mult {
                XCTAssertEqual(
                    prePointer[slot * mult + copy], expected.pre[copy], accuracy: 1e-6,
                    "slot \(slot): pre \(copy)"
                )
                XCTAssertEqual(
                    postPointer[slot * mult + copy], expected.post[copy], accuracy: 1e-6,
                    "slot \(slot): post \(copy)"
                )
            }
            for entry in 0..<(mult * mult) {
                XCTAssertEqual(
                    combPointer[slot * mult * mult + entry], expected.comb[entry],
                    accuracy: 1e-6,
                    "slot \(slot): comb \(entry)"
                )
            }
        }
    }

    /// A GEMV dispatched over a block of rows.
    ///
    /// The kernels index the input and the output by `thread_position_in_grid.y`,
    /// which is what makes a batched projection a grid height. A block whose rows
    /// each got the first row's answer would look like a plausible model and be
    /// wrong everywhere after the first token, so this pins the contract.
    func testTheGemvDispatchesOneRowPerToken() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "bf16_gemv"),
              let gemv = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device or kernel")
        }
        let rows = 3
        let inDim = 16
        let outDim = 5
        let values = (0..<(outDim * inDim)).map { index in
            DeepSeekV41SyntheticCheckpoint.truncateToBF16(Float((index % 9) - 4) / 8)
        }
        let input = (0..<(rows * inDim)).map { index in Float((index % 13) - 6) / 8 }
        let halves = values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }

        guard let weight = halves.withUnsafeBytes({ raw -> MTLBuffer? in
                  device.makeBuffer(
                      bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared
                  )
              }),
              let inputBuffer = input.withUnsafeBytes({ raw -> MTLBuffer? in
                  device.makeBuffer(
                      bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared
                  )
              }),
              let output = device.makeBuffer(
                  length: rows * outDim * MemoryLayout<Float>.stride, options: .storageModeShared
              ),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw XCTSkip("could not allocate the batched projection's buffers")
        }
        encoder.setComputePipelineState(gemv)
        encoder.setBuffer(weight, offset: 0, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var weightOffset = UInt64(0)
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)
        encoder.setBytes(&weightOffset, length: 8, index: 3)
        encoder.setBytes(&inDimension, length: 4, index: 4)
        encoder.setBytes(&outDimension, length: 4, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: outDim, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let matrix = DeepSeekV41Matrix(rows: outDim, columns: inDim, values: values)
        let pointer = output.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<rows {
            let expected = matrix.apply(Array(input[(row * inDim)..<((row + 1) * inDim)]))
            for column in 0..<outDim {
                XCTAssertEqual(
                    pointer[row * outDim + column], expected[column], accuracy: 1e-5,
                    "token \(row) row \(column)"
                )
            }
        }
    }

    /// A grouped output projection, dispatched the way the executor dispatches it.
    ///
    /// `attn.wo_a` is `groups` blocks stacked along the ROWS, each as wide as one
    /// group's slice of the attention output. Both dimensions are totals at the
    /// call site; the per-group counts are quotients of them, and dividing the row
    /// count a second time reads a fraction of each block and writes it at a
    /// fraction of the stride.
    func testTheGroupedOutputProjectionReadsOneColumnSlicePerGroup() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "bf16_gemv"),
              let gemv = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("No Metal device or kernel")
        }
        let groups = 2
        let perGroupIn = 8
        let perGroupOut = 3
        let inDim = groups * perGroupIn
        let outDim = groups * perGroupOut

        let values = (0..<(outDim * perGroupIn)).map { index in
            DeepSeekV41SyntheticCheckpoint.truncateToBF16(Float((index % 11) - 5) / 8)
        }
        let input = (0..<inDim).map { index in Float((index % 7) - 3) / 4 }
        // The kernel reads BF16 halves, so the weights go in as halves; the host
        // side keeps the same values, which are exact in BF16.
        let halves = values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        guard let weightBytes = halves.withUnsafeBytes({ raw -> MTLBuffer? in
                  device.makeBuffer(
                      bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared
                  )
              }),
              let inputBuffer = input.withUnsafeBytes({ raw -> MTLBuffer? in
                  device.makeBuffer(
                      bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared
                  )
              }),
              let output = device.makeBuffer(
                  length: outDim * MemoryLayout<Float>.stride, options: .storageModeShared
              ),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw XCTSkip("could not allocate the grouped projection's buffers")
        }
        let valueBytes = 2
        for group in 0..<groups {
            encoder.setComputePipelineState(gemv)
            encoder.setBuffer(weightBytes, offset: 0, index: 0)
            encoder.setBuffer(
                inputBuffer,
                offset: group * perGroupIn * MemoryLayout<Float>.stride,
                index: 1
            )
            encoder.setBuffer(
                output,
                offset: group * perGroupOut * MemoryLayout<Float>.stride,
                index: 2
            )
            var weightOffset = UInt64(group * perGroupOut * perGroupIn * valueBytes)
            var inDimension = UInt32(perGroupIn)
            var outDimension = UInt32(perGroupOut)
            encoder.setBytes(&weightOffset, length: 8, index: 3)
            encoder.setBytes(&inDimension, length: 4, index: 4)
            encoder.setBytes(&outDimension, length: 4, index: 5)
            encoder.dispatchThreads(
                MTLSize(width: perGroupOut, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
            if group + 1 < groups { encoder.memoryBarrier(scope: .buffers) }
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let expected = DeepSeekV41Matrix(
            rows: outDim, columns: perGroupIn, values: values
        ).applyBlockDiagonal(input, groups: groups)
        let pointer = output.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<outDim {
            XCTAssertEqual(pointer[row], expected[row], accuracy: 1e-4, "row \(row)")
        }
    }
}
