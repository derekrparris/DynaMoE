//
//  DeepSeekV41VisionEncodeDevice.swift
//  DynaMoE
//
//  Phase: DeepSeek-V4.1-Flash port -- the vision tower as one device-resident pass.
//
//  Slices #41 and #42 measured the tower's two expensive halves on the GPU: the
//  bidirectional attention runs in 6.29 s at the largest admissible image against
//  the host's 31.75 s, and the four block projections run in 0.60 s against 6.72 s.
//  Both were STAGE runners, and on their own they buy nothing: a `w1` product at
//  that size is 195 MB, so a host<->device round trip per projection costs more
//  than the host GEMM it replaces. The prize only exists if the token matrix stays
//  on the GPU across all 32 blocks.
//
//  This file is that pass. `encode` derives the patch grid exactly as
//  `DeepSeekV41VisionForward.encode` does, uploads the tower's weights once, and
//  runs the whole tower in ONE command buffer: patch embed, then per block the two
//  norms, the four projections, the split-and-rotate, the attention and the gated
//  MLP, then the final norm, the pixel-unshuffle and the aligner. The host path
//  stays untouched as the oracle (`encodeOnHost`), and the parity tests check the
//  two against each other on a tower small enough to run both.
//
//  Three rules the host path defines and this one has to keep:
//
//    * the tower's norm epsilon is 1e-6, not the text tower's 1e-20;
//    * the aligner's activation is `F.gelu`'s exact erf form, not the tanh
//      approximation;
//    * the pixel-unshuffle pads right and bottom and is channel-major inside a
//      cell, so a transposed port puts every value in the wrong slot.
//
//  One encode at a time: the class caches the uploaded weights and a token-sized
//  workspace, so it is not thread-safe. The engine encodes a turn's attachments
//  sequentially, which is the only caller.
//

import Foundation
import Metal

nonisolated public final class DeepSeekV41VisionEncodeDevice {

    /// One encode's result, with the dispatch's own GPU time beside it.
    public struct Result {
        /// One row per aligner grid cell, `textHiddenSize` wide, in reading order.
        public let rows: [[Float]]
        /// `gpuEndTime - gpuStartTime` for the single command buffer, excluding the
        /// readback and every host-side upload.
        public let gpuSeconds: Double
    }

    public let spec: DeepSeekV41VisionSpec
    public let weights: DeepSeekV41WeightSource
    public let textHiddenSize: Int
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    private let linear: DeepSeekV41VisionLinearDevice
    private let attention: DeepSeekV41VisionAttentionDevice
    private let normPipeline: MTLComputePipelineState
    private let geluPipeline: MTLComputePipelineState
    private let siluMulPipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    private let unshufflePipeline: MTLComputePipelineState

    /// The tower's weights, uploaded once. A device encode reads each tensor a
    /// single time (the host path re-reads them per encode), so this is the whole
    /// 970 MB of BF16 decoded to FP32 once per tower rather than once per picture.
    private var weightBuffers: [String: MTLBuffer] = [:]
    private var gainBuffers: [String: MTLBuffer] = [:]
    private var biasBuffers: [String: MTLBuffer] = [:]
    private var zeroBiases: [Int: MTLBuffer] = [:]

    /// The token-sized buffers, reused while the patch grid does not change.
    private var workspace: Workspace?

    private final class Workspace {
        let rows: Int
        let columns: Int
        let tokens: Int
        let hidden: Int
        let intermediate: Int
        let textHidden: Int
        let ratio: Int
        let cellCount: Int
        let cellVolume: Int
        let patchWidth: Int

        let patches: MTLBuffer
        let cosine: MTLBuffer
        let sine: MTLBuffer
        let x: MTLBuffer
        let normed: MTLBuffer
        let qkv: MTLBuffer
        let attended: MTLBuffer
        let projected: MTLBuffer
        let gateUp: MTLBuffer
        let activated: MTLBuffer
        let mlp: MTLBuffer
        let cells: MTLBuffer
        let alignerHidden: MTLBuffer
        let alignerActivated: MTLBuffer
        let output: MTLBuffer
        let planes: DeepSeekV41VisionAttentionDevice.SplitPlanes

        init(
            rows: Int, columns: Int, tokens: Int, hidden: Int, intermediate: Int,
            textHidden: Int, ratio: Int, cellCount: Int, cellVolume: Int, patchWidth: Int,
            patches: MTLBuffer, cosine: MTLBuffer, sine: MTLBuffer,
            x: MTLBuffer, normed: MTLBuffer, qkv: MTLBuffer, attended: MTLBuffer,
            projected: MTLBuffer, gateUp: MTLBuffer, activated: MTLBuffer, mlp: MTLBuffer,
            cells: MTLBuffer, alignerHidden: MTLBuffer, alignerActivated: MTLBuffer,
            output: MTLBuffer, planes: DeepSeekV41VisionAttentionDevice.SplitPlanes
        ) {
            self.rows = rows
            self.columns = columns
            self.tokens = tokens
            self.hidden = hidden
            self.intermediate = intermediate
            self.textHidden = textHidden
            self.ratio = ratio
            self.cellCount = cellCount
            self.cellVolume = cellVolume
            self.patchWidth = patchWidth
            self.patches = patches
            self.cosine = cosine
            self.sine = sine
            self.x = x
            self.normed = normed
            self.qkv = qkv
            self.attended = attended
            self.projected = projected
            self.gateUp = gateUp
            self.activated = activated
            self.mlp = mlp
            self.cells = cells
            self.alignerHidden = alignerHidden
            self.alignerActivated = alignerActivated
            self.output = output
            self.planes = planes
        }
    }

    public init(
        spec: DeepSeekV41VisionSpec,
        weights: DeepSeekV41WeightSource,
        textHiddenSize: Int,
        device: MTLDevice? = nil,
        library: MTLLibrary? = nil
    ) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no Metal device")
        }
        guard let library = library ?? device.makeDefaultLibrary() else {
            throw DeepSeekV41VisionDeviceError.unavailable("the default Metal library is missing")
        }
        guard let queue = device.makeCommandQueue() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no Metal command queue")
        }
        let heads = max(1, spec.numAttentionHeads)
        guard spec.hiddenSize > 0, spec.hiddenSize % heads == 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "hidden size \(spec.hiddenSize) is not divisible by \(heads) attention heads"
            )
        }
        guard spec.intermediateSize > 0, textHiddenSize > 0, spec.patchSize > 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "the tower needs a positive patch size, intermediate size and text width"
            )
        }
        func make(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw DeepSeekV41VisionDeviceError.unavailable("kernel \(name) is missing")
            }
            return try device.makeComputePipelineState(function: function)
        }
        self.spec = spec
        self.weights = weights
        self.textHiddenSize = textHiddenSize
        self.device = device
        self.queue = queue
        self.linear = try DeepSeekV41VisionLinearDevice(device: device, library: library)
        self.attention = try DeepSeekV41VisionAttentionDevice(device: device, library: library)
        self.normPipeline = try make("dsv41_rmsnorm_f32_gain")
        self.geluPipeline = try make("dsv41_vision_gelu_f32")
        self.siluMulPipeline = try make("dsv41_vision_silu_mul_f32")
        self.addPipeline = try make("vector_add_f32")
        self.unshufflePipeline = try make("dsv41_vision_unshuffle_f32")
    }

    /// The tower over a whole patch grid, one device pass.
    public func encode(patches: [[Float]], rows: Int, columns: Int) throws -> [[Float]] {
        try run(patches: patches, rows: rows, columns: columns).rows
    }

    /// The same encode, reporting the command buffer's own GPU time.
    ///
    /// The time is read between `waitUntilCompleted` and the readback, so neither
    /// the weight upload (once per tower) nor the `[[Float]]` materialization (once
    /// per encode) is part of it.
    public func run(patches: [[Float]], rows: Int, columns: Int) throws -> Result {
        guard rows > 0, columns > 0, patches.count >= rows * columns else {
            throw DeepSeekV41VisionForwardError.emptyImage
        }
        let heads = max(1, spec.numAttentionHeads)
        let hidden = spec.hiddenSize
        let headDim = hidden / heads
        let ropeDim = headDim / 2
        let ratio = max(1, spec.downsampleRatio)
        let patchWidth = 3 * spec.patchSize * spec.patchSize
        let tokenCount = rows * columns
        let gridRows = (rows + ratio - 1) / ratio
        let gridColumns = (columns + ratio - 1) / ratio
        let cellCount = gridRows * gridColumns
        let cellVolume = hidden * ratio * ratio

        // A ragged patch row would make a projection read the next row's values for
        // its missing tail, so it is refused rather than padded.
        for (index, patch) in patches.prefix(tokenCount).enumerated() where patch.count != patchWidth {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "patch row \(index) has \(patch.count) values, expected \(patchWidth)"
            )
        }

        let workspace = try workspaceFor(rows: rows, columns: columns)
        upload(patches: patches, into: workspace)
        try prepareWeights(for: workspace)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no command buffer")
        }

        // Patch embedding: `proj(x.flatten(1))`.
        try linear.dispatchLinear(
            on: encoder,
            inputs: workspace.patches,
            weights: try weightBuffer(.visionPatchEmbed, layer: nil, rows: hidden, columns: patchWidth),
            bias: try biasBuffer(.visionPatchEmbedBias, layer: nil, outputs: hidden),
            output: workspace.x,
            rows: tokenCount, outputs: hidden, depth: patchWidth
        )
        barrier(encoder)

        for block in 0..<max(0, spec.numHiddenLayers) {
            try dispatchNorm(
                on: encoder, input: workspace.x, output: workspace.normed,
                gain: try gainBuffer(.visionBlockNorm1, layer: block, dimension: hidden),
                width: hidden, rows: tokenCount
            )
            try linear.dispatchLinear(
                on: encoder,
                inputs: workspace.normed,
                weights: try weightBuffer(.visionBlockWqkv, layer: block, rows: 3 * hidden, columns: hidden),
                bias: try biasBuffer(.visionBlockWqkvBias, layer: block, outputs: 3 * hidden),
                output: workspace.qkv,
                rows: tokenCount, outputs: 3 * hidden, depth: hidden
            )
            barrier(encoder)
            try attention.dispatchSplitAndRotate(
                on: encoder,
                qkv: workspace.qkv, cosine: workspace.cosine, sine: workspace.sine,
                planes: workspace.planes, ropeDim: ropeDim
            )
            barrier(encoder)
            try attention.dispatchAttention(
                on: encoder, planes: workspace.planes, into: workspace.attended
            )
            barrier(encoder)
            try linear.dispatchLinear(
                on: encoder,
                inputs: workspace.attended,
                weights: try weightBuffer(.visionBlockWo, layer: block, rows: hidden, columns: hidden),
                bias: try biasBuffer(.visionBlockWoBias, layer: block, outputs: hidden),
                output: workspace.projected,
                rows: tokenCount, outputs: hidden, depth: hidden
            )
            barrier(encoder)
            try dispatchAdd(
                on: encoder, first: workspace.x, second: workspace.projected,
                output: workspace.x, width: hidden, rows: tokenCount
            )

            try dispatchNorm(
                on: encoder, input: workspace.x, output: workspace.normed,
                gain: try gainBuffer(.visionBlockNorm2, layer: block, dimension: hidden),
                width: hidden, rows: tokenCount
            )
            try linear.dispatchLinear(
                on: encoder,
                inputs: workspace.normed,
                weights: try weightBuffer(
                    .visionBlockW1, layer: block, rows: 2 * spec.intermediateSize, columns: hidden
                ),
                bias: try zeroBias(spec.intermediateSize * 2),
                output: workspace.gateUp,
                rows: tokenCount, outputs: 2 * spec.intermediateSize, depth: hidden
            )
            barrier(encoder)
            try dispatchSiluMul(
                on: encoder, gateAndUp: workspace.gateUp, output: workspace.activated,
                intermediate: spec.intermediateSize, tokens: tokenCount
            )
            try linear.dispatchLinear(
                on: encoder,
                inputs: workspace.activated,
                weights: try weightBuffer(
                    .visionBlockW2, layer: block, rows: hidden, columns: spec.intermediateSize
                ),
                bias: try zeroBias(hidden),
                output: workspace.mlp,
                rows: tokenCount, outputs: hidden, depth: spec.intermediateSize
            )
            barrier(encoder)
            try dispatchAdd(
                on: encoder, first: workspace.x, second: workspace.mlp,
                output: workspace.x, width: hidden, rows: tokenCount
            )
        }

        // `ViT.forward` closes with a norm.
        try dispatchNorm(
            on: encoder, input: workspace.x, output: workspace.normed,
            gain: try gainBuffer(.visionFinalNorm, layer: nil, dimension: hidden),
            width: hidden, rows: tokenCount
        )
        try dispatchUnshuffle(
            on: encoder, patches: workspace.normed, cells: workspace.cells,
            patchRows: rows, patchColumns: columns, width: hidden, ratio: ratio,
            cellCount: cellCount, cellVolume: cellVolume
        )
        let unshuffledWidth = cellVolume
        try linear.dispatchLinear(
            on: encoder,
            inputs: workspace.cells,
            weights: try weightBuffer(
                .visionAlignerW1, layer: nil, rows: textHiddenSize, columns: unshuffledWidth
            ),
            bias: try biasBuffer(.visionAlignerW1Bias, layer: nil, outputs: textHiddenSize),
            output: workspace.alignerHidden,
            rows: cellCount, outputs: textHiddenSize, depth: unshuffledWidth
        )
        barrier(encoder)
        try dispatchGelu(
            on: encoder, input: workspace.alignerHidden,
            output: workspace.alignerActivated, count: cellCount * textHiddenSize
        )
        try linear.dispatchLinear(
            on: encoder,
            inputs: workspace.alignerActivated,
            weights: try weightBuffer(
                .visionAlignerW2, layer: nil, rows: textHiddenSize, columns: textHiddenSize
            ),
            bias: try biasBuffer(.visionAlignerW2Bias, layer: nil, outputs: textHiddenSize),
            output: workspace.output,
            rows: cellCount, outputs: textHiddenSize, depth: textHiddenSize
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DeepSeekV41VisionDeviceError.unavailable("vision encode failed: \(error)")
        }
        let seconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        let rowsOut = linear.readRows(
            workspace.output, rows: cellCount, columns: textHiddenSize
        )
        return Result(rows: rowsOut, gpuSeconds: seconds)
    }

    // MARK: Workspace

    private func workspaceFor(rows: Int, columns: Int) throws -> Workspace {
        if let workspace, workspace.rows == rows, workspace.columns == columns {
            return workspace
        }
        let heads = max(1, spec.numAttentionHeads)
        let hidden = spec.hiddenSize
        let ratio = max(1, spec.downsampleRatio)
        let intermediate = spec.intermediateSize
        let patchWidth = 3 * spec.patchSize * spec.patchSize
        let tokenCount = rows * columns
        let gridRows = (rows + ratio - 1) / ratio
        let gridColumns = (columns + ratio - 1) / ratio
        let cellCount = gridRows * gridColumns
        let cellVolume = hidden * ratio * ratio
        let headDim = hidden / heads
        let ropeDim = headDim / 2

        let rotary = DeepSeekV41VisionForward.rotaryTables(
            rows: rows, columns: columns, ropeDim: ropeDim, theta: Float(spec.ropeTheta)
        )
        guard rotary.cosine.count == tokenCount, rotary.sine.count == tokenCount else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "the rotation built \(rotary.cosine.count) angle rows for \(tokenCount) tokens"
            )
        }
        let flatCosine = rotary.cosine.flatMap { $0 }
        let flatSine = rotary.sine.flatMap { $0 }

        let workspace = Workspace(
            rows: rows, columns: columns, tokens: tokenCount, hidden: hidden,
            intermediate: intermediate, textHidden: textHiddenSize, ratio: ratio,
            cellCount: cellCount, cellVolume: cellVolume, patchWidth: patchWidth,
            patches: try emptyBuffer(tokenCount * patchWidth, role: "patches"),
            cosine: try linear.makeBuffer(flatCosine),
            sine: try linear.makeBuffer(flatSine),
            x: try emptyBuffer(tokenCount * hidden, role: "x"),
            normed: try emptyBuffer(tokenCount * hidden, role: "normed"),
            qkv: try emptyBuffer(tokenCount * 3 * hidden, role: "qkv"),
            attended: try emptyBuffer(tokenCount * hidden, role: "attended"),
            projected: try emptyBuffer(tokenCount * hidden, role: "projected"),
            gateUp: try emptyBuffer(tokenCount * 2 * intermediate, role: "gateUp"),
            activated: try emptyBuffer(tokenCount * intermediate, role: "activated"),
            mlp: try emptyBuffer(tokenCount * hidden, role: "mlp"),
            cells: try emptyBuffer(cellCount * cellVolume, role: "cells"),
            alignerHidden: try emptyBuffer(cellCount * textHiddenSize, role: "alignerHidden"),
            alignerActivated: try emptyBuffer(cellCount * textHiddenSize, role: "alignerActivated"),
            output: try emptyBuffer(cellCount * textHiddenSize, role: "output"),
            planes: try DeepSeekV41VisionAttentionDevice.allocatePlanes(
                device: device, tokens: tokenCount, heads: heads, headDim: headDim
            )
        )
        self.workspace = workspace
        return workspace
    }

    private func emptyBuffer(_ count: Int, role: String) throws -> MTLBuffer {
        let length = max(count, 1) * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw DeepSeekV41VisionDeviceError.unavailable(
                "could not allocate the \(role) buffer (\(length) bytes)"
            )
        }
        return buffer
    }

    private func upload(patches: [[Float]], into workspace: Workspace) {
        let flat = patches.prefix(workspace.tokens).flatMap { $0 }
        let base = workspace.patches.contents()
        flat.withUnsafeBufferPointer { source in
            guard let sourceBase = source.baseAddress else { return }
            memcpy(
                base, sourceBase,
                min(flat.count, workspace.tokens * workspace.patchWidth)
                    * MemoryLayout<Float>.stride
            )
        }
    }

    // MARK: Weights

    private func weightKey(_ role: DeepSeekV41TensorRole, layer: Int?) -> String {
        "\(role.rawValue)|\(layer.map(String.init) ?? "-")"
    }

    private func weightBuffer(
        _ role: DeepSeekV41TensorRole,
        layer: Int?,
        rows: Int,
        columns: Int
    ) throws -> MTLBuffer {
        let key = weightKey(role, layer: layer)
        if let cached = weightBuffers[key] { return cached }
        guard let matrix = weights.matrix(role, layer: layer, expert: nil) else {
            throw DeepSeekV41VisionForwardError.missingTensor(role, layer)
        }
        guard matrix.rows == rows, matrix.columns == columns else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "expected \(rows)x\(columns), got \(matrix.rows)x\(matrix.columns)"
            )
        }
        guard matrix.values.count >= rows * columns else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "\(role.rawValue) has \(matrix.values.count) values for \(rows)x\(columns)"
            )
        }
        let buffer = try linear.makeBuffer(matrix.values)
        weightBuffers[key] = buffer
        return buffer
    }

    private func gainBuffer(
        _ role: DeepSeekV41TensorRole,
        layer: Int?,
        dimension: Int
    ) throws -> MTLBuffer {
        let key = weightKey(role, layer: layer)
        if let cached = gainBuffers[key] { return cached }
        guard let gain = weights.vector(role, layer: layer), gain.count == dimension else {
            throw DeepSeekV41VisionForwardError.missingTensor(role, layer)
        }
        let buffer = try linear.makeBuffer(gain)
        gainBuffers[key] = buffer
        return buffer
    }

    private func biasBuffer(
        _ role: DeepSeekV41TensorRole,
        layer: Int?,
        outputs: Int
    ) throws -> MTLBuffer {
        let key = weightKey(role, layer: layer)
        if let cached = biasBuffers[key] { return cached }
        let buffer: MTLBuffer
        if let bias = weights.vector(role, layer: layer) {
            guard bias.count == outputs else {
                throw DeepSeekV41VisionForwardError.shapeMismatch(
                    "bias has \(bias.count) values, expected \(outputs)"
                )
            }
            buffer = try linear.makeBuffer(bias)
        } else {
            buffer = try zeroBias(outputs)
        }
        biasBuffers[key] = buffer
        return buffer
    }

    private func zeroBias(_ outputs: Int) throws -> MTLBuffer {
        if let cached = zeroBiases[outputs] { return cached }
        let buffer = try emptyBuffer(outputs, role: "zero bias")
        memset(buffer.contents(), 0, buffer.length)
        zeroBiases[outputs] = buffer
        return buffer
    }

    /// Upload every weight and gain the encode will ask for, so a missing tensor is
    /// reported before any dispatch is encoded rather than halfway through a block.
    private func prepareWeights(for workspace: Workspace) throws {
        let hidden = spec.hiddenSize
        _ = try weightBuffer(
            .visionPatchEmbed, layer: nil, rows: hidden, columns: workspace.patchWidth
        )
        _ = try biasBuffer(.visionPatchEmbedBias, layer: nil, outputs: hidden)
        for block in 0..<max(0, spec.numHiddenLayers) {
            _ = try gainBuffer(.visionBlockNorm1, layer: block, dimension: hidden)
            _ = try weightBuffer(.visionBlockWqkv, layer: block, rows: 3 * hidden, columns: hidden)
            _ = try biasBuffer(.visionBlockWqkvBias, layer: block, outputs: 3 * hidden)
            _ = try weightBuffer(.visionBlockWo, layer: block, rows: hidden, columns: hidden)
            _ = try biasBuffer(.visionBlockWoBias, layer: block, outputs: hidden)
            _ = try gainBuffer(.visionBlockNorm2, layer: block, dimension: hidden)
            _ = try weightBuffer(
                .visionBlockW1, layer: block, rows: 2 * spec.intermediateSize, columns: hidden
            )
            _ = try weightBuffer(
                .visionBlockW2, layer: block, rows: hidden, columns: spec.intermediateSize
            )
        }
        _ = try gainBuffer(.visionFinalNorm, layer: nil, dimension: hidden)
        _ = try weightBuffer(
            .visionAlignerW1, layer: nil, rows: textHiddenSize, columns: workspace.cellVolume
        )
        _ = try biasBuffer(.visionAlignerW1Bias, layer: nil, outputs: textHiddenSize)
        _ = try weightBuffer(
            .visionAlignerW2, layer: nil, rows: textHiddenSize, columns: textHiddenSize
        )
        _ = try biasBuffer(.visionAlignerW2Bias, layer: nil, outputs: textHiddenSize)
    }

    // MARK: Elementwise dispatches

    private func barrier(_ encoder: MTLComputeCommandEncoder) {
        encoder.memoryBarrier(scope: .buffers)
    }

    private func dispatchNorm(
        on encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        gain: MTLBuffer,
        width: Int,
        rows: Int
    ) throws {
        guard width > 0, rows > 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch("a norm needs positive dimensions")
        }
        // One threadgroup per row; the kernel's `tgIdx` is a scalar grid position, so
        // the row index is the X axis and a grid width of `rows` is exactly one
        // threadgroup per row. The scratch tail is `lanes / 32` floats and the host
        // pads it to the 16-byte multiple Metal requires.
        let lanes = min(256, max(32, normPipeline.maxTotalThreadsPerThreadgroup))
        var dim = UInt32(width)
        var epsilon = DeepSeekV41VisionForward.normalizationEpsilon
        encoder.setComputePipelineState(normPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gain, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.stride, index: 4)
        let scratchBytes = max(
            ((lanes + 31) / 32) * MemoryLayout<Float>.stride, 16
        )
        encoder.setThreadgroupMemoryLength(scratchBytes, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
        )
        barrier(encoder)
    }

    private func dispatchAdd(
        on encoder: MTLComputeCommandEncoder,
        first: MTLBuffer,
        second: MTLBuffer,
        output: MTLBuffer,
        width: Int,
        rows: Int
    ) throws {
        guard width > 0, rows > 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch("an add needs positive dimensions")
        }
        var dim = UInt32(width)
        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(first, offset: 0, index: 0)
        encoder.setBuffer(second, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: width, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        barrier(encoder)
    }

    private func dispatchGelu(
        on encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) throws {
        guard count > 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch("a gelu needs positive dimensions")
        }
        var dimension = UInt32(count)
        encoder.setComputePipelineState(geluPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        barrier(encoder)
    }

    private func dispatchSiluMul(
        on encoder: MTLComputeCommandEncoder,
        gateAndUp: MTLBuffer,
        output: MTLBuffer,
        intermediate: Int,
        tokens: Int
    ) throws {
        guard intermediate > 0, tokens > 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch("a gated MLP needs positive dimensions")
        }
        var width = UInt32(intermediate)
        encoder.setComputePipelineState(siluMulPipeline)
        encoder.setBuffer(gateAndUp, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: intermediate, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        barrier(encoder)
    }

    private func dispatchUnshuffle(
        on encoder: MTLComputeCommandEncoder,
        patches: MTLBuffer,
        cells: MTLBuffer,
        patchRows: Int,
        patchColumns: Int,
        width: Int,
        ratio: Int,
        cellCount: Int,
        cellVolume: Int
    ) throws {
        guard patchRows > 0, patchColumns > 0, width > 0, ratio > 0, cellCount > 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "an unshuffle needs a positive patch grid and ratio"
            )
        }
        var rows = UInt32(patchRows)
        var columns = UInt32(patchColumns)
        var channels = UInt32(width)
        var downsample = UInt32(ratio)
        encoder.setComputePipelineState(unshufflePipeline)
        encoder.setBuffer(patches, offset: 0, index: 0)
        encoder.setBuffer(cells, offset: 0, index: 1)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&downsample, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: cellVolume, height: cellCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        barrier(encoder)
    }
}
