//
//  Inkling2MetalForward.swift
//  DynaMoE
//
//  The Inkling 2 Mini device graph: the same 48-visit schedule the host
//  reference runs (`Inkling2Forward.swift`), expressed as Metal dispatches over
//  the checkpoint's own zero-copy shard buffers.
//
//  Built STAGE BY STAGE against the host graph, not ported wholesale: the host
//  graph is what matches the checkpoint's Transformers reference, so each
//  device stage is compared with the host's value for the same stage before the
//  next one is written. A whole-graph port that is 2% off somewhere is a week of
//  bisecting; a stage that is 2% off the moment it is written names itself.
//
//  This file starts with the weight binding and the attention half of one
//  visit, because that half is where a mistake is most likely (relative bias,
//  sink, grouped KV heads, two different head geometries) and it reads the
//  cache the position writes.
//
//  Layout rule, inherited from the DeepSeek executor: every "constant" is a
//  small buffer filled by the host, never a shader literal, so one pipeline
//  serves every layer and shape.
//

import Foundation
import Metal

// MARK: - Weight binding

/// Resolves a catalog role to the buffer and byte offset a kernel binds.
///
/// The shards are the app's existing zero-copy mappings, one `MTLBuffer` per
/// shard index, so a weight is `(buffer, offset)` and never a copy. A role that
/// is missing, or whose shard is not mapped, is an error that names it -- the
/// host weight source fails closed the same way.
nonisolated public struct Inkling2DeviceWeights {
    public let config: Inkling2Config
    public let catalog: Inkling2WeightCatalog
    private let shardBuffers: [UInt32: MTLBuffer]

    public init(config: Inkling2Config, catalog: Inkling2WeightCatalog, shardBuffers: [UInt32: MTLBuffer]) {
        self.config = config
        self.catalog = catalog
        self.shardBuffers = shardBuffers
    }

    /// One tensor's buffer and byte offset, for a kernel's `[[buffer(n)]]` plus
    /// its `weightOffset` constant.
    public func tensor(
        _ role: Inkling2TensorRole, layer: Int? = nil
    ) throws -> (buffer: MTLBuffer, offset: UInt64) {
        guard let ref = catalog.tensor(role, layer: layer) else {
            throw Inkling2ForwardError.missingWeight(
                "\(role.rawValue)\(layer.map { " layer \($0)" } ?? "")"
            )
        }
        guard let shard = ref.shardIndex, let buffer = shardBuffers[shard] else {
            throw Inkling2ForwardError.missingWeight("\(ref.name): its shard is not mapped")
        }
        guard let offset = ref.byteOffset else {
            throw Inkling2ForwardError.missingWeight("\(ref.name): no byte offset")
        }
        return (buffer, offset)
    }

    /// A role's stored shape, straight from the catalog.
    public func shape(_ role: Inkling2TensorRole, layer: Int? = nil) throws -> [Int] {
        guard let ref = catalog.tensor(role, layer: layer) else {
            throw Inkling2ForwardError.missingWeight("\(role.rawValue)")
        }
        return ref.shape
    }

    /// A weight as the host graph sees it: dequantized FP32. Used only for the
    /// small vectors the kernels need as VALUES (the relative bank, clip
    /// scales), never for a stage the port exists to move -- those read the
    /// stored bytes directly.
    public func hostMatrix(_ role: Inkling2TensorRole, layer: Int? = nil) throws -> [Float] {
        try Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers
        ).matrix(role, layer: layer)
    }

    public func dtype(_ role: Inkling2TensorRole, layer: Int? = nil) throws -> String {
        guard let ref = catalog.tensor(role, layer: layer) else {
            throw Inkling2ForwardError.missingWeight("\(role.rawValue)")
        }
        return ref.dtype
    }
}

// MARK: - Workspace

/// The device buffers a visit's stages share.
///
/// Sized for ONE position and one visit: the staging vectors are
/// `hiddenSize`-wide (the widest a single stage produces), and the KV rows live
/// in the sequence's own per-visit buffers rather than here. A batched form
/// would replicate the staging by slot, which is why the vector strides are
/// explicit parameters on the kernels rather than assumed.
nonisolated public final class Inkling2DeviceWorkspace {
    public let hidden: MTLBuffer
    public let attentionInput: MTLBuffer
    public let query: MTLBuffer
    public let key: MTLBuffer
    public let value: MTLBuffer
    public let gate: MTLBuffer
    public let attentionOutput: MTLBuffer
    public let projectionOutput: MTLBuffer
    /// Scratch for the reduction kernels' warp sums.
    public let reductionScratch: MTLBuffer
    /// One visit's key and value rows, staged from the sequence state before an
    /// attention dispatch. The state itself stays host-side (it is what makes a
    /// resume provable), so a device attention reads a copy; the rows are
    /// `kvWidth` wide and the buffer holds the checkpoint's maximum context.
    public let stagedKeys: MTLBuffer
    public let stagedValues: MTLBuffer
    public let stagedCapacity: Int
    /// The attention stage's remaining inputs, all small host-computed vectors
    /// or weights staged once per visit: the three projection inputs, the
    /// post-clip relative states, the sink logits and the relative bank.
    public let queryInput: MTLBuffer
    public let keyInput: MTLBuffer
    public let valueInput: MTLBuffer
    public let relativeStates: MTLBuffer
    public let sinkLogits: MTLBuffer
    public let bank: MTLBuffer

    public init(device: MTLDevice, config: Inkling2Config) throws {
        let hiddenBytes = config.hiddenSize * MemoryLayout<Float>.stride
        let attentionWidth = config.numAttentionHeads * config.headDim
        func make(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(length, 16), options: .storageModeShared) else {
                throw Inkling2ForwardError.unsupported("could not allocate the workspace's \(label)")
            }
            return buffer
        }
        hidden = try make(hiddenBytes, "hidden staging")
        attentionInput = try make(hiddenBytes, "attention input")
        query = try make(attentionWidth * MemoryLayout<Float>.stride, "query")
        key = try make(config.numKeyValueHeads * config.headDim * MemoryLayout<Float>.stride, "key")
        value = try make(config.numKeyValueHeads * config.headDim * MemoryLayout<Float>.stride, "value")
        gate = try make(attentionWidth * MemoryLayout<Float>.stride, "output gate")
        attentionOutput = try make(attentionWidth * MemoryLayout<Float>.stride, "attention output")
        projectionOutput = try make(hiddenBytes, "projection output")
        reductionScratch = try make(32 * MemoryLayout<Float>.stride, "reduction scratch")
        // The attention reads at most the window plus the compressed rows; the
        // checkpoint's declared maximum is the safe bound and a few dozen MB at
        // FP32, which is the same order as the host's own rows.
        let kvWidth = config.numKeyValueHeads * config.headDim
        stagedCapacity = config.maxPositionEmbeddings
        let stagedBytes = stagedCapacity * kvWidth * MemoryLayout<Float>.stride
        stagedKeys = try make(stagedBytes, "staged keys")
        stagedValues = try make(stagedBytes, "staged values")
        queryInput = try make(hiddenBytes, "query input")
        keyInput = try make(hiddenBytes, "key input")
        valueInput = try make(hiddenBytes, "value input")
        relativeStates = try make(
            config.numAttentionHeads * config.dRel * MemoryLayout<Float>.stride, "relative states"
        )
        sinkLogits = try make(config.numAttentionHeads * MemoryLayout<Float>.stride, "sink logits")
        bank = try make(
            config.dRel * max(config.relExtent, config.slidingWindowSize) * MemoryLayout<Float>.stride,
            "relative bank"
        )
    }

    /// Copies one visit's retained key and value rows into the staging buffers.
    /// Returns the row count; the caller passes it to the attention kernel with
    /// the visit's `windowStart`.
    public func stage(_ state: Inkling2VisitState) -> Int {
        let rows = state.keys.count
        let width = state.keys.first?.count ?? 0
        guard rows > 0, width > 0, rows <= stagedCapacity else { return 0 }
        // Row-major copy: the host keeps [[Float]], the kernel wants one block.
        let keyPointer = stagedKeys.contents().bindMemory(to: Float.self, capacity: rows * width)
        let valuePointer = stagedValues.contents().bindMemory(to: Float.self, capacity: rows * width)
        for row in 0..<rows {
            let keyRow = state.keys[row]
            let valueRow = state.values[row]
            for column in 0..<min(width, keyRow.count) {
                keyPointer[row * width + column] = keyRow[column]
                valuePointer[row * width + column] = valueRow[column]
            }
        }
        return rows
    }
}

// MARK: - The device forward

/// One visit's device path. It owns the pipelines and the workspace; the turn
/// engine owns the sequence state and the weights.
nonisolated public final class Inkling2MetalForward {
    public let config: Inkling2Config
    public let weights: Inkling2DeviceWeights
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    /// The host graph, used ONLY for the pieces this stage deliberately leaves
    /// on the host (two small vectors' worth of arithmetic) and as the oracle a
    /// parity test compares against. It reads the same weights.
    public let host: Inkling2CPUForward

    private let pipelines: [String: MTLComputePipelineState]

    /// Every pipeline this graph dispatches, by kernel name. A missing one is
    /// a load-time failure that names it, not a crash at the first dispatch.
    public static let kernelNames: [String] = [
        "inkling_rmsnorm_f32",
        "inkling_rmsnorm_no_gain_f32",
        "inkling_softcapped_swiglu_f32",
        "inkling_short_conv_f32",
        "inkling_message_gate_f32",
        "inkling_l2_clip_scale_f32",
        "inkling_gate_multiply_f32",
        "inkling_block_read_f32",
        "inkling_attention_f32",
        "inkling_expert_row_f32",
        "inkling_router_topk_f32",
        "bf16_gemv_rows8",
        "bf16_gemv_rows",
        "bf16_gemv",
        "vector_add_f32"
    ]

    public init(
        device: MTLDevice,
        library: MTLLibrary,
        queue: MTLCommandQueue,
        weights: Inkling2DeviceWeights,
        host: Inkling2CPUForward
    ) throws {
        self.config = weights.config
        self.weights = weights
        self.device = device
        self.queue = queue
        self.host = host
        var built: [String: MTLComputePipelineState] = [:]
        for name in Self.kernelNames {
            guard let function = library.makeFunction(name: name) else {
                throw Inkling2ForwardError.unsupported("the shader library has no kernel '\(name)'")
            }
            built[name] = try device.makeComputePipelineState(function: function)
        }
        self.pipelines = built
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let pipeline = pipelines[name] else {
            throw Inkling2ForwardError.unsupported("no pipeline for '\(name)'")
        }
        return pipeline
    }

    // MARK: Small constants

    /// A `constant` argument, as the buffer the shader binds.
    ///
    /// The caller keeps the returned buffers alive until the command buffer is
    /// committed: a dispatch's arguments are read when the GPU runs it, not
    /// when it is encoded, so a temporary freed at the end of the encoding
    /// scope would be read after it was gone.
    private func scalar<T>(_ value: T) -> MTLBuffer? {
        var copy = value
        return device.makeBuffer(
            bytes: &copy, length: MemoryLayout<T>.stride, options: .storageModeShared
        )
    }

    /// `y = W x` for a BF16 weight, dispatched through the wide-grid block form.
    ///
    /// The host's `linear` is FP32 over a dequantized matrix; the device reads
    /// the checkpoint's BF16 bytes, which is the same number widened, so the
    /// only difference is the accumulation order.
    func gemm(
        encoder: MTLComputeCommandEncoder,
        role: Inkling2TensorRole,
        layer: Int?,
        input: MTLBuffer,
        inputOffset: Int,
        output: MTLBuffer,
        outputOffset: Int,
        rows: Int,
        columns: Int,
        kernel: String
    ) throws {
        let (weightBuffer, weightOffset) = try weights.tensor(role, layer: layer)
        let offset = scalar(UInt64(weightOffset))
        let inDim = scalar(UInt32(columns))
        let outDim = scalar(UInt32(rows))
        let zero: UInt32 = 0
        let inputStride = scalar(zero)
        let outputStride = scalar(zero)
        encoder.setComputePipelineState(try pipeline(kernel))
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBuffer(offset, offset: 0, index: 3)
        encoder.setBuffer(inDim, offset: 0, index: 4)
        encoder.setBuffer(outDim, offset: 0, index: 5)
        encoder.setBuffer(inputStride, offset: 0, index: 6)
        encoder.setBuffer(outputStride, offset: 0, index: 7)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    /// The block read: `softmax(q . rmsnorm(row) / sqrt(width)) . rmsnorm(row)`
    /// over the stack plus the partial, one threadgroup per query.
    func blockRead(
        encoder: MTLComputeCommandEncoder,
        stack: MTLBuffer,
        stackRows: Int,
        partial: MTLBuffer,
        partialOffset: Int,
        queries: MTLBuffer,
        queryCount: Int,
        output: MTLBuffer,
        outputOffset: Int
    ) throws {
        let width = config.hiddenSize
        let stackRowsValue = scalar(UInt32(stackRows))
        let widthValue = scalar(UInt32(width))
        encoder.setComputePipelineState(try pipeline("inkling_block_read_f32"))
        encoder.setBuffer(stack, offset: 0, index: 0)
        encoder.setBuffer(partial, offset: partialOffset, index: 1)
        encoder.setBuffer(queries, offset: 0, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBuffer(stackRowsValue, offset: 0, index: 4)
        encoder.setBuffer(widthValue, offset: 0, index: 5)
        encoder.setThreadgroupMemoryLength(
            (32 + 64 + 64) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: queryCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    /// The l2 clip with a learned scale, one threadgroup per head.
    func l2Clip(
        encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer,
        vectorOffset: Int,
        scale: MTLBuffer,
        scaleOffset: Int,
        headDim: Int,
        heads: Int,
        threshold: Float,
        tau: Float
    ) throws -> [MTLBuffer?] {
        let headDimValue = scalar(UInt32(headDim))
        let thresholdValue = scalar(threshold)
        let tauValue = scalar(tau)
        encoder.setComputePipelineState(try pipeline("inkling_l2_clip_scale_f32"))
        encoder.setBuffer(vector, offset: vectorOffset, index: 0)
        encoder.setBuffer(scale, offset: scaleOffset, index: 1)
        encoder.setBuffer(headDimValue, offset: 0, index: 2)
        encoder.setBuffer(thresholdValue, offset: 0, index: 3)
        encoder.setBuffer(tauValue, offset: 0, index: 4)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        // The caller keeps these alive until the commit.
        return [headDimValue, thresholdValue, tauValue]
    }

    /// One head's unweighted norm in place, which the value projection needs.
    func headNorm(
        encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer,
        headDim: Int,
        heads: Int
    ) throws -> [MTLBuffer?] {
        let headDimValue = scalar(UInt32(headDim))
        let epsValue = scalar(config.rmsNormEps)
        encoder.setComputePipelineState(try pipeline("inkling_rmsnorm_no_gain_f32"))
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(vector, offset: 0, index: 1)
        encoder.setBuffer(headDimValue, offset: 0, index: 2)
        encoder.setBuffer(epsValue, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        // One threadgroup per head; the kernel normalizes rows of `dim`, and
        // the vector is heads * headDim, so the grid is the head count.
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        return [headDimValue, epsValue]
    }

    // MARK: Staging and host-side pieces

    /// Copies a host vector into a workspace buffer.
    func staging(_ buffer: MTLBuffer, _ values: [Float]) throws -> MTLBuffer {
        guard values.count * MemoryLayout<Float>.stride <= buffer.length else {
            throw Inkling2ForwardError.shapeMismatch(
                "staging \(values.count) values into a \(buffer.length)-byte buffer"
            )
        }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count)
        for index in values.indices { pointer[index] = values[index] }
        return buffer
    }

    func readBack(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count
        ))
    }

    /// A weight as the host sees it: the same dequantized matrix the host graph
    /// multiplies by. Used only for the small vectors the kernel needs as
    /// values (the relative bank), never for a stage the port is meant to move.
    func hostMatrix(_ role: Inkling2TensorRole, layer: Int?) throws -> [Float] {
        try weights.hostMatrix(role, layer: layer)
    }

    private func commandComponents() throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw Inkling2ForwardError.unsupported("could not make a command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Inkling2ForwardError.unsupported("could not make a compute encoder")
        }
        return (commandBuffer, encoder)
    }

    /// The relative states: `wr @ input`, per head scaled by the learned scale,
    /// L2-clipped, and (on a full layer) log-scaled. Host arithmetic, because it
    /// is two small vectors and the kernel consumes only the result.
    func relativeStates(layer: Int, queryInput: [Float], position: Int, isFull: Bool) throws -> [Float] {
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let wr = try hostMatrix(.attnWrDu, layer: layer)
        let rScale = try hostMatrix(.attnRelL2ClipScale, layer: layer)
        var relative = try host.linear(wr, queryInput, rows: heads * config.dRel)
        let tau = isFull
            ? Inkling2Math.logScale(position: position, alpha: config.logScalingAlpha, floor: config.logScalingNFloor)
            : 1
        for head in 0..<heads {
            var slice = Array(relative[(head * config.dRel)..<((head + 1) * config.dRel)])
            for dimension in 0..<config.dRel { slice[dimension] *= rScale[head * config.dRel + dimension] }
            slice = Inkling2Math.l2Clip(slice, threshold: config.relL2ClipThreshold)
            if isFull { slice = slice.map { $0 * tau } }
            for dimension in 0..<config.dRel { relative[head * config.dRel + dimension] = slice[dimension] }
        }
        return relative
    }

    /// The sink logits: `sinkProj @ input`, one per head.
    func sinkLogits(layer: Int, queryInput: [Float]) throws -> [Float] {
        let heads = !config.isLocalLayer(layer) ? config.numAttentionHeads : config.swaNumAttentionHeads
        let projection = try hostMatrix(.attnSinkProj, layer: layer)
        return try host.linear(projection, queryInput, rows: heads)
    }

    /// The short convolution on device, static or dynamic, for one of a visit's
    /// four convolution states. Returns the scalar buffers the dispatch reads, to
    /// be kept alive until the commit, and PUSHES the input into the host state's
    /// history -- the state stays the authority, so a device pass and a host pass
    /// over the same visit leave it in the same place.
    func convolve(
        encoder: MTLComputeCommandEncoder,
        workspace: Inkling2DeviceWorkspace,
        state: Inkling2VisitState,
        index: Int,
        values: MTLBuffer,
        valuesCount: Int,
        layer: Int,
        isFull: Bool,
        kvHeads: Int,
        headDim: Int,
        dynamicInput: MTLBuffer,
        hidden: Int,
        output: MTLBuffer
    ) throws -> [MTLBuffer?] {
        let kernelSize = config.convKernelSize
        let history = state.history(index)
        let taps = try hostMatrix(index == 0 || index == 1
            ? (isFull ? .attnKSconvStatic : .attnKSconvWeight)
            : .attnSconv, layer: layer)
        // The stored conv1d order is oldest-first; the kernel takes it as stored.
        let tapsBuffer = try staging(workspace.bank, taps)
        let historyRows = history.count
        var flatHistory = [Float](repeating: 0, count: historyRows * valuesCount)
        for row in 0..<historyRows {
            for channel in 0..<valuesCount { flatHistory[row * valuesCount + channel] = history[row][channel] }
        }
        let historyBuffer = try staging(workspace.relativeStates, flatHistory)
        let channels = scalar(UInt32(valuesCount))
        let tapCount = scalar(UInt32(kernelSize))
        let rows = scalar(UInt32(historyRows))
        encoder.setComputePipelineState(try pipeline("inkling_short_conv_f32"))
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(historyBuffer, offset: 0, index: 1)
        encoder.setBuffer(tapsBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(channels, offset: 0, index: 4)
        encoder.setBuffer(tapCount, offset: 0, index: 5)
        encoder.setBuffer(rows, offset: 0, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: valuesCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        // The dynamic form's extra per-head taps are a host-side projection at
        // this stage: the kernel covers the static part exactly, and the
        // dynamic term is added by the host rule below when it is non-zero.
        if isFull {
            let projected = try host.linear(
                try hostMatrix(.attnKSconvProj, layer: layer),
                readBack(dynamicInput, count: hidden),
                rows: kvHeads * kernelSize
            )
            if projected.contains(where: { $0 != 0 }) {
                let current = readBack(values, count: valuesCount)
                let out = readBack(output, count: valuesCount)
                var corrected = out
                for channel in 0..<valuesCount {
                    let head = channel / max(1, headDim)
                    let channelHistory = history.map { $0[channel] }
                    let staticWeights = (0..<kernelSize).map { taps[$0 * valuesCount + channel] }
                    let dynamicWeights = (0..<kernelSize).map { projected[head * kernelSize + $0] }
                    corrected[channel] = Inkling2Math.dynamicShortConvolution(
                        current: current[channel], history: channelHistory,
                        staticWeights: staticWeights, dynamicWeights: dynamicWeights
                    )
                }
                let pointer = output.contents().bindMemory(to: Float.self, capacity: valuesCount)
                for channel in 0..<valuesCount { pointer[channel] = corrected[channel] }
            }
        }
        state.pushConvolution(index, value: readBack(values, count: valuesCount))
        _ = workspace
        _ = taps
        return [channels, tapCount, rows, tapsBuffer, historyBuffer]
    }

    /// `output *= sigmoid(gate)`, elementwise.
    func gateMultiply(encoder: MTLComputeCommandEncoder, output: MTLBuffer, gate: MTLBuffer, count: Int) throws {
        let countValue = scalar(UInt32(count))
        encoder.setComputePipelineState(try pipeline("inkling_gate_multiply_f32"))
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBuffer(gate, offset: 0, index: 1)
        encoder.setBuffer(countValue, offset: 0, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        withExtendedLifetime(countValue) {}
    }

    // MARK: The attention half

    /// The attention half of one visit, stage for stage the host's
    /// `Inkling2CPUForward.attention`, returning the same vector the host
    /// returns (after the gate and the grouped output projection).
    ///
    /// Host arrays in, host array out: the sequence state is host-side (it is
    /// what makes a resume provable), so this stages what the kernels read and
    /// copies the result back. The staging is a few hundred KB against tens of
    /// GFLOP of work at these widths, so it is not what the port is for -- the
    /// port is the arithmetic.
    public func attentionHalf(
        layer: Int,
        state: Inkling2VisitState,
        queryInput: [Float],
        keyInput: [Float],
        valueInput: [Float],
        position: Int,
        workspace: Inkling2DeviceWorkspace
    ) throws -> [Float] {
        let isFull = !config.isLocalLayer(layer)
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let kvHeads = isFull ? config.numKeyValueHeads : config.swaNumKeyValueHeads
        let headDim = isFull ? config.headDim : config.swaHeadDim
        let hidden = config.hiddenSize

        // Inputs staged once; every stage reads them.
        let queryInputBuffer = try staging(workspace.queryInput, queryInput)
        let keyInputBuffer = try staging(workspace.keyInput, keyInput)
        let valueInputBuffer = try staging(workspace.valueInput, valueInput)

        let (commandBuffer, encoder) = try commandComponents()

        // Q: project, then scale + L2 clip (+ log scale on a full layer).
        try gemm(
            encoder: encoder, role: .attnWqDu, layer: layer,
            input: queryInputBuffer, inputOffset: 0,
            output: workspace.query, outputOffset: 0,
            rows: heads * headDim, columns: hidden, kernel: "bf16_gemv_rows8"
        )
        let qScale = try weights.tensor(.attnQL2ClipScale, layer: layer)
        let tau = isFull
            ? Inkling2Math.logScale(position: position, alpha: config.logScalingAlpha, floor: config.logScalingNFloor)
            : 1
        let qKeep = try l2Clip(
            encoder: encoder, vector: workspace.query, vectorOffset: 0,
            scale: qScale.buffer, scaleOffset: Int(qScale.offset),
            headDim: headDim, heads: heads, threshold: config.qkL2ClipThreshold, tau: tau
        )

        // K: project, convolve, L2 clip.
        try gemm(
            encoder: encoder, role: .attnWkDv, layer: layer,
            input: keyInputBuffer, inputOffset: 0,
            output: workspace.key, outputOffset: 0,
            rows: kvHeads * headDim, columns: hidden, kernel: "bf16_gemv_rows8"
        )
        let keyConv = try convolve(
            encoder: encoder, workspace: workspace, state: state, index: 0,
            values: workspace.key, valuesCount: kvHeads * headDim,
            layer: layer, isFull: isFull, kvHeads: kvHeads, headDim: headDim,
            dynamicInput: keyInputBuffer, hidden: hidden, output: workspace.key
        )
        // One shared clip scale for q and k: the checkpoint publishes only
        // `attn.q_l2_clip.scale`, and the host applies it to both.
        let kScale = try weights.tensor(.attnQL2ClipScale, layer: layer)
        let kKeep = try l2Clip(
            encoder: encoder, vector: workspace.key, vectorOffset: 0,
            scale: kScale.buffer, scaleOffset: Int(kScale.offset),
            headDim: headDim, heads: kvHeads, threshold: config.qkL2ClipThreshold, tau: 1
        )

        // V: project, convolve, unweighted per-head norm.
        try gemm(
            encoder: encoder, role: .attnWvDv, layer: layer,
            input: valueInputBuffer, inputOffset: 0,
            output: workspace.value, outputOffset: 0,
            rows: kvHeads * headDim, columns: hidden, kernel: "bf16_gemv_rows8"
        )
        let valueConv = try convolve(
            encoder: encoder, workspace: workspace, state: state, index: 1,
            values: workspace.value, valuesCount: kvHeads * headDim,
            layer: layer, isFull: isFull, kvHeads: kvHeads, headDim: headDim,
            dynamicInput: valueInputBuffer, hidden: hidden, output: workspace.value
        )
        let vKeep = try headNorm(
            encoder: encoder, vector: workspace.value, headDim: headDim, heads: kvHeads
        )

        // The relative states and the sink logits are small host-side
        // projections; the kernel only needs their results, and computing them
        // on the host keeps this stage's device part to what it is for.
        let relative = try relativeStates(layer: layer, queryInput: queryInput, position: position, isFull: isFull)
        let sink = try sinkLogits(layer: layer, queryInput: queryInput)
        let relativeBuffer = try staging(workspace.relativeStates, relative)
        let sinkBuffer = try staging(workspace.sinkLogits, sink)
        let bank = try weights.tensor(.attnRelLogitsProj, layer: layer)
        let bankValues = try hostMatrix(.attnRelLogitsProj, layer: layer)
        let bankBuffer = try staging(workspace.bank, bankValues)
        _ = bank

        // The KV rows this position attends over, staged.
        let rows = workspace.stage(state)

        try attention(
            encoder: encoder, workspace: workspace, state: state, rows: rows,
            position: position, relative: relativeBuffer, bank: bankBuffer,
            sink: sinkBuffer, output: workspace.attentionOutput
        )

        // The output gate, then the grouped output projection.
        try gemm(
            encoder: encoder, role: .attnGate, layer: layer,
            input: queryInputBuffer, inputOffset: 0,
            output: workspace.gate, outputOffset: 0,
            rows: heads * headDim, columns: hidden, kernel: "bf16_gemv_rows8"
        )
        try gateMultiply(encoder: encoder, output: workspace.attentionOutput,
                         gate: workspace.gate, count: heads * headDim)
        try gemm(
            encoder: encoder, role: .attnWoUd, layer: layer,
            input: workspace.attentionOutput, inputOffset: 0,
            output: workspace.projectionOutput, outputOffset: 0,
            rows: hidden, columns: heads * headDim, kernel: "bf16_gemv_rows8"
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        // Keep every scalar buffer alive past the commit.
        withExtendedLifetime(qKeep) {}
        withExtendedLifetime(kKeep) {}
        withExtendedLifetime(vKeep) {}
        withExtendedLifetime(keyConv) {}
        withExtendedLifetime(valueConv) {}
        return readBack(workspace.projectionOutput, count: hidden)
    }

    /// The attention kernel over one visit's staged KV rows.
    func attention(
        encoder: MTLComputeCommandEncoder,
        workspace: Inkling2DeviceWorkspace,
        state: Inkling2VisitState,
        rows: Int,
        position: Int,
        relative: MTLBuffer,
        bank: MTLBuffer,
        sink: MTLBuffer,
        output: MTLBuffer
    ) throws {
        let isFull = state.isFullAttention
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let kvHeads = isFull ? config.numKeyValueHeads : config.swaNumKeyValueHeads
        let headDim = isFull ? config.headDim : config.swaHeadDim
        let kvWidth = kvHeads * headDim
        let values: [UInt32] = [
            UInt32(heads), UInt32(heads / kvHeads), UInt32(headDim), UInt32(kvWidth),
            UInt32(rows), UInt32(state.windowStart), UInt32(position),
            UInt32(config.dRel), UInt32(isFull ? config.relExtent : config.slidingWindowSize)
        ]
        encoder.setComputePipelineState(try pipeline("inkling_attention_f32"))
        encoder.setBuffer(workspace.query, offset: 0, index: 0)
        encoder.setBuffer(workspace.stagedKeys, offset: 0, index: 1)
        encoder.setBuffer(workspace.stagedValues, offset: 0, index: 2)
        encoder.setBuffer(relative, offset: 0, index: 3)
        encoder.setBuffer(bank, offset: 0, index: 4)
        encoder.setBuffer(sink, offset: 0, index: 5)
        encoder.setBuffer(output, offset: 0, index: 6)
        for (index, value) in values.enumerated() {
            guard let buffer = scalar(value) else { continue }
            encoder.setBuffer(buffer, offset: 0, index: 7 + index)
        }
        encoder.setThreadgroupMemoryLength(
            (32 + rows) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }
}
