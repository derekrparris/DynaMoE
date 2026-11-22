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

    private let pipelines: [String: MTLComputePipelineState]

    /// Every pipeline this graph dispatches, by kernel name. A missing one is
    /// a load-time failure that names it, not a crash at the first dispatch.
    public static let kernelNames: [String] = [
        "inkling_rmsnorm_f32",
        "inkling_rmsnorm_no_gain_f32",
        "inkling_softcapped_swiglu_f32",
        "inkling_short_conv_f32",
        "inkling_message_gate_f32",
        "inkling_block_read_f32",
        "inkling_attention_f32",
        "inkling_expert_row_f32",
        "inkling_router_topk_f32",
        "bf16_gemv_rows8",
        "bf16_gemv_rows",
        "bf16_gemv",
        "vector_add_f32"
    ]

    public init(device: MTLDevice, library: MTLLibrary, queue: MTLCommandQueue, weights: Inkling2DeviceWeights) throws {
        self.config = weights.config
        self.weights = weights
        self.device = device
        self.queue = queue
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
