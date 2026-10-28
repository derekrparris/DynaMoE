//
//  DeepSeekV41MetalForward.swift
//  DynaMoE
//
//  The device executor: `DeepSeekV41ForwardStages.layerPipeline` bound to the
//  Metal kernels, position by position.
//
//  It is deliberately a translation, not a second design. Everything that decides
//  *what* happens (stage order, which cache rows a position reads, when a
//  compressed row is produced, which residual a sub-layer writes) comes from the
//  same helpers the CPU reference uses, so a disagreement between the two
//  backends is a kernel bug rather than two graphs that drifted. The three shared
//  pieces are:
//
//    * `DeepSeekV41ForwardStages.layerPipeline` — the stage order,
//    * `DeepSeekV41CacheAddressing.attentionRowPlan` — the rows a position reads,
//      resolved on the host so the number of gathered rows is known before any
//      dispatch (no device-to-host readback of a count, which would be a sync
//      point per layer),
//    * `DeepSeekV41CacheAddressing.compressedRowCount` / `poolRange` — when a
//      compressed row exists and which positions it pools.
//
//  Weight formats are resolved per tensor rather than per model, because one
//  checkpoint mixes three: dense linears are E4M3 with ue8m0 scales on a 32x32
//  grid, routed experts are MXFP4, and norms / some embeddings are BF16. A format
//  with no kernel is reported at load (fail closed) instead of being dispatched to
//  a kernel that would read it at the wrong stride.
//
//  Two things are host-side by design:
//
//    * the routing top-k, because resolving an expert's tensor to a
//      `(buffer, byte offset)` pair is a host decision and the mixture needs the
//      weights as dispatch constants. One wait per layer, the same shape the
//      generic MoE loop already uses.
//    * the Engram stage. Its combination rule is
//      `DeepSeekV41Conformance.engramCombination` (unverified), its rows are a
//      ~100 GB pread stream, and a wrong combination reads the right rows and
//      mixes them wrongly, which looks like "the memory model does nothing". It
//      therefore reports rather than guesses.
//

import Foundation
import Metal

// MARK: - Weight formats

/// One linear's on-disk format, which is what picks the kernel.
nonisolated public enum DeepSeekV41LinearFormat: String, Sendable, Equatable {
    /// E4M3 values with one E8M0 scale per 32 on a 32x32 grid.
    case fp8Block32
    /// E2M1 values, two per byte, with one E8M0 scale per 32.
    case mxfp4
    /// Plain bfloat16, no scales.
    case bf16
    /// No kernel reads this format.
    case unsupported

    public var isSupported: Bool { self != .unsupported }

    /// Values per quantization block, or 0 for an unquantized format.
    public var blockWidth: Int {
        switch self {
        case .fp8Block32, .mxfp4: return 32
        case .bf16, .unsupported: return 0
        }
    }

    /// Bytes one value occupies on disk. MXFP4 packs two per byte.
    public var bytesPerValue: Int {
        switch self {
        case .fp8Block32: return 1
        case .mxfp4: return 1
        case .bf16: return 2
        case .unsupported: return 0
        }
    }
}

// MARK: - Kernel selection

/// Every kernel the executor binds, by name. Names are strings on the Swift side
/// (that is how `ComputeShaders.metal` is wired in this project), so they live in
/// one place that a test can check against the pipeline list.
nonisolated public enum DeepSeekV41Kernels {
    public static let fp8Gemv = "dsv41_fp8_block32_gemv"
    public static let fp8SwiGLU = "dsv41_fp8_block32_swiglu_gate_up"
    public static let fp8DownAccumulate = "dsv41_fp8_block32_down_accumulate"
    public static let mxfp4Gemv = "dsv41_mxfp4_gemv"
    public static let mxfp4SwiGLU = "dsv41_mxfp4_swiglu_gate_up"
    public static let mxfp4DownAccumulate = "dsv41_mxfp4_down_accumulate"
    public static let bf16Gemv = "bf16_gemv"
    public static let bf16SwiGLU = "bf16_swiglu_gate_up"
    public static let bf16DownAccumulate = "bf16_down_proj_accumulate"
    public static let bf16Embedding = "lookup_embeddings_bf16"
    public static let mxfp8Embedding = "lookup_embeddings_mxfp8"
    public static let rmsNorm = "dsv41_rmsnorm_f32_gain"
}

/// The pure half of the executor: tensor format to kernel, and the planning rules
/// that can be checked without a GPU or a checkpoint.
nonisolated public enum DeepSeekV41DevicePlan {

    /// Which format a linear resolved to.
    ///
    /// The scale companion decides between the two fp8 families the engine
    /// carries: an E8M0 scale is the 32x32 ue8m0 grid this architecture ships, a
    /// bf16 scale belongs to the 128x128 V3 grid, which has no kernel here. Both
    /// would "work" if mis-dispatched and produce plausible numbers, so the
    /// mismatch is a load-time problem instead.
    public static func linearFormat(
        tensor: DeepSeekV41DeviceTensor?,
        scale: DeepSeekV41DeviceTensor?
    ) -> DeepSeekV41LinearFormat {
        guard let tensor else { return .unsupported }
        let dtype = tensor.dtype.uppercased()
        if dtype.contains("E2M1") || dtype.contains("MXFP4") || dtype == "FP4" || dtype.hasPrefix("F4") {
            return .mxfp4
        }
        if tensor.isFP8 {
            guard let scale, scale.isScaleCompanion, !scale.isBF16, !scale.isF16 else {
                return .unsupported
            }
            return .fp8Block32
        }
        if tensor.isBF16 { return .bf16 }
        return .unsupported
    }

    public static func gemvKernel(_ format: DeepSeekV41LinearFormat) -> String? {
        switch format {
        case .fp8Block32: return DeepSeekV41Kernels.fp8Gemv
        case .mxfp4: return DeepSeekV41Kernels.mxfp4Gemv
        case .bf16: return DeepSeekV41Kernels.bf16Gemv
        case .unsupported: return nil
        }
    }

    /// The fused gate+up kernel, or nil when the format's kernel cannot apply the
    /// checkpoint's `swiglu_limit`. The clip is part of the trained function, so a
    /// kernel without it is a wrong answer rather than a rougher approximation.
    public static func swigluKernel(
        _ format: DeepSeekV41LinearFormat,
        swigluLimit: Float
    ) -> String? {
        switch format {
        case .fp8Block32: return DeepSeekV41Kernels.fp8SwiGLU
        case .mxfp4: return DeepSeekV41Kernels.mxfp4SwiGLU
        case .bf16: return swigluLimit > 0 ? nil : DeepSeekV41Kernels.bf16SwiGLU
        case .unsupported: return nil
        }
    }

    public static func downKernel(_ format: DeepSeekV41LinearFormat) -> String? {
        switch format {
        case .fp8Block32: return DeepSeekV41Kernels.fp8DownAccumulate
        case .mxfp4: return DeepSeekV41Kernels.mxfp4DownAccumulate
        case .bf16: return DeepSeekV41Kernels.bf16DownAccumulate
        case .unsupported: return nil
        }
    }

    /// The embedding lookup. A packed-MXFP4 table cannot be looked up by the
    /// existing kernels (they expect one byte per value), so it is unsupported
    /// rather than silently dequantized as E4M3.
    public static func embeddingKernel(
        tensor: DeepSeekV41DeviceTensor?,
        scale: DeepSeekV41DeviceTensor?
    ) -> String? {
        guard let tensor else { return nil }
        switch linearFormat(tensor: tensor, scale: scale) {
        case .bf16: return DeepSeekV41Kernels.bf16Embedding
        case .fp8Block32: return DeepSeekV41Kernels.mxfp8Embedding
        case .mxfp4, .unsupported: return nil
        }
    }

    /// Rows one position may gather: its window plus the compressed rows the
    /// addressing rule reads. `index_topk == 0` means "every compressed row",
    /// which grows with the context, so the candidate-block budget bounds it and a
    /// plan that provisions wider is reported instead of overflowing the buffer.
    public static func gatherRowBudget(config: DeepSeekV41Config) -> Int {
        let compressed = max(
            config.indexTopK,
            config.candidateTopKBlocks * max(1, config.candidateBlockSize)
        )
        return max(1, config.windowSize) + max(0, compressed)
    }

    /// Wiring problems the executor can see without touching a weight: a dimension
    /// a kernel cannot chew, a compress ratio wider than the window ring, the
    /// stages only the reference implements.
    public static func problems(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan]
    ) -> [String] {
        var problems: [String] = []

        let budget = gatherRowBudget(config: config)
        for layer in plan where layer.flavor == .compressed {
            if layer.compressRatio > max(1, config.windowSize) {
                problems.append(
                    "layers.\(layer.index): compress_ratio \(layer.compressRatio) is wider than the "
                        + "\(config.windowSize)-row window ring, so its pool block is not resident"
                )
            }
            if config.indexTopK <= 0 && config.candidateTopKBlocks <= 0 {
                problems.append(
                    "layers.\(layer.index): no index_topk or candidate_topk_blocks, so a compressed "
                        + "layer would have to gather its whole compressed cache"
                )
            }
            if layer.compressedKVSource < 0 {
                problems.append("layers.\(layer.index): compressed layer has no cache owner")
            }
        }
        for layer in plan where layer.engramTable != nil {
            problems.append(
                "layers.\(layer.index): Engram memory is not served on device yet "
                    + "(its combination rule is \(DeepSeekV41Conformance.engramCombination.rawValue))"
            )
        }
        let widest = max(
            config.hiddenSize,
            config.cacheRecordWidth,
            config.qLoraRank,
            config.oLoraRank,
            max(1, config.moeIntermediateSize),
            config.numAttentionHeads * config.qkHeadDim
        )
        if widest % 32 != 0 {
            problems.append(
                "a \(widest)-wide projection is not a multiple of 32, which the block-scaled kernels require"
            )
        }
        if budget < max(1, config.windowSize) {
            problems.append("gather budget \(budget) cannot hold the window")
        }
        return problems
    }

    /// Every stage this executor encodes. The switch in `encode` is exhaustive, so
    /// this set exists purely so a test can assert full coverage of the shared
    /// layer pipeline.
    public static let encodedStages: Set<DeepSeekV41ForwardStages.Stage> = [
        .mixAttentionStreams,
        .attnNorm,
        .queryDown,
        .queryUp,
        .queryNorm,
        .ropeQuery,
        .latentProject,
        .latentNorm,
        .ropeLatent,
        .writeLatent,
        .compressPool,
        .attention,
        .outputDown,
        .outputUp,
        .writeAttentionStreams,
        .mixFFNStreams,
        .ffnNorm,
        .engram,
        .router,
        .sharedExpert,
        .routedExperts,
        .writeFFNStreams,
        .collapseStreams
    ]

    /// The stages that run before the routing readback, and the ones that run
    /// after it. The split is the one host round trip a layer needs: the expert
    /// mixture binds per-expert byte offsets, which are host values.
    public static let frontHalf: [DeepSeekV41ForwardStages.Stage] = {
        let pipeline = DeepSeekV41ForwardStages.layerPipeline
        guard let split = pipeline.firstIndex(of: .router) else { return pipeline }
        return Array(pipeline[...split])
    }()

    public static let backHalf: [DeepSeekV41ForwardStages.Stage] = {
        let pipeline = DeepSeekV41ForwardStages.layerPipeline
        guard let split = pipeline.firstIndex(of: .router) else { return [] }
        return Array(pipeline[(split + 1)...])
    }()
}

// MARK: - Vector bank

/// The small FP32 vectors a layer needs as kernels' arguments: norm gains, the
/// sink logits, the router bias, and the hyper-connection read/scale/base triples.
///
/// They are resolved once, at load, by reading the mapped shard bytes on the host
/// (the shard buffers are `storageModeShared`) and decoding by dtype. Reading them
/// through per-dtype kernels instead would mean the norm gain's stride had to
/// match a kernel's assumption, and a bf16 gain read at an f16 stride is a
/// silently-wrong result rather than a crash. A few hundred KB per model buys
/// dtype-agnosticism and one binding per kernel argument.
nonisolated final class DeepSeekV41VectorBank: @unchecked Sendable {

    private let device: MTLDevice
    private let buffers: [UInt32: MTLBuffer]
    private var bank: [String: MTLBuffer] = [:]
    private var constants: [String: MTLBuffer] = [:]

    init(device: MTLDevice, buffers: [UInt32: MTLBuffer]) {
        self.device = device
        self.buffers = buffers
    }

    static func key(_ role: DeepSeekV41TensorRole, layer: Int?) -> String {
        "\(role.rawValue)|\(layer.map(String.init) ?? "-")"
    }

    func store(
        _ role: DeepSeekV41TensorRole,
        layer: Int?,
        ref: DeepSeekV41TensorRef?,
        resizeTo count: Int? = nil
    ) -> Bool {
        guard let ref, let values = read(ref, resizeTo: count) else { return false }
        guard let buffer = makeBuffer(values) else { return false }
        bank[Self.key(role, layer: layer)] = buffer
        return true
    }

    func buffer(_ role: DeepSeekV41TensorRole, layer: Int?) -> MTLBuffer? {
        bank[Self.key(role, layer: layer)]
    }

    /// The gain for a norm, falling back to ones. The reference's `rmsNorm` treats
    /// a gain whose width disagrees with the input as "no gain", and a slot the
    /// checkpoint does not publish is the same case.
    func gain(_ role: DeepSeekV41TensorRole, layer: Int?, width: Int) -> MTLBuffer? {
        if let stored = buffer(role, layer: layer), stored.length >= width * 4 { return stored }
        return constant(1, count: width)
    }

    /// A uniform vector, cached per `(value, count)`.
    func constant(_ value: Float, count: Int) -> MTLBuffer? {
        let key = "\(value)|\(count)"
        if let existing = constants[key] { return existing }
        guard let buffer = makeBuffer([Float](repeating: value, count: max(1, count))) else {
            return nil
        }
        constants[key] = buffer
        return buffer
    }

    private func makeBuffer(_ values: [Float]) -> MTLBuffer? {
        guard !values.isEmpty else { return nil }
        return values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }
    }

    private func read(_ ref: DeepSeekV41TensorRef, resizeTo count: Int?) -> [Float]? {
        guard let shard = ref.shardIndex,
              let offset = ref.offset,
              let buffer = buffers[UInt32(shard)],
              buffer.storageMode == .shared else { return nil }
        let elementCount = max(1, ref.shape.reduce(1, *))
        guard elementCount > 0, offset >= 0 else { return nil }
        let dtype = ref.dtype.uppercased()
        let contents = buffer.contents()

        var values: [Float]
        if dtype.contains("BF16") {
            guard offset + elementCount * 2 <= buffer.length else { return nil }
            let pointer = contents.advanced(by: offset).assumingMemoryBound(to: UInt16.self)
            values = (0..<elementCount).map { Float(bitPattern: UInt32(pointer[$0]) << 16) }
        } else if dtype.contains("F16") {
            guard offset + elementCount * 2 <= buffer.length else { return nil }
            let pointer = contents.advanced(by: offset).assumingMemoryBound(to: UInt16.self)
            values = (0..<elementCount).map { Float(Float16(bitPattern: pointer[$0])) }
        } else if dtype.contains("F32") {
            guard offset + elementCount * 4 <= buffer.length else { return nil }
            let pointer = contents.advanced(by: offset).assumingMemoryBound(to: Float.self)
            values = Array(UnsafeBufferPointer(start: pointer, count: elementCount))
        } else {
            return nil
        }

        guard let count, count > 0, values.count != count else { return values }
        if values.count > count { return Array(values.prefix(count)) }
        return values + [Float](repeating: 0, count: count - values.count)
    }
}

// MARK: - Executor

/// One loaded DeepSeek-V4.1 checkpoint, driven one position at a time.
///
/// `forwardToken` appends exactly one position (the cache holds every position the
/// turn has seen, ring-buffered) and returns the logits the sampler needs;
/// `prefill` walks a prompt through it. Both are synchronous: the caller is
/// already inside the engine's detached turn task, and the only waits are the
/// per-layer routing readback and the final logits readback.
nonisolated public final class DeepSeekV41MetalForward: @unchecked Sendable {

    public let config: DeepSeekV41Config
    public let plan: [DeepSeekV41LayerPlan]
    public let shapes: DeepSeekV41TensorShapes
    public let geometry: DeepSeekV41CacheGeometry
    /// Non-empty means the executor will not run. Reported by the load log and by
    /// the launch refusal, so an unwired stage reads as itself rather than as a
    /// generic "not ported".
    public let problems: [String]

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelines: DeepSeekV41Pipelines
    private let weights: DeepSeekV41DeviceWeights
    private let cache: DeepSeekV41KVCache
    private let workspace: DeepSeekV41Workspace
    private let shardBuffers: [UInt32: MTLBuffer]
    private let vectors: DeepSeekV41VectorBank
    private let frequencies: [DeepSeekV41RopeTable: MTLBuffer]
    private let mult: Int
    private let recordElements: Int
    private let swigluLimit: Float
    private let gatherRows: Int
    private let contextLength: Int

    public init(runtime: DeepSeekV41Runtime, shardBuffers: [UInt32: MTLBuffer]) {
        self.config = runtime.config
        self.plan = runtime.plan
        self.shapes = DeepSeekV41TensorShapes(config: runtime.config)
        self.geometry = runtime.cache.layout.geometry
        self.device = runtime.device
        self.commandQueue = runtime.commandQueue
        self.pipelines = runtime.pipelines
        self.weights = runtime.weights
        self.cache = runtime.cache
        self.workspace = runtime.workspace
        self.shardBuffers = shardBuffers
        self.mult = max(1, runtime.config.hcMult)
        self.recordElements = max(
            1,
            runtime.cache.layout.geometry.recordBytes
                / max(1, runtime.cache.layout.geometry.precision.valueBytesPerElement)
        )
        self.swigluLimit = runtime.config.swigluLimit ?? 0
        self.gatherRows = runtime.workspace.gatherRows
        self.contextLength = runtime.contextLength

        var problems = DeepSeekV41DevicePlan.problems(config: runtime.config, plan: runtime.plan)
        if runtime.cachePrecision != .fp16 {
            problems.append(
                "the cache is allocated at \(runtime.cachePrecision.rawValue), but the store/gather "
                    + "kernels are the FP16 pair"
            )
        }
        if runtime.workspace.gatherRows < DeepSeekV41DevicePlan.gatherRowBudget(config: runtime.config) {
            problems.append(
                "the workspace holds \(runtime.workspace.gatherRows) gathered rows for a "
                    + "\(DeepSeekV41DevicePlan.gatherRowBudget(config: runtime.config))-row budget"
            )
        }

        // RoPE tables, one buffer per rail the plan uses.
        var tables: [DeepSeekV41RopeTable: MTLBuffer] = [:]
        var seen = Set<DeepSeekV41RopeTable>()
        for layer in runtime.plan where !seen.contains(layer.ropeTable) {
            seen.insert(layer.ropeTable)
            let shape = DeepSeekV41AttentionShape.make(config: runtime.config, layer: layer)
            let values = DeepSeekV41RoPE.frequencies(for: shape, config: runtime.config)
            guard !values.isEmpty else { continue }
            let buffer = values.withUnsafeBytes { raw in
                runtime.device.makeBuffer(
                    bytes: raw.baseAddress!,
                    length: raw.count,
                    options: .storageModeShared
                )
            }
            if let buffer { tables[layer.ropeTable] = buffer }
        }
        self.frequencies = tables

        // The per-layer scalars and gains, resolved once.
        let bank = DeepSeekV41VectorBank(device: runtime.device, buffers: shardBuffers)
        let catalog = runtime.catalog
        func store(
            _ role: DeepSeekV41TensorRole,
            _ layer: Int?,
            resizeTo count: Int? = nil
        ) {
            _ = bank.store(role, layer: layer, ref: catalog.tensor(role, layer: layer), resizeTo: count)
        }
        store(.finalNorm, nil)
        for layer in runtime.plan {
            let l = layer.index
            store(.attnNorm, l)
            store(.ffnNorm, l)
            store(.attnWqANorm, l)
            store(.attnKvNorm, l)
            store(.attnSink, l)
            store(.routerBias, l)
            store(.hcAttnFn, l, resizeTo: max(1, runtime.config.hcMult))
            store(.hcAttnScale, l, resizeTo: max(1, runtime.config.hcMult))
            store(.hcAttnBase, l)
            store(.hcFfnFn, l, resizeTo: max(1, runtime.config.hcMult))
            store(.hcFfnScale, l, resizeTo: max(1, runtime.config.hcMult))
            store(.hcFfnBase, l)
        }
        self.vectors = bank

        // The embedding and the head are model-level, so their absence is a
        // different failure from a per-layer one.
        if let embed = weights.tensor(.tokenEmbedding) {
            let scale = weights.scaleTensor(.tokenEmbedding)
            if DeepSeekV41DevicePlan.embeddingKernel(tensor: embed, scale: scale) == nil {
                problems.append(
                    "embed.weight has dtype '\(embed.dtype)', which has no device lookup kernel"
                )
            }
        }
        if weights.tensor(.lmHead) == nil && !runtime.config.tieWordEmbeddings {
            problems.append("head.weight is missing and the embedding is not tied")
        }
        self.problems = problems
    }

    public var isRunnable: Bool { problems.isEmpty }

    /// One-line readiness report for the load log.
    public func describe() -> String {
        var lines = [
            "device forward: \(isRunnable ? "ready" : "not runnable") at "
                + "\(DeepSeekV41DevicePlan.gatherRowBudget(config: config)) gathered rows/position"
        ]
        lines.append(contentsOf: problems.map { "  \($0)" })
        return lines.joined(separator: "\n")
    }

    /// Drops the sequence so the next `forwardToken` starts at position 0.
    public func reset() {
        cache.reset()
    }

    /// Runs a prompt and returns the logits after the last position.
    public func prefill(tokens: [Int]) throws -> [Float] {
        var logits: [Float] = []
        for (offset, token) in tokens.enumerated() {
            logits = try forwardToken(token, position: offset)
        }
        return logits
    }

    // MARK: One position

    /// Appends one position and returns its logits.
    public func forwardToken(_ token: Int, position: Int) throws -> [Float] {
        guard problems.isEmpty else {
            throw DeepSeekV41EngineError.modelNotRunnable(problems)
        }
        guard position >= 0, position < contextLength else {
            throw DeepSeekV41EngineError.contextTooLong(position + 1, contextLength)
        }
        guard cache.storedPositions == position else {
            throw DeepSeekV41EngineError.outOfOrderPosition(position, cache.storedPositions)
        }

        try encodeEmbedding(token)
        for layer in plan {
            try runLayer(layer, at: position)
        }
        try finishBlock()

        var logits = [Float](repeating: 0, count: config.vocabSize)
        let count = min(config.vocabSize, workspace.logits.length / MemoryLayout<Float>.stride)
        let pointer = workspace.logits.contents().assumingMemoryBound(to: Float.self)
        logits.withUnsafeMutableBufferPointer { destination in
            destination.baseAddress?.update(from: pointer, count: count)
        }
        cache.noteStored(position + 1)
        return logits
    }

    // MARK: Block

    /// Embedding into `hidden`, then every hyper-connection stream starts as a copy
    /// of it (the reference's `Array(repeating: hidden, count: mult)`).
    private func encodeEmbedding(_ token: Int) throws {
        let commandBuffer = try makeCommandBuffer()
        let encoder = try makeEncoder(commandBuffer)

        let tokenPointer = workspace.tokenIDs.contents().assumingMemoryBound(to: UInt32.self)
        tokenPointer[0] = UInt32(max(0, token))

        guard let embed = weights.tensor(.tokenEmbedding),
              let kernel = DeepSeekV41DevicePlan.embeddingKernel(
                tensor: embed,
                scale: weights.scaleTensor(.tokenEmbedding)
              ),
              let embedBuffer = shardBuffer(embed) else {
            throw DeepSeekV41EngineError.missingTensor("embed.weight")
        }
        var weightOffset = UInt64(embed.byteOffset)
        var hiddenDim = UInt32(shapes.hidden)

        if kernel == DeepSeekV41Kernels.bf16Embedding {
            encoder.setComputePipelineState(pipelines.embeddingBF16)
            encoder.setBuffer(embedBuffer, offset: 0, index: 0)
            encoder.setBuffer(workspace.tokenIDs, offset: 0, index: 1)
            encoder.setBuffer(workspace.hidden, offset: 0, index: 2)
            encoder.setBytes(&weightOffset, length: 8, index: 3)
            encoder.setBytes(&hiddenDim, length: 4, index: 4)
        } else {
            guard let scale = weights.scaleTensor(.tokenEmbedding),
                  let scaleBuffer = shardBuffer(scale) else {
                throw DeepSeekV41EngineError.missingTensor("embed.weight scale")
            }
            var scaleOffset = UInt64(scale.byteOffset)
            encoder.setComputePipelineState(pipelines.embeddingMXFP8)
            encoder.setBuffer(embedBuffer, offset: 0, index: 0)
            encoder.setBuffer(workspace.tokenIDs, offset: 0, index: 1)
            encoder.setBuffer(workspace.hidden, offset: 0, index: 2)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 3)
            encoder.setBytes(&weightOffset, length: 8, index: 4)
            encoder.setBytes(&scaleOffset, length: 8, index: 5)
            encoder.setBytes(&hiddenDim, length: 4, index: 6)
        }
        encoder.dispatchThreads(
            MTLSize(width: shapes.hidden, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(shapes.hidden)
        )
        encoder.memoryBarrier(scope: .buffers)

        try encodeClear(encoder, buffer: workspace.streams, count: mult * shapes.hidden)
        try encodeStreamWrite(
            encoder,
            contribution: workspace.hidden,
            writeScales: nil
        )
        encoder.endEncoding()
        try commit(commandBuffer, stage: "embedding")
    }

    /// The block's output: collapse the streams, final norm, head.
    private func finishBlock() throws {
        let commandBuffer = try makeCommandBuffer()
        let encoder = try makeEncoder(commandBuffer)

        // The collapsed stream is stream 0, exactly like the reference's `collapse`.
        // It sits at byte 0 of the stream buffer, so no copy is needed.
        try encodeNorm(
            encoder,
            input: workspace.streams,
            gain: vectors.gain(.finalNorm, layer: nil, width: shapes.hidden),
            output: workspace.hidden,
            width: shapes.hidden
        )
        encoder.memoryBarrier(scope: .buffers)

        let head = weights.tensor(.lmHead)
            ?? (config.tieWordEmbeddings ? weights.tensor(.tokenEmbedding) : nil)
        guard let head else { throw DeepSeekV41EngineError.missingTensor("head.weight") }
        try encodeGemv(
            encoder,
            weight: head,
            scale: weights.scaleTensor(.lmHead),
            inDim: shapes.hidden,
            outDim: config.vocabSize,
            input: workspace.hidden,
            output: workspace.logits
        )
        encoder.endEncoding()
        try commit(commandBuffer, stage: "output head")
    }

    // MARK: One layer

    private func runLayer(_ layer: DeepSeekV41LayerPlan, at position: Int) throws {
        let commandBuffer = try makeCommandBuffer()
        let encoder = try makeEncoder(commandBuffer)

        for stage in DeepSeekV41DevicePlan.frontHalf {
            try encode(
                stage,
                layer: layer,
                position: position,
                encoder: encoder
            )
        }
        encoder.endEncoding()
        try commit(commandBuffer, stage: "layer \(layer.index) attention")

        // The mixture binds per-expert byte offsets, which are host values.
        let routing = readRouting()

        let expertCommand = try makeCommandBuffer()
        let expertEncoder = try makeEncoder(expertCommand)
        for stage in DeepSeekV41DevicePlan.backHalf {
            try encode(
                stage,
                layer: layer,
                position: position,
                encoder: expertEncoder,
                routing: routing
            )
        }
        expertEncoder.endEncoding()
        try commit(expertCommand, stage: "layer \(layer.index) experts")
    }

    /// Every stage the shared pipeline lists. Exhaustive on purpose: a stage added
    /// to `DeepSeekV41ForwardStages` fails to compile here rather than being
    /// silently skipped on device.
    private func encode(
        _ stage: DeepSeekV41ForwardStages.Stage,
        layer: DeepSeekV41LayerPlan,
        position: Int,
        encoder: MTLComputeCommandEncoder,
        routing: DeepSeekV41Routing? = nil
    ) throws {
        let index = layer.index
        let shape = DeepSeekV41AttentionShape.make(config: config, layer: layer)

        switch stage {
        case .mixAttentionStreams:
            try encodeStreamMix(
                encoder,
                readWeights: vectors.buffer(.hcAttnFn, layer: index)
                    ?? vectors.constant(1 / Float(mult), count: mult),
                scale: vectors.buffer(.hcAttnScale, layer: index),
                bias: vectors.buffer(.hcAttnBase, layer: index),
                output: workspace.normed
            )

        case .attnNorm:
            try encodeNorm(
                encoder,
                input: workspace.normed,
                gain: vectors.gain(.attnNorm, layer: index, width: shapes.hidden),
                output: workspace.normed,
                width: shapes.hidden
            )

        case .queryDown:
            // A checkpoint with no `wq_a` feeds the normed hidden state straight to
            // `wq_b`, which is what the reference does when the down projection is
            // absent.
            if config.qLoraRank > 0, let down = weights.tensor(.attnWqA, layer: index) {
                try encodeGemv(
                    encoder,
                    weight: down,
                    scale: weights.scaleTensor(.attnWqA, layer: index),
                    inDim: shapes.hidden,
                    outDim: max(1, config.qLoraRank),
                    input: workspace.normed,
                    output: workspace.reduced
                )
            }

        case .queryNorm:
            // A rank-normed down projection only exists with a rank; a checkpoint
            // that publishes the norm without `wq_a` still runs the norm (with a
            // ones gain, because the reference treats a gain whose width disagrees
            // with the input as absent).
            if config.qLoraRank > 0 {
                let hasDown = weights.tensor(.attnWqA, layer: index) != nil
                let width = hasDown ? max(1, config.qLoraRank) : shapes.hidden
                try encodeNorm(
                    encoder,
                    input: hasDown ? workspace.reduced : workspace.normed,
                    gain: vectors.gain(.attnWqANorm, layer: index, width: width),
                    output: hasDown ? workspace.reduced : workspace.normed,
                    width: width
                )
            }

        case .queryUp:
            let hasDown = config.qLoraRank > 0 && weights.tensor(.attnWqA, layer: index) != nil
            guard let up = weights.tensor(.attnWqB, layer: index) else {
                throw DeepSeekV41EngineError.missingTensor("attn.wq_b")
            }
            try encodeGemv(
                encoder,
                weight: up,
                scale: weights.scaleTensor(.attnWqB, layer: index),
                inDim: hasDown ? max(1, config.qLoraRank) : shapes.hidden,
                outDim: shapes.queryWidth,
                input: hasDown ? workspace.reduced : workspace.normed,
                output: workspace.query
            )

        case .ropeQuery:
            try encodeRope(
                encoder,
                vector: workspace.query,
                rowWidth: shapes.queryWidth,
                headDim: shape.qkHeadDim,
                table: layer.ropeTable,
                position: position
            )

        case .latentProject:
            guard let wkv = weights.tensor(.attnWkv, layer: index) else {
                throw DeepSeekV41EngineError.missingTensor("attn.wkv")
            }
            try encodeGemv(
                encoder,
                weight: wkv,
                scale: weights.scaleTensor(.attnWkv, layer: index),
                inDim: shapes.hidden,
                outDim: shapes.latentWidth,
                input: workspace.normed,
                output: workspace.latent
            )

        case .latentNorm:
            try encodeNorm(
                encoder,
                input: workspace.latent,
                gain: vectors.gain(.attnKvNorm, layer: index, width: shapes.latentWidth),
                output: workspace.latent,
                width: shapes.latentWidth
            )

        case .ropeLatent:
            try encodeRope(
                encoder,
                vector: workspace.latent,
                rowWidth: shapes.latentWidth,
                headDim: shape.rotaryDim,
                table: layer.ropeTable,
                position: position
            )

        case .writeLatent:
            try encodeStore(
                encoder,
                latent: workspace.latent,
                cacheBuffer: cache.windowBuffer,
                byteOffset: DeepSeekV41CacheAddressing.windowByteOffset(
                    layer: index,
                    position: position,
                    layout: cache.layout
                ) ?? -1,
                physicalRow: 0
            )

        case .compressPool:
            try encodePoolIfComplete(layer: layer, position: position, encoder: encoder)

        case .attention:
            try encodeAttention(layer: layer, position: position, encoder: encoder, shape: shape)

        case .outputDown:
            if config.oLoraRank > 0, let down = weights.tensor(.attnWoA, layer: index) {
                try encodeGemv(
                    encoder,
                    weight: down,
                    scale: weights.scaleTensor(.attnWoA, layer: index),
                    inDim: shapes.attentionOutputWidth,
                    outDim: max(1, config.oLoraRank),
                    input: workspace.attentionOut,
                    output: workspace.reduced
                )
            }

        case .outputUp:
            guard let up = weights.tensor(.attnWoB, layer: index) else {
                throw DeepSeekV41EngineError.missingTensor("attn.wo_b")
            }
            let hasDown = config.oLoraRank > 0 && weights.tensor(.attnWoA, layer: index) != nil
            try encodeGemv(
                encoder,
                weight: up,
                scale: weights.scaleTensor(.attnWoB, layer: index),
                inDim: hasDown ? max(1, config.oLoraRank) : shapes.attentionOutputWidth,
                outDim: shapes.hidden,
                input: hasDown ? workspace.reduced : workspace.attentionOut,
                output: workspace.output
            )

        case .writeAttentionStreams:
            try encodeStreamWrite(
                encoder,
                contribution: workspace.output,
                writeScales: vectors.buffer(.hcAttnScale, layer: index)
            )

        case .mixFFNStreams:
            try encodeStreamMix(
                encoder,
                readWeights: vectors.buffer(.hcFfnFn, layer: index)
                    ?? vectors.constant(1 / Float(mult), count: mult),
                scale: vectors.buffer(.hcFfnScale, layer: index),
                bias: vectors.buffer(.hcFfnBase, layer: index),
                output: workspace.normed
            )

        case .ffnNorm:
            try encodeNorm(
                encoder,
                input: workspace.normed,
                gain: vectors.gain(.ffnNorm, layer: index, width: shapes.hidden),
                output: workspace.normed,
                width: shapes.hidden
            )

        case .engram:
            // Reported at load for any layer that carries a table, so reaching here
            // with one configured means the plan was bypassed.
            if layer.engramTable != nil {
                throw DeepSeekV41EngineError.unsupportedStage("engram")
            }

        case .router:
            guard let gate = weights.tensor(.routerWeight, layer: index) else {
                throw DeepSeekV41EngineError.missingTensor("ffn.gate")
            }
            try encodeGemv(
                encoder,
                weight: gate,
                scale: weights.scaleTensor(.routerWeight, layer: index),
                inDim: shapes.hidden,
                outDim: config.numRoutedExperts,
                input: workspace.normed,
                output: workspace.routerLogits
            )
            encoder.memoryBarrier(scope: .buffers)
            try encodeRouter(encoder, layer: index)

        case .sharedExpert:
            try encodeSharedExpert(layer: layer, encoder: encoder)

        case .routedExperts:
            try encodeRoutedExperts(layer: layer, encoder: encoder, routing: routing)

        case .writeFFNStreams:
            try encodeStreamWrite(
                encoder,
                contribution: workspace.moeOut,
                writeScales: vectors.buffer(.hcFfnScale, layer: index)
            )

        case .collapseStreams:
            // The block collapses once, after the last layer, so a later stage can
            // never observe a half-collapsed state. The stage stays in the list
            // because both backends order their work by it.
            break
        }
    }

    // MARK: Attention stages

    /// Pools a completed block of latents into the owner's compressed row.
    ///
    /// A row exists only when the whole block has been seen, i.e. when
    /// `position + 1` is a multiple of the ratio. Pooling on later positions would
    /// append the same row again and let the cache grow with the context rather
    /// than with the context / ratio.
    private func encodePoolIfComplete(
        layer: DeepSeekV41LayerPlan,
        position: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        guard layer.ownsCompressedKV else { return }
        let ratio = max(1, layer.compressRatio)
        guard (position + 1) % ratio == 0 else { return }
        let rows = DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: position, ratio: ratio)
        guard rows > 0 else { return }
        let block = DeepSeekV41CacheAddressing.poolRange(row: rows - 1, ratio: ratio)
        guard block.upperBound <= position + 1 else { return }
        guard ratio <= gatherRows else {
            throw DeepSeekV41EngineError.allocationFailed(
                "a \(ratio)-row pool block does not fit the \(gatherRows)-row gather buffer"
            )
        }
        guard let windowBuffer = cache.windowBuffer,
              let base = cache.layout.windowBase[layer.index] else {
            throw DeepSeekV41EngineError.allocationFailed("the window ring")
        }
        let destinationRow = DeepSeekV41CacheAddressing.compressedByteOffset(
            ownerLayer: layer.index,
            row: rows - 1,
            layout: cache.layout
        )
        guard let destinationRow, let compressedBuffer = cache.compressedBuffer else {
            throw DeepSeekV41EngineError.allocationFailed("the compressed cache")
        }

        // Gather the block out of the ring, then pool it. The gate logits are the
        // reference's zeros: `attn.compressor.wgate` is a projection of the latent
        // whose combination rule is `DeepSeekV41Conformance.compressorPooling`
        // (unverified), so the device path pools with uniform gates exactly like the
        // reference instead of inventing a different weighting.
        try encodeRowIndices(
            Array(block).map { DeepSeekV41CacheAddressing.windowRow(for: $0, geometry: geometry) },
            at: 0
        )
        try encodeGather(
            encoder,
            cacheBuffer: windowBuffer,
            byteOffset: base,
            rowCount: ratio,
            destinationRowOffset: 0
        )
        try encodeClear(encoder, buffer: workspace.gates, count: ratio)

        encoder.setComputePipelineState(pipelines.poolLatents)
        encoder.setBuffer(workspace.gathered, offset: 0, index: 0)
        encoder.setBuffer(workspace.gates, offset: 0, index: 1)
        encoder.setBuffer(workspace.pooled, offset: 0, index: 2)
        var width = UInt32(shapes.latentWidth)
        var blockSize = UInt32(ratio)
        encoder.setBytes(&width, length: 4, index: 3)
        encoder.setBytes(&blockSize, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: shapes.latentWidth, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(shapes.latentWidth)
        )
        encoder.memoryBarrier(scope: .buffers)

        try encodeStore(
            encoder,
            latent: workspace.pooled,
            cacheBuffer: compressedBuffer,
            byteOffset: destinationRow,
            physicalRow: 0
        )
    }

    /// Gathers the window plus the selected compressed rows into one dense block and
    /// attends over it. Gathering first is what lets a single kernel serve both
    /// halves of CSA2.
    private func encodeAttention(
        layer: DeepSeekV41LayerPlan,
        position: Int,
        encoder: MTLComputeCommandEncoder,
        shape: DeepSeekV41AttentionShape
    ) throws {
        let rowPlan = DeepSeekV41CacheAddressing.attentionRowPlan(
            layer: layer,
            position: position,
            config: config,
            plan: plan,
            geometry: geometry,
            ownerRowLimit: geometry.compressedRows
        )
        let windowCount = rowPlan.windowRows.count
        let compressedCount = rowPlan.compressedRows.count
        let rowCount = windowCount + compressedCount
        guard rowCount > 0 else {
            throw DeepSeekV41EngineError.allocationFailed("attention needed zero rows")
        }
        guard rowCount <= gatherRows else {
            throw DeepSeekV41EngineError.allocationFailed(
                "position \(position) reads \(rowCount) rows but the gather buffer holds \(gatherRows)"
            )
        }
        guard let windowBuffer = cache.windowBuffer,
              let windowBase = cache.layout.windowBase[layer.index] else {
            throw DeepSeekV41EngineError.allocationFailed("the window ring")
        }

        var cursor = 0
        try encodeRowIndices(rowPlan.windowRows, at: 0)
        cursor += windowCount
        try encodeGather(
            encoder,
            cacheBuffer: windowBuffer,
            byteOffset: windowBase,
            rowCount: windowCount,
            destinationRowOffset: 0
        )
        if compressedCount > 0 {
            guard let compressedBuffer = cache.compressedBuffer,
                  let compressedBase = cache.layout.compressedBase[layer.compressedKVSource] else {
                throw DeepSeekV41EngineError.allocationFailed("the compressed cache")
            }
            try encodeRowIndices(rowPlan.compressedRows, at: cursor)
            try encodeGather(
                encoder,
                cacheBuffer: compressedBuffer,
                byteOffset: compressedBase,
                rowCount: compressedCount,
                destinationRowOffset: windowCount
            )
        }
        encoder.memoryBarrier(scope: .buffers)

        encoder.setComputePipelineState(pipelines.attention)
        encoder.setBuffer(workspace.query, offset: 0, index: 0)
        encoder.setBuffer(workspace.gathered, offset: 0, index: 1)
        if let sink = vectors.buffer(.attnSink, layer: layer.index),
           sink.length >= shape.numHeads * MemoryLayout<Float>.stride {
            encoder.setBuffer(sink, offset: 0, index: 2)
        }
        encoder.setBuffer(workspace.attentionOut, offset: 0, index: 3)
        var rows = UInt32(rowCount)
        var heads = UInt32(shape.numHeads)
        var qkHeadDim = UInt32(shape.qkHeadDim)
        var valueDim = UInt32(shape.vHeadDim)
        var width = UInt32(shapes.latentWidth)
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&heads, length: 4, index: 5)
        encoder.setBytes(&qkHeadDim, length: 4, index: 6)
        encoder.setBytes(&valueDim, length: 4, index: 7)
        encoder.setBytes(&width, length: 4, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: shape.numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(shape.numHeads)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    // MARK: Feed-forward stages

    private func encodeSharedExpert(
        layer: DeepSeekV41LayerPlan,
        encoder: MTLComputeCommandEncoder
    ) throws {
        // The shared expert runs for every token, unscaled, and accumulates into the
        // same vector the routed experts do.
        guard let w1 = weights.tensor(.sharedExpertW1, layer: layer.index),
              let w3 = weights.tensor(.sharedExpertW3, layer: layer.index),
              let w2 = weights.tensor(.sharedExpertW2, layer: layer.index) else {
            throw DeepSeekV41EngineError.missingTensor("ffn.shared_experts")
        }
        let width = shapes.sharedIntermediate
        guard width <= max(1, config.intermediateSize) || config.intermediateSize == 0 else {
            throw DeepSeekV41EngineError.allocationFailed("the shared expert intermediate")
        }
        let buffer = workspace.sharedIntermediate
        guard buffer.length >= width * MemoryLayout<Float>.stride else {
            throw DeepSeekV41EngineError.allocationFailed("the shared expert intermediate")
        }
        try encodeSwiGLU(
            encoder,
            gate: w1,
            gateScale: weights.scaleTensor(.sharedExpertW1, layer: layer.index),
            up: w3,
            upScale: weights.scaleTensor(.sharedExpertW3, layer: layer.index),
            inDim: shapes.hidden,
            outDim: width,
            input: workspace.normed,
            output: buffer
        )
        try encodeDown(
            encoder,
            weight: w2,
            scale: weights.scaleTensor(.sharedExpertW2, layer: layer.index),
            inDim: width,
            outDim: shapes.hidden,
            activation: buffer,
            output: workspace.moeOut,
            weightToAdd: 1
        )
    }

    private func encodeRoutedExperts(
        layer: DeepSeekV41LayerPlan,
        encoder: MTLComputeCommandEncoder,
        routing: DeepSeekV41Routing?
    ) throws {
        // The mixture always starts from zero: every expert accumulates into the
        // same vector.
        try encodeClear(encoder, buffer: workspace.moeOut, count: shapes.hidden)
        guard let routing, !routing.expertIDs.isEmpty else { return }
        let buffer = workspace.intermediate
        for (rank, expert) in routing.expertIDs.enumerated() {
            let weight = rank < routing.weights.count ? routing.weights[rank] : 0
            if weight == 0 { continue }
            guard let w1 = weights.tensor(.expertW1, layer: layer.index, expert: expert),
                  let w3 = weights.tensor(.expertW3, layer: layer.index, expert: expert),
                  let w2 = weights.tensor(.expertW2, layer: layer.index, expert: expert) else {
                throw DeepSeekV41EngineError.missingTensor(
                    "layers.\(layer.index).ffn.experts.\(expert)"
                )
            }
            try encodeSwiGLU(
                encoder,
                gate: w1,
                gateScale: weights.scaleTensor(.expertW1, layer: layer.index, expert: expert),
                up: w3,
                upScale: weights.scaleTensor(.expertW3, layer: layer.index, expert: expert),
                inDim: shapes.hidden,
                outDim: shapes.expertIntermediate,
                input: workspace.normed,
                output: buffer
            )
            try encodeDown(
                encoder,
                weight: w2,
                scale: weights.scaleTensor(.expertW2, layer: layer.index, expert: expert),
                inDim: shapes.expertIntermediate,
                outDim: shapes.hidden,
                activation: buffer,
                output: workspace.moeOut,
                weightToAdd: weight
            )
        }
    }

    /// The router's top-k, read back so the mixture can bind each expert's byte
    /// offset. `normalise` is on: the checkpoint scales the normalized scores by
    /// `routed_scaling_factor`, which the kernel applies after normalizing.
    private func encodeRouter(
        _ encoder: MTLComputeCommandEncoder,
        layer: Int
    ) throws {
        encoder.setComputePipelineState(pipelines.router)
        encoder.setBuffer(workspace.routerLogits, offset: 0, index: 0)
        if let bias = vectors.buffer(.routerBias, layer: layer) {
            encoder.setBuffer(bias, offset: 0, index: 1)
        }
        encoder.setBuffer(workspace.expertIndices, offset: 0, index: 2)
        encoder.setBuffer(workspace.expertWeights, offset: 0, index: 3)
        var expertCount = UInt32(config.numRoutedExperts)
        var topK = UInt32(max(1, config.numActivatedExperts))
        var scaling = config.routedScalingFactor
        var normalise: UInt32 = 1
        encoder.setBytes(&expertCount, length: 4, index: 4)
        encoder.setBytes(&topK, length: 4, index: 5)
        encoder.setBytes(&scaling, length: 4, index: 6)
        encoder.setBytes(&normalise, length: 4, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
    }

    /// The routing the router kernel wrote. `bias` is per layer, so it is bound from
    /// the layer's slot by the caller before this runs.
    private func readRouting() -> DeepSeekV41Routing {
        let count = max(0, min(config.numActivatedExperts, workspace.expertIndices.length / 4))
        let indexPointer = workspace.expertIndices.contents().assumingMemoryBound(to: UInt32.self)
        let weightPointer = workspace.expertWeights.contents().assumingMemoryBound(to: Float.self)
        var ids: [Int] = []
        var weights: [Float] = []
        ids.reserveCapacity(count)
        weights.reserveCapacity(count)
        for rank in 0..<count {
            ids.append(Int(indexPointer[rank]))
            weights.append(weightPointer[rank])
        }
        return DeepSeekV41Routing(expertIDs: ids, weights: weights, scores: weights)
    }

    // MARK: Encoders

    private func shardBuffer(_ tensor: DeepSeekV41DeviceTensor?) -> MTLBuffer? {
        guard let tensor else { return nil }
        return shardBuffers[tensor.shardIndex]
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let buffer = commandQueue.makeCommandBuffer() else {
            throw DeepSeekV41EngineError.allocationFailed("a command buffer")
        }
        return buffer
    }

    private func makeEncoder(_ commandBuffer: MTLCommandBuffer) throws -> MTLComputeCommandEncoder {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeepSeekV41EngineError.allocationFailed("a compute encoder")
        }
        return encoder
    }

    private func commit(_ commandBuffer: MTLCommandBuffer, stage: String) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DeepSeekV41EngineError.commandFailed("\(stage): \(error)")
        }
    }

    private func threadgroup(_ width: Int) -> MTLSize {
        MTLSize(width: max(1, min(width, 64)), height: 1, depth: 1)
    }

    private func encodeClear(
        _ encoder: MTLComputeCommandEncoder,
        buffer: MTLBuffer,
        count: Int
    ) throws {
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.clear)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(count)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeNorm(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        gain: MTLBuffer?,
        output: MTLBuffer,
        width: Int
    ) throws {
        guard width > 0 else { return }
        guard let gain else {
            throw DeepSeekV41EngineError.allocationFailed("a \(width)-wide norm gain")
        }
        encoder.setComputePipelineState(pipelines.rmsNormGain)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(gain, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var dim = UInt32(width)
        var epsilon = config.rmsNormEps
        encoder.setBytes(&dim, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        let lanes = max(32, min(256, pipelines.rmsNormGain.maxTotalThreadsPerThreadgroup))
        encoder.setThreadgroupMemoryLength((lanes / 32) * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreads(
            MTLSize(width: width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeGemv(
        _ encoder: MTLComputeCommandEncoder,
        weight: DeepSeekV41DeviceTensor,
        scale: DeepSeekV41DeviceTensor?,
        inDim: Int,
        outDim: Int,
        input: MTLBuffer,
        output: MTLBuffer
    ) throws {
        guard inDim > 0, outDim > 0 else {
            throw DeepSeekV41EngineError.allocationFailed("a \(outDim)x\(inDim) projection")
        }
        guard let weightBuffer = shardBuffer(weight) else {
            throw DeepSeekV41EngineError.missingTensor("a projection's shard")
        }
        let format = DeepSeekV41DevicePlan.linearFormat(tensor: weight, scale: scale)
        guard DeepSeekV41DevicePlan.gemvKernel(format) != nil else {
            throw DeepSeekV41EngineError.unsupportedDtype("a projection", weight.dtype)
        }
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)

        switch format {
        case .fp8Block32, .mxfp4:
            guard let scale, let scaleBuffer = shardBuffer(scale) else {
                throw DeepSeekV41EngineError.missingTensor("a projection's scale companion")
            }
            encoder.setComputePipelineState(
                format == .fp8Block32 ? pipelines.fp8Block32Gemv : pipelines.mxfp4Gemv
            )
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(input, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var weightOffset = UInt64(weight.byteOffset)
            var scaleOffset = UInt64(scale.byteOffset)
            encoder.setBytes(&weightOffset, length: 8, index: 4)
            encoder.setBytes(&scaleOffset, length: 8, index: 5)
            encoder.setBytes(&inDimension, length: 4, index: 6)
            encoder.setBytes(&outDimension, length: 4, index: 7)
        case .bf16:
            encoder.setComputePipelineState(pipelines.bf16Gemv)
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(input, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var weightOffset = UInt64(weight.byteOffset)
            encoder.setBytes(&weightOffset, length: 8, index: 3)
            encoder.setBytes(&inDimension, length: 4, index: 4)
            encoder.setBytes(&outDimension, length: 4, index: 5)
        case .unsupported:
            throw DeepSeekV41EngineError.unsupportedDtype("a projection", weight.dtype)
        }
        encoder.dispatchThreads(
            MTLSize(width: outDim, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(outDim)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeSwiGLU(
        _ encoder: MTLComputeCommandEncoder,
        gate: DeepSeekV41DeviceTensor,
        gateScale: DeepSeekV41DeviceTensor?,
        up: DeepSeekV41DeviceTensor,
        upScale: DeepSeekV41DeviceTensor?,
        inDim: Int,
        outDim: Int,
        input: MTLBuffer,
        output: MTLBuffer
    ) throws {
        let format = DeepSeekV41DevicePlan.linearFormat(tensor: gate, scale: gateScale)
        guard DeepSeekV41DevicePlan.swigluKernel(format, swigluLimit: swigluLimit) != nil else {
            throw DeepSeekV41EngineError.unsupportedDtype("a feed-forward gate", gate.dtype)
        }
        guard let gateBuffer = shardBuffer(gate), let upBuffer = shardBuffer(up) else {
            throw DeepSeekV41EngineError.missingTensor("a feed-forward gate")
        }
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)

        switch format {
        case .fp8Block32, .mxfp4:
            guard let gateScale, let upScale,
                  let gateScaleBuffer = shardBuffer(gateScale),
                  let upScaleBuffer = shardBuffer(upScale) else {
                throw DeepSeekV41EngineError.missingTensor("a feed-forward scale companion")
            }
            encoder.setComputePipelineState(
                format == .fp8Block32 ? pipelines.fp8SwiGLU : pipelines.mxfp4SwiGLU
            )
            encoder.setBuffer(gateBuffer, offset: 0, index: 0)
            encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(upBuffer, offset: 0, index: 2)
            encoder.setBuffer(upScaleBuffer, offset: 0, index: 3)
            encoder.setBuffer(input, offset: 0, index: 4)
            encoder.setBuffer(output, offset: 0, index: 5)
            var gateWeightOffset = UInt64(gate.byteOffset)
            var gateScaleOffset = UInt64(gateScale.byteOffset)
            var upWeightOffset = UInt64(up.byteOffset)
            var upScaleOffset = UInt64(upScale.byteOffset)
            var limit = swigluLimit
            encoder.setBytes(&gateWeightOffset, length: 8, index: 6)
            encoder.setBytes(&gateScaleOffset, length: 8, index: 7)
            encoder.setBytes(&upWeightOffset, length: 8, index: 8)
            encoder.setBytes(&upScaleOffset, length: 8, index: 9)
            encoder.setBytes(&inDimension, length: 4, index: 10)
            encoder.setBytes(&outDimension, length: 4, index: 11)
            encoder.setBytes(&limit, length: 4, index: 12)
        case .bf16:
            encoder.setComputePipelineState(pipelines.bf16SwiGLU)
            encoder.setBuffer(gateBuffer, offset: 0, index: 0)
            encoder.setBuffer(upBuffer, offset: 0, index: 1)
            encoder.setBuffer(input, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var gateWeightOffset = UInt64(gate.byteOffset)
            var upWeightOffset = UInt64(up.byteOffset)
            encoder.setBytes(&gateWeightOffset, length: 8, index: 4)
            encoder.setBytes(&upWeightOffset, length: 8, index: 5)
            encoder.setBytes(&inDimension, length: 4, index: 6)
            encoder.setBytes(&outDimension, length: 4, index: 7)
        case .unsupported:
            throw DeepSeekV41EngineError.unsupportedDtype("a feed-forward gate", gate.dtype)
        }
        encoder.dispatchThreads(
            MTLSize(width: outDim, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(outDim)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeDown(
        _ encoder: MTLComputeCommandEncoder,
        weight: DeepSeekV41DeviceTensor,
        scale: DeepSeekV41DeviceTensor?,
        inDim: Int,
        outDim: Int,
        activation: MTLBuffer,
        output: MTLBuffer,
        weightToAdd: Float
    ) throws {
        let format = DeepSeekV41DevicePlan.linearFormat(tensor: weight, scale: scale)
        guard DeepSeekV41DevicePlan.downKernel(format) != nil else {
            throw DeepSeekV41EngineError.unsupportedDtype("a feed-forward down projection", weight.dtype)
        }
        guard let weightBuffer = shardBuffer(weight) else {
            throw DeepSeekV41EngineError.missingTensor("a feed-forward down projection")
        }
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)
        var routing = weightToAdd

        switch format {
        case .fp8Block32, .mxfp4:
            guard let scale, let scaleBuffer = shardBuffer(scale) else {
                throw DeepSeekV41EngineError.missingTensor("a down projection's scale companion")
            }
            encoder.setComputePipelineState(
                format == .fp8Block32 ? pipelines.fp8DownAccumulate : pipelines.mxfp4DownAccumulate
            )
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(activation, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var weightOffset = UInt64(weight.byteOffset)
            var scaleOffset = UInt64(scale.byteOffset)
            encoder.setBytes(&weightOffset, length: 8, index: 4)
            encoder.setBytes(&scaleOffset, length: 8, index: 5)
            encoder.setBytes(&inDimension, length: 4, index: 6)
            encoder.setBytes(&outDimension, length: 4, index: 7)
            encoder.setBytes(&routing, length: 4, index: 8)
        case .bf16:
            encoder.setComputePipelineState(pipelines.bf16DownAccumulate)
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(activation, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var weightOffset = UInt64(weight.byteOffset)
            encoder.setBytes(&weightOffset, length: 8, index: 3)
            encoder.setBytes(&inDimension, length: 4, index: 4)
            encoder.setBytes(&outDimension, length: 4, index: 5)
            encoder.setBytes(&routing, length: 4, index: 6)
        case .unsupported:
            throw DeepSeekV41EngineError.unsupportedDtype("a down projection", weight.dtype)
        }
        encoder.dispatchThreads(
            MTLSize(width: outDim, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(outDim)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeStreamMix(
        _ encoder: MTLComputeCommandEncoder,
        readWeights: MTLBuffer?,
        scale: MTLBuffer?,
        bias: MTLBuffer?,
        output: MTLBuffer
    ) throws {
        guard let readWeights else {
            throw DeepSeekV41EngineError.allocationFailed("a hyper-connection read vector")
        }
        encoder.setComputePipelineState(pipelines.hcMix)
        encoder.setBuffer(workspace.streams, offset: 0, index: 0)
        encoder.setBuffer(readWeights, offset: 0, index: 1)
        if let scale { encoder.setBuffer(scale, offset: 0, index: 2) }
        if let bias { encoder.setBuffer(bias, offset: 0, index: 3) }
        encoder.setBuffer(output, offset: 0, index: 4)
        var width = UInt32(shapes.hidden)
        var streams = UInt32(mult)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&streams, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: shapes.hidden, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(shapes.hidden)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeStreamWrite(
        _ encoder: MTLComputeCommandEncoder,
        contribution: MTLBuffer,
        writeScales: MTLBuffer?
    ) throws {
        encoder.setComputePipelineState(pipelines.hcWrite)
        encoder.setBuffer(workspace.streams, offset: 0, index: 0)
        encoder.setBuffer(contribution, offset: 0, index: 1)
        if let writeScales { encoder.setBuffer(writeScales, offset: 0, index: 2) }
        var width = UInt32(shapes.hidden)
        var streams = UInt32(mult)
        encoder.setBytes(&width, length: 4, index: 3)
        encoder.setBytes(&streams, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: shapes.hidden, height: mult, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroup(shapes.hidden).width, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeRope(
        _ encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer,
        rowWidth: Int,
        headDim: Int,
        table: DeepSeekV41RopeTable,
        position: Int
    ) throws {
        guard let frequencies = frequencies[table] else {
            throw DeepSeekV41EngineError.allocationFailed("a RoPE frequency table")
        }
        let pairCount = (headDim == 0 ? rowWidth : headDim) / 2
        guard pairCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.rope)
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(frequencies, offset: 0, index: 1)
        var width = UInt32(rowWidth)
        var localHeadDim = UInt32(max(0, headDim))
        var pairs = UInt32(min(pairCount, frequencies.length / MemoryLayout<Float>.stride))
        var base = UInt32(max(0, position))
        var batch: UInt32 = 1
        encoder.setBytes(&width, length: 4, index: 2)
        encoder.setBytes(&localHeadDim, length: 4, index: 3)
        encoder.setBytes(&pairs, length: 4, index: 4)
        encoder.setBytes(&base, length: 4, index: 5)
        encoder.setBytes(&batch, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Int(pairs), height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(Int(pairs))
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    /// `byteOffset` is a byte offset into the cache buffer; the kernels address half
    /// elements, so it is divided here rather than at every call site.
    private func encodeStore(
        _ encoder: MTLComputeCommandEncoder,
        latent: MTLBuffer,
        cacheBuffer: MTLBuffer?,
        byteOffset: Int,
        physicalRow: Int
    ) throws {
        guard byteOffset >= 0, let cacheBuffer else {
            throw DeepSeekV41EngineError.allocationFailed("a cache row")
        }
        let elementOffset = byteOffset / max(1, geometry.precision.valueBytesPerElement)
        encoder.setComputePipelineState(pipelines.storeLatent)
        encoder.setBuffer(latent, offset: 0, index: 0)
        encoder.setBuffer(cacheBuffer, offset: 0, index: 1)
        var offset = UInt64(elementOffset)
        var width = UInt32(shapes.latentWidth)
        var row = UInt32(max(0, physicalRow))
        encoder.setBytes(&offset, length: 8, index: 2)
        encoder.setBytes(&width, length: 4, index: 3)
        encoder.setBytes(&row, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: shapes.latentWidth, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(shapes.latentWidth)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeGather(
        _ encoder: MTLComputeCommandEncoder,
        cacheBuffer: MTLBuffer,
        byteOffset: Int,
        rowCount: Int,
        destinationRowOffset: Int
    ) throws {
        guard rowCount > 0 else { return }
        let elementOffset = byteOffset / max(1, geometry.precision.valueBytesPerElement)
        encoder.setComputePipelineState(pipelines.gatherLatents)
        encoder.setBuffer(cacheBuffer, offset: 0, index: 0)
        encoder.setBuffer(workspace.rowIndices, offset: 0, index: 1)
        encoder.setBuffer(workspace.gathered, offset: 0, index: 2)
        var offset = UInt64(elementOffset)
        var width = UInt32(shapes.latentWidth)
        var rows = UInt32(rowCount)
        var destination = UInt64(max(0, destinationRowOffset))
        encoder.setBytes(&offset, length: 8, index: 3)
        encoder.setBytes(&width, length: 4, index: 4)
        encoder.setBytes(&rows, length: 4, index: 5)
        encoder.setBytes(&destination, length: 8, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: shapes.latentWidth, height: rowCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroup(shapes.latentWidth).width, height: 1, depth: 1)
        )
    }

    /// Writes the gather's row list. The indices are cache rows (already ring
    /// mapped), which is why this is a host fill: the ring mapping is a pure
    /// function of the position.
    private func encodeRowIndices(_ rows: [Int], at offset: Int) throws {
        guard !rows.isEmpty else { return }
        guard offset + rows.count <= workspace.rowIndices.length / MemoryLayout<UInt32>.stride else {
            throw DeepSeekV41EngineError.allocationFailed(
                "\(offset + rows.count) gather rows against a "
                    + "\(workspace.rowIndices.length / MemoryLayout<UInt32>.stride)-row buffer"
            )
        }
        let pointer = workspace.rowIndices.contents().assumingMemoryBound(to: UInt32.self)
        for (index, row) in rows.enumerated() {
            pointer[offset + index] = UInt32(max(0, row))
        }
    }
}
