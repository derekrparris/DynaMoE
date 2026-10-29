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
//    * the Engram rows. The table is a ~100 GB pread stream, not a buffer, and the
//      hash ids for a position are known before its layer runs, so the rows are
//      read host-side into a per-table staging block and everything after that
//      (`wkv`, the gate, the residual add) is a kernel. The stage order puts it
//      first in the layer because `Engram.forward` writes into the residual stream
//      ahead of the attention sub-layer.
//
//  Synchronization follows from that: the only point in a token where the CPU has
//  to see device output is the routing readback, so a layer commits its back half
//  and moves on. One command queue runs buffers in commit order, so the next
//  layer's front half is ordered behind the previous layer's experts without a
//  wait of its own; `settlePending` catches up, and surfaces any failure, at each
//  readback and at the end of the token. Waiting on every buffer instead makes the
//  host the bottleneck at two stalls per layer.
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
    ///
    /// A quantized weight can also be published as RAW BYTES rather than as a typed
    /// float format. This checkpoint's routed experts are exactly that: the
    /// safetensors header says `I8` and the row width is already packed, so the
    /// dtype alone cannot say whether a row holds one value per byte (E4M3) or two
    /// (E2M1). The published row width against the scale row does say it, because
    /// both are derived from the same logical input width: a packed MXFP4 row is
    /// `in/2` bytes and its scale row `in/32` entries, i.e. exactly 16 scale entries
    /// per weight byte, while an E4M3 row is `in` bytes, i.e. 32. Reading that ratio
    /// is what keeps the MXFP4 kernels from being dispatched on an int8 matrix (a
    /// factor-of-two error in every row) and the reverse.
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
        // Unannotated bytes with a block scale are a quantized weight whose dtype the
        // writer did not spell out.
        let isRawByte = dtype == "I8" || dtype == "U8" || dtype == "INT8"
            || dtype.contains("UINT8")
        if isRawByte, let scale, scale.isScaleCompanion, !scale.isBF16, !scale.isF16,
           scale.columns > 0 {
            if tensor.columns == scale.columns * 16 { return .mxfp4 }
            if tensor.columns == scale.columns * 32 { return .fp8Block32 }
        }
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

    /// Grid one RMSNorm row needs. `dsv41_rmsnorm_f32_gain` treats
    /// `threadgroup_position_in_grid` as the row index, so exactly one threadgroup
    /// may be dispatched per row: a grid of `width` threads would let Metal launch
    /// several threadgroups for a wide row and the extras would read and write past
    /// the row they were meant to normalize.
    public static func normDispatch(
        width: Int,
        lanes: Int
    ) -> (threadgroups: MTLSize, threadsPerThreadgroup: MTLSize) {
        let laneCount = max(32, lanes)
        return (
            MTLSize(width: 1, height: 1, depth: 1),
            MTLSize(width: laneCount, height: 1, depth: 1)
        )
    }

    /// Rows one position may gather: its window plus the compressed rows the
    /// addressing rule reads.
    ///
    /// A layer that runs an indexer keeps `min(index_topk, produced)` compressed
    /// rows, so `index_topk` is the bound there — it is not "0 means every row".
    /// A compressed layer with no indexer (`index_topk == 0`) falls back to the
    /// "all rows the candidate mask can expose" rule, which the block counts
    /// bound; pricing the indexed case off those counts instead allocated ~25×
    /// the buffer this checkpoint needs (16,512 rows for what is really 640) and
    /// made the readiness line misleading.
    public static func gatherRowBudget(config: DeepSeekV41Config) -> Int {
        let window = max(1, config.windowSize)
        if config.indexTopK > 0 { return window + config.indexTopK }
        let candidates = max(1, config.candidateTopKBlocks) * max(1, config.candidateBlockSize)
        return window + candidates
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
            let table = layer.engramTable ?? -1
            guard table >= 0, table < config.engramTables.count else {
                problems.append(
                    "layers.\(layer.index): Engram table index \(table) has no entry in "
                        + "engram_layer_ids (\(config.engramTables.count) tables)"
                )
                continue
            }
            let spec = config.engramTables[table]
            if spec.maxNgramSize < 2 || spec.headCount <= 0 {
                problems.append(
                    "layers.\(layer.index): Engram table \(table) has no hash columns to read "
                        + "(max_ngram_size \(spec.maxNgramSize), n_heads \(spec.headCount))"
                )
            }
            if spec.headDimension <= 0 {
                problems.append(
                    "layers.\(layer.index): Engram table \(table) publishes no head dimension"
                )
            }
            if spec.rowCount <= 0 {
                problems.append("layers.\(layer.index): Engram table \(table) publishes no rows")
            }
            if spec.layer != layer.index {
                problems.append(
                    "layers.\(layer.index): Engram table \(table) belongs to layer \(spec.layer)"
                )
            }
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
        // The router kernel keeps its top-k in a 64-entry stack array and clears the
        // readback buffers when it cannot, so a larger top-k would silently zero the
        // mixture instead of routing.
        if config.numActivatedExperts > 64 {
            problems.append(
                "n_activated_experts = \(config.numActivatedExperts) exceeds the router kernel's 64-way bound"
            )
        }
        return problems
    }

    /// Every stage this executor encodes. The switch in `encode` is exhaustive, so
    /// this set exists purely so a test can assert full coverage of the shared
    /// layer pipeline.
    public static let encodedStages: Set<DeepSeekV41ForwardStages.Stage> = [
        .engram,
        .hcAttentionCoefficients,
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
        .ropeOutput,
        .outputDown,
        .outputUp,
        .writeAttentionStreams,
        .hcFFNCoefficients,
        .mixFFNStreams,
        .ffnNorm,
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
nonisolated public final class DeepSeekV41MetalForward: DeepSeekV41PositionForward, @unchecked Sendable {

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
    /// Query and latent rails have different widths, so their tables differ.
    private let queryFrequencies: [DeepSeekV41RopeTable: MTLBuffer]
    private let latentFrequencies: [DeepSeekV41RopeTable: MTLBuffer]
    /// Substitutes for the optional kernel inputs, so an absent tensor is a bound
    /// identity value rather than an unbound buffer. Metal leaves an unbound slot
    /// undefined; the kernels test for `nullptr`, which an unbound buffer is not.
    private let zeros: MTLBuffer?
    /// The identity for a multiplicative slot, which is ones rather than zeros.
    private let ones: MTLBuffer?
    /// `attn_sink` absent means "no sink entry", which the kernel spells `-1e30`.
    private let sinkDisabled: MTLBuffer?
    private let mult: Int
    /// `(2 + mult) * mult`: how many logits `hc_*_fn` produces per position.
    private let mixCount: Int
    private let recordElements: Int
    private let swigluLimit: Float
    private let gatherRows: Int
    private let contextLength: Int
    /// Debug opt-out: run an Engram layer with no memory read instead of reading it.
    private let skipEngram: Bool
    /// Where each Engram table stages one position's rows. Indexed by
    /// `layer.engramTable`, which is `config.engramTables` order.
    private let engramSlots: [DeepSeekV41EngramSlot]
    private let engramReaders: [Int: DeepSeekV41EngramReader]
    private let engramLayouts: [Int: DeepSeekV41EngramLayout]
    /// The compressed vocabulary the hashes are drawn from, built at load.
    private let engramTokenMap: DeepSeekV41EngramTokenMap?
    /// Why the map is missing, so the readiness line names a reason.
    private let engramProblem: String?
    /// The token stream ending at the position being run, oldest first. The Engram
    /// hash reads look-backs from it, so the executor keeps it the way the
    /// reference forward does.
    private var tokenHistory: [Int32] = []
    /// One dead flag per history entry. A dead position (an image span) takes no
    /// part in an n-gram and its gate is zero.
    private var deadTokens: [Bool] = []
    private var pendingDead = false
    /// Which block of the compressor scratch each owner fills. Divided from the
    /// plan at load, because the block a layer writes is a property of the layer
    /// rather than of the position being run.
    private let compressorOwnerSlots: [Int: Int]
    /// Fills the arena slots from the shards. Nil means the routed experts are read
    /// through the zero-copy shard mappings, which is correct but pays a page fault
    /// per 16 KB and collapses to ~0.065 GB/s once it interleaves with GPU work.
    private let streamer: DeepSeekV41ExpertStreamer?
    /// Committed command buffers whose results the host has not needed yet. Drained
    /// by `settlePending` at every readback and at the end of a token.
    private var pending: [(buffer: MTLCommandBuffer, stage: String)] = []

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
        self.mixCount = (2 + self.mult) * self.mult
        self.recordElements = max(
            1,
            runtime.cache.layout.geometry.recordBytes
                / max(1, runtime.cache.layout.geometry.precision.valueBytesPerElement)
        )
        self.swigluLimit = runtime.config.swigluLimit ?? 0
        self.gatherRows = runtime.workspace.gatherRows
        self.compressorOwnerSlots = DeepSeekV41CacheAddressing.compressorOwnerSlots(
            plan: runtime.plan
        )
        self.contextLength = runtime.contextLength
        self.skipEngram = runtime.allowEngramSkip
        self.streamer = runtime.expertStreamer
        self.engramSlots = runtime.workspace.engramSlots
        self.engramReaders = runtime.weights.engramReaders
        self.engramLayouts = runtime.weights.engramLayouts
        self.engramTokenMap = runtime.engramTokenMap
        self.engramProblem = runtime.engramProblem

        var problems = DeepSeekV41DevicePlan.problems(
            config: runtime.config,
            plan: runtime.plan
        )
        // An Engram layer needs four pieces before it can serve a position: the
        // table file (a reader), the bucket arithmetic (a layout), the compressed
        // vocabulary (the token map) and the `wkv` projection. Each missing piece
        // is named, because the failure mode of guessing is text that reads fine.
        for layer in runtime.plan where layer.engramTable != nil && !runtime.allowEngramSkip {
            let table = layer.engramTable ?? -1
            guard table >= 0, table < runtime.workspace.engramSlots.count else {
                problems.append(
                    "layers.\(layer.index): Engram table index \(table) is outside "
                        + "the \(runtime.workspace.engramSlots.count) tables the plan carries"
                )
                continue
            }
            let slot = runtime.workspace.engramSlots[table]
            if runtime.engramTokenMap == nil {
                problems.append(
                    "layers.\(layer.index): "
                        + (runtime.engramProblem ?? "the Engram token map is unavailable")
                )
            }
            if runtime.weights.engramReaders[layer.index] == nil {
                problems.append(
                    "layers.\(layer.index): the Engram table file could not be opened, "
                        + "so its rows cannot be read"
                )
            }
            if runtime.weights.engramLayouts[layer.index] == nil {
                problems.append(
                    "layers.\(layer.index): the Engram bucket layout could not be resolved"
                )
            }
            if slot.flatWidth % 32 != 0 {
                problems.append(
                    "layers.\(layer.index): the Engram rows flatten to \(slot.flatWidth) values, "
                        + "which is not the multiple of 32 the block-scaled projections read"
                )
            }
            if let wkv = runtime.weights.tensor(.engramWkv, layer: layer.index) {
                let expectedRows = runtime.config.hiddenSize * (max(1, runtime.config.hcMult) + 1)
                if wkv.rows != expectedRows || wkv.columns != slot.flatWidth {
                    problems.append(
                        "layers.\(layer.index): engram.wkv is \(wkv.rows)x\(wkv.columns), "
                            + "but the gate needs \(expectedRows)x\(slot.flatWidth) "
                            + "((hc_mult + 1) * hidden by columns * head_dim)"
                    )
                }
            }
        }
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

        // RoPE tables: the query and latent rails rotate over different widths,
        // so each gets its own table per rail the plan uses.
        var queries: [DeepSeekV41RopeTable: MTLBuffer] = [:]
        var latents: [DeepSeekV41RopeTable: MTLBuffer] = [:]
        var seen = Set<DeepSeekV41RopeTable>()
        for layer in runtime.plan where !seen.contains(layer.ropeTable) {
            seen.insert(layer.ropeTable)
            let shape = DeepSeekV41AttentionShape.make(config: runtime.config, layer: layer)
            func buffer(_ values: [Float]) -> MTLBuffer? {
                guard !values.isEmpty else { return nil }
                return values.withUnsafeBytes { raw in
                    runtime.device.makeBuffer(
                        bytes: raw.baseAddress!,
                        length: raw.count,
                        options: .storageModeShared
                    )
                }
            }
            if let query = buffer(
                DeepSeekV41RoPE.queryFrequencies(for: shape)
            ) {
                queries[layer.ropeTable] = query
            }
            if let latent = buffer(
                DeepSeekV41RoPE.latentFrequencies(for: shape)
            ) {
                latents[layer.ropeTable] = latent
            }
        }
        self.queryFrequencies = queries
        self.latentFrequencies = latents

        // Identity buffers for the optional kernel slots, sized for every place one
        // is substituted: router bias (one per expert), the engram per-copy weights
        // (one per copy and dimension), and the attention sink (one per head).
        let zeroCount = max(
            16,
            max(
                max(runtime.config.numRoutedExperts, runtime.config.hiddenSize),
                max(max(1, runtime.config.hcMult), runtime.config.numAttentionHeads)
            )
        )
        func fill(_ value: Float, count: Int) -> MTLBuffer? {
            let values = [Float](repeating: value, count: max(1, count))
            return values.withUnsafeBytes { raw in
                runtime.device.makeBuffer(
                    bytes: raw.baseAddress!,
                    length: raw.count,
                    options: .storageModeShared
                )
            }
        }
        self.zeros = fill(0, count: zeroCount)
        self.ones = fill(1, count: max(1, runtime.config.hcMult))
        self.sinkDisabled = fill(-1e30, count: max(1, runtime.config.numAttentionHeads))

        // The per-layer scalars and gains, resolved once.
        let bank = DeepSeekV41VectorBank(device: runtime.device, buffers: shardBuffers)
        let catalog = runtime.catalog
        func store(
            _ role: DeepSeekV41TensorRole,
            _ layer: Int?,
            resizeTo count: Int? = nil
        ) {
            // `hc_*_scale` and the quantized companions classify with `isScale`
            // set, so the plain lookup misses them. Try the scale slot too; a
            // scalar that exists but cannot be decoded is reported rather than
            // silently replaced by its identity value.
            let ref = catalog.tensor(role, layer: layer)
                ?? catalog.tensor(role, layer: layer, scale: true)
            guard let ref else { return }
            if !bank.store(role, layer: layer, ref: ref, resizeTo: count) {
                problems.append(
                    "\(role.rawValue)\(layer.map { " layer \($0)" } ?? "") could not be decoded"
                )
            }
        }
        store(.finalNorm, nil)
        for layer in runtime.plan {
            let l = layer.index
            store(.attnNorm, l)
            store(.ffnNorm, l)
            store(.attnWqANorm, l)
            store(.attnKvNorm, l)
            store(.attnSink, l)
            store(.compressorNorm, l)
            store(.indexerKNorm, l)
            store(.routerBias, l)
            // The projection itself is read from the shard bytes by
            // `encodeHyperConnectionCoefficients` (it is FP32, which the quantized
            // GEMV kernels do not cover), so only the two small companions are
            // staged here. `hc_*_scale` keeps all three values: they scale the
            // pre, post and comb sections, not the copies.
            store(.hcAttnScale, l)
            store(.hcAttnBase, l)
            store(.hcFfnScale, l)
            store(.hcFfnBase, l)
            // `q_weight` and `k_weight` are ones-initialized per-copy parameters
            // that only ever appear multiplied together (`Engram.forward`), so they
            // are resolved to one FP32 vector instead of being read per token.
            if layer.engramTable != nil && !runtime.allowEngramSkip {
                let width = max(1, runtime.config.hcMult) * runtime.config.hiddenSize
                store(.engramQuery, l, resizeTo: width)
                store(.engramKey, l, resizeTo: width)
            }
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

    /// Positions the cache was provisioned for. The turn loop sizes its token cap
    /// from this so the last position cannot land past the cache.
    public var positionLimit: Int { contextLength }

    /// One-line readiness report for the load log.
    public func describe() -> String {
        var lines = [
            "device forward: \(isRunnable ? "ready" : "not runnable") at "
                + "\(DeepSeekV41DevicePlan.gatherRowBudget(config: config)) gathered rows/position"
        ]
        if skipEngram {
            let skipped = plan.filter { $0.engramTable != nil }.map(\.index)
            if !skipped.isEmpty {
                lines.append(
                    "  ENGRAM SKIPPED on layers \(skipped.map(String.init).joined(separator: ", ")): "
                        + "they run with no memory read at all (\(DeepSeekV41Runtime.engramOffDefaultKey))"
                )
            }
        } else {
            let served = plan.filter { $0.engramTable != nil }
            if !served.isEmpty {
                let rows = served.map { layer -> String in
                    let table = layer.engramTable ?? -1
                    let columns = table >= 0 && table < engramSlots.count
                        ? engramSlots[table].columns : 0
                    return "\(layer.index):\(columns)"
                }
                lines.append(
                    "  ENGRAM rows/position on layers " + rows.joined(separator: ", ")
                )
            }
        }
        lines.append(contentsOf: problems.map { "  \($0)" })
        return lines.joined(separator: "\n")
    }

    /// Drops the sequence so the next `forwardToken` starts at position 0.
    ///
    /// The token history goes with the cache: an n-gram must never span the seam
    /// between two prompts. A resumed turn skips this, which is why `resume` checks
    /// that the history it keeps really is the prompt's.
    public func reset() {
        cache.reset()
        tokenHistory.removeAll(keepingCapacity: true)
        deadTokens.removeAll(keepingCapacity: true)
        pendingDead = false
    }

    /// Positions the cache already holds.
    public var storedPositions: Int { cache.storedPositions }

    /// How much of `promptTokenIds` this cache can skip.
    ///
    /// The count arithmetic is `reusablePrefix`'s; what the count cannot say is
    /// whether the resident positions are these tokens. The pin that proposed the
    /// length belongs to the app, not to this cache: another model's turn, or an
    /// older turn of this one, can leave a pin whose length happens to fit a sequence
    /// this cache never saw. So the history is compared, and a mismatch is a full
    /// prefill rather than a wrong one.
    public func reusablePrefix(proposedPrefix: Int, promptTokenIds: [Int]) -> Int {
        let stored = cache.storedPositions
        guard deadTokens.count == stored else { return 0 }
        return DeepSeekV41CacheAddressing.reusablePrefix(
            proposed: proposedPrefix,
            storedPositions: stored,
            promptTokenIds: promptTokenIds,
            residentTokens: tokenHistory
        )
    }

    /// Keeps the resident sequence so the next `forwardToken` may continue it.
    ///
    /// Nothing is cleared: the window ring, the compressed rows, every owner's
    /// half-filled compressor group and the token stream the Engram layers hash are
    /// all addressed by position, so they describe exactly the sequence that produced
    /// them. What this does is refuse to pretend: if the resident history is not this
    /// prompt's tokens, continuing would mix two conversations inside an n-gram, and
    /// an idle cache has nothing to continue from.
    public func resume(prefixTokenIds: [Int]) throws {
        let stored = cache.storedPositions
        guard prefixTokenIds.count == stored else {
            throw DeepSeekV41EngineError.outOfOrderPosition(prefixTokenIds.count, stored)
        }
        guard tokenHistory.count == stored, deadTokens.count == stored else {
            throw DeepSeekV41EngineError.prefixStateMismatch(
                position: min(tokenHistory.count, stored)
            )
        }
        for (position, token) in prefixTokenIds.enumerated() {
            guard tokenHistory[position] == Int32(truncatingIfNeeded: token) else {
                throw DeepSeekV41EngineError.prefixStateMismatch(position: position)
            }
        }
        // A dead marker the last turn left pending belongs to that turn.
        pendingDead = false
    }

    /// Marks the next position as taking no part in an n-gram, so an Engram lookup
    /// cannot span it and contributes nothing there. An image span is the caller
    /// that needs this.
    public func markNextTokenDead() {
        pendingDead = true
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

        // The Engram layers hash the token stream, so the history is appended here
        // rather than recomputed per layer. It is cleared by `reset`, which is what
        // keeps one turn's n-grams out of the next one's.
        tokenHistory.append(Int32(truncatingIfNeeded: token))
        deadTokens.append(pendingDead)
        pendingDead = false

        try encodeEmbedding(token)
        for layer in plan {
            try runLayer(layer, at: position)
        }
        try finishBlock()
        // Every buffer committed by this token, including the deferred experts
        // halves and the output head, has to be complete before the logits are read.
        try settlePending()

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

        // Every hyper-connection copy starts as this position's embedding
        // (`Array(repeating: hidden, count: mult)`), and the stack's first collapse
        // is the identity mix (`make_identity_pre_mix`), so the mix a forward pass
        // carries is reset here beside the streams it belongs to.
        try encodeClear(encoder, buffer: workspace.streams, count: mult * shapes.hidden)
        for copy in 0..<mult {
            try encodeCopy(
                encoder,
                from: workspace.hidden,
                to: workspace.streams,
                count: shapes.hidden,
                destinationByteOffset: copy * shapes.hidden * MemoryLayout<Float>.stride
            )
        }
        encoder.setComputePipelineState(pipelines.hcIdentityMix)
        encoder.setBuffer(workspace.hcMix, offset: 0, index: 0)
        var copies = UInt32(mult)
        encoder.setBytes(&copies, length: 4, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: mult, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(mult)
        )
        encoder.memoryBarrier(scope: .buffers)
        encoder.endEncoding()
        // Nothing reads the embedding on the host, so this does not wait: the first
        // layer's front half is ordered behind it on the same queue.
        try commit(commandBuffer, stage: "embedding", waiting: false)
    }

    /// The block's output: collapse the streams, final norm, head.
    private func finishBlock() throws {
        let commandBuffer = try makeCommandBuffer()
        let encoder = try makeEncoder(commandBuffer)

        // `.collapseStreams` already collapsed the stack with the mix the last
        // block produced (`layer.hc_pre(h, pre_mix)`), and it wrote that vector into
        // `hidden`; this normalizes it and projects the head.
        try encodeNorm(
            encoder,
            input: workspace.hidden,
            gain: vectors.gain(.finalNorm, layer: nil, width: shapes.hidden),
            output: workspace.output,
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
            input: workspace.output,
            output: workspace.logits
        )
        encoder.endEncoding()
        // The caller reads the logits, so this is where the token's queue catches up.
        try commit(commandBuffer, stage: "output head", waiting: false)
    }

    // MARK: One layer

    private func runLayer(_ layer: DeepSeekV41LayerPlan, at position: Int) throws {
        // Engram rows are host reads, so they are issued before the layer's command
        // buffer exists rather than in the middle of an encode. Each table has its
        // own staging block, so this read cannot land under a kernel that is still
        // reading another table's rows.
        if layer.engramTable != nil {
            try readEngramRows(layer: layer)
        }

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
        // The routing readback below needs this half finished on the host. It also
        // drains the previous layer's expert half, which is what makes the arena
        // slots free to overwrite: a pread here can never land under a kernel that
        // is still reading the slot it replaces.
        try commit(commandBuffer, stage: "layer \(layer.index) attention")

        // The mixture binds per-expert byte offsets, which are host values.
        let routing = readRouting()

        // Filled from disk before the expert half is encoded, so the kernels read
        // the slots rather than faulting the shard mapping a page at a time.
        var residents: [DeepSeekV41ResidentExpert]?
        if let streamer, !routing.expertIDs.isEmpty {
            residents = try streamer.ensureResident(
                layer: layer.index,
                experts: routing.expertIDs
            )
        }

        let expertCommand = try makeCommandBuffer()
        let expertEncoder = try makeEncoder(expertCommand)
        for stage in DeepSeekV41DevicePlan.backHalf {
            try encode(
                stage,
                layer: layer,
                position: position,
                encoder: expertEncoder,
                routing: routing,
                residents: residents
            )
        }
        expertEncoder.endEncoding()
        // No readback of this half exists, so it is only ordered: the next layer's
        // front half commits behind it on the same queue.
        try commit(expertCommand, stage: "layer \(layer.index) experts", waiting: false)
    }

    /// Every stage the shared pipeline lists. Exhaustive on purpose: a stage added
    /// to `DeepSeekV41ForwardStages` fails to compile here rather than being
    /// silently skipped on device.
    private func encode(
        _ stage: DeepSeekV41ForwardStages.Stage,
        layer: DeepSeekV41LayerPlan,
        position: Int,
        encoder: MTLComputeCommandEncoder,
        routing: DeepSeekV41Routing? = nil,
        residents: [DeepSeekV41ResidentExpert]? = nil
    ) throws {
        let index = layer.index
        let shape = DeepSeekV41AttentionShape.make(config: config, layer: layer)

        switch stage {
        case .hcAttentionCoefficients:
            try encodeHyperConnectionCoefficients(
                encoder,
                layer: index,
                function: .hcAttnFn,
                base: .hcAttnBase,
                scale: .hcAttnScale,
                pre: workspace.hcAttentionPre,
                post: workspace.hcAttentionPost,
                comb: workspace.hcAttentionComb
            )

        case .mixAttentionStreams:
            // The sub-layer reads the stack collapsed with the mix the PREVIOUS
            // sub-layer produced. `hc_pre` also snapshots the block input, because
            // `hc_post` mixes it back in and cannot read the streams in place.
            try encodeHyperConnectionPre(
                encoder,
                mix: workspace.hcMix,
                output: workspace.normed,
                snapshotting: true
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
                sliceStart: shape.queryRopeStart,
                sliceLength: shape.queryRopeWidth,
                table: layer.ropeTable,
                latent: false,
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
                headDim: 0,
                sliceStart: shape.latentRopeStart,
                sliceLength: shape.latentRopeWidth,
                table: layer.ropeTable,
                latent: true,
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
            try encodePoolIfComplete(
                layer: layer, position: position, encoder: encoder, shape: shape
            )

        case .attention:
            try encodeAttention(layer: layer, position: position, encoder: encoder, shape: shape)

        case .ropeOutput:
            // The keys stay in the cache rotated, so the output's rope payload comes
            // back out here, before the projection. Skipping this feeds rotated values
            // into `attn.wo_a`, which is the shape of bug a parity test cannot see.
            if shape.outputCarriesRope {
                try encodeRope(
                    encoder,
                    vector: workspace.attentionOut,
                    rowWidth: shapes.attentionOutputWidth,
                    headDim: shape.vHeadDim,
                    sliceStart: shape.outputRopeStart,
                    sliceLength: shape.latentRopeWidth,
                    table: layer.ropeTable,
                    latent: true,
                    position: position,
                    inverse: true
                )
            }
        case .outputDown:
            if config.oLoraRank > 0, let down = weights.tensor(.attnWoA, layer: index) {
                try encodeBlockDiagonalProjection(
                    encoder,
                    weight: down,
                    scale: weights.scaleTensor(.attnWoA, layer: index),
                    groups: shape.outputGroups,
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
            // `wo_b` reads the CONCATENATED groups, so its input width is
            // `o_groups * o_lora_rank`, not the per-group rank.
            let reducedWidth = hasDown ? shapes.outputReducedWidth : shapes.attentionOutputWidth
            try encodeGemv(
                encoder,
                weight: up,
                scale: weights.scaleTensor(.attnWoB, layer: index),
                inDim: reducedWidth,
                outDim: shapes.hidden,
                input: hasDown ? workspace.reduced : workspace.attentionOut,
                output: workspace.output
            )

        case .writeAttentionStreams:
            try encodeHyperConnectionPost(
                encoder,
                output: workspace.output,
                gains: workspace.hcAttentionPost,
                comb: workspace.hcAttentionComb
            )

        case .hcFFNCoefficients:
            try encodeHyperConnectionCoefficients(
                encoder,
                layer: index,
                function: .hcFfnFn,
                base: .hcFfnBase,
                scale: .hcFfnScale,
                pre: workspace.hcFFNPre,
                post: workspace.hcFFNPost,
                comb: workspace.hcFFNComb
            )

        case .mixFFNStreams:
            // The feed-forward reads its stack collapsed with ATTENTION's `pre`.
            try encodeHyperConnectionPre(
                encoder,
                mix: workspace.hcAttentionPre,
                output: workspace.normed,
                snapshotting: true
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
            try encodeEngram(encoder, layer: layer)

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
            try encodeRoutedExperts(
                layer: layer, encoder: encoder, routing: routing, residents: residents
            )

        case .writeFFNStreams:
            try encodeHyperConnectionPost(
                encoder,
                output: workspace.moeOut,
                gains: workspace.hcFFNPost,
                comb: workspace.hcFFNComb
            )
            // Whatever this feed-forward decided is what the NEXT layer's attention
            // collapses with, and after the last layer what the head collapses with
            // (`Transformer.forward` returns `ffn_pre` up the stack).
            try encodeCopy(
                encoder, from: workspace.hcFFNPre, to: workspace.hcMix, count: mixCount
            )

        case .collapseStreams:
            // The block collapses once, after the last layer, so a later stage can
            // never observe a half-collapsed state; `finishBlock` then normalizes and
            // projects this vector.
            try encodeHyperConnectionPre(
                encoder,
                mix: workspace.hcMix,
                output: workspace.hidden,
                snapshotting: false
            )
        }
    }

    // MARK: Engram stage

    /// Preads one position's Engram rows into the owning table's staging block.
    ///
    /// A position reads `(max_ngram_size - 1) * n_heads` rows of ~264 bytes, which
    /// is why the whole batch goes out at once: 24 serial small reads would spend
    /// the decode step waiting on seek latency instead of using queue depth.
    ///
    /// A dead position reads nothing: `Engram.forward` zeroes the gate wherever
    /// `token_mask` is false, so the residual passes through untouched rather than
    /// taking pad rows into the stream.
    private func readEngramRows(layer: DeepSeekV41LayerPlan) throws {
        guard !skipEngram, let table = layer.engramTable else { return }
        guard deadTokens.last != true else { return }
        guard table >= 0, table < engramSlots.count else {
            throw DeepSeekV41EngineError.unsupportedStage("engram table \(table)")
        }
        let slot = engramSlots[table]
        guard let reader = engramReaders[layer.index],
              let layout = engramLayouts[layer.index],
              let tokenMap = engramTokenMap else {
            throw DeepSeekV41EngineError.missingTensor("engram.embed (layer \(layer.index))")
        }
        let requests = DeepSeekV41Engram.rowRequests(
            layer: layer.index,
            history: tokenHistory,
            dead: deadTokens,
            tokenMap: tokenMap,
            layout: layout
        )
        // The staging block is exactly one position's rows, so a request set of any
        // other size would write past it. That can only mean the plan and the
        // layout disagree about the table, which is a load-time defect the caller
        // must not discover as a heap overflow.
        guard requests.count == slot.columns else {
            throw DeepSeekV41EngineError.unsupportedStage(
                "engram rows for layer \(layer.index): \(requests.count) of \(slot.columns) columns"
            )
        }
        let destination = workspace.engramRaw.contents().advanced(by: slot.rawByteOffset)
        try reader.readRows(requests, into: destination)
    }

    /// The Engram write: `wkv` over the retrieved rows, then the gate into the
    /// residual.
    ///
    /// This is `Engram.forward` end to end. The rows concatenate into one
    /// `columns * head_dim` vector, `engram.wkv` turns it into one key per
    /// hyper-connection copy followed by the one value every copy shares, and the
    /// gate kernel mixes the value into each copy by how well that copy matches its
    /// own key.
    ///
    /// No output projection follows, because the value is already `hidden` wide.
    /// The stage runs at the TOP of the layer (first in the pipeline), before the
    /// attention sub-layer reads the residual, which is where `Model.forward` calls
    /// `layer.engram(h, ...)`.
    private func encodeEngram(
        _ encoder: MTLComputeCommandEncoder,
        layer: DeepSeekV41LayerPlan
    ) throws {
        guard !skipEngram, let table = layer.engramTable else { return }
        guard deadTokens.last != true else { return }
        guard table >= 0, table < engramSlots.count else {
            throw DeepSeekV41EngineError.unsupportedStage("engram table \(table)")
        }
        let slot = engramSlots[table]
        guard slot.columns > 0, slot.headDimension > 0 else { return }
        guard let wkv = weights.tensor(.engramWkv, layer: layer.index) else {
            throw DeepSeekV41EngineError.missingTensor("engram.wkv")
        }

        var rowCount = UInt32(slot.columns)
        var headDimension = UInt32(slot.headDimension)
        encoder.setComputePipelineState(pipelines.engramDequantRows)
        encoder.setBuffer(workspace.engramRaw, offset: slot.rawByteOffset, index: 0)
        encoder.setBuffer(workspace.engramValues, offset: slot.valueByteOffset, index: 1)
        encoder.setBytes(&rowCount, length: 4, index: 2)
        encoder.setBytes(&headDimension, length: 4, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: slot.headDimension, height: slot.columns, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threadgroup(slot.headDimension).width, height: 1, depth: 1
            )
        )
        encoder.memoryBarrier(scope: .buffers)

        // `kv.split([hc_mult * dim, dim])` is the GEMV's own row order, so the one
        // projection covers both halves.
        try encodeGemv(
            encoder,
            weight: wkv,
            scale: weights.scaleTensor(.engramWkv, layer: layer.index),
            inDim: slot.flatWidth,
            outDim: shapes.hidden * (mult + 1),
            input: workspace.engramValues,
            output: workspace.engramKV,
            inputByteOffset: slot.valueByteOffset
        )

        // `q_weight * k_weight`, elementwise per copy. Both are ones-initialized
        // parameters in the checkpoint and the reference treats an absent pair as
        // "no reweighting", so a missing one binds an explicit ones vector rather
        // than leaving the slot unbound: Metal leaves an unbound slot undefined,
        // and the kernel's `nullptr` test would not catch that.
        let weightWidth = shapes.hidden * mult
        let query: MTLBuffer
        let key: MTLBuffer
        if let storedQuery = vectors.buffer(.engramQuery, layer: layer.index),
           let storedKey = vectors.buffer(.engramKey, layer: layer.index) {
            query = storedQuery
            key = storedKey
        } else {
            guard let identity = vectors.constant(1, count: weightWidth) else {
                throw DeepSeekV41EngineError.allocationFailed("the Engram weight identity")
            }
            query = identity
            key = identity
        }
        try dispatchEngramGate(encoder, dim: shapes.hidden, query: query, key: key)
    }

    /// One threadgroup per hyper-connection copy: the score is a reduction over the
    /// whole stream before the value can be written back.
    private func dispatchEngramGate(
        _ encoder: MTLComputeCommandEncoder,
        dim: Int,
        query: MTLBuffer,
        key: MTLBuffer
    ) throws {
        guard dim > 0 else { return }
        var width = UInt32(dim)
        var streams = UInt32(mult)
        var epsilon = config.rmsNormEps
        encoder.setComputePipelineState(pipelines.engramGate)
        encoder.setBuffer(workspace.streams, offset: 0, index: 0)
        encoder.setBuffer(workspace.engramKV, offset: 0, index: 1)
        encoder.setBuffer(query, offset: 0, index: 2)
        encoder.setBuffer(key, offset: 0, index: 3)
        encoder.setBuffer(workspace.engramGates, offset: 0, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&streams, length: 4, index: 6)
        encoder.setBytes(&epsilon, length: 4, index: 7)
        let lanes = max(32, min(256, pipelines.engramGate.maxTotalThreadsPerThreadgroup))
        encoder.setThreadgroupMemoryLength((lanes / 32) * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: mult, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    // MARK: Attention stages

    /// Pools a completed block of latents into the owner's compressed row.
    ///
    /// A row exists only when the whole block has been seen, i.e. when
    /// `position + 1` is a multiple of the ratio. Pooling on later positions would
    /// append the same row again and let the cache grow with the context rather
    /// than with the context / ratio.
    /// Fills this position's slot in the compressor's group, and pools the group
    /// the moment it completes.
    ///
    /// `Compressor.forward(x)` projects the layer's post-`attn_norm` input through
    /// `attn.compressor.wkv` and `attn.compressor.wgate`, keeps the group in
    /// `kv_state` / `score_state`, and only emits a row once the group is full. The
    /// rows it pools are therefore NOT the ones the window cache holds: those have
    /// been through this layer's own `wkv` and rope.
    ///
    /// Slot addressing is `position % ratio`, which is what makes a decode step's
    /// partial group survive into the next step without a separate state machine.
    private func encodePoolIfComplete(
        layer: DeepSeekV41LayerPlan,
        position: Int,
        encoder: MTLComputeCommandEncoder,
        shape: DeepSeekV41AttentionShape
    ) throws {
        guard layer.ownsCompressedKV else { return }
        let ratio = max(1, layer.compressRatio)
        // The slot is the position's offset inside its own group, exactly as
        // `Compressor.forward` addresses `kv_state[:, start_pos % ratio]`.
        guard let slot = DeepSeekV41CacheAddressing.compressorSlot(
            position: position, ratio: ratio
        ) else { return }
        let width = shapes.latentWidth
        guard slot < ratio else { return }
        guard let owner = compressorOwnerSlots[layer.index] else {
            throw DeepSeekV41EngineError.allocationFailed(
                "layers.\(layer.index): no compressor slot block was allocated for this owner"
            )
        }
        let groupWidth = workspace.compressorGroupWidth
        guard let groupByteOffset = DeepSeekV41CacheAddressing.compressorSlotByteOffset(
                  owner: owner, slot: 0, groupWidth: groupWidth, width: width
              ),
              let slotByteOffset = DeepSeekV41CacheAddressing.compressorSlotByteOffset(
                  owner: owner, slot: slot, groupWidth: groupWidth, width: width
              ) else {
            throw DeepSeekV41EngineError.allocationFailed(
                "layers.\(layer.index): the compressor group is \(groupWidth) slots wide"
            )
        }

        guard let kvWeight = weights.tensor(.compressorWkv, layer: layer.index) else {
            throw DeepSeekV41EngineError.missingTensor("attn.compressor.wkv")
        }
        try encodeGemv(
            encoder,
            weight: kvWeight,
            scale: weights.scaleTensor(.compressorWkv, layer: layer.index),
            inDim: shapes.hidden,
            outDim: width,
            input: workspace.normed,
            output: workspace.compressorLatents,
            outputByteOffset: slotByteOffset
        )
        // A ratio of 1 is a plain projection: nothing to pool, and the checkpoint
        // publishes no gate for those layers.
        if ratio > 1 {
            guard let gateWeight = weights.tensor(.compressorWgate, layer: layer.index) else {
                throw DeepSeekV41EngineError.missingTensor("attn.compressor.wgate")
            }
            try encodeGemv(
                encoder,
                weight: gateWeight,
                scale: weights.scaleTensor(.compressorWgate, layer: layer.index),
                inDim: shapes.hidden,
                outDim: width,
                input: workspace.normed,
                output: workspace.compressorScores,
                outputByteOffset: slotByteOffset
            )
        }

        guard DeepSeekV41CacheAddressing.compressorCompletesGroup(
            position: position, ratio: ratio
        ) else { return }
        let rows = DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: position, ratio: ratio)
        guard rows > 0 else { return }
        let block = DeepSeekV41CacheAddressing.poolRange(row: rows - 1, ratio: ratio)
        guard block.upperBound <= position + 1 else { return }
        guard let compressedBuffer = cache.compressedBuffer,
              let destinationRow = DeepSeekV41CacheAddressing.compressedByteOffset(
                  ownerLayer: layer.index,
                  row: rows - 1,
                  layout: cache.layout
              ) else {
            throw DeepSeekV41EngineError.allocationFailed("the compressed cache")
        }

        encoder.setComputePipelineState(pipelines.poolLatents)
        // Bound at this owner's block: the pool reads the group from row 0 of the
        // buffers it is given, and another owner's block is not part of it.
        encoder.setBuffer(workspace.compressorLatents, offset: groupByteOffset, index: 0)
        encoder.setBuffer(workspace.compressorScores, offset: groupByteOffset, index: 1)
        encoder.setBuffer(workspace.pooled, offset: 0, index: 2)
        var poolWidth = UInt32(width)
        var blockSize = UInt32(ratio)
        encoder.setBytes(&poolWidth, length: 4, index: 3)
        encoder.setBytes(&blockSize, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: width, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(width)
        )
        encoder.memoryBarrier(scope: .buffers)

        // `Compressor.forward` returns `norm(kv)`; `_compress_kv` rotates that row
        // afterwards, at the position of the FIRST token of the group, which is
        // `row * ratio`.
        try encodeNorm(
            encoder,
            input: workspace.pooled,
            gain: vectors.gain(.compressorNorm, layer: layer.index, width: width),
            output: workspace.pooled,
            width: width
        )
        try encodeRope(
            encoder,
            vector: workspace.pooled,
            rowWidth: width,
            headDim: 0,
            sliceStart: shape.latentRopeStart,
            sliceLength: shape.latentRopeWidth,
            table: layer.ropeTable,
            latent: true,
            position: max(0, rows - 1) * ratio
        )
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
            ownerRowLimit: cache.layout.compressedRowsByOwner[layer.compressedKVSource]
                ?? geometry.compressedRows
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
        try encodeRowIndices(rowPlan.windowRows, into: workspace.rowIndices, at: 0)
        cursor += windowCount
        try encodeGather(
            encoder,
            cacheBuffer: windowBuffer,
            byteOffset: windowBase,
            rowIndices: workspace.rowIndices,
            rowIndexOffset: 0,
            rowCount: windowCount,
            destinationRowOffset: 0
        )
        if compressedCount > 0 {
            guard let compressedBuffer = cache.compressedBuffer,
                  let compressedBase = cache.layout.compressedBase[layer.compressedKVSource] else {
                throw DeepSeekV41EngineError.allocationFailed("the compressed cache")
            }
            try encodeRowIndices(rowPlan.compressedRows, into: workspace.rowIndices, at: cursor)
            try encodeGather(
                encoder,
                cacheBuffer: compressedBuffer,
                byteOffset: compressedBase,
                rowIndices: workspace.rowIndices,
                rowIndexOffset: cursor,
                rowCount: compressedCount,
                destinationRowOffset: windowCount
            )
        }
        encoder.memoryBarrier(scope: .buffers)

        encoder.setComputePipelineState(pipelines.attention)
        encoder.setBuffer(workspace.query, offset: 0, index: 0)
        encoder.setBuffer(workspace.gathered, offset: 0, index: 1)
        // The slot is always bound: an absent sink becomes a buffer of `-1e30`, the
        // kernel's own "disabled" spelling, instead of an undefined slot.
        let sink = vectors.buffer(.attnSink, layer: layer.index)
            .flatMap { $0.length >= shape.numHeads * MemoryLayout<Float>.stride ? $0 : nil }
            ?? sinkDisabled
        if let sink {
            encoder.setBuffer(sink, offset: 0, index: 2)
        } else {
            throw DeepSeekV41EngineError.allocationFailed("the attention sink placeholder")
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
        // same vector the routed experts do. A checkpoint that declares no shared
        // expert does not run one, so its absence is not a missing tensor.
        guard config.sharedExpertCount > 0 else { return }
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

    /// The routed mixture.
    ///
    /// `residents` is aligned with `routing.expertIDs` and non-nil only when the
    /// expert streamer is running: each entry is the arena slot holding that
    /// expert's bytes, already filled. Without it the weights are read through the
    /// shard mappings, which is correct and much slower (`PERF_FINDINGS.md`).
    private func encodeRoutedExperts(
        layer: DeepSeekV41LayerPlan,
        encoder: MTLComputeCommandEncoder,
        routing: DeepSeekV41Routing?,
        residents: [DeepSeekV41ResidentExpert]?
    ) throws {
        // The mixture always starts from zero: every expert accumulates into the
        // same vector.
        try encodeClear(encoder, buffer: workspace.moeOut, count: shapes.hidden)
        guard let routing, !routing.expertIDs.isEmpty else { return }
        let buffer = workspace.intermediate
        for (rank, expert) in routing.expertIDs.enumerated() {
            let weight = rank < routing.weights.count ? routing.weights[rank] : 0
            if weight == 0 { continue }
            let resident = residents.flatMap { rank < $0.count ? $0[rank] : nil }
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
                output: buffer,
                resident: resident
            )
            try encodeDown(
                encoder,
                weight: w2,
                scale: weights.scaleTensor(.expertW2, layer: layer.index, expert: expert),
                inDim: shapes.expertIntermediate,
                outDim: shapes.hidden,
                activation: buffer,
                output: workspace.moeOut,
                weightToAdd: weight,
                resident: resident
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
        guard config.scoringFunction != .softmax else {
            // `softmax` normalizes over every expert, so it cannot be evaluated one
            // expert at a time the way this single-thread scan does. Refusing is
            // the honest answer; silently routing on the wrong scores is not.
            throw DeepSeekV41EngineError.unsupportedStage(
                "scoring_func softmax (the kernel implements sigmoid and sqrtsoftplus)"
            )
        }
        encoder.setComputePipelineState(pipelines.router)
        encoder.setBuffer(workspace.routerLogits, offset: 0, index: 0)
        // An absent bias is a bound zero vector, not an unbound slot: the kernel
        // checks the pointer against `nullptr` and then dereferences it.
        let bias = vectors.buffer(.routerBias, layer: layer)
            .flatMap { $0.length >= config.numRoutedExperts * MemoryLayout<Float>.stride ? $0 : nil }
            ?? zeros
        if let bias {
            encoder.setBuffer(bias, offset: 0, index: 1)
        } else {
            throw DeepSeekV41EngineError.allocationFailed("the router bias placeholder")
        }
        encoder.setBuffer(workspace.expertIndices, offset: 0, index: 2)
        encoder.setBuffer(workspace.expertWeights, offset: 0, index: 3)
        var expertCount = UInt32(config.numRoutedExperts)
        var topK = UInt32(max(1, config.numActivatedExperts))
        var scaling = config.routedScalingFactor
        var normalise: UInt32 = config.normalizeTopKProb ? 1 : 0
        var scoring: UInt32 = config.scoringFunction == .sigmoid ? 0 : 1
        var temperature = config.gateTemp
        encoder.setBytes(&expertCount, length: 4, index: 4)
        encoder.setBytes(&topK, length: 4, index: 5)
        encoder.setBytes(&scaling, length: 4, index: 6)
        encoder.setBytes(&normalise, length: 4, index: 7)
        encoder.setBytes(&scoring, length: 4, index: 8)
        encoder.setBytes(&temperature, length: 4, index: 9)
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

    private func commit(_ commandBuffer: MTLCommandBuffer, stage: String, waiting: Bool = true) throws {
        commandBuffer.commit()
        // Held until it completes: the queue keeps a committed buffer alive, but ARC
        // would drop our reference and with it the error still to be reported.
        pending.append((buffer: commandBuffer, stage: stage))
        guard waiting else { return }
        try settlePending()
    }

    /// Waits for every buffer committed so far and reports the first failure.
    ///
    /// The queue is in-order, so the last buffer completing means the earlier ones
    /// did and their waits return immediately. Called wherever the host reads device
    /// memory (routing, logits).
    private func settlePending() throws {
        let outstanding = pending
        pending.removeAll(keepingCapacity: true)
        for entry in outstanding {
            entry.buffer.waitUntilCompleted()
            if let error = entry.buffer.error {
                throw DeepSeekV41EngineError.commandFailed("\(entry.stage): \(error)")
            }
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
        // One threadgroup per row: `tgIdx` is the row index in the kernel, so a grid
        // of `width` threads would let Metal split a wide row across threadgroups and
        // every `tgIdx >= 1` would read and write outside the row it was given.
        let grid = DeepSeekV41DevicePlan.normDispatch(width: width, lanes: lanes)
        encoder.dispatchThreadgroups(
            grid.threadgroups,
            threadsPerThreadgroup: grid.threadsPerThreadgroup
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
        output: MTLBuffer,
        weightByteOffset: UInt64 = 0,
        scaleByteOffset: UInt64 = 0,
        inputByteOffset: Int = 0,
        outputByteOffset: Int = 0
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
            encoder.setBuffer(input, offset: inputByteOffset, index: 2)
            encoder.setBuffer(output, offset: outputByteOffset, index: 3)
            var weightOffset = UInt64(weight.byteOffset) + weightByteOffset
            var scaleOffset = UInt64(scale.byteOffset) + scaleByteOffset
            encoder.setBytes(&weightOffset, length: 8, index: 4)
            encoder.setBytes(&scaleOffset, length: 8, index: 5)
            encoder.setBytes(&inDimension, length: 4, index: 6)
            encoder.setBytes(&outDimension, length: 4, index: 7)
        case .bf16:
            encoder.setComputePipelineState(pipelines.bf16Gemv)
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(input, offset: inputByteOffset, index: 1)
            encoder.setBuffer(output, offset: outputByteOffset, index: 2)
            var weightOffset = UInt64(weight.byteOffset) + weightByteOffset
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

    /// A projection whose weight is block-diagonal over `groups` head groups.
    ///
    /// `attn.wo_a` is published as `groups` dense blocks stacked along the rows
    /// (`[o_groups * o_lora_rank][heads * head_dim / o_groups]`, 8192 x 4096 at 8
    /// groups) and the reference reads it with a per-group einsum. One dense GEMV
    /// over it would read every column, which is a different matrix. Each group is
    /// therefore its own dispatch, offset into the weight, its scale, the input and
    /// the output; a single group degenerates to the ordinary GEMV.
    private func encodeBlockDiagonalProjection(
        _ encoder: MTLComputeCommandEncoder,
        weight: DeepSeekV41DeviceTensor,
        scale: DeepSeekV41DeviceTensor?,
        groups: Int,
        inDim: Int,
        outDim: Int,
        input: MTLBuffer,
        output: MTLBuffer
    ) throws {
        let groupCount = max(1, groups)
        guard groupCount > 1 else {
            try encodeGemv(
                encoder, weight: weight, scale: scale, inDim: inDim, outDim: outDim,
                input: input, output: output
            )
            return
        }
        guard inDim % groupCount == 0, outDim % groupCount == 0 else {
            throw DeepSeekV41EngineError.unsupportedStage(
                "attn.wo_a: \(outDim)x\(inDim) does not split into \(groupCount) head groups"
            )
        }
        let valueBytes = DeepSeekV41DtypeWidth.bytesPerValue(dtype: weight.dtype)
        let perGroupIn = inDim / groupCount
        let perGroupOut = outDim / groupCount
        // Both companions are contiguous by output row: the weight's rows are
        // `perGroupIn` values wide and its scale rows `perGroupIn / 32` blocks, so
        // group g starts after `g * perGroupOut` of each.
        let weightStride = Double(perGroupOut * perGroupIn) * valueBytes
        let scaleStride = Double(perGroupOut * max(1, perGroupIn / 32))
        for group in 0..<groupCount {
            try encodeGemv(
                encoder,
                weight: weight,
                scale: scale,
                inDim: perGroupIn,
                outDim: perGroupOut,
                input: input,
                output: output,
                weightByteOffset: UInt64(weightStride * Double(group)),
                scaleByteOffset: UInt64(scaleStride * Double(group)),
                inputByteOffset: group * perGroupIn * MemoryLayout<Float>.stride,
                outputByteOffset: group * perGroupOut * MemoryLayout<Float>.stride
            )
        }
    }

    /// Fused gate+up projection.
    ///
    /// `resident` binds a streamed expert's slot instead of the shard mapping: one
    /// buffer holds the pair, and the six byte offsets come from its slot geometry,
    /// so the weights are read from the bytes `pread` just wrote rather than by
    /// faulting the shard in page by page.
    private func encodeSwiGLU(
        _ encoder: MTLComputeCommandEncoder,
        gate: DeepSeekV41DeviceTensor,
        gateScale: DeepSeekV41DeviceTensor?,
        up: DeepSeekV41DeviceTensor,
        upScale: DeepSeekV41DeviceTensor?,
        inDim: Int,
        outDim: Int,
        input: MTLBuffer,
        output: MTLBuffer,
        resident: DeepSeekV41ResidentExpert? = nil
    ) throws {
        let format = DeepSeekV41DevicePlan.linearFormat(tensor: gate, scale: gateScale)
        guard DeepSeekV41DevicePlan.swigluKernel(format, swigluLimit: swigluLimit) != nil else {
            throw DeepSeekV41EngineError.unsupportedDtype("a feed-forward gate", gate.dtype)
        }
        let gateBuffer = resident?.buffer ?? shardBuffer(gate)
        let upBuffer = resident?.buffer ?? shardBuffer(up)
        guard let gateBuffer, let upBuffer else {
            throw DeepSeekV41EngineError.missingTensor("a feed-forward gate")
        }
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)

        switch format {
        case .fp8Block32, .mxfp4:
            guard let gateScale, let upScale else {
                throw DeepSeekV41EngineError.missingTensor("a feed-forward scale companion")
            }
            let gateScaleBuffer = resident?.buffer ?? shardBuffer(gateScale)
            let upScaleBuffer = resident?.buffer ?? shardBuffer(upScale)
            guard let gateScaleBuffer, let upScaleBuffer else {
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
            var gateWeightOffset = resident?.gateWeightOffset ?? UInt64(gate.byteOffset)
            var gateScaleOffset = resident?.gateScaleOffset ?? UInt64(gateScale.byteOffset)
            var upWeightOffset = resident?.upWeightOffset ?? UInt64(up.byteOffset)
            var upScaleOffset = resident?.upScaleOffset ?? UInt64(upScale.byteOffset)
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
            var gateWeightOffset = resident?.gateWeightOffset ?? UInt64(gate.byteOffset)
            var upWeightOffset = resident?.upWeightOffset ?? UInt64(up.byteOffset)
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

    /// Weighted down-projection accumulate. `resident` binds a streamed expert's
    /// slot, exactly as `encodeSwiGLU` does.
    private func encodeDown(
        _ encoder: MTLComputeCommandEncoder,
        weight: DeepSeekV41DeviceTensor,
        scale: DeepSeekV41DeviceTensor?,
        inDim: Int,
        outDim: Int,
        activation: MTLBuffer,
        output: MTLBuffer,
        weightToAdd: Float,
        resident: DeepSeekV41ResidentExpert? = nil
    ) throws {
        let format = DeepSeekV41DevicePlan.linearFormat(tensor: weight, scale: scale)
        guard DeepSeekV41DevicePlan.downKernel(format) != nil else {
            throw DeepSeekV41EngineError.unsupportedDtype("a feed-forward down projection", weight.dtype)
        }
        let weightBuffer = resident?.buffer ?? shardBuffer(weight)
        guard let weightBuffer else {
            throw DeepSeekV41EngineError.missingTensor("a feed-forward down projection")
        }
        var inDimension = UInt32(inDim)
        var outDimension = UInt32(outDim)
        var routing = weightToAdd

        switch format {
        case .fp8Block32, .mxfp4:
            guard let scale else {
                throw DeepSeekV41EngineError.missingTensor("a down projection's scale companion")
            }
            let scaleBuffer = resident?.buffer ?? shardBuffer(scale)
            guard let scaleBuffer else {
                throw DeepSeekV41EngineError.missingTensor("a down projection's scale companion")
            }
            encoder.setComputePipelineState(
                format == .fp8Block32 ? pipelines.fp8DownAccumulate : pipelines.mxfp4DownAccumulate
            )
            encoder.setBuffer(weightBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(activation, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var weightOffset = resident?.downWeightOffset ?? UInt64(weight.byteOffset)
            var scaleOffset = resident?.downScaleOffset ?? UInt64(scale.byteOffset)
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
            var weightOffset = resident?.downWeightOffset ?? UInt64(weight.byteOffset)
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

    /// `hc_*_fn` applied to the FLATTENED stack, split into the three coefficient
    /// sets.
    ///
    /// The projection is FP32 in the checkpoint — `layers.N.hc_attn_fn` is published
    /// as `[24, 20480]` — so it reads the shard bytes directly instead of going
    /// through the quantized GEMV kernels, and a checkpoint that published it at a
    /// quantized or bf16 width is refused rather than read at the wrong stride.
    private func encodeHyperConnectionCoefficients(
        _ encoder: MTLComputeCommandEncoder,
        layer: Int,
        function: DeepSeekV41TensorRole,
        base: DeepSeekV41TensorRole,
        scale: DeepSeekV41TensorRole,
        pre: MTLBuffer,
        post: MTLBuffer,
        comb: MTLBuffer
    ) throws {
        guard let projection = weights.tensor(function, layer: layer),
              let buffer = shardBuffer(projection) else {
            throw DeepSeekV41EngineError.missingTensor(function.rawValue)
        }
        guard DeepSeekV41DtypeWidth.bytesPerValue(dtype: projection.dtype) == 4 else {
            throw DeepSeekV41EngineError.unsupportedDtype(
                "\(function.rawValue) (expected FP32)", projection.dtype
            )
        }
        guard let baseBuffer = vectors.buffer(base, layer: layer),
              let scaleBuffer = vectors.buffer(scale, layer: layer) else {
            throw DeepSeekV41EngineError.missingTensor(base.rawValue)
        }

        encoder.setComputePipelineState(pipelines.hcProject)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(workspace.streams, offset: 0, index: 1)
        encoder.setBuffer(workspace.hcLogits, offset: 0, index: 2)
        var weightOffset = UInt64(projection.byteOffset)
        var normalizationEpsilon = config.rmsNormEps
        var width = UInt32(mult * shapes.hidden)
        var outDim = UInt32(mixCount)
        encoder.setBytes(&weightOffset, length: 8, index: 3)
        encoder.setBytes(&normalizationEpsilon, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&outDim, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: mixCount, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(mixCount)
        )
        encoder.memoryBarrier(scope: .buffers)

        encoder.setComputePipelineState(pipelines.hcSplit)
        encoder.setBuffer(workspace.hcLogits, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(baseBuffer, offset: 0, index: 2)
        encoder.setBuffer(pre, offset: 0, index: 3)
        encoder.setBuffer(post, offset: 0, index: 4)
        encoder.setBuffer(comb, offset: 0, index: 5)
        var copies = UInt32(mult)
        var iterations = UInt32(max(0, config.hcSinkhornIterations))
        var epsilon = config.hcEps
        encoder.setBytes(&copies, length: 4, index: 6)
        encoder.setBytes(&iterations, length: 4, index: 7)
        encoder.setBytes(&epsilon, length: 4, index: 8)
        // The split is one thread: `hc_mult` is 4 and the schedule is serial.
        encoder.dispatchThreads(
            MTLSize(width: 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    /// `hc_pre`: collapse the stack with `mix` into `output`, optionally snapshotting
    /// the stack into `workspace.hcResidual` for the `hc_post` that follows.
    private func encodeHyperConnectionPre(
        _ encoder: MTLComputeCommandEncoder,
        mix: MTLBuffer,
        output: MTLBuffer,
        snapshotting: Bool
    ) throws {
        encoder.setComputePipelineState(pipelines.hcPre)
        encoder.setBuffer(workspace.streams, offset: 0, index: 0)
        encoder.setBuffer(mix, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        if snapshotting {
            encoder.setBuffer(workspace.hcResidual, offset: 0, index: 3)
        } else {
            encoder.setBuffer(nil, offset: 0, index: 3)
        }
        var width = UInt32(shapes.hidden)
        var copies = UInt32(mult)
        encoder.setBytes(&width, length: 4, index: 4)
        encoder.setBytes(&copies, length: 4, index: 5)
        encoder.dispatchThreads(
            MTLSize(width: shapes.hidden, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(shapes.hidden)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    /// `hc_post`: expand `output` to every copy and mix the snapshotted block input
    /// back in through `comb`.
    private func encodeHyperConnectionPost(
        _ encoder: MTLComputeCommandEncoder,
        output: MTLBuffer,
        gains: MTLBuffer,
        comb: MTLBuffer
    ) throws {
        encoder.setComputePipelineState(pipelines.hcPost)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBuffer(workspace.hcResidual, offset: 0, index: 1)
        encoder.setBuffer(gains, offset: 0, index: 2)
        encoder.setBuffer(comb, offset: 0, index: 3)
        encoder.setBuffer(workspace.streams, offset: 0, index: 4)
        var width = UInt32(shapes.hidden)
        var copies = UInt32(mult)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.setBytes(&copies, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: shapes.hidden, height: mult, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threadgroup(shapes.hidden).width, height: 1, depth: 1
            )
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    private func encodeCopy(
        _ encoder: MTLComputeCommandEncoder,
        from source: MTLBuffer,
        to destination: MTLBuffer,
        count: Int,
        destinationByteOffset: Int = 0
    ) throws {
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.copy)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: destinationByteOffset, index: 1)
        var length = UInt32(count)
        encoder.setBytes(&length, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroup(count)
        )
        encoder.memoryBarrier(scope: .buffers)
    }

    /// Rotates the rope slice of every `headDim`-wide block in `vector`.
    ///
    /// `sliceStart`/`sliceLength` select the rotated slice inside each block, which is
    /// how the decoupled payload keeps the `d_nope` prefix out of the rotation, and
    /// `inverse` conjugates it so the attention output can be returned to the
    /// unrotated frame its projection was trained on.
    private func encodeRope(
        _ encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer,
        rowWidth: Int,
        headDim: Int,
        sliceStart: Int,
        sliceLength: Int,
        table: DeepSeekV41RopeTable,
        latent: Bool,
        position: Int,
        inverse: Bool = false
    ) throws {
        let tables = latent ? latentFrequencies : queryFrequencies
        guard let frequencies = tables[table] else {
            throw DeepSeekV41EngineError.allocationFailed("a RoPE frequency table")
        }
        let pairCount = max(0, sliceLength) / 2
        guard pairCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.rope)
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(frequencies, offset: 0, index: 1)
        var width = UInt32(rowWidth)
        var localHeadDim = UInt32(max(0, headDim))
        var pairs = UInt32(min(pairCount, frequencies.length / MemoryLayout<Float>.stride))
        var base = UInt32(max(0, position))
        var batch: UInt32 = 1
        var offset = UInt32(max(0, sliceStart))
        var conjugate: UInt32 = inverse ? 1 : 0
        encoder.setBytes(&width, length: 4, index: 2)
        encoder.setBytes(&localHeadDim, length: 4, index: 3)
        encoder.setBytes(&pairs, length: 4, index: 4)
        encoder.setBytes(&base, length: 4, index: 5)
        encoder.setBytes(&batch, length: 4, index: 6)
        encoder.setBytes(&offset, length: 4, index: 7)
        encoder.setBytes(&conjugate, length: 4, index: 8)
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
        rowIndices: MTLBuffer,
        rowIndexOffset: Int,
        rowCount: Int,
        destinationRowOffset: Int
    ) throws {
        guard rowCount > 0 else { return }
        // The kernel indexes `rowIndices[row]` from zero, so the bound buffer has
        // to start at this gather's first row. Binding offset 0 here is what made
        // the compressed gather re-read the window rows: the compressed indices
        // are written after the window's in the shared list.
        let indexByteOffset = rowIndexOffset * MemoryLayout<UInt32>.stride
        let elementOffset = byteOffset / max(1, geometry.precision.valueBytesPerElement)
        encoder.setComputePipelineState(pipelines.gatherLatents)
        encoder.setBuffer(cacheBuffer, offset: 0, index: 0)
        encoder.setBuffer(rowIndices, offset: indexByteOffset, index: 1)
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
    private func encodeRowIndices(
        _ rows: [Int],
        into buffer: MTLBuffer,
        at offset: Int
    ) throws {
        guard !rows.isEmpty else { return }
        guard offset + rows.count <= buffer.length / MemoryLayout<UInt32>.stride else {
            throw DeepSeekV41EngineError.allocationFailed(
                "\(offset + rows.count) gather rows against a "
                    + "\(buffer.length / MemoryLayout<UInt32>.stride)-row buffer"
            )
        }
        let pointer = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for (index, row) in rows.enumerated() {
            pointer[offset + index] = UInt32(max(0, row))
        }
    }
}
