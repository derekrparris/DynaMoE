//
//  DeepSeekV41Engine.swift
//  DynaMoE
//
//  The runtime: resolves the checkpoint into device tensors, owns the CSA2 cache,
//  and drives the block forward on Metal.
//
//  This is the piece `CompletionEngine.launchTurn` delegates to for a DeepSeek-V4.1
//  checkpoint. It is deliberately separate from the generic decode loop: that loop
//  binds per-head K/V caches, a single QKV projection and a `model.`-rooted tensor
//  catalog, none of which this architecture has. Running it anyway produces text
//  from the wrong graph, which is why the launch refuses rather than guesses.
//
//  Two properties are load-bearing for correctness and both are enforced here
//  rather than assumed:
//
//    * THE CATALOG FAILS CLOSED. Every tensor is resolved to a role at load and
//      any name the catalog cannot place, and any role the plan needs but the
//      checkpoint lacks, is reported. A silently-ignored tensor is a model that
//      runs at full speed and answers wrongly.
//    * THE GRAPH IS THE REFERENCE'S. The stage order, the cache rows a position
//      reads, and the residual bookkeeping are the ones `DeepSeekV41CPUForward`
//      implements and the tests assert, so a kernel disagreement is a kernel bug
//      rather than a second, divergent re-derivation.
//
//  Residency: the compressed caches and window rings are the only long-lived
//  allocations (a few hundred MB at a million tokens). The dense backbone lives in
//  the shard buffers the loader already mapped, and the routed experts are read
//  through those same mappings a page at a time, so the arena only decides what to
//  prefetch.
//

import Foundation
import Metal

// MARK: - Device tensor

/// One resolved tensor, addressed as `(bucket, byte offset)` into the segmented
/// shard buffers plus the shape the graph expects.
nonisolated public struct DeepSeekV41DeviceTensor: Sendable, Equatable {
    public let shardIndex: UInt32
    public let byteOffset: Int
    /// Rows (`out` dimension) for a matrix, or the element count for a vector.
    public let rows: Int
    public let columns: Int
    public let dtype: String

    public var isFP8: Bool {
        let upper = dtype.uppercased()
        return upper.contains("F8") || upper == "FP8" || upper.contains("E4M3") || upper.contains("E5M2")
    }
    public var isBF16: Bool { dtype.uppercased().contains("BF16") }
    /// True only for IEEE half. `"BF16".contains("F16")` is true, so the bf16 check
    /// has to come first or a bf16 tensor is dispatched as if it were fp16.
    public var isF16: Bool {
        guard !isFP8, !isBF16 else { return false }
        let upper = dtype.uppercased()
        return upper.contains("F16") || upper.contains("FLOAT16")
    }
    public var isScaleCompanion: Bool {
        let upper = dtype.uppercased()
        return upper.contains("E8M0") || upper.contains("U8") || upper.contains("UINT8")
    }

    /// Elements in one matrix row, which is the count a `gemv` binds as `inDim`.
    public var inDimension: Int { columns }
    public var outDimension: Int { rows }
}

// MARK: - Device weights

/// Resolves catalog roles to device tensors, and owns the Engram readers.
nonisolated public final class DeepSeekV41DeviceWeights: @unchecked Sendable {

    public let catalog: DeepSeekV41WeightCatalog
    /// Non-nil when the checkpoint cannot be run; the launch surfaces these.
    public let problems: [String]
    /// One reader per Engram table, keyed by layer.
    public let engramReaders: [Int: DeepSeekV41EngramReader]
    public let engramLayouts: [Int: DeepSeekV41EngramLayout]

    private var tensors: [String: DeepSeekV41DeviceTensor]
    private var scales: [String: DeepSeekV41DeviceTensor]

    public init(
        snapshotDir: URL?,
        summary: ModelSummary,
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan]
    ) {
        let catalog = DeepSeekV41WeightCatalog.from(summary: summary, layerCount: config.numHiddenLayers)
        self.catalog = catalog
        var problems = catalog.validate(config: config, plan: plan)

        var tensors: [String: DeepSeekV41DeviceTensor] = [:]
        var scales: [String: DeepSeekV41DeviceTensor] = [:]
        // Tensor names are unique in a safetensors checkpoint, so the name is the
        // address. `offsetStart` is relative to the shard its index names, which is
        // exactly what the segmented buffers are keyed by.
        for tensor in summary.tensors {
            let shape = DeepSeekV41WeightCatalog.shape(from: tensor.shapeDisplay)
            // Vectors are resolved as one row with `columns` 1 so a 1-D lookup
            // (a norm, a bias) resolves instead of silently returning nil. A rank
            // above 2 has no kernel at all, so it is reported rather than being
            // flattened into a matrix that would be read at the wrong stride.
            let rows: Int
            let columns: Int
            switch shape.count {
            case 0:
                rows = 1
                columns = 1
            case 1:
                rows = shape[0]
                columns = 1
            case 2:
                rows = shape[0]
                columns = shape[1]
            default:
                problems.append(
                    "\(tensor.name) is rank \(shape.count), which no DeepSeek V4.1 kernel reads"
                )
                continue
            }
            let device = DeepSeekV41DeviceTensor(
                shardIndex: tensor.shardIndex,
                byteOffset: Int(tensor.offsetStart),
                rows: rows,
                columns: columns,
                dtype: tensor.dtype
            )
            if tensor.name.contains("scale") || tensor.name.contains("scales") {
                scales[tensor.name] = device
            } else {
                tensors[tensor.name] = device
            }
        }
        self.tensors = tensors
        self.scales = scales
        self.problems = problems

        // Engram tables are opened here so a truncated download fails at load rather
        // than mid-generation. They are never mapped: the reader preads single rows.
        var readers: [Int: DeepSeekV41EngramReader] = [:]
        var layouts: [Int: DeepSeekV41EngramLayout] = [:]
        if let snapshotDir {
            for spec in config.engramTables {
                if let layout = try? DeepSeekV41EngramLayout(spec: spec) {
                    layouts[spec.layer] = layout
                }
                guard let slot = catalog.tensor(.engramTable, layer: spec.layer),
                      let tensor = summary.tensors.first(where: { $0.name == slot.name }) else {
                    continue
                }
                let shardName = summary.shards.first(where: { $0.index == tensor.shardIndex })?.filename
                    ?? "model-\(tensor.shardIndex).safetensors"
                let tableURL = snapshotDir.appendingPathComponent(shardName)
                // Requests are table-relative; the tensor's own offset inside the
                // shard is the base the reader adds before every pread.
                if let reader = try? DeepSeekV41EngramReader(
                    fileURL: tableURL,
                    spec: spec,
                    baseOffset: Int(tensor.offsetStart)
                ) {
                    readers[spec.layer] = reader
                }
            }
        }
        self.engramReaders = readers
        self.engramLayouts = layouts
    }

    public func tensor(
        _ role: DeepSeekV41TensorRole,
        layer: Int? = nil,
        expert: Int? = nil
    ) -> DeepSeekV41DeviceTensor? {
        guard let slot = catalog.tensor(role, layer: layer, expert: expert) else { return nil }
        return tensors[slot.name]
    }

    public func scaleTensor(
        _ role: DeepSeekV41TensorRole,
        scope: DeepSeekV41TensorScope = .backbone,
        layer: Int? = nil,
        expert: Int? = nil
    ) -> DeepSeekV41DeviceTensor? {
        let key = DeepSeekV41WeightCatalog.key(
            role, scope: scope, layer: layer, expert: expert, scale: true
        )
        guard let slot = catalog.slot(matchingKey: key) else { return nil }
        return scales[slot.ref.name]
    }

    public func close() {
        for reader in engramReaders.values { reader.closeNow() }
    }
}

// MARK: - Pipelines

/// Every pipeline the forward binds. Built once per load; a missing function is a
/// hard load failure rather than a silently skipped stage.
nonisolated public struct DeepSeekV41Pipelines {
    public let fp8Block32Gemv: MTLComputePipelineState
    public let mxfp4Gemv: MTLComputePipelineState
    public let mxfp4SwiGLU: MTLComputePipelineState
    public let mxfp4DownAccumulate: MTLComputePipelineState
    public let engramDequantRows: MTLComputePipelineState
    public let hcSinkhorn: MTLComputePipelineState
    public let rope: MTLComputePipelineState
    public let storeLatent: MTLComputePipelineState
    public let gatherLatents: MTLComputePipelineState
    public let attention: MTLComputePipelineState
    public let poolLatents: MTLComputePipelineState
    public let indexerScores: MTLComputePipelineState
    public let router: MTLComputePipelineState
    public let hcMix: MTLComputePipelineState
    public let hcWrite: MTLComputePipelineState
    public let clear: MTLComputePipelineState
    public let vectorAdd: MTLComputePipelineState
    /// Gain-vector RMSNorm the executor binds (the shard-offset `rmsnorm_bf16`
    /// assumes a bf16 gain, and the executor resolves every gain to FP32 once).
    public let rmsNormGain: MTLComputePipelineState
    /// Dense E4M3 32x32 ue8m0 FFN pair. The V3 128x128 kernels cannot be
    /// reuse-pointed at this grid, and the routed experts are MXFP4, so the dense
    /// half needs its own two.
    public let fp8SwiGLU: MTLComputePipelineState
    public let fp8DownAccumulate: MTLComputePipelineState
    /// BF16 fallback for a checkpoint whose FFN was not quantized.
    public let bf16Gemv: MTLComputePipelineState
    public let bf16SwiGLU: MTLComputePipelineState
    public let bf16DownAccumulate: MTLComputePipelineState
    public let embeddingBF16: MTLComputePipelineState
    public let embeddingMXFP8: MTLComputePipelineState

    public init(library: MTLLibrary, device: MTLDevice) throws {
        func make(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw DeepSeekV41EngineError.missingKernel(name)
            }
            return try device.makeComputePipelineState(function: function)
        }
        self.fp8Block32Gemv = try make("dsv41_fp8_block32_gemv")
        self.mxfp4Gemv = try make("dsv41_mxfp4_gemv")
        self.mxfp4SwiGLU = try make("dsv41_mxfp4_swiglu_gate_up")
        self.mxfp4DownAccumulate = try make("dsv41_mxfp4_down_accumulate")
        self.engramDequantRows = try make("dsv41_engram_dequant_rows")
        self.hcSinkhorn = try make("dsv41_hc_sinkhorn")
        self.rope = try make("dsv41_rope_latent")
        self.storeLatent = try make("dsv41_store_latent_f16")
        self.gatherLatents = try make("dsv41_gather_latents_f16")
        self.attention = try make("dsv41_attention_csa2_f32")
        self.poolLatents = try make("dsv41_pool_latents_f32")
        self.indexerScores = try make("dsv41_indexer_scores_f32")
        self.router = try make("dsv41_router_topk")
        self.hcMix = try make("dsv41_hc_mix_f32")
        self.hcWrite = try make("dsv41_hc_write_f32")
        self.clear = try make("clear_vector_f32")
        self.vectorAdd = try make("vector_add_f32")
        self.rmsNormGain = try make("dsv41_rmsnorm_f32_gain")
        self.fp8SwiGLU = try make("dsv41_fp8_block32_swiglu_gate_up")
        self.fp8DownAccumulate = try make("dsv41_fp8_block32_down_accumulate")
        self.bf16Gemv = try make("bf16_gemv")
        self.bf16SwiGLU = try make("bf16_swiglu_gate_up")
        self.bf16DownAccumulate = try make("bf16_down_proj_accumulate")
        self.embeddingBF16 = try make("lookup_embeddings_bf16")
        self.embeddingMXFP8 = try make("lookup_embeddings_mxfp8")
    }
}

// MARK: - Errors

nonisolated public enum DeepSeekV41EngineError: Error, CustomStringConvertible {
    case missingKernel(String)
    case modelNotRunnable([String])
    case allocationFailed(String)
    case missingTensor(String)
    case unsupportedDtype(String, String)
    case contextTooLong(Int, Int)
    case outOfOrderPosition(Int, Int)
    case unsupportedStage(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .missingKernel(let name):
            return "DeepSeek V4.1 runtime is missing the Metal kernel '\(name)'"
        case .modelNotRunnable(let problems):
            let shown = problems.prefix(6).joined(separator: "; ")
            let more = problems.count > 6 ? " (+\(problems.count - 6) more)" : ""
            return "DeepSeek V4.1 checkpoint cannot be run: \(shown)\(more)"
        case .allocationFailed(let what):
            return "DeepSeek V4.1 runtime could not allocate \(what)"
        case .missingTensor(let name):
            return "DeepSeek V4.1 checkpoint is missing the tensor '\(name)'"
        case let .unsupportedDtype(tensorName, tensorDtype):
            return "DeepSeek V4.1 tensor '\(tensorName)' has dtype '\(tensorDtype)', which the engine has no kernel for"
        case let .contextTooLong(requested, maximum):
            return "DeepSeek V4.1 turn needs \(requested) positions but the cache was provisioned for \(maximum)"
        case let .outOfOrderPosition(requested, stored):
            return "DeepSeek V4.1 forward was asked for position \(requested) while the cache holds \(stored)"
        case .unsupportedStage(let stage):
            return "DeepSeek V4.1 stage '\(stage)' has no device implementation"
        case .commandFailed(let detail):
            return "DeepSeek V4.1 Metal work failed: \(detail)"
        }
    }
}

// MARK: - Runtime

/// Last runtime-initialization failure, behind a lock so a raced load cannot tear
/// the string.
nonisolated private final class InitErrorBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: String?

    var stored: String? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
        set {
            lock.lock()
            value = newValue
            lock.unlock()
        }
    }
}

/// One loaded DeepSeek-V4.1 model plus its caches. Single-flight, like the rest of
/// the engine: a turn owns it for its duration.
nonisolated public final class DeepSeekV41Runtime: @unchecked Sendable {

    public let config: DeepSeekV41Config
    public let plan: [DeepSeekV41LayerPlan]
    public let catalog: DeepSeekV41WeightCatalog
    public let weights: DeepSeekV41DeviceWeights
    public let cache: DeepSeekV41KVCache
    public let memoryPlan: DeepSeekV41MemoryPlan
    public let arena: DeepSeekV41ExpertArena
    public let dspark: DeepSeekV41DSparkPlan
    public let contextLength: Int
    /// The precision the KV cache is actually allocated at. The checkpoint's
    /// design is FP8; the runtime runs FP16 until the FP8 quantize/dequantize
    /// pair is checked against the reference, and reports the difference.
    public let cachePrecision: DeepSeekV41CachePrecision

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let pipelines: DeepSeekV41Pipelines
    /// The zero-copy shard mappings the executor binds tensors through.
    public let shardBuffers: [UInt32: MTLBuffer]

    /// The device executor, built on first use: resolving every norm gain and
    /// RoPE table is load work a turn needs and a pure diagnostic does not.
    private var cachedForward: DeepSeekV41MetalForward?

    /// Reused per-token workspace, allocated once. Public because the device
    /// executor binds it directly rather than re-allocating per turn.
    public let workspace: DeepSeekV41Workspace

    /// Why the last `init` returned nil. The initializer is failable and its caller
    /// only sees "could not be built", so without this the difference between a
    /// missing kernel, a missing Metal device and a failed allocation is lost.
    public static var lastInitializationError: String? { initializationError.stored }

    private static let initializationError = InitErrorBox()

    static func record(_ reason: String) {
        initializationError.stored = reason
    }

    public init?(
        snapshotDir: URL?,
        summary: ModelSummary,
        config: DeepSeekV41Config,
        shardBuffers: [UInt32: MTLBuffer],
        contextLength: Int,
        cachePrecision: DeepSeekV41CachePrecision = .fp16
    ) {
        guard let plan = try? config.makeLayerPlan() else {
            DeepSeekV41Runtime.record("the per-layer plan could not be built from config.json")
            return nil
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            DeepSeekV41Runtime.record("no Metal device is available")
            return nil
        }
        guard let commandQueue = device.makeCommandQueue() else {
            DeepSeekV41Runtime.record("no Metal command queue could be created")
            return nil
        }
        guard let library = device.makeDefaultLibrary() else {
            DeepSeekV41Runtime.record("the app's default Metal library is missing")
            return nil
        }
        let pipelines: DeepSeekV41Pipelines
        do {
            pipelines = try DeepSeekV41Pipelines(library: library, device: device)
        } catch {
            DeepSeekV41Runtime.record("\(error)")
            return nil
        }

        let weights = DeepSeekV41DeviceWeights(
            snapshotDir: snapshotDir,
            summary: summary,
            config: config,
            plan: plan
        )
        let geometry = DeepSeekV41CacheGeometry.make(
            config: config,
            plan: plan,
            contextLength: contextLength,
            precision: cachePrecision
        )
        let layout = DeepSeekV41CacheLayout(geometry: geometry, plan: plan)
        guard let cache = DeepSeekV41KVCache(device: device, layout: layout) else {
            DeepSeekV41Runtime.record(
                "the \(layout.totalBytes)-byte KV cache could not be allocated"
            )
            return nil
        }
        let shapes = DeepSeekV41TensorShapes(config: config)
        guard let workspace = DeepSeekV41Workspace(device: device, shapes: shapes, config: config) else {
            DeepSeekV41Runtime.record("the per-token workspace could not be allocated")
            return nil
        }

        self.config = config
        self.plan = plan
        self.catalog = weights.catalog
        self.weights = weights
        self.cache = cache
        self.memoryPlan = DeepSeekV41MemoryPlan.make(
            config: config,
            plan: plan,
            catalog: weights.catalog,
            contextLength: contextLength,
            recordBytesOverride: cachePrecision.recordBytes(width: config.cacheRecordWidth)
        )
        self.arena = DeepSeekV41ExpertArena(slotCount: max(1, config.numActivatedExperts * 2))
        self.dspark = DeepSeekV41DSparkPlan.make(config: config)
        self.contextLength = contextLength
        self.cachePrecision = cachePrecision
        self.device = device
        self.commandQueue = commandQueue
        self.pipelines = pipelines
        self.workspace = workspace
        self.shardBuffers = shardBuffers
    }

    public var isRunnable: Bool { weights.problems.isEmpty }

    /// The device executor for this checkpoint. Built once and kept: it holds the
    /// per-layer gain vectors and RoPE tables.
    public func forward() -> DeepSeekV41MetalForward {
        if let cachedForward { return cachedForward }
        let built = DeepSeekV41MetalForward(runtime: self, shardBuffers: shardBuffers)
        cachedForward = built
        return built
    }

    /// One-line readiness report for the load log.
    public func describe() -> String {
        var lines: [String] = []
        lines.append("cache: \(cache.layout.describe()) at \(cachePrecision.rawValue)")
        lines.append("memory: \(memoryPlan.describe(expertSlots: arena.capacity))")
        lines.append(
            "dspark: " + (dspark.isEnabled
                ? "block \(dspark.blockSize) over layers \(dspark.targetLayerIDs)"
                : "disabled")
        )
        if let vision = config.visionTowerPlan {
            lines.append(vision.describe())
        }
        lines.append(forward().describe())
        if !weights.problems.isEmpty {
            lines.append("not runnable: \(DeepSeekV41EngineError.modelNotRunnable(weights.problems))")
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Workspace

/// Per-token scratch. Sized from the graph so the reference and the device path
/// agree, and allocated once because a per-token allocation would dominate decode.
nonisolated public final class DeepSeekV41Workspace: @unchecked Sendable {
    public let hidden: MTLBuffer
    public let normed: MTLBuffer
    public let query: MTLBuffer
    public let latent: MTLBuffer
    public let gathered: MTLBuffer
    public let attentionOut: MTLBuffer
    public let output: MTLBuffer
    public let routerLogits: MTLBuffer
    public let expertIndices: MTLBuffer
    public let expertWeights: MTLBuffer
    public let intermediate: MTLBuffer
    /// The shared expert's intermediate width, which is not the routed width.
    public let sharedIntermediate: MTLBuffer
    /// One pooled compressed row, before it is written to the cache.
    public let pooled: MTLBuffer
    /// The compressor's per-position `wkv(x)` results for the group being filled,
    /// `maxRatio * latentWidth` floats. The compressor projects the layer input, so
    /// these are not the same rows the window cache holds.
    public let compressorLatents: MTLBuffer
    /// The compressor's per-position `wgate(x)` scores, the same geometry. The
    /// softmax that pools them runs per DIMENSION over the group.
    public let compressorScores: MTLBuffer
    /// The `wq_a` / `wo_a` rank-width scratch.
    public let reduced: MTLBuffer
    public let moeOut: MTLBuffer
    public let streams: MTLBuffer
    public let logits: MTLBuffer
    public let rowIndices: MTLBuffer
    public let tokenIDs: MTLBuffer
    /// Rows the gather allocates for one layer's attention: window + compressed.
    public let gatherRows: Int

    public init?(device: MTLDevice, shapes: DeepSeekV41TensorShapes, config: DeepSeekV41Config) {
        let mult = max(1, config.hcMult)
        let maxCompressed = max(1, config.compressRatios.max() ?? 1)
        // Rows one position gathers: its window plus whatever the addressing rule
        // may select. `index_topk == 0` would mean "the whole compressed cache", so
        // the candidate-block budget bounds it and the executor reports a cache
        // provisioned wider than this.
        let gather = DeepSeekV41DevicePlan.gatherRowBudget(config: config)
        let float4: (Int) -> Int = { $0 * MemoryLayout<Float>.stride }

        func make(_ bytes: Int) -> MTLBuffer? {
            device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)
        }
        guard let hidden = make(float4(shapes.hidden)),
              let normed = make(float4(shapes.hidden)),
              let query = make(float4(shapes.queryWidth)),
              let latent = make(float4(shapes.latentWidth)),
              let gathered = make(float4(gather * shapes.latentWidth)),
              let attentionOut = make(float4(shapes.attentionOutputWidth)),
              let output = make(float4(shapes.hidden)),
              let routerLogits = make(float4(config.numRoutedExperts)),
              let expertIndices = make(max(4 * config.numActivatedExperts, 16)),
              let expertWeights = make(float4(config.numActivatedExperts)),
              let intermediate = make(float4(shapes.expertIntermediate)),
              let sharedIntermediate = make(float4(shapes.sharedIntermediate)),
              let pooled = make(float4(shapes.latentWidth)),
              let compressorLatents = make(float4(maxCompressed * shapes.latentWidth)),
              let compressorScores = make(float4(maxCompressed * shapes.latentWidth)),
              let reduced = make(float4(max(1, max(shapes.qLoraRank, shapes.outputReducedWidth)))),
              let moeOut = make(float4(shapes.hidden)),
              let streams = make(float4(mult * shapes.hidden)),
              let logits = make(float4(config.vocabSize)),
              let rowIndices = make(4 * gather),
              let tokenIDs = make(4) else {
            return nil
        }

        self.hidden = hidden
        self.normed = normed
        self.query = query
        self.latent = latent
        self.gathered = gathered
        self.attentionOut = attentionOut
        self.output = output
        self.routerLogits = routerLogits
        self.expertIndices = expertIndices
        self.expertWeights = expertWeights
        self.intermediate = intermediate
        self.sharedIntermediate = sharedIntermediate
        self.pooled = pooled
        self.compressorLatents = compressorLatents
        self.compressorScores = compressorScores
        self.reduced = reduced
        self.moeOut = moeOut
        self.streams = streams
        self.logits = logits
        self.rowIndices = rowIndices
        self.tokenIDs = tokenIDs
        self.gatherRows = gather
    }
}
