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
        // The primes are drawn once across every Engram layer, so the layouts are
        // built together; one table in isolation would address the wrong rows.
        let layouts = (try? DeepSeekV41EngramLayout.make(specs: config.engramTables)) ?? [:]
        if let snapshotDir {
            for spec in config.engramTables {
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
    /// The Engram gate and residual add, one threadgroup per hyper-connection copy.
    public let engramGate: MTLComputePipelineState
    public let hcSinkhorn: MTLComputePipelineState
    /// The mHC rule: project the flattened stack, split the logits, collapse with a
    /// mix, expand back through `comb`.
    public let hcProject: MTLComputePipelineState
    public let hcSplit: MTLComputePipelineState
    public let hcPre: MTLComputePipelineState
    public let hcPost: MTLComputePipelineState
    public let hcIdentityMix: MTLComputePipelineState
    public let rope: MTLComputePipelineState
    public let storeLatent: MTLComputePipelineState
    public let gatherLatents: MTLComputePipelineState
    /// CSA2 attention over the gathered block. The block is FP16, so this is the
    /// half-key kernel; the FP32 form is kept in the shader library as the baseline
    /// the measurement A/Bs against (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #17").
    public let attention: MTLComputePipelineState
    public let poolLatents: MTLComputePipelineState
    public let indexerScores: MTLComputePipelineState
    public let router: MTLComputePipelineState
    public let clear: MTLComputePipelineState
    public let copy: MTLComputePipelineState
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
    public let bf16GemvRows: MTLComputePipelineState
    public let bf16GemvBlock: MTLComputePipelineState
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
        self.engramGate = try make("dsv41_engram_gate_f32")
        self.hcSinkhorn = try make("dsv41_hc_sinkhorn")
        self.hcProject = try make("dsv41_hc_project_f32")
        self.hcSplit = try make("dsv41_hc_split_f32")
        self.hcPre = try make("dsv41_hc_pre_f32")
        self.hcPost = try make("dsv41_hc_post_f32")
        self.hcIdentityMix = try make("dsv41_hc_identity_mix_f32")
        self.rope = try make("dsv41_rope_latent")
        self.storeLatent = try make("dsv41_store_latent_f16")
        self.gatherLatents = try make("dsv41_gather_latents_f16")
        self.attention = try make("dsv41_attention_csa2_f16")
        self.poolLatents = try make("dsv41_pool_latents_f32")
        self.indexerScores = try make("dsv41_indexer_scores_f32")
        self.router = try make("dsv41_router_topk")
        self.clear = try make("clear_vector_f32")
        self.copy = try make("dsv41_copy_f32")
        self.vectorAdd = try make("vector_add_f32")
        self.rmsNormGain = try make("dsv41_rmsnorm_f32_gain")
        self.fp8SwiGLU = try make("dsv41_fp8_block32_swiglu_gate_up")
        self.fp8DownAccumulate = try make("dsv41_fp8_block32_down_accumulate")
        self.bf16Gemv = try make("bf16_gemv")
        self.bf16GemvRows = try make("bf16_gemv_rows")
        self.bf16GemvBlock = try make("bf16_gemv_rows8")
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
    /// A resumed turn found resident state that is not this prompt's.
    case prefixStateMismatch(position: Int)
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
        case .prefixStateMismatch(let position):
            return "DeepSeek V4.1 cannot continue the cached sequence: its token history diverges at position \(position)"
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
    /// Fills the arena slots from the shard files. Nil means the routed experts are
    /// bound through the zero-copy shard mappings instead, which is correct and
    /// several times slower: every expert matrix is a page-fault walk.
    public let expertStreamer: DeepSeekV41ExpertStreamer?
    /// Why there is no streamer, so the readiness line names a reason instead of
    /// leaving "no streaming" to be read as "streaming is fine here".
    public let expertStreamerProblem: String?
    public let dspark: DeepSeekV41DSparkPlan
    public let contextLength: Int
    /// The precision the KV cache is actually allocated at. The checkpoint's
    /// design is FP8; the runtime runs FP16 until the FP8 quantize/dequantize
    /// pair is checked against the reference, and reports the difference.
    public let cachePrecision: DeepSeekV41CachePrecision

    /// Debug-only: let the two Engram layers run with no memory read at all
    /// instead of reading their tables. The opt-out is deliberate (a wrong
    /// combination reads the right rows and mixes them wrongly, which looks like
    /// "the memory model does nothing"), so this is off unless the user opts in
    /// with `dynamoe_dsv41_engram_off`, and the load log and readiness report say
    /// so every time. It exists to exercise the other 38 layers on a machine that
    /// has already paid for the 510 GB download.
    public let allowEngramSkip: Bool

    /// The compressed vocabulary the Engram hashes are drawn from, built from the
    /// checkpoint's tokenizer at load. Nil when the checkpoint has no Engram table
    /// or no tokenizer could be read, in which case `engramProblem` says why.
    public let engramTokenMap: DeepSeekV41EngramTokenMap?
    /// Why the token map is missing. The device executor turns it into a wiring
    /// problem rather than reading rows through a stand-in map.
    public let engramProblem: String?

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

    /// `defaults write DRP.DynaMoE dynamoe_dsv41_engram_off -bool YES`
    ///
    /// Debug-only opt-out: run the Engram layers with no memory read instead of
    /// refusing the turn. Off by default, and the readiness report names every
    /// layer it applies to, because the whole point of the refusal is that a
    /// missing or miscombined memory reads as "the model is fine, just worse".
    public static let engramOffDefaultKey = "dynamoe_dsv41_engram_off"

    public static var engramSkipRequested: Bool {
        UserDefaults.standard.bool(forKey: engramOffDefaultKey)
    }

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
        cachePrecision: DeepSeekV41CachePrecision = .fp16,
        allowEngramSkip: Bool = false
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
        guard let workspace = DeepSeekV41Workspace(
                  device: device,
                  shapes: shapes,
                  config: config,
                  compressedOwners: plan.filter(\.ownsCompressedKV).count,
                  // Prefill runs a block of positions per dispatch, so every
                  // per-position vector is replicated for the block. The ring
                  // caps the block: it holds `window` rows, and a wider block
                  // would write a row before the query that needs it attended.
                  slots: DeepSeekV41DevicePlan.chunkSlots(
                      config: config, contextLength: max(1, contextLength)
                  )
              ) else {
            DeepSeekV41Runtime.record("the per-token workspace could not be allocated")
            return nil
        }

        // One arena slot holds one whole `(layer, expert)` pair, so sizing it from
        // RAM is what turns a cold cache on every token into a working set: an
        // active expert is ~19 MB of MXFP4, and a 128 GB machine can hold a few
        // thousand of the 15,360 pairs that exist. Twelve slots (top-k x 2) was a
        // placeholder that guaranteed every expert was a miss.
        let memoryPlan = DeepSeekV41MemoryPlan.make(
            config: config,
            plan: plan,
            catalog: weights.catalog,
            contextLength: contextLength,
            recordBytesOverride: cachePrecision.recordBytes(width: config.cacheRecordWidth)
        )
        let physicalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let slotCount = min(
            DeepSeekV41MemoryPlan.affordableExpertSlots(plan: memoryPlan, physicalGB: physicalGB),
            max(1, config.numRoutedExperts * max(1, config.numHiddenLayers))
        )
        let arena = DeepSeekV41ExpertArena(slotCount: slotCount)

        // The routed mixture is a working set this machine reads, not a weight it
        // holds: 15,360 whole-expert pairs at ~19 MB each. Filling the arena from
        // the shards with pread is what makes a token's expert traffic a linear read
        // instead of one page fault per 16 KB, so the streamer is built here rather
        // than lazily. It is optional on purpose: a checkpoint whose experts do not
        // resolve to one slot layout still runs through the shard mappings, just
        // slowly, and the readiness line says which path is in use.
        var streamer: DeepSeekV41ExpertStreamer?
        var streamerProblem: String?
        if let snapshotDir {
            do {
                streamer = try DeepSeekV41ExpertStreamer(
                    device: device,
                    weights: weights,
                    summary: summary,
                    snapshotDir: snapshotDir,
                    arena: arena,
                    config: config,
                    plan: plan
                )
            } catch {
                streamerProblem = "\(error)"
            }
        } else {
            streamerProblem = "the snapshot directory is unknown, so the expert shards cannot be read"
        }

        self.config = config
        self.plan = plan
        self.catalog = weights.catalog
        self.weights = weights
        self.cache = cache
        self.memoryPlan = memoryPlan
        self.arena = arena
        self.expertStreamer = streamer
        self.expertStreamerProblem = streamerProblem
        self.dspark = DeepSeekV41DSparkPlan.make(config: config)
        self.contextLength = contextLength
        self.cachePrecision = cachePrecision
        self.allowEngramSkip = allowEngramSkip
        // The Engram hash reads the checkpoint's compressed vocabulary, so the map
        // is compiled from the shipped tokenizer once, at load. A checkpoint with no
        // table needs neither the parse nor the map.
        if !config.engramTables.isEmpty, !allowEngramSkip {
            if let snapshotDir {
                do {
                    let map = try DeepSeekV41EngramTokenMap.make(
                        snapshotDir: snapshotDir, config: config
                    )
                    self.engramTokenMap = map
                    self.engramProblem = nil
                } catch {
                    self.engramTokenMap = nil
                    self.engramProblem = "the Engram token map could not be built: \(error)"
                }
            } else {
                self.engramTokenMap = nil
                self.engramProblem = "the snapshot directory is unknown, so the Engram token map is unavailable"
            }
        } else {
            self.engramTokenMap = nil
            self.engramProblem = nil
        }
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

    /// Releases the descriptors and slot buffers this runtime owns.
    ///
    /// Called when a new runtime replaces it (a different checkpoint, or a turn that
    /// needs more context). Without it the Engram tables and the 48 expert shards
    /// stay open for the life of the process.
    public func shutdown() {
        cachedForward = nil
        weights.close()
        expertStreamer?.close()
        arena.reset()
    }

    /// One-line readiness report for the load log.
    public func describe() -> String {
        var lines: [String] = []
        lines.append("cache: \(cache.layout.describe()) at \(cachePrecision.rawValue)")
        lines.append("memory: \(memoryPlan.describe(expertSlots: arena.capacity))")
        if let expertStreamer {
            lines.append(expertStreamer.describe())
        } else {
            lines.append(
                "experts: read through the shard mappings (\(expertStreamerProblem ?? "no streamer"))"
            )
        }
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

// MARK: - Engram staging

/// Where one Engram table's rows are staged for the device path.
///
/// The table itself is never a buffer (~100 GB is past every `maxBufferLength`),
/// so one position's rows are preaded into `engramRaw` and dequantized into
/// `engramValues`. Both buffers carry one block PER TABLE rather than one shared
/// block, which is the double buffering the memory plan prices: the two Engram
/// layers are far apart in the stack, and giving each its own block means a
/// read for one table can never land under a kernel still reading the other.
nonisolated public struct DeepSeekV41EngramSlot: Sendable, Equatable {
    /// Layer that owns the table.
    public let layer: Int
    /// Index into `config.engramTables`, which is what `layer.engramTable` holds.
    public let table: Int
    /// `(max_ngram_size - 1) * n_heads` rows: the hash columns of one position.
    public let columns: Int
    public let headDimension: Int
    /// E4M3 values plus one E8M0 scale per 32.
    public let rowBytes: Int
    /// Byte offset of this table's raw row block in `engramRaw`.
    public let rawByteOffset: Int
    /// Element offset of this table's decoded rows in `engramValues`.
    public let valueOffset: Int

    public var rawByteCount: Int { columns * rowBytes }
    public var valueCount: Int { columns * headDimension }
    public var valueByteOffset: Int { valueOffset * MemoryLayout<Float>.stride }
    /// Width of the concatenated rows the `wkv` projection reads.
    public var flatWidth: Int { columns * headDimension }

    /// One slot per Engram table, in `config.engramTables` order so the index is
    /// `layer.engramTable`. A table with no hash columns gets an empty slot rather
    /// than being dropped: the caller looks slots up by index, and shifting the
    /// array would hand a layer its neighbour's rows.
    public static func slots(config: DeepSeekV41Config) -> [DeepSeekV41EngramSlot] {
        var rawCursor = 0
        var valueCursor = 0
        return config.engramTables.enumerated().map { table, spec in
            let columns = max(1, spec.maxNgramSize - 1) * spec.headCount
            let slot = DeepSeekV41EngramSlot(
                layer: spec.layer,
                table: table,
                columns: columns,
                headDimension: spec.headDimension,
                rowBytes: spec.rowBytes,
                rawByteOffset: rawCursor,
                valueOffset: valueCursor
            )
            rawCursor += slot.rawByteCount
            valueCursor += slot.valueCount
            return slot
        }
    }

    public init(
        layer: Int,
        table: Int,
        columns: Int,
        headDimension: Int,
        rowBytes: Int,
        rawByteOffset: Int,
        valueOffset: Int
    ) {
        self.layer = layer
        self.table = table
        self.columns = columns
        self.headDimension = headDimension
        self.rowBytes = rowBytes
        self.rawByteOffset = rawByteOffset
        self.valueOffset = valueOffset
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
    /// one block per compressed-KV owner, `groupCount * maxRatio * latentWidth`
    /// floats. The compressor projects the layer input, so these are not the same
    /// rows the window cache holds, and the state is per layer rather than shared:
    /// two owners pooling out of one block mixed their latents.
    public let compressorLatents: MTLBuffer
    /// The compressor's per-position `wgate(x)` scores, the same geometry. The
    /// softmax that pools them runs per DIMENSION over the group.
    public let compressorScores: MTLBuffer
    /// How many compressed-KV owners the compressor scratch is split across.
    public let compressorGroups: Int
    /// Slots one owner's compressor block holds, i.e. the widest compress ratio.
    public let compressorGroupWidth: Int
    /// The `wq_a` / `wo_a` rank-width scratch.
    public let reduced: MTLBuffer
    public let moeOut: MTLBuffer
    public let streams: MTLBuffer
    /// The `hc_*_fn` logits for the sub-layer being computed, `(2 + mult) * mult`.
    public let hcLogits: MTLBuffer
    /// The collapse weight the NEXT sub-layer reads. Reset to the identity mix at
    /// the start of every forward pass (`make_identity_pre_mix`).
    public let hcMix: MTLBuffer
    public let hcAttentionPre: MTLBuffer
    public let hcAttentionPost: MTLBuffer
    public let hcAttentionComb: MTLBuffer
    public let hcFFNPre: MTLBuffer
    public let hcFFNPost: MTLBuffer
    public let hcFFNComb: MTLBuffer
    /// The block input `hc_post` mixes back in. `hc_pre` writes it in the same pass
    /// that collapses the stack, because `hc_post` cannot read the streams in place.
    public let hcResidual: MTLBuffer
    public let logits: MTLBuffer
    public let rowIndices: MTLBuffer
    public let tokenIDs: MTLBuffer
    /// One staging block per Engram table: this position's raw rows, preaded.
    public let engramRaw: MTLBuffer
    /// The same rows dequantized to FP32, which is what the `wkv` projection reads.
    public let engramValues: MTLBuffer
    /// The `wkv` output for one Engram layer: one key per hyper-connection copy,
    /// followed by the one value every copy shares.
    public let engramKV: MTLBuffer
    /// The gate each copy's residual was mixed with, kept for a probe readback.
    public let engramGates: MTLBuffer
    /// Where each Engram table's rows stage. Empty when the checkpoint has none.
    public let engramSlots: [DeepSeekV41EngramSlot]
    /// Rows the gather allocates for one layer's attention: window + compressed.
    public let gatherRows: Int
    /// Positions the vectors are replicated for.
    ///
    /// A decode step uses slot 0 only. A prefilled chunk runs its dense stages for
    /// several positions per dispatch — the kernels index the token from
    /// `thread_position_in_grid.y` — so every per-position vector lives in `slots`
    /// consecutive copies and a stage binds the copy it is working on. The
    /// per-position scratch (`intermediate`, the compressor block, the Engram
    /// staging) stays single: those are used one position at a time no matter how
    /// wide the dispatch is. `gathered` and `rowIndices` do NOT: both are per
    /// position, because the gather's row list is a host fill and a host fill is not
    /// ordered against a dispatch that has not executed yet.
    public let slots: Int

    /// Byte offset of one position's copy of a `width`-wide vector.
    public func slotOffset(_ slot: Int, width: Int) -> Int {
        min(max(slot, 0), max(0, slots - 1)) * width * MemoryLayout<Float>.stride
    }

    /// First row of one position's gathered block, and of its row list.
    ///
    /// `gathered` holds one `gatherRows`-row block per position, so this is the row
    /// offset to gather into; `rowIndices` holds one `gatherRows`-long list per
    /// position, so it is also the element offset to write that list at.
    public func gatheredRowOffset(_ slot: Int) -> Int {
        min(max(slot, 0), max(0, slots - 1)) * gatherRows
    }

    public init?(
        device: MTLDevice,
        shapes: DeepSeekV41TensorShapes,
        config: DeepSeekV41Config,
        compressedOwners: Int,
        slots: Int = 1
    ) {
        let mult = max(1, config.hcMult)
        let slots = max(1, slots)
        let maxCompressed = max(1, config.compressRatios.max() ?? 1)
        // `Compressor` keeps one group per module instance, so a layer's slots are
        // its own. The block is the widest ratio wide; a narrow owner leaves the
        // remaining slots unused rather than pooling another layer's latents.
        let compressorGroups = max(1, compressedOwners)
        // Rows one position gathers: its window plus whatever the addressing rule
        // may select. `index_topk == 0` would mean "the whole compressed cache", so
        // the candidate-block budget bounds it and the executor reports a cache
        // provisioned wider than this.
        let gather = DeepSeekV41DevicePlan.gatherRowBudget(config: config)
        let float4: (Int) -> Int = { $0 * MemoryLayout<Float>.stride }
        // The gathered block holds the cache's own records, so its values are two
        // bytes and not four: the block is a copy of cache rows, and the executor
        // refuses any cache that is not FP16 (`DeepSeekV41DevicePlan.problems`), so
        // widening each record on the way out doubled both the block and every
        // head's read of it (`dsv41_attention_csa2_f16`).
        let gatheredValueBytes = MemoryLayout<UInt16>.stride
        // Engram staging is sized for exactly the rows one position reads, which is
        // well under the 64-row block the memory plan prices, so the allocation can
        // never outgrow the budget the load log reports.
        let engramSlots = DeepSeekV41EngramSlot.slots(config: config)
        let engramRawBytes = engramSlots.reduce(0) { $0 + $1.rawByteCount }
        let engramValueCount = engramSlots.reduce(0) { $0 + $1.valueCount }
        let engramKVCount = (mult + 1) * shapes.hidden

        func make(_ bytes: Int) -> MTLBuffer? {
            device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)
        }
        func slotBytes(_ width: Int) -> Int {
            float4(max(1, width)) * slots
        }
        guard let hidden = make(slotBytes(shapes.hidden)),
              let normed = make(slotBytes(shapes.hidden)),
              let query = make(slotBytes(shapes.queryWidth)),
              let latent = make(slotBytes(shapes.latentWidth)),
              let gathered = make(gather * shapes.latentWidth * slots * gatheredValueBytes),
              let attentionOut = make(slotBytes(shapes.attentionOutputWidth)),
              let output = make(slotBytes(shapes.hidden)),
              let routerLogits = make(slotBytes(config.numRoutedExperts)),
              let expertIndices = make(max(4 * config.numActivatedExperts, 16) * slots),
              let expertWeights = make(slotBytes(config.numActivatedExperts)),
              let intermediate = make(float4(shapes.expertIntermediate)),
              let sharedIntermediate = make(float4(shapes.sharedIntermediate)),
              let pooled = make(float4(shapes.latentWidth)),
              let compressorLatents = make(
                  float4(maxCompressed * compressorGroups * shapes.latentWidth)
              ),
              let compressorScores = make(
                  float4(maxCompressed * compressorGroups * shapes.latentWidth)
              ),
              let reduced = make(slotBytes(max(1, max(shapes.qLoraRank, shapes.outputReducedWidth)))),
              let moeOut = make(slotBytes(shapes.hidden)),
              let streams = make(slotBytes(mult * shapes.hidden)),
              let hcLogits = make(slotBytes((2 + mult) * mult)),
              let hcMix = make(slotBytes(mult)),
              let hcAttentionPre = make(slotBytes(mult)),
              let hcAttentionPost = make(slotBytes(mult)),
              let hcAttentionComb = make(slotBytes(mult * mult)),
              let hcFFNPre = make(slotBytes(mult)),
              let hcFFNPost = make(slotBytes(mult)),
              let hcFFNComb = make(slotBytes(mult * mult)),
              let hcResidual = make(slotBytes(mult * shapes.hidden)),
              let logits = make(float4(config.vocabSize)),
              let rowIndices = make(4 * gather * slots),
              let tokenIDs = make(4),
              let engramRaw = make(engramRawBytes),
              let engramValues = make(float4(engramValueCount)),
              let engramKV = make(slotBytes(engramKVCount)),
              let engramGates = make(slotBytes(mult)) else {
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
        self.compressorGroups = compressorGroups
        self.compressorGroupWidth = maxCompressed
        self.reduced = reduced
        self.moeOut = moeOut
        self.streams = streams
        self.hcLogits = hcLogits
        self.hcMix = hcMix
        self.hcAttentionPre = hcAttentionPre
        self.hcAttentionPost = hcAttentionPost
        self.hcAttentionComb = hcAttentionComb
        self.hcFFNPre = hcFFNPre
        self.hcFFNPost = hcFFNPost
        self.hcFFNComb = hcFFNComb
        self.hcResidual = hcResidual
        self.logits = logits
        self.rowIndices = rowIndices
        self.tokenIDs = tokenIDs
        self.engramRaw = engramRaw
        self.engramValues = engramValues
        self.engramKV = engramKV
        self.engramGates = engramGates
        self.engramSlots = engramSlots
        self.gatherRows = gather
        self.slots = slots
    }
}
