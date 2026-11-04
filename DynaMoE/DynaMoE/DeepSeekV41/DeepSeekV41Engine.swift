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
    /// Bytes this tensor occupies in its shard, which is what the dense pin reads
    /// and what a range check needs. Not derivable from `rows * columns`: an
    /// E4M3 tensor is one byte per value and a packed MXFP4 one is half, and the
    /// scale companion is a separate tensor.
    public let byteLength: Int

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

    /// The same tensor, addressed somewhere else. Used by the dense pin, which
    /// moves a tensor's bytes into a resident buffer and has to repoint it.
    public func addressing(shardIndex: UInt32, byteOffset: Int) -> DeepSeekV41DeviceTensor {
        DeepSeekV41DeviceTensor(
            shardIndex: shardIndex,
            byteOffset: byteOffset,
            rows: rows,
            columns: columns,
            dtype: dtype,
            byteLength: byteLength
        )
    }
}

// MARK: - Device weights

/// Resolves catalog roles to device tensors, and owns the Engram readers.
nonisolated public final class DeepSeekV41DeviceWeights: @unchecked Sendable {

    public let catalog: DeepSeekV41WeightCatalog
    /// Non-nil when the checkpoint cannot be run; the launch surfaces these.
    public let problems: [String]
    /// One reader per Engram table, keyed by layer.
    public let engramReaders: [Int: DeepSeekV41EngramReader]
    /// Why a table has no reader, keyed by layer, in the words of the thing that
    /// refused it. A missing shard, an offset past the file end and a wrong scale
    /// shape all read as "could not be opened" without this.
    public let engramReaderProblems: [Int: String]
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
                dtype: tensor.dtype,
                byteLength: Int(tensor.offsetEnd - tensor.offsetStart)
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
        // Why a table could not be opened, keyed by layer. The readiness report
        // prints this instead of a generic "could not be opened": a missing shard,
        // a wrong offset and a refused descriptor are different downloads.
        var readerProblems: [Int: String] = [:]
        // The primes are drawn once across every Engram layer, so the layouts are
        // built together; one table in isolation would address the wrong rows.
        let layouts = (try? DeepSeekV41EngramLayout.make(specs: config.engramTables)) ?? [:]
        if let snapshotDir {
            for spec in config.engramTables {
                let layer = spec.layer
                guard let valuesRef = catalog.tensor(.engramTable, layer: layer),
                      let values = summary.tensors.first(where: { $0.name == valuesRef.name }) else {
                    readerProblems[layer] = "layers.\(layer): the Engram table tensor is missing "
                        + "from the catalog"
                    continue
                }
                // `engram.embed.scale` is a SEPARATE tensor in the checkpoint, so the
                // reader has to be told where its block starts. Its absence is not
                // fatal only when the table stores each row's scales inline, which no
                // published Engram table does; a table whose rows carry block scales
                // without a companion tensor is refused rather than read as garbage.
                let scaleRef = catalog.tensor(.engramTable, layer: layer, scale: true)
                let scale = scaleRef.flatMap { ref in
                    summary.tensors.first(where: { $0.name == ref.name })
                }
                guard let scale else {
                    // The reference stores the block scales in their own tensor
                    // (`ParallelEngramEmbedding.scale`). Assuming they are inline
                    // would read the next row's values as this row's exponents, which
                    // decodes to plausible garbage, so a table without them is
                    // refused instead.
                    readerProblems[layer] = "layers.\(layer): the table has no "
                        + "engram.embed.scale companion, so its rows' block scales "
                        + "are not addressable"
                    continue
                }
                let shape = DeepSeekV41WeightCatalog.shape(from: scale.shapeDisplay)
                let expected = [spec.rowCount, spec.rowBytes - spec.headDimension]
                if shape != expected {
                    readerProblems[layer] = "layers.\(layer): \(scale.name) is \(shape), "
                        + "but the table's \(spec.rowCount) rows carry \(expected.last ?? 0) "
                        + "block scales each"
                    continue
                }
                guard let filename = DeepSeekV41EngramTableLocator.shardFilename(
                    forTensorNamed: valuesRef.name,
                    in: snapshotDir
                ) else {
                    readerProblems[layer] = "layers.\(layer): no shard in the snapshot publishes "
                        + "\(valuesRef.name) (looked in the index and in every header)"
                    continue
                }
                do {
                    readers[layer] = try DeepSeekV41EngramReader(
                        fileURL: snapshotDir.appendingPathComponent(filename),
                        spec: spec,
                        valuesOffset: Int(values.offsetStart),
                        // Streamed tensors are passed through segmentation untouched,
                        // so their offsets stay relative to the whole FILE.
                        scalesOffset: Int(scale.offsetStart)
                    )
                } catch {
                    readerProblems[layer] = "layers.\(layer): \(error)"
                }
            }
        }
        self.engramReaders = readers
        self.engramReaderProblems = readerProblems
        self.engramLayouts = layouts
    }

    /// A tensor by role.
    ///
    /// The namespace is a parameter for the same reason it is on the catalog: the
    /// DSpark stages under `mtp.N.*` are full blocks with the same stage names and
    /// their own layer indices, so `mtp.0.attn.wkv` and `layers.0.attn.wkv` are two
    /// different tensors under one `(role, layer)` pair. The default is the backbone
    /// because that is the graph every existing call site belongs to.
    public func tensor(
        _ role: DeepSeekV41TensorRole,
        scope: DeepSeekV41TensorScope = .backbone,
        layer: Int? = nil,
        expert: Int? = nil
    ) -> DeepSeekV41DeviceTensor? {
        guard let slot = catalog.tensor(role, scope: scope, layer: layer, expert: expert) else {
            return nil
        }
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

    /// The tensors the dense pin may move, in the catalog's own terms.
    ///
    /// The exclusions are the reason this is here rather than at the call site:
    /// whether a tensor is streamed, an expert, or part of a graph nothing runs
    /// is a property of its ROLE, and a name-prefix rule would get the Engram
    /// table's neighbours wrong the way it did for the summary (`engram.wkv` is a
    /// dense 157 MB matrix, not a table).
    public func pinnableTensors() -> [DeepSeekV41DensePinCandidate] {
        var slots: [String: DeepSeekV41TensorSlot] = [:]
        for slot in catalog.slots { slots[slot.ref.name] = slot }
        var candidates: [DeepSeekV41DensePinCandidate] = []
        candidates.reserveCapacity(tensors.count + scales.count)
        for (name, tensor) in tensors {
            guard let slot = slots[name], Self.isPinnable(slot) else { continue }
            candidates.append(Self.candidate(name: name, tensor: tensor))
        }
        for (name, tensor) in scales {
            guard let slot = slots[name], Self.isPinnable(slot) else { continue }
            candidates.append(Self.candidate(name: name, tensor: tensor))
        }
        return candidates
    }

    /// Repoints every pinned tensor at the buffer that now holds its bytes.
    ///
    /// Only the address changes: the shape, dtype and byte length stay what the
    /// summary said, so every kernel binds the same offsets it did before.
    @discardableResult
    public func pin(_ plan: DeepSeekV41DensePinPlan) -> Int {
        var moved = 0
        for range in plan.ranges {
            let index = plan.pinnedShardIndex(chunk: range.chunk)
            if let tensor = tensors[range.tensorName] {
                tensors[range.tensorName] = tensor.addressing(
                    shardIndex: index, byteOffset: range.chunkOffset
                )
                moved += 1
            } else if let tensor = scales[range.tensorName] {
                scales[range.tensorName] = tensor.addressing(
                    shardIndex: index, byteOffset: range.chunkOffset
                )
                moved += 1
            }
        }
        return moved
    }

    private static func candidate(
        name: String,
        tensor: DeepSeekV41DeviceTensor
    ) -> DeepSeekV41DensePinCandidate {
        DeepSeekV41DensePinCandidate(
            name: name,
            shardIndex: tensor.shardIndex,
            byteOffset: tensor.byteOffset,
            byteLength: tensor.byteLength
        )
    }

    private static func isPinnable(_ slot: DeepSeekV41TensorSlot) -> Bool {
        // The Engram table is streamed row by row and never becomes a buffer.
        guard !slot.role.isStreamed else { return false }
        // Routed experts live in the arena, which the streamer fills.
        guard slot.expert == nil else { return false }
        // Nothing in the text graph reads the vision tower or the draft block.
        guard !slot.role.isVision, !slot.role.isDraft else { return false }
        // The embedding is read one row at a time, so pinning 1.3 GB to save two
        // page faults per token is memory spent for nothing.
        guard slot.role != .tokenEmbedding else { return false }
        return true
    }
}

// MARK: - Pipelines

/// Every pipeline the forward binds. Built once per load; a missing function is a
/// hard load failure rather than a silently skipped stage.
nonisolated public struct DeepSeekV41Pipelines {
    public let fp8Block32Gemv: MTLComputePipelineState
    /// The same 32x32 ue8m0 dot product with one output row per threadgroup, so a
    /// warp's weight loads are adjacent rather than a row apart.
    public let fp8Block32GemvRows: MTLComputePipelineState
    /// The same projection over a BLOCK of positions, with the row's weight requested
    /// once for all of them (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #73").
    public let fp8Block32GemvRowsSlots: MTLComputePipelineState
    public let mxfp4Gemv: MTLComputePipelineState
    public let mxfp4SwiGLU: MTLComputePipelineState
    public let mxfp4DownAccumulate: MTLComputePipelineState
    /// The same two with one output row per threadgroup (coalesced weight loads).
    public let mxfp4SwiGLURows: MTLComputePipelineState
    public let mxfp4DownAccumulateRows: MTLComputePipelineState
    public let engramDequantRows: MTLComputePipelineState
    /// The Engram gate and residual add, one threadgroup per hyper-connection copy.
    public let engramGate: MTLComputePipelineState
    public let hcSinkhorn: MTLComputePipelineState
    /// The mHC rule: project the flattened stack, split the logits, collapse with a
    /// mix, expand back through `comb`.
    public let hcProject: MTLComputePipelineState
    /// The four-accumulator form of the projection, which a decode step ships: the
    /// walk's 24-threadgroup grid has one load in flight per lane without it
    /// (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #62").
    public let hcProjectX4: MTLComputePipelineState
    public let hcSplit: MTLComputePipelineState
    /// The same split with the `mult == 4` Sinkhorn held in registers, which is the
    /// form a decode step ships (`#62`).
    public let hcSplitRegs: MTLComputePipelineState
    public let hcPre: MTLComputePipelineState
    public let hcPost: MTLComputePipelineState
    public let hcIdentityMix: MTLComputePipelineState
    /// The router's top-k with one SIMD group per position, which a decode step
    /// ships: the one-thread scan is a `sqrtsoftplus` per expert (`#63`).
    public let routerTopKLanes: MTLComputePipelineState
    public let rope: MTLComputePipelineState
    public let storeLatent: MTLComputePipelineState
    public let gatherLatents: MTLComputePipelineState
    /// CSA2 attention over the gathered block. The block is FP16, so this is the
    /// half-key kernel; the FP32 form is kept in the shader library as the baseline
    /// the measurement A/Bs against (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #17").
    public let attention: MTLComputePipelineState
    /// The same graph over FP32 keys. The DSpark draft block reads its own ring,
    /// which the seed writes as floats, so it takes this form: a half-key ring would
    /// put an fp16 rounding floor under the parity bar that hides which row or which
    /// rope slice a mistake is in.
    public let attentionF32: MTLComputePipelineState
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
    /// The same pair with one output row per threadgroup (coalesced weight loads).
    public let fp8SwiGLURows: MTLComputePipelineState
    public let fp8DownAccumulateRows: MTLComputePipelineState
    /// The same pair over a BLOCK of positions, with the row's weight requested once:
    /// the shared expert is the one weight every position of a block reads.
    public let fp8SwiGLURowsSlots: MTLComputePipelineState
    public let fp8DownAccumulateRowsSlots: MTLComputePipelineState
    /// BF16 fallback for a checkpoint whose FFN was not quantized.
    public let bf16Gemv: MTLComputePipelineState
    public let bf16GemvRows: MTLComputePipelineState
    public let bf16GemvBlock: MTLComputePipelineState
    /// The block form over two output rows per threadgroup (`attn.wo_a`, `attn.wo_b`).
    public let bf16GemvBlockPair: MTLComputePipelineState
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
        self.fp8Block32GemvRows = try make("dsv41_fp8_block32_gemv_rows")
        self.fp8Block32GemvRowsSlots = try make("dsv41_fp8_block32_gemv_rows_slots")
        self.mxfp4Gemv = try make("dsv41_mxfp4_gemv")
        self.mxfp4SwiGLU = try make("dsv41_mxfp4_swiglu_gate_up")
        self.mxfp4DownAccumulate = try make("dsv41_mxfp4_down_accumulate")
        self.mxfp4SwiGLURows = try make("dsv41_mxfp4_swiglu_gate_up_rows")
        self.mxfp4DownAccumulateRows = try make("dsv41_mxfp4_down_accumulate_rows")
        self.engramDequantRows = try make("dsv41_engram_dequant_rows")
        self.engramGate = try make("dsv41_engram_gate_f32")
        self.hcSinkhorn = try make("dsv41_hc_sinkhorn")
        self.hcProject = try make("dsv41_hc_project_f32")
        self.hcProjectX4 = try make("dsv41_hc_project_f32_x4")
        self.hcSplit = try make("dsv41_hc_split_f32")
        self.hcSplitRegs = try make("dsv41_hc_split_f32_regs")
        self.hcPre = try make("dsv41_hc_pre_f32")
        self.hcPost = try make("dsv41_hc_post_f32")
        self.hcIdentityMix = try make("dsv41_hc_identity_mix_f32")
        self.routerTopKLanes = try make("dsv41_router_topk_lanes")
        self.rope = try make("dsv41_rope_latent")
        self.storeLatent = try make("dsv41_store_latent_f16")
        self.gatherLatents = try make("dsv41_gather_latents_f16")
        self.attention = try make("dsv41_attention_csa2_f16")
        self.attentionF32 = try make("dsv41_attention_csa2_f32")
        self.poolLatents = try make("dsv41_pool_latents_f32")
        self.indexerScores = try make("dsv41_indexer_scores_f32")
        self.router = try make("dsv41_router_topk")
        self.clear = try make("clear_vector_f32")
        self.copy = try make("dsv41_copy_f32")
        self.vectorAdd = try make("vector_add_f32")
        self.rmsNormGain = try make("dsv41_rmsnorm_f32_gain")
        self.fp8SwiGLU = try make("dsv41_fp8_block32_swiglu_gate_up")
        self.fp8DownAccumulate = try make("dsv41_fp8_block32_down_accumulate")
        self.fp8SwiGLURows = try make("dsv41_fp8_block32_swiglu_gate_up_rows")
        self.fp8DownAccumulateRows = try make("dsv41_fp8_block32_down_accumulate_rows")
        self.fp8SwiGLURowsSlots = try make("dsv41_fp8_block32_swiglu_gate_up_rows_slots")
        self.fp8DownAccumulateRowsSlots = try make("dsv41_fp8_block32_down_accumulate_rows_slots")
        self.bf16Gemv = try make("bf16_gemv")
        self.bf16GemvRows = try make("bf16_gemv_rows")
        self.bf16GemvBlock = try make("bf16_gemv_rows8")
        self.bf16GemvBlockPair = try make("bf16_gemv_rows8x2")
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
    /// A supplied residual row is not the model's hidden width.
    case embeddingWidthMismatch(expected: Int, actual: Int)
    /// A resumed turn found resident state that is not this prompt's.
    case prefixStateMismatch(position: Int)
    /// The checkpoint cannot turn pixels into residual rows.
    case visionUnavailable(String)
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
        case let .embeddingWidthMismatch(expected, actual):
            return "DeepSeek V4.1 was given a \(actual)-wide residual row where the model is "
                + "\(expected) wide"
        case .prefixStateMismatch(let position):
            return "DeepSeek V4.1 cannot continue the cached sequence: its token history diverges at position \(position)"
        case .visionUnavailable(let detail):
            return "DeepSeek V4.1 cannot see images here: \(detail)"
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
    /// The dense backbone pread into resident buffers, or nil when those tensors
    /// are still read through the shard mappings. See `DeepSeekV41DensePin`.
    public let densePin: DeepSeekV41DensePinPlan?
    /// Wall time the pin's reads took, and why there is no pin when there is none.
    public let densePinSeconds: Double
    public let densePinProblem: String?
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
    /// The zero-copy shard mappings the executor binds tensors through, plus one
    /// entry per dense-pin chunk when the backbone was pinned. The pinned entries
    /// live at `DeepSeekV41DensePinPlan.pinnedShardBase` and up.
    public let shardBuffers: [UInt32: MTLBuffer]

    /// The device executor, built on first use: resolving every norm gain and
    /// RoPE table is load work a turn needs and a pure diagnostic does not.
    private var cachedForward: DeepSeekV41MetalForward?

    /// The vision tower, built on first use. Kept for its weight resolver and its
    /// device pass: the host encode is per call and cache-free, because one encode
    /// reads each tensor exactly once, while the device pass uploads the tower's
    /// weights once and keeps the token matrix on the GPU across all blocks.
    private var cachedVision: DeepSeekV41VisionForward?
    /// Why the device tower was not built, when one was asked for. Nil means either
    /// the device pass is in use or no image has needed the tower yet.
    public private(set) var visionDeviceProblem: String?
    /// The directory the shards were indexed from, which the tower's `pread` reads
    /// need. Nil for a runtime built without one (a test fixture).
    public let snapshotDir: URL?

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

    /// `defaults write DRP.DynaMoE dynamoe_dsv41_pin_dense_off -bool YES`
    ///
    /// Debug-only opt-out: read the dense backbone through the shard mappings
    /// instead of pinning it. The pin is the difference between DRAM traffic and
    /// a GPU page fault per 16 KB, so this exists to A/B it, not to be shipped
    /// as a setting.
    public static let densePinOffDefaultKey = "dynamoe_dsv41_pin_dense_off"

    public static var densePinDisabled: Bool {
        UserDefaults.standard.bool(forKey: densePinOffDefaultKey)
    }

    /// `defaults write DRP.DynaMoE dynamoe_dsv41_arena_fraction -float 0.72`
    ///
    /// Debug-only seam for A/B-ing how much of RAM the expert arena may hold. The arena
    /// is a cache, so a bigger one means fewer experts a token pays the SSD for; the
    /// counterweight is memory pressure, which is exactly what a measurement has to
    /// settle rather than assume. Not a shipped setting.
    public static let arenaFractionDefaultKey = "dynamoe_dsv41_arena_fraction"

    public static var arenaFractionOverride: Double? {
        let value = UserDefaults.standard.double(forKey: arenaFractionDefaultKey)
        return value > 0 ? value : nil
    }

    /// `defaults write DRP.DynaMoE dynamoe_dsv41_wire_arena_off -bool YES`
    ///
    /// Debug-only opt-out from the arena's residency policy. Wiring the slots is the
    /// `#56` lever: a wired page cannot become compressor input, which took a
    /// 128-position prefill of this checkpoint from 330-375 to 171-174 ms/position.
    /// It is on where the memory plan can afford it, so this exists to A/B the
    /// policy and to get an unwired machine back, not to be a setting.
    public static let wireArenaOffDefaultKey = "dynamoe_dsv41_wire_arena_off"

    public static var wireArenaDisabled: Bool {
        UserDefaults.standard.bool(forKey: wireArenaOffDefaultKey)
    }

    /// `defaults write DRP.DynaMoE dynamoe_dsv41_arena_lru -bool YES`
    ///
    /// Debug-only opt-out from the arena's frequency replacement rule. `#58` measured
    /// the rule on a warm 128-position prefill: 2,190-2,394 misses and 38.3-41.9 GB read
    /// against LRU's 3,117-3,239 and 54.6-56.7 GB at the same 3,760 slots, i.e. 13-14%
    /// of the pass. It is on because the cold pass is unaffected -- routing does not
    /// consult the arena, so a first touch is a miss under any rule -- so this exists to
    /// get the old order back for an A/B, not to be a setting.
    public static let arenaLRUDefaultKey = "dynamoe_dsv41_arena_lru"

    public static var arenaLRUForced: Bool {
        UserDefaults.standard.bool(forKey: arenaLRUDefaultKey)
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
        allowEngramSkip: Bool = false,
        pinDenseWeights: Bool? = nil,
        wireArena: Bool? = nil,
        // How much of RAM the expert arena may hold, resolved ahead of the
        // `dynamoe_dsv41_arena_fraction` seam. `#58` needs two capacities in one
        // process, one runtime at a time, without writing the developer's defaults.
        arenaFraction: Double? = nil
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
                  // The widest owner's row count, not the geometry's base one: the
                  // base is sized from the WIDEST ratio, and a ratio-1 owner pools
                  // one row per token, so it needs the whole context.
                  indexerRows: plan
                      .filter(\.ownsCompressedIndex)
                      .map { layout.geometry.compressedRows(forRatio: max(1, $0.compressRatio)) }
                      .max() ?? 0,
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
            DeepSeekV41MemoryPlan.affordableExpertSlots(
                plan: memoryPlan,
                physicalGB: physicalGB,
                fraction: arenaFraction ?? DeepSeekV41Runtime.arenaFractionOverride ?? 0.6
            ),
            max(1, config.numRoutedExperts * max(1, config.numHiddenLayers))
        )
        let arena = DeepSeekV41ExpertArena(slotCount: slotCount)
        // `#58`: a warm pass misses mostly on pairs the arena already loaded and evicted,
        // so the replacement rule is a lever at a fixed capacity. The frequency rule
        // ships; this is the one debug seam back to LRU.
        if DeepSeekV41Runtime.arenaLRUForced { arena.victimPolicy = .leastRecentlyUsed }

        // The dense backbone is read in full on every token. Through the shard
        // mappings that is a GPU page fault per 16 KB, and the expert stream
        // evicts those pages between tokens, so the same few gigabytes are
        // re-faulted every step. Pinning it once makes it DRAM traffic. The
        // budget is a third of RAM: the memory plan already prices these bytes as
        // resident, so this is the same claim on memory expressed as a cap, and a
        // machine too small to hold the whole backbone pins a prefix of it rather
        // than refusing to load.
        var buffers = shardBuffers
        var densePin: DeepSeekV41DensePinPlan?
        var densePinSeconds: Double = 0
        var densePinProblem: String?
        if pinDenseWeights ?? !DeepSeekV41Runtime.densePinDisabled {
            if let snapshotDir {
                let pinPlan = DeepSeekV41DensePinPlan.make(
                    summary: summary,
                    candidates: weights.pinnableTensors(),
                    budgetBytes: Int(physicalGB * 0.3 * 1_073_741_824),
                    chunkLimit: min(1 << 30, Int(device.maxBufferLength)),
                    bufferLimit: Int(device.maxBufferLength)
                )
                do {
                    let loaded = try DeepSeekV41DensePinLoader.load(
                        plan: pinPlan, snapshotDir: snapshotDir, device: device
                    )
                    buffers.merge(loaded.buffers) { current, _ in current }
                    weights.pin(pinPlan)
                    densePin = pinPlan
                    densePinSeconds = loaded.seconds
                } catch {
                    densePinProblem = "\(error)"
                }
            } else {
                densePinProblem = "the snapshot directory is unknown, so the backbone cannot be read"
            }
        } else {
            densePinProblem = "disabled by \(DeepSeekV41Runtime.densePinOffDefaultKey)"
        }

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

        // The arena's slots are the destination a fill writes, and `#56` measured the
        // same bytes into them at 4.6-6.7 GB/s unwired against 13.0 wired, i.e. the
        // page compressor reconstructing every page a pass rewrote. The policy is
        // all-or-nothing and the budget is the arena's own size, granted only when the
        // memory plan says the rest of the machine still fits beside it.
        if let streamer, wireArena ?? !DeepSeekV41Runtime.wireArenaDisabled {
            let arenaBytes = streamer.capacity * streamer.geometry.slotBytes
            if memoryPlan.wireBudgetBytes(
                arenaBytes: arenaBytes, physicalGB: physicalGB
            ) > 0 {
                streamer.enableSlotWiring(budgetBytes: arenaBytes)
            }
        }

        self.config = config
        self.plan = plan
        self.catalog = weights.catalog
        self.weights = weights
        self.snapshotDir = snapshotDir
        self.cache = cache
        self.memoryPlan = memoryPlan
        self.arena = arena
        self.expertStreamer = streamer
        self.expertStreamerProblem = streamerProblem
        self.densePin = densePin
        self.densePinSeconds = densePinSeconds
        self.densePinProblem = densePinProblem
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
        self.shardBuffers = buffers
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

    /// The vision tower over the checkpoint's own weights, built on first use.
    ///
    /// Nil when `config.json` publishes no `vision_config`, the snapshot directory is
    /// unknown, or the tower's tensors could not be resolved from the shards. The
    /// tower carries a device pass when Metal can build one: its weights are read
    /// once and the whole encode stays on the GPU, which is what makes #41's and
    /// #42's measured stages pay. `dynamoe_dsv41_vision_device_off` forces the host
    /// path, and a device tower that cannot be built leaves `visionDeviceProblem`
    /// set rather than failing the image.
    public func visionForward() -> DeepSeekV41VisionForward? {
        if let cachedVision { return cachedVision }
        guard let spec = config.vision, let snapshotDir else { return nil }
        guard let source = DeepSeekV41VisionWeights(
            snapshotDir: snapshotDir, catalog: catalog
        ), let textHidden = source.textHiddenSize else { return nil }
        var deviceTower: DeepSeekV41VisionEncodeDevice?
        if UserDefaults.standard.bool(forKey: "dynamoe_dsv41_vision_device_off") {
            visionDeviceProblem = "disabled by dynamoe_dsv41_vision_device_off"
        } else {
            do {
                deviceTower = try DeepSeekV41VisionEncodeDevice(
                    spec: spec, weights: source, textHiddenSize: textHidden
                )
            } catch {
                visionDeviceProblem = "\(error)"
            }
        }
        let built = DeepSeekV41VisionForward(
            spec: spec, weights: source, textHiddenSize: textHidden, device: deviceTower
        )
        cachedVision = built
        return built
    }

    /// Pixels to the residual rows an image span puts in the prompt.
    ///
    /// The span's length is the prompt cost (`rows * (columns + 1) + 2`), and
    /// `DeepSeekV41ImagePrompt.expand` is what binds those rows to positions.
    public func imageSpan(
        rgb: [UInt8],
        width: Int,
        height: Int
    ) throws -> DeepSeekV41ImagePrompt.ImageSpan {
        guard let tower = visionForward() else {
            throw DeepSeekV41EngineError.visionUnavailable(
                config.vision == nil
                    ? "config.json publishes no vision_config"
                    : "the tower's weights could not be read from the snapshot"
            )
        }
        let fitted = DeepSeekV41VisionLayout.fit(width: width, height: height, spec: tower.spec)
        let rows = try tower.imageSpan(rgb: rgb, width: width, height: height)
        return DeepSeekV41ImagePrompt.ImageSpan(
            rows: rows, columns: fitted.columns, gridRows: fitted.rows
        )
    }

    /// What an image of this size would cost, without running the tower.
    ///
    /// The resize plan is the checkpoint's own (`plan_image_grid` then `safe_resize`),
    /// so this is the number the prompt and the KV pricing must agree on. It is split
    /// out because pricing an image must not require decoding 970 MB of weights.
    public func imageSpanPlan(width: Int, height: Int) -> (
        width: Int, height: Int, columns: Int, rows: Int, tokens: Int
    )? {
        guard let spec = config.vision else { return nil }
        return DeepSeekV41VisionLayout.fit(width: width, height: height, spec: spec)
    }

    /// Releases the descriptors and slot buffers this runtime owns.
    ///
    /// Called when a new runtime replaces it (a different checkpoint, or a turn that
    /// needs more context). Without it the Engram tables and the 48 expert shards
    /// stay open for the life of the process.
    public func shutdown() {
        cachedForward = nil
        // The tower's resolver holds one descriptor per shard it reads from; dropping
        // it here is what closes them with everything else this runtime owns.
        cachedVision = nil
        visionDeviceProblem = nil
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
        if let densePin {
            let gb = String(format: "%.2f", Double(densePin.totalBytes) / 1_073_741_824)
            let rate = densePinSeconds > 0
                ? String(
                    format: " at %.0f MB/s",
                    Double(densePin.totalBytes) / 1_048_576 / densePinSeconds
                )
                : ""
            let left = densePin.skipped.isEmpty
                ? ""
                : ", \(densePin.skipped.count) left on the mapping ("
                    + densePin.skipped.prefix(3).joined(separator: "; ")
                    + (densePin.skipped.count > 3 ? "; ..." : "") + ")"
            lines.append(
                "dense: pinned \(gb) GB into \(densePin.chunkLengths.count) buffer(s), "
                    + "\(densePin.ranges.count) tensors\(rate)\(left)"
            )
        } else {
            lines.append("dense: read through the shard mappings (\(densePinProblem ?? "no pin"))")
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
    /// Positions this table's staging block is replicated for.
    ///
    /// A prefilled block reads every position's rows BEFORE the layer's command
    /// buffer exists, so one shared block left every position of the block with the
    /// LAST position's n-gram memory. One block per position is ~6 KB of raw rows
    /// and ~24 KB decoded at the published table, which is nothing against the
    /// workspace around it.
    public let positionSlots: Int

    public var rawByteCount: Int { columns * rowBytes }
    public var valueCount: Int { columns * headDimension }
    public var valueByteOffset: Int { valueOffset * MemoryLayout<Float>.stride }
    /// Width of the concatenated rows the `wkv` projection reads.
    public var flatWidth: Int { columns * headDimension }

    /// Bytes this table owns in `engramRaw`, every position included.
    public var rawByteStride: Int { rawByteCount * positionSlots }
    /// Elements this table owns in `engramValues`, every position included.
    public var valueStride: Int { valueCount * positionSlots }

    /// Where one position's raw rows start inside `engramRaw`.
    public func rawBytes(atPositionSlot slot: Int) -> Int {
        rawByteOffset + min(max(slot, 0), max(0, positionSlots - 1)) * rawByteCount
    }

    /// Where one position's decoded rows start inside `engramValues`, in BYTES.
    public func valueBytes(atPositionSlot slot: Int) -> Int {
        (valueOffset + min(max(slot, 0), max(0, positionSlots - 1)) * valueCount)
            * MemoryLayout<Float>.stride
    }

    /// One slot per Engram table, in `config.engramTables` order so the index is
    /// `layer.engramTable`. A table with no hash columns gets an empty slot rather
    /// than being dropped: the caller looks slots up by index, and shifting the
    /// array would hand a layer its neighbour's rows.
    public static func slots(
        config: DeepSeekV41Config,
        positions: Int = 1
    ) -> [DeepSeekV41EngramSlot] {
        let positionSlots = max(1, positions)
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
                valueOffset: valueCursor,
                positionSlots: positionSlots
            )
            rawCursor += slot.rawByteStride
            valueCursor += slot.valueStride
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
        valueOffset: Int,
        positionSlots: Int = 1
    ) {
        self.layer = layer
        self.table = table
        self.columns = columns
        self.headDimension = headDimension
        self.rowBytes = rowBytes
        self.rawByteOffset = rawByteOffset
        self.valueOffset = valueOffset
        self.positionSlots = max(1, positionSlots)
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
    /// The indexer's own query for the position being scored, `indexHeadDim` wide
    /// per index head, rotated. One copy per slot.
    public let indexQuery: MTLBuffer
    /// `weights_proj(x)` for the position, one weight per index head, unscaled: the
    /// kernel multiplies by these, so the `index_head_dim ** -0.5 *
    /// index_n_heads ** -0.5` factor is applied on the way in.
    public let indexWeights: MTLBuffer
    /// One index key while it is being built from a pooled latent.
    public let indexKey: MTLBuffer
    /// The indexer's score for every compressed row, one block per slot. Shared so
    /// the host can read the selection it implies.
    public let indexScores: MTLBuffer
    /// Rows `indexScores` holds per slot.
    public let indexerRows: Int
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
        indexerRows: Int = 0,
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
        let engramSlots = DeepSeekV41EngramSlot.slots(config: config, positions: slots)
        let engramRawBytes = engramSlots.reduce(0) { $0 + $1.rawByteStride }
        let engramValueCount = engramSlots.reduce(0) { $0 + $1.valueStride }
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
              // One staging row per position: a block's shared expert projects every
              // slot's normed input in one dispatch, so this block is slot-sized rather
              // than one position wide.
              let sharedIntermediate = make(slotBytes(shapes.sharedIntermediate)),
              let pooled = make(float4(shapes.latentWidth)),
              let compressorLatents = make(
                  float4(maxCompressed * compressorGroups * shapes.latentWidth)
              ),
              let compressorScores = make(
                  float4(maxCompressed * compressorGroups * shapes.latentWidth)
              ),
              let indexQuery = make(
                  slotBytes(max(1, config.indexNHeads * config.indexHeadDim))
              ),
              let indexWeights = make(slotBytes(max(1, config.indexNHeads))),
              let indexKey = make(float4(max(1, config.indexHeadDim))),
              let indexScores = make(
                  float4(max(1, indexerRows) * slots)
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
              // One vocabulary vector per position, because a block that captures
              // its logits runs the head over every slot at once and leaves them
              // side by side. A decode step and a plain prefill only read slot 0.
              let logits = make(float4(config.vocabSize * slots)),
              let rowIndices = make(4 * gather * slots),
              // One token id per position: a prefilled block writes its whole
              // prompt slice here and binds one id per embedding dispatch, so a
              // single entry overruns the buffer at position 4 of 8.
              let tokenIDs = make(slotBytes(1)),
              let engramRaw = make(engramRawBytes),
              // One decoded block per position: `engramValueCount` already counts
              // every position of the block.
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
        self.indexQuery = indexQuery
        self.indexWeights = indexWeights
        self.indexKey = indexKey
        self.indexScores = indexScores
        self.indexerRows = max(1, indexerRows)
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
