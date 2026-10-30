//
//  DeepSeekV41SyntheticCheckpoint.swift
//  DynaMoETests
//
//  A whole DeepSeek-V4.1 checkpoint at a size a test can hold in memory.
//
//  The device forward has never been run against anything: the local shards are
//  header-only stubs of a ~510 GB checkpoint, so every kernel is validated by a
//  kernel-level golden and nothing validates the GRAPH that binds them — which
//  mix a stage reads, what offset a buffer is bound at, whether the residual the
//  collapse snapshots is the one `hc_post` mixes back in. Those are exactly the
//  failures a kernel test cannot see, and a port that has them produces plausible
//  logits at full speed.
//
//  This builds the missing thing: the same tensor NAMES the checkpoint publishes,
//  at widths the config derives, as one aligned byte blob that the executor wraps
//  in a `storageModeShared` buffer the way it wraps a zero-copy shard mapping. The
//  host reference is fed the identical values, so `DeepSeekV41CPUForward` and
//  `DeepSeekV41MetalForward` can be asked the same question about the same model.
//
//  Two rules keep the comparison honest:
//
//  - Every BF16 value is truncated to a BF16-representable float before it is
//    written anywhere, so the host reads the numbers the GPU reads. Comparing a
//    full-precision float against its own truncation would put a floor under the
//    error that has nothing to do with the graph.
//  - The config declares no Engram tables, no compressed layers, no DSpark
//    stages, and no shared expert. Each of those is a separate wiring question
//    with its own hand-fed fixtures; a parity test that has to get all of them
//    right at once says nothing about which one is wrong.
//

import Foundation
import Metal
@testable import DynaMoE

// MARK: - Shape

/// The dimensions of the synthetic model.
///
/// Chosen so every rule has somewhere to go wrong rather than to be as small as
/// possible: two heads in two output groups (a block-diagonal reduction that is
/// not accidentally a dense one), a decoupled rope payload narrower than the head
/// (so the `d_nope` prefix must not rotate), a value head as wide as a cache
/// record (so the output conjugate runs), two hyper-connection copies (the count
/// at which a transposed mixing matrix is not hidden by symmetry), and four
/// routed experts taking two (so the mixture really is a mixture).
nonisolated struct DeepSeekV41SyntheticShape {
    var hidden = 64
    var layers = 2
    var heads = 2
    /// `head_dim`, which the config requires to be nope + rope.
    var headDim = 48
    var ropeHeadDim = 16
    var valueHeadDim = 48
    var kvLoraRank = 32
    var qLoraRank = 32
    var oLoraRank = 16
    var oGroups = 2
    var vocab = 128
    var moeIntermediate = 64
    var routedExperts = 4
    var activatedExperts = 2
    var window = 4
    var hcMult = 2
    var hcIterations = 3
    var contextLength = 48
    /// Largest magnitude a weight is drawn from. Small enough that a two-layer
    /// residual stream stays in a range where a 1e-4 tolerance means something.
    var weightMagnitude: Float = 0.35
    /// Magnitude for norm gains, which want to be near one.
    var gainMagnitude: Float = 0.15
    /// Forces a role's every value to one constant. A bisecting aid: with the
    /// hyper-connection scale and base pinned, the split's `pre`/`post`/`comb` are a
    /// pure function of the projection's logits, so a disagreement has to be in the
    /// projection rather than in how the three sections are read.
    var constantRoles: [DeepSeekV41TensorRole: Float] = [:]

    var nopeHeadDim: Int { headDim - ropeHeadDim }
    var queryWidth: Int { heads * headDim }
    var attentionOutputWidth: Int { heads * valueHeadDim }
    var latentWidth: Int { kvLoraRank + ropeHeadDim }
    var outputReducedWidth: Int { oGroups * oLoraRank }
}

// MARK: - Tensors

/// One published tensor: its checkpoint name, the slot it resolves to, and the
/// values both backends read.
nonisolated struct DeepSeekV41SyntheticTensor {
    let name: String
    let role: DeepSeekV41TensorRole
    let layer: Int?
    let expert: Int?
    /// `"BF16"` or `"F32"`. The hyper-connection projections are FP32 in the
    /// checkpoint, and the executor refuses a bf16 one, so the format follows the
    /// role rather than being uniform.
    let dtype: String
    let shape: [Int]
    let values: [Float]

    var elementCount: Int { shape.reduce(1, *) }
    var isMatrix: Bool { shape.count == 2 }

    var rows: Int { shape.count == 2 ? shape[0] : shape[0] }
    var columns: Int { shape.count == 2 ? shape[1] : 1 }

    /// The Rust core's `Debug` spelling, which is what a summary carries.
    var shapeDisplay: String {
        "[" + shape.map(String.init).joined(separator: ", ") + "]"
    }
}

// MARK: - Checkpoint

/// A complete synthetic checkpoint: the config, the tensors, one shard blob, the
/// `ModelSummary` that indexes it, and the host-side weight source built from the
/// same numbers.
nonisolated final class DeepSeekV41SyntheticCheckpoint {

    let shape: DeepSeekV41SyntheticShape
    let config: DeepSeekV41Config
    let plan: [DeepSeekV41LayerPlan]
    let tensors: [DeepSeekV41SyntheticTensor]

    init(shape: DeepSeekV41SyntheticShape = DeepSeekV41SyntheticShape()) {
        self.shape = shape
        self.config = Self.makeConfig(shape)
        self.plan = (try? config.makeLayerPlan()) ?? []
        self.tensors = Self.makeTensors(shape: shape, config: config, plan: plan)
    }

    // MARK: Config

    private static func makeConfig(_ shape: DeepSeekV41SyntheticShape) -> DeepSeekV41Config {
        let text: [String: Any] = [
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "dtype": "bfloat16",
            "hidden_size": shape.hidden,
            "num_hidden_layers": shape.layers,
            "num_attention_heads": shape.heads,
            "num_key_value_heads": 1,
            "vocab_size": shape.vocab,
            "head_dim": shape.headDim,
            "qk_rope_head_dim": shape.ropeHeadDim,
            "v_head_dim": shape.valueHeadDim,
            "q_lora_rank": shape.qLoraRank,
            "kv_lora_rank": shape.kvLoraRank,
            "o_groups": shape.oGroups,
            "o_lora_rank": shape.oLoraRank,
            "intermediate_size": 0,
            "moe_intermediate_size": shape.moeIntermediate,
            "n_shared_experts": 0,
            "n_routed_experts": shape.routedExperts,
            "n_activated_experts": shape.activatedExperts,
            "moe_router_enable_expert_bias": false,
            // Absent on purpose: the BF16 SwiGLU kernel has no clip, and the
            // executor refuses to run a checkpoint that asks for one through it.
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "window_size": shape.window,
            // Every layer window-only: a compressed layer brings the compressor,
            // the indexer and the owner/addressing rules, each with its own
            // fixtures.
            "compress_ratios": [Int](repeating: 0, count: shape.layers),
            "hc_mult": shape.hcMult,
            "hc_sinkhorn_iters": shape.hcIterations,
            "hc_eps": 1e-6,
            "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.5,
            "num_nextn_predict_layers": 0,
            "max_position_embeddings": shape.contextLength,
            "tie_word_embeddings": false,
            "engram_layer_ids": [Int](),
            "engram_num_embeddings": [Int](),
            "bos_token_id": 0,
            "eos_token_id": [1],
            "pad_token_id": 2
        ]
        let object: [String: Any] = ["text_config": text]
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let config = try? DeepSeekV41Config.decode(jsonData: data) else {
            fatalError("the synthetic config does not decode; fix the fixture")
        }
        return config
    }

    // MARK: Tensor table

    private static func makeTensors(
        shape: DeepSeekV41SyntheticShape,
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan]
    ) -> [DeepSeekV41SyntheticTensor] {
        var out: [DeepSeekV41SyntheticTensor] = []

        func add(
            _ name: String,
            _ role: DeepSeekV41TensorRole,
            layer: Int? = nil,
            expert: Int? = nil,
            dtype: String = "BF16",
            shape dims: [Int],
            magnitude: Float? = nil
        ) {
            out.append(
                DeepSeekV41SyntheticTensor(
                    name: name,
                    role: role,
                    layer: layer,
                    expert: expert,
                    dtype: dtype,
                    shape: dims,
                    values: shape.constantRoles[role].map { constant in
                        [Float](repeating: constant, count: dims.reduce(1, *))
                    } ?? Self.values(
                        name: name,
                        count: dims.reduce(1, *),
                        magnitude: magnitude ?? shape.weightMagnitude,
                        bf16: dtype == "BF16"
                    )
                )
            )
        }

        let hidden = shape.hidden
        let gain = shape.gainMagnitude

        // Model level.
        add("embed.weight", .tokenEmbedding, shape: [shape.vocab, hidden])
        add("norm.weight", .finalNorm, shape: [hidden], magnitude: gain)
        add("head.weight", .lmHead, shape: [shape.vocab, hidden])

        let mult = max(1, shape.hcMult)
        let mixCount = (2 + mult) * mult

        for layer in plan.map(\.index) {
            let prefix = "layers.\(layer)"
            add("\(prefix).attn_norm.weight", .attnNorm, layer: layer, shape: [hidden], magnitude: gain)
            add("\(prefix).ffn_norm.weight", .ffnNorm, layer: layer, shape: [hidden], magnitude: gain)

            // Query rail: down to `q_lora_rank`, normed there, back up to every
            // head.
            if shape.qLoraRank > 0 {
                add(
                    "\(prefix).attn.wq_a.weight", .attnWqA, layer: layer,
                    shape: [shape.qLoraRank, hidden]
                )
                add(
                    "\(prefix).attn.q_norm.weight", .attnWqANorm, layer: layer,
                    shape: [shape.qLoraRank], magnitude: gain
                )
                add(
                    "\(prefix).attn.wq_b.weight", .attnWqB, layer: layer,
                    shape: [shape.queryWidth, shape.qLoraRank]
                )
            } else {
                add(
                    "\(prefix).attn.wq_b.weight", .attnWqB, layer: layer,
                    shape: [shape.queryWidth, hidden]
                )
            }

            add(
                "\(prefix).attn.wkv.weight", .attnWkv, layer: layer,
                shape: [shape.latentWidth, hidden]
            )
            add(
                "\(prefix).attn.kv_norm.weight", .attnKvNorm, layer: layer,
                shape: [shape.latentWidth], magnitude: gain
            )

            // `wo_a` is published block-diagonal: the column count is PER GROUP.
            if shape.oLoraRank > 0 {
                add(
                    "\(prefix).attn.wo_a.weight", .attnWoA, layer: layer,
                    shape: [
                        shape.outputReducedWidth,
                        shape.attentionOutputWidth / shape.oGroups
                    ]
                )
                add(
                    "\(prefix).attn.wo_b.weight", .attnWoB, layer: layer,
                    shape: [hidden, shape.outputReducedWidth]
                )
            } else {
                add(
                    "\(prefix).attn.wo_b.weight", .attnWoB, layer: layer,
                    shape: [hidden, shape.attentionOutputWidth]
                )
            }

            // Router: `bias` selects but does not weight, and the kernel binds a
            // zero placeholder when the checkpoint does not publish one.
            add(
                "\(prefix).ffn.gate.weight", .routerWeight, layer: layer,
                shape: [shape.routedExperts, hidden]
            )

            for expert in 0..<shape.routedExperts {
                let base = "\(prefix).ffn.experts.\(expert)"
                add("\(base).w1.weight", .expertW1, layer: layer, expert: expert,
                    shape: [shape.moeIntermediate, hidden])
                add("\(base).w3.weight", .expertW3, layer: layer, expert: expert,
                    shape: [shape.moeIntermediate, hidden])
                add("\(base).w2.weight", .expertW2, layer: layer, expert: expert,
                    shape: [hidden, shape.moeIntermediate])
            }

            for (suffix, base) in [
                ("hc_attn_fn", DeepSeekV41TensorRole.hcAttnFn),
                ("hc_ffn_fn", DeepSeekV41TensorRole.hcFfnFn)
            ] {
                add("\(prefix).\(suffix)", base, layer: layer, dtype: "F32",
                    shape: [mixCount, mult * hidden])
            }
            for (suffix, base) in [
                ("hc_attn_base", DeepSeekV41TensorRole.hcAttnBase),
                ("hc_ffn_base", DeepSeekV41TensorRole.hcFfnBase)
            ] {
                add("\(prefix).\(suffix)", base, layer: layer, dtype: "F32",
                    shape: [mixCount], magnitude: gain)
            }
            // The three read/scale/base constants the split multiplies through.
            for (suffix, base) in [
                ("hc_attn_scale", DeepSeekV41TensorRole.hcAttnScale),
                ("hc_ffn_scale", DeepSeekV41TensorRole.hcFfnScale)
            ] {
                add("\(prefix).\(suffix)", base, layer: layer, dtype: "F32",
                    shape: [3], magnitude: 0.5)
            }
        }
        return out
    }

    // MARK: Values

    /// Deterministic draws, so a failure reproduces and two runs compare.
    private static func values(name: String, count: Int, magnitude: Float, bf16: Bool) -> [Float] {
        var state = seed(name)
        var out: [Float] = []
        out.reserveCapacity(count)
        for _ in 0..<count {
            var value = next(&state) * magnitude
            if bf16 { value = truncateToBF16(value) }
            out.append(value)
        }
        return out
    }

    private static func seed(_ name: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in name.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01B3
        }
        return hash == 0 ? 0x9E37_79B9_7F4A_7C15 : hash
    }

    private static func next(_ state: inout UInt64) -> Float {
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        let scrambled = state &* 0x2545_F491_4F6C_DD1D
        let unit = Double(scrambled >> 11) / Double(UInt64(1) << 53)
        return Float(unit * 2 - 1)
    }

    /// The float a BF16 store reads back. Truncation, not round-to-nearest: it is
    /// what `as_type<float>(bits << 16)` returns, and the host has to read the
    /// same number rather than a better one.
    static func truncateToBF16(_ value: Float) -> Float {
        Float(bitPattern: value.bitPattern & 0xFFFF_0000)
    }

    // MARK: Blob and summary

    /// One shard, every tensor in table order, each start aligned so a
    /// `setBuffer(offset:)` binding is legal for the widest load the kernels do.
    private(set) lazy var layout: [(offset: Int, tensor: DeepSeekV41SyntheticTensor)] = {
        var cursor = 0
        var out: [(offset: Int, tensor: DeepSeekV41SyntheticTensor)] = []
        for tensor in tensors {
            cursor = (cursor + 255) & ~255
            out.append((cursor, tensor))
            cursor += tensor.elementCount * Self.bytesPerValue(tensor.dtype)
        }
        return out
    }()

    private(set) lazy var blob: [UInt8] = {
        var bytes = [UInt8](repeating: 0, count: (totalByteCount + 255) & ~255)
        for (offset, tensor) in layout {
            let width = Self.bytesPerValue(tensor.dtype)
            for (index, value) in tensor.values.enumerated() {
                let at = offset + index * width
                if width == 2 {
                    let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
                    bytes[at] = UInt8(bits & 0xFF)
                    bytes[at + 1] = UInt8(bits >> 8)
                } else {
                    let bits = value.bitPattern
                    bytes[at] = UInt8(bits & 0xFF)
                    bytes[at + 1] = UInt8((bits >> 8) & 0xFF)
                    bytes[at + 2] = UInt8((bits >> 16) & 0xFF)
                    bytes[at + 3] = UInt8((bits >> 24) & 0xFF)
                }
            }
        }
        return bytes
    }()

    private var totalByteCount: Int {
        guard let last = layout.last else { return 0 }
        return last.offset + last.tensor.elementCount * Self.bytesPerValue(last.tensor.dtype)
    }

    static func bytesPerValue(_ dtype: String) -> Int {
        dtype.uppercased().contains("F32") ? 4 : 2
    }

    /// The summary the Rust indexer would produce for the blob above.
    func summary() -> ModelSummary {
        var metadata: [TensorMetadata] = []
        for (offset, tensor) in layout {
            let bytes = tensor.elementCount * Self.bytesPerValue(tensor.dtype)
            metadata.append(
                TensorMetadata(
                    name: tensor.name,
                    shapeDisplay: tensor.shapeDisplay,
                    dtype: tensor.dtype,
                    sizeMb: Double(bytes) / 1_048_576,
                    shardIndex: 0,
                    offsetStart: UInt64(offset),
                    offsetEnd: UInt64(offset + bytes),
                    category: tensor.role.rawValue,
                    layerIndex: tensor.layer.map(UInt32.init),
                    expertId: tensor.expert.map(UInt32.init)
                )
            )
        }
        return ModelSummary(
            sizeGb: Double(blob.count) / 1_073_741_824,
            tensorCount: UInt32(metadata.count),
            layerCount: UInt32(config.numHiddenLayers),
            maxExpertId: UInt32(max(0, shape.routedExperts - 1)),
            shards: [
                ShardMetadata(
                    index: 0,
                    filename: "synthetic.safetensors",
                    baseAddress: 0,
                    length: UInt64(blob.count)
                )
            ],
            tensors: metadata,
            layers: []
        )
    }

    /// The zero-copy shard mapping, with the same storage mode a real one has.
    func shardBuffer(device: MTLDevice) -> MTLBuffer? {
        blob.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
        }
    }

    // MARK: Host weights

    /// The same numbers, as the host reference reads them.
    ///
    /// Every tensor is stored under BOTH spellings. The reference reaches almost
    /// everything through a `scope:` lookup, but the model-level head is read
    /// unscoped, and a source holding only scoped keys answers that one with "no
    /// head" — which the reference turns into `hidden` instead of logits, i.e. a
    /// 64-wide vector where 128 logits belong.
    func weightSource() -> DeepSeekV41DenseWeightSource {
        let source = DeepSeekV41DenseWeightSource()
        for tensor in tensors {
            if tensor.isMatrix {
                let matrix = DeepSeekV41Matrix(
                    rows: tensor.shape[0],
                    columns: tensor.shape[1],
                    values: tensor.values
                )
                source.set(matrix, role: tensor.role, layer: tensor.layer, expert: tensor.expert)
                source.set(
                    matrix,
                    role: tensor.role,
                    scope: .backbone,
                    layer: tensor.layer,
                    expert: tensor.expert
                )
            } else {
                source.set(tensor.values, role: tensor.role, layer: tensor.layer)
                source.set(
                    tensor.values,
                    role: tensor.role,
                    scope: .backbone,
                    layer: tensor.layer
                )
            }
        }
        return source
    }

    /// Why the catalog refused this checkpoint, if it did.
    func catalogProblems() -> [String] {
        let catalog = DeepSeekV41WeightCatalog.from(
            summary: summary(),
            layerCount: config.numHiddenLayers
        )
        return catalog.validate(config: config, plan: plan)
            + catalog.unconsumed.map { "unconsumed: \($0)" }
    }
}
