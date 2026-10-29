//
//  DeepSeekV41Weights.swift
//  DynaMoE
//
//  Tensor-name resolution for DeepSeek-V4.1-Flash checkpoints.
//
//  Three things make this model's naming different from every family DynaMoE
//  already runs, and each one is why this file exists instead of reusing
//  `InferenceEngine.buildCachedLayers`:
//
//    1. The checkpoint has NO `model.` root. Names start at `layers.N.`,
//       `embed.`, `norm.`, `head.` and `mtp.N.`, so every existing
//       `contains("self_attn.q_proj")` style binding misses.
//    2. The attention block is latent (wq_a -> wq_b, wkv, wo_a -> wo_b) with a
//       per-head sink logit and two optional side networks (indexer,
//       compressor) that exist only on layers the plan marks as compressed.
//    3. Routed experts are `ffn.experts.{E}.w{1,2,3}` with a paired scale
//       tensor each, and the Engram tables (`engram.embed`, one per
//       `engram_layer_ids` entry) are ~100 GB single tensors that must never be
//       handed to Metal. `engram.wkv` is a dense projection OVER the retrieved
//       rows, not the table, and must not be confused with it.
//
//  The catalog therefore does two jobs:
//    * resolve every name to a (scope, role, layer, expert) slot with tolerant
//      suffix matching, because the exact scale suffix (`weight_scale` vs
//      `scale`) is not fixed across DeepSeek's checkpoints, and
//    * FAIL CLOSED: any tensor the catalog cannot place, and any slot the plan
//      requires but the checkpoint lacks, is reported. A silently-ignored
//      tensor is how a model runs at full speed and emits the wrong
//      distribution.
//

import Foundation

// MARK: - Roles

nonisolated public enum DeepSeekV41TensorRole: String, Sendable, Equatable, CaseIterable {
    // Model-level
    case tokenEmbedding
    case finalNorm
    case lmHead

    // Attention block
    case attnNorm
    case attnWqA
    case attnWqANorm
    case attnWqB
    case attnWkv
    case attnKvNorm
    case attnWoA
    case attnWoB
    case attnSink

    // Sparse-attention side networks
    case indexerWqB
    case indexerWk
    case indexerWeights
    case indexerKNorm
    case compressorWkv
    case compressorWgate
    case compressorNorm

    // Feed-forward
    case ffnNorm
    case routerWeight
    case routerBias
    case routerVisionBias
    case sharedExpertW1
    case sharedExpertW2
    case sharedExpertW3
    case expertW1
    case expertW2
    case expertW3

    // Hyper-connections (mHC)
    case hcAttnFn
    case hcAttnBase
    case hcAttnScale
    case hcFfnFn
    case hcFfnBase
    case hcFfnScale

    // Engram
    case engramTable
    case engramWkv
    case engramQuery
    case engramKey

    // DSpark draft block
    case draftMainProj
    case draftMainNorm
    case draftNorm
    case draftMarkovEmbed
    case draftMarkovHead
    case draftConfidence

    // Vision tower, its projector and the three image markers. The checkpoint
    // publishes both alongside the text tower and the text graph never reads them,
    // but they are classified tensor by tensor rather than collapsed into one
    // "tower" role: the tower forward has to address `wqkv` separately from `wo`,
    // and one slot per block would silently keep only the last one.
    case visionPatchEmbed
    case visionPatchEmbedBias
    case visionBlockNorm1
    case visionBlockWqkv
    case visionBlockWqkvBias
    case visionBlockWo
    case visionBlockWoBias
    case visionBlockNorm2
    case visionBlockW1
    case visionBlockW2
    case visionFinalNorm
    case visionAlignerW1
    case visionAlignerW1Bias
    case visionAlignerW2
    case visionAlignerW2Bias
    case visionImageStart
    case visionImageNewline
    case visionImageEnd

    /// Roles whose resolved tensor is streamed from disk and never mapped into
    /// a Metal buffer (the Engram table is one ~100 GB tensor).
    public var isStreamed: Bool { self == .engramTable }

    /// Roles indexed by a DSpark draft block rather than a backbone layer.
    public var isDraft: Bool {
        switch self {
        case .draftMainProj, .draftMainNorm, .draftNorm, .draftMarkovEmbed,
             .draftMarkovHead, .draftConfidence:
            return true
        default:
            return false
        }
    }

    /// Roles that exist only when the plan says the layer has a compressor.
    public var requiresCompressor: Bool {
        switch self {
        case .compressorWkv, .compressorWgate, .compressorNorm,
             .indexerWqB, .indexerWk, .indexerWeights, .indexerKNorm:
            return true
        default:
            return false
        }
    }

    /// Roles from the vision tower or its projector. Nothing in the text graph
    /// reads them.
    public var isVision: Bool {
        switch self {
        case .visionPatchEmbed, .visionPatchEmbedBias, .visionBlockNorm1,
             .visionBlockWqkv, .visionBlockWqkvBias, .visionBlockWo,
             .visionBlockWoBias, .visionBlockNorm2, .visionBlockW1, .visionBlockW2,
             .visionFinalNorm, .visionAlignerW1, .visionAlignerW1Bias,
             .visionAlignerW2, .visionAlignerW2Bias, .visionImageStart,
             .visionImageNewline, .visionImageEnd:
            return true
        default:
            return false
        }
    }
}

// MARK: - Slots

/// One resolved tensor.
nonisolated public struct DeepSeekV41TensorRef: Sendable, Equatable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let shardIndex: Int?
    public let offset: Int?

    public init(name: String, dtype: String, shape: [Int], shardIndex: Int? = nil, offset: Int? = nil) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.shardIndex = shardIndex
        self.offset = offset
    }

    /// Whether the safetensors dtype is one the DeepSeek V4.1 kernels understand.
    public var isSupportedDtype: Bool {
        let upper = dtype.uppercased()
        return upper.contains("F8_E4M3")
            || upper.contains("I8")
            || upper.contains("BF16")
            || upper.contains("F32")
            || upper.contains("F16")
            || upper.contains("U8")
            || upper.contains("F8_E8M0")
    }
}

/// Which namespace a tensor name lives in. The DSpark stages under `mtp.N.*` are
/// full backbone blocks (attention, MoE, hyper-connections) with their own layer
/// indices, so without a scope `mtp.0.attn.wkv` and `layers.0.attn.wkv` would
/// resolve to the same slot and one of them would be silently dropped.
nonisolated public enum DeepSeekV41TensorScope: String, Sendable, Equatable {
    case backbone
    case draft
    case vision
}

/// Where a tensor lives in the model: a role, optionally per layer and per
/// expert, plus whether it is the paired quantization-scale companion.
nonisolated public struct DeepSeekV41TensorSlot: Sendable, Equatable {
    public let role: DeepSeekV41TensorRole
    /// Namespace the name came from; `layer` is a backbone layer index, a draft
    /// block index, or a vision block index depending on it.
    public let scope: DeepSeekV41TensorScope
    /// Backbone layer, or the draft block index for `draft*` roles, or nil for
    /// model-level tensors.
    public let layer: Int?
    public let expert: Int?
    public let isScale: Bool
    public let ref: DeepSeekV41TensorRef

    public init(
        role: DeepSeekV41TensorRole,
        scope: DeepSeekV41TensorScope = .backbone,
        layer: Int?,
        expert: Int?,
        isScale: Bool,
        ref: DeepSeekV41TensorRef
    ) {
        self.role = role
        self.scope = scope
        self.layer = layer
        self.expert = expert
        self.isScale = isScale
        self.ref = ref
    }

    public var key: String {
        "\(scope.rawValue)|\(role.rawValue)|\(layer.map(String.init) ?? "-")|\(expert.map(String.init) ?? "-")|\(isScale ? "s" : "w")"
    }
}

// MARK: - Catalog

nonisolated public struct DeepSeekV41WeightCatalog: Sendable {

    public let slots: [DeepSeekV41TensorSlot]
    /// Names the catalog could not place. The load refuses when this is not empty.
    public let unconsumed: [String]
    /// Two different tensors that resolve to the same role. The later name used to
    /// win silently, which is how a checkpoint with both `attn.kv_norm` and
    /// `attn.wkv_norm` would load one and ignore the other.
    public let duplicateRoles: [String]
    /// Backbone tensors naming a layer the config does not have. `layerCount`
    /// used to be accepted and never read, so a truncated manifest placing
    /// tensors past the last layer validated clean.
    public let outOfRangeLayers: [String]

    private let index: [String: DeepSeekV41TensorSlot]

    public init(named tensors: [DeepSeekV41TensorRef], layerCount: Int) {
        var slots: [DeepSeekV41TensorSlot] = []
        var outOfRange: [String] = []
        var unconsumed: [String] = []
        var duplicates: [String] = []
        var index: [String: DeepSeekV41TensorSlot] = [:]
        for tensor in tensors {
            guard let slot = Self.classify(name: tensor.name, ref: tensor) else {
                unconsumed.append(tensor.name)
                continue
            }
            if let layer = slot.layer, slot.scope == .backbone,
               layer < 0 || layer >= max(0, layerCount) {
                outOfRange.append("\(tensor.name) names layer \(layer) of \(layerCount)")
            }
            if let existing = index[slot.key] {
                duplicates.append(
                    "\(existing.ref.name) and \(tensor.name) both resolve to \(slot.scope.rawValue) \(slot.role.rawValue)\(slot.layer.map { " layer \($0)" } ?? "")"
                )
                slots.append(slot)
                continue
            }
            slots.append(slot)
            index[slot.key] = slot
        }
        self.slots = slots
        self.unconsumed = unconsumed
        self.duplicateRoles = duplicates
        self.outOfRangeLayers = outOfRange
        self.index = index
    }

    /// Whether a tensor name is one the loader must never map into a Metal
    /// buffer. Swift keys this off its own classifier so the decision does not
    /// depend on the display string the Rust core happens to emit.
    public static func isStreamedName(_ name: String) -> Bool {
        let ref = DeepSeekV41TensorRef(name: name, dtype: "", shape: [])
        return classify(name: name, ref: ref)?.role.isStreamed == true
    }

    public func tensor(
        _ role: DeepSeekV41TensorRole,
        scope: DeepSeekV41TensorScope = .backbone,
        layer: Int? = nil,
        expert: Int? = nil,
        scale: Bool = false
    ) -> DeepSeekV41TensorRef? {
        let key = Self.key(role, scope: scope, layer: layer, expert: expert, scale: scale)
        return index[key]?.ref
    }

    /// The role-index key. Every lookup and the slot's own key come from here, so
    /// the two cannot drift apart.
    public static func key(
        _ role: DeepSeekV41TensorRole,
        scope: DeepSeekV41TensorScope = .backbone,
        layer: Int?,
        expert: Int? = nil,
        scale: Bool = false
    ) -> String {
        "\(scope.rawValue)|\(role.rawValue)|\(layer.map(String.init) ?? "-")|\(expert.map(String.init) ?? "-")|\(scale ? "s" : "w")"
    }

    /// Looks a slot up by its `role|layer|expert|w|s` key. Used by the device-side
    /// resolver, which has already turned a role into a slot and needs the tensor
    /// behind it.
    public func slot(matchingKey key: String) -> DeepSeekV41TensorSlot? {
        index[key]
    }

    /// Per-expert tensors that were published for a layer, ascending.
    public func expertIndices(layer: Int, scope: DeepSeekV41TensorScope = .backbone) -> [Int] {
        Set(
            slots.filter { $0.role == .expertW1 && $0.layer == layer && $0.scope == scope }
                .compactMap(\.expert)
        ).sorted()
    }

    /// Streamed slots: the Engram table and its scale companion. Both live in
    /// the same ~100 GB blob, which is never mapped, so both names have to be
    /// excluded before Metal segmentation (`maxBufferLength` is smaller than one
    /// table).
    public var streamedSlots: [DeepSeekV41TensorSlot] {
        slots.filter { $0.role.isStreamed }
    }

    /// Unique tensor names that must not be handed to Metal.
    public var streamedTensorNames: [String] {
        var seen = Set<String>()
        return streamedSlots.map(\.ref.name).filter { seen.insert($0).inserted }
    }

    // MARK: Classification

    /// Resolves one tensor name. Tolerant of the suffix variants DeepSeek's
    /// converters emit: `x.weight` / bare `x`, and the scale companion as
    /// `x.weight_scale` / `x.scale` / `x.scales`.
    static func classify(name: String, ref: DeepSeekV41TensorRef) -> DeepSeekV41TensorSlot? {
        // Model-level tensors.
        switch name {
        case "embed.weight", "embed_tokens.weight":
            return slot(.tokenEmbedding, layer: nil, expert: nil, scale: false, ref)
        case "norm.weight":
            return slot(.finalNorm, layer: nil, expert: nil, scale: false, ref)
        case "head.weight", "lm_head.weight":
            return slot(.lmHead, layer: nil, expert: nil, scale: false, ref)
        default:
            break
        }

        // Draft blocks (`mtp.N.…`): full backbone stages in their own namespace.
        if let rest = remainder(of: name, prefix: "mtp.") {
            let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, let block = Int(parts[0]), block >= 0 else { return nil }
            return classifyBlock(String(parts[1]), scope: .draft, layer: block, ref: ref)
        }

        // Vision tower, its projector, and the three image markers. The text graph
        // never reads them, but they are published in the same checkpoint, so an
        // unclassified name here would refuse an otherwise complete text load.
        if let rest = remainder(of: name, prefix: "vision.") {
            if let role = visionTowerRole(rest) {
                return slot(role, scope: .vision, layer: nil, expert: nil, scale: false, ref)
            }
            // `vision.blocks.N.<stage>`: one model per stage, addressed by block.
            guard let block = visionBlockIndex(rest) else { return nil }
            let body = String(rest.split(separator: ".", maxSplits: 2).last ?? "")
            guard let role = visionBlockRole(body) else { return nil }
            return slot(role, scope: .vision, layer: block, expert: nil, scale: false, ref)
        }
        if remainder(of: name, prefix: "aligner.") != nil {
            switch name {
            case "aligner.w1.weight":
                return slot(.visionAlignerW1, scope: .vision, layer: nil, expert: nil, scale: false, ref)
            case "aligner.w1.bias":
                return slot(.visionAlignerW1Bias, scope: .vision, layer: nil, expert: nil, scale: false, ref)
            case "aligner.w2.weight":
                return slot(.visionAlignerW2, scope: .vision, layer: nil, expert: nil, scale: false, ref)
            case "aligner.w2.bias":
                return slot(.visionAlignerW2Bias, scope: .vision, layer: nil, expert: nil, scale: false, ref)
            default:
                return nil
            }
        }
        switch name {
        case "image_start":
            return slot(.visionImageStart, scope: .vision, layer: nil, expert: nil, scale: false, ref)
        case "image_newline":
            return slot(.visionImageNewline, scope: .vision, layer: nil, expert: nil, scale: false, ref)
        case "image_end":
            return slot(.visionImageEnd, scope: .vision, layer: nil, expert: nil, scale: false, ref)
        default:
            break
        }

        // Backbone layer.
        guard let rest = remainder(of: name, prefix: "layers.") else { return nil }
        let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, let layer = Int(parts[0]), layer >= 0 else { return nil }
        return classifyBlock(String(parts[1]), scope: .backbone, layer: layer, ref: ref)
    }

    /// The `vision.blocks.N.…` stage index, nil for the tower-wide tensors
    /// (`vision.patch_embed.*`, `vision.norm.*`).
    private static func visionBlockIndex(_ rest: String) -> Int? {
        guard let blockRest = remainder(of: rest, prefix: "blocks.") else { return nil }
        let parts = blockRest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, let block = Int(parts[0]), block >= 0 else { return nil }
        return block
    }

    /// Tower-wide vision tensors: the patch embedding and the final norm.
    private static func visionTowerRole(_ rest: String) -> DeepSeekV41TensorRole? {
        switch rest {
        case "patch_embed.proj.weight":
            return .visionPatchEmbed
        case "patch_embed.proj.bias":
            return .visionPatchEmbedBias
        case "norm.weight":
            return .visionFinalNorm
        default:
            return nil
        }
    }

    /// One vision transformer block's stages, keyed by the name below
    /// `vision.blocks.N.`.
    private static func visionBlockRole(_ body: String) -> DeepSeekV41TensorRole? {
        switch body {
        case "norm1.weight":
            return .visionBlockNorm1
        case "attn.wqkv.weight":
            return .visionBlockWqkv
        case "attn.wqkv.bias":
            return .visionBlockWqkvBias
        case "attn.wo.weight":
            return .visionBlockWo
        case "attn.wo.bias":
            return .visionBlockWoBias
        case "norm2.weight":
            return .visionBlockNorm2
        case "mlp.w1.weight":
            return .visionBlockW1
        case "mlp.w2.weight":
            return .visionBlockW2
        default:
            return nil
        }
    }

    /// Classifies the part of a name below its layer prefix. Backbone layers and
    /// DSpark stages publish the same block (attention, MoE, hyper-connections),
    /// so they share this table and differ only in scope.
    private static func classifyBlock(
        _ body: String,
        scope: DeepSeekV41TensorScope,
        layer: Int,
        ref: DeepSeekV41TensorRef
    ) -> DeepSeekV41TensorSlot? {
        let (base, isScale) = splitScale(body)

        if let expertRest = remainder(of: base, prefix: "ffn.experts.") {
            let expertParts = expertRest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard expertParts.count == 2, let expert = Int(expertParts[0]), expert >= 0 else { return nil }
            guard let role = expertRole(String(expertParts[1])) else { return nil }
            return slot(role, scope: scope, layer: layer, expert: expert, scale: isScale, ref)
        }

        let roles: [String: DeepSeekV41TensorRole] = [
            "attn_norm": .attnNorm,
            "ffn_norm": .ffnNorm,
            "attn.wq_a": .attnWqA,
            "attn.q_norm": .attnWqANorm,
            "attn.wq_a_norm": .attnWqANorm,
            "attn.wq_b": .attnWqB,
            "attn.wkv": .attnWkv,
            "attn.kv_norm": .attnKvNorm,
            "attn.wkv_norm": .attnKvNorm,
            "attn.wo_a": .attnWoA,
            "attn.wo_b": .attnWoB,
            "attn.attn_sink": .attnSink,
            "attn.indexer.wq_b": .indexerWqB,
            "attn.indexer.wk": .indexerWk,
            "attn.indexer.weights_proj": .indexerWeights,
            "attn.indexer.k_norm": .indexerKNorm,
            "attn.compressor.wkv": .compressorWkv,
            "attn.compressor.wgate": .compressorWgate,
            "attn.compressor.norm": .compressorNorm,
            "ffn.gate": .routerWeight,
            "ffn.gate.bias": .routerBias,
            "ffn.gate.bias_vl": .routerVisionBias,
            "ffn.shared_experts.w1": .sharedExpertW1,
            "ffn.shared_experts.w2": .sharedExpertW2,
            "ffn.shared_experts.w3": .sharedExpertW3,
            "hc_attn_fn": .hcAttnFn,
            "hc_attn_base": .hcAttnBase,
            "hc_attn_scale": .hcAttnScale,
            "hc_ffn_fn": .hcFfnFn,
            "hc_ffn_base": .hcFfnBase,
            "hc_ffn_scale": .hcFfnScale,
            // `engram.embed` is the n-gram table itself (`engram_num_embeddings`
            // rows by `engram_head_dim`, plus its per-block scale companion).
            // `engram.wkv` is a dense projection over the retrieved rows, so it
            // must not be mistaken for the table: mapping it as one would map a
            // 25 600 x 6 144 matrix where a ~98 GB table lives.
            "engram.embed": .engramTable,
            "engram.wkv": .engramWkv,
            "engram.q_weight": .engramQuery,
            "engram.k_weight": .engramKey,
            // DSpark heads. Stage 0 carries the target-layer projection, the last
            // stage carries the norm and the two output heads.
            "main_proj": .draftMainProj,
            "main_norm": .draftMainNorm,
            "markov_head.embed": .draftMarkovEmbed,
            "markov_head.head": .draftMarkovHead,
            "confidence_head.proj": .draftConfidence
        ]
        if let role = roles[base] {
            // Norms carry no scale companion; only quantized linears do.
            return slot(role, scope: scope, layer: layer, expert: nil, scale: isScale, ref)
        }
        // `norm` is the final RMSNorm of the last DSpark stage. It is spelled like
        // the model-level `norm.weight`, which is why it only means anything here.
        if scope == .draft && base == "norm" {
            return slot(.draftNorm, scope: scope, layer: layer, expert: nil, scale: isScale, ref)
        }
        return nil
    }

    private static func expertRole(_ name: String) -> DeepSeekV41TensorRole? {
        switch name {
        case "w1": return .expertW1
        case "w2": return .expertW2
        case "w3": return .expertW3
        default: return nil
        }
    }

    /// Splits a tensor body into its base name and whether it is the paired
    /// quantization scale. Covers the three spellings in circulation.
    static func splitScale(_ body: String) -> (String, Bool) {
        for suffix in [".weight_scale", ".scales", ".scale"] {
            if body.hasSuffix(suffix) {
                return (String(body.dropLast(suffix.count)), true)
            }
        }
        if body.hasSuffix(".weight") {
            return (String(body.dropLast(".weight".count)), false)
        }
        return (body, false)
    }

    private static func remainder(of name: String, prefix: String) -> String? {
        guard name.hasPrefix(prefix) else { return nil }
        return String(name.dropFirst(prefix.count))
    }

    private static func slot(
        _ role: DeepSeekV41TensorRole,
        scope: DeepSeekV41TensorScope = .backbone,
        layer: Int?,
        expert: Int?,
        scale: Bool,
        _ ref: DeepSeekV41TensorRef
    ) -> DeepSeekV41TensorSlot {
        DeepSeekV41TensorSlot(
            role: role,
            scope: scope,
            layer: layer,
            expert: expert,
            isScale: scale,
            ref: ref
        )
    }
}

// MARK: - Completeness

/// Why a checkpoint cannot be run, phrased for the load log.
nonisolated public enum DeepSeekV41WeightError: Error, Equatable, CustomStringConvertible {
    case unconsumedTensors([String])
    case missingRequired([String])
    case unsupportedDtypes([String])
    case streamedTensorTooSmall(String)

    public var description: String {
        switch self {
        case .unconsumedTensors(let names):
            let shown = names.prefix(6).joined(separator: ", ")
            let more = names.count > 6 ? " (+\(names.count - 6) more)" : ""
            return "DeepSeek V4.1 checkpoint carries \(names.count) tensor(s) this engine never reads: \(shown)\(more)"
        case .missingRequired(let names):
            let shown = names.prefix(8).joined(separator: ", ")
            let more = names.count > 8 ? " (+\(names.count - 8) more)" : ""
            return "DeepSeek V4.1 checkpoint is missing \(names.count) required tensor(s): \(shown)\(more)"
        case .unsupportedDtypes(let names):
            let shown = names.prefix(6).joined(separator: ", ")
            return "DeepSeek V4.1 checkpoint uses dtypes this engine has no kernel for: \(shown)"
        case .streamedTensorTooSmall(let name):
            return "DeepSeek V4.1 Engram tensor '\(name)' is smaller than one row"
        }
    }
}

extension DeepSeekV41WeightCatalog {

    /// Which indexer tensors a layer has to publish.
    ///
    /// The reference builds an `Indexer` only on an index-source layer
    /// (`Attention.__init__`: `if self.is_index_source: self.indexer = Indexer(...)`),
    /// and inside one only a layer that compresses its own KV can produce index
    /// keys (`Indexer.owns_k = layer_id in kv_source_layers`). Every layer in
    /// between reuses the source's `topk_idxs` (`_compress_topk_idxs` returns
    /// `shared_attn.topk_idxs`) and publishes no indexer tensors at all.
    ///
    /// The rule lives here so it can be asserted without a checkpoint: demanding
    /// the full set on every compressed layer rejects the real 384-expert
    /// checkpoint, which publishes exactly this split (2/8/14/20 own the
    /// compressed KV and the index keys, 24/28/32/36 own index keys only, every
    /// other compressed layer owns neither).
    static func requiredIndexerRoles(
        layer: DeepSeekV41LayerPlan,
        config: DeepSeekV41Config
    ) -> [DeepSeekV41TensorRole] {
        guard layer.ownsCompressedIndex else { return [] }
        var roles: [DeepSeekV41TensorRole] = [.indexerWqB, .indexerWeights]
        if config.kvSourceLayerIDs.contains(layer.index) {
            // `Indexer.forward` normalises the projected key before rotating it.
            roles.append(contentsOf: [.indexerWk, .indexerKNorm])
        }
        return roles
    }

    /// The checkpoint's spelling for an indexer role, for the missing-tensor line.
    static func indexerLabel(_ role: DeepSeekV41TensorRole) -> String {
        switch role {
        case .indexerWqB: return "attn.indexer.wq_b.weight"
        case .indexerWk: return "attn.indexer.wk.weight"
        case .indexerWeights: return "attn.indexer.weights_proj.weight"
        case .indexerKNorm: return "attn.indexer.k_norm.weight"
        default: return role.rawValue
        }
    }

    /// Every requirement the plan imposes on the checkpoint, checked against the
    /// catalog. Returns human-readable problems; empty means the checkpoint can
    /// be run.
    public func validate(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan]
    ) -> [String] {
        var problems: [String] = []

        for duplicate in duplicateRoles {
            problems.append("two tensors claim one role: \(duplicate)")
        }
        problems.append(contentsOf: outOfRangeLayers)

        // Model-level.
        for (role, label) in [(DeepSeekV41TensorRole.tokenEmbedding, "embed.weight"),
                              (.finalNorm, "norm.weight"),
                              (.lmHead, "head.weight")] where tensor(role) == nil {
            problems.append("missing \(label)")
        }
        if let embed = tensor(.tokenEmbedding), embed.shape.count == 2,
           embed.shape[0] != config.vocabSize {
            problems.append(
                "embed.weight has \(embed.shape[0]) rows for vocab_size \(config.vocabSize)"
            )
        }

        for layer in plan {
            let l = layer.index
            var required: [(DeepSeekV41TensorRole, String)] = [
                (.attnNorm, "attn_norm.weight"),
                (.ffnNorm, "ffn_norm.weight"),
                (.attnWqA, "attn.wq_a.weight"),
                (.attnWkv, "attn.wkv.weight"),
                (.attnWoA, "attn.wo_a.weight"),
                (.attnWoB, "attn.wo_b.weight"),
                (.routerWeight, "ffn.gate.weight")
            ]
            if config.sharedExpertCount > 0 {
                required.append(contentsOf: [
                    (.sharedExpertW1, "ffn.shared_experts.w1.weight"),
                    (.sharedExpertW2, "ffn.shared_experts.w2.weight"),
                    (.sharedExpertW3, "ffn.shared_experts.w3.weight")
                ])
            }
            if config.qLoraRank > 0 {
                required.append((.attnWqANorm, "attn.wq_a_norm.weight"))
            }
            required.append((.attnWqB, "attn.wq_b.weight"))
            // `kv_norm` is unconditional in the reference (`RMSNorm(head_dim)` over the
            // whole cache record), independent of whether the checkpoint also publishes
            // a separate `kv_lora_rank`.
            required.append((.attnKvNorm, "attn.kv_norm.weight"))
            if config.hcMult > 1 {
                required.append(contentsOf: [
                    (.hcAttnFn, "hc_attn_fn"),
                    (.hcAttnBase, "hc_attn_base"),
                    (.hcFfnFn, "hc_ffn_fn"),
                    (.hcFfnBase, "hc_ffn_base")
                ])
            }
            // Only the four `kv_source_layer_ids` layers publish compressor
            // weights: the other compressed layers read the rows their source
            // layer wrote. `wgate` is ratio-specific too, because a ratio of 1 is
            // a plain projection with no gate (``Compressor.forward``).
            if layer.ownsCompressedKV {
                required.append((.compressorWkv, "attn.compressor.wkv.weight"))
                if layer.compressRatio > 1 {
                    required.append((.compressorWgate, "attn.compressor.wgate.weight"))
                }
                required.append((.compressorNorm, "attn.compressor.norm.weight"))
            }
            for role in Self.requiredIndexerRoles(layer: layer, config: config) {
                required.append((role, Self.indexerLabel(role)))
            }
            if layer.engramTable != nil {
                required.append(contentsOf: [
                    (.engramTable, "engram.embed.weight"),
                    (.engramWkv, "engram.wkv.weight"),
                    (.engramQuery, "engram.q_weight"),
                    (.engramKey, "engram.k_weight")
                ])
                if let ref = tensor(.engramTable, layer: l), (ref.shape.last ?? 0) <= 0 {
                    problems.append("layers.\(l): Engram table has no columns")
                }
            }
            for (role, label) in required where tensor(role, layer: l) == nil {
                problems.append("layers.\(l): missing \(label)")
            }

            // Routed experts: every layer needs at least one expert, and the
            // count must match the config so the router's top-k can never index
            // past the published set.
            let experts = expertIndices(layer: l)
            if experts.isEmpty {
                problems.append("layers.\(l): no routed experts (ffn.experts.*)")
            } else {
                if experts.count != config.numRoutedExperts {
                    problems.append(
                        "layers.\(l): \(experts.count) routed experts, config says \(config.numRoutedExperts)"
                    )
                }
                if experts != Array(0..<experts.count) {
                    problems.append("layers.\(l): expert ids are not contiguous from 0")
                }
                for role in [DeepSeekV41TensorRole.expertW1, .expertW2, .expertW3] {
                    for expert in experts where tensor(role, layer: l, expert: expert) == nil {
                        problems.append("layers.\(l): expert \(expert) is missing \(role.rawValue)")
                    }
                }
            }
        }

        // DSpark stages. Each `mtp.N.*` is a full `Block` (attention, MoE,
        // hyper-connections) in the draft namespace; only stage 0 carries the
        // target-layer projection and only the last stage carries the norm and
        // the two output heads (``DSparkBlock.__init__``).
        for block in 0..<config.numNextNPredictLayers {
            var required: [(DeepSeekV41TensorRole, String)] = [
                (.attnNorm, "attn_norm.weight"),
                (.attnWqA, "attn.wq_a.weight"),
                (.attnWqB, "attn.wq_b.weight"),
                (.attnWkv, "attn.wkv.weight"),
                (.attnKvNorm, "attn.kv_norm.weight"),
                (.attnWoA, "attn.wo_a.weight"),
                (.attnWoB, "attn.wo_b.weight"),
                (.ffnNorm, "ffn_norm.weight"),
                (.routerWeight, "ffn.gate.weight")
            ]
            if config.qLoraRank > 0 {
                required.append((.attnWqANorm, "attn.q_norm.weight"))
            }
            if config.hcMult > 1 {
                required.append(contentsOf: [
                    (.hcAttnFn, "hc_attn_fn"),
                    (.hcAttnBase, "hc_attn_base"),
                    (.hcFfnFn, "hc_ffn_fn"),
                    (.hcFfnBase, "hc_ffn_base")
                ])
            }
            if block == 0 {
                required.append(contentsOf: [
                    (.draftMainProj, "main_proj.weight"),
                    (.draftMainNorm, "main_norm.weight")
                ])
            }
            if block == config.numNextNPredictLayers - 1 {
                required.append(contentsOf: [
                    (.draftNorm, "norm.weight"),
                    (.draftMarkovEmbed, "markov_head.embed.weight"),
                    (.draftMarkovHead, "markov_head.head.weight"),
                    (.draftConfidence, "confidence_head.proj.weight")
                ])
            }
            for (role, label) in required where tensor(role, scope: .draft, layer: block) == nil {
                problems.append("mtp.\(block): missing \(label)")
            }
            let draftExperts = expertIndices(layer: block, scope: .draft)
            if draftExperts.isEmpty {
                problems.append("mtp.\(block): no routed experts (ffn.experts.*)")
            } else if draftExperts.count != config.draftExpertCounts.routed {
                problems.append(
                    "mtp.\(block): \(draftExperts.count) routed experts, config says \(config.draftExpertCounts.routed)"
                )
            }
        }

        // Dtypes: the kernels are per-format, so an unknown dtype is a load-time
        // refusal rather than a wrong answer at token 1.
        let unsupported = slots
            .filter { !$0.ref.isSupportedDtype }
            .map { $0.ref.name + " (\($0.ref.dtype))" }
        if !unsupported.isEmpty {
            problems.append(contentsOf: unsupported.map { "unsupported dtype: \($0)" })
        }

        // Names the catalog could not place are a hard error: a tensor nobody
        // reads is either a new architecture feature or a corrupted manifest.
        if !unconsumed.isEmpty {
            problems.append(contentsOf: unconsumed.map { "unconsumed tensor: \($0)" })
        }
        return problems
    }
}
