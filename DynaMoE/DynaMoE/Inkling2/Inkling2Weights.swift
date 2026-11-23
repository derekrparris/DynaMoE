//
//  Inkling2Weights.swift
//  DynaMoE
//
//  Tensor-name resolution for Inkling 2 Mini checkpoints.
//
//  Why this exists instead of reusing the generic binder:
//
//    1. Every name is prefixed `model.llm.` or `model.visual.`, and the text
//       tower's attention is not a conventional GQA block: it is
//       `wq_du` / `wk_dv` / `wv_dv` / `wr_du` / `wo_ud` plus a per-head sink and
//       a relative-logit projection, with the post-attention and post-MLP
//       convolutions as separate tensors.
//    2. Routed experts are FUSED banks: one `w13_weight` holds all 128 experts
//       (gate and up interleaved along the output dimension), with one scale
//       tensor per layout. There is no per-expert tensor to bind.
//    3. The same physical layer runs several times, so a role is addressed by
//       PHYSICAL layer and the visit schedule decides how often it is read.
//       Cross-pass controllers exist only on the core layers.
//    4. The vision tower lives in the same file under `model.visual.`, and the
//       DSpark sidecar in its own `mtp.safetensors` under a different root.
//
//  So the catalog does two jobs:
//    * resolve every name to an exact (scope, role, layer) slot, and
//    * FAIL CLOSED: a tensor it cannot place, a slot the plan requires and the
//      checkpoint lacks, a scale companion that should not be there, or a
//      dtype/shape that disagrees with the config is reported with its name.
//      A silently-ignored tensor is how a model runs at full speed and emits
//      the wrong distribution.
//
//  Classification is EXACT-MATCH on purpose. Tolerant suffix matching is right
//  for DeepSeek's converters, which spell scales three ways; here
//  `w13_weight` and `w13_weight_scale` are two different tensors of one bank,
//  and a suffix rule would fold the second into the first.
//

import Foundation

// MARK: - Scope

/// Which namespace a tensor name lives in.
nonisolated public enum Inkling2TensorScope: String, Sendable, Equatable {
    /// `model.llm.*` -- the text tower, its residual stack and the cross-pass
    /// queries.
    case text
    /// `model.visual.*` -- the vision tower and its projector.
    case vision
    /// The `mtp.safetensors` sidecar's rootless names (`layers.N.*`,
    /// `markov_head.*`, `fuse_weight`, ...).
    case sidecar

    public var isMainFile: Bool { self != .sidecar }
}

// MARK: - Roles

nonisolated public enum Inkling2TensorRole: String, Sendable, Equatable, CaseIterable {
    // Model level
    case tokenEmbedding
    case embeddingNorm
    case finalNorm
    case lmHead

    // Block attention residuals and cross-pass queries
    case attnResAttnQuery
    case attnResMlpQuery
    case attnResOutputQuery
    case crossPassQuery
    case crossPassKVQuery

    // Per-layer attention
    case attnNorm
    case qInputNorm
    case kInputNorm
    case vInputNorm
    case attnWqDu
    case attnWkDv
    case attnWvDv
    case attnWrDu
    case attnWoUd
    case attnGate
    case attnSinkProj
    case attnRelLogitsProj
    case attnQL2ClipScale
    case attnRelL2ClipScale
    case attnKSconvWeight
    case attnKSconvStatic
    case attnKSconvProj
    case attnVSconvWeight
    case attnVSconvStatic
    case attnVSconvProj
    case attnSconv
    case mlpSconv

    // Per-layer feed-forward
    case mlpNorm
    case routerWeight
    case routerBias
    case expertW13
    case expertW13Scale
    case expertW13Amax
    case expertW13OuterScaleDenominator
    case expertW2
    case expertW2Scale
    case expertW2Amax
    case expertW2OuterScaleDenominator

    // Cross-pass message controllers (core layers only)
    case controllerAttnDown
    case controllerAttnForget
    case controllerAttnGain
    case controllerKDown
    case controllerKForget
    case controllerKGain
    case controllerVDown
    case controllerVForget
    case controllerVGain
    case controllerMlpDown
    case controllerMlpForget
    case controllerMlpGain

    // Vision
    case visionClassEmbedding
    case visionClsPositionEmbedding
    case visionPatchEmbedding
    case visionPositionEmbedding
    case visionNormPreWeight
    case visionNormPreBias
    case visionBlockLn1Weight
    case visionBlockLn1Bias
    case visionBlockLn2Weight
    case visionBlockLn2Bias
    case visionBlockScale1
    case visionBlockScale2
    case visionBlockQProjWeight
    case visionBlockQProjBias
    case visionBlockKProjWeight
    case visionBlockKProjBias
    case visionBlockVProjWeight
    case visionBlockVProjBias
    case visionBlockOutProjWeight
    case visionBlockOutProjBias
    case visionBlockFcWeight
    case visionBlockFcBias
    case visionBlockProjWeight
    case visionBlockProjBias
    case visionProjIn
    case visionProjOut

    // DSpark sidecar
    case draftFuseWeight
    case draftInputNorm
    case draftPostAttentionNorm
    case draftDflareNorm
    case draftQProj
    case draftKProj
    case draftVProj
    case draftOProj
    case draftQNorm
    case draftKNorm
    case draftKProjContext
    case draftVProjContext
    case draftKNormContext
    case draftGateProj
    case draftUpProj
    case draftDownProj
    case draftMarkovDflareNorm
    case draftMarkovHiddenOutWeight
    case draftMarkovHiddenOutBias
    case draftMarkovW1
    case draftMarkovRnnProjWeight
    case draftMarkovRnnProjBias
    case draftMaskEmbed
    case draftFinalNorm

    public var isExpertBank: Bool {
        switch self {
        case .expertW13, .expertW2:
            return true
        default:
            return false
        }
    }

    /// The scale/amax/denominator tensors that describe an expert bank.
    public var isExpertCompanion: Bool {
        switch self {
        case .expertW13Scale, .expertW13Amax, .expertW13OuterScaleDenominator,
             .expertW2Scale, .expertW2Amax, .expertW2OuterScaleDenominator:
            return true
        default:
            return false
        }
    }

    public var isVision: Bool {
        switch self {
        case .visionClassEmbedding, .visionClsPositionEmbedding, .visionPatchEmbedding,
             .visionPositionEmbedding, .visionNormPreWeight, .visionNormPreBias,
             .visionBlockLn1Weight, .visionBlockLn1Bias, .visionBlockLn2Weight,
             .visionBlockLn2Bias, .visionBlockScale1, .visionBlockScale2,
             .visionBlockQProjWeight, .visionBlockQProjBias, .visionBlockKProjWeight,
             .visionBlockKProjBias, .visionBlockVProjWeight, .visionBlockVProjBias,
             .visionBlockOutProjWeight, .visionBlockOutProjBias, .visionBlockFcWeight,
             .visionBlockFcBias, .visionBlockProjWeight, .visionBlockProjBias,
             .visionProjIn, .visionProjOut:
            return true
        default:
            return false
        }
    }

    public var isDraft: Bool {
        switch self {
        case .draftFuseWeight, .draftInputNorm, .draftPostAttentionNorm, .draftDflareNorm,
             .draftQProj, .draftKProj, .draftVProj, .draftOProj, .draftQNorm, .draftKNorm,
             .draftKProjContext, .draftVProjContext, .draftKNormContext, .draftGateProj,
             .draftUpProj, .draftDownProj, .draftMarkovDflareNorm,
             .draftMarkovHiddenOutWeight, .draftMarkovHiddenOutBias, .draftMarkovW1,
             .draftMarkovRnnProjWeight, .draftMarkovRnnProjBias, .draftMaskEmbed,
             .draftFinalNorm:
            return true
        default:
            return false
        }
    }

    /// Roles only the core (looped) layers carry.
    public var isCrossPassController: Bool {
        switch self {
        case .controllerAttnDown, .controllerAttnForget, .controllerAttnGain,
             .controllerKDown, .controllerKForget, .controllerKGain,
             .controllerVDown, .controllerVForget, .controllerVGain,
             .controllerMlpDown, .controllerMlpForget, .controllerMlpGain:
            return true
        default:
            return false
        }
    }

    /// Roles only full-attention layers carry: the separately normalized Q/K/V
    /// inputs and the dynamic convolution projections.
    public var isFullAttentionOnly: Bool {
        switch self {
        case .qInputNorm, .kInputNorm, .vInputNorm,
             .attnKSconvStatic, .attnVSconvStatic, .attnKSconvProj, .attnVSconvProj:
            return true
        default:
            return false
        }
    }

    /// Roles only sliding (local) layers carry: the single attention norm and
    /// the static K/V convolutions.
    public var isLocalOnly: Bool {
        switch self {
        case .attnNorm, .attnKSconvWeight, .attnVSconvWeight:
            return true
        default:
            return false
        }
    }

    /// Expert-bank companions, which only the NVFP4 layout publishes.
    public var isNVFP4OnlyCompanion: Bool {
        self == .expertW13Amax || self == .expertW13OuterScaleDenominator
            || self == .expertW2Amax || self == .expertW2OuterScaleDenominator
    }
}

// MARK: - Slots

/// One tensor as the catalog sees it: name, storage dtype and logical shape.
nonisolated public struct Inkling2TensorRef: Sendable, Equatable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let shardIndex: UInt32?
    public let byteOffset: UInt64?

    public init(
        name: String,
        dtype: String,
        shape: [Int],
        shardIndex: UInt32? = nil,
        byteOffset: UInt64? = nil
    ) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.shardIndex = shardIndex
        self.byteOffset = byteOffset
    }

    /// Parses the Rust summary's `shapeDisplay` ("[128, 2304, 1152]").
    public init(metadata: TensorMetadata) {
        let digits = metadata.shapeDisplay
            .split(whereSeparator: { $0 == "[" || $0 == "]" || $0 == "," || $0 == " " })
            .compactMap { Int($0) }
        self.init(
            name: metadata.name,
            dtype: metadata.dtype,
            shape: digits,
            shardIndex: metadata.shardIndex,
            byteOffset: metadata.offsetStart
        )
    }

    /// Whether the safetensors dtype is one the Inkling 2 kernels understand.
    /// `U8` is here because NVFP4 values and E8M0 scales are both stored as it.
    public var isSupportedDtype: Bool {
        let upper = dtype.uppercased()
        return upper.contains("BF16") || upper.contains("F32") || upper.contains("F16")
            || upper.contains("F8_E4M3") || upper == "U8" || upper.contains("F8_E8M0")
    }

    public var elementCount: Int {
        shape.reduce(1) { $0 * max(1, $1) }
    }
}

/// Where a tensor lives in the model: a role, a scope and optionally a layer.
nonisolated public struct Inkling2TensorSlot: Sendable, Equatable {
    public let role: Inkling2TensorRole
    public let scope: Inkling2TensorScope
    /// Text/vision block index, or the draft layer for sidecar roles, or nil
    /// for model-level tensors.
    public let layer: Int?
    public let ref: Inkling2TensorRef

    public var key: String {
        "\(scope.rawValue)|\(role.rawValue)|\(layer.map(String.init) ?? "-")"
    }
}

// MARK: - Catalog

nonisolated public struct Inkling2WeightCatalog: Sendable {

    public let slots: [Inkling2TensorSlot]
    /// Names the catalog could not place. The load refuses when this is not empty.
    public let unconsumed: [String]
    /// Two different tensors that resolve to the same role.
    public let duplicateRoles: [String]
    /// Tensors naming a layer the config does not have.
    public let outOfRangeLayers: [String]

    private let index: [String: Inkling2TensorSlot]

    /// Builds the catalog from every tensor in the main file and, when it is
    /// supplied, the DSpark sidecar. `layerCount` is the text tower's physical
    /// layer count; the vision block count is taken from the tensors themselves.
    public init(
        mainFile tensors: [Inkling2TensorRef],
        sidecar: [Inkling2TensorRef] = [],
        layerCount: Int
    ) {
        var collected: [Inkling2TensorSlot] = []
        var duplicates: [String] = []
        var outOfRange: [String] = []
        var unconsumed: [String] = []
        var byKey: [String: Inkling2TensorSlot] = [:]

        func place(_ tensor: Inkling2TensorRef, scope: Inkling2TensorScope) {
            guard let slot = Self.classify(name: tensor.name, ref: tensor, scope: scope) else {
                unconsumed.append(tensor.name)
                return
            }
            if slot.scope == .text, let layer = slot.layer, layer < 0 || layer >= max(0, layerCount) {
                outOfRange.append("\(tensor.name) names layer \(layer) of \(layerCount)")
            }
            if let existing = byKey[slot.key] {
                duplicates.append(
                    "\(existing.ref.name) and \(tensor.name) both resolve to "
                        + "\(slot.scope.rawValue) \(slot.role.rawValue)"
                        + (slot.layer.map { " layer \($0)" } ?? "")
                )
                collected.append(slot)
                return
            }
            collected.append(slot)
            byKey[slot.key] = slot
        }

        for tensor in Self.coalescingExpertBanks(tensors) {
            // The main file holds text and vision names; the `model.llm.` /
            // `model.visual.` prefix decides which, never a substring search.
            if tensor.name.hasPrefix("model.visual.") {
                place(tensor, scope: .vision)
            } else if tensor.name.hasPrefix("model.llm.") {
                place(tensor, scope: .text)
            } else {
                unconsumed.append(tensor.name)
            }
        }
        for tensor in sidecar {
            place(tensor, scope: .sidecar)
        }

        self.slots = collected
        self.unconsumed = unconsumed
        self.duplicateRoles = duplicates
        self.outOfRangeLayers = outOfRange
        self.index = byKey
    }

    /// Reassembles the fused expert banks the Rust summary reports split.
    ///
    /// The generic loader expands every stacked 3D tensor whose first dimension
    /// is an expert count into one `TensorMetadata` per expert
    /// (`...mlp.experts.0.w13_weight_scale`, `...experts.1.w13_weight_scale`, ...),
    /// because the generic engine binds experts one at a time. This family's
    /// banks are read WHOLE -- one scale tensor serves all 128 experts' rows --
    /// so the split view is coalesced back to the checkpoint's own name, with
    /// the bank's shape restored from the sub-shape and its offset from expert 0
    /// (the entries are contiguous, which the per-expert offsets themselves are
    /// checked to confirm).
    static func coalescingExpertBanks(_ tensors: [Inkling2TensorRef]) -> [Inkling2TensorRef] {
        struct GroupKey: Hashable {
            let prefix: String
            let suffix: String
            let dtype: String
        }
        var order: [GroupKey] = []
        var groups: [GroupKey: [(index: Int, ref: Inkling2TensorRef, subshape: [Int])]] = [:]
        var passthrough: [Inkling2TensorRef] = []

        for tensor in tensors {
            guard let split = splitExpertEntry(tensor.name) else {
                passthrough.append(tensor)
                continue
            }
            let key = GroupKey(prefix: split.prefix, suffix: split.suffix, dtype: tensor.dtype)
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append((split.index, tensor, tensor.shape))
        }

        var reassembled: [Inkling2TensorRef] = passthrough
        for key in order {
            guard var entries = groups[key] else { continue }
            entries.sort { $0.index < $1.index }
            let indices = entries.map(\.index)
            let contiguous = indices == Array(0..<indices.count)
            let subshape = entries.first?.subshape ?? []
            // Every entry must describe the same sub-shape, and expert E must sit
            // exactly E banks after expert 0: otherwise this is a name family
            // that merely looks split, and fusing it would invent an offset.
            let uniform = entries.allSatisfy { $0.subshape == subshape }
            let firstOffset = entries.first?.ref.byteOffset ?? 0
            let perExpert = entries.first.map { ($0.ref.shape.reduce(1, *) ) } ?? 0
            let stridesAgree = entries.allSatisfy { entry in
                (entry.ref.byteOffset ?? 0) == firstOffset + UInt64(entry.index) * UInt64(perExpert * Self.storageWidth(entry.ref.dtype))
            }
            guard contiguous, uniform, stridesAgree, !subshape.isEmpty else {
                passthrough.append(contentsOf: entries.map(\.ref))
                continue
            }
            reassembled.append(
                Inkling2TensorRef(
                    name: key.prefix + key.suffix,
                    dtype: key.dtype,
                    shape: [entries.count] + subshape,
                    shardIndex: entries.first?.ref.shardIndex,
                    byteOffset: firstOffset
                )
            )
        }
        // Preserve the file's own order as far as the grouping allows: a
        // catalog is order-independent, but a diagnostic that prints an
        // unplaced name reads better beside its neighbours.
        return reassembled
    }

    /// `model.llm.layers.N.mlp.experts.E.<suffix>` -> (prefix, E, suffix), nil
    /// when the name is not a per-expert entry.
    static func splitExpertEntry(_ name: String) -> (prefix: String, index: Int, suffix: String)? {
        guard let range = name.range(of: ".mlp.experts.") else { return nil }
        let prefix = String(name[..<range.upperBound])
        let rest = name[range.upperBound...]
        let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, let index = Int(parts[0]), index >= 0 else { return nil }
        return (prefix, index, String(parts[1]))
    }

    /// Bytes per stored element, for the contiguity check above.
    static func storageWidth(_ dtype: String) -> Int {
        let upper = dtype.uppercased()
        if upper == "U8" || upper.contains("F8") { return 1 }
        if upper.contains("BF16") || upper.contains("F16") { return 2 }
        if upper.contains("F32") { return 4 }
        return 2
    }

    public func tensor(
        _ role: Inkling2TensorRole,
        scope: Inkling2TensorScope? = nil,
        layer: Int? = nil
    ) -> Inkling2TensorRef? {
        index[Self.key(role, scope: scope ?? Self.defaultScope(role), layer: layer)]?.ref
    }

    /// Which scope a role lives in when the caller does not say. Vision and
    /// draft roles only ever exist in their own scope, and a lookup that named
    /// no scope defaulted to `.text` found nothing -- which is how
    /// `visionPatchEmbedding` came back missing from a checkpoint that has it.
    public static func defaultScope(_ role: Inkling2TensorRole) -> Inkling2TensorScope {
        if role.isVision { return .vision }
        if role.isDraft { return .sidecar }
        return .text
    }

    public static func key(
        _ role: Inkling2TensorRole,
        scope: Inkling2TensorScope? = nil,
        layer: Int? = nil
    ) -> String {
        let resolved = scope ?? defaultScope(role)
        return "\(resolved.rawValue)|\(role.rawValue)|\(layer.map(String.init) ?? "-")"
    }

    /// Slots of one role, ascending by layer, for the "which layers have this" checks.
    public func layers(of role: Inkling2TensorRole, scope: Inkling2TensorScope? = nil) -> [Int] {
        let resolved = scope ?? Self.defaultScope(role)
        return Set(slots.filter { $0.role == role && $0.scope == resolved }.compactMap(\.layer)).sorted()
    }

    /// Bytes of every tensor that is not an expert bank, i.e. what a dense pin
    /// or a residency plan would keep.
    public var nonExpertBytes: Int {
        slots.filter { !$0.role.isExpertBank && !$0.role.isExpertCompanion }
            .reduce(0) { $0 + Self.byteCount($1.ref) }
    }

    public var expertBankBytes: Int {
        slots.filter { $0.role.isExpertBank || $0.role.isExpertCompanion }
            .reduce(0) { $0 + Self.byteCount($1.ref) }
    }

    /// Bytes of one tensor, at the width its dtype stores.
    public static func byteCount(_ ref: Inkling2TensorRef) -> Int {
        let values = ref.elementCount
        let upper = ref.dtype.uppercased()
        if upper == "U8" || upper.contains("F8") { return values }
        if upper.contains("BF16") || upper.contains("F16") { return values * 2 }
        if upper.contains("F32") { return values * 4 }
        return values * 2
    }

    // MARK: Classification

    /// Resolves one tensor name within its scope.
    static func classify(
        name: String,
        ref: Inkling2TensorRef,
        scope: Inkling2TensorScope
    ) -> Inkling2TensorSlot? {
        func slot(_ role: Inkling2TensorRole, layer: Int? = nil) -> Inkling2TensorSlot {
            Inkling2TensorSlot(role: role, scope: scope, layer: layer, ref: ref)
        }
        switch scope {
        case .text:
            guard let rest = remainder(of: name, prefix: "model.llm.") else { return nil }
            switch rest {
            case "embed.weight": return slot(.tokenEmbedding)
            case "embed_norm.weight": return slot(.embeddingNorm)
            case "norm.weight": return slot(.finalNorm)
            case "unembed.weight": return slot(.lmHead)
            case "middle_cycle_cross_pass_queries": return slot(.crossPassQuery)
            case "middle_cycle_cross_pass_kv_queries": return slot(.crossPassKVQuery)
            default: break
            }
            if let queryRest = remainder(of: rest, prefix: "attn_res.attn_queries.") {
                guard let index = Int(queryRest) else { return nil }
                return slot(.attnResAttnQuery, layer: index)
            }
            if let queryRest = remainder(of: rest, prefix: "attn_res.mlp_queries.") {
                guard let index = Int(queryRest) else { return nil }
                return slot(.attnResMlpQuery, layer: index)
            }
            if rest == "attn_res.output_query" { return slot(.attnResOutputQuery) }
            guard let layerRest = remainder(of: rest, prefix: "layers.") else { return nil }
            let parts = layerRest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, let layer = Int(parts[0]), layer >= 0 else { return nil }
            guard let role = textLayerRole(String(parts[1])) else { return nil }
            return slot(role, layer: layer)

        case .vision:
            guard let rest = remainder(of: name, prefix: "model.visual.") else { return nil }
            switch rest {
            case "backbone.embeddings.class_embedding": return slot(.visionClassEmbedding)
            case "backbone.embeddings.cls_position_embedding": return slot(.visionClsPositionEmbedding)
            case "backbone.embeddings.patch_embedding.weight": return slot(.visionPatchEmbedding)
            case "backbone.embeddings.position_embedding": return slot(.visionPositionEmbedding)
            case "backbone.ln_pre.weight": return slot(.visionNormPreWeight)
            case "backbone.ln_pre.bias": return slot(.visionNormPreBias)
            case "proj_in.weight": return slot(.visionProjIn)
            case "proj_out.weight": return slot(.visionProjOut)
            default: break
            }
            guard let blockRest = remainder(of: rest, prefix: "backbone.layers.") else { return nil }
            let parts = blockRest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, let block = Int(parts[0]), block >= 0 else { return nil }
            guard let role = visionBlockRole(String(parts[1])) else { return nil }
            return slot(role, layer: block)

        case .sidecar:
            switch name {
            case "fuse_weight": return slot(.draftFuseWeight)
            case "mask_embed": return slot(.draftMaskEmbed)
            case "norm.weight": return slot(.draftFinalNorm)
            case "markov_head.dflare_norm.weight": return slot(.draftMarkovDflareNorm)
            case "markov_head.hidden_out.weight": return slot(.draftMarkovHiddenOutWeight)
            case "markov_head.hidden_out.bias": return slot(.draftMarkovHiddenOutBias)
            case "markov_head.markov_w1.weight": return slot(.draftMarkovW1)
            case "markov_head.rnn_proj.weight": return slot(.draftMarkovRnnProjWeight)
            case "markov_head.rnn_proj.bias": return slot(.draftMarkovRnnProjBias)
            default: break
            }
            guard let layerRest = remainder(of: name, prefix: "layers.") else { return nil }
            let parts = layerRest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, let layer = Int(parts[0]), layer >= 0 else { return nil }
            guard let role = sidecarLayerRole(String(parts[1])) else { return nil }
            return slot(role, layer: layer)
        }
    }

    /// The body of a `model.llm.layers.N.` name.
    static func textLayerRole(_ body: String) -> Inkling2TensorRole? {
        switch body {
        case "attn_norm.weight": return .attnNorm
        case "q_input_norm.weight": return .qInputNorm
        case "k_input_norm.weight": return .kInputNorm
        case "v_input_norm.weight": return .vInputNorm
        case "attn.wq_du.weight": return .attnWqDu
        case "attn.wk_dv.weight": return .attnWkDv
        case "attn.wv_dv.weight": return .attnWvDv
        case "attn.wr_du.weight": return .attnWrDu
        case "attn.wo_ud.weight": return .attnWoUd
        case "attn.gate_attn.weight": return .attnGate
        case "attn.attn_sink_proj.weight": return .attnSinkProj
        case "attn.rel_logits_proj.proj": return .attnRelLogitsProj
        case "attn.q_l2_clip.scale": return .attnQL2ClipScale
        case "attn.rel_l2_clip.scale": return .attnRelL2ClipScale
        case "attn.k_sconv.weight": return .attnKSconvWeight
        case "attn.k_sconv.static_w": return .attnKSconvStatic
        case "attn.k_sconv.weight_proj": return .attnKSconvProj
        case "attn.v_sconv.weight": return .attnVSconvWeight
        case "attn.v_sconv.static_w": return .attnVSconvStatic
        case "attn.v_sconv.weight_proj": return .attnVSconvProj
        case "attn_sconv.weight": return .attnSconv
        case "mlp_sconv.weight": return .mlpSconv
        case "mlp_norm.weight": return .mlpNorm
        case "mlp.gate.weight": return .routerWeight
        case "mlp.gate.bias": return .routerBias
        case "mlp.experts.w13_weight": return .expertW13
        case "mlp.experts.w13_weight_scale": return .expertW13Scale
        case "mlp.experts.w13_weight_amax": return .expertW13Amax
        case "mlp.experts.w13_weight_outer_scale_denominator": return .expertW13OuterScaleDenominator
        case "mlp.experts.w2_weight": return .expertW2
        case "mlp.experts.w2_weight_scale": return .expertW2Scale
        case "mlp.experts.w2_weight_amax": return .expertW2Amax
        case "mlp.experts.w2_weight_outer_scale_denominator": return .expertW2OuterScaleDenominator
        case "controllers.attn.down_weight": return .controllerAttnDown
        case "controllers.attn.forget_weight": return .controllerAttnForget
        case "controllers.attn.gain_weight": return .controllerAttnGain
        case "controllers.k.down_weight": return .controllerKDown
        case "controllers.k.forget_weight": return .controllerKForget
        case "controllers.k.gain_weight": return .controllerKGain
        case "controllers.v.down_weight": return .controllerVDown
        case "controllers.v.forget_weight": return .controllerVForget
        case "controllers.v.gain_weight": return .controllerVGain
        case "controllers.mlp.down_weight": return .controllerMlpDown
        case "controllers.mlp.forget_weight": return .controllerMlpForget
        case "controllers.mlp.gain_weight": return .controllerMlpGain
        default: return nil
        }
    }

    /// The body of a `model.visual.backbone.layers.N.` name.
    static func visionBlockRole(_ body: String) -> Inkling2TensorRole? {
        switch body {
        case "ln_1.weight": return .visionBlockLn1Weight
        case "ln_1.bias": return .visionBlockLn1Bias
        case "ln_2.weight": return .visionBlockLn2Weight
        case "ln_2.bias": return .visionBlockLn2Bias
        case "ls_1.gamma": return .visionBlockScale1
        case "ls_2.gamma": return .visionBlockScale2
        case "self_attn.q_proj.weight": return .visionBlockQProjWeight
        case "self_attn.q_proj.bias": return .visionBlockQProjBias
        case "self_attn.k_proj.weight": return .visionBlockKProjWeight
        case "self_attn.k_proj.bias": return .visionBlockKProjBias
        case "self_attn.v_proj.weight": return .visionBlockVProjWeight
        case "self_attn.v_proj.bias": return .visionBlockVProjBias
        case "self_attn.out_proj.weight": return .visionBlockOutProjWeight
        case "self_attn.out_proj.bias": return .visionBlockOutProjBias
        case "mlp.c_fc.weight": return .visionBlockFcWeight
        case "mlp.c_fc.bias": return .visionBlockFcBias
        case "mlp.c_proj.weight": return .visionBlockProjWeight
        case "mlp.c_proj.bias": return .visionBlockProjBias
        default: return nil
        }
    }

    /// The body of a sidecar `layers.N.` name.
    static func sidecarLayerRole(_ body: String) -> Inkling2TensorRole? {
        switch body {
        case "dflare_norm.weight": return .draftDflareNorm
        case "input_layernorm.weight": return .draftInputNorm
        case "post_attention_layernorm.weight": return .draftPostAttentionNorm
        case "mlp.down_proj.weight": return .draftDownProj
        case "mlp.gate_proj.weight": return .draftGateProj
        case "mlp.up_proj.weight": return .draftUpProj
        case "self_attn.k_norm.weight": return .draftKNorm
        case "self_attn.k_norm_ctx.weight": return .draftKNormContext
        case "self_attn.k_proj.weight": return .draftKProj
        case "self_attn.k_proj_ctx.weight": return .draftKProjContext
        case "self_attn.o_proj.weight": return .draftOProj
        case "self_attn.q_norm.weight": return .draftQNorm
        case "self_attn.q_proj.weight": return .draftQProj
        case "self_attn.v_proj.weight": return .draftVProj
        case "self_attn.v_proj_ctx.weight": return .draftVProjContext
        default: return nil
        }
    }

    private static func remainder(of name: String, prefix: String) -> String? {
        guard name.hasPrefix(prefix) else { return nil }
        return String(name.dropFirst(prefix.count))
    }
}

// MARK: - Query unpacking

/// How the checkpoint's packed query tensors map onto the reference's logical
/// parameters. Every one is pinned by a test: the counts alone would leave the
/// row order ambiguous, and a swapped row is a silent numeric error.
nonisolated public enum Inkling2QueryUnpacking {

    /// `attn_res.attn_queries.N` belongs to text layer `N + 1`: the reference
    /// gives layer 0 no attention-input read (`attention_input is None`), so
    /// there are 23 tensors for 24 layers. Full-attention layers publish THREE
    /// queries (Q, K, V of the block read) and sliding layers ONE.
    public static func attentionInputLayer(forQueryIndex index: Int) -> Int { index + 1 }

    /// `attn_res.mlp_queries.N` belongs to text layer `N`: every layer reads its
    /// MLP input, so there are 24, each one query.
    public static func mlpInputLayer(forQueryIndex index: Int) -> Int { index }

    /// Queries one attention-input read carries, by layer kind.
    public static func attentionInputQueryCount(isFullAttention: Bool) -> Int {
        isFullAttention ? 3 : 1
    }

    /// The four logical cross-pass messages of core layer `i`, in the
    /// reference's own order `(query, key, value, mlp)`.
    ///
    /// The checkpoint stores them in TWO 24-row tensors: rows `2i` and `2i + 1`
    /// of `middle_cycle_cross_pass_queries` are the Q and MLP queries, and the
    /// corresponding rows of `middle_cycle_cross_pass_kv_queries` are K and V.
    public struct CrossPassQueryRows: Sendable, Equatable {
        public let query: Int
        public let key: Int
        public let value: Int
        public let mlp: Int

        public init(coreLayer: Int) {
            let row = 2 * coreLayer
            self.query = row
            self.key = row
            self.value = row + 1
            self.mlp = row + 1
        }

        /// The row each logical message reads, in `(query, key, value, mlp)` order.
        public var rows: [Int] { [query, key, value, mlp] }

        /// Which tensor each logical message reads, in the same order: true for
        /// `middle_cycle_cross_pass_queries`, false for the `_kv_` one. Q and MLP
        /// come from the first, K and V from the second.
        public var fromMainTensor: [Bool] { [true, false, false, true] }
    }
}

// MARK: - Completeness

/// Why a checkpoint cannot be run, phrased for the load log.
nonisolated public enum Inkling2WeightError: Error, Equatable, CustomStringConvertible {
    case unconsumedTensors([String])
    case missingRequired([String])
    case unsupportedDtypes([String])
    case shapeMismatch([String])
    case quantCompanionMismatch([String])
    case duplicateRoles([String])

    public var description: String {
        switch self {
        case .unconsumedTensors(let names):
            let shown = names.prefix(6).joined(separator: ", ")
            let more = names.count > 6 ? " (+\(names.count - 6) more)" : ""
            return "Inkling 2 checkpoint carries \(names.count) tensor(s) this engine never reads: \(shown)\(more)"
        case .missingRequired(let names):
            let shown = names.prefix(8).joined(separator: ", ")
            let more = names.count > 8 ? " (+\(names.count - 8) more)" : ""
            return "Inkling 2 checkpoint is missing \(names.count) required tensor(s): \(shown)\(more)"
        case .unsupportedDtypes(let names):
            return "Inkling 2 checkpoint uses dtypes this engine has no kernel for: \(names.prefix(6).joined(separator: ", "))"
        case .shapeMismatch(let details):
            return "Inkling 2 checkpoint shapes disagree with its config: \(details.prefix(4).joined(separator: "; "))"
        case .quantCompanionMismatch(let details):
            return "Inkling 2 expert quantization metadata is inconsistent: \(details.prefix(4).joined(separator: "; "))"
        case .duplicateRoles(let details):
            return "Inkling 2 checkpoint has two tensors for one role: \(details.prefix(4).joined(separator: "; "))"
        }
    }
}

extension Inkling2WeightCatalog {

    /// Every requirement the config imposes on the checkpoint, checked against
    /// the catalog. Returns human-readable problems; empty means it can be run.
    public func validate(config: Inkling2Config) -> [String] {
        var problems: [String] = []
        problems.append(contentsOf: duplicateRoles.map { "two tensors claim one role: \($0)" })
        problems.append(contentsOf: outOfRangeLayers)
        problems.append(contentsOf: unconsumed.map { "unconsumed tensor: \($0)" })

        // Model level.
        for role in [Inkling2TensorRole.tokenEmbedding, .embeddingNorm, .finalNorm, .lmHead] {
            if tensor(role) == nil {
                problems.append("missing \(role.rawValue) (\(checkpointLabel(role)))")
            }
        }
        // The unembedding is padded to vocab_size; the embedding table is
        // indexed by input ids and is the same width.
        for (role, label) in [(Inkling2TensorRole.tokenEmbedding, "embed.weight"),
                              (.lmHead, "unembed.weight")] {
            if let ref = tensor(role), ref.shape.count == 2, ref.shape[0] != config.vocabSize {
                problems.append("\(label) has \(ref.shape[0]) rows for vocab_size \(config.vocabSize)")
            }
        }

        // Attention-residual queries, one per layer (attention read from layer 1).
        for layer in 1..<config.numHiddenLayers where tensor(.attnResAttnQuery, layer: layer - 1) == nil {
            problems.append("missing attn_res.attn_queries.\(layer - 1) for layer \(layer)")
        }
        for layer in 0..<config.numHiddenLayers where tensor(.attnResMlpQuery, layer: layer) == nil {
            problems.append("missing attn_res.mlp_queries.\(layer)")
        }
        if config.numPasses > 1 {
            for role in [Inkling2TensorRole.attnResOutputQuery, .crossPassQuery, .crossPassKVQuery]
            where tensor(role) == nil {
                problems.append("missing \(checkpointLabel(role))")
            }
        }
        // The published queries are [layers_with_attention_input][hidden] with
        // three rows on full layers; the row count is the check that catches a
        // swapped mapping.
        for layer in 1..<config.numHiddenLayers {
            let ref = tensor(.attnResAttnQuery, layer: layer - 1)
            let expectedRows = Inkling2QueryUnpacking.attentionInputQueryCount(
                isFullAttention: !config.isLocalLayer(layer)
            )
            // A single query is stored as a 1-D vector, a three-query read as [3, hidden].
            if let ref, (ref.shape.count == 1 ? 1 : ref.shape[0]) != expectedRows {
                problems.append(
                    "attn_res.attn_queries.\(layer - 1) is \(ref.shape), expected \(expectedRows) row(s): layer \(layer) is \(config.isLocalLayer(layer) ? "sliding" : "full") attention"
                )
            }
        }
        if let ref = tensor(.crossPassQuery), ref.shape.count == 2,
           ref.shape[0] != 2 * config.coreLayerCount {
            problems.append(
                "middle_cycle_cross_pass_queries is \(ref.shape), expected \(2 * config.coreLayerCount) rows (two per core layer)"
            )
        }
        if let ref = tensor(.crossPassKVQuery), ref.shape.count == 2,
           ref.shape[0] != 2 * config.coreLayerCount {
            problems.append(
                "middle_cycle_cross_pass_kv_queries is \(ref.shape), expected \(2 * config.coreLayerCount) rows (two per core layer)"
            )
        }

        // Per-layer requirements.
        for layer in 0..<config.numHiddenLayers {
            let isLocal = config.isLocalLayer(layer)
            let isCore = layer >= config.loopCoreStart && layer < config.loopCoreEnd
            var required: [Inkling2TensorRole] = [
                .attnWqDu, .attnWkDv, .attnWvDv, .attnWrDu, .attnWoUd, .attnGate,
                .attnSinkProj, .attnRelLogitsProj, .attnQL2ClipScale, .attnRelL2ClipScale,
                .attnSconv, .mlpSconv, .mlpNorm, .routerWeight, .routerBias
            ]
            if isLocal {
                required.append(contentsOf: [.attnNorm, .attnKSconvWeight, .attnVSconvWeight])
            } else {
                required.append(contentsOf: [
                    .qInputNorm, .kInputNorm, .vInputNorm,
                    .attnKSconvStatic, .attnVSconvStatic, .attnKSconvProj, .attnVSconvProj
                ])
            }
            if isCore {
                required.append(contentsOf: [
                    .controllerAttnDown, .controllerAttnForget, .controllerAttnGain,
                    .controllerKDown, .controllerKForget, .controllerKGain,
                    .controllerVDown, .controllerVForget, .controllerVGain,
                    .controllerMlpDown, .controllerMlpForget, .controllerMlpGain
                ])
            }
            for role in required where tensor(role, layer: layer) == nil {
                problems.append("layers.\(layer): missing \(checkpointLabel(role))")
            }

            // The split must be exact in BOTH directions: a sliding layer
            // carrying a full-only tensor is as wrong as the reverse, and
            // would mean the layer-kind rule disagrees with the checkpoint.
            for role in [Inkling2TensorRole.attnNorm, .attnKSconvWeight, .attnVSconvWeight] {
                if !isLocal, tensor(role, layer: layer) != nil {
                    problems.append("layers.\(layer): full-attention layer publishes \(checkpointLabel(role))")
                }
            }
            for role in [Inkling2TensorRole.qInputNorm, .kInputNorm, .vInputNorm,
                         .attnKSconvStatic, .attnVSconvStatic, .attnKSconvProj, .attnVSconvProj] {
                if isLocal, tensor(role, layer: layer) != nil {
                    problems.append("layers.\(layer): sliding layer publishes \(checkpointLabel(role))")
                }
            }
            if !isCore {
                for role in Inkling2TensorRole.allCases where role.isCrossPassController {
                    if tensor(role, layer: layer) != nil {
                        problems.append("layers.\(layer): non-core layer publishes \(checkpointLabel(role))")
                        break
                    }
                }
            }
            problems.append(contentsOf: expertProblems(config: config, layer: layer))
        }

        // Vision: the tower is either complete or refused. The markers are one
        // added token (`<|unused_200054|>`) whose embedding lives in the main
        // embedding table, so the tower's own files are the check.
        if let vision = config.vision {
            let required: [Inkling2TensorRole] = [
                .visionClassEmbedding, .visionClsPositionEmbedding, .visionPatchEmbedding,
                .visionPositionEmbedding, .visionNormPreWeight, .visionNormPreBias,
                .visionProjIn, .visionProjOut
            ]
            for role in required where tensor(role, scope: .vision) == nil {
                problems.append("vision: missing \(role.rawValue)")
            }
            for block in 0..<vision.numHiddenLayers {
                for role in [Inkling2TensorRole.visionBlockLn1Weight, .visionBlockLn1Bias,
                             .visionBlockLn2Weight, .visionBlockLn2Bias,
                             .visionBlockScale1, .visionBlockScale2,
                             .visionBlockQProjWeight, .visionBlockQProjBias,
                             .visionBlockKProjWeight, .visionBlockKProjBias,
                             .visionBlockVProjWeight, .visionBlockVProjBias,
                             .visionBlockOutProjWeight, .visionBlockOutProjBias,
                             .visionBlockFcWeight, .visionBlockFcBias,
                             .visionBlockProjWeight, .visionBlockProjBias]
                where tensor(role, scope: .vision, layer: block) == nil {
                    problems.append("vision block \(block): missing \(role.rawValue)")
                }
            }
        } else if !slots.filter({ $0.scope == .vision }).isEmpty {
            problems.append(
                "vision: the snapshot ships tower tensors but config.json declares no vision_config"
            )
        }

        // DSpark sidecar: only what the config declares.
        if let mtp = config.mtp {
            for layer in 0..<mtp.numHiddenLayers {
                for role in [Inkling2TensorRole.draftInputNorm, .draftPostAttentionNorm,
                             .draftDflareNorm, .draftQProj, .draftKProj, .draftVProj,
                             .draftOProj, .draftQNorm, .draftKNorm, .draftKProjContext,
                             .draftVProjContext, .draftKNormContext, .draftGateProj,
                             .draftUpProj, .draftDownProj]
                where tensor(role, scope: .sidecar, layer: layer) == nil {
                    problems.append("mtp block \(layer): missing \(role.rawValue)")
                }
            }
            for role in [Inkling2TensorRole.draftFuseWeight, .draftMaskEmbed, .draftFinalNorm,
                         .draftMarkovDflareNorm, .draftMarkovHiddenOutWeight,
                         .draftMarkovHiddenOutBias, .draftMarkovW1,
                         .draftMarkovRnnProjWeight, .draftMarkovRnnProjBias]
            where tensor(role, scope: .sidecar) == nil {
                problems.append("mtp: missing \(role.rawValue)")
            }
        } else if !slots.filter({ $0.scope == .sidecar }).isEmpty {
            problems.append("mtp: the sidecar is present but config.json declares no mtp_config")
        }

        // Dtypes.
        let unsupported = slots.filter { !$0.ref.isSupportedDtype }
            .map { "\($0.ref.name) (\($0.ref.dtype))" }
        problems.append(contentsOf: unsupported.map { "unsupported dtype: \($0)" })
        return problems
    }

    /// One layer's expert banks: shapes, scale companions and quant metadata,
    /// all inferred from the tensors rather than from a parameter-group label.
    func expertProblems(config: Inkling2Config, layer: Int) -> [String] {
        var problems: [String] = []
        guard let w13 = tensor(.expertW13, layer: layer),
              let w13Scale = tensor(.expertW13Scale, layer: layer),
              let w2 = tensor(.expertW2, layer: layer),
              let w2Scale = tensor(.expertW2Scale, layer: layer) else {
            return ["layers.\(layer): expert banks are incomplete"]
        }
        guard w13.shape.count == 3, w2.shape.count == 3,
              w13Scale.shape.count == 3, w2Scale.shape.count == 3 else {
            return ["layers.\(layer): expert banks are not 3D"]
        }
        let experts = w13.shape[0]
        if experts != config.numRoutedExperts {
            problems.append("layers.\(layer): \(experts) routed experts, config says \(config.numRoutedExperts)")
        }
        let inter = config.moeIntermediateSize
        let hidden = config.hiddenSize
        // w13's logical output is gate and up interleaved: 2 * intermediate.
        if w13.shape[1] != 2 * inter {
            problems.append("layers.\(layer): w13 output \(w13.shape[1]) != 2 x intermediate \(2 * inter)")
        }
        if w2.shape[1] != hidden || w2.shape[2] < 1 {
            problems.append("layers.\(layer): w2 is \(w2.shape), expected [\(experts), \(hidden), ...]")
        }
        let layout = Inkling2ExpertQuantLayout.infer(
            w13: w13, w13Scale: w13Scale, w2: w2, w2Scale: w2Scale, config: config
        )
        switch layout {
        case .failure(let reason):
            problems.append("layers.\(layer): \(reason)")
        case .success(let quant):
            if w13.shape[2] != quant.packedInputWidth(hidden) {
                problems.append(
                    "layers.\(layer): w13 input stride \(w13.shape[2]) != \(quant.packedInputWidth(hidden)) for \(quant.rawValue)"
                )
            }
            if w2.shape[2] != quant.packedInputWidth(inter) {
                problems.append(
                    "layers.\(layer): w2 input stride \(w2.shape[2]) != \(quant.packedInputWidth(inter)) for \(quant.rawValue)"
                )
            }
            if w13Scale.shape[2] != quant.scaleInputWidth(hidden) {
                problems.append(
                    "layers.\(layer): w13 scale stride \(w13Scale.shape[2]) != \(quant.scaleInputWidth(hidden)) for \(quant.rawValue)"
                )
            }
            if w2Scale.shape[2] != quant.scaleInputWidth(inter) {
                problems.append(
                    "layers.\(layer): w2 scale stride \(w2Scale.shape[2]) != \(quant.scaleInputWidth(inter)) for \(quant.rawValue)"
                )
            }
            let companions = [
                Inkling2TensorRole.expertW13Amax, .expertW13OuterScaleDenominator,
                .expertW2Amax, .expertW2OuterScaleDenominator
            ]
            for role in companions {
                let present = tensor(role, layer: layer) != nil
                if quant.requiresAmaxDenominator && !present {
                    problems.append("layers.\(layer): \(quant.rawValue) expert is missing \(checkpointLabel(role))")
                }
                if !quant.requiresAmaxDenominator && present {
                    problems.append("layers.\(layer): \(quant.rawValue) expert publishes \(checkpointLabel(role))")
                }
            }
            if let amax = tensor(.expertW13Amax, layer: layer), amax.shape != [experts] {
                problems.append("layers.\(layer): w13 amax is \(amax.shape), expected [\(experts)]")
            }
            if let denominator = tensor(.expertW13OuterScaleDenominator, layer: layer),
               denominator.shape != [1], denominator.elementCount != 1 {
                problems.append("layers.\(layer): w13 outer scale denominator is \(denominator.shape), expected [1]")
            }
        }
        return problems
    }

    func checkpointLabel(_ role: Inkling2TensorRole) -> String {
        switch role {
        case .tokenEmbedding: return "model.llm.embed.weight"
        case .embeddingNorm: return "model.llm.embed_norm.weight"
        case .finalNorm: return "model.llm.norm.weight"
        case .lmHead: return "model.llm.unembed.weight"
        case .attnResAttnQuery: return "model.llm.attn_res.attn_queries.N"
        case .attnResMlpQuery: return "model.llm.attn_res.mlp_queries.N"
        case .attnResOutputQuery: return "model.llm.attn_res.output_query"
        case .crossPassQuery: return "model.llm.middle_cycle_cross_pass_queries"
        case .crossPassKVQuery: return "model.llm.middle_cycle_cross_pass_kv_queries"
        case .attnNorm: return "attn_norm.weight"
        case .qInputNorm: return "q_input_norm.weight"
        case .kInputNorm: return "k_input_norm.weight"
        case .vInputNorm: return "v_input_norm.weight"
        case .attnWqDu: return "attn.wq_du.weight"
        case .attnWkDv: return "attn.wk_dv.weight"
        case .attnWvDv: return "attn.wv_dv.weight"
        case .attnWrDu: return "attn.wr_du.weight"
        case .attnWoUd: return "attn.wo_ud.weight"
        case .attnGate: return "attn.gate_attn.weight"
        case .attnSinkProj: return "attn.attn_sink_proj.weight"
        case .attnRelLogitsProj: return "attn.rel_logits_proj.proj"
        case .attnQL2ClipScale: return "attn.q_l2_clip.scale"
        case .attnRelL2ClipScale: return "attn.rel_l2_clip.scale"
        case .attnKSconvWeight: return "attn.k_sconv.weight"
        case .attnKSconvStatic: return "attn.k_sconv.static_w"
        case .attnKSconvProj: return "attn.k_sconv.weight_proj"
        case .attnVSconvWeight: return "attn.v_sconv.weight"
        case .attnVSconvStatic: return "attn.v_sconv.static_w"
        case .attnVSconvProj: return "attn.v_sconv.weight_proj"
        case .attnSconv: return "attn_sconv.weight"
        case .mlpSconv: return "mlp_sconv.weight"
        case .mlpNorm: return "mlp_norm.weight"
        case .routerWeight: return "mlp.gate.weight"
        case .routerBias: return "mlp.gate.bias"
        case .expertW13: return "mlp.experts.w13_weight"
        case .expertW13Scale: return "mlp.experts.w13_weight_scale"
        case .expertW13Amax: return "mlp.experts.w13_weight_amax"
        case .expertW13OuterScaleDenominator: return "mlp.experts.w13_weight_outer_scale_denominator"
        case .expertW2: return "mlp.experts.w2_weight"
        case .expertW2Scale: return "mlp.experts.w2_weight_scale"
        case .expertW2Amax: return "mlp.experts.w2_weight_amax"
        case .expertW2OuterScaleDenominator: return "mlp.experts.w2_weight_outer_scale_denominator"
        case .controllerAttnDown: return "controllers.attn.down_weight"
        case .controllerAttnForget: return "controllers.attn.forget_weight"
        case .controllerAttnGain: return "controllers.attn.gain_weight"
        case .controllerKDown: return "controllers.k.down_weight"
        case .controllerKForget: return "controllers.k.forget_weight"
        case .controllerKGain: return "controllers.k.gain_weight"
        case .controllerVDown: return "controllers.v.down_weight"
        case .controllerVForget: return "controllers.v.forget_weight"
        case .controllerVGain: return "controllers.v.gain_weight"
        case .controllerMlpDown: return "controllers.mlp.down_weight"
        case .controllerMlpForget: return "controllers.mlp.forget_weight"
        case .controllerMlpGain: return "controllers.mlp.gain_weight"
        default: return role.rawValue
        }
    }
}

// MARK: - Expert quantization layout

/// Which packed arithmetic one layer's expert banks are stored in, INFERRED
/// from the tensors. `hf_quant_config.json` groups them by name, but the plan's
/// rule is that shapes and quant metadata are authoritative: the MXFP8 group's
/// declared `weight_block_shape` is [32, 32] while its stored scales are one per
/// output row per 32 inputs, so the declaration alone would mis-size the read.
nonisolated public enum Inkling2ExpertQuantLayout: String, Sendable, Equatable {
    /// E2M1 values, two per byte, with E4M3 scales every 16 inputs, a per-expert
    /// FP32 amax and one outer-scale denominator.
    case nvfp4
    /// E4M3 values with E8M0 scales every 32 inputs.
    case mxfp8

    public var blockSize: Int {
        switch self {
        case .nvfp4: return 16
        case .mxfp8: return 32
        }
    }

    public var valuesPerByte: Int {
        switch self {
        case .nvfp4: return 2
        case .mxfp8: return 1
        }
    }

    public var scaleFormat: Inkling2ElementFormat {
        switch self {
        case .nvfp4: return .e4m3
        case .mxfp8: return .e8m0
        }
    }

    public var valueFormat: Inkling2ElementFormat {
        switch self {
        case .nvfp4: return .e2m1
        case .mxfp8: return .e4m3
        }
    }

    public var requiresAmaxDenominator: Bool { self == .nvfp4 }

    /// Bytes one logical input row occupies, for `inputWidth` logical values.
    public func packedInputWidth(_ inputWidth: Int) -> Int {
        (inputWidth + valuesPerByte - 1) / valuesPerByte
    }

    /// Scale entries one logical input row carries.
    public func scaleInputWidth(_ inputWidth: Int) -> Int {
        (inputWidth + blockSize - 1) / blockSize
    }

    /// Infers the layout from storage dtypes and shapes. The two layouts are
    /// told apart by what only one of them has: a U8 VALUE bank with an E4M3
    /// scale is NVFP4; an E4M3 value bank with a U8 scale is MXFP8.
    public static func infer(
        w13: Inkling2TensorRef,
        w13Scale: Inkling2TensorRef,
        w2: Inkling2TensorRef,
        w2Scale: Inkling2TensorRef,
        config: Inkling2Config
    ) -> Result<Inkling2ExpertQuantLayout, Inkling2WeightError> {
        func isU8(_ ref: Inkling2TensorRef) -> Bool { ref.dtype.uppercased() == "U8" }
        func isE4M3(_ ref: Inkling2TensorRef) -> Bool { ref.dtype.uppercased().contains("F8_E4M3") }

        let layout: Inkling2ExpertQuantLayout
        if isU8(w13), isU8(w2), isE4M3(w13Scale), isE4M3(w2Scale) {
            layout = .nvfp4
        } else if isE4M3(w13), isE4M3(w2), isU8(w13Scale), isU8(w2Scale) {
            layout = .mxfp8
        } else {
            return .failure(.quantCompanionMismatch([
                "expert banks are \(w13.dtype)/\(w13Scale.dtype) and \(w2.dtype)/\(w2Scale.dtype), "
                    + "which is neither NVFP4 (U8 values, E4M3 scales) nor MXFP8 (E4M3 values, U8 scales)"
            ]))
        }
        return .success(layout)
    }
}

// MARK: - Packed expert arithmetic (host reference)

/// One expert's stored bank: the packed weight, its block scales and (NVFP4
/// only) the per-expert amax and outer-scale denominator.
nonisolated public struct Inkling2PackedExpertMatrix: Sendable, Equatable {
    public let layout: Inkling2ExpertQuantLayout
    /// Raw stored bytes, row-major `[rows][packedColumns]`.
    public let weight: [UInt8]
    public let weightRowStride: Int
    /// Raw scale bytes, row-major `[rows][scaleColumns]`.
    public let scales: [UInt8]
    public let scaleRowStride: Int
    public let rows: Int
    public let logicalColumns: Int
    /// NVFP4 only: the expert's amax, folded in as `blockScale * (amax / denominator)`.
    public let expertAmax: Float
    public let outerScaleDenominator: Float

    public init(
        layout: Inkling2ExpertQuantLayout,
        weight: [UInt8],
        weightRowStride: Int,
        scales: [UInt8],
        scaleRowStride: Int,
        rows: Int,
        logicalColumns: Int,
        expertAmax: Float = 1,
        outerScaleDenominator: Float = 1
    ) {
        self.layout = layout
        self.weight = weight
        self.weightRowStride = weightRowStride
        self.scales = scales
        self.scaleRowStride = scaleRowStride
        self.rows = rows
        self.logicalColumns = logicalColumns
        self.expertAmax = expertAmax
        self.outerScaleDenominator = outerScaleDenominator
    }

    /// The dequantized value at (row, column), in FP32, before any BF16
    /// rounding: `code * blockScale * outerScale`.
    public func value(row: Int, column: Int) -> Float {
        guard row >= 0, row < rows, column >= 0, column < logicalColumns else { return 0 }
        let block = column / layout.blockSize
        guard block < scaleRowStride else { return 0 }
        let scale = Inkling2Element.decode(scales[row * scaleRowStride + block], as: layout.scaleFormat)
        let outerScale = layout.requiresAmaxDenominator && outerScaleDenominator > 0
            ? expertAmax / outerScaleDenominator
            : 1
        switch layout {
        case .nvfp4:
            let byte = weight[row * weightRowStride + column / 2]
            let nibble = column % 2 == 0 ? byte & 0xF : byte >> 4
            return Inkling2Element.decodeE2M1(nibble) * scale * outerScale
        case .mxfp8:
            return Inkling2Element.decodeE4M3(weight[row * weightRowStride + column]) * scale
        }
    }

    /// One logical row, dequantized and rounded to BF16 -- the HF-reference
    /// profile's view of the weight. Rounding is to the BF16 grid (eight
    /// explicit mantissa bits) because Transformers dequantizes to BF16 before
    /// an ordinary BF16 projection.
    public func bf16Row(_ row: Int) -> [Float] {
        (0..<logicalColumns).map { Inkling2Element.roundToBF16(value(row: row, column: $0)) }
    }

    /// `y = W @ x` for one output row, in the HF-reference profile.
    ///
    /// The block scale is decoded once per `blockSize` columns rather than per
    /// column: 32 consecutive values share one scale byte, and decoding it per
    /// column was the inner loop of every expert projection.
    public func project(row: Int, input: [Float]) -> Float {
        guard row >= 0, row < rows else { return 0 }
        let count = min(input.count, logicalColumns)
        var total: Float = 0
        let outerScale = layout.requiresAmaxDenominator && outerScaleDenominator > 0
            ? expertAmax / outerScaleDenominator
            : 1
        let weightBase = row * weightRowStride
        let scaleBase = row * scaleRowStride
        var column = 0
        while column < count {
            let block = column / layout.blockSize
            let scale = block < scaleRowStride
                ? Inkling2Element.decode(scales[scaleBase + block], as: layout.scaleFormat) * outerScale
                : 0
            let end = min(count, (block + 1) * layout.blockSize)
            switch layout {
            case .nvfp4:
                while column < end {
                    let byte = weight[weightBase + column / 2]
                    let nibble = column % 2 == 0 ? byte & 0xF : byte >> 4
                    total += Inkling2Element.roundToBF16(Inkling2Element.decodeE2M1(nibble) * scale) * input[column]
                    column += 1
                }
            case .mxfp8:
                while column < end {
                    let value = Inkling2Element.decodeE4M3(weight[weightBase + column])
                    total += Inkling2Element.roundToBF16(value * scale) * input[column]
                    column += 1
                }
            }
        }
        return total
    }

    /// The fused `w13` bank interleaves gate and up along its output dimension:
    /// even raw rows are gate, odd raw rows are up. The two logical matrices are
    /// therefore read by parity, never as two contiguous halves.
    public func gateUpRow(_ index: Int) -> (gate: Int, up: Int) {
        (gate: 2 * index, up: 2 * index + 1)
    }
}
