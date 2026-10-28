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
//       tensor each, and the Engram tables (`engram.wkv`) are ~100 GB single
//       tensors that must never be handed to Metal.
//
//  The catalog therefore does two jobs:
//    * resolve every name to a (role, layer, expert) slot with tolerant
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
    case compressorWkv
    case compressorWgate

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
    case engramQuery
    case engramKey

    // DSpark draft block
    case draftMainProj
    case draftMainNorm
    case draftNorm
    case draftMarkovEmbed
    case draftMarkovHead
    case draftConfidence
    case draftExpertW1
    case draftExpertW2
    case draftExpertW3

    /// Roles whose resolved tensor is streamed from disk and never mapped into
    /// a Metal buffer (the Engram table is one ~100 GB tensor).
    public var isStreamed: Bool { self == .engramTable }

    /// Roles that exist only when the plan says the layer has a compressor.
    public var requiresCompressor: Bool {
        switch self {
        case .compressorWkv, .compressorWgate, .indexerWqB, .indexerWk, .indexerWeights:
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

/// Where a tensor lives in the model: a role, optionally per layer and per
/// expert, plus whether it is the paired quantization-scale companion.
nonisolated public struct DeepSeekV41TensorSlot: Sendable, Equatable {
    public let role: DeepSeekV41TensorRole
    /// Backbone layer, or the draft block index for `draft*` roles, or nil for
    /// model-level tensors.
    public let layer: Int?
    public let expert: Int?
    public let isScale: Bool
    public let ref: DeepSeekV41TensorRef

    public init(
        role: DeepSeekV41TensorRole,
        layer: Int?,
        expert: Int?,
        isScale: Bool,
        ref: DeepSeekV41TensorRef
    ) {
        self.role = role
        self.layer = layer
        self.expert = expert
        self.isScale = isScale
        self.ref = ref
    }

    public var key: String {
        "\(role.rawValue)|\(layer.map(String.init) ?? "-")|\(expert.map(String.init) ?? "-")|\(isScale ? "s" : "w")"
    }
}

// MARK: - Catalog

nonisolated public struct DeepSeekV41WeightCatalog: Sendable {

    public let slots: [DeepSeekV41TensorSlot]
    /// Names the catalog could not place. The load refuses when this is not empty.
    public let unconsumed: [String]

    private let index: [String: DeepSeekV41TensorSlot]

    public init(named tensors: [DeepSeekV41TensorRef], layerCount: Int) {
        var slots: [DeepSeekV41TensorSlot] = []
        var unconsumed: [String] = []
        var index: [String: DeepSeekV41TensorSlot] = [:]
        for tensor in tensors {
            guard let slot = Self.classify(name: tensor.name, ref: tensor) else {
                unconsumed.append(tensor.name)
                continue
            }
            slots.append(slot)
            index[slot.key] = slot
        }
        self.slots = slots
        self.unconsumed = unconsumed
        self.index = index
    }

    public func tensor(
        _ role: DeepSeekV41TensorRole,
        layer: Int? = nil,
        expert: Int? = nil,
        scale: Bool = false
    ) -> DeepSeekV41TensorRef? {
        let key = "\(role.rawValue)|\(layer.map(String.init) ?? "-")|\(expert.map(String.init) ?? "-")|\(scale ? "s" : "w")"
        return index[key]?.ref
    }

    /// Looks a slot up by its `role|layer|expert|w|s` key. Used by the device-side
    /// resolver, which has already turned a role into a slot and needs the tensor
    /// behind it.
    public func slot(matchingKey key: String) -> DeepSeekV41TensorSlot? {
        index[key]
    }

    /// Per-expert tensors that were published for a layer, ascending.
    public func expertIndices(layer: Int) -> [Int] {
        Set(slots.filter { $0.role == .expertW1 && $0.layer == layer }.compactMap(\.expert)).sorted()
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

        // Draft block (`mtp.N.…`).
        if let rest = remainder(of: name, prefix: "mtp.") {
            let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, let block = Int(parts[0]) else { return nil }
            return classifyDraft(String(parts[1]), block: block, ref: ref)
        }

        // Backbone layer.
        guard let rest = remainder(of: name, prefix: "layers.") else { return nil }
        let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, let layer = Int(parts[0]) else { return nil }
        let body = String(parts[1])
        let (base, isScale) = splitScale(body)

        if let expertRest = remainder(of: base, prefix: "ffn.experts.") {
            let expertParts = expertRest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard expertParts.count == 2, let expert = Int(expertParts[0]) else { return nil }
            let role = expertRole(String(expertParts[1]))
            guard let role else { return nil }
            return slot(role, layer: layer, expert: expert, scale: isScale, ref)
        }

        let roles: [String: DeepSeekV41TensorRole] = [
            "attn_norm": .attnNorm,
            "ffn_norm": .ffnNorm,
            "attn.wq_a": .attnWqA,
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
            "attn.compressor.wkv": .compressorWkv,
            "attn.compressor.wgate": .compressorWgate,
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
            "engram.wkv": .engramTable,
            "engram.wkv.scale": .engramTable,
            "engram.q_weight": .engramQuery,
            "engram.k_weight": .engramKey
        ]
        if let role = roles[base] {
            // Norms carry no scale companion; only quantized linears do.
            return slot(role, layer: layer, expert: nil, scale: isScale, ref)
        }
        return nil
    }

    private static func classifyDraft(
        _ body: String,
        block: Int,
        ref: DeepSeekV41TensorRef
    ) -> DeepSeekV41TensorSlot? {
        let (base, isScale) = splitScale(body)
        if let expertRest = remainder(of: base, prefix: "ffn.experts.") {
            let expertParts = expertRest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard expertParts.count == 2, let expert = Int(expertParts[0]) else { return nil }
            guard let role = expertRole(String(expertParts[1])) else { return nil }
            let mapped: DeepSeekV41TensorRole
            switch role {
            case .expertW1: mapped = .draftExpertW1
            case .expertW2: mapped = .draftExpertW2
            default: mapped = .draftExpertW3
            }
            return slot(mapped, layer: block, expert: expert, scale: isScale, ref)
        }
        let roles: [String: DeepSeekV41TensorRole] = [
            "main_proj": .draftMainProj,
            "main_norm": .draftMainNorm,
            "norm": .draftNorm,
            "markov_head.embed": .draftMarkovEmbed,
            "markov_head.head": .draftMarkovHead,
            "confidence_head.proj": .draftConfidence
        ]
        guard let role = roles[base] else { return nil }
        return slot(role, layer: block, expert: nil, scale: isScale, ref)
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
        layer: Int?,
        expert: Int?,
        scale: Bool,
        _ ref: DeepSeekV41TensorRef
    ) -> DeepSeekV41TensorSlot {
        DeepSeekV41TensorSlot(role: role, layer: layer, expert: expert, isScale: scale, ref: ref)
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

    /// Every requirement the plan imposes on the checkpoint, checked against the
    /// catalog. Returns human-readable problems; empty means the checkpoint can
    /// be run.
    public func validate(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan]
    ) -> [String] {
        var problems: [String] = []

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
                (.routerWeight, "ffn.gate.weight"),
                (.sharedExpertW1, "ffn.shared_experts.w1.weight"),
                (.sharedExpertW2, "ffn.shared_experts.w2.weight"),
                (.sharedExpertW3, "ffn.shared_experts.w3.weight")
            ]
            if config.qLoraRank > 0 {
                required.append((.attnWqANorm, "attn.wq_a_norm.weight"))
            }
            required.append((.attnWqB, "attn.wq_b.weight"))
            if config.kvLoraRank > 0 {
                required.append((.attnKvNorm, "attn.kv_norm.weight"))
            }
            if config.hcMult > 1 {
                required.append(contentsOf: [
                    (.hcAttnFn, "hc_attn_fn"),
                    (.hcAttnBase, "hc_attn_base"),
                    (.hcFfnFn, "hc_ffn_fn"),
                    (.hcFfnBase, "hc_ffn_base")
                ])
            }
            if layer.flavor == .compressed {
                required.append(contentsOf: [
                    (.compressorWkv, "attn.compressor.wkv.weight"),
                    (.compressorWgate, "attn.compressor.wgate.weight")
                ])
                if layer.ownsCompressedIndex || config.indexTopK > 0 {
                    required.append(contentsOf: [
                        (.indexerWqB, "attn.indexer.wq_b.weight"),
                        (.indexerWk, "attn.indexer.wk.weight"),
                        (.indexerWeights, "attn.indexer.weights_proj.weight")
                    ])
                }
            }
            if layer.engramTable != nil {
                required.append(contentsOf: [
                    (.engramTable, "engram.wkv"),
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

        // DSpark draft blocks.
        for block in 0..<config.numNextNPredictLayers {
            for (role, label) in [(DeepSeekV41TensorRole.draftMainProj, "main_proj.weight"),
                                  (.draftMainNorm, "main_norm.weight"),
                                  (.draftNorm, "norm.weight"),
                                  (.draftMarkovEmbed, "markov_head.embed.weight"),
                                  (.draftMarkovHead, "markov_head.head.weight"),
                                  (.draftConfidence, "confidence_head.proj.weight")] {
                if tensor(role, layer: block) == nil {
                    problems.append("mtp.\(block): missing \(label)")
                }
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
