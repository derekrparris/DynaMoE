//
//  Inkling2Config.swift
//  DynaMoE
//
//  Inkling 2 Mini ("Inkling2Loop") hyperparameters and the derived visit plan.
//
//  Phase 1 of the Inkling 2 port: pure data. No Metal, no tokenizer, no I/O
//  beyond decoding a `config.json` the caller already read. Everything the hot
//  path needs is resolved ONCE here -- the 48-visit schedule, which visits own a
//  full-attention cache, which layer's weights each visit runs, which physical
//  layers carry a cross-pass controller -- so the forward loop never re-derives
//  architecture strings per token.
//
//  Three things make this config unlike every other family in the repo:
//
//    1. It is NESTED three ways: `text_config`, `vision_config`, `mtp_config`,
//       each a full sub-config, plus a `loop_config` inside the text tower. A
//       flat read finds none of the geometry.
//    2. The model is a FIXED LOOP: 24 physical decoder layers executed as 48
//       visits (0...5 once, 6...17 three times, 18...23 once). Weights are shared
//       across the passes; KV and the four convolution states are NOT, so the
//       executor state is per VISIT while the weights are per PHYSICAL layer.
//    3. Attention has no RoPE. Positions enter through a relative-distance bias
//       bank and a logarithmic query scale on full-attention layers.
//
//  Field names are the checkpoint's own. The reference's `layer_types` is
//  derived when absent (`hybrid` every sixth layer, `hybrid_sliding` otherwise),
//  which is exactly what the published `local_layer_ids` says, so both spellings
//  decode to the same plan and a checkpoint that publishes neither still gets
//  the checkpoint's own answer.
//
//  Strictness: an unknown key anywhere in the tree is refused rather than
//  ignored. `hf_quant_config.json`'s parameter groups and this config are the
//  only description of what the checkpoint contains, and a silently-ignored key
//  is how a new architecture feature (a new tensor role, a new loop knob) turns
//  into a graph that runs at full speed and computes something else.
//

import Foundation

// MARK: - Errors

nonisolated public enum Inkling2ConfigError: Error, Equatable, CustomStringConvertible {
    case notInkling2
    case missingField(String)
    case invalidField(String, String)
    case unknownField(String, String)
    case inconsistentLayout(String)

    public var description: String {
        switch self {
        case .notInkling2:
            return "config.json does not describe an Inkling 2 model"
        case .missingField(let field):
            return "Inkling 2 config is missing '\(field)'"
        case .invalidField(let field, let detail):
            return "Inkling 2 config field '\(field)' is invalid: \(detail)"
        case .unknownField(let field, let where_):
            return "Inkling 2 config has an unknown field '\(field)' in \(where_)"
        case .inconsistentLayout(let detail):
            return "Inkling 2 config is inconsistent: \(detail)"
        }
    }
}

// MARK: - JSON access

/// Tolerant reader for HF config values, which mix `6`, `6.0`, `[1, 14]` and
/// strings freely. Mirrors `DSV41JSON` in shape so the two ports read a
/// checkpoint the same way.
nonisolated struct Inkling2JSON {
    let raw: [String: Any]

    init(_ raw: [String: Any]) { self.raw = raw }

    func has(_ key: String) -> Bool { raw[key] != nil }
    var keys: Set<String> { Set(raw.keys) }

    func int(_ key: String) -> Int? {
        guard let value = raw[key] else { return nil }
        if let i = value as? Int { return i }
        if let d = value as? Double { return Int(exactly: d) }
        if let n = value as? NSNumber {
            let wide = n.int64Value
            guard wide >= Int64(Int.min), wide <= Int64(Int.max) else { return nil }
            return Int(wide)
        }
        if let s = value as? String { return Int(s) }
        return nil
    }

    func float(_ key: String) -> Float? {
        guard let value = raw[key] else { return nil }
        if let f = value as? Float { return f }
        if let d = value as? Double { return Float(d) }
        if let n = value as? NSNumber { return n.floatValue }
        if let s = value as? String, let f = Float(s) { return f }
        return nil
    }

    func bool(_ key: String) -> Bool? {
        guard let value = raw[key] else { return nil }
        if let b = value as? Bool { return b }
        if let n = value as? NSNumber { return n.boolValue }
        return nil
    }

    func string(_ key: String) -> String? { raw[key] as? String }

    func ints(_ key: String) -> [Int]? {
        guard let value = raw[key] else { return nil }
        if let list = value as? [Int] { return list }
        if let list = value as? [Double] { return list.compactMap { Int(exactly: $0) } }
        if let list = value as? [NSNumber] {
            let converted = list.compactMap { Int(exactly: $0.doubleValue) }
            return converted.count == list.count ? converted : nil
        }
        if let single = int(key) { return [single] }
        return nil
    }

    func strings(_ key: String) -> [String]? { raw[key] as? [String] }

    func object(_ key: String) -> Inkling2JSON? {
        guard let nested = raw[key] as? [String: Any] else { return nil }
        return Inkling2JSON(nested)
    }

    /// Every key of `allowed`, checked against this object's own keys. Returns
    /// the unknown ones so a caller can refuse with a real message.
    func unknownKeys(allowing allowed: Set<String>) -> [String] {
        keys.subtracting(allowed).sorted()
    }
}

// MARK: - Visit plan

/// What one EXECUTED VISIT of the 48 does. `physicalLayer` is which weights it
/// runs; `pass` is which loop pass it belongs to (0 for the pre/loop-first/
/// post layers, 0...2 for the core's three passes); `cacheSlot` is which KV and
/// convolution-state bundle it owns, and it is unique per visit -- that is the
/// whole point of the plan.
nonisolated public struct Inkling2VisitPlan: Sendable, Equatable {
    /// 0...47: the visit's place in the schedule.
    public let visit: Int
    /// 0...23: the physical decoder layer whose weights this visit runs.
    public let physicalLayer: Int
    /// 1...3 for the core layers (the loop pass), 0 outside the looped range.
    public let pass: Int
    /// Unique per visit; the index into every per-visit state bundle.
    public let cacheSlot: Int
    /// True for the eight visits that own a growing full-attention cache.
    public let isFullAttention: Bool
    /// True when the physical layer carries cross-pass controllers (core only).
    public let hasMessageControllers: Bool
    /// The physical layer index this visit's block-attention residual queries
    /// belong to, or nil for the very first visit (layer 0 has no attention
    /// input read: `attention_input` is nil at `layer_idx == 0`).
    public let attentionInputLayer: Int?

    public init(
        visit: Int,
        physicalLayer: Int,
        pass: Int,
        cacheSlot: Int,
        isFullAttention: Bool,
        hasMessageControllers: Bool,
        attentionInputLayer: Int?
    ) {
        self.visit = visit
        self.physicalLayer = physicalLayer
        self.pass = pass
        self.cacheSlot = cacheSlot
        self.isFullAttention = isFullAttention
        self.hasMessageControllers = hasMessageControllers
        self.attentionInputLayer = attentionInputLayer
    }
}

// MARK: - Config

nonisolated public struct Inkling2Config: Sendable, Equatable {

    // Identity
    public var architectures: [String]
    public var modelType: String
    public var textModelType: String
    public var dtype: String
    public var quantizationMethod: String?

    // Text geometry
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var swaNumAttentionHeads: Int
    public var swaNumKeyValueHeads: Int
    public var swaHeadDim: Int
    public var slidingWindowSize: Int
    public var vocabSize: Int
    public var unpaddedVocabSize: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var convKernelSize: Int
    /// Number of convolution state bundles per visit: K, V, post-attention, post-MLP.
    public var numberOfConvStates: Int
    /// Whether the Q/O projections carry a bias. False on this checkpoint, and
    /// the catalog has no slot for a bias tensor, so a checkpoint declaring one
    /// is refused rather than loaded without it.
    public var qBias: Bool
    public var oBias: Bool
    /// Value normalization is gamma-free in the reference
    /// (`use_value_norm_no_gamma`); a checkpoint with a learned gamma would need
    /// a tensor this catalog does not know.
    public var useValueNormNoGamma: Bool

    // Relative attention
    public var dRel: Int
    public var relExtent: Int
    public var logScalingNFloor: Int
    public var logScalingAlpha: Float
    public var qkL2ClipThreshold: Float
    public var relL2ClipThreshold: Float

    // Layer kinds
    public var layerTypes: [String]
    public var localLayerIDs: [Int]
    public var nLayersPerAttnResBlock: Int
    public var attnResNormalizeValues: Bool

    // Loop
    public var loopCoreStart: Int
    public var loopCoreEnd: Int
    public var numPasses: Int
    public var gateRank: Int
    /// The reference's own flag. False here: the VARIABLE-loop research
    /// checkpoint is a different model, and its config must be refused rather
    /// than run with a fixed schedule.
    public var isVariableLoop: Bool

    // MoE
    public var numRoutedExperts: Int
    public var numExpertsPerToken: Int
    public var numSharedExperts: Int
    /// The routed experts' intermediate width (the reference's
    /// `moe_intermediate_size`; the config's `intermediate_size`).
    public var moeIntermediateSize: Int
    public var routeScale: Float
    public var mlpGateSoftcap: Float
    public var mlpUpSoftcap: Float
    public var scoringFunction: String
    public var useGateBias: Bool

    // Output
    public var finalLogitSoftcapping: Float

    // Tokenizer
    public var eosTokenID: Int
    public var padTokenID: Int
    public var imageTokenID: Int

    // Vision (validated here, executed in a later phase)
    public var vision: Inkling2VisionSpec?

    // DSpark sidecar (validated here, executed in a later phase)
    public var mtp: Inkling2MTPSpec?

    // MARK: Derived counts

    /// Visits executed per token: 24 physical layers + 2 extra passes of 12 core layers.
    public var executedVisitCount: Int {
        numHiddenLayers + max(0, numPasses - 1) * coreLayerCount
    }

    public var coreLayerCount: Int { max(0, loopCoreEnd - loopCoreStart) }

    /// Every visit's plan, in execution order. Throws when the loop bounds are
    /// not on block boundaries, because a loop that starts mid-block would
    /// restart the residual stack from a state that is not a block boundary.
    public func makeVisitPlan() throws -> [Inkling2VisitPlan] {
        let block = max(1, nLayersPerAttnResBlock)
        guard loopCoreStart > 0, loopCoreStart % block == 0 else {
            throw Inkling2ConfigError.invalidField(
                "loop_config.core_start", "\(loopCoreStart) must be a positive multiple of \(block)"
            )
        }
        guard loopCoreEnd % block == 0 else {
            throw Inkling2ConfigError.invalidField(
                "loop_config.core_end", "\(loopCoreEnd) must be a multiple of \(block)"
            )
        }
        guard loopCoreEnd <= numHiddenLayers else {
            throw Inkling2ConfigError.inconsistentLayout(
                "loop_config.core_end \(loopCoreEnd) is past num_hidden_layers \(numHiddenLayers)"
            )
        }
        guard numPasses >= 1 else {
            throw Inkling2ConfigError.invalidField("loop_config.num_passes", "must be at least 1")
        }
        var plan: [Inkling2VisitPlan] = []
        var slot = 0
        func append(_ layer: Int, pass: Int) {
            let inCore = layer >= loopCoreStart && layer < loopCoreEnd
            plan.append(
                Inkling2VisitPlan(
                    visit: plan.count,
                    physicalLayer: layer,
                    pass: inCore ? pass + 1 : 0,
                    cacheSlot: slot,
                    isFullAttention: !isLocalLayer(layer),
                    hasMessageControllers: inCore,
                    attentionInputLayer: layer > 0 ? layer : nil
                )
            )
            slot += 1
        }
        for layer in 0..<loopCoreStart { append(layer, pass: 0) }
        for pass in 0..<numPasses {
            for layer in loopCoreStart..<loopCoreEnd { append(layer, pass: pass) }
        }
        for layer in loopCoreEnd..<numHiddenLayers { append(layer, pass: 0) }
        return plan
    }

    /// True when a physical layer attends over a bounded window ring rather than
    /// a growing cache. `local_layer_ids` is the published authority; the
    /// derived `layer_types` default (`hybrid` every sixth layer) agrees with it,
    /// so a config that publishes only one of them still answers the same.
    public func isLocalLayer(_ layer: Int) -> Bool {
        if !localLayerIDs.isEmpty {
            return localLayerIDs.contains(layer)
        }
        if layer < layerTypes.count {
            return layerTypes[layer] == "hybrid_sliding"
        }
        let block = max(1, nLayersPerAttnResBlock)
        return (layer + 1) % block != 0
    }

    /// Physical layers that own a growing full-attention cache.
    public var fullAttentionLayers: [Int] {
        (0..<numHiddenLayers).filter { !isLocalLayer($0) }
    }

    /// The reference's derived `layer_types`, which is also what the checkpoint
    /// publishes as `local_layer_ids`.
    public var derivedLayerTypes: [String] {
        let block = max(1, nLayersPerAttnResBlock)
        return (0..<numHiddenLayers).map { (($0 + 1) % block == 0) ? "hybrid" : "hybrid_sliding" }
    }

    /// Rows per KV head per token in BF16/FP16: key and value.
    public var kvBytesPerTokenPerFullVisit: Int { 2 * headDim * 2 }

    // MARK: Loading

    /// Cheap identity probe used by discovery before the full decode runs.
    public static func identifies(_ object: [String: Any]) -> Bool {
        let root = Inkling2JSON(object)
        let text = root.object("text_config")
        let types = [root.string("model_type"), text?.string("model_type")].compactMap { $0?.lowercased() }
        let archs = (root.strings("architectures") ?? []).map { $0.lowercased() }
        return types.contains { $0.hasPrefix("inkling2") || $0.hasPrefix("inkling_2") }
            || archs.contains { $0.contains("inkling2") || $0.contains("inkling_2") }
    }

    public static func decode(jsonData: Data) throws -> Inkling2Config {
        guard let object = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw Inkling2ConfigError.invalidField("config.json", "not a JSON object")
        }
        return try decode(object: object)
    }

    public static func decode(contentsOf url: URL) throws -> Inkling2Config {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw Inkling2ConfigError.missingField("config.json (\(error.localizedDescription))")
        }
        return try decode(jsonData: data)
    }

    public static func decode(object: [String: Any]) throws -> Inkling2Config {
        guard identifies(object) else { throw Inkling2ConfigError.notInkling2 }
        let root = Inkling2JSON(object)
        try refuseUnknownKeys(in: root, allowing: Self.rootKeys, where: "config.json")
        guard let text = root.object("text_config") else {
            throw Inkling2ConfigError.missingField("text_config")
        }
        try refuseUnknownKeys(in: text, allowing: Self.textKeys, where: "text_config")

        func requireInt(_ json: Inkling2JSON, _ key: String, where where_: String) throws -> Int {
            guard let value = json.int(key) else {
                throw Inkling2ConfigError.missingField("\(where_).\(key)")
            }
            return value
        }

        let loop = text.object("loop_config") ?? Inkling2JSON([:])
        try refuseUnknownKeys(in: loop, allowing: Self.loopKeys, where: "text_config.loop_config")

        let numHiddenLayers = try requireInt(text, "num_hidden_layers", where: "text_config")
        let hiddenSize = try requireInt(text, "hidden_size", where: "text_config")
        let numAttentionHeads = try requireInt(text, "num_attention_heads", where: "text_config")
        let numKeyValueHeads = try requireInt(text, "num_key_value_heads", where: "text_config")
        let headDim = try requireInt(text, "head_dim", where: "text_config")
        let vocabSize = try requireInt(text, "vocab_size", where: "text_config")
        let blockSize = text.int("n_layers_per_attn_res_block") ?? 6
        let layerTypes = text.strings("layer_types")
            ?? (0..<numHiddenLayers).map { (($0 + 1) % max(1, blockSize) == 0) ? "hybrid" : "hybrid_sliding" }

        var visionSpec: Inkling2VisionSpec?
        if let visionJSON = root.object("vision_config") {
            visionSpec = try Inkling2VisionSpec.decode(visionJSON, textHiddenSize: hiddenSize)
        }

        var mtpSpec: Inkling2MTPSpec?
        if let mtpJSON = root.object("mtp_config") {
            mtpSpec = try Inkling2MTPSpec.decode(mtpJSON)
        }

        let config = Inkling2Config(
            architectures: root.strings("architectures") ?? ["Inkling2LoopForConditionalGeneration"],
            modelType: root.string("model_type") ?? "inkling2_loop",
            textModelType: text.string("model_type") ?? "inkling2_loop_text",
            dtype: root.string("dtype") ?? root.string("torch_dtype") ?? "bfloat16",
            quantizationMethod: root.object("quantization_config")?.string("quant_method"),
            hiddenSize: hiddenSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            swaNumAttentionHeads: text.int("swa_num_attention_heads") ?? numAttentionHeads,
            swaNumKeyValueHeads: text.int("swa_num_key_value_heads") ?? numKeyValueHeads,
            swaHeadDim: text.int("swa_head_dim") ?? headDim,
            slidingWindowSize: text.int("sliding_window_size") ?? text.int("sliding_window") ?? 0,
            vocabSize: vocabSize,
            unpaddedVocabSize: text.int("unpadded_vocab_size") ?? vocabSize,
            maxPositionEmbeddings: text.int("max_position_embeddings") ?? text.int("model_max_length") ?? 0,
            rmsNormEps: text.float("rms_norm_eps") ?? 1e-6,
            convKernelSize: text.int("conv_kernel_size") ?? text.int("sconv_kernel_size") ?? 4,
            numberOfConvStates: text.int("number_of_conv_states") ?? 4,
            qBias: text.bool("q_bias") ?? false,
            oBias: text.bool("o_bias") ?? false,
            useValueNormNoGamma: text.bool("use_value_norm_no_gamma") ?? true,
            dRel: text.int("d_rel") ?? 0,
            relExtent: text.int("rel_extent") ?? 0,
            logScalingNFloor: text.int("log_scaling_n_floor") ?? 0,
            logScalingAlpha: text.float("log_scaling_alpha") ?? 0,
            qkL2ClipThreshold: text.float("qk_l2_clip_threshold") ?? 0,
            relL2ClipThreshold: text.float("rel_l2_clip_threshold") ?? 0,
            layerTypes: layerTypes,
            localLayerIDs: text.ints("local_layer_ids") ?? [],
            nLayersPerAttnResBlock: blockSize,
            attnResNormalizeValues: text.bool("attn_res_normalize_values") ?? true,
            loopCoreStart: loop.int("core_start") ?? text.int("loop_layer_index_start") ?? 0,
            loopCoreEnd: loop.int("core_end") ?? text.int("loop_layer_index_end") ?? numHiddenLayers,
            numPasses: loop.int("num_passes") ?? text.int("loop_num_passes") ?? 1,
            gateRank: loop.int("gate_rank") ?? text.int("loop_gate_rank") ?? 0,
            isVariableLoop: loop.bool("is_varloop") ?? false,
            numRoutedExperts: text.int("n_routed_experts") ?? text.int("num_local_experts") ?? 0,
            numExpertsPerToken: text.int("num_experts_per_tok") ?? 0,
            numSharedExperts: text.int("n_shared_experts") ?? 0,
            moeIntermediateSize: text.int("intermediate_size") ?? text.int("moe_intermediate_size") ?? 0,
            routeScale: text.float("route_scale") ?? 1,
            mlpGateSoftcap: text.float("mlp_gate_softcap") ?? 4,
            mlpUpSoftcap: text.float("mlp_up_softcap") ?? 25,
            scoringFunction: text.string("scoring_func") ?? "sigmoid",
            useGateBias: text.bool("use_gate_bias") ?? true,
            finalLogitSoftcapping: text.float("final_logit_softcapping") ?? 30,
            eosTokenID: tokenID(root, "eos_token_id") ?? text.int("eos_token_id") ?? -1,
            padTokenID: tokenID(root, "pad_token_id") ?? 0,
            imageTokenID: root.int("image_token_id") ?? 200054,
            vision: visionSpec,
            mtp: mtpSpec
        )
        try config.validate()
        return config
    }

    private static func tokenID(_ json: Inkling2JSON, _ key: String) -> Int? {
        if let list = json.ints(key), let first = list.first { return first }
        return json.int(key)
    }

    static func refuseUnknownKeys(in json: Inkling2JSON, allowing allowed: Set<String>, where where_: String) throws {
        let unknown = json.unknownKeys(allowing: allowed)
        if let first = unknown.first {
            throw Inkling2ConfigError.unknownField(first, where_)
        }
    }

    /// Hard invariants: anything failing here means no plan can be built.
    func validate() throws {
        guard hiddenSize > 0 else {
            throw Inkling2ConfigError.invalidField("hidden_size", "must be positive")
        }
        guard numHiddenLayers > 0 else {
            throw Inkling2ConfigError.invalidField("num_hidden_layers", "must be positive")
        }
        guard headDim > 0, numAttentionHeads > 0, numKeyValueHeads > 0,
              numAttentionHeads % numKeyValueHeads == 0 else {
            throw Inkling2ConfigError.invalidField(
                "num_attention_heads/num_key_value_heads",
                "\(numAttentionHeads) heads over \(numKeyValueHeads) KV heads must be a positive multiple"
            )
        }
        guard swaHeadDim > 0, swaNumAttentionHeads > 0, swaNumKeyValueHeads > 0,
              swaNumAttentionHeads % swaNumKeyValueHeads == 0 else {
            throw Inkling2ConfigError.invalidField(
                "swa_num_attention_heads/swa_num_key_value_heads",
                "\(swaNumAttentionHeads) heads over \(swaNumKeyValueHeads) KV heads must be a positive multiple"
            )
        }
        guard vocabSize > 0, unpaddedVocabSize > 0, unpaddedVocabSize <= vocabSize else {
            throw Inkling2ConfigError.invalidField(
                "unpadded_vocab_size",
                "\(unpaddedVocabSize) must be in 1...\(vocabSize) (the checkpoint pads the unembedding)"
            )
        }
        guard layerTypes.count == numHiddenLayers else {
            throw Inkling2ConfigError.inconsistentLayout(
                "layer_types has \(layerTypes.count) entries for \(numHiddenLayers) layers"
            )
        }
        let known = Set(["hybrid", "hybrid_sliding", "full_attention", "sliding_attention"])
        if let bad = layerTypes.first(where: { !known.contains($0) }) {
            throw Inkling2ConfigError.invalidField("layer_types", "unknown layer type '\(bad)'")
        }
        if !localLayerIDs.isEmpty {
            if let bad = localLayerIDs.first(where: { $0 < 0 || $0 >= numHiddenLayers }) {
                throw Inkling2ConfigError.inconsistentLayout(
                    "local_layer_ids contains \(bad), outside 0..<\(numHiddenLayers)"
                )
            }
            if Set(localLayerIDs).count != localLayerIDs.count {
                throw Inkling2ConfigError.inconsistentLayout("local_layer_ids contains duplicates")
            }
        }
        guard slidingWindowSize > 0 else {
            // Every layer keeps a bounded ring or a growing cache, and a zero
            // window would make a local layer attend over nothing at all.
            throw Inkling2ConfigError.invalidField("sliding_window_size", "must be positive")
        }
        guard convKernelSize >= 2 else {
            throw Inkling2ConfigError.invalidField("conv_kernel_size", "must be at least 2")
        }
        guard !qBias, !oBias else {
            throw Inkling2ConfigError.invalidField(
                "q_bias/o_bias",
                "a biased Q or O projection is not the checkpoint's shape and has no tensor role here"
            )
        }
        guard useValueNormNoGamma else {
            throw Inkling2ConfigError.invalidField(
                "use_value_norm_no_gamma",
                "false: a learned value-norm gamma is not in this checkpoint's tensor set"
            )
        }
        guard numberOfConvStates == 4 else {
            throw Inkling2ConfigError.invalidField(
                "number_of_conv_states",
                "\(numberOfConvStates): the graph has exactly four (K, V, post-attention, post-MLP) and a different count means a different graph"
            )
        }
        // KV convolution has one tap per head dimension, so the head widths have
        // to be the kernel's multiple or a tap would read the next head.
        guard dRel > 0, relExtent > 0 else {
            throw Inkling2ConfigError.invalidField(
                "d_rel/rel_extent", "both must be positive: relative attention is this model's only position signal"
            )
        }
        guard logScalingNFloor > 0, logScalingAlpha >= 0 else {
            throw Inkling2ConfigError.invalidField(
                "log_scaling_n_floor/log_scaling_alpha", "a positive floor and a non-negative alpha"
            )
        }
        guard qkL2ClipThreshold > 0, relL2ClipThreshold > 0 else {
            throw Inkling2ConfigError.invalidField(
                "qk_l2_clip_threshold/rel_l2_clip_threshold", "both must be positive"
            )
        }
        guard numRoutedExperts > 0, numExpertsPerToken > 0, numExpertsPerToken <= numRoutedExperts else {
            throw Inkling2ConfigError.invalidField(
                "num_experts_per_tok", "must be in 1...\(numRoutedExperts)"
            )
        }
        guard numSharedExperts == 0 else {
            // The graph has no shared expert; a checkpoint that declares one is
            // a different graph and its extra tensors would be unconsumed.
            throw Inkling2ConfigError.invalidField(
                "n_shared_experts",
                "\(numSharedExperts): this checkpoint has none and the graph has no shared-expert path"
            )
        }
        guard moeIntermediateSize > 0, moeIntermediateSize % 2 == 0 else {
            // w13 interleaves gate and up along its output dimension, so the
            // logical width is even by construction.
            throw Inkling2ConfigError.invalidField(
                "intermediate_size", "\(moeIntermediateSize) must be a positive even width"
            )
        }
        guard gateRank > 0 else {
            throw Inkling2ConfigError.invalidField("loop_config.gate_rank", "must be positive")
        }
        guard !isVariableLoop else {
            throw Inkling2ConfigError.invalidField(
                "loop_config.is_varloop",
                "true: this is the variable-loop research checkpoint, not the fixed-loop one this runtime implements"
            )
        }
        guard numPasses >= 1, loopCoreStart > 0, loopCoreEnd > loopCoreStart, loopCoreEnd <= numHiddenLayers else {
            throw Inkling2ConfigError.invalidField(
                "loop_config",
                "core_start \(loopCoreStart) < core_end \(loopCoreEnd) <= \(numHiddenLayers) with num_passes >= 1"
            )
        }
        guard eosTokenID >= 0, eosTokenID < vocabSize else {
            throw Inkling2ConfigError.invalidField("eos_token_id", "\(eosTokenID) is outside the vocabulary")
        }
        if let vision, vision.textHiddenSize != hiddenSize {
            throw Inkling2ConfigError.inconsistentLayout(
                "vision_config.text_hidden_size \(vision.textHiddenSize) != text hidden_size \(hiddenSize)"
            )
        }
    }

    // MARK: Allowed keys

    /// The top-level keys this decoder understands. Everything else is refused
    /// by name, so a new checkpoint key is a load error rather than a graph that
    /// silently ignores it.
    static let rootKeys: Set<String> = [
        "architectures", "model_type", "dtype", "torch_dtype", "eos_token_id", "pad_token_id",
        "bos_token_id", "image_token_id", "text_config", "vision_config", "mtp_config",
        "quantization_config", "tie_word_embeddings", "transformers_version"
    ]

    static let textKeys: Set<String> = [
        "model_type", "model_max_length", "torch_dtype", "hidden_size", "num_hidden_layers",
        "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim", "d_rel",
        "rel_extent", "q_bias", "o_bias", "log_scaling_n_floor", "log_scaling_alpha",
        "rms_norm_eps", "use_value_norm_no_gamma", "qk_l2_clip_threshold",
        "rel_l2_clip_threshold", "local_layer_ids", "sconv_kernel_size", "conv_kernel_size",
        "unpadded_vocab_size", "final_logit_softcapping", "swa_head_dim",
        "swa_num_attention_heads", "swa_num_key_value_heads", "sliding_window_size",
        "sliding_window", "n_layers_per_attn_res_block", "attn_res_normalize_values",
        "n_routed_experts", "num_local_experts", "num_experts_per_tok", "n_shared_experts",
        "intermediate_size", "moe_intermediate_size", "mlp_gate_softcap", "mlp_up_softcap",
        "route_scale", "routed_scaling_factor", "use_gate_bias", "scoring_func", "loop_config",
        "layer_types", "loop_layer_index_start", "loop_layer_index_end", "loop_num_passes",
        "loop_gate_rank", "number_of_conv_states", "max_position_embeddings",
        "eos_token_id", "attention_dropout", "initializer_range", "hidden_act",
        "tie_word_embeddings"
    ]

    static let loopKeys: Set<String> = [
        "core_start", "core_end", "num_passes", "gate_rank", "is_varloop"
    ]
}

// MARK: - Vision spec

/// `vision_config`: the tower's geometry and what it costs to price a picture.
/// Phase 1 validates it and derives the span arithmetic; the forward is a later
/// phase.
nonisolated public struct Inkling2VisionSpec: Sendable, Equatable {
    public var modelType: String
    public var textHiddenSize: Int
    public var temporalPatchSize: Int
    public var nativePatchSize: Int
    public var adapterMergeSize: Int
    public var projectionHiddenDim: Int
    public var projectorBias: Bool
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var mlpRatio: Float
    public var posembGridSize: Int
    public var layerNormEps: Float

    /// The image processor's fixed cell, in source pixels.
    public static let cellSize = 40

    /// Language positions one `width x height` picture occupies:
    /// `ceil(height / 40) * ceil(width / 40)`.
    public func languagePositions(width: Int, height: Int) -> Int {
        let rows = (height + Self.cellSize - 1) / Self.cellSize
        let columns = (width + Self.cellSize - 1) / Self.cellSize
        return max(1, rows) * max(1, columns)
    }

    /// ViT tokens the same picture produces: each language cell is
    /// `adapterMergeSize^2` native subpatches of `nativePatchSize` pixels.
    public func towerTokens(width: Int, height: Int) -> Int {
        languagePositions(width: width, height: height)
            * max(1, adapterMergeSize) * max(1, adapterMergeSize)
    }

    /// Rows the tower emits per language position: one merged output row.
    public var rowsPerLanguagePosition: Int { 1 }

    static func decode(_ json: Inkling2JSON, textHiddenSize: Int) throws -> Inkling2VisionSpec {
        try Inkling2Config.refuseUnknownKeys(
            in: json,
            allowing: [
                "model_type", "text_hidden_size", "temporal_patch_size", "native_patch_size",
                "adapter_merge_size", "projection_hidden_dim", "projector_bias", "hidden_size",
                "num_hidden_layers", "num_attention_heads", "mlp_ratio", "posemb_grid_size",
                "layer_norm_eps"
            ],
            where: "vision_config"
        )
        guard let hidden = json.int("hidden_size"),
              let layers = json.int("num_hidden_layers"),
              let heads = json.int("num_attention_heads") else {
            throw Inkling2ConfigError.missingField(
                "vision_config.hidden_size / num_hidden_layers / num_attention_heads"
            )
        }
        let spec = Inkling2VisionSpec(
            modelType: json.string("model_type") ?? "inkling2_vision",
            textHiddenSize: json.int("text_hidden_size") ?? textHiddenSize,
            temporalPatchSize: json.int("temporal_patch_size") ?? 1,
            nativePatchSize: json.int("native_patch_size") ?? 20,
            adapterMergeSize: json.int("adapter_merge_size") ?? 2,
            projectionHiddenDim: json.int("projection_hidden_dim") ?? 0,
            projectorBias: json.bool("projector_bias") ?? false,
            hiddenSize: hidden,
            numHiddenLayers: layers,
            numAttentionHeads: heads,
            mlpRatio: json.float("mlp_ratio") ?? 4,
            posembGridSize: json.int("posemb_grid_size") ?? 32,
            layerNormEps: json.float("layer_norm_eps") ?? 1e-5
        )
        guard spec.temporalPatchSize >= 1 else {
            throw Inkling2ConfigError.invalidField(
                "vision_config.temporal_patch_size", "must be at least 1"
            )
        }
        guard hidden % heads == 0 else {
            throw Inkling2ConfigError.invalidField(
                "vision_config.num_attention_heads", "\(heads) must divide \(hidden)"
            )
        }
        return spec
    }
}

// MARK: - MTP spec

/// `mtp_config`: the optional DSpark sidecar. Validated here so a checkpoint
/// that ships it is either understood or refused by name; the draft runtime is
/// a later phase and nothing in the base turn reads it.
nonisolated public struct Inkling2MTPSpec: Sendable, Equatable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var localLayerIDs: [Int]
    public var slidingWindowSize: Int
    public var blockSize: Int
    public var markovRank: Int
    public var markovHeadType: String
    public var targetLayerIDs: [Int]

    static func decode(_ json: Inkling2JSON) throws -> Inkling2MTPSpec {
        try Inkling2Config.refuseUnknownKeys(
            in: json,
            allowing: [
                "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "intermediate_size", "max_position_embeddings", "torch_dtype",
                "q_bias", "o_bias", "rms_norm_eps", "rope_theta", "local_layer_ids",
                "sliding_window_size", "block_size", "markov_rank", "markov_head_type",
                "target_layer_ids"
            ],
            where: "mtp_config"
        )
        guard let hidden = json.int("hidden_size"),
              let layers = json.int("num_hidden_layers"),
              let heads = json.int("num_attention_heads"),
              let kvHeads = json.int("num_key_value_heads"),
              let headDim = json.int("head_dim") else {
            throw Inkling2ConfigError.missingField(
                "mtp_config.hidden_size / num_hidden_layers / num_attention_heads / num_key_value_heads / head_dim"
            )
        }
        let spec = Inkling2MTPSpec(
            hiddenSize: hidden,
            numHiddenLayers: layers,
            numAttentionHeads: heads,
            numKeyValueHeads: kvHeads,
            headDim: headDim,
            intermediateSize: json.int("intermediate_size") ?? 0,
            maxPositionEmbeddings: json.int("max_position_embeddings") ?? 0,
            ropeTheta: json.float("rope_theta") ?? 10_000,
            localLayerIDs: json.ints("local_layer_ids") ?? [],
            slidingWindowSize: json.int("sliding_window_size") ?? 0,
            blockSize: json.int("block_size") ?? 0,
            markovRank: json.int("markov_rank") ?? 0,
            markovHeadType: json.string("markov_head_type") ?? "rnn_hidden",
            targetLayerIDs: json.ints("target_layer_ids") ?? []
        )
        guard heads % kvHeads == 0 else {
            throw Inkling2ConfigError.invalidField(
                "mtp_config.num_attention_heads", "\(heads) must be a multiple of \(kvHeads)"
            )
        }
        guard spec.blockSize > 0, spec.markovRank > 0 else {
            throw Inkling2ConfigError.invalidField(
                "mtp_config.block_size/markov_rank", "both must be positive when the sidecar is declared"
            )
        }
        guard spec.targetLayerIDs.count == 4 else {
            // The reference captures at three visits plus the final state.
            throw Inkling2ConfigError.invalidField(
                "mtp_config.target_layer_ids",
                "\(spec.targetLayerIDs.count) entries; the draft captures at three visits plus final hidden state"
            )
        }
        return spec
    }
}
