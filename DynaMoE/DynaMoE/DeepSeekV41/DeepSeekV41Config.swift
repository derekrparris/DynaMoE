//
//  DeepSeekV41Config.swift
//  DynaMoE
//
//  DeepSeek-V4.1-Flash ("CED + CSA2 MoE") hyperparameters and the derived
//  per-layer plan.
//
//  Phase 1 of the DeepSeek V4.1 port: pure data, no Metal, no tokenizer, no
//  I/O beyond decoding a `config.json` the caller already read. Everything the
//  hot path needs is resolved ONCE here (layer flavor, compressed-KV source
//  layers, RoPE table, Engram table index) so the forward loop never re-derives
//  architecture strings per token.
//
//  Field names follow the checkpoint's `config.json`. The multimodal wrapper
//  carries the text tower under `text_config`; the checkpoint's own
//  `inference/config.json` is the same fields flattened at the root, and both
//  decode through the same accessor path.
//
//  Layer topology, from the reference forward pass:
//    * `compress_ratios[l]` is 0 for layers that see only the sliding window,
//      and 1 or 2 for layers that pool every 1 or 2 tokens into one compressed
//      row and attend over those rows (CSA2) in addition to the window.
//    * `kv_source_layer_ids` / `index_source_layer_ids` name the layers that
//      OWN those caches; the decoder half of the CED stack reads the encoder
//      half's rows instead of writing its own.
//    * Compressed layers rotate with `compress_rope_theta`; window layers use
//      `rope_theta`. The reference keeps two frequency tables and picks by
//      `compress_ratios[layer]`.
//

import Foundation

// MARK: - Errors

nonisolated public enum DeepSeekV41ConfigError: Error, Equatable, CustomStringConvertible {
    case notDeepSeekV41
    case missingField(String)
    case invalidField(String, String)
    case inconsistentLayout(String)

    public var description: String {
        switch self {
        case .notDeepSeekV41:
            return "config.json does not describe a DeepSeek-V4.1 model"
        case .missingField(let field):
            return "DeepSeek V4.1 config is missing '\(field)'"
        case .invalidField(let field, let detail):
            return "DeepSeek V4.1 config field '\(field)' is invalid: \(detail)"
        case .inconsistentLayout(let detail):
            return "DeepSeek V4.1 layer layout is inconsistent: \(detail)"
        }
    }
}

// MARK: - JSON access

/// Tolerant dictionary reader. Accepts Int/Double/Bool/String and arrays of
/// Int, because HF configs mix `6`, `6.0` and `[1, 14]` freely, and the flat
/// `inference/config.json` uses a few different key spellings than the wrapper.
nonisolated private struct DSV41JSON {
    let raw: [String: Any]

    init(_ raw: [String: Any]) { self.raw = raw }

    /// A JSON number or numeric string as an `Int`, refusing values an `Int`
    /// cannot hold. The Engram row counts are in the 10^9 range, so a 32-bit read
    /// would truncate them silently, and `Int(Double)` traps on overflow.
    static func intValue(_ value: Any) -> Int? {
        if let i = value as? Int { return i }
        if let d = value as? Double {
            guard d.isFinite, d >= Double(Int.min), d <= Double(Int.max) else { return nil }
            return Int(d)
        }
        if let n = value as? NSNumber {
            let wide = n.int64Value
            guard wide >= Int64(Int.min), wide <= Int64(Int.max) else { return nil }
            return Int(wide)
        }
        if let s = value as? String { return Int(s) }
        return nil
    }

    func int(_ keys: String?...) -> Int? { int(keys.compactMap { $0 }) }
    func int(_ keys: [String]) -> Int? {
        for key in keys {
            guard let value = raw[key] else { continue }
            if let parsed = DSV41JSON.intValue(value) { return parsed }
        }
        return nil
    }

    func float(_ keys: String?...) -> Float? { float(keys.compactMap { $0 }) }
    func float(_ keys: [String]) -> Float? {
        for key in keys {
            guard let value = raw[key] else { continue }
            if let f = value as? Float { return f }
            if let d = value as? Double { return Float(d) }
            if let n = value as? NSNumber { return n.floatValue }
            if let s = value as? String, let f = Float(s) { return f }
        }
        return nil
    }

    func bool(_ keys: String?...) -> Bool? { bool(keys.compactMap { $0 }) }
    func bool(_ keys: [String]) -> Bool? {
        for key in keys {
            guard let value = raw[key] else { continue }
            if let b = value as? Bool { return b }
            if let n = value as? NSNumber { return n.boolValue }
        }
        return nil
    }

    func string(_ keys: String?...) -> String? { string(keys.compactMap { $0 }) }
    func string(_ keys: [String]) -> String? {
        for key in keys {
            if let s = raw[key] as? String { return s }
        }
        return nil
    }

    func strings(_ keys: String?...) -> [String]? { strings(keys.compactMap { $0 }) }
    func strings(_ keys: [String]) -> [String]? {
        for key in keys {
            if let s = raw[key] as? [String] { return s }
        }
        return nil
    }

    func ints(_ keys: String?...) -> [Int]? { ints(keys.compactMap { $0 }) }
    func ints(_ keys: [String]) -> [Int]? {
        for key in keys {
            guard let value = raw[key] else { continue }
            if let list = value as? [Int] { return list }
            if let list = value as? [Double] {
                let converted = list.compactMap { DSV41JSON.intValue($0) }
                return converted.count == list.count ? converted : nil
            }
            if let list = value as? [NSNumber] {
                let converted = list.compactMap { DSV41JSON.intValue($0) }
                return converted.count == list.count ? converted : nil
            }
        }
        return nil
    }

    func object(_ keys: String?...) -> DSV41JSON? { object(keys.compactMap { $0 }) }
    func object(_ keys: [String]) -> DSV41JSON? {
        for key in keys {
            if let d = raw[key] as? [String: Any] { return DSV41JSON(d) }
        }
        return nil
    }
}

// MARK: - Layer flavor

/// Which KV/attention machinery a backbone layer runs.
nonisolated public enum DeepSeekV41LayerFlavor: String, Sendable, Equatable, Codable {
    /// `compress_ratio == 0`: the layer sees only the sliding window.
    case windowOnly
    /// `compress_ratio > 0`: CSA2 attention over compressed rows plus the window.
    case compressed
}

/// Which RoPE frequency table a layer rotates with.
nonisolated public enum DeepSeekV41RopeTable: String, Sendable, Equatable, Codable {
    case window
    case compress
}

// MARK: - Layer plan

/// Everything the forward pass needs to know about one backbone layer, resolved
/// at load time.
nonisolated public struct DeepSeekV41LayerPlan: Sendable, Equatable {
    public let index: Int
    public let flavor: DeepSeekV41LayerFlavor
    /// `compress_ratios[layer]`; 0 for a window-only layer.
    public let compressRatio: Int
    /// Layer that OWNS the compressed K/V rows this layer attends over, or -1
    /// for a window-only layer (it has nothing to compress).
    public let compressedKVSource: Int
    /// Layer that owns the indexer keys / top-k this layer reuses, or -1.
    public let compressedIndexSource: Int
    /// True on the single layer whose block scores seed candidate pre-filtering.
    public let isCandidateSource: Bool
    /// Index into `DeepSeekV41Config.engramTables` when this layer carries an
    /// Engram table, nil otherwise.
    public let engramTable: Int?
    public let ropeTable: DeepSeekV41RopeTable

    /// Every layer keeps its own bounded window ring, so the only caches that
    /// scale with context are the compressed ones a layer owns.
    public var ownsCompressedKV: Bool { compressedKVSource == index }
    public var ownsCompressedIndex: Bool { compressedIndexSource == index }

    public init(
        index: Int,
        flavor: DeepSeekV41LayerFlavor,
        compressRatio: Int,
        compressedKVSource: Int,
        compressedIndexSource: Int,
        isCandidateSource: Bool,
        engramTable: Int?,
        ropeTable: DeepSeekV41RopeTable
    ) {
        self.index = index
        self.flavor = flavor
        self.compressRatio = compressRatio
        self.compressedKVSource = compressedKVSource
        self.compressedIndexSource = compressedIndexSource
        self.isCandidateSource = isCandidateSource
        self.engramTable = engramTable
        self.ropeTable = ropeTable
    }
}

// MARK: - Engram

/// One Engram table (one per `engram_layer_ids` entry). The table is far larger
/// than RAM and is never mapped: rows are pread on demand, one hash column at a
/// time. See `DeepSeekV41EngramLayout` for the bucket arithmetic.
nonisolated public struct DeepSeekV41EngramSpec: Sendable, Equatable {
    /// Layer that carries this table.
    public let layer: Int
    /// Rows in the table, as published by `engram_num_embeddings`. The bucket
    /// ranges the hasher uses are carved out of this many rows.
    public let rowCount: Int
    /// Largest n-gram the table hashes (the table owns `maxNgramSize - 1`
    /// hash columns per head: bigrams and up).
    public let maxNgramSize: Int
    public let headCount: Int
    public let headDimension: Int
    /// Token that fills n-gram slots with no history (matches training).
    public let padTokenID: Int
    /// Size of the compressed token vocabulary the hashes are drawn from.
    public let compressedVocabSize: Int
    /// Bucket size each (n-gram size, head) column starts searching primes at.
    public let bucketStart: Int

    /// Bytes of one table row: E4M3 values plus one E8M0 scale per 32.
    public var rowBytes: Int { headDimension + (headDimension + 31) / 32 }
    public var totalBytes: Int { rowCount * rowBytes }

    public init(
        layer: Int,
        rowCount: Int,
        maxNgramSize: Int,
        headCount: Int,
        headDimension: Int,
        padTokenID: Int,
        compressedVocabSize: Int,
        bucketStart: Int
    ) {
        self.layer = layer
        self.rowCount = rowCount
        self.maxNgramSize = maxNgramSize
        self.headCount = headCount
        self.headDimension = headDimension
        self.padTokenID = padTokenID
        self.compressedVocabSize = compressedVocabSize
        self.bucketStart = bucketStart
    }
}

// MARK: - Config

nonisolated public struct DeepSeekV41Config: Sendable, Equatable {

    // Identity
    public var architectures: [String]
    public var modelType: String
    public var dtype: String
    public var quantizationMethod: String?

    // Sizes
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var vocabSize: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var sharedExpertCount: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var tieWordEmbeddings: Bool

    // Latent attention (MLA / CSA2)
    public var qLoraRank: Int
    public var kvLoraRank: Int
    public var qkNopeHeadDim: Int
    public var qkRopeHeadDim: Int
    public var vHeadDim: Int
    public var oGroups: Int
    public var oLoraRank: Int

    // RoPE
    public var ropeTheta: Float
    public var compressRopeTheta: Float
    public var ropeFactor: Float
    public var ropeOriginalLength: Int
    public var ropeBetaFast: Float
    public var ropeBetaSlow: Float

    // Attention shape
    public var windowSize: Int
    public var compressRatios: [Int]
    public var kvSourceLayerIDs: [Int]
    public var indexSourceLayerIDs: [Int]
    public var candidateSourceLayerID: Int
    public var candidateBlockSize: Int
    public var candidateTopKBlocks: Int
    public var indexNHeads: Int
    public var indexHeadDim: Int
    public var indexTopK: Int

    // Hyper-connections (mHC)
    public var hcMult: Int
    public var hcSinkhornIterations: Int
    public var hcEps: Float
    public var hcPreFromPrevSublayer: Bool

    // Engram
    public var engramLayerIDs: [Int]
    public var engramRowCounts: [Int]
    public var engramMaxNgramSize: Int
    public var engramVocabSize: Int
    public var engramHeadCount: Int
    public var engramHeadDimension: Int
    public var engramPadTokenID: Int
    public var engramCompressedVocabSize: Int

    // MoE routing
    public var numRoutedExperts: Int
    public var numActivatedExperts: Int
    public var routedScalingFactor: Float
    public var swigluLimit: Float?
    public var routerHasBias: Bool

    // DSpark drafter
    public var numNextNPredictLayers: Int
    public var dsparkBlockSize: Int
    public var dsparkNoiseTokenID: Int
    public var dsparkTargetLayerIDs: [Int]

    // Tokenizer
    public var bosTokenID: Int
    public var eosTokenIDs: [Int]
    public var padTokenID: Int
    public var imageTokenID: Int?

    // Vision
    public var vision: DeepSeekV41VisionSpec?

    /// Whether the checkpoint carries a vision tower. Derived from the decoded
    /// `vision_config` rather than a separate flag, so the tower's geometry is
    /// never "present but unparsed".
    public var hasVisionTower: Bool { vision != nil }

    /// The vision tower's dispatch plan, or nil for a text-only checkpoint.
    public var visionTowerPlan: DeepSeekV41VisionTowerPlan? {
        guard let vision else { return nil }
        return DeepSeekV41VisionTowerPlan(spec: vision, projectorOutputWidth: hiddenSize)
    }

    // MARK: Derived

    public var engramTables: [DeepSeekV41EngramSpec] {
        zip(engramLayerIDs, engramRowCounts).map { layer, rows in
            DeepSeekV41EngramSpec(
                layer: layer,
                rowCount: rows,
                maxNgramSize: engramMaxNgramSize,
                headCount: engramHeadCount,
                headDimension: engramHeadDimension,
                padTokenID: engramPadTokenID,
                compressedVocabSize: engramCompressedVocabSize,
                bucketStart: engramVocabSize
            )
        }
    }

    /// Size of one query head after RoPE (`d_nope + d_rope`).
    public var qkHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    /// Values in one cached latent record: the compressed `kv_lora_rank` latent
    /// plus any decoupled rope payload. This is the width the cache stores, which
    /// is the latent width rather than the query width (they coincide on the
    /// published checkpoint, but a checkpoint with a decoupled rope payload would
    /// diverge and this must not silently agree with `qkHeadDim`).
    public var cacheRecordWidth: Int {
        kvLoraRank > 0 ? kvLoraRank + qkRopeHeadDim : qkHeadDim
    }

    /// Bytes one token costs in the FP8 main cache: one E4M3 per latent
    /// dimension plus one E8M0 scale per 32 (the footer form the reference ships).
    public var fp8CacheRecordBytes: Int {
        let width = cacheRecordWidth
        return width + (width + 31) / 32
    }

    /// Bytes one token costs in the FP4 main cache: two values per byte plus one
    /// E4M3 scale per 16. The published checkpoint's designed cache is this one,
    /// 288 bytes for a 512-wide latent; the FP8 form is the safe first target.
    public var fp4CacheRecordBytes: Int {
        let width = cacheRecordWidth
        return (width + 1) / 2 + (width + 15) / 16
    }

    /// Layers that own a compressed K/V cache (the only caches that grow with
    /// context; window rings are bounded by `windowSize`).
    ///
    /// Derived from the same owner rule as `makeLayerPlan` but without building
    /// the plan: swallowing a layout error into an empty plan would make this
    /// report zero caches, and `cacheBytesPerToken` silently zero.
    public var compressedKVOwnerLayerIndices: [Int] {
        guard numHiddenLayers > 0 else { return [] }
        var owners: [Int] = []
        for layer in 0..<numHiddenLayers {
            let ratio = layer < compressRatios.count ? compressRatios[layer] : 0
            guard ratio > 0 else { continue }
            if owner(in: kvSourceLayerIDs, at: layer, field: "kv_source_layer_ids") == layer {
                owners.append(layer)
            }
        }
        return owners
    }

    /// Bytes per token across every compressed cache, at the given precision.
    public func cacheBytesPerToken(useFP4Main: Bool) -> Int {
        let main = useFP4Main ? fp4CacheRecordBytes : fp8CacheRecordBytes
        return compressedKVOwnerLayerIndices.count * main
    }

    /// Bytes one window ring holds per layer (FP8 window cache).
    public var windowRingBytesPerLayer: Int {
        windowSize * fp8CacheRecordBytes
    }

    /// Largest single parameter block the streamed Engram reader must address.
    public var largestEngramTableBytes: Int {
        engramTables.map(\.totalBytes).max() ?? 0
    }

    /// Total Engram bytes on disk, the dominant streaming cost of this model.
    public var totalEngramBytes: Int {
        engramTables.reduce(0) { $0 + $1.totalBytes }
    }

    // MARK: Layer plan

    /// Resolves the per-layer plan. Throws when the layout is internally
    /// inconsistent, because every one of these relations is load-bearing for
    /// the forward pass and a silently-wrong plan produces plausible garbage.
    public func makeLayerPlan() throws -> [DeepSeekV41LayerPlan] {
        guard numHiddenLayers > 0 else {
            throw DeepSeekV41ConfigError.invalidField("num_hidden_layers", "must be positive")
        }
        if !compressRatios.isEmpty && compressRatios.count < numHiddenLayers {
            throw DeepSeekV41ConfigError.inconsistentLayout(
                "compress_ratios has \(compressRatios.count) entries for \(numHiddenLayers) layers"
            )
        }
        if engramLayerIDs.count != engramRowCounts.count {
            throw DeepSeekV41ConfigError.inconsistentLayout(
                "engram_layer_ids has \(engramLayerIDs.count) entries but engram_num_embeddings has \(engramRowCounts.count)"
            )
        }
        for (offset, layer) in engramLayerIDs.enumerated() {
            guard layer >= 0, layer < numHiddenLayers else {
                throw DeepSeekV41ConfigError.inconsistentLayout(
                    "engram_layer_ids[\(offset)] = \(layer) is outside 0..<\(numHiddenLayers)"
                )
            }
            if offset > 0 && engramLayerIDs[offset - 1] >= layer {
                throw DeepSeekV41ConfigError.inconsistentLayout(
                    "engram_layer_ids is not strictly ascending"
                )
            }
        }
        for (name, ids) in [("kv_source_layer_ids", kvSourceLayerIDs),
                            ("index_source_layer_ids", indexSourceLayerIDs)] {
            guard Set(ids).count == ids.count else {
                throw DeepSeekV41ConfigError.inconsistentLayout("\(name) contains duplicates")
            }
            if let bad = ids.first(where: { $0 < 0 || $0 >= numHiddenLayers }) {
                throw DeepSeekV41ConfigError.inconsistentLayout(
                    "\(name) contains \(bad), outside 0..<\(numHiddenLayers)"
                )
            }
            if ids != ids.sorted() {
                throw DeepSeekV41ConfigError.inconsistentLayout("\(name) is not ascending")
            }
        }

        var plans: [DeepSeekV41LayerPlan] = []
        plans.reserveCapacity(numHiddenLayers)
        for layer in 0..<numHiddenLayers {
            let ratio = compressRatios.isEmpty ? 0 : compressRatios[layer]
            guard ratio >= 0 else {
                throw DeepSeekV41ConfigError.invalidField(
                    "compress_ratios[\(layer)]", "must not be negative"
                )
            }
            let flavor: DeepSeekV41LayerFlavor = ratio > 0 ? .compressed : .windowOnly
            let kvSource = flavor == .compressed
                ? owner(in: kvSourceLayerIDs, at: layer, field: "kv_source_layer_ids")
                : -1
            let indexSource = flavor == .compressed
                ? owner(in: indexSourceLayerIDs, at: layer, field: "index_source_layer_ids")
                : -1
            plans.append(
                DeepSeekV41LayerPlan(
                    index: layer,
                    flavor: flavor,
                    compressRatio: ratio,
                    compressedKVSource: kvSource,
                    compressedIndexSource: indexSource,
                    isCandidateSource: layer == candidateSourceLayerID,
                    engramTable: engramLayerIDs.firstIndex(of: layer),
                    ropeTable: flavor == .compressed ? .compress : .window
                )
            )
        }
        return plans
    }

    /// Most recent published source layer at or before `layer`. Every compressed
    /// layer must resolve one: without it there is no cache to attend over.
    private func owner(in ids: [Int], at layer: Int, field: String) -> Int {
        let published = ids.last(where: { $0 <= layer })
        if let published { return published }
        // No published list (or none before this layer): the first compressible
        // layer owns the cache and every later compressed layer reads it. This
        // reproduces the reference default where each compressed layer owns its
        // own rows only when the checkpoint says so.
        if ids.isEmpty {
            if let firstCompressed = compressRatios.prefix(layer + 1).firstIndex(where: { $0 > 0 }) {
                return firstCompressed
            }
            return layer
        }
        return layer
    }

    /// Observations worth putting in the load log: shapes that are legal per the
    /// schema but that the forward pass does not model yet.
    public func warnings(layerPlan: [DeepSeekV41LayerPlan]) -> [String] {
        var out: [String] = []
        if numKeyValueHeads > 1 {
            out.append(
                "num_key_value_heads = \(numKeyValueHeads): the latent cache holds one record per position"
            )
        }
        if numAttentionHeads % oGroups != 0 {
            out.append(
                "o_groups = \(oGroups) does not divide num_attention_heads = \(numAttentionHeads)"
            )
        }
        if numHiddenLayers % 2 != 0 {
            out.append(
                "odd layer count \(numHiddenLayers): the CED encoder/decoder split assumes an even count"
            )
        }
        if layerPlan.allSatisfy({ $0.flavor == .windowOnly }) {
            out.append("no layer declares compress_ratios > 0: the checkpoint is not a CSA2 model")
        }
        if windowSize <= 0 {
            out.append("window_size = \(windowSize): every layer would attend over the full context")
        }
        if numNextNPredictLayers > 0 && dsparkTargetLayerIDs.count != numNextNPredictLayers {
            out.append(
                "num_nextn_predict_layers = \(numNextNPredictLayers) but dspark_target_layer_ids has \(dsparkTargetLayerIDs.count) entries"
            )
        }
        return out
    }
}

// MARK: - Loading

nonisolated public extension DeepSeekV41Config {

    /// Decodes a `config.json`. Accepts both shapes the checkpoint ships: the
    /// multimodal wrapper (`text_config` nested) and the flat inference config.
    /// Throws `notDeepSeekV41` when the file belongs to another family, so
    /// callers can tell "not mine" from "mine but broken".
    static func decode(jsonData: Data) throws -> DeepSeekV41Config {
        guard let object = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw DeepSeekV41ConfigError.invalidField("config.json", "not a JSON object")
        }
        return try decode(object: object)
    }

    static func decode(contentsOf url: URL) throws -> DeepSeekV41Config {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw DeepSeekV41ConfigError.missingField("config.json (\(error.localizedDescription))")
        }
        return try decode(jsonData: data)
    }

    /// Cheap identity probe used by discovery before the full decode runs.
    nonisolated static func identifies(_ object: [String: Any]) -> Bool {
        let root = DSV41JSON(object)
        let text = root.object("text_config")
        let types = [root.string("model_type"), text?.string("model_type")]
            .compactMap { $0?.lowercased() }
        let archs = (root.strings("architectures") ?? []).map { $0.lowercased() }
        return types.contains { $0.contains("deepseek_v41") || $0.contains("deepseek_v4.1") }
            || archs.contains { $0.contains("deepseekv41") }
    }

    static func decode(object: [String: Any]) throws -> DeepSeekV41Config {
        guard identifies(object) else {
            throw DeepSeekV41ConfigError.notDeepSeekV41
        }
        let root = DSV41JSON(object)
        // The text tower is authoritative when present; the flat inference
        // config puts the same fields at the root.
        let text = root.object("text_config") ?? root

        func requireInt(_ keys: String...) throws -> Int {
            guard let value = text.int(keys) ?? root.int(keys) else {
                throw DeepSeekV41ConfigError.missingField(keys[0])
            }
            return value
        }

        let numHiddenLayers = try requireInt("num_hidden_layers", "n_layers")
        let ropeScaling = text.object("rope_scaling") ?? root.object("rope_scaling")
        // Decoded before the config is built: the image token id can live in either
        // the root or the vision block, and a tower without a token id cannot be
        // prompted for.
        let visionSpec = DeepSeekV41VisionSpec.decode(root.object("vision_config")?.raw)
            ?? DeepSeekV41VisionSpec.decode(text.object("vision_config")?.raw)

        let config = DeepSeekV41Config(
            architectures: root.strings("architectures") ?? ["DeepseekV41ForCausalLM"],
            modelType: root.string("model_type") ?? text.string("model_type") ?? "deepseek_v41",
            dtype: root.string("dtype", "torch_dtype") ?? "bfloat16",
            quantizationMethod: root.object("quantization_config")?.string("quant_method"),
            hiddenSize: try requireInt("hidden_size", "dim"),
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: try requireInt("num_attention_heads", "n_heads"),
            numKeyValueHeads: text.int("num_key_value_heads") ?? 1,
            vocabSize: try requireInt("vocab_size"),
            intermediateSize: text.int("intermediate_size") ?? 0,
            moeIntermediateSize: text.int("moe_intermediate_size") ?? 0,
            sharedExpertCount: text.int("n_shared_experts") ?? 0,
            maxPositionEmbeddings: text.int("max_position_embeddings", "context_length") ?? 1_048_576,
            rmsNormEps: text.float("rms_norm_eps", "norm_eps") ?? 1e-6,
            tieWordEmbeddings: text.bool("tie_word_embeddings") ?? false,
            qLoraRank: text.int("q_lora_rank") ?? 0,
            kvLoraRank: text.int("kv_lora_rank") ?? 0,
            qkNopeHeadDim: text.int("qk_nope_head_dim") ?? text.int("head_dim") ?? 512,
            qkRopeHeadDim: text.int("qk_rope_head_dim") ?? 0,
            vHeadDim: text.int("v_head_dim") ?? text.int("head_dim") ?? 512,
            oGroups: text.int("o_groups") ?? 1,
            oLoraRank: text.int("o_lora_rank") ?? 0,
            ropeTheta: text.float("rope_theta") ?? 10_000,
            compressRopeTheta: text.float("compress_rope_theta") ?? text.float("rope_theta") ?? 10_000,
            ropeFactor: ropeScaling?.float("factor") ?? 1,
            ropeOriginalLength: ropeScaling?.int("original_max_position_embeddings") ?? 0,
            ropeBetaFast: ropeScaling?.float("beta_fast") ?? 32,
            ropeBetaSlow: ropeScaling?.float("beta_slow") ?? 1,
            windowSize: text.int("window_size") ?? 0,
            compressRatios: text.ints("compress_ratios") ?? root.ints("compress_ratios") ?? [],
            kvSourceLayerIDs: text.ints("kv_source_layer_ids") ?? root.ints("kv_source_layer_ids") ?? [],
            indexSourceLayerIDs: text.ints("index_source_layer_ids") ?? root.ints("index_source_layer_ids") ?? [],
            candidateSourceLayerID: text.int("candidate_source_layer_id", "candidate_source_layer") ?? -1,
            candidateBlockSize: text.int("candidate_block_size") ?? 0,
            candidateTopKBlocks: text.int("candidate_topk_blocks") ?? 0,
            indexNHeads: text.int("index_n_heads", "indexer_n_heads") ?? 0,
            indexHeadDim: text.int("index_head_dim", "indexer_head_dim") ?? 0,
            indexTopK: text.int("index_topk") ?? 0,
            hcMult: text.int("hc_mult", "hc_count") ?? 1,
            hcSinkhornIterations: text.int("hc_sinkhorn_iters") ?? 0,
            hcEps: text.float("hc_eps") ?? 1e-6,
            hcPreFromPrevSublayer: text.bool("hc_pre_from_prev_sublayer") ?? true,
            engramLayerIDs: text.ints("engram_layer_ids") ?? [],
            engramRowCounts: text.ints("engram_num_embeddings") ?? [],
            engramMaxNgramSize: text.int("engram_max_ngram_size") ?? 1,
            engramVocabSize: text.int("engram_vocab_size") ?? 0,
            engramHeadCount: text.int("engram_n_heads") ?? 0,
            engramHeadDimension: text.int("engram_head_dim") ?? 0,
            engramPadTokenID: text.int("engram_pad_token_id", "engram_pad_id") ?? root.int("pad_token_id") ?? 2,
            engramCompressedVocabSize: text.int("engram_compressed_vocab_size") ?? 0,
            numRoutedExperts: text.int("n_routed_experts", "num_experts", "num_local_experts") ?? 0,
            numActivatedExperts: text.int("n_activated_experts", "num_experts_per_tok", "top_k_experts") ?? 0,
            routedScalingFactor: text.float("routed_scaling_factor", "route_scale") ?? 1,
            swigluLimit: text.float("swiglu_limit"),
            routerHasBias: text.bool("moe_router_enable_expert_bias", "router_bias") ?? true,
            numNextNPredictLayers: text.int("num_nextn_predict_layers") ?? 0,
            dsparkBlockSize: text.int("dspark_block_size") ?? 0,
            dsparkNoiseTokenID: text.int("dspark_noise_token_id") ?? -1,
            dsparkTargetLayerIDs: text.ints("dspark_target_layer_ids") ?? [],
            bosTokenID: root.int("bos_token_id") ?? 0,
            eosTokenIDs: DeepSeekV41Config.tokenIDs(root, "eos_token_id") ?? [1],
            padTokenID: root.int("pad_token_id") ?? 2,
            imageTokenID: root.int("image_token_id")
                ?? visionSpec.flatMap { $0.imageTokenID >= 0 ? $0.imageTokenID : nil },
            vision: visionSpec
        )

        try config.validate()
        return config
    }

    private static func tokenIDs(_ json: DSV41JSON, _ key: String) -> [Int]? {
        if let list = json.ints(key), !list.isEmpty { return list }
        if let single = json.int(key) { return [single] }
        return nil
    }

    /// Hard invariants: anything failing here means no plan can be built.
    func validate() throws {
        guard hiddenSize > 0 else {
            throw DeepSeekV41ConfigError.invalidField("hidden_size", "must be positive")
        }
        guard numAttentionHeads > 0 else {
            throw DeepSeekV41ConfigError.invalidField("num_attention_heads", "must be positive")
        }
        guard qkNopeHeadDim > 0 else {
            throw DeepSeekV41ConfigError.invalidField("qk_nope_head_dim", "must be positive")
        }
        // A decoupled rope payload is the one shape the two backends disagree on:
        // the reference rotates the latent as a single vector, while the kernel
        // rotates it per `rotaryDim` block. Refusing here is the only honest option
        // until one rule is chosen, because the checkpoint would otherwise load and
        // run silently different numerics on CPU and GPU.
        guard qkRopeHeadDim == 0 else {
            throw DeepSeekV41ConfigError.invalidField(
                "qk_rope_head_dim",
                "must be 0: the CSA2 path rotates the whole \(qkHeadDim)-wide latent, and a decoupled rope payload has no agreed rotation"
            )
        }
        guard oGroups > 0 else {
            throw DeepSeekV41ConfigError.invalidField("o_groups", "must be positive")
        }
        guard numRoutedExperts > 0 else {
            throw DeepSeekV41ConfigError.invalidField("n_routed_experts", "must be positive")
        }
        guard numActivatedExperts > 0, numActivatedExperts <= numRoutedExperts else {
            throw DeepSeekV41ConfigError.invalidField(
                "n_activated_experts", "must be in 1...\(numRoutedExperts)"
            )
        }
        guard hcMult >= 1 else {
            throw DeepSeekV41ConfigError.invalidField("hc_mult", "must be at least 1")
        }
        if engramLayerIDs.isEmpty {
            guard engramHeadCount == 0 else {
                throw DeepSeekV41ConfigError.invalidField(
                    "engram_n_heads", "set with no engram_layer_ids"
                )
            }
        } else {
            guard engramHeadCount > 0, engramHeadDimension > 0 else {
                throw DeepSeekV41ConfigError.invalidField(
                    "engram_n_heads/engram_head_dim", "must be positive when engram_layer_ids is set"
                )
            }
            guard engramMaxNgramSize >= 2 else {
                throw DeepSeekV41ConfigError.invalidField(
                    "engram_max_ngram_size", "must be at least 2 (it hashes bigrams and up)"
                )
            }
            guard engramCompressedVocabSize > 0 else {
                throw DeepSeekV41ConfigError.invalidField(
                    "engram_compressed_vocab_size", "must be positive"
                )
            }
            guard engramVocabSize > 0 else {
                throw DeepSeekV41ConfigError.invalidField(
                    "engram_vocab_size", "must be positive (bucket ranges start here)"
                )
            }
            if engramRowCounts.contains(where: { $0 <= 0 }) {
                throw DeepSeekV41ConfigError.invalidField(
                    "engram_num_embeddings", "every table needs a positive row count"
                )
            }
        }
        if hasVisionTower && imageTokenID == nil {
            throw DeepSeekV41ConfigError.missingField("image_token_id")
        }
    }
}
