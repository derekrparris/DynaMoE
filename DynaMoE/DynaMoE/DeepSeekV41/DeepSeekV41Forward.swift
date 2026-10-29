//
//  DeepSeekV41Forward.swift
//  DynaMoE
//
//  The block forward, written once against a backend so the same control flow runs
//  on the GPU and on the CPU reference.
//
//  Why a CPU reference at all: every other architecture in this repo can be
//  checked against a model that is already on disk, but DeepSeek-V4.1's checkpoint
//  is 510 GB and normal work happens before it is downloaded. A CPU forward over
//  synthetic weights is the only way to prove the parts that do not depend on the
//  weights at all:
//
//    * the layer order and where each sub-layer writes its residual,
//    * which cache rows a position reads and which it writes,
//    * that the compressed-row set for a position is what the ratio says,
//    * that routing, the expert mixture and the hyper-connection streams reshape
//      the hidden state exactly once each,
//    * that the final norm and head see the collapsed stream, not one of the
//      intermediate ones.
//
//  Those are the bugs that a wrong port ships with, and none of them needs 510 GB
//  of weights to catch. The GPU executor binds the same stages to the Metal kernels
//  (`DeepSeekV41MetalForward`), so a disagreement between the two is a kernel bug
//  rather than a re-derivation of the graph.
//
//  Layout conventions, all from the checkpoint:
//    * matrices are `[out][in]` row-major, so `apply` is a dot product per row,
//    * the latent record (key AND value) has `cacheRecordWidth` values,
//    * every query head has `qkHeadDim` values, and only its trailing rope payload
//      rotates (the whole head on the legacy non-decoupled shape),
//    * the residual is `hcMult` copies of the hidden vector.
//

import Foundation

// MARK: - Matrix

nonisolated public struct DeepSeekV41Matrix: Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    /// Row-major, `rows * columns` values.
    public let values: [Float]

    public init(rows: Int, columns: Int, values: [Float]) {
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    /// y = W x. A missing or mismatched matrix yields zeros so a plan-level bug
    /// shows up as a zero block instead of an out-of-range read.
    public func apply(_ x: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        guard columns > 0, values.count >= rows * columns, x.count >= columns else { return out }
        for row in 0..<rows {
            let base = row * columns
            var total: Float = 0
            for column in 0..<columns {
                total += values[base + column] * x[column]
            }
            out[row] = total
        }
        return out
    }

    /// Row `row` as a vector, for `[1][n]` matrices used as norms.
    public func row(_ index: Int) -> [Float] {
        guard index >= 0, index < rows else { return [] }
        let base = index * columns
        guard base + columns <= values.count else { return [] }
        return Array(values[base..<(base + columns)])
    }

    /// y = W x where `W` is block-diagonal over `groups` head groups: the matrix
    /// is `groups` dense blocks stacked along the rows, and block `g` reads only
    /// columns `g * columns / groups ..< (g + 1) * columns / groups`.
    ///
    /// `attn.wo_a` is published this way ([o_groups * o_lora_rank] rows by
    /// [heads * head_dim / o_groups] columns, 8192 x 4096 at 8 groups) and the
    /// reference reads it with an einsum per group, not as one `Linear`. Applying
    /// it dense would multiply each group's rows into every other group's heads.
    public func applyBlockDiagonal(_ x: [Float], groups: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        let g = max(1, groups)
        guard values.count >= rows * columns, x.count >= columns else { return out }
        guard rows % g == 0, columns % g == 0 else { return out }
        let rowsPerGroup = rows / g
        let columnsPerGroup = columns / g
        for group in 0..<g {
            let rowBase = group * rowsPerGroup
            let columnBase = group * columnsPerGroup
            for row in 0..<rowsPerGroup {
                let offset = (rowBase + row) * columns + columnBase
                var total: Float = 0
                for column in 0..<columnsPerGroup {
                    total += values[offset + column] * x[columnBase + column]
                }
                out[rowBase + row] = total
            }
        }
        return out
    }
}

// MARK: - Weight source

/// Where the forward reads its matrices. The CPU implementation holds them in
/// memory; the Metal one resolves them to `(shard buffer, byte offset)` and never
/// materializes a `[Float]`.
nonisolated public protocol DeepSeekV41WeightSource: AnyObject {
    func matrix(_ role: DeepSeekV41TensorRole, layer: Int?, expert: Int?) -> DeepSeekV41Matrix?
    func vector(_ role: DeepSeekV41TensorRole, layer: Int?) -> [Float]?
}

public extension DeepSeekV41WeightSource {
    /// Layer-level tensors, which are the overwhelming majority.
    func matrix(_ role: DeepSeekV41TensorRole, layer: Int?) -> DeepSeekV41Matrix? {
        matrix(role, layer: layer, expert: nil)
    }
}

/// In-memory weight store, keyed by `role|layer|expert` exactly like the catalog.
///
/// It also answers Engram lookups from rows a test pins with `setEngramRow`, which
/// is how the reference forward is exercised without a 100 GB table on disk. The
/// real path is `DeepSeekV41EngramStreamingLookup`.
nonisolated public final class DeepSeekV41DenseWeightSource: DeepSeekV41WeightSource,
                                                            DeepSeekV41EngramLookup {
    private var matrices: [String: DeepSeekV41Matrix] = [:]
    private var vectors: [String: [Float]] = [:]
    private var engram: [String: [Float]] = [:]
    /// Highest column index pinned per layer, so the lookup can report the shape
    /// the caller pinned rather than the shape the config declares.
    private var engramColumns: [Int: Int] = [:]

    public init() {}

    public static func key(_ role: DeepSeekV41TensorRole, layer: Int?, expert: Int? = nil) -> String {
        "\(role.rawValue)|\(layer.map(String.init) ?? "-")|\(expert.map(String.init) ?? "-")"
    }

    public func set(_ matrix: DeepSeekV41Matrix, role: DeepSeekV41TensorRole, layer: Int?, expert: Int? = nil) {
        matrices[Self.key(role, layer: layer, expert: expert)] = matrix
    }

    public func set(_ vector: [Float], role: DeepSeekV41TensorRole, layer: Int?) {
        vectors[Self.key(role, layer: layer)] = vector
    }

    public func setEngramRow(_ row: [Float], layer: Int, column: Int) {
        engram["\(layer)|\(column)"] = row
        engramColumns[layer] = max(engramColumns[layer] ?? 0, column + 1)
    }

    public func matrix(_ role: DeepSeekV41TensorRole, layer: Int?, expert: Int?) -> DeepSeekV41Matrix? {
        matrices[Self.key(role, layer: layer, expert: expert)]
    }

    public func vector(_ role: DeepSeekV41TensorRole, layer: Int?) -> [Float]? {
        vectors[Self.key(role, layer: layer)]
    }

    public func engramRows(layer: Int, history: [Int32], dead: [Bool]) throws -> [[Float]]? {
        guard let columns = engramColumns[layer], columns > 0 else { return nil }
        return (0..<columns).map { engram["\(layer)|\($0)"] ?? [] }
    }
}

// MARK: - Forward state

/// The buffers one forward pass owns. Sizes are derived from the config so the
/// reference and the GPU path allocate identically.
nonisolated public struct DeepSeekV41ForwardState: Sendable, Equatable {
    public let hidden: Int
    public let qWidth: Int
    public let latentWidth: Int
    public let outputWidth: Int
    public let qLoraRank: Int
    public let oLoraRank: Int
    public let moeIntermediate: Int
    public let sharedIntermediate: Int
    public let heads: Int
    public let hcMult: Int
    public let vocab: Int

    public init(config: DeepSeekV41Config) {
        let shapes = DeepSeekV41TensorShapes(config: config)
        self.hidden = config.hiddenSize
        self.qWidth = shapes.queryWidth
        self.latentWidth = config.cacheRecordWidth
        self.outputWidth = shapes.attentionOutputWidth
        self.qLoraRank = config.qLoraRank
        self.oLoraRank = config.oLoraRank
        self.moeIntermediate = shapes.expertIntermediate
        self.sharedIntermediate = shapes.sharedIntermediate
        self.heads = config.numAttentionHeads
        self.hcMult = max(1, config.hcMult)
        self.vocab = config.vocabSize
    }
}

// MARK: - Derived shapes

/// Every projection's `(in, out)` derived from the config.
///
/// The catalog is the authority on the checkpoint; this is what the graph expects.
/// `DeepSeekV41WeightCatalog.validate` compares the two, and the difference is
/// reported rather than silently absorbed by a transposed multiply.
nonisolated public struct DeepSeekV41TensorShapes: Sendable, Equatable {
    public let hidden: Int
    public let queryWidth: Int
    public let attentionOutputWidth: Int
    public let latentWidth: Int
    public let qLoraRank: Int
    public let oLoraRank: Int
    /// Head groups `attn.wo_a` is block-diagonal over (`o_groups`).
    public let outputGroups: Int
    /// Width of the `wo_a` result: `o_groups * o_lora_rank`, NOT `o_lora_rank`.
    /// Each group projects its own heads to `o_lora_rank` and the groups are
    /// concatenated, which is `o.flatten(2)` in the reference; `attn.wo_b` then
    /// reads all of them.
    public let outputReducedWidth: Int
    public let expertIntermediate: Int
    public let sharedIntermediate: Int

    public init(config: DeepSeekV41Config) {
        self.hidden = config.hiddenSize
        self.queryWidth = config.numAttentionHeads * config.qkHeadDim
        self.attentionOutputWidth = config.numAttentionHeads * config.vHeadDim
        self.latentWidth = config.cacheRecordWidth
        self.qLoraRank = config.qLoraRank
        self.oLoraRank = config.oLoraRank
        self.outputGroups = max(1, config.oGroups)
        self.outputReducedWidth = config.oLoraRank > 0
            ? max(1, config.oLoraRank) * max(1, config.oGroups)
            : 0
        self.expertIntermediate = max(1, config.moeIntermediateSize)
        // The shared expert runs at `n_shared_experts` times the routed width when
        // the checkpoint publishes only the routed width, which is how every
        // DeepSeek checkpoint spells it.
        self.sharedIntermediate = config.intermediateSize > 0
            ? config.intermediateSize
            : max(config.moeIntermediateSize, 1) * max(1, config.sharedExpertCount)
    }
}

// MARK: - Stage list

/// The ordered stages of one layer, shared by both backends.
nonisolated public enum DeepSeekV41ForwardStages {

    public enum Stage: String, Sendable, Equatable {
        case mixAttentionStreams
        case attnNorm
        case queryDown
        case queryUp
        case queryNorm
        case ropeQuery
        case latentProject
        case latentNorm
        case ropeLatent
        case writeLatent
        case compressPool
        case attention
        case ropeOutput
        case outputDown
        case outputUp
        case writeAttentionStreams
        case mixFFNStreams
        case ffnNorm
        case engram
        case router
        case sharedExpert
        case routedExperts
        case writeFFNStreams
        case collapseStreams
    }

    /// Every stage a layer runs, in order. A window-only layer skips the compressor
    /// and reads no compressed rows but still runs the same attention stage over
    /// just its window.
    public static let layerPipeline: [Stage] = [
        .mixAttentionStreams,
        .attnNorm,
        .queryDown,
        .queryNorm,
        .queryUp,
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
        .mixFFNStreams,
        .ffnNorm,
        .engram,
        .router,
        .routedExperts,
        .sharedExpert,
        .writeFFNStreams,
        .collapseStreams
    ]
}

// MARK: - Engram combination

/// How retrieved Engram rows enter the layer.
///
/// The checkpoint's `Engram.forward` forms `kv = wkv(embed(hashes).flatten(-2))`,
/// splits it into one key per `hc_mult` copy plus a single shared value, and gates
/// that value by a normalized dot of the stream against its own key. There is no
/// output projection, so nothing is projected back to the hidden width: the value
/// is already `dim` wide.
///
/// It writes into the RESIDUAL stream, at the top of the layer, before the block's
/// attention sub-layer (`Model.forward` calls `layer.engram(h, ...)` ahead of
/// `layer(...)`). Adding it to the post-norm FFN input instead would put the
/// contribution through the FFN's weights, which is not what the checkpoint does.
///
/// Gated by `DeepSeekV41Conformance.engramCombination`: the rule matches the
/// reference, but no reference vector has been checked in against it yet.
nonisolated public enum DeepSeekV41EngramCombination {

    /// What one position's lookup writes into the residual, plus the gate it used
    /// so a probe run can compare against the reference tensor.
    public struct Contribution: Sendable, Equatable {
        /// `hc_mult` copies of the input stream, `dim` wide, with the memory added.
        public let streams: [[Float]]
        /// One gate per copy. The value is shared across copies, the gate is not.
        public let gates: [Float]

        public init(streams: [[Float]], gates: [Float]) {
            self.streams = streams
            self.gates = gates
        }
    }

    /// `Engram.forward`: write the retrieved rows into the residual stream, gated
    /// by how well each copy of that stream matches the retrieved key.
    ///
    /// - Parameters:
    ///   - residual: the stream the layer is about to read, `hc_mult` copies of
    ///     `dim` values.
    ///   - rows: the rows the hash columns addressed, `engram_head_dim` each, in
    ///     column order.
    ///   - wkv: `[dim * (hc_mult + 1)][n_hash_cols * head_dim]`, which turns the
    ///     concatenated rows into one key per copy followed by one shared value.
    ///   - queryWeight: `[hc_mult * dim]`, a plain parameter that is only ever
    ///     used multiplied by `keyWeight`.
    ///   - keyWeight: `[hc_mult * dim]`, see `queryWeight`.
    public static func contribution(
        residual: [[Float]],
        rows: [[Float]],
        wkv: DeepSeekV41Matrix?,
        queryWeight: [Float]?,
        keyWeight: [Float]?,
        epsilon: Float
    ) -> Contribution? {
        let copies = residual.count
        guard copies > 0, let dim = residual.first?.count, dim > 0, !rows.isEmpty else { return nil }
        guard let wkv else { return nil }

        // `embed(hashes).flatten(-2)`: the rows concatenate along one axis, so the
        // projection reads them as a single `n_hash_cols * head_dim` vector.
        var flat: [Float] = []
        flat.reserveCapacity(rows.reduce(0) { $0 + $1.count })
        for row in rows { flat.append(contentsOf: row) }

        // `kv.split([hc_mult * dim, dim])`: a key per copy, then one value shared
        // by every copy.
        let kv = wkv.apply(flat)
        let keyWidth = copies * dim
        guard kv.count >= keyWidth + dim else { return nil }
        let value = Array(kv[keyWidth..<(keyWidth + dim)])

        // `q_weight * k_weight`, elementwise per copy. Both are ones-initialized
        // parameters, so a checkpoint that omits them means "no reweighting".
        var weight = [Float](repeating: 1, count: keyWidth)
        if let queryWeight, let keyWeight,
           queryWeight.count >= keyWidth, keyWeight.count >= keyWidth {
            for index in 0..<keyWidth {
                weight[index] = queryWeight[index] * keyWeight[index]
            }
        }

        let inverseRootDim = 1 / sqrt(Float(dim))
        var gates: [Float] = []
        gates.reserveCapacity(copies)
        var streams: [[Float]] = []
        streams.reserveCapacity(copies)

        for copy in 0..<copies {
            var stream = residual[copy]
            if stream.count < dim {
                stream.append(contentsOf: [Float](repeating: 0, count: dim - stream.count))
            }
            let base = copy * dim
            // Normalized per (token, copy) over `dim`, never jointly over the copies.
            var streamSquares: Float = 0
            var keySquares: Float = 0
            var dot: Float = 0
            for dimension in 0..<dim {
                let hidden = stream[dimension]
                let key = kv[base + dimension]
                streamSquares += hidden * hidden
                keySquares += key * key
                dot += hidden * weight[base + dimension] * key
            }
            let streamRstd = 1 / sqrt(streamSquares / Float(dim) + epsilon)
            let keyRstd = 1 / sqrt(keySquares / Float(dim) + epsilon)
            // `dot * rstd * dim**-0.5`, then the signed square root before the
            // sigmoid, matching the training kernel.
            let score = dot * streamRstd * keyRstd * inverseRootDim
            let magnitude = max(abs(score), 1e-6)
            let gate = 1 / (1 + exp(-copysign(magnitude.squareRoot(), score)))
            gates.append(gate)
            for dimension in 0..<dim {
                stream[dimension] += gate * value[dimension]
            }
            streams.append(stream)
        }
        return Contribution(streams: streams, gates: gates)
    }
}

// MARK: - Reference forward

/// The CPU forward: the graph itself, with no Metal.
///
/// One instance per turn. `prefill` walks a prompt and leaves the cache holding
/// every position; `forwardToken` appends one position and returns the logits.
nonisolated public final class DeepSeekV41CPUForward {

    public let config: DeepSeekV41Config
    public let plan: [DeepSeekV41LayerPlan]
    public let shapes: DeepSeekV41TensorShapes
    /// The ring geometry the row plan addresses against. The reference keeps
    /// every position it has ever written, so it never clamps a read, but the
    /// ring width and record layout here are the ones the device cache uses.
    public let geometry: DeepSeekV41CacheGeometry
    private let weights: DeepSeekV41WeightSource
    /// RoPE inverse frequencies per rope table, resolved once. The query rail
    /// rotates the trailing rope slice of each `qkHeadDim`-wide head and the latent
    /// rail rotates the trailing slice of each cache record, so their tables are
    /// built over those slice widths (they coincide when the payload is decoupled,
    /// and differ on the legacy whole-record shape).
    private let queryFrequencyTable: [DeepSeekV41RopeTable: [Float]]
    private let latentFrequencyTable: [DeepSeekV41RopeTable: [Float]]
    /// Written positions, in order. Mirrors the device cache so the reference can
    /// read the same rows the kernels would.
    public private(set) var latents: [[[Float]]] = []
    /// Post-`attn_norm` attention inputs per layer, in position order. The
    /// compressor projects these, so they are kept beside the latents.
    public private(set) var attentionInputs: [[[Float]]] = []
    /// Compressed rows per owner layer.
    public private(set) var compressed: [Int: [[Float]]] = [:]
    /// Candidate block scores from the candidate-source layer, per position.
    /// Recorded rather than consumed: row selection goes through
    /// `DeepSeekV41CacheAddressing.attentionRowPlan` so the number of rows a
    /// position reads is known before any dispatch (see that type for why). These
    /// stay because they are what a probe run compares against.
    public private(set) var candidateScores: [[Float]] = []
    /// Engram gates per layer, one entry per position that carried a lookup. The
    /// value is written into the residual at the top of the layer, before
    /// attention; keeping the gate is what a probe run compares against.
    public private(set) var engramGates: [Int: [[Float]]] = [:]
    /// Where this forward gets Engram rows. Nil leaves Engram layers skipped, which
    /// is what a config without tables does anyway.
    private let engramLookup: DeepSeekV41EngramLookup?
    /// Token ids seen so far, oldest first: the n-gram hashes are built from them.
    public private(set) var tokenHistory: [Int32] = []
    /// Positions that take no part in an n-gram (image spans). Parallel to
    /// `tokenHistory`; a lookup stops a look-back at the first dead token.
    public private(set) var deadTokens: [Bool] = []
    /// Set by `markNextTokenDead`, consumed by the next `forwardToken`.
    private var pendingDead = false

    public init(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan],
        weights: DeepSeekV41WeightSource,
        engramLookup: DeepSeekV41EngramLookup? = nil
    ) {
        self.config = config
        self.plan = plan
        self.shapes = DeepSeekV41TensorShapes(config: config)
        self.geometry = DeepSeekV41CacheGeometry.make(
            config: config,
            plan: plan,
            contextLength: max(1, config.maxPositionEmbeddings),
            precision: .fp16
        )
        self.weights = weights
        self.engramLookup = engramLookup
        var queries: [DeepSeekV41RopeTable: [Float]] = [:]
        var latents: [DeepSeekV41RopeTable: [Float]] = [:]
        for layer in plan {
            guard queries[layer.ropeTable] == nil else { continue }
            let shape = DeepSeekV41AttentionShape.make(config: config, layer: layer)
            queries[layer.ropeTable] = DeepSeekV41RoPE.queryFrequencies(for: shape)
            latents[layer.ropeTable] = DeepSeekV41RoPE.latentFrequencies(for: shape)
        }
        self.queryFrequencyTable = queries
        self.latentFrequencyTable = latents
    }

    public var storedPositions: Int { latents.first?.count ?? 0 }

    public func reset() {
        latents = []
        attentionInputs = []
        compressed = [:]
        candidateScores = []
        engramGates = [:]
        tokenHistory = []
        deadTokens = []
    }

    /// Embeds a token id, or zeros when the table is missing.
    public func embed(_ token: Int) -> [Float] {
        guard let matrix = weights.matrix(.tokenEmbedding, layer: nil, expert: nil) else {
            return [Float](repeating: 0, count: shapes.hidden)
        }
        return matrix.row(token)
    }

    /// Runs a whole prompt and returns the logits after the last position.
    public func prefill(tokens: [Int]) -> [Float] {
        guard !tokens.isEmpty else { return [] }
        guard storedPositions == 0 else { return [Float](repeating: 0, count: shapes.hidden) }
        for (position, token) in tokens.enumerated() {
            let logits = forwardToken(token, position: position)
            if position == tokens.count - 1 { return logits }
        }
        return []
    }

    /// Marks the next position as taking no part in an n-gram, so an Engram lookup
    /// never spans it. An image span sets this for every token it expands to.
    public func markNextTokenDead() {
        pendingDead = true
    }

    /// Appends one position and returns the output logits.
    public func forwardToken(_ token: Int, position: Int) -> [Float] {
        forwardEmbedding(embed(token), token: token, position: position)
    }

    /// Appends one position whose embedding the caller supplies, which is how an
    /// image span enters the residual stream: `Model.merge_image_embeddings`
    /// overwrites the span's rows before the layers run, so those positions never
    /// come from the token embedding table.
    ///
    /// `token` is still recorded in the history because the n-gram hashes are built
    /// from token ids; mark the position dead (`markNextTokenDead`) so an Engram
    /// lookup cannot span it and contributes nothing there.
    @discardableResult
    public func forwardEmbedding(
        _ embedding: [Float],
        token: Int,
        position: Int
    ) -> [Float] {
        // The hash inputs are the token stream itself, so the history is kept here
        // rather than recomputed by each Engram layer.
        tokenHistory.append(Int32(truncatingIfNeeded: token))
        deadTokens.append(pendingDead)
        pendingDead = false

        var hidden = embedding
        let mult = max(1, config.hcMult)
        var streams = Array(repeating: hidden, count: mult)

        for layer in plan {
            runLayer(layer, at: position, streams: &streams)
        }

        // The block's output is the collapsed stream, not an average of the streams.
        hidden = collapse(streams)
        hidden = rmsNorm(hidden, role: .finalNorm, layer: nil, epsilon: config.rmsNormEps)
        // A tied checkpoint has no separate head: the embedding matrix is the head,
        // transposed. The catalog reports the same tensor either way, so the tie flag
        // from the config is what decides which multiply runs.
        let head = weights.matrix(.lmHead, layer: nil, expert: nil)
            ?? (config.tieWordEmbeddings ? weights.matrix(.tokenEmbedding, layer: nil, expert: nil) : nil)
        guard let head else { return hidden }
        return head.apply(hidden)
    }

    // MARK: One layer

    private func runLayer(
        _ layer: DeepSeekV41LayerPlan,
        at position: Int,
        streams: inout [[Float]]
    ) {
        let index = layer.index
        let shape = DeepSeekV41AttentionShape.make(config: config, layer: layer)

        // Engram memory lands in the residual stream at the TOP of the layer, before
        // the attention sub-layer reads it (`Model.forward` calls `layer.engram(h,
        // ...)` ahead of `layer(...)`).
        if let engramTable = layer.engramTable {
            writeEngram(layer: index, table: engramTable, streams: &streams)
        }

        // Attention sub-layer.
        let attnInput = mixStreams(
            streams,
            readRole: .hcAttnFn,
            scaleRole: .hcAttnScale,
            baseRole: .hcAttnBase,
            layer: index
        )
        let normed = rmsNorm(attnInput, role: .attnNorm, layer: index, epsilon: config.rmsNormEps)

        var query = queryPath(normed, layer: index, shape: shape)
        DeepSeekV41RoPE.rotateHeads(
            &query,
            heads: shape.numHeads,
            headDim: shape.qkHeadDim,
            sliceStart: shape.queryRopeStart,
            sliceLength: shape.queryRopeWidth,
            position: position,
            frequencies: queryFrequencyTable[layer.ropeTable] ?? []
        )

        var latent = latentPath(normed, layer: index)
        DeepSeekV41RoPE.rotateSlice(
            &latent,
            start: shape.latentRopeStart,
            length: shape.latentRopeWidth,
            position: position,
            frequencies: latentFrequencyTable[layer.ropeTable] ?? []
        )
        appendLatent(layer: layer, position: position, latent: latent)
        appendAttentionInput(layer: index, position: position, input: normed)
        pooledLatent(layer: layer, position: position, shape: shape)

        let keys = attentionKeys(layer: layer, position: position)
        let sink = weights.vector(.attnSink, layer: index)
        var attended = DeepSeekV41AttentionReference.attend(
            query: query,
            keys: keys,
            shape: shape,
            sink: sink
        )
        // The keys stay stored rotated, so the output's rope payload has to come back
        // out before the output projection. The reference conjugates the SAME slice it
        // rotated into the keys (`apply_rotary_emb(o[..., -rd:], freqs_cis, True)`);
        // leaving it in rotates every value the projection reads.
        if shape.outputCarriesRope {
            DeepSeekV41RoPE.rotateHeads(
                &attended,
                heads: shape.numHeads,
                headDim: shape.vHeadDim,
                sliceStart: shape.outputRopeStart,
                sliceLength: shape.latentRopeWidth,
                position: position,
                frequencies: latentFrequencyTable[layer.ropeTable] ?? [],
                inverse: true
            )
        }
        let attnOut = outputPath(attended, layer: index, shape: shape)
        DeepSeekV41HyperConnections.write(
            streams: &streams,
            contribution: attnOut,
            writeScales: weights.vector(.hcAttnScale, layer: index)
        )

        // Feed-forward sub-layer.
        let ffnInput = mixStreams(
            streams,
            readRole: .hcFfnFn,
            scaleRole: .hcFfnScale,
            baseRole: .hcFfnBase,
            layer: index
        )
        let ffnNormed = rmsNorm(ffnInput, role: .ffnNorm, layer: index, epsilon: config.rmsNormEps)
        let ffnOut = feedForward(ffnNormed, layer: index)
        DeepSeekV41HyperConnections.write(
            streams: &streams,
            contribution: ffnOut,
            writeScales: weights.vector(.hcFfnScale, layer: index)
        )
    }

    // MARK: Attention pieces

    private func queryPath(
        _ normed: [Float],
        layer: Int,
        shape: DeepSeekV41AttentionShape
    ) -> [Float] {
        var reduced = normed
        if let down = weights.matrix(.attnWqA, layer: layer, expert: nil) {
            reduced = down.apply(normed)
        }
        if shape.qLoraRank > 0 {
            reduced = rmsNorm(reduced, role: .attnWqANorm, layer: layer, epsilon: config.rmsNormEps)
        }
        guard let up = weights.matrix(.attnWqB, layer: layer, expert: nil) else {
            return [Float](repeating: 0, count: shape.queryWidth)
        }
        return up.apply(reduced)
    }

    private func latentPath(_ normed: [Float], layer: Int) -> [Float] {
        var latent = weights.matrix(.attnWkv, layer: layer, expert: nil)?.apply(normed)
            ?? [Float](repeating: 0, count: shapes.latentWidth)
        latent = rmsNorm(latent, role: .attnKvNorm, layer: layer, epsilon: config.rmsNormEps)
        return latent
    }

    /// `wo_a` is block-diagonal over the head groups, so it is read group by
    /// group; `wo_b` is an ordinary dense projection over the concatenated
    /// groups (`torch.einsum("bsgd,grd->bsgr")` then `o.flatten(2)`).
    private func outputPath(
        _ attended: [Float],
        layer: Int,
        shape: DeepSeekV41AttentionShape
    ) -> [Float] {
        var reduced = attended
        if let down = weights.matrix(.attnWoA, layer: layer, expert: nil) {
            reduced = down.applyBlockDiagonal(attended, groups: shape.outputGroups)
        }
        guard let up = weights.matrix(.attnWoB, layer: layer, expert: nil) else {
            return [Float](repeating: 0, count: shapes.hidden)
        }
        return up.apply(reduced)
    }

    /// Every key the attention stage reads for one position: the live window, then
    /// the compressed rows the addressing rule selects.
    func attentionKeys(layer: DeepSeekV41LayerPlan, position: Int) -> [[Float]] {
        var keys: [[Float]] = []
        let plan = DeepSeekV41CacheAddressing.attentionRowPlan(
            layer: layer,
            position: position,
            config: config,
            plan: self.plan,
            geometry: geometry
        )
        for row in plan.windowPositions {
            guard let written = latent(layer: layer.index, position: row) else { continue }
            keys.append(written)
        }
        guard !plan.compressedRows.isEmpty else { return keys }
        let rows = compressed[layer.compressedKVSource] ?? []
        for row in plan.compressedRows where rows.indices.contains(row) {
            keys.append(rows[row])
        }
        return keys
    }

    /// Records the latent for a position. The reference keeps every position (it is
    /// the oracle, and a test wants to compare any of them); the device cache is the
    /// one that rings.
    private func appendLatent(layer: DeepSeekV41LayerPlan, position: Int, latent: [Float]) {
        let index = layer.index
        if latents.indices.contains(index) {
            latents[index].append(latent)
        } else {
            while latents.count <= index { latents.append([]) }
            latents[index] = [latent]
        }
    }

    /// The post-`attn_norm` block input for a position. The compressor needs it:
    /// `Compressor.forward(x)` projects the attention input, so pooling from the
    /// written latents would feed it a different vector.
    private func appendAttentionInput(layer: Int, position: Int, input: [Float]) {
        if attentionInputs.indices.contains(layer) {
            attentionInputs[layer].append(input)
        } else {
            while attentionInputs.count <= layer { attentionInputs.append([]) }
            attentionInputs[layer] = [input]
        }
    }

    func attentionInput(layer: Int, position: Int) -> [Float]? {
        guard attentionInputs.indices.contains(layer) else { return nil }
        guard attentionInputs[layer].indices.contains(position) else { return nil }
        return attentionInputs[layer][position]
    }

    private func latent(layer: Int, position: Int) -> [Float]? {
        guard latents.indices.contains(layer) else { return nil }
        guard latents[layer].indices.contains(position) else { return nil }
        return latents[layer][position]
    }

    /// Pools a completed block of latents into the owner's compressed row, and the
    /// indexer's block scores at the candidate-source layer.
    private func pooledLatent(
        layer: DeepSeekV41LayerPlan,
        position: Int,
        shape: DeepSeekV41AttentionShape
    ) {
        guard layer.flavor == .compressed, layer.ownsCompressedKV else {
            // A non-owner still contributes candidate scores at the source layer.
            if layer.isCandidateSource { recordCandidateScores(layer: layer, position: position) }
            return
        }
        let ratio = max(1, layer.compressRatio)
        // A compressed row is produced only when a whole block has been seen, i.e.
        // exactly on the positions where `position + 1` is a multiple of the ratio.
        // Pooling on every later position would append the same row again and let
        // the cache grow with the context instead of with the context / ratio.
        guard (position + 1) % ratio == 0 else { return }
        let rows = DeepSeekV41CacheAddressing.compressedRowCount(afterPosition: position, ratio: ratio)
        let completed = DeepSeekV41CacheAddressing.poolRange(row: rows - 1, ratio: ratio)
        guard rows > 0, completed.upperBound <= position + 1 else { return }
        // The compressor projects the layer's post-`attn_norm` input, not the
        // latents already in the window ring: `Compressor.forward(x)` takes the
        // attention input, and pooling the written records would re-use a vector
        // that already carries this layer's own `wkv` + rope.
        var inputs: [[Float]] = []
        inputs.reserveCapacity(completed.count)
        for row in completed {
            guard let input = attentionInput(layer: layer.index, position: row) else { return }
            inputs.append(input)
        }
        var pooled = DeepSeekV41Compressor.pooledLatent(
            inputs: inputs,
            kvWeight: weights.matrix(.compressorWkv, layer: layer.index),
            gateWeight: ratio > 1 ? weights.matrix(.compressorWgate, layer: layer.index) : nil
        )
        // `Compressor.forward` returns `norm(kv)`, pre-rope; `_compress_kv` rotates
        // it afterwards, at the position of the FIRST token of its group.
        pooled = rmsNorm(
            pooled, role: .compressorNorm, layer: layer.index, epsilon: config.rmsNormEps
        )
        DeepSeekV41RoPE.rotateSlice(
            &pooled,
            start: shape.latentRopeStart,
            length: shape.latentRopeWidth,
            position: max(0, rows - 1) * ratio,
            frequencies: latentFrequencyTable[layer.ropeTable] ?? []
        )
        compressed[layer.index, default: []].append(pooled)
        if layer.isCandidateSource { recordCandidateScores(layer: layer, position: position) }
    }

    private func recordCandidateScores(layer: DeepSeekV41LayerPlan, position: Int) {
        let rows = compressed[layer.compressedKVSource] ?? []
        guard !rows.isEmpty else { return }
        let queries = [Array(rows[rows.count - 1].prefix(max(1, config.indexHeadDim)))]
        let scores = DeepSeekV41Indexer.blockScores(
            queries: queries,
            keys: rows,
            blockSize: max(1, config.candidateBlockSize),
            keyWidth: max(1, config.indexHeadDim)
        )
        candidateScores.append(scores)
    }

    // MARK: FFN

    private func feedForward(_ normed: [Float], layer: Int) -> [Float] {
        let limit = config.swigluLimit ?? 0
        var out = [Float](repeating: 0, count: shapes.hidden)

        if let routing = route(normed, layer: layer) {
            for (rank, expert) in routing.expertIDs.enumerated() {
                guard let w1 = weights.matrix(.expertW1, layer: layer, expert: expert),
                      let w3 = weights.matrix(.expertW3, layer: layer, expert: expert),
                      let w2 = weights.matrix(.expertW2, layer: layer, expert: expert) else { continue }
                let activated = DeepSeekV41Activation.swiglu(
                    gates: w1.apply(normed),
                    ups: w3.apply(normed),
                    limit: limit
                )
                let projected = w2.apply(activated)
                let weight = rank < routing.weights.count ? routing.weights[rank] : 0
                for dim in 0..<min(out.count, projected.count) {
                    out[dim] += weight * projected[dim]
                }
            }
        }

        // The shared expert runs for every token, unscaled, when the checkpoint
        // declares one.
        if config.sharedExpertCount > 0,
           let s1 = weights.matrix(.sharedExpertW1, layer: layer),
           let s3 = weights.matrix(.sharedExpertW3, layer: layer),
           let s2 = weights.matrix(.sharedExpertW2, layer: layer) {
            let activated = DeepSeekV41Activation.swiglu(
                gates: s1.apply(normed),
                ups: s3.apply(normed),
                limit: limit
            )
            let projected = s2.apply(activated)
            for dim in 0..<min(out.count, projected.count) {
                out[dim] += projected[dim]
            }
        }
        return out
    }

    func route(_ normed: [Float], layer: Int) -> DeepSeekV41Routing? {
        guard let gate = weights.matrix(.routerWeight, layer: layer) else { return nil }
        let logits = gate.apply(normed)
        let bias = weights.vector(.routerBias, layer: layer)
        return DeepSeekV41Router.route(
            logits: logits,
            bias: bias,
            topK: config.numActivatedExperts,
            scalingFactor: config.routedScalingFactor,
            normalise: config.normalizeTopKProb,
            scoringFunction: config.scoringFunction,
            gateTemp: config.gateTemp
        )
    }

    /// Writes one position's Engram lookup into `streams`.
    ///
    /// The lookup does the hashing and the row IO; this only assembles what the
    /// projection reads. A column with no row is zero-filled rather than skipped:
    /// the concatenation is positional, so a missing row is a zero row in its own
    /// slot, and dropping it would shift every later column.
    private func writeEngram(layer: Int, table: Int, streams: inout [[Float]]) {
        guard let engramLookup else { return }
        // `Engram.forward` zeroes the gate wherever `token_mask` is false, so a dead
        // position (an image span) passes through untouched rather than reading pad
        // rows into the stream.
        guard deadTokens.last != true else { return }
        let spec = config.engramTables[table]
        // `(max_ngram_size - 1) * n_heads` hash columns, one row each.
        let columns = max(1, spec.maxNgramSize - 1) * spec.headCount
        guard let retrieved = try? engramLookup.engramRows(
            layer: layer, history: tokenHistory, dead: deadTokens
        ), !retrieved.isEmpty else { return }
        var rows = retrieved
        if rows.count < columns {
            rows.append(contentsOf: Array(
                repeating: [Float](repeating: 0, count: spec.headDimension),
                count: columns - rows.count
            ))
        }
        guard let contribution = DeepSeekV41EngramCombination.contribution(
            residual: streams,
            rows: rows,
            wkv: weights.matrix(.engramWkv, layer: layer),
            queryWeight: weights.vector(.engramQuery, layer: layer),
            keyWeight: weights.vector(.engramKey, layer: layer),
            epsilon: config.rmsNormEps
        ) else { return }
        streams = contribution.streams
        if engramGates[layer] == nil { engramGates[layer] = [] }
        engramGates[layer]?.append(contribution.gates)
    }

    // MARK: Streams

    private func mixStreams(
        _ streams: [[Float]],
        readRole: DeepSeekV41TensorRole,
        scaleRole: DeepSeekV41TensorRole,
        baseRole: DeepSeekV41TensorRole,
        layer: Int
    ) -> [Float] {
        let mult = streams.count
        let readRaw = weights.matrix(readRole, layer: layer)
        let read = readRaw.map { resize($0.row(0), to: mult) }
            ?? [Float](repeating: 1 / Float(max(1, mult)), count: mult)
        let scale = weights.vector(scaleRole, layer: layer)
        let bias = weights.vector(baseRole, layer: layer)
        return DeepSeekV41HyperConnections.mix(
            streams: streams,
            readWeights: read,
            scale: scale,
            bias: bias
        )
    }

    private func collapse(_ streams: [[Float]]) -> [Float] {
        streams.first ?? []
    }

    private func resize(_ values: [Float], to count: Int) -> [Float] {
        guard values.count != count else { return values }
        if values.count > count { return Array(values.prefix(count)) }
        return values + [Float](repeating: 0, count: count - values.count)
    }

    // MARK: Norm

    /// RMSNorm with the checkpoint's per-channel gain.
    func rmsNorm(
        _ input: [Float],
        role: DeepSeekV41TensorRole,
        layer: Int?,
        epsilon: Float
    ) -> [Float] {
        guard !input.isEmpty else { return [] }
        var sumSquares: Float = 0
        for value in input { sumSquares += value * value }
        let rms = 1 / sqrt(sumSquares / Float(input.count) + epsilon)
        let gain = weights.vector(role, layer: layer)
        var out = [Float](repeating: 0, count: input.count)
        for index in input.indices {
            let scale = (gain?.count ?? 0) == input.count ? gain![index] : 1
            out[index] = input[index] * rms * scale
        }
        return out
    }
}
