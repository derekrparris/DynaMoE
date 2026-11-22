//
//  Inkling2Forward.swift
//  DynaMoE
//
//  The host reference graph for Inkling 2 Mini, shared by tests and by the
//  device comparison.
//
//  One control flow drives everything: `runVisit` executes ONE visit of the
//  48-visit schedule, and the loop around it is `Inkling2LoopTextModel.forward`
//  transcribed line for line -- pre-loop layers, the first core pass, the two
//  message-carrying passes that restart from the pre-loop state, then the
//  post-loop layers and the final residual read.
//
//  Two details of that loop are easy to "clean up" and must not be:
//
//    * At a block boundary the layer appends its INPUT to the residual stack
//      (before the MLP read), and the loop then appends the same input AGAIN
//      (it appends `hidden_states` before assigning the layer's output). Two
//      entries per boundary layer, and the stack's last entry equals the
//      vector the next layer also passes as its current partial.
//    * A pass restart slices `block_states[num_pre_loop_blocks + 1:]`, so the
//      message read sees the previous pass's stack minus its first entry.
//
//  Everything numeric is FP32, because that is what the reference keeps in
//  FP32: the convolutions, the block reads, the message gates, the router, the
//  mixture accumulation, the L2 clips and the final logits. The HF-reference
//  numerical profile rounds each dequantized expert weight to BF16, exactly
//  what Transformers does when it loads this checkpoint without a quantization
//  backend (see `Inkling2PackedExpertMatrix.bf16Row`).
//
//  This file is not wired to Metal. A later phase runs the same per-visit graph
//  on the device and is checked against this one; a CPU reference that is only
//  ever compared with itself proves nothing.
//

import Foundation

// MARK: - Weight source

/// Everything one visit reads. A snapshot-backed source serves the real
/// checkpoint; a synthetic fixture serves tests. Routed experts are addressed
/// one (layer, expert) at a time because their banks are fused.
nonisolated public protocol Inkling2WeightSource {
    /// A 2-D weight `[rows][columns]`, dequantized to FP32, or a 1-D vector.
    func matrix(_ role: Inkling2TensorRole, layer: Int?) throws -> [Float]
    func shape(_ role: Inkling2TensorRole, layer: Int?) throws -> [Int]
    /// One expert's packed bank (weight bytes, scales, amax).
    func expertMatrix(_ role: Inkling2TensorRole, layer: Int, expert: Int) throws -> Inkling2PackedExpertMatrix
    /// One row of a 2-D tensor. A REQUIREMENT, not an extension-only helper: a
    /// call through the protocol dispatches statically to an extension method,
    /// so a source with an efficient override would never be asked for it -- and
    /// the default reads the whole table.
    func row(_ role: Inkling2TensorRole, layer: Int?, index: Int) throws -> [Float]
}

public extension Inkling2WeightSource {
    /// The default slices `matrix`; a source backed by a file overrides it to
    /// read the row instead of the whole table, which is what makes a per-token
    /// embedding read cost 9 KB rather than 900 MB.
    func row(_ role: Inkling2TensorRole, layer: Int?, index: Int) throws -> [Float] {
        let tensorShape = try shape(role, layer: layer)
        guard tensorShape.count == 2, index >= 0, index < tensorShape[0] else {
            throw Inkling2ForwardError.shapeMismatch("row \(index) of \(tensorShape) is out of range")
        }
        let width = tensorShape[1]
        let all = try matrix(role, layer: layer)
        guard all.count >= (index + 1) * width else {
            throw Inkling2ForwardError.shapeMismatch("matrix is shorter than \(tensorShape)")
        }
        return Array(all[(index * width)..<((index + 1) * width)])
    }
}

nonisolated public enum Inkling2ForwardError: Error, Equatable, CustomStringConvertible {
    case missingWeight(String)
    case shapeMismatch(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .missingWeight(let name):
            return "Inkling 2 forward is missing weight '\(name)'"
        case .shapeMismatch(let detail):
            return "Inkling 2 forward shape mismatch: \(detail)"
        case .unsupported(let detail):
            return "Inkling 2 forward cannot run: \(detail)"
        }
    }
}

// MARK: - Options

nonisolated public struct Inkling2ForwardOptions: Sendable, Equatable {
    public var activationProfile: Inkling2ActivationProfile
    /// Positions the sequence may hold, or nil for the checkpoint's declared
    /// maximum. A full-attention visit keeps every position's K/V, so this is
    /// what turns a KV BUDGET (see `Inkling2MemoryPlan`) into a refusal by name
    /// instead of a process that grows until it is killed. A ring visit is not
    /// affected: it holds at most its window.
    public var maxPositions: Int?

    public init(activationProfile: Inkling2ActivationProfile = .hfReference, maxPositions: Int? = nil) {
        self.activationProfile = activationProfile
        self.maxPositions = maxPositions
    }
}

// MARK: - The host graph

nonisolated public final class Inkling2CPUForward {
    public let config: Inkling2Config
    public let plan: [Inkling2VisitPlan]
    public let weights: Inkling2WeightSource
    public let options: Inkling2ForwardOptions
    /// The visit `runVisit` is currently executing, so the attention and mixture
    /// helpers can label their captures without threading it through.
    private var visitIndexForDebug = 0
    /// The loop pass `runVisit` is executing, for the captures that must name
    /// the LAST pass (the reference's message dump is the final pass's).
    private var passForDebug = 0
    private var currentPosition = 0
    /// Every visit's KV and convolution state, plus the token history.
    public let sequence: Inkling2SequenceState
    /// One state bundle per visit, indexed by `cacheSlot`.
    public var states: [Inkling2VisitState] { sequence.visits }
    /// Called after each visit completes, for the golden comparison. Off in
    /// production: it is a diagnostic seam, like DeepSeek's parity probes.
    public var onVisitComplete: ((_ visitIndex: Int, _ layer: Int, _ pass: Int, _ output: [Float]) -> Void)?
    /// Called with the raw attention output and the raw mixture output of each
    /// visit, for the sub-stage comparison. Off in production.
    public var onVisitSubStages: ((_ visitIndex: Int, _ attention: [Float], _ mixture: [Float]) -> Void)?
    /// Called with the post-attention convolution of each visit, i.e. the
    /// attention half's final value before the residual stack.
    public var onPostAttention: ((_ visitIndex: Int, _ value: [Float]) -> Void)?
    /// Called with named intermediates of one visit, at the same points the
    /// reference exposes as modules. Off in production; used to bisect a
    /// mismatch inside a layer instead of only at its output.
    public var onInternal: ((_ visitIndex: Int, _ name: String, _ value: [Float]) -> Void)?
    /// Runs one visit in place of `runVisit`, or nil for the host graph.
    ///
    /// The loop SCHEDULE (block boundaries, the loop passes, the previous
    /// pass's stack) is subtle enough that it must have one copy, so an
    /// executor substitutes a VISIT rather than a token: `forwardHidden` keeps
    /// deciding what to run, and this decides how. A substitution must leave the
    /// visit state exactly where the host would -- the KV rows, the four
    /// convolution histories -- or a later visit reads a different sequence.
    public typealias VisitRunner = (
        _ layer: Int, _ pass: Int, _ hidden: [Float], _ blockStates: inout [[Float]],
        _ messages: [[Float]]?, _ position: Int
    ) throws -> (output: [Float], attentionQueries: Int)
    public var visitRunner: VisitRunner?

    public init(
        config: Inkling2Config,
        weights: Inkling2WeightSource,
        options: Inkling2ForwardOptions = Inkling2ForwardOptions()
    ) throws {
        self.config = config
        self.plan = try config.makeVisitPlan()
        self.weights = weights
        self.options = options
        let kvWidth = config.numKeyValueHeads * config.headDim
        self.sequence = Inkling2SequenceState(
            plan: plan,
            window: config.slidingWindowSize,
            kvWidth: kvWidth,
            hidden: config.hiddenSize,
            historyDepth: max(0, config.convKernelSize - 1),
            maxPositions: options.maxPositions ?? config.maxPositionEmbeddings
        )
    }

    /// Clears every visit's KV and convolution state: a new sequence.
    public func reset() {
        sequence.reset()
    }

    /// Runs one token at `position`, returning the unpadded logits.
    public func forwardToken(token: UInt32, position: Int) throws -> [Float] {
        let row = Int(token)
        let table = try weights.shape(.tokenEmbedding, layer: nil)
        guard table.count == 2, row >= 0, row < table[0] else {
            throw Inkling2ForwardError.shapeMismatch("token \(token) is outside the embedding table")
        }
        var hidden = try weights.row(.tokenEmbedding, layer: nil, index: row)
        hidden = Inkling2Math.rmsNorm(
            hidden, gain: try weights.matrix(.embeddingNorm, layer: nil), eps: config.rmsNormEps
        )
        let logits = try forwardHidden(hidden, position: position)
        sequence.record(token: row)
        return logits
    }

    // MARK: The schedule

    /// The 48-visit schedule over one position's embedding, then the final
    /// residual read, norm, unembedding and softcap.
    public func forwardHidden(_ embedding: [Float], position: Int) throws -> [Float] {
        currentPosition = position
        var hidden = embedding
        // The attention-residual stack: one entry per append, as the reference
        // builds it (a boundary layer contributes its input twice).
        var blockStates: [[Float]] = []
        let coreStart = config.loopCoreStart
        let coreEnd = config.loopCoreEnd
        let block = config.nLayersPerAttnResBlock

        for layer in 0..<coreStart {
            let (output, _) = try runVisitOrSubstitute(
                layer: layer, pass: 0, hidden: hidden, blockStates: &blockStates,
                messages: nil, position: position
            )
            // The loop appends the layer's INPUT at a boundary, BEFORE assigning
            // its output: `if decoder_layer.is_block_boundary: block_states =
            // cat(...); hidden_states = new_hidden_states`. The layer's own
            // append inside the MLP read is local and never reaches the caller.
            if layer % block == 0 { blockStates.append(hidden) }
            hidden = output
        }
        let preLoopHidden = hidden
        let preLoopBlocks = blockStates
        let preLoopBlockCount = blockStates.count

        for layer in coreStart..<coreEnd {
            let (output, _) = try runVisitOrSubstitute(
                layer: layer, pass: 0, hidden: hidden, blockStates: &blockStates,
                messages: nil, position: position
            )
            if layer % block == 0 { blockStates.append(hidden) }
            hidden = output
        }

        onInternal?(0, "endOfPassOne@\(position)", Array(hidden.prefix(4)))
        for pass in 1..<config.numPasses {
            // Both are read from the PREVIOUS pass's leftovers, so they are
            // recomputed at the top of every pass, not once before the loop:
            // `previous_block_states = block_states[..., num_pre_loop_blocks + 1:, :]`
            // and `previous_hidden_states = hidden_states`, both before the reset.
            let previousBlocks = Array(blockStates.dropFirst(preLoopBlockCount + 1))
            if pass == 1, let onInternal {
                onInternal(0, "loop_stackCount", [Float(blockStates.count), Float(preLoopBlockCount), Float(previousBlocks.count)])
            }
            let previousHidden = hidden
            onInternal?(0, "previousHidden@\(position)", Array(previousHidden.prefix(4)))
            hidden = preLoopHidden
            blockStates = preLoopBlocks
            for layer in coreStart..<coreEnd {
                let coreIndex = layer - coreStart
                let messages = try messageInput(
                    blockStates: previousBlocks, partial: previousHidden,
                    coreIndex: coreIndex, pass: pass
                )
                let (output, _) = try runVisitOrSubstitute(
                    layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                    messages: messages, position: position
                )
                if layer % block == 0 { blockStates.append(hidden) }
                hidden = output
            }
        }

        for layer in coreEnd..<config.numHiddenLayers {
            let (output, _) = try runVisitOrSubstitute(
                layer: layer, pass: 0, hidden: hidden, blockStates: &blockStates,
                messages: nil, position: position
            )
            if layer % block == 0 { blockStates.append(hidden) }
            hidden = output
        }

        let outputQuery = try weights.matrix(.attnResOutputQuery, layer: nil)
        hidden = blockRead(blockStates: blockStates, partial: hidden, queries: [outputQuery])[0]
        // The read over the WHOLE stack, which is what makes it the check on the
        // stack's contents and order: one wrong entry, one entry too many, or a
        // boundary layer appended after its output instead of before all move
        // this vector and nothing else that is named.
        onInternal?(0, "finalRead", hidden)
        hidden = Inkling2Math.rmsNorm(
            hidden, gain: try weights.matrix(.finalNorm, layer: nil), eps: config.rmsNormEps
        )
        let head = try weights.matrix(.lmHead, layer: nil)
        let unpadded = config.unpaddedVocabSize
        var logits = [Float](repeating: 0, count: unpadded)
        for row in 0..<unpadded {
            var total: Float = 0
            let base = row * config.hiddenSize
            for column in 0..<config.hiddenSize { total += head[base + column] * hidden[column] }
            logits[row] = total
        }
        return Inkling2Math.finalLogitSoftcap(logits, softcap: config.finalLogitSoftcapping)
    }

    /// One visit, through the installed executor when there is one.
    func runVisitOrSubstitute(
        layer: Int,
        pass: Int,
        hidden: [Float],
        blockStates: inout [[Float]],
        messages: [[Float]]?,
        position: Int
    ) throws -> (output: [Float], attentionQueries: Int) {
        if let visitRunner {
            return try visitRunner(layer, pass, hidden, &blockStates, messages, position)
        }
        return try runVisit(
            layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
            messages: messages, position: position
        )
    }

    /// One visit. Returns its output and the block-read query count, which the
    /// tests use to check the packed-query unpacking.
    ///
    /// The order is the reference's, and the two appends are its own:
    ///   1. attention input read (layer > 0 only; 3 queries on a full layer)
    ///   2. attention, with per-kind Q/K/V input norms
    ///   3. post-attention convolution
    ///   4. residual: at a boundary append the layer's INPUT to the stack and
    ///      keep only the convolution output; otherwise add the input
    ///   5. MLP input read over the stack plus that residual
    ///   6. mixture, post-MLP convolution, add
    @discardableResult
    public func runVisit(
        layer: Int,
        pass: Int,
        hidden initial: [Float],
        blockStates: inout [[Float]],
        messages: [[Float]]?,
        position: Int
    ) throws -> (output: [Float], attentionQueries: Int) {
        let visitIndex = try visitIndex(layer: layer, pass: pass)
        visitIndexForDebug = visitIndex
        passForDebug = pass
        let state = states[visitIndex]
        let isFull = !config.isLocalLayer(layer)
        let block = config.nLayersPerAttnResBlock
        let blockResidual = initial

        var hidden = initial
        var attentionQueries = 0
        // A full-attention layer reads the residual stack THREE times (one query
        // each for Q, K and V) and its three projections then take those reads
        // separately; a sliding layer reads it once and shares the result.
        var qRead = hidden
        var kRead = hidden
        var vRead = hidden
        if layer > 0 {
            let packed = try weights.matrix(.attnResAttnQuery, layer: layer - 1)
            let queries = unpackRows(packed, count: isFull ? 3 : 1)
            attentionQueries = queries.count
            let reads = blockRead(blockStates: blockStates, partial: hidden, queries: queries)
            qRead = reads[0]
            kRead = isFull && reads.count > 1 ? reads[1] : reads[0]
            vRead = isFull && reads.count > 2 ? reads[2] : reads[0]
        }
        hidden = qRead
        onInternal?(visitIndex, "attn_input", hidden)

        // The message gates act on the block-read result, BEFORE the input
        // norms: the reference writes `self.input_layernorm(self.q_message_gate(
        // query_hidden_states, query_message))`, so a gate applied after the norm
        // sees different inputs.
        var queryRead = qRead
        var keyRead = kRead
        var valueRead = vRead
        if let messages, messages.count == 4 {
            queryRead = try messageGate(role: .controllerAttnDown, layer: layer, current: queryRead, message: messages[0])
            keyRead = try messageGate(role: .controllerKDown, layer: layer, current: keyRead, message: messages[1])
            valueRead = try messageGate(role: .controllerVDown, layer: layer, current: valueRead, message: messages[2])
        }

        // The three projections' inputs, each normalized differently.
        let queryInput: [Float]
        let keyInput: [Float]
        let valueInput: [Float]
        if isFull {
            queryInput = Inkling2Math.rmsNorm(queryRead, gain: try weights.matrix(.qInputNorm, layer: layer), eps: config.rmsNormEps)
            keyInput = Inkling2Math.rmsNorm(keyRead, gain: try weights.matrix(.kInputNorm, layer: layer), eps: config.rmsNormEps)
            valueInput = Inkling2Math.rmsNorm(valueRead, gain: try weights.matrix(.vInputNorm, layer: layer), eps: config.rmsNormEps)
        } else {
            // One shared norm on a sliding layer, but applied to each gated value
            // SEPARATELY: the reference writes three `self.input_layernorm(...)`
            // calls, and with messages the three gates make the three inputs
            // different. Sharing one result was invisible without messages and
            // wrong with them.
            let gain = try weights.matrix(.attnNorm, layer: layer)
            queryInput = Inkling2Math.rmsNorm(queryRead, gain: gain, eps: config.rmsNormEps)
            keyInput = Inkling2Math.rmsNorm(keyRead, gain: gain, eps: config.rmsNormEps)
            valueInput = Inkling2Math.rmsNorm(valueRead, gain: gain, eps: config.rmsNormEps)
        }

        let attentionOutput = try attention(
            layer: layer, state: state, queryInput: queryInput,
            keyInput: keyInput, valueInput: valueInput, position: position
        )
        let postAttention = try shortConvolution(layer: layer, state: state, index: 2, values: attentionOutput)
        onPostAttention?(visitIndex, postAttention)
        onInternal?(visitIndex, "postAttention", postAttention)

        // The boundary check is on `blockResidual`, the layer's INPUT, and the
        // append is LOCAL: the reference appends it inside the layer to read the
        // MLP input, and the persistent stack entry is the loop's own append
        // after the layer returns. Appending to the caller's stack here would
        // count every boundary layer twice.
        // The residual the mixture is added back to is `block_residual +
        // postAttention` on an ordinary layer, and the post-attention value
        // ALONE at a block boundary (the boundary's block input is what gets
        // stacked instead). The layer's own append is local to this MLP read:
        // the persistent stack entry is the loop's, after the layer returns.
        var residual = postAttention
        var mlpStack = blockStates
        if layer % block == 0 {
            mlpStack.append(blockResidual)
        } else {
            residual = zip(blockResidual, postAttention).map(+)
        }

        let mlpPacked = try weights.matrix(.attnResMlpQuery, layer: layer)
        var mlpHidden = blockRead(blockStates: mlpStack, partial: residual, queries: [mlpPacked])[0]
        if let messages, messages.count == 4 {
            mlpHidden = try messageGate(role: .controllerMlpDown, layer: layer, current: mlpHidden, message: messages[3])
        }
        onInternal?(visitIndex, "mlp_input", mlpHidden)
        let mixtureInput = Inkling2Math.rmsNorm(mlpHidden, gain: try weights.matrix(.mlpNorm, layer: layer), eps: config.rmsNormEps)
        var mixture = try mixture(hidden: mixtureInput, layer: layer)
        onInternal?(visitIndex, "mixtureOut", mixture)

        onVisitSubStages?(visitIndex, attentionOutput, mixture)
        mixture = try shortConvolution(layer: layer, state: state, index: 3, values: mixture)
        onInternal?(visitIndex, "postMlpConv", mixture)
        onInternal?(visitIndex, "residual", residual)
        let output = zip(residual, mixture).map(+)
        if visitIndex == 0 {
            onInternal?(visitIndex, "add_residual", Array(residual.prefix(4)))
            onInternal?(visitIndex, "add_mixture", Array(mixture.prefix(4)))
            onInternal?(visitIndex, "add_sum", Array(output.prefix(4)))
        }
        onVisitComplete?(visitIndex, layer, pass, output)
        return (output, attentionQueries)
    }

    public func isCore(_ layer: Int) -> Bool {
        layer >= config.loopCoreStart && layer < config.loopCoreEnd
    }

    /// The visit's cache slot for a (layer, pass) pair. A non-core layer appears
    /// once; a core layer once per pass, in order.
    public func visitIndex(layer: Int, pass: Int) throws -> Int {
        let wantedPass = isCore(layer) ? pass + 1 : 0
        let matches = plan.filter { $0.physicalLayer == layer && $0.pass == wantedPass }
        guard !matches.isEmpty else {
            throw Inkling2ForwardError.unsupported("no visit for layer \(layer) pass \(pass)")
        }
        return matches[0].cacheSlot
    }

    // MARK: Residual stack

    /// The block-attention residual read: FP32 softmax of the queries over the
    /// RMS-normalized stack plus the current partial, `1/sqrt(hidden)` scaled.
    public func blockRead(blockStates: [[Float]], partial: [Float], queries: [[Float]]) -> [[Float]] {
        Inkling2Math.blockResidualRead(blockStates: blockStates, partial: partial, queries: queries)
    }

    func unpackRows(_ tensor: [Float], count: Int) -> [[Float]] {
        let width = config.hiddenSize
        guard count > 1 else { return [tensor] }
        guard tensor.count >= count * width else { return [tensor] }
        return (0..<count).map { row in
            Array(tensor[(row * width)..<((row + 1) * width)])
        }
    }

    // MARK: Message gates

    /// One message gate: `(1 - tanh(forget)) * current + tanh(gain) * message`,
    /// where both projections read a rank-64 bottleneck of
    /// `[norm(current); norm(message)]`.
    public func messageGate(
        role: Inkling2TensorRole,
        layer: Int,
        current: [Float],
        message: [Float]
    ) throws -> [Float] {
        let eps = config.rmsNormEps
        let joined = Inkling2Math.rmsNorm(current, eps: eps) + Inkling2Math.rmsNorm(message, eps: eps)
        let down = try weights.matrix(role, layer: layer)
        let gate = try linear(down, joined, rows: config.gateRank)
        let forget = try linear(try weights.matrix(forgetRole(for: role), layer: layer), gate, rows: config.hiddenSize)
        let gain = try linear(try weights.matrix(gainRole(for: role), layer: layer), gate, rows: config.hiddenSize)
        return Inkling2Math.messageGate(current: current, message: message, forget: forget, gain: gain)
    }

    func forgetRole(for role: Inkling2TensorRole) -> Inkling2TensorRole {
        switch role {
        case .controllerAttnDown: return .controllerAttnForget
        case .controllerKDown: return .controllerKForget
        case .controllerVDown: return .controllerVForget
        case .controllerMlpDown: return .controllerMlpForget
        default: return role
        }
    }

    func gainRole(for role: Inkling2TensorRole) -> Inkling2TensorRole {
        switch role {
        case .controllerAttnDown: return .controllerAttnGain
        case .controllerKDown: return .controllerKGain
        case .controllerVDown: return .controllerVGain
        case .controllerMlpDown: return .controllerMlpGain
        default: return role
        }
    }

    /// `y = W x` for a `[rows][input.count]` matrix.
    public func linear(_ matrix: [Float], _ input: [Float], rows: Int) throws -> [Float] {
        guard matrix.count >= rows * input.count else {
            throw Inkling2ForwardError.shapeMismatch("linear has \(matrix.count) values for \(rows) x \(input.count)")
        }
        var output = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var total: Float = 0
            let base = row * input.count
            for column in 0..<input.count { total += matrix[base + column] * input[column] }
            output[row] = total
        }
        return output
    }

    /// Four messages for one core layer, from the previous pass's stack slice
    /// plus its final hidden state.
    /// `pass` is the loop's own 0-based counter, passed IN rather than read from
    /// `passForDebug`: this runs before `runVisit`, so the field still holds the
    /// PREVIOUS visit's pass and a capture keyed on it is tagged one pass late.
    public func messageInput(
        blockStates: [[Float]], partial: [Float], coreIndex: Int, pass: Int
    ) throws -> [[Float]] {
        let unpacking = Inkling2QueryUnpacking.CrossPassQueryRows(coreLayer: coreIndex)
        let main = try weights.matrix(.crossPassQuery, layer: nil)
        let kv = try weights.matrix(.crossPassKVQuery, layer: nil)
        let width = config.hiddenSize
        let queries: [[Float]] = unpacking.rows.enumerated().map { offset, row in
            let source = unpacking.fromMainTensor[offset] ? main : kv
            return Array(source[(row * width)..<((row + 1) * width)])
        }
        let read = blockRead(blockStates: blockStates, partial: partial, queries: queries)
        if let onInternal {
            // A read happens once per core layer per pass PER POSITION, so the
            // label carries the position and the PASS: naming the read by the
            // core layer alone made pass 1's and pass 2's reads share a key,
            // and the later one silently overwrote the earlier.
            //
            // The visit index is deliberately NOT in the key: this runs before
            // `runVisit`, so `visitIndexForDebug` still holds the PREVIOUS
            // visit's index and a visit-keyed capture is filed one visit late.
            // The key names the layer, the channel and the pass, which is what
            // a read is.
            let visit = 0
            if coreIndex == 0 {
                onInternal(visit, "message0@\(currentPosition)", read[0])
                onInternal(visit, "message0_stackCount@\(currentPosition)", [Float(blockStates.count)])
                onInternal(visit, "message0_partial@\(currentPosition)", Array(partial.prefix(4)))
            }
            // Every channel of every core layer, on the LAST pass: the
            // reference's `messages` dump is the final pass's four reads per
            // core layer, in layer order, so this is the only pass whose
            // values can be compared with it (an earlier pass reads a shorter
            // stack). Keyed by layer and channel, not by visit.
            //
            // `passForDebug` is the LOOP's counter (0-based); the plan's `pass`
            // is 1-based for core layers, which is why the two numbers do not
            // match and must not be used interchangeably.
            if config.numPasses > 1 {
                for channel in read.indices {
                    onInternal(visit, "message\(coreIndex)_\(channel)_p\(pass)", read[channel])
                }
            }
        }
        return read
    }

    // MARK: Convolutions

    /// The static short convolution `(conv4(x) + x) / sqrt(5)`, feeding its
    /// input into the visit's history.
    public func shortConvolution(
        layer: Int,
        state: Inkling2VisitState,
        index: Int,
        values: [Float]
    ) throws -> [Float] {
        try shortConvolution(layer: layer, state: state, index: index, values: values,
                             weightsRole: index == 2 ? .attnSconv : .mlpSconv)
    }

    /// The static convolution over a K/V projection or the post-attention /
    /// post-MLP stream, whose weight lives under its own role.
    ///
    /// The stored `conv1d` weight is in `F.conv1d` order -- oldest tap first --
    /// while `Inkling2Math.staticShortConvolution` takes backward-distance taps,
    /// so it is REVERSED once here. Per-channel tap and history vectors are built
    /// once per channel rather than re-derived inside the tap loop.
    func shortConvolution(
        layer: Int,
        state: Inkling2VisitState,
        index: Int,
        values: [Float],
        weightsRole: Inkling2TensorRole
    ) throws -> [Float] {
        let kernel = config.convKernelSize
        let taps = try weights.matrix(weightsRole, layer: layer)
        let history = state.history(index)
        var output = [Float](repeating: 0, count: values.count)
        for channel in 0..<values.count {
            var channelTaps = [Float](repeating: 0, count: kernel)
            for tap in 0..<kernel {
                channelTaps[tap] = taps[channel * kernel + (kernel - 1 - tap)]
            }
            let channelHistory = history.map { $0[channel] }
            output[channel] = Inkling2Math.staticShortConvolution(
                current: values[channel], history: channelHistory, weights: channelTaps
            )
        }
        state.pushConvolution(index, value: values)
        return output
    }

    /// The dynamic K/V convolution of a full-attention layer: static taps plus
    /// per-token taps projected from the projection's input, shared within each
    /// KV head.
    public func dynamicConvolution(
        state: Inkling2VisitState,
        index: Int,
        values: [Float],
        dynamicInput: [Float],
        staticTaps: [Float],
        projected: [Float],
        kvHeads: Int,
        headDim: Int
    ) -> [Float] {
        let kernel = config.convKernelSize
        let history = state.history(index)
        var output = [Float](repeating: 0, count: values.count)
        for channel in 0..<values.count {
            let head = channel / max(1, headDim)
            let staticWeights = (0..<kernel).map { staticTaps[$0 * values.count + channel] }
            let dynamicWeights = (0..<kernel).map { tap in
                projected[(head * kernel + tap)]
            }
            let channelHistory = history.map { $0[channel] }
            output[channel] = Inkling2Math.dynamicShortConvolution(
                current: values[channel], history: channelHistory,
                staticWeights: staticWeights, dynamicWeights: dynamicWeights
            )
        }
        _ = dynamicInput
        _ = kvHeads
        state.pushConvolution(index, value: values)
        return output
    }

    // MARK: Attention

    /// The full attention block: Q/K/V projections with their norms and
    /// convolutions, relative bias, sink, softmax and the gated output.
    public func attention(
        layer: Int,
        state: Inkling2VisitState,
        queryInput: [Float],
        keyInput: [Float],
        valueInput: [Float],
        position: Int
    ) throws -> [Float] {
        let isFull = !config.isLocalLayer(layer)
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let kvHeads = isFull ? config.numKeyValueHeads : config.swaNumKeyValueHeads
        let headDim = isFull ? config.headDim : config.swaHeadDim
        let groups = heads / kvHeads
        let eps = config.rmsNormEps

        // Q: project, split, learned scale, L2 clip, then log scale on full.
        let wq = try weights.matrix(.attnWqDu, layer: layer)
        var query = try linear(wq, queryInput, rows: heads * headDim)
        onInternal?(visitIndexForDebug, "q_linear", query)
        let qScale = try weights.matrix(.attnQL2ClipScale, layer: layer)
        for head in 0..<heads {
            var slice = Array(query[(head * headDim)..<((head + 1) * headDim)])
            for dimension in 0..<headDim {
                slice[dimension] *= qScale[head * headDim + dimension]
            }
            slice = Inkling2Math.l2Clip(slice, threshold: config.qkL2ClipThreshold)
            if isFull {
                let tau = Inkling2Math.logScale(position: position, alpha: config.logScalingAlpha, floor: config.logScalingNFloor)
                slice = slice.map { $0 * tau }
            }
            for dimension in 0..<headDim { query[head * headDim + dimension] = slice[dimension] }
        }
        onInternal?(visitIndexForDebug, "q_norm", query)

        // K: project, convolve, L2 clip.
        let wk = try weights.matrix(.attnWkDv, layer: layer)
        var key = try linear(wk, keyInput, rows: kvHeads * headDim)
        onInternal?(visitIndexForDebug, "k_linear", key)
        if isFull {
            let staticTaps = try weights.matrix(.attnKSconvStatic, layer: layer)
            let projected = try linear(try weights.matrix(.attnKSconvProj, layer: layer), keyInput, rows: kvHeads * config.convKernelSize)
            key = dynamicConvolution(
                state: state, index: 0, values: key, dynamicInput: keyInput,
                staticTaps: staticTaps, projected: projected, kvHeads: kvHeads, headDim: headDim
            )
        } else {
            key = try shortConvolution(layer: layer, state: state, index: 0, values: key, weightsRole: .attnKSconvWeight)
        }
        onInternal?(visitIndexForDebug, "k_sconv", key)
        for head in 0..<kvHeads {
            let slice = Inkling2Math.l2Clip(
                Array(key[(head * headDim)..<((head + 1) * headDim)]),
                threshold: config.qkL2ClipThreshold
            )
            for dimension in 0..<headDim { key[head * headDim + dimension] = slice[dimension] }
        }
        onInternal?(visitIndexForDebug, "k_norm", key)

        // V: project, convolve, unweighted RMS norm.
        let wv = try weights.matrix(.attnWvDv, layer: layer)
        var value = try linear(wv, valueInput, rows: kvHeads * headDim)
        onInternal?(visitIndexForDebug, "v_linear", value)
        if isFull {
            let staticTaps = try weights.matrix(.attnVSconvStatic, layer: layer)
            let projected = try linear(try weights.matrix(.attnVSconvProj, layer: layer), valueInput, rows: kvHeads * config.convKernelSize)
            value = dynamicConvolution(
                state: state, index: 1, values: value, dynamicInput: valueInput,
                staticTaps: staticTaps, projected: projected, kvHeads: kvHeads, headDim: headDim
            )
        } else {
            value = try shortConvolution(layer: layer, state: state, index: 1, values: value, weightsRole: .attnVSconvWeight)
        }
        for head in 0..<kvHeads {
            let slice = Inkling2Math.rmsNorm(
                Array(value[(head * headDim)..<((head + 1) * headDim)]), eps: eps
            )
            for dimension in 0..<headDim { value[head * headDim + dimension] = slice[dimension] }
        }
        onInternal?(visitIndexForDebug, "v_norm", value)

        // A full-attention visit keeps every position's K/V, so the KV budget is
        // enforced here, where a row would be added: one position past it is a
        // refusal by name, not a process that keeps growing.
        if state.isAtCapacity {
            throw Inkling2ForwardError.unsupported(
                "position \(position) is past the KV budget: this sequence holds "
                    + "\(state.committedPositions) positions and the plan provisions "
                    + "\(state.maxRows ?? 0). Start a new conversation or raise the context budget."
            )
        }
        _ = state.append(key: key, value: value)

        // Relative bias: R from the pre-attention hidden, banked by backward
        // distance, zero outside the extent.
        let extent = isFull ? config.relExtent : config.slidingWindowSize
        let wr = try weights.matrix(.attnWrDu, layer: layer)
        var relative = try linear(wr, queryInput, rows: heads * config.dRel)
        onInternal?(visitIndexForDebug, "rel_linear", relative)
        let rScale = try weights.matrix(.attnRelL2ClipScale, layer: layer)
        for head in 0..<heads {
            var slice = Array(relative[(head * config.dRel)..<((head + 1) * config.dRel)])
            for dimension in 0..<config.dRel { slice[dimension] *= rScale[head * config.dRel + dimension] }
            slice = Inkling2Math.l2Clip(slice, threshold: config.relL2ClipThreshold)
            if isFull {
                let tau = Inkling2Math.logScale(position: position, alpha: config.logScalingAlpha, floor: config.logScalingNFloor)
                slice = slice.map { $0 * tau }
            }
            for dimension in 0..<config.dRel { relative[head * config.dRel + dimension] = slice[dimension] }
        }
        // The kernel's own input: captured AFTER the learned scale, the L2 clip
        // and (on a full layer) the log scale, because `rel_linear` above is
        // only the raw projection and a device comparison needs the value the
        // bias is actually mixed from.
        onInternal?(visitIndexForDebug, "relative_states", relative)
        let bank = try weights.matrix(.attnRelLogitsProj, layer: layer)   // [d_rel][extent]

        let sinkProjection = try weights.matrix(.attnSinkProj, layer: layer)  // [heads][hidden]
        var sinkLogits = [Float](repeating: 0, count: heads)
        for head in 0..<heads {
            var total: Float = 0
            let base = head * config.hiddenSize
            for column in 0..<config.hiddenSize { total += sinkProjection[base + column] * queryInput[column] }
            sinkLogits[head] = total
        }
        onInternal?(visitIndexForDebug, "sink", sinkLogits)
        onInternal?(visitIndexForDebug, "bias_head0", (0..<state.keys.count).map { row in
            let distance = position - (state.windowStart + row)
            guard distance >= 0, distance < extent else { return 0 }
            var total: Float = 0
            for dimension in 0..<config.dRel {
                total += relative[dimension] * bank[dimension * extent + distance]
            }
            return total
        })

        let scaling = 1 / Float(headDim).squareRoot()
        var attentionOutput = [Float](repeating: 0, count: heads * headDim)
        for head in 0..<heads {
            let kvHead = head / groups
            let count = state.keys.count
            var scores = [Float](repeating: 0, count: count)
            for row in 0..<count {
                let keyRow = state.keys[row]
                var dot: Float = 0
                for dimension in 0..<headDim {
                    dot += query[head * headDim + dimension] * keyRow[kvHead * headDim + dimension]
                }
                let distance = position - (state.windowStart + row)
                var bias: Float = 0
                if distance >= 0 && distance < extent {
                    for dimension in 0..<config.dRel {
                        bias += relative[head * config.dRel + dimension] * bank[dimension * extent + distance]
                    }
                }
                scores[row] = dot * scaling + bias
            }
            // Sink: one extra logit with a zero value row; softmax includes it
            // and the output uses only the real values.
            if head == 0 {
                onInternal?(visitIndexForDebug, "scores_head0", scores)
            }
            var combined = scores
            combined.append(sinkLogits[head])
            let maximum = combined.max() ?? 0
            let exponentials = combined.map { exp($0 - maximum) }
            let denominator = exponentials.reduce(0, +)
            if head == 0 {
                onInternal?(visitIndexForDebug, "weights_head0", exponentials.map { $0 / denominator })
            }
            for row in 0..<count {
                let weight = exponentials[row] / denominator
                let valueRow = state.values[row]
                for dimension in 0..<headDim {
                    attentionOutput[head * headDim + dimension] += weight * valueRow[kvHead * headDim + dimension]
                }
            }
        }

        onInternal?(visitIndexForDebug, "attn_raw", attentionOutput)
        // Output gate, then wo.
        let gate = try linear(try weights.matrix(.attnGate, layer: layer), queryInput, rows: heads * headDim)
        onInternal?(visitIndexForDebug, "gate", gate)
        for index in 0..<attentionOutput.count {
            attentionOutput[index] *= Inkling2Math.sigmoid(gate[index])
        }
        onInternal?(visitIndexForDebug, "attn_gated", attentionOutput)
        return try linear(try weights.matrix(.attnWoUd, layer: layer), attentionOutput, rows: config.hiddenSize)
    }

    // MARK: Mixture

    /// The mixture: FP32 sigmoid router over the fused expert bank, the selected
    /// experts' softcapped SwiGLU, and the weighted FP32 accumulation.
    public func mixture(hidden: [Float], layer: Int) throws -> [Float] {
        let router = try weights.matrix(.routerWeight, layer: layer)      // [128][hidden]
        let bias = try weights.matrix(.routerBias, layer: layer)
        let experts = config.numRoutedExperts
        var scores = [Float](repeating: 0, count: experts)
        for expert in 0..<experts {
            var total: Float = 0
            let base = expert * config.hiddenSize
            for column in 0..<config.hiddenSize { total += router[base + column] * hidden[column] }
            scores[expert] = Inkling2Math.sigmoid(total)
        }
        let route = Inkling2Math.route(
            scores: scores, bias: bias, topK: config.numExpertsPerToken, routeScale: config.routeScale
        )
        onInternal?(visitIndexForDebug, "route_indices", route.indices.map { Float($0) })
        onInternal?(visitIndexForDebug, "route_weights", route.weights)
        onInternal?(visitIndexForDebug, "route_scores", scores)
        onInternal?(visitIndexForDebug, "mixture_input", hidden)
        var accumulator = [Float](repeating: 0, count: config.hiddenSize)
        for (slot, expert) in route.indices.enumerated() {
            let weight = route.weights[slot]
            let bank = try weights.expertMatrix(.expertW13, layer: layer, expert: expert)
            let down = try weights.expertMatrix(.expertW2, layer: layer, expert: expert)
            var gate = [Float](repeating: 0, count: config.moeIntermediateSize)
            var up = [Float](repeating: 0, count: config.moeIntermediateSize)
            for index in 0..<config.moeIntermediateSize {
                let rows = bank.gateUpRow(index)
                gate[index] = bank.project(row: rows.gate, input: hidden)
                up[index] = bank.project(row: rows.up, input: hidden)
            }
            var activated = [Float](repeating: 0, count: config.moeIntermediateSize)
            for index in 0..<config.moeIntermediateSize {
                activated[index] = Inkling2Math.softcappedSwiGLU(
                    gate: gate[index], up: up[index],
                    gateSoftcap: config.mlpGateSoftcap, upSoftcap: config.mlpUpSoftcap
                )
            }
            if slot == 0 {
                onInternal?(visitIndexForDebug, "expert0_gate", Array(gate.prefix(4)))
                onInternal?(visitIndexForDebug, "expert0_up", Array(up.prefix(4)))
                onInternal?(visitIndexForDebug, "expert0_act", Array(activated.prefix(4)))
                onInternal?(visitIndexForDebug, "expert0_down", (0..<4).map { down.project(row: $0, input: activated) })
            }
            var contribution = [Float](repeating: 0, count: config.hiddenSize)
            for row in 0..<config.hiddenSize {
                contribution[row] = down.project(row: row, input: activated) * weight
            }
            onInternal?(visitIndexForDebug, "slot\(slot)_expert", [Float(expert), weight])
            onInternal?(visitIndexForDebug, "slot\(slot)_norm", [Float(contribution.reduce(0) { $0 + $1 * $1 }.squareRoot())])
            onInternal?(visitIndexForDebug, "slot\(slot)_row0", Array(contribution.prefix(4)))
            for row in 0..<config.hiddenSize {
                accumulator[row] += contribution[row]
            }
        }
        return accumulator
    }
}
