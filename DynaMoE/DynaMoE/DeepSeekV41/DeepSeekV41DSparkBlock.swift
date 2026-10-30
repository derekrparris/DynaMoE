//
//  DeepSeekV41DSparkBlock.swift
//  DynaMoE
//
//  The DSpark draft BLOCK: the `mtp.N.*` stage forward, on the host.
//
//  `DeepSeekV41DSpark` owns the chain — the Markov bias, the draw, the confidence
//  gate, and the acceptance rule — and takes each step's logits from its caller.
//  This is that caller. It runs the stages the checkpoint stores under `mtp.N.*`:
//
//    1. `main_x = main_norm(main_proj(main_hidden))`, where `main_hidden` is the
//       target layers' concatenated means (see `DeepSeekV41CPUForward.mainHidden`).
//    2. A WINDOW-CACHE attention that is unlike every other layer here: a prefill
//       call (`startPosition == 0`) only SEEDS the ring from the target's own
//       `wkv(main_x)` rows and returns, and every later call appends one row and
//       attends over the ring plus the whole draft block at once. The block is not
//       causal within itself — one index list is shared by every query — so a port
//       that masks draft position i from draft position i+1 computes a different
//       graph (see `Tools/dsv41_goldens.py dspark_block`).
//    3. A full `Block` around that attention: hyper-connection mixes, the MoE under
//       the DSpark expert counts, and the `pre` mix the next stage collapses with.
//    4. Only on the LAST stage, `norm`, the model head, the Markov head (an
//       additive full-vocabulary bias) and the confidence head.
//
//  Nothing calls this from the turn loop yet, and the reason is cost rather than
//  correctness: the device executor forwards one position per call, so verifying a
//  chain of n drafts costs n target forwards — more work than decoding n tokens.
//  The rule is here, and matched against the reference, so a batched verify has
//  something to draft with when it lands.
//

import Foundation

nonisolated public final class DeepSeekV41DSparkBlock {

    public let config: DeepSeekV41Config
    public let plan: DeepSeekV41DSparkPlan
    /// The geometry a draft stage runs under: window-only attention, so no
    /// compressor and no YaRN (`Attention.__init__` takes `original_seq_len = 0`
    /// at `compress_ratio == 0` and falls back to the base theta).
    public let shape: DeepSeekV41AttentionShape
    /// One ring per stage, `[stage][slot][latentWidth]`. Per stage, because every
    /// `mtp.N` block owns its own window cache, filled from its own projection.
    public private(set) var windows: [[[Float]]]
    /// What the last round kept. The chain's tokens are the API; these are the
    /// numbers a golden compares, recorded rather than consumed for the same reason
    /// `DeepSeekV41CPUForward.candidateScores` is.
    public private(set) var lastTrace: Trace?

    /// One round's intermediate state, position-major.
    public struct Trace: Sendable, Equatable {
        /// `[stage][position][copy][dim]`: the residual stream after that stage's
        /// attention AND ffn.
        public let stageOutputs: [[[[Float]]]]
        /// `[position][dim]`: stage 0's `attn_norm(hc_pre(x, pre_mix))`.
        public let attentionInputs: [[Float]]
        /// `[position][numHeads * qkHeadDim]`: stage 0's rotated queries.
        public let queries: [[Float]]
        /// `[position][numHeads * vHeadDim]`: stage 0's `sparse_attn` output, before
        /// the output's rope payload is conjugated back out.
        public let attended: [[Float]]
        /// `[position][dim]`: stage 0's attention output after the grouped `wo_a`
        /// and `wo_b`.
        public let projected: [[Float]]
        /// `[position][copy][dim]`: stage 0's stack after its attention sub-layer.
        public let postAttention: [[[Float]]]
        /// `[position][dim]`: stage 0's `ffn_norm` output, the MoE's input.
        public let ffnInputs: [[Float]]
        /// `[position][dim]`: stage 0's MoE output, before `hc_post` mixes it in.
        public let ffnOutputs: [[Float]]
        /// `[position][hcMult]`: the mix the last stage hands the head.
        public let preMix: [[Float]]
        /// `[position][dim]`: the collapse the head normalizes and projects.
        public let headInputs: [[Float]]
        /// `[position][vocab]`: the draft head's own logits, before any Markov bias.
        public let logits: [[Float]]

        public init(
            stageOutputs: [[[[Float]]]],
            attentionInputs: [[Float]],
            queries: [[Float]],
            attended: [[Float]],
            projected: [[Float]],
            postAttention: [[[Float]]],
            ffnInputs: [[Float]],
            ffnOutputs: [[Float]],
            preMix: [[Float]],
            headInputs: [[Float]],
            logits: [[Float]]
        ) {
            self.stageOutputs = stageOutputs
            self.attentionInputs = attentionInputs
            self.queries = queries
            self.attended = attended
            self.projected = projected
            self.postAttention = postAttention
            self.ffnInputs = ffnInputs
            self.ffnOutputs = ffnOutputs
            self.preMix = preMix
            self.headInputs = headInputs
            self.logits = logits
        }
    }

    private let forward: DeepSeekV41CPUForward
    private let queryFrequencies: [Float]
    private let latentFrequencies: [Float]

    public init(forward: DeepSeekV41CPUForward) {
        let config = forward.config
        self.config = config
        self.plan = DeepSeekV41DSparkPlan.make(config: config)
        self.shape = DeepSeekV41DSparkBlock.draftShape(config: config)
        self.forward = forward
        self.queryFrequencies = DeepSeekV41RoPE.queryFrequencies(for: shape)
        self.latentFrequencies = DeepSeekV41RoPE.latentFrequencies(for: shape)
        let slots = max(1, config.windowSize)
        let row = [Float](repeating: 0, count: max(1, shape.latentWidth))
        let stages = max(0, plan.draftBlockCount)
        self.windows = (0..<stages).map { _ in Array(repeating: row, count: slots) }
    }

    public var stageCount: Int { windows.count }

    /// `DSparkBlock.forward_embed`'s `main_x`: the target layers' stacked means
    /// through the stage-0 projection and norm. Row per position, so a prefix goes
    /// through in one call.
    public func mainX(_ hidden: [[Float]]) -> [[Float]] {
        guard let projection = matrix(.draftMainProj, stage: 0) else { return [] }
        return hidden.map { row in
            forward.rmsNorm(
                projection.apply(row),
                role: .draftMainNorm,
                scope: .draft,
                layer: 0,
                epsilon: config.rmsNormEps
            )
        }
    }

    /// The draft stack `forward_embed` builds: the token being drafted from, then
    /// `blockSize - 1` noise tokens, one hyper-connection copy each.
    public func draftStack(token: Int) -> [[[Float]]] {
        let copies = max(1, config.hcMult)
        let noise = Array(repeating: plan.noiseTokenID, count: max(0, plan.blockSize - 1))
        return ([token] + noise).map { id in
            Array(repeating: forward.embed(id), count: copies)
        }
    }

    /// Seeds every stage's ring from a prefix's `main_x` rows.
    ///
    /// This is the reference's `start_pos == 0` branch, and it is all that branch
    /// does: it writes the target's own latents and returns the residual untouched,
    /// which is why a seed that stored the wrong rows stays invisible until the
    /// first draft call reads them. A prefix longer than the window keeps its LAST
    /// `window` rows, rotated so slot `len % window` holds the oldest of them.
    public func seed(_ mainX: [[Float]]) {
        guard plan.isEnabled, !mainX.isEmpty else { return }
        for stage in windows.indices {
            var latents: [[Float]] = []
            latents.reserveCapacity(mainX.count)
            for (position, row) in mainX.enumerated() {
                latents.append(latent(row, stage: stage, position: position))
            }
            windows[stage] = seededRing(latents, slots: windows[stage].count)
        }
    }

    /// One drafting round: the block forward, then the chain.
    ///
    /// `startPosition` is how many positions the TARGET holds, which is also where
    /// the draft block's own positions begin: the reference takes its frequencies
    /// from `start_pos + seqlen` and writes the new ring row at
    /// `start_pos % window_size`. Zero means "seed instead", which this refuses
    /// rather than pretending to draft from nothing.
    public func draft(
        token: Int,
        mainX: [Float],
        startPosition: Int,
        temperature: Float = 0,
        nextUniform: @escaping () -> Float = DeepSeekV41DSpark.randomUniform
    ) -> DeepSeekV41DSpark.Draft {
        guard plan.isEnabled, startPosition > 0 else { return .empty }
        var stack = draftStack(token: token)
        var preMix = Array(
            repeating: DeepSeekV41HyperConnections.identityMix(copies: max(1, config.hcMult)),
            count: stack.count
        )

        // The ring is written BEFORE any query reads it, and the row written is the
        // target's newest position: `window_kv_cache[:bsz, start_pos % win] =
        // main_kv`, then `kv = cat([window_kv_cache, kv])` and the queries gather
        // slots `0..<min(win, start_pos + 1)` — which includes the slot just
        // overwritten. Reading first would attend over the position it replaced.
        for stage in windows.indices {
            let row = latent(mainX, stage: stage, position: startPosition)
            let slots = windows[stage].count
            windows[stage][startPosition % slots] = row
        }

        var stageOutputs: [[[[Float]]]] = []
        var attentionInputs: [[Float]] = []
        var queries: [[Float]] = []
        var attended: [[Float]] = []
        var projected: [[Float]] = []
        var postAttention: [[[Float]]] = []
        var ffnInputs: [[Float]] = []
        var ffnOutputs: [[Float]] = []
        for stage in windows.indices {
            let pass = runStage(stage, stack: stack, preMix: preMix, startPosition: startPosition)
            stack = pass.stack
            preMix = pass.preMix
            stageOutputs.append(pass.stack)
            if stage == windows.startIndex {
                attentionInputs = pass.attentionInputs
                queries = pass.queries
                attended = pass.attended
                projected = pass.projected
                postAttention = pass.postAttention
                ffnInputs = pass.ffnInputs
                ffnOutputs = pass.ffnOutputs
            }
        }
        let outcome = head(
            stack: stack,
            preMix: preMix,
            token: token,
            temperature: temperature,
            nextUniform: nextUniform
        )
        lastTrace = Trace(
            stageOutputs: stageOutputs,
            attentionInputs: attentionInputs,
            queries: queries,
            attended: attended,
            projected: projected,
            postAttention: postAttention,
            ffnInputs: ffnInputs,
            ffnOutputs: ffnOutputs,
            preMix: preMix,
            headInputs: outcome.headInputs,
            logits: outcome.logits
        )
        return outcome.draft
    }

    // MARK: One stage

    private struct StagePass {
        let stack: [[[Float]]]
        let preMix: [[Float]]
        let attentionInputs: [[Float]]
        let queries: [[Float]]
        let attended: [[Float]]
        let projected: [[Float]]
        let postAttention: [[[Float]]]
        let ffnInputs: [[Float]]
        let ffnOutputs: [[Float]]
    }

    private func runStage(
        _ stage: Int,
        stack: [[[Float]]],
        preMix: [[Float]],
        startPosition: Int
    ) -> StagePass {
        let positions = stack.count
        // The block's own positions are the ones after the target's last: with a
        // single `main_x` row in this call, `seqlen == 1`.
        let first = startPosition + 1

        var coefficients: [DeepSeekV41HyperConnections.Coefficients] = []
        var residuals: [[[Float]]] = []
        var normedInputs: [[Float]] = []
        for offset in 0..<positions {
            let input = stack[offset]
            coefficients.append(
                forward.hyperConnectionCoefficients(
                    layer: stage, scope: .draft, entry: .attention, streams: input
                )
            )
            residuals.append(input)
            let attentionInput = DeepSeekV41HyperConnections.pre(
                streams: input, mix: preMix[offset]
            )
            normedInputs.append(
                forward.rmsNorm(
                    attentionInput,
                    role: .attnNorm,
                    scope: .draft,
                    layer: stage,
                    epsilon: config.rmsNormEps
                )
            )
        }

        var queries: [[Float]] = []
        queries.reserveCapacity(positions)
        for (offset, normed) in normedInputs.enumerated() {
            var query = forward.queryPath(normed, scope: .draft, layer: stage, shape: shape)
            DeepSeekV41RoPE.rotateHeads(
                &query,
                heads: shape.numHeads,
                headDim: shape.qkHeadDim,
                sliceStart: shape.queryRopeStart,
                sliceLength: shape.queryRopeWidth,
                position: first + offset,
                frequencies: queryFrequencies
            )
            queries.append(query)
        }

        // The block's own latents come from the DRAFT stack, not from `main_x`: the
        // cache holds the target's keys, and these are the drafter's.
        var ownLatents: [[Float]] = []
        ownLatents.reserveCapacity(positions)
        for (offset, normed) in normedInputs.enumerated() {
            ownLatents.append(latent(normed, stage: stage, position: first + offset))
        }

        // One index list for every query: the written window slots in ring order,
        // then the block's own rows. Draft positions see each other.
        let window = windows[stage]
        var keys = Array(window.prefix(min(window.count, startPosition + 1)))
        keys.append(contentsOf: ownLatents)

        let sink = forward.weights.vector(.attnSink, scope: .draft, layer: stage)
        var streams = stack
        var attendedOut: [[Float]] = []
        attendedOut.reserveCapacity(positions)
        var projectedOut: [[Float]] = []
        projectedOut.reserveCapacity(positions)
        for offset in 0..<positions {
            var attended = DeepSeekV41AttentionReference.attend(
                query: queries[offset],
                keys: keys,
                shape: shape,
                sink: sink
            )
            attendedOut.append(attended)
            // The keys stay stored rotated, so the output's payload is conjugated
            // back out before the projection — at ITS OWN position, like the keys.
            if shape.outputCarriesRope {
                DeepSeekV41RoPE.rotateHeads(
                    &attended,
                    heads: shape.numHeads,
                    headDim: shape.vHeadDim,
                    sliceStart: shape.outputRopeStart,
                    sliceLength: shape.latentRopeWidth,
                    position: first + offset,
                    frequencies: latentFrequencies,
                    inverse: true
                )
            }
            let output = forward.outputPath(attended, scope: .draft, layer: stage, shape: shape)
            projectedOut.append(output)
            streams[offset] = DeepSeekV41HyperConnections.post(
                output: output,
                residual: residuals[offset],
                gains: coefficients[offset].post,
                comb: coefficients[offset].comb
            )
        }

        let afterAttention = streams
        var nextMix = preMix
        var ffnInputs: [[Float]] = []
        ffnInputs.reserveCapacity(positions)
        var ffnOutputs: [[Float]] = []
        ffnOutputs.reserveCapacity(positions)
        for offset in 0..<positions {
            let input = streams[offset]
            let ffnCoefficients = forward.hyperConnectionCoefficients(
                layer: stage, scope: .draft, entry: .feedForward, streams: input
            )
            let ffnInput = DeepSeekV41HyperConnections.pre(
                streams: input, mix: coefficients[offset].pre
            )
            let normed = forward.rmsNorm(
                ffnInput,
                role: .ffnNorm,
                scope: .draft,
                layer: stage,
                epsilon: config.rmsNormEps
            )
            ffnInputs.append(normed)
            let output = forward.feedForward(
                normed,
                scope: .draft,
                layer: stage,
                topK: config.draftExpertCounts.activated
            )
            ffnOutputs.append(output)
            streams[offset] = DeepSeekV41HyperConnections.post(
                output: output,
                residual: input,
                gains: ffnCoefficients.post,
                comb: ffnCoefficients.comb
            )
            nextMix[offset] = ffnCoefficients.pre
        }
        return StagePass(
            stack: streams,
            preMix: nextMix,
            attentionInputs: normedInputs,
            queries: queries,
            attended: attendedOut,
            projected: projectedOut,
            postAttention: afterAttention,
            ffnInputs: ffnInputs,
            ffnOutputs: ffnOutputs
        )
    }

    // MARK: The head

    /// `DSparkBlock.forward_head`: one collapse, one norm, then the chain.
    ///
    /// The chain itself is `DeepSeekV41DSpark.runChain` — the Markov bias is ADDED
    /// to the draft logits and each step conditions on the token it drew — so this
    /// only supplies, per step, the block's own logits, the bias for the token
    /// before it, and the confidence of that position.
    private struct HeadOutcome {
        let draft: DeepSeekV41DSpark.Draft
        let headInputs: [[Float]]
        let logits: [[Float]]
    }

    private func head(
        stack: [[[Float]]],
        preMix: [[Float]],
        token: Int,
        temperature: Float,
        nextUniform: @escaping () -> Float
    ) -> HeadOutcome {
        guard let stage = windows.indices.last else {
            return HeadOutcome(draft: .empty, headInputs: [], logits: [])
        }
        var collapsed: [[Float]] = []
        var logits: [[Float]] = []
        for offset in 0..<stack.count {
            // `x = hc_pre(h, pre_mix)` is the collapse BOTH the head and the
            // confidence head read; the norm is only on the way to the logits.
            let x = DeepSeekV41HyperConnections.pre(streams: stack[offset], mix: preMix[offset])
            let normed = forward.rmsNorm(
                x,
                role: .draftNorm,
                scope: .draft,
                layer: stage,
                epsilon: config.rmsNormEps
            )
            collapsed.append(x)
            logits.append(forward.headLogits(normed))
        }
        let draft = DeepSeekV41DSpark.runChain(
            steps: min(plan.blockSize, logits.count),
            seedToken: token,
            temperature: temperature,
            nextUniform: nextUniform,
            step: { index, previous in
                (
                    draftLogits: logits[index],
                    markovBias: markovBias(for: previous, stage: stage),
                    confidence: confidence(collapsed: collapsed[index], previous: previous, stage: stage)
                )
            }
        )
        return HeadOutcome(draft: draft, headInputs: collapsed, logits: logits)
    }

    /// `markov_head(token)`: `head_embed(head)`, a full-vocabulary bias.
    private func markovBias(for token: Int, stage: Int) -> [Float] {
        guard let embed = matrix(.draftMarkovEmbed, stage: stage),
              let head = matrix(.draftMarkovHead, stage: stage) else { return [] }
        return head.apply(embed.row(token))
    }

    /// `confidence_head([x, markov_embed(token)])`, a scalar per step.
    private func confidence(collapsed: [Float], previous: Int, stage: Int) -> Float {
        guard let embed = matrix(.draftMarkovEmbed, stage: stage),
              let projection = matrix(.draftConfidence, stage: stage) else { return 0 }
        var joined = collapsed
        joined.append(contentsOf: embed.row(previous))
        return projection.apply(joined).first ?? 0
    }

    // MARK: Cache rows

    /// One cache row from an attention input: `kv_norm(wkv(.))` rotated at the
    /// position it will be stored at. The reference also quantizes it to FP8 here
    /// (`act_quant(..., inplace=True)`), which is the cache's storage format rather
    /// than part of the rule.
    private func latent(_ input: [Float], stage: Int, position: Int) -> [Float] {
        var latent = forward.latentPath(input, scope: .draft, layer: stage)
        DeepSeekV41RoPE.rotateSlice(
            &latent,
            start: shape.latentRopeStart,
            length: shape.latentRopeWidth,
            position: position,
            frequencies: latentFrequencies
        )
        return latent
    }

    /// The seed branch, both halves: a prefix that fits lands in slot order, and a
    /// longer one keeps its last `slots` rows rotated so slot `length % slots` holds
    /// the oldest survivor.
    private func seededRing(_ rows: [[Float]], slots: Int) -> [[Float]] {
        let blank = [Float](repeating: 0, count: max(1, shape.latentWidth))
        var ring = Array(repeating: blank, count: max(1, slots))
        guard !rows.isEmpty else { return ring }
        let window = ring.count
        if rows.count <= window {
            for (slot, row) in rows.enumerated() { ring[slot] = row }
            return ring
        }
        let cutoff = rows.count % window
        let tail = Array(rows.suffix(window))
        for slot in 0..<(window - cutoff) { ring[cutoff + slot] = tail[slot] }
        for slot in 0..<cutoff { ring[slot] = tail[(window - cutoff) + slot] }
        return ring
    }

    private func matrix(_ role: DeepSeekV41TensorRole, stage: Int) -> DeepSeekV41Matrix? {
        forward.weights.matrix(role, scope: .draft, layer: stage, expert: nil)
    }

    private static func draftShape(config: DeepSeekV41Config) -> DeepSeekV41AttentionShape {
        DeepSeekV41AttentionShape(
            numHeads: config.numAttentionHeads,
            headsPerGroup: max(1, config.numAttentionHeads / max(1, config.oGroups)),
            qkHeadDim: config.qkHeadDim,
            vHeadDim: config.vHeadDim,
            latentWidth: config.cacheRecordWidth,
            qLoraRank: config.qLoraRank,
            oLoraRank: config.oLoraRank,
            isCompressed: false,
            compressRatio: 0,
            ropeTheta: config.ropeTheta,
            rotaryDim: config.qkRopeHeadDim,
            ropeScaling: .none
        )
    }
}
