//
//  DeepSeekV41DSpark.swift
//  DynaMoE
//
//  DSpark: the checkpoint's built-in multi-token drafter.
//
//  Every other speculative path in this repo is JetSpec, which builds a TREE from
//  the model's own logits and verifies it with a tree-causal mask. DSpark is a
//  different animal: the checkpoint ships draft blocks (`mtp.N.*`) that produce a
//  linear CHAIN of candidate tokens, one per step, each conditioned on the draft
//  token chosen at the previous step through a Markov head. There is no tree to
//  build, so verification is a straight longest-prefix match against the target's
//  own choices.
//
//  The chain itself is the reference's, not a design of ours. `DSparkBlock` in the
//  checkpoint's `inference/model.py` does:
//
//    1. `main_x = main_norm(main_proj(main_hidden))`, where `main_hidden` is the
//       mean over the hyper-connection copies of the ATTENTION INPUT at each of
//       `dspark_target_layer_ids`, concatenated.
//    2. Seed the draft block with `[last token, noise, noise, ...]`.
//    3. Per step i: `logits[i] += markov_head(output_ids[i])` — the Markov head is
//       an ADDITIVE full-vocab bias, not a mixture — then draw one token with
//       Gumbel-max at `temperature` (argmax at 0), and that drawn token, not the
//       target's, is what step i+1 conditions on.
//    4. `confidence[i] = confidence_head([x_i, markov_embed(output_ids[i])])`,
//       with `x_i` the block's own pre-norm collapsed stream at step i.
//
//  Two details decide whether this is a speedup or a slowdown:
//
//    * WHICH token a draft step consumes. Feeding the target's token back would
//      make the chain a copy of the target and every draft would "verify",
//      reporting an acceptance rate that is a lie.
//
//    * WHEN to submit. `confidence_head.proj` gives a per-step confidence, and a
//      chain whose leading steps are confident is worth verifying even when its
//      tail is not: the cost of a failed verification is one extra target forward,
//      and the cost of not drafting is one target forward per token anyway. So the
//      gate is on the whole chain's leading confidence, not on every step. That
//      gate, and the acceptance/emission rule below, are this port's: the
//      reference defines the drafter but nothing in its generator calls
//      `forward_spec`, so there is no reference speculative loop to match.
//
//  The draft block's own attention and MoE are not implemented here, and until a
//  BATCHED multi-position verify exists they would not pay for themselves even if
//  they were: `DeepSeekV41MetalForward` forwards one position per call, so
//  verifying a chain of n drafts costs n target forwards — more work than
//  decoding n tokens. That is why the turn loop still reports `jetSpecActive:
//  false` for this family.
//

import Foundation

// MARK: - Plan

nonisolated public struct DeepSeekV41DSparkPlan: Sendable, Equatable {
    /// Draft tokens produced per verification round. Zero means the checkpoint has
    /// no drafter and the engine decodes one token at a time.
    public let blockSize: Int
    /// The token injected to seed a drafting round.
    public let noiseTokenID: Int
    /// Backbone layers whose hidden states feed the draft block, ascending.
    public let targetLayerIDs: [Int]
    /// Number of `mtp.N` blocks the checkpoint carries.
    public let draftBlockCount: Int

    public var isEnabled: Bool { blockSize > 0 && draftBlockCount > 0 }

    public init(blockSize: Int, noiseTokenID: Int, targetLayerIDs: [Int], draftBlockCount: Int) {
        self.blockSize = blockSize
        self.noiseTokenID = noiseTokenID
        self.targetLayerIDs = targetLayerIDs
        self.draftBlockCount = draftBlockCount
    }

    public static func make(config: DeepSeekV41Config) -> DeepSeekV41DSparkPlan {
        DeepSeekV41DSparkPlan(
            blockSize: max(0, config.dsparkBlockSize),
            noiseTokenID: config.dsparkNoiseTokenID,
            targetLayerIDs: config.dsparkTargetLayerIDs.sorted(),
            draftBlockCount: max(0, config.numNextNPredictLayers)
        )
    }

    /// Positions a drafting round covers, in speculative-attention terms: the
    /// verified tokens plus the `blockSize` drafts.
    public func verificationWindow(verifiedCount: Int) -> Range<Int> {
        guard blockSize > 0 else { return 0..<0 }
        return verifiedCount..<(verifiedCount + blockSize + 1)
    }
}

// MARK: - Drafting

nonisolated public enum DeepSeekV41DSpark {

    /// Minimum leading-step confidence for a chain to be worth verifying.
    ///
    /// A single target forward costs the same whether it verifies one token or
    /// `blockSize` tokens, so the gate only has to keep chains that are likely to
    /// be wrong from being submitted; it does not have to be strict.
    public static let defaultConfidenceThreshold: Float = 0.3

    /// Result of one drafting round.
    public struct Draft: Sendable, Equatable {
        /// Candidate tokens, in generation order.
        public let tokens: [Int]
        /// Confidence of each candidate, same order.
        public let confidences: [Float]

        public init(tokens: [Int], confidences: [Float]) {
            self.tokens = tokens
            self.confidences = confidences
        }

        public static let empty = Draft(tokens: [], confidences: [])
    }

    /// Adds the Markov head's bias to a step's draft logits.
    ///
    /// The reference writes `logits[:, i].add_(logits_bias)` over the full vocab,
    /// so this is a sum and not a mixture: a convex blend would scale the draft
    /// block's own logits down and change which token wins, which is a quiet way to
    /// make every draft worse. Widths that disagree mean the caller handed over a
    /// different head's row; the draft logits are returned untouched rather than
    /// truncated, because a partial bias would be just as wrong and harder to see.
    public static func applyMarkovBias(_ draftLogits: [Float], bias: [Float]) -> [Float] {
        guard draftLogits.count == bias.count else { return draftLogits }
        var out = draftLogits
        for index in out.indices {
            out[index] += bias[index]
        }
        return out
    }

    /// Greedy choice from biased logits, with ties broken by lowest token id so a
    /// draft is deterministic and reproducible.
    public static func pick(_ logits: [Float]) -> Int? {
        guard let best = logits.indices.max(by: { lhs, rhs in
            logits[lhs] == logits[rhs] ? lhs > rhs : logits[lhs] < logits[rhs]
        }) else {
            return nil
        }
        return best
    }

    /// A uniform draw for the sampler, kept out of `(0, 1)` so `log` stays finite.
    public static func randomUniform() -> Float {
        Float.random(in: 0.000_001...0.999_999)
    }

    /// Gumbel noise from one uniform draw: `-log(-log u)`.
    ///
    /// `-log(Exp(1))` is Gumbel, so this is the same distribution the reference's
    /// `probs.div_(empty_like(probs).exponential_(1)).argmax()` draws from — but
    /// driven by an injected uniform, which is what makes a reference vector
    /// replayable off a torch box.
    public static func gumbel(_ uniform: Float) -> Float {
        let clamped = min(max(uniform, 1e-6), 1 - 1e-6)
        return -log(-log(clamped))
    }

    /// One draft draw: greedy at temperature 0, otherwise Gumbel-max.
    ///
    /// The reference's `sample` is `argmax` at `temperature == 0` and otherwise
    /// `argmax(p / -log u)`, i.e. `argmax(logits / max(temperature, 1e-5) +
    /// gumbel(u))` — the same rule, written to stay finite. `nextUniform` is called
    /// once per vocabulary entry, in index order, so a caller that replays a
    /// recorded uniform table reproduces the draw exactly.
    public static func sample(
        logits: [Float],
        temperature: Float,
        nextUniform: () -> Float
    ) -> Int? {
        guard !logits.isEmpty else { return nil }
        guard temperature > 0 else { return pick(logits) }
        let scale = max(temperature, 1e-5)
        var bestIndex = 0
        var bestScore = -Float.infinity
        for (index, logit) in logits.enumerated() {
            let score = logit / scale + gumbel(nextUniform())
            if score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        return bestIndex
    }

    /// Whether a chain should be submitted for verification.
    ///
    /// The gate is on the FIRST step: a chain whose first draft is unlikely is a
    /// pure cost (the target has to run anyway and will reject it), while a chain
    /// that starts well and degrades is still a win for as long as it holds.
    public static func shouldSubmit(
        _ draft: Draft,
        threshold: Float = defaultConfidenceThreshold
    ) -> Bool {
        guard let first = draft.confidences.first else { return false }
        return first >= threshold
    }

    /// Longest prefix of `draft` that the target model agrees with.
    ///
    /// `targetChoices` is the target's own greedy choice at each position, starting
    /// at the position the first draft token occupies. Verification stops at the
    /// first disagreement: everything after it is conditioned on a token the target
    /// would not have produced, so it cannot be accepted.
    public static func acceptedCount(
        draft: [Int],
        targetChoices: [Int]
    ) -> Int {
        var accepted = 0
        for (index, candidate) in draft.enumerated() {
            guard index < targetChoices.count else { break }
            guard targetChoices[index] == candidate else { break }
            accepted += 1
        }
        return accepted
    }

    /// Runs one drafting round.
    ///
    /// `step` is supplied by the runtime so this stays pure: the runtime runs the
    /// draft block for one step and returns `(draftLogits, markovBias,
    /// confidence)`, and the chain machinery below is what is left to get wrong.
    /// The draft's own chosen token — not the target's — is what `previous` is set
    /// to for the next step.
    public static func runChain(
        steps: Int,
        seedToken: Int,
        temperature: Float = 0,
        nextUniform: @escaping () -> Float = DeepSeekV41DSpark.randomUniform,
        step: (Int, Int) -> (draftLogits: [Float], markovBias: [Float], confidence: Float)?
    ) -> Draft {
        guard steps > 0 else { return .empty }
        var tokens: [Int] = []
        var confidences: [Float] = []
        tokens.reserveCapacity(steps)
        confidences.reserveCapacity(steps)
        var previous = seedToken
        for index in 0..<steps {
            guard let result = step(index, previous) else { break }
            let biased = applyMarkovBias(result.draftLogits, bias: result.markovBias)
            // A step that produced nothing usable ends the chain: continuing would
            // condition the rest of it on a token that does not exist.
            guard let token = sample(
                logits: biased, temperature: temperature, nextUniform: nextUniform
            ) else { break }
            tokens.append(token)
            confidences.append(result.confidence)
            previous = token
        }
        return Draft(tokens: tokens, confidences: confidences)
    }

    /// Tokens to emit for a verification round: the accepted prefix plus the
    /// target's own token at the first disagreement. The target's choice is always
    /// emitted, which is what keeps the round a net win: a fully accepted chain
    /// still ends on a token the drafter did not produce.
    public static func emission(
        draft: [Int],
        targetChoices: [Int]
    ) -> [Int] {
        let accepted = acceptedCount(draft: draft, targetChoices: targetChoices)
        return Array(targetChoices.prefix(min(accepted + 1, targetChoices.count)))
    }
}
