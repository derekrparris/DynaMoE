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
//  Two details decide whether this is a speedup or a slowdown:
//
//    * WHICH token a draft step consumes. The draft block's input is the target
//      model's hidden state at the verified position, but the token it conditions
//      on is the one the draft chain produced itself, not the target's. Feeding the
//      target's token back would make the chain a copy of the target and every
//      draft would "verify", reporting an acceptance rate that is a lie.
//
//    * WHEN to submit. `confidence_head.proj` gives a per-step confidence, and a
//      chain whose leading steps are confident is worth verifying even when its
//      tail is not: the cost of a failed verification is one extra target forward,
//      and the cost of not drafting is one target forward per token anyway. So the
//      gate is on the whole chain's leading confidence, not on every step.
//
//  The block/target layer ids come from `dspark_block_size` and
//  `dspark_target_layer_ids`; the exact conditioning rule is gated by
//  `DeepSeekV41Conformance.dsparkDrafting` until the probe settles it.
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

    /// Mixes the draft block's own head with the Markov head's contribution.
    ///
    /// `weight` is the Markov weight in [0, 1]: 0 leaves the draft block's logits
    /// untouched, 1 is the Markov head alone. Out of range is clamped rather than
    /// trusted, because a bad weight silently turns the chain into noise.
    public static func combineLogits(
        draftLogits: [Float],
        markovLogits: [Float],
        weight: Float
    ) -> [Float] {
        guard !draftLogits.isEmpty else { return [] }
        guard draftLogits.count == markovLogits.count else { return draftLogits }
        let clamped = min(max(weight, 0), 1)
        guard clamped > 0 else { return draftLogits }
        var out = [Float](repeating: 0, count: draftLogits.count)
        for index in out.indices {
            out[index] = (1 - clamped) * draftLogits[index] + clamped * markovLogits[index]
        }
        return out
    }

    /// Greedy choice from mixed logits, with ties broken by lowest token id so a
    /// draft is deterministic and reproducible.
    public static func pick(_ logits: [Float]) -> Int? {
        guard let best = logits.indices.max(by: { lhs, rhs in
            logits[lhs] == logits[rhs] ? lhs > rhs : logits[lhs] < logits[rhs]
        }) else {
            return nil
        }
        return best
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
    /// draft block for one step and returns `(draftLogits, markovLogits,
    /// confidence)`, and the chain machinery below is what is left to get wrong.
    /// The draft's own chosen token — not the target's — is what `prevToken` is set
    /// to for the next step.
    public static func runChain(
        steps: Int,
        seedToken: Int,
        markovWeight: Float,
        step: (Int, Int) -> (draftLogits: [Float], markovLogits: [Float], confidence: Float)?
    ) -> Draft {
        guard steps > 0 else { return .empty }
        var tokens: [Int] = []
        var confidences: [Float] = []
        tokens.reserveCapacity(steps)
        confidences.reserveCapacity(steps)
        var previous = seedToken
        for index in 0..<steps {
            guard let result = step(index, previous) else { break }
            let mixed = combineLogits(
                draftLogits: result.draftLogits,
                markovLogits: result.markovLogits,
                weight: markovWeight
            )
            // A step that produced nothing usable ends the chain: continuing would
            // condition the rest of it on a token that does not exist.
            guard let token = pick(mixed) else { break }
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
