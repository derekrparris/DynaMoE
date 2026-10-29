//
//  DeepSeekV41Conformance.swift
//  DynaMoE
//
//  Which parts of the port have been matched against the checkpoint's own
//  reference implementation, and which are still inferred from the architecture
//  description.
//
//  The distinction exists because the two failure modes are not symmetric. A wrong
//  tensor name or a wrong dtype fails closed at load. A wrong *numeric rule* (the
//  Engram hash combination, the compressor's pooling, a prompt marker spelling)
//  runs at full speed and produces plausible text, which is far worse: the model
//  looks like it is working.
//
//  So each such rule carries a gate here. The load log prints the unverified ones,
//  `Tools/dsv41_probe.py` dumps the checkpoint's own files (it does not compute
//  reference outputs), and flipping a gate to `.verifiedAgainstReference` is a
//  one-line change accompanied by a test that carries vectors produced by a real
//  reference run. The gates are diagnostics rather than switches, with
//  one deliberate exception: the device forward refuses an Engram layer while the
//  hash combination is unverified, because reading the wrong rows is silent.
//

import Foundation

/// Whether the family can serve a turn yet. Detection and the clear refusal stay
/// on unconditionally; discovery and the graded-effort UI read this so a user is
/// not invited to download a ~510 GB checkpoint that can only fail closed.
nonisolated public enum DeepSeekV41Availability {
    public static let isRuntimeAvailable = false
}

nonisolated public enum DeepSeekV41Conformance {

    public enum Gate: String, Sendable, Equatable {
        /// Inferred from the architecture description and the tensor shapes.
        case unverified
        /// Matched against the checkpoint's own reference implementation, with the
        /// reference vectors checked into `DeepSeekV41LogicTests`.
        case verifiedAgainstReference

        public var isVerified: Bool { self == .verifiedAgainstReference }
    }

    /// How a run of positions pools into one compressed row.
    ///
    /// The RULE now matches `Compressor.forward` in the checkpoint's
    /// `inference/model.py` (project the layer input through `wkv` and `wgate`,
    /// softmax the gate over the group PER DIMENSION, `RMSNorm`, then rotate at the
    /// group's first position), and both backends implement that one rule. What is
    /// still missing is a numeric vector produced by a reference run, so the gate
    /// stays open: reading the reference and agreeing with it is not the same as
    /// having compared outputs.
    public static let compressorPooling: Gate = .unverified

    /// How a raw gate logit becomes a routing score (`scoring_func`) and how the
    /// selection bias is applied.
    ///
    /// Matches `Gate.forward` (`score + bias` selects, `score` weights), but the
    /// CPU reference and the kernel are only self-consistent until a real routing
    /// vector is checked in.
    public static let routerScoring: Gate = .unverified

    /// How the compressed-row candidate blocks are scored and truncated.
    ///
    /// Both backends currently read rows through
    /// `DeepSeekV41CacheAddressing.attentionRowPlan`, which is value-free: the
    /// count a position reads is known on the host before any dispatch. The
    /// block-score path stays a recorded diagnostic (the reference keeps
    /// `candidateScores`, the device path does not compute them) because a
    /// value-dependent selection needs a per-layer readback of the selected count,
    /// and paying that before the rule is verified would slow every token down to
    /// reproduce a guess.
    public static let candidateSelection: Gate = .unverified

    /// How retrieved Engram rows are combined into the layer's residual.
    ///
    /// The RULE now matches `Engram.forward` in the checkpoint's
    /// `inference/model.py`: `kv = wkv(embed(hashes).flatten(-2))` splits into a
    /// per-hc-copy key and one shared value, the value is gated by
    /// `sigmoid(copysign(sqrt(dot), dot))` over a normalized dot of the stream
    /// against its key, and the result is written into the residual stream at the
    /// top of the layer (before attention) rather than into the post-`ffn_norm`
    /// input. What is still missing is a numeric vector produced by a reference
    /// run, so the gate stays open, and the device forward keeps refusing an Engram
    /// layer until it flips: reading the right rows and mixing them wrongly looks
    /// like "the memory model does nothing".
    public static let engramCombination: Gate = .unverified

    /// The mHC hyper-connection mixing rule.
    public static let hyperConnections: Gate = .unverified

    /// DSpark's draft-chain construction.
    public static let dsparkDrafting: Gate = .unverified

    /// The vision tower's patch grid and image token layout.
    ///
    /// The GEOMETRY now matches `image_processor.py`: the resize plan is
    /// `plan_image_grid` + `safe_resize`, the patch grid is `llm_grid`, the span is
    /// `num_image_tokens = rows * (columns + 1) + 2`, and the patches are emitted in
    /// the checkpoint's `reshape(3, n_h, p, n_w, p)` order. Two things still keep the
    /// gate open: no tower runner exists (there is no 32-block ViT/aligner forward),
    /// and the token IDENTIFIER is unverified, because the reference writes the same
    /// `image_token_id` at every span position and trusts the tokenizer for it,
    /// while this port renders the span from a marker string.
    public static let visionPatching: Gate = .unverified

    /// Every gate, paired with the label the load log prints.
    public static var all: [(name: String, gate: Gate)] {
        [
            ("Engram hashing", DeepSeekV41EngramHasher.conformance == .verifiedAgainstReference
                ? .verifiedAgainstReference : .unverified),
            ("compressor pooling", compressorPooling),
            ("router scoring", routerScoring),
            ("Engram combination", engramCombination),
            ("candidate selection", candidateSelection),
            ("hyper-connections", hyperConnections),
            ("DSpark drafting", dsparkDrafting),
            ("vision patching", visionPatching)
        ]
    }

    /// The unverified rules, phrased for the load log. Empty means every rule was
    /// matched against the reference.
    public static var unverifiedSummary: [String] {
        all.filter { !$0.gate.isVerified }.map(\.name)
    }
}
