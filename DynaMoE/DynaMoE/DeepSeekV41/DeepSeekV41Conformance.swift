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
//  reference run. The gates are diagnostics rather than switches: a stage the port
//  has implemented runs whether or not its rule has been matched against a numeric
//  reference, and the gate is how far to trust the numbers it produces.
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
    /// MATCHED AGAINST `Compressor.forward` in the checkpoint's
    /// `inference/model.py`: project the layer input through `wkv` and `wgate`,
    /// softmax the gate over the group PER DIMENSION, weight and sum, then
    /// `RMSNorm`. `Tools/dsv41_goldens.py` instantiates that module at a tiny
    /// config (dense FP32 weights, seeded parameters, four positions that
    /// disagree about which one should dominate) and
    /// `DeepSeekV41CompressorGoldenTests` carries the resulting vectors, so the
    /// two backends are checked against the reference's numbers rather than
    /// against each other.
    ///
    /// What it does NOT cover: the decode path's slot addressing
    /// (`position % ratio`), which is host code and is pinned by its own tests,
    /// and the fp8 storage width, which is exercised by the kernel tests.
    public static let compressorPooling: Gate = .verifiedAgainstReference

    /// How a raw gate logit becomes a routing score (`scoring_func`) and how the
    /// selection bias is applied.
    ///
    /// MATCHED AGAINST `Gate.forward` in the checkpoint's `inference/model.py`,
    /// for both per-logit scorings: `sqrtsoftplus` (the shipped one) and
    /// `sigmoid`. `DeepSeekV41RouterGoldenTests` carries vectors from
    /// `Tools/dsv41_goldens.py` for the same weights under both, which is what
    /// makes the comparison worth something: they select DIFFERENT experts, so
    /// picking the wrong map cannot pass. The temperature in those vectors is not
    /// 1, because the reference divides the logits before the scoring function.
    ///
    /// The bias is confirmed to steer selection only (`scores + bias` ranks,
    /// `scores` weights) and the renormalization to happen only for `topk > 1`.
    public static let routerScoring: Gate = .verifiedAgainstReference

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
    /// MATCHED AGAINST `Engram.forward` in the checkpoint's `inference/model.py`:
    /// `kv = wkv(embed(hashes).flatten(-2))` splits into a per-hc-copy key and one
    /// shared value, the value is gated by
    /// `sigmoid(copysign(sqrt(dot), dot))` over a normalized dot of the stream
    /// against its key, and the result is written into the residual stream at the
    /// top of the layer (before attention) rather than into the post-`ffn_norm`
    /// input.
    ///
    /// `Tools/dsv41_goldens.py` instantiates that module at a tiny config and
    /// `DeepSeekV41EngramGoldenTests` carries the vectors: two positions, two
    /// hyper-connection copies, two hash columns of 32-wide rows. The comparison
    /// covers the parts that are invisible in output, i.e. that the gate
    /// normalizes per (token, copy) over `dim` and not jointly over the copies,
    /// and that `kv` splits as `[hc_mult * dim, dim]` rather than interleaved.
    ///
    /// The device path is checked against the same rule by
    /// `DeepSeekV41EngramGateKernelTests`, which runs `dsv41_engram_gate_f32` on
    /// the CPU rule's inputs.
    public static let engramCombination: Gate = .verifiedAgainstReference

    /// The mHC hyper-connection mixing rule.
    public static let hyperConnections: Gate = .unverified

    /// DSpark's draft-chain construction.
    public static let dsparkDrafting: Gate = .unverified

    /// The vision tower's patch grid, image token layout, and the tower forward.
    ///
    /// The GEOMETRY matches `image_processor.py`: the resize plan is
    /// `plan_image_grid` + `safe_resize`, the patch grid is `llm_grid`, the span is
    /// `num_image_tokens = rows * (columns + 1) + 2`, and the patches are emitted in
    /// the checkpoint's `reshape(3, n_h, p, n_w, p)` order. `DeepSeekV41VisionForward`
    /// runs the tower itself and matches an independent mirror of `vision.py` on a
    /// fixture.
    ///
    /// Two things still keep the gate open: the token IDENTIFIER is unverified,
    /// because the reference writes the same `image_token_id` at every span position
    /// and trusts the tokenizer for it while this port renders the span from a
    /// marker string; and no reference vectors for a REAL image exist, so the tower
    /// is checked against a transcription rather than against the checkpoint's own
    /// output.
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
