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
    /// What it does NOT cover: the fp8 storage width, which is exercised by the
    /// kernel tests, and the accumulator state the group is pooled from, which is
    /// `compressorAccumulation`.
    public static let compressorPooling: Gate = .verifiedAgainstReference

    /// Where a decode position's latent waits, and when it is pooled.
    ///
    /// MATCHED AGAINST `Compressor.forward`'s decode branch: the position fills slot
    /// `start_pos % ratio` of ITS OWN module's state, and a compressed row comes out
    /// only when `(start_pos + 1) % ratio == 0`. `Tools/dsv41_goldens.py
    /// compressor_state` runs the reference module over a seven-position sequence at
    /// ratio 3, and `DeepSeekV41CompressorStateGoldenTests` pins what came out: rows
    /// at positions 2 and 5 and nothing between, each slot holding the projection of
    /// the newest position congruent to it, and each row pooled from its whole group
    /// in ascending position order.
    ///
    /// This gate exists because a bug shipped past the pooling gate and produced
    /// plausible text: the port kept one per-position group shared between the four
    /// compressed-KV owners, and the ratio-1 owner rewrote slot 0 every position, so
    /// the ratio-2 owners pooled their latent together with another layer's.
    ///
    /// The per-owner partition itself (`compressorOwnerSlots`) is host addressing
    /// rather than a numeric rule, and is pinned by its own tests; what this gate
    /// covers is the geometry those offsets must obey.
    public static let compressorAccumulation: Gate = .verifiedAgainstReference

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
    ///
    /// The CHAIN is matched against `DSparkBlock.forward_head` in the checkpoint's
    /// `inference/model.py`: the Markov head is a full-vocab bias projected from the
    /// token before each draft and ADDED to the draft head's logits, the draw is
    /// argmax at temperature 0 and Gumbel-max above it, and each step conditions on
    /// the token it drew itself. `Tools/dsv41_goldens.py` runs that chain at a tiny
    /// config with the draws driven by a recorded uniform table (torch's own RNG is
    /// not replayable), and `DeepSeekV41DSparkGoldenTests` carries both the greedy
    /// and the sampled chain; the two disagree, so a port that implements only one
    /// of them cannot pass.
    ///
    /// What keeps this gate open is the draft BLOCK: `main_proj`/`main_norm`, its
    /// window-cache attention, and the MoE under the DSpark expert counts. Those run
    /// only through `forward_spec`, and nothing in the checkpoint's generator calls
    /// it, so there is no reference output for a draft block forward over a real
    /// prefix. The acceptance/emission rule and the confidence gate are this port's
    /// too: the reference defines the drafter but ships no speculative loop.
    public static let dsparkDrafting: Gate = .unverified

    /// The vision tower's patch grid, image token layout, and the tower forward.
    ///
    /// The GEOMETRY matches `image_processor.py`: the resize plan is
    /// `plan_image_grid` + `safe_resize`, the patch grid is `llm_grid`, the span is
    /// `num_image_tokens = rows * (columns + 1) + 2`, and the patches are emitted in
    /// the checkpoint's `reshape(3, n_h, p, n_w, p)` order.
    ///
    /// The TOWER matches the reference's own `ViT`/`Aligner`: `Tools/dsv41_goldens.py
    /// vision` instantiates both at a tiny config, feeds a given patch grid, and
    /// records the patch projection, the attention branch, the MLP branch, the block
    /// output, the final norm and the aligner rows.
    /// `DeepSeekV41VisionGoldenTests` carries them. The fixture uses two heads (the
    /// 2D rotary is shared, so one head would hide a per-head table), three patch
    /// columns (odd, so the aligner's padding to the ratio is exercised) and one
    /// block.
    ///
    /// The token IDENTIFIER is matched too, which is the part that used to be a
    /// guess. An image span is a run of one token: `prepare_vl_inputs` writes
    /// `image_token_id` at every position and keeps the start/image/newline/end
    /// distinction in a parallel `token_types` array. `encoding/encoding.py` names
    /// that spelling `IMAGE_PLACEHOLDER`, the tokenizer lists that exact string as
    /// added token 129,264, and `config.json`'s `image_token_id` is 129,264;
    /// `prepare_vl_inputs` asserts the last two agree. The same case emits those
    /// codepoints and `DeepSeekV41VisionMarkers.familyDefault` is asserted against
    /// them, so the marker cannot drift into a spelling the tokenizer splits.
    public static let visionPatching: Gate = .verifiedAgainstReference

    /// Every gate, paired with the label the load log prints.
    public static var all: [(name: String, gate: Gate)] {
        [
            ("Engram hashing", DeepSeekV41EngramHasher.conformance == .verifiedAgainstReference
                ? .verifiedAgainstReference : .unverified),
            ("compressor pooling", compressorPooling),
            ("compressor accumulation", compressorAccumulation),
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
