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

    /// Which cache rows each query of a chunk reads, and what a chunk pools.
    ///
    /// MATCHED AGAINST `get_window_topk_idxs` and both branches of
    /// `Compressor.forward`. This is the rule a batched prefill needs and a
    /// per-position walk does not: the walk addresses one query at a time, while a
    /// chunk has to know, for every query in it at once, which window rows that query
    /// may read and which compressed rows already exist.
    ///
    /// `Tools/dsv41_goldens.py chunk` asks the reference for both: a five-position
    /// chunk over a three-slot window (so every query's window differs, and a query
    /// at the end of the chunk reads rows the ring has since rotated), and the same
    /// window as a decode step lists it, rotated oldest-first with unwritten slots
    /// marked. It also runs `Compressor.forward` over the chunk and position by
    /// position, which is the equality a batched pool claims: `len // ratio` rows,
    /// pooled from the same positions, with the trailing remainder left unpooled.
    /// `DeepSeekV41ChunkGoldenTests` carries the vectors.
    ///
    /// What it does NOT cover: attended values, because `sparse_attn` is a tilelang
    /// kernel and cannot run here. The row lists are the rule; the arithmetic over
    /// them is `DeepSeekV41AttentionReference`'s own and is pinned separately.
    public static let chunkAddressing: Gate = .verifiedAgainstReference

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

    /// The mHC residual mixing rule.
    ///
    /// MATCHED AGAINST `Block.hc_mixes`, `Block.hc_pre` and `Block.hc_post` in the
    /// checkpoint's `inference/model.py`. `hc_*_fn` projects the FLATTENED stack,
    /// scaled by `rsqrt(mean(x^2) + norm_eps)`, onto `(2 + mult) * mult` logits; the
    /// first `mult` become `pre = sigmoid(...) + eps`, the next `mult` become
    /// `post = 2 * sigmoid(...)`, and the `mult × mult` tail becomes `comb` through
    /// the Sinkhorn split. A sub-layer collapses the stack with the `pre` the
    /// PREVIOUS sub-layer produced, expands its own output back out through `post`
    /// and `comb`, and hands its own `pre` on; the head finally collapses the stack
    /// with the last one. `Tools/dsv41_goldens.py hc` runs those three block methods
    /// as written (two positions, two copies, a three-round split) and
    /// `DeepSeekV41HyperConnectionGoldenTests` carries the vectors.
    ///
    /// What it does NOT cover, and the reason this gate took longer to close than the
    /// rest: the split itself is a tilelang kernel (`hc_split_sinkhorn`), so the
    /// goldens transcribe its body from `inference/kernel.py` instead of executing it.
    /// Its schedule is pinned separately (a one-round case where the columns are exact
    /// and the rows are not), and everything around it — the projection, the RMS
    /// scaling, the sigmoid pre/post, the collapse and the expansion — is the
    /// reference's own code. The softmax / `+eps` / column-normalize order is this
    /// port's reading of that kernel, not a number the reference produced.
    ///
    /// The device path runs the same rule in four kernels: `dsv41_hc_project_f32`
    /// (the FP32 projection of the flattened stack), `dsv41_hc_split_f32` (the
    /// three-way split, sharing the Sinkhorn schedule with `dsv41_hc_sinkhorn`),
    /// `dsv41_hc_pre_f32` (the collapse, which also snapshots the block input) and
    /// `dsv41_hc_post_f32` (the expansion). `DeepSeekV41HyperConnectionKernelTests`
    /// compares all four with this host rule on the same inputs, so a disagreement
    /// is a kernel bug. What that cannot show, because the checkpoint has never been
    /// run, is the wiring: which mix each stage reads and where the residual is
    /// snapshotted.
    public static let hyperConnections: Gate = .verifiedAgainstReference

    /// DSpark's draft-chain construction and the block it drafts with.
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
    /// The BLOCK is matched by `Tools/dsv41_goldens.py dspark_block`: two
    /// `DSparkBlock` stages run at a tiny config, with the window cache SEEDED by a
    /// prefill call and then written and read by a draft call, stage 0's attention
    /// (its input, its rotated queries and `sparse_attn`'s output) recorded
    /// alongside both stages' residual streams, the collapse the head reads, the
    /// head's own logits and the chain they produce.
    /// `DeepSeekV41DSparkBlockGoldenTests` carries the vectors. What that covers
    /// that a 2-copy hyper-connection test cannot: `hc_post` sums its mixing matrix
    /// over the FIRST axis, so a port that walks a row instead of a column mixes
    /// every copy with the wrong neighbours — invisible at two copies with a nearly
    /// symmetric Sinkhorn matrix, and this case caught it.
    ///
    /// What stays this port's own: the acceptance/emission rule and the confidence
    /// gate. The reference defines the drafter but ships no speculative loop, and
    /// nothing in its generator calls `forward_spec`.
    ///
    /// `sparse_attn` is the one rule transcribed rather than executed here (it is a
    /// tilelang kernel), on the same terms as `hyperConnections`' split.
    public static let dsparkDrafting: Gate = .verifiedAgainstReference

    /// Whether a prompt is prefilled as one batched pass on the device.
    ///
    /// The RULE a batched pass needs is matched: `chunkAddressing` pins which cache
    /// rows each query of a chunk reads and which rows a chunk pools, and
    /// `DeepSeekV41ChunkGoldenTests` carries the reference vectors.
    ///
    /// The GRAPH around the rule now exists. `DeepSeekV41MetalForward.prefillChunk`
    /// runs a prompt in blocks of `workspace.slots` positions, and each block is one
    /// pass: the norm, the linear stages and the RoPE dispatches cover every position
    /// at once (the kernels read the token from `thread_position_in_grid.y`, except
    /// the norm and embedding kernels, whose index is scalar and therefore X), while
    /// the stages that cannot batch run per position inside the same command buffer —
    /// the cache write and the attention that reads it, the mHC coefficients, the
    /// Engram stage, and the routed mixture.
    ///
    /// Two bounds decide the shape rather than convenience. The cache is a RING of
    /// `window` rows, so writing position `p + 1` evicts a row a query at `p` still
    /// needs: the write, the pool and the attention therefore stay in one per-position
    /// loop, and a block is capped at the window. And the expert ARENA holds a handful
    /// of experts, so a block's union does not fit: the mixture stays per position,
    /// with the wait that keeps the next read out of the slot the previous kernels are
    /// still reading. Batching the mixture needs an arena redesign, not a wider grid.
    ///
    /// What changed about the gate's own reasoning: the local shards are still
    /// header-only stubs, but the device forward is no longer unrun.
    /// `DeepSeekV41SyntheticCheckpoint` publishes the checkpoint's own tensor names at
    /// widths the config derives, feeds that graph and the host reference identical
    /// values, and `DeepSeekV41DeviceGraphParityTests` compares a block with the walk
    /// (bit for bit, across a block boundary and a ring wrap) and the walk with the
    /// reference stage by stage. That is what found the three wiring bugs the old
    /// note predicted nothing could — a byte offset read as floats, a grouped output
    /// projection splitting its rows twice, and a RoPE batch rotating one row.
    ///
    /// What stays open: no comparison against the checkpoint's own numbers at real
    /// weights, and no measurement of what the pass is worth on real weights. The turn
    /// loop already routes a from-position-0 prompt through
    /// `DeepSeekV41PositionForward.prefillChunk`, so nothing outside the executor has
    /// to change either way.
    public static let batchedDevicePrefill: Gate = .unverified

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
            ("chunk addressing", chunkAddressing),
            ("router scoring", routerScoring),
            ("Engram combination", engramCombination),
            ("candidate selection", candidateSelection),
            ("hyper-connections", hyperConnections),
            ("DSpark drafting", dsparkDrafting),
            ("batched device prefill", batchedDevicePrefill),
            ("vision patching", visionPatching)
        ]
    }

    /// The unverified rules, phrased for the load log. Empty means every rule was
    /// matched against the reference.
    public static var unverifiedSummary: [String] {
        all.filter { !$0.gate.isVerified }.map(\.name)
    }
}
