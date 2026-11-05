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
//  reference run -- or, for a gate about a PATH rather than a numeric rule, by a
//  comparison against the walk on the published checkpoint (see
//  `batchedDevicePrefill`). The gates are diagnostics rather than switches: a stage
//  the port has implemented runs whether or not its rule has been matched against a
//  reference, and the gate is how far to trust the numbers it produces.
//

import Foundation

/// Whether the family can serve a turn, which it can: a rendered prompt goes
/// through `DeepSeekV41TurnRunner` on the published checkpoint, and every numeric
/// rule in the graph it runs has a reference (see `DeepSeekV41Conformance`).
/// Discovery reads this to omit an "unavailable" caveat from the model row, and
/// `ModelConfig.supportsThinkingLevels` reads it because the dialect renders the
/// checkpoint's own `Reasoning Effort:` line.
///
/// What it advertises is deliberately TEXT turns: this is the gate discovery and
/// the thinking-level UI read. The vision tower is a separate promise
/// (`ModelConfig.supportsVision`), and as of `#43` it runs as one device pass; an
/// image reaches an in-app turn, but the API path still has no span cache and is
/// not promised. DSpark is not wired either: `#26` measured the speculative round
/// losing to walking at every width, so it is an optimization this flag does not
/// promise, and cache rewind for a rejected suffix is unbuilt.
///
/// The value is a constant rather than derived from `unverifiedSummary`: a gate
/// records how far to trust a number, and one open rule for a stage the text path
/// never runs must not disable a text turn that works. It is also not a download
/// gate; discovery lists whatever is in the hub cache either way.
nonisolated public enum DeepSeekV41Availability {
    public static let isRuntimeAvailable = true
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

    /// Which compressed rows a query reads: the indexer's two-level top-k.
    ///
    /// MATCHED AGAINST `Indexer.forward` and `select_candidate_blocks`. Three rules
    /// decide the answer and each is pinned separately by
    /// `Tools/dsv41_goldens.py indexer` and `DeepSeekV41IndexerGoldenTests`:
    ///
    /// * the score. `(relu(dot) * weight).sum(heads)` against the shared index key,
    ///   with `index_head_dim ** -0.5 * index_n_heads ** -0.5` on the WEIGHTS. The
    ///   rectifier is per head, so a head that votes against a row removes only its
    ///   own contribution, and the heads are summed rather than maxed.
    /// * the reach mask. A compressed row becomes visible to a query only once the
    ///   query has passed its last token (`arange(1, seqlen + 1) // ratio`), and the
    ///   cut runs BEFORE that filter, so a row a query cannot see can take a slot and
    ///   then leave the selection one short.
    /// * the two cuts. A block scores as its best row, the partly-filled block at the
    ///   frontier is pinned at `+inf` so it cannot lose to an older full block, and
    ///   only the candidate source publishes that mask: the layers after it score
    ///   with their OWN query and weights inside it.
    ///
    /// The port read the newest `index_topk` rows until this was verified, which is
    /// the same list only while nothing competes for the slots. On the published
    /// checkpoint `index_topk` is 512 and the twenty ratio-1 layers read one
    /// compressed row per token, so the two disagreed from position 513 on.
    ///
    /// VERIFIED AT THE PUBLISHED GEOMETRY, on the checkpoint's own tensors: a real
    /// 576-position prefill of layer 20 is dumped through
    /// `DeepSeekV41MetalForward.indexerInputRequest` and
    /// `Tools/dsv41_goldens.py indexer_real` runs `Indexer.forward` on those exact
    /// activations with the real `wq_b` / `weights_proj` / `wk` / `k_norm`. The
    /// port's 576 scores differ from the reference's by at most 2.6e-6 in absolute
    /// terms -- float summation order, since the scale is folded into the weights in
    /// one and onto each term in the other -- and the 512 selected rows are
    /// identical. `DeepSeekV41RealIndexerReScareTests` carries those numbers, which a
    /// reproducible dump made worth checking in: two independently prefilled dumps are
    /// byte-identical.
    ///
    /// What it does NOT cover, now measured rather than assumed: the FP4 activation
    /// quantization of the indexer's query and keys, which the dense path skips
    /// everywhere else too. Emulating the reference's own `fp4_act_quant` (E2M1, one
    /// power-of-two E8M0 scale per 32-value block) moves exactly 3 of the 512 selected
    /// rows at that position and drops 3 others -- the same six on both dumps, and all
    /// of them by less than a quarter of a score unit, against a cut that separates its
    /// neighbours by 0.0025 -- so the deviation is real and bounded. The order among
    /// exactly equal scores is also unspecified: a row every head votes against scores
    /// zero, and the port breaks that tie to the lower row index
    /// (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #30" and "#32").
    public static let candidateSelection: Gate = .verifiedAgainstReference

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
    /// pass: the norm, the linear stages, the RoPE, the mHC coefficients, the router
    /// and the routed mixture cover every position at once (the linear kernels read
    /// the token from `thread_position_in_grid.y`, the slot-grid ones from the axis
    /// their own doc names, and the norm and embedding kernels are scalar and
    /// therefore X), while the stages that cannot batch run per position inside the
    /// same command buffer — the cache write and the attention that reads it, the mHC
    /// pre/post mix, the shared expert, and the Engram stage.
    ///
    /// Two bounds decide the shape rather than convenience. The cache is a RING of
    /// `window` rows, so writing position `p + 1` evicts a row a query at `p` still
    /// needs: the write, the pool and the attention therefore stay in one per-position
    /// loop, and a block is capped at the window. The second bound is the expert
    /// ARENA: a slot may only be refilled once the kernels reading it have finished,
    /// so the mixture is one submission only when the block's deduplicated expert
    /// union can be resident all at once. The arena the memory plan sizes holds
    /// hundreds of slots, and a block of `chunkSlots` (at most eight) routes at most
    /// 48 at top-k 6, so the union fits and one command buffer covers the block; an
    /// arena kept below that falls back to one fill-and-wait per position instead of
    /// overwriting a slot a kernel is still reading.
    ///
    /// The device forward was first checked without a checkpoint:
    /// `DeepSeekV41SyntheticCheckpoint` publishes the checkpoint's own tensor names at
    /// widths the config derives, feeds that graph and the host reference identical
    /// values, and `DeepSeekV41DeviceGraphParityTests` compares a block with the walk
    /// (bit for bit, across a block boundary and a ring wrap) and the walk with the
    /// reference stage by stage. That is what found the three wiring bugs the old
    /// note predicted nothing could — a byte offset read as floats, a grouped output
    /// projection splitting its rows twice, and a RoPE batch rotating one row.
    ///
    /// THE PUBLISHED CHECKPOINT, which is what closed the gate.
    /// `DeepSeekV41PublishedTurnTests
    /// .testABatchedPrefillMatchesTheWalkOnThePublishedCheckpoint` runs two
    /// consecutive blocks of the real model and asks the head for EVERY slot, then
    /// compares each position with a per-position walk. The synthetic fixture cannot
    /// produce the published shapes, the 128-row ring, the ratio-2 compressor or the
    /// real expert arena; that run does, and it covers a continuation (the second
    /// block appends to a cache the first filled) rather than only a prompt from
    /// position 0. The capture is also the shape a verify needs: the target's choice
    /// for draft `p` comes from the logits at position `p - 1`, so one block returns
    /// the whole run side by side, with the head projecting all slots in one grid.
    ///
    /// What this does NOT claim: that a batched verify is faster than walking. `#26`
    /// measured the drafter's acceptance and the per-width costs, and a narrow verify
    /// loses to the walk at the widths a chain actually submits. No drafter is wired
    /// into the turn loop, and cache rewind for a rejected suffix is still unbuilt,
    /// because those are costs in service of a negative result. The turn loop keeps
    /// its one batching use — a from-position-0 prompt through
    /// `DeepSeekV41PositionForward.prefillChunk` — while a decode step walks
    /// (numbers in `Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #1" to "#5" and "#33").
    public static let batchedDevicePrefill: Gate = .verifiedAgainstReference

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
