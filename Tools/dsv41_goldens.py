#!/usr/bin/env python3
"""DeepSeek-V4.1 reference vectors, produced by the checkpoint's own modules.

The port's conformance gates (`DeepSeekV41Conformance`) stay `.unverified` until a
numeric vector from the reference implementation is checked in. That used to read
as "someone with a torch box has to send us numbers". It does not: the checkpoint
repository ships its reference source under `inference/`, and the modules can be
instantiated at a tiny config and run on CPU. No weights are needed -- 510 GB is
irrelevant to the question "does this rule compute what their rule computes".

So this script is the other half of the gates. It imports
`inference/model.py` from a downloaded snapshot, builds the module under test
with deterministic parameters, runs one forward pass, and prints the inputs
alongside the reference output. `DeepSeekV41GoldensTests` carries those floats as
literals and asserts the Swift rule reproduces them; flipping a gate is then a
one-line change that is actually backed by a comparison.

What is deliberately NOT exercised here:

  * the fused GPU kernels in `inference/kernel.py` (tilelang). They are stubbed
    out, so every `Linear` is built as a dense FP32 matrix and `linear()` takes
    the plain `F.linear` path. A golden that ran through their fused kernel would
    validate *their* kernel against *our* kernel, not the rule against the rule.
    Dense FP32 is also what makes the vectors cheap to read: the numbers here are
    the rule, not a quantization accident.
  * the real weights. Parameters are filled from a seeded generator, so the run
    needs the snapshot's `inference/` directory and nothing else.

Usage:

    python3 Tools/dsv41_goldens.py                 # every case, newest snapshot
    python3 Tools/dsv41_goldens.py engram          # one case
    python3 Tools/dsv41_goldens.py all --json out.json
    python3 Tools/dsv41_goldens.py all --snapshot <snapshot dir>

Cases: `engram` (Engram.forward), `compressor` (Compressor.forward, one block),
`compressor_state` (Compressor.forward over a decode sequence: which slot a
position fills and when a row comes out), `chunk` (which window slots each query
of a prefill chunk reads, `get_window_topk_idxs` for both of its branches, plus
the chunk branch of `Compressor.forward` against its decode branch), `router`
(Gate.forward, under both per-logit scorings), `dspark` (`DSparkBlock`'s
`forward_embed` seed plus its `forward_head` chain: the Markov bias, the
per-step draw, and the confidence ordering), and `vision` (`ViT.forward` +
`Aligner.forward` over a given patch grid, plus the spelling and id of the token
that addresses an image span).

Two gates deliberately have no case yet. `hyperConnections` needs
`kernel.hc_split_sinkhorn`, the real Sinkhorn, which is a tilelang kernel and so
is exactly the transcription this script exists to avoid repeating;
`candidateSelection` needs live cache state.

`dspark` covers the half of the drafter that decides which tokens a chain
contains. What it cannot cover, and what therefore keeps the gate open, is the
draft block's own attention and MoE: those run only through `forward_spec`, which
nothing in the checkpoint's own generator calls, so there is no reference output
for a draft block forward over a real prefix.

`chunk` exists because a batched prefill is the first place the port would run
more than one position through a layer, and the two rules that decide whether that
pass sees what a per-position walk sees are positional: the sliding-window mask
over a chunk, and the compressor splitting a chunk into complete groups plus a
trailing remainder. `get_window_topk_idxs` is pure torch, so the mask is asked
rather than restated; the attention that consumes it is not, which is why this
case pins the row lists and not attended values.

`vision` starts from a patch grid rather than an image: `load_image`'s PIL
bicubic resample is preprocessing, and pinning it here would pin Pillow instead
of the tower. Everything the tower and the aligner do to patches is covered,
including the resize-free half of `plan_image_grid` that `DeepSeekV41VisionLayout`
implements separately.

torch and numpy are required. `brew install pytorch` puts the Python package
inside the formula's venv rather than on `sys.path`, so point PYTHONPATH at it:

    PYTHONPATH=$(brew --prefix pytorch)/libexec/lib/python3.*/site-packages \
        python3 Tools/dsv41_goldens.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import types

# ---------------------------------------------------------------------------
# Import shims
# ---------------------------------------------------------------------------
#
# `inference/model.py` imports four things a CPU goldens run cannot have: the
# fused-kernel DSL (tilelang), the image helpers (Pillow), a primality library
# (sympy, used by the Engram bucket layout) and the vision tower. Each is
# replaced by the smallest honest stand-in rather than by installing a stack the
# goldens do not need.


def _isprime(value: int) -> bool:
    """Trial division, the same test `engram.find_next_prime` asks for."""
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    if value % 3 == 0:
        return value == 3
    factor = 5
    while factor * factor <= value:
        if value % factor == 0 or value % (factor + 2) == 0:
            return False
        factor += 6
    return True


def install_shims() -> None:
    if "sympy" not in sys.modules:
        sympy = types.ModuleType("sympy")
        sympy.isprime = _isprime
        sys.modules["sympy"] = sympy

    # `image_processor.py` needs Pillow at module scope for the marker constants
    # the model imports; nothing here reads an image.
    for name in ("PIL", "PIL.Image", "PIL.ImageOps"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.Image = module
            module.ImageOps = module
            sys.modules[name] = module

    def unavailable(name: str):
        def _refuse(*_args, **_kwargs):
            raise RuntimeError(
                f"{name} is a fused GPU kernel; the goldens run the dense path "
                "so the rule is under test rather than two kernels"
            )

        return _refuse

    if "kernel" not in sys.modules:
        kernel = types.ModuleType("kernel")
        for name in (
            "act_quant",
            "fp4_act_quant",
            "fp4_gemm",
            "fp8_gemm",
            "hc_split_sinkhorn",
            "sparse_attn",
        ):
            setattr(kernel, name, unavailable(name))
        sys.modules["kernel"] = kernel


# ---------------------------------------------------------------------------
# Snapshot and module setup
# ---------------------------------------------------------------------------


def find_snapshot(explicit: str | None) -> str:
    if explicit:
        return explicit
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/*"
    )
    candidates = [path for path in glob.glob(pattern) if os.path.isfile(
        os.path.join(path, "inference", "model.py")
    )]
    if not candidates:
        raise SystemExit(
            "no snapshot with inference/model.py found; pass --snapshot <dir> "
            "(the reference source is what these goldens come from)"
        )
    return sorted(candidates)[-1]


def load_reference(snapshot: str):
    """Imports the checkpoint's own `model` module with dense FP32 linears."""
    inference = os.path.join(snapshot, "inference")
    if not os.path.isfile(os.path.join(inference, "model.py")):
        raise SystemExit(f"{snapshot}: no inference/model.py")
    install_shims()
    if inference not in sys.path:
        sys.path.insert(0, inference)
    import torch

    import model as reference

    # One rank: the sharded imports are all `world_size`-parameterised and 1
    # makes them the identity.
    reference.world_size = 1
    reference.rank = 0
    # Dense FP32 rather than the checkpoint's fp8: see the module docstring.
    reference.default_dtype = torch.float32
    return torch, reference


def fill(module, torch, seed: int = 20251017, scale: float = 0.35):
    """Deterministic parameters, so the vectors are a property of the rule.

    A norm gain is filled around 1 rather than around 0 (a zero gain would make
    every normed output zero and hide a wrong reduction), and an E8M0 scale is
    filled with a power of two, which is the only kind of value it can hold.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.dtype == torch.float8_e8m0fnu:
                parameter.copy_(torch.full(parameter.shape, 2.0**-2).to(parameter.dtype))
            elif name.endswith("weight") and parameter.dim() == 1:
                parameter.copy_(
                    (1 + scale * torch.randn(parameter.shape, generator=generator)).to(parameter.dtype)
                )
            else:
                parameter.copy_(
                    (scale * torch.randn(parameter.shape, generator=generator)).to(parameter.dtype)
                )
    return module


class _UpcastRows:
    """Wraps the Engram table so its rows leave as FP32.

    `ParallelEngramEmbedding` returns bf16 because the table's storage is fp8 and
    the checkpoint's fused GEMM takes bf16 activations directly. That quantization
    belongs to the kernel path, not to the rule: `Engram.forward` is fp32 from the
    projection's split onward. Upcasting the row keeps the golden about the rule,
    and bf16 to fp32 is lossless, so nothing about the reference's numbers moves.
    """

    def __new__(cls, torch, inner):
        class _Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, indices):
                return self.inner(indices).float()

        return _Wrapper()


def rows_of(tensor) -> list:
    """A tensor as nested Python floats."""
    return tensor.detach().float().cpu().tolist()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def engram_case(torch, reference) -> dict:
    """`Engram.forward`: retrieved rows to a gated residual write.

    Small enough to read: dim 8, two hyper-connection copies, two hash columns
    (a bigram table with two heads), one 32-wide block per row (the table's fp8
    scale block size, so the dequantize is exercised), two positions.
    """
    from engram import EngramLayout

    args = reference.ModelArgs(
        dim=8,
        hc_mult=2,
        norm_eps=1e-6,
        engram_layer_ids=(1,),
        engram_num_embeddings=(4096,),
        engram_max_ngram_size=2,
        engram_vocab_size=512,
        engram_n_heads=2,
        engram_head_dim=32,
    )
    layout = EngramLayout.from_args(args)
    module = fill(reference.Engram(args, 1, layout), torch)
    module.embed = _UpcastRows(torch, module.embed)

    n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
    # Two positions, each with its own n_hash_cols row indices into the table.
    hash_ids = torch.tensor([[[5, 3000], [17, 4095]]], dtype=torch.long)
    # [B, L, hc_mult, dim], the residual the memory is written into.
    x = torch.tensor(
        [[
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
             [0.5, -1.0, 2.0, -3.0, 0.25, -0.75, 1.5, -2.0]],
            [[-1.0, 0.5, 1.25, -2.5, 3.0, -0.125, 2.0, 1.0],
             [2.0, 2.0, -2.0, -2.0, 1.0, 1.0, -1.0, -1.0]],
        ]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        out = module(x, hash_ids)
        rows = module.embed(hash_ids)
        kv = module.wkv(rows.flatten(-2))
        key, value = kv.split([module.hc_mult * module.dim, module.dim], dim=-1)
        key = key.float().unflatten(-1, (module.hc_mult, module.dim))
        weight = module.q_weight.float() * module.k_weight.float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + module.eps) * torch.rsqrt(
            key.square().mean(-1) + module.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * module.dim**-0.5
        gates = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(module.clamp_value).sqrt(), dot)
        )

    return {
        "gate": "engramCombination",
        "reference": f"inference/model.py Engram.forward ({snapshot_note()})",
        "config": {
            "dim": module.dim,
            "hc_mult": module.hc_mult,
            "head_dim": layout.head_dim,
            "n_heads": layout.n_heads,
            "max_ngram_size": layout.max_ngram_size,
            "n_hash_cols": n_hash_cols,
            "eps": module.eps,
        },
        "input": {
            "residual": rows_of(x),
            "hash_ids": rows_of(hash_ids),
            "rows": rows_of(rows),
            "wkv": rows_of(module.wkv.weight),
            "q_weight": rows_of(module.q_weight),
            "k_weight": rows_of(module.k_weight),
        },
        "expected": {
            # [L][hc][dim]: the residual with the gated value added.
            "streams": rows_of(out),
            # [L][hc]: the gate each copy was mixed with.
            "gates": rows_of(gates),
        },
    }


def compressor_case(torch, reference) -> dict:
    """`Compressor.forward`: one block of positions to one compressed latent.

    The softmax is over the group axis PER DIMENSION, which is the whole point of
    the gate: a kernel that shared one weight per row would produce the group
    mean instead. The four positions below disagree about which one should
    dominate, so a wrong axis is visible in every component.
    """
    args = reference.ModelArgs(
        dim=8,
        head_dim=4,
        norm_eps=1e-6,
        compress_ratios=[0, 2],
        max_batch_size=1,
    )
    module = fill(reference.Compressor(args, 1), torch)
    # Two groups of two positions: [B, L, dim].
    x = torch.tensor(
        [[
            [1.0, 0.0, 2.0, -1.0, 0.5, 1.5, -0.5, 3.0],
            [0.0, 1.0, -1.0, 2.0, 4.0, -2.0, 1.0, 0.25],
            [2.0, -3.0, 0.5, 1.0, -1.5, 0.75, 2.5, -0.25],
            [-1.0, 2.0, 3.0, 0.5, 1.0, -4.0, -2.0, 1.5],
        ]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        post_norm = module(x, start_pos=0)
        # The same forward with the norm removed: this is what isolates the
        # pooling rule, since `RMSNorm` is pinned by its own tests.
        original_norm = module.norm
        module.norm = torch.nn.Identity()
        pooled = module(x, start_pos=0)
        module.norm = original_norm

    return {
        "gate": "compressorPooling",
        "reference": f"inference/model.py Compressor.forward ({snapshot_note()})",
        "config": {
            "dim": args.dim,
            "head_dim": args.head_dim,
            "compress_ratio": module.compress_ratio,
            "norm_eps": args.norm_eps,
        },
        "input": {
            "positions": rows_of(x),
            "wkv": rows_of(module.wkv.weight),
            "wgate": rows_of(module.wgate.weight),
            "norm_gain": rows_of(module.norm.weight),
        },
        "expected": {
            # [groups][head_dim], still un-normalised: `pooledLatent`'s output.
            "pooled": rows_of(pooled),
            # [groups][head_dim], the module's own return value.
            "post_norm": rows_of(post_norm),
        },
    }


def router_case(torch, reference) -> dict:
    """`Gate.forward`: logits to selected experts and their mixture weights.

    Two scorings over the SAME weights, because they are not rescalings of each
    other: `sqrtsoftplus` (what the checkpoint ships) and `sigmoid` rank experts
    differently, and a port that picks the wrong one runs the wrong experts while
    every number still looks plausible.

    `gate_temp` is not 1 on purpose: the divide happens before the scoring
    function, so a port that divided after it would agree on a temperature of 1.
    """
    import copy

    args = reference.ModelArgs(
        dim=8,
        n_layers=1,
        n_routed_experts=6,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        gate_temp=0.7,
        norm_topk_prob=True,
        route_scale=2.5,
    )
    gate = fill(reference.Gate(1, args), torch)
    # [tokens, dim]: three tokens that disagree about which expert should win.
    x = torch.tensor(
        [
            [1.0, 0.5, -2.0, 1.5, 0.0, 2.0, -1.0, 0.25],
            [-1.0, 2.0, 1.0, -0.5, 1.5, -2.0, 0.5, 1.0],
            [0.25, -0.75, 0.5, 2.0, -1.5, 1.0, 1.0, -1.0],
        ],
        dtype=torch.float32,
    )

    cases = []
    with torch.no_grad():
        for scoring in ("sqrtsoftplus", "sigmoid"):
            module = copy.deepcopy(gate)
            module.score_func = scoring
            weights, indices = module(x)
            cases.append(
                {
                    "scoring": scoring,
                    "indices": rows_of(indices),
                    "weights": rows_of(weights),
                }
            )

    return {
        "gate": "routerScoring",
        "reference": f"inference/model.py Gate.forward ({snapshot_note()})",
        "config": {
            "dim": args.dim,
            "experts": args.n_routed_experts,
            "topk": args.n_activated_experts,
            "gate_temp": args.gate_temp,
            "route_scale": args.route_scale,
            "norm_topk_prob": args.norm_topk_prob,
        },
        "input": {
            "tokens": rows_of(x),
            "weight": rows_of(gate.weight),
            "bias": rows_of(gate.bias),
        },
        "cases": cases,
    }


def dspark_case(torch, reference) -> dict:
    """`DSparkBlock`: the target-state projection, then the draft chain.

    Two halves, because they fail differently. `forward_embed` turns the target
    model's concatenated hidden states into the draft block's conditioning row and
    seeds the chain's inputs with `[last token, noise, noise, ...]`. `forward_head`
    then walks the block, adds the Markov head's FULL-VOCAB bias for the token
    before each draft, draws one token per step, and scores each step's confidence.

    The chain below is written out here rather than taken from `forward_head`'s
    return value for one reason: `forward_head` draws with torch's global RNG, and
    a vector pinned to that RNG cannot be replayed by any other implementation. So
    the draws here come from a fixed uniform table that travels with the vectors,
    using the same rule the reference's `sample` implements — `argmax(p / -log u)`,
    i.e. Gumbel-max, since `-log(Exponential(1))` is Gumbel.
    """
    args = reference.ModelArgs(
        dim=8,
        head_dim=4,
        norm_eps=1e-6,
        hc_mult=2,
        max_batch_size=1,
        vocab_size=16,
        n_layers=1,
        n_mtp_layers=1,
        n_heads=2,
        rope_head_dim=2,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=2,
        moe_inter_dim=16,
        n_routed_experts=4,
        n_activated_experts=2,
        window_size=8,
        compress_ratios=(0, 0),
        dspark_block_size=3,
        dspark_noise_token_id=1,
        dspark_target_layer_ids=(0,),
        dspark_markov_rank=4,
        # The draft block's MoE is never run here, and an fp4 expert weight cannot
        # be seeded by a generator, so the dense path is the only one that works.
        expert_dtype=None,
    )
    block = fill(reference.DSparkBlock(args.n_layers, args), torch)
    # A Transformer hands the stage the backbone's embedding and head; this case
    # builds them itself so the vectors do not depend on a whole model.
    block.embed = fill(reference.ParallelEmbedding(args.vocab_size, args.dim), torch, seed=7)
    block.head = fill(
        reference.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps), torch, seed=8
    )

    vocab = args.vocab_size
    block_size = args.dspark_block_size
    seed_token = 11
    uniforms = torch.rand(
        (block_size, vocab), generator=torch.Generator().manual_seed(20251017), dtype=torch.float32
    )

    with torch.no_grad():
        main_hidden = torch.arange(
            1.0, 1.0 + args.dim * len(args.dspark_target_layer_ids)
        ).view(1, 1, -1)
        input_ids = torch.tensor([[seed_token]], dtype=torch.long)
        draft_input_ids = input_ids.new_full([1, block_size], args.dspark_noise_token_id)
        draft_input_ids[:, 0] = input_ids

        x, main_x = block.forward_embed(main_hidden, input_ids)
        collapsed = block.hc_pre(x, reference.make_identity_pre_mix(x, args.hc_mult))
        # What the block's own head would produce per draft position, before the
        # Markov bias: the chain's only dependence on the block forward.
        base_logits = block.head(block.norm(collapsed), full_logits=True)

        # The chain, with the draws driven by `uniforms` instead of torch's RNG.
        markov_logits, markov_embeds = [], []
        chosen = [seed_token]
        for step in range(block_size):
            # `output_ids[:, i]` is 1-D in the reference, which is what keeps the
            # bias [b, vocab] and the embedding [b, markov_rank].
            bias, embed = block.markov_head(torch.tensor([chosen[step]], dtype=torch.long))
            markov_logits.append(bias[0])
            markov_embeds.append(embed)
            step_logits = base_logits[0, step] + bias[0]
            probs = torch.softmax(step_logits / max(args.temperature, 1e-5), dim=-1, dtype=torch.float32)
            chosen.append(int(torch.argmax(probs / -torch.log(uniforms[step]))))
        confidence = block.confidence_head(
            collapsed, torch.stack(markov_embeds, dim=1)
        )

        # The same chain with the reference's own `sample` at temperature 0
        # (argmax), which is the shape a greedy port has to reproduce.
        greedy = [seed_token]
        for step in range(block_size):
            bias, _ = block.markov_head(torch.tensor([greedy[step]], dtype=torch.long))
            greedy.append(int(torch.argmax(base_logits[0, step] + bias[0])))

    return {
        "gate": "dsparkDrafting",
        "reference": f"inference/model.py DSparkBlock.forward_embed/forward_head ({snapshot_note()})",
        "config": {
            "dim": args.dim,
            "hc_mult": args.hc_mult,
            "vocab_size": vocab,
            "block_size": block_size,
            "noise_token_id": args.dspark_noise_token_id,
            "markov_rank": args.dspark_markov_rank,
            "temperature": args.temperature,
            "norm_eps": args.norm_eps,
        },
        "input": {
            "main_hidden": rows_of(main_hidden),
            "input_ids": seed_token,
            "main_proj": rows_of(block.main_proj.weight),
            "main_norm_gain": rows_of(block.main_norm.weight),
            "markov_embed": rows_of(block.markov_head.embed.weight),
            "markov_head": rows_of(block.markov_head.head.weight),
            "confidence_proj": rows_of(block.confidence_head.proj.weight),
            "uniforms": rows_of(uniforms),
            # The chain's block half: per draft position, the head's logits before
            # any Markov contribution.
            "base_logits": rows_of(base_logits[0]),
            "collapsed": rows_of(collapsed),
        },
        "expected": {
            "main_x": rows_of(main_x[0, 0]),
            "markov_logits": rows_of(torch.stack(markov_logits)),
            "confidence": rows_of(confidence[0]),
            "chained_ids": chosen,
            "greedy_ids": greedy,
            "mixed_logits": rows_of(
                torch.stack([base_logits[0, step] + markov_logits[step] for step in range(block_size)])
            ),
        },
    }


def compressor_state_case(torch, reference) -> dict:
    """`Compressor.forward` across a decode sequence: the accumulator lifecycle.

    `compressor` pins one block's pooling arithmetic. This pins what the SAME module
    does when positions arrive one at a time, which is where a real bug lived: the
    state is per module instance (`kv_state` / `score_state` are buffers), a position
    fills slot `start_pos % ratio`, and a row comes out only when
    `(start_pos + 1) % ratio == 0`. A port that shares one per-position block across
    the owners, or that counts slots instead of taking the position modulo the ratio,
    pools one layer's latent together with another's and produces plausible text.

    The norm is replaced by `Identity` so the returned rows are the pooled latent
    itself — the normalisation is pinned by the `compressor` case, and comparing
    against `pooledLatent` requires the raw form.

    What makes this suite discriminating is `kv_state` at the end: it holds one row
    per slot, each the projection of a DIFFERENT position, so a port that shares one
    block between owners, or that advances a slot counter instead of taking the
    position modulo the ratio, cannot reproduce it.
    """
    args = reference.ModelArgs(
        dim=8,
        head_dim=4,
        norm_eps=1e-6,
        compress_ratios=[0, 3],
        max_batch_size=1,
    )
    module = fill(reference.Compressor(args, 1), torch)
    module.norm = torch.nn.Identity()
    ratio = module.compress_ratio
    positions = 7

    # One distinct row per position, so a slot written twice or by the wrong
    # position is visible in `kv_state` rather than merely in the pooled output.
    x = (
        torch.arange(1.0, 1.0 + positions * args.dim, dtype=torch.float32).view(1, positions, args.dim)
        * 0.25
    )

    with torch.no_grad():
        outputs = []
        for position in range(positions):
            out = module(x[:, position : position + 1], position)
            outputs.append(None if out is None else rows_of(out[0, 0]))
        state = rows_of(module.kv_state[0])

    return {
        "gate": "compressorAccumulation",
        "reference": f"inference/model.py Compressor.forward, decode branch ({snapshot_note()})",
        "config": {
            "dim": args.dim,
            "head_dim": args.head_dim,
            "compress_ratio": ratio,
            "positions": positions,
            "norm_eps": args.norm_eps,
        },
        "input": {
            "positions": rows_of(x[0]),
            "wkv": rows_of(module.wkv.weight),
            "wgate": rows_of(module.wgate.weight),
        },
        "expected": {
            # One entry per position: null while the group is still filling.
            "outputs": outputs,
            # [ratio][head_dim]: the slots the sequence left behind. Slot i must hold
            # `wkv(x[p])` for the last p with `p % ratio == i`.
            "kv_state": state,
        },
    }


def chunk_case(torch, reference) -> dict:
    """What a batch of positions reads, and what it pools: the prefill-chunk rules.

    The port walks a prompt one position at a time today and is about to take the
    checkpoint's own shape, where a chunk of positions goes through the block in one
    pass. Two rules decide whether that pass sees what the walk sees, and both are
    positional rather than numeric:

      * the sliding-window mask. In a chunk every query attends over its own causal
        window, which `get_window_topk_idxs` lists as chunk positions, with -1 where
        a position has not been written yet. A decode step reuses the same ring but
        has to rotate it, oldest slot first, and report the slots still unwritten at
        the start of a sequence.
      * the compressor's group split. `Compressor.forward` at `start_pos=0` pools the
        chunk's complete groups and leaves the trailing remainder in its state, so a
        chunk whose length is not a multiple of the ratio produces `len // ratio`
        rows. Running the SAME module position by position through its decode branch
        has to produce the same rows, and that equality is what a batched port claims
        when it stops calling the module once per position.

    Both halves are asked of the reference rather than restated: the slot lists come
    from `get_window_topk_idxs` and the rows from `Compressor.forward` under both of
    its branches. `chunk_state` is recorded because the remainder the chunk did not
    pool is visible there.
    """
    window = 3
    positions = 5
    ratio = 2
    args = reference.ModelArgs(
        dim=8,
        head_dim=4,
        norm_eps=1e-6,
        window_size=window,
        compress_ratios=[0, ratio],
        max_batch_size=1,
    )

    chunk_slots = rows_of(reference.get_window_topk_idxs(window, 1, positions, 0))[0]
    decode = [
        {
            "start_pos": start_pos,
            "slots": rows_of(reference.get_window_topk_idxs(window, 1, 1, start_pos))[0][0],
        }
        for start_pos in (0, 1, 3, 7)
    ]

    # One distinct row per position, so a group pooled from the wrong positions is
    # visible in its values rather than only in the row count.
    x = (
        torch.arange(1.0, 1.0 + positions * args.dim, dtype=torch.float32).view(1, positions, args.dim)
        * 0.25
    )
    chunk_module = fill(reference.Compressor(args, 1), torch)
    chunk_module.norm = torch.nn.Identity()
    step_module = fill(reference.Compressor(args, 1), torch)
    step_module.norm = torch.nn.Identity()

    with torch.no_grad():
        chunk_rows = rows_of(chunk_module(x, 0)[0])
        step_rows = []
        produced_at = []
        for position in range(positions):
            out = step_module(x[:, position : position + 1], position)
            produced_at.append(out is not None)
            if out is not None:
                step_rows.append(rows_of(out[0, 0]))
        chunk_state = rows_of(chunk_module.kv_state[0])

    return {
        "gate": "chunkAddressing",
        "reference": (
            "inference/model.py get_window_topk_idxs and Compressor.forward, both branches "
            f"({snapshot_note()})"
        ),
        "config": {
            "window_size": window,
            "positions": positions,
            "compress_ratio": ratio,
            "dim": args.dim,
            "head_dim": args.head_dim,
        },
        "input": {
            "positions": rows_of(x[0]),
            "wkv": rows_of(chunk_module.wkv.weight),
            "wgate": rows_of(chunk_module.wgate.weight),
        },
        "expected": {
            # [positions][min(positions, window)]: the chunk positions each query reads,
            # -1 where that query has not reached the slot yet.
            "chunk_window_slots": chunk_slots,
            # Ring SLOTS, not positions: where a query's window sits once the ring has
            # been written. Rotated oldest first, -1 where a slot is still unwritten.
            "decode_window_slots": decode,
            # [positions // ratio][head_dim], pooled by the chunk branch.
            "chunk_rows": chunk_rows,
            # The same rows from the decode branch, one position at a time.
            "step_rows": step_rows,
            # One entry per position: whether that position completed a group.
            "step_produced_at": produced_at,
            # [ratio][head_dim]: the trailing remainder the chunk left unpooled.
            "chunk_state": chunk_state,
        },
    }


def _vision_identifier(snapshot: str) -> dict:
    """What token an image span is spelled with, from the checkpoint's own files.

    The port renders an image span as a run of one marker string, so the marker's
    spelling IS the position's token id. Three files agree about it and none of them
    is the port's source: `encoding/encoding.py` defines `IMAGE_PLACEHOLDER`, the
    tokenizer lists that spelling among its added tokens, and the config's
    `image_token_id` is the id it was added under. `image_processor.prepare_vl_inputs`
    cross-checks the last two at runtime and raises when they disagree.
    """
    with open(os.path.join(snapshot, "config.json"), "r", encoding="utf-8") as handle:
        config = json.load(handle)
    with open(os.path.join(snapshot, "tokenizer.json"), "r", encoding="utf-8") as handle:
        tokenizer = json.load(handle)
    with open(os.path.join(snapshot, "encoding", "encoding.py"), "r", encoding="utf-8") as handle:
        encoding_source = handle.read()

    match = re.search(r'^IMAGE_PLACEHOLDER = "([^"]*)"', encoding_source, re.MULTILINE)
    if match is None:
        raise SystemExit("encoding/encoding.py no longer defines IMAGE_PLACEHOLDER")
    placeholder = match.group(1)

    added = {entry["content"]: entry.get("id") for entry in tokenizer.get("added_tokens") or []}
    image_token_id = config.get("image_token_id")
    if image_token_id is None:
        image_token_id = (config.get("text_config") or {}).get("image_token_id")
    spelled_at_config_id = next(
        (text for text, ident in added.items() if ident == image_token_id), None
    )

    return {
        "placeholder": placeholder,
        "placeholder_codepoints": [ord(character) for character in placeholder],
        "placeholder_token_id": added.get(placeholder),
        "config_image_token_id": image_token_id,
        "token_spelled_at_config_id": spelled_at_config_id,
        "token_spelled_at_config_id_codepoints": (
            [ord(character) for character in spelled_at_config_id] if spelled_at_config_id else None
        ),
    }


def vision_case(torch, reference) -> dict:
    """`ViT.forward` then `Aligner.forward`: patch grid to language-width rows.

    Small enough to read and wide enough to be discriminating: hidden 8 with two
    heads (so the 2D rotary has a real frequency vector and is shared across heads),
    a 2x3 patch grid of 2x2 pixels (an ODD patch-column count, so the aligner's
    padding to the next multiple of the ratio is exercised), one block, and a
    downsample ratio of 2 (two aligner rows of two cells over hidden*4-wide cells).

    The patches are given, not resized from an image: `load_image`'s PIL bicubic
    resample is preprocessing, and pinning it would pin Pillow. Everything the tower
    does to a patch grid is here.

    Stage outputs come from forward hooks on the reference's own modules rather than
    from a re-implementation of `Block.forward`, so a failing vector names the stage
    that broke without this script restating the rule it is checking.

    The identifier half is here too, because the gate has two reasons: the tower's
    arithmetic (below) and the spelling of the token that addresses an image span.
    """
    args = reference.ModelArgs(
        dim=5,
        hc_mult=1,
        max_batch_size=1,
        vision_n_layers=1,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=4,
        vision_patch_size=2,
        vision_rope_theta=100.0,
        vision_downsample_ratio=2,
        vision_max_n_token=1024,
        vision_min_pixels=0,
    )
    vit = fill(reference.ViT(args), torch)
    aligner = fill(reference.Aligner(args), torch)

    patch_size = args.vision_patch_size
    n_h, n_w = 2, 3
    # One row per patch of `3 * patchSize^2` values, channel-major, as
    # `load_image`'s `reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4)` emits them.
    patches = torch.tensor(
        [
            ((token * 7 + dimension * 3) % 5 - 2) * 0.25
            for token in range(n_h * n_w)
            for dimension in range(3 * patch_size * patch_size)
        ],
        dtype=torch.float32,
    ).view(n_h * n_w, 3, patch_size, patch_size)

    stages = {}
    handles = []
    for name, module in (
        ("patch_embed", vit.patch_embed),
        ("attn", vit.blocks[0].attn),
        ("mlp", vit.blocks[0].mlp),
        ("block", vit.blocks[0]),
        ("norm", vit.norm),
    ):

        def capture(_module, _inputs, output, name=name):
            stages[name] = output.detach().float()

        handles.append(module.register_forward_hook(capture))

    with torch.no_grad():
        # `ViT.forward(patches, n_h, n_w)` builds the rotary table itself; the aligner
        # then unshuffles the tower's output on the SAME patch grid.
        tower = vit(patches, n_h, n_w)
        aligned = aligner(tower, n_h, n_w)

    for handle in handles:
        handle.remove()
    if set(stages) != {"patch_embed", "attn", "mlp", "block", "norm"}:
        raise SystemExit(f"vision hooks did not all fire: {sorted(stages)}")
    if not torch.equal(stages["norm"], tower):
        raise SystemExit("the final-norm hook and the tower's return value disagree")

    block = vit.blocks[0]
    return {
        "gate": "visionPatching",
        "reference": (
            f"inference/vision.py ViT.forward + Aligner.forward ({snapshot_note()})"
        ),
        "config": {
            "hidden": args.vision_dim,
            "heads": args.vision_n_heads,
            "intermediate": args.vision_inter_dim,
            "patch_size": patch_size,
            "downsample_ratio": args.vision_downsample_ratio,
            "rope_theta": args.vision_rope_theta,
            "rope_dim": vit.rope_dim,
            "text_hidden": args.dim,
            "rows": n_h,
            "columns": n_w,
            "norm_eps": 1e-6,
        },
        "input": {
            "patches": rows_of(patches.view(n_h * n_w, -1)),
            "patch_embed_weight": rows_of(vit.patch_embed.proj.weight),
            "patch_embed_bias": rows_of(vit.patch_embed.proj.bias),
            "block_norm1_gain": rows_of(block.norm1.weight),
            "block_wqkv_weight": rows_of(block.attn.wqkv.weight),
            "block_wqkv_bias": rows_of(block.attn.wqkv.bias),
            "block_wo_weight": rows_of(block.attn.wo.weight),
            "block_wo_bias": rows_of(block.attn.wo.bias),
            "block_norm2_gain": rows_of(block.norm2.weight),
            "block_w1_weight": rows_of(block.mlp.w1.weight),
            "block_w2_weight": rows_of(block.mlp.w2.weight),
            "final_norm_gain": rows_of(vit.norm.weight),
            "aligner_w1_weight": rows_of(aligner.w1.weight),
            "aligner_w1_bias": rows_of(aligner.w1.bias),
            "aligner_w2_weight": rows_of(aligner.w2.weight),
            "aligner_w2_bias": rows_of(aligner.w2.bias),
        },
        "expected": {
            # [tokens][hidden]: the patch projection, before any block.
            "patch_embed": rows_of(stages["patch_embed"]),
            # [tokens][hidden]: attention's output after `wo`, rotary included.
            "attn": rows_of(stages["attn"]),
            # [tokens][hidden]: the MLP branch alone, after `w2`.
            "mlp": rows_of(stages["mlp"]),
            # [tokens][hidden]: the residual stream after the block.
            "block": rows_of(stages["block"]),
            # [tokens][hidden]: `ViT.forward`'s return value, i.e. the final norm.
            "normalized": rows_of(stages["norm"]),
            # [cells][text_hidden]: one language-width row per aligner cell.
            "aligner": rows_of(aligned),
        },
        "identifier": _vision_identifier(_SNAPSHOT),
    }


CASES = {
    "engram": engram_case,
    "compressor": compressor_case,
    "compressor_state": compressor_state_case,
    "chunk": chunk_case,
    "router": router_case,
    "dspark": dspark_case,
    "vision": vision_case,
}

_SNAPSHOT = ""


def snapshot_note() -> str:
    return _SNAPSHOT or "unknown snapshot"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "case", nargs="?", default="all", choices=["all", *CASES], help="which vectors to emit"
    )
    parser.add_argument("--snapshot", help="snapshot directory holding inference/")
    parser.add_argument("--json", help="write the vectors here instead of stdout")
    options = parser.parse_args(argv)

    global _SNAPSHOT
    snapshot = find_snapshot(options.snapshot)
    _SNAPSHOT = snapshot
    torch, reference = load_reference(snapshot)

    names = list(CASES) if options.case == "all" else [options.case]
    payload = {
        "meta": {
            "torch": torch.__version__,
            "snapshot": snapshot,
            "note": (
                "Dense FP32 module weights and one rank. Generated by "
                "Tools/dsv41_goldens.py from the checkpoint's own inference/ "
                "source; no checkpoint weights were read."
            ),
        }
    }
    for name in names:
        payload[name] = CASES[name](torch, reference)

    text = json.dumps(payload, indent=2)
    if options.json:
        with open(options.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {options.json}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
