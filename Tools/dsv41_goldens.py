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

Cases: `engram` (Engram.forward), `compressor` (Compressor.forward), `router`
(Gate.forward, under both per-logit scorings), and `dspark` (`DSparkBlock`'s
`forward_embed` seed plus its `forward_head` chain: the Markov bias, the per-step
draw, and the confidence ordering).

Two gates deliberately have no case yet. `hyperConnections` needs
`kernel.hc_split_sinkhorn`, the real Sinkhorn, which is a tilelang kernel and so
is exactly the transcription this script exists to avoid repeating;
`candidateSelection` needs live cache state.

`dspark` covers the half of the drafter that decides which tokens a chain
contains. What it cannot cover, and what therefore keeps the gate open, is the
draft block's own attention and MoE: those run only through `forward_spec`, which
nothing in the checkpoint's own generator calls, so there is no reference output
for a draft block forward over a real prefix.

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


CASES = {
    "engram": engram_case,
    "compressor": compressor_case,
    "router": router_case,
    "dspark": dspark_case,
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
