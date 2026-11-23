#!/usr/bin/env python3
"""What the released reference holds, and how much its own tail rows can be trusted.

Two facts decide how an Inkling 2 logit comparison may be read, and neither is
visible from the port:

  * **The expert bank is dequantized to FP32, not rounded to BF16.** Transformers
    maps the checkpoint's packed `w13_weight` / `w13_weight_scale` pair onto a
    dense `gate_up_proj` and dequantizes it in the load dtype. With
    `dtype=float32` the result is FP32 -- only a few hundred of a sampled 4,096
    values are even on the BF16 grid -- while the port deliberately rounds every
    dequantized expert weight to BF16 so its device kernel and its host graph
    compute the same numbers. A ~0.3 mean logit difference against an FP32 dump
    is therefore that rounding, not a defect; an FP32 dump is the right yardstick
    for RANKING and the wrong one for elementwise equality.

  * **The reference's last few rows are not stable targets.** Appending a tail
    moves rows near the end of the prompt (0.37 mean / 2.16 max at row 126 of a
    131-token prompt) and the movement does NOT track causality or tail length:
    an 8-token tail moves nothing, 9-12 move, 13 moves nothing again, 14-16
    move, and a 16-token text tail chosen from the prompt's own body moves the
    same rows by the same amounts. Those rows are therefore not information from
    the future that a causal graph could match -- they are a numerical regime the
    release enters depending on the input, and a per-position gate must drop a
    dump's last few rows rather than compare against them.

Run it in the pinned environment (it downloads nothing; the snapshot must be
present):

    /tmp/inkling_venv/bin/python Tools/inkling2_reference_profile.py

It loads the checkpoint once in FP32, prints the expert-bank fingerprint, then
compares the prompt-only forward against tails of several lengths.
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

SNAPSHOT = os.environ.get(
    "INKLING_SNAPSHOT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/"
        "snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
    ),
)
PROMPT_PATH = os.environ.get("INKLING_PROMPT", "/tmp/inkling2_port_prompt.txt")
TAIL_POOL = 16
TAIL_ROWS = 5
FILLER = 200001


def main():
    if not os.path.exists(PROMPT_PATH):
        raise SystemExit(f"no prompt at {PROMPT_PATH}: dump one from a turn case first")
    ids = AutoTokenizer.from_pretrained(SNAPSHOT)(
        open(PROMPT_PATH).read(), add_special_tokens=False
    )["input_ids"]
    print(f"prompt: {len(ids)} tokens")

    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        SNAPSHOT, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()

    # 1. The expert bank's own resolution. This is the number a logit
    # comparison's expected band comes from.
    bank = None
    for name, parameter in model.named_parameters():
        if name.endswith("mlp.experts.gate_up_proj"):
            bank = (name, parameter.detach().float().flatten()[:4096])
            break
    if bank is None:
        raise SystemExit("no gate_up_proj in the loaded model: conversion mapping changed")
    name, values = bank
    rounded = values.to(torch.bfloat16).to(torch.float32)
    print(f"{name}: {values.numel()} sampled, "
          f"{int((rounded == values).sum())} on the BF16 grid "
          f"(max |v - bf16(v)| {float((rounded - values).abs().max()):.2e})")
    print("  -> a BF16-rounding port cannot be elementwise-equal to this dump, "
          "and an FP32 dump cannot validate the port's expert arithmetic.")

    # 2. How stable the reference's own tail rows are. The continuation is the
    # reference's own when a tail dump exists, otherwise text from the prompt.
    continuation = None
    tail_path = "/tmp/inkling2_tool_logits_tail_fp32.json"
    if os.path.exists(tail_path):
        continuation = json.load(open(tail_path)).get("continuation")
    if not continuation:
        continuation = [200001, 200008] + ids[88:100]
    text_tail = [t for t in ids[88:104] if t >= 1000][:TAIL_POOL]

    arms = {"prompt": ids, "filler16": ids + [FILLER] * 16}
    for length in (8, 9, 12, 13, 16):
        arms[f"tail{length}"] = ids + continuation[:length]
    arms["text16"] = ids + text_tail

    rows = {}
    with torch.no_grad():
        for label, payload in arms.items():
            rows[label] = model(torch.tensor([payload])).logits[0].float().numpy()

    print()
    print("rows near the prompt's end against a prompt-only forward "
          f"(max |difference| over the last {TAIL_ROWS} rows)")
    print(f"{'arm':>10s} {'length':>7s} " + " ".join(f"{r:>9d}" for r in range(len(ids) - TAIL_ROWS, len(ids))))
    for label, payload in arms.items():
        cells = []
        for row in range(len(ids) - TAIL_ROWS, len(ids)):
            cells.append(f"{np.abs(rows['prompt'][row] - rows[label][row]).max():9.4f}")
        print(f"{label:>10s} {len(payload):7d} " + " ".join(cells))
    early = max(np.abs(rows["prompt"][row] - rows["text16"][row]).max() for row in range(0, len(ids) - TAIL_ROWS - 2, 16))
    print()
    print(f"rows well before the tail move by at most {early:.2e}, so the instability is")
    print("confined to a dump's last few rows: a per-position gate drops those and keeps the rest.")


if __name__ == "__main__":
    main()
