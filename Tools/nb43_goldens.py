#!/usr/bin/env python3
"""Reference Llama forward for the Nanbeige 4.3 checkpoint, in float32 numpy.

This exists because "the engine printed gibberish" is not a diagnosis: the same
prompt has to be scored from the checkpoint's own bytes by an implementation
that was not written by the engine, so the two answers can be compared. It is
the oracle the engine's greedy continuation is checked against.

It streams one layer at a time out of the safetensors shards (never holding the
66 GB), applies textbook Llama arithmetic (1-mean RMSNorm, rotate_half RoPE over
the full head dimension, GQA with headsPerKv = numHeads / numKvHeads, SiLU MLP),
and prints the next-token ranking for the last position.

    python3 Tools/nb43_goldens.py --prompt-ids 166100,2714,13,764

Nothing here is used by the app; it is a measurement tool.
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Nanbeige--Nanbeige4.3-33B/"
    "snapshots/df919e1e2cf077a22204c78b6c826bf3db18bc08"
)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """BF16 is the top half of an FP32: widening is a 16-bit left shift."""
    bits = raw.astype(np.uint32) << 16
    return bits.view(np.float32)


class Shards:
    """Byte-range reads out of the shard files, plus the header index."""

    def __init__(self, snapshot: str):
        with open(os.path.join(snapshot, "model.safetensors.index.json")) as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self.snapshot = snapshot
        self.headers = {}
        self.header_sizes = {}
        self.handles = {}

    def _header(self, filename):
        if filename not in self.headers:
            with open(os.path.join(self.snapshot, filename), "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.header_sizes[filename] = length
                self.headers[filename] = json.loads(handle.read(length))
        return self.headers[filename]

    def _handle(self, filename):
        if filename not in self.handles:
            self.handles[filename] = open(os.path.join(self.snapshot, filename), "rb")
        return self.handles[filename]

    def tensor(self, name: str) -> np.ndarray:
        filename = self.weight_map[name]
        entry = self._header(filename)[name]
        assert entry["dtype"] == "BF16", f"{name} is {entry['dtype']}, not BF16"
        start, end = entry["data_offsets"]
        handle = self._handle(filename)
        handle.seek(8 + self.header_sizes[filename] + start)
        raw = np.frombuffer(handle.read(end - start), dtype=np.uint16)
        assert raw.size * raw.itemsize == end - start, f"short read for {name}"
        return bf16_to_f32(raw).reshape(entry["shape"])


def rms_norm(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope_tables(head_dim, positions, theta):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.outer(positions.astype(np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--prompt-ids", required=True, help="comma separated token ids")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--layers", type=int, default=0, help="0 = every layer")
    parser.add_argument("--probe", default="", help="comma separated ids to print logits for")
    parser.add_argument("--greedy", type=int, default=0,
                        help="also generate this many tokens greedily, re-running the "
                             "forward per token (each pass re-reads the checkpoint)")
    args = parser.parse_args()

    with open(os.path.join(args.snapshot, "config.json")) as handle:
        config = json.load(handle)
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"] if args.layers <= 0 else args.layers
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", hidden // heads)
    intermediate = config["intermediate_size"]
    eps = config["rms_norm_eps"]
    theta = float(config.get("rope_theta", config.get("rotary_base", 10000.0)))
    vocab = config["vocab_size"]
    heads_per_kv = heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    ids = [int(part) for part in args.prompt_ids.split(",")]
    tokens = len(ids)
    print(f"prompt {tokens} tokens, {layers} layers, head_dim {head_dim}, "
          f"heads {heads}/{kv_heads} (x{heads_per_kv}), theta {theta:.0f}")

    shards = Shards(args.snapshot)
    cos, sin = rope_tables(head_dim, np.arange(tokens), theta)
    cos_q, sin_q = cos[:, None, :], sin[:, None, :]

    x = shards.tensor("model.embed_tokens.weight")[ids].astype(np.float32)
    print(f"embedding rows: |x| rms {np.sqrt(np.mean(x * x)):.4f}")

    causal = np.triu(np.ones((tokens, tokens), dtype=bool), k=1)
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        norm1 = shards.tensor(prefix + "input_layernorm.weight")
        norm2 = shards.tensor(prefix + "post_attention_layernorm.weight")

        normed = rms_norm(x, norm1, eps)
        q = (normed @ shards.tensor(prefix + "self_attn.q_proj.weight").T).reshape(tokens, heads, head_dim)
        k = (normed @ shards.tensor(prefix + "self_attn.k_proj.weight").T).reshape(tokens, kv_heads, head_dim)
        v = (normed @ shards.tensor(prefix + "self_attn.v_proj.weight").T).reshape(tokens, kv_heads, head_dim)

        q = (q * cos_q + rotate_half(q) * sin_q).astype(np.float32)
        k = (k * cos_q + rotate_half(k) * sin_q).astype(np.float32)

        k = np.repeat(k, heads_per_kv, axis=1)
        v = np.repeat(v, heads_per_kv, axis=1)
        scores = np.einsum("thd,shd->hts", q, k) * scale
        scores = np.where(causal, -np.inf, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        context = np.einsum("hts,shd->thd", weights, v).reshape(tokens, heads * head_dim)
        attn = context @ shards.tensor(prefix + "self_attn.o_proj.weight").T
        x = x + attn.astype(np.float32)

        normed = rms_norm(x, norm2, eps)
        gate = normed @ shards.tensor(prefix + "mlp.gate_proj.weight").T
        up = normed @ shards.tensor(prefix + "mlp.up_proj.weight").T
        act = (gate / (1.0 + np.exp(-gate))) * up
        down = act @ shards.tensor(prefix + "mlp.down_proj.weight").T
        x = x + down.astype(np.float32)
        if layer % 13 == 0 or layer == layers - 1:
            print(f"  layer {layer:2d}: |x| rms {np.sqrt(np.mean(x * x)):.4f}", flush=True)

    final = rms_norm(x, shards.tensor("model.norm.weight"), eps)
    logits = final[-1] @ shards.tensor("lm_head.weight").T
    order = np.argsort(-logits)[: args.top]
    print("next-token ranking:")
    for rank, index in enumerate(order):
        print(f"  {rank:2d}  id {int(index):6d}  logit {float(logits[index]):9.4f}")
    print(f"vocab {vocab}, logits finite: {bool(np.isfinite(logits).all())}")
    if args.probe:
        for part in args.probe.split(","):
            index = int(part)
            print(f"  probe id {index}: logit {float(logits[index]):9.4f}")

    if args.greedy > 0:
        drawn = [int(order[0])]
        consumed = list(ids)
        for step in range(args.greedy - 1):
            consumed = consumed + [drawn[-1]]
            tokens_now = len(consumed)
            cos_n, sin_n = rope_tables(head_dim, np.arange(tokens_now), theta)
            cos_q, sin_q = cos_n[:, None, :], sin_n[:, None, :]
            state = shards.tensor("model.embed_tokens.weight")[consumed].astype(np.float32)
            causal = np.triu(np.ones((tokens_now, tokens_now), dtype=bool), k=1)
            for layer in range(layers):
                prefix = f"model.layers.{layer}."
                normed = rms_norm(state, shards.tensor(prefix + "input_layernorm.weight"), eps)
                q = (normed @ shards.tensor(prefix + "self_attn.q_proj.weight").T).reshape(tokens_now, heads, head_dim)
                k = (normed @ shards.tensor(prefix + "self_attn.k_proj.weight").T).reshape(tokens_now, kv_heads, head_dim)
                v = (normed @ shards.tensor(prefix + "self_attn.v_proj.weight").T).reshape(tokens_now, kv_heads, head_dim)
                q = (q * cos_q + rotate_half(q) * sin_q).astype(np.float32)
                k = (k * cos_q + rotate_half(k) * sin_q).astype(np.float32)
                k = np.repeat(k, heads_per_kv, axis=1)
                v = np.repeat(v, heads_per_kv, axis=1)
                scores = np.einsum("thd,shd->hts", q, k) * scale
                scores = np.where(causal, -np.inf, scores)
                scores = scores - scores.max(axis=-1, keepdims=True)
                weights = np.exp(scores)
                weights = weights / weights.sum(axis=-1, keepdims=True)
                context = np.einsum("hts,shd->thd", weights, v).reshape(tokens_now, heads * head_dim)
                state = state + (context @ shards.tensor(prefix + "self_attn.o_proj.weight").T).astype(np.float32)
                normed = rms_norm(state, shards.tensor(prefix + "post_attention_layernorm.weight"), eps)
                gate = normed @ shards.tensor(prefix + "mlp.gate_proj.weight").T
                up = normed @ shards.tensor(prefix + "mlp.up_proj.weight").T
                state = state + ((gate / (1.0 + np.exp(-gate))) * up @ shards.tensor(prefix + "mlp.down_proj.weight").T).astype(np.float32)
            final = rms_norm(state, shards.tensor("model.norm.weight"), eps)
            step_logits = final[-1] @ shards.tensor("lm_head.weight").T
            drawn.append(int(np.argmax(step_logits)))
            print(f"  greedy step {step + 1}: {drawn[-1]}", flush=True)
        print(f"greedy continuation: {drawn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
