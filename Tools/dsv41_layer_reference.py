"""One-layer NumPy reference for DeepSeek V4.1, built from the published shards.

Why this exists: the port's own reference (`DeepSeekV41CPUForward`) is a
transcription of the checkpoint's `inference/model.py`, so a rule both of them
share cannot be caught by comparing them. This computes layer 0 from the real
tensors, in the checkpoint's own arithmetic, so the device path can be compared
against something it did not write.

The layer is the published one: `hc_mult` residual streams with Sinkhorn mixing,
latent attention over a single sliding-window row with the per-head sink, grouped
output projection, `sqrtsoftplus` routing with the selection bias, six MXFP4
routed experts, one shared E4M3 expert, and the `ParallelHead`.

At position 0 the RoPE is the identity (cos = 1, sin = 0), which is why a
one-position reference needs no YaRN ramp; that is verified separately.

    python3 Tools/dsv41_layer_reference.py [--token 47493] [--dump /tmp/ref]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP8_BLOCK = 32
FP4_BLOCK = 32


def e4m3(raw: np.ndarray) -> np.ndarray:
    b = raw.astype(np.uint8)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    exp = (b >> 3) & 0x0F
    mant = (b & 0x07).astype(np.float32)
    normal = (1.0 + mant / 8.0) * np.exp2(exp.astype(np.float32) - 7.0)
    subnormal = (mant / 8.0) * np.float32(2.0 ** -6)
    return sign * np.where(exp == 0, subnormal, normal)


def e8m0(raw: np.ndarray) -> np.ndarray:
    b = raw.astype(np.uint8)
    return np.where(b == 0, 0.0, np.exp2(b.astype(np.float32) - 127.0)).astype(np.float32)


def bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def e2m1(raw: np.ndarray, shape) -> np.ndarray:
    b = raw.astype(np.uint8)

    def unpack(nibble: np.ndarray) -> np.ndarray:
        magnitude = E2M1[nibble & 0x07]
        return np.where(nibble & 0x08, -magnitude, magnitude).astype(np.float32)

    interleaved = np.stack([unpack(b & 0x0F), unpack((b >> 4) & 0x0F)], axis=-1).reshape(-1)
    return interleaved.reshape(shape)


class Checkpoint:
    def __init__(self, snapshot: str):
        self.snapshot = snapshot
        with open(os.path.join(snapshot, "model.safetensors.index.json")) as handle:
            self.index = json.load(handle)["weight_map"]
        self._headers = {}

    def header(self, shard: str):
        if shard not in self._headers:
            with open(os.path.join(self.snapshot, shard), "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                self._headers[shard] = json.loads(handle.read(size))
        return self._headers[shard]

    def raw(self, name: str) -> np.ndarray:
        shard = self.index[name]
        header = self.header(shard)
        start, end = header[name]["data_offsets"]
        with open(os.path.join(self.snapshot, shard), "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(8 + size + start)
            return np.frombuffer(handle.read(end - start), dtype=np.uint8)

    def shape(self, name: str):
        return self.header(self.index[name])[name]["shape"]

    def bf16(self, name: str) -> np.ndarray:
        return bf16(self.raw(name)).reshape(self.shape(name))

    def fp8(self, name: str) -> np.ndarray:
        weights = e4m3(self.raw(name)).reshape(self.shape(name))
        scales = e8m0(self.raw(name.replace(".weight", ".scale"))).reshape(self.shape(name.replace(".weight", ".scale")))
        rows, cols = weights.shape
        expanded = np.repeat(np.repeat(scales, FP8_BLOCK, axis=0), FP8_BLOCK, axis=1)
        return (weights * expanded[:rows, :cols]).astype(np.float32)

    def fp4(self, name: str) -> np.ndarray:
        shape = self.shape(name)
        weights = e2m1(self.raw(name), (shape[0], shape[1] * 2))
        scale_name = name.replace(".weight", ".scale")
        scales = e8m0(self.raw(scale_name)).reshape(self.shape(scale_name))
        expanded = np.repeat(scales, FP4_BLOCK, axis=1)
        return (weights * expanded).astype(np.float32)


def rmsnorm(x: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    return (x * (1.0 / np.sqrt((x.astype(np.float64) ** 2).mean() + eps))).astype(np.float32) * gain


def sinkhorn(mixes, hc_scale, hc_base, mult, iters, eps):
    pre = 1.0 / (1.0 + np.exp(-(mixes[:mult] * hc_scale[0] + hc_base[:mult]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mixes[mult:2 * mult] * hc_scale[1] + hc_base[mult:2 * mult])))
    comb = (mixes[2 * mult:].reshape(mult, mult) * hc_scale[2]
            + hc_base[2 * mult:].reshape(mult, mult)).astype(np.float64)
    comb = np.exp(comb - comb.max(axis=1, keepdims=True))
    comb = comb / comb.sum(axis=1, keepdims=True) + eps
    comb = comb / (comb.sum(axis=0, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=0, keepdims=True) + eps)
    return pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--token", type=int, default=47493)
    parser.add_argument("--dump", default="/tmp/ref")
    args = parser.parse_args()

    ckpt = Checkpoint(args.snapshot)
    config = json.load(open(os.path.join(args.snapshot, "config.json")))
    text = config.get("text_config", config)
    dim = text["hidden_size"]
    heads = text["num_attention_heads"]
    head_dim = text["head_dim"]
    rope_dim = text["qk_rope_head_dim"]
    groups = text["o_groups"]
    o_lora = text["o_lora_rank"]
    inter = text["moe_intermediate_size"]
    experts = text["n_routed_experts"]
    topk = text["num_experts_per_tok"]
    mult = text["hc_mult"]
    eps = text["rms_norm_eps"]
    hc_eps = text["hc_eps"]
    limit = text["swiglu_limit"]
    scale = text["routed_scaling_factor"]

    embed = ckpt.bf16("embed.weight")[args.token].astype(np.float32)
    stack = np.tile(embed, (mult, 1))
    print("embed norm %.4f" % np.linalg.norm(embed))

    # --- hyper-connection coefficients from the block input ---
    def hc(which):
        fn = ckpt.raw("layers.0.hc_%s_fn" % which).copy().view(np.float32).reshape(24, mult * dim)
        base = ckpt.raw("layers.0.hc_%s_base" % which).copy().view(np.float32)
        hc_scale = ckpt.raw("layers.0.hc_%s_scale" % which).copy().view(np.float32)
        flat = stack.reshape(-1).astype(np.float64)
        rstd = 1.0 / np.sqrt((flat ** 2).mean() + eps)
        mixes = fn.astype(np.float64) @ flat * rstd
        print("%s mixes" % which, np.round(mixes, 4))
        return sinkhorn(mixes, hc_scale, base, mult, text["hc_sinkhorn_iters"], hc_eps)

    attn_pre, attn_post, attn_comb = hc("attn")
    print("attn pre", np.round(attn_pre, 4), "post", np.round(attn_post, 4))

    residual = stack
    x = (np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[:, None] * stack).sum(axis=0)

    def attention(x):
        normed = rmsnorm(x, ckpt.bf16("layers.0.attn_norm.weight"), eps)
        qr = ckpt.fp8("layers.0.attn.wq_a.weight") @ normed
        qr = rmsnorm(qr, ckpt.bf16("layers.0.attn.q_norm.weight"), eps)
        q = (ckpt.fp8("layers.0.attn.wq_b.weight") @ qr).reshape(heads, head_dim)
        kv = ckpt.fp8("layers.0.attn.wkv.weight") @ normed
        kv = rmsnorm(kv, ckpt.bf16("layers.0.attn.kv_norm.weight"), eps)
        kv = kv[-(head_dim):]  # window K is the whole latent
        sink = ckpt.raw("layers.0.attn.attn_sink").copy().view(np.float32)
        scores = (q @ kv) * (head_dim ** -0.5)
        # softmax over {score, sink}: the sink has no value row
        weights = 1.0 / (1.0 + np.exp(sink - scores))
        o = weights[:, None] * kv[None, :]
        # inverse rope at position 0 is the identity
        o = o.reshape(groups, -1)
        wo_a = ckpt.fp8("layers.0.attn.wo_a.weight").reshape(groups, o_lora, -1)
        y = np.einsum("gd,grd->gr", o, wo_a)
        return ckpt.fp8("layers.0.attn.wo_b.weight") @ y.reshape(-1), o, kv, q, weights

    y, attended, kv, q, p = attention(x)
    print("q norm %.4f  kv norm %.4f" % (np.linalg.norm(q), np.linalg.norm(kv)))
    print("attended norm %.4f  p[:8] %s" % (np.linalg.norm(attended), np.round(p[:8], 4)))
    print("q[:6]", np.round(q[0][:6], 4), " kv[:6]", np.round(kv[:6], 4))
    print("y norm %.4f" % np.linalg.norm(y))
    np.asarray(attended, dtype=np.float32).tofile("/tmp/ref/attention_out.bin")
    np.asarray(q, dtype=np.float32).tofile("/tmp/ref/query.bin")
    np.asarray(kv, dtype=np.float32).tofile("/tmp/ref/latent.bin")

    def post(out, res, post_c, comb_c):
        return post_c[:, None] * out[None, :] + np.einsum("sk,kd->sd", comb_c, res)

    stack = post(y, residual, attn_post, attn_comb)
    print("stack after attention norm %.4f" % np.linalg.norm(stack))
    os.makedirs(args.dump, exist_ok=True)
    stack.astype(np.float32).tofile(os.path.join(args.dump, "after_attention.bin"))

    # --- feed forward ---
    ffn_pre, ffn_post, ffn_comb = hc("ffn")
    x = (attn_pre[:, None] * stack).sum(axis=0)
    normed = rmsnorm(x, ckpt.bf16("layers.0.ffn_norm.weight"), eps)

    gate = ckpt.bf16("layers.0.ffn.gate.weight")
    bias = ckpt.raw("layers.0.ffn.gate.bias").copy().view(np.float32)
    scores = np.sqrt(np.log1p(np.exp(gate.astype(np.float32) @ normed)))
    chosen = np.argsort(scores + bias)[-topk:][::-1]
    weights = scores[chosen]
    weights = weights / (weights.sum() + 1e-20) * scale
    print("experts", chosen.tolist())
    print("weights", np.round(weights, 4))

    def swiglu(w1, w3, w2, vec):
        g = w1 @ vec
        u = w3 @ vec
        u = np.clip(u, -limit, limit)
        g = np.minimum(g, limit)
        return w2 @ (g / (1.0 + np.exp(-g)) * u)

    routed = np.zeros(dim, dtype=np.float32)
    for expert, weight in zip(chosen, weights):
        routed += weight * swiglu(
            ckpt.fp4("layers.0.ffn.experts.%d.w1.weight" % expert),
            ckpt.fp4("layers.0.ffn.experts.%d.w3.weight" % expert),
            ckpt.fp4("layers.0.ffn.experts.%d.w2.weight" % expert),
            normed,
        )
    shared = swiglu(
        ckpt.fp8("layers.0.ffn.shared_experts.w1.weight"),
        ckpt.fp8("layers.0.ffn.shared_experts.w3.weight"),
        ckpt.fp8("layers.0.ffn.shared_experts.w2.weight"),
        normed,
    )
    y = routed + shared
    print("ffn out norm %.4f (routed %.4f shared %.4f)" % (
        np.linalg.norm(y), np.linalg.norm(routed), np.linalg.norm(shared)))
    stack = post(y, stack, ffn_post, ffn_comb)

    hidden = (ffn_pre[:, None] * stack).sum(axis=0)
    final = rmsnorm(hidden, ckpt.bf16("norm.weight"), eps)
    head = ckpt.bf16("head.weight")
    logits = (head.astype(np.float32) @ final).astype(np.float32)
    top = np.argsort(-logits)[:5]
    print("hidden norm %.3f" % np.linalg.norm(hidden))
    print("logits top5", [(int(i), round(float(logits[i]), 3)) for i in top])
    stack.astype(np.float32).tofile(os.path.join(args.dump, "final_stack.bin"))
    np.save(os.path.join(args.dump, "logits.npy"), logits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
