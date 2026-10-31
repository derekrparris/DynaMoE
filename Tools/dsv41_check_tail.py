"""Checks the last three stages of a DeepSeek V4.1 forward against the dumps a
published-checkpoint run leaves in /tmp/dsv41_*.bin.

Run `DeepSeekV41PublishedTurnTests.testTwoDifferentTokensProduceDifferentLogits`
with `/tmp/dynamoe_dsv41_real_turn` present, then this. Every stage is compared
against the checkpoint's own tensors, so a mismatch names the stage instead of
leaving "the model is wrong" as the diagnosis.
"""
import json
import os
import struct
import sys

import numpy as np


def main(snapshot: str) -> int:
    def tensor(shard: str, name: str):
        path = os.path.join(snapshot, shard)
        with open(path, "rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            start, end = header[name]["data_offsets"]
            handle.seek(8 + header_size + start)
            return header[name], handle.read(end - start)

    streams = np.fromfile("/tmp/dsv41_streams.bin", dtype=np.float32)
    mix = np.fromfile("/tmp/dsv41_mix.bin", dtype=np.float32)
    hidden = np.fromfile("/tmp/dsv41_hidden.bin", dtype=np.float32)
    normed = np.fromfile("/tmp/dsv41_normed.bin", dtype=np.float32)
    logits = np.fromfile("/tmp/dsv41_logits.bin", dtype=np.float32)

    copies = len(mix)
    width = len(hidden)
    stack = streams.reshape(copies, width)
    collapsed = (mix[:, None] * stack).sum(axis=0)
    print("mix", mix)
    print("collapse      max delta %.5f  (hidden norm %.3f)" % (
        np.abs(collapsed - hidden).max(), np.linalg.norm(hidden)))

    meta, raw = tensor("model-00043-of-00048.safetensors", "norm.weight")
    gain = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    normed_expected = hidden / np.sqrt((hidden.astype(np.float64) ** 2).mean() + 1e-20) * gain
    print("rmsnorm       max delta %.6f  (std %.4f vs %.4f)" % (
        np.abs(normed_expected - normed).max(), normed_expected.std(), normed.std()))

    meta, raw = tensor("model-00043-of-00048.safetensors", "head.weight")
    head = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    head = head.reshape(meta["shape"])
    logits_expected = head.astype(np.float64) @ normed.astype(np.float64)
    print("head          max delta %.4f" % np.abs(logits_expected - logits).max())
    for label, values in (("device", logits), ("numpy", logits_expected)):
        top = np.argsort(-values)[:5]
        print("  %-6s top5 %s" % (label, [(int(i), round(float(values[i]), 3)) for i in top]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
        "1bae0e1088993b9b82de08f59de5be5a34e42170")))
