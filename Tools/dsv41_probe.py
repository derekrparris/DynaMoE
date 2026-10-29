#!/usr/bin/env python3
"""DeepSeek-V4.1 checkpoint probes.

Three questions the port cannot answer without the real download, answered from
the files themselves instead of from guesswork:

  tensors   What tensor names, dtypes and shapes does this checkpoint actually
            ship? Emits a name-template histogram so the Swift catalog can be
            checked for coverage, and a JSON table for reference.

  engram    What does the checkpoint's own `inference/engram.py` hash, and how
            are its tables laid out? Dumps the module's public surface with
            signatures plus the Engram tensor shapes. It does NOT run the
            module, so it cannot verify a hash output by itself: the bucket
            indices it prints as next steps are what a later run has to
            reproduce before `conformance` moves off `.unverified`.

  encoding  What do the shipped prompt-encoding goldens look like? Dumps
            `encoding/tests/test_output_*.txt` with every non-ASCII codepoint
            escaped, so prompt markers can be written as `\\u{...}` literals
            instead of retyped glyphs (AGENTS.md: "Reading files safely").

Usage:
    python3 Tools/dsv41_probe.py tensors  ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<sha>
    python3 Tools/dsv41_probe.py engram   <snapshot>
    python3 Tools/dsv41_probe.py encoding <snapshot>

Standard library only: runs on a machine that has the weights but no Python
environment set up for the model itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Tensor table
# ---------------------------------------------------------------------------


def read_safetensors_header(path: str) -> dict:
    """Returns the parsed JSON header of a .safetensors file (tensor metadata
    only; the data section is never read, so this works on 10 GB shards)."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (truncated length prefix)")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"{path}: implausible header length {header_len}")
        header = handle.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path}: truncated header ({len(header)} of {header_len} bytes)")
    return json.loads(header.decode("utf-8"))


def snapshot_index(snapshot: str) -> dict:
    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise SystemExit(f"no model.safetensors.index.json in {snapshot}")
    with open(index_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# The role rules the Swift catalog implements, mirrored here so the probe can
# report coverage. Kept in one place: if the catalog gains a role, add it here
# too, and the probe's "unclassified" list stays meaningful.
LAYER_ROLES = {
    "attn_norm": "attnNorm",
    "ffn_norm": "ffnNorm",
    "attn.wq_a": "attnWqA",
    "attn.wq_a_norm": "attnWqANorm",
    "attn.wq_b": "attnWqB",
    "attn.wkv": "attnWkv",
    "attn.kv_norm": "attnKvNorm",
    "attn.wkv_norm": "attnKvNorm",
    "attn.wo_a": "attnWoA",
    "attn.wo_b": "attnWoB",
    "attn.attn_sink": "attnSink",
    "attn.indexer.wq_b": "indexerWqB",
    "attn.indexer.wk": "indexerWk",
    "attn.indexer.weights_proj": "indexerWeights",
    "attn.compressor.wkv": "compressorWkv",
    "attn.compressor.wgate": "compressorWgate",
    "ffn.gate": "routerWeight",
    "ffn.gate.bias": "routerBias",
    "ffn.gate.bias_vl": "routerVisionBias",
    "ffn.shared_experts.w1": "sharedExpertW1",
    "ffn.shared_experts.w2": "sharedExpertW2",
    "ffn.shared_experts.w3": "sharedExpertW3",
    "hc_attn_fn": "hcAttnFn",
    "hc_attn_base": "hcAttnBase",
    "hc_attn_scale": "hcAttnScale",
    "hc_ffn_fn": "hcFfnFn",
    "hc_ffn_base": "hcFfnBase",
    "hc_ffn_scale": "hcFfnScale",
    "engram.wkv": "engramTable",
    "engram.q_weight": "engramQuery",
    "engram.k_weight": "engramKey",
}
DRAFT_ROLES = {
    "main_proj": "draftMainProj",
    "main_norm": "draftMainNorm",
    "norm": "draftNorm",
    "markov_head.embed": "draftMarkovEmbed",
    "markov_head.head": "draftMarkovHead",
    "confidence_head.proj": "draftConfidence",
}
MODEL_ROLES = {"embed.weight": "tokenEmbedding", "norm.weight": "finalNorm", "head.weight": "lmHead"}
SCALE_SUFFIXES = (".weight_scale", ".scales", ".scale")


def split_scale(body: str) -> tuple[str, bool]:
    for suffix in SCALE_SUFFIXES:
        if body.endswith(suffix):
            return body[: -len(suffix)], True
    if body.endswith(".weight"):
        return body[: -len(".weight")], False
    return body, False


def classify(name: str) -> str | None:
    if name in MODEL_ROLES:
        return MODEL_ROLES[name]
    if name.startswith("mtp."):
        rest = name[len("mtp."):]
        block, _, body = rest.partition(".")
        if not block.isdigit():
            return None
        base, _ = split_scale(body)
        if base.startswith("ffn.experts."):
            return "draftExpert*"
        return DRAFT_ROLES.get(base)
    if not name.startswith("layers."):
        return None
    rest = name[len("layers."):]
    layer, _, body = rest.partition(".")
    if not layer.isdigit():
        return None
    base, _ = split_scale(body)
    if base.startswith("ffn.experts."):
        return "expertW*"
    return LAYER_ROLES.get(base)


def template(name: str) -> str:
    """Collapses layer and expert indices so names group by shape."""
    name = re.sub(r"^mtp\.\d+\.", "mtp.{k}.", name)
    name = re.sub(r"^layers\.\d+\.ffn\.experts\.\d+\.", "layers.{L}.ffn.experts.{E}.", name)
    return re.sub(r"^layers\.\d+\.", "layers.{L}.", name)


def cmd_tensors(args) -> int:
    index = snapshot_index(args.snapshot)
    weight_map = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    metadata = index.get("metadata") or {}
    total_size = metadata.get("total_size")

    print(f"snapshot : {args.snapshot}")
    print(f"shards   : {len(shards)}")
    if total_size:
        print(f"total    : {total_size / 1024 ** 3:.2f} GiB (metadata.total_size)")

    missing = [s for s in shards if not os.path.exists(os.path.join(args.snapshot, s))]
    present = [s for s in shards if s not in missing]
    present_bytes = sum(os.path.getsize(os.path.join(args.snapshot, s)) for s in present)
    print(f"present  : {len(present)}/{len(shards)} shards, {present_bytes / 1024 ** 3:.2f} GiB on disk")
    if missing:
        print(f"MISSING  : {len(missing)} shard(s), e.g. {', '.join(missing[:4])}")

    tensors: dict[str, dict] = {}
    for shard in present:
        path = os.path.join(args.snapshot, shard)
        try:
            header = read_safetensors_header(path)
        except (OSError, ValueError) as error:
            print(f"WARN: {shard}: {error}", file=sys.stderr)
            continue
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            tensors[name] = {
                "shard": shard,
                "dtype": entry.get("dtype"),
                "shape": entry.get("shape"),
                "bytes": (entry.get("data_offsets") or [0, 0])[1]
                - (entry.get("data_offsets") or [0, 0])[0],
            }

    if not tensors:
        print("no tensors read; is the download complete?", file=sys.stderr)
        return 1

    dtypes = Counter(t["dtype"] for t in tensors.values())
    print(f"\ntensors  : {len(tensors)}")
    print("dtypes   : " + ", ".join(f"{k}={v}" for k, v in dtypes.most_common()))

    templates = Counter(template(n) for n in tensors)
    print(f"\nname templates ({len(templates)}):")
    for name in sorted(templates):
        print(f"  {templates[name]:6d}  {name}")

    unclassified = sorted(n for n in tensors if classify(n) is None)
    print(f"\nunclassified by the Swift catalog: {len(unclassified)}")
    for name in unclassified[:40]:
        entry = tensors[name]
        print(f"  {name}  {entry['dtype']} {entry['shape']}")
    if len(unclassified) > 40:
        print(f"  (+{len(unclassified) - 40} more)")

    # Engram tables are the ones that must never be mapped.
    engram = {n: t for n, t in tensors.items() if ".engram." in n}
    if engram:
        print(f"\nEngram tensors ({len(engram)}):")
        for name in sorted(engram):
            entry = engram[name]
            print(f"  {name}  {entry['dtype']} {entry['shape']} {entry['bytes'] / 1024 ** 3:.2f} GiB")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"shards": shards, "missing": missing, "tensors": tensors}, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


# ---------------------------------------------------------------------------
# Engram
# ---------------------------------------------------------------------------


def cmd_engram(args) -> int:
    inference = os.path.join(args.snapshot, "inference")
    engram_py = os.path.join(inference, "engram.py")
    config_paths = [
        os.path.join(inference, "config.json"),
        os.path.join(args.snapshot, "config.json"),
    ]
    for path in config_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            text = config.get("text_config", config)
            print(f"config: {path}")
            for key in (
                "engram_layer_ids",
                "engram_num_embeddings",
                "engram_max_ngram_size",
                "engram_vocab_size",
                "engram_n_heads",
                "engram_head_dim",
                "engram_pad_token_id",
                "engram_pad_id",
                "engram_compressed_vocab_size",
            ):
                if key in text:
                    print(f"  {key} = {text[key]}")
            break

    if not os.path.exists(engram_py):
        print(f"\nno {engram_py}: the checkpoint does not ship its reference implementation")
        return 1

    source = open(engram_py, "r", encoding="utf-8").read()
    print(f"\n{engram_py}: {len(source.splitlines())} lines")
    print("top-level definitions:")
    for line_number, line in enumerate(source.splitlines(), start=1):
        match = re.match(r"^(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if match:
            print(f"  {line_number:5d}  {match.group(1)} {match.group(2)}")

    # The hash itself: show the definition and the surrounding arithmetic with
    # codepoints escaped where needed, so it can be transcribed exactly.
    for keyword in ("def hash", "def maybe_hash", "HASH", "PRIME", "prime", "token_map", "compress"):
        hits = [i + 1 for i, line in enumerate(source.splitlines()) if keyword in line]
        if hits:
            print(f"\nlines mentioning {keyword!r}: {hits[:20]}")
    print(
        "\nNext step: run this module against the checkpoint's own `inference/config.json`\n"
        "with a fixed token sequence, and record the 24 bucket indices per Engram layer.\n"
        "Those integers are the reference vectors that let DeepSeekV41EngramHasher.seed\n"
        "be corrected and `conformance` flipped to .verifiedAgainstReference."
    )
    return 0


# ---------------------------------------------------------------------------
# Prompt encoding goldens
# ---------------------------------------------------------------------------


def escape(text: str) -> str:
    """Escapes text into the body of a single Swift string literal.

    Newlines, tabs and carriage returns become their short escapes and every other
    non-ASCII codepoint becomes a unicode escape, so the emitted literal compiles
    and a reader never has to trust a rendered glyph.
    """
    short = {"\n": "\\n", "\t": "\\t", "\r": "\\r", '"': '\\"', "\\": "\\\\"}
    out = []
    for char in text:
        if char in short:
            out.append(short[char])
            continue
        code = ord(char)
        if 0x20 <= code < 0x7F:
            out.append(char)
        elif code <= 0xFFFF:
            out.append(f"\\u{{{code:04X}}}")
        else:
            out.append(f"\\u{{{code:X}}}")
    return "".join(out)


def cmd_encoding(args) -> int:
    tests = os.path.join(args.snapshot, "encoding", "tests")
    if not os.path.isdir(tests):
        print(f"no {tests}: the checkpoint ships no encoding goldens")
        return 1
    names = sorted(os.listdir(tests))
    outputs = [n for n in names if n.startswith("test_output")]
    inputs = [n for n in names if n.startswith("test_input")]
    print(f"{tests}: {len(inputs)} input(s), {len(outputs)} expected output(s)")
    for name in outputs:
        path = os.path.join(tests, name)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        print(f"\n=== {name} ({len(text)} chars) ===")
        print(f'Swift literal: "{escape(text)}"')
        print("byte preview:")
        print(repr(text)[:2000])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("tensors", cmd_tensors), ("engram", cmd_engram), ("encoding", cmd_encoding)):
        child = sub.add_parser(name)
        child.add_argument("snapshot", help="path to the HF snapshot directory")
        if name == "tensors":
            child.add_argument("--json", help="write the full tensor table here")
        child.set_defaults(handler=handler)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
