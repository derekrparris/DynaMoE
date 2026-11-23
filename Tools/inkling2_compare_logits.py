#!/usr/bin/env python3
"""Per-position comparison of the Inkling 2 port against the reference.

This is the offline half of the logit-fidelity investigation: the port dumps its
logits from `testTheHostRankingMatchesTheReferencePositionByPosition`
(`/tmp/inkling2_tool_logits_mine.json`) and the reference dumps its own from
`/tmp/inkling_tool_logits_ref.py` -- twice, in BF16 (the checkpoint's shipped
dtype) and in FP32. The FP32 run is the yardstick, not the BF16 one: the BF16
reference disagrees with its own FP32 run by ~0.27 mean / 3.7 max, flipping its
argmax at one position and its top-4 set at two, so a port that merely differs
from BF16 has not been shown wrong.

    /tmp/inkling_venv/bin/python Tools/inkling2_compare_logits.py

Inputs (all optional except the port's dump; a missing one is skipped):
    /tmp/inkling2_tool_logits_mine.json    the port
    /tmp/inkling2_tool_logits_full.npy     reference, BF16
    /tmp/inkling2_tool_logits_full_fp32.npy reference, FP32

Reads as a diagnostic, never as a gate: a near-tie (a small margin between the
4th and 5th logit) is expected to reorder, and the plan's rule is that an
adequate margin must agree while a near tie gets an explanation, not a verdict.
"""
import json
import os

import numpy as np

MINE = "/tmp/inkling2_tool_logits_mine.json"
# The device path's dump, which is the one a turn actually runs and the one the
# full-range bisect wants: every prompt position, written raw so 130 rows cost
# no JSON. Preferred when present; the host JSON above is the fallback.
MINE_BIN = "/tmp/inkling2_port_device_rows.bin"
MINE_INDEX = "/tmp/inkling2_port_device_rows.json"
# FP32 first and BF16 second: the FP32 run is the yardstick, and the BF16 one is
# kept beside it only to show the reference's own rounding (they disagree with
# each other, so neither alone can judge the port).
REFERENCE = {
    "fp32": "/tmp/inkling2_tool_logits_full_fp32.npy",
    "bf16": "/tmp/inkling2_tool_logits_full.npy",
}


def load(path):
    """A dump as floats. A `null` in the port's JSON is a non-finite logit the
    writer could not encode, so it reads back as NaN rather than as zero."""
    if path.endswith(".json"):
        return np.array(json.load(open(path)), dtype=np.float64)
    return np.load(path).astype(np.float64)


def load_mine():
    if os.path.exists(MINE_BIN) and os.path.exists(MINE_INDEX):
        index = json.load(open(MINE_INDEX))
        rows = np.fromfile(MINE_BIN, dtype="<f4")
        rows = rows.reshape(index["positionCount"], index["rowWidth"])
        print(f"port: {MINE_BIN} ({rows.shape[0]} positions, device path)")
        return rows.astype(np.float64)
    return load(MINE)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def top4(row):
    return set(np.argsort(-row)[:4].tolist())


def bisect(mine, fp32, bf16=None):
    """Where the port leaves the band the reference itself occupies.

    The bar is not a constant: the BF16 reference's cosine against FP32 at the
    same position is how sure the reference is of itself there, so a position
    where the port is FURTHER from FP32 than BF16 is by a real margin is the
    first one worth inspecting. Ranges are printed too, because the profile is
    what distinguishes a drift that grows with position (accumulated rounding)
    from one that jumps at a specific depth (a structural bug).
    """
    count = mine.shape[0]
    cs = np.array([cosine(mine[p], fp32[p]) for p in range(count)])
    if bf16 is not None:
        span = min(count, len(bf16))
        bs = np.array([cosine(bf16[p], fp32[p]) for p in range(span)])
    else:
        bs = None
    print("cosine against FP32, by range (port / reference-bf16)")
    edges = list(range(0, count, 16)) + [count]
    for lo, hi in zip(edges, edges[1:]):
        segment = cs[lo:hi]
        line = f"  {lo:3d}-{hi - 1:3d}: min {segment.min():.6f}  mean {segment.mean():.6f}"
        reference = bs[lo:hi] if bs is not None else None
        if reference is not None and len(reference):
            line += f"   | bf16 min {reference.min():.6f}  mean {reference.mean():.6f}"
        print(line)
    if bs is not None:
        # A margin, not an absolute bar: near ties move both runs.
        span = len(bs)
        gap = bs - cs[:span]
        worst = int(np.argmax(gap))
        print()
        print(f"port is furthest below the reference at position {worst}: "
              f"cos {cs[worst]:.6f} against bf16's {bs[worst]:.6f} (gap {gap[worst]:.4f})")
        first = [p for p in range(span) if gap[p] > 0.01]
        print("positions where the port trails bf16 by more than 0.01:",
              first[:12] if first else "none")
    return cs


def main():
    import sys
    bisect_only = "--bisect" in sys.argv
    if not os.path.exists(MINE) and not os.path.exists(MINE_BIN):
        raise SystemExit(
            f"no port dump at {MINE} (or {MINE_BIN}): run a ranking case in "
            "Inkling2DeviceKernelTests first"
        )
    mine = load_mine()
    others = {name: load(path) for name, path in REFERENCE.items() if os.path.exists(path)}
    if not others:
        raise SystemExit("no reference dump: run /tmp/inkling_tool_logits_ref.py (BF16 and FP32)")
    # The reference covers the PROMPT only; a port dump from the device walk
    # also carries the generated tokens, which are compared by the tail script
    # and the test case's own table, not here. The length is the FP32 dump's,
    # NOT the shortest reference's: the BF16 run is a few positions shorter and
    # clamping to it threw away 114 of the 131 positions for no reason.
    primary = "fp32" if "fp32" in others else next(iter(others))
    count = min(mine.shape[0], len(others[primary]))
    mine = mine[:count]
    for name in list(others):
        if len(others[name]) < count:
            print(f"  ({name} reference covers {len(others[name])} of {count} positions)")
        others[name] = others[name][:count]

    print("per-position magnitude (mean / max absolute logit difference)")
    for name, reference in others.items():
        span = min(len(mine), len(reference))
        difference = np.abs(mine[:span] - reference[:span])
        print(f"  port vs {name:4s}: {difference.mean():.4f} / {difference.max():.3f}  (over {span} position(s))")
    if len(others) == 2:
        span = min(len(others["bf16"]), len(others["fp32"]))
        difference = np.abs(others["bf16"][:span] - others["fp32"][:span])
        print(f"  bf16 vs fp32: {difference.mean():.4f} / {difference.max():.3f}  (the reference against itself)")

    if bisect_only:
        if "fp32" in others:
            bisect(mine, others["fp32"], others.get("bf16"))
        return

    print()
    print("pos | argmax mine/bf16/fp32 | top4 mine==32 bf16==32 | cos mine-32 bf16-32 | margin 4th-5th")
    tally = {name: 0 for name in ("mine", *others)}
    for position in range(count):
        rows = {"mine": mine[position]}
        rows.update({name: r[position] for name, r in others.items() if position < len(r)})
        argmax = "/".join(str(int(np.argmax(rows[n]))) for n in ("mine", "bf16", "fp32") if n in rows)
        agreement = ""
        if "fp32" in rows:
            for name in ("mine", "bf16"):
                if name in rows:
                    same = top4(rows[name]) == top4(rows["fp32"])
                    tally[name] += same
                    agreement += f" {str(same):5s}"
            order = np.argsort(-rows["fp32"])
            margin = f"{rows['fp32'][order[3]] - rows['fp32'][order[4]]:7.4f}"
            spread = f" {cosine(rows['mine'], rows['fp32']):.6f} {cosine(rows['bf16'], rows['fp32']):.6f}"
        else:
            margin, spread = "      -", ""
        print(f"{position:3d} | {argmax:>20s} |{agreement} |{spread} | {margin}")

    if bisect_only and "fp32" in others:
        bisect(mine, others["fp32"], others.get("bf16"))
        return

    if "fp32" in others:
        print()
        argmax_mine = sum(
            np.argmax(mine[p]) == np.argmax(others["fp32"][p]) for p in range(count)
        )
        argmax_bf16 = sum(
            np.argmax(others["bf16"][p]) == np.argmax(others["fp32"][p])
            for p in range(min(count, len(others["bf16"])))
        ) if "bf16" in others else None
        print(f"argmax agrees with FP32: port {argmax_mine}/{count}"
              + (f", bf16 reference {argmax_bf16}/{count}" if argmax_bf16 is not None else ""))
        print(f"top-4 agrees with FP32:  port {tally['mine']}/{count}"
              + (f", bf16 reference {tally['bf16']}/{count}" if "bf16" in tally else ""))


if __name__ == "__main__":
    main()
