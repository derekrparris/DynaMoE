# DynaMoE SSD-Streaming Bottleneck Analysis — M1 Pro (MacBookPro18,1, 16 GB, 512 GB SSD)

**Model under test:** Ornith 1.5 35B A3B FP8 (`models--ornith-ai--Ornith-1.5-35B-A3B-FP8`, FlashMoE-repacked)
**Tool:** `Benchmarks/dynamoe_stream_bench.swift` (standalone CLI, uses the app's real Metal kernels + exact IO patterns)
**Date:** 2026-09-18

---

## TL;DR — The remaining bottleneck is NOT the pread pool

The `ExpertIOThreadPool` packed-expert path is healthy: it streams 8 random
3.0 MB expert slices at **3.7–4.0 GB/s cold / 19–23 GB/s warm**, essentially the
device ceiling for ≥8-way parallel reads. The bytes dragging sustained
throughput below 1 GB/s are the **backbone attention/SSM weights (2.30 GB) and
the lm_head (0.95 GB)**, which are read **every token by the GPU straight
through the VM page-fault path**. In FlashMoE mode the Rust core maps
`model_weights.bin` as shard 0 (core/src/lib.rs:351-366) and the app wraps it
with `MTLBuffer(bytesNoCopy: .storageModeShared)`; all Phase-A backbone
projections and the logits GEMV read from those file-backed mappings. Measured
fault throughput: **0.065–2.5 GB/s** depending on dispatch shape — and when
GPU work on *other* buffers is interleaved between backbone reads (exactly what
the app does every layer), the file-backed GPU mappings are dropped and every
dispatch re-faults at ~250 μs/page. Only 0.94 GB/token (22%) of the traffic
goes through the fast pool; 77% goes through the fault path.

Per-token data movement (parsed from `model_weights.json` + `layout.json`):

| Stream | Bytes/token | Path | Measured throughput |
|---|---|---|---|
| 8 routed experts × 40 layers | 0.94 GB | POSIX pread pool → staging MTLBuffer | **3.7–4.0 GB/s cold**, 19–23 GB/s warm |
| Backbone (q/k/v/o, GDN, conv, norms) | 2.30 GB | GPU reads mmap'd `model_weights.bin` (page faults) | **0.065–2.5 GB/s** |
| lm_head GEMV (248K vocab) | 0.95 GB | GPU reads mmap'd `model_weights.bin` (page faults) | same fault path |
| **Total** | **~4.19 GB** | | |

With the fault path running at its bad end, a decode token costs seconds →
~0.3 tok/s → **the "effective SSD bandwidth" the user observes collapses below
1 GB/s** even while the pool itself is running at 4 GB/s.

---

## Measured results (all numbers from this machine, Ornith FP8 packed layout)

### Storage layer

| Test | Pattern | Result |
|---|---|---|
| T1 | Single-stream sequential `pread`, 256 KB chunks, `F_NOCACHE`, cold | **1.39–1.56 GB/s** |
| T2 | 8-thread parallel sequential, same file, cold | **3.67–4.30 GB/s** |
| T3 | App's exact decode pattern: 40 layers × 8 random experts × 3.0 MB, `concurrentPerform` from `Task.detached(.userInitiated)`, page-cache fds | **3.75–4.01 GB/s** pass 1 (cold), **19–23 GB/s** pass 2 (warm); layer IO mean 6.0 ms cold / 1.0 ms warm |
| T3b | Same reads issued serially on one thread | 5.9–6.2 GB/s warm (≈8 × single stream) |
| T11 | Prefill pattern: K-way concurrent expert reads, one layer file (warm) | K=32: 4.3 / K=64: 4.7 / K=128: 3.5 / K=256: 4.2–5.6 GB/s |

**Conclusions:**
1. The M1 Pro SSD delivers ~4.3 GB/s **only with ≥8-way parallel DMA**. A single
   serialized read stream tops out at **~1.4–1.55 GB/s**. Anything that ends up
   single-streaming (or CPU page-faulting) is 3× slower than the user's
   expectation before the SSD is even saturated.
2. The app's pread pool is **not** the bottleneck. Random 3 MB expert reads hit
   near-ceiling rates cold, and warm hits are ~6× faster.
3. Prefill's 256-way reads are fine (4–5.6 GB/s). Reading the *entire* 807 MB
   layer file for any prompt ≥ ~32 tokens is inherent to the layout (256 experts
   × 3.0 MB), giving ~32 GB of IO per prefill at ~8 s — acceptable, but worth
   noting for long agent prompts.

### GPU compute (real kernels from `ComputeShaders.metal`, headless process)

| Test | Phase | Result |
|---|---|---|
| T4 | Decode MoE phase: 8 experts, 16 kernel dispatches + barriers (`fp8_swiglu_gate_up_simd` + `fp8_down_proj_accumulate_simd`), incl. commit+wait | **1.0–1.5 ms/layer → 42–61 ms/token** |
| T4b | Fused variant (all 8 experts in 2 dispatches, custom kernel) | **0.6–1.3 ms/layer → 23–52 ms/token** (1.2–2× faster) |
| T5 | Router phase: `rmsnorm_bf16` + `moe_router_topk_bf16` + commit+wait | **0.42–0.9 ms/layer → 17–36 ms/token** (dominated by sync overhead) |
| T9 | lm_head GEMV warm/resident (`bf16_gemv_simd`, 248K×2048) | **7.8–8.8 ms/token** (108–122 GB/s effective) |
| T12 | 807 MB prefill staging `makeBuffer` + first-touch | 70–84 ms fixed per prefill |

### The fault path (the bottleneck)

| Test | Pattern | Result |
|---|---|---|
| T8 | GPU reader over mmap'd safetensors, one 1 GB dispatch, 65K faults, cold | **2.45–3.6 GB/s** (0.41 s) — then 1400+ GB/s cached |
| T8 | CPU stride-touch (what `primeSlices` fallback does), cold | **0.66 GB/s** (one run), 12–20 GB/s cached |
| T10b [A] | GPU read of same 62 MB window ×5, no eviction | rep0 **928 ms**, reps 1–4 **0.2 ms** (fault once → resident) |
| T10b [B] | 62 MB window after 1.2 GB pread churn | survives: 0.6–0.7 ms (still cached) |
| T10 | App-shaped loop: per layer, fault-read 62 MB window interleaved with router/preads/MoE GPU work | **~945 ms PER LAYER (0.065 GB/s), every token, never recovers** |

**T10 vs T10b[A] is the smoking gun.** A 62 MB file-backed window read by a
single dedicated dispatch is fast *after the first fault* (pages stay wired).
But when the same read is **interleaved with other Metal command buffers using
other buffers** — which is exactly what the real `runTokenForward()` does every
layer — the GPU mappings for the file-backed shared buffers are dropped and
every dispatch re-faults at ~250 μs/page. The app's per-token access pattern
(backbone of all 40 layers + lm_head through `bytesNoCopy` maps over the 37 GB
of safetensors, interleaved with MoE/staging/attention work) reproduces this
condition continuously on a 16 GB machine.

### End-to-end pipeline

| Test | Structure | Result |
|---|---|---|
| T6 | Current app structure: router GPU → commit+wait → 8 preads → MoE GPU → commit+wait, ×40 layers (warm IO) | **130–149 ms/token ≈ 6.7–7.7 tok/s**, GPU busy ~63–68%, IO busy ~31% |
| T7 | Pipelined upper bound: layer l+1 experts pread'd during layer l MoE GPU (oracle prefetch, double-buffered staging) | **94–100 ms ≈ 10.0–10.7 tok/s** |

Even with warm IO, the serialized router→IO→MoE structure with **2–3 blocking
`waitUntilCompleted()` syncs per layer (≈120 syncs/token)** costs ~35–40% of the
achievable throughput. With cold/faulty IO the serialization penalty is worse
because the GPU idles completely during every IO phase.

---

## Root causes, ranked

1. **Backbone + lm_head re-fault every token through the mmap path (dominant).**
   In FlashMoE mode the core maps `model_weights.bin` as shard 0 and — despite
   the file being *fully read every token* (every layer's backbone projections
   plus the 0.95 GB lm_head are exercised once per token) — those reads go
   through `MTLBuffer(bytesNoCopy:)` over the mmap. `runTokenForward()` Phase A
   reads every layer's q/k/v/o/GDN/conv/norm tensors from `shardBuffers[...]`,
   and the logits GEMV reads `lm_head.weight` the same way. The result is
   3.24 GB/token flowing through the slowest path the hardware has (16 KB VM
   faults, serialized whenever interleaved GPU work unwires the file-backed
   mappings — T10). The pread pool already exists and is idle during Phase A.
   Additionally, the expert layer files are *also* all mmap'd (shards 1..40,
   ~32 GB of file-backed VM) even though experts are streamed via separate
   `pread` fds — the redundant mappings only add eviction pressure to the same
   page cache the backbone faults depend on.
   `WorkingSetManager.initialize()` is skipped in FlashMoE mode
   (ContentView.swift:10967), so the fd-based prefetch machinery that could
   rescue this path is dead code for packed models.

2. **Fully serialized GPU↔IO pipeline.** Per layer: GPU router pass → blocking
   sync → 8 blocking preads (SSD idle during GPU phases, GPU idle during IO) →
   blocking sync → residual pass. Zero overlap. T7 shows pipelining recovers
   ~35–50% even before touching the fault path.

3. **Per-layer command-buffer churn.** ~3 command buffers + 2–3
   `waitUntilCompleted()` per layer × 40 layers ≈ 0.3–0.6 ms/layer of pure sync
   overhead (visible as the gap between T5's 0.42 ms floor and its 0.55–0.9 ms
   measured cost).

4. **Kernel dispatch granularity in the MoE phase.** 16 dispatches + barriers
   per layer vs 2 fused (T4 vs T4b): 1.2–2× GPU-time reduction available.

5. **Speculative prefetch is dead weight in packed mode.** `prefetchLayerExperts`
   / `prefetchLayerBackbone` operate on `WorkingSetManager.expertSlices`, which
   is never initialized for packed models ( ContentView.swift:10967 skips
   `initialize()`), so the prefetch degrades to `primeSlices`' mmap stride-touch
   fallback — CPU page-faulting backbone pages at 0.66 GB/s, competing with the
   critical-path expert preads for NVMe bandwidth, while the GPU still re-faults
   them anyway.

---

## Recommendations (priority order)

1. **Pin the invariant weights in RAM once at load (~3.3 GB).** Backbone +
   lm_head + embeddings are touched every token but never change.
   `model_weights.bin` is already the perfect contiguous 4.5 GB container —
   at model load, `pread` it through the existing pool into dedicated
   *anonymous* `MTLBuffer` staging (it fits comfortably on 16 GB) and point the
   Phase-A / logits kernels at those buffers instead of the mmap. This converts
   3.24 GB/token of fault traffic into resident DRAM reads (~68 GB/s → ~34 ms
   per token) — expected decode impact: **from ~0.3–0.5 tok/s to ~3–4 tok/s**
   before any other change. The 32 GB expert files remain streamed; they are
   the only weights that truly need streaming. Also drop the redundant
   `bytesNoCopy` mappings of the 40 expert layer files (the pread pool uses its
   own fds), freeing ~32 GB of file-backed VM pressure.

2. **Software-pipeline the expert streaming.** Double-buffer the expert staging
   (two 32 MB `MTLBuffer`s, slot l%2), and after layer l's router completes,
   kick the 8 preads for layer l+1's *predicted* experts (the Markov
   `ExpertTransitionTracker` machinery already exists at ContentView.swift:6010)
   on a background queue while the GPU computes layer l's MoE phase. T7 shows
   this alone is worth ~10 tok/s (vs 6.7 serialized). Only cache-miss experts
   (predictor misses) should block.

3. **Fuse the MoE expert kernels.** One `bench_fp8_moe_gate_up_fused`-style
   dispatch (grid 8×512 threadgroups) + one down-projection dispatch replaces
   16 dispatches + 17 barriers. ~1.2–2× GPU-phase reduction.

4. **Cut sync points.** Merge the residual/next-layer-prep command buffer
   (`nextCmd`) into the same command buffer as the MoE pass, and ideally move
   the expert pread *kickoff* to before `moeCmd.waitUntilCompleted()` (reads for
   the *next* layer can be issued once its router logits land; only
   `staging[l+1 % 2]` writes need synchronization before the next Phase A read).

5. **Wire `F_RDADVISE` into the packed layer fds** (`ExpertIOThreadPool`).
   Issuing `fcntl(fd, F_RDADVISE)` for the next layer's predicted expert byte
   ranges costs nothing and lets the kernel readahead overlap GPU compute.
   The architecture doc claims this exists; it currently does not.

6. **Prefill niceties (secondary):** reuse a persistent staging buffer across
   prefills (saves the 70–84 ms alloc+touch), and skip reading experts with
   zero routed tokens (already done — `expertTokenMap` guard). At K≥128 the
   pool already runs 3.5–5.6 GB/s, so prefill is near the device ceiling once
   prompts exceed ~100 tokens; the 32 GB/prompt floor is a layout property.

## Expected end state

With backbone pinned (1) + pipelining (2): per-token cost ≈ max(expert IO
0.94 GB / 4 GB/s ≈ 235 ms, GPU ~60–90 ms) + fault-free fixed overhead ≈
**~4–5 tok/s sustained streaming** with the SSD saturated near 4 GB/s —
finally letting the hardware's parallel read rate show up in the app. Further
out, the architectural ceiling on this hardware is ~4 tok/s while experts must
be streamed (0.94 GB/token), or ~10+ tok/s if the expert working set can be
partially retained across tokens (expert-hit-rate telemetry already tracks this).

---

## IMPLEMENTATION — Fix #1: backbone pinning (SHIPPED)

**Changes:**

1. `DynaMoE/DynaMoE/ExpertIOThreadPool.swift` — new
   `ExpertIOThreadPool.preadFileIntoBuffer(fd:dst:length:threads:)`: N-way
   parallel POSIX pread of a whole file into a pre-allocated anonymous
   `.storageModeShared` buffer. Returns elapsed seconds only if the transfer is
   complete (a short read returns nil — partial data would silently corrupt
   weights, so callers must fall back).

2. `DynaMoE/DynaMoE/ContentView.swift` (`loadAndBridgeToMetal`): in FlashMoE
   mode, after the shard mapping loop, shard 0 (`model_weights.bin`) is pread
   once into an anonymous MTLBuffer (8 threads, ~1.5 s for 4.19 GB) and
   `buffers[0]` is replaced with it. All Phase-A backbone projections, the
   final norm, and the lm_head GEMV then read from resident anonymous memory
   instead of the file-backed mmap. Controlled by
   `@AppStorage("dynamoe_pin_backbone_weights")` (default **true**; toggle via
   `defaults write <bundle-id> dynamoe_pin_backbone_weights -bool false` until
   a Settings UI row is added). On alloc failure, `maxBufferLength` overflow,
   or incomplete transfer it logs and falls back to the existing mmap path.

**Benchmark verification (T10 vs T13, same app-shaped loop, 2 tokens):**

| Variant | Backbone reads | Per-token | Throughput |
|---|---|---|---|
| T10 — mmap (previous behavior) | re-faulted every layer, every token | ~36.7–38.7 s | **0.03 tok/s** (0.11 GB/s effective) |
| T13 — pinned (shipped fix) | 1.45 s one-time load pin, then **38.3 ms/token** steady-state | **394 ms** | **2.54 tok/s** (still serialized + cold expert IO) |

Steady-state backbone cost went from ~945 ms/layer (0.065 GB/s fault path) to
~1 ms/layer (DRAM). The residual 394 ms/token in T13 is the serialized expert
streaming + command-buffer sync structure — exactly what fixes #2 (software
pipelining), #3 (fused kernels), and #4 (sync-point reduction) address next.

Note: the 4.19 GB pinned buffer replaces ~2–3 GB of file-cache residency that
the mmap path held in steady state, so net committed memory grows by roughly
1.5–2 GB; on 16 GB machines this is the difference between a stable resident
working set and a thrashing page cache. The toggle exists as an escape hatch
for tighter-memory configurations.

---

## IMPLEMENTATION — Fix #2: software-pipelined expert streaming (SHIPPED)

**Changes (all in `ContentView.swift` unless noted):**

1. **Double-buffered expert staging.** Second 64 MB staging buffer
   (`expertStagingBufferB`) allocated next to the original. Layer l consumes
   buffer `l & 1`; the prefetcher writes buffer `(l+1) & 1`.

2. **Pipelined decode consumption** (`runTokenForward`, packed branch): per layer,
   1. wait for the in-flight prefetch of layer l (no-op when absent),
   2. consume prefetched experts by slot map (`expertId -> slot`), synchronously
      pread **only predictor misses** into free slots,
   3. kick the background pread of layer l+1's Markov-predicted experts
      (top-10 by aggregated transition counts, `ExpertTransitionTracker`) into the
      opposite buffer — it overlaps this layer's MoE GPU phase,
   4. GPU phase reads from the parity buffer with per-expert slot offsets.

   Synchronization: one fresh `DispatchSemaphore` per kick (no stale-signal reuse);
   wait has a 5 s timeout and the slot map is tagged with the resident layer so
   stale maps can never be consumed. Capacity is guaranteed by construction
   (prefetch slots ≤ capacity − maxTopK; misses ≤ maxTopK).

3. **Adaptive gate.** Kicks are disabled for the rest of the generation if the
   predictor's observed hit rate falls below 20% over the first 64 consumed
   experts (prediction-useless routing would otherwise waste NVMe competing
   with critical-path miss IO). Retried next generation as the tracker learns.

4. **Backbone prefetch disabled for packed models.** All
   `prefetchLayerBackbone` call sites are gated on `packedExpertsDir == nil` —
   with fix #1 the backbone is resident; the old WSM fallback only CPU-faulted
   safetensors pages and competed with the expert preads.

5. **Telemetry.** `[DIAG] Layer ... (prefetch H/T)` per layer and
   `| Prefetch H/T (N%)` in the token summary.

**Benchmark verification (T14 — identical sticky routing replayed in both
variants, 2 tokens, Markov predictor in the pipelined variant):**

| Variant | Per-token (warm tokens) | Sync IO | Predicted-hit rate |
|---|---|---|---|
| T14 serialized (cold cache) | 303 ms (3.3 tok/s) | 234 ms/token | 0% (no prefetcher) |
| T14 pipelined | **66 ms (15.1 tok/s)** | **0.8 ms/token** | **98%** |

That is a **~4.5× improvement over the serialized cold path** and **~2.3× over
the serialized warm path** (T6: 147 ms/token with fully warm expert cache).
Combined with fix #1 (backbone pinning), the projected steady-state decode on
this machine is ~10–15 tok/s while experts stream at NVMe line rate. The real
hit rate depends on the model's routing locality — the `[DIAG]` prefetch
telemetry now reports it live, and the adaptive gate bounds the worst case to
the previous (all-miss) behavior.

*Benchmark note:* the T10 fault-storm diagnostic section (now ordered last)
can destabilize the bench process after hundreds of GPU fault dispatches on
16 GB machines (SIGSEGV in the pread workers, seen in all orderings); the
pipeline tests T13/T14 complete cleanly and are unaffected. Run with
`--tokens 2` for the full suite.

---

## CORRECTNESS FIX (post-ship, SHIPPED)

Initial QA on the live app produced garbage output (mixed-language tokens) —
reproduced and root-caused with a new bench harness (T15: replicates the exact
consume/kick shape and verifies every consumed expert's staged bytes against a
direct read of the layer file). Two compounding bugs in the original consume
path:

1. **Slot-space leak → out-of-bounds staging writes.** `nextSlot` was only
   reset by the kick's synchronous part — which never fires when the Markov
   prediction is empty (cold tracker on the first tokens, or after the adaptive
   gate turns the prefetcher off). Miss allocations then accumulated across
   layers sharing the parity buffer, eventually allocating slots beyond the
   staging capacity (21 slots × 3.0 MB = 64 MB) and preading weights past the
   end of the buffer.

2. **Silent wrong-expert fallback.** Skipped experts (capacity guard `continue`)
   and any missing slot-map entry fell back to `slotOf[expert.id] ?? slot` in
   the GPU loop, feeding the GPU **a different expert's weights** — the direct
   cause of the gibberish output.

**Fixes (app + bench):**

- The consumer now **owns the slot space**: when the resident map is stale or
  absent for this layer, `nextSlot[bufIdx]` is reset to 0, so misses always fill
  from slot 0 and the slot count is bounded by construction
  (kick slots ≤ capacity − maxTopK, misses ≤ maxTopK).
- The capacity guard is now a **hard abort** instead of a silent `continue`
  (unreachable by construction; aborting beats corrupting).
- The kick publishes **only completed preads** (`result == size`); a short read
  is no longer advertised as resident (the consumer re-reads it as a miss).
- Removed the `?? slot` fallback hazard by the above (every consumed expert is
  guaranteed a real slot).

**Verification:** new bench harness T15 (app-shaped consume/kick + byte-exact
verification of every consumed expert against a direct file read): **960
consumed experts, 0 mismatches** after the fix (the pre-fix shape reproduced
out-of-bounds slots 21/22 with wrong weights at byte level — exactly the
in-app corruption).

---

## PREDICTOR FIX (post-QA #2, SHIPPED)

Live-app QA #2: output correct, but the Markov (layer l → l+1 transition)
predictor measured **0/320 hits** on real Ornith decode — the cross-layer
routing signal is genuinely not predictive for this model's traffic, so the
adaptive gate fell back to all-miss sync IO (~400 ms/token).

**Fix: temporal-first prediction.** Consecutive tokens in fluent text route to
largely the same experts **at the same layer** (content locality — the
"Pre-gated MoE" insight; MoE-Infinity's temporal expert-reuse profile). The
signal already exists in the runtime: `previousLayerActiveExperts[l + 1]`
holds the previous token's expert set at layer l+1 (the dict persists across
tokens; the current token only writes entries for layers it has reached).

New prediction for layer l+1 (kicked at layer l, after router l resolves):

1. **Temporal (primary):** previous token's full 8-expert set at layer l+1.
2. **Markov fill:** cross-layer transition top-10 appended (dedup) into any
   remaining prefetch budget (capacity − maxTopK slots).

T14 benchmark (temporal-first predictor, sticky routing, same as before):
**pipelined 13.4–17.1 tok/s vs serialized 3.1–3.3 tok/s (~4–5×), 98% hit
rate**, sync IO ~1 ms/token. T15 byte-verification: 0 mismatches with the
temporal predictor in the loop.

Fix: gate judgement (and observation accumulation) now requires
`hadTemporalHistory` (captured at token start: `!previousLayerActiveExperts.isEmpty`).
Token 0's legitimate cold-start 0% can no longer latch the gate; from token 2
onward the accumulated rate reflects real prediction quality, and the gate
only latches off on genuine evidence of a unpredictable-routing workload.

---

## PREDICTOR RESULTS + 2-TOKEN HISTORY (post-QA #4, SHIPPED)

Live QA #4 (Ornith, real prompt, tokens 240-245): temporal predictor running —
**38-54% hit rate** (trending up as the session progresses; best token 54%),
per-token **IO 120-160 ms** (down from ~230 ms), token time 299-400 ms.
The adaptive gate no longer false-trips.

Next lever shipped: **2-token temporal union**. Single-token history caps
coverage near 50%; the union of the last two tokens' sets at the target layer
(router top-k order is weight-ranked, so earlier entries are
higher-confidence) pushed the measured coverage to ~70-75% in the bench.
Prefetch budget unchanged (capacity − maxTopK = 13 slots: t-1's 8 + t-2's
top-5). Expected effect on the user's numbers: misses drop from ~4/layer to
~2/layer → sync IO from ~120-160 ms toward ~60-90 ms → token time toward
~250-280 ms (~3.6-4 tok/s).

Remaining cost structure at this point (from QA #4 logs):
- RouterGPU (Phase A attention/SSM + router): ~106 ms/token — now the largest
  single block; real attention/GDN compute + 3 blocking syncs per layer.
- MoE GPU: ~44 ms/token.
- Other (logits GEMV + sampling): ~31 ms/token.
- IO: hit-rate dependent (120 ms at ~50% hits; floor ~30-45 ms at 75-100%).

Fixes #3 (fused MoE kernels, 1.2-2x on the MoE block) and #4 (sync-point
reduction) are the next structural wins once IO is fully hidden.

---

## FIX #3 SHIPPED: fused MoE expert kernels

Long-context QA (#5, the 1,685-token "explain MoE" turn at 2.0 tok/s) showed a
NEW dominant cost: **RouterGPU (Phase A attention/SSM + router) grew from
~106 ms/token at short context to ~484 ms/token at ~1,900 tokens** — 72% of
the token. Two causes identified:

1. **GQA attention decode kernels dispatch one thread per query head**
   (16 threads total, each serially looping the whole KV cache) — latency-bound
   at long context, near-zero GPU occupancy. Affects the 10 full-attention
   layers.
2. UI/compositor contention from re-rendering the accumulating markdown view
   every ~80 ms.

Shipped in this round: **fused MoE expert kernels** (fix #3) — new
`fp8_moe_gate_up_fused` + `fp8_moe_down_fused` MSL kernels in
`ComputeShaders.metal`, dispatching all routed experts in **2 dispatches
instead of 16 + barriers**. Design notes:

- Grid `(intermediateDim, topK)` / `(hiddenDim, topK)`; the kernels walk an
  explicit `slotList[rank] -> staging slot` mapping because the consume path
  assigns slots in prediction-hit order, not router-rank order; weights are
  rank-ordered (`fusedSlotWeightsBuffer`, filled at consume time).
- `interBuffer` enlarged to topK × intermediateDim floats (dense per-rank
  intermediate); safe for other paths (they use offset 0 only).
- Applies to per-row-scaled FP8 layouts; block-scaled and Q4/BF16 keep the
  existing per-expert path; automatic fallback when pipelines are missing.

**Verification:** T16 in the bench runs both paths on identical staged data
and routing — **max relative diff 0.00e+00 across all 2048 outputs** (byte
equivalent). Bench timing: fused MoE GPU phase **0.72 ms/layer → ~29 ms/token**
vs 1.14 ms/layer → ~46 ms/token per-expert (T4) — **~37% less GPU time on the
MoE block**, and fewer dispatch/barrier round-trips to be preempted by the
compositor.

Next round: fix #4 (merge the per-layer command buffers to cut ~120 blocking
GPU syncs/token — the direct lever on the RouterGPU scheduling-gap exposure),
and a parallelized long-context GQA attention kernel (flash-decoding style
split-K over the KV cache) for the 10 full-attention layers.

---

## QA #6 + DIAGNOSTICS (shipped)

QA #6 (the joke turn, 112 tokens): **3.2 tok/s** end-to-end, tokens 316-367 ms.
Fused-path MoE GPU measured ~39-40 ms (bench floor 29 ms + shared expert +
dispatch overhead). Hit rate 30-47% on the short/novel response (weaker
temporal locality than the long technical response's 65%) — the IO cost tracks
the hit rate exactly as designed. MoEGPU ~39 ms.

Added instrumentation for the next round:
- `[DIAG] Layer N(gqa): PhaseA=X.XXms IO=...` — per-layer Phase-A time with a
  marker on the 10 full-attention layers, to pinpoint whether the long-context
  RouterGPU growth is attention-kernel time (gqa layers spike) or uniform
  stretch (compositor/clock contention).
- One-time `⚡ [MoE] fused expert path: ACTIVE|off (reason)` line per
  generation, so the fused path's activation status is visible in logs.

---

## FIX #4a SHIPPED: chunked (flash-decoding style) GQA attention

QA #6 diagnostics nailed the long-context whale: the 10 full-attention layers
cost **23 ms each at ~1,000 tokens of context → 41-52 ms each at ~1,700
tokens** (GDN layers stay ~1.25 ms) — the one-thread-per-head kernel is
latency-bound with ~zero GPU occupancy. Separately, end-of-session logs showed
system-wide inflation (IO 124→1,012 ms, MoE 39→102-149 ms, Other 27→150-187 ms)
— memory-pressure/swap fighting the pinned backbone + KV + UI; separate
mitigation, see next section.

**Shipped: `gqa_attention_decode_headgate_f16_chunked` + combine kernel.**
Flash-decoding split-K: grid `(numQHeads, numChunks)` with ~64 keys per chunk;
each thread computes an online-softmax partial (m, l, acc) for its chunk; a
combine kernel log-sum-exp-merges the partials and applies the per-head sigmoid
gate. Scratch: heads × ≤256 chunks × (2 + headDim) floats (~4.2 MB max). Wired
into the `.fp16` KV gated path with a fallback to the original kernel.

**Verification (T17 in the bench, synthetic FP16 KV, 16 Q-heads / 2 KV-heads /
256 dim):** outputs match the original kernel to max rel diff ~4e-3 (FP
reduction-order noise), and the timing at decode-relevant contexts:

| Context | Original (1 thread/head) | Chunked | Speedup |
|---|---|---|---|
| 1,000 | 23.0 ms | 2.9 ms | 7.9× |
| 2,000 | 45.7 ms | 4.8 ms | 9.6× |
| 4,000 | 91.2 ms | 13.5 ms | 6.8× |
| 8,000 | 182.1 ms | 27.2 ms | 6.7× |

Expected live impact at ~2,000-token context: RouterGPU drops from ~484 ms
toward **~270-280 ms** (attention 45→5 ms × 10 layers), and long responses stop
degrading linearly. Combined with the fused MoE path (shipped) the fixed GPU
cost per token is now ~40-60 ms of attention + ~29 ms MoE.

Remaining known items: the end-of-session memory-pressure inflation (IO/Other
spikes at 1,600+ tokens — likely the 4.5 GB pinned backbone + KV + UI working
set saturating 16 GB RAM), and the O(context) repetition-penalty scan in
sampling (the Other bucket growth).

---

## QA #7 + DISPATCH-SITE FIX (shipped)

QA #7 (~1,100-token response): gqa layers still at ~26 ms — the chunked
attention was wired into the WRONG dispatch site. The edit had landed in
`runLayerWisePrefill`'s batched-attention branch (single-token kernel operating
on multi-token batched buffers — both ineffective and a correctness hazard),
while the decode site (`runTokenForward`, fp16 KV) kept the original kernel.

Fixed: the chunked block is removed from the prefill branch (original batched
dispatch restored) and inserted at the decode site as the first branch of the
fp16 gated path with a fall-through to the original headgate kernel. The
prefill path must NOT use the chunked kernel: it is single-token decode-only
(q/k/v/out offsets assume tokenIdx=0).

Also identified from QA #7 logs: token 1087's hit rate dropped to 13% with
IO back to ~245 ms — the temporal predictor degrades when the model's output
is fully novel (the joke's creative punchline region); the 2-token history
plus markov fill covers ~30-65% of traffic. The prefetch wait remains bounded.

---

## QA #8 + THE ACTUAL PATH FIX (shipped)

QA #8 (fresh prompt, user confirms FP16 KV): gqa layers STILL ~23 ms —
the chunked dispatch was wired to the HEADGATE branch, but **Ornith-class
models have no `self_attn.g_proj` tensor** (verified in model_weights.json),
so `attnGateProjTensor == nil` and decode actually falls through to the
**fused Q+Gate fallback** (`gqaDecodeF16Pipeline` /
`gqa_attention_decode_fused_f16`): per head, a fused vector of Q[256] +
Gate[256] with an elementwise sigmoid gate — also one-thread-per-head.

**Shipped: `gqa_attention_decode_fused_f16_chunked` + combine.** Mirrors the
fused path exactly: partial kernel reads the fused Q vector and writes
(m, l, acc) partials; combine log-sum-exp-merges and applies the elementwise
sigmoid gate from the fused layout. Wired as the first branch of the
`gqaDecodeF16 ?? gqaDecode` fallback with fall-through.

**Verification (T17b, fused Q+Gate synthetic, FP16 KV):**

| Context | Original | Chunked | Speedup | max rel diff |
|---|---|---|---|---|
| 1,000 | 23.5 ms | 2.8 ms | 8.4× | 3.1e-3 |
| 2,000 | 46.9 ms | 4.9 ms | 9.6× | 2.6e-3 |
| 4,000 | 92.8 ms | 14.1 ms | 6.6× | 1.3e-3 |
| 8,000 | 184.9 ms | 27.1 ms | 6.8× | 5.2e-4 |

Also observed in QA #8: a fresh long session ran with IO at ~500 ms/token
(expert reads 13-15 ms/layer, prefetch 8-25%) — the memory-pressure signature
appearing EARLY now; the SSD itself delivering ~1-1.4 GB/s on expert reads vs
~4 GB/s in earlier sessions. Likely causes: swap/compressor traffic from the
4.19 GB pinned backbone + KV cache + UI working set, or page-cache eviction by
the app's own allocations. Investigation queued separately.

---

## GATE WARMUP FIX (post-QA #3, SHIPPED)

Live-app QA #3: output correct, but `[DIAG]` showed the prefetcher disabled
from the first token onward (0/320 forever, IO back to ~6 ms/layer sync).

Root cause: the adaptive gate evaluated on the **first 64 observations of
token 0** — which by design has *no* temporal history (`previousLayerActiveExperts`
empty → 0% prediction) — and latched the prefetcher off for the entire
generation. Every later token inherited the disabled state, so the temporal
predictor never ran past token 0.

Fix: gate judgement (and observation accumulation) now requires
`hadTemporalHistory` (captured at token start: `!previousLayerActiveExperts.isEmpty`).
Token 0's legitimate cold-start 0% can no longer latch the gate; from token 2
onward the accumulated rate reflects real prediction quality, and the gate
only latches off on genuine evidence of a unpredictable-routing workload.

---

## FIX #5: PREFILL EXPERT-READ PIPELINING (TTFT, SHIPPED)

**Problem.** TTFT on a fresh chat was ~14-15s. Prefill (`runLayerWisePrefill`)
Phase B was fully serialized per layer: `ExpertIOThreadPool.dispatchSync`
reads ALL active experts (up to 256 x 3.15 MB = ~806 MB/layer) -> encode MoE
cmd -> commit -> `waitUntilCompleted`. IO and GPU never overlapped: total
prefill time ~= sum(IO) + sum(GPU). With ~32 GB streamed (256 experts x 40
layers) at ~4-5.6 GB/s parallel pread (~6.4s) plus ~4-6s of GPU MoE/attention
and 80 sync points, the two never overlapped.

**Why prediction works here.** Router output for layer l+1 is not known until
layer l's MoE completes (it needs `nextHBuf`), so exact next-layer sets are
impossible — but temporal prediction is: layer l's active set predicts layer
l+1's. For P >= ~64 prompt tokens nearly every expert is active on every layer
(8 draws/token; coupon collector saturates 256 experts at ~200 tokens), so
overlap between adjacent layers is near-perfect and the predicted set IS the
read set. For small P (delta prefill of a short follow-up), overlap is partial
and misses are read synchronously — still no worse than today.

**Shipped design (packed-expert path only):**
- Double staging buffers `prefillStagingBufferA/B` (each the same size as the
  old single buffer: min(P*topK, numExperts) * expert_size). `prefillStagingBuffer`
  is now a var rebound per layer; all existing kernel binding sites unchanged.
- Per MoE layer (packed branch):
  1. Drain stale kicks targeting earlier layers (dense-layer skips) by waiting
     their semaphores — guarantees buffers are quiescent before reuse.
  2. If a speculative kick exists for this layer (`prefetchedFor[l]`): wait its
     semaphore, adopt its slot map as hits, read only the misses synchronously
     into slots after the predicted ones. Otherwise read all (as before) into
     the buffer NOT used by the last MoE GPU command.
  3. Encode + `commit()` the MoE command, then — while the GPU runs — kick an
     async pread of layer l+1's predicted set (this layer's active set, sorted,
     into slots 0..K-1) into the OTHER staging buffer via the new
     `ExpertIOThreadPool.dispatchAsync(tasks:done:)` (serial kick queue, one
     in-flight kick per parity; completion signals a DispatchSemaphore).
  4. `waitUntilCompleted()`.
- `defer` drains any unconsumed kicks before function exit (cancellation and
  error paths) so no in-flight pread writes into freed buffers.
- Toggle: `dynamoe_prefill_pipeline` UserDefaults (default ON). When OFF no
  kicks are issued and behavior matches the old sync path.
- Telemetry: per-layer `[PREFILL-DIAG] Layer l: act= hits= miss= ioWait=` and
  an end-of-prefill `[PREFILL-DIAG] Summary: ioWait= gpuWait= hitRate=
  prefillTotal=` line.

**Expected effect.** Per-layer cost becomes max(IO_window, GPU_window) instead
of IO + GPU. Estimated TTFT ~14-15s -> ~8-10s (IO-bound floor ~6.4s + Phase A +
non-overlappable head/tail). Decode path unaffected.

### QA #10 — Fix #5 live results (continued session, delta prefill)

Per-layer: L0 cold (act=245, ioWait=191.9ms); L1-39 act=157-238, hits 55-90%,
ioWait 10-45ms (~4-6 GB/s on miss reads). Pipeline works.

| Block | Time |
|---|---|
| ioWait (expert streaming) | 1.48s (was ~6s serialized) |
| gpuWait (prefill MoE cmds) | 3.62s (~90ms/layer) |
| Unaccounted (Phase A: batched attention over ~12k ctx, 30 GDN layers' sequential recurrence, per-token dispatch loops, embed, tail) | ~7.0s |
| **prefillTotal** | **12.11s** |

TTFT 14.2 -> 12.81s on this turn. Remaining whales, biggest first:
1. **Phase A ~7s** — unmeasured until now. Suspects: GDN per-token sequential
   recurrence (P tokens x 30 layers of per-token kernels), per-token
   shared-gate/routing dispatch loops (Q8 path dispatches P kernels per layer),
   batched attention over long context. Added `PhaseA=` timing to the per-layer
   line + `other=` to the summary to split it on the next run.
2. **MoE gpuWait 3.62s** — prefill uses the per-expert batched kernels
   (~2 dispatches + barrier per expert per layer = ~19k dispatches). A fused
   slot-list kernel (like decode's fused path) could cut dispatch overhead,
   but measured ~90 GFLOP/s suggests poor per-expert threadgroup shapes
   (count=1-3 tokens/expert) dominate — needs a batched-over-tokens rewrite,
   not just dispatch fusion.
3. Decode-side memory-pressure spikes (QA #9 token 1146: IO 504ms, prefetch
   12%, per-layer IO 12-16ms vs 2-6ms baseline) — swap/compressor stealing
   SSD bandwidth; separate lane.

---

## FIX #6a: MULTI-ROW CHUNKED PREFILL ATTENTION (SHIPPED, then REVERTED — see below)

**Problem identified.** Prefill gqa layers fall to `gqaDecodeF16Pipeline` (the
old per-thread full-KV-loop kernel — one thread per (head,row), 12k+ serial
iterations) because `isStandardGqa=false` (fused Q+Gate) and there is no
g_proj tensor. Measured prefill PhaseA = 5.66s of 12.11s TTFT; gqa layers
~140ms each (~1.2s of it), GDN seq-scan ~135ms x 30 layers (~4.05s, the
bigger half).

**First attempt (rows-chunked kernels) FAILED bench verification.** Wrote
`gqa_attention_decode_fused_f16_chunked_rows(+_combine)` (3D grid head x
chunk x row, per-row causal limit). T17c results:

- Numerically fine (error growth with row count is max-over-more-samples
  statistics; per-element error ~2e-3, same noise floor as T17b; CPU ground
  truth confirms ref kernel within 2e-3).
- But 0.2-0.3x SLOWER than the old kernel at rows=64/128 (681-2948ms vs
  121-262ms ref) and only 1.5x faster at rows=8.

Root cause (measured, not fully explained): one thread per (head, chunk, row)
with a serial chunk loop is latency-bound garbage on M1 — both the old kernel
(~23us per serial KV iteration at 16-1024 threads) and the new one scale
terribly with thread count above ~1-2k. T17b's decode win (chunkSize~64,
1008 threads) does not extrapolate: 32k threads of this shape run 25x slower
than thread-count-proportional. The correct prefill design needs SIMD-lane
tiling (32 lanes split the headDim dot + online-softmax state per threadgroup),
which is a bigger kernel rewrite.

**Action: REVERTED the prefill wiring to the old kernel path** (kernels remain
compiled; decode's chunked path untouched and verified). Wiring kept behind a
dead branch to be replaced by the tiled kernel in a follow-up.
