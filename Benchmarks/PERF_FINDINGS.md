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

---

## FIX #7: DECODE PREFETCH BUDGET + TOKEN-BOUNDARY LOOKAHEAD (SHIPPED)

**Diagnosis.** Decode hit rate was capped by two structural limits:
1. `expertStagingSize = 16 slots` → prediction budget = capacity - maxTopK
   = **8 slots/layer** — exactly the previous token's set. The Markov and
   prev-prev-history sources were starved by construction (budget filled by
   the first history source alone). Hit rate 42-56% == P(prev set ∩ active).
2. Every token starts cold: layer 0-1's kicks are issued mid-token
   (B(l) kicks l+1), so Layer 0 showed a persistent "prefetch 0/8" and its
   8 misses read synchronously at token start (5-10ms in DIAG), while the
   logits GEMV + sampling ("Other" ~27ms) left the SSD completely idle.

**Shipped changes (packed decode path only):**
- Staging capacity 16 → 32 slots per buffer (~100.8 MB/buffer, ~202 MB both;
  was ~50 MB/buffer).
- Prediction budget capped at 16 candidates (larger lists waste IO on false
  positives that compete with true-positive reads for SSD bandwidth — the
  absolute floor is ~1.0 GB/token of needed reads at 4-6 GB/s, so the window
  fits ~4-8 completions; ordering quality beats list length).
- Intersection-promoted ordering: experts present in BOTH history tokens
  (stable core) first, weight-ordered; then prev-only; then prev-prev-only;
  then Markov.
- Token-boundary lookahead kick: after the last layer's MoE completes, before
  the final norm + logits GEMV, kick predicted sets for layers 0 and 1 (from
  the current token's sets) into the two staging buffers — overlapping the
  ~25-30ms Other window. The per-layer kick for layer 1 then skips (new
  resident-layer guard in `kickPackedPrefetch`) when the boundary prediction
  is still resident, avoiding duplicate reads of an identical prediction.
- `packedPrefetchPending` converted from a single (layer, sem) to a per-layer
  dict to support the two in-flight boundary kicks (one per staging buffer).
- Consume logic unchanged: wait for the layer's pending sem, adopt resident
  slots, sync-read misses into slots after the kick's.

**Expected effect.** Layers 0-1 stop being cold every token (~10-25ms/token
saved); hit rate should rise from 42-56% toward 55-70% on content-following
turns, cutting IO toward ~80-110ms/token → ~4.5-5.0 tok/s. Watch: Layer 0/1
"prefetch" counts in [DIAG] (no longer 0/8), the per-token Prefetch X/320
percentage, and the IO block.

**Note on theoretical ceiling.** Every needed expert is read from SSD/page
cache exactly once per token (kick or sync), so total SSD traffic is ~1.0
GB/token regardless of hit rate; the sync-wait portion is what hit rate
removes. Page cache makes repeat experts cheap. With ~5.6 GB/s effective
random-read throughput the absolute decode floor is ~180-200ms/token (~5
tok/s) at this expert granularity; beating that needs smaller read
granularity (component-level streaming), a different project.

### QA #11 — Fix #7 REGRESSION, REVERTED

Tokens 251-305ms -> 574-686ms (1.6 tok/s). IO 119-160 -> 343-439ms; RouterGPU
67-85 -> 132-151ms; per-layer IO 1-6 -> 9-28ms. Hit rate ROSE (48-77%) but net
throughput collapsed.

**Root cause (confirmed by pattern):** budget 8 -> 16 doubled every kick to
~50MB, but the per-layer overlap window (MoE GPU ~1-1.7ms + next PhaseA
~1.3-3ms) only fits ~2-8 expert reads. Kicks no longer complete before
consume; the all-or-nothing per-kick semaphore makes EVERY layer wait the
FULL 16-expert kick (~15-20ms) even when it needs only 8. The wait eats the
window. The old 8-expert kicks (~5ms) fit the window; 16-expert kicks never
do.

Also unresolved: the token-boundary kick produced 0/8 at Layer 0 in every
DIAG (data not adopted or prediction source ineffective); combined with the
concurrent-kick clobber race on shared buffers (per-layer kick resets
resident[] while a still-reading boundary kick owns the buffer). Fixing
properly needs per-slot or per-kick-tail tracking, not bigger budgets.

**Action: REVERTED all Fix #7 changes** (staging 32->16, budget cap 16->8,
ordering simplification, boundary kick, pending dict, resident guard).
Decode returns to the QA #10 state (~250-300ms tokens, 42-56% hits).

---

## HOTFIX: SPARK 2.5 TOKEN-SOUP REGRESSION (chunked headgate gate binding)

**Symptom.** Spark X2.5-4B (dense DeepSeek-style: 36 layers = 27 sliding@512 +
9 full, head_dim 256, 16 Q / 4 KV heads, headwise sigmoid output gate via a
separate g_proj) produced multilingual token soup from the first thinking
token after the chunked-attention commit (5ce99f2).

**Root cause.** Decode's chunked headgate branch bound the combine kernel's
gate input (buffer 4) to `qGateBuffer` (the fused QKV projection output) while
the kernel reads `gateVector[qHeadIdx]` — a per-head scalar. Spark's real
per-head gate logits live in `bVectorBuffer` (g_proj output, the same buffer
the OLD headgate kernel binds). The combine therefore scaled every head by
sigmoid(Q-vector bytes) — garbage gates, soup output. The fused Q+Gate chunked
path (Ornith) was unaffected: its gate IS inside the fused vector.

**Why T17 missed it.** The bench bound a dedicated synthetic gate buffer as
buffer(4) for BOTH ref and chunked kernels — self-consistent, so equivalence
held; it validated the KERNELS, not the app's argument binding.

**Fix.** One line in the decode fp16 headgate chunked combine: bind
`bVectorBuffer` (matching the non-chunked headgate branch). Rebuilt green.
Lesson recorded: kernel-level bench equivalence does not cover wiring-level
buffer mismatches; future chunk ports need an app-binding review or a bench
that mirrors the app's exact buffer roles.

---

## FIX #8: PER-SLOT COMPLETION TRACKING FOR DECODE PREFETCH (SHIPPED)

**Design (incorporating QA #11's lessons):**
1. `ExpertIOThreadPool.dispatchAsync(tasks:onTaskDone:done:)` — per-task
   completion signals. Each finished pread marks its slot OK and signals the
   slot's semaphore; `done` signals after the whole batch.
2. `PackedExpertKick` (class): targetLayer, bufIdx, staging buffer, slotBase,
   candidate list, per-slot semaphores + OK flags, all-done semaphore.
   Published SYNCHRONOUSLY at kick time so the consumer can always see it;
   reads then run on the background queue.
3. Consume (per layer): if the kick targeting this layer is resident, wait
   ONLY the needed slots (15ms timeout each, then synchronous duplicate read
   into a miss slot — correct because the kick writes a different slot);
   false-positive reads never block. Stale kicks (skipped dense layer, token
   boundary) are retired with a full wait before their slot space is reused.
4. **Kick moved after `moeCmd.commit()`** — the retire-wait for the previous
   kick into the same parity buffer (2 layers apart) now overlaps the GPU
   execution of the current MoE command instead of serializing the CPU (the
   QA #11 regression shape).
5. Capacity 32 slots (~100.8 MB/buffer); prediction budget 16 (QA #7 showed
   66-77% hits at this size; per-slot tracking removes the waits that made
   16 candidates a regression).

**Safety invariants.** GPU reads only slots that are sem-waited+OK (kick) or
synchronously read (misses). Kick writes and consume miss-writes never share
slots. Buffer reuse is guarded by the retire-wait. A short/failed pread
signals its sem with OK=false and the consumer falls back to a duplicate sync
read, so garbage slots are never advertised.

**Expected effect.** Hit rate toward 55-70% (more candidates + ordered
histories) without the QA #11 wait penalty; IO block down from ~120-160ms
toward ~80-110ms/token. Risk watch: the retire-wait tail (kick ~50MB ≈ 10ms
vs ~4-9ms elapsed per 2 layers) — overlapped with MoE GPU (~1-1.7ms) but may
net-stall if SSD bandwidth drops; tune budget down to 12 if per-layer IO
inflates.

### QA #12 — Fix #8 (per-slot tracking) results: IO fixed, GPU inflates

| | QA #10 (baseline) | QA #12 (now) |
|---|---|---|
| tokens | 251-305ms | 300-436ms |
| IO | 119-160ms | **93-132ms** ✓ |
| hit rate | 42-56% | 44-59% ✓ |
| RouterGPU | 67-85ms | **89-149ms** ✗ |
| MoEGPU | 39-41ms | **46-65ms** ✗ |
| gqa PhaseA | 2.7-2.9ms | **5.1-8.9ms** ✗ |

Per-slot consumption works exactly as designed (no full-kick waits; IO down
~30%; hit rate up ~5-10pts). But 16-candidate kicks (50MB/layer, ~2.0GB/token
of total SSD reads) saturate the disk (only ~5.6GB/s) and correlate 1:1 with
GPU-phase inflation: concurrent 8-thread pread bursts during attention phases
coincide with RouterGPU wall-time doubling (driver submission starvation /
memory-controller contention). MoEGPU inflation includes the retire-wait
tails (~+0.5ms/layer, bounded ✓).

**Action: keep the per-slot machinery, shrink the kick budget to the SSD
window: 16 -> 10 candidates** (31.5MB ≈ 5.6ms ≈ the 2-layer overlap window).
SSD load per token drops from ~2.4GB to ~1.7GB; expected hit rate ~50-58%,
tokens back toward ~260-290ms. The 77% hollow-hit-rate lesson: candidate
counts beyond the window's read capacity buy late hits, not speed.

### QA #13 — Fix #8 + budget 10: baseline restored, decode lane at diminishing returns

Token 850 (best): **256ms** | RouterGPU=80.4ms ✓ (baseline) | IO=103.2ms ✓ |
MoEGPU=40.4ms ✓ | 40% hits — the per-slot machinery is clean, GPU inflation
gone. Token 849: 349ms @ 29% (topic drift); token 851: 377ms @ **2%** — the
whole token's prediction collapsed (all layers 0/8-1/8): a mid-generation
topic switch (code block) invalidates every temporal-history predictor at
once. That failure mode is fundamental to history-based routing prediction,
not a fixable bug.

**State of the decode hit-rate lane:** hits bounded by temporal routing
stability (~40-60% fluent text, collapses on topic switches); the remaining
structural levers are (a) component-level expert streaming (breaks the
~5 tok/s SSD floor, big project), (b) fused sampler/logits for the ~30ms
"Other" block (~10-15ms recoverable), or (c) return to TTFT (GDN kernel
threading, ~2-2.5s of TTFT — the clearest remaining win).

---

## FIX #9: WEIGHT-STATIONARY BATCHED GEMV FOR PREFILL (SHIPPED)

**Measurement (T18).** The GDN recurrence kernels are NOT the prefill whale:
seq kernel = 68us/token (48 TGs x 32 lanes); a 128-lane re-thread gained
nothing (0.9-1.5x). The real prefill cost is the **per-token weight re-reads
in the GEMV kernels**: bf16_gemv_simd traffic = P x weights (measured 3.2ms @
P=1 -> 305ms @ P=1024, linear in P; ~170 GB/s effective). Every prefill layer
runs ~4-10 GEMVs (in_projQKV 41.9MB, in_proj_z 25MB, out_proj 25MB, q/k/v/o
...), so prefill re-reads ~9GB of weights per token for P~100.

**Shipped kernel: `bf16_gemv_batched`** — one threadgroup per (output row,
16-token block); ushort4 chunked weight loads (coalesced, 8 bf16/lane), 16
token activations per chunk iteration, 16 simd_sum reductions. Weight traffic
= ceil(P/16) x weights. Verified in T18b: identical to bf16_gemv_simd
(rel <= 4e-4, float32 reordering), 1.6-2.3x faster at P=64..1024, slower at
P=1 -> wired only for batchSize > 1; decode keeps bf16_gemv_simd.

Wired in dispatchLinear's BF16 branch (before the simd branch). Covers
Ornith's linear_attn in_projQKV/Z/A/B + out_proj and lm_head batched paths;
Spark 2.5 (all-BF16 backbone) prefill benefits too.

**Expected.** Ornith prefill GDN layers ~60ms of GEMV -> ~35ms; PhaseA 5.5s ->
~3.5-4s; TTFT ~11.5s -> ~9-9.5s. The attention layers' projections are
F8_E4M3 (fp8_gemv_simd, likely the same per-token re-read) — an
`fp8_gemv_batched` clone is the follow-up if the PREFILL-DIAG confirms.
Summary now prints P= and startPos= to confirm the actual prefill size.

---

## FIX #10: KV-PREFIX REUSE FOR HYBRID (GDN) MODELS (SHIPPED)

**QA #14 (tools-on turn) exposed the TTFT multiplier:** P=1928,
**startPos=0** — the app re-prefilled the ENTIRE conversation, and TTFT=390s
vs prefillTotal=89s showed the web-search tool loop re-prefilled the full
context after every tool result (each ~60-90s at P~1900-2900).

Root cause (2 stacked bugs):
1. `let prefixTokensReused = hasRecurrence ? 0 : findCommonPrefix(...)` —
   prefix reuse was FORCE-DISABLED for any model with linear-attention layers
   (Ornith: 30 GDN layers).
2. `KVCacheManager.reset` zeroed linearStateBuffer + convStateBuffer on every
   reset for recurrence models — so even a reused prefix would have run the
   delta prefill with EMPTY recurrent states (the reason the disable existed).

**Shipped mechanism:**
- `KVCacheManager.captureLinearStates()` — memcpy the live GDN + conv states
  into prefix snapshot buffers. Called at the exact moment `recordTurn` pins
  the token list (turn end), so the snapshot corresponds to
  `prompt + generated` by construction.
- `reset(...)` restores the snapshots when `preservePrefixCount > 0 &&
  linearStatesPinned` (instead of zeroing); zeroing remains for fresh or
  partially-matched prefixes; `linearStatesPinned` clears on fresh resets.
- ContentView reuse gate for recurrence models: reuse only when
  `findCommonPrefix == currentPinnedCount` (FULL match — a partial match would
  need states from mid-recurrence, which we cannot reconstruct).

**Expected effect.** With the full conversation matching: turn-2+ prefill
prefills only the delta (~50-1000 tokens for tool results) instead of
~1900-2900. The web-search loop's TTFT drops from ~6.5min to ~1-2min
(4x re-prefills of ~2900 tokens -> deltas); normal turns from ~89s to
~3-5s. The MoE gpuWait (32.9s at P=1928) still re-runs per re-prefill and
remains the next prefill lever (batched-over-tokens MoE).

Also added a load-time warning if bf16_gemv_batched is unavailable from the
metallib (Fix #9 did not engage in QA #14's build — GDN layers measured at
the old per-token GEMV rate; verify the .metal file recompiled).

### QA #15 — Fix #10 live results: THE PREFIX REUSE WIN

- Tools-off follow-up turn: prefill **2-3s** (was ~89s). Fix #10 confirmed.
- Tools-on web-search turn: re-prefill after the first search was
  `P=868 startPos=2154`, hitRate=87%, ioWait=1.06s — the tool loop now
  prefills tool-result deltas. TTFT 51-95s (was 390s).
- GDN prefill layers still measure the OLD per-token GEMV rate
  (475ms @ P=868 = 169 GB/s; batched would be ~35ms) — **Fix #9 still not
  engaging in the user's build**. Added affirmative load print
  ("⚡ [GEMV] bf16_gemv_batched pipeline ready") to diagnose; if absent, the
  metallib didn't recompile (clean build needed).
- gqa prefill layers ~1.7s each at P=868: FP8 attention GEMVs likely dominate
  -> `fp8_gemv_batched` clone queued next.
- Fixed the adaptive-gate warning spam (printed every layer once tripped;
  now prints once at the flip). Gate behavior correct: after a tool-result
  prefill the routing is genuinely novel (0-4% hits), kicks are disabled and
  misses read sync — the right trade for unpredictable content.

### FIX #9b: FP8 BATCHED GEMV (SHIPPED, VERIFIED)

`fp8_gemv_batched` cloned from bf16_gemv_batched: uchar4 weight chunk loads +
unpack_e4m3 dequant once per chunk + 16 token activations per iteration +
per-row bf16 scale applied at the final sum. One threadgroup per (output,
16-token block). T18c verification: identical to fp8_gemv_simd
(rel <= 8.6e-06 — the e4m3 dequant is deterministic, only float32 reordering),
1.7x faster at P=64 and P=1024; P=1 stays on fp8_gemv_simd.

Wired in dispatchLinear's FP8 branch (before the simd branch, batchSize > 1).
Covers Ornith's self_attn q/k/v/o projections (F8_E4M3) during prefill — the
gqa prefill layers measured ~1.7s each at P=868 in QA #15, dominated by the
per-token FP8 GEMV re-reads. Expected: those drop toward ~0.3-0.5s, cutting
delta-prefill totals by roughly a third. Load prints confirm which pipelines
are live. Note: the bench's experiment copies of the batched kernels were
removed once they became production kernels in ComputeShaders.metal (double
definition otherwise).

### QA #16 — batched GEMVs live; NEW REGRESSION: tool-loop truncation

Both ⚡ [GEMV] pipeline prints confirmed. Delta prefill improved (P=616:
35.16s total, PhaseA=20.9s, gpuWait=10.8s — was 50.8s at P=868). BUT:
- GDN prefill layers still ~0.54ms/token — the batched kernels measure only
  ~17GB/s effective in T18b (activation L2 re-reads dominate), so the
  in-app win is modest, not the traffic-model's 10x. A tiled GEMM is the
  real fix if we continue this lane.
- Decode speed down (2.8 tok/s at ~2.8k ctx) — partly the adaptive gate
  correctly disabling prefetch on novel post-search content (IO 130-190ms).
- **NEW BUG: agent tool loop truncates** — model does one tool call, intends
  another, stops (85-token turn). Reproduced 2x. Prime suspect: the Fix #10
  GDN-state restore desyncing the recurrence state vs the pinned token list
  (turn 1 without restore is fine; truncated turns all follow a restore).

Shipped diagnostics: restore self-check (memcmp live-vs-snapshot at reset;
a nonzero diff means the state ran ahead of the pin between turns),
pin-length log at capture (📌 [PREFIX] pinned N tokens), and an A/B toggle:
`defaults write DRP.DynaMoE dynamoe_disable_prefix_reuse -bool YES`.

### QA #17 — logs A/B analysis

- Self-check: ZERO GDN-state divergence in both runs — the snapshot/restore
  is byte-consistent; no smoking gun for the truncation in the state path.
- Pins grow exactly right: A 2044->2795->3609; B 2067->2402->3161->6515.
- Turns reuse correctly (startPos=2044/2067/2402 with 79-89% hit rates,
  ioWait ~1s). Prefills now: P=698-721 in ~40s (batched GEMV + pipelining).
- **B turn 4: startPos=0, P=5540, 286s** — the full-pin gate fell back. The
  final synthesis turn's prompt likely diverges from the raw generated
  tokens (thinking stripped / re-rendered) or passes a different sessionId;
  added a reuse-gate print (partial-match/no-common-prefix reasons) to
  identify which on the next run.
- A's truncation (92-token gen, then loop end): no diagnostic smoking gun;
  tokens healthy (285ms, 46% prefetch), attention ctx-consistent. Decisive
  test remains the dynamoe_disable_prefix_reuse toggle A/B (not yet run).

### QA #18 — E.txt analysis: the truncation chain fully explained

Break reasons logged: turn 1 tool-parser-freeze (normal, search call fired);
turns 2+3 **degenerate-cycle** (the model repeating `parameter>`-style
fragments while writing the curl shell_run call); turn 4 clean eos-token at
612 tokens (final answer completed).

Chain: model degenerates mid-tool-call -> cycle guard breaks the generation
-> parsed tool call is a fragment (unterminated quote) -> zsh exits != 0 ->
"Failed" curl runs -> the mangled assistant turn diverges from the pinned
tokens -> strict full-pin gate falls back to startPos=0 -> 134-142s full
re-prefills (the "slower last prefill").

Key exonerations: turn 3 degenerated with NO state restore (fresh full
prefill) -> Fix #10 restore NOT the cause (consistent with C/D A/B).
Grammar masks only tool/parameter NAMES (the curl value is unconstrained),
so not grammar-forced either. Remaining suspects: sampling loops on
structured output / the search-result echo content.

Shipped diagnostics: shell_run exit/stderr print (curl failures become
visible: DNS vs 4xx vs syntax); the cycle-break tail log extended to ~48
tokens/320 chars to see exactly what the model was repeating; A/B toggle
`defaults write DRP.DynaMoE dynamoe_disable_grammar -bool YES` (grammar only
constrains names, so this is a weaker suspect, but cheap to test).
Note: same URL succeeds from the shell on this machine (HTTP 200) -> the
app's curl failures stem from the mangled/parsed call, not the network.

### QA #19 — F.txt: ROOT CAUSE of the tool-loop truncation found and fixed

The extended 🛑 tail logs exposed the mechanism:
- Turn 1 froze normally on `<function=tools_load><parameter=names>["web_search"]</parameter>ieldquo` (tools_load fired, web_search loaded).
- Turn 2 (and E's turns 2-3) emitted `<function>` — an UNNAMED tool call —
  then looped on empty `<parameter>` tags until the degenerate-cycle guard
  broke the generation. Everything downstream (failed curls, prompt
  divergence, startPos=0 re-prefills) follows from that truncation.

Root cause: GrammarConstrainedSampler.applyLogitMask's `>` escape hatch
allowed any token starting with '>' to bypass the name-constraint EVEN when
the tool/parameter name was incomplete — so the model could legally emit
`<function>` with an empty name, then (in the unconstrained parameter-body
states) loop empty `<parameter>` tags forever.

Fix: the `>` escape is now allowed only when the current prefix IS a
complete registered tool name (function) or a complete parameter key
(parameter name). The model is forced to name the tool and each parameter
key; values remain unconstrained. This should eliminate the degenerate
tool-call truncations, the failed curl fragments, and the resulting
startPos=0 re-prefills.

### QA #20 — G.txt verifies the unnamed-call fix; tools.txt exposes a second loop; raw-token splice shipped

G.txt (7 generations, 6 tool calls): the unnamed `<function>` loop is gone.
All calls parsed and executed (shell_run exit=0, no curl failures), final
1171-token answer ended on clean EOS, zero GDN divergence warnings, pins
exact every turn. But turns 4/5/7 still looped `</parameter>` after a
successfully closed value (turn 5 exited via an EOS-family token mid-loop,
masking as a clean EOS). Cost: two degenerate-cycle exits left the turn text
ending mid-tag -> next prompt re-tokenized one token off the pin -> hybrid
gate failed -> 3.5s/4.7s full re-prefills (vs 30-400ms suffix prefills on
healthy turns).

tools.txt exposed the same disease one level up: the four "between-tags"
grammar states (enteringToolCall, enteringParameterTag) were unconstrained
(default: return). Observed: turn 1 emitted a malformed `<parameter(names>`
tag; turn 2 looped 21 consecutive empty tool-call open tags.

Fixes shipped:
1. Grammar between-tags masking: all four between-tag states now constrain
   continuation toward the legal tag(s), with prefix canonicalization at the
   last structural boundary so repeated close tags re-anchor instead of
   dead-ending into the unmask valve. Ling/Bailing native format is gated off
   via enforceStructuralTagContinuation (set from modelConfig at the call site).
   Verified: 27/27 logic assertions + full-project typecheck.
2. Raw-token splice for agent continuations: continuation turns now assemble
   the next prompt from the previous turn's ACTUAL prompt+generated token ids
   (buildSplicedContinuationTokens) plus a freshly tokenized suffix (end tag +
   tool-response/continuation text), instead of re-encoding the accumulated
   string. Re-tokenization could diverge by one token whenever generation was
   cut mid-tag; the splice makes the pin match byte-exact, so the hybrid
   full-pin gate passes and the suffix-only prefill runs (TTFT for
   continuation turns drops to the suffix length). Wired at all six
   continuation sites (tool response, uncalled-action, loop-guard synthesis,
   subagent relay, force-synthesis, empty-reply recovery); falls back to the
   string path if suffix tokenization fails. Verified: 13/13 property
   assertions (prefix continuity, pin full-match, endTag append semantics,
   fallback paths, character-sequence equivalence vs the string path).
   Residual: the empty-reply recovery site replays the bare prompt, which is
   shorter than the pin -> hybrid full re-prefill remains correct there.

### QA #21 — 1.txt: splice works end-to-end; prefill attention identified as the last big hot spot

First full run with the raw-token splice: all agent continuation turns
prefilled only their suffix — 42-44ms (143-147 tokens) vs 3.5-4.7s full
re-prefills before. Zero 🔁 full re-prefills across 6 tool calls; clean EOS
finish; run never quit mid-turn.

The one huge prefill (post first shell_run): PhaseA total 419.6s (~7 min),
with the 10 gqa/attention layers at ~30s each vs ~4s for MoE layers. The
splice math proves the suffix was 11348-4058 = 7290 tokens — the curl'd
GitHub HTML page (~29KB markup) injected as the tool result. MoE layers stay
linear (~0.55ms/token, same rate as turn 1); attention goes superlinear
(quadratic in context length) on a kernel path that is far off GPU peak
(~0.1 TFLOP of attention work per gqa layer taking ~30s => ~1-3% of peak).
Two levers:
1. Tool-result hygiene: strip/trim HTML in shell_run renders (or steer the
   model to raw.githubusercontent URLs) — would have cut this turn ~10x.
2. Batched multi-row attention prefill kernel (the reverted FIX #6a idea) is
   the remaining big TTFT win for large tool-result turns.

Also: a silent mid-run generation restart was identified (8 pipeline-ready
prints vs 7 [GEN] lines): user interrupt (stop + new message) => fresh
re-render whose system prompt embedded the post-tools_load tool list =>
diverged from the pin at token 24 => one full 3004-token re-prefill (~85s).
Inherent to interrupts (genuinely new prompt), but silent cancel-exits now
log 🛑 reason=cancelled with the tail text, and interruptAndSendMessage logs
⏹ [INT] so future logs tell the full story.

### QA #22 — shell_run HTML conversion shipped (tool-result hygiene, lever 1)

shell_run now detects HTML in stdout (doctype/html/body markers definitive;
otherwise >512B + >=12 tags + at least one HTML-vocabulary tag; XML
declaration opts out so config files stay intact) and converts pages to
plain text before truncation: script/style/head/noscript/svg/template blocks
and comments dropped wholesale, block-level closers become newlines, tags
strip to spaces, numeric+named entities decoded, boilerplate phrases/lines
removed, whitespace collapsed. Non-HTML output (JSON, XML, logs) passes
through untouched; conversion only engages above 2KB. shell_run's tool
description now steers the model toward raw text endpoints.

Verified with 30+ standalone assertions incl. two regressions caught during
development: (1) the tag-stripping alternation matched the `head` prefix
inside `<header>`, swallowing the whole page body until the next `</script>`
(fixed with a `(?=[\s/>])` name-boundary lookahead); (2) tag-density alone
flagged large XML configs as HTML (fixed with the vocabulary requirement).
Real-world-shape test: 17.7KB page (mostly inline JS/CSS) -> 72 chars (246x).
The 1.txt incident shape (7290-token suffix from one curl) would compress to
a ~50-token suffix, i.e. that ~7-minute attention-bound prefill becomes
sub-second. Remaining lever: batched multi-row prefill attention kernel.

### QA #23 — schema-aware required-parameter enforcement in the tool grammar

2.txt showed the between-tags masks working (4/4 breaks clean, zero cycles,
zero re-prefills, suffix-only prefills of 414/393/53ms) but exposed a
semantic hole: the mask legitimately allows `</function>` right after the
name (some tools take no params), so `<function=shell_run></function>` is
structurally legal yet carries no `command`. The harness guard
(hasEmptyRequiredArguments) caught it, skipped execution and the model
recovered — but the call was wasted.

Fix: registerTools now records each tool's `required` keys (ToolDefinition
already carried them), updateState tracks the live call context
(currentToolName + parameter keys already opened, scanned from the function
body), and the tag-choice masks withhold `</function>` while any required key
is missing — forcing the model to emit `<parameter=command>` before closing.
Optional params still omit freely; tools with no required keys and unknown
tools stay ungated (fail-open; the harness guard remains the backstop).
Verified: 19 new standalone assertions (gate matrix, body scanning,
end-to-end empty-call shape from 2.txt) plus the existing 27; full-project
typecheck clean.

### QA #24 — boilerplate removal restricted to navigation links (review follow-up)

An external review flagged that the unanchored boilerplate pass removed
"sign in" / "log in" / "advertisement" anywhere in a converted page, silently
mutating legitimate prose, commands, and code examples.

Compromise shipped: boilerplate is now defined structurally, not lexically.
`<a>...</a>` spans are bracketed with sentinels during conversion, and a noise
phrase is dropped only when it constitutes an ENTIRE link span (or an entire
line, as before). Prose sentences keep their words; a descriptive link like
"Sign in with your company SSO" is kept; real nav items ("Sign in", "Share",
"Skip to main content", "advertisement" as link text) still disappear, so the
original problem (nav items concatenated into one long line after tag
stripping) stays fixed. Sentinels never reach the output.

Verified: suite now 40+ assertions, incl. prose/code survival, nav removal,
partial-link retention, and sentinel-leak checks. Typecheck clean.
Alternative if strict review parity is preferred: delete the link-span pass
entirely and keep line-scoped removal only.

### QA #25 — required-param scan must skip parameter values (review follow-up)

Review flagged that parameterKeysTyped regex-scanned the whole function body,
so a literal `<parameter=command>` inside a VALUE (shell command echoing tool
markup, file_write content, a cwd string) counted as an opened required key
and opened the `</function>` gate early. Valid, though low severity: the
harness's hasEmptyRequiredArguments guard is the real backstop, so the worst
case is a wasted call + recovered error turn, not a wrong result.

Fix: parameterKeysTyped is now a small stateful walk — find `<parameter=`,
read the key to `>`, jump past the next `</parameter>`, repeat. Tag-like text
inside values is never scanned; unterminated values stop the walk, which
keeps the gate engaged (safe direction). This also matches parser semantics
(first `</parameter>` closes the value) and is stricter than the parser can
be, never looser.

Verified: new cases include a literal `<parameter=command>` in a cwd value
(not counted, gate holds), a real command after such a value (counted, gate
opens), an entire embedded tool-call block inside a content value (not
counted), unterminated values, and early-close semantics. Full suite passes;
typecheck clean.

### QA #26 — uncalled-action nudge false positive: "look into" in a closing pleasantry

Symptom: after the user sent "very cool, thanks!", the app produced TWO
assistant bubbles. The first was a normal close — "You're welcome! Let me know
if there's anything else you want to look into — happy to help." — then, after
a 22.7s "thinking" pause, a second bubble: "There isn't any outstanding action
or task to complete here…".

Cause: `detectUncalledActionIntent` matched actionPatterns unconditionally over
the finished text. The close contains the substring "look into", so the
detector returned true, `hasUncalledIntent` set `willContinueAgent`, and
`formatActionContinuationTurn()` was appended — a synthetic `<|im_start|>user`
turn asking the model to execute an action if it meant one. The model then
*answered the harness directive* instead of the user, which is why the second
bubble reads like a meta-reply.

Cost from the log: the close pinned 3369 tokens (prompt=3318 gen=51); the
synthetic turn spliced to 3440 raw tokens (P=70, startPos=3369,
`prefillTotal=8.30s`, PhaseA=3.55s at a 50% expert hit rate) plus 22.7s of
thinking and 119 new tokens, before pinning again at 3559. Well over a minute
of work to answer "thanks!". The splice itself worked correctly end-to-end on
this path (a live verification of splice site 6).

Fix, in `AgentHarness.detectUncalledActionIntent`: three narrow signals replace
the broad pattern list.

1. `commitmentPhrases` — explicit first-person starts ("i'll start by",
   "let me start by", "first, let me", "i'm going to start", …) matched
   anywhere in content or thinking: always an intent.
2. `closingPleasantryRegex` — wrap-up language ("let me know", "happy to
   help", "anything else", "you're welcome", …) **vetoes** detection. A
   pleasantry is never a promise to act even when it borrows action words.
3. `commitmentActionRegex` — a first-person cue ("i'll", "let me" but not
   "let me know", "i should", "i need to", …) followed within 30 characters,
   in the same clause, by action vocabulary. Ordinary prose ("you can check
   the docs", "the ranges look at combined cycle numbers") no longer fires.

Curly apostrophes (U+2019) are normalized so "I’ll start by" matches. The
existing >400-character bail is preserved and runs first.

Verified: 20-case standalone suite (`/tmp/detect_test.swift`) — 9 positives
including thinking-only cues and the curly-apostrophe form; 8 negatives
including the exact regression string, "happy to help — let me know if you'd
like me to look into the charging curve", bare action nouns, and second-person
advice; the >400-char bail; and the documented ordering rule that an explicit
commitment phrase still wins over a pleasantry in the same message
("I'll start by reading the spec, and I'll let you know what I find"). All
pass. Full-project typecheck clean.

Follow-up in the same pass: the nudge itself was framed as a synthetic
`<|im_start|>user` turn, which is *why* a fired continuation reads as a second
assistant bubble answering a meta-question. Both formatters now emit a system
directive — `<|im_start|>system` for ChatML models and `<role>SYSTEM</role>`
for Ling — matching the framing the app already uses for its system prompt
(`ContentView.swift:1706`, `:3546`, `:3513`). This is a behavior change, so
treat any change in recovery rates as the thing to watch; the detector fix
means the path now only runs on genuine narrated intents.

Review follow-up (Copilot, High): `testLingNativePromptAndToolResponseFormatting`
still asserted `continuation.contains("<role>HUMAN</role>")` at
`DynaMoETests.swift:6505`, so the role change would have failed the suite.
Valid catch. Updated to assert `<role>SYSTEM</role>` and added an explicit
`XCTAssertFalse(... "<role>HUMAN</role>")` so the intent is locked. The ChatML
formatter had no coverage at all, so `testAgentMultiTurnToolCallParsingAndContinuation`
now also asserts `formatActionContinuationTurn` emits `<|im_start|>system` with
no `<|im_start|>user`. `formatToolResponseTurn` was deliberately not changed —
tool results stay user-role — and its assertion is untouched. Both tests are
pure formatter/parser checks with no model dependency, so they run on any host.

Review follow-up (Copilot, High): the new assertion
`continuationTurn.hasSuffix("<|im_start|>assistant\n thinking")` could never
pass. Correct, and the cause is worth recording because it is a tooling trap,
not a logic slip. `formatActionContinuationTurn` appends the literal
`" thinking\n"` (real angle brackets), so the turn ends with the tag plus a
trailing newline. The assertion as written expected `\n`, a space, and the bare
word `thinking`. Verified at byte level: the needle's codepoints were
`0x5c 0x6e 0x20 0x74 0x68 0x69 0x6e 0x6b 0x69 0x6e 0x67` (backslash, n, space,
`thinking`) where every neighbouring assertion uses
`0x3c 0x74 0x68 0x69 0x6e 0x6b 0x3e` (` thinking`) — including the assertion two
lines below it that I did not write.

Root cause: the literal was copied from a rendered file view, and the rendering
pipeline strips `<`/`>` from unrecognized tags, so ` thinking` displays as
` thinking`. The same sanitizing affects comments and prose, which is harmless,
but makes any tag-bearing *code or test literal* untrustworthy when read
through a rendering. Every tag literal written in this branch was re-audited at
byte level: all production tags (`<|im_start|>system`, `<|im_start|>user`,
`<|role_end|>`, `<role>SYSTEM</role>`, `<role>OBSERVATION</role>`) and the
tool-call template embedded in the continuation directive (`<tool_call>`,
`<function=`, `</function>`, `<parameter=`, `</parameter>` — one each) are
intact. The test was the only casualty.

Also checked the neighbouring assertions that share this shape: the Ling
`hasSuffix("<role>ASSISTANT</role>\n thinking")` is correct because the Ling
formatter ends with `"\n thinking"` and no trailing newline, so it was left
alone. Added a note that the trailing newline is part of the ChatML
`" thinking\n"` handoff. Confirmed by simulating both forms against the real
tail: the corrected assertion returns true, the original returns false.

Review follow-up (Copilot, Medium): the pleasantry veto ran after the
commitment-phrase check but *before* the cue+action check, so any turn that
paired a pleasantry with a real narrated action was suppressed — "Happy to help
— I'll read the config now." returned false. That is a regression I introduced,
and it is the expensive direction: a false negative drops a narrated action the
model never executed, which is the exact failure the detector exists to catch.
Before the rewrite the old unconditional pattern list would have matched "read
the" and fired.

Fix: the cue+action check now runs before any wrap-up consideration, and the
pleasantry veto is deleted rather than reordered. Reordering alone would have
left the veto unreachable — once the cue+action test fails the answer is false
either way — so keeping it would have been dead code. The veto also turned out
to be unnecessary: the original regression is already fenced off because its
action words ("look into") have no first-person cue in front of them, since
"let me know" is excluded as a cue. Confirmed by test: the exact regression
string, "Let me know if you want me to look into it", and "The EPA ranges look
at combined cycle numbers" all still return false without any veto.

Documented cost: a conditional offer ("I'll look into it if you want") now
counts as an intent, because the cue+action pair is genuinely present. That is
one cheap extra turn, and with the system framing from earlier in this entry it
reads as a resumed action rather than a phantom user question — a better trade
than the false negatives the blanket veto caused. Recorded in the code comment
on the check so the next reviewer sees it was deliberate.

Verified: standalone suite now 25 cases including four pleasantry-plus-action
positives and the conditional-offer cost; added
`testUncalledActionIntentDetection` to the XCTest suite so the detector has
real coverage (pure logic, no model dependency, runs on any host). App
typecheck clean under the project's real flags.

### QA #27 — grammar masking state is shared across generations (review follow-up)

Review (Copilot, High): the live-context fields added for the required-parameter
gate (`currentToolName`, `seenParameterKeys`) are shared through
`GrammarConstrainedSampler.shared`, mutated from the detached generation task
and reset on the main actor with no synchronization.

Verified, and worse than the comment implies in one respect. The app target sets
`SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so the sampler was implicitly
`@MainActor` while the generation loop calls it from
`Task.detached` (`ContentView.swift:4442`) without `await`. That is an unchecked
isolation violation that Swift 5 mode only warns about, so nothing enforced
mutual exclusion at runtime — the state machine really was unsynchronized.

A second, purely logical hazard sits on top: cancellation gives a previous
generation up to one more token iteration to unwind, and a token takes ~350ms on
the reference machine while the requeue path waits only ~80ms. During that
window the old task keeps calling `updateState` with its own accumulated text,
so the new generation's state could be overwritten by its predecessor.

Fix:

- `nonisolated` on `GrammarConstrainedSampler` and `TokenTrieNode` — states the
  real threading contract instead of claiming main-actor isolation that was
  never honoured. This is what the compiler was trying to tell us.
- `stateLock` (`NSRecursiveLock`) guards every mutable grammar field;
  `currentState`, `isEnabled` and `enforceStructuralTagContinuation` are now
  lock-backed accessors, so off-actor reads and writes are synchronized.
- `beginGeneration()` clears the context and issues a monotonically increasing
  token. `updateStateAndApplyLogitMask(..., token:)` advances the state machine
  and applies the mask under a single lock hold, and drops any write bearing a
  stale token. A cancelled generation can no longer clobber its successor, and
  state + mask can no longer be interleaved by another task.
- `applyLogitMaskLocked` also holds `registrationLock` for its duration:
  `registerTools` rebuilds the schema in place (removeAll + refill), and a torn
  read mid-rebuild silently dropped the tool-name mask for a token. Lock order
  is always stateLock then registrationLock; `registerTools` never takes
  stateLock, so the ordering cannot invert.

Blast radius is unchanged from QA #23/#25: this is prevention, and the harness's
`hasEmptyRequiredArguments` guard remains the backstop, so the worst case was
always a wasted call plus a recovered error turn, not a wrong result.

Verified: new test `testGrammarGenerationTokenIsolatesStaleWriters` drives two
generations through the shared sampler and asserts the second generation's state
survives a late write carrying the first generation's token, that
`beginGeneration` issues distinct tokens, and that it clears the context. App
typecheck clean under the project's real flags; new `nonisolated` shape probed
in isolation to confirm it produces no actor-isolation diagnostics.

Tooling caveat found while verifying this: `xcrun swiftc -typecheck` on the app
globs only reports the `no such module 'Sparkle'` error and then stops, so it
suppresses every warning in every other file — including exactly the isolation
warnings this review is about. Pass the project's flags
(`-swift-version 5 -default-isolation MainActor`) and treat a clean CLI
typecheck as "no errors", not "no warnings".

### QA #28 — a cancelled generation ran the whole finalization path (review follow-up)

Review (Copilot, High): cancellation only breaks the token loop. The detached
task then continues through finalization, which updates shared UI state,
captures/records the prefix and can schedule another agent turn. Since
`interruptAndSendMessage` starts the replacement after only 60ms, the cancelled
task can overwrite the replacement's `generationTask`/prefix cache and race its
shared KV buffers.

Confirmed, and it is reachable. `stopAutoregressiveGeneration()` (the
Stop/interrupt path, and the only caller that owns the UI cleanup) sets
`isGeneratingText = false`, cancels and nils `generationTask`. The replacement
then starts 60-80ms later and assigns a fresh `generationTask`. The cancelled
task reaches its finalization block afterwards — up to one token later, ~350ms
on the reference machine — and that block sets `isGeneratingText = false` and
`generationTask = nil` again, silently **disarming the replacement**: the Stop
button and any later interrupt would no longer be able to cancel the running
generation. The same block also runs

- `KVCacheManager.shared.captureLinearStates()` and
  `PrefixCacheManager.shared.recordTurn(...)`, pinning a truncated turn into the
  prefix cache while the replacement is prefilling against the same KV cache, and
- `dequeueAndRunNextPromptIfNeeded(sessionId:)`, which can start yet another
  generation on top of the replacement.

Fix, one early return plus one generation identity:

- The detached task now returns immediately when `Task.isCancelled`, before the
  finalization block. Nothing is lost: the partial reply is already committed to
  the message by the per-token streaming updates, and the caller has already set
  the status line. Logs `⏹ [GEN] cancelled at N tokens — skipping finalization`.
- Cancellation alone is not a sufficient licence to mutate shared generation
  state, because `interruptAndSendMessage` supersedes the task. Added
  `@State generationId`, bumped once per `startAutoregressiveGeneration` and
  captured by that task, plus `ownsGeneration(_:)`. Every other place that
  cleared `generationTask`/`isGeneratingText` on cancellation
  (`ContentView.swift` prefill-failure guard, tool-execution guard, and the two
  agent-continuation guards) now returns early unless it still owns the
  generation. Those sites all shared the same defect; guarding only the flagged
  line would have left three live instances.

Deliberately NOT done here: awaiting the previous task before starting the
replacement. Written at the time on the belief that a cancelled task sitting
inside `runLayerWisePrefill` does not observe cancellation until it returns to
the token loop. **That belief was wrong** — corrected and acted on in QA #30.

Verified: app typecheck clean under the project's real flags. Behaviour is
observable in the run log; the user should see `⏹ [INT]` followed by
`⏹ [GEN] cancelled at N tokens — skipping finalization` and then only the
replacement's own generation.

### QA #29 — a superseded generation sampled from unmasked logits (review follow-up)

Review (Copilot, High): returning early from `updateStateAndApplyLogitMask` on a
stale token silently treats the stale generation as if masking had succeeded —
the logits are left unmasked and `sampleNextToken` gets no signal to abort, so
the cancelled task can sample one unmasked token and still reach the later
finalization/prefix-recording path.

The unmasked-draw half is correct and worth fixing: the staleness guard exists to
protect the *shared* state machine, but it also silently drops the mask, and the
sampling loop had no way to tell the difference between "masked" and "mask not
applied". The loop only re-checked cancellation at the top of each iteration, so
a cancel arriving during a token forward (a ~350ms window, and the mask is
dropped for the whole remainder of that iteration) let the task draw a token
from raw logits and commit it into its message and the stream buffer.

The "still reach the finalization/prefix-recording path" half was already closed
by QA #28, which returns a cancelled task before finalization. That also means
this is narrower than the review suggests: the stray token lands only in the
task's own superseded message, and the task returns before tool parsing, so it
can neither be executed as a tool call nor recorded as a prefix.

Fix, making staleness visible rather than silent:

- `GrammarConstrainedSampler.isCurrent(_:)` — the generation token is now
  queryable, so a caller can tell that its mask was dropped.
- The decode loop stops before sampling when the task is cancelled *or* the
  token is no longer current, logging
  `⏹ [GEN] superseded at N tokens — stopping before sampling`. Checking before
  the draw rather than after the commit means no unmasked token is drawn at all.
- The finalization block is additionally gated on `ownsGeneration`, so a
  superseded-without-cancelled task cannot touch the UI flags, the task handle,
  the prefix cache, or schedule another turn. `Task.isCancelled` was the only
  check there before; ownership is the stronger and more precise condition.

Deliberately not done: propagating the stale result through the mask closure as
a Bool. The closure is a `(UnsafeMutablePointer<Float>, Int) -> Void` hook shared
with `InferenceEngine.sampleNextToken`, which returns a bare `UInt32` with no
error channel, so that route needs either a sentinel value or an optional return
threaded through two signatures and the single call site. Querying staleness
where the loop can see it achieves the same outcome without inventing a sentinel
protocol. Revisit only if another caller of the mask hook appears.

Verified: app typecheck clean under the project's real flags;
`testGrammarGenerationTokenIsolatesStaleWriters` now also asserts
`isCurrent` is true for the live token and false for the superseded one.

### QA #30 — replacement generation reset shared KV buffers under a live task

Review (Copilot, High): the generation token isolates grammar/UI bookkeeping but
not the shared inference buffers. `KVCacheManager.shared.reset` immediately
before `beginGeneration` can nil, reallocate or copy the singleton KV/state
buffers while the cancelled detached task is still inside
`runLayerWisePrefill`/`runTokenForward`, and the replacement then races the old
task on the same Metal buffers. The 60ms interrupt delay is not a
synchronization barrier.

Confirmed, and pre-existing rather than introduced by the token change — the
reset call has always sat there, and the interrupt path has always started the
replacement on a fixed 60ms guess. Reading `KVCacheManager.reset` shows it is
worse than a plain race: with `preservePrefixCount == 0` it sets
`kCacheBuffer`/`vCacheBuffer` to nil and allocates fresh backing, so a task that
is still mid-layer does not merely touch freed memory, it dispatches against
whatever the singleton now points at — the replacement's brand new KV cache. The
`memcpy` prefix-preservation branch additionally copies from the old buffers
while the old task writes to them.

This is the residual that QA #28 deferred, and the deferral rested on a wrong
fact. This journal claimed a cancelled task inside `runLayerWisePrefill` does not
observe cancellation until it returns to the token loop, and cited a 419.6s
prefill as the cost of waiting. Both layer loops actually check cancellation at
every layer boundary (`ContentView.swift:5104` decode, `:7400` prefill, plus
`:9328` on the JetSpec path), so a cancelled task stops dispatching after the
current layer. The 419.6s figure was the whole 40-layer prefill, not the
remaining work. Corrected here.

Fix: the interrupt path now waits for the previous generation to actually finish
instead of sleeping for 60ms — `let previous = generationTask` captured before
`stopAutoregressiveGeneration()` nils it, then `await previous?.value` before
`handleSendMessage`. Because the layer loops bail at the next boundary this is
bounded by one layer, which is one token during decode (the common interrupt)
and the worst case is a single layer of a very large prefill. The await
suspends rather than blocking the main actor, so the old task's own
`await MainActor.run` hops still complete.

Audited the other paths that reach the same reset, and they are already
correctly ordered, so no identity machinery was needed there:
`dequeueAndRunNextPromptIfNeeded` and the agent-continuation sites are all called
from inside the outgoing task's finalization, which is past its buffer work, and
the replacement starts on a later main-actor hop. Sends from the UI cannot start
a concurrent generation either: `ChatDetailView` routes Enter to the queue and
Cmd+Enter to `interruptAndSendMessage` while generating, and the send button is
only built in the `else` branch of its `isGenerating` check
(`ChatDetailView.swift:738`), so `handleSendMessage` is unreachable mid-run. No
defensive guard was added to it — a synchronous guard could not fix the race
anyway, since only an await can postpone the reset.

Not addressed: a Metal dispatch cannot be preempted, so the wait floor is one
layer. During a huge-prefill interrupt that is seconds rather than milliseconds.
That is the right trade against silently corrupting the KV cache, but it is a
real responsiveness cost and the reason to keep an eye on interrupt latency in
large-prefill runs.

Verified: app typecheck clean under the project's real flags. Watch for
`⏹ [INT] interrupt requested` followed by the replacement's own
`⚡ [GEMV]` pipeline-ready pair only after the previous generation's
`⏹ [GEN] ended ... reason=cancelled` line.

### QA #31 — 404-era log + prompt dumps: prefix reuse exonerated, the "echo done" gesture found

Shipped a prompt-dump diagnostic (`dynamoe_dump_prompts` or a marker file at
`~/Downloads/DynaMoe-dump-prompts`, written to `~/Downloads/DynaMoePromptDumps/`)
that records the exact decoded token stream each turn attends over, plus a
splice-vs-reencode equivalence verdict. Two instructive things about enabling it:
the app is sandboxed, so prefs live in
`~/Library/Containers/DRP.DynaMoE/.../Preferences/DRP.DynaMoE.plist`, and
`cfprefsd` can keep serving a stale cached value for a running app — the marker
file bypasses both. (The bundle id is `DRP.DynaMoE`, not `com.drp.DynaMoE`;
earlier docs entries had it wrong and the A/B toggles were therefore never
reachable.)

What the dumps show across a clean 5-turn run:

- `prefixReused == pin` on every continuation, suffix-only prefills, no
  `🔁` full re-prefills, and the splice-vs-reencode check diverges by exactly one
  token at the first turn boundary (token 2081) and never structurally. That is
  the known-benign boundary case the splice exists to avoid — prefix reuse is
  healthy, pin math is exact, and nothing is duplicated or dropped in context.
- The WEB RESULTS themselves were fine (MSN/Fortune/real articles, correct URLs),
  the search loop guard fired correctly when the model re-searched the same query,
  and the model's final answer was accurate and well-reasoned.

The real defect was turn management: after delivering its final answer the model
emitted a `<tool_call><function=shell_run><parameter=command>echo done`, the
harness executed it, and the agent loop therefore continued — producing a second
bubble that re-summarized the same answer. A no-op shell command cannot advance
any task; the model emits one as a "task finished" gesture because the system
prompt told it when it MUST call tools but never how to end a turn (`complete`
exists but nothing points at it).

Fixes:
1. Prompt: a new "# Ending Your Turn" section — plain text ends a turn; use
   `complete` (with summary) only when the whole task is done; never emit
   placeholder/no-op tool calls; one final answer per task. Also fixed a literal
   concatenation bug that glued the format-guidance paragraph onto the end of the
   preceding one ("...because of them.If you choose to call a function ONLY...").
2. Harness backstop: `AgentHarness.isNoOpGestureToolCall` drops gesture calls
   (`shell_run` with a bare `echo`/`echo <short>`/`true`/`:`/`exit`, no
   redirection/pipe/chaining/substitution) before the continue decision and before
   execution, so the turn ends instead of looping.

Verified: 17-case unit test (gesture shapes vs real commands incl. `echo done >
file`, `echo done && ls`, `echo $HOME`), plus the existing grammar/agent suites.

### QA #32 — prompt dumps expose the freeze's dirty context: unterminated calls → format drift

The prompt dumps from a 9-step Ornith agent run show the structural defect that
QA #31's turn-management fix did not address. Counting tags in a mid-run dump:
9 `<tool_call>` openers, 2 closers, 11 `<parameter=` openers, 9 closers.

The freeze fires the moment `</function>` closes (the fallback branch in
`shouldFreezeGeneration`), which is the common case — so 7 of 9 calls were
committed to the pinned context with NO `</tool_call>` closer. The model then
reads back its own history as a stack of structurally invalid examples of the
format it is being asked to follow. Within a few turns it drifted:

    <tool_call>
    <function=web_fetch>
    <parameter=url>
    <parameter=url>
    https://www.thdrive.com
    </parameter>
    </function>

— the parameter tag re-opened inside its own value, with a bogus URL. The harness
parsed the value as literal markup, `web_fetch` rejected it ("Invalid URL"), the
model repeated the identical malformed call on the next turn, and the run
spiraled. Prefix reuse was healthy throughout (`prefixReused == pin` every
continuation; splice-vs-reencode diverges only by the known-benign boundary
tokens, growing ~2/turn). This is context hygiene, not KV or prefill.

Fixes:
1. `StreamingToolParser.hasUnclosedToolCallBlock` + `closedAssistantTurnText`:
   every continuation now closes an unterminated `<tool_call>` block before the
   turn is spliced into the next prompt (token path and string path both, so the
   fallback agrees). The model only ever reads back valid examples of its format.
2. Parser tolerance: `AgentHarness.unwrapNestedParameterTag` descends into a
   re-opened `<parameter=…>` inside a value, so the observed shape resolves to
   the inner payload instead of poisoning the tool call.
3. `web_fetch`'s invalid-URL error now echoes the offending value and points at
   the `url` field from the search results, instead of a bare "Invalid URL" the
   model can only respond to by repeating itself.
4. Tests: freeze/no-closer detection (incl. closed-block-then-prose), the exact
   malformed shape from the dump, and the well-formed control.

Note for agent runs: the Ornith "Precise" profile ships repetitionPenalty 1.00 and
presencePenalty 0.00, while the model's own Agentic-Loop profile uses presence
1.50. Verbatim repeats (this run's duplicated malformed call) are what presence
penalty targets; worth an A/B on long agent runs.

### QA #33 — the closure fix regressed the token path; the unwrap fix landed in the wrong parser

Two self-inflicted bugs, both caught by the prompt dumps from the next run.

1. `buildSplicedContinuationTokens` was fed the ALREADY-normalized turn text. Its
   "is the block unclosed?" test therefore read the closer the caller had just
   added and skipped appending it to the token stream — and the end-tag test read
   the appended `<|im_end|>` and skipped that too. Result: every continuation in
   that run spliced turns with NO `</tool_call>` and NO `<|im_end|>`, i.e. the
   model's own turn boundary was invisible in context. The dump shows it directly:
   the splice window reads `</function><|im_start|>user` while the re-encoded
   window reads `</function></tool_call><|im_end|><|im_start|>user`.
   Fix: the token path always takes the RAW decoded turn; the closure decision
   lives in one shared `StreamingToolParser.turnClosureSuffix`, and
   `closedAssistantTurnText` mirrors it for the string path. Closure is now
   count-based (`unclosedToolCallCount`) so a turn that froze two calls closes
   both.

2. `unwrapNestedParameterTag` was added to `AgentHarness.parseAllXMLFunctionCalls`,
   but the live agent loop parses with `StreamingToolParser.parseStreamingToolCalls`
   — so the fix never ran where it mattered. The dump shows the consequence in
   exact bytes: a `shell_run` whose value re-opened
   `<parameter=command>` executed that markup as a shell redirect
   (`zsh:1: no such file or directory: parameter=command`, exit 1), and the
   `web_fetch` twin returned "Invalid URL" — each repeated verbatim.
   Fix: one public implementation in `StreamingToolParser`, called by
   `decodeParameterValue` (the live dialect path) and by the AgentHarness parser.

Both are now pinned by tests that run the observed bytes through the LIVE parser
entry point, not just the AgentHarness one. Also observed in the same run and not
yet addressed: a `web_fetch` of a fabricated URL returned a soft-404 page whose
body was ~300 tokens of "Oops! Page not found" plus unrelated trending-story
headlines, which the model then had to read and reason about. Soft-404 detection
(compact error instead of the noise body) is the obvious next hygiene win.

### QA #34 — soft-404 pages no longer ship as content

Observed in the 15:xx dumps: a fabricated Fortune URL
(`fortune.com/2025/10/106932.html`) answered HTTP 200 with "Oops! Page not
found" plus a list of unrelated trending headlines. `web_fetch` returned that
body as success, so the model spent ~300 tokens reading nav chrome and pop
finance links, then kept constructing URLs.

Fix: `AgentHarness.softNotFoundSignal(title:cleanedContent:)`, checked in
`web_fetch` right after cleaning and before any content is assembled. Title
signals are precise ("404", "page not found", "page unavailable", "not found")
— bare "error" is deliberately excluded so "Error handling in Swift" stays real.
Body signals ("oops! page not found", "the page you are looking for",
"article not found", …) only count within the first 1500 chars of a page under
8000 chars, because a not-found shell is always small while a long article may
mention the phrase in passing. On match, `web_fetch` returns a compact error that
names the URL, the matched signal, and points the model back at the `url` field
from the search results instead of the page body.

Verified: 9 assertions covering the exact observed shape (site-name title, marker
buried behind nav chrome), several title/body variants, and three negatives
(long article with an incidental phrase, a real "Error handling in Swift"
article, and an ordinary news sentence).

### QA #35 — the format fixes hold; the loop moved to tool SELECTION (tools_load dialect)

The 16:22 run's dumps are structurally CLEAN — every dump has balanced
`<tool_call>`/`</tool_call>` (2/2 … 16/16) and `<|im_start|>`/`<|im_end|>`
off-by-one as expected for a live turn, and splice-vs-reencode diverges only by
the benign boundary tokens. So the closure, end-tag, and unwrap fixes did their
job; no format invention anywhere in 13 steps.

The new failure mode is tool choice, and the trigger is one rejected call:

    [tools_load] ERROR: Missing 'name' or 'names' parameter in tools_load.

The model wrote `{"web_search", "web_fetch"}` — JSON-set braces, not an array.
`tools_load` only accepted `[String]`/`String`, so the value decoded as a plain
string and the load was rejected. Consequence for the whole run: **web_fetch was
never loaded** (dumps show 0 `function=web_fetch` in all 13 steps), so the model
read every page with `curl` through `shell_run`, then hand-wrote
`grep -oE 'price[^ ]*"content[^ ]*' | ...` extractors, hit zsh quoting errors
("unmatched \""), retried the same shape, and degenerated into
`articles/articles/articles/...` inside a URL until the cycle guard cut it.

Fixes:
1. `AgentHarness.normalizeToolNameList` accepts the dialects models actually
   emit: `["a","b"]`, `[Any]` of strings, `{a, b}`, `a, b`, `a b`, `a`, with
   quotes/brackets attached. `tools_load` uses it for both `names` and `name`,
   de-duplicates, and its error now echoes what it received plus the array shape.
2. `shell_run`'s description no longer advertises fetching pages ("prefer raw
   text endpoints…") — that wording invited curl-based scraping. It now says:
   read pages with `web_fetch`; do not pipe curl into grep.
3. A zsh quoting/syntax failure now comes back with a hint naming the likely
   cause and the alternative (`web_fetch`), instead of a bare "unmatched \"".
4. `tools_load`'s schema description shows the literal array form.

Verified: 9 dialect cases plus an end-to-end `tools_load` test asserting the
braced form actually loads web_search+web_fetch and exposes them to the model.
Full test classes re-run against a stashed clean tree: identical failure sets
(9 pre-existing dogfood failures, 2 pre-existing DynaMoETests failures — none
introduced).

Review follow-up (Copilot, Medium on QA #34): the title heuristic matched a
substring, so a legitimate article titled "How to Fix a Page Not Found Error" or
"Why Was the Page Not Found?" was rejected as a soft 404 before its content was
ever read. Valid — the whole point of the check is to skip error shells, not to
veto articles that discuss the phrase.

Fix: titles are now split into separator-delimited segments (`|`, `:`, `·`, `–`,
`—`, or a spaced hyphen) and matched per segment. Strong markers ("page not
found", "404 not found", "page unavailable", …) must equal a whole segment;
their bare forms ("404", "error", "oops") only count when the title IS that
marker or every other segment looks like a site name (no spaces), so
"404: A Story of Loss" and "Troubleshooting 404 Responses in Express" fall
through to the body check and read normally, while "Example | 404" and
"Page Not Found - Example" still match. The standalone check caught the
"Example | 404" regression in the first cut of the fix.

Verified: the case list grew to 19 (help-article title, question title, prose
after a "404:" prefix, troubleshooting title as negatives; four title-segment
error forms plus five body-signal forms as positives) and all pass. Note for the
next session on this machine: an Xcode/macOS update here drops the Metal
toolchain component (`xcodebuild -downloadComponent MetalToolchain`) and leaves
the previously built test bundle unsigned (CodeSign: "code object is not signed
at all" in DynaMoETests.xctest) — a `clean` plus fresh build fixes that.

Review follow-up (Copilot, Medium on QA #32's backfill loop): breaking out of the
backfill loop on cancellation did not stop finalization — execution fell through
to `recordTurn` and could schedule the next agent step on top of the interrupt,
and a failed backfill forward would pin a prefix whose tail slots were never
written (exactly the hole the backfill exists to close).

Fix in `ContentView` after the loop:
- Cancellation (checked after the loop as well as inside it) returns immediately
  with `⏹ [GEN] cancelled during prefix backfill — skipping finalization`,
  matching the turn's earlier cancel guard, so no pin is recorded and no
  continuation is scheduled.
- A failed `runTokenForward` logs
  `⚠️ [GEN] prefix backfill forward failed at N/total — dropping the pinned prefix`
  and invalidates the session's pin (MainActor, behind the generation-ownership
  guard) before returning. The next turn full re-prefills instead of trusting
  slots that were never computed. The partial reply is already committed to the
  message by the streaming updates, so nothing user-visible is lost.

Review follow-up (Copilot, Medium on the gesture filter): filtering
`parsedResult.calls` only changed execution — `finalDecoded` (and, more
importantly, the token splice) still contained the dropped no-op call. In a turn
with both a gesture and a real call, the assistant turn carried two calls while
`toolResponseTurn` carried one result, leaving an unmatched call in the
transcript to invite a retry.

Note on the fix: removing the call from the assistant TEXT cannot work. The next
prompt is assembled from the raw generated token ids, so a dropped call is in the
transcript regardless of what the string says — the transcript has to gain a
matching result instead.

Fix: `AgentHarness.splitGestureCalls` returns the actionable call indices plus a
skip notice per dropped call (keyed by its index), and the agent loop keeps one
response slot per emitted call, stamping results into their transcript positions
and compacting at the end. Gesture calls get no UI chip and are never executed;
their slot carries
`[tool] success / skipped: true / reason: no-op gesture … do not re-issue it`.
Ordering is preserved for mixed turns, so call N always has result N.

Verified: 3 assertions on a [real, gesture, real] turn (indices [0, 2], one skip
notice at 1, a full 1:1 slot mapping after compaction) plus all-gesture and
no-gesture turns. Gesture detection suite still passes.

Review follow-up (Copilot, Medium on the overshoot rule): `overshootTerminatesWord`
accepted any remainder that merely STARTED with '>', so a single tokenizer token
("rch>junk", "rch></function>") was admitted and carried text past the state this
mask exists to constrain.

Why this is more than cosmetic: the required-parameter gate works by withholding
the `</function>` TAG CHOICE (`optionsRequiringRequiredParams`), so it cannot see
a tag smuggled in behind a '>' inside one token. Observed shape it would allow:
a token like `name></function>` closes a call whose required parameters were never
written — exactly the structurally-legal-but-empty call the gate was added to
prevent (QA #23). Since values are unconstrained, "name>junk" also began the body
early, bypassing the parameter-name state entirely.

Fix: the overshoot must END the word — optional space/tab, then '>' and nothing
else. The tail now has to arrive as its own token, where the tag-choice masks can
gate it.

Also added a dead-end valve, since the strict rule is the first one that can
conceivably mask an entire vocab (a tokenizer with no bare '>' token would have
no legal continuation while typing a name): the mask pass counts allowed tokens
and remembers the logit of the first loosely-allowed candidate; if nothing at all
was allowed, that one token is released. Sampling on all -inf logits silently
produces garbage rather than an error, so failing open is the safe direction.

Verified: grammar suite extended with '>'-terminated vs '>'+tail cases for both
the name and parameter-key states, plus a valve case (vocab of ["_fetch", ">junk"]
keeps exactly one continuation alive). Note: the first cut of these assertions
paired a complete prefix with a spanning token (candidate "web_searchrch>"),
which cannot match — corrected. Full fast class: only the two pre-existing
failures.

Review follow-up (Copilot, Low): a subagent's unknown-tool error printed the
subagent WHITELIST as "Available tools". The whitelist is not the available set —
it can name tools that were never registered and omits installed tools outside it —
so the recovery hint sent the model after tools it could never call.

Fix: `SubagentToolExecutor.callableToolNames` is the whitelist INTERSECTED with the
harness's installed catalog, and the unknown-tool error now reads
"Unknown tool 'X'. Allowed tools: <callable>." (or says the whitelist names no
installed tool, instead of listing nothing). The separate un-whitelisted error
still reports the declared whitelist, which is accurate under its own label.

Verified: test constructs a real executor for a generated subagent instance and
checks a fabricated whitelist entry never leaks into the list, the callable
intersection is advertised, installed-but-un-whitelisted tools keep their distinct
message, and an all-fabricated whitelist explains itself.

### QA #36 — tools_load must repeat schemas for already-loaded tools

Live 15:20 run (weather lookup) went off the rails: the model loaded web_search +
web_fetch, then never used web_search — it curled wttr.org (wrong TLD, parked
page), a dead noaa.gov path, a malformed URL, and re-called tools_discover twice.
Context hygiene was perfect throughout (splice-vs-reencode IDENTICAL on every
turn, balanced tags), so this was a tool-INTERFACE problem, not corruption.

Root cause is the schema channel. The pinned system prompt's tool block is fixed
at the CORE set (loads do not rewrite it — that is why the tool description's
"load/unload rewrites the tool block and triggers a re-prefill" is aspirational),
so the tools_load RESPONSE is the only place a loaded tool's schema can arrive.
`loadedTools` persists in-process across sessions, so a second run that asks for
web_search gets `already_loaded: true` and — before this fix — NO schema. The
model then knows the tool exists (tools_discover lists it; its own thinking said
"Actually there's a tool web_search loaded. Let me try it") but has never seen its
parameters, so it hand-rolls curl instead. Same disease as QA #35, different
symptom.

Fix: `ToolLoadTool` now repeats the schema for already-loaded tools and says so in
the message ("Schema(s) repeated for the already-loaded tool(s) so you can call
them correctly").

Verified: test loads web_search, calls tools_load again, and asserts the response
carries web_search's schema (max_results/query), reports already_loaded, and says
"already loaded" in the message. (First cut of that test asserted
`"status": "success"`, which never matches JSONSerialization's pretty-printed
`"status" : "success"` — match on values, not on compacted key separators.)

Residual, smaller hygiene item seen in the same run: the model once wrote
`</function>` without `</parameter>`; the closure adds `</tool_call>` but cannot
insert a `</parameter>` BEFORE an already-emitted `</function>` without splitting
the generated token range, which would break the pin as a prefix (one full
re-prefill per occurrence). Left alone deliberately; the parser tolerates it and
the required-param walk stops at unterminated values (safe direction).

Unrelated: `testWorkingSetManagerTokenBoundaryEviction` failed once in a
full-class run and passes in isolation — order-dependent flake in shared
WorkingSetManager state, not touched by this change.

### QA #37 — invented-host fetch loop: schema fix landed, new guardrail added

The 16:57 run is the first where the model actually used web_search AND web_fetch
(QA #36's schema repeat working — before it, a loaded web_search had no interface
and the model curled instead). Context hygiene stayed clean: splice-vs-reencode
diverges only by the benign boundary tokens (4→6, at token 2724), tags balanced.

New failure mode: the model wanted a curl-friendly weather endpoint it knows from
pretraining (wttr.in) and burned six consecutive steps on hosts that do not exist —
wttr.info, wttri.info, "wt.tr.info" (one shell_run even ran a bare URL as the
command, exit 127) — with a web_fetch of a nonexistent host returning
content_length 0. Meanwhile the search results already in context listed real
forecast pages (accuweather.com, …). So the model ignores the URLs it was given and
constructs its own; the prompt rule ("NEVER invent URLs") does not hold under this
temptation.

Fix: a failure-driven guardrail, mirroring the web-search loop guard.
- `AgentHarness.recentSearchResultURLs` records what web_search returned (reset per
  run in `beginAgentSearchGuard`).
- `recordWebFetchOutcome(succeeded:)` counts consecutive web-read failures;
  `looksLikeWebFetchCommand` makes curl/wget through shell_run count too, so a
  fetch-loop through either tool is caught.
- After `webFetchFailureLimit` (2) consecutive failures, `applyWebFetchFailureGuard`
  appends a notice listing up to six URLs from this run's searches, says the host
  was not from search results, and tells the model to answer from the snippets
  instead of fetching again. It is applied in `executeTool`, so it covers web_fetch
  and shell_run identically, and a success resets the counter (one-off failures
  never trigger it).

Verified: guardrail unit test (below limit untouched; at limit lists the returned
URL; empty-search case explains itself; success resets; curl/wget detection
positives and negatives). Full fast class: the two pre-existing failures only.

### QA #38 — structural `<tool_call>` counting: a marker inside a value is data (review follow-up)

Review follow-up (Copilot, High on `StreamingToolParser.unclosedToolCallCount`):
counting raw `<tool_call>` / `</tool_call>` substrings cannot tell a structural
opener from the same literal printed inside a parameter value. Confirmed real:
the grammar leaves `.insideParameterValue` unconstrained (`default: return` in
`GrammarConstrainedSampler.applyLogitMaskLocked`), so a shell command, URL, or
search query that echoes the marker inflates `opens`. One extra `</tool_call>`
was then appended to the turn — an UNMATCHED closer spliced into the model's own
history, the same corruption class QA #32/#33 spent the run eliminating (the
code's "one extra (harmless) closer" note was wrong).

Fix: `unclosedToolCallCount` now scans left-to-right with a depth counter and
skips value spans — `<parameter=…>`…`</parameter>`, the bare `<parameter>`
dialect, and Ling's `<arg_value>…</arg_value>` — so tag-like text inside a value
is treated as data. Nesting is tracked (depth floors at 0) instead of the former
`max(0, opens - closes)` subtraction.

Tests: `testToolCallHygieneAndNestedParameterTolerance` adds the literal-in-value
case (count stays 1; appending one closer balances to 0), the `<arg_value>`
variant, and a closed block whose value mentions the marker (stays 0). Existing
frozen/multi-call/closed-plus-prose assertions unchanged. Verified: test passes.

### QA #39 — a failed prefix backfill left the app stuck "generating" (review follow-up)

Review follow-up (Copilot, Medium on the backfill failure branch in
`ContentView.swift`): when the backfill forward returns false, the branch invalidated the
pinned prefix and returned, but did not clear `isGeneratingText` /
`generationTask`. Confirmed: cancellation has its own reset path (the canceller
clears both flags), but a Metal forward failure is not a cancellation and has no
other reset, so the spinner and the generation scheduler stayed in an
active-generation state and new prompts could not start.

Fix: the failure branch now resets the same state as the prefill-failure path
(`ContentView.swift:10683`), in the same `ownsGeneration`-guarded MainActor block
that invalidates the prefix: clears `isGeneratingText`, `generationTask`,
status text, and the message's `prefillStatus` / `isThinking`.

`Task.isCancelled` early-returns in the surrounding code were left as-is: the
canceller already owns flag reset, and the `ownsGeneration` guard keeps a
superseded task from touching its replacement.

Verified: `swiftc -typecheck` clean, `xcodebuild build` succeeded.

### QA #40 — skipped gestures get no result in durable history (review follow-up)

Review follow-up (Copilot, Medium on the gesture split): `responseSlots` fed only
the immediate continuation turn; it was never persisted to
`ChatMessage.toolCalls`, and `initialRecords` was built from `actionableCalls`
alone, so a skipped gesture had no record. The assistant message content still
carried the raw gesture call (the display path strips `<tool_call>` blocks from
`content`, confirming they live there), so a LATER user turn re-encoded an
assistant turn whose call had no matching `<tool_response>` — the truncated-
exchange shape that invites the model to re-issue the call.

Fix: every skipped gesture now gets a persisted record via
`AgentHarness.gestureSkipRecord(for:)` — its output is the same model-facing skip
notice the live turn reports, rendered once so live and rebuilt turns agree — and
the record is flagged `isGesture` so `ChatDetailView` hides it from the tool
timeline. Record building moved ahead of the `!actionableCalls.isEmpty` guard and
is laid out by call index, so a gesture-only turn (which ends the run) persists
too and every call keeps its transcript order.

The same path exposed a second bug: the Qwen history-reconstruction branch was
missing the `!cleanMsg.contains("tool_call>")` guard the Spark and Ling branches
already had, so whenever `content` still carried the raw call it was appended
again by the rebuild loop — every prior tool call (and now every gesture) would
appear TWICE in the reconstructed prompt for a new user turn. Guard added; it
falls back to rebuilding from the record when the content no longer carries the
call.

Tests: `testNoOpGestureToolCallDetection` now asserts the gesture record's shape
(isGesture, name, status, rendered notice) and that `splitGestureCalls` reports
the skipped index. Verified: test passes, full fast class shows only the two
pre-existing failures.

### QA #41 — "off the rails" after a large tool dump: nested-tag value loop (release + debug)

Both regressions reported after the release are the SAME task — "analyze this CSV
in ~/Downloads" — and both start right after a nearly-complete large tool result
that got truncated.

- Session `2260612F` (19:16–19:26): step0 emitted a clean `head -5`; step1 then
  degenerated into `<parameter=command>` followed by a run of bare `<parameter>`
  openers with no value and no closers. The harness closed the block with one
  `</tool_call>`; the parser handed `shell_run` a value of literal `<parameter>`
  markup; zsh failed (`parse error near '\n'`); the malformed turn entered history
  and the model re-emitted the same shape for seven steps.
- Session `A884D96F` (19:34–19:51): step0 clean `cat`; the model then minted a
  wrong filename (`/Users/derek Harris/…`, dropping the `_`/`,`) and re-issued the
  failing `wc -l` until an `ls` finally showed the real name.

Mechanical checks were clean: system prompt unchanged pre/post merge; prompts
well-formed; tag balance correct (the "extra closers" were only the dump's
divergence-window header); `splice-vs-reencode` IDENTICAL for every session-B
turn and a benign 1-token divergence for session A. Crucially, `unclosedToolCallCount`
returns the same value for these turns under the new scanner and the old
substring counter, so the recent closure/splice work does NOT change these
continuations — the prompts are byte-identical to the previous build's.

Root cause: the parameter-VALUE state was unconstrained, so once the model began
echoing `<parameter` inside the value it could loop there forever. The active
Ornith profile (`ModelProfileManager`, ornith `.coder`) uses `temperature 0.60,
repetitionPenalty 1.00` — no repetition penalty — so nothing discouraged the
repeat, and the `<<<TRUNCATED>>>` marker (with no `next_start_line` for
`shell_run`) is exactly what prompted the model to try again.

Fixes:
1. `GrammarConstrainedSampler` now masks the value state so it can never re-open
   `<parameter`/`<arg_value`. Single-token openers are withheld from a set built
   once per tokenizer (`ensureOpenerTokenIds`); a split opener (tail sitting on
   `<`, `<pa`, …) triggers an O(vocab) completion scan (`endsWithOpenerPartial`).
   Nested tags are never valid — the parser already rewrites them as a repair.
2. `shell_run` truncation now adds a `note` telling the model NOT to re-run the
   same command, but to narrow (head/tail/grep/sed) or write a summary to a file
   and page it with `file_read`. The bare marker was the retry trigger.
3. `AgentHarness.consecutiveFailedToolCalls` tracks same-tool failure streaks
   (`recordToolCallOutcome`); at `failedToolCallLimit` (4) the run ends with the
   forced-synthesis turn instead of burning the remaining steps. Mirrors the
   existing empty-arg / search / web-fetch guards.

Tests: `testGrammarMaskBlocksNestedTagInParameterValue` (whole opener, split
opener, and a legitimate `</parameter>` that must stay allowed) and
`testConsecutiveFailedToolCallStreak`. Verified: `swiftc -typecheck` clean,
`xcodebuild build` succeeded, full fast class shows only the two pre-existing
failures.

### QA #42 — second CSV retest: path-copy fidelity + thrash-with-successes (release)

Re-run of the same "analyze this CSV" task, twice (two step0 dumps → sessions
`2260612F` and `8E56D734`). The grammar fix held — the `<parameter>` loop is gone
— but a new shape of the same disease appeared in `8E56D734`:

- step1 is a clean `head -5`; the ~10k-token CSV dump comes back truncated.
- step2 the model writes a Python heredoc, but the path is mangled to
  `/Users/derekparris/Downloads/LDW End of Engagement Survey.csv` — the
  `_September 25, 2026_14.06` suffix dropped. FileNotFound.
- It then thrashes for six more steps: `/Users/derekarris/...` (missing `p`),
  literal `...` placeholders, then the invented `/Users/derek Harris/...` that
  earlier session B also produced, and heredocs with DUPLICATED terminators
  (`PYEOF\nPYEOF\nEOF`, `EOF\nEOF`) so even successful-looking writes error with
  `zsh: command not found: PYEOF`.

Not a prompt bug: `splice-vs-reencode` is IDENTICAL for every step, and the exact
filename is in the prompt verbatim twice (user message + step0 call). It is the
model losing copy fidelity after the big tool result — and the consecutive
failure streak never tripped because partial successes (writing a script,
listing a dir) reset it every few steps.

Fixes:
1. `AgentHarness.missingPathRepairHint` — on a "no such file" failure it finds
   the non-existent `/Users/…` or `/home/…` path in the command, lists the real
   names in that directory (up to six, ranked by shared prefix), and tells the
   model to copy the exact name. Wired into `shell_run`'s error hint. This is the
   direct break for the path-mangling loop.
2. `AgentHarness.totalFailedToolCalls` (limit 6) — counts ALL failed calls in the
   run regardless of interspersed successes, and escalates to the forced-synthesis
   turn like the consecutive streak. The streak alone cannot bound a thrash that
   succeeds every few steps.

Tests: `testMissingPathRepairHintSuggestsRealSibling` and an extended
`testConsecutiveFailedToolCallStreak` (total survives successes; switching tools
restarts only the consecutive). Verified: `swiftc -typecheck` clean,
`xcodebuild build` succeeded, full fast class shows only the two pre-existing
failures.

### QA #43 — grammar mask redirected the loop; repair the value, and give the model a repetition penalty

Re-run of the same task (`2A60624E`), now with a short path (`Survey_Data.csv`)
so the path-mangling is out of the picture. The model still degenerated, but in a
new shape — and it points at the grammar fix from QA #41:

    <tool_call>
    <function=shell_run>
    <parameter=command>
    command>
    python3 << 'EOF'
    ...
    EOF
    </parameter>

The value's first line is `command>` — the model re-emitted the key terminator.
The parsed command became `command>\npython3 …`, so zsh failed with
`parse error near '\n'`, and it repeated for every following step. `command>`
appears ONLY in this session's dumps (not before QA #41), so masking `<parameter`
inside the value did not remove the degeneration — it redirected it to the next
token. Whack-a-mole.

Two fixes:

1. `StreamingToolParser.unwrapNestedParameterTag` now strips a value's leading
   spurious markup: a bare `<parameter>` opener AND a lone `key>` line (the
   duplicated key terminator). A legitimate multi-line value or a real redirect
   (`> out.txt`) does not match, so real content is untouched. With this the
   `command>` case parses to the intended `python3 << 'EOF' …` and RUNS, so the
   loop breaks by success rather than by another guard.

2. `ModelProfileManager` Ornith `.coder` repetitionPenalty 1.00 → 1.10 (the only
   deviation from the official 1.00 profile; documented in the code and the
   profile test). At 1.00 there is no penalty and the model freely repeats the
   just-emitted tag/key/terminator. This is the lever that should stop the
   degeneration at the source; it is a one-line revert if 35B quality suffers.

Tests: `testSpuriousKeyTerminatorInParameterValue` (both parsers recover the
command; normal value and real redirect untouched) and the Ornith profile
assertion updated to 1.10. Verified: `swiftc -typecheck` clean, `xcodebuild
build` succeeded, full fast class shows only the two pre-existing failures.

### QA #44 — the loop token morphs (`<parameter>` → `command>` → `<command>`); recover the value, don't chase the token

The 12:38–13:58 dumps are ONE session (`CDCDBA27`): step0 at 12:38 ran well
(steps 1–5 wrote and ran scripts successfully), fumbled around step 7–9 (a
`/tmp/bin/python` attempt), and after a follow-up message the model degenerated
into a `<command>` loop:

    <parameter=command>
    <command>
    <command>
    … ten lines …
    <</tool_call>

The value had NO real command — the model's thinking was coherent ("let me check
the python@3.14 install directory") but it failed to emit the command, and the
free-form value state repeated tag-like tokens. Note the loop token keeps
changing between sessions: `<parameter>` (QA #31/A), `command>` (QA #43),
`<command>` (here). Chasing each one with grammar rules is whack-a-mole; the
repetition penalty (1.10) was too weak to overcome a sharply peaked repeat.

Fixes:

1. `StreamingToolParser.unwrapNestedParameterTag` now strips leading tag-shaped
   markup lines generically — `<command>`, `</command>`, `command>` — not just
   `<parameter…>`. A value that is ONLY such noise collapses to EMPTY, which the
   existing empty-argument guard catches: the call is not executed, it counts
   toward `consecutiveEmptyToolCalls` (>= 2) and the failure caps, so the run ends
   with a synthesis turn after two loops instead of seven parse errors. When real
   content follows the noise it is recovered as before. Real redirects (`> out.txt`,
   `< input.txt`) and `<<'EOF'` do not match (whole line, single token, no spaces).

2. `ModelProfileManager` Ornith `.coder` repetitionPenalty 1.10 → 1.20 (still the
   only deviation from the official 1.00 profile).

Not fixed, and worth stating plainly: the model is failing to PRODUCE the
command. Harness repairs only stop the damage; they cannot invent the command.
If this persists, the next lever is sampling (temperature) or disabling grammar
masking as a diagnostic — and ultimately the model may be the limit on this task.

Tests: `testSpuriousKeyTerminatorInParameterValue` extended (`<command>`/`</command>`/
`command>` collapse to empty; the live parser surfaces an empty command so the
guard catches it) and the profile assertion updated to 1.20. Verified:
`swiftc -typecheck` clean, `xcodebuild build` succeeded, full fast class shows
only the two pre-existing failures.

### QA #45 — agents must check Homebrew before calling a package missing

Field report: the agent repeatedly concludes a tool/library is "not installed"
when it is actually brew-managed, because it only checks `which <tool>`, the
system interpreter, or `/usr/local`. On this machine that is the common case
(e.g. `python@3.14` under `/opt/homebrew/opt`, brew formula tools), so the model
wastes steps trying to install or work around something that already exists.

Fix (prompt/tool guidance only, no behavior change):

- New system-prompt section `# Local Environment & Homebrew` (all models,
  including Ling): check `brew --version`, `brew list --formula` / `--cask`,
  `brew --prefix <formula>`, and `which -a <tool>` BEFORE concluding anything is
  missing; enumerate language runtimes (e.g. the several `python3`s) and query a
  specific interpreter directly; never install/upgrade unless asked; when
  reporting a missing dependency, state that Homebrew was checked.
- `shell_run`'s tool description now carries the same rule, so it is visible even
  if a model skims the system prompt.

Tests: `testAgentHarnessToolCalling` asserts the section and its key commands are
present in the built prompt. Verified: `swiftc -typecheck` clean, `xcodebuild
build` succeeded, full fast class shows only the two pre-existing failures.

### QA #46 — reverted the Ornith repetition-penalty bump (it broke verbatim copying)

Follow-up to QA #43/#44. Field report: the agent copied the attached path
correctly under the `.assistant` profile, but mangled it immediately under
`.coder` (`/Users/derekparris/…` → `derekeparris`, `derekarris`) after a rebuild.

The comparison is confounded — the well-behaved run was `.assistant` on a build
WITHOUT the QA #45 Homebrew section, the failing run was `.coder` on the build
WITH it. But the Homebrew section is static brew guidance and cannot change how
the model copies a username. The real difference is the profile, and every coder
run in the dumps mangles the path.

The repetition penalty I added is the wrong lever: it penalizes tokens already in
the recent window, and the username sits right there, so it nudges the model to
ALTER a verbatim copy. It also never stopped the template loops (they occur at
1.00 as well). Reverted `ModelProfileManager` Ornith `.coder` repetitionPenalty
1.20 → 1.00 (official). Loop damage stays handled by the parser repair + failure
caps from QA #42–#44.

Remaining profile delta from the assistant config that worked: temperature 0.60
vs 1.00 and presencePenalty 0.00 vs 1.50. If coder still mangles the path at
1.00, the next step is to align those rather than touch penalties.

Tests: profile assertion back to 1.00. Verified: `swiftc -typecheck` clean,
`xcodebuild build` succeeded, full fast class shows only the two pre-existing
failures. (`testModelSpecificProfilesCoderAndAssistant` flaked once in a full
run, then passed on re-run; it reads shared UserDefaults.)

### QA #47 — model load clobbered the active profile's system prompt (UI/profile desync)

Field report: on app launch the Assistant profile was shown as active (it is set
from the persisted `dynamoe_model_global_active_profile_v1` / per-model map and
the UI reflects it), yet the run's own prompt dump showed the CODER system prompt
(`"You are an expert software engineer and programming assistant…"`) and coder
sampling. The Assistant profile's prompt (`"You are a helpful, respectful, and
honest AI assistant."`) never appeared in the prompt.

Root cause (ordering bug in `ContentView.loadAndBridgeToMetal`, NOT in
`ModelProfileManager`): the loader captured `self.systemPrompt` before starting
its detached task, then wrote that captured value BACK on completion:

    let currentSystemPrompt = self.systemPrompt   // captured at call time
    ...
    await MainActor.run { ... self.systemPrompt = sysPrompt }

Launch order is `onAppear -> switchModel -> loadAndBridgeToMetal` (starts the
async load) followed immediately by `applyProfile(.assistant, …)` (sets the
correct prompt). Because the load completes LATER, its stale write won: the
profile-applied prompt was overwritten by whatever `self.systemPrompt` held
before `applyProfile` ran — the persisted user-default prompt, which had been
set to the coder text via "Save as Default". Result: `activeProfile == .assistant`
while the model ran the coder prompt.

Fix: the load completion no longer restores a pre-load prompt. It only fills the
user default when `systemPrompt` is empty. The profile (via `applyProfile`, which
every `switchModel`/session-change/open-panel path already calls) is authoritative.

Verified against the live machine state: `~/Library/Preferences/DRP.DynaMoE.plist`
(non-sandboxed store — the sandbox container plist is stale and lacks profile
keys) holds `dynamoe_user_default_system_prompt` = the coder text and
`dynamoe_model_active_profiles_v1["ornith-ai/ornith-1.5-35b-a3b-fp8"] = "Assistant"`,
exactly matching the reported desync. Dump `prompt-step0-…19-18-48Z.txt` contains
`expert software engineer` and not `helpful, respectful`.

Tests: new `testOrnith35BFamilyProfileResolution` (35B key normalizes to
`ornith-ai/ornith-1.5-35b-a3b-fp8`, snapshot path normalizes to the same key, and
`.assistant` resolution returns temp 1.00 / presence 1.50 / the assistant prompt).
Also repaired a dangling edit in `testModelSpecificProfilesCoderAndAssistant`
(missing `let manager = ModelProfileManager.shared`). Verified: `swiftc -parse`
clean, both profile tests pass via `xcodebuild test`.

### QA #48 — duplicate parameter executed the wrong command; malformed call at step>0 ended the run silently

Two defects surfaced in one run (session 753AF4D4, `prompt-step0-…20-06-54Z.txt`);
the profile fix from QA #47 held (every dump carries the Assistant system prompt).

**(a) Duplicate `<parameter=command>` — last value won.** At step 3 the model emitted
two command parameters in one `shell_run`:

    <parameter=command>
    python3 -c "import pandas; print(pandas.__version__)" && ls -la …/Survey_Data.csv
    </parameter>
    <parameter=command>
    echo check python pandas availability
    </parameter>

Both parsers (`StreamingToolParser.parseStreamingToolCalls` dialect path and
`AgentHarness.parseAllXMLFunctionCalls`) assigned with `args[key] = value`, so the
trailing human-language echo OVERWROTE the real command. The harness ran the echo,
the intended pandas/ls check never happened, and the model got a useless result.
Fix: `StreamingToolParser.assignParameter` keeps the FIRST non-empty value; a later
value may still replace an earlier one that collapsed to empty (noise-only), so both
orderings recover the real value.

**(b) A malformed tool call at step > 0 ended the run with no recovery.** The turn
that followed produced no further dump: the parser returned zero usable calls, and the
only recovery branches for "model tried to call a tool but it did not parse" were gated
to `agentStep == 0` (`hasTruncatedToolCall` / `detectUncalledActionIntent`). At step 3
the run simply fell out of the harness with `willContinueAgent == false`, the model's
raw output left in the bubble, no re-prompt and no answer. Fix: new
`AgentHarness.hasToolCallMarkup(in:)` (true when the text contains `<tool_call>` or
`<function=`) and the recovery branch now fires at ANY agent step when a turn carried
tool-call markup but produced no actionable call — provided no gesture call ended the
turn. Bounded by `maxAgentSteps`.

Tests: `testDuplicateParameterPrefersRealValue` (real value wins over the trailing
echo, via both the live parser and the canonical parser; noise-first ordering still
recovers a later real value; `hasToolCallMarkup` true for attempted calls, false for a
plain answer). Verified: `swiftc -parse` clean, parser tests pass.

**(c) The actual stop was a no-op gesture — the console confirmed it.** The app log
from the run ends with `🧹 [AGENT] dropped 1 no-op gesture call(s) — treating turn as
finished`, followed by `📌 [PREFIX] pinned 6678 tokens (prompt=6437 gen=241)` — a
6437-token prompt matching exactly `prompt-step3`'s `promptTokens=6437`. So the turn
after the bad tool result emitted a bare no-op `shell_run` (a "task finished" gesture per
QA #31) and the harness ended the run by design, with no final answer.

QA #31's gesture handling is right when the model gestures AFTER writing its answer
(otherwise a second re-summary bubble). It is wrong when the model gestures WITHOUT any
answer — the run just stops and the user gets nothing. Fix: after the gesture drop, if
the model wrote no answer this turn (and there is no prior content in the bubble), force
the existing tools-disabled synthesis turn so the user gets an answer. `responseTextWithoutToolCalls`
strips `<tool_call>…</tool_call>` blocks so a gesture commingled with a real answer still
ends normally. The synthesis directive now names both causes (empty-argument calls or a
no-op end-of-turn call).

Tests: gesture assertions added to `testDuplicateParameterPrefersRealValue` (answer+gesture
still counts as an answer; gesture-only strips to empty). Verified: `swiftc -parse` clean,
`testDuplicateParameterPrefersRealValue` and `testNoOpGestureToolCallDetection` pass.

Review follow-up (Copilot, Medium on `hasToolCallMarkup`): the predicate checked only the
Qwen markers, but the parser also accepts Llama 3 calls introduced by `<|python_tag|>`
(`StreamingToolParser.llamaTagOpen`), so a malformed Llama call would skip recovery. Added
that marker (hermetic/Ling JSON already sits inside `<tool_call>`, so it was covered).
Test asserts the Llama opener is detected.

Review follow-up (Copilot, Medium on `GrammarConstrainedSampler.ensureOpenerTokenIds`): the
nested-opener cache was keyed on `vocabSize` alone, but the decoder closure comes from the
ACTIVE tokenizer and the sampler is a singleton that outlives model switches. Two models
can share a vocab size with different token→text mappings, so a stale cache would mask the
old tokenizer's ids and leave the new tokenizer's opener tokens unmasked. Added
`GrammarConstrainedSampler.invalidateTokenizerCaches()`, called wherever `self.tokenizer`
is reassigned (`loadTokenizer` and the model-load completion). Test
`testGrammarOpenerCacheInvalidatedOnTokenizerChange` demonstrates the stale-cache mask on
the wrong id and the corrected mask after invalidation.

Review follow-up (Copilot, High on the opener mask): the value-state mask withheld any
token whose decoded text merely CONTAINED the prefix `<parameter`, which also rejects
legitimate values such as `grep -F '<parameter' file` (and the split-boundary scan used
the same over-broad `contains`). Narrowed both to COMPLETED openers — `<parameter=`,
`<parameter>`, `<arg_value>` — so a `<parameter` prefix can appear in a value while the
re-open loop stays blocked. To keep the blocked path closed when the opener arrives split
across tokens (Ornith's vocab has no `<parameter` token at all; it is `<` + `parameter` +
`=`), the boundary scan's probe now holds the whole `<parameter=`/`<parameter>` boundary
(suffix 16) and `endsWithOpenerPartial` also recognizes the FULL opener, so the completing
`=`/`>` token is still withheld. Updated `testGrammarMaskBlocksNestedTagInParameterValue`
accordingly (bare `<parameter` and a value containing it now pass; `=`/`>` completion is
masked).

Review follow-up (Copilot, Medium on `responseTextWithoutToolCalls`): it stripped only the
Qwen `<tool_call>…</tool_call>` block, so a Llama 3 gesture turn (`<|python_tag|>…`) left
raw markup in the response, made the caller believe an answer was written, and suppressed
the gesture synthesis fallback — the run could still end with no user-visible answer. The
regex now strips every dialect the parser accepts: `<tool_call>…</tool_call>`, the Llama
`<|python_tag|>…</|python_tag|>` (or unclosed), and a bare `<function=…>…</function>`
block. Test covers each dialect (and answer+Llama-call still counting as an answer).

### QA #49 — exit-0 shell runs that failed outright still counted as successes, so the loop guard never escalated

The latest dumps (`prompt-step0-…18-51-20Z.txt` through `step5`) show the same
mangled-path spiral QA #31/#32 named (`/Users/derekparris/Downloads/survey.csv` →
`/Users/derek Harris Downloads` → `/Users Derek` → `/Path Space`), but the run was
never stopped. Six `shell_run` calls in a row, none of which inspected the data:
`head - 30 <path>` (stray space), `cd <mangled>; awk …` (cd failed, awk ran in the old
cwd), and two `cd <mangled>; python3 …` heredocs with `NameError`s. Every one exited 0,
so `recordToolCallOutcome(succeeded: execResult.record.status == .success)` saw six
successes, `consecutiveFailedToolCalls` stayed 0, and the forced-synthesis escalation
(`escalateToSynthesis`) never fired.

Prompt hygiene was NOT the cause: prefix reuse was full on steps 1-3 and 5 (step4's
`prefixReused=0/pin=6864` is the intended `contentChanged → re-encode` fallback from
`buildSplicedContinuationTokens`, not a fault), the tag counts were balanced every step
(QA #32 holding), and step5's `splice-vs-reencode DIVERGES at 8307` is the known-benign
case (splice=string=8841; the two decode windows are byte-identical).

The defect was failure classification: `shell_run` returns `stderr: nil` on exit 0
(deliberately, so benign brew/pip chatter never trips the guard), which also swallowed
the failed elements. Fix: `AgentHarness.shellRunSoftFailure(stderr:)` — true only when an
exit-0 stderr line is attributed to a command/builtin (`zsh:cd:1: too many arguments`,
`head: -: No such file or directory`, `zsh: command not found: …`) and is not a benign
`warning:`/`note:`/`notice:`/`info:` line. `ShellRunTool` now returns that stderr as the
soft-failure signal, so the model still reads the success JSON + stderr text (unchanged
context) while the streak increments and the guard escalates; `succeeded` flips on the
UI/guard side only. The synthesis directive now also names "a bad path, a missing binary"
as a cause.

Tests: `testShellRunSoftFailureDetection` (the three observed diagnostics count; empty,
whitespace, a `warning:` line, a download-progress line, and an unattributed phrase do
not). Verified: `swiftc -parse` clean, `testShellRunSoftFailureDetection` and
`testNoOpGestureToolCallDetection` pass.

**Follow-up — the model drops to a bare relative filename.** The next run
(`prompt-step5-…20-49-03Z.txt`) showed the mirror-image problem: handed
`/Users/derekparris/Downloads/survey.csv`, the model ran `open('survey.csv')`. That
worked only because the app's process cwd happened to contain the file
(`dynamoe_agent_working_directory` was unset, so `shell_run` fell back to
`FileManager.default.currentDirectoryPath`, `AgentHarness.swift:166`). Under a different
cwd the relative command is a guaranteed FileNotFound, and `missingPathRepairHint` —
which only inspects `/Users/…` and `/home/…` tokens — offered no guidance. Two fixes:

1. Prompt rule (both tool-format branches): "For any file the user names, use its EXACT
   absolute path … Never `cd` into a folder and then reference a bare filename, and never
   assume the working directory already contains the file."
2. `missingPathRepairHint` second pass: a bare relative token that (a) carries a dot or an
   interior slash and (b) does not exist under the cwd is matched by basename against the
   absolute paths in the new `AgentHarness.lastPromptText` (set from `formattedPrompt` in
   `startAutoregressiveGeneration`) and echoed back as the exact path to use. Non-path
   tokens (`csv`, `import`) are skipped because they carry no dot and never match.

Tests: `testMissingPathRepairHintResolvesRelativeName` (echoes the user's absolute path
for `open('<name>')`; a dotless module token yields no hint). Verified: `swiftc -parse`
clean, test passes.

Also widened `missingPathRepairHint`'s trigger: a failed `cd` writes
`zsh:cd:1: too many arguments` (the mangled `/Users/derek Harris Downloads`) with no
missing-path phrase, so the guard bailed and the hint never ran on the very first
mangled-path command. A line containing `cd:` plus a cd diagnostic
(`too many arguments`, `no such file`, `not a directory`, `permission denied`,
`string not in pwd`, `invalid option`) now counts as a missing-path signal, so the
absolute-path pass names the real sibling directory. Test:
`testMissingPathRepairHintHandlesFailedCd`. Verified: passes.

### QA #50 — RAM telemetry displayed RSIZE as the "Activity Monitor" number and double-counted every resident weight page

Found by probing the app's exact zero-copy strategy (read-only `memmap2` mmap of a
3 GiB file wrapped in `MTLBuffer(bytesNoCopy:.storageModeShared)`, then faulting
"backbone"/"expert" ranges), with every number cross-checked against `footprint`,
`vmmap --summary`, and `ps` (all tools agreed at every phase):

1. **"Process Heap (Activity Monitor / Xcode)" showed RSIZE.** The metric was
   `mach_task_basic_info.resident_size`, which counts every resident page the task
   maps — including the clean file-backed weight pages held by the unified page
   cache. Activity Monitor's "Memory" column and Xcode's memory gauge display
   `phys_footprint` (dirty + compressed + IOKit only). Probe: with a 512 MB heap
   plus 2.25 GB of weights faulted in through the mapping, the displayed metric
   read 2.79 GB while every ground-truth tool read **0.54 GB** — a 5.2x
   overstatement. In-vivo (real Ornith 35B FP8 decode via the stream bench):
   RSIZE 8.8–10.0 GB vs footprint 4.6–5.8 GB.
2. **"Working Set RAM" = RSIZE + (denseBytes + residentExpertBytes) double-counted
   the whole tracked catalogue.** RSIZE already contains every resident weight
   page, file-backed (touching 1.25 GB of mapped weights raised RSIZE by exactly
   1.25 GB) or anonymous (pinned backbone, ExpertIOThreadPool staging). The probe
   state above displayed **5.04 GB against a true 2.79 GB** working set, and a
   live 12.9 GB sidebar reading decomposes as 5.2 GB RSIZE + 7.7 GB tracked.
   `trimToBudget` also made its over-budget trim decision on the same inflated
   figure (ContentView.swift `effectiveResidentMemoryGB`/`trimToBudget`), evicting
   experts early.

Fixes (ContentView.swift, SidebarView.swift, SettingsSheetView.swift):

- Added `getActivityMonitorFootprintGB()` (`task_vm_info.phys_footprint`);
  the sidebar tooltip, the settings "Heap:" line, and both tooltips now show it
  for the "(Activity Monitor / Xcode)" number.
- `effectiveResidentMemoryGB` and `trimToBudget` now use RSIZE alone; tracked
  dense/expert bytes remain LRU bookkeeping, not RAM added on top.
- The "Unified Memory Cache" tooltip line now measures residency directly: a
  mincore() walk over the registered shard mappings counts the resident weight
  pages. RSIZE - footprint is only an accounting delta (it also contains clean
  framework/library mappings and is offset by compressed/IOKit charges), so the
  subtraction is not used; the displayed value supersedes the catalogue delta
  shipped in the first round of this fix and is an exact page count.
- Side finding from the same probe: the fd "fast path" in `primeSlices`/
  `preFaultAll` `pread()`s into the PROT_READ shard mapping, which returns
  EFAULT on the first chunk (`if n <= 0 { break }` swallowed it) — a silent
  no-op, and `primeSlices` returned before ever running its own touch fallback,
  so decode-time backbone prefetch never primed anything. Removed; the stride
  touch (T8: 13–20 GB/s warm) now always runs. Also verified:
  `posix_madvise(DONTNEED)` does not reclaim clean mapped pages until system
  pressure exists (RSIZE held flat for 30 s after an "eviction"), so post-eviction
  RSIZE is the honest view while the catalogue's "evicted" bookkeeping is
  optimistic until the kernel actually reclaims.
- Copilot review round on PR #25: the now-dead fd/path cache is removed
  (`initialize` no longer opens one descriptor per shard;
  `closeAllFileDescriptors` became `releaseShardMappings`, its only fd readers
  having been the removed pread paths), and every "Process Heap" label is now
  "Process Footprint" — phys_footprint includes compressed memory and IOKit
  allocations, not just heap.
- Second Copilot round: the injected resident-memory seam assertions now pin the
  no-double-count contract exactly (the injected 2.5 GB must pass through even
  with 480 experts tracked; the mock catalogue alone adds ~480 KiB of dense
  bytes, an order of magnitude above the 1e-4 GB tolerance), and
  `testPrimeSlicesFaultsReadOnlyMmapPages` proves the stride touch on a REAL
  F_NOCACHE-cold file read-only-mmap'd and wrapped in
  `MTLBuffer(bytesNoCopy:)`: mincore reports ~0 residency before priming and the
  full 1 MB after, so the old pread EFAULT no-op (which left it cold) fails the
  test instead of passing silently.
- Third Copilot round: the residency walk is now cached. The O(#pages) mincore
  scan was evaluated eagerly inside the SwiftUI body — tooltips build their
  format strings during every render and currentRssGB ticks ~10x/s during
  prefill, so a 37 GB model would repeatedly scan ~2.4M page flags on the main
  thread even with no tooltip visible. `residentShardBytesGB` is now an O(1)
  cached read; a utility-queue loop walks at a bounded 2 s cadence (armed at
  `initialize`, cancelled by `releaseShardMappings` via a generation counter),
  and `refreshShardResidencyNow()` provides the synchronous walk for callers
  that need determinism (tests assert both the walk result and its publication
  into the view-facing cache). The read-only priming regression test also now
  aborts before mmap when the temp-file write is short, instead of risking a
  SIGBUS stride read over an undersized file.
- Fourth Copilot round: FlashMoE loads bypass `WorkingSetManager.initialize`
  (only the regular path calls it), so the residency registry was never adopted
  on a first packed load and kept walking the previous model's stale mappings
  across a switch. Registration is now an independent call:
  `registerShardMappings(_:)` bumps the walk generation, replaces the mapping
  set, resets the cache, and (idempotently per generation) schedules the
  utility-queue loop — `initialize` calls it, and the FlashMoE branch calls it
  directly after backbone pinning, excluding the pinned backbone buffer (its
  anonymous pages are dirty heap that the footprint line already counts; they
  are not clean unified-cache pages). The stale-publication race Copilot
  flagged is closed by snapshotting the generation with the mappings inside
  `refreshShardResidencyNow` and publishing only if the generation is still
  current, so a walk in flight across a registry swap or release cannot write
  the old model's residency into the new model's cache. Regression test:
  `testRegisterShardMappingsSwapsRegistryIndependently`.
- Fifth Copilot round: registry adoption now happens inside the guarded load
  install (after the `modelLoadToken` check), not before it — a superseded task
  registering its engine's mappings and then being rejected would leave the
  residency loop walking a dropped engine's pointers, stranded if the next
  load also failed. `initialize` is back to paging-catalog-only duty; registry
  adoption is exclusively `registerShardMappings`, called from the guarded
  install for every load path (a pinned backbone still excluded). The walker
  now probes a mapping's unaligned tail page separately instead of flooring
  the length to page granularity, so partial pages are no longer omitted.
  Tests release the singleton's registry when they're done with it, so the
  background cadence never walks a completed test's dead mock buffers.
- Sixth Copilot round (high finding): the failed-load catch path clears the
  installed engine and `shardBuffers` but left the adopted registry probing
  those addresses on the 2 s cadence — and if the next load reuses the virtual
  addresses, mincore would report unrelated memory as shard residency.
  `releaseShardMappings()` now runs in that catch block (teardown of the
  installed set), alongside the engine/buffer clear.

Verification: probe on macOS 26.7.1 / M1 Pro (in-process metrics matched the
`footprint` CLI and vmmap at every phase); Xcode build green;
`testWorkingSetManagerTokenBoundaryEviction`,
`testWorkingSetManagerBulkPreadPriming`, and
`testPrimeSlicesFaultsReadOnlyMmapPages` pass. The first pins the
no-double-count contract through an injected resident-memory seam
(`effectiveResidentMemoryGB` must equal the provided RSS exactly while 480
experts are tracked); the latter exercises the mincore() walker over a
fully-touched mock shard buffer and verifies release cleanup. The mock-buffer
tests were also why the old pread path looked tested — their writable
`makeBuffer(length:)` shards always allowed the pread that the real read-only
mmap forbids.

---

## FIX #11: FP8 KV FLASH-DECODING SPLIT-K (SHIPPED)

The FP8 KV-cache path never received the Fix #4a flash-decoding kernels —
`.fp8` decode ran the one-thread-per-head serial kernels
(`gqa_attention_decode_headgate_fp8` / `_fused_fp8` / `_standard_fp8`), each
thread dequantizing the INT8 cache sequentially over the whole context, so
FP8 was latency-bound and dramatically slower than FP16 at long context (the
"FP8 makes models run super slow" report).

**Shipped:** `gqa_attention_decode_headgate_fp8_chunked` and
`gqa_attention_decode_fused_fp8_chunked` scans (grid `(numQHeads, numChunks)`,
per-head FP16 scales applied inside the chunk loop), reusing the
precision-agnostic F16 chunked combine kernels (they only touch FP32 partials
plus gate vectors). Wired into the `.fp8` decode branch mirroring the F16
branch order (chunked headgate → serial headgate → serial standard → chunked
fused → serial fused), with the same scan→combine memory barrier.

Also fixed alongside (Copilot review round 1):
- The flash-decoding partial buffers were hardcoded `16 * 256` heads×chunks
  while dispatch used the model's `numHeads` — they are sized from `numHeads`
  now (the F16 path shared the same latent overflow for >16-head models).
- JetSpec is now disabled for FP8 KV: `runJetSpecTreeForward`'s tree
  store/verify kernels are F16/FP32-only, and the FP32 fallback would write
  4-byte elements into a 1-byte-per-element cache. `isJetSpecEligible(...)`
  is the single source of truth, covered by
  `testJetSpecEligibilityExcludesFP8KVCache`.

**Verification (T17d in the bench, synthetic INT8 KV with per-head FP16
scales, 16 Q-heads / 2 KV-heads / 256 dim):** outputs match the serial kernel
to max rel diff ~4e-3 (FP reassociation noise, same as T17), and timing:

| Context | Window | Serial (1 thread/head) | Chunked | Speedup |
|---|---|---|---|---|
| 1,000 | — | 21.4 ms | 2.8 ms | 7.7× |
| 2,000 | — | 42.7 ms | 4.6 ms | 9.3× |
| 4,000 | — | 85.6 ms | 11.0 ms | 7.8× |
| 8,000 | — | 170.9 ms | 20.7 ms | 8.3× |
| 4,000 | 512 | 10.9 ms | 2.4 ms | 4.6× |
| 8,000 | 1,024 | 21.5 ms | 2.6 ms | 8.2× |

(The first recorded run of T17d had the sliding-window span inverted — it
dispatched `seqLen - win` chunks instead of production's `win`-sized span,
so most chunks scanned empty ranges and understated the windowed speedups
as 3.2×/4.7×. Copilot review round 2 caught it; the bench now mirrors
production's `spanLen = curLen - winStart` dispatch. With the window
active the chunk count is small (8/16), yet the chunked path still wins
~5-8× because the serial kernel's one thread per head remains
latency-bound; full-attention layers win ~8×.) Xcode build green and
`testJetSpecEligibilityExcludesFP8KVCache` passes on M1 Pro.

**Verification (T17e, fused Q+Gate FP8 pair — the branch Ornith-class
models take, added after Copilot review round 3 flagged it as previously
missed; synthetic INT8 KV, 16 Q-heads / 2 KV-heads / 256 dim):** outputs
match `gqa_attention_decode_fused_fp8` to max rel diff ~4e-3, and timing
(no sliding window on this path — the kernels take no window parameter):

| Context | Serial (1 thread/head) | Chunked + fused combine | Speedup |
|---|---|---|---|
| 1,000 | 21.5 ms | 2.7 ms | 7.9× |
| 2,000 | 42.6 ms | 4.8 ms | 8.8× |
| 4,000 | 85.5 ms | 12.6 ms | 6.8× |
| 8,000 | 171.5 ms | 28.0 ms | 6.1× |

## FIX #12: FP8 SCALE PRESERVATION ON PREFIX-SPLICED RESET (SHIPPED)

Symptom (live agent runs, FP8 KV, Ornith/spark-class models): spliced
continuation turns degenerated into word salad ("within within within …") or
silent token-burn with no tool calls, while full re-prefill turns of the same
prompt were perfectly coherent. The prompt dumps proved the context itself was
clean — `splice-vs-reencode` diverged only by the known-benign 2-token
live-turn boundary merge, and a forced `prefixReused=0` retry of the exact
3834-token step produced a good web_search call — so the corruption was in
KV state, not the prompt build.

**Root cause.** Agent prompts grow every step, so `neededSeqLen`
(`max(8192, prompt + maxTokens + 512)`) grows too and
`KVCacheManager.reset(preservePrefixCount:)` reallocates the cache on
*spliced* turns specifically. The preserve path memcpy'd the pinned INT8
K/V prefix into the new buffers, but the FP8 per-(token, kv-head) dequant
scale buffers (`kScaleBuffer`/`vScaleBuffer`) were simply re-`makeBuffer`d —
fresh zero-filled, old data dropped on the floor. Every restored prefix slot
then dequantized against scale 0: all K scores and V reads collapse, and
attention over the majority of the context returns noise. Full re-prefills
rewrote the scales and stayed clean, which is why the bug only ever hit
spliced steps (Spark's long chats, Ornith step ≥1) and why step0 was fine.

**Why T17d/T17e missed it.** The bench tests the chunked FP8 *kernels*
standalone with offset-0 synthetic buffers that live for one dispatch — the
scale-restore bug wasn't in a kernel at all, it was in the transition between
turns (buffer lifecycle), which no kernel-equality bench can see.

**Fix.** `reset()` now snapshots the old scale buffers before any rebuild and,
for `preservePrefixCount > 0`, restores each slot's scale prefix with the same
old-stride/new-stride memcpy shape as the K/V restore (copy only when the
buffers were rebuilt or the per-slot offset moved; verified skip when the
buffers are retained in place). Also added a belt-and-braces guard at the
reuse decision: a pin minted under one KV precision is invalidated when the
selected precision changes before the turn starts (the preserve math sizes
slots with the new elementBytes against the old layout).

**Tests:** `testKVCacheFP8ScalePrefixSurvivesSpliceRealloc` — allocates an
FP8 cache at maxSeqLen 1024, fills both scale buffers with a distinct nonzero
bit pattern per element, grows to 2048 with `preservePrefixCount: 512`, and
asserts every restored slot's scale prefix round-trips byte-exact while the
tail of the rebuilt buffer stays zero. Pure Metal-buffer logic, no model
snapshot needed. Verified headless on M1 Pro:
`xcodebuild test -destination 'platform=macOS,arch=arm64' -only-testing:
DynaMoETests/DynaMoETests/testKVCacheFP8ScalePrefixSurvivesSpliceRealloc`
passed (0.07s), and `testKVCacheManagerPrefixPreservation` still passes
through the reworked reset preserve path. Two headless-launch footguns hit
while verifying, worth recording: a two-component
`-only-testing:Target/testName` filter silences into `TEST SUCCEEDED` with
**zero** test cases (use the three-component
`-only-testing:Target/Class/method` form), and launching the runner fails
with Runningboard error 5 while a debug session has the app open.

**Follow-up round (second live run, 11:57–12:20Z — same session symptoms, decoded
as the old binary + threshold):** a fresh agent run on the model still produced
spliced-step degradation (byte-exact `<parameter=query>` triplication at step1,
progressive confusion through steps 2–6, word salad at step7), while a forced
fresh re-prefill mid-run (step6 retry) was clean. The per-step pattern fits the
pre-fix scale wipe exactly at that run's agent max_tokens: step1's 3558-token
prompt stayed under the 8192 budget → no realloc → healthy splice (its only
fault, the triplication, is FP8 KV quantization noise — the grammar allows
re-typing a parameter and the model retried it byte-identically); from step2 on,
prompt growth crossed the realloc threshold → every spliced step wiped the
pinned prefix's scales on rebuild → the observed progressive collapse; the fresh
retry escaped it by rewriting the whole cache. The debug session running that
test predated the fix's build, i.e. the app hadn't been rebuilt with it yet.
Added splice-turn telemetry (`🔍 [KVCACHE] splice reset: pin=… maxSeq A->B
kvRealloc=… scaleRebuilt=…`) to `reset()` so any future run's binary vintage and
cache-lifecycle history are readable straight from the Xcode console instead of
reverse-engineered from prompt dumps. A/B switches for bisecting, should it ever
recur on a fixed build: `dynamoe_disable_prefix_reuse` (all turns full
re-prefill) and a temporary FP16 KV setting (isolates quantization noise from
splice mechanics).

### QA #51 — KV provisioning ignored the machine's RAM, and "Full RAM" mode removed the ceiling

Field report: on the 16 GB M1 Pro, the sidebar tooltip read "Working Set RAM:
0.6 GB / Process Footprint: 20.6 GB / Unified Memory Cache: 7.7 GB" while idle
with Spark X2.5 4B loaded — 20.6 GB on a 16 GB machine.

Measured against the live Release process (`footprint -p`, `vmmap`, `ps`, and a
standalone probe), not inferred:

- `phys_footprint` 20.6 GB (peak 26.4 GB); `vmmap` reported writable regions
  `resident 39.6 G / swapped_out 20.5 G (51 %)`.
- `mach_task_basic_info.resident_size` 0.39 GB — the "Working Set RAM" number.
  A probe proved RSIZE *does* count mapped file pages and CPU-written
  `storageModeShared` Metal buffers (2 GB touched through a mapping → +2.00 GB
  RSIZE; 1 GB anon + 2×2 GB shared MTLBuffers → +5.00 GB). So the 0.6 GB reading
  is not "RSIZE misses Metal" — it is RSIZE counting only *resident* pages while
  the OS had swapped the app's GPU allocations out. Under pressure the working-set
  display collapses to a meaningless number while the real footprint stays 20.6 GB.
- `IOAccelerator (graphics)`: **19.4 GB resident, 113 regions**, in repeated
  identical sets — 8 × 3.0 GB, 4 × 2.5 GB, 4 × 1.2 GB. Those sizes match KV
  caches (`kvStride = 4 heads × 256 = 1024`, fp16 → 2 KB/position; 2.5 GB ≈
  36 slots × 32768 positions).

Root cause: `neededSeqLen = max(minSeq, min(32768, prompt + maxTokens + 512))`
capped provisioning at a flat 32k tokens regardless of the model's per-token KV
cost or the machine. For Spark that is 36 slots × 32768 × 1024 × 2 B ≈ **4.8 GB
per K+V pair** — and Auto/Smart mode resolves this model on this machine to
"Full RAM" (`physicalRAM >= model × 1.25` → 16 ≥ 9.6), which mapped to
`budgetMode == .unrestricted` and therefore disabled every budgeting rule. A
small model on a roomy-looking machine was the one case with no KV ceiling at
all.

Fix (`EngineRuntime.swift`, `CompletionEngine.swift`):

- New `KVCacheProvisioning` enum holding the layout math once — `geometry(...)`
  (slot count, ring flags/len, K/V strides, element bytes), `positions`,
  `requiredBytes`, `ringSlotCount`, `clampedSeqLen`, and `budgetGB`. `reset()`
  now *derives its layout from the same `geometry(...)`* it previously computed
  inline (with a debug `assert` that its slot bookkeeping still matches
  `positions`), so the clamp and the allocation can never drift.
- The launch path clamps provisioning before `reset()`:
  `budgetGB = (budgetMode == .unrestricted ? physicalRAM × 0.75 : targetMaxRssGB)
  − modelFootprintGB − 2.5 GB working reserve`, floored at 0.25 GB; the clamped
  length never goes below `max(minSeq, prefixTokensReused)` so a pinned prefix is
  never invalidated. Clamping prints a `⚠️ [KVCACHE]` line with the budget, the
  resulting KV size, and the machine's RAM.

On the reported machine (16 GB, 7.7 GB model): Full-RAM target 12 GB − 7.7 − 2.5
= **1.8 GB KV budget** → ~12k tokens instead of 32k (~4.8 GB); balanced16GB gives
1.3 GB → ~9k tokens. A 64 GB machine with a 37 GB model still gets 8.5 GB (no
clamp at 32k). Tests: `KVCacheProvisioningLogicTests` (bytes/positions/precision
math, ring slots, clamp floors and pass-throughs, budget modes).

**Display metric fixed (same day).** The user-facing memory figure no longer
reports RSIZE. `WorkingSetManager.attributedMemoryGB` = `phys_footprint` +
`residentShardBytesGB` (cached weight pages), which cannot double count because
footprint excludes clean file-backed pages and the shard registry excludes the
pinned backbone. Shown in the sidebar pill (`SidebarView.swift`), the settings
stats row (relabelled `DYNAMOE MEMORY`, `SettingsSheetView.swift`), with tooltips
breaking out footprint / cache (reclaimable) / resident pages. The eviction budget
is untouched: `effectiveResidentMemoryGB` and `trimToBudget` still compare RSIZE
against the mode target, so eviction timing is unchanged — the display metric is
the accurate "how much is DynaMoE using" number, the budget input remains the
resident one. Docs updated (`SETTINGS_GUIDE.md` "Live Diagnostics", README UMA
bullet). Tests: `MemoryTelemetryLogicTests` pins all three contracts.

**Follow-up (measured live after the clamp, 2 turns, Spark 4B):** footprint
**20.6 GB → 6.4 GB** (peak 26.4 → 6.4 GB) and `IOAccelerator (graphics)`
**19.4 GB → 5.5 GB**.

**Correction to an earlier read of that follow-up.** The residual 5.5 GB was
first decoded as a stale KV allocation (3 GiB is exactly `48 slots × 32768
positions × 2048 B`), but the console disproved it: the app's own log shows
`🔍 [KVCACHE] splice reset: pin=470 maxSeq 4809->5090 … slots=36` — KV is
**36 slots × ~5k positions (~750 MB)**, and the clamp never fires because
`neededSeqLen` stays far under the budget. The 5.5 GB is the FlashMoE backbone
pin, printed as `⚡ [FlashMoE] Pinned backbone (5.53 GB) into anonymous GPU
memory`, i.e. intended behaviour (dense backbone pread once into an anonymous
`.storageModeShared` buffer). The 48 × 32768 arithmetic was an over-fit
coincidence; do not repeat that inference without a matching console line.

That leaves the 20.6 GB unexplained by KV: it is ~4 × the 5.53 GB pin, so the
open hypothesis is that **backbone pins stack across model (re)loads** (each
FlashMoE load preads `model_weights.bin` into a fresh 5.5 GB buffer, and the pin
has no release log). The load path already serialises loads, waits generations
out, and replaces `shardBuffers`/the engine's buffer dict, so the retainer is not
obvious. A probe was added: each pin now prints its ordinal, how many *previous*
pins are still alive through a weak reference, and the process footprint
before/after (`… (pin #N; prior pins alive: k = X GB; footprint A -> B GB,
delta +C)`), which makes stacking vs. genuine one-shot allocation black-and-white.

**CONFIRMED — pins stack** (console, three consecutive loads):
`5.53 GB`, `5.53 GB`, `4.19 GB` all pinned, i.e. **15.25 GB resident across three
pins**, one of them a different model's backbone size. With KV only ~750 MB, that
accounts for the ~20 GB footprint on a 16 GB machine. No engine or view property
holds shard buffers (`grep` for `[UInt32: MTLBuffer]` finds only the two dicts
replaced on every load, plus locals captured by in-flight turn closures), so the
fix chosen is structural rather than a hunt for the retainer:

Fix (`ContentView.swift`) — `BackbonePinPool`: one retained pin buffer, reused
when a later model's backbone fits and grown only when it does not, released on
model teardown (load failure and Apple Foundation Model activation). Pin memory is
therefore bounded by the *largest* backbone loaded in a session, not by the number
of loads. The pin log now says `(reused retained pin)` or `(new allocation)` next
to the ordinal/footprint fields, so the effect is verifiable per load. Caveat
noted in the code path: reuse rewrites the buffer in place, which is safe because
`awaitGenerationTeardown()` (engine `awaitIdle`) runs before a load — no live turn
holds a previous model's buffers.

**Verified** (same session, Spark 4B → 4.19 GB-backbone model → Spark 4B):

| pin | backbone | path | footprint |
| --- | --- | --- | --- |
| #1 | 5.53 GB | new allocation | 0.30 → 5.70 GB |
| #2 | 4.19 GB | reused retained pin | 5.86 → 5.86 GB (+0.00) |
| #3 | 5.53 GB | reused retained pin | 5.81 → 5.81 GB (+0.00) |

Footprint holds at ~5.8 GB across model switches instead of climbing ~5.5 GB per
load (20.6 GB before). The telemetry's live-pin accounting reports *distinct*
buffers (`ObjectIdentifier`-deduped): with one reused buffer, a per-pin sum read
`2 = 9.72 GB` for a single 5.53 GB allocation.

Separately, the reuse check *was* capacity-only and is fixed as a correctness
issue (it could reuse a buffer laid out for a different model's slot count,
stride, or precision): the resident allocation now records its geometry
(`allocatedSlotCount`/`KStride`/`VStride`/`elementBytes`) and any change forces a
realloc, for the FP8 scale buffers too. Test:
`testKVCacheManagerReleasesStaleGeometryOnModelSwitch`.

### QA #52 — Server turns on prompts the machine cannot serve: silent KV overrun, minute-long SSE silence, and unexplained "forward pass failed"

Field report (server testing with crush as the client): a request starts, then
the turn "just dies in different ways" — once on Spark, once on Ornith. Live
reproduction with the exact failed-run prompt (19,352 tokens on Ornith, loaded
from the server prompt dumps) pinned three concrete defects, each confirmed on
the 16 GB M1 Pro with the Release binary:

1. **The KV clamp does not bound prefill.** `neededSeqLen` shrinks to the
   machine's KV budget (⚠️ `[KVCACHE]` line), but prefill kept ingesting all
   19,352 tokens into a cache provisioned for fewer rows. Attention reads and
   KV stores index rows past the allocation: adjacent slots absorb the
   overflow as silent corruption, and the final slot's rows fall off the end
   of the buffer. Nothing prints; the turn may still "succeed" with garbage.
2. **The SSE stream goes silent for the whole prefill.** The streaming relay
   cancelled the queue heartbeat the moment the previous turn drained (and the
   generic batched prefill path emits nothing until a layer-chunk boundary),
   so a client saw zero bytes — not even HTTP headers flushing — while the
   engine ground through a prefill that measured at 10–100 s per layer under
   memory compression (`waitUntilCompleted` parked for 40+ minutes; process
   footprint 9.1 GB with 5.5 GB compressed; no crash report, because the app
   never crashed — the turn just ran for hours). To the client this is exactly
   "the process died in different ways".
3. **Silent allocation-failure sites return false with no diagnostic.** The
   per-token Spark runs failed mid-prefill (step 862 / step 225, two runs) with
   only the useless "see prior diagnostics" line — there were no prior
   diagnostics because every Metal allocation-failure guard along that path
   returns without printing. With ~4 GB of FP16 KV provisioned against a
   3.8 GB budget on a 16 GB machine plus the pinned backbone, an allocator
   failure mid-prefill is plausible; the code hid it.

Fixes (CompletionEngine.swift, EngineRuntime.swift, ServerController.swift):

- `KVCacheProvisioning.turnFit(...)` — pure fit check at launch: computes the
  engine's exact post-clamp provision and either `.ok`, `.maxTokensReduced`
  (headroom capped to a floored 64 tokens when the affirmative budget would
  push generation negative), or `.promptTooLong` → new `CompletionLaunchError.promptTooLong`. Rejection happens before any scratch
  or KV allocation, so a bad turn costs nothing. The server maps it to a
  400-shaped `invalid_request_error` in both protocols' streaming and
  non-streaming envelopes (`launchFailed` now carries the full `ServerAPIError`
  so status/type render correctly).
- Runtime row guards: `runTokenForward` refuses `step >= allocatedSeqLen` and
  `runLayerWisePrefill` refuses chunks extending past the provisioned length,
  each with a `❌ [KVCACHE]` line naming the numbers.
- Every silent `return false` guard in `runTokenForward` / `runLayerWisePrefill`
  (command-buffer nil, encoder nil) now prints the failing site, layer, and
  step; the generic prefill fails through the same path.
- The queue heartbeat now runs for the lifetime of the turn (cancel moved into
  the task's `defer`), so supervision pings flow while the engine is IN
  prefill, not only while waiting behind earlier turns.

Measured on the post-fix binary (16 GB M1 Pro, Ornith):

| probe | before | after |
| --- | --- | --- |
| 19,352-token prompt | prefill ran 40+ min past a 8k-row KV clamp, zero frames to client | rejected in **0.01 s** with `invalid_request_error` envelope ("19,360 prompt tokens exceed the 3,276-token KV capacity this turn can provision (KV budget 0.25 GB on a 16 GB machine)") |
| SSE first byte (idle queue) | ~50 s of silence, then 10 buffered pings at once | **0.04 s** |
| SSE pings while queue busy behind a long turn | none until buffered flush | **5.3 s cadence**, continuous |
| short streaming turn (191 tokens) | worked | works end-to-end: role chunk at turn start, `:prefill:` progress comments every ~1 s (12 → 15 tok/s), then reasoning deltas, `[DONE]` |

Tests: `TurnFitLogicTests` (seven cases: ample budget, prompt overrun rejection,
floored max-token ratchet, near-full agent prompt, agent-session floor,
needed-length formula), and `ServerQueueLogicTests` updated for the typed
`launchFailed` error. All existing pure-logic suites still green
(engine/adapters/queue/hardening). The per-token prefill failure print is
cancellation-aware so a user stop reports `🛑 stopped by user` instead of a
forward-pass error.

## FIX #13: STALE TOOL BLOCK AFTER tools_load (SHIPPED)

Third live-run forensics round (13:03–13:14Z, post scale-fix binary — steps were
coherent until `tools_load(["web_fetch"])`, then the model streamed 500+ tokens
without ever committing to a call). Prompt dumps showed two prompt-side defects:

**Cracked open by the dumps:** after `tools_load` executed, the next step's
prompt REUSED the previous turn's system text verbatim
(`nextPrompt = formattedPrompt + …`), so the authoritative "Only the following
functions are currently loaded and callable" block never gained `web_fetch`
(step5's `<tools>` still listed the 5 fundamentals, and `prefixReused==pin`
proved the block bytes never changed — a block edit at the prompt head would
have broken the common prefix). The tools_load result told the model "you may
now call it directly using the provided schema" while the block denied it
existed — the model burned tokens reconciling the contradiction. Same run
revealed the sibling bug: `beginAgentSearchGuard()` (which default-loads
`web_search`) ran AFTER `buildSystemPrompt` at send time, so the first prompt
of a fresh process advertised no web_search at all (previous sessions showed
it only because an earlier send in the same process had loaded it).

**Fix:** `AgentHarness.buildToolsSection()` extracted so a freshly rendered
block and a mid-run refresh are byte-identical by construction;
`refreshingLoadedToolsSection(inPrompt:)` re-renders the "# Tools … NOTE"
segment against the live registry at every post-tool-execution continuation
(normal next step + both synthesis paths); when the section changed the
continuation passes `promptTokens: nil` (full re-encode/re-prefill — the
cost the NOTE text already tells the model to batch around). And
`beginAgentSearchGuard()` now runs BEFORE the system prompt is built at send
time. Telemetry: `🔄 [TOOLS] tool block changed since this prompt was built —
rewriting system tools section (loaded=N)`.

**Tests:** `testToolsLoadRewritesPromptToolSectionForNextStep` — builds a
prompt, loads web_fetch, and asserts the refreshed stale prompt matches a
freshly built one byte-for-byte, is idempotent, and passes non-agent prompts
through untouched. Verified headless on M1 Pro alongside
`testKVCacheFP8ScalePrefixSurvivesSpliceRealloc` and
`testKVCacheManagerPrefixPreservation`. Pre-existing red tests discovered
while verifying (fail identically on the pre-fix tree — stale prompt-section
expectations + a suite-order-dependent isolated failure, documented in AGENTS
notes): `testAgentHarnessBuildSystemPromptIncludesDateTime` (in isolation),
`testAgentHarnessWebResearchPromptGuardrails`,
`testAgentHarnessStructuredSectionsAndReleaseCyclePromptGuardrails`.

## FIX #14: WEB_RESEARCH READINESS HINT (SHIPPED) + ROUND 4 LIVE FORENSICS

Fourth live-run round (15:41Z Ornith / 16:19Z Spark dumps, post FIX #12/#13
binary). The two runs failed for two DIFFERENT reasons — and neither was the
Round 3 stale-tool-block bug (no tools_load was issued; the tool block tracked
the registry correctly at every step, proven by Spark step6 re-encoding fresh
at prefixReused=901 the moment web_search left the registry).

**Spark run (FF5A426B, 6 steps, FP8 KV, 3.0k→8.2k):** generation stayed fully
coherent on every spliced turn through 7.4k context — the splice + FP8 KV +
scale-restore path demonstrably holds for Spark at these lengths. The failure
was behavioral: the model wanted page content ("Let me fetch the detailed
pages"), but web_fetch was never loaded and its schema was hidden, so it
re-issued web_search query after query (six, all distinct enough to dodge the
repeat limit) until `applyWebSearchLoopGuard`'s per-task budget (6) fired,
unloaded web_search, and told it to answer now; with web_search gone from the
block the grammar masks it, and the model burned tokens unable to do the
thing it was told to do. Prompt-side trap: shell_run's description says "To
READ A WEB PAGE, use web_fetch" unconditionally — an instruction referencing a
tool the prompt simultaneously says is not loaded.

**Fix (shipped):** `buildToolsSection()` now appends a targeted "Web research
readiness" hint when web_search is loaded and web_fetch is not — telling the
model to tools_load web_fetch BEFORE it needs pages and that re-searching
cannot yield page content. Byte-shared with the refresh path by construction
(same renderer). Test:
`testToolsSectionCarriesWebFetchReadinessHintOnlyWhenNeeded` (hint appears in
the search-without-fetch state, disappears when web_fetch loads or web_search
unloads, refreshed prompt stays byte-identical to a fresh build).

**Ornith run (BECE59D6, 5 steps, 35B-A3B-FP8, FP8 KV, 2.8k→6.4k):** genuine
progressive token-level degradation, all on spliced turns, starting at turn1
(3.5k, NO realloc — under the 8192 budget, so FIX #12's path is not involved):
duplicated empty parameter blocks, then triplicated parameter blocks with
garbled think text ("Zhiho", "GLK"), then duplicate command params with
fabricated URLs. Tool-response echoes prove the garbling was in the live
emission, not the history build. Differential analysis against the clean
Spark run (identical splice/scale mechanics, no GDN layers): the shared
attention-KV splice path is exonerated; GDN linear/conv state splicing is sound
by construction (the sequence kernels are serial scans over the in-place state
buffer, and capture/restore plus the pin backfill align state to the pinned
token list). Remaining suspects, in priority order:

1. Ornith-only, spliced-only mechanism in the 35B MoE expert path — the
   repacked FP8 experts (ExpertRepacker + ExpertIOThreadPool staging) run with
   a warm working set on splice turns but cold on fresh re-prefills; Spark is
   dense and never touches it.
2. FP8 KV dequant noise specific to Ornith's full-attn GQA layers at 3.5k+
   (Spark tolerates the same quantization, but different model/sensitivity).
3. Prefill-batched vs decode-simd GEMV numeric drift (rel ≤ 4e-4, T18b)
   accumulating through the GDN state across mixed prefill/decode turns.

**Not yet done (needs live console capture or A/B):** rerun the same Ornith
conversation with (a) console capture (`🔍 [KVCACHE]`, `⚠️ [PREFIX] GDN state
diverged`, working-set/resident-expert lines), (b) `dynamoe_disable_prefix_reuse`
(all-turns-fresh — isolates the splice/warm-set path), (c) temporary FP16 KV
(isolates quantization noise). (a)+(b)+(c) on the same prompt pinpoints which
of the three suspects remains standing.

**Follow-up (post-FIX #14):** rather than relying on the hint alone, web_fetch
now ships in `coreLoadedToolNames` (default-loaded from session start, still
unloadable via tools_unload if the model wants the tokens back). Live evidence:
models will NOT tools_load web_fetch on their own even when instructed to
read pages via it, so leaving it opt-in just re-creates the loop.
`testToolsLoadRewritesPromptToolSectionForNextStep` now drives the refresh
regression with git_diff (a non-core tool) since web_fetch is core.

**Review hardening (FIX #12 second round, Copilot review):** the scale-buffer
allocation condition was capacity-based, so an earlier, longer conversation
could leave scale buffers large enough for BOTH the old and new strides. A
spliced reset that changed the stride then relaid out slots inside the SAME
buffer — and the forward slot loop wrote slot 1's 2048-stride destination over
slot 2's 1024-stride source before slot 2 was copied (unreachable with a
max(8192, prompt+…) budget within one session, but reachable across sessions
in a shared process, and memmove would not help — the overlap is between
separate slot regions). Scale buffers now rebuild on every preserve-with-
stride-change so the restore is always buffer-to-buffer; fresh resets keep the
capacity-retention fast path. Test hardening: the FP8 splice test nils the
singleton's scale buffers before setup (its assertions previously depended on
leftover capacity from earlier tests) and adds a retained-capacity phase —
fresh-drop to 1024 inside retained 2048 capacity, then spliced grow back to
2048 — asserting a rebuilt buffer and byte-exact restore of every slot.

**Review hardening round 2 (two more Copilot findings, both valid):**

1. **Scale restore used the padded head count.** The FP8 store/attention
   kernels address scales with the LOGICAL head count the generation path
   resolves (`config?.effectiveNumKeyValueHeads ?? numKvHeads` — 2 for a
   nil-config, non-Nanbeige run), packing each slot as
   `[maxSeq x logicalHeads]` at `slot * maxSeq * logicalHeads`. reset()'s
   `kvHeads` is only an allocation pad (>= 8 when config is nil), and using it
   for the splice restore's offsets/sizes copied the wrong regions after a
   stride change, giving later slots zeroed or foreign pinned scales. No live
   model was affected (every supported model's config head count equals the
   pad, so the numbers coincide), but the nil-config path laid the trap. The
   restore now computes offsets/sizes with the logical count
   (`scaleLayoutHeads`) while sizing/zeroing keep the padded count.
   `testKVCacheFP8ScaleRestoreUsesLogicalKvHeadCount` pins it with a 2-head,
   config-nil, whole-buffer-poison layout.

2. **K/V prefix was not relaid out on spliced stride SHRINK.** neededSeqLen
   can shrink turn-over-turn (tools_unload shortening the prompt above the
   min-sequence floor, or a lower max-tokens setting), and the capacity check
   alone kept the old, larger K/V buffers on such splices — pinned prefix at
   the old slot stride while generation indexes by the new allocatedSeqLen,
   AND while the scales were being moved to the new stride (mismatched pair).
   `spliceStrideChanged` now also forces the K/V realloc+relayout, mirroring
   the scale-buffer condition.
   `testKVCachePrefixRelayoutsOnSplicedStrideShrink` reconstructs a shrink
   splice (512 -> 384, pin 128) with retained-capacity K/V buffers and asserts
   reallocation plus byte-exact prefix restoration and a zeroed tail.

Known remaining caveat (out of scope, unreachable with current models): for a
hypothetical nil-config model whose logical K/V row stride
(numKvHeads x headDim) falls below the 1024-element floor, reset()'s
`effectiveKvStride` pad and the generation-side `kvStride` would disagree; no
supported model is in that regime (all are >= 1024 logical strides).


## FIX #15: UNBOUNDED LAYOUT WHITESPACE AT THE TAG-CHOICE BOUNDARY (SHIPPED)

**Symptom (live, both post-FIX-#14 Spark attempts, 18:21Z + 18:38Z sessions):**
the model thinks, closes its think block, writes its response line, opens a
tool call... and then nothing. Tokens keep streaming at ~2 tok/s, no visible
output appears, no step-1 dump is ever written, and no freeze ever fires. The
persisted transcripts pin it exactly: both turns end mid-call, the last visible
text being the tool-call opener, with 302/555 total tokens against only
~110/200 visible ones — hundreds of tokens emitted past the opener decode to
nothing a user can see.

**Isolation (new model-gated diagnostic,
`testSparkAgentToolCallEmissionDiagnostic`):** replays the byte-exact failing
step-0 prompt (read at runtime from the live dump, 3203/3203 token round-trip)
through the app-faithful FP8 decode path with the real sampler, grammar mask
and parser-freeze wiring. BOTH passes (greedy and the assistant-profile
sampled settings) emit a clean, well-formed web_search call and the parser
freeze fires (g249 greedy / g88 sampled). So prompt, model emission, grammar
mask and parser were all healthy under decode-path numerics — the app-only
failure had to be a marginal trajectory falling into a hole the replication
didn't visit. The transcripts supplied the hole: the ~2 tok/s crawl is the
per-token O(vocab) tag-choice scan, i.e. the mask WAS engaged the whole time,
and the only tokens that pass the engaged mask while rendering as nothing are
whitespace (the response is whitespace-trimmed every frame, and the
degenerate-cycle guard deliberately ignores units with no letters or digits).

**Root cause:** `applyTagChoiceMask` exempts whitespace tokens
unconditionally at the whitespace boundary, and its `allowed()` drops ALL
leading whitespace from candidates, so an empty candidate is a "prefix" of
every option. A trajectory whose top logits at the opener boundary land on
whitespace (the app's chunked-prefill numerics differ from the decode path by
<= 4e-4/projection, T18b — plenty to flip a near-tie) can draw whitespace
forever: every draw keeps the prefix whitespace-only, the exemption re-arms,
nothing is ever visible, no closer is ever typed so the parser can never
freeze, and the loop runs to max-tokens or the user's Stop.

**Fix:** the layout-whitespace allowance at a tag-choice boundary is now
bounded to two characters (one generous newline — the canonical format uses
exactly one). Past the allowance, pure-whitespace tokens are masked too, so
the mask forces the next structural tag (a merged whitespace+tag token still
passes at any time). Healthy calls are unaffected: both diagnostic passes used
a single newline, well inside the allowance.

**Test:** `testGrammarMaskBoundsWhitespaceAtTagChoiceBoundary` (pure logic,
runs anywhere) pins fresh-boundary legality, the allowance window, the bound
beyond it, and the same bound between a parameter closer and the next tag;
it fails on the pre-fix mask by construction (whitespace past the boundary
was unmasked). All six grammar-sampler tests re-run green, plus both
tools-section prompt regressions.

**Residual note:** the diagnostic's own prefill is the decode-path forward,
not the app's chunked prefill, so the trap trajectory itself is not directly
reproducible headlessly — the diagnosis is transcript forensics plus
mechanism analysis, and the fix is verified structurally. Live re-run of the
same conversation is the confirmation step. The per-token O(vocab)
tag-choice scan (the ~2 tok/s crawl whenever the grammar engages) is a known
cost, out of scope here.


## FIX #16: SPARK CONTINUATION PROMPT ENDED AT THE TOOL TURN (SHIPPED)

**Context:** the first live run with the FIX #15 binary was a partial success - the
whitespace hole is closed at the boundary that mattered. Step 0 emitted a clean,
well-formed web_search call for the first time on this prompt, the parser froze,
the tool executed (5 results), and the FIX #12 splice machinery carried the
pinned prefix into step 1 perfectly (prefixReused=3328, splice-vs-reencode
IDENTICAL).

**Symptom (step 1):** 21 generated tokens, then freeze with garbage. The exact
emission: Spark's `<｜end▁of▁text｜>` added token (id 7, NOT eos id 1/2) followed by
a FABRICATED tool-result block - it wrote the `[web_search] success / count: 5 /
query: Kimi K3 coding capabilities comparison` header for the SECOND search it had
planned but not run - then the parser's structural close glyphs, which froze the
turn as a broken fragment. No actionable call, run over, garbage shown to the user.

**Root cause:** `formatSparkToolResponseTurn` (the agent-loop continuation turn for
Spark) ended at the Tool role turn's close. The Qwen and Ling equivalents both
re-open an assistant turn (`<|im_start|>assistant`, `<role>ASSISTANT</role>`), but
Spark's did not - the continuation prompt's last token was the Tool-turn end tag
with no generation prompt behind it (verified byte-exact in the step-1 dump).
Per Spark's own chat_template.jinja, after a tool turn the generation prompt must
be added (`add_generation_prompt`): a fresh Bot turn with the think marker. With
nothing there, the model pattern-completed the transcript instead of answering.

**Fix (three parts):**

1. `formatSparkToolResponseTurn` gained `includeAssistantPrefix` +
   `thinkingEnabled` and, when the prefix is requested, appends the exact
   generation-prompt suffix reused verbatim from the first-turn prompt builder
   (DeepSeek-style Bot reopen + think marker; the close variant when thinking is
   off). History embedding keeps the bare form.
2. The agent-loop call site passes `includeAssistantPrefix: true,
   thinkingEnabled: thinkingEnabled` (mirroring the Qwen/Ling branches).
3. The decode loop's end-tag-in-text break now also recognizes Spark's
   `<｜end▁of▁text｜>` (id 7 decodes visibly; without this the model could emit it
   mid-turn and keep generating through the marker).

**Test:** `testSparkToolResponseTurnReopensAssistantTurn` pins the continuation
form (assistant reopen AFTER the Tool-turn close, think-marker variants, no
ChatML leakage) and the bare history form. Ling/Qwen formatting tests and the
tools-section prompt regressions re-run green.

**Note:** the fabricated-result freeze path itself behaved defensively - the
broken fragment was parsed, found non-actionable, and the run ended cleanly
rather than executing a phantom call. The failure mode was output quality, not
harness integrity.


**Review hardening (PR #29, post-4c09aeb canary finding):** Copilot caught that
`testKVCachePrefixRelayoutsOnSplicedStrideShrink`'s UInt16 canary
(`0x2A00 &+ i`) wraps every 65,536 elements while both of that test's slot
strides (512 x 1024 and 384 x 1024) are multiples of that period — so every slot
carried identical bytes and the byte-exact relayout assertions would still pass
if the restore read the wrong source slot or used the new stride for source
offsets. The fill and expectation now share a `kvCanary` helper that XORs the
wrapped high index bits into the low ones
(`UInt16(i) ^ UInt16(i >> 16) ^ 0x2A00`), making each absolute element distinct
so the regression detects wrong-source relayouts, not just reallocation and tail
clearing. The sibling canary tests are not affected: their slot strides
(2048/4096/8192/16384) are not wraparound multiples, and the logical-head test
pairs its compact-layout canary with whole-buffer foreign poison
(0xDEAD/0xBEEF), so a wrong-source copy there surfaces as poison bytes. All
three scale/splice regressions re-run green.

---

## FIX #17: GEMMA 4 SLIDING-WINDOW RING KV CACHE (SHIPPED)

**Problem identified.** Gemma 4 26B A4B is 30 layers of which 25 are
`sliding_attention` (window 1024): those layers only ever attend within
`window` of the current position, yet the uniform KV layout gave every slot
`maxSeqLen` positions — 30 x maxSeq x 2048 elements x 2 (K+V) x elemBytes:
~2.0 GB at an 8k agent sequence, ~8.1 GB at 32k (fp16). The 5 global layers
are the only ones that ever read deep history. The follow-up note to the
lazy-commit fix flagged this as the remaining principled cut.

**Design.**

- Ring slots: the 25 sliding slots hold `window + slack` positions (R = 1024
  + 512 = 1536) addressed `position % R`; the 5 global slots keep the linear
  `maxSeqLen` layout. `KVCacheManager` now carries per-slot position counts
  and prefix-sum region bases (empty arrays = the old uniform layout, so
  every other model's math is byte-identical), plus helpers every dispatch
  site uses for region/scale byte offsets.
- All 8 Gemma kernels take a `ringLen` arg (0 = linear): the store and
  decode families cover fp32/fp16/fp8, and tree verification covers
  fp32/fp16 only (JetSpec excludes fp8 KV caches). The decode/tree read
  loops carry a running ring slot with a wrap check, and the fp8 scale
  index uses the same ring row.
- JetSpec tree scratch: node k lives at ring slot `(step + k) % R` — it may
  land on dead prefix rows (positions below the read window) but never on a
  live window row; the acceptance compaction blit copies ring-mapped rows.

**Invariants (all three are load-bearing; breaking any produces silent
garbage or races):**

1. **A single store batch must be strictly smaller than ringLen.** An
   in-batch wrap makes two rows of the same dispatch race on one ring row
   (measured maxErr 0.73 in a standalone repro; the win is a nondeterministic
   interleave). Hence the Gemma batched-prefill chunk cap (`gemmaPrefillChunkCap`
   = 512) and the JetSpec node count (8) both sit well under R - window.
2. **Ring slack (R - window) must cover the largest prefill chunk and the
   tree scratch** so in-flight batches never overwrite a row a later row in
   the same pass still reads.
3. **Prefix splices validate exactly, not by length.** Every store site
   records its positions in per-ring-slot last-position bookkeeping
   (`noteRingStores`); a splice is only taken when every reused row still
   holds its own position. A turn that ran past the pin boundary and wrapped
   onto a needed row rejects the splice and takes a full re-prefill (the
   same policy as the GDN "cannot rewind" rule) — this also covers failed
   turns and precision/layout changes, where a length check would happily
   splice clobbered rows. The preserve itself relocates only the last
   `window` rows, ring-mapped, for K/V and FP8 scales alike.

**Memory (fp16, gemmaKvStride 2048, 25+5 slots).** 8k agent sequence:
25x1536 + 5x8192 positions = ~650 MB vs ~2.0 GB uniform (~1.36 GB cut);
32k: ~1.66 GB vs ~8.06 GB (~6.4 GB cut); fp8 halves both. The transient
per-call prefill buffers also stop scaling with the whole delta prompt
(~250 KB/token): the chunked prefill caps them at ~128 MB.

**Bugs caught by the new tests before shipping:** the ring preserve
relocation initially added position-count bases to byte offsets (only
slot 0's zero base masked it; the layout test's canary rows caught the
shifted copies), and the first kernel-equivalence draft fed an 8-token
batch into a 6-row ring, racing the batch with itself.

**Tests.** `testGemma4SlidingRingKVLayoutAndPrefixValidation` (layout sums,
uniform fallback, bookkeeping accept/reject around a wrap, preserved-reset
relocation of exactly the live rows, bookkeeping surviving the splice);
`testGemma4SlidingRingKernelEquivalence` (ring vs linear caches: wrapped
store placement, decode through a wrapped window, tree verify whose node
slots reuse dead rows, chunked-prefill batch store + per-row attention —
all maxErr 0.0); the FIX #12 splice-canary trio re-run green byte-exact;
the real-model Gemma suite (`testGemma4VerbatimAppForward`,
`testGemma4BatchedPrefillMatchesReference` incl. fp8, `testGemma4PromptAndGeneration`)
re-run green. `testKVCacheManagerPrefixPreservation`'s stale tail-zeroing
assertions (obsolete since the lazy-commit fix removed the tail fill) were
updated to pin the retained-tail behavior.

**Follow-up (shipped same day): chunked FP8 prefill died at chunk 3 + retried
forever.** Live run on the 16 GB machine, fresh 3071-token prompt at FP8:
chunks 1-2 (positions 0..<1024) passed, chunk 3 (1024..<1536) failed with no
`❌ [METAL ERROR]` line, and the agent loop kept restarting the generation
(full re-prefill each time) — "prefill looping over again." Root cause: every
chunk re-allocated the full per-call transient set — dominated by the
double-buffered expert staging (2 x 128 slots x 11.9 MB = 3.05 GB) — while the
budget-capped weight residency grew toward its cap; by the third cycle
`device.makeBuffer` returned nil and the big per-call `guard` failed silently.
(The ring kernels themselves are exact through chunk 3: a standalone repro at
real dims — W=1024, R=1536, kvStride=2048, sliding ring + global linear, FP8
store/decode — matched the linear cache with maxDiff 0.0 and no Metal errors.)

Fix: `PrefillTransientPool` — every per-call transient in
`runLayerWisePrefill` now comes from a named pool that allocates once per
prefill (first chunk, while residency is still low) and is reused across
chunks; the generation loop drains it the moment prefill completes or fails,
so the decode phase never holds the staging. The allocation-per-chunk profile
is now one allocation per prefill (the pre-chunking profile) while the P-sized
buffers stay bounded by the chunk cap. Every previously-silent guard on the
path (the pooled buffer guard, the embed/Phase A/Phase B command-buffer
guards) now prints a named diagnostic, so any recurrence is identifiable from
the console. Test: `testPrefillTransientPoolReuseAndDrain` pins the
reuse/grow/drain contract.

Note for future runs: a `❌ [PREFILL] transient buffer '...' failed to
allocate` line means the budget-capped residency plus staging genuinely
exceeds the machine — lower the memory budget mode (8 GB => 5.5 GB cap) or
the model's resident set, not the chunking.

---

### QA #53 — the KV agent floor overruled the memory budget, so under-budget turns were accepted and thrashed until they died

Follow-up to the settings wiring (server turns now read the in-app memory
policy, QA #52's fit check) and the same 16 GB thread of observations:
Spark on `balanced16GB` budgets 0.25 GB of KV (~910 rows at the engine's
8-head shape), `neededSeqLen`'s 8,192-row agent floor deliberately won over
that budget in `clampedSeqLen` (`max(floor, min(requested, affordable))`),
so the allocator provisioned the floor anyway — roughly 9x the KV budget in
a real repro: a 8,798-token turn was accepted, ran at 4.3 s/layer, and died
at token 913 on runtime row guards, instead of failing fast with a
diagnosable message. The same floor-wins math also let a turn that passed
the fit check under the REQUESTED KV precision survive the mid-launch
Ling/Gemma fp8 -> fp16 overrides and still over-provision with the
effective (2x-byte) cache.

Fix (EngineRuntime.swift, CompletionEngine.swift): the memory budget is now
the authority, the floor is a secondary preference.

- `KVCacheProvisioning.affordableSeqLen(geometry:budgetGB:)` extracted from
  the clamp arithmetic (the pure "rows the budget can hold" number; the
  `clampedSeqLen` contract itself is unchanged for its other callers).
- `turnFit` rejects the turn outright as `.promptTooLong(provisioned:
  affordable)` when the budget cannot hold even the `minSeq` floor — pure
  math, milliseconds, before any allocation. `clampedSeqLen` never gets
  asked to raise a request above what the budget affords on this path.
- The launch-time clamp re-derives affordability from the EFFECTIVE KV
  precision (after the fp16 overrides, which the fit check cannot see) and
  throws `CompletionLaunchError.promptTooLong` when the budget cannot hold
  the session floor or the pinned prefix, plus a prompt-vs-provision guard
  for the same drift; both run before the multi-GB `KVCacheManager.reset`.

Behavior after the fix on the 16 GB M1 Pro: agent turns (8,192-row floor)
fail fast with "KV budget 0.25 GB ... cannot hold the 8192-row floor ...
affords 910 rows ... raise the memory budget mode or lower the KV cache
precision" instead of a 4.3 s/layer spiral and a dead generation; a budget
that holds the floor keeps the previous ratchet semantics intact (thin
headroom floors max_tokens at 64 rather than rejecting).

Tests: `TurnFitLogicTests` reworked — the budget-below-floor cases now
expect `.promptTooLong` with the affordable provision (910 rows), the
floor-holds cases pin their budgets to exactly the floor's cost
(0.5625 / 1.25 GB shapes) so the max-token ratchet coverage survives the
law change; new `testBudgetBelowAgentFloorFailsFastEvenForShortPrompts`
pins the fail-fast for prompts the budget could otherwise nominally fit;
`KVCacheProvisioningLogicTests.testAffordableSeqLenDerivesRowsFromBudget`
covers the helper (including the non-finite-budget no-trap guard); the
launch-side repudiation of old clampedSeqLen expectations is intentional —
`clampedSeqLen`'s own floor-wins contract tests are unchanged because the
fit/launch gates render that path unreachable for turns that allocate.
Suites: TurnFitLogicTests (9), KVCacheProvisioningLogicTests (10),
ServerQueueLogicTests (5), MemorySettingsPersistenceTests (3) — all green.

---

### QA #54 — a fresh `Date()` per server turn split the pinned prefix inside the temporal block, so every opencode step re-prefilled ~7.7K tokens

Field report ("one response takes forever"): opencode's log showed
`🔁 [PREFIX] partial match 32/815 pinned — hybrid state cannot rewind; full
re-prefill`, and every agent step of a Spark session re-prefilled the full
~8K prompt (~32 s per step, ~30 s of it prefill). The system prompt LIST
was byte-identical every turn, so the pin should have matched at ~7.7K
tokens — the divergence at token 32 was inside the FIRST sentence.

**Root cause.** The `# Current Date & Time` block sits at the head of the
effective system prompt (ModelConfig.buildEffectiveSystemPrompt ->
AgentHarness.formattedDateTimeContext) and renders minute+second precision
("Wednesday, October 15, 2026 at 3:41 PM -07:00 (2026-10-15T15:41:23-07:00)").
The interactive chat path already threads a per-conversation date
(`sessions[idx].createdAt` — cache-friendly by construction), but the server
turn planner passed a fresh `Date()` per request, so every turn's prompt
byte-diverged ~32 tokens in — inside the temporal block — and a hybrid (GDN)
session, whose pins require the ENTIRE pinned prefix to match (linear state
cannot rewind), got zero reuse for the whole conversation. Attention models
suffer the same partial-reuse loss (32 of 7,700 tokens).

**Fix (PrefixCacheManager.swift, ServerController.swift).** The temporal
context is now pinned alongside the KV prefix:

- `PrefixCacheManager` gained `asOfDate(sessionId:)` — a date
  (`pinnedAsOfDate`) owned by the session that will record the next pin.
  Same-session asks return the inherited date (byte-stable prompts across
  turns); different sessions or a cleared pin mint a fresh one;
  `invalidate(...)` clears it, so a new prefill carries fresh temporal
  context instead of a growing-stale one. When pinning is disabled it
  simply serves the current time. The class is now `nonisolated` (the
  lock-guarded singletons' existing pattern, e.g. GrammarConstrainedSampler)
  so the nonisolated server planner can call it without an actor hop, and
  the metrics struct became `nonisolated` for its initializer.
- `ServerController.buildTurnPlan` renders the temporal block from
  `PrefixCacheManager.shared.asOfDate(sessionId:)` instead of `Date()`.
  The server session key is a deterministic FNV-1a hash over (model, first
  message), so the same opencode conversation mints once and every later
  turn reproduces the identical temporal block -> the tokenized prompt
  extends the pin by only the appended turn -> `findCommonPrefix` matches
  at the full pinned length -> hybrid reuse ("cannot rewind" gate satisfied).

**Net effect.** Each opencode step now prefills only the appended
conversation delta (tens of tokens) instead of the full ~8K prompt, turning
the ~30 s step into seconds — the perceived-speed decomposition the field
report asked for. Only a full re-prefill event (pin invalidated, model or
KV-precision change) mints a new date, which is exactly when the pin is
rewritten anyway.

Tests: `PrefixAsOfDateTests` (5 — same-session stability across an ask ->
repeat-ask and an ask -> recordTurn -> ask cycle, fresh mint for a different
session, invalidation reset, disabled-pinning freshness). All pure-logic
suites re-run green (32 tests: PrefixAsOfDate 5, MemorySettings 3,
ServerQueue 5, TurnFit 9, KVCacheProvisioning 10).

---

### QA #55 — the fit check priced a sliding-ring turn as all-linear and turned the agent floor into a veto, so tools-on turns died in-app

Field report: the in-app chat on Gemma 4 26B (16 GB M1 Pro, FP8 KV) refused a
plain prompt the moment tools were on, in ~0.01 s:

`⚠️ Prompt too long: 3065 prompt tokens exceed the 2184-token KV capacity this
turn can provision (KV budget 0.25 GB on a 16 GB machine; shorten the prompt,
lower max_tokens, or raise the memory budget).`

3,065 tokens did not exceed 2,184 because the prompt was long — the message was
the wrong explanation for the right refusal. The same model had run a
multi-step CSV analysis on the same machine before QA #52/#53 shipped.

**Two defects in the QA #53 law, plus two ordering defects that fed it.**

1. **The gate charged a uniform linear layout.** `turnFit` was called before
   the Gemma sliding-ring derivation, with `slidingSlotFlags: nil,
   slidingRingLen: 0`, so all 30 slots were priced as linear: 30 x 4,096 B =
   122,880 B/row, and a 0.25 GB budget "afforded" 2,184 rows. The allocator
   (`KVCacheManager.reset`, which does get the flags) holds 25 sliding slots at
   `window + 512` rows = 1,536 regardless of sequence length and 5 global slots
   at the sequence length: `(268,435,456 - 25*1,536*4,096) / (5*4,096)` =
   **5,427 rows**. The gate demanded 2.5x the memory the allocation pays, and
   the 3,065-token agent prompt was refused against a capacity that did not
   exist. (The 2,184 in the message is the exact arithmetic of 0.25 GB over 30
   uniform slots — that number identified the code path from the field report.)
2. **The agent floor was a veto, not a preference.** `affordable (2,184) <
   minSeq (8,192)` returned `.promptTooLong` before the prompt was compared to
   the provision, so *every* tools-bearing turn on an under-floor machine was
   rejected — a 200-token prompt included. Tools-off took the 2,048-row chat
   floor, passed, and that is the entire tools-on/tools-off asymmetry the
   report described.
3. `turnFit` was passed the raw `dynamoe_memory_budget_mode` while the launch
   clamp used the resolved mode (`.unrestricted` on Auto/Full RAM), so the gate
   could be stricter than the allocator on a roomy machine.
4. The Ling/Gemma fp8 -> fp16 precision overrides ran *after* the gate, so the
   gate priced the requested precision and the allocation paid the fallback.

Fix (CompletionEngine.swift, EngineRuntime.swift, ServerController.swift):

- The ring layout (window, slack, per-slot flags, ring length) is derived
  before the fit check and passed into `turnFit`, so the estimate and the
  allocation price the same rows. The ring-intact prefix validation stays where
  it was (it needs `prefixTokensReused`).
- `KVCacheProvisioning.sessionFloor(isAgentSession:configured:)` resolves the
  reserve (8,192 agent / 2,048 chat, overridable through
  `dynamoe_agent_session_floor`), and `affordableFloor` caps it to
  `affordableSeqLen`. A turn is refused only when the prompt itself (plus the
  64-row slack) does not fit the provision; the budget is still never exceeded,
  which is QA #53's actual invariant.
- The launch-side re-gate now defends only the pinned prefix (the one hard
  floor the relayout cannot drop) plus the same prompt guard, and both gates
  share one message that reports budget, model footprint, precision and the
  affordable rows instead of claiming the prompt is longer than the cache.
- The gate uses the resolved budget mode and runs *after* the precision
  overrides, so both sides of the launch see one budget and one precision.
- The server maps this rejection onto `400` with
  `"code": "context_length_exceeded"` (new `ServerAPIError.code`, previously a
  hard-coded nil) so clients can classify it instead of pattern-matching prose.

**Numbers on the field machine** (16 GB, Gemma 4 26B, FP8 KV, 0.25 GB budget,
3,065-token in-app agent prompt): before the fix, rejected with
`.promptTooLong(2,184)`; after it, provisioned at 5,427 rows and generation
ratcheted to 1,850 tokens. Tools-off is unchanged (2,048-row chat floor). The
honest ceiling on this box is ~5.4k rows: the model's `size_gb` (26 GB) exceeds
physical RAM in every budget mode, so `budgetGB` sits at its 0.25 GB floor.
Budgeting the *resident* streaming footprint instead of the on-disk size is the
follow-up that would restore ~8k contexts, and it changes eviction pressure
globally, so it is deliberately not part of this fix.

Tests: `TurnFitLogicTests` reworked to the derived-floor law, plus
`testAgentFloorIsCappedByTheBudgetForShortPrompts`,
`testSessionFloorOverrideReplacesOnlyTheAgentReserve`, and the field pair
`testUniformGeometryOverpricesGemmaSlidingRingTurn` (2,184 rows, rejected) /
`testRingAwareGeometryServesTheFieldPrompt` (5,427 rows, 1,850-token cap);
`KVCacheProvisioningLogicTests.testAffordableSeqLenPricesSlidingRingsAsFixedRows`
pins ring vs uniform affordability and the ring byte layout.

Review follow-ups (same change): the rejection text moved out of `launchTurn`
into `KVCacheProvisioning.capacityMessage`, because a nested builder could not
be tested and the server test could only assert against a literal it wrote
itself. The message now names a reserve the budget capped ("agent sessions
reserve 8192 rows, capped by the budget to 910") instead of implying the
requested reserve is what was provisioned, and renders `Int.max` affordability
(all-ring layouts) as "unbounded" rather than a 19-digit row count.
`sessionFloor` also clamps `dynamoe_agent_session_floor` to the 32,768-row
`maxProvisionedRows` ceiling, so the override can lower or raise the reserve
inside the range the sequence cap allows but can no longer request a provision
larger than any other path on the machine. New coverage:
`testCapacityMessageBlamesTheBindNotThePromptLength`,
`testCapacityMessageNamesACappedReserve`,
`testCapacityMessageRendersUnboundedCapacityAndChatRemedy`, and the server test
now builds its message through the real builder.

## DSV4.1 #1: BLOCK PREFILL VS THE WALK (MEASURED, NOT YET WORTH BATCHING MORE)

**Why this entry exists before any more work on the path.** `prefillChunk` now
runs a prompt as blocks of positions: the dense stages (norms, linears, rope)
dispatch once over the whole block, while the cache write, the attention that
reads it, the mHC coefficients, the Engram rows and the routed mixture stay per
position. That is a claim about cost. This journal already records
optimizations shipped on the strength of such a claim and reverted once
somebody measured them (Fix #6a, Fix #7), so the claim got a number first.

**Setup.** `DeepSeekV41SyntheticCheckpoint` is the only runnable model in the
repo — the local DeepSeek-V4.1 shards are 14 MB header-only stubs of a ~510 GB
checkpoint, so nothing runs at published widths. Synthetic shape: hidden 64,
head_dim 48, 2 layers, 2 heads in 2 output groups, 4 routed experts taking 2,
no compressed layers, no Engram tables, FP16 KV cache. Prompt 32 positions,
best of 3 runs after 1 warmup, one `reset()` per run. Machine: **M5 Max, 128 GB,
macOS 27.2** (not the M1 Pro the budgets are tuned for). Reproduce with:

```bash
touch /tmp/dynamoe_dsv41_prefill_measure
xcodebuild test-without-building -project DynaMoE.xcodeproj -scheme DynaMoE \
  -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO \
  -only-testing:DynaMoETests/DeepSeekV41PrefillMeasurementTests
```

| window | slots | walk ms | chunk ms | speedup | walk ms/pos | chunk ms/pos |
|-------:|------:|--------:|---------:|--------:|------------:|-------------:|
| 2 | 2 | 31.10 | 24.72 | 1.26x | 0.972 | 0.773 |
| 4 | 4 | 21.17 | 21.20 | 1.00x | 0.662 | 0.663 |
| 8 | 8 | 23.46 | 21.27 | 1.10x | 0.733 | 0.665 |
| 16 | 8 | 25.91 | 25.11 | 1.03x | 0.810 | 0.785 |

(An earlier run of the same build gave 1.41x / 1.05x / 1.10x / 1.08x, so
anything under ~1.15x is inside this harness's noise.)

**Attribution.** A block costs 0.346 ms/position at one layer and 0.680 at two:
**0.334 ms per position per layer**, linear in the layer count. A block of one
position costs 0.934 ms, which is two layers of that plus the embedding block
and the head. Scaling with layers means the cost is work the block still does
per position; the block's fixed cost is gone. What remains per position per
layer is one command buffer, one submission and one `waitUntilCompleted` for
the expert arena — the mixture cannot batch because `ensureResident` may only
refill a slot after the kernels reading it have finished
(`DeepSeekV41MetalForward.runBlockLayer`).

**Conclusion.** The dense batching is real but small at these widths: it removes
dispatches that were never the floor. **The floor is the per-position mixture
round trip**, roughly 85% of a block's per-position time. On a checkpoint whose
arithmetic is 7,168-wide that round trip is a smaller share, so the real
speedup at published widths is *larger* than these numbers and still not
knowable here. Two honest consequences:

- Do not flip the `batchedDevicePrefill` conformance gate on the strength of
  this: it is port-against-port parity, not a number from real weights.
  **Updated in `#33`:** the gate was flipped on a real-weight comparison, not on
  this synthetic one (`#33`, "The real-weight comparison").
- The next lever on this path is batching the **mixture** — an arena that holds
  a block's expert union (or deduplicates residency across the block's
  routings), so one submission covers the block instead of one per position.
  That is an arena-size and streaming question, not a wider grid, and it is
  worth ~0.33 ms/position/layer here before any real weights are involved.

**Caveats worth keeping in mind.** The walk pays a per-position logits readback
(`forwardToken` finishes a block of one each time) and the chunk reads logits
only at the end; that difference is part of the measured win and is included
rather than netted out. Varying the window changes the ring, so a row's
walk-vs-chunk ratio is the only comparison valid within this table. The
synthetic model has no compressed layers, no Engram and no DSpark stages, so
none of their per-position costs are in these numbers.

Tests: `DeepSeekV41PrefillMeasurementTests` (three opt-in cases: the width
sweep, the per-layer attribution, and block-of-one versus block-of-N). They
skip unless `/tmp/dynamoe_dsv41_prefill_measure` exists or
`DYNAMOE_DSV41_PREFILL_MEASURE=1`, because `xcodebuild` does not forward the
shell environment into an app-hosted test, and they assert only the correctness
invariant (the block returns the walk's logits) rather than a timing threshold.

## DSV4.1 #2: BATCHING THE MIXTURE, AND WHAT THE ROUND TRIP REALLY COST

**What changed.** `DSV4.1 #1` measured the block prefill and named the
per-position mixture round trip as the floor: `runBlockLayer` created one
command buffer per position, submitted it, and waited, because
`DeepSeekV41ExpertStreamer.ensureResident` may only refill an arena slot after
the kernels reading it have finished. `DeepSeekV41MetalForward.runBlockLayer`
now fills the arena once for the block's deduplicated expert union and encodes
every position's mixture stages into **one** command buffer with no wait
between positions. The wait moves to the block boundary: the half is committed
unordered, and the next layer's front-half wait drains it — the same ordering a
single decode position already relies on. When the union does not fit the arena
(a deliberately small one), the layer keeps the old per-position fill-and-wait,
because a slot overwritten under a running kernel is a wrong answer with no
error.

**Setup.** Unchanged from `DSV4.1 #1`: synthetic checkpoint (hidden 64, 2
layers, 4 routed experts taking 2, FP16 KV, no compressed layers / Engram /
DSpark), 32 positions, best of 3 after 1 warmup, one `reset()` per run. Machine
**M5 Max, 128 GB, macOS 27.2**. Same reproduce command
(`DeepSeekV41PrefillMeasurementTests`, opt-in via
`/tmp/dynamoe_dsv41_prefill_measure`).

**Before → after, same synthetic model.**

| window | slots | walk ms | chunk ms | speedup | chunk ms/pos before | chunk ms/pos after |
|-------:|------:|--------:|---------:|--------:|--------------------:|-------------------:|
| 2 | 2 | 27.56 | 13.36 | 2.06x | 0.773 | 0.418 |
| 4 | 4 | 20.57 | 10.24 | 2.01x | 0.663 | 0.320 |
| 8 | 8 | 21.85 | 11.12 | 1.97x | 0.665 | 0.347 |
| 16 | 8 | 25.22 | 14.04 | 1.80x | 0.785 | 0.439 |

Two further runs of the same build: 2.08x / 1.99x / 2.00x / 1.80x and 2.11x /
2.04x / 2.02x / 1.77x. The walk times moved by ~2% between sessions, so the
walk column is stable enough for the ratio to be read; the spread inside the
after column is under 2%.

Per-layer attribution, same case: a block of 32 positions costs **0.185
ms/position at 1 layer and 0.317 at 2** (three runs: 0.181/0.319, 0.181/0.324),
so **~0.138 ms per position per layer**, down from 0.334. A block of one
position costs 0.551-0.751 ms, down from 0.934.

**Correction to `DSV4.1 #1`.** That entry inferred the round trip was "roughly
85%" of a block's per-position cost, from the layer slope plus the block-of-one
number. The direct before/after says **~50-60%**: at window 4 the block went
from 0.663 to 0.320 ms/position (52% removed), and per layer from 0.334 to
0.138 (59% removed). The inference was directionally right and numerically
optimistic; this entry's number is the one to quote.

**What is still per position** (~0.138 ms/position/layer here): the dense
half's per-slot stages (`hcAttentionCoefficients`, the mHC pre-mix, the
compressor writes, the gathered rows, the cache write, and the attention that
reads it are all encoded per slot inside one command buffer), plus the
mixture itself, which is still encoded per position: a clear, two dispatches
per routed expert, the shared expert, and the hyper-connection write and
collapse. Those are *dispatches*, not round trips: batching them further would
need the kernels to take a position index where they currently take a slot
pointer (`DeepSeekV41DevicePlan.chunkSlots` stays the ceiling at 8). Not
measured separately, so not recommended yet — the next step on this path is
another measurement, not another rewrite.

**Caveats.**

- This run had **no streamer**: the synthetic runtime is built with
  `snapshotDir: nil`, so the experts come from the shard mapping and the
  measurement isolates the round trip and the command-buffer count. On real
  weights the same change also replaces `count` fills of top-k experts with one
  fill of the block's union, which is a hit-rate question this harness cannot
  see (at 384 experts and top-k 6 the union of 8 positions is up to 48, and the
  arena the memory plan sizes holds hundreds, so it fits).
- The arena-union path itself is unexercised by any test: the synthetic arena is
  8 slots for 4 experts *and* the test runtime has no streamer, so the union
  always fits and the fallback never runs. The fallback is the old code path,
  kept verbatim.
- The `batchedDevicePrefill` conformance gate still stays `.unverified`: this is
  port-against-port parity on a synthetic model, with no real-weight number
  anywhere on the machine.
  **Updated in `#33`:** it is now `.verifiedAgainstReference`, on the published
  checkpoint's own weights; this bullet describes the state before `#33`.
- Everything from `#1` about the walk paying a per-position logits readback,
  window rows not being comparable to each other, and the absent compressed /
  Engram / DSpark stages still holds.

Tests: `DeepSeekV41DeviceGraphParityTests.testTheChunkSeamReturnsTheWalksLogits`
is the guard — it walks `2 * window - 1` tokens and requires the block path to
return bit-identical logits (worst delta 0, asserted at 1e-6), and every parity
case asserts its tolerance through the same batched path. Full DeepSeek suite
after the change: 399 tests, 73 classes, 0 failures.

## DSV4.1 #3: WHERE THE BLOCK'S REMAINING TIME ACTUALLY GOES

**Why another measurement.** `DSV4.1 #2` batched the mixture and halved a
block's per-position cost to ~0.14 ms/position/layer, then left the remainder
unattributed. Two explanations were on the table and they want opposite fixes:
host time spent encoding dispatches (fewer and wider dispatches), or device
time spent running them (fewer or cheaper kernels). Guessing here is what this
journal exists to prevent, so the executor grew a switch.

**Method.** `DeepSeekV41MetalForward.recordsBlockTiming` marks four points per
layer — layer start, dense half encoded, dense half waited on, mixture encoded
— and accumulates host nanoseconds per bucket. That split still cannot say
which HALF the device time belongs to, because the dense half's wait also
absorbs whatever the queue had not finished (the previous layer's mixture), so
`DeepSeekV41BlockTiming` also sums the command buffers' own
`gpuStartTime`/`gpuEndTime`. Then `recordsStageTiming` changes the LAYOUT, not
the work: each dense stage is committed in its own command buffer (the queue's
submission order and every hazard it protects are unchanged), so every stage
reports its own device window. Finally a `calibration` stage dispatches `count`
trivial clears, which is what turn "device window" into "overhead" versus
"work".

Both switches are off in every production path; the cases that use them are
opt-in like the rest of this file's measurements. Machine **M5 Max, 128 GB,
macOS 27.2**, synthetic checkpoint (hidden 64, 2 layers, window 4, block width
4), prompt 32 positions, best of `1 + 3` runs.

**Host split, ms per position per layer** (three runs):

| bucket | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| dense half encoded (host) | 0.0237 | 0.0235 | 0.0241 |
| dense half waited on | 0.2064 | 0.2101 | 0.1480 |
| mixture encoded (host) | 0.0186 | 0.0188 | 0.0181 |

Host encode is ~17-22% of that total. The rest is blocked on the device.

**Device time, ms per position per layer** (same runs):

| half | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| dense | 0.1709 | 0.1707 | 0.1187 |
| mixture | 0.0307 | 0.0303 | 0.0215 |

**The dense half is 85% of the device time**, in every run. So the mixture is
no longer the lever — it is ~15% — and batching more of it would buy little.

**Inside the dense half** (instrumented layout, two runs, share of dense device
time):

| stage | ms/position | share |
|---|---:|---:|
| `writeLatent` (cache write + pool + gather + attention) | 0.155-0.166 | 42% |
| `hcFFNCoefficients` (mHC project + split) | 0.060-0.067 | 16.5% |
| `hcAttentionCoefficients` (the same pair) | 0.060-0.064 | 16.4% |
| `outputDown` (block-diagonal projection) | 0.017 | 4.6% |
| `router` | 0.013 | 3.4% |
| everything else (norms, GEMVs, rope, mHC pre/post) | ≤0.006 each | ≤1.7% each |

Two honesty notes about that table. The per-stage layout inflates absolute
numbers: the same dense half measured 0.17 ms/position in one buffer and 0.36-
0.39 spread over 22 buffers, so read the shares, not the milliseconds. And
`calibration` — four trivial dispatches in one buffer — costs 0.004
ms/position (16 µs per layer-block, ~4 µs per dispatch), which means dispatch
overhead alone does NOT explain the mHC stages: each of those two stages costs
~30x the calibration for two dispatches that compute a handful of numbers.

**What that points at.** `dsv41_hc_split_f32` returns immediately for every
thread except `tid == 0` — the whole split and its Sinkhorn schedule run on one
GPU thread, and the block pays that serial chain `2 * count` times per layer.
`hc_project` is `mixCount` (8) threads doing two 128-element passes each, which
is the same shape of problem. The fix is not a wider grid over one position; it
is a slot dimension, so the block's positions run their serial chains in
parallel. `writeLatent` is the largest stage and the least tractable: the cache
write, the gather and the attention are interleaved because the ring tears if a
position's successor writes before its query reads (see
`DeepSeekV41Conformance.batchedDevicePrefill`), so it needs its own reasoning
before it is touched.

**Caveats.** The synthetic widths make fixed per-dispatch cost dominate the
arithmetic, so this ranking is a ranking of per-position WORK ITEMS, not a
prediction at 7,168 hidden: `outputDown` and the mHC projections both grow with
hidden, while their dispatch counts do not. Nothing here is measured at real
weights, because the local shards are header-only stubs.

Tests: `DeepSeekV41PrefillMeasurementTests.testWhereABlocksRemainingTimeGoes`
(the three host buckets and the two device halves) and
`DeepSeekV41PrefillMeasurementTests.testWhichDenseStageHoldsTheDeviceTime` (the
stage table plus the calibration), both opt-in and both asserting the
correctness invariant — the block must still return the walk's logits through
the instrumented paths — rather than a timing threshold.

## DSV4.1 #4: A SLOT DIMENSION ON THE mHC PAIR AND THE ROUTER

**What changed, and why it was the next thing.** `DSV4.1 #3` split a block's
remaining cost and found the dense half holding 85% of the device time, with
the two mHC coefficient stages at 33% of it. The reason is visible in the
kernels: `dsv41_hc_split_f32` returns immediately for every thread except
`tid == 0`, so a block paid the whole Sinkhorn schedule once per position per
sub-layer, and `dsv41_router_topk` is the same shape (one thread scans every
expert). `dsv41_hc_project_f32` was a `mixCount`-wide grid at one position.

All three now take a slot dimension — `thread_position_in_grid.y` for the
projection, `.x` for the split and the router — and the executor dispatches
them once per layer instead of once per position, with the slot applied as an
offset inside the kernel. The arithmetic, the Sinkhorn schedule and the router
rule are untouched; only the grid and the addressing changed. A decode step
passes one slot, so the walk runs the same kernels it always did.

**Before → after, same harness** (synthetic checkpoint, 32 positions, best of
3 after a warmup, block widths from the ring; three runs after the change):

| window | slots | speedup before | speedup after | chunk ms/pos before | chunk ms/pos after |
|-------:|------:|---------------:|--------------:|--------------------:|-------------------:|
| 2 | 2 | 2.06x | 1.91-1.99x | 0.418 | 0.442-0.458 |
| 4 | 4 | 2.01x | 2.39-2.44x | 0.320 | 0.273-0.282 |
| 8 | 8 | 1.97x | 2.50-2.56x | 0.347 | 0.284-0.293 |
| 16 | 8 | 1.80x | 2.10-2.15x | 0.439 | 0.377-0.389 |

Per position per layer the block went from 0.133-0.143 ms to 0.117-0.121 ms.
Blocks wider than two positions gained 15-25%; **width 2 is slightly WORSE**,
which is what a two-wide grid should do — there is almost nothing to
parallelize, and the kernels pay one more constant to read.

**The attribution moved a long way, and not only where expected.**

| measure (ms per position per layer) | before | after |
|---|---:|---:|
| dense half encoded (host) | 0.0237 | 0.0174 |
| dense half waited on | 0.2064 | 0.0879 |
| mixture encoded (host) | 0.0186 | 0.0127 |
| dense device | 0.1709 | 0.0646 |
| mixture device | 0.0307 | 0.0159 |
| `hc{Attn,FFN}Coefficients` (per-stage layout) | 0.060-0.067 each | 0.009 each |
| `router` | 0.013 | 0.004 |
| `writeLatent` | 0.155-0.166 | 0.079 |

The dense half's device time fell 2.6x, which is more than the 33% the mHC pair
held before. The explanation is that a single-thread dispatch is not only slow
itself, it stalls everything queued behind it: `writeLatent` was never touched
and still halved, and the host wait fell further than the device work did. So
the honest reading is "removing one-thread dispatches unblocks the queue", and
a per-stage table measured around such a stage attributes its neighbours' lost
time to them. Read each table as a ranking, and the whole-block wall as the
result.

**What is left.** `writeLatent` — the cache write, the row gather and the
attention — is now 54% of a dense half that is itself 2.6x cheaper, at 0.079
ms/position/layer. It cannot be batched the same way: the ring tears if a
position's successor writes before its predecessor's query reads, and the row
plan (which rows a position attends to) is resolved on the host per position.
It needs its own reasoning, so it stays the open item.

**Caveats.** Synthetic widths again: at hidden 64 the mHC projections are a
handful of multiplies, so part of what this change removed is fixed per-dispatch
cost that a 7,168-wide model would amortize on its own. What transfers is the
serialization — a one-thread Sinkhorn per position and a one-thread expert scan
per position are the same cost at any width. Nothing here is measured at real
weights.

Tests: the existing mHC kernel and graph parity cases, plus
`DeepSeekV41DeviceGraphParityTests.testTheHyperConnectionCoefficientsDispatchOneSlotPerPosition`,
which drives two slots with different stacks through the project and split
kernels and compares each against `DeepSeekV41HyperConnections.coefficients` —
the failure mode a slot grid introduces is a slot reading its neighbour's
stack. `testTheChunkSeamReturnsTheWalksLogits` still requires bit-identical
logits from the batched path. Full DeepSeek suite: 402 tests, 0 failures.

## DSV4.1 #5: THE ATTENTION KERNEL WAS AS WIDE AS THE HEAD COUNT

**What changed, and why the harness had to leave the synthetic model.** `DSV4.1 #4`
left `writeLatent` as the largest per-position stage of a block, at 0.079
ms/position/layer. Reading it says the cost is not distributed the way the other
stages are: `dsv41_attention_csa2_f32` was dispatched with
`MTLSize(width: shape.numHeads)` and one thread per head, and that thread walked
every gathered row THREE times (maximum, denominator, output) over a 512-wide latent
record. On the published checkpoint a compressed layer reads `window + index_topk` =
128 + 512 rows, so a head was a serial chain of roughly 1.3 million multiply-adds and
the entire kernel was 128 threads wide.

That shape cannot be timed by the synthetic checkpoint, which is two heads over at
most a handful of rows: its attention is a dispatch, and a dispatch is what the
harness already spends most of its budget on. So this slice added a case that times
the kernel alone, at the geometry `config.json` publishes — 128 heads, `head_dim`
512, `v_head_dim` 512, a 512-wide cache record (`kv_lora_rank` is absent from the
checkpoint, so the record is the head) — for the three row counts a position can
have: the 128-row window, a window with a few compressed rows, and the 640-row
worst case of a full compressed cache.

The case is self-validating the same way the block cases are: before it prints a
number it compares two heads' outputs with `DeepSeekV41AttentionReference.attend` at
1e-4 relative, so the timing is never reported for a different computation.

**What a single dispatch cost, before** (one command buffer, `gpuStartTime` /
`gpuEndTime`, best of 5-20 dispatches; M5 Max, 128 GB, macOS 27.2):

| rows | ms/dispatch | µs per head |
|---:|---:|---:|
| 128 | 13.479 | 105.3 |
| 136 | 14.326 | 111.9 |
| 640 | 68.308 | 533.7 |

One position of one compressed layer, on an idle GPU, in a kernel that is 0.2% of
the die. The arithmetic does not explain 68 ms — 1.3 million multiply-adds is a
microsecond at any lane — and that is the point: there were 128 of them in flight,
each paying a strided scalar load per multiply-add.

**What it costs now.** One threadgroup per head, one thread per row for the dot
products, and the row scores kept in threadgroup memory so the three passes share ONE
dot product per row instead of recomputing it:

| rows | before | after (128-thread threadgroup) | speedup |
|---:|---:|---:|---:|
| 128 | 13.479 | 0.092-0.095 | ~145x |
| 136 | 14.326 | 0.120-0.124 | ~118x |
| 640 | 68.308 | 0.453-0.466 | ~148x |

The threadgroup width is a measured choice rather than a guess; the three widths
tried, same case, same machine:

| rows | 128 threads | 256 threads | 512 threads |
|---:|---:|---:|---:|
| 128 | 0.095 | 0.087 | 0.318 |
| 136 | 0.124 | 0.097 | 0.092 |
| 640 | 0.466 | 0.489 | 0.450 |

512 buys 3% on the worst case and loses 3.4x on the 128-row one (128 heads' worth of
idle threads still occupy warp slots), and 256 is inside the run-to-run noise of 128
on both ends. 128 threads per head is four SIMD groups, which covers a 640-row
attention's rows and a 512-wide record's dimensions with room to spare, and it is
what ships.

**What the block pass made of it** (same harness as `#2`-`#4`: synthetic checkpoint,
32 positions, best of 3 after a warmup, block widths from the ring):

| window | slots | speedup before | speedup after | chunk ms/pos before | chunk ms/pos after |
|-------:|------:|---------------:|--------------:|--------------------:|-------------------:|
| 2 | 2 | 1.91-1.99x | 1.62-2.68x | 0.442-0.458 | 0.291-0.352 |
| 4 | 4 | 2.39-2.44x | 2.44-2.96x | 0.273-0.282 | 0.193-0.205 |
| 8 | 8 | 2.50-2.56x | 3.14-3.93x | 0.284-0.293 | 0.149-0.154 |
| 16 | 8 | 2.10-2.15x | 3.11-3.73x | 0.377-0.389 | 0.150-0.160 |

The synthetic model is two heads wide, so this is the effect of giving each head a
threadgroup at a width where the rows are nearly free: the win is the parallelism,
not the arithmetic, and it is the same win at 7,168 hidden. Widths 8 and 16 are
steady across every run; widths 2 and 4 move with the machine's state (one run of
the same binary reported a 97.89 ms walk for window 4 while the other runs reported
15-19 ms, so the spread above is the honest range rather than the best number).

**The stage table moved again, and is now close to its own floor.**

| stage (ms/position, per-stage layout) | before `#5` | after `#5` |
|---|---:|---:|
| `writeLatent` | 0.079 | 0.041 |
| `outputDown` | — | 0.026 |
| `hc{Attn,FFN}Coefficients` | 0.009 each | 0.015 each |
| `calibration(4 dispatch)` | 0.004 | 0.011 |
| `router`, the norms, the mHC pre/post mixes | 0.003-0.006 | 0.003-0.012 |

Read that as a ranking only, and note what happened to the calibration row: the
trivial four-dispatch buffer used to cost 0.004 ms/position and now costs 0.011, and
every norm and every mHC pre/post mix lands on exactly 0.011. The per-stage layout
commits one command buffer per stage, and with the work below it collapsed that
per-buffer cost IS the measurement. The honest reading is that `writeLatent` (0.041)
and `outputDown` (0.026) are the two stages still measurably above the floor; the
`hc*Coefficients` rows rising from 0.009 to 0.015 in a table that measures less work
than before is the floor moving, not the stages getting slower.

**What is left.** The per-position remainder of a block: the cache write, the row
gather and the mHC pre/post mix, each still one dispatch per position inside the
block; `outputDown`, which loops `encodeBlockDiagonalProjection` over slots; and the
`writeLatent` interleave itself, which cannot be batched without either a staging
copy of the block's own rows or a wider ring, because writing position `p + 1` evicts
a row a query at `p` still needs. None of it is measured at real weights, because the
local shards are header-only stubs.

**Caveats.** The attention numbers are device time from one command buffer, so they
exclude the per-dispatch latency a block pays in a real submission; they are the part
of the cost that does not change with how the buffer is built. The block numbers
above are the synthetic widths and the machine was contended during part of the run
(a clean run and a loaded run disagree by 2.4x on the walk), which is why the table
carries ranges. And the kernel's row loop is memory bound at 640 rows: 168 MB of key
records are read per dispatch across the heads, so the remaining headroom there is
about how many heads share a loaded row, not about arithmetic.

Tests: `DeepSeekV41PrefillMeasurementTests.testWhatOneAttentionDispatchCostsAtThePublishedWidths`
(the timing above, opt-in, gated on the marker file and asserting the host rule
first), `DeepSeekV41AttentionKernelTests.testTheAttentionKernelMatchesTheHostRuleAcrossItsWholeThreadgroup`
(200 rows across four SIMD groups with a value head wider than the record, so the
cross-group reduction and the tail zeroing are both exercised),
`DeepSeekV41AttentionKernelTests.testASingleRowAttentionIsThatRow` (a row count far
below the threadgroup width), `DeepSeekV41DeviceGraphParityTests.testTheChunkSeamReturnsTheWalksLogits`
(bit-identical logits from the batched path) and
`testTheDeviceAndTheHostAgreeStageByStage`. Full DeepSeek suite: 405 tests, 6
skipped, 0 failures.

## DSV4.1 #6: SHARING A GATHERED ROW ACROSS HEADS MADE ATTENTION SLOWER

**The idea, and why it looked free.** `#5` left the attention kernel with one
threadgroup per head and recorded, as its remaining headroom, "how many heads share
a loaded row". The rows are there to be shared: `workspace.gathered` is ONE dense
block that every head reads, so at the published geometry a dispatch reads 640 rows
of 512 floats — 1.3 MB of distinct rows — once per head, 168 MB in total. A
threadgroup carrying `g` heads, with one thread holding `g` dot products through a
single pass over the row and `g` accumulators through a single pass over a value
dimension, divides that traffic by `g` and leaves the arithmetic per head identical
(it still sums over dimensions in order, so the result is bit-identical).

That was implemented, measured at the same published geometry as `#5`, and
reverted. The measurement is below because the premise — that those 168 MB are
memory traffic worth removing — is the part that turned out to be wrong, and the
next person to read "how many heads share a loaded row" should know that.

**What it cost** (same case as `#5`: one command buffer, `gpuStartTime`/
`gpuEndTime`, best of 5-20 dispatches, M5 Max 128 GB macOS 27.2; `G` is the heads
per threadgroup, and the register depth is how wide the unrolled accumulator loop
compiles):

| rows | shipped `#5` kernel | rewrite, G=1, depth 8 | rewrite, G=1, depth 2 | rewrite, G=2, depth 2 | rewrite, G=4, depth 8 |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.092-0.095 | 0.240 | 0.117 | 0.137 | 0.292 |
| 136 | 0.120-0.124 | 0.348 | 0.158 | 0.187 | 0.428 |
| 640 | 0.453-0.466 | 1.189 | 0.755 | 0.683 | 1.455 |

Two things are in there, and the second is the interesting one.

**The register state cost 2.5x before the group did anything.** The first rewrite
carried a fixed eight-deep unrolled accumulator loop (so the per-head accumulators
stay in registers rather than local memory), and `G=1` — the shipped behavior — came
out 2.5x SLOWER than the kernel it replaced: 0.240 ms at 128 rows against 0.095,
1.189 at 640 against 0.466. Eight live accumulators, eight maximums, eight
denominators and eight inverses per thread is a register bill that costs resident
threads, and attention at this shape is latency-bound enough that occupancy shows up
directly. Narrowing the unroll to two brought `G=1` back to 0.117 / 0.158 / 0.755,
still slower than shipped because a two-wide predicated inner loop does two loads
and two multiply-adds per dimension where the shipped kernel does one.

**The traffic did not matter.** After paying that overhead, `G=2` at the narrowed
depth was 17% WORSE at 128 rows (0.137 against 0.117) and only 10% better at 640
(0.683 against 0.755). Halving the record traffic moved the 640-row time by 10%,
which is the answer: those 168 MB were never DRAM traffic, they were served from the
cache hierarchy, so removing half of them bought a tenth of the kernel. The group
also divides the threadgroup COUNT by `g` — 64 threadgroups at `G=2` — and that
costs occupancy at exactly the widths where the kernel was already fast.

A kernel templated on a compile-time group width would remove the predication and
project to about 0.41 ms at 640 rows (9% better than shipped) and 0.111 ms at 128
(17% worse), for the price of template entry points, more than one pipeline and a
dispatch-side choice. Nine percent on the worst row count against seventeen on the
common one is not a trade worth a second kernel, so the rewrite was reverted and
the shipped kernel stands.

**What this closes.** The "share a loaded row" branch of the `#5` follow-up. The
remaining per-position cost of a block is the ring-interleaved `writeLatent` (cache
write + pool + gather + attention, 0.041 ms/position), which cannot be batched
without a staging copy of the block's own rows or a wider ring, and `outputDown`
(0.026), whose eight head-group projections cannot be merged into fewer dispatches
because each group reads different weight bytes. Neither is a traffic problem of the
kind measured here.

**Caveats.** The numbers come from the rewrite, which is no longer in the tree: the
published-geometry case in the tree measures the shipped kernel, and reproducing a
`G` row of the table means re-applying the change. Same instrument caveats as `#5`:
device time from one command buffer, so submission latency is excluded, and the
machine was contended during part of the run.

Tests: unchanged from `#5` — 405 tests, 6 skipped, 0 failures. The rewrite was
verified before it was timed (`DeepSeekV41AttentionKernelTests` extended to dispatch
the kernel with G=1, 2, 4 and 8 against the host rule, partial tail group included),
so the table above is a timing of a correct kernel, not of a broken one.

## DSV4.1 #7: THE STAGE TABLE WAS RANKING A TWO-HEAD MODEL

**Why the ranking had to be re-taken.** `#5` and `#6` both closed with the same two
names — `writeLatent` at 0.041 ms/position and `outputDown` at 0.026 — and both
numbers came from a synthetic checkpoint that is TWO heads wide over a four-row
window at hidden 64. A ranking of per-position stages taken from that model is a
ranking of a different model: it has no 512-wide record to re-read, no 65,536-wide
attention output to project down, and no four-copy residual stack to mix.

So the harness got a second shape, `DeepSeekV41SyntheticShape.publishedWidths`:
7,168 hidden, 128 heads of 512 (448 nope + 64 rope, so the cache record is the head
width the published config derives), 8 output groups with `o_lora_rank` 1,024,
`q_lora_rank` 1,536, `hc_mult` 4 with 20 Sinkhorn iterations, and a 640-row window so
the ring hands attention the same row count a compressed layer sees. One layer, a
128-token vocabulary, `moe_intermediate_size` 32 and four routed experts keep it
holdable — the mixture was batched in `#2` and is not what this table ranks. That
makes the DENSE half the shipped half down to the byte, which is what a per-stage
ranking needs.

**What one position of one layer costs at those widths** (`testWhereABlocksRemainingTimeGoesAtThePublishedWidths`,
640-position prompt, block of 8, per-stage command buffers, M5 Max 128 GB macOS 27.2,
wall 3.22 ms/position of which 3.24 ms is dense device time and 0.42 mixture):

| stage | ms/position | share |
|---|---:|---:|
| `outputDown` | 1.864 | 57.6% |
| `hcAttentionCoefficients` | 0.391 | 12.1% |
| `hcFFNCoefficients` | 0.350 | 10.8% |
| `writeLatent` | 0.249 | 7.7% |
| `queryUp` | 0.097 | 3.0% |
| `latentNorm` | 0.083 | 2.6% |
| `outputUp` | 0.081 | 2.5% |
| `queryDown`, `latentProject`, `router` | 0.024 each | 0.7% each |
| the norms, ropes, mHC mixes, calibration | 0.001-0.011 | below 0.3% |

**The finding.** `outputDown` is 57.6% of the dense half, and it was ranked seventh
at 0.026 ms/position on the synthetic model — wrong by a factor of about seventy.
The cause is a shape the synthetic model cannot express: `wo_a` is
`o_lora_rank=1,024` by `attentionOutputWidth=65,536`, split into eight head groups,
and `encodeBlockDiagonalProjection` is dispatched inside `eachSlot`, so each of a
block's eight positions re-reads the entire 134 MB weight matrix in BF16 through 64
GEMV dispatches (eight groups by eight slots). 1.864 ms for 134 MB per position is
about 72 GB/s of weight traffic, and the weights are identical for every position of
the block.

**Why that is the next lever, and why the earlier arithmetic was wrong.** `#6`'s
close-out said `outputDown` "cannot be merged into fewer dispatches because each
group reads different weight bytes". That is true and it was the wrong measure: the
cost is not the dispatch, it is the weight BYTES, and those are re-read once per
POSITION rather than once per BLOCK. The fix is not fewer dispatches — it is a grid
height of `count` on each of the eight per-group GEMVs, which reads every weight byte
once per block instead of eight times. `queryUp` shows what that is worth on the same
table: `wq_b` is 201 MB and already takes a `rows:` height, so it reads 25 MB per
position and costs 0.097 ms — 258 GB/s of effective weight traffic, 3.5x the
per-slot path's.

The one thing the batched form needs is a ROW STRIDE on the GEMV kernels: within a
block the input rows are contiguous at `attentionOutputWidth` stride while a group
reads `perGroupIn` = 8,192 of those columns, so the kernel has to be told the stride
rather than deriving it from `inDim`.

**Caveats.** The synthetic weights are BF16; the published checkpoint's attention
projections are quantised, so the real byte count per position is smaller — at FP8
`outputDown` would be about 0.93 ms/position, at MXFP4 about 0.47, both still the
largest term by a wide margin, and the 8x re-read factor is independent of dtype.
The `hc*Coefficients` rows are each one FP32 GEMV of 24 rows by 28,672 columns, and
were never measured at these widths before either: 0.74 ms/position between them is
the second-largest term. They are already batched over the block — the projection
takes a grid height of `slots` — so their cost is not repeated weight traffic: 2.75 MB
of weights per block arriving in 0.39 ms is about 1 GB/s, which says the grid, not
memory, is the limit. `hc_mult` 4 makes the output width 24, so the kernel dispatches
24 threads and each one walks a 28,672-wide reduction serially. That is the same
class of defect `#5` fixed in attention: a grid as wide as a small output dimension,
with the whole reduction in one lane.

Attention reads the window half of its row budget here (no compressed layers in the
synthetic config), which is why `writeLatent` is 0.249 and the compressed-row half is
measured separately by `#5`/`#6`. The case takes ~40 s because
the synthetic blob is 190M values; it is opt-in like the rest. Device time is one
command buffer per stage, so submission cost is excluded.

Tests: `testWhereABlocksRemainingTimeGoesAtThePublishedWidths` (the table above,
opt-in, so it is the seventh skipped case in a default run) plus the
`publishedWidths` shape preset it runs on; every pre-existing case unchanged — 406
tests, 7 skipped, 0 failures.

## DSV4.1 #8: ONE DISPATCH PER HEAD GROUP PER BLOCK, NOT PER POSITION

**What the published-widths table said.** `#7` ranked the dense half at the widths
the checkpoint runs and found `outputDown` at 1.864 ms/position — 57.6% of it —
because `encodeBlockDiagonalProjection` was dispatched inside the per-slot loop:
eight head groups by eight positions, each dispatch reading that group's whole weight
slab, so every position re-read all 134 MB of BF16 `attn.wo_a`.

**The change.** Each group's projection now runs ONCE for the block, with the grid's
height carrying the positions. The GEMV kernels gained two row strides
(`bf16_gemv` at buffers 6 and 7, `dsv41_fp8_block32_gemv` and `dsv41_mxfp4_gemv` at 8
and 9), because a band of a block-diagonal projection reads `perGroupIn` = 8,192 of a
row that is `attentionOutputWidth` = 65,536 wide and writes `perGroupOut` = 1,024 of
an `outputReducedWidth` = 8,192 vector: `inDim`/`outDim` no longer describe the row
layout the way they do for a dense projection. A stride of zero means "the dimension",
so every other caller of those kernels is unchanged — including the generic engine
path in `ContentView`, which now binds the two strides explicitly rather than relying
on that fallback.

**What it cost before and after** (same case, same machine, 640-position prompt,
block of 8, per-stage command buffers):

| stage | before | after | change |
|---|---:|---:|---:|
| `outputDown` | 1.864 | 0.214 | 8.7x |
| whole block, ms/position | 3.22 | 1.57 | 2.05x |
| dense device, ms/position | 3.236 | 1.581 | 2.05x |

`outputDown` is now 13.5% of the dense half instead of 57.6%, and a block's per
position cost halved. The 8.7x matches the eightfold traffic reduction the change was
made for, which is the check that the cost really was the re-read and not something
else that moved with it.

**The table it leaves** (ms/position, published widths):

| stage | ms/pos | share |
|---|---:|---:|
| `hcAttentionCoefficients` | 0.393 | 24.8% |
| `hcFFNCoefficients` | 0.350 | 22.1% |
| `writeLatent` | 0.248 | 15.7% |
| `outputDown` | 0.214 | 13.5% |
| `queryUp` | 0.095 | 6.0% |
| `latentNorm` | 0.081 | 5.1% |
| `outputUp` | 0.079 | 5.0% |

The mHC coefficient projections are now the largest term between them, at 0.74
ms/position — 47% of the dense half — and `#7` already described why: `hc_mult` 4
makes the output 24 wide, so `dsv41_hc_project_f32` dispatches 24 threads and each
walks a 28,672-wide reduction serially. That is a grid shape, not a traffic problem,
and the fix is the one `#5` applied to attention: make the reduction parallel across
a threadgroup and keep the partials in threadgroup memory.

**Caveats.** The synthetic weights are BF16, so the absolute `outputDown` numbers are
larger than the checkpoint's own (its attention projections are quantised): the win
is the eightfold re-read that the change removes, and the fraction of a block it
represents scales with the dtype. `writeLatent` reads only the window half of its row
budget here (no compressed layers), so its 0.248 is a floor rather than the guarantee.
The case is opt-in and takes ~40 s to build the 190M-value blob. Device time is one
command buffer per stage.

Tests: the batched path is pinned by `DeepSeekV41DeviceGraphParityTests` —
`testTheDeviceAndTheHostAgreeStageByStage` and
`testTheChunkSeamReturnsTheWalksLogits` (bit-identical) — and the whole DeepSeek suite
is green: 406 tests, 7 skipped, 0 failures.

## DSV4.1 #9: THE mHC PROJECTION KEPT ITS REDUCTION IN ONE LANE

**The shape of the defect.** `#7` left the two hyper-connection coefficient
projections as the largest term of a block's dense half — 0.74 ms/position between
them, 47% of it — and `dsv41_hc_project_f32` showed why in its dispatch rather than
its traffic. The projection is `(2 + mult) * mult` logits over the flattened residual
stack, which at `hc_mult` 4 is 24 logits over `mult * hidden` = 28,672 values per
position. It ran one THREAD per logit, and each thread walked that reduction twice:
once for the RMS statistic and once for the dot product. So a position was 24 lanes,
against 2.75 MB of weight rows to read, and the block already handed it a grid height
of `slots` — the parallelism that was missing was inside the reduction, not across
positions.

**The change.** One THREADGROUP per (slot, logit). Lanes stride the reduction,
accumulate both operands in ONE pass over the stream, `simd_sum` in each warp, and the
partials meet in threadgroup memory for a serial finish by lane 0. The grid is flat
(`mixCount * slots` threadgroups, row-major) because a kernel's built-in position
attributes are either all scalar or all vector, and the lane indices are scalars — a
`uint2` threadgroup position next to a `uint` thread id does not compile.

**The kernel alone** (`testWhatOneHyperConnectionProjectionCostsAtThePublishedWidths`,
2.75 MB of weights over 8 slots, one command buffer, best of 20, self-checking three
logits against the host rule before it prints):

| threadgroup | ms/dispatch | ms/position |
|---:|---:|---:|
| 24 (one thread per logit) | 2.253 | 0.282 |
| 32 | 0.188 | 0.023 |
| 64 | 0.088 | 0.011 |
| 128 | 0.045 | 0.006 |
| **256** | **0.031** | **0.004** |
| 512 | 0.034 | 0.004 |

73x on the dispatch at 256 threads, which is what the checkpoint's shape affords:
28,672 values over 256 lanes is 112 each, and the accumulators are two floats, so the
width costs nothing until the threadgroup outgrows the reduction. 512 is 10% worse and
250 wider for no gain.

**What it did to the table** (published widths, `#7`'s case, same machine):

| stage | `#8` | now |
|---|---:|---:|
| `hcAttentionCoefficients` | 0.393 | 0.011 |
| `hcFFNCoefficients` | 0.350 | 0.012 |
| dense device, ms/position | 1.581 | 0.775 |
| whole block, ms/position | 1.57 | 1.22 |

The dense half of a block is now 2.04x cheaper than before this slice, and the whole
block — dense plus a mixture that is deliberately tiny in the synthetic model — is
1.22 ms/position against 3.22 two slices ago.

**The table it leaves** (ms/position): `writeLatent` 0.246 (30.3%), `outputDown` 0.213
(26.1%), `queryUp` 0.096 (11.8%), `outputUp` 0.078 (9.7%), the calibration buffer
0.038, `router`/`queryDown`/`latentProject` 0.024 each, then everything else at or
under the floor. Note the calibration row: it is a buffer of eight no-op dispatches and
it read 0.001 before this slice and 0.038 after, which is what a per-stage layout costs
in submission latency once the work around it is gone. Anything at or below that row in
this table is the instrument, not the stage.

**What is left, and it is all one defect.** `outputDown` (0.213) now reads its 134 MB
of weights once per block, which is right, but at 79 GB/s it is still three times off
what `queryUp` gets from the same kind of traffic (262 GB/s at 0.096 ms/position over a
201 MB matrix). Both are single-thread-per-output-row GEMVs whose reduction runs in one
lane, which is exactly what this slice fixed in the mHC projection — `wq_b` is 65,536
output rows over a 1,536-wide reduction, so its lanes are the output and the serial part
is short, while `wo_a`'s bands are 1,024 rows over 8,192, where it is not. Widening the
GEMV reduction is therefore the next slice, and it touches `bf16_gemv` — which the
generic engine path also uses — rather than a DeepSeek-only kernel.

**Caveats.** Same as `#7`/`#8`: BF16 synthetic weights (the checkpoint's attention
projections are quantised, so the real mixture of costs differs while the ranking holds),
no compressed layers so attention reads the window half of its row budget, one layer, and
the mixture's 0.417 ms/position is dispatch overhead rather than expert arithmetic because
the synthetic intermediate is 32. Device time is one command buffer per stage. The case is
opt-in and takes ~40 s.

Tests: `testWhatOneHyperConnectionProjectionCostsAtThePublishedWidths` (the kernel table
above, opt-in) and the three direct kernel dispatches of `dsv41_hc_project_f32` in
`DeepSeekV41HyperConnectionKernelTests` / `DeepSeekV41DeviceParityTests` now dispatch
threadgroups with 256 threads and the reduction scratch. Whole DeepSeek suite: 407
tests, 8 skipped, 0 failures — including `testTheDeviceAndTheHostAgreeStageByStage` and
the bit-identical chunk seam, so the reordered reduction produces the same coefficients.

## DSV4.1 #10: THE GEMV PUT ONE THREAD ON A WHOLE OUTPUT ROW

`bf16_gemv` gives a thread an output row and the reduction is a serial loop, so the grid
is `outDim * rows` threads and nothing else. That is the right shape when there are a
lot of rows — `attn.wq_b` is 73,728 of them over a 1,536-wide reduction, and its lanes
are the output, so the machine is covered. It is the wrong shape when there are very few
and the reduction is wide: `attn.wo_a` is eight bands of 128 rows over 8,192 columns, so
a block of eight positions is **1,024 threads in total** on a 40-core GPU, and `attn.wkv`
is 448 rows, which is 3,584. `#9` had just shown what one lane per reduction costs in the
mHC projection (73x). This is the same defect in five stages.

**The two forms.** `bf16_gemv_rows` (new) gives an output row a threadgroup: lanes stride
the reduction, `simd_sum` per warp, partials in `[[threadgroup(0)]]`, lane 0 finishes. The
grid is flat (`outDim * rows`) so `row = groupIdx % outDim` and the token is
`groupIdx / outDim` — the same flat-grid constraint `dsv41_hc_project_f32` ran into,
where a `threadgroup` buffer cannot be declared beside vector grid attributes. Striding
the reduction across lanes also makes a lane's load adjacent to its neighbours' instead
of 16 KB away, which is part of what the one-thread form pays for. Both kernels take the
same two row strides, so a band-driven caller binds them and every other caller leaves
them zero. The choice is `DeepSeekV41DevicePlan.bf16GemvWantsWideGrid(outDim:rows:)` — a
plan rule, not a constant — and the shipped `bf16_gemv` is untouched, because the app's
generic decode path dispatches it and no measurement here covers that path.

**Kernel alone, published shapes, one block of eight positions** (`testWhatOneProjection
CostsAtThePublishedWidths`, self-validated against a host dot product at 1e-4 before
timing). Weights are hammered 20x per window, so these are cache-warm:

| stage | shape | one thread per row | threadgroup per row | |
|---|---|---:|---:|---:|
| `latentProject` (`attn.wkv`) | 448 x 7,168 | 0.0340 | **0.0065** | 5.2x |
| `outputDown`, one band (`attn.wo_a`) | 128 x 8,192 | 0.1416 | **0.0132** | 10.7x |
| `queryDown` (`attn.wq_a`) | 1,536 x 7,168 | 0.0419 | **0.0227** | 1.8x |
| `outputUp` (`attn.wo_b`) | 7,168 x 1,024 | **0.0080** | 0.0145 | 0.6x |
| `queryUp` (`attn.wq_b`) | 73,728 x 1,536 | **0.1637** | 0.3911 | 0.4x |

and at a batch height of one, every shape prefers the threadgroup form, because a grid
one row tall never has a second position to amortize a row's bytes over: 20.2x on
`latentProject`, 22.1x on `outputDown`'s band, 6.2x on `queryDown`, 1.6x on `outputUp`,
1.4x on `queryUp`. So the rule has two clauses — a decode step takes the wide grid at any
row count, and a block takes it up to 8,192 threads.

**The crossover, measured rather than guessed** (8,192-column reduction, eight
positions, so the row count is the only variable):

| rows | threads | one thread per row | threadgroup per row | |
|---:|---:|---:|---:|---:|
| 96 | 768 | 0.0182 | **0.0013** | 13.9x |
| 192 | 1,536 | 0.0183 | **0.0022** | 8.3x |
| 384 | 3,072 | 0.0184 | **0.0034** | 5.4x |
| 768 | 6,144 | 0.0226 | **0.0099** | 2.3x |
| 1,536 | 12,288 | **0.0230** | 0.0252 | 0.9x |
| 3,072 | 24,576 | **0.0281** | 0.0567 | 0.5x |
| 6,144 | 49,152 | **0.0676** | 0.1654 | 0.4x |

The threshold sits at 8,192 threads: between a 2.3x win and a 10% loss. A 64-lane
threadgroup with only a few float4 loads each spends more time at its barrier than on the
reduction, which is why the tall shapes lose.

**Lane width** at the same shapes: 64 and 128 are within noise of each other, 32 costs
the narrow bands up to 11%, 256 up to 50%. 64 shipped.

**What it did to the table** (published widths, `#7`'s case, same machine):

| stage | `#9` | now |
|---|---:|---:|
| `outputDown` | 0.213 | 0.145 |
| `latentProject` | 0.024 | 0.012 |
| `queryDown` | 0.024 | 0.025 |
| `queryUp` | 0.096 | 0.096 |
| `outputUp` | 0.078 | 0.079 |
| dense device, ms/position | 0.775 | 0.674 |
| whole block, ms/position | 1.22 | 1.12 |

The in-situ gain is far smaller than the kernel table suggests, and the reason is worth
writing down: the isolated case hammers one warm weight 20 times, while a stage inside a
block reads weights that fell out of cache since the last block, and each stage's number
in the block table includes the tail of its predecessor (they run back to back in one
command buffer). `outputUp` reads 0.008 alone and 0.079 in place, which is not the stage
being slow — it is the stage waiting. Treat the isolated table as a ranking of the two
kernel forms and the block table as the authority on what a stage costs.

`queryDown` and `queryUp` did not move because they kept the shipped kernel by the rule
above, so the 0.9x crossover row is what protects them.

**The table it leaves** (ms/position): `writeLatent` 0.246 (34.4%), `outputDown` 0.145
(20.3%), `queryUp` 0.096 (13.4%), `outputUp` 0.079 (11.0%), the calibration buffer 0.042,
`queryDown` 0.025, `latentProject` 0.012, then `hcFFNCoefficients`/`hcAttentionCoefficients`
/`ffnNorm`/`attnNorm` 0.011 each, which is the floor. Dense device 0.674, mixture 0.416 —
the mixture is now 38% of device time on a model whose expert arithmetic is fake.

**What is left.** Two things, and both are visible in this table. `outputDown` is still
14.5 GB/s of weight traffic even after the fix, because it is **eight dispatches** of one
band each and the calibration row says eight dispatches cost 0.042 ms/position on their
own; a single flat grid over all eight bands would remove seven launches, and the
threadgroup-per-row form makes that a small change rather than a redesign (the band
becomes arithmetic on the flat index, exactly as the slot does in `dsv41_hc_project_f32`).
`writeLatent` at 34.4% is still the ring's serialization and still needs a design
decision before it can be touched.

**Caveats.** Same as `#7`-`#9`: BF16 synthetic weights, no compressed layers, one layer,
a synthetic mixture that is dispatch overhead. Plus one new one — the isolated kernel
numbers are cache-warm and are not additive into the block table. Device time is one
command buffer per stage. The case is opt-in and takes ~25 s, and it now sweeps both
batch heights and the crossover, so it is self-validating at every configuration it
times.

Tests: `testWhatOneProjectionCostsAtThePublishedWidths` (both tables above, opt-in),
`testTheWideGemvGridIsChosenByHowMuchWorkTheProjectionHas` (the plan rule, pure logic)
and `testTheWideGemvGridKeepsTheTokenAndTheRowStrides` (the batched flat grid and the two
row strides, against a host block-diagonal product). Whole DeepSeek suite: 410 tests, 9
skipped, 0 failures — including `testTheDeviceAndTheHostAgreeStageByStage` and the
bit-identical chunk seam, so the stages that moved to the second kernel produce the same
logits.

## DSV4.1 #11: ONE GRID FOR EVERY BAND DID NOT BUY ANYTHING

`#10` left `outputDown` at 0.145 ms/position in a table whose calibration row — eight
no-op dispatches — was 0.040. The projection is eight band dispatches of `attn.wo_a`, so
the obvious next move was to collapse them: `bf16_gemv_rows` already derives the row and
the token from a flat index, so a third term (`band = withinToken / rowCount`) makes the
whole block-diagonal projection one grid, and eight launches become one. It was
implemented, self-validated against the host block-diagonal product, measured, and
**reverted**.

**What it measured.** Isolated, published shapes, one block of eight positions:

| form | ms/position | weight traffic |
|---|---:|---:|
| eight wide-grid dispatches, one per band | **0.0133** | 158 GB/s |
| one grid over all eight bands | 0.0172 | 123 GB/s |

The single grid is 29% SLOWER. At a batch height of one it is the other way round and by
more: 0.0518 against **0.0231**, 2.24x better, because eight launches of 1,024
threadgroups each cannot overlap their ramp with each other the way one 8,192-threadgroup
grid can. So the form is a trade, block against decode — and blocks are what this table
ranks.

**What it did not measure.** In place, in the published-widths table, the projection went
from 0.145 to **0.146 ms/position** and the whole block stayed at 1.12. Eight launches
collapsed into one and nothing moved. That is the finding worth keeping: `outputDown`'s
0.145 is not launch count.

**Which means the calibration row is not a floor you can subtract.** It reads 0.040 for
eight dispatches that do nothing, and the tempting arithmetic is "eight launches cost
0.040, so collapsing them saves a third of the stage". A stage with real work hides its
launches under that work. The calibration row prices the per-stage command-buffer layout
for a stage that has no work to overlap with, and nothing else.

**What the stage is actually made of.** The same projection reads 16.8 MB of weights and
costs 0.0133 isolated against 0.145 in place — a factor of eleven that neither kernel form
touches. The whole dense half of a block moves 286 MB in 5.39 ms, which is 53 GB/s, while
the isolated kernels reach 150-300 GB/s on the same bytes. Whatever that gap is — cold
weights, DRAM latency rather than bandwidth, the shared-storage buffer being read through
the same path as its writes — it is not the dispatch shape, and it is bigger than every
remaining dispatch-level item in the table combined.

**Caveats.** Same warnings as `#10`: the isolated numbers are cache-warm (20 dispatches
over one weight in one command buffer) and are not additive into the block table; the
synthetic weights are BF16; the mixture is fake. The banded form's grid was
self-validated before it was timed, so neither number is measuring the wrong computation.

**What this closes.** "Collapse the eight band launches" is off the remaining-work list,
and with it the idea that the calibration row is recoverable time. `#10`'s eight-band loop
is what ships. What is left in the dense half is `writeLatent` 0.246 (34.4%, still the
ring's serialization) and the eleven-fold gap between isolated and in-place projection
cost, which needs a measurement of its own before it needs a change.

Tests: unchanged from `#10` — the experiment touched `bf16_gemv_rows`,
`encodeBlockDiagonalProjection` and two test files, and all four were reverted with
`git checkout --`; only this entry is committed. Whole DeepSeek suite: 410 tests, 9
skipped, 0 failures.

## DSV4.1 #12: THE PROJECTION WAS NOT ELEVEN TIMES ITS ISOLATED COST, IT WAS 2.2

`#11` ended on an anomaly: the attention output projection measured 0.0133 ms/position in
isolation and 0.145 inside the block, and the whole dense half appeared to move its
weights at 53 GB/s against 150-300 GB/s for the isolated kernels. Both numbers were
wrong, and they were wrong in a way a fixture can be wrong: the isolated case priced
"eight bands" by timing one 2.1 MB band EIGHT TIMES.

Fixed, with the geometry held still and only the number of distinct bands growing, so the
two throughputs separate — requests per second, which a warm working set can serve from
cache, against distinct bytes per second, which the memory system actually has to deliver:

| bands | distinct | ms/position | GB/s unique | GB/s requested |
|---:|---:|---:|---:|---:|
| 1 | 2.0 MB | 0.0087 | 30 | 242 |
| 2 | 4.0 MB | 0.0167 | 31 | 251 |
| 4 | 8.0 MB | 0.0314 | 33 | 268 |
| **8** | **16.0 MB** | **0.0660** | **32** | **254** |
| 16 | 32.0 MB | 0.1381 | 30 | 243 |
| 32 | 64.0 MB | 0.1204 | 70 | 557 |
| 64 | 128.0 MB | 0.2447 | 69 | 549 |

So the real isolated cost of the projection is **0.0660**, not 0.0133, and the in-place
figure is **2.2x** that rather than 11x. The cost is linear in the number of bands (eight
bands is 7.6x one band, for 8x the work) and it does not care about the footprint: 30-33
GB/s unique holds from 2 MB to 32 MB with no cliff, which rules out the cache story `#11`
reached for. The jump at 32 bands is the loop being long enough to amortize the dispatch
ramp, not a memory effect.

**What the stage is actually limited by.** Every position re-reads the same weight row:
eight positions over one band is 16.8 MB of requests for 2.1 MB of weight, and that is
where the 254 GB/s goes. The kernel is request-bound, not DRAM-bound — 30 GB/s of unique
bytes is what 254 GB/s of requests buys when eight of them are the same row served from
L2 rather than L1, because the eight threadgroups that want a row land on different cores.

Per stage, in place, as distinct bytes per second (`weights / (8 x msPerPosition)`):
`queryUp` 294 GB/s, `queryDown` 110, `latentProject` 67, `outputUp` 23, `outputDown` 14.5.
The spread is not a rank of kernel quality — it is what fraction of each stage's traffic a
row re-read inflates. `queryUp` reads 226 MB of weight per block, so its eight-fold
redundancy is a small multiple of a big number and it lands near peak DRAM; `outputDown`
reads 16.8 MB and its redundancy is the whole cost. So `#11`'s "the dense half runs at 53
GB/s" is also withdrawn: pooled over the stages that dominate, the dense half moves
254 MB at 240 GB/s, and the two small stages are slow for their own reason.

**What it opens.** The re-read is removable and the shape of the reuse is the opposite of
`#6`'s: there, sharing a loaded record row across heads needed eight deep accumulators and
cost 2.5x in occupancy. Here the shared operand is the WEIGHT and the per-position state is
one float — a threadgroup could take one output row for every position of the block, load
each weight value once, and keep `slots` accumulators, cutting the request traffic eight
times for eight registers. That is the next slice, and it is measurable the same way.

**Caveats.** Cache-warm repetitions again; BF16 synthetic weights; one layer; a 32-wide
synthetic mixture. The footprint rows 32 and 64 use a 134 MB buffer and are the only ones
whose working set exceeds what the part can cache, and they are the FASTEST per byte,
which is the strongest evidence that the footprint is not the variable.

Tests: `testHowAStagesCostScalesWithTheWeightItReads` (the table above, opt-in, ~2 s).
Whole DeepSeek suite: 411 tests, 10 skipped, 0 failures.

## DSV4.1 #13: A BLOCK'S PROJECTIONS NOW READ EACH WEIGHT ROW ONCE

`#12` ended on a description of the defect rather than a fix: every position of a block
re-reads the same weight row, so `attn.wo_a`'s eight bands ask for 134 MB of loads to
read 16.8 MB, and the stage settles at the request rate. The reuse it suggested is the
opposite register trade to `#6`'s: there, sharing a loaded record row across heads needed
eight deep per-head accumulator state and cost 2.5x in occupancy; here the shared operand
is the weight and the per-position state is ONE float.

`bf16_gemv_rows8` gives a threadgroup one output row and streams it once, keeping eight
named accumulators — named because an array indexed by a runtime bound does not stay in
registers, and eight is `chunkSlots`' ceiling. `DeepSeekV41DevicePlan.bf16GemvForm` picks
between it, `bf16_gemv_rows` and `bf16_gemv`: an eight-position block always takes the
block form, a shorter batch has no block kernel and falls back to `#10`'s two-way rule
(decode prefers the wide grid at any row count).

**Isolated, published shapes, one block of eight positions** (self-validated against a host
dot at 1e-4 for all three forms before timing):

| stage | shape | one thread per row | threadgroup per (row, position) | threadgroup per row |
|---|---|---:|---:|---:|
| `latentProject` | 448 x 7,168 | 0.0264 | 0.0038 | **0.0023** |
| `queryDown` | 1,536 x 7,168 | 0.0222 | 0.0220 | **0.0058** |
| `queryUp` | 73,728 x 1,536 | 0.1119 | 0.3833 | **0.0486** |
| `outputUp` | 7,168 x 1,024 | 0.0080 | 0.0146 | **0.0029** |
| `outputDown`, one band | 128 x 8,192 | 0.1462 | 0.0154 | **0.0095** |
| `crossover` | 6,144 x 8,192 | 0.0688 | 0.1668 | **0.0369** |
| `crossover` | 96 x 8,192 | 0.0182 | 0.0014 | **0.0012** |

The block form wins at every shape measured, from 96 rows to 73,728, which is why it needs
no threshold: 1.6x to 3.8x over whichever form shipped before it. The row-count crossover
that `#10` had to find disappears at an eight-position batch, because the weight is read
once whatever the row count is.

**What it did to the table** (published widths, same case):
| stage | `#10` | now |
|---|---:|---:|
| `queryUp` | 0.096 | 0.046 |
| `outputDown` | 0.145 | 0.046 |
| `outputUp` | 0.079 | 0.044 |
| `queryDown` | 0.025 | 0.013 |
| `latentProject` | 0.012 | 0.014 |
| `writeLatent` | 0.246 | 0.247 |
| dense device, ms/position | 0.674 | 0.480 |
| whole block, ms/position | 1.12 | **0.93** |

Dense device time is 1.40x cheaper and the whole block 1.20x, because the dense half is no
longer most of it: the mixture is 0.417 against 0.480 now, so the deliberately-tiny
synthetic mixture is 46% of device time and further dense wins buy little until it stops
being fake.

**The table it leaves.** `writeLatent` is **0.247 ms/position and 51.2% of the dense half**
— it was 34.4% before this slice purely because the stages around it got cheaper. Every
other dense stage is now at or near the floor, `queryUp`/`outputDown`/`outputUp` at 0.044-0.046
having been 0.079-0.145.

A note on the floor: the calibration row read 0.040 in `#10`'s table and reads 0.003 here,
with no change to the calibration dispatch. It is a per-stage layout cost, and it moves with
whatever else is in flight, so it has to be read off the same table as the stages rather than
carried between them.

**Caveats.** The isolated numbers are cache-warm, and one fixture quirk survives: the
`outputDown` row times one band eight times, so its absolute figures are optimistic for both
forms — the ratio is still measured like for like. BF16 synthetic weights (the checkpoint
quantises its attention projections), one layer, no compressed layers, a 32-wide synthetic
mixture.

Tests: `testTheBlockGemvSharesOneLoadAcrossEightPositions` (all eight positions of all rows
of a banded projection against the host product, with the group offsets a block-diagonal
caller binds), `testTheWideGemvGridIsChosenByHowMuchWorkTheProjectionHas` extended to the
three-way choice, and `testWhatOneProjectionCostsAtThePublishedWidths` grown a third column.
Whole DeepSeek suite: 412 tests, 10 skipped, 0 failures — including
`testTheDeviceAndTheHostAgreeStageByStage` and the bit-identical chunk seam.

## DSV4.1 #14: THE CACHE LOOP IS THE ATTENTION, AND NOTHING ELSE SHOWS UP

After `#13` the published-widths table reads `writeLatent` 0.247 ms/position, **51.2% of
the dense half**, with everything else at or near the floor. `writeLatent` is one loop per
position doing four things — store the latent row into the ring, pool it into the
compressed cache when its group completes, gather the row budget, run attention — and
until now nobody had said which of them the 0.247 belongs to.

**Answered without new code, from a number that already existed.** `#5`'s
`testWhatOneAttentionDispatchCostsAtThePublishedWidths` times this kernel alone at the
published geometry (128 heads, head_dim 512, record 512):

| window rows | ms/dispatch | us/head |
|---:|---:|---:|
| 128 | 0.092 | 0.72 |
| 136 | 0.126 | 0.99 |
| 640 | 0.453 | 3.54 |

A 640-position prompt's window is `min(position + 1, 640)` rows, which averages ~320, and
between 136 and 640 rows the cost is ~0.65 us/row. So one position's attention is about
**0.245 ms**, eight of them are 1.96 ms per block, and `writeLatent` is 1.976. The store
(448 halves), the gather (at most 640 rows of 512) and the loop's own bookkeeping are
inside the noise floor of that comparison.

**One of the four costs nothing at all, here.** `encodePoolIfComplete` returns
immediately when the layer does not own a compressed KV cache, and the synthetic
published-widths checkpoint has no compressed layers. That is a caveat with teeth: on a
real compressed layer the pool runs TWO GEMVs per position inside the loop
(`attn.compressor.wkv` and `attn.compressor.wgate`, 7,168 wide by 448), so this table
understates `writeLatent` for exactly the layers the checkpoint's second half is made of.
The synthetic model cannot price that, and no change here should be justified by this
table's pool term being invisible.

**So the lever is the attention's SHAPE, not the loop's plumbing.** One position is one
dispatch of 128 threadgroups — 16,384 threads, a fifth of what the part can hold — and
eight of them run back to back in one command buffer with a gather in front of each. The
gathers cannot be hoisted: a gather for position p must follow the store for p (a causal
query reads its own row) and precede the store for p + 1 (which evicts the oldest row p
still needs), so the only way to launch the block's eight attends at once is eight gathered
buffers, 5.2 MB at 640 rows of 512 half-precision values.

**What to measure before building that.** The whole premise is that more threadgroups per
launch is cheaper than more launches, and that is a two-line test on the existing case:
time one dispatch at 8x the heads against eight dispatches at 1x, with the same rows and
the same total work. If the batched launch is not clearly cheaper, eight gathered buffers
and a slot dimension on the attention kernel buy nothing.

**Caveats.** The attention numbers are cache-warm repetitions of one 640-row key block;
the in-situ row counts differ per position and the estimate above interpolates between two
measured row counts rather than measuring an average window. Same synthetic-weight and
one-layer caveats as `#13`. Device time is one command buffer per stage.

Tests: none added — the table above is `#5`'s existing opt-in case, re-read against the
current table. Whole DeepSeek suite: 412 tests, 10 skipped, 0 failures.

## DSV4.1 #15: ONE ATTENTION LAUNCH PER BLOCK, AND THE ROW LIST IT EXPOSED

`#14` priced `writeLatent` at 0.247 ms/position — 51.2% of a block's dense half — and
attributed nearly all of it to the attention, then named the next measurement: one
dispatch at 8x the grid against eight dispatches at 1x, same rows, same total work. The
premise was that a position is 128 threadgroups, "a fifth of the machine", and a block
runs eight of those back to back.

**The measurement first, and it could not be done without the kernel.** The kernel returns
early when its threadgroup index is past the head count, so widening the grid measures
nothing. So `dsv41_attention_csa2_f32` grew a slot dimension before anything was wired:
with `slotPlan` bound (buffer 9) the grid is `slotCount * headCount`, `slot = groupIdx /
headCount`, `head = groupIdx % headCount`, and slot `s` reads its gathered block at row
`s * keyRowStride`. The slot-major query and output offsets are then still `groupIdx *
qkHeadDim` and `groupIdx * valueDim`, and the head part of the index selects the sink.
Buffer 10 holds one row count per slot, because a block's positions do not read the same
number of rows. Leaving `slotPlan` unbound is the old form, so nothing else changed.

**The A/B** (`testWhetherOneBatchedAttentionLaunchBeatsEight`, opt-in): published geometry,
8 slots with ramping row counts (128, 192, … 576 — 2,816 rows, 352 per position), eight
DISTINCT gathered blocks in both arms, both arms validated against
`DeepSeekV41AttentionReference.attend` at 1e-4 before anything is printed.

| form | ms/block | ms/position | |
|---|---:|---:|---:|
| 8 launches of 128 threadgroups | 2.118 | 0.265 | |
| 1 launch of 1,024 threadgroups | **0.964** | **0.120** | **2.20x** |

0.265 ms/position is also the first direct confirmation of `#14`'s 0.245: the arithmetic
that interpolated ~0.65 us/row over an average 320-row window lands within 8%.

**The bug the slot dimension could not avoid.** A gather's row list is a host fill into
`workspace.rowIndices`, and a host fill is not ordered against a dispatch that has not
executed yet: every gather in a stage is encoded, and only then is the command buffer
committed. While that buffer had ONE region for the whole block, every gather in a block
read the list written for the block's LAST position. It is invisible in everything
normally looked at — attention writes only that position's own stream, and
`prefillChunk` samples only the last position's logits — and it is invisible even to a
parity test for two more reasons: with the synthetic default (window 4, block 4) the
blocks are window-aligned, so a successor's list is a permutation of the rows its
predecessor needs and attention does not care about row order, and the probe that would
have shown it was wired only by the walk, never by the block path. So the block path now
writes the row list into its own per-position slice of `rowIndices` and gathers into its
own per-position block of `gathered` (both `workspace.gatheredRowOffset(slot)` apart), and
the block path's `.writeAttentionStreams` now honours `parityProbes.afterAttention` for
slot 0.

`testEveryPositionOfABlockAttendsOverItsOwnRows` pins it with a ring WIDER than the block
(10 over 8, the checkpoint's own ratio is 128 over 8), three blocks, and a comparison of
the last block's slot 0 against the walk at that position. With the row list temporarily
put back on one shared region the case fails at 2.1e-3 against a 1e-5 bar; with the
per-position region it passes. 2.1e-3 is small because the wrong rows are neighbours of
the right ones and the softmax barely moves — the reason this needs a test rather than a
number.

**What it did to the table** (published widths, same case as `#13`/`#14`, same machine):

| stage | `#13` | now | change |
|---|---:|---:|---:|
| `writeLatent` | 0.247 | **0.112** | 2.2x |
| dense device, ms/position | 0.480 | **0.345** | 1.39x |
| whole block, ms/position | 0.93 | **0.79** | 1.18x |

`writeLatent` is 2.2x cheaper, which is the isolated A/B's own factor — the check that the
in-situ win is the launch shape and not something that moved with it. It is now 30.4% of
the block instead of 51.2%, and the mixture (0.416, unchanged) is 55% of the device half.

**The table it leaves** (ms/position, whole block 0.79; dense device 0.3448, mixture
0.4163): `writeLatent` 0.112 (30.4%), `queryUp` 0.047 (12.7%), `outputDown` 0.045 (12.3%),
`outputUp` 0.044 (12.0%), the calibration buffer 0.024, `latentProject` 0.014, `queryDown`
0.012, then `hcFFNCoefficients`/`hcAttentionCoefficients`/`ffnNorm`/`attnNorm` 0.011 each
and everything else at 0.005 or under. The calibration row moved 0.003 → 0.024 with no
change to its dispatch, which is `#11`'s warning repeating: it is an instrument, not a
stage.

Host split: dense encode 0.0699, front wait 0.2624, mixture encode 0.0114 — encode is now
24% of accounted time.

**What is left in the cache loop.** The gathers are still one dispatch each, and by `#14`'s
constraint they cannot be hoisted past a store: gather for p must follow store p and
precede store p + 1. Nothing measured says eight gather launches cost anything — they are
512-wide by `rowCount` threads each — but the same argument that made the attention worth
batching has not been tested on them, and now that each gather owns its own destination
block in `gathered`, one slot-aware gather would be a strict drop-in if it did.

**Caveats.** Same as `#13`/`#14`: BF16 synthetic weights (the checkpoint's attention
projections are quantised, so absolute numbers differ while rankings hold), no compressed
layers so the pool term is invisible and the positive margin is real but unmeasured, one
layer, and the mixture's 0.416 ms/position is dispatch overhead because the synthetic
intermediate is 32. The isolated A/B is cache-warm and its two arms share one key buffer
allocation, so it ranks the two forms and is not additive into the block table. Device
time is one command buffer per stage. Both cases are opt-in (~7 s and ~37 s).

Tests: `testWhetherOneBatchedAttentionLaunchBeatsEight` (the table above, opt-in) and
`testEveryPositionOfABlockAttendsOverItsOwnRows` (the regression). Whole DeepSeek suite:
414 tests, 11 skipped, 0 failures, including `testTheChunkSeamReturnsTheWalksLogits` at
bit-identical logits and `testTheDeviceAndTheHostAgreeStageByStage`.
## DSV4.1 #16: THE COMPRESSED HALF, WHICH EVERY TABLE UP TO HERE LEFT OUT

`#15` closed with one caveat it could not retire: the synthetic checkpoint declares
`compress_ratios` all zeros, so attention reads the window half of its row budget, the
compressor's two per-position projections never run, and `writeLatent` carries no pool
term. Every figure in `#13`-`#15` is therefore a LOWER bound for a compressed layer, and
the stage those entries promote may not be the stage the checkpoint pays for.

This retires the caveat. A compressed fixture now exists, the graph it drives is compared
against the host, and the stage table is taken with the compressed half switched on.

**The fixture** (`DeepSeekV41SyntheticShape.compressedPublishedWidths`): 7,168 hidden, 128
heads of 512, window 640, `index_topk` 512, so the gather budget is 1,152 rows — the
checkpoint's own bound — over a 1,024-position context (at ratio 2 the last positions keep
the full 512 compressed rows). Two layers, one window-only and one compressed at ratio 2,
i.e. the 40-layer checkpoint's 50/50 encoder/decoder split at depth 2. The indexer's
`index_n_heads`/`index_head_dim` are deliberately tiny: no backend reads those weights,
because row selection is a function of the position and the ratio, so they only size a
cache region nothing attends over. `index_topk` is NOT tiny, because it is the row count.

A second, cheap shape (`.compressedSmall`, the default widths plus two compressed layers)
exists because `compressedPublishedWidths` costs about a minute just to build its weight
blob, and a default-suite test cannot pay that.

**The graph, checked before anything was timed.** No parity fixture before this could run
a compressed layer at all, so four separate rules had never been compared with the
reference: the compressor's two projections over the post-`attn_norm` input, the pool that
emits a row only when a group closes, the rope rotation applied at the group's FIRST
position rather than at the pooling position, and the gather that mixes pooled rows in
with the live window. `testTheDeviceAndTheHostAgreeOnCompressedRows` walks six positions
on the small compressed model and compares logits with `DeepSeekV41CPUForward` at the
standard decode tolerance, and `testACompressedBlockPrefillsLikeTheWalk` runs the same
tokens as one block and compares against the walk at 1e-5.

Sensitivity was checked in the usual direction: pointing the device's pooled-row rotation
at `position` instead of `(rows - 1) * ratio` — i.e. rotating a compressed row at the
position that closed its group rather than at its first position — fails both cases (5
assertions). That is the failure this fixture exists to catch, and the first positions
still match, which is why it needs a test rather than an eyeball.

**The A/B** (`testWhatTheCompressedHalfCostsOverTheWindowOnlyHalf`, opt-in): one layer
each, same 1,024 positions, same window, same context, identical weights everywhere except
`compress_ratios`. The compressed layer is an OWNER at ratio 2, so it runs the compressor
and pools rows, and it reads `window + index_topk` rows where the window-only arm reads the
window.

| stage | window-only | compressed | change |
|---|---:|---:|---:|
| `writeLatent`, ms/pos | 0.149 | **0.271** | +0.122 (1.82x) |
| dense device, ms/pos | 0.382 | **0.504** | +0.123 |
| whole block, ms/pos | 0.83 | **0.97** | +17% |
| gather rows | 641 | 1,152 | |

So the compressed half costs 17% of a block, all of it inside `writeLatent`.

**Where that 0.122 goes, and what it says the stage IS.** Over 1,024 positions the
window-only arm averages 440.3 window rows (the ramp to 640, then 384 full ones: 450,880
rows in total) and the compressed arm averages 440.3 plus 256.0 compressed rows
(`min(512, (p + 1) / 2)` over the same span), i.e. +58% rows.

An attention head reads every gathered row at the record width, so the stage's traffic is
`heads * rows * record * 2` bytes: 57.7 MB per position for the window-only arm and 91.3 MB
for the compressed one. 57.7 MB in the measured 0.149 ms is **387 GB/s**, and 91.3 MB at
that same rate predicts 0.236 ms against 0.271 measured. So `writeLatent` is, to first
order, a pure key-read bandwidth cost, and the residual 0.035 is the compressor's two
448-wide projections (cf. `latentProject`'s one 512-wide row at 0.014) plus pool, norm,
rope and store, which run once per group.

That reframes the stage more than the split does. The compressor is a quarter of the
delta and it is not the lever: **the cost is 128 heads each reading the whole gathered
block**, so it comes down by reading fewer BYTES per row, or by serving one row read to
several heads. The second was tried and reverted (`#6`: one threadgroup sharing a loaded
row across heads lost 17% at 128 rows once occupancy was paid for), so any retry has to
be a different shape rather than the same idea with fewer registers.

**The model-shaped table** (`testWhereABlocksRemainingTimeGoesWithCompressedLayers`, opt-in,
1,024 positions, two layers, 0.88 ms/position/layer = **1.75 ms/position whole block**;
dense device 0.4535, mixture 0.4119):

| stage | ms/pos/layer | share |
|---|---:|---:|
| `writeLatent` | **0.211** | 45.5% |
| `outputDown` | 0.047 | 10.1% |
| `queryUp` | 0.046 | 9.9% |
| `outputUp` | 0.044 | 9.6% |
| `queryDown` | 0.018 | 3.9% |
| `latentProject` | 0.017 | 3.7% |
| `hcFFNCoefficients`, `hcAttentionCoefficients`, `ffnNorm`, `attnNorm` | 0.011-0.012 each | ~2.5% |
| `calibration(8 dispatch)` | 0.010 | 2.1% |
| everything else | 0.005 and below | |

`writeLatent` is 45.5% of the dense half here, against 30.4% in `#15` — the compressed
layer is what moved it, and it is once again the stage to attack. `compressPool` and
`attention` both print 0.000 because they run INSIDE `writeLatent`; the two are separated
only by the A/B above.

**Caveats.** Same synthetic-weight caveat as before: the checkpoint's attention and
compressor projections are quantised, so absolute numbers differ while rankings hold. The
fixture's compressed layer is an owner at ratio 2 with no reading layers, so the "a
compressed layer reads its owner's rows" case is priced by the host/device parity only, not
by this table. The synthetic mixture is still `moe_intermediate` 32, so its 0.41
ms/position is dispatch overhead and is 52% of device time. One command buffer per stage;
the isolated tables are cache-warm and are not additive into the block table. Both cases
are opt-in (`testWhereABlocksRemainingTimeGoesWithCompressedLayers` ~1 min, mostly building
the two-layer blob; `testWhatTheCompressedHalfCostsOverTheWindowOnlyHalf` ~1 min for the
same reason).

Tests: `testTheCompressedFixturePublishesWhatThePlanDemands` (the catalog accepts the
compressed table: three compressor tensors on the owner, the indexer's weights on the
index source, and the gather budget is `window + index_topk`),
`testTheDeviceAndTheHostAgreeOnCompressedRows`, `testACompressedBlockPrefillsLikeTheWalk`,
and the two opt-in tables above. Whole DeepSeek suite: 419 tests, 13 skipped, 0 failures.

## DSV4.1 #17: THE ATTENTION WAS CHAIN-BOUND, NOT BANDWIDTH-BOUND

`#16` closed by naming the lever: `writeLatent` is 128 heads each reading the whole
gathered block, so it comes down "by reading fewer BYTES per row". That was a model, not a
measurement — 57.7 MB over the measured 0.149 ms is 387 GB/s, which is a believable number
for this machine, and believable is not the same as checked. The check was cheap: halve the
record and change nothing else.

Halving the record did nothing. That falsified the model, and the stage turned out to be
bound by something the arithmetic had hidden.

**The A/B** (`testWhatAnFP16KeyBlockSavesOverAnFP32One`, opt-in, published geometry: 128
heads, 512-wide record, sink on). Both arms are fed the SAME key values: the block is
sampled as `half` and the FP32 arm reads those values widened, which is exactly what the
gather did to a cache row. Row counts 136 (one window plus the first compressed row), 640
(`window + index_topk`) and 1,152 (what a block's late positions read). The arms INTERLEAVE
inside each attempt and every number is a minimum of six attempts, because GPU clocks ramp
across a sweep and a first-timed arm reads slow for reasons that have nothing to do with its
code; without interleaving, the first row count read 14% slow purely for being first.

| form | 136 rows | 640 rows | 1,152 rows |
|---|---:|---:|---:|
| FP32 records, one accumulator (what `#16` shipped) | 0.121 | 0.456 | **0.820** |
| FP16 records, one accumulator | 0.094 | 0.531 | 0.808 |
| FP32 records, four accumulators | 0.045 | 0.339 | 0.299 |
| FP16 records, four accumulators | 0.040 | 0.314 | 0.274 |
| FP16, four accumulators, `half4` score pass (shipped) | **0.038** | **0.149** | **0.265** |

All in ms per dispatch. The 1,152 row is the one to read: it is stable to 0.001 across four
runs, where the 640 row moves up to 50% between runs. As a ratio, the shipped form is
roughly 3.2x / 3.1x / 3.1x faster than the form `#16` measured.

**Reading the table.** Row 2 against row 1 is the byte hypothesis alone: 0.820 → 0.808, i.e.
1.01x, at every row count. Halving the bytes moved by the dominant read bought nothing, so
the stage was not paying for bandwidth. Row 3 against row 1 is the thing it WAS paying for:
the same FP32 records, with the dot product and the value sum split across four independent
accumulators, are 2.74x faster at 1,152 rows. `dot += q * k` is a `width`-long chain of
dependent FMAs — 512 of them, ~4 cycles each, in one lane per row — and that chain, not the
memory system, set the cost. Row 4 against row 3 says the bytes are worth 9% once the chain
is gone, so both changes are real but they are not additive in the way the byte model
implied, and the byte half only pays because the chain was fixed first.

The last row is the shipped kernel. `half4` vectorisation of the score pass — a row is
contiguous and the record width is a multiple of four, so it is one 8-byte load per four
values instead of four 2-byte loads — is worth a further 3-4%, measured at both large row
counts.

Both arms are validated against `DeepSeekV41AttentionReference.attend` before any number is
printed, and against each other at a float-rounding bar: the two kernels can sum in
different orders, so they agree to rounding rather than to the bit. Correctness of the
graph is unaffected because the rule, the addressing, the slot dimension and the threadgroup
shape are all unchanged.

**What shipped.** `dsv41_attention_csa2_f16`: FP16 key records, four accumulators in both
passes, `half4` score pass. The gather now writes the cache's own 2-byte records instead of
widening them to FP32 (`dsv41_gather_latents_f16` is a plain copy), which also halves the
gathered block — 10 MB to 5 MB at eight slots of a 1,152-row budget — and the executor
dispatches the FP16 kernel. The FP32 kernel stays in the library untouched, as the baseline
this A/B measured against.

**In situ** (the two opt-in tables from `#16`, re-taken, 1,024 positions):

| stage | window-only `#16` | window-only `#17` | compressed `#16` | compressed `#17` |
|---|---:|---:|---:|---:|
| `writeLatent`, ms/pos | 0.149 | **0.070** | 0.271 | **0.136** |
| dense device, ms/pos | 0.382 | **0.303** | 0.504 | **0.370** |
| whole block, ms/pos | 0.83 | **0.75** | 0.97 | **0.84** |

2.13x on the window-only `writeLatent` and 1.99x on the compressed one, and the compressed
penalty over its window-only arm falls from +0.122 to +0.066 ms/pos (+17% → +12% on the
block). The model-shaped table (two layers, 0.77 ms/position/layer) goes from 1.75 to
**1.54 ms/position whole block**, with `writeLatent` 0.211 → **0.103** — 45.5% of the dense
half down to 29.4%.

In situ is 2.0x where the isolated A/B is 3.1x, and the gap is not noise: `writeLatent`
carries the store, the pool, the gather and the latent rope as well, and the surrounding
dispatches evict the block, so the in-situ arm is not the cache-warm one the isolated table
measures. Neither number is the other's prediction; the isolated table ranks kernel FORMS,
the block table measures the stage.

**The ranking it leaves.** `writeLatent` is still first at 29.4%, but it is no longer in a
class of its own: `queryUp` 0.046, `outputDown` 0.045 and `outputUp` 0.044 are ~13% each,
i.e. three weight-read GEMV stages that together are 39% of the dense half. Those are now
what the attention was before this entry — the biggest single thing left, and worth a
measurement of their own rather than an inference from the attention's.

**Caveats.** Synthetic weights, as always: the checkpoint's attention and compressor
projections are quantised, so absolute numbers differ while rankings hold, and the synthetic
mixture (`moe_intermediate` 32) is still dispatch overhead at 0.41 ms/position, which caps
how far a 12% block reduction can look. The isolated table is cache-warm and is not additive
into the block table. The published-geometry timing case
(`testWhatOneAttentionDispatchCostsAtThePublishedWidths`) reads the shipped kernel too now,
but its harness averages two attempts of five dispatches and spreads far wider than the
interleaved A/B — it says 0.061 / 0.056 / 0.426 ms at 128 / 136 / 640 rows — so treat the
A/B above as the decision instrument and that case as a coarse cross-check.

**Tests.** The kernels that ship are the kernels that are pinned:
`DeepSeekV41AttentionKernelTests` (both cases, including the 200-row threadgroup reduction
and the value head wider than the record) and `DeepSeekV41GatherKernelTests` now read FP16
records. Sensitivity was checked in the usual direction: transposing one lane of the
`half4` load (`record.w` → `record.z`) fails 3 of the 5 attention and parity cases it was
run against. The parity suite that matters for the graph —
`testTheDeviceAndTheHostAgreeStageByStage`, `testTheChunkSeamReturnsTheWalksLogits`,
`testEveryPositionOfABlockAttendsOverItsOwnRows`, `testTheDeviceAndTheHostAgreeOnCompressedRows`,
`testACompressedBlockPrefillsLikeTheWalk` — passes unchanged at its existing bars. Whole
DeepSeek suite: 420 tests, 14 skipped, 0 failures.

## DSV4.1 #18: THE DENSE PROJECTIONS ARE INPUT-BOUND, AND A WIDER BLOCK IS NOT FREE

`#17` closed with three stages level with each other — `queryUp` 0.046, `outputDown`
0.045, `outputUp` 0.044 ms/pos/layer — reading 201, 134 and 117 MB of weights apiece, i.e.
at 580, 408 and 342 GB/s. Those are plausible streaming rates, so the reading was that the
stages buy bytes and that the only way to buy fewer per position is to amortize one weight
read over more positions. `bf16_gemv_rows8` already amortizes over the eight positions of a
block. The obvious next step was sixteen.

Sixteen is worse. That was the whole of this entry's first result, and it inverted the
model.

**Block width** (`bf16_gemv_rows16`, since removed), same matrices, same band structure,
same caller layout, only the block width differing:

| projection | weight | block 8 | block 16 | per position |
|---|---:|---:|---:|---:|
| `queryUp` (1,536 -> 65,536) | 192 MB | 0.346 ms | 0.715 ms | 0.97x |
| `outputDown` (8 x 8,192 -> 1,024) | 128 MB | 0.330 ms | 1.053 ms | 0.63x |
| `outputUp` (8,192 -> 7,168) | 112 MB | 0.343 ms | 1.269 ms | 0.54x |

A wider block reads the SAME weight bytes for twice the positions, so if the weight read
were the binding cost the per-position figure would halve. Instead the cost grew, and grew
most where the row is longest. What doubles with the block is the INPUT: a threadgroup
reading one output row loads every position's vector for the length of that row, once per
row, over tens of thousands of rows — 1.9 GB of activation reads per block for `outputUp`
against 112 MB of weights. The stage is not streaming weights at the memory limit; it is
re-reading activations out of cache, and the weight stream rides along.

**Rows per threadgroup** is the lever that follows, because it divides that input traffic
without touching the weight read. Two rows per threadgroup (`bf16_gemv_rows8x2`, shipped)
halves it; four (`bf16_gemv_rows8x4`, removed) quarters it but costs thirty-two
accumulators:

| projection | rows8 | rows8x2 | rows8x4 |
|---|---:|---:|---:|
| `queryUp` | 0.347 | 0.353 (0.98x) | 0.422 (0.82x) |
| `outputDown` | 0.329 | 0.275 (**1.20x**) | 0.399 (0.82x) |
| `outputUp` | 0.344 | 0.256 (**1.34x**) | 0.264 (1.30x) |

So the answer to "how many" is two, and it depends only on the LENGTH of the row. Sweeping
input width at a fixed 7,168-row output:

| inDim | 1,024 | 1,536 | 2,048 | 3,072 | 4,096 | 6,144 | 8,192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows8x2 vs rows8, run A | 0.94x | 1.00x | 1.06x | 1.05x | 0.97x | 1.20x | 1.20x |
| rows8x2 vs rows8, run B | 0.95x | 1.03x | 1.11x | 1.22x | 1.32x | 1.08x | 1.26-1.31x |

The two runs agree at both ends and disagree in the middle, so `rowSharingWidth` is set to
6,144: below it the pair form is neutral to 6% worse, and above it every measurement wins
by 20-34%. The published attention stack lands on the right side of that line without
being asked to — `attn.wq_b` is 1,536 wide and keeps the one-row form, `attn.wo_a`,
`attn.wo_b` and `attn.wq_a` are 7,168-8,192 wide and take the pair.

**In situ** (`testWhereABlocksRemainingTimeGoesWithCompressedLayers`, two layers, 1,024
positions):

| stage | `#17` | `#18` | change |
|---|---:|---:|---:|
| `outputDown` | 0.045 | **0.037** | 1.22x |
| `outputUp` | 0.044 | **0.033** | 1.33x |
| `queryDown` | 0.019 | **0.013** | 1.46x |
| `queryUp` | 0.046 | 0.046 | unchanged |
| dense device | 0.3457 | **0.3179** | 8% |
| whole block | 1.54 | **1.50** | 2.6% |

Close to what the isolated A/B predicted, which is the point of taking both. `queryDown` is
the pleasant one: it was not in the A/B's three named geometries, and it took the pair form
because its input is `hidden` (7,168), so a rule keyed on the measured quantity picked up a
stage nobody aimed at.

**What is left of the dense half, and the question it opens.** `writeLatent` is 30.9%,
then `queryUp` at 14.0% — unchanged by everything above, at 65,536 output rows of a 1,536
input. By this entry's own model it should have been the BEST candidate for row sharing
(small input, huge output) and it measured 0.98x. Something else dominates there, and the
shape of it is the next thing to measure rather than assume: 65,536 threadgroups of 384
iterations each is a grid a fifth of a millisecond wide, so per-row scheduling and the
`outDim`-threadgroups-per-SM tail are the suspects.

**Caveats.** Same synthetic-weight caveat. The absolute times in the two crossover runs
differ by 1.65x (0.347 vs 0.574 ms for the same `queryUp` dispatch on the same fixtures),
which is why every decision here is a ratio taken from interleaved arms inside one run and
why the crossover itself is reported as a pair of runs rather than one line. Four rows per
threadgroup was measured once; two was measured twice. The M5 Max was carrying other work
during at least one of these runs, and the isolated table is cache-warm and is not additive
into the block table.

**Tests.** The pair kernel's contract is pinned by
`testThePairGemvFillsBothRowsOfItsTile`, which deliberately uses an ODD output count so the
last tile carries one row: the published projections are all even, so nothing else exercises
the guard that stops a threadgroup reading past the end of the matrix. Sensitivity checked
in the usual direction — pointing the second row's weight read at the first row's bytes
fails it with 14 assertion failures. `testTheWideGemvGridIsChosenByHowMuchWorkTheProjectionHas`
now asserts the form rule's new parameter: a 1,536-wide block stays `.block`, an 8,192-wide
block takes `.blockPair`, and a short batch still falls back to the two-way rule. Whole
DeepSeek suite: 422 tests, 15 skipped, 0 failures.

## DSV4.1 #19: THE DENSE BACKBONE WAS A PAGE-FAULT WALK, NOT A COMPUTE COST

Every entry above measured the device graph on `DeepSeekV41SyntheticCheckpoint`, which
publishes a whole model as one aligned blob. That is the right fixture for ranking kernels
and the wrong one for asking where a real turn's seconds go: a synthetic blob is already
resident, so no measurement in this journal had ever priced the *address* a tensor lives at.
On the published checkpoint the dense backbone is ~170 MB per layer plus a 1.32 GB output
head, mapped zero-copy out of 48 shards that are 99% routed experts, and those pages do not
survive a token: the expert stream reads 23 GB of them in between.

The first real turn (`#none`, before this entry) was 60.7 s of prefill and 348 s of decode
for seven tokens — ~58 s/token, i.e. 1.45 s per layer for ~200 MB of dense reads, or
~140 MB/s. That is the GPU page-fault path, not DRAM, and it is 160x off what the same
weights cost once resident.

**The fix** is the one the packed-format families already use, generalised:
`DeepSeekV41DensePin` packs the non-expert, non-streamed backbone into 1 GiB resident
`.storageModeShared` buffers at load (a tensor wider than a chunk gets its own), rewrites
each tensor's `(shardIndex, byteOffset)` to address its new home, and merges the chunks into
the buffer registry the executor binds through. Nothing in the graph changed: a GEMV still
binds a buffer at offset 0 and passes its `byteOffset` as an argument, so a rewritten
address is indistinguishable from a shard address.

**What is pinned** (measured from the index, not assumed):

| category | bytes | tensors | pinned |
|---|---:|---:|---|
| routed experts | 296.0 GB | 94,464 | no, the arena streams them |
| Engram tables | 202.8 GB | 4 | no, read row by row |
| dense layers (40) | 6.8 GB | 1,236 | yes |
| output head | 1.32 GB | 1 | yes |
| embedding | 1.32 GB | 1 | no, read as single rows |
| vision tower + draft block | 1.68 GB | 360 | no, the text graph never reads them |

**Load** (`testAPublishedPromptCompletesATurn`, "hey"):

```
dense: pinned 8.11 GB into 8 buffer(s), 1335 tensors at 27921 MB/s,
       8 left on the mapping (layers.1.engram.k_weight: no shard names
       model-00047-of-00048.safetensors at file offset 0; ...)
```

27.9 GB/s is the parallel pread of 8 files at once; the pin costs a few seconds of load and
nothing afterwards. The eight tensors left on the mapping are the two Engram shards' dense
half: those files are 94.56 GiB and `segmentShardsForMetal` collapses each to ONE segment
(a ~100 GB streamed table plus a few hundred MB that maps), so the segment's start in the
file is not in the summary. Recovering it needs the safetensors header, which is a follow-up
worth ~314 MB of 8.4 GB, i.e. 3.7%.

**The turn**, same prompt, same machine, only the address of the weights differing:

| | mapping | pinned, head skipped | pinned, head included |
|---|---:|---:|---:|
| prefill (5 positions) | 60.7 s | 3.12 s | **1.18 s** |
| decode (6 steps) | 348 s | 4.01 s | **2.19 s** |
| whole test (load + turn) | ~700 s | 13.06 s | **8.43 s** |
| answer | "Hey! How's it going?" | same | same |

~58 s/token → 0.37 s/token. The middle column isolates the head: 1.32 GB read in full once
per token cost 0.30 s/token on its own (4.4 GB/s, faster than the layer path because one
contiguous gigabyte faults more sequentially), which is why it is pinned first in the plan's
order and gets a dedicated chunk.

**Why it was not the GPU.** `#13`-`#18` put the whole dense half of a block at 0.32
ms/position on the synthetic fixture, i.e. 40 layers ≈ 13 ms/token. The real turn was 1,450
ms/layer. The gap was never in the kernels, and no amount of block-width or row-sharing work
would have found it: every table in this journal was taken on weights that were already in
RAM.

**Tests.** `DeepSeekV41DensePinPlanTests` pins the packing (alignment, no chunk straddle,
segment file offsets, the budget, the head-first order, a dedicated buffer for a tensor
wider than a chunk, refusal past the buffer limit). The end-to-end guarantee is
`testAPinnedRuntimeComputesTheSameLogitsAsTheMappedOne`: two runtimes over the same
synthetic shard file, one pinned and one mapped, must return BIT-IDENTICAL logits for three
decode positions — a dropped tensor, a forgotten segment start or a misaligned slot moves
them by O(1). Whole DeepSeek suite: 450 tests, 20 skipped, 0 failures.

**Caveats.** The pinned bytes are dirty pages rather than clean file-backed ones, which is
why the budget is a third of RAM and why the memory plan already prices them as resident; on
a machine that cannot hold the whole backbone the plan pins a prefix (head first) instead of
refusing to load. And the numbers above are one turn of seven tokens on an M5 Max, so they
say the fault path is gone, not what the new steady state is. What is left is visible in the
same log: the turn read 23.25 GB of experts at 8.6 GB/s with a 40.5% arena hit rate, i.e.
~2.6 GB of expert misses per token, about 0.30 s of the 0.37 s/token. The dense half is no
longer the cost; the arena's hit rate is, and that is the next measurement rather than
another kernel.

## DSV4.1 #20: THE DECODE STEP WAS TWO KERNEL SHAPES, NOT THE DISK

`#19` closed by naming the expert arena's hit rate as the next lever, on the strength of a
turn that spent 0.30 s of 0.37 s/token waiting on 2.6 GB of expert misses. Measuring it said
something else, and the difference matters because it sends the next slice somewhere else
entirely.

**The instrument.** The block path had `DeepSeekV41BlockTiming`; a decode step runs the walk,
which commits two command buffers per layer and blocks on the host twice, so a token's cost
can sit anywhere between the host and the GPU. `DeepSeekV41WalkTiming` (off unless
`DeepSeekV41MetalForward.recordsWalkTiming` is on) now splits it into front encode, front
wait, expert fill, mixture encode, and each half's own `gpuStartTime`/`gpuEndTime` window, and
`DeepSeekV41PublishedTurnTests.testWhatADecodePositionCosts` runs the real checkpoint over a
53-token prompt+answer sequence and prints the split. Per decoded position, before anything
below was changed:

| bucket | ms/position |
|---|---:|
| front encode (host) | 4.9 |
| front wait (host, blocked) | 253.3 |
| expert fill (host, pread) | 136.2 |
| mixture encode (host) | 3.6 |
| front half, device | 38.7 |
| mixture half, device | 39.1 |
| whole position | 398.0 |

**The disk was already at its ceiling.** A raw measurement of the same read pattern
against `model-00047` (5.9 MB ranges, distinct spans, `F_NOCACHE`): 13.43 GB/s at concurrency
1, 13.84 at 4, 16, and 36; 13.19 GB/s for a batch of only 36 ranges (0.18 GB); 13.79 for
0.5 GB. The streamer's own pread timers report 13-14 GB/s over a fill at the start of a
sequence. A fill is therefore worth exactly the bytes it moves, and the bytes are the arena's
hit rate — but the fill was 136 ms of 398, not 300 of 370: the token had moved on.

**What the token was actually spending.** Each half read its weights through a kernel that
gives a thread a WHOLE OUTPUT ROW and walks it block by block. With 32 lanes per warp a row
apart, every lane's load pulls a 32-byte sector for one to sixteen useful bytes, and the
reduction is one dependent chain per row. This is the same defect `#10`, `#13` and `#17`
fixed for BF16 and for attention, and it had simply never been fixed for the two
block-scaled formats this checkpoint actually ships.

Isolated A/B of the new forms (`threadgroup` per output row, lanes striding the blocks,
`simd_sum` reduction, threadgroup memory sized one float per SIMD group), one dispatch
amortized over six distinct weight regions in one command buffer, best of 8:

FP8 32x32 ue8m0, decode (one position):

| projection | one thread/row | coalesced, 32 lanes | coalesced, 64 lanes |
|---|---:|---:|---:|
| `attn.wq_b` 32,768x1,280 | **0.308 ms** | 0.394 (0.78x) | 0.351 (0.88x) |
| `attn.wo_b` 5,120x8,192 | 0.261 | **0.108 (2.42x, 370 GB/s)** | 0.110 (2.37x) |
| `attn.wo_a` band 1,024x4,096 | 0.128 | **0.033 (3.93x)** | 0.033 (3.94x) |
| `attn.wq_a` 1,280x5,120 | 0.156 | 0.039 (4.00x) | **0.038 (4.13x)** |
| `attn.wkv` 512x5,120 | 0.153 | 0.032 (4.74x) | **0.028 (5.41x)** |
| `shared_experts.w1` 2,304x5,120 | 0.174 | 0.052 (3.38x) | **0.045 (3.88x)** |

MXFP4 routed expert, decode:

| kernel | one thread/row | coalesced, 32 lanes | coalesced, 64 lanes |
|---|---:|---:|---:|
| fused gate+up 2,304x5,120 | 0.472 ms (48 GB/s) | **0.075 (6.33x, 300 GB/s)** | 0.092 (5.13x) |
| down 5,120x2,304 | 0.133 (42 GB/s) | **0.057 (2.32x)** | 0.059 (2.26x) |

Two things the table says that a rule needs:

- **`wq_b` keeps the one-thread form.** It is 32,768 rows over 1,280 columns: 40 blocks, so a
  32-lane row has one of them and only 40 lanes of a 64-lane group are ever busy, while the
  grid becomes 32,768 threadgroups of almost no work. At a block of eight positions it is
  worse still (0.588 → 0.850 ms, 0.69x). `fp8GemvWantsWideGrid` therefore requires
  `inDim >= 1,024` AND leaves tall-narrow shapes on the old form until a rows-per-threadgroup
  kernel exists; `wq_b` is ~12 ms/token of the dense half, so that is worth doing.
- **32 lanes wins for the mixture and 64 for FP8.** A 32-lane group is one SIMD group, so
  `simd_sum` finishes the reduction and the kernel returns without touching threadgroup
  memory or a barrier at all; the fused gate+up pays for four accumulators, which is where the
  wider group's barrier hurts most. Both widths measured, both shipped as measured
  (`mixtureLaneWidth` 32, `gemvLaneWidth` 64).

**After.** Same sequence, same tokens, only the kernels differing:

| bucket | before | after |
|---|---:|---:|
| front half, device | 38.7 ms | **28.7 ms** |
| mixture half, device | 39.1 ms | **27.0 ms** |
| expert fill | 136.2 ms | 67.1 ms |
| whole position | 398.0 ms | **204.6 ms** |
| 53 positions, average | 315.6 ms | **220.1 ms** |

The fill moved too, and not because of the kernels: the run above had the page cache serving
part of it. The honest reading is that the GPU halves fell by 2.7x and 1.5x and the fill is a
function of the arena's hit rate on the day.

**The turn**, `Write three sentences about why the ocean is salty.`, greedy, 39 tokens:

| | before | after |
|---|---:|---:|
| prefill (14 positions) | 7.15 s | **2.49 s** (2.04 s on an idle machine) |
| decode (39 tokens) | 19.02 s (**488 ms/token**) | **8.35 s (214 ms/token)** |
| answer | identical | identical |

A second run immediately after another app-hosted test measured 12.42 s of decode, i.e. 318
ms/token, with identical read counts — the difference is machine state (a 70 GB arena being
released by the previous process and the page cache reclaiming), which is why the tables
above are A/Bs inside one run rather than comparisons across runs.

**`F_NOCACHE` on the expert descriptors was implemented, measured, and REVERTED.** The arena
is the expert cache (~70 GB of whole pairs) and each pread also inserts its bytes into the
unified buffer cache, so bypassing it looks like a way to stop two caches competing for the
same RAM. It is a 2.2x loss: fills ran at 2-4 GB/s against 13.4-13.9 through the page cache,
and the GPU halves slowed to 60 ms each as well, so the time was going somewhere other than
the disk. A cold read of the same ranges through a plain buffer measures 13.8 GB/s either way,
which says the cost is this destination (Metal shared memory), not the file cache. Do not
re-propose it without a number that beats 13.9.

**Tests.** `DeepSeekV41FP8GemvKernelTests` pins the coalesced FP8 form against a host rule
(including the 32x32 scale grid, where 32 consecutive rows share one scale row), against the
one-thread form, on a batch with input/output strides wider than the shape, and on a row
shorter than the block count of a lane. `testWhatTheCoalescedFP8FormIsWorth` and
`testWhatTheCoalescedMixtureIsWorth` are the A/Bs above, opt-in like the rest of the
measurement file. The published turn answers identically and runs clean under
`MTL_DEBUG_LAYER=1`; whole DeepSeek suite 457 tests, 23 skipped, 0 failures.

**What is left, in the order the buckets now rank.** The expert fill is the largest single
item (67 ms of 205) and it is disk-bound at 13-14 GB/s, so what improves it is fewer misses:
the arena is `0.6 * RAM` (3,760 slots on 128 GB, ~71 GB) against 15,360 pairs, and one run
measured the fill degrading from 13.9 GB/s to 8 GB/s as the arena filled, which is memory
pressure rather than the disk. Second is `front wait` (130 ms of which ~56 ms is the two
halves' own device time): a decode step makes 80 host round trips per token, and the queue's
turnaround at ~1.9 ms/layer is now larger than the work it is waiting for. Third is the two
shapes the rules deliberately left alone: `wq_b` (~12 ms/token) and the shared experts'
FP8 swiglu and down (~21 ms/token), both of which want the same treatment with several rows
per threadgroup rather than one.

## DSV4.1 #21: THE TWO-ROW FP8 FORM IS A WASH, AND THE ARENA NEVER EVICTS

#20 left two named candidates: the shapes the coalesced form still lost on (`attn.wq_b`
and the shared experts' FFN), which looked like the `bf16_gemv_rows8x2` "several rows per
threadgroup" fix, and the expert arena's hit rate, which looked like an eviction-policy
problem. Both were measured. Neither was what it looked like, and the second question
turned out to have no eviction in it at all.

### Two rows per threadgroup, on FP8

`dsv41_fp8_block32_gemv_rows2` gives a threadgroup two consecutive output rows and one
input token: each lane reads its input floats once, then both rows' weight bytes and
scales, and the grid halves from `outDim * tokens` to `ceil(outDim / 2) * tokens`. Same
buffer indices as the one-row form, two floats per SIMD group of partials, the 32-lane
fast path still returns without a barrier. `DeepSeekV41FP8GemvKernelTests` pins it against
the host rule at both lane widths, on an odd row count (the last pair has no second row),
and on a batch with input/output strides wider than the shape.

The A/B runs three arms — one-thread, one-row, pair — interleaved inside each of eight
repetitions, six distinct weight regions per command buffer, and refuses to time a pair
whose outputs disagree. One decode position (ms):

| shape | one-thread | one-row | pair 32 | pair 64 |
|---|---:|---:|---:|---:|
| `attn.wq_b` 32,768x1,280 | **0.140** | 0.151 (0.93x) | 0.148 | 0.146 |
| `shared_experts.w1` 2,304x5,120 | 0.170 | **0.049 (3.48x)** | 0.051 | 0.055 |
| `shared_experts.w2` 5,120x2,304 | 0.087 | **0.052 (1.67x)** | 0.061 | 0.059 |
| `attn.wo_b` 5,120x8,192 | 0.279 | **0.108 (2.59x)** | 0.124 | 0.113 |

and a block of eight positions:

| shape | one-thread | one-row | pair 32 | pair 64 |
|---|---:|---:|---:|---:|
| `attn.wq_b` | **0.585** | 0.934 (0.63x) | 0.849 | 0.893 |
| `shared_experts.w1` | 0.239 | **0.192 (1.24x)** | 0.190 | 0.196 |
| `shared_experts.w2` | **0.205** | 0.238 (0.86x) | 0.230 | 0.234 |
| `attn.wo_b` | 0.644 | **0.583 (1.10x)** | 0.585 | 0.584 |

The pair form is between 0.85x and 1.10x of the one-row form everywhere, i.e. inside the
noise of the thing it was meant to replace. The reason is the opposite of the BF16 result:
there, a block of eight positions re-reads every position's activation once per output row
(1.9 GB of READ requests per block on `outputUp`), so halving the input traffic halved the
stage. Here the input is a single 5 KB vector that every threadgroup reads from cache; what
the pair form adds instead is four more accumulators and twice the E4M3 unpack per lane. Do
not wire it, and do not re-propose "more rows per threadgroup" for FP8 without a shape whose
input traffic is the measured cost. It stays in the shader library as the baseline the
sweep measures, and it is deliberately not in `DeepSeekV41Pipelines`.

**The rule moved instead.** A width sweep at a fixed 5,120 rows puts the one-thread form
ahead below 64 blocks (1,024 values: one-row still wins 1.42x at one position but loses at
eight; 1,280 loses at both) and the coalesced form clearly ahead from 2,048 up. That is one
32-value block per lane at 64 lanes, which is now the rule: `inDim >= 2_048` rather than
`1_024`. The published shape this changes is `attn.wq_b` (40 blocks), the one projection
where #20's own numbers already said the coalesced form was 0.78x — it now takes the
one-thread form, worth 1.07x at one position and 1.6x at eight.

The GB/s column in the measurement prints was also wrong by a factor of a million: it
divided a MILLISECOND figure by 1e9 twice. Both helpers are fixed, so the printed number is
now the number the journal quotes.

### The arena has no evictions to tune

#20 ranked the fill first and asked whether the arena's degradation was reclaim, and whether
a smaller arena or a different eviction order would fill faster. Instrumenting
`DeepSeekV41ExpertStreamer` answered the question before it was optimised: over a 53-position
walk of the published checkpoint, 12,720 acquisitions at a 70.6% hit rate touched 3,739
distinct pairs against a capacity of 3,760. **Every miss was a first touch and nothing was
ever evicted.** Raising the budget from 0.6 to 0.72 of RAM (3,760 -> 4,637 slots) left the
per-position read byte-for-byte identical (806.8 MB at position 52 in both arms) and the hit
rate identical (70.6%), because the extra capacity was never needed. A 15-draw turn at 0.72
ends at 1,790 of 4,637 resident.

So the fill is cold-start bound, not capacity bound: a token routes six of 384 experts in
each of forty layers, and after a few dozen tokens most of what it asks for is still new.
The steady-state cost over these runs is 900-950 MB/token at 78% hit, which the disk serves
at 10-13.7 GB/s (13.8 is the machine's measured ceiling), i.e. 70-90 ms of a ~270 ms
position. The earlier "13.9 -> 8 GB/s as the arena filled" was machine state, not eviction:
two runs of the same token list read exactly the same bytes and varied by 2x in wall time.

Two changes came out of the investigation anyway. Placement is now a dictionary lookup
(`(layer, expert) -> slot`) plus one pass for a free slot or the LRU victim, instead of
`slots.firstIndex(where:)` twice per miss over 3,760 elements; the timings printed by the
walk are 75.7 us per pair before and 72.0 after in a Debug test build, which says the scan
was not the cost in that build (Release would be a fraction of a millisecond per token) —
the instrument is what proves it rather than the optimisation being assumed. And
`dynamoe_dsv41_arena_fraction` is the debug seam the A/B above used, alongside the existing
`dynamoe_dsv41_pin_dense_off`.

**What is left.** The two GPU halves are ~57 ms of a 270 ms position and read ~9.4 GB at
190-230 GB/s, so their remaining headroom over the best measured GEMV rate (~370 GB/s) is
tens of milliseconds at most, and the shapes that are left are single-percent slices. The
host is the story: `front wait` is 2.4-3.2 ms per layer against ~1.4 ms of device work, and
the fill is 70-90 ms that no kernel overlaps because the routing readback sits between the
two halves. A decode step is 80 host round trips whose turnaround is now larger than the
work it waits for, and the fill cannot move off the critical path without either a second
arena whose slots the previous layer's mixture is not reading, or a way to know layer N's
routing before layer N's attention has run. Neither is a kernel change; both are a graph
change and both need their own measurement.

## DSV4.1 #22: THE DECODE STEP IS 72 ms OF WORK AND ~190 ms OF WAITING, AND A BATCHED VERIFY IS 2.16x CHEAPER

#21 ranked the host round trip first and named two graph changes. This slice measured
where the round trip actually goes before building either, and the answer moved the
target: the walk's cost is not the wait as such, it is that a decode step cannot batch.

### What the round trip is made of

`DeepSeekV41WalkTiming` already split a position into encode, wait, fill and device
windows. What it could not say is whether `front wait` (126-207 ms of a ~270 ms position)
was the GPU still working or the CPU and GPU handing the queue back and forth. Three
counters now answer it, all off unless `recordsWalkTiming` is on:

- `frontDrainNanoseconds` / `mixtureDrainNanoseconds` — wall time inside each
  `waitUntilCompleted`, split by the half the buffer belongs to.
- `frontLateStartNanoseconds` / `mixtureLateStartNanoseconds` — how long after the host
  began waiting the buffer's own `gpuStartTime` was stamped, i.e. a drain decomposed into
  late start + device window + tail.
- `frontUnwindowedBuffers` / `mixtureUnwindowedBuffers` — buffers Metal reported an empty
  window for, so the device totals can be read as a floor rather than a measurement.

One published 52-position walk, position 48 (`/tmp/walk_late.log`):

| bucket | ms/position |
|---|---|
| front encode | 4.5 |
| front wait | 100.6 |
| — drained front | 29.5 |
| — drained mixture | 78.5 |
| — late start, front | **0.0** |
| — late start, mixture | **49.0** |
| expert fill | 69.2 |
| mixture encode | 3.6 |
| front device | 27.5 |
| mixture device | 26.0 |

The dense half's wait is its own device window: the host never catches that buffer
unstarted. The mixture half's wait is 78.5 ms against a 26.0 ms window, and **49.0 ms of
the difference is the buffer not having started when the host asked for it** — 1.2 ms per
layer. That is the fill's shadow: the routing readback needs the dense half finished, the
fill runs on the host for ~1.7 ms per layer with nothing on the GPU, and then the next
commit is picked up late.

A microbenchmark (`DeepSeekV41CommandStartCostTests`) prices that pickup on its own, with
no model in the way — one trivial dispatch, committed after an idle gap:

| queue state | late start | drain |
|---|---|---|
| fed | 0.09 ms | 0.17 ms |
| idle 2 ms | 0.46 ms | 0.53 ms |
| idle 10 ms | 0.59 ms | 0.68 ms |

So a ~2 ms idle gap costs ~0.5 ms before the next buffer starts, and the walk's 1.2 ms is
that same effect on a buffer that binds an arena, an index list and a dozen vectors. The
number matters because it is **not** recoverable by rearranging dispatches: no kernel
change can keep a queue fed across a readback that has to happen.

### The walk with nothing to read

The same walk, timed over tokens the arena already held (zero expert misses), costs
**71.8 ms/position** against 263 ms with a realistic miss rate. Everything above the
72 ms is the expert stream and the stalls it causes, so a decode step is ~72 ms of work
and ~190 ms of paying for one position at a time. Per layer: one readback, one ~1.7 ms
fill, one late commit.

### What batching is worth

Speculative decoding is the only structure that lets a decode step batch, and the block
path it would verify through already exists (`prefillChunk`, compared with the walk on the
published checkpoint). `testWhatABatchedVerifyWouldSaveOverAWalk` prices both arms on the
published checkpoint over eight drafted positions, each arm on its own token runs so
neither is served by what the other already read:

```
batched verify vs walk, 8 drafted positions:
  walk  1226.6 ms (153.3 ms/position, 10256 MB read),
  block  567.4 ms ( 70.9 ms/position,  1357 MB read) -> 2.16x
```

The block reads **7.6x fewer bytes**: one expert union per block instead of one set per
position, which is the steady-state ~1 GB/token this model streams at a 70-78% arena hit
rate. The wall-clock win is 2.16x rather than 7.6x because the block still pays the head
and the same kernels.

That ratio is also the acceptance bar: a verify of eight costs 567 ms, so it beats
walking the same eight (1226 ms) once **more than ~3.7 of the eight are accepted**, i.e.
above a ~47% acceptance rate. The number is what a draft head has to clear, and it is the
first measured bar this port has for DSpark.

Two earlier fixtures are worth recording as traps. A block/walk A/B over tokens the
arena already held read **0 MB in both arms** and priced only kernel work: 47.7 vs
71.8 ms/position, 1.51x, with the fill that batching actually amortizes absent. And the
same test re-timed over the same tokens reads nothing at all, because the arena is shared
by the whole runtime — hence one token run per arm per repetition.

### What landed

`prefillChunk` now starts at the cache's own position instead of 0, so a chunk CONTINUES a
sequence rather than only opening one. That is the verify seam, and the position is what
makes it correct: the ring slot, the compressor's group slot (`position % compress_ratio`,
filled by the positions that ran before the call) and whether the group completes are all
derived from the absolute position, so a mid-sequence chunk is the same graph as a
mid-sequence walk. `testAChunkContinuesFromTheCacheLikeTheWalk` pins it on a fixture with
a compressed layer, starting the continuation one position INTO a group so the chunk has
to read a slot an earlier position left behind.

Nothing else changed: the verify still needs the draft head on the device (the target
layers' `mainHidden` capture), per-position logits capture in the block (`finishBlock`
currently writes the last slot's), and cache rewind on rejection. This slice measured the
prize and opened the seam.

**What is left.** The batched verify, now with a bar to clear (~47% acceptance over eight
drafts) and a measured multiplier (2.16x per position, 7.6x less I/O). The other two
levers #21 named are dead ends for the same reason the numbers here show: the per-layer
round trip is dominated by the idle gap the fill leaves, which only batching removes, and
the arena's capacity has already been shown not to be the constraint.

## DSV4.1 #23: THE DRAFTER'S CONDITIONING ROW, CAPTURED ON THE DEVICE

#22 measured the prize a batched verify is playing for and left one question open:
whether the checkpoint's own drafter clears the bar. Answering it needs the draft
block's forward, and the first thing that forward consumes is `main_hidden` — the
target layers' block inputs, means over the hyper-connection copies, concatenated in
ascending layer order. That row did not exist anywhere but the host reference, so it is
now captured as the walk runs.

### What the row is, and why the capture point is the whole rule

The reference appends to `Model.forward`'s `main_hiddens` between the Engram write and
the layer's own work, so the row is the BLOCK INPUT of each of `dspark_target_layer_ids`
(`[37, 38, 39]` on this checkpoint) and not the layer's output: an output-captured row
conditions every draft on a state the target had already moved past, and nothing about
the result looks wrong until an acceptance rate comes out near zero. The mean is over
the hyper-connection copies (`h.mean(dim=2)`), which is a plain average and not the
attention sub-layer's weighted collapse.

On the device that average needs no new kernel: `dsv41_hc_pre_f32` computes
`sum_copy mix[copy] * x[copy][dim]`, so a mix of `1 / hc_mult` IS the mean. The capture
is one dispatch of that kernel per target layer per position, encoded at the top of
`.hcAttentionCoefficients` — the first stage that reads the streams without writing
them, i.e. the buffer still holds exactly the block input. It is off unless
`capturesDraftInput` is set, and the buffer is `[slot][target][hidden]` (60 KB for
3 x 5,120, times the block width), allocated at load only when the checkpoint names
target layers.

### Verified against the host, not against itself

`testTheDeviceCapturesTheDraftersConditioningRow` runs the same tokens through
`DeepSeekV41CPUForward` and the executor on a synthetic checkpoint whose target list is
written DESCENDING (`[1, 0]`), so a capture that followed the config's spelling instead
of the sorted order would come out swapped, and compares `mainHidden(position:)` with
the captured row for every position of a walk and for every slot of a block. The block
half is the one that catches a shared offset: one row written to slot 0 for all slots
passes the walk and leaves the block holding its last position only.

The bar is relative and 5e-4, which is where the measurement put the noise: 1.4e-4 at
the first position and 6e-5 after, against the 2e-3 the walk's own logits are held to.
Every way of getting this row WRONG is definitional — the output instead of the input, a
weighted collapse instead of the mean, the config's order instead of the sorted one —
and each of those is O(1), not O(1e-4).

### On the published checkpoint

`testWhatTheDraftersConditioningRowHoldsOnThePublishedCheckpoint` prints the row for 14
prompt positions. Its width is `3 x 5,120 = 15,360`, which is exactly what
`mtp.0.main_proj` is published as ([5120, 15360]), so the geometry agrees with the
weights the drafter will actually read.

The magnitudes are worth recording because they are NOT uniform, and a drafter fed the
raw row has to cope with that:

| position | layer 37 \|x\| | layer 37 max | top logit |
|---|---|---|---|
| 0 | 906,452 | 278,807 | 13.2 |
| 1 | 148 | 7.98 | 23.1 |
| 2 | 429,057 | 117,249 | 12.9 |
| 3 | 156 | 12.97 | 21.5 |
| 6 | 187 | 20.35 | 22.6 |

A residual stream two to three orders of magnitude wider than its neighbours' is not a
capture artifact — the synthetic fixture matches the host to 1e-4 — and it is not a
broken graph either: the logits at those positions are in the same 13-24 range as
everywhere else, which is what the per-sub-layer RMSNorm is for. The scale lives in the
residual stream alone, each sub-layer normalizes it away, and the mHC mixes are
Sinkhorn-normalized rows, so the graph is scale-robust by construction. The drafter is
too: `main_proj` is linear and `main_norm` is an RMSNorm, so the row can be fed as-is.
Anything that compares rows by magnitude (a similarity, a distance, a threshold) would
not be, which is why this is written down.

**What is left.** The draft block's forward, which is where the acceptance number comes
from: `mtp.N.*` is a full `Block` with its own window-cache attention, and on the device
that means a `main_proj`/`main_norm` stage, the draft attention (a seeded ring plus the
block's own rows, one shared index list, not causal within the block), the MoE under the
draft expert counts (128 routed, 3 active), and the last stage's head plus the Markov
bias and confidence head. The host route is not available at this size: the draft stage
is 5,120 wide with a 129,280-row head, so its matrices do not fit the in-memory
`DeepSeekV41DenseWeightSource` the host reference reads, and the host has no shard-backed
source at all.

## DSV4.1 #24: THE DRAFTER'S WINDOW RING IS SEEDED ON THE DEVICE, AND A TOKEN'S WORTH COSTS 0.46 ms

#23 captured the row the drafter conditions on. This slice makes the row usable, and
it turned up two collisions that would have made every draft silently wrong rather than
slow.

### What the checkpoint actually publishes under `mtp.N.*`

The host block reads roles; the checkpoint publishes names. 2,401 `mtp.*` tensors, and
the split is not uniform:

| under `mtp.0` | count | under `mtp.2` | count |
|---|---|---|---|
| `attn.*` (wq_a/wq_b/wkv/wo_a/wo_b/sink/norms) | 13 | `attn.*` | 13 |
| `ffn.experts.E.w1/w2/w3` (+ scales) | 768 | `ffn.experts.E.*` | 768 |
| `ffn.gate.{weight,bias,bias_vl}` | 3 | `ffn.gate.*` | 3 |
| `ffn.shared_experts.w1/w2/w3` | 6 | `ffn.shared_experts.*` | 6 |
| `hc_attn_*`, `hc_ffn_*` | 6 | `hc_*` | 6 |
| `attn_norm`, `ffn_norm` | 2 | `attn_norm`, `ffn_norm` | 2 |
| `main_proj`, `main_norm` | 2 | `norm`, `markov_head.embed/head`, `confidence_head.proj` | 4 |

That is `DSparkBlock.__init__`: stage 0 owns the target projection, the LAST stage owns
the output norm and both heads, and `embed`/`head` are the backbone's own modules aliased
into every stage (`self.mtp[-1].embed = self.embed`). The catalog already classified all
of it — the names resolve, nothing was missing — so this was a mapping to confirm rather
than a mapping to invent.

### Two collisions, both silent

`DeepSeekV41DeviceWeights.tensor(_:layer:expert:)` had no namespace. `mtp.0.attn.wkv`
and `layers.0.attn.wkv` are two tensors under one `(role, layer)` pair, and every draft
lookup would have resolved to the backbone's bytes: the same shape, no error, a draft
conditioned on the wrong matrix.

The vector bank keyed gains by `role|layer` as well, so `mtp.0.attn_norm` and
`layers.0.attn_norm` shared one buffer — whichever was stored last won, and the norm
before every draft attention would have used the backbone's gain.

Both now carry a scope. The device lookup defaults to `.backbone`, so every existing call
site is unchanged, and the bank's draft entries are staged from the CATALOG's stage list
rather than the config's count, because the two disagree in exactly the case that
matters (a checkpoint that publishes `mtp.N.*` without declaring the stages).

### `main_x`, and a ring instead of a slot

`main_x = main_norm(main_proj(main_hidden))` is one FP8 block-scaled GEMV (5,120 x 15,360
on the published checkpoint) plus an RMSNorm, both already proven by the backbone. The
input is the capture itself: `main_proj` reads the target layers' means laid end to end,
which is how the capture stores them.

`testTheDeviceProjectsTheDraftersConditioningRow` checks it against
`DeepSeekV41DSparkBlock.mainX` on a two-stage synthetic checkpoint whose draft expert
counts differ from the backbone's (3 routed / 2 active), and whose `mtp.0` tensors are
byte-distinct from `layers.0`'s, so a lookup that ignored the namespace fails at the
first position rather than at the first draft. Relative 5e-4, the same bar the capture
uses.

The capture itself had to change shape to serve the ring. It held one row per BLOCK
slot; the seed needs the last `window_size` positions, which is 128 on the published
checkpoint and far more than a block. It is now `[position % window][target][hidden]`
(7.9 MB at the published widths), and `capturedDraftInput(atPosition:)` returns nil for a
position the ring no longer holds rather than a row from an earlier turn.

### The seed is one rule, not two

`DSparkAttention.forward`'s `start_pos == 0` branch writes the prefix's latents and
returns, and for a prefix longer than the window it splits the tail around a cut of
`seqlen % window`. Written as a position walk, both cases are the same statement: ring
slot `p % window` holds `kv_norm(wkv(main_x[p]))` with its trailing rope payload rotated
at `p`. The rotation in the reference exists only because it fills the ring in one shot.

A device run is therefore "for each position the rings do not hold, project and store",
and a run that crosses the end of the ring becomes two runs because a run is contiguous
in both the ring and the positions. `testTheDeviceSeedsTheDraftRingsFromTheTargetsLatents`
seeds six positions over a four-row window — so the wrap is exercised, not assumed — and
compares every slot of every stage against `DeepSeekV41DSparkBlock.seed`, relative 5e-4.

### On the published checkpoint

`testWhatTheDraftRingsHoldOnThePublishedCheckpoint` walks eight positions, seeds, and
prints each stage's ring. The geometry checks itself: `mtp.0.main_proj` is [5120, 15360]
for three 5,120-wide targets, `attn.wkv` is [512, 5120] with an E8M0 companion, so the
seed takes the block-32 path the synthetic (bf16, 64 x 48) cannot reach.

| stage | slot 0 \|x\| | slot 0 max | slot 7 \|x\| | slot 7 max |
|---|---|---|---|---|
| 0 | 10.08 | 2.94 | 11.06 | 4.09 |
| 1 | 11.21 | 1.70 | 11.65 | 1.90 |
| 2 | 10.82 | 1.85 | 12.22 | 1.82 |

Over 512 values that is an RMS of 0.44-0.54, i.e. the latents of a normalized record —
and unlike the conditioning row (#23, magnitudes to 906,452), these are uniform, which is
what an RMSNorm applied before the rope should produce.

Timing: the window is **128**, not the 640 the measurement fixtures use, so a ring is
256 KB per stage and three stages hold 768 KB. A cold seed of 8 rows costs 14.76 ms
(first dispatches, pipeline warm-up, first touch of every buffer); one WARM row costs
**0.46 ms**, which is the number a live turn pays, because a decode step adds exactly one
position. Against a ~150 ms decode step, maintaining the drafter's ring is free.

**What is left.** The draft attention itself (the ring plus the block's own five rows,
one shared index list for every query, not causal within the block), the MoE under the
draft expert counts, and the last stage's head plus the Markov bias — then an acceptance
rate, which is the number that decides whether any of this ships.

## DSV4.1 #24b: THE DRAFT BLOCK'S ATTENTION, AND WHY ITS INDEX LIST IS THE WHOLE RULE

The draft attention is the one attention in this file that reads no cache. Its keys
are the stage's own ring (`seedDraftRings`, #24) followed by the block's five rows, and
ONE list serves every query: `get_dspark_topk_idxs` builds `arange(min(window,
start_pos + 1)) ++ window + arange(block_size)` once and expands it across the block, so
draft position 0 attends over draft position 4. Every other attention here is causal
within a position's own write, which is exactly the assumption a port carries in by
habit.

Three consequences, all of them load-bearing:

- **The block's rows come from the draft stack, not the ring.** The ring holds what the
  TARGET saw (`kv_norm(wkv(main_x))`); the block's rows are
  `kv_norm(wkv(attn_norm(hc_pre(stack))))` of the drafter's own tokens. Mixing them up
  is a silent swap of one 512-wide record for another.
- **Ring order, not position order.** After a wrapped seed, slot order and position
  order disagree. The reference gathers slots `0..<min(win, start_pos + 1)`, so the port
  copies the ring from slot 0. The parity fixture runs six positions over a four-row
  window for exactly this: its ring at the draft's position is slots 0-3 = positions
  4, 5, 2, 3, and a position-ordered copy fails.
- **Every slot reads the SAME rows.** The kernel's slot dimension exists so a block can
  share one launch, and `keyRowStride` is what separates slots when they read different
  rows. The draft's index list is identical for every query, so the stride is ZERO and
  the per-slot row count repeats. That is a real use of the kernel, not a workaround:
  the alternative is five launches of the same dispatch.

The block is `dspark_block_size` positions (5 on the published checkpoint), and their
query positions are `start_pos + 1 ..< start_pos + 1 + block_size` because the ring row
for `start_pos` is already written when the drafter runs.

### Verified against the host block

`testTheDraftAttentionMatchesTheHostBlock` compares the stack `hc_post` leaves behind
(what the host records as `Trace.postAttention`) for every position and every
hyper-connection copy, relative 5e-4, on a fixture with a 4-row window so the ring is
wrapped and out of position order. That one comparison covers the query path
(`wq_a` → `q_norm` → `wq_b` → rope), the block's own latents (`wkv` → `kv_norm` → rope),
the key block's two halves, the attention, the output's rope conjugate and the grouped
`wo_a`/`wo_b`. It passed first try, which is the point of having built the seed and the
projection in the two slices before it.

The device takes the FP32 attention kernel (`dsv41_attention_csa2_f32`, now registered as
`pipelines.attentionF32`) rather than the shipping FP16 one: the ring is floats, and an
fp16 key block would put a rounding floor under the parity bar that hides which row or
which rope slice a mistake is in. The ring is 128 x 512 x 4 = 256 KB per stage, so the
bytes are not the issue; if a later slice wants the half-key kernel it is a drop-in once
the ring is stored as halves.

### On the published checkpoint

The ring probe now also runs one draft attention: 5 positions over an 8-row prefix in
**1.72 ms**, all finite, |x| 15-19 per 5,120-wide copy for the noise positions and 132 for
the first (which embeds the real token, where the others embed
`dspark_noise_token_id` 128,799). That is the shipped 64-head, 512-wide geometry against
the 128-row ring, i.e. the shape the acceptance number will be taken with.

**What is left.** The stage's FFN (`hc_ffn_fn` coefficients, `ffn_norm`, the router under
`draftExpertCounts` — 128 routed, 3 active — and the MXFP4 mixture), the chaining of the
three stages, and the last stage's head plus the additive Markov bias and confidence
head. Then the acceptance rate.

## DSV4.1 #24c: THE DRAFT BLOCK'S MIXTURE, AND THE ORDERING BUG THE PARITY CHECK FOUND

With the attention in place (#24b) the draft block is one feed-forward short of a full
round: `hc_ffn_fn` coefficients, the FFN's `hc_pre`, `ffn_norm`, the router under
`draftExpertCounts`, the routed mixture, the shared expert, and the `pre` mix the next
stage collapses with. All of it is a backbone layer's machinery with two deliberate
differences.

### Two differences, both from the checkpoint

- **The counts are DSpark's.** The published stages route 128 experts and activate 3,
  against the backbone's 384 and 6. `encodeRouter` and `readRouting` now take them as
  parameters instead of reading the backbone's, and the parity fixture sets 3/2 against
  a backbone of 4/2 so a stage that read the wrong pair routes differently and says so.
- **The experts come straight from the shard mappings, not the arena.** The arena is
  keyed by `(layer, expert)`, and a draft stage's layer 0 IS the backbone's layer 0 — one
  key, two different matrices. Sharing it would serve one model's expert to the other,
  which is the same class of bug as the two namespaced lookups #24 removed. The draft
  therefore passes `residents: nil` and pays the mmap fault path; at 9 experts a round
  (3 stages x 3 active x 17.7 MB) it is a cost to revisit only if the drafter ships.

### The bug: the router reads the FFN's NORMED input, not the attention's

The first version of `draftStageOutputs` ran the router as its own step before the
feed-forward, which put it BEFORE `hc_pre`/`ffnNorm` — so the router scored the drafter's
attention input. The symptom was not an exception or a wrong shape: the mixture came out
0.05 off on a 0.37 scale, which reads like a kernel rounding problem, and the routing
carried it: device `[2, 0]` against the host's `[0, 1]`.

The bisect that found it is worth keeping: `Trace.ffnInputs` matched the device's
`workspace.normed` to 6e-8 (so the mix, the coefficient and the norm were right), while
`Trace.ffnOutputs` was 0.05 off (so the mixture was wrong), and printing both sides'
router logits showed the device's logits for the last two slots were IDENTICAL — the same
input row, twice. A stack dump read after a two-stage run is useless here, by the way:
`workspace.normed` and `moeOut` are reused by every stage, so the dump has to be taken
with a one-stage fixture.

The fix is the backbone's own pipeline order, split into two command buffers so the
readback still lands between them: coefficients, mix and norm first, then the router,
then the mixture.

### Verified

`testTheDraftBlockChainsItsStagesLikeTheHostBlock` compares every stage's residual
(`Trace.stageOutputs`) for every position and copy at relative 5e-4, with shared experts
published on both the backbone and the draft side so `encodeSharedExpert`'s namespaced
lookup is exercised too. The mixture itself lands at 1.6e-8 against the host, i.e. float
rounding rather than a rule difference.

**What is left.** The last stage's head — `hc_pre` with the final mix, `norm`, the
129,280-row projection (the backbone's own `head.weight`), the additive Markov bias at
each step and the confidence head — and then the acceptance rate this was all for.

## DSV4.1 #25: THE DRAFTER ACCEPTS 31%, AND THE BLOCK IS TOO WIDE FOR IT

This was the number the last three slices were built to reach, and it is a NO for the
drafter as the checkpoint declares it.

### The full round is on the device and matches the reference

Everything the acceptance number needed now runs on the GPU: the block's attention
(#24b), its mixture and the three-stage chain (#24c), and — `#25` — the head. The head is
two pieces a port gets wrong separately, and both are now pinned:

- **It is the backbone's head.** `DSparkBlock.forward_head` normalizes the LAST stage's
  collapse with the mix that stage's feed-forward handed on and projects with the shared
  `head` module: nothing is published under `mtp.N.head`. The logits are kept for every
  position (a verify needs all five), unlike `finishBlock`, which only reads the last.
- **The collapse is scored BEFORE the norm.** `confidence_head` reads `x =
  hc_pre(h, pre_mix)` — the pre-norm vector — so the head must normalize into a different
  buffer. Normalizing in place passes the logits comparison and then reports every
  confidence as ~0.7 off, which is how this was found.

`testTheDraftHeadAndItsChainMatchTheHostBlock` compares the five logits rows against
`Trace.logits` (relative 5e-4) and then the CHAIN — the drawn tokens and every confidence
— against `DeepSeekV41DSparkBlock.draft`. The chain assertion is the one that catches a
bias applied to the wrong step: a Markov bias from the target's token instead of the
draft's own choice changes which token wins, and the tokens diverge.

### The measurement

`testWhatDSparkAcceptsOnThePublishedCheckpoint` drafts a round, walks the target greedily
over the same five positions, and counts the agreeing prefix. Two continuations, ten
rounds each, so 100 drafts:

| continuation | accepted | per step (s0..s4) | gated | draft cost |
|---|---|---|---|---|
| p0 | 14/50 (28%) | 80%, 30%, 10%, 10%, 10% | 35% | 155 ms/round |
| p1 | 17/50 (34%) | 80%, 60%, 30%, 0%, 0% | 38% | 158 ms/round |

**31% overall against a ~47% bar.** The bar is `#22`'s: a batched verify costs 70.9
ms/position against a walk's 153.3, so a round of five drafts needs ~2.3 accepted to pay
for the verify alone — and 155 ms of drafting on top moves it further.

Three things in that table matter more than the total:

- **The first draft is accepted 80% of the time on BOTH continuations.** That is the one
  content-independent number here, and it is high.
- **The curve dies at step 2.** s1 is 30% on one continuation and 60% on the other, s2 is
  10-30%, and s3/s4 are at the floor. A five-wide chain spends 60% of its drafting and
  80% of its verification on positions that almost never land.
- **The confidence gate helps but does not save it.** Restricting to rounds
  `shouldSubmit` keeps moves 28% → 35% and 34% → 38%: the gate is selecting real signal
  and still an order of magnitude short of the bar.

### What the numbers say to do instead

The value is in the first one or two drafts, so the question is what a NARROW verify
costs. `testWhatAVerifyCostsAtTheWidthsAChainWouldSubmit` prices `prefillChunk` at
widths 1-8 on the same runtime, and it does not settle it — three runs of the same
procedure, the last interleaved one-sample-per-arm:

| run | w1 | w2 | w3 | w4 | w5 | w8 |
|---|---|---|---|---|---|---|
| sequential | — | 154.5 | 56.5 | — | 61.6 | 59.0 |
| sequential | 147.9 | 62.9 | 56.6 | 58.9 | 59.1 | 57.5 |
| interleaved | 111.3 | 105.8 | 85.5 | 88.4 | 86.2 | 79.2 |

Two things are stable and one is not. Stable: width 1 gets NOTHING from batching (a
one-position block is a walk with different bookkeeping — 111-148 ms/position), and
width 3 and up sit on a roofline well under the walk's 153.3 (57-86 ms/position
depending on the run). Not stable: width 2, which one run puts at 2.35x faster than
width 1 and another at 5% faster. The absolute levels also move ~40% between runs, which
is larger than the effect being chased, so the width-2 question needs better
instrumentation before anyone acts on it - the arms have to be isolated (each on its own
runtime, interleaved, several samples) rather than measured in one sequence against a
shared arena.

So the honest economics use the roofline and the acceptance curve, and both sides are
now measured facts rather than projections:

- B=3 costs 3 x 57-86 ms of verify plus ~93 ms of drafting and emits `E[accepted] + 1`
  = ~2.45 tokens, against 2.45 x 153.3 = 376 ms of walking. That is 1.3-1.5x FASTER.
- B=5 costs 5 x 59-86 ms plus 155 ms and emits ~2.55 tokens, against 391 ms. That is
  between a wash and 15% slower, with the acceptance rate (31%) as the reason.
- B=2 is the case the curve likes most and the case the verify measurement cannot
  resolve.

What is NOT in doubt is the direction: the drafter's declared block is wider than its
acceptance curve can fill, and the useful configuration is one the checkpoint's own
`dspark_block_size` does not describe. Running a 2- or 3-wide block is legal — the block
size only decides how many positions the draft block runs — but it changes the graph
(the block's own rows are part of every query's key list, so a narrower block attends
over a shorter list and its acceptance is NOT the per-step rate measured at width 5).
That measurement is the next step, not the verify.

The drafting cost is also pessimistic as measured: the draft mixture reads experts
through the shard mappings with `residents: nil` rather than from the arena, because the
arena is keyed by `(layer, expert)` and collides with the backbone (see #24c). That is
worth 100 ms of the 155 and would be the first thing to fix if the drafter ships.

**What was deliberately NOT built.** Per-position logits in `finishBlock` and a cache
rewind for a rejected suffix. Both exist only to serve a verify, and a verify at the
declared width loses ~30% against walking. Building them now would be building the losing
half of a design the measurement has already rejected.

## DSV4.1 #26: A TWO-WIDE DRAFT ACCEPTS 50-55%, AND STILL LOSES TO WALKING

`#25` ended on a question the accept curve made: the chain dies at its second step, so a
narrower block should accept a larger share of what it drafts. It does. It then loses to
walking anyway, because a two-wide verify buys almost nothing and the drafter's own
forward costs more than it buys. This slice measured both halves on the published
checkpoint and the answer is a NO for every width, not just the declared one.

### The acceptance, by width

`testWhatANarrowerDraftBlockAcceptsOnThePublishedCheckpoint` runs `#25`'s protocol at
widths 2, 3 and 5. Each width gets its OWN runtime, because the width sizes the draft key
block, the head's logits buffer and every block workspace at construction: the config's
`dspark_block_size` is overridden before the runtime is built, which is exactly the graph
a checkpoint declaring that width produces. Width 5 is in the sweep as the control.

| width | p0 | p1 | p0 per step | p1 per step | gated | draft ms/round |
|---|---|---|---|---|---|---|
| 2 | 10/20 (50%) | 11/20 (55%) | 70%, 30% | 60%, 50% | 67% of 12, 69% of 16 | 90, 56 |
| 3 | 11/30 (37%) | 12/30 (40%) | 60%, 30%, 20% | 60%, 30%, 30% | 42% of 24, 50% of 24 | 73, 84 |
| 5 | 14/50 (28%) | 17/50 (34%) | 80%, 30%, 10%, 10%, 10% | 80%, 60%, 30%, 0%, 0% | 35% of 40, 38% of 45 | 155, 154 |

The width-5 row reproduces `#25` exactly, which is what makes the narrow rows comparable:
same protocol, same prompts, same rounds, one variable.

Two things in that table matter more than the totals:

- **The first draft gets WORSE when the block narrows** — 70%/60% at width two against
  80%/80% at width five. This is not noise; it is the graph. The block attends over its
  own rows as well as the ring (it is not causal inside itself), so removing siblings
  changes what position 0 sees. A narrow block is a different drafter, not a truncation
  of the wide one.
- **The aggregate still roughly doubles**, 31% to 50-55%, because the tail the curve dies
  in no longer exists. The win is where the chain stops, not a better first token. The
  second step is 30-50% at width two and 30-60% at width five — statistically the same
  marginal draft.

The confidence gate is much stronger here: it keeps 67-69% acceptance among the rounds it
would submit, against 35-38% at width five. It still only submits 60-80% of rounds.

### The verify, measured like-for-like

`#25`'s width sweep printed width-2 and width-3 verify costs of 105.8 and 85.5 ms/position
and invited the reading "a narrow verify is 2x cheaper than a walk". That reading was
wrong, and `#26` shows why: those blocks ran over a 24-token list the arena already held
after the first round, so they read no expert bytes at all, while the 153.3 ms/position
baseline they were compared against was a realistic walk WITH misses. It is the trap `#22`
recorded, and it cost a projection.

`testWhatANarrowVerifyCostsAgainstAWalkInTheSameRuntime` fixes the procedure: every sample
resets the executor and re-walks the same prefix, the arm order rotates, and each round
times a one-position WALK so every width is reported as a ratio to a walk measured in the
same state. Medians of six samples:

| arm | ms/pos | ratio to walk | MB/pos |
|---|---|---|---|
| walk | 64.8 | 1.00 | 0 |
| w2 | 53.5 | 0.83 | 0 |
| w3 | 52.0 | 0.80 | 0 |

That is the ZERO-MISS regime — the byte counter reads 0 in both arms, as it must, because
the arm tokens repeat. A live turn never repeats a position, so
`testWhatANarrowVerifyCostsWhenItsExpertsAreCold` prices the same two arms over DISTINCT
two- and three-token runs (a fixed list, not a continuation the test just walked, which
would pre-load exactly the experts the arms are supposed to miss), re-walking the same
prompt and seed each sample so both arms start from identical cache and arena state:

| arm | ms/pos | ratio to walk | MB/pos |
|---|---|---|---|
| walk | 194.8 | 1.00 | 645 |
| w2 | 173.6 | 0.89 | 601 |
| w3 | 211.0 | 1.08 | 998 |

Both regimes agree on the ratio to within a few percent (0.83-0.89 at width two), which is
the useful part: **a two-wide block is ~11-17% cheaper per position than walking it, not
2.16x cheaper.** `#22`'s 2.16x is a property of width EIGHT, where one expert union
replaces eight per-position sets and the block reads 7.6x fewer bytes. Width two has two
positions to amortize one round trip over, and width three is actively worse than walking
(1.08x, 55% MORE bytes per position) because the union it fills up front is wider than the
per-position sets it replaces. The byte model in `#17` warned that the block's traffic is
what to watch; here it goes the wrong way.

### The economics, at last with both sides measured

A round emits `1 + E[accepted]` tokens and costs `draft + width x verify`. Walking the same
tokens costs `(1 + E[accepted]) x walk`.

| width | emits/round | round cost | walking the same | result |
|---|---|---|---|---|
| 2 | 2.05 | 73 + 2x173.6 = 420 ms | 2.05 x 194.8 = 399 ms | 1.05x SLOWER |
| 3 | 2.15 | 78.5 + 3x211.0 = 712 ms | 2.15 x 194.8 = 419 ms | 1.70x SLOWER |

At the zero-miss walk (64.8 ms) width two is 1.4x slower. Break-even for width two needs a
walk of ~270 ms/position — beyond any walk this port has measured (153.3 realistic at
`#22`, 194.8 here, 263 at the very coldest) — and that is with the drafting cost as
measured, which is itself the pessimistic one (see below).

So the answer is not "the block is too wide". Narrowing the block DOES raise acceptance,
roughly doubles it, and the round still loses, because the batching it gives up is worth
more than the acceptance it gains. `#25`'s B=3 projection of 1.3-1.5x came from the
mixed-regime verify number and is withdrawn.

### What this means and what was NOT built

The drafter would need BOTH of these to break even at width two, and neither is close:

- The draft's own forward cut from 56-90 ms to under ~50 ms per round. Its mixture reads
  experts through the shard mappings with `residents: nil` (the arena collides on
  `(layer, expert)`, `#24c`), which is most of its cost; making that resident might reach
  ~25-30 ms.
- A walk slower than ~270 ms/position in the steady state, i.e. persistent arena misses.

Even with both, the margin is single-digit percent, which is not worth per-position logits
in `finishBlock` plus a cache rewind for a rejected suffix (`noteStored` only appends; ring
position, `tokenHistory`, `deadTokens` and the compressor group phase would all have to
rewind together). Those two pieces were already deliberately unbuilt at `#25`; the narrower
block does not change that, it closes the question.

What is now on record: DSpark's greedy chain verification is a NO at widths 2, 3 and 5 on
an M5 Max with this SSD, and the reason is not acceptance (50-55% is fine) but that a
narrow batch is the regime where batching stops paying. The drafter's remaining value would
have to come from somewhere other than a linear chain's verify, and no such use is
proposed here.

## DSV4.1 #27: THE TOOL DIALECT IS MEASURED, AND IT IS NOT THE ONE THE VOCABULARY SUGGESTS

The open item was "the tool dialect". Nothing in the checkpoint defines it: `tokenizer_config.json`
carries no `chat_template`, `encoding/` has no tool path, and the README says nothing about tools.
What the checkpoint DOES carry is a vocabulary: nine `tool`-joined special tokens (calls begin/end,
call begin/end, sep, outputs begin/end, output begin/end, ids 128806-128814) and one bare DSML token
(128825, U+FF5C "DSML" U+FF5C). Which of those the model actually uses, and what grammar surrounds
them, is a question about weights, so it was answered by asking the model.

### The outbound half: the PROMPT SHAPE picks the dialect

`testWhatToolDialectTheCheckpointEmits` renders four prompts over one tool (`get_current_time`,
`city` required) and runs a greedy turn on each. Three of them hand the model a call SCHEMATIC built
from the marker strings; the fourth hands it the instruction block this app actually ships. Same
catalogue, same `## Tools` header, same preamble, same question: only the call grammar differs.

| variant | prompt tokens | finish | generated | emitted |
|---|---|---|---|---|
| schematic+chat | 168 | eos | 42 | plain `tool_call_name` / `tool_call_arguments` tags |
| no-schematic | 86 | eos | 37 | `tool_call` tag around `{"name": …, "arguments": …}` |
| schematic+think | 168 | eos | 51 | reasoning, then the plain tags |
| **instructions-only** | 306 | eos | 38 | **the DSML tags** |

The first three reproduce what they were shown, which is what a chat prompt with a template in it is
supposed to do: a model shown a schematic copies the schematic. The fourth is the one that matters,
because it is the prompt this app builds, and it answers in DSML. The generated pieces of that turn,
one token each, are the proof of how the tag is assembled:

```
"\n\n" "<" "<bar>DSML<bar>" " calls" ">\n" "<" "<bar>DSML<bar>" " invoke" " name" "=\"" …
```

So an opening DSML tag is `<`, then ONE vocabulary id, then ordinary text with a LEADING SPACE
(`" invoke name=\""`). The closer is `</`, the same id, then `" invoke>"`. The id is the whole tag
machinery; everything else is spelled out as text, which means the tags cannot be matched by token
id and must be matched as strings. (`<bar>` stands for U+FF5C, written that way here because a
rendered view strips it.)

### The discovery: an agent turn does not freeze on an eos

`testAnAgentTurnFreezesOnAToolCall` runs the whole agent path (the harness-shaped system prompt, the
turn runner, the engine's freeze hook, the call rewrite) on real weights. Before this slice the
dialect knew only the plain-text form, and the turn came back `eos` after 41 tokens with zero calls
parsed:

```
\n\n<calls>\n<invoke name="shell_run">\n<parameter name="command" string="true">ls -la /tmp</parameter>\n</invoke>\n</calls>
```

(also with the DSML id elided). It is a MULTI-CALL container: one `calls` wrapper holding one
`invoke` per tool, each carrying `parameter` elements whose `name` is the argument's JSON name and
whose `string="true"` marks a literal value (`string="false"` marks JSON, so a number stays a
number). That is a richer grammar than the plain-text form: typed arguments, no JSON-in-text
parsing, and a name that cannot be confused with prose.

The failure is the reason the test exists. A parser that only parses its own fixture is a parser for
the fixture: this one parsed every hand-written plain-text case and then dropped the generation it
was written for.

### The inbound half: not discriminating

`testWhatToolResultShapeTheCheckpointReads` gives the same conversation the same payload
(`{"local_time": "14:05"}`) inside five different containers: the vocabulary's `output` pair, its
`outputs` pair, a plain `tool_call_output` tag, a `tool_response` tag, and bare JSON. All five
produced the IDENTICAL answer, "It's currently **14:05 (2:05 PM)** in Paris.", in 16 tokens each.
The checkpoint reads the payload wherever it is, so the container is free to be the one that costs
least: the family's own `output` pair, one id on each side.

Note what this does NOT support: an earlier note in `AGENTS.md` claimed the model's tool results are
`<tool_result>…</tool_result>` inside a user turn. That spelling appears nowhere in the checkpoint's
files (no `.py`, no `.md`), and the measurement above shows the model does not need it.

### What was wired

All of it is a daisy-chain into the canonical call block
`<tool_call><function=NAME>JSON</function></tool_call>`, because that is the only shape every
downstream consumer (executor, HITL gate, gesture split, timeline, history) understands.

- `DeepSeekV41ToolDialect` grew the DSML grammar: `dsmlTag` / `dsmlCallsOpen` / `dsmlInvokeTag(name:)`
  / `dsmlParameterTag(name:isString:)`, a `canonicalizeDSMLBlocks` scan that turns each `invoke` into
  a canonical block and then drops the container tags, and a parameter reader that keeps
  `string="false"` values as JSON. The tag constants are BUILDERS, never literals: a rendered view
  strips U+FF5C, so a typed tag looks right and matches nothing.
- Truncation still does not execute. An invoke with no closer, or a parameter with no closer, is left
  as text: a call the model never finished asking for must not run with a half-built argument list.
- `shouldFreeze` now stops on the `calls` closer, or on an `invoke` closer when the model wrote a
  single call with no container. The container close is one token after the last invoke close, which
  is the cheapest complete point (`#26`'s economics: a token costs ~263 ms).
- `instructions()` now TEACHES DSML, built from the tag builders. Teaching a dialect the model was
  not post-trained on is how a call becomes prose, and the measurement says which one it was
  post-trained on.
- The grammar mask stays OFF for this family. Its marker vocabulary is Qwen's, and forcing a
  `<function=>` shape on a model that writes DSML tags is how a valid call becomes unsamplable.
  Argument validation happens at execution instead.
- The chat bubble truncates at the first DSML tag, beside the existing plain-tag strip.
- The plain-text and JSON forms remain PARSEABLE (they are what the schematic prompt draws), they are
  just no longer what the instructions ask for.

After the change the same agent turn ends `toolFreeze` after 41 tokens and reports
`shell_run["command": ls -la /tmp]`. The 41 tokens are unchanged: freezing on the container close
costs nothing here because the container close IS the model's last token.

### The numbers that are worth remembering

| item | value |
|---|---|
| DSML token id | 128825, U+FF5C "DSML" U+FF5C, no word-joining space inside |
| tool special tokens (unused by the model) | 128806-128814, one id each |
| schematic prompt → | 42 tokens, plain-text tags (37 with no schematic) |
| shipped instruction block → | 38 tokens, DSML |
| inbound container | five shapes, all read, 16 tokens each |
| agent turn after the fix | `toolFreeze`, 41 tokens, 1 call parsed |


## DSV4.1 #28: THE COMPRESSED ROWS A QUERY READS ARE A SCORED TOP-K, AND THE PORT WAS READING THE NEWEST ONES

**Question.** `attentionRowPlan` selected compressed rows with a rule that never
looked at a value: read the newest `index_topk` rows, because those are the ones a
position can see. `DeepSeekV41Conformance.candidateSelection` was `.unverified`
and the note next to it said the reference "keeps `candidateScores`" — but nothing
compared the two. Does the reference select the newest rows?

**Answer: no, and the difference starts at ordinary context.**

`inference/model.py` `Indexer.forward` is a small side attention: it projects the
layer's own `qr` into `index_n_heads` of `index_head_dim`, scores every compressed
key with `(relu(dot) * weight).sum(heads)`, masks the rows the query has not
reached yet, and keeps the best `index_topk`. `select_candidate_blocks` is the
level above it, and only the candidate source runs it.

| rule | published value | when it starts to bind |
| --- | --- | --- |
| `index_topk` (level two) | 512 | position 513 on the twenty ratio-1 layers, 1025 on the ratio-2 half |
| `candidate_topk_blocks` (level one) | 2048 blocks x 8 rows | 16,384 compressed positions, i.e. 16,384 tokens on the ratio-1 half |

So the level the port got wrong is the one that binds first, and the level it
never implemented (blocks) is inert on this checkpoint for any ordinary context.
That is worth stating plainly: 20 of the 40 layers read one compressed row per
token, so "newest 512" and "best 512" diverge from token 513.

**What changed.**

* `DeepSeekV41Indexer` is now an exact port of the reference: `positionScores`
  (per-head rectified dot, weighted, summed), `candidateBlocks` (block score is
  the block's best row, and the partly-filled frontier block is pinned at `+inf`
  so it cannot lose to an older full block), and `selectedRows` (the cut, then the
  reach filter — in that order, which can leave the selection one row short).
* `Tools/dsv41_goldens.py indexer` runs `Indexer.forward` and
  `select_candidate_blocks` at a tiny config for four cases: a prefill chunk whose
  queries reach different prefixes, a decode step, a candidate source publishing a
  mask, and the next layer rescoring inside it. `DeepSeekV41IndexerGoldenTests`
  pins the scores AND the selections.
* The host reference computes the index keys at the pooling point (`k_norm(wk(latent))`
  on the RoPE-free latent, rotated at the group's first position), scores per
  position, and reuses the source's list for the layers between two sources.
* The device path does the same with `dsv41_indexer_scores_f32` (rewritten: it was
  the Qwen max-over-heads block scorer and nothing called it), and reads the score
  vector back. That readback is the cost the old note was avoiding, and it is one
  wait per index-source layer: the dense half now ends after the indexer and a
  second submission carries the gather, because the row list the gather binds is a
  host value.

**Numbers worth remembering.**

| | value |
| --- | --- |
| index sources on the published checkpoint | 8 (`[2, 8, 14, 20, 24, 28, 32, 36]`), of which 4 own the keys |
| first position where the old rule is wrong | 513 (ratio-1 layers), 1,025 (ratio-2) |
| level one binds at | 16,384 compressed positions — never at ordinary context |
| golden fixture tie rows | 0 (a tie would be a row every head voted against) |
| DeepSeek suite | 479 passed, 0 failed, 35 skipped across 82 classes |
| Metal validation | 0 errors on the same 479 |

**Two things the golden caught that a fixture would not.** The cut runs before the
reach filter, so a row a query cannot see can take a slot and then be dropped: with
three visible rows and `topk` 2 the reference selects ONE row, not two. And the
source layer does not mask itself — `is_candidate_source` publishes the blocks and
`uses_candidates` (strictly later layers) applies them, which is why a consumer's
selection differs from its source's on the same keys.

**Still open.** The indexer's FP4 query/key quantization is skipped like every other
activation quantization in the dense path, and exactly equal scores break to the
lower row index where `torch.topk` is unspecified. Neither was measured against real
weights: the whole slice is synthetic-checkpoint coverage, and the published
geometry's first real run has not happened.

## DSV4.1 #29: THE INDEXER'S KEY CACHE WAS PRICED IN HALF THE WIDTH IT IS ADDRESSED IN

**Question.** `#28` replaced the port's invented "the newest `index_topk` compressed
rows" scorer with the checkpoint reference's two-level rule, and pinned that rule
against `Indexer.forward` and `select_candidate_blocks`. Every vector behind it was
generated at a six-wide config: `index_topk` 3 over 8 produced rows. Below the
budget every reachable row is selected, so the scored cut is invisible there. Does
the rule still change the answer at the published geometry, where `index_topk` is
512 and a ratio-1 layer produces one row per position?

**Answer: it does from position 512, and the first real-weight run found a cache bug
that no synthetic fixture could see.**

The probe prefills 576 positions of the real checkpoint through the block path and
reads back, per index-source layer, the score vector and the row list the gather
bound. The first run scored **exactly zero for 287 of the 576 rows, at all five
ratio-1 sources, at the same rows** — and the number of *distinct* scores was 290 at
every layer, which no learned arithmetic produces. That is a region with nothing
written in it, not a rectifier.

The cause is one number in two files. `DeepSeekV41CacheGeometry.indexerKeyBytes`
priced an index key at `MemoryLayout<UInt16>.stride` while every access used four
bytes: `dsv41_copy_f32` writes `k_norm(wk(latent))` as floats and
`dsv41_indexer_scores_f32` reads the region as `device const float*`. So each key
row overran its own 256-byte slot by half into the next row's, the last row of every
owner wrote past the end of its region, and the score kernel read half-widths back as
floats, which are denormals. `DeepSeekV41MemoryPlan` priced the same two bytes, so
the load log agreed with the allocation and both disagreed with the traffic. The
synthetic checkpoint never noticed because its rows still fit inside the region it
was given.

Fixed by pricing the cache at the width both kernels address, in both files, and
pinned by `DeepSeekV41IndexerKeyLayoutTests`: one key is `index_head_dim` floats, the
stride between rows is that width, the last row ends exactly at its owner's region
boundary, and the memory plan's price equals the layout's allocation.

**Where it bites.** Nothing on this model is optional about it: a compressed layer's
gather binds the row list the scores produce, so before the fix the port was reading
287 rows nobody had written plus aliased halves of written ones. On the twenty
ratio-1 layers the whole history beyond row 288 was effectively invisible.

**Numbers, after the fix** (576 positions, one prefill, `M5 Max`, 475 GB checkpoint):

| | value |
| --- | --- |
| prefill of 576 positions | 105 s (~0.18 s/position) |
| index sources recorded | 5 ratio-1 (`20, 24, 28, 32, 36`) of 8; total 8 with the ratio-2 half |
| positions past the budget | 64 (position 512 onward) x 5 layers = 320 selections |
| those that are not the newest 512 rows | **320 of 320** |
| exact-zero scores per layer | 287 before the fix, **0** after |
| distinct scores per layer | 290 before the fix, **576** after (no ties anywhere) |
| score range, layer 20 at position 575 | -3.331 .. 3.802, median -1.011, 135 positive / 441 negative |
| rank of the newest row, layer 20 / 24 at 575 | 62 / 464 (of 576) |
| rank of row 0, layer 20 / 24 at 575 | 43 / 39 |
| distinct selections at position 575 | 6 across 8 sources (the 3 ratio-2 sources keep every row) |
| probe run | 1 test, 0 failures, 111.8 s |

What the selection actually does at 576 rows is less dramatic than "it reads the
oldest rows" and more useful: the cut keeps 512 of 576, the 64 it drops are the
lowest-scoring rows wherever they sit, and the oldest rows survive on their scores
(row 0 ranks 43rd, the newest row 62nd at layer 20). The set differs from a recency
window at **every** budget-exceeding position, which is the property `#28` claimed and
this run is the first evidence for.

Level one is confirmed inert at this checkpoint rather than assumed inert: the
candidate source's mask came back all-`true` at every position, because 2,048 blocks
of 8 rows is 16,384 rows and the run reaches 576.

**A second blind spot, in the fixture rather than the port.** With the stride fixed,
the device/host parity case for compressed rows failed at position 5 of
`compressedSmall` — the first position where `index_topk` 2 binds over 3 produced
rows. The cause was the fixture: it published
`layers.N.attn.indexer.wk.weight` as `[index_head_dim, hidden]`, where the checkpoint
publishes `[128, 512]` and the reference builds `wk = Linear(head_dim,
index_head_dim)` over the compressor's pooled LATENT. `DeepSeekV41Matrix.apply`
returns zeros when the vector is narrower than the matrix row (deliberately, so a
plan-level bug reads as a zero block rather than an out-of-range read), so the host's
index keys were all zero and its scores were noise around 1e-4. Every fixture before
this one had `index_topk` at or above the produced row count, so the cut never bound
and a zero score vector selected exactly the rows a real one would. The fixture now
publishes the latent width, and the parity case compares a binding cut, which is the
comparison it always meant to make.

**Two things the run now asserts that a fixture cannot.** The newest row is not
preferred: it ranks 62nd of 576 at one source and 464th at another, so a query at
position 575 does not weight its own newest compressed row above everything older.
And two index sources over the same key cache choose different rows at the same
position (6 distinct lists across 8 sources), so the selection depends on the query
rather than on the row geometry.

**Still open.** The indexer's FP4 query/key quantization is still skipped, exactly
equal scores still break to the lower row index where `torch.topk` is unspecified
(the real run has no ties to break), and `candidateSelection`'s golden vectors remain
synthetic: this contribution is the rule and the cache it reads, not a comparison of
the port's scores against the checkpoint's own `Indexer.forward` at real weights.

## DSV4.1 #30: THE INDEXER'S SCORES ARE THE REFERENCE'S, AND TWO LONG PREFILLS DO NOT AGREE

**Question.** `#29` proved the port now reads the rows its own scores rank first on the
real checkpoint. It did not prove those scores are `Indexer.forward`'s: every vector
behind the rule was generated at a six-wide config, where `index_topk` 3 over 8 rows
makes the cut invisible, and `Indexer.forward`'s own FP4 activation quantization was
never measured either. So: on the checkpoint's tensors and a real prefill's
activations, does the port score what the reference scores -- and does the
quantization the port skips change which rows it reads?

**Answer: yes to the first (worst |delta| 2.4e-6 over 576 rows, the selected 512 rows
identical), 3 of 512 rows to the second, and along the way the forward pass turned out
not to be reproducible at 576 positions.**

### The comparison

`Indexer.forward` cannot be run end to end here: 475 GB does not fit, so the reference
cannot produce `x`, `qr` or the pooled latents itself. It CAN be handed them.
`DeepSeekV41MetalForward.indexerInputRequest` is an opt-in probe that copies one
layer's pooled latent per position into a caller's buffer as each group completes, and
`indexerInputDump` carries the two query-side vectors. `DeepSeekV41PublishedTurnTests`
`testWhatTheIndexerWasGivenOnThePublishedCheckpoint` prefills 576 positions of the real
checkpoint, dumps layer 20, and `Tools/dsv41_goldens.py indexer_real` reads the
checkpoint's own `layers.20.attn.indexer.{wq_b, weights_proj, wk, k_norm}` out of the
shards, dequantizes `wq_b`'s FP8/E8M0 32x32 blocks, and runs the reference's `Indexer`
at the published geometry on those activations. Both sides then see the SAME numbers,
which is the only way the difference between them can be the indexer's arithmetic.

Two independent dumps were scored, because the forward is not reproducible (below),
and the agreement is the same both times:

| | dump A | dump B |
|---|---|---|
| layer / position | 20 / 575, of 576 prefilled | same |
| produced rows | 576 | 576 |
| worst absolute score delta | **2.380e-6** (row 385; reference -2.881097, port -2.881099) | **2.149e-6** |
| rows differing by more than 1e-5 | **0** of 576 | **0** of 576 |
| worst relative delta | 1.195e-4 | 1.264e-4, both at near-zero scores: a cancellation, not a mismatch |
| selected rows | **512 vs 512, identical** | **512 vs 512, identical** |
| port score range | -3.3318 .. 3.8019 | -3.3509 .. 3.7777 |
| cut score (512th) | -2.2896 | -2.2631 |

The remaining 2.4e-6 is the summation order: the port's kernel multiplies each head's
`relu(dot)` by `weight * (128**-0.5 * 32**-0.5)` inside the sum while the reference
folds the scale into the weights first, its 128-wide dot product uses four
accumulators where `einsum` uses another order, and `wq_b` runs as a dequantized GEMV
here against an FP8 GEMM there. Nothing about the selection depends on any of it.

### What the FP4 quantization is worth

`Indexer.forward` rounds both the index keys and the index query to E2M1 with one
power-of-two E8M0 scale per 32-value block (`fp4_act_quant`), in place, before
scoring. The port skips every activation quantization in this model, so this is a
real and documented deviation -- and now a measured one. `fp4_act_quant` is a tilelang
kernel that cannot run on this machine, but its arithmetic is not ambiguous:
`2 ** ceil(log2(amax / 6))` per block and then round-to-nearest-even into
`{0, .5, 1, 1.5, 2, 3, 4, 6}`. `_fp4_quantizer` in `Tools/dsv41_goldens.py` is that,
in torch, and the case runs a second arm through it.

| arm | dump A | dump B |
|---|---|---|
| identity quant (what the port implements) | **0** of 512 rows differ | **0** of 512 |
| the reference's FP4 quant | **3** of 512 (163, 436, 491 out for 289, 305, 464) | **5** of 512 (163, 379, 436, 470, 491 out for 236, 289, 339, 464, 487) |

So skipping the quantization costs 3 to 5 of 512 rows at the deepest position of one
source, and every row involved sits within 0.17 of the cut score: the quantization
moves scores enough to reshuffle rows that are already tied with the boundary, and
nothing else. Individual scores move much further (up to 1.9 relative) among rows that
nearly cancel across heads, which is why a relative delta is the wrong statistic here
and the row list is the right one. That range is the whole justification for the port
skipping the quantization rather than a guess; a future slice that wants the last
0.6-1% has a target and a test. The count moved between dumps because the activations
moved between runs -- the FP4 error lands on rows that are near the boundary, so which
of them cross depends on the boundary's exact position. **Updated in `#31` and `#32`:** the
activations moved because of the reduction race, not the boundary. With it fixed two dumps
are byte-identical and both move the SAME three rows, so the range above is 3 of 512 and
the rows are 163 / 436 / 491 in for 289 / 305 / 464 out.

### The defect this found

Two prefills of the same 576 tokens through the same executor, with `reset()` between
them, do not agree. All 576 layer-20 scores move, worst |delta| 1.84e-3 against a
3.80 maximum, and the selected rows with them. Two separate processes differ the same
way, so it is not state carried across a `reset()`.

The first position whose pooled latent differs is 86, and its latent differs by
**0.4886** against a 4.24 maximum -- a discrete jump, not accumulating rounding. From
87 on the differences are ~1e-2, which is what a single wrong residual write
downstream of an attention would leave behind. Block width is 8, so position 86 is
slot 6 of block 10, and slots 0-5 of that block are bit-identical.

**The same test is bit-identical under Metal's debug layer with shader validation**
(`MTL_DEBUG_LAYER=1`, `MTL_DEBUG_LAYER_ERROR_MODE=assert`,
`MTL_DEBUG_LAYER_VALIDATE_ENABLE=1`, `MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS=1`,
`MTL_SHADER_VALIDATION=1`): 0 of 576 moved. The value it settles on, 3.8018694, is a
third distinct answer, which is what a read of memory nothing wrote looks like when
the validation layer pins the contents. At 128 positions the same test is
bit-identical WITHOUT validation, in one process and across two, so whatever is read
unwritten is reached past the sliding window rather than in it.

Not localised yet, and the reason this is written down rather than fixed: every
ordering the code relies on was re-read and holds. `commit` waits with
`waitUntilCompleted` on every pending buffer, so `ensureResident`'s preads cannot land
under a live kernel in either the walk or the block path; every `encodeGemv` form and
`encodeNorm` ends with `memoryBarrier(scope: .buffers)`; the gather's host row list
is filled before its command buffer is committed; the arena's LRU is a counter, not a
hash. What is left is a region some kernel reads and no kernel writes, and the
validation layer is the tool that will point at it: run
`DeepSeekV41PublishedTurnTests/testTheIndexerSelectsPastItsBudgetOnThePublishedCheckpoint`
or the 576-position form of the determinism test under the flags above and watch for
the first stage that reports differently.

The reason this matters beyond bookkeeping: every real-weight numeric claim in `#28`
and `#29` is a single run, and a single run of this forward is one sample. The
structural results in `#29` (320/320 budget-exceeding selections differing from the
newest window, 0 zero-score rows, 576 distinct scores) reproduce across runs; the
exact score census does not, and the difference between runs is larger than the
difference between the port and the reference on a fixed set of activations. That is
why the parity work above is done on a dump: comparing two live runs would have
measured this defect instead of the indexer.

**Solved in `#31`.** The nondeterminism was a threadgroup reduction race in the Engram
gate, not a read of unwritten memory: see below. The FP4 count above is still a range
for its own reason -- those rows sit on the cut, so which of them cross depends on
where the boundary lands -- and the rule arm does not have that problem, which is the
shape the two results should have: an arithmetic comparison survives a moving input, a
which-rows-cross-the-cut claim does not.

## DSV4.1 #31: THE 576-POSITION PREFILL IS REPRODUCIBLE, AND THE RACE THAT MADE IT NOT

`#30` ended with an open defect: two prefills of the same 576 tokens through the same
executor did not agree. All 576 layer-20 scores moved (worst |delta| 1.84e-3 on the run
that found it, 0.4195 on one taken later), the selected rows with them; at 128 positions
the same test was bit-identical, and under Metal's validation layer it was bit-identical
at 576 but settled on a third distinct value. That last part is what made it read like a
peek at memory nothing wrote. It was not.

### Localising it

The instrument is `DeepSeekV41MetalForward.slotDigests` plus three diagnostic hooks:
`onBlockEmbedded`, `onBlockStageFinished` and `onBlockLayerFinished`. The digests hash the
workspace one POSITION SLOT at a time, so two runs compare equal only if every value in a
slot is bit-identical -- a whole-buffer hash cannot say which position moved, and a
position is what names the stage. The stage hook needs the `recordsStageTiming` layout,
the only one where a stage is complete before the next one is encoded.

Three runs, each one narrowing the answer:

| Observation | What it ruled out |
|---|---|
| At block ends everything was bit-identical except three per-slot scratch TAILS: `gathered` rows past a slot's row count, `indexScores` past the rows a position produced, `rowIndices` past theirs | A wrong value anywhere the graph reads: those tails are never read, at any row count |
| The first live divergence was always at an `.engram` stage (layers 1 and 14) | Every path except the Engram write |
| `engramRaw[T]` (the host pread), `engramValues[T]` (the dequant) and `engramKV` (the `wkv` projection) were identical at that stage, while `engramGates` and `streams` moved | The retrieved rows, their decode and their projection: the gate's own arithmetic was what moved |
| The onset moved between runs (block 0 slots 6-7, then block 15, then block 28) and always left slot 0 alone | Anything geometric: slot 0 is the dispatch that has the queue to itself |

The same probe also retracted an earlier reading of our own. `#30`'s first measurement
put the first divergent pooled latent at position 86, slot 6 of block 10; that was a
single run of a moving target, and with the position-level digests in place the onset
was somewhere else every time. A session that had kept chasing position 86 would have
chased a coincidence.

### The race

`threadgroup_reduce_sum` ended by having every thread read `sharedWarpSums[0]` after its
trailing barrier. That barrier orders this call's WRITE before every thread passes;
nothing then stops a fast thread from entering the next call and overwriting
`sharedWarpSums[warpIdx]` -- for warp 0, element 0 again -- while a slow thread is still
reading this call's result. A kernel that reduces once per dispatch cannot hit it, which
is why rmsnorm and the GEMVs were never affected. The Engram gate is the port's only
kernel that reduces three times in a row:

    streamSquares = threadgroup_reduce_sum(localStream, ...)
    keySquares    = threadgroup_reduce_sum(localKey,    ...)
    dot           = threadgroup_reduce_sum(localDot,    ...)

so `keySquares` could return a partial of `dot`, or `streamSquares` one of `keySquares`.
The gate then differs, the residual it writes differs, and every layer downstream differs
-- which is why the reported first divergence kept moving and why all 576 scores moved
together rather than one region's worth of them.

The fix is one barrier at the START of the helper, before the partials are written: it
orders "read the previous result" against "write the next reduction's partials" for every
caller, at the cost of a barrier a single-call kernel will not notice.

### What it is worth

| | before | after |
|---|---|---|
| `testTwoPrefillsOfOneStreamAgreeOnTheIndexScores`, 576 positions | 576/576 scores moved, worst \|delta\| 0.4195 | **0/576, worst 0.0** |
| `testEveryStageOfTwoPrefillsOfOneStreamAgrees`, 192 positions, every stage of every layer | first divergence at an `.engram` stage | **0 differences** |

The parity results in `#30` do not move: they were taken on a dump, where both sides
score the same activations. What changes is what a LIVE run means. `#29`'s score census
moving between runs (cut score -2.2880 -> -2.2704 -> -2.2442, layer-24 newest-row rank
464 -> 476 -> 483) was this race and not the checkpoint, so a single real-weight run is
now a sample of a stable distribution. The FP4 row count in `#30` (3 of 512 on one dump,
5 on another) was left here as a range because it still needed two dumps to be honest.
`#32` took them: two dumps after the fix are byte-identical and both count 3.

The validation layer's apparent determinism was a timing artefact, not a shield: making
the racing thread consistently early or late makes the SAME wrong partial get picked
every time, so the run looks correct and reproducible. That is the general lesson and the
reason this entry exists -- a wrong threadgroup partial looks exactly like garbage memory,
and a debug layer that changes timing will hide it rather than report it.

## DSV4.1 #32: TWO DUMPS ARE ONE FILE NOW, AND THE FP4 COUNT IS ONE NUMBER

**Question.** `#31` fixed the reduction race that made a long prefill unreproducible.
`#30` had measured the reference's FP4 activation quantization at "3 of 512 rows on one
dump, 5 on another" and left that range standing, because those rows sit on the cut and
the boundary was moving. Now that it should not move: is the spread gone (i.e. was the
range the race?), and are the real-weight numbers quotable as single values?

**Answer: two dumps written by two separate processes are byte-identical, the FP4 arm
moves exactly 3 of 512 rows on both, and `#29`'s census reproduces exactly.**

### The dumps

`testWhatTheIndexerWasGivenOnThePublishedCheckpoint` prefills the real checkpoint's 576
positions and writes layer 20's activations. Run twice, in two processes, with
`DSV41_INDEXER_INPUT_DUMP` pointed at two paths (through the xctestrun's
`EnvironmentVariables`, since xcodebuild does not forward the shell's):

| | value |
|---|---|
| size | 3,740,641 bytes, both |
| sha256 | `08095c9e61871f07cdaf79fd6a722cece4adc3b302a305e7b7cadab0561581c1`, both |
| `cmp` | identical |
| cost | 93.7 s and 98.0 s, one prefill each |

That is a stronger statement than `#31`'s: it is not the derived score vector that agrees
but every byte, across 576 pooled latents plus the normed input and rank-normed query at
the deepest position.

### The reference on the deterministic dump

`Tools/dsv41_goldens.py indexer_real` on either file gives one set of numbers:

| | value |
|---|---|
| layer / position | 20 / 575, of 576 prefilled |
| produced rows | 576 |
| worst absolute score delta | **2.621e-6** (row 153: reference 1.778044701, port 1.778042080) |
| rows over 1e-5 / over 1e-6 | **0** / 134 |
| port score range | -3.330896 .. 3.801967 |
| reference score range | -3.330895 .. 3.801969 |
| selected rows | **512 vs 512, identical** |
| cut score | -2.287968 (port) against -2.287969 (reference); the next row down is -2.290512 |
| top row | 41: port 3.801966670 against 3.801968575 |

### The FP4 count

| arm | rows moved |
|---|---|
| identity quant (what the port implements) | 0 of 512 |
| the reference's `fp4_act_quant` | **3 of 512**: selects 163 / 436 / 491, drops 289 / 305 / 464 |

The same three rows on both dumps, which is the answer: `#30`'s 3-then-5 was the race
changing the activations, not the boundary landing somewhere else.

The mechanism is worth stating precisely, because `#30` described the affected rows as
"sitting within 0.17 of the cut" and that understates what happens. The cut separates the
kept rows from the dropped ones by 0.0025. Quantizing moves those six rows by 0.074 to
0.206:

| row | unquantized | quantized | moved | side of the cut |
|---|---|---|---|---|
| 289 | -2.173703 | -2.379647 | -0.206 | kept, 0.114 above |
| 305 | -2.037593 | -2.171350 | -0.134 | kept, 0.250 above |
| 464 | -2.207995 | -2.281592 | -0.074 | kept, 0.080 above |
| 163 | -2.327516 | -2.213908 | +0.114 | dropped, 0.040 below |
| 436 | -2.305418 | -2.151766 | +0.154 | dropped, 0.017 below |
| 491 | -2.364699 | -2.177117 | +0.188 | dropped, 0.077 below |

Across all 576 rows the quantization error has a median of 0.066 and a p90 of 0.161, so
569 of 576 rows move by more than the 0.0025 the cut separates its neighbours by -- yet
only three rows cross, because the error is correlated across rows (it perturbs the query
and the keys, so scores shift together rather than independently). The deviation is
therefore not a tie-break artifact: the frontier sits deep inside the quantization error,
and only the rank correlation keeps the damage to three rows of 512.

### The re-scare

`#30` declined to check the reference's scores into the suite, because the activations were
not reproducible and a literal would have pinned one sample of a moving quantity. They are
reproducible now, so `DeepSeekV41RealIndexerReScareTests` carries the reference's 576
rule-arm scores and the 64 rows it drops, and re-checks the port against them in 25 ms for
all three cases. It skips when the dump is absent (producing it needs the checkpoint and
the opt-in marker), and its doc comment says what a failure can mean: the scoring rule
moved, or something upstream of the dump did. Two negative controls were run before it was
committed -- a dump with one score moved by 0.01 fails it and names the row, and no dump at
all skips it -- because an assertion that has never failed is not yet evidence.

### The census, as one value

`testWhatTheIndexerSelectsPastItsBudgetOnThePublishedCheckpoint`, run twice after the fix
(96.9 s and 97.9 s), printed the same lines both times:

| layer at position 575 | cut score | range | median | positive / negative | row 0 ranks | newest row ranks |
|---|---|---|---|---|---|---|
| 20 | -2.2880 | -3.331 .. 3.802 | -1.0109 | 135 / 441 | 43 | 62 |
| 24 | -2.6140 | -3.073 .. 3.127 | -2.2484 | 59 / 517 | 39 | 464 |
| 28 | -1.8919 | -2.360 .. 3.997 | -1.0060 | 130 / 446 | 47 | 94 |
| 32 | -1.8990 | -2.677 .. 3.980 | -0.7575 | 175 / 401 | 46 | 110 |
| 36 | -1.9725 | -3.059 .. 4.487 | -0.6498 | 190 / 386 | 62 | 66 |

576 distinct scores at every source, 0 zero-score rows of 576, and 320 of 320
budget-exceeding selections differ from the newest 512 rows. These are `#29`'s FIRST-run
numbers exactly -- cut -2.2880, layer-24 newest rank 464, layer-20 row-0 rank 43 -- so the
two runs that moved were corrupted by the race and the original census was the correct
one. `#29`'s table can be quoted as it stands; the `+/-` range `#30` introduced is what is
retired.

### What it leaves

Nothing numeric is open any more: the rule arm matches the reference to 2.6e-6, the cut is
identical, the deviation the port accepts by skipping the activation quantization is three
rows of 512 and pinned by name, and both a live prefill and the numbers derived from it
are reproducible. `batchedDevicePrefill` remains the only unverified gate and it needs real
weights, not another dump.

## DSV4.1 #33: THE BATCHED PREFILL RUNS ON THE PUBLISHED CHECKPOINT, AND THE LAST GATE CLOSES

The only conformance rule still open was `batchedDevicePrefill`, and its stated reason was
not a rule: the block graph had been compared with the host on a synthetic checkpoint, but
never on the real one, where the tensor shapes, the 128-row ring, the ratio-2 compressor and
the expert arena are the checkpoint's own. `#26` had already closed the performance question
(a narrow verify loses to walking at widths 2, 3 and 5), so this slice's target was the gate,
not a faster decode.

### What changed

- `prefillChunk` gained `capturingLogitsAt:`, relative to the chunk like the CPU forward's
  argument. It maps each named position into the block that owns it and returns the usual
  `last` beside a `captured` dictionary. A chunk that continues the cache is the same path,
  so a verify can ask for a drafted run in one pass.
- `finishBlock(rows:)` now normalizes and projects a whole block in one grid instead of one
  slot: the final norm loops its rows and the `vocabSize` head runs with a row stride. The
  head is the one stage whose cost grows with the capture, and a per-position dispatch would
  read its 1.3 GB matrix once per position; one row grid reads it once for the block.
- `workspace.logits` is one vocabulary vector per position slot, so a block's logits sit
  side by side and a decode step still reads slot 0.

### The real-weight comparison

`DeepSeekV41PublishedTurnTests.testABatchedPrefillMatchesTheWalkOnThePublishedCheckpoint`
runs 16 positions as two consecutive blocks of eight, asks for every slot, and compares each
position with a per-position walk. The second block continues a cache the first filled, so
the ring slot and the compressor's group slot are derived from the absolute position.

```
batched prefill vs walk: 16 positions in two blocks of 8,
  worst relative delta 1.701e-04 at position 12 (scale 20.184);
  plain blocks (no capture) worst 1.147e-04 at position 12 (scale 20.184)
```

Under the full Metal validation layer the same case printed 1.500e-04 and 1.339e-04, so the
delta is not a fixed offset: it is what a re-associated reduction costs at this dynamic
range.

The residual is float re-association, and the control says so: the capture-free block path,
which keeps the walk's one-row head, is already at 1.15e-4, and the batched head moves it by
another 0.55e-4 on the normal run (less when the validation layer changes the timing). Both
are far below the 5e-4 bar, which is itself four times below the
graph's 2e-3 decode tolerance; a wrong row list or a head that wrote every slot to slot 0
would be orders of magnitude larger. The synthetic cases still carry the bit-exact contract
(`1e-6` at the chunk seam) because their reductions are small enough for the orders to
coincide.

### What it means

`batchedDevicePrefill` flips to `verifiedAgainstReference`; `unverifiedSummary` is empty for
the first time. The port's conformance ledger now has no open rule: every gate is either a
reference vector from `Tools/dsv41_goldens.py` or, for the batched graph, a real-checkpoint
comparison against the path the turn loop already uses.

What this does NOT mean: DSpark is not wired into the turn loop. `#26`'s economics still
hold, and the two pieces a verify still needs are cache rewind for a rejected suffix and the
accept/emit loop itself. The per-position logits are now the easy half; rewind (ring
position, `tokenHistory`, `deadTokens` and the compressor group phase together) remains
deliberately unbuilt because it buys a feature that loses to walking at every width measured.
`DeepSeekV41Availability.isRuntimeAvailable` was also still false here, and that was a
separate question about what the flag promises, not about the graph. **Updated in `#34`:**
it is now true, with the promise scoped to text turns (see below).

## DSV4.1 #34: THE FLAG NOW SAYS TEXT TURNS, AND THE EFFORT CONTROL IT SWITCHES ON REACHES THE PROMPT

### The question

`DeepSeekV41Availability.isRuntimeAvailable` had been `false` since the family was first
detected. Its comment said the flag existed so "a user is not invited to download a ~510 GB
checkpoint that can only fail closed" — a premise `#33` retired: every conformance gate is
verified, `DeepSeekV41TurnRunner` runs the published checkpoint, and a real prompt answers.
So the question was not "can it run" but "what does advertising it promise?".

### What changed

- `isRuntimeAvailable` is `true`, and its doc states the promise as TEXT turns and names what
  is outside it: the vision tower has no device path, DSpark is unwired, cache rewind is
  unbuilt, and the composer still refuses images because no local model advertises vision.
  It stays a constant rather than being derived from `unverifiedSummary`: a gate records how
  far to trust a number, and one open rule for a stage the text path never runs must not
  disable a working text turn. It is also not a download gate; discovery lists the hub cache
  either way.
- The model row's `, runtime not available` caveat is now gated off by the flag (the string
  stays for a future regression).
- `ModelConfig.supportsThinkingLevels` detects DeepSeek V4.1 from the checkpoint identity
  while the runtime is advertised, because its formatter renders the family's own
  `Reasoning Effort:` line. That switches on the High/Low/Off menu Spark already had; every
  prompt path (chat, agent continuation, both API adapters) was already threading the level
  into the dialect.
- The server's turn-memory digest now covers the rendered system block for V4.1 too.
  `ChatPromptFormatter.renderedSystemBody` is the single authority for "the system text the
  model saw" (Spark's body is unchanged by the extraction), so switching High to Low
  invalidates a recorded raw turn instead of trusting one generated under a different system
  prompt.
- The DSV4.1 load log gains a `runtime: text turns (...)` line, so the promise is visible at
  the same place the checkpoint's plan is printed.

### Evidence at real weights

`testAPublishedPromptCompletesATurn` now runs both shapes on one runtime: the thinking-off
prompt it always ran, and the app's default (thinking on, effort high), whose 31-token prompt
carries the effort instruction and opens the reasoning channel.

```
turn 1 (thinking off): lengthCapped after 8 token(s), prefill 1.85 s, decode 2.13 s
  text: "Hey! How's it going? What"
thinking turn (effort 100): lengthCapped after 7 token(s), prefill 9.27 s, decode 2.09 s
  text: "The user just said \"hey\"."
```

The second turn is a fresh prefill because the effort line changes the prompt's first
tokens; the generated text is the reasoning channel's own opening, which is what the
checkpoint's encoder asks a thinking turn to produce. Its 9.27 s against the first prompt's
1.85 s is not the flag's cost: the prompt is 31 tokens instead of 5 and starts over at
position 0.

Pure logic pins the wiring: `testTheRuntimeAdvertisesTextTurnsAndTheirEffortLevels` (flag
true, the published config advertises levels, a plain `llama` config still does not),
`ModelConfigThinkingLevelTests.testDeepSeekV41IsDetectedWithoutATemplateKnob`, and
`ChatPromptFormatterConversationTests.testRenderedSystemBodyCarriesTheGradedEffortLevel`
(High and Low differ, thinking off emits nothing, the block is a substring of the actually
rendered prompt, and Spark's block is byte-identical to `sparkSystemBody`).

### What it does not mean

Vision is still CPU-reference only: no device tower, no image span reaches a turn, and the
composer refuses attachments for every local model. **Updated in `#35`:** an image span now
DOES reach a real turn (`DeepSeekV41TurnRunner.run(imageEmbeddings:)`), and the host tower
measures ~10.4 hours for the smallest image the checkpoint accepts, so the composer still
refuses attachments — for a measured reason now, not an assumed one. DSpark is still unwired,
still a measured loss at every width (`#26`), and cache rewind is still unbuilt. The flag
advertises the text path that exists, not every head the checkpoint publishes.

## DSV4.1 #35: AN IMAGE SPAN NOW REACHES A REAL TURN, AND THE HOST TOWER MEASURES AT TEN HOURS A PICTURE

### The question

`#34` advertised text turns and named vision as outside the promise. The remaining
candidates were a second indexer source, the image path, the device vision tower, and the
shelved drafter. `image_prefill` was chosen: the tower's CPU reference was already matched
against the checkpoint's own `ViT`/`Aligner`, `imageSpan(rgb:width:height:)` was verified,
and the image token was pinned to the tokenizer's own added token 129,264, so what was
missing looked like plumbing. The open question was whether plumbing is all it is.

### What changed

- **The device executor takes a supplied residual row.** `DeepSeekV41MetalForward
  .forwardEmbedding(_:token:position:)` uploads the row into the workspace's embedding slot,
  seeds every hyper-connection copy from it, records the position DEAD for the Engram hashes,
  and then walks exactly what `forwardToken` walks. This is the seam every future vision path
  enters through, device tower or not.
- **The turn loop carries rows.** `DeepSeekV41TurnRunner.run(imageEmbeddings:)` walks a
  prompt that has any, and `DeepSeekV41PositionForward` grew the matching requirement with a
  default that REFUSES: an executor that cannot take a row says so, because silently running
  the token embedding would answer about nothing.
- **An image prompt never resumes a prefix.** The app-wide pin is keyed on token ids, and
  every position of an image carries the same id whatever the pixels were. A resumed span
  would keep a previous picture's rows in the cache. This is a deliberate cost (a follow-up
  turn re-prefills) until the span is hashed into the pin.
- **`DeepSeekV41ImagePrompt`** expands one placeholder id into its span and binds a row to
  each position, reporting an image with no placeholder and a placeholder with no image
  instead of dropping either, because dropping one shifts every position after it.
- **`DeepSeekV41VisionWeights`** reads the tower's 266 tensors (263 block tensors plus the
  three markers) by name out of the shard headers, decodes BF16/FP32 on demand, and never
  pins: one encode touches each tensor once.
- **The CPU forward's two entry points now share one walk.** `forwardToken` used to delegate
  to `forwardEmbedding`, so making a supplied row dead-by-construction made EVERY token dead
  and 70 tests failed on Engram hashes. They now both call a private `runPosition(dead:)`.

### Evidence at real weights

The first time the tower has run on the published weights at all (every previous vision test
is a synthetic fixture against a numpy mirror). `testAnImageEntersAPublishedTurn`, behind
`/tmp/dynamoe_dsv41_real_turn`:

```
🔎 [dsv41 real] image prompt: 12 tokens, image id 129264
🔎 [dsv41 real] vision weights: 266 tensors
🔎 [dsv41 real] tower: 4 patch tokens in 127.11 s, 16 in 422.55 s; span 4 rows for the 2x2
   grid, prompt 12 -> 15 tokens
🔎 [dsv41 real] image turn: lengthCapped after 2 token(s), prompt 15, reused 0,
   prefill 4.07 s
🔎 [dsv41 real] image turn text: "This picture"
```

The placeholder tokenized to ONE token (id 129,264) out of 12, the span expanded the prompt
from 12 to 15 tokens, the turn walked all 15 with the supplied row at its own position, and
`reused 0` is the no-resume rule holding even though a 11-token prefix was proposed.

### What the tower costs, and why this is the headline

Two patch grids, same weights, same process:

| patch tokens | encode |
|---|---|
| 4 (2x2) | 127.11 s |
| 16 (4x4) | 422.55 s |

That is **24.6 s per marginal patch token** and a fixed **~29 s** (the 970 MB of BF16 read
and widened into 1.9 GB of `Float`, once per encode). The checkpoint's own floor is
`min_pixels` 295,936, i.e. at least 544x544, which the patch grid rounds up to 546x546 =
**1,521 patch tokens** for the smallest image it will accept: 28.6 + 1521 * 24.6 = **~10.4 hours** of host tower before the span's
184 positions even reach the backbone (which then costs ~50 s at the measured ~270 ms a
position). The shape is the problem, not the CPU: a projection re-reads its whole matrix per
token through bounds-checked scalar dots, which is a GEMM being done as `rows` dot products.

So `image_prefill` is DONE and vision is NOT shippable on it. Two ways forward, in order of
cost: give the host tower a real GEMM (Accelerate is already linked; the same arithmetic at
~100 GFLOP/s turns a picture into tens of seconds), or build the device tower (`vision_device`
as originally scoped). The composer must keep refusing attachments until one of them lands;
the `supportsVision` flag stays false.

Pure logic pins the wiring: `DeepSeekV41ImagePromptTests` (span length is the grid plus its
delimiters, one placeholder expands to its span and binds every row, two images keep their
order, a placeholder with no image survives, an image with no placeholder is reported, a
wrong span length and a wrong row width are refused, a text-only prompt is untouched), the
five new `DeepSeekV41TurnLoopTests` cases (row runs only at its own position, an image turn
does not resume even when the prefix matches, a text-only turn still resumes and chunks, an
out-of-range row is refused, an executor without image support refuses the row), and
`DeepSeekV41DeviceGraphParityTests.testASuppliedRowComputesTheSameLogitsAsTheHost` (the
supplied row AND the text position after it, `2e-3` normalized).

### What it does not mean

The image path exists end to end and is unusably slow. No `supportsVision` flag was turned on,
no composer path was opened, no chat or API request can carry an image yet: the engine seam is
proven, the serving cost is not. `vision_device` is now justified by a number rather than a
premise, and the cheaper alternative (a host GEMM) is named above so it can be chosen first.
**Updated in `#36`:** the host GEMM was chosen and shipped. The same harness measures the
smallest image at 68.06 s (from ~10.4 h), the slope falls from 24.6 s to 20.8 ms per marginal
patch token, and an isolated `-O` mirror prices the tower's arithmetic at 2.00 s, so what is
left of the harness number is the `-Onone` widen. `vision_device` survives as an optimisation,
not a prerequisite.

## DSV4.1 #36: ONE GEMM PER BLOCK TURNS TEN HOURS INTO SECONDS, AND WHAT IS LEFT IS A `-Onone` ARTIFACT

### The question

`#35` priced the host vision tower at **~10.4 hours for the smallest image the checkpoint
accepts** and named the shape rather than the CPU: `DeepSeekV41VisionForward.linear` re-read
its whole matrix once per token and reduced it through bounds-checked scalar dots, which is a
GEMM being done as `rows` dot products (24.6 s per marginal patch token). Two ways forward
were named: give the host tower a real GEMM, or build the device tower. Routing put
`vision_gemm` at 0.99 (confidence 0.99) and "GEMM before device" at 0.999, on the ground that
the CPU tower already has an oracle (the checkpoint's own `ViT`, and the pinned aligner rows
from `#34`) while the device tower has no device code at all.

### What changed

- **`linearBatched`**: one `cblas_sgemm` for the whole block of tokens,
  `CblasRowMajor/NoTrans/Trans` because `DeepSeekV41Matrix` is row-major `[out][in]`, with the
  bias folded into the copy-out. `linearRowByRow` keeps the definition beside it as the oracle.
- **`attentionBatched`**: two GEMMs per head (`alpha` carries `1/sqrt(head_dim)` on the score
  GEMM), and the 2D rotation sliced once per token instead of once per head per token.
- **`softmaxRow`**: `vDSP_maxv` / `vDSP_vsadd` / `vvexpf` / `vDSP_sve` / `vDSP_vsmul` over one
  flat row. `softmax` keeps its scalar form for the oracle.
- **`rmsNorm`** through `vDSP_svesq` + `vDSP_vsmul` + `vDSP_vmul`: it was three bounds-checked
  scalar passes over every patch, twice a layer.
- **`DeepSeekV41VisionWeights.decode`** widens through a typed destination instead of two
  bounds-checked byte loads per value.
- `encode` still validates every row and now REFUSES a ragged one (`shapeMismatch`) instead of
  zeroing it, because a GEMM cannot take a short row.

The numeric contract is unchanged: `attention` and `softmax` are untouched, and the
numpy-pinned aligner rows from `#34` still match at 1e-4. `cblas_sgemm` is deprecated in macOS
13.3 in favour of the ILP64 interface (`-DACCELERATE_NEW_LAPACK`); the build warns twice, and
the call is the same one every Accelerate consumer makes.

### Evidence at real weights

`testAnImageEntersAPublishedTurn`, published checkpoint, same harness and machine as `#35`:

```
🔎 [dsv41 real] image prompt: 12 tokens, image id 129264
🔎 [dsv41 real] indexed 96085 tensors in 48 shard(s), 475.2508442029357 GB
🔎 [dsv41 real] mapped 48/48 segment(s), 286.4 GB
🔎 [dsv41 real] vision weights: 266 tensors
🔎 [dsv41 real] tower: 4 patch tokens in 25.35 s, 16 in 25.60 s, 1521 in 68.06 s; span 4 rows for the 2x2 grid, prompt 12 -> 15 tokens
🔎 [dsv41 real] tower floor: 20.3 ms per marginal patch token, smallest image 68.06 s (184 span positions)
🔎 [dsv41 real] image turn: lengthCapped after 2 token(s), prompt 15, reused 0, prefill 2.53 s
🔎 [dsv41 real] image turn text: This picture
```

| patch tokens | `#35` | `#36` |
|---|---|---|
| 4 (2x2) | 127.11 s | 25.35 s |
| 16 (4x4) | 422.55 s | 25.60 s |
| 1,521 (39x39, the smallest admissible image) | ~10.4 h, extrapolated | **68.06 s** |

The slope between the two small grids falls from **24.6 s to 20.8 ms per marginal patch
token** (1,180x). The 1,521-token point implies 28.2 ms/token rather than 20.8, which is the
attention: its cost per token grows with the token count, so a slope measured at 16 tokens
understates the picture. The whole smallest image goes from ~10.4 hours to **68 seconds** in
the same harness, and the turn's prefill beside it from 4.07 s to 2.53 s.

### What the remaining 68 seconds are made of

Two constants, both measured in isolation rather than inferred from the slope, because the
test bundle is a `-Onone` build: BLAS and vDSP are precompiled and their numbers are
Build-settings-independent, while every Swift per-element loop pays the Debug penalty.

**The fixed cost is the BF16 widen.** The same widening loop over 960 MB (480 million values),
built both ways:

| form | `-Onone` (the test bundle) | `-O` (the shipped build) |
|---|---|---|
| old: two bounds-checked byte loads per value | 27.35 s (35 MB/s) | 0.05 s (17 GB/s) |
| new: typed destination + `loadUnaligned` | 23.93 s (40 MB/s) | **0.03 s (34 GB/s)** |

So the ~25 s that does not move with patch count IS the widen, and at `-O` it is nothing.

**The arithmetic is 2 seconds.** A standalone `-O` mirror of the tower at the shipped shapes
(1,521 tokens, hidden 1,024, 16 heads of 64, intermediate 2,816, 32 layers):

| stage | time |
|---|---|
| linear GEMMs (qkv, wo, w1, w2 per layer) | 0.86 s |
| attention GEMMs (scores + context per head) | 0.49 s |
| softmax (vDSP) | 0.53 s |
| norms (vDSP) | 0.01 s |
| glue (SiLU, copy-out, residual adds) | 0.11 s |
| **total** | **2.00 s** |

One `w1` GEMM at this shape runs at **1,608 GFLOP/s**, which is what a real GEMM buys: the
same product that `#35` measured at ~17 MMAC/s.

The rest of the harness's 68 s is `-Onone` per-element Swift: the q/k/v fill and copy-out
(~200 million values per encode), the gate/up activation (~137 million), and the residual adds
(~100 million) all run at the 40 MB/s the widen table shows, and the mirror measures the same
work at 0.11 s when it is compiled.

### What it means

The scaling law is fixed, so the next slice is **`composer_image_attachment`**, not
`vision_device`. At `-O` the smallest image is ~2 s of arithmetic plus one 970 MB read of the
tower's shards per encode; a turn that already spends ~270 ms per backbone position can afford
that. `vision_device` is now an optimisation with a named prize rather than a prerequisite:
it would replace the per-encode read with a bind and the 2 s of CPU arithmetic with GPU
dispatches, and it is the only way to serve the LARGEST admissible image, because the cap is
1,024 aligner tokens, about 9,216 patch tokens, six times the floor, and attention is
quadratic (0.49 s becomes ~18 s).

Caching the widened weights is not justified either: 1.9 GB of pinned `Float` would buy 0.03 s.

### What it does not mean

Nothing was opened in the app: `supportsVision` is still false, the composer still refuses
attachments, and no chat or API request can carry an image. The 2.00 s is a standalone `-O`
mirror of the same shapes and the same calls, not a Release build of the app, whose own glue
was not timed; the 68.06 s is the number the harness actually measured.

**Updated in `#37`:** both gaps are closed. The app's own build, compiled `-O`, measures the
smallest accepted image at **2.74 s** (24.8x the 68.06 s above), the largest image the
checkpoint accepts at **41.71 s**, and 76% of that 41.71 s is the quadratic attention term. The
2.00 s mirror was 21% optimistic against the 2.52 s of measured arithmetic; see `#37`.

Pure-logic pins: `DeepSeekV41VisionForwardTests.testTheBatchedLinearMatchesItsDefinition` (an
odd 3x7x5 shape so a transposed matrix or a wrong `lda` cannot cancel out),
`testTheBatchedLinearHandlesASingleToken`, `testTheBatchedAttentionMatchesItsDefinition`,
`testTheBatchedAttentionWithoutRotationMatchesItsDefinition` (a missing table must not become a
rotation by zero), `testARaggedPatchRowIsRefused`, and the new
`DeepSeekV41VisionWeightDecodeTests` (BF16 is the high half of its float, the discarded low
half is not reinvented, FP32 passes through bit for bit, a short buffer yields zeros, the dtype
widths). The suite is **556 tests, 0 failures, 41 skipped**, and 0 Metal validation errors
under `MTL_DEBUG_LAYER`.

## DSV4.1 #37: THE REAL TOWER AT `-O` TAKES 2.74 s, AND THE LARGEST IMAGE THE CHECKPOINT ACCEPTS IS 76% ATTENTION

### The question

`#36` left one claim in this journal that was a mirror rather than a measurement: it priced the
tower's arithmetic at **2.00 s** by compiling a standalone `-O` harness of the same shapes and
calls, while the 68.06 s everyone would actually see came from a `-Onone` test bundle. Two
things were therefore unknown, and a third was named but unmeasured: what the app's OWN code
costs per image at `-O` (its glue inside `encode` was never timed there), and what the LARGEST
image the checkpoint accepts costs, which `#36` could only extrapolate from a quadratic
attention term (0.49 s at 1,521 patch tokens becomes ~18 s at 9,216). Routing put
`vision_release_measure` at **0.84** (confidence 0.80), "measure at `-O` before opening the
composer" at **0.98**, and "the app's untimed glue is a real risk" at 0.81.

### What changed

Test-only. Production code is untouched, so every number below is the code that shipped in
`#36` — just compiled the way a release build compiles it.

- **The `-O` run is the app's own build**, not a mirror: `build-for-testing` with
  `SWIFT_OPTIMIZATION_LEVEL=-O` into a SEPARATE derived data directory
  (`-derivedDataPath /tmp/dd_dsv41_opt`), so the Debug products other runs depend on are left
  alone. It produced a single 127 MB `DynaMoE` binary instead of the Debug configuration's
  59 KB launcher plus 154 MB `DynaMoE.debug.dylib`, and a 19.0 MB `DynaMoETests` bundle against
  Debug's 23.0 MB. That the flag took effect is not inferred from the size: the same test that
  took 126.57 s takes 63.24 s.
- **`testAnImageEntersAPublishedTurn` grew an opt-in tail** gated on
  `/tmp/dynamoe_dsv41_vision_max`, because the quadratic term makes the big grids a
  many-minute measurement at `-Onone`. It measures the largest image the checkpoint accepts
  **through `fit`** (a 20,000 pixel square collapses onto the token budget) rather than
  hard-coding a grid, so the measurement cannot drift from what an oversized real image would
  become; it re-times the 4-token encode to get the fixed cost with the page cache warm; it
  times a middle grid; and it solves `seconds = fixed + perToken * tokens + squared * tokens^2`
  through the three points by Cramer's rule (`costModel(tokens:seconds:)`), returning nil rather
  than inventing a split when the points cannot separate the growth terms.
- The three small grids (`2x2`, `4x4`, `39x39`) still run on every invocation, so the `#36`
  series stays comparable line for line.

### Evidence at real weights

Published checkpoint, `-O` build, `/tmp/slice37_o2.log`:

```
🔎 [dsv41 real] image prompt: 12 tokens, image id 129264
🔎 [dsv41 real] vision weights: 266 tensors
🔎 [dsv41 real] tower: 4 patch tokens in 0.24 s, 16 in 0.24 s, 1521 in 2.74 s; span 4 rows for the 2x2 grid, prompt 12 -> 15 tokens
🔎 [dsv41 real] tower warm: a repeated 4-token call in 0.22 s (the first call pays the cold read, so this is the fixed cost)
🔎 [dsv41 real] tower cap: 8649 patch tokens (93x93) -> 31x31 aligner grid, 994 span positions in 41.71 s (15.2x the smallest image, 4.82 ms per patch token)
🔎 [dsv41 real] tower cost model: 0.22 s fixed + 1.126 ms per token + 4.245e-07 s per token^2; at 8649 tokens the quadratic term is 76% of the 41.71 s (mid grid 63x63 = 3969 tokens in 11.38 s)
🔎 [dsv41 real] image turn: lengthCapped after 2 token(s), prompt 15, reused 0, prefill 2.15 s
🔎 [dsv41 real] image turn text: This picture
Test Case '...testAnImageEntersAPublishedTurn' passed (63.240 seconds)
```

| patch tokens | aligner grid | span positions | `#36` `-Onone` | `#37` `-O` |
|---|---|---|---|---|
| 4 (2×2) | 1×1 | 4 | 25.35 s | 0.24 s |
| 16 (4×4) | 2×2 | 8 | 25.60 s | 0.24 s |
| 1,521 (39×39) | 13×13 | 184 | 68.06 s | **2.74 s** |
| 3,969 (63×63) | 21×21 | 464 | not measured | 11.38 s |
| 8,649 (93×93) | 31×31 | 994 | not measured | **41.71 s** |

The smallest image the checkpoint accepts is **24.8x** faster than the number `#36` published,
and the whole test — three small towers, the warm repeat, the middle grid, the largest grid and
a real image turn with 2 decoded tokens — goes 126.57 s → **63.24 s**.

A second clean run reproduces every point: 2.75 s smallest, 0.22 s warm, 11.43 s middle,
41.51 s largest, 2.14 s prefill, 62.71 s wall clock, and a model of
`0.22 s + 1.170 ms/token + 4.167e-07 s/token^2` with the quadratic term at 75%. The spread is
1% on the cache-warm points. A third run, taken while the same machine was busy with another
process, read 3.82 s and 43.71 s — **+39% on the smallest image and +5% on the largest** — so
the numbers quoted above are the quiet pair, and `-O` does not make the measurement immune to
contention, only much faster than it. The image turn's own
prefill (2.15 s against Debug's 2.53 s) is the backbone's Metal path and barely moves, which is
the expected shape: `-O` cannot improve GPU dispatches.

### The cost model, and three checks that it is not curve fitting

`0.22 s fixed + 1.126 ms per token + 4.245e-07 s per token^2` is solved exactly through the
4-token, 3,969-token and 8,649-token points, so it reproduces those three by construction. The
checks are the parts it does NOT fit:

1. **The fixed term is independently measured.** It comes out at 0.22 s, and the repeated
   4-token encode — a separate call, timed for exactly this reason — is 0.22 s. It is the
   tower's 970 MB of weights being `pread` once per encode (4.4 GB/s), not the widen, which
   `#36` measured at 0.03 s when compiled.
2. **An unused point agrees.** 1,521 tokens is not one of the three fit points; the model
   predicts 0.22 + 1.71 + 0.98 = 2.91 s against 2.74 s measured, 6% high.
3. **The quadratic term matches `#36`'s mirror.** The model attributes 0.98 s of the 1,521-token
   encode to the `t^2` term, and the standalone mirror measured attention GEMMs at 0.49 s plus
   softmax at 0.53 s, i.e. 1.02 s — 4% apart, from two completely different code paths.

So the tower is three things at real weights: a 0.22 s weight read, 1.126 ms per patch token
(35 µs/token/layer across 32 layers) of linear GEMM, norm and glue work, and attention plus
softmax growing with the square of the token count.

### What it means

**The composer is now a measured decision, not an estimate.** The smallest accepted image is
2.74 s of tower plus 2.15 s of prefill — about 5 s to the first token — and the largest
admissible one is 41.71 s plus its prefill, about 45 s. That is a 15x range across legal
images, so the composer should price an attachment from `DeepSeekV41Runtime.imageSpanPlan`
before encoding it and say what it will cost, rather than discovering it afterwards.

**`vision_device` now has a number instead of an adjective.** Its prize is the `t^2` term:
31.75 s of a 41.71 s cap image, 0.98 s of a 2.74 s floor image. It does not touch the 0.22 s
read and does not touch the 9.7 s linear term (unless the weights are also bound instead of
`pread`). So the device tower is worth ~76% of a large image and ~36% of the smallest — a real
prize for a big picture, a small one for a thumbnail. That is the honest ranking, and it keeps
`vision_device` an optimisation rather than a prerequisite.

**An image is encoded per ATTACHMENT, not per turn, and nothing caches that yet.** Every
`imageSpan` call re-reads the 970 MB and re-runs the tower, and an image prompt never resumes a
prefix, so a conversation that keeps a picture in context re-prefills every turn. The composer
slice should cache the span by `ImageAttachmentRef`; nothing in the current code does.

**Pinning the widened weights is still not justified**: 1.9 GB of `Float` buys 0.03 s, and the
0.22 s fixed term is the read, which pinning would remove. That trade is 1.9 GB of resident
memory for 0.22 s per image; worth revisiting only if the tower ever runs per turn.

### What it does not mean

- **`-O` here is the Debug configuration compiled with `SWIFT_OPTIMIZATION_LEVEL=-O`**, which is
  the same sources and the same optimizer as a Release build of these paths. No Release build of
  the app was archived or notarized, and the shipped Release configuration was not run.
- **Nothing was opened.** `supportsVision` is still false, the composer still refuses
  attachments, and no chat or API request can carry an image.
- **8,649 patch tokens is the largest a SQUARE image becomes.** A non-square image gets a
  smaller `columns * (rows + 1) + 2`, so it is the worst case, not the only case.
- **What a developer runs from Xcode still costs 68 s**, because Debug is `-Onone`. Both numbers
  are real; they belong to different builds.
- The `-Onone` gate on the marker is load-bearing: at `-Onone` the largest grid would be tens of
  minutes of interpreter overhead and would say nothing about the tower.

### Verification

The full DeepSeek suite is **556 tests, 0 failures, 41 skipped** on the Debug build, and 0 Metal
validation errors under `MTL_DEBUG_LAYER` (`dsv41_validation43.xctestrun`). The vision suites
(`DeepSeekV41VisionForwardTests`, `DeepSeekV41VisionWeightDecodeTests`,
`DeepSeekV41VisionGoldenTests`, 25 tests) also pass in the `-O` build, which is the non-obvious
one: `-O` must not move the GEMM parity bars, and it does not. Production code was not touched
in this slice, so those bars are `#36`'s.


## DSV4.1 #38: THE COMPOSER CAN SEND A PICTURE, AND THE PATH IS FOUR SEAMS

### The question

`#37` measured what an image costs and left the composer as the next slice, on the ground that
the cost objection was retired and the UI half of the seam already existed. Routing agreed
(`composer_image_attachment` 0.993, confidence 0.99; "probe the app's own tokenizer path first"
0.998; "price the attachment in the composer" 0.996). Four gaps stood between a staged
`PreparedImage` and a turn the model can see: the request had no image field, the engine never
called the tower, the app's prompt text carried no placeholder, and no local model reported
`supportsVision`.

### What changed

- **The probe came first, and it was worth it.**
  `testTheComposedPromptCarriesOneImageTokenPerAttachment` renders a user turn through
  `ChatPromptFormatter.formatPrompt` (the app's path, not `dialect.render`), tokenizes it with
  the checkpoint's own tokenizer, and expands it: **10 tokens plain, 12 with one image, 14 with
  two**; the marker alone tokenizes to exactly `[129264]`; `expand` binds 8 rows with 0
  unfilled and 0 unused. The first two fixtures were wrong in the same instructive way —
  a newline is its own token, and *two adjacent newlines are one* — so the control is now one
  ordinary token placed where the marker goes, which is the only comparison that isolates the
  marker.
- **`DeepSeekV41ImagePixels`** turns encoded bytes into the row-major RGB the tower samples:
  ImageIO decode with EXIF orientation applied and a 4,096-pixel side cap (the token budget is
  a 1,344-pixel box, so the cap refuses a 100-megapixel photo that would cost 300 MB of RGB to
  end up as 31x31 cells). A 24-bit sRGB `CGContext` is refused by Core Graphics on this OS
  (measured), so the draw goes through 32-bit RGBA and is compacted.
- **`DeepSeekV41TurnImages`** splits the two halves the engine needs in that order: `price`
  answers "how many prompt positions will this become" from `DeepSeekV41VisionLayout.fit` alone,
  and `bind` runs the tower and places its rows. Both fail closed on no tower, an image with no
  placeholder, a placeholder with no image, a malformed buffer, and a span whose row count
  disagrees with its grid.
- **The engine prices before provisioning and binds after.** The KV cache has to be sized for the
  *expanded* prompt, which is why the price cannot wait for the tower; the bind then runs with
  the runtime in hand and its length is asserted against the price, because a disagreement means
  a cache sized for one sequence and a turn run on another. The expanded ids feed the prefix pin,
  the prepared-length bookkeeping and the sampler's context. Any family without a tower now
  refuses a turn that carries images instead of dropping them.
- **`ModelConfig.supportsVision(modelPath:)`** is the composer's gate, backed by
  `DeepSeekV41VisionWeights.directoryHasTower`, which reads the shard index for a block, the
  aligner and the three marker embeddings — the markers live in a different shard from the
  blocks, so a partial download fails this check rather than advertising a tower it cannot run.
- **The composer prices the attachment.** `DeepSeekV41VisionPricing` turns the resize plan into
  span positions and seconds on the model `#37` measured, and the chip reads
  `546×546 · 12 KB · about 2.9 s`.
- **One encode per attachment, not per turn.** A chat turn re-renders the whole conversation, so
  a picture in context would re-run the tower every turn; the engine now caches encoded spans by
  `snapshot path | sha256`, bounded to four, and drops them with the checkpoint.

### Evidence at real weights

`testAnRGBImagePricesAndBindsOnThePublishedWeights`, published checkpoint, `-Onone` harness:

```
🔎 [dsv41 real] rgb bind: 56x56 pixels -> 1521 patch tokens, 184 span positions, priced 2.91 s against 68.52 s measured
```

2.91 s is the estimate `#37`'s model gives for 1,521 patch tokens; 68.52 s is what the same grid
costs in the debug harness, which `#37` measured at **2.74 s** when the app's own code is
compiled `-O`. So the estimate is a release number, and the harness number is a debug one — said
out loud because they differ by 25x.

### What it means

The composer's path is now measurable end to end: stage a picture, see what it will cost, send,
and the tower runs once, the rows land at the placeholder's positions, and a follow-up turn
reuses the cached span. **Vision is no longer blocked on anything numeric.**

What it is still blocked on is a prompt renderer. For a DeepSeek V4.1 chat turn the app builds a
chatml transcript and `formatPrompt` then wraps that transcript inside ONE V4.1 user turn, so an
attachment's markers land inside a nested blob rather than in the checkpoint's own turn
structure. The rows bind and the model sees the picture, but the surrounding text is not what the
checkpoint was trained on. That is a pre-existing rendering gap (`#34` exercised the API path and
the formatter, not `ContentView`'s builder), and it is the next slice: the family should render
through `ChatPromptFormatter.formatConversation`, as the server path already does.

### What it does not mean

- The API server does not carry images. A stateless request re-renders its conversation, so it
  would need the same span cache to avoid re-encoding per request; out of scope here.
- Nothing was measured on a full-size photograph. The largest image the checkpoint accepts was
  priced in `#37` (41.71 s at the cap) and is what the chip reports, but only the 56x56 smallest
  admissible case was bound here.
- A turn that carries a picture never resumes a prefix, by design; the cache saves the *encode*,
  not the prefill.

### Verification

`DeepSeekV41TurnImagesTests` (9), `DeepSeekV41VisionCostTests` (5), `DeepSeekV41ImagePixelsTests`
(5) and `DeepSeekV41VisionDirectoryTests` (1) are the new pure-logic pins: the price equals the
bound length, a stray placeholder is refused, a span that disagrees with its grid is refused, the
decoded buffer starts at the image's top-left (an off-diagonal white pixel, so a flip, a
transpose and a BGR decode each fail differently), the cost model reproduces the measured points,
the cap is 31x31 cells, and a snapshot without the marker shards is not vision-capable. The full
suite is **578 tests, 0 failures, 43 skipped**, and 0 Metal validation errors under
`MTL_DEBUG_LAYER`.

## DSV4.1 #39: THE CHAT UI NOW RENDERS THE CHECKPOINT'S OWN TURNS, AND THE CONTINUATION WAS MISSING ITS ASSISTANT MARKER

### The question

`#38` closed the composer and named the next gap in its own journal section: for a DeepSeek V4.1
chat turn, `ContentView.handleSendMessage` builds a chatml transcript (`<|im_start|>user\n…`) for
every family in its `else` branch, and only then calls `formatPrompt`. The V4.1 branch of
`formatPrompt` probes `isAlreadyDeepSeekTemplated` — does the prompt *begin* with one of the
checkpoint's own role markers — and chatml begins with `<|im_start|>system`, so the probe is
false and the entire transcript is wrapped inside ONE V4.1 user turn. The API server has always
rendered through `ChatPromptFormatter.formatConversation` (`ServerController.swift:1107`), so the
two entry paths disagreed about what the model reads. Routing was decisive (`chat_prompt_renderer`
0.986, confidence 0.98; scope `full` 0.863/0.79; "does converting tool history risk regressing the
measured tool loop" 0.44, i.e. not a reason to defer).

### What changed

- **The chat builder has a V4.1 branch.** It walks the session's messages into
  `ChatPromptFormatter.ConversationTurn`s and calls `formatConversation`, the same function the
  server calls, with the live dialect (`CompletionEngine.shared.promptDialect`, resolved from the
  checkpoint's own `chat_template.jinja`). The rendered string begins with the checkpoint's
  begin-of-sentence marker, so `formatPrompt`'s passthrough probe returns it untouched: the
  prompt is templated exactly once.
- **A completed turn is flattened the way the API path records it** —
  `thinkingContent + canonicalReasoningCloseTag + content` — because the renderer's job is to
  split reasoning back out. That is what makes the rebuild reproduce a turn that was generated
  with thinking on even after thinking is switched off.
- **A tool result is not a role turn.** The checkpoint has no tool role: a result is the family's
  `tool▁output` pair, placed after the call turn's end-of-sentence marker, and the *next*
  assistant turn opens after it. `ConversationTurn` therefore gained a `kind` (`.text` /
  `.toolResult(name:)`) and `DeepSeekV41PromptDialect.Turn` gained `isToolOutput`; the other
  families ignore both and keep rendering `role` + `text` exactly as before, which is why a tool
  result still carries `role == .user`.
- **A stored call the content no longer carries is rebuilt in DSML.**
  `DeepSeekV41ToolDialect.callBlock(name:arguments:)` emits the container the model itself
  writes, from the dialect's own tag builders, and neutralizes a parameter value that spells the
  parameter closer. Every turn this app produced keeps its raw call in `content`, so this is the
  reload/foreign-turn path.
- **The live continuation was missing its assistant prefix — found while making the two paths
  agree.** `#27`'s probe builds "the continuation the app would build" as
  `…call + endOfSentence + result + <|Assistant|><|end|>`, and production emitted only the result
  pair. The comment beside the call site claimed `startAutoregressiveGeneration` would re-render
  the turn and add the marker; it does not, because the prompt already begins with the dialect's
  marker and `formatPrompt` passes it through verbatim. So the loop was asking the model to
  continue immediately after `</tool▁output>` with no assistant turn opened.
  `formatDeepSeekV41ToolResponseTurn` now appends `assistantPrefix(thinkingEnabled:)` and takes
  the live dialect, and the history rebuild ends a tool-using turn the same way — which is what
  lets a post-tool chat turn extend its pin instead of re-prefilling.

### Evidence

Pure logic (`DeepSeekV41ChatPromptFormatterTests`, 7 new cases): no chatml reaches the checkpoint;
a completed turn closes with the checkpoint's sentence marker; a tool result sits between the
call turn's closer and the next assistant prefix and is never rendered as a user turn; a rebuilt
`callBlock` parses back into the same call through `canonicalized` + `StreamingToolParser`; a
rendered history is not wrapped a second time; and the live continuation's suffix is *identical*
to the history render's, including a payload that spells a role marker (so the escaping half is
compared too). `ChatPromptFormatterConversationTests`'s family list gained `deepseek_v41`, which
immediately put it under the single-turn equivalence, prefix-extension and effort tests that every
other family already had — and exposed one stale premise: `testNonSparkFamiliesIgnoreThinkingEffort`
asserted that only Spark consumes the level, which was never true of V4.1 (it renders a graded
budget), and had only passed because the family was absent from the list.

At the checkpoint's own tokenizer (`testAChatTurnsTokensExtendThePreviousPromptPlusGeneration`,
snapshot-gated):

```
🔎 [dsv41 real] chat turn: 13 prompt ids + 2 generated, turn 2 is 21
🔎 [dsv41 real] chat turn: 13 prompt ids + 7 generated, turn 2 is 26
```

That is the property the prefix pin needs at the level it is actually used: turn 2's ids extend
turn 1's prompt ids *plus the generation ids*, not merely the same string. The boundary is a
special token, which is why no BPE merge crosses it.

Suite: **542 passed, 0 failed, 44 skipped** (586 executed; the 8 new cases are 7 here plus the
snapshot-gated one), and **0 Metal validation errors** under `MTL_DEBUG_LAYER` on the same run.

### What it means

An in-app DeepSeek V4.1 turn is now the same shape as an API turn: one system turn, one turn per
message with the checkpoint's own markers, a per-turn reasoning channel, and tool results in the
output pair. The model reads its own history instead of a foreign transcript nested inside a
single request, and a second chat turn re-encodes to the same ids as the first prompt plus its
generation, so the resident prefix is reusable.

### What it does not mean

- The continuation-after-a-tool-result shape was never run against the real model. It now matches
  `#27`'s probe helper, which is the only reference for it, and it is pinned as a string and as a
  token prefix; an end-to-end generated turn after a tool result is still unmeasured.
- A `tools_load`/`tools_unload` result carries a `[TOOL_REGISTRATION]` line that the live loop
  prepends from the raw JSON, and a recorded turn stores only the rendered output, so a rebuild
  cannot reproduce that one line. The turn still carries the registration's schema as metadata;
  the effect is a lost prefix on the next turn, and every family's history rebuild has the same
  gap.
- The two emergency injections (the tool-loop synthesis directive and the subagent relay
  directive) still append a `<system>…</system>` block as plain text after the assistant turn.
  They are fallback paths, they are chatml-flavoured, and they were left alone rather than
  re-shaped without a measurement.
- The API server still renders a client's tool result as a plain user turn
  (`Function result (name): …`). That is a protocol-adapter choice, not this renderer's, and it
  is unchanged.
- The composer's image-only turn still survives on the marker keeping the prompt non-empty; the
  guard at `startAutoregressiveGeneration` was not touched.

### Verification

`DeepSeekV41ChatPromptFormatterTests` (11), `ChatPromptFormatterConversationTests` (20),
`DeepSeekV41ToolDialectTests` (28) pass; the full suite is 542/0/44 and the Metal-validation run
is clean.

## DSV4.1 #40: A TURN AFTER A TOOL RESULT READS IT, AND THE CONTINUATION IS THE TURN'S OWN TOKEN STREAM

### The question

`#39` closed the continuation-after-a-tool-result SHAPE and named what it had not done in its own
section: the shape was pinned as a string (`testTheLiveContinuationSuffixMatchesTheHistoryRender`)
and as a token prefix (`testAChatTurnsTokensExtendThePreviousPromptPlusGeneration`), and `#27` had
measured which result container the checkpoint can READ, but no token had been generated from the
continuation the app actually builds. A model that answers five containers in a probe is not a loop
that answers one. Routing was decisive (`tool_result_end_to_end` 0.876, confidence 0.855; "measure
before build" 0.969), so this slice measured rather than built.

### What was measured

One walk of the loop the app walks, on the published checkpoint:

- the checkpoint is asked for a shell command with `DeepSeekV41ToolDialect.instructions()` in the
  system prompt, and it freezes on its OWN call rather than a hand-written one;
- the response is `shell_run`'s JSON, rendered by `renderToolResultForModel` and wrapped by
  `AgentHarness.formatDeepSeekV41ToolResponseTurn` — the same call `ContentView` makes;
- the continuation is `formattedPrompt + closed call turn + the result turn`, and the answer is
  generated from it.

### Evidence

`DeepSeekV41PublishedTurnTests.testATurnAnswersAfterAToolResultOnThePublishedCheckpoint`:

```
🔎 [dsv41 real] tool continuation: 299 prompt + 41 generated + 1 end + 75 suffix = 416 spliced, 416 re-encoded, splice==reencode true
🔎 [dsv41 real] tool continuation resumes 339 of 416
🔎 [dsv41 real] tool answer: lengthCapped after 64 token(s), reused 339, prefill 24.72 s, decode 20.78 s
🔎 [dsv41 real] tool answer text: "Here are the files in `/tmp`:\n\n| Permissions | Owner | Group | Size | Date | Name |\n|---|---|---|---|---|---|\n| `-rw-r--r--` | derek | wheel | 42 B | Oct 31 09:14 | `file_a.txt` |\n| `-rw"
```

Three things are asserted, not printed:

- the continuation the app splices (prompt ids + the real generation ids + the closure + the
  suffix) IS the re-encoded string, 416 ids both ways, so the pin and the prompt agree token for
  token;
- the cache resumes 339 of 416, which is the prompt plus 40 of the 41 generated ids (the freeze
  token is drawn and never forwarded), so a post-tool turn pays a 77-position walk rather than a
  full prefill;
- the answer names `file_a.txt`, which the question never mentioned: it read the result.

Two runs produced the same 64 ids and the same text, at temperature 0.

### What it means

The live loop answers after a tool result with the shape `#39` shipped: the checkpoint reads the
result the app renders, the assistant turn the continuation opens is the one the model answers in,
and the pin extends by exactly the ids that went through the cache. On this machine a continued
turn costs ~0.32 s for each of its 77 prefilled positions and ~0.32 s per generated token
(24.72 s + 20.78 s for the 64-token answer above).

### What it does not mean

- Only the thinking-off continuation was generated. The thinking-on suffix differs by its last
  marker only, but a branch whose prompt is SHORTER than the resident cache cannot reuse a partial
  prefix (`reusablePrefix` refuses when the stored sequence is longer than the prompt), so measuring
  it after the thinking-off turn would pay a full 416-position prefill for one special token. The
  marker itself is pinned in `#39`.
- The API path still renders a client's tool result as a plain user turn (`Function result
  (name): …`). That is a protocol-adapter choice, and it is unchanged.
- A `tools_load`/`tools_unload` result still cannot reproduce its `[TOOL_REGISTRATION]` line on a
  history rebuild; the gap is every family's, not this checkpoint's.
- The turn was length-capped at the test's own 64-token budget while writing its table. That is the
  budget, not the model failing to stop.

### Verification

`testATurnAnswersAfterAToolResultOnThePublishedCheckpoint` is new and opt-in (it reads the 475 GB
checkpoint) and passes in 144.5 s. The full DeepSeek suite is **542 passed / 0 failed / 45 skipped**
(587 executed; the new case skips without the marker), and the Metal-validation run is the same
with **0 validation errors**. The new case also ran under `MTL_DEBUG_LAYER` with the marker and
reproduced the same 64 ids and text; its timings roughly double there (48.85 s + 39.22 s), which is
the validation layer and not the model.

## DSV4.1 #41: THE VISION TOWER'S QUADRATIC TERM RUNS ON THE GPU, 5.05x FASTER

### The question

`#37` priced the host vision tower on the app's own `-O` build: 2.74 s for the smallest
admissible image and 41.71 s for the largest, split `0.22 s + 1.126 ms/token +
4.245e-07 s/token^2`, with attention plus softmax **76% of the large one** (31.75 s). The host
attention keeps a `tokens * tokens` score block per head while it computes a bidirectional
softmax, so the largest image is 74.8 million (query, key) pairs per head per block and the
score block alone is 299 MB. That term is the whole prize, and it is exactly what a GPU should
not be doing with a score matrix. Routing was decisive: `vision_scope` chose `attention_first`
at 0.959 with 0.945 confidence, and the alternatives (`whole_tower` 0.002,
`linear_stages_first` 0.007, `not_vision` 0.032) were not close.

### What landed

- **`dsv41_vision_rope_split_f32`** turns the tower's interleaved `3 * hidden` row per patch
  token into three `[heads, tokens, headDim]` planes, applying the checkpoint's split-half
  rotation to q and k and copying v untouched. One thread per (token, head, pair), so a pair's
  two halves are written by the same thread.
- **`dsv41_vision_attention_f32`** is the stage itself, in 32-key tiles. The dot product needs
  one lane per key; the value sum needs one lane per output dimension. A tile is what makes
  both true at once: the tile's 32 scores are reduced with `simd_max` and `simd_sum`, and each
  lane's weight is broadcast with `simd_shuffle` so every lane folds all 32 value rows into the
  two dimensions it owns. The running maximum and denominator are uniform, a new tile maximum
  rescales the two lane-local accumulators, no score row is stored, and no dot product is
  computed twice.
- **`DeepSeekV41VisionAttentionDevice`** is the stage runner: `splitAndRotate` builds the
  planes, `attention` runs the stage, and `runAttention(_:into:)` returns the dispatch's GPU
  time without the readback (the readback of a `tokens * hidden` matrix costs more than the
  dispatch it describes, and `gpuEndTime - gpuStartTime` keeps that out of the number).

### Evidence

Parity, small shapes, against the host definition (`DeepSeekV41VisionForward.attention`) and the
batched form the tower actually calls:

```
🔎 [dsv41 vision] attention parity: worst 2.98e-07 against the serial rule, 2.38e-07 against the batched one
🔎 [dsv41 vision] narrow-head parity: worst 1.79e-07
```

The published geometry, one block's attention (`runsAttention` GPU time, three attempts, best):

```
🔎 [dsv41 vision] attention smallest: 1521 tokens per head, 6.72 ms per block, 0.21 s for 32 blocks
🔎 [dsv41 vision] attention smallest row 0: worst 1.25e-06 against the host rule
🔎 [dsv41 vision] attention smallest row 760: worst 1.61e-06 against the host rule
🔎 [dsv41 vision] attention largest: 8649 tokens per head, 196.66 ms per block, 6.29 s for 32 blocks
🔎 [dsv41 vision] attention largest row 0: worst 2.21e-06 against the host rule
🔎 [dsv41 vision] attention largest row 4324: worst 2.38e-06 against the host rule
```

So the term `#37` measured at **31.75 s** now costs **6.29 s** for the 8,649-token image
(**5.05x**) and the 0.98 s term for the 1,521-token image costs **0.21 s** (4.55x), with a
host-row check at each size so the timing is of the right kernel and not of a fast wrong one.
If the surrounding path were unchanged and the transfers free, the largest image's encode would
go from 41.71 s to about 16.25 s; that is an estimate, not a measurement, and the wiring is not
in yet.

Two defects stood between the first draft and that number, and both were caught by the cases
rather than by inspection:

1. **The first design summed value accumulators ACROSS lanes**, which added dimension `l` to
   dimension `m`; every output row came back constant across its dimensions. The tile plus
   `simd_shuffle` is the fix, and it is why the kernel looks like flash attention.
2. **A lane with no output dimension still wrote its accumulator.** `laneInSimd` is a dimension
   index only while it is below `headDim`; at a 12-wide head lanes 12..31 wrote into the next
   head's row. The published 64-wide head has a dimension for every lane, so this was invisible
   at the shape that ships and fatal at the narrow one. The write, the first-dimension value
   read and the accumulate are all guarded now.

### What it means

The vision tower has a device path for the stage that actually costs the money, and it is
checked against the host rule at both the shape that ships and the shapes that break lane
assumptions. The GPU number is a kernel number, not an integration number.

### What it does not mean

- The rest of the tower -- patch embed, the block projections, pixel-unshuffle, the aligner --
  is still on the host. An encode would still pay those costs and would add the marshal of
  `qkv` and the readback per block, which is why the 16.25 s figure is an estimate.
- 196.66 ms is 1.56 TFLOP/s of arithmetic, well under what the GPU can do; the stage is likely
  limited by re-reading each head's key and value planes for every query block. That is the next
  measurement if the stage is ever integrated, not a claim that this is the fastest form.
- The unit is one block at the published geometry on synthetic values; the tower's real sequence
  of 32 blocks shares nothing across blocks, so 32x is the right extrapolation, but the number
  was not taken with the tower in the loop.
- The validation layer runs the same kernel about 6.7x slower (1,310.39 ms at the largest size).
  Those are debug-layer numbers and must not be quoted as the stage's cost.

### Verification

`DeepSeekV41VisionDeviceTests` is new: three parity cases run with the default suite (2.98e-07
and 1.79e-07 worst differences) and the published-geometry timing is opt-in behind
`/tmp/dynamoe_dsv41_vision_measure`. The full DeepSeek suite is **545 passed / 0 failed / 46
skipped** (591 executed) without the marker; with it, under `MTL_DEBUG_LAYER`, **546 passed / 0
failed / 45 skipped and 0 validation errors**.

## DSV4.1 #42: THE VISION TOWER'S PROJECTIONS RUN ON THE GPU, 11.16x FASTER

### The question

`#41` left the largest image at an estimated 16.25 s, with the per-token term -- `1.126 ms/token`
over 8,649 tokens, i.e. **9.74 s** -- the majority of what was left. That term was never split by
stage: it came out of `#37`'s three-point solve, and `#37` only checked it against the fixed and
quadratic terms, not against its own parts. What the tower calls there is not scalar Swift:
`DeepSeekV41VisionForward.linearBatched` flattens the token block and runs **one
`cblas_sgemm`** per projection, so at 838 MFLOP per patch token in 1.126 ms the host is running
near **744 GFLOP/s**. Moving that to the GPU is therefore a bet on device headroom, not a repair,
and it is also what any device-resident tower needs -- a token matrix that leaves for the host
between blocks costs a transfer per stage.

Router: `vision_priority` (is the tower the highest-value target ahead of the protocol surfaces?)
came back **0.933**; the first `next_slice` request was mid-band (`vision_device_linear_stages`
0.395 at 0.308 confidence) and was re-filed with the measured costs in `state`, which raised it to
**0.578** and put `new_kernel_worth_it` at **0.819** -- a decisive "a GEMM kernel is justified".
The smaller alternatives were `attention_queries_per_group` 0.098, `wire_the_shipped_kernel_only`
0.115 and `host_stage_split_first` 0.209.

### What landed

- **`dsv41_vision_gemm_f32`** computes `C = A * Bᵀ + bias` over a whole block of patch tokens.
  `A` is `[rows][depth]`, `B` is `[outputs][depth]` (exactly how a `Linear`'s weight is stored), `C`
  is `[rows][outputs]` -- the host GEMM's own layout, so neither operand is transposed.
  - One threadgroup owns a **64x64** output tile. Its 256 threads are eight SIMD groups laid out
    2x4, and each group owns 32 rows by 16 columns, i.e. **eight `simdgroup_float8x8`
    accumulators**; a 64x16 chunk of each operand is staged once per depth step and read by all
    eight groups.
  - The weight tile is `[output][depth]`, so the 8x8 the accumulator needs is its TRANSPOSE and the
    fragment is loaded with `simdgroup_load`'s transpose flag rather than by staging a second copy.
  - Bounds are handled by **zero-filling the staged tiles** and guarding only the write-out, which
    is what lets a ragged `rows`, `outputs` or `depth` take the same path: a zero outside the matrix
    contributes nothing to a dot product. The bias is added on the write, where the guards are.
- **`DeepSeekV41VisionLinearDevice`** is the stage runner: `runLinear` takes device buffers and
  returns `gpuEndTime - gpuStartTime` (no readback), `linear` takes the tower's `[[Float]]` and a
  `DeepSeekV41Matrix` and returns rows, and `readRows`/`makeBuffer` are the plumbing a measurement
  needs. A ragged input row is REFUSED, because the kernel would read the next row's values for
  the missing tail.

### The tile had to be BOUND, not declared, and the validation layer is how that was found

The first form declared its three tiles in the kernel body. On the app's own build the offline
`metal` compiler allocated **49,152 bytes** for a kernel that names **24,576** -- it duplicates the
whole set: `xcrun metal` + `metallib` on the same source reports 49,152 while
`makeLibrary(source:)` at runtime reports 24,576, and a minimal kernel with the same three
declarations reports 24,576 through the offline compiler too, so it is this kernel's shape rather
than the declarations. The driver ran it anyway and produced correct numbers -- the three `-O`
measurement runs below passed before the fix -- but `MTL_DEBUG_LAYER` refused the dispatch against
the device's `maxThreadgroupMemoryLength` of **32,768** and killed the test process, which is
exactly the class of defect the validation run exists for. The tiles are now
`[[threadgroup(0..2)]]` parameters sized by the host (`setThreadgroupMemoryLength`, 4 KB + 4 KB +
16 KB = 24,576), so `staticThreadgroupMemoryLength` is **0** and the number in the shader is the
number the device sees. It is also faster, because the compiler is no longer carrying a duplicate
set: `wo` at the largest size went 6.88 ms -> 1.71 ms.

### Evidence

Parity, `-O`, against both host forms (relative, and the definition is the only check that would
catch a shared transposed operand):

```
🔎 [dsv41 vision] linear parity (ragged): worst relative 0.0
🔎 [dsv41 vision] linear parity (definition): worst relative 1.1920929e-07
🔎 [dsv41 vision] linear parity (published width): worst relative 0.0
🔎 [dsv41 vision] linear threadgroup memory: declared 0, bound 24576, device limit 32768
```

The four block projections at the two published token counts, app's own `-O` build
(`/tmp/dd_dsv41_opt`), device arm = `runLinear` GPU time on pre-uploaded buffers, host arm =
`linearBatched` as the tower calls it (flatten and slice included), best of three:

| shape | smallest (1,521 tokens) | largest (8,649 tokens) |
|---|---|---|
| qkv `1024 -> 3072` | 3.52 ms vs 9.16 ms (2.60x) | 4.38 ms vs 55.31 ms (12.63x) |
| wo `1024 -> 1024` | 0.57 ms vs 3.66 ms (6.43x) | 1.71 ms vs 18.78 ms (10.97x) |
| w1 `1024 -> 5632` | 3.10 ms vs 16.06 ms (5.19x) | 7.92 ms vs 99.21 ms (12.53x) |
| w2 `2816 -> 1024` | 3.57 ms vs 6.58 ms (1.84x) | 4.81 ms vs 36.66 ms (7.63x) |
| **32 blocks** | **0.34 s vs 1.13 s (3.30x)** | **0.60 s vs 6.72 s (11.16x)** |

Runs before the tile was bound, same harness, read 0.36 s vs 1.11 s (3.12x) and 0.91 s vs 6.38 s
(7.04x), and a second after it read 3.30x and 11.16x; the device arm moves 12-27% between runs
because the machine was carrying a load average of 2-7, so the honest statement is **3.1-3.9x at
the smallest image and 7.0-11.2x at the largest**.

The same test on a DEBUG build reports the host arm at 34.76 s and 197.14 s, because the flatten
and the per-token slice are `-Onone` while `cblas` is not. That is a build artifact and the reason
every host number here and in `#37` is quoted from `-O`.

### What it means

The per-token term is now SPLIT, which it had never been: the four block projections are **6.72 s
of the 9.74 s** at the largest image -- 69%, independently confirming the three-point solve -- and
the device runs them in **0.60 s**. Arithmetic on measured parts, not a measured encode: with
`#41`'s attention also in place the largest image would be `0.22 + (9.74 - 6.72 + 0.60) + 6.29` =
**10.13 s** against the 41.71 s that ships, i.e. 4.1x. Nothing is wired, so that number is a sum of
three measurements of different things and must not be quoted as an encode.

### What it does not mean

- **`encode` still calls `linearBatched`.** This is a stage with an oracle. Swapping the stage in
  without moving the rest of the tower would be a LOSS: at the largest size `w1`'s product is
  8,649 x 5,632 floats (195 MB) and its input 35 MB, so a host<->device round trip per projection
  costs more than the 99 ms host GEMM it replaces. The win requires keeping the token matrix
  device-resident across all 32 blocks, which is a device encode, not a substitution.
- A device encode still needs four things this file does not have: an FP32 gelu and silu, the fused
  `silu(gate) * up` split of the 2x-intermediate projection (the MLP emits gate and up halves in
  one matrix), the residual add (an FP32 `vector_add_f32` exists, but nothing fuses it with a
  projection), and pixel-unshuffle into the aligner's channel-major cells. `dsv41_rmsnorm_f32_gain`
  already exists for the two norms per block.
- The weight upload is outside the timed arm, which is the steady state an integrated tower would
  create (970 MB of BF16 tower weights, 1.9 GB once decoded to FP32, uploaded once and not per
  block). A per-call uploader would add a read plus a copy to every encode and the table would not
  describe it.
- `wo`'s host rate at the smallest size is the weakest of the four (1.84x on `w2`, 2.60x on `qkv`
  against 12.63x at the large size): the same kernel, so it is the shape -- 16 column tiles against
  88, i.e. 2,176 threadgroups against 11,968 on a machine with ~40 cores. Named as the next tuning
  target if the tower is ever integrated, not fixed here.
- These are kernel numbers on one block's projection at synthetic values; the tower's 32 blocks
  share nothing but the weight, so 32x is the right extrapolation for the DISPATCH, and no run took
  it with the tower in the loop.

### Verification

`DeepSeekV41VisionLinearDeviceTests` is new: five parity/refusal cases and the threadgroup-memory
guard run with the default suite, and the published-geometry timing is opt-in behind
`/tmp/dynamoe_dsv41_linear_measure`. The class is now in `/tmp/args41m37.txt` (93 entries), so the
full DeepSeek suite is **551 passed / 0 failed / 47 skipped** (598 executed) without the marker,
and **551 passed / 0 failed / 47 skipped with 0 validation errors** under `MTL_DEBUG_LAYER`. The
run BEFORE the tiles were bound is the one that exposed the 49,152-byte allocation: it lost the
four parity cases and killed the process each time.


## DSV4.1 #43: THE WHOLE VISION TOWER RUNS IN ONE DEVICE PASS, 8.81x TO 6.02x FASTER

### The question

`#41` and `#42` measured the tower's two expensive halves on the GPU and left both as STAGE
runners: `encode` still called `linearBatched`, so neither stage ran in a turn. `#42`'s own
journal said why wiring alone would be a LOSS -- at the largest image `w1`'s product is 195 MB, so
a host<->device round trip per projection costs more than the host GEMM it replaces -- and the
prize only exists if the token matrix stays on the GPU across all 32 blocks. That is a device
encode: patch embed, the two norms, the four projections, the split-and-rotate, the attention, the
gated MLP, the final norm, the pixel-unshuffle and the aligner, all in one command buffer.

Routing agreed decisively. The first request (`/tmp/route43.json`) put `vision_device_encode` at
**0.954 @ 0.947**, `device_encode_now` at **0.989** and `measure_before_build` at **0.935**; every
other candidate was below 0.04. The last of those three is why this section opens with a host
stage split: the middle 3.02 s of the per-token term had never been seen.

### What landed

Three elementwise kernels in `ComputeShaders.metal`, all FP32 and all `dsv41_vision_*`:

- `dsv41_vision_gelu_f32` -- the aligner's activation. MSL has no `erf`, so the kernel carries
  Abramowitz & Stegun 7.1.26 (`|error| <= 1.5e-7`), which is far below the parity bar and is the
  only approximation the device pass introduces.
- `dsv41_vision_silu_mul_f32` -- one thread per output unit, reading the gate and up halves of one
  `w1` row.
- `dsv41_vision_unshuffle_f32` -- channel-major cells with the padded rows and columns written as
  zero.

`DeepSeekV41VisionEncodeDevice` composes those with the two existing stage runners
(`dsv41_vision_gemm_f32`, `dsv41_vision_rope_split_f32` + `dsv41_vision_attention_f32`) and the
existing `dsv41_rmsnorm_f32_gain` / `vector_add_f32`. One command buffer per encode, one
`waitUntilCompleted`, no host round trip between stages. The weights are read through the same
`DeepSeekV41VisionWeights` resolver the host path uses and uploaded once per tower (0.35 s), and
the token-sized workspace is reused while the patch grid does not change. `encodeOnHost` gained an
`onStage` closure (nil in every production path) so the host split could be measured without
perturbing the oracle; the host path itself is unchanged.

The device pass is now what `DeepSeekV41VisionForward.encode` calls when it carries one, and the
engine builds one in `visionForward()`. `dynamoe_dsv41_vision_device_off` forces the host path.

### The host split, measured (`-O`, load average ~2.7)

The two smaller sizes exist to separate the quadratic term; the largest is the one that matters.
Each number is the sum over 32 blocks of `encodeOnHost`'s stage clock, from the first of the two
runs (the second moved the host total by 0.9%).

| stage | 39x39 / 1,521 tokens | 63x63 / 3,969 tokens | 93x93 / 8,649 tokens |
|---|---|---|---|
| attention | 1.241 s (44.0%) | 7.822 s (67.9%) | 34.034 s (81.4%) |
| w1 | 0.418 s | 0.946 s | 1.954 s |
| qkv | 0.355 s | 0.833 s | 1.745 s |
| silu(gate) * up | 0.215 s | 0.564 s | 1.237 s |
| w2 | 0.212 s | 0.488 s | 1.019 s |
| wo | 0.132 s | 0.290 s | 0.611 s |
| residual add 2 | 0.075 s | 0.200 s | 0.439 s |
| residual add 1 | 0.076 s | 0.196 s | 0.439 s |
| norm1 | 0.016 s | 0.044 s | 0.102 s |
| norm2 | 0.014 s | 0.037 s | 0.082 s |
| aligner w1 | 0.036 s | 0.051 s | 0.081 s |
| aligner w2 | 0.020 s | 0.029 s | 0.049 s |
| unshuffle | 0.003 s | 0.007 s | 0.017 s |
| patch embed | 0.003 s | 0.007 s | 0.016 s |
| aligner gelu | 0.001 s | 0.003 s | 0.007 s |
| final norm | 0.001 s | 0.001 s | 0.003 s |
| **host total** | **2.82 s** | **11.52 s** | **41.83 s** |

The answer to the measurement question: at 8,649 tokens the host's non-attention work is **7.80 s**,
of which the four projections are **5.33 s** and the elementwise stages (silu * up, the two
residual adds, unshuffle and gelu) are **2.13 s**, with the norms at 0.18 s. The elementwise stages
were not negligible -- 5.1% of a largest-image encode -- and the three new kernels were worth
writing.

### The measured encode

Same run, same tower, same checkpoint (`-O`, app's own code):

| image | patch tokens | host | device (GPU) | speedup | parity (relative) |
|---|---|---|---|---|---|
| 546x546 (the pixel floor) | 1,521 | 2.82-2.84 s | **0.32 s** | **8.81-8.93x** | 1.7e-4 |
| 882x882 | 3,969 | 11.52-11.76 s | **1.70 s** | **6.78-6.90x** | 4.0e-5 |
| the largest admissible | 8,649 | 41.83-42.20 s | **6.90-6.95 s** | **6.02-6.12x** | 4.8e-5 |

The ranges are two runs of the same test: the first before and the second after a barrier was
added at each stage boundary (a no-op on the values, and the device arm moved under 1%). The
weight upload is 0.35-0.36 s once per tower and is outside the device arm. Across the two runs the
device three-point solve reads `-0.06 to -0.08 s + 0.13-0.15 ms/token + 7.6-7.9e-08 s/token^2`, and
the host `-0.11 to -0.28 s + 1.30-1.44 ms/token + 4.0-4.1e-07 s/token^2`. The negative constant is
the fit's noise (the GPU time excludes the upload), so `DeepSeekV41VisionCostModel.measured` is
fitted with a zero constant from the two extreme points -- `8.4e-5 * t + 8.32e-8 * t^2` -- which
reproduces 0.32 s, 1.70 s and 6.90-6.95 s to within 4%. The `#37` host constants are kept as
`hostReference`.

### What it means

- **The tower's shipped range is now 0.32-6.90 s, not 2.82-41.83 s.** The composer's estimate
  moved with it; the smallest image is under a third of a second and the largest under seven.
- **The device total is almost entirely #41's attention.** 6.90-6.95 s at the largest size against
  `#41`'s measured 6.29 s of attention, i.e. everything else -- patch embed, all 128 projections,
  all 64 norms, the elementwise work, the unshuffle and the aligner -- is **~0.6-0.7 s**. The
  ceiling for this pass is now the attention kernel, not the graph around it.
- **The speedup FALLS as the image grows** (8.93x -> 6.12x) precisely because the host's attention
  share rises (44.0% -> 81.4%): the already-device half cannot be accelerated twice. Quoting one
  number for the tower without the size is not meaningful.
- **The elementwise stages cost the device tens of milliseconds, not seconds.** The host paid
  2.13 s for silu * up, two residual adds, unshuffle and gelu at the largest size; the device's
  whole non-attention remainder is smaller than that.

### What it does not mean

- **Not a 6x faster image TURN.** The encode is one part of an image turn; prefill and decode of
  the expanded prompt are untouched.
- **Not a faster attention kernel.** #41's 6.29 s is the floor this pass inherits, and the attention
  is 81.4% of the largest host encode.
- **Not the isolated #42 projection numbers.** #42 timed `linearBatched` alone on warm synthetic
  weights and read 6.72 s for 32 blocks; the in-situ host tower reads 5.33 s for the same
  projections at the same size, because the isolated harness also timed the `[[Float]]`
  materialization around each GEMM. The encode table is the number to quote for the tower.
- **The parity bars are relative to the row scale, not absolute.** 1.7e-4 at the smallest size and
  4.8e-5 at the largest are the host's own summation-order difference from a device pass, through
  32 blocks, including the erf approximation.
- **The 0.35 s upload is a real first-image cost.** It is not in the GPU time and not in the cost
  model, which prices a warm tower; a session's first picture pays it.
- These numbers are from one `-O` build on an M5 Max at load average ~2.7. The device arm was
  reproducible to under 1% across two runs; the host arm moved with load.

### Verification

`DeepSeekV41VisionEncodeDeviceTests` is new: seven parity/refusal cases run with the default
suite (the whole small tower against the host, a padded grid, two blocks, workspace reuse across
grids, a ragged patch row, a missing weight, and a text-width mismatch), and the published-
geometry measurement is opt-in behind `/tmp/dynamoe_dsv41_vision_encode_measure` or
`DYNAMOE_DSV41_VISION_ENCODE_MEASURE=1`. The tiny tower agrees with the host to **2.2e-7
relative**, and every timed published arm checks one grid of rows against the host before
printing. `DeepSeekV41PublishedTurnTests.testAnRGBImagePricesAndBindsOnThePublishedWeights` now
drives the engine's own tower (`runtime.visionForward()`), asserts it carries a device pass, and
therefore exercises the wiring and the one-time upload end to end; a Debug bundle reports that
upload at `-Onone` (~26 s for 1.9 GB of FP32), which is why the app's `-O` figure is the only one
quoted.

The full DeepSeek suite on this code is **556 passed / 0 failed / 51 skipped** (607 executed),
and the same run under `MTL_DEBUG_LAYER` is **556 / 0 / 51 with 0 validation errors**.

## DSV4.1 #44: THE VISION ATTENTION IS AT 10% OF THIS GPU'S FP32 RATE, AND NOTHING THAT RESHAPES ITS LANES MOVES IT

### Question

`#43` left one kernel as the tower's floor: `dsv41_vision_attention_f32` is 6.29 s of a 6.90 s
device encode at the largest admissible image, and everything else -- patch embed, all 128
projections, all 64 norms, the elementwise work, the unshuffle and the aligner -- is ~0.6-0.7 s.
The kernel was written for two reasons (one load instruction per key row is `headDim` floats wide,
i.e. 32 cache lines for 128 bytes of use; and a threadgroup's several query SIMD groups can share a
staged tile), and neither had been tested. So the question for this slice was: which shape should
that kernel have, and is it even off the machine's pace?

### What was measured

Six shape arms at the published geometry (16 heads of 64, 1,521 and 8,649 patch tokens), best of
three, on the app's own `-O` build, load average ~2.6:

| arm | 1,521 tokens | 8,649 tokens |
|---|---|---|
| `direct/128` (shipped, 4 queries per threadgroup) | 6.15 ms/block, 1.54 TFLOP/s | 194.1 ms/block, 1.58 TFLOP/s |
| `direct/256` (8 queries) | 6.11 ms, 1.55 | 194.2 ms, 1.58 |
| `direct/512` (16 queries) | 6.11 ms, 1.55 | 195.6 ms, 1.57 |
| `staged/128` (key tile in threadgroup memory) | 9.90 ms, 0.96 | 303.7 ms, 1.02 |
| `staged/256` | 7.86 ms, 1.21 | 244.9 ms, 1.25 |
| `staged/512` | 7.42 ms, 1.28 | 229.6 ms, 1.33 |

- **Staging the key tile LOSES** 18% at its best (229.6 against 194.1) and 55% at its worst. It
  fixes the coalescing it was built for and pays two threadgroup barriers per tile for it, and a
  threadgroup of four queries does not reuse the tile enough to get that back.
- **The threadgroup width is irrelevant** -- 4, 8 and 16 queries per threadgroup are the same
  kernel to within 1%. So the dispatch is not limited by how many queries share a key/value read,
  which is what kills every "more reuse" hypothesis before it is built (including a retry of `#6`).
- Every staged arm agrees with the shipped one **bit for bit** (worst 0) on the whole
  `tokens x heads x headDim` output, so the negative is a measurement of the right rule.

### Where the dispatch actually goes

The ablation kernel runs the two halves of a step separately. `mode` 2 runs both and reproduces the
shipped kernel bit for bit (asserted), so the split is of this kernel and not another:

| 8,649 tokens | ms/block | share |
|---|---|---|
| dot product + row softmax | 83-89 | 30-34% |
| value fold (over a synthetic score) | 161-171 | 66-70% |
| both | 196-207 | |

Solving the three arms together (`both`, `dot+softmax`, `fold`) puts the per-tile softmax and loop
overhead at **~53 ms**, the dot's arithmetic at **~36 ms** and the fold's at **~119 ms**. The fold
is the stage.

Then two more probes on the fold alone, which is what made this slice's answer unambiguous:

- **removing the weight broadcast** (`simd_shuffle` for all 32 keys) is **1.00x** -- the broadcast is
  free;
- **removing half the row's loads** (one scalar instead of two) is **1.04-1.05x** -- the loads are
  nearly free.

### The machine's pace, measured

`dsv41_vision_fma_peak_f32` is eight independent FMA chains per thread and no memory traffic at
all, 34.4 GFLOP over 1,048,576 threads: **15.23-15.74 TFLOP/s**. The attention's 1.56 TFLOP/s is
therefore **10% of this GPU's SIMT FP32 rate**, not 90% of it -- four reshapes landing within 5% of
each other was not the ceiling, it was a shared defect those reshapes do not touch. What survives
that list is a per-lane cost that every arm pays: the fold's per-key addressing (a 64-bit
multiply-add and a 64-bit pointer add per key), its load instruction issue, and the loop itself.

### What it means

- **One reshape does ship.** `dsv41_vision_attention_pairfold_f32` gives a lane two ADJACENT
  dimensions of the fold -- so its pair is one aligned `float2` -- and splits the tile across four
  accumulator pairs instead of one 32-long dependent chain. At 1,521 tokens it is **1.28-1.39x**
  (6.69 -> 5.24, 6.29 -> 4.52 across runs); at 8,649 it is **1.03-1.15x** (195.5 -> 189.2,
  196.8 -> 191.3, 217.5 -> 189.2 under load). It is never slower. It is the default form now, and
  the attention the tower runs measures **5.14 ms/block at 1,521 tokens (0.16 s for 32 blocks) and
  184.6 ms at 8,649 (5.91 s)** against `#41`'s 6.29 s.
- **The tower's own encode moves with it.** Re-measured on the shipped path at load average 1.5:
  39x39 **0.31 s** device (host 3.03 s, 9.73x), 63x63 **1.60 s** (12.32 s, 7.68x), 93x93 **6.74 s**
  (44.91 s, 6.66x), parity 1.4e-4 / 3.5e-5 / 8.2e-5, upload 0.37 s -- against `#43`'s
  0.32 / 1.70 / 6.90 s.
- **The 4-way split alone bought nothing.** The first version of that kernel made the split a
  runtime `foldUnroll` branch and measured 1.02x, which measured a branch: the shipped form is
  unconditionally four-way with no data-dependent exit from the body.
- **The variable that does NOT matter is how a lane's dimensions are laid out away from
  memory-width terms, or how many queries a threadgroup holds.** Both were the kernel's stated
  reasons for its shape, and both are now measured dead.
- **The next lever is not a SIMT shape.** At 10% of the FP32 rate with the loads, the broadcast and
  the chain all cleared, the remaining cost is per-element instruction issue, and the matrix units
  do 512 multiply-adds per instruction. That is a different kernel (8x8 score tiles in registers,
  P x V as an MMA), not another reshape of this one, and it is its own slice.

### What it does not mean

- **Not a faster image turn.** The tower's encode is 6.9 s at the largest image and this takes
  ~0.4 s of it; prefill and decode of the 994-position span are untouched.
- **Not a claim that the fold is understood.** Removing the loads, the broadcast, the vectorization
  and the dependency chain each failed to move it, which is a measurement of what it is NOT, not an
  explanation of what it is. Do not build a fifth SIMT shape on a theory from this table.
- **Not stable to the last digit.** The 8,649-token arm moved 195 -> 217 between a quiet and a
  loaded run (`uptime` 2.6 against 4.4); every range above is quoted as a range for that reason.
- **Not a `float2` win alone or an unroll win alone.** They landed in one kernel.

### Two device faults this exposed

- **`simd_shuffle` is a group collective and cannot be run by part of a group.** The paired fold
  guards its accumulation on `ownsPair`, and a narrow head leaves most lanes owning nothing; with
  the broadcast inside that guard, participating lanes were handed another lane's weight. Full
  width (64 dimensions, every lane owns a pair) was exact, so the bug only appeared on the 8- and
  32-wide heads: **0.62 and 0.705 absolute error where the shipped kernel was 2.4e-7**. The
  broadcast is unconditional now and only the load and the accumulation are guarded.
- **A refused dispatch must still close its encoder.** `runAttention` created an encoder and let a
  `shapeMismatch` escape before `endEncoding`, and Metal asserts on release: a Debug-build abort
  inside a test that was only asking for a refusal.

### Verification

The default suite gains `testTheShippedPairedFoldMatchesTheDirectKernelAndTheHost` -- three shapes
(narrow head 8 at 37 tokens, full width 64 at 5 tokens, 32 at 33 tokens) against both the direct
kernel and the serial host rule, worst **2.4e-7 / 1.2e-7 / 3.0e-7** -- and
`testThePairedFoldRefusesAnOddHeadDimension`, which also asserts the direct form still runs the odd
head it always ran. Four opt-in cases join the published-geometry harness behind
`/tmp/dynamoe_dsv41_vision_measure`: the six-arm shape sweep, the ablation, the fold probes and the
FMA peak, and every timed arm checks rows against the host before it prints. The parity case of the
existing narrow-head test still passes through the new default.

The full DeepSeek suite on this code is **558 passed / 0 failed / 55 skipped** (613 executed, 94
classes), and the same run under `MTL_DEBUG_LAYER` is **558 / 0 / 55 with 0 validation errors** and
no failed assertions -- which is what a new kernel and a changed default form have to survive before
they are trusted.

## DSV4.1 #45: THE MATRIX UNITS RUN AT THE FMA RATE, SO AN MMA ATTENTION LOSES BY ARITHMETIC

### Question

`#44` closed with a named next lever: the attention runs at 1.56 TFLOP/s, 10% of this GPU's SIMT
FP32 peak, the loads / the broadcast / the dependency chain had all been measured away, so "the
remaining cost is per-element instruction issue, and the matrix units do 512 multiply-adds per
instruction". That is a theory about a number nobody had measured -- whether Apple's FP32 matrix
units actually run FASTER than the FP32 FMA pipes -- plus a cost model that ignores what a fragment
cannot do (it cannot be indexed, so it cannot be rescaled). This slice measured both before writing
the kernel, then wrote it and swept its two shape knobs.

### The units' own rate, measured first

`dsv41_vision_mma_peak_f32`: eight independent `simdgroup_multiply_accumulate` chains per SIMD
group and nothing else, 8,192 groups, 256 iterations, 17.2 GFLOP, best of three. `dsv41_vision_
fma_peak_f32` was re-run beside it in the same process so the two are comparable:

| probe | run 1 (load 2.2) | run 2 (load 2.0) |
|---|---|---|
| FP32 matrix units, product only | 2.19 ms, **7.84 TFLOP/s** | 2.33 ms, **7.38 TFLOP/s** |
| the same loop fed from threadgroup memory | 2.19 ms (1.00x) | 2.19 ms (1.06x) |
| SIMT FP32 FMA chains | **8.58 TFLOP/s** | **7.73 TFLOP/s** |

**A matrix unit is not a faster pipe on this part: it is 0.91-0.95x the FMA pipes.** Feeding it from
threadgroup memory is free, so the units themselves are not the constraint. (The absolute numbers
moved by ~2x between this session and `#44`'s 15.2-15.7 TFLOP/s for the same FMA kernel -- 8.58
against 15.74 in a run whose load average was 2.2 -- which is why only the ratio is quoted here.
Both arms were measured in the same process, in the same run, back to back.)

That fixes the ceiling before any kernel exists. An 8x8 score or value tile needs 64 dimensions of
reduction per 8x8 output, and a fragment accumulator cannot be indexed -- so a raised row maximum
cannot rescale it the way flash attention does, and the keys have to be walked **twice** (pass one
for the row maximum and denominator, pass two for the weights and the fold). That is 24 products per
tile where 16 suffice: **1.5x the multiply-adds at 0.95x the rate, i.e. a hard ceiling of 0.63x the
shipped form**, whichever way the kernel is written.

### What the kernel actually measured

`dsv41_vision_attention_mma_f32` is that form: one SIMD group per (head, 8-query tile), 8x8 f32
fragments for both products, lanes 0-7 carrying a query row's running maximum and sum in registers,
`simdgroup_load`'s transpose flag turning the staged `[key][dim]` tile into the `[dim][key]` the
score product needs. Two knobs were swept at the published geometry (16 heads of 64, 1,521 and 8,649
patch tokens, best of three, app's own `-O` build, load average 2.94):

| arm | 1,521 tokens | 8,649 tokens |
|---|---|---|
| `pairedFold` (shipped) | **4.65 ms/block** | **190.39 ms/block** |
| staged tiles, 1 group (32 lanes) | 14.97 ms (0.31x) | 443.94 ms (0.43x) |
| direct fragment reads, 1 group | 11.72 ms (0.40x) | 343.41 ms (0.55x) |
| staged, 4 groups (128 lanes) | 13.04 ms (0.36x) | 364.90 ms (0.52x) |
| direct, 4 groups | 11.82 ms (0.39x) | 339.38 ms (0.56x) |
| staged, 8 groups (256 lanes) | 12.09 ms (0.38x) | 358.97 ms (0.53x) |
| direct, 8 groups | 12.46 ms (0.37x) | 348.31 ms (0.55x) |

Every arm agrees with the host rule (worst 1.73e-6 at the smallest image, 2.98e-6 at the largest), so
the negatives are measurements of the right rule. An earlier run with only the staged arm landed at
0.32x / 0.44x, which is the same story.

- **The form loses everywhere, and the best arm is 84% of its own arithmetic ceiling** (0.56x against
  the 0.63-0.67x the product count allows). So the kernel is not badly written; the arithmetic it
  multiplies by is against it.
- **Per multiply-add, the matrix unit is 0.84x the hand-written fold** (0.56 x 1.5): even after
  deleting the fold's shuffle, its 64-bit addressing and its loads, a matrix product does LESS per
  MAC than the shipped kernel does. That is the finding, and it kills the variant that would have
  avoided the second pass: a one-pass MMA using a diagonal-fragment rescale does the same products at
  the same rate, so its ceiling is parity and its measured per-MAC rate puts it at ~0.84x.
- **Staging is a 10-20% tax, not the reason it loses.** Reading fragments straight out of the plane
  beats staging the tile and sharing it: 0.55x against 0.43x at the largest image with one group.
  Staging pays (device read -> threadgroup write -> threadgroup read) for the same bytes and buys the
  next group's reuse; at 4-8 groups that reuse recovers only 0.09x (0.43 -> 0.52).
- **Group width helps the staged arm and does nothing for the direct one** (0.55 -> 0.56 -> 0.55
  across 1, 4 and 8 groups), which is what sharing a staged tile between groups should look like.
- **The smallest image is where the form is worst** (0.40x against 0.55x). Fewer keys per query tile
  means the once-per-threadgroup query staging and the two passes over a short key row are a larger
  share of the work.

### What it means

- **Do not build a fifth SIMT shape, and do not build an MMA one either.** `#44`'s "the next lever is
  the matrix units" is now falsified by measurement, and the falsification is arithmetic rather than
  a bad kernel: same rate per pipe, 1.5x the products, and a per-MAC deficit even when the pass count
  is free.
- **The shipped `pairedFold` stays the default** and is untouched. `.matrixTiles` joins
  `.stagedKeyTile` as a measured negative kept in `AttentionForm`, with its numbers in its own doc
  comment: the two knobs are `threadgroupWidth` (SIMD groups sharing a staged tile) and
  `stagesTiles` (stage a tile, or read fragments straight from the plane).
- **The attention's 10% of peak is not an instruction-issue defect that a bigger instruction fixes.**
  What is left for that stage is fewer bytes or fewer products -- and the byte hypotheses are already
  dead (`#17`'s record halving, `#44`'s load halving, both ~1.0x), which leaves the algorithm: a
  query tile that shares one head's key rows with another head's, or a cheaper score rule.
- **Planes now carry one zeroed tile of tail.** A direct `simdgroup_load` reads 8 rows at a time, so
  the last partial tile of the last head runs past `tokens * heads * headDim`. `allocatePlanes`
  allocates one extra 8-row tile (zeroed, `matrixTile * headDim` floats) and the addressing formula is
  unchanged, so every other form and the rope split are unaffected.

### Verification

The default suite gains `testTheMatrixTileAttentionMatchesTheDirectKernelAndTheHost` (heads 3/dim 8
at 37 tokens, 2/64 at 5, 4/32 at 33, and 2/64 at 16 -- a partial last key tile, a single-tile prompt,
a single-fragment head and the published head, worst **8.2e-7** against the direct kernel and
**4.2e-7** against the serial host rule) and `testTheMatrixTileAttentionRefusesAHeadThatIsNotAFragment`
(dim 12 is refused by the tile form and still runs on every SIMT form). Two opt-in cases join the
published-geometry harness behind `/tmp/dynamoe_dsv41_vision_measure`:
`testWhatThisGPUsFP32MatrixUnitsReach` and `testWhatTheMatrixTileAttentionCostsAtThePublishedGeometry`
(seven arms, every timed arm's rows checked against the host before it prints).

The full DeepSeek suite on this code is **560 passed / 0 failed / 57 skipped** (617 executed, 94
classes), and the same run under `MTL_DEBUG_LAYER` is **560 / 0 / 57 with 0 validation errors** and no
failed assertions -- which a new kernel and a changed plane allocation have to survive before they are
trusted.

## DSV4.1 #46: THE LARGEST IMAGE COSTS FIVE MINUTES, NOT SEVEN SECONDS, AND THE PREFILL IS 98% OF IT

### Question

Routing picked this slice at 0.963 with 0.957 confidence, and it was the largest unmeasured number
left in the port: what does the biggest picture the composer accepts actually cost the person who
attached it? The tower's own cost is published (`#43`: 6.74 s for the largest admissible span) and
every mechanical lever on its attention is measured dead (`#44`, `#45`), so the open question was not
the tower at all -- it was the rest of the turn.

### What the largest picture actually costs

The composer's path is `price` (no weight read) -> provision the cache for the EXPANDED prompt ->
`runtime.imageSpan` (the device tower) -> `TurnImages.bind` -> the turn loop. A 4,096x4,096 image is
the largest `ImagePixels.decode` lets through (one side above that is refused before the tower sees
it), and every side from there up collapses onto the checkpoint's own token budget, so it is the
worst case an attachment can reach:

| stage | measured |
|---|---|
| resize plan | 8,649 patch tokens, 31x31 aligner grid, **994 span positions**, prompt 12 -> 1,005 tokens |
| price | **6.95 s** estimated, in 0.0001 s (no weight read) |
| encode (device, cold process: weight upload included) | **7.23 s** |
| encode (device, repeated) | **6.88 s** |
| prefill of the 1,005-position prompt | **255.98 / 268.05 / 297.98 s** in three runs (254.7 / 266.7 / 296.5 ms/position, 218,886 MB of experts, 95% hit) |
| first generated token | 0.00 s |
| **to the first visible token** | **263.29 / 275.28 / 305.20 s** |

**The encode is 2.4-2.6% of the wait; the prefill of the span is 97-98%.** A reader who trusts the
published 6.74 s would predict a seven-second picture and be wrong by a factor of 39-45.

### Why the prefill walked

`DeepSeekV41TurnRunner` hands a whole prompt to `prefillChunk` only when every position is a token,
so an image prompt was WALKED: one full layer pass and one expert set per position. Two separate
reasons, and the second one was a latent bug:

- the chunk form had **no way to take a supplied row** (`prefillChunk(_ tokens: [Int])`);
- the dead flag was a **single pending mark** per call, and the block's Engram stage gated on
  `deadTokens.last` -- the block's LAST position. A block that mixes image and text positions could
  not say which position was dead, so the whole block inherited one position's answer.

### The batched image prefill

- `DeepSeekV41PositionForward.prefillChunk(_:embeddings:)`, with a protocol default that WALKS. A
  partial executor owes correct numbers here rather than a refusal, because the walk and the block are
  the same graph.
- Device: each block gets only its own rows; the host copy happens BEFORE the block's command buffer
  exists (a shared buffer written from the host is not ordered against work that has not executed, and
  here the copy simply precedes every dispatch that reads the slot); a supplied row REPLACES the token
  lookup instead of adding to it; a row that is out of range or the wrong width is refused.
- Dead flags are read off the row map, one per position: **a supplied residual row IS an image
  position and takes no part in an n-gram**.
- `encodeEngram` gained the per-position dead test (`position: Int?`), which is the fix for the latent
  bug above: a walk names none and reads `deadTokens.last`, a block names each position it carries.
- The CPU oracle's `prefillChunk(_:embeddings:)` replaces `embed(token)` with the row and derives the
  dead flag the same way, so the device has a reference that is not itself.

### The latent Engram bug, caught by the parity test

`testABatchedImagePrefillMatchesTheWalkOnThePublishedCheckpoint` compares a two-block image prefill
with a walk of the same positions on the checkpoint's own weights and tower rows. With the block-level
Engram gate it measured **2.81e-01 relative** at a logit scale of 22.0 -- the live text positions at
the front of the span lost their memory contribution because the block's last position was an image
position. With the gate moved per position the same case measures **1.08e-04**. The synthetic fixture
could not have found it: it publishes no Engram layers at all.

### Where batching pays, and where it does not

Same runtime, same prompt, one arm each, arena reset before every arm, `block`/`per-position` counting
which mixture layout actually ran (5,040 block, 0 per-position in every arm below). The prompt is the
span alone at four lengths:

| positions | batched | walk | ratio | expert bytes (batched / walk) | hit rate |
|---|---|---|---|---|---|
| 16 | 2,180.8 ms (136.3 ms/pos) | 1,656.3 ms (103.5 ms/pos) | 0.76x | 26,052 / 26,016 MB | 21% / 62% |
| 128 | 9,220.9 ms (72.0 ms/pos) | 17,731.9 ms (138.5 ms/pos) | **1.92x** | 47,836 / 47,514 MB | 78% / 91% |
| 512 | 47,184.0 ms (92.2 ms/pos) | 76,360.7 ms (149.1 ms/pos) | **1.62x** | 74,211 / 73,691 MB | 92% / 97% |
| 997 | 121,980.2 ms (122.3 ms/pos) | 153,242.6 ms (153.7 ms/pos) | **1.26x** | 99,546 / 99,062 MB | 95% / 98% |

The sixteen-position arm is the outlier in the other direction and it is not a result: the first arm of
a process pays the arena's slot allocations and the page cache. The sweep warms up on a discarded
eight-position chunk for exactly that reason, and 16 still reads 0.76x because it is next.

Then the composer's own largest image, same protocol:

| prompt | batched | walk (shipped) | ratio | expert bytes | hit rate |
|---|---|---|---|---|---|
| 1,005 positions, 4,096x4,096 pixels | 345.62 / 355.22 / 386.90 s | **255.98 / 268.05 / 278.38 s** | **0.75x / 0.74x / 0.72x** | 218,312 / 218,886 MB | 91% / 95% |

So the same span is 1.26x faster batched at 997 positions and 0.74x faster walked at 1,005 -- and the
difference is not the length, it is the **working set**. The bare-span prompt reads 99,546 MB over 997
positions; the composer's prompt reads 218,886 MB over the same span, i.e. it touches 2.2x as many
distinct `(layer, expert)` pairs (a different picture and eight extra text positions at the front move
the routing). Against an arena of 3,760 slots the ratio tracks the working set monotonically:

| distinct pairs touched | working set vs arena | ratio |
|---|---|---|
| 2,660 | 0.71x | 1.92x |
| 4,118 | 1.10x | 1.62x |
| 5,533 | 1.47x | 1.26x |
| 12,228 | 3.25x | 0.74x |

That is the mechanism: when the working set fits, the block form replaces eight per-position fills
with one and wins; when it does not, EVERY fill evicts, and a union fill evicts 20-48 pairs at once
where the walk's 6-pair fill keeps what it just used -- the hit rate falls from 95% to 91% and the
bulk eviction costs more than the round trips save.

### What it means

- **The walk stays the shipped path for an image prompt**, and `DeepSeekV41TurnRunner` says so with
  this table in its comment. `testAnImageRowRunsOnlyAtItsOwnPosition` pins the decision, and the
  end-to-end case asserts the batched form is SLOWER at the largest span, so a later change to either
  has to re-measure.
- **The batched image prefill stays, as a capability with a number.** It is correct (1.08e-04 against
  the walk on real weights), it is 1.26-1.92x faster while the span's expert working set fits the
  arena, and the entry point is one call. What it needs next is not more batching but a fill policy
  that does not discard the locality the walk had -- which is a targeted-invalidation question for
  `DeepSeekV41ExpertArena`, not a prefill-layout one.
- **The composer's estimate is right about the wrong thing.** `DeepSeekV41VisionCostModel.measured`
  prices the TOWER (6.95 s predicted, 6.88-7.23 s measured), and the tower is 2% of the wait. Anything
  user-facing that quotes a picture's cost from that model is quoting the encode.
- **The 6.74 s encode is not the problem, and neither is anything `#44`/`#45` measured.** The two
  levers that would move this number are the expert fill's behaviour above the arena's capacity and
  the per-position work of a long prompt -- both outside the vision tower entirely.

### Verification

The default suite gains `testABatchedImageSpanMatchesTheWalk` (one block, rows at every other
position, against the host's own `forwardEmbedding` walk AND against the host's chunk form, so a bug
in the oracle's new entry point cannot cancel a bug in the device's) and
`testABatchedImageSpanMapsItsRowsIntoTheRightBlocks` (two blocks, rows only from the second, which a
per-block slice that dropped its offset would answer with the block's first slot). Both run on the
synthetic checkpoint; `DeepSeekV41TurnLoopTests` is unchanged in shape and pins that an image prompt
still walks.

The published-checkpoint cases are opt-in with the rest of that file:
`testABatchedImagePrefillMatchesTheWalkOnThePublishedCheckpoint` (16 positions, real tower rows, the
Engram bug above), `testWhatBatchingAnImagePrefillSavesAtLength` (the four-length sweep, gated on
`/tmp/dynamoe_dsv41_vision_max_turn`) and `testWhatTheLargestImageCostsEndToEnd` (gated on the same
marker, ~10 minutes at `-O`).

The end-to-end numbers above are the app's own `-O` build (`/tmp/dd_dsv41_opt`), load average
3.2-4.8, and the two arms of every pair were measured in one process back to back.

The full DeepSeek suite on this code is **562 passed / 0 failed / 60 skipped** (622 executed, 94
classes), and the same run under `MTL_DEBUG_LAYER` is **562 / 0 / 60 with 0 validation errors** and no
failed assertions. That run matters more than usual here: a per-position Engram gate and a host write
into a shared buffer before a command buffer exists are exactly the two changes a validation layer
catches and nothing else does.

## DSV4.1 #47: A WALKED POSITION IS 200 ms OF IDLE GPU, AND THE EXPERT STREAM IS ONLY A FIFTH OF IT

### Question

`#46` established the number a person actually waits for -- the largest picture the composer accepts
costs 255.98-297.98 s of prefill, i.e. 254-297 ms per position -- and named the arena's fill policy as
the lever. That name was an inference from a byte count, not an observation: 218,886 MB at the disk's
13-14 GB/s ceiling is seventeen seconds, so 250 of those 255 milliseconds per position had to be
somewhere else. Routing picked this slice at 0.900 (confidence 0.886) to find out where, and closed the
shared path to behavioural change for the round (0.43): measure first.

### What a walked position costs, in buckets

Four arms, one runtime, one process, each from a reset arena and a cleared accumulator, each printing
its own bytes, misses and evictions. The prompt is the composer's own (12 text tokens then a
994-position span, 1,005 positions), rows from the device tower on a 4,096x4,096 picture, app's `-O`
build, load average 3.2-3.4:

| arm | positions | ms/position | MB read | hit | misses | evictions | pread | front encode | front wait | indexer wait | fill | mix encode | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| the shipped walk, real image rows | 1,005 | **269.4** | 218,886 | 94.9% | 12,208 | 8,448 | 57.4 s (3,812 MB/s) | 2.8 | 160.6 | 39.3 | 57.5 | 3.0 | 37.5+31.2 |
| the same arm again | 1,005 | **275.7** | 218,886 | 94.9% | 12,208 | 8,448 | 60.6 s (3,615 MB/s) | 2.7 | 162.8 | 40.2 | 60.6 | 3.0 | 37.3+31.3 |
| the same length, TEXT tokens | 1,005 | **228.2** | 111,415 | 97.4% | 6,214 | 2,454 | 30.2 s (3,690 MB/s) | 3.6 | 144.6 | 39.6 | 30.3 | 2.9 | 37.4+30.5 |
| `#46`'s sweep fixture, synthetic rows | 997 | **149.0** | 99,062 | 97.7% | 5,525 | 1,765 | 25.7 s (3,860 MB/s) | 2.4 | 92.8 | 21.0 | 25.9 | 2.8 | 33.2+27.1 |

An earlier run of the same four arms, before the indexer split was separated into its own bucket, read
244.7 / 257.0 / 194.9 / 139.5 ms, so quote the ranges: **245-276 ms for the real span, 195-228 ms for
a text walk of the same length, 140-149 ms for the synthetic fixture.** The byte columns are identical
across runs to the megabyte, because routing is a deterministic function of the activations.

Three facts fall out of the table before anything else:

- **The host's own encoding is 2.4-3.6 ms per position for all forty layers** (about 70 microseconds a
  layer). Every "fewer dispatches per layer" argument in this port's history is arguing over 1% of a
  position.
- **The expert fill is 21% of a walked position** (57.5 of 269.4), and the indexer readback another
  15% (39.3, i.e. 4.9 ms on each of the eight compressed-index layers).
- **A same-length text walk costs 195-228 ms with the identical shape**, so an image span is 1.18-1.26x
  a text walk of its own length and nothing more. The difference is the real tower rows' routing
  breadth: 12,208 misses against 6,214 at the same position count.

### Where the idle actually sits

Buckets say how long the host blocked, not what the GPU was doing, so `recordsWalkTimeline` now keeps
each committed buffer's commit time, the moment the host began waiting on it, and its own device
window. A steady-state position (400 of a walked text run, i.e. after warm-up) is 88 committed buffers:
40 front halves, 40 mixtures and 8 indexer splits.

| | position 400 | position 401 |
|---|---|---|
| wall | 297.5 ms | 224.3 ms |
| device windows | 94.3 ms (32%) | **68.0 ms (30%)** |
| host blocked in `waitUntilCompleted` | 205.5 ms (69%) | 148.7 ms (66%) |
| GPU idle between buffers | 200.3 ms | 153.8 ms |
| commit-to-start, summed | 257.2 ms | 184.4 ms |
| commit-to-start, per buffer | 2.92 ms | **2.10 ms** |

**A committed buffer's device window opens 2.1-2.9 ms after `commit()` returns, while the GPU has
nothing else to do.** The first ten buffers of position 400, in order:

```
layer 0 attention: committed  0.00 ms, started at  0.02 for 1.14 ms, host waited 1.141 ms
layer 0 experts:   committed 14.40 ms, started at 16.26 for 0.95 ms, host waited 2.270 ms
layer 1 attention: committed 15.02 ms, started at 17.21 for 1.43 ms, host waited 1.408 ms
layer 1 experts:   committed 18.78 ms, started at 22.96 for 0.96 ms, host waited 5.184 ms
layer 2 indexer:   committed 18.81 ms, started at 23.93 for 0.61 ms, host waited 0.614 ms
layer 2 attention: committed 24.70 ms, started at 24.79 for 0.71 ms, host waited 0.842 ms
layer 2 experts:   committed 25.61 ms, started at 29.61 for 0.89 ms, host waited 4.928 ms
layer 3 attention: committed 25.66 ms, started at 30.50 for 0.98 ms, host waited 0.953 ms
layer 3 experts:   committed 37.64 ms, started at 38.38 for 1.12 ms, host waited 1.878 ms
layer 4 attention: committed 37.70 ms, started at 39.51 for 1.58 ms, host waited 1.577 ms
```

Layer 1's mixture is handed to the driver at 18.78 ms, the GPU is idle (layer 1's front window ended
at 18.64), and it starts at 22.96: **4.2 ms for a buffer holding one layer's mixture kernels.** That is
not the fill -- the gap between that front's window ending and the mixture's commit is 0.14 ms, against
the arm's 1.4 ms a layer average -- and it is not the host, whose entire front encode for the layer is
30 microseconds.

So a walked position decomposes as, per position at the real span:

| | ms | share |
|---|---|---|
| device windows (front + mixture) | 68.7 | 26% |
| GPU idle inside a wait, between buffers | ~131 | ~49% |
| GPU idle during the expert fill | 57.5 | 21% |
| host encoding (both halves) | 5.8 | 2% |

**Seventy percent of a walked position the GPU does nothing at all**, and only a fifth of that idle is
the disk.

### Why `#46`'s sweep table was twice as fast as the composer's turn

The sweep fixture reproduces `#46` exactly: 99,062 MB read (the same megabyte count), 97.7% hit
against 98%, 140-149 ms/position against 154. It was a real measurement of a real graph on real
weights. What it was not is a measurement of a real picture: its rows come from
`imageSpan(patches:)` over a deterministic sawtooth, whose routing is 5,525 distinct pairs where the
tower's own rows are 12,208. Both touch a 3,760-slot arena that cannot hold either working set, so the
fixture's misses (and therefore its fill) are 45% of the real span's. That is why the sweep's
`1.26x`-`1.92x` block-vs-walk ratios and the composer's `0.74x` are both true statements about
different working sets, and why the composer's number is the one that describes a picture.

### What it means (and what is not the lever)

- **The lever is the number of command buffers per position, not the expert stream.** 88 buffers at
  2.1-2.9 ms of startup each is 184-257 ms, against 63-94 ms of actual device work. Nothing in the
  kernels touches this; it is a property of how many submissions a position is cut into.
- **The fill is real but second-order**: 57.5 ms, and a fifth of the position. `#46`'s "the next lever
  is the arena's fill policy" was directionally right and quantitatively wrong -- halving the misses
  would buy at most ~29 ms and would not touch the 131 ms of restart idle.
- **A surgical, unmeasured candidate**: `runLayer` commits the mixture of layer L and then the front of
  layer L+1 as two buffers, with nothing between them but 30 microseconds of encoding. Merging them
  into one submission halves the buffer count (the eight compressed-index layers keep an extra split),
  and the two halves are adjacent in time anyway. That is a pure host-side change, it needs no new
  kernel, and by this table it addresses half of the largest term.
- **Batching already removes the restart term entirely and still loses.** The chunk form runs 5,040
  mixture submissions for the same 1,005 positions -- about 5 buffers a position -- so the 184-257 ms
  of restarts is nearly gone, and yet it measured 345.62-386.90 s against the walk's 255.98-297.98 s.
  Whatever the block form's own cost is, it is therefore at least the restart saving plus its 84 ms
  deficit: **~215 ms/position that `#47` has not explained.** That, not the fill, is where the next
  measurement belongs.
- **The vision cost model still prices 2% of a turn** (`#46`), and this slice does not change that: an
  image attachment's user-visible cost is the walk of its span, and the walk is CPU/GPU hand-off.

### Verification

Two new opt-in published-checkpoint cases (`/tmp/dynamoe_dsv41_walk_cost`, on top of that class's
`/tmp/dynamoe_dsv41_real_turn` gate): `testWhereAWalkedImagePrefillSpendsItsTime` (the four arms above,
943 s at `-O`) and `testWhereAWalkedPositionsIdleTimeSits` (the timeline at positions 400-401, 101 s).
The instruments themselves are pinned on the synthetic checkpoint by
`testTheWalkInstrumentCoversEveryBufferOfAPosition`, which asserts that a position produces exactly two
timeline entries per layer in commit order, that no window starts before its commit, that the device
total cannot exceed the wall, and that the new indexer bucket is zero on a shape with no compressed
index source and non-zero on one that has it -- because both instruments are off in every production
path, and a bucket that silently stops covering part of a position would still print.

The full DeepSeek suite on this code is **563 passed / 0 failed / 62 skipped** (94 classes; one new
case plus the two new opt-in published ones, which skip without their markers), and the same run under
`MTL_DEBUG_LAYER` is **563 / 0 / 62 with 0 validation errors and no failed assertions**. The `-O` arms
are the app's own optimized build in `/tmp/dd_dsv41_opt`, load average 3.2-3.4; the same arms read
245-276 ms under lighter load, which is why every number in this section is a range.

One pre-existing warning is untouched: `DeepSeekV41IndexerInputRequest.latents` holds an `MTLBuffer` in
a `Sendable` struct, which the Swift 6 language mode will make an error. It is not from this slice.

## DSV4.1 #48: A SUBMITTED BUFFER STARTS IN A TENTH OF A MILLISECOND, SO THE WALK'S IDLE IS NOT ITS SUBMISSION COUNT

### Question

`#47` measured a walked position at 245-276 ms with the GPU executing 63-94 ms of it, attributed the
difference to `88 command buffers per position, each starting 2.1-2.9 ms after it was handed over`, and
named the submission count as the lever -- with the surgical candidate being a merge of layer L's
mixture buffer into layer L+1's front buffer. Before anything was changed, the question this slice had
to answer was whether a committed buffer really costs 2-3 ms to start, and if so what it is
proportional to: a fixed cost per BUFFER (merge them), a cost per DISPATCH (cut dispatches), or a cost
of the QUEUE HAVING GONE EMPTY (keep more than one buffer in flight). Those have three different
fixes and the walk's own numbers could not tell them apart.

Two instruments answer it. `DeepSeekV41CommitPickupCostTests` drives command buffers of chosen shapes
through one queue and splits each one's time four ways; the published timeline case sums the same
split by boundary class over a real position.

### What the walk's idle actually is

`#47`'s per-buffer figure was `commit -> the buffer's own device window opening`, and that is not idle:
a buffer committed while another is still queued "pays" the predecessor's remaining window. The
published case now reports both, so the same two positions read:

| | position 400 | position 401 |
|---|---|---|
| wall | 232.2 ms | 218.3 ms |
| device windows (88 buffers) | 67.3 ms (29%) | 66.3 ms (30%) |
| host blocked in `waitUntilCompleted` | 162.2 ms (70%) | 154.6 ms (71%) |
| **idle with nothing queued** | **93.7 ms (40%)** | **87.4 ms (40%)** |
| commit-to-start, summed over 88 buffers | 212.9 ms (2.42/buffer) | 197.8 ms (2.25/buffer) |
| idle into the front half (40 boundaries) | 2.2 ms (0.05/buffer) | 2.9 ms (0.07/buffer) |
| idle into the indexer split (8) | 0.0 ms | 0.0 ms |
| **idle into the mixture half (40)** | **91.5 ms (2.29/buffer)** | **84.5 ms (2.11/buffer)** |
| queue wait, front half | 94.2 ms (2.36/buffer) | 98.2 ms (2.46/buffer) |
| host gap before the mixture commit | 0.15-0.21 ms/layer | 0.15-0.21 ms/layer |

So the position has 87-94 ms of genuine idle, **97% of it in one gap per layer, immediately before the
expert half starts**, and the host is already finished when that gap opens: it commits the mixture
0.2 ms after the front half retires and then blocks on it. The 2.36 ms of "commit to start" the front
half reports is queue wait behind that same mixture, i.e. real GPU work, not idle.

That is the correction `#47` needs, and it changes what a fix has to do: there are not 88 late starts,
there are 40, and they are all in the same place. (The timeline covers the layers' two buffers each;
the embedding and output-head commits are not in `walkBuffers`, so the device column understates a
position by those two submissions and would read a few milliseconds higher with them in.)

### What a submission costs to start, across every shape a probe can hold

Medians, `-O`-free Debug test build, load average 3.2-4.0, `gpuStartTime` against the mach clock the
commit stamp is taken from:

| arm | pickup | window |
|---|---|---|
| 1 launch, 1 small resource, queue idle | 0.080 ms | 0.002 ms |
| the same, with no `scheduled` handler registered (instrument control) | 0.083 ms | 0.004 ms |
| the same, after 2 ms of host idle | 0.071 ms | 0.005 ms |
| the same, after 5 / 10 / 20 / 50 ms of host idle | 0.682 / 0.572 / 0.768 / 0.843 ms | 0.011 ms |
| 8 / 32 / 128 launches in one buffer | 0.166 / 0.668 / 0.561 ms | 0.032 / 0.119 / 0.455 ms |
| one launch over a 1,048,576-lane grid | 0.134 ms | 0.020 ms |
| 64 launches over 64 resident 64 MiB resources | 0.470 ms | 0.236 ms |
| 64 launches over 8 resident 512 MiB resources (same 4 GiB) | 0.495 ms | 0.238 ms |
| 64 launches over 64 zero-copy mapped 64 MiB resources | 0.088 ms | 0.221 ms |
| 20 launches naming 0 / 8 / 64 / 128 / 256 / 512 resources | 0.110 / 0.087 / 0.086 / 0.110 / 0.136 / 0.173 ms | 0.07 -> 0.04 ms |
| 20 launches, 128 resources, a buffer barrier after each launch | 0.101 ms | 0.043 ms |
| queue fed with 2 / 4 / 8 buffers before any wait: follower's gap from its predecessor's end | **0.001 ms** | |
| a co-tenant saturating the GPU on another queue, 2 buffers at a time | **0.009 ms** (0.078 alone) | |
| the walk's own alternation: 40 layers of front-commit-and-wait, host work, mixture committed and left in flight (80 buffers) | idle 6.6 ms of 12.3 (0.08/boundary) | |
| ... with 128 MB of host writes into the mixture's own buffers every layer | pickup 0.088 ms (the added idle is the host's own `memset`) | |
| ... with a different pipeline for the mixture than for the front | idle 8.5 ms of 15.3 (0.11/boundary) | |
| ... (one merged pattern) all 80 buffers queued before any is waited on | idle 0.6 ms of 6.4 | |
| a 64 MB read of memory the host had just written, every page touched | 0.096 ms (0.110 untouched) | 0.316 ms |

Nothing reproduces 2.3 ms. The delay is not the submission's shape (launch count 1-128, grids to a
million lanes, 2 GiB of bound memory, 512 named resources, buffer barriers), not its size, not the
pipeline switch between halves, not the host's own writes into the memory the next submission reads,
not the queue having gone empty (a fed queue starts a follower 1 microsecond after its predecessor
retires), and not the machine state a 402-position walk leaves behind: **the same probe run in the
walk's process, after the walk, still reads 0.053-0.073 ms.**

### What it is not, and what it means

- **The submission count is not the lever, and merging buffers would not buy the idle.** The 2.3 ms is
  one gap per layer, before the expert half. Merging layer L's mixture into layer L+1's front would
  remove one of the two submissions per layer and turn the front half's 2.36 ms of queue wait into
  work inside a longer buffer -- but the merged buffer would still start where the mixture starts, so
  the 91.5 ms of idle stays. `#47`'s surgical candidate is measured dead before it was built.
- **The one lever the probe does confirm is keeping the queue fed**: a follower committed before its
  predecessor retires starts 1 microsecond later, i.e. 2000x sooner than an isolated submission's
  0.08 ms and 2000x sooner than the walk's 2.3 ms. Whether the walk can be arranged that way is a
  different question from how many buffers it commits, but it is the shape with a measured payoff.
- **What is left unexplained is specific and narrow**: one submission per layer, the expert half, is
  the only submission in the port that (a) follows a `pread` into the memory it reads and (b) runs the
  expert kernels. Both were held constant in a probe and neither moved the number, at 128 MB of write
  and 64 MB of read. The next instrument therefore has to be inside the walk rather than beside it:
  a GPU-side split of the buffer's own start (`MTLCounterSampleBuffer`, or the first launch's
  timestamp) plus the layer's fill/eviction counts, so the 40 gaps can be correlated with what the
  layer did -- 40 samples, one per layer, is a correlation the timeline can already carry.

### Verification

`DeepSeekV41CommitPickupCostTests` (seven cases, all default-suite, all on the GPU with no model): the
pickup decomposition with the `scheduled`-handler control and the idle-gap curve, the queue-depth
sweep with its in-order invariant, the dispatch/grid/residency sweep, the co-tenant arm, the walk's
alternation with its host-write and pipeline variants, and the read-after-host-write arm. Each asserts
its sample count, that every buffer opened a window, that `drain >= pickup + window` (the one
structural relation the clocks guarantee), that a follower in a one-queue round never starts before its
predecessor finished, and that the `scheduled` handler fired for every completed buffer -- because the
first version of `#47`'s timeline stamped `waitEnd` before the wait and reported a clean 0.000 ms.

The published case grew the boundary attribution (idle, queue wait and window per class, plus a
twenty-boundary sequence) and the post-walk probe arm; it is still opt-in behind
`/tmp/dynamoe_dsv41_walk_cost` on top of the class's `/tmp/dynamoe_dsv41_real_turn`.

The full DeepSeek suite on this code is **570 passed / 0 failed / 62 skipped** (95 classes), and the
same run under `MTL_DEBUG_LAYER` is **570 / 0 / 62 with 0 validation errors and no failed assertions**.
The published numbers are the app's own `-O` build in `/tmp/dd_dsv41_opt` (positions 400-401, 102 s),
load average 3.2-3.6; the probe arms are Debug and run in under 3 s together.

Nothing in the production path changed: no kernel, no encoder, no submission pattern.

## DSV4.1 #49: THE GAP BEFORE A LAYER'S EXPERT HALF IS THE WALK'S I/O, AND IT COSTS A SECOND READING OF THE SAME BYTES

### Question

`#48` left the port with exactly one unexplained number. A walked position's genuine idle --
the GPU with nothing queued -- was 87-94 ms of a 218-232 ms position, and 97% of it sat at a
single boundary per layer: the hand-off from the layer's dense half to its expert half, 2.11-
2.29 ms across 40 boundaries, with the host already finished. A probe beside the walk could not
reproduce it in any shape it could hold (1-128 launches, a million-lane grid, 2 GiB of bound
memory, 512 named resources, a barrier per launch, zero-copy mapped memory, a 64 MB read of what
the host had just written, the walk's own alternation, a co-tenant saturating the GPU, and the
same probe inside this process after 402 positions): every arm started in a tenth of a
millisecond. The one thing a probe beside the walk cannot reproduce is the LAYER.

So this slice instrumented the layer. Three instruments, none of them beside the walk:

1. `DeepSeekV41MetalForward.recordsLayerGaps` -- one `DeepSeekV41LayerGapSample` per layer, closed
   when that layer's expert half is drained by the next layer's dense half, carrying the dense
   half's window end, the expert half's commit stamp and window, and the layer's own fill numbers
   (bytes, misses, slots allocated, pread seconds, placement seconds) plus its routing read and
   its encode.
2. A zero-byte-fill arm: the same 40 boundaries and the same submissions over an eight-token
   cycle, which leaves the arena holding every pair the walk asks for, so the fill reads nothing.
3. A one-dispatch command buffer committed immediately before each expert half, carrying its own
   commit stamp, which separates "the queue will not start work" from "this buffer will not
   start".

A fourth instrument -- reading every page of the fill back on the committing thread, to move the
dirty lines to the core that commits -- was built, measured, and WITHDRAWN (below).

### The gap tracks how much the walk read, and nothing else about the layer

Four fill levels, one runtime, one process, one `-O` build (`/tmp/dd_dsv41_opt`), each arm
recording one steady-state position's 40 boundaries. `idle` is the GPU with nothing queued, from
the later of the dense half's retirement and the expert half's commit:

| arm | fill / layer | idle mean | idle median | idle total over 40 layers | host gap mean |
|---|---|---|---|---|---|
| zero-byte fill (8-token cycle) | 0.0 MB | **0.145 ms** | **0.091 ms** | 5.8 ms | 0.167 ms |
| partial fill (96-token cycle) | 7.2 MB | 1.343 ms | 0.932 ms | 53.7 ms | 1.381 ms |
| full fill, plain A | 13.9 MB | 2.146 ms | 1.318 ms | 85.9 ms | 2.581 ms |
| full fill, plain B | 13.9 MB | 2.314 ms | 1.410 ms | 92.6 ms | 2.357 ms |

Two points can only say "the gap needs I/O". Three say it scales with it, and the fit is
uncomfortably tight to the disk: the median slope is **0.088-0.117 ms of GPU idle per MB read**,
and one MB off a 13.8 GB/s disk is 0.0725 ms. So the hand-off after a fill idles for **1.2-1.6x
the time the bytes themselves took**, and the mean slope (0.144-0.166 ms/MB) says up to 2.3x
because the distribution is spiky rather than smooth.

What the layer itself did barely matters. Within a full-fill arm, 19 of the 40 layers read
NOTHING -- their pairs were resident -- and they still idle 1.56-1.65 ms each, against 0.091 ms
for the same boundary with no I/O anywhere in the walk. The correlation between a layer's own
fill and its gap is +0.24. What predicts the gap is that the WALK is reading, not that this LAYER
read.

### It is not the submission, and it is not the writeback

The probe settles the first half. A one-dispatch clear carrying 16 bytes, committed into the same
empty queue a few hundred microseconds before the expert half, starts in **0.1019 ms** (max 0.645)
while the expert half takes **1.573 ms** in the same position; in 0 of 40 layers did the probe
still be running when the expert half started. `#48` was right that the queue is willing: a
trivial buffer committed at that exact moment starts at once. What does not start at once is the
expert half itself -- and no bound resource, grid, barrier or memory kind makes a buffer slow to
start (`#48`), so what is left is the state the machine is in, not the shape of the work.

The writeback hypothesis was the obvious second suspect: the fill writes from
`DispatchQueue.concurrentPerform`'s worker threads, so the lines the expert half reads are dirty
in other cores' caches, and `#21` had already learned that `F_NOCACHE` on these descriptors is a
2.2x LOSS. `DeepSeekV41ExpertStreamer.touchesFilledRanges` reads every page of a fill back on the
calling thread, and the arms were run as an ALTERNATING PAIR (off, on, off, on) on the same
tokens and a warm page cache, because the first attempt at this comparison was confounded: a
read-back arm that ran fourth looked 39% better than a cold arm that ran first, and most of that
was the page cache, not the read-back.

| paired arm | idle mean | idle median |
|---|---|---|
| plain A | 2.146 ms | 1.318 ms |
| read-back A | 2.453 ms | 1.531 ms |
| plain B | 2.314 ms | 1.410 ms |
| read-back B | 2.421 ms | 1.813 ms |

**The read-back is a measured negative.** The two read-back-off arms differ from each other by
0.168 ms, and read-back moves the mean the WRONG way by 0.207 ms (2.230 -> 2.437, 0.91x). Medians
agree: 1.318/1.410 without against 1.531/1.813 with. So the delay is not CPU cache ownership, and
the same is true of an explicit flush or fence -- do not re-propose either without a new
argument.

### What it means

The expert stream costs about **twice its own byte time**, and only half of that is the `pread`.
At this arm's fill (13.9 MB/layer, 556 MB/position) the 40 hand-offs idle 85.9-92.6 ms on top of
the host's own 94-103 ms gap, against 29.8-30.6 ms of expert-half device time and (from `#48`) a
~66 ms device total for the position. The zero-fill arm's floor is 5.8 ms, so essentially all of
that idle is the stream.

Applying the median slope to `#21`'s measured steady state of 900-950 MB/token gives **79-111 ms
of hand-off idle per position**, which is the number `#48` measured by a completely different
route (87-94 ms). So the two agree, and the consequence is a correction to `#21`'s priority:
halving the arena's misses does not buy the ~29 ms of `pread` that `#21` credited it with. It buys
that, AND a comparable amount of the idle that follows it -- roughly **78 ms of a 205-260 ms
position, or 1.5-1.6x decode**, which is the largest single lever measured on this port so far
and it is the arena's hit rate, not a kernel.

### What it is not (do not re-propose)

- **Not the submission shape.** `#48`'s twenty arms, plus this slice's probe at 0.1019 ms in the
  walk's own process.
- **Not the CPU cache writeback, and not a flush/fence.** Measured negative in an alternating
  pair with a noise estimate.
- **Not the layer's own bytes.** 19 zero-read layers in a full-fill arm idle 1.56-1.65 ms.
- **Not merging buffers.** `#48` measured that dead before it was built.
- **Not `F_NOCACHE`.** 2.2x loss, `#21`.

### Verification

`DeepSeekV41PublishedTurnTests.testWhatEachLayersExpertHalfHandoffCosts` is opt-in behind
`/tmp/dynamoe_dsv41_layer_gaps` on top of the class's `/tmp/dynamoe_dsv41_real_turn`, because it
walks the real checkpoint eight times: a discarded warm-up (so the paired arms are not the
first-read arm paying for a cold disk), the alternating read-back pair, the zero-fill arm, the
partial-fill arm, and the probe arm. It asserts every layer produced a window (`40/40 windowed`
in all eight arms), that the probe was dispatched and ran (`40 one-dispatch buffer(s)`), and it
prints each arm's per-layer rows so the correlations above can be recomputed from the log rather
than trusted.

Nothing in the production path changed: no kernel, no encoder, no submission pattern, no arena
policy. The three new flags (`recordsLayerGaps`, `recordsMixtureStartProbe`,
`touchesFilledRanges`) are off everywhere, and the clock reads they gate are inside
`commit()`/`settlePending()` behind them, so a turn pays nothing -- mechanism 7's rule from
`#47`, kept.

The full DeepSeek suite on this code is **570 passed / 0 failed / 63 skipped** (95 classes, 633
cases: `#48`'s 632 plus this slice's new opt-in case), and the same run under `MTL_DEBUG_LAYER`
is **570 / 0 / 63 with 0 validation errors and no failed assertions**. The published numbers are
the app's own `-O` build in `/tmp/dd_dsv41_opt`, eight arms of 202 positions each in ~5.5 minutes,
load average 2.6-3.8. One caveat worth carrying: the suite reads **573 / 0 / 60** whenever
`/tmp/dsv41_indexer_inputs.json` is present, because the three `DeepSeekV41RealIndexerReScareTests`
cases run instead of skipping -- an opt-in published case writes that dump as a side effect, so a
suite count that moves by three is that file, not a regression.

## DSV4.1 #50: THE HAND-OFF IS 71-75% OF A WALKED POSITION, IT ACCOUNTS FOR ITSELF, AND NO COMMAND BUFFER A PROBE CAN BUILD REPRODUCES THE STALL AFTER THE COMMIT

### Question

`#49` attributed the walk's last unexplained stall to the walk's I/O and priced the expert stream at
about twice its own byte time. Three things were still unmeasured when it closed, and one of them was
an assumption rather than a number:

1. **Nothing had checked that forty layer records add up to the position they came from.** `#49`
   reported 85.9-92.6 ms of post-commit idle and 94-103 ms of pre-commit host gap for one arm, plus
   29.8-30.6 ms of expert-half device time; with `#48`'s ~66 ms of dense device time that is a 285 ms
   position, and `#48` measured 218-232 ms for one. Two of those numbers had to be the same interval,
   and only a per-position clock can say which.
2. **`#49`'s single probe localised the stall to "the expert buffer" by elimination**, and a probe
   that binds nothing is not the expert half. `#48` had already failed to make a buffer slow to start
   in twenty arms, so "the expert buffer's own content" was the surviving candidate by default.
3. **The fill's throughput inside a walk had never been compared against `#21`'s isolated 13.8 GB/s.**

### What was added (measurement only, every flag off in production)

`recordsLayerGaps` grew four things; no kernel, encoder, submission pattern or arena policy changed:

- `layerGapPositions` -- one `DeepSeekV41GapPositionSample` per recorded position: its WALL time and
  how many layer records were taken during it. This is the control.
- `frontWindowStartNanoseconds` on the layer record, so the dense half's own device time is separable
  from the hand-off (only its end was recorded before).
- `mixtureStartProbeDepth` and `mixtureStartProbeDispatches` -- how many trivial buffers to commit
  back to back, and how many trivial dispatches each holds.
- `mixtureStartProbeBinding` / `mixtureStartProbeDirtyBytes` -- what a probe BINDS: nothing, one
  resident expert's arena slot, the workspace's gathered rows, thirty distinct shard allocations, or
  a 16 MB shared buffer the host has just `memset`.

### The parts tile the position, and the hand-off is most of it

Fourteen arms, one runtime, one process, one `-O` build (`/tmp/dd_dsv41_opt`), each recording one
steady-state position's 40 boundaries. `hand-off` is the host's gap (dense retire -> expert commit)
plus the pickup (commit -> expert window start). They TILE rather than overlap: the host is blocked on
the dense half until it retires, so `mixtureCommitNanoseconds` is always the later of the two and
`idleMilliseconds` IS `pickupMilliseconds` -- the `max` in it never binds.

| arm | position | dense halves | hand-off | expert halves | accounted |
|---|---|---|---|---|---|
| zero-byte fill (8-token cycle) | 71.0 ms | 26.5 | **12.7** | 24.3 | 89.3% |
| partial fill (96-token cycle) | 211.9 ms | 30.6 | **142.5** | 30.5 | 96.1% |
| full fill, plain A | 275.0 ms | 30.1 | **205.0** | 30.7 | 96.7% |
| full fill, plain B | 241.4 ms | 28.1 | **176.5** | 27.9 | 96.3% |
| eight probe arms (below) | 236.7-281.9 ms | 28.6-34.3 | **171.1-208.7** | 28.9-33.0 | 96.3-96.8% |

So `#49`'s accounting is sound: the four pieces tile a position to 96-97%, and the remainder is the
host's own work between one layer's expert half and the next layer's dense half -- ~8-9 ms of a
position, and 7.5 ms of it is visible even in the zero-fill arm. What the control adds is that the
hand-off is **167-209 ms of a 236-305 ms position (71-75%)** on this arm's fill, against 61-66 ms of
device time in total. `#48`'s 40% was a lighter arm; this one reads 556 MB/position.

### The walk's fill does not reach the disk

555.8 MB per position in 74.2-97.8 ms of `pread` is **5.5-7.3 GB/s**, where `#21` measured an isolated
fill at 13-14 GB/s against a 13.8 GB/s ceiling. One miss (17.9 MB, one shard, six ranges) takes 3.8-
4.7 ms and three misses across three shards take 10.2-10.7 ms. `DeepSeekV41ExpertStreamer` accumulates
`secondsReading` as the WALL time of the concurrent batch, so this is not a per-thread sum showing up
as wall time. The host's half of the hand-off is 74-98 ms of the position and it is running at about
half the disk.

### Every probe starts in a tenth of a millisecond, whatever it binds

The probe's own pickup against the expert half's, per arm, median over the 40 layers of one position:

| probe | window | probe pickup | the expert half behind it |
|---|---|---|---|
| one clear, nothing bound | 0.0027 ms | **0.1040 ms** | 1.7743 ms |
| three of them, back to back | 0.0025 ms each | **0.1040 / 0.1018 / 0.0968** | 1.6418 ms |
| one, eight dispatches | 0.0093 ms | **0.1072 ms** | 1.4803 ms |
| one, reading a resident expert's arena slot | 0.0027 ms | **0.1094 ms** | 1.6229 ms |
| one, reading the gathered key rows | 0.0025 ms | **0.1033 ms** | 1.3044 ms |
| one, binding thirty shard allocations | 0.0027 ms | **0.1000 ms** | 1.2872 ms |
| one, reading 4 KB the host just wrote | 0.0025 ms | **0.0993 ms** | 1.6303 ms |
| one, reading 16 MB the host just wrote | 0.0027 ms | **0.1026 ms** | 1.4455 ms |

Read that as a set of eliminations. The stall after the commit is not the queue (a probe is handed the
same empty one), not the cadence of committing into it (three commits in a row, each 0.1 ms), not how
much work the buffer carries (eight dispatches, a 3.4x wider window), not one arena binding, not the
workspace, not the length of the resource list (thirty allocations against a probe's two), and not
CPU-dirtied shared memory (16 MB `memset` on the host immediately before the commit). In 0 of 40
layers was any probe still running when the expert half started -- so the queue was idle again, in
both senses, before the expert half was.

### And it is still the walk's I/O

The same arm separates the two sides. A layer whose experts were all resident reads NOTHING and still
shows a host gap of 0.15-0.20 ms and a pickup of 1.14-2.76 ms, while in the zero-fill ARM -- where no
layer of the position reads anything at all -- all forty pickups are 0.069-0.547 ms (median 0.095).
Within a reading arm a layer's own fill predicts nothing: correlation +0.05 with its own bytes, -0.19
with the previous layer's, +0.07 with its own read seconds, +0.00 with slots allocated (plain A).

So the stall is a property of the MACHINE while the walk has been reading, not of the buffer that
waits in front of it. That is `#49`'s own closing sentence -- "what is left is the state the machine
is in, not the shape of the work" -- now with the "shape of the work" half measured instead of
assumed.

### What it means

- **The stream's price is confirmed, and it is larger than `#49`'s headline.** On this arm the hand-off
  is 205 ms of a 275 ms position: 98 ms of host gap (90 ms of it `pread`) and 107 ms of post-commit
  stall. Any miss the arena stops making is worth its bytes AND the stall that follows them.
- **The fill has a second lever nobody has priced: it runs at 5.5-7.3 GB/s in a walk.** Its reads are
  issued per shard batch, `concurrentPerform` over the batches, and a one-miss layer is one batch of
  six ranges -- so a layer that misses one expert reads 17.9 MB at 3.8-4.7 GB/s where the disk serves
  13.8. That is 40-50 ms of a position, and unlike the arena's hit rate it needs no policy.
- **The post-commit stall has no in-buffer lever left.** Every shape a probe can hold starts in a
  tenth of a millisecond; the only remaining attacks on it are structural -- fewer hand-offs per
  position (the batched block path), or giving the GPU work from somewhere else while a hand-off
  happens (two streams).

### What it is not (do not re-propose)

- **Not the submission, the cadence, the dispatch count or the resource list.** Measured above, one
  arm each.
- **Not one arena binding, not the gathered rows, not CPU-dirtied shared memory.**
- **Not the CPU cache writeback or a flush/fence.** `#49`'s alternating pair.
- **Not `F_NOCACHE`.** 2.2x loss, `#21`. **Not merging buffers.** `#48`.

### Verification

`DeepSeekV41PublishedTurnTests.testWhatEachLayersExpertHalfHandoffCosts` is opt-in behind
`/tmp/dynamoe_dsv41_layer_gaps` on top of the class's `/tmp/dynamoe_dsv41_real_turn`. It now walks the
real checkpoint fifteen times -- a discarded warm-up, the alternating read-back pair, the zero-fill
arm, the partial-fill arm, and eight probe arms -- asserting 40/40 windows in every arm and printing
each arm's per-layer rows and its parts-versus-wall line, so every number here can be recomputed from
the log rather than trusted.

The full DeepSeek suite is **570 passed / 0 failed / 63 skipped** (95 classes, 633 cases), and the same
run under `MTL_DEBUG_LAYER` is **570 / 0 / 63 with 0 validation errors and no failed assertions**. The
published numbers are the app's own `-O` build in `/tmp/dd_dsv41_opt`, fourteen arms of 202 positions
in ~11 minutes, load average 2.4-3.7. The usual caveat stands: the suite reads **573 / 0 / 60**
whenever `/tmp/dsv41_indexer_inputs.json` is present, so a count that moves by three is that file.

## DSV4.1 #51: THE FILL'S FAN-OUT IS NOT WHY A WALK'S FILL IS SLOW -- THE SHIPPED FORM IS THE FASTEST ONE, AND EVERY DEEPER QUEUE LOSES 1.3-2.7x IN SITU

`#50` closed with one lever nobody had priced: a walk's expert fill runs at **5.5-7.3 GB/s** in
555.8 MB per position, where the isolated fill the arena was tuned against measures 13-14 GB/s
(`#21`), and a one-miss layer's 17.9 MB takes 3.8-4.7 ms where the disk would take 1.3. The
hypothesis that followed from the code was the fan-out's depth: `ensureResident` groups misses by
shard and one shard's ranges fan out inside the reader, so a layer that misses ONE expert issues
**six** requests of 5.5 MB and no more, however large they are.

### Question

Is the walk's fill slow because it is too shallow, or because of the machine it runs on?

### What was added

- **`DeepSeekV41FillForm`** (`.shardBatched` shipped, `.chunked(bytes:)`, `.serial`) on
  `DeepSeekV41ExpertStreamer.fillForm`, and `chunkBytes`/`sequential` on
  `DeepSeekV41ExpertShardReader.read`. `.chunked` splits each range into pieces and fans the PIECES
  out (deeper queue for the same bytes); `.serial` issues one read at a time. `chunkBytes = 0` is the
  shipped path, unchanged.
- **`DeepSeekV41FillSample`** (`misses`, `shards`, `ranges`, **true bytes requested from the file**,
  `seconds`) behind `recordsFillSamples`. The byte count is the ranges' own lengths, not the slot's:
  pricing traffic off the slot count (17.93 MB) instead of the bytes asked for (17.5 MB) makes a
  comparison against the disk's rate wrong by 2%.
- `DeepSeekV41FillThroughputTests`, opt-in behind `/tmp/dynamoe_dsv41_fill_measure`, two cases: the
  isolated decomposition and the in-walk A/B.

### The harness's own control caught the harness first

The first sweep ran the arms in a **fixed order** and reported the shipped form at 4.26 ms for
17.9 MB while `.chunk 8 MB` reported 1.30 ms. Those are the **same code path**: every published range
is smaller than 8 MB, so splitting is the identity. The difference was position in the round — the
first arm of each round reads against a colder page cache. With the order **rotated by the round** and
six rounds (one per arm, so every arm is measured at every position) the two agree exactly:

| form (identity pair) | misses 1 best | misses 1 median | misses 6 best |
|---|---|---|---|
| `.shardBatched` (shipped) | 0.86 ms (21.8 GB/s) | 1.67 ms (10.8) | 2.63 ms (42.8) |
| `.chunk 8 MB` (the identity) | 0.80 ms (23.6 GB/s) | 1.70 ms (10.5) | 2.58 ms (43.7) |

That pair is kept in the table on purpose. **Any A/B of a fill form that does not reproduce it is
measuring the order.**

### The isolated decomposition

Same distinct `(layer, expert)` pairs, same rounds, same reader, into a Metal arena slot
(`.shardBatched`) and into a malloc'd host buffer (the control), best of six rounds:

| arm | misses 1 | misses 3 | misses 6 |
|---|---|---|---|
| `.shardBatched` (shipped) | 0.86 ms (21.8 GB/s) | 1.57 (35.9) | 2.63 (42.8) |
| `.serial` | 1.24 (15.2) | 3.84 (14.7) | 7.32 (15.4) |
| `.chunk 2 MB` | 1.33 (14.1) | 2.64 (21.4) | 4.23 (26.7) |
| `.chunk 512 KB` | 1.46 (12.9) | 3.72 (15.2) | 7.28 (15.5) |
| `.chunk 128 KB` | 1.42 (13.2) | 4.16 (13.5) | 8.35 (13.5) |
| host fan-out (6) | | | 6.74 ms (16.7) |
| host serial (6) | | | 11.35 (9.9) |
| host chunk 512 KB (6) | | | 5.97 (18.9) |

Three things fall out. **Concurrency helps** (serial is 2-3x slower), **deeper is not better** (128 KB
is the worst arm at every count), and the **destination is not the problem**: the same ranges into a
malloc'd buffer through the same reader are not faster than into an arena slot at the same shape.

### In situ, the shipped form wins every round

Three walks of 42 positions per arm, one arm order per round, a real tokenized text stream per walk
(distinct streams, because the arena serves a second walk over the same tokens for free -- `#26`):

| round | `.shardBatched` | `.chunk 2 MB` | `.chunk 512 KB` |
|---|---|---|---|
| 0 | **12.8 GB/s, 205.0 ms/position** | 7.3, 331.2 | 4.8, 448.3 |
| 1 | **5.9, 387.8** | 6.0, 357.8 | 4.6, 476.9 |
| 2 | **6.7, 339.0** | 6.0, 353.6 | 4.8, 405.3 |

`.chunk 512 KB` is 1.4-2.7x slower per byte and costs the walk **240 ms per position** in round 0.
The mechanism is the one the isolated table hints at and the walk makes obvious: a 5.5 MB `pread` gets
the file's sequential coalescing and readahead, and 512 KB of it does not. **Deepening the queue
trades coalescing for parallelism on a device where the parallelism was already there.**

### What the numbers say the fill's rate actually is

Rate is a function of **misses per fill**, not of the form. In the walk, `.shardBatched` at 1.7-2.2
misses per fill reads at 5.9-15.2 GB/s; in the same walk at 1 miss it reads at 3.8-4.7 GB/s (`#50`).
The isolated arms say the same thing from the other side: 1 miss 11-22 GB/s, 3 misses 15-36, 6 misses
16-43. A fill is **latency-bound while it carries one expert and bandwidth-bound once it carries
three**, and that is a property of how many experts the router picked, not of the code that reads them.

### What it means

- **The fill's fan-out is closed.** The shipped form is the fastest form measured, in isolation and in
  a walk, and the deeper forms lose 1.3-2.7x. The seam stays (`.chunked`/`.serial` are the baselines
  those numbers were taken against) and `.shardBatched` stays the default.
- **`#50`'s "5.5-7.3 GB/s" is not a code defect.** It is what a one-miss fill costs on a machine whose
  page cache is being consumed by the arena's own reads. There is no form of that read that makes it
  faster; there is only **fewer of them**, which is the batched block path's 2.16x and 7.6x less I/O
  (`#22`) and is unchanged as the next lever.
- **A probe that reports a rate must report misses per fill beside it.** Every fill number in this
  project before this slice was an average over fills carrying between one and six experts, which is
  the one variable that moves the rate by 3x.

### What it is not (do not re-propose)

- **Not the fan-out's depth.** Measured, isolated and in situ, both directions.
- **Not the destination.** Arena slots and malloc'd host memory read at the same rate at the same
  shape (16.7 against 16.6 GB/s for six experts); `#20`'s "the cost is this destination" is refined,
  not contradicted: the cost is the machine's state, and `F_NOCACHE` (2.2x loss, `#21`) is still dead.
- **Not the arena's slot allocation.** `#21` measured ~1 ms/position; it is inside the "total" column
  here and never inside the fill's own seconds.
- **Not thread-pool wake-up.** `.serial` exists so that claim could be checked: no concurrency is
  *slower*, so the shipped path is not paying for a thread to be scheduled.

### Verification

`DeepSeekV41FillThroughputTests` (opt-in, `/tmp/dynamoe_dsv41_fill_measure`) runs
`testWhatAFillsShapeCostsOnThePublishedCheckpoint` -- six forms x three miss counts x six rotated
rounds, plus three host-buffer arms each with its own pairs -- and
`testWhatAFillFormIsWorthInAWalk` -- three arms x three rounds of 42 real positions, reporting each
walk's own bytes, fills, misses per fill, fill seconds and wall. The isolated case costs 4 s; the
walk case 145 s at `-O`.

The full DeepSeek suite is **570 passed / 0 failed / 65 skipped** (96 classes, 635 cases; the new
class's two cases skip without the marker), and the same run under `MTL_DEBUG_LAYER` is
**570 / 0 / 65 with 0 validation errors and no failed assertions**. Two harness bugs were found and
fixed on the way, both worth carrying: an engine returned out of scope unmaps the `bytesNoCopy` shard
buffers under the runtime, which surfaces as a SIGSEGV at an address in the middle of nowhere (484 GB
here) inside a gain-vector widen; and a stale test bundle in `-derivedDataPath /tmp/dd_dsv41_opt`
reported a line number from the pre-edit source, which is what made a build look like a crash.

## DSV4.1 #52: A BATCHED PREFILL'S TIME IS NOW FULLY ACCOUNTED, AND IT IS STILL 57-77% WAITING

### Question

Routing picked `block_prefill_cost` at 0.997 / 0.996, on `#51`'s conclusion that the only remaining
lever on the expert stream is FEWER fills, and on the fact that `DeepSeekV41BlockTiming` had never
run on a published span. That type had three host buckets, folded the routing readback, the union
fill and the mixture encode into one, carried no block wall time, and had none of the
drain/late-start attribution `#50` gave the walk -- so "~215 ms per position of the batched form's
own cost is unexplained" was a statement about the INSTRUMENT as much as about the code.

### What was added

Additive, off in every production path, and nothing in the shipped walk changed:

- `DeepSeekV41BlockTiming` gained `wallNanoseconds` (the block's own wall, embedding pass included),
  `embeddingNanoseconds`, `routingReadbackNanoseconds`, `expertFillNanoseconds` (split out of
  `mixtureEncodeNanoseconds`), `fillMisses`/`fillBytes` (from the streamer's own counters),
  `blocks`, `positionCount`, and the four drain/late-start fields -- the same attribution the walk
  got in `#48`/`#50`, computed in `settlePending` under `recordsBlockTiming`. The block path commits
  its mixture WITHOUT a wait, so a mixture's drain lands in the next layer's settle: that is the
  block's hand-off, and a bucket total cannot say so.
- Two published-checkpoint cases: `testWhereABatchedPrefillsTimeGoesOnThePublishedCheckpoint` (a
  tokenized text stream at 128 and 1,005 positions, both layouts, two rounds with the arm order
  rotated) and `testWhereAPicturesPrefillGoesOnThePublishedCheckpoint` (the composer's own largest
  picture, real pixels through the tower, one round -- each arm reads 218 GB, larger than this
  machine's RAM, so there is no page cache for the rotation to protect against).

### The block's whole cost, on the checkpoint's own weights

Three regimes, `-O`, load 2.6-3.7. Every figure is ms per position; "host" is the sum of the host
buckets, which is also the harness's own whole-vs-parts control:

| regime | block | walk | ratio | host/wall | dense encode | front wait | expert fill | mixture encode | device |
|---|---|---|---|---|---|---|---|---|---|
| 128 text positions, 39 misses/pos | **197.0** | 288.4 | 1.46x | 98% | 28.6 | 96.3 | 65.9 | 1.6 | 52.3 |
| 1,005 text positions, 8.5 misses/pos | **128.7** | 159.2 | 1.24x | 98% | 23.0 | 77.5 | 23.7 | 1.6 | 46.0 |
| 1,005 positions, largest picture, 12 misses/pos | **117.1** | 183.8 | 1.57x | 98% | 17.6 | 67.3 | 28.5 | 1.6 | 41.5 |

**Nothing is unexplained.** The buckets sum to 98% of the block's wall in all three regimes -- the
same tile the walk got in `#50` (96-97%) -- and the answer to the open question is that the batched
form spends 57-77% of a position in `frontWait`, not in its kernels. The mixture's own device work is
25-38 ms of that position; the rest is the hand-off.

**The mixture's LATE START survives batching.** Per position: 65.4 ms at 128 text positions, 50.2 at
1,005 text, 40.5 on the picture -- i.e. 33%, 39% and 35% of the block's whole position is time the
queue had not yet begun the mixture after the host started waiting. `#48` found the same stall on a
walked position (69-101 ms) and `#49`/`#50` traced it to the expert half and ruled out every command
buffer a probe can build. Batching removes dispatches, round trips and (below) bytes; it does not
touch this, because the readback that empties the queue is still between the two halves.

**Batching's real, attributable win is the fill's shape.** The union fill carries 1.1-2.4 misses per
fill where a walk's carries ~0.3, and `#51`'s law says a fill reads faster the more it carries: the
picture's 213.2 GB costs **28.5 ms/position batched at 7.4 GB/s against 39.2 ms/position walked at
5.4 GB/s**, on byte counts that differ by 0.3% (218,312 against 218,886 MB). That is 10.7 ms/position
of the 66.7 ms gap, measured rather than argued.

### `#46`'s 0.74x does not reproduce, and that is the finding

`#46` measured the composer's largest picture at 345.62 / 355.22 / 386.90 s batched against
255.98 / 268.05 / 278.38 s walked -- 0.74x -- and that number is why the turn still WALKS an image
prompt. This run reproduces the prompt exactly (1,005 positions, 994 image rows, 12,176 against
12,208 misses, 218,312 against 218,886 MB) and gets **117.7 s batched against 184.8 s walked, 1.57x**.
Both arms here are ~2x faster in absolute terms than in `#46`, on the same bytes.

The order argument cannot rescue either number, and it is why this matters:

| run | first arm | second arm | result |
|---|---|---|---|
| `#46` | walk | block (cache-warm) | block 0.74x |
| `#52` | block (cache-cold) | walk (cache-warm) | block 1.57x |

The arm that ran second had the page-cache advantage in both, and the outcome reversed. So the ratio
at this span is not stable across runs, and **the shipped-path decision rests on a measurement that
flipped sign between two runs of the same prompt** -- not on a mechanism. Until a single process
alternates the arms over several rounds and reports both, neither layout can claim this span, and the
`DeepSeekV41TurnRunner` comment that cites 0.74x should be read as "measured once, under one machine
state".

What the regime table DOES show is that `#46`'s mechanism was wrong in a way that matters: it
attributed the loss to bulk eviction costing more than the round trips save, but the two layouts read
the same bytes on that prompt (0.3% apart), so no eviction count can explain an 84 ms/position gap.
The gap batched wins is the fill's rate plus the round trips, and the gap it pays is a wider union.

### The first real-weight per-stage table

On one block of eight positions, the dense half's own device windows, largest first (ms for the whole
block, share of the dense half):

| stage | ms | share |
|---|---|---|
| `queryUp` | 45.1 | 20.0% |
| `outputUp` | 43.3 | 19.2% |
| `outputDown` | 37.8 | 16.8% |
| `indexer` | 24.3 | 10.8% |
| `queryDown` | 10.6 | 4.7% |
| `latentProject` | 8.7 | 3.9% |
| `engram` | 8.7 | 3.8% |
| `router` | 8.0 | 3.5% |
| `writeLatent` | 4.3 | 1.9% |

Every stage table in this journal before now was taken on the synthetic fixture, where `writeLatent`
was 29.4% of the dense half. On the checkpoint's own weights it is 1.9%, and 67% sits in three BF16
GEMVs plus the indexer. Two caveats, both measured: the table is not additive into the block table
(its total is ~2x the in-situ dense device, because each stage now owns a command buffer and pays its
own pickup -- `#16`'s warning), and its `calibration(8 dispatch)` row (1.15 ms) is a floor the table
cannot be corrected by, only read beside.

### What it means

- **The batched prefill's cost is accounted, and it is not the batched prefill's kernels.** 57-77% of
  a batched position is the same hand-off the walk pays. A change that removes dispatches from the
  dense or mixture halves cannot move these numbers; only fewer hand-offs or another stream's work
  can (mechanism 109, unchanged).
- **`#46`'s shipped-path decision is unstable, not wrong-when-measured.** Do not flip the turn to the
  batched path on this run either: alternate the arms inside one process, several rounds, and require
  the same sign twice.
- **Any future block number must quote misses per fill and misses per position beside its rate**, for
  the same reason `#51` requires it on a fill: the block's fill reads at 6.3-10.4 GB/s across these
  three regimes purely because the router put 1.1-2.4 experts in each of them.

### What it is not (do not re-propose)

- Not a denser dense half: `denseEncode` is 12-22% of a batched position and part of that is the
  indexer's second submission, which is a host value by contract (mechanism 11).
- Not a wider block: width is capped at 8 by `chunkSlots`, and `#18` measured sixteen positions
  losing per position on the synthetic fixture.
- Not the arena: `#21` moved capacity 3,760 -> 4,637 with identical bytes on a walk, and this run's
  union fills (1.1-2.4 misses each) are already the shape `#51` says reads fastest.
- Not the vision tower: it is 6.9 s of a 118-185 s prefill on the largest picture, and its mechanical
  levers are closed in `#44`/`#45`.

### Verification

Two new cases in `DeepSeekV41PublishedTurnTests`, both opt-in:
`testWhereABatchedPrefillsTimeGoesOnThePublishedCheckpoint` (`/tmp/dynamoe_dsv41_block_span`, 128
positions, 152 s; `/tmp/dynamoe_dsv41_block_span_long` adds the 1,005-position arm, 728 s) and
`testWhereAPicturesPrefillGoesOnThePublishedCheckpoint`
(`/tmp/dynamoe_dsv41_block_span_picture`, 315 s). The shipped walk and the block path are byte-identical
in behaviour: the additions are clock reads behind `recordsBlockTiming`, which is off in production,
plus an `if let residents {` that binds the same value the old `if let residents = try ...` did.

Two unit bugs were caught by the harness's own control rather than by the numbers looking wrong, and
both are worth carrying: `millisecondsPerPosition()` divided by `positions`, which counts one entry per
LAYER per position, so every bucket read as "ms per position-layer" (5.3 against a wall of 197.0, i.e.
40x low); and the fix, deriving the position count from `positions / layers`, truncated on a
1,004-position prefill's short last block and inflated the 1,005-position arm by 14%. `positionCount`
is stored now, and `describe()` prints both the wall it must tile and the share it tiles.

## DSV4.1 #53: THE PREFILL DECISION REPRODUCES, THE MARGINS DO NOT, AND A FIRST PASS IN A YOUNG PROCESS IS WORTH MORE THAN EITHER LAYOUT

### Question

`#52` left the turn's walk-vs-block decision resting on a number that flipped sign between two runs of
the same prompt: `#46` measured the composer's largest picture at 0.74x batched (walk first in the
process), `#52` re-ran it at 1.57x (block first), on the same 1,005 positions, the same misses and the
same 213 GB. Routing picked `prefill_decision_revalidation` at 0.961 / 0.952 and said do not touch a
shipped path (0.010), so this slice alternates the two layouts inside ONE process over four rounds on
that exact prompt, prints every round instead of a best-of, and adds the identity pairs the harness
needs to know its own noise floor.

### What was added

`testWhetherTheBatchedPrefillDecisionHoldsInsideOneProcess` (the composer's own largest picture) and
`testWhetherBatchingATextPrefillPaysInsideOneProcess` (the `#52` text stream at 128 positions, plus
1,005 behind a second marker), both opt-in, sharing one private `measurePrefillSign` helper: arm order
alternates every round, every round is printed, each arm prints its own whole-vs-parts line, and the
harness prints its identity pairs, its per-arm spread, its byte check and, from `#53` on, the slot
`MTLBuffer`s each pass created. No shipped path changed.

### The result, three processes

`walk / block`, so a figure above 1 means the batched form won. Round 1 of each case is marked: it is
the pass that ran first on a freshly created runtime, and it is the one both earlier measurements
quoted.

| prompt | round 1 | rounds 2-4 | steady state | spread (block / walk) |
|---|---|---|---|---|
| text, 128 positions (3 runs) | 1.39-1.42 | 1.03-1.07 | **1.04-1.05** | 1.48x / 1.11-1.13x |
| text, 1,005 positions (young process) | 1.25 | 1.24, 1.24, 1.23 | **1.24** | 1.04x / 1.02x |
| text, 1,005 positions (aged process) | 1.26 | 1.05, 0.95, 0.95 | 0.98 | 1.66x / 1.26x |
| largest picture, 1,005 positions (run A) | 1.55 | 1.00, 0.96, 0.92 | **0.96** | 1.73x / 1.04x |
| largest picture, 1,005 positions (run C) | 0.87 | 0.84, 0.77, 0.79 | **0.80** | 1.07x / 1.12x |

- **The text prefill decision holds, at 4-24% rather than 46%.** At 128 positions three independent
  processes agree on 1.04, 1.05 and 1.05, in every round after the first. At 1,005 positions one
  process measured 1.23-1.25 in all four rounds with identity pairs of 0.98 and 0.99; `#52`'s 1.24x
  reproduces exactly, and so does its 1.46x -- as the 128-position arm's ROUND ONE.
- **The image prefill decision holds, at 4-20% rather than 26%.** The walk is ahead in every round of
  both picture runs once past the first pass (0.92-1.00 and 0.77-0.87), so `#46`'s direction was right
  and `#52` was wrong to overturn it. The block's only win on that prompt (1.55x) is a round one.
- **`#46`'s 0.74x understated it and `#52`'s 1.57x overstated the other way; both are first-pass
  numbers.** 0.74x is close to run C's steady 0.80; 1.57x is close to run A's round-one 1.55.

### A first pass in a young process is 1.3-1.6x faster, in every bucket of the batched arm

The batched arm's own clock, per position, first pass against the same arm later in the same runtime:

| case | dense encode | expert fill | device | fill rate |
|---|---|---|---|---|
| text 128, round 1 -> 2 | 31.5 -> 42.0 ms | 66.2 -> 102.9 ms | 54.0 -> 69.0 ms | 10.4 -> 6.7 GB/s |
| largest picture, run A round 1 -> 2 | 17.8 -> 29.0 | 28.5 -> 41.4 | 42.3 -> 55.4 | 7.4 -> 5.1 GB/s |

Every bucket moves together -- host command encoding, pread throughput and the GPU's own windows -- on
byte counts and miss counts that are identical to the digit (218,312 MB and 12,176 misses in all four
picture rounds; 90,025 MB and 5,021 at 128 text positions). The walk is nearly insensitive to it
(device 62.3-63.9 flat, fill 38.7-41.3 flat on the picture), because its wall is the per-layer
hand-off rather than a throughput term, so the artefact always hands the win to whichever arm runs
first. That is exactly what `#46` (walk first) and `#52` (block first) each measured.

The slot counters close one hypothesis and leave another open. A runtime allocates its 3,760 slot
`MTLBuffer`s lazily on first write and never frees them, so a fresh runtime's first prefill creates
them: 2,554 at 128 text positions, 2,913 on the picture, and **0 in every later pass**. But run C's
picture case created 2,913 of them in round one and was NOT fast (316.5 ms/position, 3.5 GB/s), while
run A's identical round one was (119.3 ms/position, 7.4 GB/s). The difference between the two runs is
the process's age (5 minutes against 45), so the state that matters is the process's, not the
runtime's, and what turns it is not established here. What is established is that quoting a runtime's
first prefill as a layout's ratio is invalid.

### What it means

- **Both shipped decisions survive.** Text prefills batch (1.04-1.05x at 128 positions, 1.23-1.25x at
  1,005); image prefills walk (the walk is ahead by 4-20%). Neither margin is what the code comments
  claimed, and the comments now quote this table.
- **A prefill A/B needs three things this one now has**: every runtime's first pass discarded,
  arm order alternated per round, and the identity pairs printed beside the ratio. The identity pairs
  here range 0.60-0.68x for a batched arm and 0.79-1.12x for a walked one, i.e. wider than every
  effect measured -- so a single-pass comparison cannot resolve the layout at ANY of these spans.
- **The margins are single-digit on the picture and on 128-position text, and 24% at length.** That
  reframes what a batching change is worth: not the 1.5-2x `#46` reported at short spans, but a few
  percent there and ~20% on a long prompt whose working set still fits the arena.

### What it is not (do not re-propose)

- Not a reason to flip either shipped path. The routed instruction was measurement-only, and the
  steady state agrees with what ships.
- Not thermal: `ProcessInfo.thermalState` read `nominal` in all twelve rounds.
- Not the page cache: the two layouts read within 0.26-0.63% of the same bytes in every case, and the
  runs whose bytes exceed RAM show the same first-pass effect as the ones that fit.
- Not the arena's capacity or its eviction policy (`#21`), and not the fill's form or fan-out (`#51`).

### Verification

Both cases are opt-in and both skip without their markers:
`testWhetherTheBatchedPrefillDecisionHoldsInsideOneProcess` (`/tmp/dynamoe_dsv41_prefill_sign`, 1,495 s
and 2,365 s across two runs) and `testWhetherBatchingATextPrefillPaysInsideOneProcess`
(`/tmp/dynamoe_dsv41_prefill_sign_text`, 302-1,754 s; `_text_long` adds the 1,005-position arm;
`_quick` halves the rounds for a harness check). Every run used `touch` and `rm -f` in the same
command. Three processes over ~70 minutes: `/tmp/test53_sign3.log` (text 128 + picture),
`/tmp/test53_sign4.log` (text 128 + 1,005) and `/tmp/test53_sign5.log` (all three, with the slot
counters).

## DSV4.1 #54: THE FIRST-PASS ADVANTAGE IS NOT THE PROMPT, NOT THE ALLOCATION, AND NOT THERMAL, SWAP OR THE PAGE CACHE

### Question

`#53` found every throughput bucket of a batched prefill 1.3-1.6x slower once a process had warmed,
gave the effect a precise magnitude and left its cause open. The app lives in the slow state for every
prefill after its first, so a cause here is worth 30-60% on every batched prefill. Routing picked
`first_pass_state_cause` at 0.992 / 0.990 and said do not touch a shipped path (0.008), so this slice
reads the kernel's own memory accounting (`task_vm_info` plus `host_statistics64`) beside the
existing buckets and tests each candidate with a controlled pass sequence.

### The passes, on one runtime, on identical work

Every pass below reads the same prompt, and every one of them reads 90,025 MB in 5,021 misses, i.e.
7.8 misses per fill (`#51`'s law held fixed by design). Two prompts (P and Q, the same paragraph
rotated differently) alternate so a pass never carries the previous pass's expert set. Run A
(`/tmp/test54_state4.log`):

| pass | ms/position | dense encode | front wait | expert fill | fill rate | device | slot buffers created |
|---|---|---|---|---|---|---|---|
| P, 1st | **234.1** | 33.3 | 117.9 | 75.2 | **9.1 GB/s** | 60.0 | 3,760 |
| Q, 1st | 310.7 | 41.5 | 139.5 | 119.4 | 5.7 GB/s | 63.6 | 0 |
| P, 2nd | 308.5 | 46.9 | 142.8 | 110.4 | 6.2 GB/s | 65.0 | 0 |
| Q, 2nd | 314.8 | 46.4 | 144.6 | 114.9 | 6.0 GB/s | 63.9 | 0 |

Two more runs of the same case reproduce the shape with different absolute levels: 9.6 -> 7.4 -> 6.9
-> 6.7 GB/s (`/tmp/test54_state3.log`) and 8.6 -> 6.4 -> 6.2 -> 5.4 under the layer hook
(`/tmp/test54_state5.log`). The first pass of a fresh runtime has measured 197.1, 232.4, 234.1, 301.6
and 316.5 ms/position on this identical work, i.e. a 1.6x spread BETWEEN processes, while the decay
inside a process is always 1.3-1.5x.

### What the candidates do

| candidate | test | verdict |
|---|---|---|
| the prompt / its expert set | a second pass on a DIFFERENT prompt (Q) is as slow as a second pass on P; misses 4,998 against 5,021 | **out** |
| the arena's slot allocation | the allocating pass is not reliably fast: first passes created 2,554-3,760 slot buffers and ranged 197-317 ms/position, and the aged picture pass in `#53` created 2,913 and was the slowest of its set | **out** |
| thermal throttling | `ProcessInfo.thermalState` read `nominal` in every round of every run | **out** |
| swapping | 0 swapins in run A and 131,072 in another; swap used 2.0-2.1 GB unchanged across passes | **out** |
| the page cache | every pass reads the same 90,025 MB from the shards; the byte count does not move | **out** |
| misses per fill (`#51`) | 7.8 per fill in every 128-position pass, 5,021 misses each | **ruled constant** |
| the arena's size | the quarter-arena control (1,202 slots) doubles misses per fill to 16.2 and read 6.0-6.5 GB/s at HIGHER compression and pressure 2, so its rate says nothing about compression | **withdrawn, uninterpretable** |
| unified-memory compression | every memory counter is FLAT or FALLING while throughput falls: task `compressed` 37.5 -> 36.4 GB and the GRAPHICS ledger's compressed share 13.1 -> 11.9 GB in one run, 53.4 -> 51.6 and 29.7 -> 25.3 GB in the other, the machine's compressor 13.8 -> 12.5 and 30.1 -> 25.8 GB | **out** |

The process's memory is genuinely squeezed, and that is worth knowing on its own: after one pass the
footprint is 74.5 GB, 36-53 GB of it compressed, the machine's compressor holds 12-30 GB, and every
128-position pass costs 4.7-8.4 MILLION decompressions. But the squeeze eases as the run proceeds
while throughput falls, in both runs that carry the counters, so the decompressions are a consequence
of the arena existing rather than what the fill pays for.

**What the decay is made of is the FILL, and almost only the fill.** The last run's buckets, per
position: expert fill 91.3 -> 120.2 -> 124.3 -> 123.3 ms (+35%), front wait 31.3 -> 36.6 -> 36.2 ->
35.9 (+17%), device 60.6 -> 67.0 -> 63.8 -> 64.6 (+7%). So the same 90,025 MB is re-read at 7.5 GB/s
on a runtime's first pass and at 5.5 GB/s on its fourth, in one process, with the same 5,021 misses and
the same 7.8 misses per fill. That is a disk-path term, not a memory term and not a kernel term.

### Two things worth carrying

- **A per-block split cannot be taken with the existing hook, and the hook is not free.** It ends with
  its own `settlePending()`, so installing it serialises the path a layer at a time. Two runs minutes
  apart, one without the hook and one with it, put the same arm at 234.1 and 232.4 ms/position, so the
  WALL does not move, but the recorded front wait collapses from 117.9 to 27.1 ms/position, because
  there is then never queued work to wait on, and the buckets stop tiling the wall (49% against 96%).
  Its absolute numbers are therefore not comparable with an unhooked run, and its per-block split was
  noisy enough to disagree with itself between runs, so no block-level claim is made here.
- **The block path's per-layer `settlePending` is free, and its overlap buys nothing.** The diagnostic
  hooks settle after every layer, which serialises the path; the same arm's wall time did not move
  (232.4 against 234.1 ms/position measured both ways) while the recorded front wait collapsed from
  117.9 to 27.1 ms/position, because there is then never queued work to wait on. That is `#52`'s
  "57-77% waiting" seen from the other side: the queue is empty anyway, so nothing was being overlapped.

### What it means

- **Do not quote a prefill number without the process's position in it.** A runtime's first pass is a
  different machine, and the effect is larger than every layout margin `#53` measured.
- **The next experiment has a shape, and it is about the disk path.** The decay is a pread rate, so
  the lever is whatever makes the same bytes read 1.35x slower on a runtime's fourth pass than on its
  first: sample per-fill rates through `recordsFillSamples` across passes (the streamer already records
  chicken bytes, pread seconds and placement separately), then separate "the volume of I/O this process
  has issued" from "the file cache's state" by repeating the passes after a rest, and by reading a
  DIFFERENT set of shard ranges (a prompt whose experts live in other files) so a per-file effect
  cannot hide behind a per-process one. If the rate recovers after a rest, it is I/O throttling or
  thermal on the controller rather than anything in this code.

### What it is not (do not re-propose)

- Not the fill's form, fan-out, chunk size, `F_NOCACHE` or destination (`#51`), not the arena's
  capacity or eviction (`#21`), not a wider block or a denser dense half (`#52`), not the attention's
  shape (`#17`), not a second command queue without a probe (`#48`).
- Not a reason to add a shipped knob. The routed instruction was measurement-only and nothing here
  changes a production path; the probe installs its hook only inside the test.
- Not unified-memory compression, which was the leading candidate going in and is now ruled out by its
  own counters falling as throughput falls.

### Verification

`testWhatTurnsABatchedPrefillsStateInsideOneProcess` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_state_probe`), 154-507 s at `-O`. It prints, per pass, the block buckets, the
streamer's counters and slot allocations, the task's footprint/graphics/compressed/purgeable ledgers,
decompressions, swapins, the pressure level, `MTLDevice.currentAllocatedSize`, the machine's compressor
and free pages and swap used, and the first block's cost apart from the rest. Runs:
`/tmp/test54_state1.log` (8 passes, no machine counters), `_state2.log` (the withdrawn quarter-arena
phase), `_state3.log` (machine counters), `_state4.log` (the P/Q alternation), `_state5.log` (with the
layer hook). Every run used `touch` and `rm -f` in the same command, and the quarter-arena phase's
`UserDefaults` seam was removed with the phase (verified with `defaults read`, which finds no key).

## DSV4.1 #55: THE WARMED PREFILL PENALTY IS THE ARENA'S DESTINATION PAGES, NOT THE DISK -- AND THE FILL DECAYS WITHIN ITS OWN FIRST PASS

`#54` left one sentence standing: the expert fill re-reads the same 90,025 MB at 7.5 GB/s on a
runtime's first pass and 5.5 on its fourth, with misses per fill unchanged, and no counter the runtime
owns explains it. Every explanation it could name had been tested from inside the fill. This slice asks
from outside the fill instead: it probes a fixed 1.1 GB of expert ranges through its own descriptors,
before the first pass and after every one, into four kinds of destination, and it samples every fill of
every pass.

### The probe

One runtime, the `#52`/`#53`/`#54` 128-position stream, four passes with the arena reset before each,
and a 1.1 GB probe (64 fixed `(layer, expert)` pairs, six contiguous ranges each, grouped per shard and
read concurrently exactly as a fill reads) at six points: before any pass, after each of the first
three, after a 20 s rest, and after the fourth.

| phase | pass ms/pos | fill GB/s | host cold | host repeat | metal resident | scratch | fresh host | fresh metal | free GB | machine compressor GB |
|---|---|---|---|---|---|---|---|---|---|---|
| before any pass | - | - | 14.7 | 96.0 | 75.6 | 16.5 | 55.3 | 59.2 | 64.8 | 0.1 |
| P1 | 238.2 | **8.4** | | | | | | | | |
| after P1 | | | 0.7 | 61.7 | 97.7 | 10.5 | 21.2 | **3.4** | 0.1 | 38.9 |
| P2 | 315.4 | **6.4** | | | | | | | | |
| after P2 | | | 14.1 | 89.1 | 95.3 | 9.9 | 25.7 | **2.5** | 0.1 | 30.1 |
| P3 | 368.8 | **5.5** | | | | | | | | |
| after P3 | | | 15.2 | 86.6 | 90.0 | 10.0 | 27.9 | **0.8** | 0.1 | 44.4 |
| after a 20 s rest | - | - | 67.1 | 67.2 | 80.8 | 10.2 | 49.9 | **48.2** | 3.4 | 41.1 |
| P4 | 325.0 | **6.3** | | | | | | | | |
| after P4 | | | 13.6 | 83.6 | 92.9 | 10.0 | 23.7 | 10.6 | 0.1 | 28.2 |

An earlier run of the same case without the two `fresh` arms reproduced the pass column (274.9, 340.5,
332.1, 341.6 ms/position at 7.7, 5.9, 6.4, 6.0 GB/s) and the same flat probe columns (host cold 15.6 /
15.1 / 12.7 / 14.7 / 16.1, host repeat 83-96, metal resident 72-98, scratch 5.6-10.9).

### What it says

- **The decay is inside the first pass, not between passes.** P1's fill runs 15.3 GB/s over its first
  byte quartile, 13.4 over its second, 4.5 over its third and 8.9 over its fourth (the earlier run:
  16.7 / 11.1 / 4.6 / 6.5). Every later pass starts where P1 ended (first quartile 10.4, 6.4, 10.0) and
  stays there (6.4 / 6.0 / 6.3 / 4.2). So "a first pass is 1.3-1.6x faster" is really "the first ~20 GB
  this process reads are fast, and it never comes back": a fourth pass after a 20 s rest is still 6.3
  GB/s.
- **Nothing outside the fill decays.** The checkpoint's own bytes read cold at 13.6-15.2 GB/s at every
  phase, including after four passes (0.7 once, in a phase whose other arms read 61.7 and 97.7, i.e. a
  stall rather than a decay). A file this test writes itself reads at 10.0-16.5 GB/s. Warm reads of the
  same ranges run at 83.6-96.0 and resident Metal slots at 72.4-97.7. The disk, the page cache, the
  process's read path and the destination TYPE are all flat across the run.
- **The destination's RESIDENCY is what changed.** The same warm bytes written into fresh arena-shaped
  slots run at **3.4, 2.5 and 0.8 GB/s** after the first three passes, against 90.0-97.7 into slots this
  process already owns. Into a fresh HOST buffer the same bytes run at 21.2-27.9. That is the fill's own
  rate (5.5-6.4) and no other arm is anywhere near it: the fill is reading from a fine source into a
  destination the kernel has to make available first.
- **One rest is enough to prove it is memory pressure and not Metal's allocation cost.** After 20 s idle
  the machine had 3.4 GB free, and the same fresh-slot arm ran at 48.2 GB/s instead of 0.8. The
  allocation path had not changed; the free memory had.
- **What the arena costs is the machine's free memory.** Free memory goes 64.8 GB before P1 to 0.1 GB
  after it, the machine's compressor 0.1 to 28-44 GB, the process's compressed total 0.0 to 53-64 GB,
  the graphics ledger's compressed share 0.00 to 15-44 GB, and decompressions 3.5M to 27.3M over the
  four passes. Those are the only counters that move with the fill, and they move with it in the right
  direction; `#54` compared their TOTALS across passes, which drift either way, and so found them
  "flat or falling" while the fill slowed.
- **This is why the walked arm was insensitive in `#53`.** A walk's fill is already at 5.5-7.3 GB/s on
  its first position (`#50`), because the arena has been filled and the machine squeezed before a single
  token is decoded; the batched arm was measured while its arena was still growing, so the same decay
  showed up as a 1.3-1.6x difference between its first round and its later ones.

### What it means

The turn's expert traffic is not waiting on the disk. It is waiting on the kernel to materialize pages in
`storageModeShared` slot buffers while the machine has no free memory and is compressing those same
buffers. The lever is therefore the arena's residency, not its capacity, its eviction order, its
placement, or the read that fills it. The candidate to measure next (NOT to build on this evidence) is
whether the slots can stay uncompressed or wired: any fix has to survive the two traps `#54` recorded
(misses per fill must be held constant, and a smaller arena changes the rate by itself) plus the one this
slice adds (`fresh` and `resident` destinations must both be measured, because they differ by 30x).

### What it is not (do not re-propose)

- Not the disk: 13.6-15.2 GB/s cold at every point in the process's life, and an unrelated file at
  10.0-16.5.
- Not the page cache: the same bytes read warm at 83-96 GB/s throughout, and a probe that nothing evicts
  reads at 88.5 and 67.1 after rests.
- Not the runtime's identity or its slot allocation: P2 allocates 0 slots and is as slow as P3, and a
  second runtime built in the aged process ran at 3.3 GB/s (`#54`'s confound applies in reverse here:
  that phase was at pressure level 2 with 138.7 GB compressed, so it is recorded, not relied on).
- Not thermal (`nominal` in every round `#54` sampled), not swap (1.9 GB used, unchanged, and 0-131k
  swapins across runs), not the prompt or its expert set (`#54`).
- Not `F_NOCACHE`, a deeper fan-out, a different fill form, a wider block, a second queue or the
  attention's shape (measured negatives, `#20`, `#21`, `#48`, `#50`, `#51`, `#52`).

### Verification

`testWhatMakesAProcessesReadsSlowerThanItsFirst` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_fill_decay`, plus `/tmp/dynamoe_dsv41_fill_decay_fresh` for the second-runtime
phase), 192 s at `-O` for four passes and six probes. It prints, per pass, the block buckets, the
streamer's counters, the per-fill minimum/median/maximum and the fill rate bucketed by cumulative bytes
read; and per probe, six destination arms plus the task and machine memory accounting. Runs:
`/tmp/test55_decay1.log` (four arms, with the second runtime) and `/tmp/test55_decay2.log` (six arms, no
second runtime). Both used `touch` and `rm -f` in the same command, and no marker survived either.

## DSV4.1 #56: THE FILL'S DESTINATION IS COMPRESSOR INPUT -- WIRING THE ARENA TAKES A PREFILL FROM 371 TO 172 ms/POSITION AND PUTS THE FILL BACK ON THE DISK'S CEILING

`#55` left one sentence standing: the same warm bytes run at 0.8-3.4 GB/s into freshly made
`storageModeShared` slots on a squeezed machine and 90-97 into slots the process owns, while the disk, the
page cache and the process's read path stay flat. Its counters named the compressor, but a counter that
moves with the fill is a correlation. A wired page is the one destination the compressor cannot take, so
this slice makes that the variable.

### The probe

One runtime, the `#52`-`#55` 128-position stream, and a fixed 1.12 GB of expert ranges (64 pairs, 17.9 MB
each) that are read cold once and then WARM for every arm, so the source is constant and only the
destination differs. The arms: `resident metal` (written at every previous probe, cold to this one),
`fresh metal` (`#55`'s arm), `touched metal` (the same new slots zeroed before the clock starts), `wired
host` and `wired metal` (`mmap` + `mlock`, once as plain memory and once as the backing of arena-shaped
`bytesNoCopy` Metal buffers), `fresh host` and `touched host`, and `resident footprint` (warm bytes
rotated through a 4.48 GB destination four times the probe's size). Every arm reports the decompressions
it caused. After each pass the arena's own slots are wired in rungs and the same pass is timed again,
with misses per fill printed beside the rate.

Two runs, and they agree to the millisecond where it matters. Run A (three rungs) and run B (four):

| pass | run A ms/pos | run A fill GB/s | run B ms/pos | run B fill GB/s | misses/fill | decompressions (run B) |
|---|---|---|---|---|---|---|
| P1, unwired | 444.7 | 4.6 | 370.6 | 6.0 | 8.1 | +8.1M |
| P2, 8 GB wired | 364.6 | 6.0 | 426.9 | 5.2 | 8.2 | +13.2M |
| P3, 24 GB wired | 361.5 | 6.5 | 330.7 | 6.7 | 8.1 | +8.3M |
| P4, **whole arena wired** (89.82 GB) | - | - | **173.8** | **13.0** | 8.2 | **+1.4M** |

Run C repeated run B end to end: P1 324.3 / 6.3, P2 412.7 / 5.3, P3 374.6 / 6.2, P4 **171.8 / 13.0** with
`device` 47.2 ms and `decompressions` +1.3M. So the fully wired pass is reproducible to 1% and the
intermediate rungs are not: 8 GB of wiring is worse than none in both runs, 24 GB is a wash, and the
whole arena is a different machine.

The destination arms, after P1 in run B (pressure level 2, 0.1 GB free, compressor 62 GB):

| arm | GB/s | decompressions | what it says |
|---|---|---|---|
| cold host (disk) | 12.6 | 0 | the disk is fine |
| warm host | 84.3 | 0 | the page cache is fine |
| resident metal | 47.5 | 14,787 | an unwired destination under pressure |
| fresh metal | **1.9** | 0 | `#55` reproduced; the cost is materializing pages |
| touched metal | 41.2 | 1 | pre-faulting recovers part of it |
| wired host | **92.6** | 0 | the same bytes, wired |
| wired metal | **99.8** | 0 | the same bytes, wired, as Metal slots |

and after P3 in run A, which is the cleaner reading because the arm had been idle through a pass:

| arm | GB/s | decompressions |
|---|---|---|
| resident metal | 13.1 | **73,050** |
| touched metal | 41.8 | 0 |
| wired host / wired metal | 88.0 / 96.0 | 0 |
| fresh metal | 1.0 | 0 |

### What it says

- **An unwired destination is compressed between uses, and a rewrite pays to reconstruct it.** 73,050
  decompressions against 1.12 GB of destination is 1.05 decompressions per 16 KB page: essentially every
  page of the arm had been compressed since the previous probe and every write reconstructed one. The
  same arm on a machine with room reads 96.2 GB/s with **0** decompressions (run C, quiet machine). The
  count tracks pressure, not the code.
- **A wired destination is untouched by it.** 79.2-99.8 GB/s at every phase in all three runs, including
  pressure level 2 with 0.1 GB free, and 0 decompressions every time. Host memory and Metal slots agree,
  so it is not Metal's allocation path; `fresh host` is the noisy arm (5.5-97.9, depending on which freed
  region a 1.12 GB allocation lands on) but it never beats wired.
- **Pre-faulting alone is not enough.** A zeroed-but-unwired slot recovers to 24.7-44.4 GB/s (0
  decompressions), i.e. 1/3 to 1/2 of the gap. The rest is the pages being compressible again.
- **The footprint decays on its own.** The 4.48 GB resident destination reads 65.7 GB/s in a quiet
  machine and 59.3 / 31.8 / 19.0 / 19.9 window by window after one pass, so a rate falls with how much
  destination the process holds, on a destination that is wired to nothing.
- **The whole arena, wired, is a different regime.** `mlock` took **6.0-6.7 s for 89.82 GB with zero
  refusals** at 0.1 GB free, and then: fill 13.0 GB/s (against the disk's own 13.6-15.2 from `#55`, i.e.
  the fill is finally disk-bound), `decompressions` 8.3M -> 1.4M, the machine's compressor 44-71 GB ->
  11-16 GB, the process's compressed total 58.7 -> 16.5 GB (the graphics ledger's compressed share
  46.3 -> 8.1 GB), and the pass 330.7-374.6 -> 171.8-173.8
  ms/position. The **device** halves moved too, 63.9-73.6 -> 47.2-47.7 ms, which is what a GPU reading
  slot pages that no longer need reconstructing should look like.
- **Wiring a fraction is not a fraction of the win.** 8 GB is *worse* than none in both full runs (412-427
  against 324-371 ms/pos), and 24 GB is inside the 5.5-7.5 GB/s band `#55` measured unwired. A pass
  writes all 90 GB of destinations, so every slot it touches that is still compressible costs a
  decompression; the rungs do not remove those, they only take room away from the compressor that the
  rest of the arena then needs (compressor stays 41-52 GB at 24 GB wired). The decompression column says
  the same thing non-linearly: 6.5-13.2M at 0-24 GB wired, 1.3-1.4M at 90.

### What it means

The turn's expert traffic was never waiting on NVMe. It was waiting on pages the kernel had compressed
and that the fill then had to reconstruct one by one, and a wired arena removes the mechanism entirely at
a cost of 6 s and no refusals. That makes the arena's RESIDENCY POLICY the first production-shaped lever
this port has measured above 2x on the prefill path: 1.9-2.2x against the passes in these runs and
against `#55`'s unwired 315-369 ms/position.

It is a measurement, not a fix, and three things have to be answered before it ships. A wired arena is
memory the machine cannot reclaim, so a policy needs a budget derived from the memory plan and a refusal
path -- `mlock` here claimed 89.82 GB because the evicted thing was the page cache, which is
re-readable; a machine whose other occupants need it has no such slack. The rungs say the transition is
not linear, so a partial wiring policy is not obviously worth having. And the wired pass is now disk
bound, so the next lever behind it is the miss rate, which is where `#21` already stopped.

### What it is not (do not re-propose)

- Not the disk: 12.6-15.3 GB/s cold in every phase of every run, and the wired fill reads at 13.0, at
  that ceiling.
- Not the page cache: warm reads 51.4-90.8 throughout, and the same arm reads 96.2 with 0 decompressions
  when the machine has room.
- Not Metal's allocation path: a wired HOST destination is just as fast (88.0-94.0), and Metal over the
  same wired memory reads 91.6-99.8.
- Not the arena's capacity, eviction order or placement, and not the prompt: misses per fill are printed
  on every row and are 8.1-8.2 in all four rungs of all three runs (`#54`'s first trap), and no rung
  changes the arena's size (`#54`'s second).
- Not the read form, the fan-out, `F_NOCACHE`, the block width, the command queue or the attention
  (`#20`, `#21`, `#48`, `#50`-`#52`).
- Not a shipped knob. `wireResidentSlots(budgetBytes:)` and its counters are measurement seams in the
  streamer, in the style of `recordsFillSamples`/`touchesFilledRanges`; production never calls them and
  no `UserDefaults` key was added (verified with `defaults read`, which finds none).

### Verification

`testWhetherAnArenasSlotsCanStayOutOfTheCompressor` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_slot_residency`, plus the class's own `/tmp/dynamoe_dsv41_real_turn`), 187 s at `-O`
for four passes and five probes. It prints, per probe, ten destination arms with each arm's decompression
count, the footprint windows, and the task and machine memory accounting; and per pass, ms/position, the
fill rate, bytes, misses and misses per fill, device time, decompressions, compressor before and after,
and the arena's wired bytes. Runs: `/tmp/test56_residency2.log` (three rungs), `/tmp/test56_residency3.log`
(four rungs) and `/tmp/test56_residency4.log` (the confirmation). All used `touch` and `rm -f` in the same
command, and no marker survived any of them.

## DSV4.1 #57: THE RESIDENCY POLICY SHIPS -- WIRING AT ALLOCATION REPRODUCES 108 ms/POSITION, AND WITHDRAWING IT COSTS 2.05x

`#56` ended with one sentence it could not answer: the win was measured by wiring the arena
RETROACTIVELY, every slot locked before the timed pass, and a shipping policy cannot do that. Slots are
allocated on first use, so a policy wires each slot inside the fill that creates it, and the first pass
pays for it. `#56` also said the rungs were not linear and that a budget had to come from the memory plan
rather than `Int.max / 2`. This slice turns the measurement into that policy and runs it.

### The policy

- **The grant is the whole arena or nothing.** `DeepSeekV41MemoryPlan.wireBudgetBytes(arenaBytes:physicalGB:floorGB:)`
  answers with the arena's own bytes when the plan's other claims (`cacheBytes + residentWeightBytes +
  engramStagingBytes`) plus a 16 GB floor for the OS, other applications and the page cache still fit in
  physical memory, and with 0 otherwise. A partial grant is not offered, because `#56` measured 8 GB
  wired as worse than none: a pass writes every slot, so a still-compressible slot costs a decompression
  whichever rung the policy stopped at.
- **Wiring happens when the slot is allocated.** `DeepSeekV41ExpertStreamer.enableSlotWiring(budgetBytes:)`
  makes `slotBuffer(_:)` apply `DeepSeekV41SlotWiring` to each buffer it creates. The `#56` retroactive
  seam (`wireResidentSlots`) stays as the measurement it was, and does not silently wire the slots a
  later pass allocates.
- **A refusal is all-or-nothing.** The first `mlock` that refuses hands back every slot already wired and
  latches the policy off, so the arena is either entirely wired or entirely not. The memory is valid
  either way; the fill runs in the unwired regime. `DeepSeekV41SlotWiring` takes its page primitives as
  a parameter, which is the only way a test can make the kernel refuse on purpose, and one case drives
  the real `mlock`/`munlock` pair as well.
- **The machine decides, with a debug opt-out.** `DeepSeekV41Runtime` grants the budget when
  `dynamoe_dsv41_wire_arena_off` is unset; `close()` hands the pages back with the streamer, and the load
  log names what happened (`arena wired 0.00 of 65.84 GB (0 slots)` at construction, then the running
  figure in `describe()`).

### The run

One runtime through the app's own loading path, the `#52`-`#56` 128-position rotated stream, a 65.84 GB
grant (3,760 slots of 17.9 MB), and five passes: three with the policy on, one after `disableSlotWiring()`,
and one after restoring it retroactively the way `#56` did. Two independent runs:

| pass | run 1 ms/pos | run 1 fill GB/s | run 2 ms/pos | run 2 fill GB/s | decompressions (run 2) | compressor (run 2) |
|---|---|---|---|---|---|---|
| P1, policy on (allocates and wires 65.84 GB) | 136.5 | 10.4 | 128.4 | 11.0 | 0 | 1.7 -> 1.8 GB |
| P2, after wiring | 110.8 | 14.4 | 108.5 | 14.4 | 0 | 1.8 |
| P3, after wiring | 111.8 | 14.5 | 108.3 | 14.4 | 0 | 1.8 -> 1.7 |
| P4, policy withdrawn | - | - | **222.4** | **8.8** | **+4,592,426** | 1.7 -> 13.2 GB |
| P5, retroactively restored (65.84 GB in 2.9 s) | - | - | 123.6 | **14.4** | +59,251 | 4.4 -> 4.1 GB |

`mlock` took **2.16-2.27 s for 65.84 GB with zero refusals**, in both runs, inside P1: 17-18 ms/position
of the 20-25 ms/position P1 pays over the wired band. So the policy costs one pass and repays it on the
next; the rest of P1's excess is slot allocation and a cold page cache, which an unwired first pass pays
too.

### What it says

- **The shipping shape reproduces the wired regime.** P2 and P3 read 14.4-14.5 GB/s in both runs (108.3-111.8
  ms/position), i.e. at or above the disk's own 13.6-15.2 and the same regime `#56` reached by wiring
  before the pass (13.0). Across two processes the wired pass is reproducible to 0.3-3%.
- **Withdrawing the policy is the A/B, and it is 2.05x.** The same runtime, stream, arena and slot
  contents, with only `disableSlotWiring()` between P3 and P4: fill 8.8 GB/s against 14.4, the pass 222.4
  against 108.4 ms/position, the device halves 55.3 against 39.6-39.8, and **4.59M decompressions** where
  the wired passes made none. The machine's compressor grows 1.7 -> 13.2 GB in that one pass. Re-wiring
  the same 65.84 GB takes 2.9 s and puts the fill rate back at exactly 14.4.
- **8.8 GB/s is a FLOOR on the unwired arm, not the ceiling.** This machine started P4 with 57 GB free and
  a 1.7 GB compressor, so the compressor only reached 13.2 GB. `#56`'s squeezed machine (0.1 GB free,
  compressor 44-71 GB) measured 4.6-6.7 GB/s unwired, i.e. the policy's win grows with the pressure it
  prevents, and the 2.05x here is what a quiet machine shows.
- **Wiring at allocation is cheaper than restoring afterwards.** Fresh slots cost 2.16-2.27 s for 65.84 GB;
  the retroactive restore after P4 compressed them costs 2.9 s. The pages exist and are unshared when the
  slot is created, which is the moment the policy is designed to take them.
- **P5's pass is still 14% above the wired band while its fill rate is fully recovered.** The compressor
  is unwinding the 13 GB P4 made it hold, so the recovery is the fill's and not the whole pass's. No
  second lever is claimed from that number.

### What it is not

- Not the disk: the wired fill is at or above the disk's 13.6-15.2, and the unwired one is far below it.
- Not a partial policy: a grant smaller than the arena is not offered, and the refusal path takes the
  whole policy back. `#56`'s rungs still say why.
- Not a memory-pressure valve. One was considered and rejected: `#56`'s win exists exactly when pressure
  is high, so a policy that unwired on `.warning` would undo the fix at the moment it is needed and hand
  65 GB back to the compressor. The release paths are the static floor gate, the `mlock` refusal, and
  `close()`.
- Not a shipped setting: `dynamoe_dsv41_wire_arena_off` exists to A/B the policy, in the style of
  `dynamoe_dsv41_pin_dense_off`.
- Not applied to the older measurement harnesses: `DeepSeekV41PublishedTurnTests.makeRuntime` defaults to
  `wireArena: false` and `DeepSeekV41FillThroughputTests` pins the same regime, because every number
  `#20`-`#56` logged was taken unwired and a run that silently wired 65 GB would move them.

### Verification

`testWhatTheArenaResidencyPolicyIsWorthIncludingItsLock` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_wired_policy`, plus the class's own `/tmp/dynamoe_dsv41_real_turn`), 98 s at `-O`
for five passes. It asserts the grant is enabled, every allocated slot is wired, the wired passes read
above 8 GB/s, and nothing was refused. Policy logic is pinned without a checkpoint by
`DeepSeekV41SlotWiringTests` (a slot is wired once, a budget is never exceeded, a refusal unlocks exactly
what it wired and latches, a zero budget is off rather than a refusal, and the system `mlock` pair wires
a real page) and `DeepSeekV41ArenaWiringBudgetTests` (a whole arena granted, one that would eat the floor
refused entirely, and the floor proven to be the reason). Runs: `/tmp/test57_policy1.log` (three passes)
and `/tmp/test57_policy2.log` (the five-pass A/B). Both used `touch` and `rm -f` in the same command, and
no marker survived.

The full `DynaMoETests` target ran 1,471 tests with 89 skipped and 33 failures in 131.0 s; all 98
`DeepSeekV41*` suites passed with 0 failures, including the 9 new policy cases and the opt-in case
skipped. The 14 failed cases are the pre-existing environmental ones (`DynaMoETests`: help-topic count,
Qwen3.8 Flash Next forward, tool-discover lifecycle, truncated tool-call recovery; `ModelDogfoodAndPrefixCacheTests`:
the 9 prompt/web-fetch/headless-Chrome cases) plus `ServerQueueLogicTests.testTimedOutTurnKeepsFIFOBarrierForSuccessors`,
which passes in isolation in 0.24 s and fails only under full-suite load, so it is a timing flake and not
this change.

## DSV4.1 #58: THE RESIDUAL MISSES ARE THE ARENA'S OWN EVICTIONS -- A WARM PASS RE-READS 54.6-56.7 GB IT ALREADY HAD, AND A FREQUENCY RULE TAKES THAT TO 38.6-39.3

`#57` shipped the residency policy and left one number standing: the wired fill reads at 14.4-14.5
GB/s, the disk's own 13.6-15.2, with **8.1-8.2 misses per fill** that nothing in the policy moved.
That made the miss rate the next lever, and `#21` had already closed it -- "the fill is cold-start
bound, not capacity bound" -- from a 53-position walk whose 3,739 distinct pairs *just* fit the
3,760 slots. Those two readings are compatible only if a pass's misses are its COMPULSORY working
set, pairs read once and never needed again, because no replacement rule can remove a read that has
never happened before. This slice makes that answerable instead of arguable, and the answer is no:
the cold pass is compulsory, and every warm pass after it is the arena throwing away what it had.

### The instrument

`DeepSeekV41ExpertArena.recordsMissComposition` classifies each miss since the last `reset()` as a
pair this window had never loaded (`firstTouchMisses`) or one it had loaded and then evicted
(`refetchMisses`), with `distinctPairs` as the window's working set. It costs one `Set` insert per
miss and is off in every production path. The two columns are the whole question: the first is the
floor a perfect oracle would still pay on the same slots, and only the second is what a policy can
win back. Every arm asserts `firstTouch + refetch == acquisitions - hits` and that the streamer's
own `fillMisses` equals the arena's, so the decomposition cannot silently miss a miss.

There is a second, policy-free decomposition in the same table: a pass from `arena.reset()` is a
COLD arena, and a pass without one is the steady state a live turn is in.

### The run

One runtime through the app's own loading path, the `#52`-`#57` 128-position rotated stream, the
shipped 65.84 GB arena (3,760 slots of 17.9 MB) wired by the shipping policy, and six passes: three
under LRU and three under frequency, with identical rotations on both sides. Then a seventh and
eighth pass on a second runtime at a 0.72 arena fraction. `-O`, one process per runtime.

| arm | policy | arena | ms/pos | fill GB/s | read GB | misses | first touch | re-fetch | evictions |
|---|---|---|---|---|---|---|---|---|---|
| A1 cold | LRU | 65.84 GB | 126.6 | 10.9 | 87.9 | 5,021 | 4,784 | 237 (4.7%) | 1,261 |
| A2 warm | LRU | 65.84 GB | **84.2** | 13.8 | 54.6 | 3,117 | 527 | 2,590 (83.1%) | 3,117 |
| A3 warm | LRU | 65.84 GB | **84.8** | 13.8 | 56.7 | 3,239 | 0 | 3,239 (100%) | 3,239 |
| C1 cold | frequency | 65.84 GB | 104.3 | 14.5 | 87.9 | 5,022 | 4,784 | 238 (4.7%) | 1,262 |
| C2 warm | frequency | 65.84 GB | **72.7** | 14.4 | 39.3 | 2,247 | 527 | 1,720 (76.5%) | 2,247 |
| C3 warm | frequency | 65.84 GB | **73.2** | 14.4 | 38.6 | 2,203 | 0 | 2,203 (100%) | 2,203 |
| B1 cold | LRU | 81.19 GB | **239.0** | 6.4 | 83.8 | 4,786 | 4,784 | 2 | 149 |
| B2 warm | LRU | 81.19 GB | 79.5 | 13.1 | 32.4 | 1,852 | 527 | 1,325 (71.5%) | 1,852 |

A1 pays the first pass's slot allocation and `mlock` (2.2 s), which is why its rate is 10.9 against
the 13.8-14.5 band the passes after it read; that is `#57`'s shape, not a new observation.

### What it says

- **The working set EXCEEDS the arena, and by enough to matter.** A 128-position pass touches 4,784
  distinct pairs -- 83.8 GB -- against a 3,760-slot arena, 21.4% short. `#21`'s 53-position walk
  touched 3,739, i.e. it stopped one position short of the regime where capacity starts to bite, and
  its 0.6-vs-0.72 A/B compared bytes at a position that fit both sizes, so it could not have seen a
  difference by construction. Quote the working-set column before quoting any capacity conclusion.
- **A cold pass is compulsory and a policy cannot help it.** 95.3% of A1's misses are first touches
  (4,784 of 5,021) and the arena's own share of the read is 4.1 GB of 87.9. The frequency rule
  changes the cold pass by one miss (5,021 against 5,022, 4,784 first touches both), which is also
  the control that says routing does not consult the arena: the two policies were measured on the
  same 16,658 acquisitions and the same 4,784 pairs.
- **A warm pass is almost entirely re-fetches.** 83.1-100% of a warm pass's misses are pairs the
  arena already held, and A3 is the pure case: a warm arena that has already seen this exact stream
  re-reads 56.7 GB with **zero** first touches. The arena is not cold, it is thrashing. That is what
  the 8.1-8.2 misses per fill `#57` left standing actually is.
- **Capacity fixes the re-fetches, and costs more than they are worth.** At 4,637 slots the cold
  pass's re-fetch column collapses from 237 to 2 and its evictions from 1,261 to 149, and a warm
  pass reads 32.4 GB against 54.6. But `mlock` took the whole 81.19 GB with **zero refusals** and
  left the machine with **0 GB free**, so the cold pass fell to 6.4 GB/s with 1.91M decompressions
  and 239.0 ms/position against the shipped arena's 126.6 -- everything the process did not wire
  (the 10.95 GB dense pin, the shard mappings, the workspace) became compressor input. The 40% read
  win is real and the wall-clock win is 4.7 ms/position, against a 112 ms/position loss on the pass
  that fills it. The static `wireBudgetBytes` gate granted it, correctly: the bytes do fit. What
  does not is everything else the machine was holding.
- **The replacement rule is the cheap lever, and it wins.** At the SAME 3,760 slots and the same
  stream, frequency takes a warm pass from 84.2-84.8 to 72.7-73.2 ms/position (-13.7%) and its read
  from 54.6-56.7 GB to 38.6-39.3 (-28%). It is cheaper than capacity on both count and time: 72.7
  ms/position at 65.84 GB beats 79.5 at 81.19 GB, with no extra memory and no slower first pass.
- **The rule is what LRU cannot know.** A pass touches its 4,784 pairs 3.5 times each on average, so
  the recurring set is exactly the one with a count above one. LRU sees only recency, so a singleton
  evicts a pair this layer-block will want again; frequency evicts the lowest count and breaks ties
  by LRU, and a capacity-worth of misses halves every count so a stale hot pair is retired instead
  of pinned forever. 2,203-2,247 misses remain at 3,760 slots, i.e. 1,024 pairs cannot fit whoever
  chooses the victim -- the floor a policy is measured against.

### What it is not (do not re-propose)

- Not the disk, the page cache, the read form, the fan-out, `F_NOCACHE`, the block width, the
  command queue or the attention: every arm filled at 13.1-14.5 GB/s in its steady state, at the
  disk's own rate, and `#51` closed the forms.
- Not the prompt: both cold passes touched the same 4,784 pairs on the same stream, and the warm
  passes are the same two rotations.
- Not a bigger arena by default. The capacity win exists and is measured here, and so is its price;
  the arena fraction stays at 0.6 and `dynamoe_dsv41_arena_fraction` remains the debug seam.
- Not a memory-pressure valve, a partial wiring policy, or arena eviction measured without a
  working-set column: `#56`/`#57` still say why, and this slice says why the column is mandatory.
- Not `mlock` refusing: the wider arena locked all 81.19 GB with zero refusals. The policy did its
  job; the machine had nothing left for anything else.

### Verification

`testWhatTheResidualMissesAreMadeOf` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_miss_composition`, plus the class's own `/tmp/dynamoe_dsv41_real_turn`), 126 s
at `-O` for six passes and two runtime builds. It asserts the decomposition covers every miss and
matches the streamer, that both cold passes touched an identical working set, that the shipped
regime's warm passes read above 8 GB/s, and that the wider arena stayed wired with no refusal.
`DeepSeekV41ExpertArenaTests` pins the policy without a checkpoint: frequency keeps the pair it has
seen more often, the tie-break among equals is still LRU, aging retires a pair nothing has touched
since, frequency is the default, and the composition counters classify once each and reset with the
arena. Runs: `/tmp/test58_probe1.log` (the decomposition), `/tmp/test58_probe2.log` and
`/tmp/test58_probe3.log` (with residency and the policy arms) and `/tmp/test58_probe4.log` (the
paired rotations, the green log). Every run used `touch` and `rm -f` in the same command and no
marker survived any of them.

The frequency rule ships as the arena's default; `dynamoe_dsv41_arena_lru` is the debug opt-out, and
every measurement harness pins LRU explicitly so `#20`-`#57`'s tables stay comparable. Shipping it
was a judgment call the numbers only half settled -- SubQ put "should frequency be the default?" at
0.58 (mid-band, low confidence) and "should the arena fraction be raised?" at 0.005 -- so the record
is that the decision was made on the mechanism: the cold pass is provably unaffected, the warm pass
is 13.7% faster on two rotations in two processes, and the rule cannot change what a slot holds,
only which pair loses one.

The full `DynaMoETests` target ran 1,478 tests with 90 skipped and 34 failures in 132.6 s; all 98
`DeepSeekV41*` suites passed with 0 failures, including the six new arena cases and the opt-in probe
skipped. The 14 failed cases are the pre-existing environmental ones (`DynaMoETests`: help-topic
count, Qwen3.8 Flash Next forward, tool-discover lifecycle, truncated tool-call recovery;
`ModelDogfoodAndPrefixCacheTests`: the 9 prompt/web-fetch/headless-Chrome cases) plus
`ServerQueueLogicTests.testTimedOutTurnKeepsFIFOBarrierForSuccessors` and — new to this run's list —
`ServerQueueLogicTests.testPerformTimeoutFiresWhileLeaderGateClosed`, both of which pass in isolation
(13/13 in the suite in 2.8 s) and are timing cases under full-suite load.

## DSV4.1 #59: THE REPLACEMENT RULE TRANSFERS TO THE WALK -- A WARM DECODE READS 31.9 GB INSTEAD OF 56.2 AND SAVES 15-19% OF A POSITION

`#58` shipped the frequency rule on a 128-position PREFILL and named its own caveat: a live turn's
decode loop is the walk, one position at a time through the same arena, and no walk had ever run
under the rule. `#21` had measured a 53-position walk whose 3,739 distinct pairs *just* fit 3,760
slots, which is how it concluded there were no evictions to tune; `#58` then showed a 128-position
working set does not fit. This asks the walk side of that question at the length where the overflow
exists, with no production change: the rule already ships, and this is what it is worth to a decode.

### The run

One runtime through the app's own loading path, one 128-position token stream (the `#52`-`#58` text,
one rotation), six walks. Every arm replays the SAME stream from position 0, so every arm issues the
identical 30,720 acquisitions and only the pair a miss replaces changes. Cold arms `reset()` the
arena; warm arms do not, which is the steady state a live turn decodes in. `-O`, shipped 65.84 GB
arena (3,760 slots of 17.9 MB), 128 positions.

| arm | policy | ms/pos | read GB | fill GB/s | misses | first touch | re-fetch | evictions |
|---|---|---|---|---|---|---|---|---|
| A1 cold | LRU | 158.3 | 87.6 | 13.8 | 5,005 | 4,790 | 215 (4.3%) | 1,245 |
| A2 warm | LRU | 121.9 | 56.2 | 12.4 | 3,210 | 0 | 3,210 (100%) | 3,210 |
| A3 warm | LRU | 122.0 | 56.2 | 12.4 | 3,210 | 0 | 3,210 (100%) | 3,210 |
| C1 cold | frequency | 140.8 | 87.6 | 13.3 | 5,004 | 4,790 | 214 (4.3%) | 1,244 |
| C2 warm | frequency | 106.4 | 35.1 | 12.3 | 2,002 | 0 | 2,002 (100%) | 2,002 |
| C3 warm | frequency | 100.8 | 28.7 | 12.2 | 1,640 | 0 | 1,640 (100%) | 1,640 |

Warm mean: **LRU 122.0 ms/position and 56.2 GB read; frequency 103.6 and 31.9, i.e. -15.0% wall and
-43.3% read.** The worst frequency warm arm (C2, 106.4 ms and 35.1 GB) still beats the best LRU one
(121.9 and 56.2), so the ordering does not depend on picking an arm. The probe ran three times; the
other two gave -14.9% and -19.4% wall with the SAME -43.3% read, so the byte win is the reproducible
one and the wall win is 15-19%, depending on page-cache state (see the C3 note below).

### What it says

- **A warm walk's misses are re-fetches, all of them.** `firstTouch` is 0 in every warm arm, exactly
  as `#58`'s prefill A3 found: an arena that has already seen the stream re-reads what it evicted
  and nothing it has never seen. The compulsory column is the cold pass, and it is untouched.
- **The rule transfers, and by a wider read margin than on a prefill.** Frequency takes a warm walk
  from 56.2 to 31.9 GB (-43.3%) against the prefill's 54.6-56.7 to 38.6-39.3 (-28%), and the wall
  by 14.9-19.4% against the prefill's -13.7%. A decode position draws 240 acquisitions (6 experts x
  40 layers), so it repeats its working set more often per byte read, which is the direction the rule
  wins in.
- **The win is the fill bucket, and the graph is untouched.** Comparing the first warm pass of each
  policy (A2 against C2): front encode 2.5 + front wait 61.4 + expert fill 38.6 + mixture encode 2.6
  ms/position under LRU, against 2.5 + 60.5 + 24.5 + 2.7 under frequency (device 53.8 both). The fill
  drops 38.6 to 24.5, i.e. 14.1 ms/position, and the host round trip and both device halves are flat.
  Replacement changes how much I/O is repeated, not what is computed.
- **The fill stays disk-bound in this run: all six arms read at 12.2-13.8 GB/s**, the disk's own
  rate. One earlier run of the same probe had C3 at 26.0 GB/s, ABOVE the measured 13.8 ceiling, so
  part of that arm's 28.7 GB came from the page cache rather than the disk; that is an order effect
  the engine does not control (all six arms replay one stream, so a later arm's bytes may still be
  cached), and it is why the wall win is quoted as a range. The miss/byte columns are the structural
  result: they were identical in all three runs (3,210 under LRU against 2,002 and 1,640 under
  frequency).
- **The walk's working set is the prefill's regime, not `#21`'s.** 4,790 distinct pairs (85.7 GB)
  against 3,760 slots (65.84 GB), 27.4% over, and 1,030 pairs cannot fit whoever picks the victim.
  Frequency leaves 1,640-2,002 misses at that capacity, which is the floor a future rule is measured
  against.
- **Frequency converges over warm passes; LRU is flat.** C2 (2,002 misses, 35.1 GB) is the first warm
  walk after a cold one, C3 (1,640, 28.7 GB) the second, because the use counts a cold pass seeds are
  all ones. LRU misses the same 3,210 and reads the same 56.2 GB both times. The mean above uses both
  arms; a reader quoting a single warm arm should say which one. A live turn's arena is warmed by its
  prefill and then by every position of its answer, so the converged number is the one it reaches,
  not the one it starts at.
- **The cold pass is still a control, not a result.** Both cold walks touched the same 4,790 pairs,
  missed within one of each other (5,005 against 5,004) and read the same 87.6 GB, so the rule does
  not move compulsory traffic. The 17.5 ms/position wall gap between A1 and C1 is not the policy: A1
  is the pass that creates and wires the arena's 3,760 slot buffers (fill 72.8 against 56.4
  ms/position, about 2.1 s over 128 positions; the other two runs gave 22.9-23.9 ms), which is
  `#57`'s first-pass note that the lock is paid inside the pass that allocates rather than up front.
  C1 reuses the buffers `arena.reset()` does not free.
- **What a decode now costs.** 101-122 ms/position on a warm arena, of which 59-63 ms is the host
  round trip and about 25 ms the mixture's device window. `#22`'s 205-324 ms/position was a colder
  regime; this is the steady state, and it says the next decode lever is still the round trip
  (`#22`'s late start) rather than replacement.

### What it is not (do not re-propose)

- Not a production change. No source outside the test harness and this journal moved; the rule
  shipped in `#58` and this is its decode-side measurement.
- Not a case for a bigger arena: capacity was priced in `#58`'s B1 (81.19 GB wired, 0 GB free, cold
  pass at 6.4 GB/s), and this slice varies no capacity.
- Not a licence to change the host round trip from a replacement result: `front wait` is 59.3-62.9
  ms/position across the six arms of the quoted run (58.9-65.7 across all three) and does not move
  with the policy.
- Not evidence that a cold pass got faster: the arms differ by one miss and identical bytes, and the
  wall gap is slot creation and wiring in the first arm.
- Not a page-cache argument: one run's C3 hit the cache at 26.0 GB/s, but the engine does not control
  that, and the rule's contribution is the 24.3 GB per pass it stops asking the cache for.
- Not a re-run of `#21`'s capacity A/B and not a partial wiring or memory-pressure valve argument;
  `#56`-`#58` still say why.

### Verification

`testWhatTheReplacementRuleIsWorthToAWalk` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_walk_policy`, plus the class's own `/tmp/dynamoe_dsv41_real_turn`), 102 s at
`-O`. It asserts that the composition covers every miss, that the fills the streamer sampled equal
the arena's misses, that both cold walks touched an identical working set, and that every warm arm
fills above 8 GB/s. No new unit case: the rule's replacement, aging, composition and default are
already pinned by `DeepSeekV41ExpertArenaTests` (`#58`), and this slice adds no production code.
Runs: `/tmp/test59_walk3.log` (the one quoted above), plus `/tmp/test59_walk1.log` and
`/tmp/test59_walk2.log` for the wall range. Markers were touched and removed in the same command;
none survived.

The full `DynaMoETests` target ran 1,479 tests with 91 skipped and 30 failures in 119.1 s on the
final `-O` test build. Every `DeepSeekV41*` suite passed with 0 failures, including the new probe when
it was skipped and the existing `DeepSeekV41ExpertArenaTests`. The 13 failed cases are exactly the
pre-existing environmental ones (`DynaMoETests`: help-topic count, Qwen3.8 Flash Next forward,
tool-discover lifecycle, truncated tool-call recovery; `ModelDogfoodAndPrefixCacheTests`: the 9
prompt/web-fetch/headless-Chrome cases); both `ServerQueueLogicTests` timing flakes that appeared in
`#58`'s run passed here. Run: `/tmp/test59_suite2.log`.

## DSV4.1 #60: THE EXPERT STAGES ARE 43% OF A WALKED POSITION'S DEVICE TIME, AND THE BLOCK'S STAGE RANKING DOES NOT TRANSFER

`#59` closed the arena's replacement question and left a warm decode at 101-122 ms/position with
53.8 ms of device windows -- the largest single bucket, and one no published table had ever attributed.
Every stage table in this journal comes from a synthetic fixture, and the one table taken on the real
checkpoint (`#52`) is a BLOCK of eight positions. This asks the walk's side of it, because the block's
answer cannot be the walk's: its three biggest stages are GEMV projections whose cost there is the
per-position activation traffic a block re-requests once per output row (`#18` priced it at 1.9 GB of
requests for one block), and that traffic grows with the block width while a walk asks for one
position's worth.

### The run

One runtime through the app's own loading path, one 128-position token stream (the `#52`-`#59` text,
one rotation), the shipping frequency rule, `-O`, the shipped 65.84 GB arena (3,760 slots of 17.9 MB).
A warm-up pass is discarded (a young process pays slot creation, wiring and a cold page cache), then
two arms walk the SAME stream without an arena reset between them, so both are in the steady state a
live turn decodes in -- `#59`'s C2/C3 regime:

1. **baseline**: one command buffer per half, the decomposition `#59` quoted;
2. **stages**: one command buffer per stage (`recordsWalkStageTiming`), so every stage reports its own
   `gpuStartTime`/`gpuEndTime`.

| arm | wall | device | dense device | mixture device | misses | read |
|---|---|---|---|---|---|---|
| baseline, one buffer per half | 105.6 ms/pos | 54.3 | 28.7 | 25.6 | 2,002 | 35.1 GB |
| stages, one buffer per stage | 82.1 ms/pos | 58.5 | 32.9 | 25.5 | 1,640 | 28.7 GB |

The two walls are NOT comparable and the difference is not the layout: the second arm's fill came out
of the page cache (11.1 ms/position against the first's 24.0, 28.7 GB at ~20 GB/s), exactly the order
effect `#59` saw on its third arm. The miss and byte columns are the structural ones and they are
`#59`'s own (2,002 / 35.1 then 1,640 / 28.7). Only the device columns are read below.

### The layout is the same graph

- **The answer is unchanged to the bit**: worst logit delta 0.00e+00 absolute on a 24.96 scale, after
  128 positions. Splitting a half into per-stage buffers changes no kernel, no binding and no
  submission order.
- **The mixture's four stages add up to the mixture half**: 25.51 against 25.6, and 24.99 against
  25.3 in the second run -- 0.4-1.2%. Each carries ~3 ms of device time per layer, so a per-buffer
  cost is invisible there.
- **The dense half's twenty stages add up to MORE than its one-buffer total**: 32.9 against 28.7, and
  32.85 against 28.6 in the second run -- +4.2 ms/position, +14.6%, every time. The one-buffer layout
  overlaps adjacent dense stages (or the split pays ~5 us per boundary); either way the dense shares
  below are shares of a SERIALIZED 32.9 ms whose overlapped cost is 28.7, and the table is read as a
  ranking with that upward bias, not as absolute milliseconds.
- The per-stage windows sum to exactly the half totals the same arm reported
  (`stageTotal == frontDevice + mixtureDevice`), which is what makes the division a division of one
  population rather than of two.

### The table

Milliseconds per walked position and share of the 58.45 ms of serialized device time, 27 stages that
report a window (`.compressPool` dispatches nothing -- the pool runs inside `.indexer`).

| stage | ms/pos | share | half |
|---|---|---|---|
| sharedExpert | 12.963 | 22.2% | mixture |
| routedExperts | 12.198 | 20.9% | mixture |
| queryUp | 4.357 | 7.5% | dense |
| outputDown | 4.165 | 7.1% | dense |
| hcAttentionCoefficients | 4.105 | 7.0% | dense |
| router | 3.621 | 6.2% | dense |
| outputUp | 3.505 | 6.0% | dense |
| hcFFNCoefficients | 3.373 | 5.8% | dense |
| latentProject | 1.585 | 2.7% | dense |
| attention | 1.409 | 2.4% | dense |
| indexer | 1.170 | 2.0% | dense |
| queryDown | 0.955 | 1.6% | dense |
| ropeOutput | 0.900 | 1.5% | dense |
| ropeQuery | 0.884 | 1.5% | dense |
| engram | 0.635 | 1.1% | dense |
| ffnNorm | 0.538 | 0.9% | dense |
| attnNorm | 0.507 | 0.9% | dense |
| queryNorm | 0.237 | 0.4% | dense |
| writeFFNStreams | 0.205 | 0.4% | mixture |
| ropeLatent | 0.191 | 0.3% | dense |
| latentNorm | 0.190 | 0.3% | dense |
| writeLatent | 0.179 | 0.3% | dense |
| mixFFNStreams | 0.153 | 0.3% | dense |
| writeAttentionStreams | 0.150 | 0.3% | dense |
| collapseStreams | 0.143 | 0.2% | mixture |
| mixAttentionStreams | 0.132 | 0.2% | dense |
| compressPool | 0.001 | 0.0% | dense |

The second run reproduced every figure within 2% (total 57.84, dense 32.85, mixture 24.99; the top
eight in the same order), so this is not a one-run table.

### What it says

- **The two expert stages are 43.1% of a walked position's device time and 99% of the mixture half.**
  The SHARED expert (12.96) is the single largest stage of a decode -- larger than all six routed
  experts together (12.20). Nothing in the dense half comes close.
- **The routed experts run at the machine's measured GEMV ceiling; the shared expert does not.** One
  position reads 240 experts (6 x 40 layers) x 17.9 MB = 4.30 GB in 12.20 ms, i.e. **352 GB/s**,
  which is `#20`'s 300-370 GB/s in isolation. The shared expert reads 40 x 3 x 5120 x 2304 values =
  1.42 GB (1.46 GB with its 32x32 block scales) in 12.96 ms, i.e. **109-113 GB/s** -- about a third
  of the routed kernels on the same shapes and at twice their per-byte rate. `#20` measured the
  coalesced FP8 form in isolation at 262 GB/s on the shared `w1` alone, so this is an in-situ gap,
  not a form that was never measured.
- **The block's ranking does not transfer, and the direction is the one `#18` predicted.** `queryUp`
  is 20.4% of a block's dense half and 7.5% of a walk's device time; `outputUp` 19.1% -> 6.0%;
  `outputDown` 17.7% -> 7.1%. Between them 57% -> 20.6%. A block re-reads every position's activation
  vector once per output row; a walk reads one 20 KB vector and the weight once.
- **Those three GEMVs are themselves bandwidth-bound.** `queryUp` reads 32,768 x 1,280 values =
  1.73 GB/position in 4.36 ms (**397 GB/s**), `outputDown` 1.38 GB in 4.17 ms (**332 GB/s**); both are
  at the ceiling, so their cost is the bytes they must read and no kernel shape will take it away.
- **The mHC coefficient pair is 7.48 ms/position, 12.8% of device time** -- 4.1 for the attention
  sub-layer's projection and 3.4 for the FFN's. The block table put the same pair at 1.9 ms/position,
  and the difference is the batching: a block projects ONCE for eight positions, a walk once per
  position, 120 dispatches per position of 24 threadgroups each (four slots x six logits).
- **`router` is 3.62 ms/position, 6.2%**, for the same reason: one dispatch per position scoring 384
  experts, amortized over a block and paid per token by a walk.
- **The attention itself is not a target.** `attention` is 1.41 ms (2.4%) and `writeLatent` 0.18 ms
  (0.3%). `#17` closed the attention at these widths and the walk agrees.
- **The compressor is free on a walk** (0.001 ms): it runs inside `.indexer`, which is 1.17 ms
  (2.0%) including the selection.
- **What is left is a tail of small per-position kernels.** mHC pair 7.48 + router 3.62 + ropes 1.98
  + norms 1.47 + stream mixes 0.58 = **15.1 ms/position, 26% of device time and 14% of the wall**,
  all of it latency-bound rather than bandwidth-bound. A block pays each once per eight positions;
  a walk pays each forty times.
- **The wall around it** (baseline arm): 101.7 ms/position of host buckets -- front encode 2.5, front
  wait 60.0, indexer wait 12.5, expert fill 24.0, mixture encode 2.7 -- against a 105.6 ms wall, so
  ~3.9 ms is outside every bucket. The wait's own split is drained front 32.0 / mixture 44.0 with
  late starts 0.7 / 14.7 ms; those drains cross window boundaries (the indexer's readback and the
  token's final settle are settles that are not front waits), so they do not partition the 60.0 and
  are not a fourth bucket.

### What it is not (do not re-propose)

- **Not a case for the attention, the cache write or the compressor.** 1.41, 0.18 and 0.001
  ms/position, measured at the published widths.
- **Not evidence that the three GEMVs are the walk's problem.** They are 20.6% together, and at
  332-397 GB/s they are already at the ceiling. The block table's 57% is a block property.
- **Not a rows-per-threadgroup or "wider block" argument for the walk.** `#18` and `#21` both closed
  those on the shapes they were measured on, and nothing here reopens them.
- **Not a page-cache or fill result.** The two arms' walls differ by 12-17 ms/position purely because
  the second arm's fill was served from cache; only the device columns are compared.
- **Not a capacity or replacement result.** The arena is the shipped one and the policy is the
  shipping rule; this slice varies neither.
- **Not a claim that the block table was wrong.** It was a correct table of a block. It is simply
  not a decode's ranking, and this is the decode's.

### Verification

`testWhereAWalkedPositionsDeviceTimeGoes` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_walk_stages` plus the class's own `/tmp/dynamoe_dsv41_real_turn`), 49-61 s at
`-O`. It asserts that the two layouts return identical logits, that the stage windows cover every
position and add up to the half totals, that no half buffer is unwindowed, and that every stage the
checkpoint dispatches appears in the table. Runs: `/tmp/test60_walk2.log` (the one tabulated above)
and `/tmp/test60_walk3.log` (the reproduction); `/tmp/test60_walk1.log` is a first cut whose arms were
arena-reset (cold), and it produced the same ranking. The same case also passes under the Metal debug
layer (`/tmp/test60_walk_mtl.log`, 50.1 s, no validation errors), where it reports 57.65 ms/position
of device time (32.77 dense, 24.88 mixture) -- a third run of the same table.

The instrument itself is pinned without the checkpoint by
`DeepSeekV41DeviceGraphParityTests.testThePerStageWalkLayoutIsTheSameWalk` (synthetic fixture): the
split layout must return bit-identical logits, its stage windows must equal the half totals, and the
table must name every stage the fixture dispatches. That case also passes under the debug layer.

The full `DynaMoETests` target ran 1,481 tests with 92 skipped and 30 failures in 119.7 s on the `-O`
build: every `DeepSeekV41*` suite passed with 0 failures (the two new cases included), and the 13
failed cases are exactly the pre-existing environmental set (4 in `DynaMoETests`: help-topic count,
Qwen3.8 Flash Next forward, tool-discover lifecycle, truncated tool-call recovery; 9
`ModelDogfoodAndPrefixCacheTests` prompt/web-fetch/headless-Chrome cases). Run:
`/tmp/test60_suite.log`.

No kernel, encoder, dispatch shape or production path changed: `recordsWalkStageTiming` is a flag
that is off in every production path, and the single-buffer layout a turn uses is byte-for-byte the
one `#59` measured.

No kernel, encoder, dispatch shape or production path changed: `recordsWalkStageTiming` is a flag
that is off in every production path, and the single-buffer layout a turn uses is byte-for-byte the
one `#59` measured.

## DSV4.1 #61: THE SHARED EXPERT WAS ONE THREAD PER ROW, AND IT IS 15% OF A DECODE

`#60` attributed a walked position's device time and found the shared expert the largest single
stage of a decode: 12.96 ms of 54.3 ms, reading its 1.42 GB/position at 109-113 GB/s while the
routed experts read 352 GB/s on the same kind of bytes. `#20` had already fixed the shape that
causes it -- a thread per whole output row, so a warp's 32 lanes sit a full row width apart and
every lane's load pulls its own sector -- for the plain FP8 GEMV and for the MXFP4 mixture. It had
never been fixed for the two kernels the shared expert actually calls: `dsv41_fp8_block32_swiglu_gate_up`
(fused gate+up) and `dsv41_fp8_block32_down_accumulate`. The checkpoint's shared expert is
`F8_E4M3` `[2304, 5120]` gate and up with `F8_E8M0` `[72, 160]` 32x32 scales, and a `[5120, 2304]`
down, on forty layers, so this one omission was ~1.4 GB of a decoded token's reads.

**The fix.** Two coalesced siblings, `dsv41_fp8_block32_swiglu_gate_up_rows` and
`dsv41_fp8_block32_down_accumulate_rows`: one threadgroup per output row, lane `l` covering blocks
`l, l + lanes, ...`, so a warp's 32 lanes read one contiguous 1 KB window per step, and the
reduction is a `simd_sum` plus (past 32 lanes) one lane-0 pass over threadgroup memory. Both bind
the SAME buffer indices as their one-thread siblings, so the call site swaps the pipeline and the
dispatch shape and nothing else. `fp8GemvWantsWideGrid` already sends both published shapes that
way (`inDim` 5,120 and 2,304 against its 2,048 crossover); the lanes are the new decision, below.

**Isolated A/B, the checkpoint's own shapes** (six distinct weight regions in one command buffer,
best of 8, both forms checked against each other first):

| projection | one thread/row | coalesced 32 lanes | coalesced 64 lanes |
|---|---:|---:|---:|
| fused gate+up 2,304x5,120 | 0.506 ms (47 GB/s) | **0.123 (4.12x, 192 GB/s)** | 0.122 (4.13x, 193 GB/s) |
| down 5,120x2,304 | 0.153 (77 GB/s) | **0.084 (1.81x, 140 GB/s)** | 0.092 (1.66x, 128 GB/s) |

The fused row counts BOTH matrices (gate and up are separate 11.2 MB tiles); the traffic numbers
above include both, which is why they are not the 262-370 GB/s `#20` quoted for a single-matrix
GEMV. 32 lanes wins the down projection by 9% and ties the fused pair, and 32 is one SIMD group, so
the kernel returns without touching threadgroup memory at all -- the same argument that made
`mixtureLaneWidth` 32. `gemvLaneWidth` stays 64 for the plain FP8 GEMV: that kernel measured the
other way round (`#20`) and it reduces one matrix rather than a fused pair.
`DeepSeekV41DevicePlan.fp8FusedLaneWidth` is 32 and the GEMV keeps its own.

**In situ, one runtime, one 128-position stream, the shipping frequency rule**, all three arms under
`recordsWalkStageTiming` so the `sharedExpert` window is measured the same way in each:

| arm | device | mixture device | sharedExpert | misses | read |
|---|---:|---:|---:|---:|---:|
| one-thread (what shipped until now) | 58.0 ms/pos | 25.1 | **12.785 ms** | 2,007 | 35.1 GB |
| coalesced, 32 lanes (shipped) | 49.2 | 16.4 | **4.061 ms** | 1,681 | 29.4 GB |
| coalesced, 64 lanes (control) | 49.4 | 16.6 | 4.235 | 1,562 | 27.3 GB |

**The shared expert goes from the largest stage of a decode to the fifth**, 12.785 -> 4.061
ms/position (3.15x, 8.72 ms saved, 15.0% of the one-thread arm's device time), and the mixture half
of a position goes 25.1 -> 16.4 ms. `routedExperts` (11.97) is the mixture's remaining cost and the
largest stage again; the new order is routedExperts 11.97, queryUp 4.42, outputDown 4.12,
hcAttentionCoefficients 4.09, sharedExpert 4.06.

**The answer was checked three ways, because a kernel swap that is 23x noisier than a lane-width
change has to be explained rather than excused.**

1. *The kernel's own arithmetic on the checkpoint's activations.* `DeepSeekV41ParityProbes` grew
   `ffnNormed` / `sharedIntermediate` / `sharedExpertCaptureLayer`, which copy the FFN's normed
   input and the fused pair's output at one layer (off in production). At layer 0, position 0, over
   a 128-position walk, the two arms feed the shared expert a **bit-identical** normed vector and
   its fused output differs by **2.98e-07 absolute on a 1.85 scale, 1.61e-07 relative** -- FP32
   rounding, nothing else.
2. *A reduction-tree control.* The shipped 32-lane form and the 64-lane form are the same arithmetic
   with two tree shapes; they drift **6.88e-04 relative** at the last position. The one-thread form
   drifts **1.58e-02**, 23.0x more, and that is the OLD form's error: it sums 160 blocks in one
   dependent chain where a threadgroup sums a shallow tree, so its rounding is ~log2(160) times
   worse, and forty layers of mHC Sinkhorn and routed selection amplify it.
3. *The decision itself.* The greedy token is identical at every sampled position of the stream
   (0, 1, 2, 3, 7, 15, 31, 63, 127), so the form changes no answer. The drift is not monotone along
   the stream (7.8e-04 at position 0, 1.9e-04 at 15, 7.1e-03 at 63), which is what amplification of
   a 1e-7 stage difference looks like rather than a bias.

`DeepSeekV41FP8SharedExpertKernelTests` pins both kernels against a HOST rule at the published
shapes, at both lane widths, with a row count off the 32-row scale boundary (65 rows, which needs
the partial scale group), an input shorter than a lane's block count, the token axis, and
`swiglu_limit` (a 0.25 limit must change the answer, so a kernel that dropped the clip fails).

### What it says

- **The shared expert was never a bandwidth problem, it was a dispatch shape**, and it was the same
  defect `#20` named, one kernel family later. The evidence `#20` left behind -- a coalesced A/B of
  `shared_experts.w1` at 3.88x -- was measured on the plain GEMV, which the shared expert does not
  call. This closes that gap: the two kernels it DOES call now have their sibling.
- **What is left of a walked position is the routed mixture and the small per-position kernels.**
  After this: routedExperts 11.97, queryUp 4.42, outputDown 4.12, hcAttentionCoefficients 4.09,
  sharedExpert 4.06, router ~3.6, the rest under 2. The dense GEMVs are at 332-397 GB/s (`#60`) and
  the routed mixture at 352, so the remaining large items are **latency**, not bytes: the mHC
  coefficient pair and the router are per-position dispatches a block amortizes, and 60 ms of a
  ~64 ms wall is still `front wait` (41.3) plus the expert fill (6.3) plus two encodes.
- **32 lanes beat 64 on this pair and lose to it on the plain GEMV**, so the lane width is a property
  of the kernel family, not of the format.

### What it is not (do not re-propose)

- **Not a claim that the one-thread forms were reading the wrong bytes.** They are arithmetically
  equivalent (1.6e-7); they are slow, not wrong. Nothing here changes what an answer is.
- **Not a licence to switch the plain FP8 GEMV to 32 lanes.** It measured 15% the other way (`#20`).
- **Not a reason to add rows per threadgroup.** `#21` measured that a wash on FP8 (0.85-1.10x) with
  a different explanation, and this slice does not touch the question: the fused pair's input is
  one cached 5 KB vector, so sharing it across rows buys nothing.
- **Not an arena, replacement, fill, page-cache or prompt result.** The arena is the shipped one and
  the policy is the shipping rule; the arms' miss counts differ (2,007 vs 1,681 vs 1,562) because a
  rounding-level difference occasionally flips a routing decision, and their walls are not compared
  across arms for that reason plus the fill's page-cache share (23.2 vs 6.3 vs 4.3 ms/position).
  Only the `sharedExpert` and device columns are results.
- **Not a wide-block or batched-verify argument.** A block already amortizes this stage differently
  (`#18`); this is a decode-shaped fix measured on a decode.

### Verification

`testWhatTheCoalescedSharedExpertIsWorthToAWalk` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_shared_stage` plus the class's own `/tmp/dynamoe_dsv41_real_turn`), ~53 s at
`-O`. It asserts the layer-0 kernel equivalence (input bit-identical, fused output < 1e-5), the
greedy token at every sampled position, that both arms' instruments covered all 128 positions, and
that the stage table reports the shared expert. Runs: `/tmp/test61_walk7.log` (the one tabulated
above), with `/tmp/test61_walk4.log` and `/tmp/test61_walk5.log` as reproductions of the shape
(12.73-12.78 -> 4.05-4.26 ms/position, 3.0-3.15x). The isolated form sweep is
`DeepSeekV41PrefillMeasurementTests.testWhatTheCoalescedSharedExpertIsWorth` (opt-in,
`/tmp/dynamoe_dsv41_prefill_measure`, 0.09 s): `/tmp/test61_shared2.log`. The kernels are pinned
without the checkpoint by `DeepSeekV41FP8SharedExpertKernelTests` (6 cases,
`/tmp/test61_kernels.log`).

Production code changed: two kernels, two pipelines, `DeepSeekV41Kernels`, the selection in
`encodeSwiGLU`/`encodeDown` (FP8 only; MXFP4 keeps its existing rule and lane width), the
`fp8FusedLaneWidth` plan constant, and two measurement-only seams on the executor
(`disablesFP8FusedRows`, `fp8FusedLaneWidthOverride`) plus three `DeepSeekV41ParityProbes` fields,
all off in every production path.
## DSV4.1 #62: THE mHC PAIR WAS A ONE-THREAD SINKHORN, NOT A PROJECTION (7.0x)

`#60` attributed a walked position and left the two hyper-connection coefficient stages as the
largest items not at the machine's ceiling: `hcAttentionCoefficients` 4.09 ms/position and
`hcFFNCoefficients` 3.37, i.e. 102 us per layer for a 1.97 MB FP32 weight read, **19 GB/s** where
the dense GEMVs run at 332-397 and the routed experts at 352. `#61` then closed the shared expert,
which ranked above them, so this pair was what was left.

It is TWO dispatches, and `#60` never separated them: `dsv41_hc_project_f32` (one threadgroup per
(slot, logit), 24 of them at a walk) and `dsv41_hc_split_f32` (ONE thread per position, running the
checkpoint's twenty Sinkhorn iterations over a 4x4 matrix). This slice prices each half.

### The run

One runtime through the app's own loading path, one 128-position token stream (the `#52`-`#61`
text, one rotation), the shipping frequency rule, `-O`, the shipped 65.84 GB arena. A warm-up pass
is discarded and the measured arms share the arena, as in `#59`-`#61`. Every arm repeats the SAME
walk; the only difference is which kernel a stage dispatches.

### The projection, isolated

`testWhatOneHyperConnectionProjectionCostsAtThePublishedWidths`, re-run at the checkpoint's own
geometry (hidden 5,120, `hc_mult` 4, so a 20,480-wide reduction and 24 logits) and with **40
distinct weight regions cycled per dispatch** (79 MB, close to the 157 MB a position reads), so the
rate is a read rather than one region's cache hit. Best of five, both forms self-checked against
the host rule before printing; `ms/marginal` is 20 dispatches in one buffer over 20, i.e. the
kernel, and `ms/alone` is one dispatch, i.e. kernel plus launch:

| lanes | one chain, 1 slot | four chains, 1 slot | one chain, 8 slots | four chains, 8 slots |
|---:|---:|---:|---:|---:|
| 64 | 0.091 ms (21 GB/s) | 0.023 (81) | 0.075 | 0.023 |
| 128 | 0.040 (47) | 0.013 (145) | 0.039 | 0.016 |
| 256 | 0.022 (87) | 0.008 (240) | 0.021 | **0.015** |
| 512 | 0.012 (154) | **0.006 (335)** | 0.018 | 0.014 |
| 1,024 | 0.008 (234) | **0.006 (337)** | 0.019 | 0.015 |

Two things fall out. At a WALK's occupancy (one slot = 24 threadgroups) the projection is
0.022 ms one-chain at 256 lanes and 0.006 with four independent accumulator pairs at 512-1,024 —
**3.7x**, and both halves of that are real (four chains alone is 2.75x at 256 lanes; 1,024 lanes
alone is 2.75x with one chain) because each buys memory-level parallelism and the second buys less
once the first is in. At a BLOCK's occupancy (eight slots = 192 threadgroups) four chains are worth
1.4x and the width is flat — 0.015 ms at 256, 0.014 at 512, 0.015 at 1,024, inside a block's
eight-way amortization — so 256 stays. **The lane width is a property of the grid, not of the
kernel**: `hyperConnectionLaneWidth(slots:)` is 1,024 for a walk and 256 for a block.

### The split, in situ

An isolated table cannot price this one honestly, so the split is measured where it runs: a
*repeat* of its dispatch (idempotent — it writes the same `pre`/`post`/`comb` twice, so the
graph's answer cannot move) makes the slope of the stage window the dispatch's in-situ cost. The
same trick on the projection, and on the Sinkhorn's iteration count, separates the chain from the
launch:

| what | slope per extra dispatch |
|---|---:|
| the projection, four chains | 5.0-7.4 us |
| the split, device-memory Sinkhorn | **75.7 us** |
| the split, register Sinkhorn (shipped) | 3.3 us |
| the split at 32 lanes instead of 1 | -0.03 us (nothing) |
| one Sinkhorn iteration instead of 20 (device-memory form) | 16.8 us for the window (a stage is 73.8) |

So the 20-iteration schedule was 3.0 us per iteration and 59 us of a 74 us stage: the 4x4 matrix
lives in a `device float*`, so each of the schedule's ~160 normalization passes is a chain of
dependent 4-byte loads and stores on ONE thread. A threadgroup width of 32 rather than 1 changes
nothing, which is what says it is the chain and not the dispatch.

**The fix.** `dsv41_hc_split_f32_regs` holds the matrix in REGISTERS for the `mult == 4` path
(`dsv41_hc_sinkhorn4`, every sum folded in the general helper's order), and falls back to the
general helper for any other `hc_mult`. `dsv41_hc_project_f32_x4` adds the four accumulator pairs,
and the lane width comes from the plan above.

### In situ, the pair

Milliseconds per walked position; the pair is 80 dispatches (a projection and a split, once per
sub-layer, twice per layer, forty layers); all arms under `recordsWalkStageTiming`:

| arm | hcAttn | hcFFN | pair | device |
|---|---:|---:|---:|---:|
| one chain, 256 lanes, device-memory split (what shipped) | 4.164 | 3.426 | **7.590** | 49.8 |
| four chains, 256, device-memory split | 3.413 | 2.732 | 6.145 | 48.7 |
| one chain, 256, register split | 1.636 | 1.284 | 2.920 | 45.7 |
| four chains, 512, register split | 0.535 | 0.527 | 1.063 | 43.1 |
| four chains, 1,024, register split (shipped) | 0.539 | 0.544 | **1.084** | 43.2 |

**7.590 -> 1.084 ms/position: 7.00x, 6.506 ms saved, 13.1% of the one-chain arm's device time.**
The baseline reproduces `#60`'s 7.48 within half a percent. The two changes are strongly
interacting — each is worth less than half alone (6.145 and 2.920) because the projection is only
visible once the Sinkhorn stops dominating, and the register split only shows its full value once
the projection stops dominating. A stage window is now 13.55 us: ~6 projection, ~4 split, ~4 fixed.

**The production layout agrees**, which is the number that matters: in the arm that commits one
command buffer per half (no per-stage windows, the layout a turn runs), the dense half goes from
**28.626 to 23.044 ms/position, 5.582 ms won**, i.e. 86% of the stage-window figure. Of that,
`Sinkhorn 1 iteration` is only 0.166 ms cheaper than 20 — the chain is gone — and a repeated split
costs 4.12 us, which is what the register form is worth per dispatch in situ.

### The answer was checked three ways

1. *The projection's own arithmetic on real activations.* `DeepSeekV41ParityProbes` grew
   `hyperConnectionCaptureLayer` (the coefficient probes previously copied at EVERY layer, so they
   reported the last layer's values, forty layers of amplification away from the projection being
   compared). At layer 0 the two arms' logits differ by 3.05e-05 absolute of 374.35 (**8.15e-08
   relative**) for the attention sub-layer and 2.67e-05 of 155.79 (**1.71e-07**) for the FFN,
   against a 1e-5 bar.
2. *A reduction-tree control.* The shipped form against the same kernels at 512 lanes drifts
   2.17e-04 relative at the last position; the old form drifts 7.83e-04, 3.6x more, and that is
   the OLD form's error — one dependent chain over 80 blocks where a threadgroup sums a shallow
   tree.
3. *The decision itself.* The greedy token is identical at every sampled position of the stream
   (0, 1, 2, 3, 7, 15, 31, 63, 127).

The split itself is pinned without the checkpoint:
`DeepSeekV41HyperConnectionKernelTests.testTheRegisterResidentSinkhornComputesTheSameSplit` runs
both kernels over the same logits at `mult` 4 (the fast path) and 2 (the fallback) and at 0, 1, 3
and 20 iterations, and requires agreement to 1e-7 on a normalized matrix — the register form is one
ulp from the device form because the seed's product+add rounds once in a register where a store
rounds it twice, and the schedule is contractive after that. The projection's general path keeps
its agreement with `DeepSeekV41HyperConnections` through
`testProjectionAndSplitMatchTheHostRule`, and both forms are checked against the host rule inside
the measurement case before it prints.

### What it says

- **The pair's cost was never the projection.** 76 of the 102 us per layer was a Sinkhorn over
  SIXTEEN numbers, executed by one thread, keeping its matrix in device memory. `#60` called the
  pair "latency-bound rather than bandwidth-bound" and was right for a reason nobody had priced:
  not the dispatch count, not the grid, but ~160 dependent device round trips per position.
- **The projection still needed its own fix, and its cross-over is occupancy.** 24 threadgroups is
  not enough lanes to cover a 2 MB read's latency, so a walk wants the widest threadgroup it can
  get and four independent chains, while a block (192 threadgroups) is already covered and prefers
  256 lanes. `#9`'s lane table was a correct table of a block.
- **The stage table is a table of the graph, and the production layout agrees with it to 86%.** The
  remaining 14% is the per-stage buffer boundary `#60` priced at ~5 us.
- **What is left of a walked position is now the routed mixture and the router.** After this:
  routedExperts 11.95 (at 352 GB/s), queryUp 4.29 and outputDown 4.16 (at 332-397), sharedExpert
  4.06 (at the ceiling since `#61`), **router 3.64** — and the mHC pair, which used to be second,
  is now 1.08 and out of the top five. `router` is the last large item that is not at the ceiling:
  90 us per layer for a 3.93 MB BF16 GEMV plus a serial top-k scan, i.e. 44 GB/s where the same
  bytes in `queryUp` run at 397.

### What it is not (do not re-propose)

- **Not a case for the projection's shape.** Four chains and 1,024 lanes are shipped; the sweep
  above is the record and a block keeps 256 lanes.
- **Not a case for a wider threadgroup on the split.** 32 lanes instead of 1 measured -0.03 us.
  The kernel is one thread because the schedule is serial, and that is now correct rather than
  slow.
- **Not a case for changing `hc_sinkhorn_iters`.** Twenty is the checkpoint's schedule and the
  register form makes the other nineteen cost 0.166 ms/position.
- **Not a dispatch-launch result.** A one-thread dispatch with ONE iteration still costs 16.8 us of
  window, so the launch/fixed part is ~14 us and the chain was 59.
- **Not the per-stage instrument.** The one-buffer-per-half arm wins 5.582 of the 6.506.
- **Not an arena, fill, page-cache or prompt result.** The arena is the shipped one and the policy
  is the shipping rule; the arms' reads differ by up to 8 GB per position because a rounding-level
  change occasionally flips a routing decision (`35.1` on the baseline arm against `27.3` on the
  shipped one), so the walls are NOT compared across arms. Only the stage windows and the device
  halves are results.
- **Not a wide-block or batched-verify argument.** A block already amortized this stage (one
  dispatch per eight positions); this is a decode-shaped fix measured on a decode.

### Verification

`testWhatTheHyperConnectionProjectionAndSplitAreWorthToAWalk` in `DeepSeekV41PublishedTurnTests`,
opt-in (`/tmp/dynamoe_dsv41_hcwalk_stage` plus the class's own `/tmp/dynamoe_dsv41_real_turn`),
~3 min at `-O` because it runs fifteen arms. It asserts the layer-0 coefficient equivalence, the
greedy token at every sampled position, that both instruments covered all 128 positions, and that
the stage table names both coefficient stages. Runs: `/tmp/test62_walk4.log` (the one tabulated
above, 7.590 -> 1.084, dense half 28.626 -> 23.044) and `/tmp/test62_walk5.log`, an independent
reproduction after the doc edits (7.607 -> 1.073, 7.09x, dense half 28.697 -> 23.059, the same
8.15e-08 / 1.71e-07 layer-0 equivalence and the same 7.83e-04 / 2.17e-04 answer control), with
`/tmp/test62_walk2.log` and `/tmp/test62_walk3.log` as the instrument's own development (the repeat
slopes and the Sinkhorn-iteration sweep against the device-memory split).
The isolated form sweep is
`DeepSeekV41PrefillMeasurementTests.testWhatOneHyperConnectionProjectionCostsAtThePublishedWidths`
(opt-in, `/tmp/dynamoe_dsv41_prefill_measure`, 0.16 s): `/tmp/test62_hc2.log`. The kernels are
pinned without the checkpoint by
`DeepSeekV41HyperConnectionKernelTests.testTheRegisterResidentSinkhornComputesTheSameSplit`
(`/tmp/test62_kernels3.log`).

The full `DynaMoETests` target ran 1,491 tests with 95 skipped and 30 failures in 117.5 s on the
`-O` build: every `DeepSeekV41*` suite passed (594 cases, 0 failures), and the 13 failed cases are
exactly the pre-existing environmental set (4 in `DynaMoETests`: help-topic count, Qwen3.8 Flash
Next forward, tool-discover lifecycle, truncated tool-call recovery; 9
`ModelDogfoodAndPrefixCacheTests` prompt/web-fetch/headless-Chrome cases). Run:
`/tmp/test62_suite1.log`.

Production code changed: two kernels (`dsv41_hc_project_f32_x4`, `dsv41_hc_split_f32_regs` plus the
inline `dsv41_hc_sinkhorn4`), two pipelines, `DeepSeekV41DevicePlan.hyperConnectionLaneWidth`, the
selection in `encodeHyperConnectionCoefficients`, and seven measurement-only seams on the executor
(`disablesHyperConnectionX4`, `hyperConnectionLaneWidthOverride`,
`hyperConnectionProjectionRepeats`, `hyperConnectionSplitRepeats`,
`hyperConnectionSplitLanesOverride`, `hyperConnectionSinkhornIterationsOverride`,
`disablesHyperConnectionSplitRegisters`) plus one `DeepSeekV41ParityProbes` field, all off in every
production path.
## DSV4.1 #63: THE ROUTER'S TOP-K WAS ONE THREAD OF TRANSCENDENTALS (3.5x)

`#62` closed the mHC pair and left `router` as the last large stage of a walked position that is not
at a bandwidth ceiling: 3.63 ms/position, 91 us per layer for a 3.93 MB BF16 projection plus a top-k
scan over 384 experts. The stage is two dispatches and nothing had separated them.

### The run

Same harness as `#62`: one runtime, one 128-position stream, the shipping rule, `-O`, and each
dispatch repeated in place — both are idempotent, so the graph's answer cannot move and the slope of
the stage window IS the dispatch's in-situ cost. The repeats also make a free correctness check: the
answer must be **bit-identical** across every arm, and it is.

| arm | router stage |
|---|---:|
| one thread, what shipped | 3.632 ms/position (90.81 us/layer) |
| **lane-parallel top-k (shipped)** | **1.035 (25.88 us/layer)** |
| projection dispatched twice | 1.322 |
| projection dispatched four times | 1.736 |
| top-k dispatched twice | 1.453 |

**The split was 6 us of projection against 72 us of top-k.** The scan is one thread, and each of 384
experts costs three transcendentals (`sqrtsoftplus` = `exp`, `log`, `sqrt`); on top of that, the
top-`k` arrays were declared at their bound (64) and indexed dynamically, so they lived in local
memory. 72 us per layer is 6.6% of a decode's device time for the *selection* of six experts.

**The fix.** `dsv41_router_topk_lanes` gives each position one SIMD group: lane `l` scans experts
`l, l + lanes, ...` (a coalesced read), keeps its own `count` candidates in registers, and publishes
them to threadgroup memory. The merge is the k-way one, run by every lane: each round takes the best
remaining head with `simd_max`, then the winning expert's id with `simd_min` — which resolves a tie
to the LOWER expert id, exactly the rule the one-thread insertion uses, without needing the lower
lane to win. The winner's weight comes back through `simd_sum`, and the selection is staged in the
output buffer's weight slot because lane 0's candidate table is still being read by later rounds.
`DeepSeekV41DevicePlan.routerTopKWantsLanes(topK:)` caps the lane form at a top-k of eight (the
published value is six) and sends anything wider to the one-thread scan rather than spilling the
merge.

The first version of this kernel merged SERIALLY on lane 0 and was only 1.37x (91 -> 67 us/layer):
the merge's lists are dynamically indexed thread arrays, i.e. local memory, and 192 candidates
through them cost 48 us. The k-way merge with shuffles is what makes the whole dispatch 10.6 us.

### What it says

- **The router goes 3.632 -> 1.035 ms/position, 3.51x and 2.597 ms saved**, and it leaves the top
  five: routedExperts 11.85, queryUp 4.32, outputDown 4.13, sharedExpert 4.04, outputUp 3.47.
- **The production layout agrees**: one command buffer per half puts the dense half at 20.424
  ms/position against 23.022 before, i.e. **2.598 ms won**, 99.9% of the stage-window figure.
- **The answer is unchanged to the bit**, which the repeats and the lane form both require and the
  probe asserts at every sampled position. Nothing is re-associated: each expert's score is computed
  by whichever lane owns it, with the same expression, and the selection order is a total order.
- **A dispatch of one SIMD group still costs ~10 us**, of which the projection's 6 is a real weight
  read and the merge is a few hundred cycles. What is left in the stage is a per-dispatch floor that
  a block amortizes over eight positions and a walk cannot.

### What it is not (do not re-propose)

- **Not a case for a wider or narrower SIMD group.** 32 lanes is one warp and the merge's shuffles
  need exactly that; nothing here swept the width.
- **Not a case for changing the routing rule.** The bias ranks and the weight is unbiased, the
  normalization is the checkpoint's `1e-20` and not `norm_eps`, and the tie-break is the lower expert
  id; all of that is `#13`'s rule, untouched, and `DeepSeekV41RouterKernelTests` still pins it.
- **Not a projection result.** The projection is 6 us in situ, already at ~600 GB/s; the stage was
  never the GEMV.
- **Not a reason to parallelize the mHC split the same way.** That kernel is one thread because its
  schedule is serial, and `#62` fixed it by moving the matrix to registers. Here the work IS
  parallel (384 independent scores), which is why the shape is different.
- **Not an arena, fill, page-cache or prompt result.** As in `#62`, the arms' reads differ (35.0
  against 28.8 GB) because a rounding-level change occasionally flips a routing decision, so the
  walls are not compared; only the stage windows and the device halves are results.

### Verification

`testWhatARouterStageIsMadeOfOnAWalk` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_router_stage` plus the class's own `/tmp/dynamoe_dsv41_real_turn`), ~2 min at
`-O`. It asserts that the repeats and the lane form leave the logits bit-identical at the last
position and at every sampled position, and that the stage table reports the router. Runs:
`/tmp/test63_router3.log` (the one tabulated above) and `/tmp/test63_router4.log`, an independent
reproduction after the merge's dead-code cleanup (3.637 -> 1.038, 3.50x, dense half 23.015 ->
20.410), with `/tmp/test63_router2.log` (the serial-merge form, 1.37x) and `/tmp/test63_router1.log`
(the instrument's first cut, both dispatches on the one-thread kernels) as its development.

The kernel is pinned without the checkpoint by
`DeepSeekV41RouterKernelTests.testTheLaneFormOfTheTopKIsTheOneThreadFormsResult`, which runs both
kernels over the same logits at the published shape (384 experts, top-k 6), at eight slots, and at a
5-expert shape shorter than a SIMD group, with a deliberate equal-score pair so the tie-break has to
agree — and requires **exact** equality of both the indices and the weights, because nothing in the
lane form re-associates a sum. `/tmp/test63_kernels4.log`.

Production code changed: one kernel (`dsv41_router_topk_lanes`), one pipeline,
`DeepSeekV41DevicePlan.routerTopKWantsLanes`, the selection in `encodeRouter`, and three
measurement-only seams on the executor (`routerProjectionRepeats`, `routerTopKRepeats`,
`disablesRouterTopKLanes`), all off in every production path.
## DSV4.1 #64: THE INDEXER'S DRAIN IS THE GPU, SO A DEVICE SELECTION WOULD BUY NOTHING

After `#61`-`#63` the walked position's device time is **40.35 ms/position** (24.14 dense, 16.21
mixture) against `#60`'s 58.45, and the stage ordering has changed underneath it: routedExperts
11.82 (29.3%, 352 GB/s), queryUp 4.33, outputDown 4.13, sharedExpert 4.03, outputUp 3.45 -- all at
their DRAM ceilings -- then a tail in which nothing is above 1.8 ms: indexer 1.77, attention 1.39,
queryDown 1.15, latentProject 1.13, router 1.04, ropes 1.80 between them, norms 1.48, the mHC pair
1.07, the stream mixes 0.88, engram 0.63.

The one item left that is WALL rather than device is the indexer: 20 of the 40 layers own a
compressed index, and each one stops the dense half for a score readback because the row list the
attention gathers is a HOST value (`readIndexSelections`). `#60` measured that drain at 12.5
ms/position and this run at 9.4, of a ~80 ms wall.

So the obvious slice is to compute the selection on the device and remove the round trip. This
prices the decision before it is taken, by splitting the bucket rather than guessing:
`DeepSeekV41WalkTiming.indexerSelectNanoseconds` times the host's own work inside the round trip
(readback + `candidateBlocks` + `selectedRows`), and `describe()` prints it as `indexer host X of
that wait`.

**The answer is 0.0 ms/position.** The entire 9.4 is `settlePending` waiting for the GPU, i.e. the
layer's own dense work up to and including the score kernel -- 470 us per index-source layer against
a 515 us-per-layer dense half, so the drain is not overhead at all, it is the work that the front
wait would otherwise have covered. A device-side selection would remove the ~0.0-0.1 ms of host CPU
and nothing else, while reimplementing a selection rule that `#28`-`#32` pinned. **Do not port it
for this reason.** What it would still buy is *ordering*: the drain serializes the host against the
GPU, and the pickup after the queue empties is `#22`'s ~0.5-1.2 ms/layer -- but that cost is inside
the 9.4 and cannot be separated from it by this instrument, and it is bounded by the same GPU work.

The instrument is the whole change. Runs: `/tmp/test64_walk2.log` (indexer wait 9.4, indexer host
0.0) with `/tmp/test64_walk1.log` as the fresh whole-walk attribution taken the run before (device
40.35, stage table above; the two runs' walls differ by 1.3 ms for page-cache reasons and the
indexer wait is 9.4 in both).
## DSV4.1 #65: A DEFERRED-BIND EXPERT HALF DOES NOT START UNDER THE FILL, SO THE HANDOFF IS THE MACHINE, NOT THE COMMIT

`#64` closed the indexer with a negative and left the walk's structure as the last lever. Every layer
stops the queue for the routing readback, then streams its experts, and only then hands the GPU the
half that reads them. The GPU is idle for the whole fill, and by the time the half is committed the
queue has been empty long enough that its own window opens late: `mixtureLateStartNanoseconds` is
11.6-12.5 ms/position in the warm regime below, and `#22`/`#50` had already shown that number is not
the buffer's shape, its bindings, its resource list, or CPU-dirtied memory.

The one change that can move that pickup under the fill is a **deferred bind**. The expert half binds
only what the arena PLACED, so it can be encoded and committed before its bytes exist, gated on a
`MTLSharedEvent` the read signals. `DeepSeekV41ExpertStreamer.placeResident` splits placement from the
read to make that possible (`ensureResident` is literally the two composed), and
`DeepSeekV41MetalForward.gatesMixtureOnExpertFill` makes the walk use it.

### The answer: the gate buys nothing, and it is priced twice

**On the driver** (`DeepSeekV41CommandStartCostTests.testWhatAnEventGateSavesOnACommitIntoAnEmptyQueue`,
`/tmp/test65_gate4.log`). Two arms at each gap: `idle` is today's shape (queue empty, host sleeps inside
a `pread`-shaped gap, then commits), `gated` commits first and signals when the same sleep ends. Both
are timed from the moment the DATA was ready — the commit for `idle`, the signal for `gated`.

| gap | dispatches | idle ready->window | gated ready->window | window |
|---|---|---|---|---|
| 0 ms | 1 | 0.083 ms | 0.099 | 0.007 |
| 0 ms | 20 | 0.106 | 0.108 | 0.071 |
| 0 ms | 64 | 0.103 | 0.102 | 0.229 |
| 0.6 ms | 1 | **0.122** | **0.112** | 0.009 |
| 0.6 ms | 20 | **0.161** | **0.163** | 0.074 |
| 0.6 ms | 64 | **0.155** | **0.144** | 0.238 |

**On the graph** (`DeepSeekV41PublishedTurnTests.testWhatAGatedMixtureCommitSavesOnAWalk`,
`/tmp/test65_gatewalk3.log`). One runtime, one arena, the 128-position stream `#52`-`#64` used, four
arms in a rotation so the page cache cannot be read as a win:

| arm | wall | device | misses | read | fill | mixture late start |
|---|---|---|---|---|---|---|
| control, arm 1 (cold) | 82.4 | 36.9 | 2001 | 35.0 GB | 21.7 | 12.5 |
| gated, arm 1 | 64.4 | 36.7 | 1646 | 28.8 GB | 5.0 | 11.6 |
| gated, arm 2 | 63.8 | 36.6 | 1559 | 27.3 GB | 4.3 | 11.7 |
| control, arm 2 | **63.5** | 36.5 | **1559** | **27.3 GB** | 4.2 | **11.6** |

The two SECOND arms are the comparison: they run back to back, so they read the same 1,559 misses and
the same 27.3 GB, and the wall differs by 0.2 ms — in the CONTROL's favour. The gated arms' apparent
win is entirely the cold arm's 35.0 GB. And the answer is bit-identical at every sampled position of
all four arms, so the gate is correctly ordered (see the two traps below); it simply has nothing to
recover.

### What it says

- **The pickup is not a handover delay.** A gate changes WHICH call hands the work over and not how
  long the machine takes to begin it. An ungated commit into an empty queue and a gated buffer whose
  event has just fired cost the same 0.11-0.16 ms for a trivial buffer, and the same 0.3 ms/layer on
  the real graph. `#50`'s reading — a property of the machine after it has been reading, not of the
  buffer — survives the one structural change that could have explained it away.
- **Do not re-propose the deferred bind, an event gate, or a second command queue for the expert
  half.** All three change when the host hands the work over; the number that matters is the machine's
  cost to start it, and this slice measured it from both sides.
- **What is left is the dispatch floor, and it is device time.** The sweep shows the START is flat in
  dispatch count (0.083-0.163 ms for 1, 20 or 64 dispatches) while the WINDOW grows by ~3.5 us per
  dispatch (0.007 -> 0.071 -> 0.229). A walked position issues ~20 dispatches per expert half and
  ~1,080 overall, so ~3.8 ms/position of the 36.6 ms of device time is per-dispatch overhead. That is
  a FUSION question (fewer, larger kernels), not a scheduling one.
- The warm position's remaining wall is 63.5 ms: device 36.5, mixture late start 11.6, indexer wait
  9.4 (GPU drain, `#64`), fill 4.2, encodes 4.6, front late start 0.7.

### Two traps, both of which make a gate look good

1. **A wait encoded after the dispatches does not hold them.** `encodeWaitForEvent` orders everything
   encoded AFTER it in the buffer, so the first cut of the production change appended the wait at the
   commit and every dispatch in front of it ran free. The walk caught it as a RACE (three positions of
   128 off, 10-88% relative on the logits, `/tmp/test65_gatewalk1.log`), which is why the probe asserts
   BIT-identical logits rather than a tolerance: a scheduling change that moves a logit is a bug.
2. **A shared event's value never goes back down.** An arm that restarts its values at 1 finds its gate
   already satisfied and measures the UNGATED path under the gated arm's name — which is how the first
   version of the driver's sweep "measured" a gate that started work 0.001 ms after the signal. The
   event needs one monotonic counter for the whole test (`/tmp/test65_gate2.log` and
   `/tmp/test65_gate3.log` are the two wrong versions, kept by name so the mistake is recognisable).

### Verification

`/tmp/test65_gate4.log` (driver sweep, monotonic event), `/tmp/test65_gatewalk3.log` (the walk, four
arms, bit-identical logits), `/tmp/test65_gatewalk1.log` (the racy first cut). Production default is
`gatesMixtureOnExpertFill = false`: the seam and its probe stay as the baseline any future deferred-bind
idea has to beat, and `DeepSeekV41ExpertStreamer.placeResident` is the split that idea would need.

## DSV4.1 #66: THE DISPATCH FLOOR IS 2,072 DISPATCHES AT ~1.07 us EACH, SO FUSION CAN RETURN AT MOST 1.06 ms

`#65` closed the deferred-bind expert half and handed the next slice a lever with two inferences
behind it: "fuse kernels rather than reschedule them", priced at ~1,080 dispatches/position (27
pipeline stages x 40 layers) and ~3.5 us per dispatch of WINDOW growth from a driver sweep. The
first is a stage count, not a dispatch count — `.routedExperts` issues one clear plus two per active
expert — and the second was measured with trivial dispatches going into an EMPTY queue, where nothing
can hide their cost. So both were measured in the graph, on the 128-position stream `#52`-`#65` used.

**The count is 2,071.8 dispatches per walked position**, identical in every arm and in both runs
(265,195 per 128 positions), i.e. 1.9x the estimate. `countsDispatches` counts at the encoder call
itself — every dispatch in the file now goes through a `dispatch(encoder, threads:/threadgroups:)`
wrapper — so this is what the GPU is handed, not what the stage switch happens to name.

| stage | dispatches/position | share | device ms | of which dispatch |
|---|---|---|---|---|
| routedExperts | 520.0 | 25.1% | 11.831 | 0.56 |
| outputDown | 320.0 | 15.4% | 4.130 | 0.34 |
| attention | 117.9 | 5.7% | 1.394 | 0.13 |
| hcFFNCoefficients | 80.0 | 3.9% | 0.540 | 0.09 |
| sharedExpert | 80.0 | 3.9% | 4.030 | 0.09 |
| writeFFNStreams | 80.0 | 3.9% | 0.186 | 0.09 |
| router | 80.0 | 3.9% | 1.038 | 0.09 |
| hcAttentionCoefficients | 80.0 | 3.9% | 0.533 | 0.09 |
| indexer | 59.0 | 2.8% | 1.780 | 0.06 |
| 16 stages at 40.0 each | 640.0 | 30.9% | 17.117 | 0.68 |
| embedding / engram / head | 15.0 | 0.7% | 0.631 | 0.02 |

(`routedExperts` is 1 clear + 2 x 6 experts; `outputDown` is one dispatch per block-diagonal group,
8 per layer; `attention` is ~3 per layer because the compressed layers also gather and store. The
16 stages at exactly 40.0 are the ones that issue one dispatch per layer.)

**The marginal dispatch costs ~1.07 us**, not 3.5. `injectedDispatchRepeats` adds K clears of a
4-float scratch buffer to every stage's own command buffer, so an injected arm's census is the
control's plus exactly 27 x 40 x K — asserted, not assumed — and the slope across three arms is the
per-dispatch cost:

| arm | wall ms/position | device ms/position | misses | read |
|---|---|---|---|---|
| control | 63.9 | 36.7 | 1,559 | 27.3 GB |
| +1 clear/stage (3,151.8 dispatches) | 65.0 | 38.0 | 1,559 | 27.3 GB |
| +4 clears/stage (6,391.8 dispatches) | 68.9 | 41.0 | 1,559 | 27.3 GB |
| control again | 63.9 | 36.7 | 1,559 | 27.3 GB |

That is **1.03 and 1.17 us of wall, 1.18 and 0.98 us of device** per dispatch (a second run:
0.83/1.18 wall, 1.16/0.98 device; control 63.7 both times). The two arms agree, the repeated control
lands on the first to 0.1 ms, and every arm returns bit-identical logits because the extra dispatches
write a buffer nobody reads.

**So the whole dispatch floor is 2,072 x 1.07 us = 2.2 ms** of a 36.7 ms device position and 2.1 ms
of a 63.7 ms wall, 3.3%. Fusing every stage of every layer into a SINGLE dispatch — the bound no
kernel can beat — returns **1.06 ms**, 1.7% of a walked position. The driver sweep's 3.5 us would
have predicted 7.25 ms and a redesign; in the graph a marginal dispatch costs a third of that
because it hides under the work already in its buffer, which is the same lesson `#65` learned about
a commit: the queue's own state is what the isolated instrument cannot see.

What that reframes: a warm walked position's 36.7 ms of device time is 34.5 ms of WORK and 2.2 of
dispatch. There is nothing structural left in the kernel count. The remaining wall (63.7 against
36.7) is the buckets `#65` named and none of them is a dispatch: mixture late start 11.6, indexer
drain 9.4 (which `#64` showed is the GPU's own work), fill ~4, encodes ~4.6.

**What it is not.** It is not a reason to fuse the eight `outputDown` band dispatches: `#11` tried
exactly that, measured 29% slower in isolation and no change in place, and the census now explains
why — the whole stage's dispatch overhead is 0.34 ms and the bands are 4.1 ms of weight traffic.
It is not a reason to fuse the mHC pair or the router either: 0.09 ms each. And it does not say a
fusion can never pay — it says a fusion has to remove WORK (bytes read, a redundant residual read,
a barrier-separated pass over the same vector), because the dispatches themselves are worth 1.7%.

**Two traps, both of which produced a plausible win.** (1) The first run compared a control that ran
FIRST, so it paid the arena's cold fill (2,001 misses, 35.0 GB) against arms at 1,559 misses and
27.3 GB, and the injected arms "won" 20 ms by reading 7 GB less; `#65` recorded the same shape. The
probe now runs three discarded warm-ups, asserts equal miss counts and equal bytes, and repeats the
control last. (2) The per-dispatch figure was first divided by `count * 1_000` instead of
`count`, which printed 0.001 us and made the floor look like 0.02 ms — a units error that agreed
with the "dispatches are free" hypothesis, so nothing looked wrong.

Verification: `/tmp/test66_census2.log` and `/tmp/test66_census3.log` (two full runs, arms and stage
table above), `/tmp/test66_census1.log` (the cold-arm first cut). The instrument lives behind
`countsDispatches` / `resetDispatchCensus()` and `injectedDispatchRepeats`, both off in every
production path; the dispatch wrapper is a load and a compare, and the control's 63.7-63.9 ms/position
matches `#65`'s 63.5, so the census costs nothing measurable.

## DSV4.1 #67: THE EXPERT STREAM IS A FLOOR, NOT A LEVER — AND EVERY MILLISECOND OF IT IS NOW NAMED

`#66` closed the dispatch floor and left the ranking it could not settle: a warm walked position is
36.7 ms of device work, a mixture late start of 11.6, an indexer drain of 9.4, an encode of 4.6 and a
fill of **4.2 ms**. `#22` and `#49` had measured a live turn's fill at 67 ms and 556 MB/position with
86-93 ms of hand-off idle behind it, and `#59` at 301-307 MB/position. Four probes, four fills, and
every later slice quotes whichever one it happened to run. The handover routed the next slice to the
expert read volume and told it to re-measure the live regime first; SubQ agreed
(`expert_read_volume` 0.814, confidence 0.751; "price it before building it" 0.917).

### Question

What does a live turn's expert fill actually cost, in the regime a live turn is in, and is any part of
it a lever rather than the disk's own floor?

### The trap in every earlier number

A probe that re-walks one stream is warm in **two** independent ways, and a turn's text is warm in
neither:

1. **The arena.** `#59` showed a cold pass is 95.3% compulsory first touches while a warm pass
   re-fetches 83-100% of its misses. The same bytes, with different causes, and only the second is a
   replacement question.
2. **Slot wiring.** `#57`'s shipping policy `mlock`s a slot when it is allocated, so the pass that
   creates the arena pays for it and the passes after it do not.

`#66`'s 4.2 ms was the fifth pass of seven over one stream, with three discarded warm-ups in front of
it: its expert bytes came out of the page cache and its slots were already wired. It is the *warmest*
number the journal has, quoted as if it were a live one.

### The run

One runtime through the app's own loading path, the `#52`-`#66` 128-position rotated stream, the
shipping `leastFrequentlyUsed` rule and the shipping wiring grant (`wireArena: nil`), `-O`, one
process. Four arms, each recording every position's layer records (`recordsLayerGaps`, which carry the
fill's own bytes, misses, resolve, `pread` and placement seconds beside the four intervals that tile a
position), with the arena's miss classifier on:

| arm | ms/pos | dense | hand-off | expert | reads | misses | 1st touch | GB/s | evictions |
|---|---|---|---|---|---|---|---|---|---|
| **cold** (reset arena, unread bytes) | **121.8** | 17.9 | 79.5 | 17.4 | 701 MB | 39.1 | **95.8%** | 15.1 | 1,248 |
| **warm** (same stream again) | **82.2** | 17.6 | 41.3 | 16.4 | 280 MB | 15.6 | 0.0% | 13.2 | 2,001 |
| **floor** (eight-token cycle, already resident) | **54.8** | 17.4 | 14.6 | 16.0 | 101 MB | 5.6 | 0.0% | 46.6 | 723 |

The hand-off decomposes exactly, which is what makes the rest of this section possible:

| arm | routing | allocate | wire | resolve | pread | place | idle | encode | unattributed |
|---|---|---|---|---|---|---|---|---|---|
| cold | 0.0 | 0.3 | **11.1** | 0.3 | **45.5** | 0.5 | **16.5** | 2.3 | 3.0 |
| warm | 0.0 | 0.0 | 0.0 | 0.1 | **20.8** | 0.2 | 14.8 | 2.3 | 3.0 |
| floor | 0.0 | 0.0 | 0.0 | 0.1 | 2.1 | 0.1 | 7.0 | 2.2 | 3.0 |

So **hiding the entire expert fill would take a cold position from 121.8 to 54.8 ms — 67.1 ms, 55% —
and a warm one from 82.2 to 54.8, 27.4 ms, 33%.** That is the largest single prize this port has
measured. It is also, component by component, already closed.

### What it says

- **The read volume is compulsory and the rate is the ceiling.** 701 MB/position in 45.5 ms is 15.1
  GB/s, the machine's own limit (13.6-15.2 in `#56`-`#59`), and 95.8% of a cold pass's misses are
  pairs this arena never held. `#21` found the same at 53 positions, `#59` at 128, and `#58` priced the
  capacity that would hold them (81.19 GB wired left 0 GB free and fell to 6.4 GB/s). Fewer bytes is
  closed on every side: the arena (`#58`, `#59`), the read form, the fan-out, `F_NOCACHE` and the disk
  (`#51`), and the fact that all six ranges of an expert are read by a kernel that wants them.
- **The 11.1 ms of wiring is the shipping policy paying for itself in the arm that creates the
  slots.** 0.28 ms per miss, ~1.1 ms per newly wired 17.9 MB slot, i.e. `mlock` page-touching at
  ~15.7 GB/s. `#57` measured 17-18 ms/position for a pass that wires all 65.84 GB and showed the
  alternative: P4, wired withdrawn, was **222.4 ms/position with 4.6M decompressions**. This is a cost
  that pays back on the next pass and it is not a lever.
- **Resolving a miss is nearly free — 0.006-0.011 ms each.** The bucket that looked like "catalog
  lookups per expert" (a 12.0 ms cold arm, 120x the warm one) was wiring, not resolution. Worth
  stating because the opposite was the plausible reading of the first cut.
- **The idle does not scale with the bytes.** 16.5 / 14.8 / 7.0 ms against 45.5 / 20.8 / 2.1 of read.
  `#49` fit 1.2-2.3x of the byte time; here the read-heavy part is at most 0.2, and most of the bucket
  is a ~7 ms per-position floor present with no I/O at all. `#48`-`#50` closed the causes (not the
  submission shape, not CPU cache writeback, not the resource count) and `#65` closed the one remedy
  that would have removed it (the deferred gate, measured at matched read volume).
- **The remaining GPU work is at ceilings.** dense 17.9 + expert halves 17.4 = 35.3 ms of the 121.8,
  and `#60`-`#66` put every stage of both halves within tens of a percent of the best measured GEMV
  rate. Against a floor of 54.8 ms, a cold position is 35.3 ms of work under 74 ms of streaming.

**So the expert stream is a floor, and the only lever of any size left is to stop paying it once per
position.** That is batching, and it is priced on both sides: `#22` measured a block of eight at 70.9
ms/position against a 153.3 ms walk (2.16x, 7.6x less I/O), and `#26` measured the width sweep that
makes it unusable — acceptance roughly doubles at widths two and three but a narrow verify is only
0.83-0.89x a walk per position, and a width-2 round emits 2.05 tokens for 420 ms against 399 ms of
walking. Note how that arithmetic moves with this slice: our *cold* walk is 121.8, so a verify of eight
would need ~47% acceptance to break even, and `#25` measured 31% at width five. Do not re-propose a
verify of width 2-3 without a new acceptance number, and do not re-propose the read itself.

### What it is not (do not re-propose)

- **Not a bigger or differently-sized arena.** `#58`'s B1 and `#59`'s replacement A/B, re-confirmed
  here: 1,248 evictions in the cold arm and 2,001 in the warm one, all refetches, with capacity
  already priced.
- **Not dropping or partially applying slot wiring.** `#56`-`#58`, and P4's 222.4 ms above.
- **Not the resolve path.** 0.006-0.011 ms per miss; the tempting 12 ms was `mlock`.
- **Not the hand-off idle alone.** `#48`-`#50` closed its causes and `#65` closed the deferred gate.
- **Not a page-cache argument for the warm arm.** 13.2 GB/s is disk rate, not cache rate; the floor
  arm is the one served from cache, at 46.6 GB/s, which is what makes it a floor for time rather than
  for bytes.

### Verification

`testWhatALiveTurnsExpertFillCosts` in `DeepSeekV41PublishedTurnTests`, opt-in
(`/tmp/dynamoe_dsv41_live_fill`, plus the class's own `/tmp/dynamoe_dsv41_real_turn`), 51 s at `-O`.
Runs: `/tmp/test67_live3.log` (the cold/warm/quiet cut), `/tmp/test67_live4.log` (the decomposed
cut), `/tmp/test67_live5.log` (the quoted one), `/tmp/test67_live6.log` (the committed one). Four runs
of the same arms: cold **121.8-125.0**, warm **82.2-83.4**, floor **54.8-55.4** ms/position, i.e. 2.6%
apart end to end and 0.5% apart on the floor. Quote the range: the decomposition reproduces to ~0.3 ms
a bucket (wire 10.4-11.1, pread 45.5-48.6, idle 16.5-17.2) while the cold arm's absolute figure moves
with the machine, which is why the prize is stated as 55-56% of it and not as 67.1 ms. It asserts the cold and warm arms return bit-identical
logits, that a cold arm is >90% compulsory first touches, that the floor arm misses <25% as often and
reads at a higher rate, that the dense halves and expert halves move <2 ms between arms, and that
every position's layer records tile it and that the miss composition covers every miss.

The instrument is `DeepSeekV41ExpertStreamer.prepareTiming` (new: wall time inside `placeResident`
short of the arena's own placement, billed through a `defer` so a throw is counted too) plus two new
`DeepSeekV41LayerGapSample` fields, `fillAllocateSeconds` and `fillWireSeconds`, which split it into
per-new-slot and per-miss halves. All are behind `recordsLayerGaps`, off in every production path.

A fully zero-byte floor arm is not reachable under the shipping rule and the assertion says so
instead of pretending otherwise: `leastFrequentlyUsed`'s aging halves every count once a
capacity-worth of misses has gone by, so even an eight-token cycle's own set is retired and re-read —
101 MB at 46.6 GB/s, 2.3 ms. That is a floor for the position's time without being a floor for bytes.

The full `DynaMoETests` target ran 1,497 tests with 99 skipped and 33 failures in 124.1 s on the final
`-O` build (`/tmp/test67_suite2.log`). Zero `DeepSeekV41*` failures. 30 are the known pre-existing
environmental set (5 XCTest failures in 4 `DynaMoETests` cases, 25 in 9 `ModelDogfoodAndPrefixCacheTests`
cases); the other 3 are `ServerQueueLogicTests`, which pass in isolation (13/13,
`/tmp/test67_queue.log`) and are the timing flake `#58` recorded.

## DSV4.1 #68: THE BAR FOR A DRAFT IS 45-75% EVEN IF DRAFTING WERE FREE, AND #22's 2.16x WAS A WARM-ARENA ARTIFACT

#67 closed the expert stream as a floor and left one lever standing: a decode step cannot batch, so
speculative decoding is the only structure that stops paying the stream once per position. #22 had
priced it at 2.16x per position and 7.6x less I/O, which put the bar at ~47% acceptance over eight
drafts; #25 and #26 measured the checkpoint's own drafter at 31% and then at 50-55% when the block was
narrowed, and closed it. This slice asks the one question that ordering never had to answer -- what
acceptance would a draft need, AT ALL, against the walk a live turn actually pays -- and the answer
invalidates the premise under the 47%: a block does not read fewer bytes, and the bar is 45-75% before
a drafter has spent a millisecond.

### The run

One runtime, the published checkpoint, the shipping arena rule, `arena.reset()` before every arm so
every arm is compulsory (100% first touches, asserted) and no arm inherits another's pairs or its
page-cache lines. Arms of 48 positions, rotated between rounds, two measured rounds after a discarded
warm-up: a WALK arm and a VERIFY arm at widths 2, 3, 5 and 8, each on its own disjoint slice of the
token pool. `testWhatAcceptanceADraftWouldHaveToReachToPayForItsVerify`, logs `/tmp/test68_bar1.log`
and `/tmp/test68_bar2.log`.

| arm | ms/position | MB/position | misses/position | first touch |
|---|---|---|---|---|
| walk | 96.8-100.9 | 876-928 | 48.8-51.8 | 100% |
| w2 | 89.1-92.7 | 967-973 | 53.9-54.3 | 100% |
| w3 | 75.7-79.0 | 911-939 | 50.8-52.4 | 100% |
| w5 | 79.8-81.2 | 966-972 | 53.9-54.2 | 100% |
| w8 | 83.1-84.4 | 999-1022 | 55.7-57.0 | 100% |

Two runs, and the byte columns reproduce exactly while the walls move ~1%: w2 0.95/0.96x a walk, w3
0.78/0.79x, w5 0.82/0.84x, w8 0.86/0.87x. That is the whole saving batching has -- 4.5 to 22.2 ms of a
~98 ms position, one host round trip per layer that a block does not pay per position.

### The 7.6x was never there

Against a walk's 928 MB, the blocks read **1.01-1.10x as much**: w2 973 MB, w3 939, w5 972, w8 1,022.
Not fewer. `#22`'s "one expert union per block instead of one set per position" is not what a resident
arena does, because the arena is already the union: a walked position hits ~80% of the time (48.8-51.8
misses against ~280 requests) since the positions before it in the same turn read the same experts, so
the set a block fills up front is very nearly the set the walks would have read anyway -- and the block
pays for its union's whole width in ONE burst, which is why `#26` measured width three reading 55% MORE
per position than walking it.

Where `#22`'s 1,357 MB came from is visible in its own test: it walks runs 0, 1 and 2 first, then
blocks runs **1, 2 and 3** -- two of the three block runs are runs the walk arm had just loaded. The
block byte count is the walk's own bytes served from the arena, which is precisely the trap `#22`
recorded for a different fixture and then walked into. Its 2.16x is therefore mostly warm-arena, and
every projection resting on it -- including the ~47% bar for a verify of eight -- is withdrawn.

### The bar, at last without a premise

A round costs `draft + width * verify * W` and emits `1 + a` tokens against `(1 + a) * W` of walking,
so it pays when

    a >= draft/W + width * verify_ratio - 1

and with a FREE drafter that is `width * ratio - 1` accepted drafts:

| width | verify ratio | saves | free bar | at #26's drafting | #26 measured |
|---|---|---|---|---|---|
| 2 | 0.95-0.96 | 4.5 ms/pos | **45-46%** | 82% (73 ms) | 52% |
| 3 | 0.78-0.79 | 22.2 | **45-46%** | 71% (78.5 ms) | 38% |
| 5 | 0.82-0.84 | 17.9 | **62-64%** | 93% (154.5 ms) | 31% |
| 8 | 0.86-0.87 | 14.3 | **73-75%** | not measured | not measured |

Read down the free column: a drafter that costs NOTHING has to be right 45-46% of the time at width
two or three and 62-75% at five or eight, because a verify only ever buys back the per-layer round
trip and the batching that would have bought more is worth nothing here. Width two is the only
configuration with any headroom, and the checkpoint's own drafter delivers 52% there -- ABOVE the free
bar. What kills it is its own forward: 73 ms of a 98 ms walk position. Break-even at width two needs
`draft <= (1.05 - 0.91) * W`, i.e. **~14 ms per round**, which is 4-6x cheaper than the 56-90 ms `#26`
measured and still 2x cheaper than the ~25-30 ms that slice projected if the draft mixture were made
fully arena-resident. At widths three and five the round cannot break even at ANY draft cost, since the
free bar already exceeds the acceptance the checkpoint delivers.

### The one drafter that costs nothing

So the remaining candidate is a drafter with no model in it: the continuation of the nearest earlier
occurrence of the turn's own last n tokens. It is scored EXACTLY rather than approximately -- the
protocol advances a round by `width` whatever was accepted (which is what `dsparkAcceptance` does, so
the two drafters face the same target path) and acceptance always stops at the first disagreement, so
every position before it is the model's own greedy path and a single walked continuation answers the
question offline. The continuation is a real 96-token greedy generation from a rendered prompt, not
fixture text.

| width | 2-token lookup | 3-token lookup | free bar |
|---|---|---|---|
| 2 | 3/12 (25%), drafted in 6 of 48 rounds | 1/6 (17%), 3 of 48 | 46% |
| 3 | 4/12 (33%), 4 of 32 | 0/3 (0%), 1 of 32 | 45% |
| 5 | 0/5 (0%), 1 of 19 | 0/5 (0%), 1 of 19 | 62% |
| 8 | 0/0, 0 of 12 | 0/0, 0 of 12 | 75% |

On ordinary prose it does not have a draft to offer: it proposed one in 11 of 111 rounds across the
four widths, and the ones it proposed were right a quarter to a third of the time. Its expected
accepted tokens per round is a fortieth of what width two needs. Reproducible to the token across both
runs.

### What it says

Speculative decoding is closed on this port at every width, and now for a reason that does not depend
on any one drafter: the verify's whole prize is the per-layer host round trip (0.78-0.96x), the bar
that prize sets is 45-75% with no drafting cost at all, and no drafter measured here -- the
checkpoint's own chain at 31-52%, or a free n-gram lookup at 0-33% on prose -- comes within reach
except at width two, where the checkpoint's 52% clears a 46% bar and its 73 ms of drafting moves the
bar to 82%.

The relationship is worth stating because it generalises: **the faster the walk, the higher the bar**,
since `draft/W` grows as `W` falls and the round-trip saving it is chasing shrinks with it. `#67`
pushed `W` down to 121.8 ms/position cold and 55 to the floor; a verify that could not pay at 153-194
therefore cannot pay at 122, and at the 55 ms floor there is nothing to batch at all because the fill
has already left the critical path. Speculative decoding and the expert stream are the same lever
pointing in opposite directions.

### What it is not (do not re-propose)

- **A batched verify as a way to read fewer expert bytes.** It does not read fewer; it reads
  1.01-1.10x as many, because the arena already amortises the stream across positions. Do not quote
  `#22`'s 7.6x again.
- **A width-2 verify on the strength of 52% acceptance.** 52% clears the free bar by 6 points and the
  drafter's own 56-90 ms of forward turns that into a 82% bar. The number that would have to move is
  the draft cost, to under ~14 ms/round, and `#26`'s best projection for a fully resident mixture is
  double that.
- **Widths 3, 5 and 8 at any drafting cost.** The free bar exceeds the measured acceptance, so no
  optimisation of the drafter helps.
- **An n-gram or prompt-lookup drafter on prose.** 11 drafts in 111 rounds; it is a no-op that would
  still have to be verified. (It may be worth revisiting only for text that repeats by construction,
  and then it must be priced on that text and not on this one.)
- **Per-position logits in `finishBlock` and cache rewind for a rejected suffix.** Still exactly what
  `#25` and `#26` said: the losing half of a design the measurement rejects.
- Everything `#22`, `#26` and `#67` already closed: the fill's read form, fan-out, `F_NOCACHE`, disk
  and page cache (`#51`); arena capacity and replacement (`#58`/`#59`); slot wiring and its withdrawal
  (`#56`/`#57`); the hand-off idle, deferred bind, event gate and second command queue (`#48`-`#50`,
  `#65`, `#67`); dispatch count and fusion (`#66`); the resolve path (`#67`); the dense GEMVs and the
  wider block (`#60`); the shared expert, mHC and router kernels (`#61`-`#63`).

### Verification

Ranges over `/tmp/test68_bar1.log` and `/tmp/test68_bar2.log`: the walls agree to ~1% between runs and
the byte columns to the megabyte, and the width ratios (0.95/0.96, 0.78/0.79, 0.82/0.84, 0.86/0.87) are
the quoted figures. The case asserts that every arm is >80% compulsory first touches (so no arm read
another's pairs), that a round records exactly one walk arm and one arm per width, and that the
required-acceptance arithmetic reproduces the printed bar. It reports the bar at the measured walk and
quotes `#26`'s two drafter numbers rather than re-running them, because the draft width sizes buffers at
construction and a sweep needs one runtime per width.

The regime differs from `#26`'s in one deliberate way and it is worth knowing which is which: `#26`
walked a 14-position prefix and then timed ONE round against it, which is a round whose experts are
almost all cold, while these arms are 48 positions long so each block is filled against the arena the
turn so far left -- the ~80% hit rate `#67` measured on a live turn. That is why width three reads
0.78x here and 1.08x there: neither is wrong, and the one that answers "does a verify pay in a live
turn" is this one.

## DSV4.1 #69: WHAT A WHOLE TURN COSTS, END TO END — AND WHY THE PREFILL BAR NEVER MOVED

### Question

Every slice from `#60` on measured a PIECE of a decoded position: one stage of the dense
half (`#60`-`#63`), the indexer's wait (`#64`), the deferred bind (`#65`), a dispatch
(`#66`), the expert fill (`#67`), a verify that turned out not to pay (`#68`). The only
end-to-end figure anywhere in this journal is `#22`'s 205-324 ms/position, and it was
taken before the dense pin, the coalesced FP8/MXFP4 GEMV family, the arena's replacement
rule and `#67`'s re-measurement of its own regime -- so nothing since is comparable with
it, or with each other. There was no common reference.

There was also a user-visible symptom of the same gap. In the chat UI the prefill status
did not appear on the first prompt at all, and on a follow-up turn it appeared as a single
late count. Both came from the same place: a whole prompt is ONE call into the executor.

### The run

`DeepSeekV41PublishedTurnTests.testWhatAWholeTurnActuallyCosts` runs
`DeepSeekV41TurnRunner` exactly as `CompletionEngine.launchTurn` does -- one chunked
prefill from position 0, then a walk per drawn token -- over the published checkpoint,
and reuses the runtime across turns so the arena's behaviour is the app's. Two prompt
shapes, each followed by its own continuation (the prompt the model saw plus what it
wrote, which is what the app's pin proposes):

| turn | prompt | reused | run | prefill | steps | decode | ms/position | first token |
|---|---|---|---|---|---|---|---|---|
| 1 thinking off ("hey") | 5 | 0 | 5 | 1.41 s (281 ms/pos) | 1 | 1.29-1.32 s | 161.6-164.4 | 1.41 s |
| 2 its continuation | 13 | 12 | 1 | 0.14 s (136-138) | 1 | 0.97-0.98 s | 138.5-139.7 | 0.14 s |
| 3 app default (thinking on, high) | 31 | 0 | 31 | 5.79-5.85 s (187-189) | 4 (8,8,8,7) | 1.40-1.46 s | 175.2-182.6 | 5.79-5.85 s |
| 4 its continuation | 39 | 38 | 1 | 0.18 s (177-182) | 1 | 1.32-1.36 s | 164.9-169.7 | 0.18 s |

The decode, decomposed by the executor's own buckets, per walked position (host buckets
are mutually exclusive; the device windows are a SUBSET of the host waits, so they are
reported as "of that host time" rather than added):

| turn | wall | host | encode | front wait | expert fill | indexer wait | mixture encode | device windows | outside the walk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 184.7-187.9 | 180.8-184.2 | 3.1 | 52.4-52.9 | 112.0-112.6 | 10.4-13.8 | 2.4 | 42.0-43.8 | 3.7-3.8 |
| 3 | 200.2-208.7 | 195.4-204.4 | 3.5-3.6 | 79.5-88.9 | 90.5-92.3 | 17.2-18.8 | 2.8 | 44.0-47.3 | 4.2-4.8 |

Four turns read 81.35 GB into 4,646 slots, ended 3,760/3,760 resident at 62.0% hit over
12,224 acquisitions, and the streamer served it at 11.7-11.8 GB/s.

### What it says

**1. The composer's blank prefill was plumbing, and it is fixed.** A whole prompt goes
through one `prefillChunk` call, and the turn loop reported progress only AFTER that call
returned (`onPrefill(newPositions, newPositions)`), while the app's own `done % 8 == 0`
guard dropped every report a short turn produced. So the first prompt emitted nothing
until the prefill was over and then jumped to a finished count, and a resumed turn -- one
or two positions -- emitted exactly one line. Both are the same defect: a report with no
granularity behind it. `DeepSeekV41PositionForward` now carries
`prefillChunk(_:onProgress:)`; the device executor fires it once per `runBlock` (a block
is one command buffer, so that is the honest granularity), the walk-based executors
report once at the end, and the app emits a 0/N line before the call and then a
time-throttled, rate-bearing status (`Prefilling 24/31 (+38 resident) • 5.3 pos/s`). The
cost of the fix is one closure call per block.

**2. A 31-token default prompt is 5.8 seconds of nothing.** That is the number the user
stares at, and it is now on the screen as it happens. `user waits 5.79-5.85 s for the
first token` is the column to watch for any future prefill work.

**3. The batched prefill buys essentially nothing on a prompt of this size, measured
through the app rather than in an arm.** Its per-position cost is 187-189 ms against a
cold decode position's 175-183 ms, i.e. **1.02-1.07x**, which is `#53`'s 1.05x at 128
positions arriving from the app's own path. The block form's win was never per-position
expert traffic (`#68` withdrew that premise); it is the per-layer host round trip, and a
31-position prompt has four of them to amortise. There is nothing here to speed up by
batching harder.

**4. A live decode position is the expert fill plus the routing round trip, and nothing
else.** Of turn 3's ~205 ms: 90-92 ms is the fill (44%), 80-89 ms is the front wait
(39-43%, of which ~44-47 ms is device windows), 17-19 ms is the indexer's second wait,
and 6 ms is host encoding. **3.4-5.2 ms is outside the walk altogether** -- the head, the
sampling, the text decode and the stream hop together -- so no amount of work on the
non-walk path is worth measuring. This is `#67`'s conclusion (the fill is the floor) and
`#22`'s conclusion (the readback empties the queue and the next commit is picked up late)
arriving through a third instrument, and it is the first time either has been shown on
the path the chat UI actually runs.

**5. The resumed turn is nearly free, which is the prefix cache working.** 12 of 13 and
38 of 39 prompt positions were skipped, so a follow-up's first token arrives in 0.14-0.18 s
against a first prompt's 5.8 s, and that single walked position costs the same as any
other (136-182 ms).

### What it is not (do not re-propose)

- **Not a contradiction of `#67`.** Its arms were 48-position walks against a 14-position
  prefix; these turns start at an empty arena and fill it as they go (62% hit by the end
  of the fourth). The 162-183 ms/position here is what a turn pays from cold, which is
  why it sits above `#67`'s 121.8. Both are right and neither is the other's regime.
- **Not a case for a different prefill shape.** 1.02-1.07x is `#53`'s number; a wider
  block, a narrower one or a device-side pool all have to beat that, and `#53` already
  measured the first two.
- **Not a target in the 3-5 ms outside the walk.** The head, sampling and text decode
  together are 2% of a position.
- Everything `#60`-`#68` closed: the dense GEMVs and the wider block, the shared expert,
  mHC and router kernels, the indexer's host work, the deferred bind, the dispatch count
  and fusion, the expert read and arena, the hand-off idle, and speculative decoding at
  every width.

### Verification

`/tmp/test69_turn_run1.log` and `/tmp/test69_turn_run2.log` (two runs of the same case in
the same process order): every wall agrees to 1-3%, every byte column is IDENTICAL
(24,563.7 / 32,112.1 / 74,282.7 / 83,301.3 MB), and the generated text is token-identical
in both. The case asserts, per turn, that prefill reports are monotonic, that the last one
equals the positions the turn actually ran, and that the count is one per block when
chunked (3's four reports for 31 positions at a block width of 8) and one per position
when walked (a resumed turn's single position) -- which is the composer's contract, and
was the thing that was broken. Run with the class marker and the opt-in file:

```bash
touch /tmp/dynamoe_dsv41_real_turn && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild test-without-building -project DynaMoE.xcodeproj -scheme DynaMoE \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath /tmp/dd_dsv41_61_opt \
  -parallel-testing-enabled NO \
  -only-testing:DynaMoETests/DeepSeekV41PublishedTurnTests/testWhatAWholeTurnActuallyCosts
```

The progress plumbing is pinned without any model as well:
`DeepSeekV41TurnLoopTests.testAWholePromptReportsProgressBlockByBlock` (a scripted
forward that reports per block, asserting the counts climb and end on the chunk) and
`DeepSeekV41DeviceGraphParityTests.testAPrefillReportsEveryBlockItRuns` (the real device
executor over 9 positions at a 4-wide block, asserting `[4, 8, 9]`).
## DSV4.1 #70: A PREFILL POSITION IS ONLY CHEAPER THAN A DECODE ONE WHEN ITS WORKING SET FITS — AND THE WIDTH THAT WINS FOLLOWS THAT, NOT THE LAYOUT

### Question

`#69` measured the app's own turn and found a prefill position costs what a decode position
costs (187-281 against 138-183 ms), on a layout that pays ONE dense-half wait, ONE routing
readback, ONE indexer split and ONE union fill per layer for a whole block of eight. Eight
positions for the price of one should be a large win; the numbers say the layout is worth
2-5%, and nothing since `#53` has measured the SHAPE of that. Every prefill table in this
journal is also older than the dozen optimisations from `#56` to `#67`, all of which tuned
the WALK a turn resumes into rather than the block a prompt is prefilled with.

Routing asked which prefill slice was worth the next machine hour and ranked a width-scale
attribution at 0.84 (confidence 0.80) over instrumenting the block's encode (0.07), a
16-slot block form (0.02) and batching the per-slot dispatches (0.06). It also put "is the
width already saturated at eight" at 0.80, which turned out to be the one thing this slice
overturned.

### What was added

Nothing on a shipped path until the last section; three instrument repairs and one probe:

- `DeepSeekV41BlockTiming` gained `indexerWaitNanoseconds`. `encodeBlockStage` submits the
  dense half in two parts around the indexer, because the row selection is a HOST value,
  and that drain plus readback was being counted as `denseEncode`. It made a block read
  28.6 ms of "encode" per position against a walked position's 3.5. The walk has kept the
  same column separate since `#48`; the two layouts' tables can now be read side by side.
  `describe()` prints it and includes it in its own whole-vs-parts control.
- The dispatch census learned the block path. `encodeBlockStage` never labelled its stages
  (so every block dispatch was attributed to whatever the walk set last) and
  `dispatchCensusPositions` never counted a block's positions.
- `prefillChunk(_:blockWidth:)` — a public overload naming the block width, so a sweep can
  compare widths without changing what production runs at. `prefillChunk`'s private form
  takes it and falls back to `DeepSeekV41DevicePlan.prefillBlockWidth`.
- `testWhereATextPrefixsPerPositionCostGoesOnThePublishedCheckpoint` (opt-in,
  `/tmp/dynamoe_dsv41_prefix_time`): the `#52`-`#59` 128-token stream, the walk and widths
  1, 2, 4 and 8, three rounds with the arm order rotated, arena reset before every arm, and
  the same instruments on each. Then the same comparison at 31 positions (the app's own
  DEFAULT prompt, which `#69` measured at 2.9 s of prefill) and at 64 (to bracket the
  crossover), then the two widths again with the arena left WARM, which a follow-up turn
  meets and a cold arm cannot see.

### The width curve: four wins on a long prompt, eight wins on a short one

Cold arena, steady state (rounds 2-3, round 1 discarded), ms per position. The first table
is one process that had the machine to itself and swept all three lengths; the second is
how three earlier processes reproduced it, because a prefill arm is cheap enough to repeat
and every earlier prefill table in this journal was a single pass:

| positions | walk | width 1 | width 2 | width 4 | width 8 |
|---|---|---|---|---|---|
| 128 | 266.7 | 258.9 | 235.1 | **222.9** | 264.2 |
| 64 | 306.5 | - | - | **278.4** | 345.8 |
| 31 | 372.1 | - | - | 422.6 | **399.9** |

| process | 128 walk / w1 / w2 / w4 / w8 | 64 walk / w4 / w8 | 31 walk / w4 / w8 |
|---|---|---|---|
| run 1 | 273.7 / 266.1 / 250.1 / 232.3 / 269.1 | - | - |
| run 2 | 276.7 / 282.8 / 249.7 / 242.4 / 270.9 | - | 390.8 / 411.6 / 406.5 |
| run 3 | 280.7 / 273.5 / 253.9 / 238.2 / 263.7 | - | 402.3 / 463.7 / 418.6 |

Every arm at a given length reads the same bytes (89,971-90,545 MB over 5,018-5,050 misses
at 128; 67,541 MB over 3,767 at 64; 47,352 MB over 2,641 at 31), which is what makes the
columns comparable at all. Run 4's 64-position arms are left out on purpose: they overlapped
builds and a parity suite and read every arm 10-15% slow while keeping the order, which is
the failure mode the rotation rule exists to catch and cannot always catch on its own.

- **Eight is not the optimum, and it is barely better than walking.** At 128 positions width
  eight is 1.01-1.05x the walk across four processes — `#53`'s 1.04-1.05 reproduced exactly —
  while width four is 1.18-1.20x. A width-1 block, which batches nothing, is 2-4% ahead of
  the walk, so the block path's own overhead is small and the curve is about width.
- **The curve is not monotonic, and it turns over between 31 and 64 positions.** At 31
  positions width eight beats width four by 6-12% and both are slightly BEHIND the walk
  (0.88-0.93x); at 64 four is 1.10x ahead of the walk and 1.24x ahead of eight; at 128 four
  is 1.19-1.20x ahead of the walk and 1.18x ahead of eight.
- **The crossover is the working set, not the length and not the warmness.** A prompt's
  routed `(layer, expert)` pairs saturate as it lengthens — 2,641 at 31 positions, 3,767 at
  64, 5,045 at 128 — against an arena of 3,760 slots, so 64 positions is exactly where a
  prompt stops fitting. Before that, a prefill is pure per-layer overhead and the widest
  block amortises it best. After it, every block pays a fill and a wider block costs more
  per layer than its extra positions buy back.
- **A warm arena does not change which width wins, only how much is left to win.** With the
  arena left as the previous pass filled it, at 31 positions (0 MB read, pure compute) width
  eight is still ahead, 1.05-1.19x across two processes (83.1 against 86.9, and 69.4 against
  84.4 ms/position), and at 128 positions even a warm pass still reads 56 GB and width four
  is 1.09-1.12x ahead (189.7 against 212.8, and 184.1 against 201.1). That matters because
  it is the arm that killed the obvious guess: a short prompt is warm, so eight is right
  there, and a long prompt cannot be warm, so four is right there.

### Where a block's position actually goes

Same 128-position stream, round 2 of run 3, per position in ms (this is the whole answer
to "why does batching not amortise"):

| bucket | walk | w1 | w2 | w4 | w8 |
|---|---|---|---|---|---|
| **wall** | 259.5 | 248.3 | 230.8 | **218.5** | 265.6 |
| dense encode (host) | 3.0 | 3.00 | 1.91 | 1.38 | 0.98 |
| indexer wait | 33.8 | 29.4 | 30.0 | 30.2 | 35.4 |
| front wait | 125.7 | 117.0 | 107.6 | 99.8 | 129.6 |
| expert fill | 94.2 | 89.6 | 84.5 | 81.1 | 87.0 |
| mixture encode (host) | 2.8 | 2.95 | 2.37 | 1.91 | 1.64 |
| **device, both halves** | 51.9 | 46.5 | 43.7 | 42.1 | 50.7 |
| - dense half | 28.1 | 23.1 | 19.4 | **17.3** | 20.9 |
| - mixture half | 23.8 | 23.3 | 24.3 | 24.9 | **29.8** |
| mixture late start | 105.2 | 93.5 | 88.3 | 82.8 | 103.0 |
| dispatches / position | 2,072 | 2,072 | 1,630 | 1,410 | 1,299 |

- **The dense weight read is the only thing that amortises.** Its device half goes 28.1 →
  17.3 ms per position from walk to width four, −39%, because the dense backbone is resident
  and read once per block instead of once per position. That is the entire prize: ~11 ms of
  260, and half of it goes back at width eight (20.9).
- **The mixture half gets WORSE with width, monotonically.** 23.8 walked, 23.3 at width one,
  24.3 at two, 24.9 at four, 29.8 at eight. Batching the mixture was supposed to be its win
  (`#52` measured the union fill carrying 1.1-2.4 experts where a walk's carries 0.3); on
  this checkpoint the per-position expert computation is what a mixture costs, and a wider
  block spreads it over more slots per submission without making it cheaper.
- **The fill is the largest single bucket at every width and it barely moves.** 94.2 ms
  walked, 81.1 at width four, 87.0 at eight, on identical bytes: 88-90 GB per 128-position
  prefill at 7.7-8.5 GB/s. `#67` measured this machine's ceiling at 13-14 GB/s, so a cold
  text prefill sits at ~60% of it, and the layout moves it by 14%.
- **The hand-off stops improving past width four.** The mixture's late start falls 105 →
  82.8 from walk to four and then rises to 103.0 at eight: past four the block's own mixture
  takes long enough that the next layer's dense half is waiting for it again.
- **Every bucket tiles.** The block's host buckets sum to 96-98% of its own wall in all six
  regimes, the same control `#50` and `#52` used.

### The block path's own dispatch census, for the first time

At width eight over 128 positions: 166,299 dispatches, 1,299 per position, and 56.2% of them
are the router (93,440). Then `writeLatent` 9.5%, `indexer` 4.5%, and eight stages at 3.1%
each (`attnNorm`, `queryNorm`, `latentNorm`, `ffnNorm`, `mixAttentionStreams`,
`mixFFNStreams`, `writeAttentionStreams`, `outputDown`).

- **The per-slot stages are emitted once per position, so the count per position falls only
  37% across an eight-fold width** (2,072 → 1,299). A width-1 block and a walk emit exactly
  the same 2,072, which is the block path's overhead and nothing else.
- **It buys nothing to care.** The encode bucket at width eight is 0.98 ms per position and
  at width one 3.00, i.e. those dispatches cost ~0.5 µs each, already under `#66`'s 1.07 µs
  floor. Batching the per-slot dispatches is not where a prefill's time is; the journal's
  `#15` note that "the gathers are the remaining candidates and unmeasured" is answered:
  measured, they are 9.5% of a dispatch count that is itself 1% of a position.

### What it means

- **Batching a text prefill is worth 0-20% per position, not 2x**, and where the optimum
  sits is decided by whether the prompt's expert working set fits the arena.
- **The shipped width was eight, which is right for the app's own prompt and ~19% off for a
  long one.** `prefillBlockWidth` now returns the workspace's width below
  `prefillFillBound` (64) positions and four above it. The threshold is where the measured
  working set stops fitting (3,767 distinct pairs at 64 against 3,760 slots), so it is a
  derivation rather than a tuned constant, and it is `min`-capped by the workspace so a
  narrow-context fixture cannot ask for a block its buffers do not hold.
- **A prefill position is a decode position because the mixture is a decode position.** The
  dense half does amortise; the experts do not, and they are 33-37% of the wall plus most of
  the device half.
- **The fill is at ~60% of the disk's ceiling on a cold text prefill** (7.7-8.5 against
  `#67`'s 13-14 GB/s). `#67` named every millisecond of a walked position's fill and found
  it at the ceiling; a block's union fill, at 3.9-7.8 misses per fill, does not reach it.
  That is a ~31 ms/position gap at width four and it is the largest single open number this
  slice leaves.

### What it is not (do not re-propose)

- Not the bytes: every width reads within 0.6% of the same expert bytes on the same stream.
- Not the dispatch count or fusion (`#66`): the block path's encode is 1.0-1.4 ms/position
  at every width past one.
- Not a block wider than the workspace's eight: measured, it is 10% worse at 128 positions
  and only wins when the working set fits.
- Not the arena's capacity or eviction policy (`#21`, `#58`, `#59`), not a second command
  queue or a deferred bind (`#65`), not the indexer (`#64`), and not the mixture's kernels
  (`#60`-`#63`).
- Not warm-versus-cold as the discriminator for width: measured both ways at both ends of
  the threshold, and the sign is the same in both regimes at each length.

### Verification

`testWhereATextPrefixsPerPositionCostGoesOnThePublishedCheckpoint` is opt-in and skips
without `/tmp/dynamoe_dsv41_prefix_time`; it draws no slot `MTLBuffer` beyond what a normal
prefill allocates and changes no shipped path. Five processes over ~100 minutes:
`/tmp/test70_prefix_run1.log` (128 positions, three rounds), `run2.log` (128 + 31),
`run3.log` (128 + 31 + the warm arms), `run4.log` (adds 64, overlapped other work) and
`final2.log` (the three-length sweep every table above is taken from).

The word "clean" matters here, and one run had to be killed for it. A second process left
running from an earlier invocation put two runtimes over one machine, which is 2 x 74 GB
against 128 and 52 GB of swap: the affected sweep read 1,040 ms/position against a normal
176-266, i.e. 4-6x, and every arm kept its ORDER. A rotated sweep only protects against an
effect that moves with arm position; it cannot see a constant load, so quote one only from a
process that had the machine and say so.

Two unit bugs were caught by the harness's own controls rather than by the numbers looking
wrong, and both are worth carrying: the arms identified themselves by `width`, which is 1
for a walk, so the walk's census assertion read the block-1 arm's; and the parity bar was
5e-4, taken from `#33`'s block-versus-walk comparison at 16 positions, where legitimate
widths on this stream disagree by 5.3e-3 at 128 positions and 1.1e-2 at 31. The bar is 5e-2
with an exact argmax check beside it, and a skipped stage or a slot reading another
position's rows is orders of magnitude larger than either.

Shipped path: one rule (`prefillBlockWidth`) plus the seam it is decided through. Pinned by
`DeepSeekV41DeviceGraphParityTests.testThePrefillWidthFollowsTheWorkingSet` (the rule at
both ends of the threshold, and the report cadence it produces driven through a real
prefill) and `testEveryPositionOfABlockAttendsOverItsOwnRows` now names its width
explicitly, because its subject is a block as wide as the workspace and not what a prompt
ships at. `DeepSeekV41DeviceGraphParityTests` 37 tests pass; `DeepSeekV41TurnLoopTests` 26
pass; `testWhatAWholeTurnActuallyCosts` is unchanged where it counts: the app's 31-token prompt
still prefills in FOUR reports at width eight, with the same 74,282.7 MB read, the same
12,224 arena acquisitions at the same 62.0% hit rate and the same generated text as `#69`.
Its wall moved from 5.79-5.85 s to 8.57 s and back to 6.69 s on a build that ran the same
work at width four, because that turn is 74 GB of cold expert read and the disk's rate is
what varies (11,368 against 9,995 MB/s reported by the streamer on the same bytes). Quote
that turn's wall as a range or not at all.

## DSV4.1 #71: THE PREFILL FILL IS AT THE DISK'S CEILING — #70's 31 ms GAP WAS THE HARNESS'S UNWIRED ARENA, AND THE WIDTH THRESHOLD IT SHIPPED DOES NOT SURVIVE

`#70` closed by naming one open number: at the best prefill width a 128-position prompt
spends 81.1 ms/position filling the expert arena, 703 MB/position at **7.7-8.5 GB/s**
against this machine's measured 13-14 GB/s ceiling, i.e. ~31 ms/position of apparent
headroom, and the largest single open figure the journal had. The same journal had already
measured a cold WALKED position — same bytes, same misses — at **15.1 GB/s** (`#67`), and
`#70` itself recorded the app's own turn moving 5.79 s to 8.57 s to 6.69 s on
byte-identical reads. Two processes, one workload, 2x apart, and nobody had asked which
process was normal.

### Question

Is a prefill's fill slower than a walk's, or is the bucket not the read, or is the number
a property of the process that measured it?

### What was added

Nothing on a shipped path except the rule at the end. One probe, and the seam it needs:

- `testWhatRegimeAPrefillsFillIsMeasuredInOnThePublishedCheckpoint` (opt-in,
  `/tmp/dynamoe_dsv41_fill_regime`): one runtime through the app's own loading path, the
  `#52`-`#70` 128-token stream, `recordsFillSamples` on, the walk and widths 4 and 8 at 128
  and at 31 positions, three rounds with the arm order rotated and round 1 discarded, the
  arena dropped before every arm so every arm reads the same bytes.
- It bills each arm TWICE. The outer bucket is `DeepSeekV41BlockTiming.expertFillNanoseconds`
  (or the walk's), the wall around `ensureResident` — which is what `#70` divided its bytes
  by. The inner one is the streamer's own clocks: `readMisses`'s wall (`timing.reading`),
  placement (`placementTiming`), the rest of `placeResident` (`prepareTiming`, which
  CONTAINS allocation and wiring), the true bytes of the ranges, and one
  `DeepSeekV41FillSample` per fill (`misses`, `shards`, `ranges`, true `bytes`, `seconds`),
  which also gives the rate per fill SHAPE.
- And it CALIBRATES: the same `ensureResident` the prefill calls, cold, eight fills of six
  experts and eight of twenty-four, from distinct pairs so nothing is read twice. The only
  honest ceiling for an in-situ rate is one the same process measured.

### The answer is the arena's wiring, and it is not close

The first cut ran the harness's default runtime — `makeRuntime(wireArena: false)` — and got
`#70`'s numbers. The same probe with the app's own runtime, which wires every slot
(`wireArena: nil`), gets:

| 128 positions | wall | fill bucket | read | GB/s | bytes | misses |
|---|---|---|---|---|---|---|
| walk | 132.9 | 58.4 | 57.8 | **12.2** | 90,455 MB | 5,045 |
| width 4 | **114.0** | 54.1 | 53.5 | **13.2** | 90,025 MB | 5,021 |
| width 8 | 113.6 | 55.0 | 54.5 | **12.9** | 89,971 MB | 5,018 |

| 31 positions | wall | fill bucket | read | GB/s | bytes | misses |
|---|---|---|---|---|---|---|
| walk | 207.5 | 122.8 | 121.7 | **12.6** | 47,352 MB | 2,641 |
| width 4 | 182.5 | 114.7 | 113.7 | **13.4** | 47,352 MB | 2,641 |
| width 8 | 187.0 | 112.2 | 111.4 | **13.7** | 47,406 MB | 2,644 |

(Those are process 1's walls. Three processes measured the same arms and the fill read rate
is stable across them — 12.2-13.9 GB/s at every width and length — while the walls move
5-30% with the machine, which is why the width comparison below is per process.)

Both sides are the same bytes and the same misses, so the only difference between these
rows and `#70`'s is the regime:

| regime | prefill fill rate (128 positions) | same-process calibration |
|---|---|---|
| harness, `wireArena: false` (`#70`, and a fresh process of this probe) | 5.9-8.5 GB/s | — |
| app, `wireArena: nil` (this probe, three processes) | **12.2-13.9 GB/s** | **13.2-17.0 GB/s** at 6 experts, **14.8-17.0** at 24 |

`#56` had already measured it and said so: the same bytes into the same slots run at
**4.6-6.7 GB/s unwired against 13.0 wired**, because an unwired slot is a page the
compressor reconstructs after every pass that rewrote it. Every prefill table in this
journal since `#20` was taken unwired — the harness default exists so `#20`-`#56` compare
like with like — and the app wires. `#67`'s 15.1 was the shipping regime; `#70`'s 7.7-8.5
was not.

**So the 31 ms/position is withdrawn, and with the wiring on the bucket is 99% read:**
at width four on the 128-position stream, placement is 0.29 ms/position, preparation 0.28
(and allocation and wiring inside it are 0.00, because the slots exist by then), and 0.13
is unaccounted — 0.70 ms of 54.1. The fill's bookkeeping is not a lever because there is
none to find. At 77-87% of what the same process can do in an isolated fill, with 100% of
its misses compulsory (5,021 misses against 5,045 distinct pairs), a prefill's expert read
is the disk doing its job.

### Where a wired prefill position actually goes

128 positions, round 2, per position, ms — the same table `#70` printed, in the regime the
app runs:

| bucket | walk | width 4 | width 8 |
|---|---|---|---|
| **wall** | 132.9 | **114.0** | **113.6** |
| dense encode (host) | 2.8 | 1.38 | 1.10 |
| indexer wait | 11.9 | 11.24 | 11.44 |
| front wait | 52.5 | 43.87 | 43.73 |
| expert fill | 59.0 | 53.48 | 54.93 |
| mixture encode (host) | 2.4 | 1.75 | 1.51 |
| **device, both halves** | 39.7 | 31.78 | 33.66 |
| - dense half | 21.7 | **13.24** | 13.67 |
| - mixture half | 18.0 | 18.54 | 19.99 |
| mixture late start | 22.9 | 19.3 | 17.9 |

`#75` retargets one row: `front wait` is one `settlePending`, so 63-65% of it is the previous layer's
mixture still on the queue -- already billed in `- mixture half` above -- and only the remainder is the
dense half's own drain. Read the two together, never add them.

Read it against `#70`'s unwired table and every bucket moves, because the fill is what the
whole layer was waiting on: the indexer split falls 30.2 → 11.2 (it drains the previous
mixture, which is now cheap), the mixture's late start 82.8 → 19.3 (the queue is idle for a
1.3 ms fill instead of a 2 ms one), and the dense half's device window 17.3 → 13.2. What
does NOT move is the shape: the fill is still the largest single bucket, the mixture's
device half is still the second, and the dense half is still the only thing batching
amortises.

### The width curve in the app's regime, and the rule

Three processes, cold arena, arm order rotated, round 1 discarded, mean of the measured
rounds, ms per position. The winner column is the two widths against each other:

| positions | process | walk | width 4 | width 8 | winner |
|---|---|---|---|---|---|
| 128 | 1 | 133.0 | 114.0 | **113.6** | eight, 0.4% |
| 128 | 2 | 120.2 | 102.1 | **97.0** | eight, 5.3% |
| 128 | 3 | 121.5 | 102.6 | **99.7** | eight, 2.9% |
| 64 | 3 | 154.5 | 133.7 | 133.4 | tied, 0.3% |
| 31 | 1 | 207.5 | **182.5** | 187.0 | four, 2.5% |
| 31 | 2 | 252.2 | 183.8 | **180.5** | eight, 1.8% |
| 31 | 3 | 190.4 | 173.6 | **165.7** | eight, 4.8% |

Both widths beat the walk everywhere, by 1.10-1.22x, which is the same prize `#70` found
for width four. Between the widths the whole effect is 0.3-5.3%, eight is ahead in five of
the six comparisons, and the one loss is the app's own 31-token prompt in process 1, the
process whose machine state was slowest (its walk is 207.5 against 190.4 in process 3).

`#70` shipped a threshold on the strength of the opposite reading: the workspace width below
64 positions and four above, on the theory that a short prompt's routed experts fit the
arena so per-layer overhead dominates and the widest block amortises it best. In the
shipping regime that theory has nothing left to explain, and the threshold's crossover is
where the two widths are closest (64 positions: 133.7 against 133.4). **So the threshold is
gone and `prefillBlockWidth` returns the workspace's width at every prompt length** — what
the engine ran before `#70`, which is the conservative direction to revert in, and what the
measurements support (eight or tied in five of six). `prefillFillBound` is deleted; the
function stays as the seam a future sweep can move without touching `prefillChunk`.

The first cut of this slice shipped a CONSTANT FOUR on process 1's numbers alone, where
four was 2.5% ahead at 31 positions and 0.4% behind at 128. Processes 2 and 3 reversed both
readings. That is the honest lesson about a 2-5% effect measured on one machine: n=1 is not
a measurement, and the only defensible rule at this margin is the one with the best central
tendency across processes.

### What it means for the app

`#69`'s live-turn numbers are this regime's counterpart and are ~2x pessimistic. Re-run
with the app's arena, the same 31-token default prompt prefills in **6.67 s** (215.3
ms/position over 31, four reports at the shipped width) and the whole turn is **7.74 s of
wall for 74,283 MB read** with 133.4 ms per decoded token, against `#69`'s 8.57 s unwired
on the same bytes; a second shipping process at width four prefilled in 5.80 s and turned in
6.85 s, which is the same 5-30% machine variance the width table shows. The 5-token `"hey"`
prompt prefills in 1.80 s and turns in 3.29 s. The streamer's own report over those four
turns is **14,084 MB/s** with the arena wired 65.84 of 65.84 GB, which is the same rate the
probe measured and the same policy `#57` shipped.

The lesson that generalises past this slice: `makeRuntime` defaults to the unwired regime,
so **a test that prices the app must pass `wireArena: nil`**, and `#69`'s turn case now
does. Anything else is a number about the harness.

### What it is not (do not re-propose)

- **Not the fill's bookkeeping.** 0.70 ms of a 54.1 ms bucket at width four: placement,
  preparation, allocation, wiring and resolve are all accounted for and all under 1.5%.
- **Not the bytes, the arena or the layout.** 5,021 misses against 5,045 distinct pairs on
  a 128-position prompt is 100% compulsory; there is no refetch left for capacity, policy
  or a different block shape to remove.
- **Not the wiring policy.** It is the shipping default, it is worth 2x on every fill, and
  `#57` already showed the alternative (P4, unwired, 222.4 ms/position).
- **Not a narrower block.** The two widths are within 0.3-5.3% in the shipping regime and
  eight leads in five of six comparisons, so a rule that narrows a prompt is buying noise.
  The workspace's eight is both the ceiling on the buffers and the running width.
- **Not a second arena in one process.** The unwired arm, run after a wired one in the same
  process, collapsed to 0.9 GB/s (3,108 ms/position): that is 2 x 65.84 GB against 128 GB,
  not a regime. The probe runs one regime per process for that reason.
- **Not the disk.** 12.2-13.7 GB/s in situ against 15.2-15.9 isolated is the machine, and
  the streamer's own four-turn report agrees at 13,479 MB/s.

### Verification

`testWhatRegimeAPrefillsFillIsMeasuredInOnThePublishedCheckpoint` is opt-in and draws no
slot `MTLBuffer` beyond what a normal prefill allocates. Runs: `/tmp/test71_fill1.log` (the
first cut, unwired, which is where 5.9-6.9 GB/s came from), `/tmp/test71_regime1.log` (the
shipping regime every table above is taken from, and the contaminated unwired arm it
replaced) and `/tmp/test71_turn2.log` (the live turn). It asserts, per arm: that the fill
samples cover every miss the bucket was charged for, that the read is shorter than the
bucket that contains it, that the ranges and the slot count agree on the bytes within 5%,
that every arm read the same bytes across rounds, and that widths 4 and 8 land on the
walk's token (relative 5e-2, exact argmax). The calibration asserts it read every pair it
asked for.

Shipped: `DeepSeekV41DevicePlan.prefillBlockWidth` returns the workspace's width at every
prompt length, and `prefillFillBound` is deleted. Pinned by
`DeepSeekV41DeviceGraphParityTests.testThePrefillWidthIsTheWorkspaceWidth`, which checks
the rule at seven prompt lengths and at a workspace narrower than the ceiling, and drives a
real prefill at 3, 9, 64 and 128 positions to assert the report cadence.
`DeepSeekV41DeviceGraphParityTests` 37 tests pass and `DeepSeekV41TurnLoopTests` 26 pass.
`testWhatAWholeTurnActuallyCosts` passes with its cadence expectation derived from the
shipped rule instead of from `chunkSlots` directly; the indirection is what caught the
first cut, which asserted a block count the engine no longer produced.

## DSV4.1 #72: THE MIXTURE HALF DOES NOT AMORTISE WITH WIDTH BECAUSE BOTH ITS BIG STAGES ARE PER POSITION -- AND THE SHARED EXPERT'S REPETITION WAS REQUESTS, NOT BYTES

### Question

`#71` closed the fill and listed what is left of a 128-position prefill at the shipped width:
99.7 ms/position of which fill 52.2 (read 51.8 at 13.6 GB/s), **mixture device 19.3-20.0**, dense
device 13.4, indexer wait 11.3, front wait 43.7. The mixture half is the largest bucket that is not
the disk (`#75` later showed ~63-65% of that 43.7 IS the mixture half's own device time, drained inside
the next layer's front wait, so the two rows overlap), and it is the one bucket that did NOT fall with
the block width -- `#71`'s table had it
rising 18.0 (walk) -> 18.5 (width 4) -> 20.0 (width 8). Every stage table in this journal before now
covers the DENSE half (`#52` measured one on a real block; `#60` did the walk's), because
`recordsStageTiming` only ever gave those stages a command buffer each. So "the mixture half costs
19-20 ms and does not batch" was a statement about a bucket, not about a stage.

### What was added

Measurement only, off in every production path:

- `recordsStageTiming` now splits a BLOCK's mixture half the same way it already split its dense
  half: one command buffer per back-half stage, every position of the block inside it. The order
  becomes stage-major rather than position-major, and that is safe rather than
  equivalent-by-luck: the slots are disjoint, and the only cross-stage hazard (`routedExperts`
  clears and accumulates `moeOut`, `writeFFNStreams` reads it) stays ordered because every buffer
  is on one in-order queue. The probe asserts the two layouts return the same answer.
- `testWhereABlocksMixtureTimeGoesOnThePublishedCheckpoint` (opt-in,
  `/tmp/dynamoe_dsv41_mixture_stages`, 135 s): 128 positions in the app's own regime
  (`wireArena: nil`), widths 8 and 4, three rounds with the arm order rotated and round 1 discarded
  (`#53`), the shipped one-buffer layout beside the stage split. It prints the four mixture stages
  per width and asserts: identical logits between layouts (relative 1e-6), the same argmax, all four
  stages reported, and that the two layouts read the same bytes (a guard, not a result).

### The table, before any change

Milliseconds per position, 128 positions, arena 3,760 slots wired (65.8 GB), two layouts measured in
the same process:

| width | mixture device (split / one buffer) | routedExperts | sharedExpert | writeFFNStreams | collapseStreams |
|---|---|---|---|---|---|
| 8 | 16.17 / 17.06 | **12.272 (75.9%)** | **3.655 (22.6%)** | 0.152 | 0.095 |
| 4 | 16.37 / 17.32 | **12.409 (75.8%)** | **3.703 (22.6%)** | 0.157 | 0.100 |

Two stages are 98.5% of the half. Their bytes say what they are:

- **`routedExperts` is the machine's GEMV ceiling and is per position by construction.** One
  position routes 6 of 384 experts on each of 40 layers, 6 x 17.93 MB x 40 = 4.30 GB, read in
  12.22 ms -- **352 GB/s**, which is `#20`'s 300-370 GB/s and `#60`/`#61`'s 352 reproduced at a
  block. Eight positions of a block touch about 45 DISTINCT experts of the 48 they route (a 6%
  union saving), so batching can remove almost nothing: the bytes are the routing.
- **`sharedExpert` reads the same 33.8 MB per layer for every position** -- 1.32 GB/position -- and
  it was doing so at 390 GB/s, i.e. at the ceiling, once per position. That is the one repetition
  in the mixture half that batching can remove.

**`#71`'s "it rises with width" does not reproduce.** The same instrument in two processes puts the
mixture half at 15.5 (width 8) and 15.8 (width 4); the 18.0 -> 18.5 -> 20.0 row was instrument
spread in the bucket table, not a width effect.

### The batched shared expert, and what it is actually worth

Two additive kernels, `dsv41_fp8_block32_swiglu_gate_up_rows_slots` and
`dsv41_fp8_block32_down_accumulate_rows_slots`: one threadgroup per output row with the POSITION
loop inside it, so the row's bytes are requested once and the later positions' passes come from the
caches in front of DRAM. Buffer indices are their `_rows` siblings' plus `slotCount`; the output is
`[slot][outDim]`, and lane 0 accumulates each slot's row exactly as the `_rows` form accumulates
one. The block path now encodes the mixture stage-major and hands `.sharedExpert` ONE dispatch with
`rows: count`; every other format (and a narrow fixture, where `fp8GemvWantsWideGrid` is false)
keeps the per-position loop. `workspace.sharedIntermediate` is one row per slot.

Predicted from the byte model: `(1 - 1/8) x 3.66 = 3.2 ms/position`. Measured:

| width | sharedExpert before | sharedExpert after | mixture device after (split / one buffer) |
|---|---|---|---|
| 8 | 3.655 | **3.060 (1.19x)** | 15.53 / 15.50 |
| 4 | 3.703 | **3.177 (1.17x)** | 15.78 / 15.90 |

**So 84% of the predicted saving was not there, and the reason matters more than the number.** If
those 1.32 GB/position had been DRAM traffic, removing 7/8 of them would have taken 3.2 ms.
Removing them took 0.56 ms, which means the per-position re-reads were being served by the caches
in front of DRAM: a layer's shared expert is 33.8 MB, and eight consecutive passes over it fit.
The stage was never bandwidth-bound on its WEIGHT.

What it IS bound on is its INPUT. A threadgroup owns one output row and reads EVERY position's
normed vector for the length of that row:

    outDim x slots x inDim x 4 bytes = 2,304 x 8 x 5,120 x 4 = 377 MB per layer-block

which is 1.88 GB per position, 14x the weight, and 377 MB in the 0.61 ms a layer-block takes is
**616 GB/s**. That is the stage, and batching the weight does not touch it. `#18` found exactly this
for the block's BF16 GEMVs (a block of sixteen was WORSE than eight because the input, not the
weight, is what a threadgroup re-requests per output row); this is the same defect in the shared
expert, which is the one stage of the mixture that all eight positions share.

In situ, the shipped layout's wall moved with it, in two processes and at both widths: width 8
82.6 / 80.7 -> 78.5 / 77.7, width 4 88.6 / 87.4 -> 83.6 / 84.7 ms/position, i.e. **~3.5
ms/position, 4%**. Part of that is the stage (0.56) and part is the layout (the mixture half is now
15.50 one-buffer against 17.06, and the two layouts agree after the change where they differed by
0.9 before), so quote the stage number as the result and the wall as the direction. The live turn
agrees in size: the app's 31-position default prompt went 6.67 s (215.3 ms/position, `#71`) to
**6.29 s (203.0 ms/position)**, turn 7.74 -> 7.35 s, on a machine whose absolute walls move 5-30%
between processes.

### What it says

- **The mixture half's floor is `routedExperts` + whatever the shared expert's input traffic
  costs**, and at a block that is 12.2 + ~3.1 of a 15.5 ms half. Neither is a kernel-form
  question: the routed bytes are the routing, and the shared expert's stage is its activations.
- **A block does not amortise the routed mixture, and that is a property of the model.** Six of 384
  experts per position means two positions share an expert about 1.5% of the time.
- **The one lever left inside the mixture half is rows per threadgroup on the shared expert**, and
  `#21`'s prohibition has an exception written into it: "do not re-propose rows-per-threadgroup for
  FP8 without a shape whose INPUT traffic is the measured cost". This is such a shape (616 GB/s of
  activation re-reads). `#18` measured the two-row form at 1.20-1.34x on `attn.wo_a`/`attn.wo_b`,
  which here would put the stage at ~2.4 ms/position -- under 1% of a prefill. It is a named,
  unbuilt lever with a size, not a recommendation.

### What it is not (do not re-propose)

- **Not evidence that the mixture half should fall with width.** It is flat (15.5 at 8, 15.8 at 4)
  because both its stages are per-position; `#71`'s rising row was spread.
- **Not a batching argument for the routed experts.** The union dedup saves 6% of the slots and
  nothing of the bytes, and the bytes are at the ceiling.
- **Not a caller-side fix.** The block already submits its mixture as one command buffer with no
  round trip between positions (`#52`).
- **Not the dispatch count or fusion.** `writeFFNStreams` and `collapseStreams` are 0.25 ms/position
  together; `#66` already put the whole graph's dispatch floor at ~1.06 ms.
- **Not the arena, the fill, the page cache or the width.** Every arm here read the same bytes
  (56,281 MB / 3,139 misses at width 8, 56,963 / 3,177 at width 4 across all layouts and rounds)
  and the fill was closed in `#71`.
- **Not a reason to touch the walk.** The batched form is a block form; a decode step has one
  position and `rows: 1` keeps the `_rows` kernel it already shipped.

### Verification

`testWhereABlocksMixtureTimeGoesOnThePublishedCheckpoint` in `DeepSeekV41PublishedTurnTests`
(opt-in, `/tmp/dynamoe_dsv41_mixture_stages` plus the class's `/tmp/dynamoe_dsv41_real_turn`, 135 s
at `-O`): runs `/tmp/test72_mix1.log` (before), `/tmp/test72_mix2.log` and `/tmp/test72_mix3.log`
(after, reproducing every stage within 2%). The kernel change is pinned without the checkpoint by
`DeepSeekV41FP8SharedExpertKernelTests.testTheSlotFormsProjectEveryPositionFromOneWeightRead`:
five slots (so no form can assume four or eight), the published shapes, the batched output against
the per-slot `_rows` form AND -- at a small shape -- the host rule, plus a check that each slot lands
in its own row. The class's 7 cases pass. `DeepSeekV41DeviceGraphParityTests`,
`DeepSeekV41DeviceParityTests`, `DeepSeekV41TurnLoopTests`, `DeepSeekV41SlotWiringTests` and
`DeepSeekV41ExpertStreamTests` pass, 68 cases with 0 failures; those compare the device block and
walk against `DeepSeekV41CPUForward`, which is what makes the stage-major reorder and the batched
shared expert checked against the host graph rather than against themselves. The live turn is
`/tmp/test72_turn1.log`.

Production changed: two kernels, two pipelines, `DeepSeekV41Kernels` names, `encodeSwiGLU` /
`encodeDown`'s `rows:`, `encodeSharedExpert`'s batch decision, `encodeBlockMixtureStage` (stage-major
with one shared-expert dispatch), and `workspace.sharedIntermediate` at one row per slot (8 x 9 KB
instead of 9 KB). The split layout and `sharedExpertCapture`/`ffnNormed` probes are measurement-only
and still off in every production path.

## DSV4.1 #73: THE DENSE HALF IS THREE FP8 PROJECTIONS, AND READING THEIR WEIGHT ONCE PER BLOCK INSTEAD OF ONCE PER POSITION COSTS 2.42x

### Question

`#72` closed the mixture half and left the dense half as the largest remaining bucket of a
128-position prefill at the shipped width: 78-80 ms/position of which fill ~52 (`#71`, at the disk's
13.6 GB/s), **dense device 13.4-15.4**, mixture device 15.5 (`#72`), indexer wait 11.3, front wait
43.7 (of which `#75` later found ~63-65% is the mixture's own device time, drained inside the wait).
Every dense stage table in this journal was taken on the SYNTHETIC checkpoint at published
widths (`#16`-`#18`), where the ranking was `writeLatent` 45.5% and then three GEMVs, `compressPool`
and `attention` printed zero because the fixture had no compressed layers, and every weight was
already resident. The published plan is half compressed (8 of 40 layers own a compressed index), so
no table had ever said which stage holds the half where it ships.

### What was added

Measurement only:

- `testWhereABlocksDenseHalfGoesOnThePublishedCheckpoint` (opt-in, `/tmp/dynamoe_dsv41_dense_stages`
  plus the class's `/tmp/dynamoe_dsv41_real_turn`, ~200 s at `-O`): 128 positions in the app's own
  regime (`wireArena: nil`, `#71`), widths 8 and 4, three arms -- the shipped one-buffer layout,
  the per-position stage split, and a third arm that enables the candidate form below -- three
  rounds with the arm order rotated and round 1 discarded (`#53`). Every front-half stage is priced
  against the catalog the runtime actually bound: the weights a stage reads (values plus their E8M0
  scale companion) are resolved by role per layer and summed over the runtime's own plan, so a stage
  that reads no weight (a norm, a rope, a mix, the attention) prints without a rate instead of with
  an invented one.
- A candidate kernel for the one thing the table suggested, and the plumbing to A/B it in situ:
  `dsv41_fp8_block32_gemv_rows_slots` (one threadgroup per output row, `slotCount` positions inside
  a compile-time-eight loop, 8 accumulator pairs, the row's 32 weight bytes read once and held in
  registers), one pipeline entry, `fp8GemvWantsSlotForm`, and `enablesFP8SlotGemv` -- OPT-IN and off
  in every production path, because the answer was no.

### The table

Milliseconds per position, 128 positions, arena 3,760 slots wired (65.8 GB), the shipped
per-position form, 40 layers:

| stage | ms/position | share of the half |
|---|---|---|
| `queryUp` (`attn.wq_b`) | **2.992** | **19.8%** |
| `outputUp` (`attn.wo_b`) | **3.028** | **20.0%** |
| `outputDown` (`attn.wo_a`, 8 bands) | **2.673** | **17.7%** |
| `indexer` | 1.715 | 11.3% |
| `queryDown` (`attn.wq_a`) | 0.747 | 4.9% |
| `engram` | 0.689 | 4.6% |
| `writeLatent` (attention in the window) | 0.607 | 4.0% |
| `latentProject` (`attn.wkv`) | 0.472 | 3.1% |
| `latentNorm` | 0.289 | 1.9% |
| `attnNorm` | 0.376 | 2.5% |
| `ffnNorm` | 0.367 | 2.4% |
| `mixFFNStreams`, `mixAttentionStreams`, `router`, `queryNorm`, `hcAttentionCoefficients`, `hcFFNCoefficients`, `ropeOutput`, `writeAttentionStreams`, `ropeQuery` | 0.100-0.152 each | 0.7-1.0% each |
| `compressPool`, `attention` | 0.000 | 0.0% |
| **dense device** | **15.12** (one buffer 11.06) | |

Width 4 is the same table within 3% (`queryUp` 3.090, `outputUp` 3.076, `outputDown` 2.859, half
16.13). The split layout's own overhead is priced on its own row: 8 trivial dispatches cost 0.219
ms/position, 1% of the half.

### What it says

- **Three FP8 projections are 58% of the dense half**, and they are stage for stage the ones `#18`
  and `#20` named from isolated tables. 8.7 of 15.1 ms/position. Everything else in the half is
  under 1.7 ms, and the two biggest of those (`indexer`, `engram`) have no weight read to speak of.
- **They do not batch.** 2.992 / 3.028 / 2.673 at width 8 against 3.090 / 3.076 / 2.859 at width 4:
  +2% to +7% per position when the block halves. Like the mixture's routed half, their cost is per
  position, which is why the dense half is the one bucket `#70`'s width rule could not shrink.
- **The journal's predicted ranking does not hold at this prefix, and the reason is the prefix.**
  `writeLatent` is 4.0% here against 29-45% in `#17`, because at 128 positions attention can only
  reach 128 rows; `#17` measured 1,024 positions and 1,152 gathered rows. `compressPool` and
  `attention` print 0.000 because both run inside `writeLatent` (`#16`), and the 8 compressed layers
  are not visible in this table at all. A long-prefix prefill has a different shape.
- **`indexer` is 1.68 ms/position with 40 MB of weight**: it is not a weight read, so it is not
  priced by bytes. Eight of 40 layers own a compressed index, so that is ~0.21 ms per owning layer
  per position, and the selection readback is booked separately as `indexer wait` (11.3 ms/position
  in `#71`). It is the one dense stage whose cost is neither a GEMV nor a norm, and it is worth its
  own slice.
- **The byte arithmetic fits a ceiling it is not at.** One pass over a stage's weight per POSITION
  is 1.25-1.56 GB per layer-block at width 8; at the measured windows that is 425-523 GB/s for all
  three, which is this machine's memory bandwidth and looks like a wall. That is exactly the
  coincidence `#17` warned about, and the form below is how it was tested.

### The form that was built, measured and rejected

`dsv41_fp8_block32_gemv_rows_slots` reads each weight row once per block instead of once per
position -- `bf16_gemv_rows8`'s trade (one accumulator per position against one weight read) in the
FP8 block-scaled form, and the same shape `#72` shipped for the shared expert. Enabled through
`enablesFP8SlotGemv`, in the same process, over the same tokens and the same arena:

| width | dense device, per-position form | dense device, slot form | ratio |
|---|---|---|---|
| 8 | 15.12 | **36.57** | **2.42x worse** |
| 4 | 16.13 | **38.56** | **2.39x worse** |

and it is not an artefact of the stage-split layout: in the shipped one-buffer layout the same arm
reads 11.06 without the form and 24.05 with it (2.18x, `/tmp/test73_dense6.log` against
`/tmp/test73_dense2.log`). Per stage, the slot form costs `queryUp` 3.05x, `outputDown` 2.51x,
`outputUp` 2.06x, `queryDown` 4.62x, `latentProject` 7.82x its per-position time, while every stage
that reads no weight is unchanged (0.93-1.00x). It also moves
the final logits 9.6e-3 relative -- the split layout itself is bit-identical (0.0 difference), so
that drift is the projection's reordered summation amplifying through 40 layers of router near-ties;
`DeepSeekV41FP8GemvKernelTests` holds the same kernel to 1e-4 against the host rule on the published
shapes with full-range bytes, so the form is correct and simply slower.

**So the repetition was free.** Eight visits to the same 1.25-8 KB row are served by the caches in
front of DRAM, and the eight threadgroups that make those visits are what the stage was using. This
is `#72`'s lesson one level up: there the removal of the repetition was free and worth 1.19x; here
the removal costs 2.42x because a plain GEMV has nothing else to overlap with, and it is the one
place a batching change is a LOSS.

### What it is not (do not re-propose)

- **Rows-per-threadgroup, slot forms or a "read the weight once" variant on the plain FP8 GEMV at a
  prefill block.** Measured: 2.42x worse, both layouts, both widths, five projections. `#21`
  forbade it on FP8 until a shape whose INPUT traffic is the measured cost turned up; `#72` found
  one and it paid 1.19x. This is the other case: the input is 164 KB per layer-block against 39 MB
  of weight, so the form removes the cheap half and spends its parallelism.
- **"The dense half is weight-bandwidth bound."** Its 425-523 GB/s is arithmetic that fits a
  per-position model, not a measurement of traffic; the experiment that would have confirmed it is
  the one that lost by 2.42x.
- **`#17`'s 29-45% for `writeLatent` as a short-prefix statement.** At 128 positions it is 4.0%.
  Prefix length is the variable, and the two tables are not comparable.
- **Not the width, the fill, the arena or the prompt.** Every arm read the same bytes (56,281 MB /
  3,139 misses at width 8, 56,963 / 3,177 at width 4), and `#70`/`#71` closed the width and the fill.
- Everything from `#51`-`#72` still stands.

### Verification

`testWhereABlocksDenseHalfGoesOnThePublishedCheckpoint` in `DeepSeekV41PublishedTurnTests`
(`/tmp/test73_dense1.log` before the kernel existed, `/tmp/test73_dense6.log` after, every stage
reproduced within 3%): it asserts the split layout returns bit-identical logits to the one-buffer
layout (relative 1e-6, measured 0.0), the same argmax, the same bytes within 2%, that every
front-half stage is reported, and that the slot arm keeps the argmax while drifting no more than
5e-2 (measured 9.6e-3 at width 8, 5.8e-3 at width 4). `DeepSeekV41FP8GemvKernelTests` is 12 cases,
0 failures: the four new ones pin the slot form's position strides, a short block of three against
the compile-time eight, the published width, and both published shapes with FULL-RANGE bytes (every E4M3 exponent, E8M0 scales
over 2^-11..2^+5, activations that cancel) against the host rule at 1e-4 -- the narrow fixture
passes a kernel that reads the wrong scale row, and the first version of that test overflowed
because its scale bytes reached 2^127.

Production changed: nothing. The kernel, pipeline and rule are in the tree as a measured baseline
behind `enablesFP8SlotGemv`, which is false in every production path; the graph still dispatches
`dsv41_fp8_block32_gemv` / `dsv41_fp8_block32_gemv_rows` exactly as `#20` and `#21` left them.

## DSV4.1 #74: THE INDEXER STAGE IS THREE RAILS, AND ITS PROJECTION RAIL WAS STARVED OF THREADS

### Question

`#73` billed `.indexer` at 1.715 ms/position, 11.3% of a prefill position's dense half and the
fourth largest row of it, and stopped there: the stage is ONE command buffer holding three
unrelated rails. Four of forty layers own a compressed KV cache and run the compressor, eight own
a compressed index and run the query-side projections, and only the row scores depend on how many
rows a position can see. 40 MB of weight is all the stage touches, so "1.7 ms" had no mechanism
behind it and no way to tell a wall from a shape.

### What was added

- `DeepSeekV41IndexerSection` (`.pool` / `.project` / `.select`) and `encodeIndexerSection`.
  Production is unchanged in shape and in order: `encodeIndexerStage` loops the three sections into
  the one encoder it always used.
- Under `recordsStageTiming` each rail gets its own command buffer, and only on the layers that RUN
  it. The other thirty-two of forty encode none of the three, and an empty buffer reports no device
  window at all, so giving them one would have billed nothing while pretending to. They are labelled
  `indexer.idle` and measure 0.000, which is the finding that retires a hypothesis rather than
  confirming one: `#73`'s single `indexer` window was NOT thirty-two empty windows (see below).
- `forcesFP8WideGrid`, a measurement-only flag, so the coalesced form could be priced on a shape the
  plan rule refuses without changing what the rule decides for everything else.
- `testWhatTheIndexerStageIsMadeOfOnThePublishedCheckpoint` in `DeepSeekV41PublishedTurnTests`
  (`touch /tmp/dynamoe_dsv41_indexer_stages`): 128 positions at the shipped width, the app's regime
  (`wireArena: nil`), three arms rotated and round 1 discarded — shipped split, forced-coalesced
  split, and the one-buffer production layout for the answer and the byte count.
- One shape, `attn.indexer.wq_b` at 4,096 rows over 1,280 columns, in the existing FP8 form A/B
  (`testWhatTwoRowsPerThreadgroupSavesOnTheFP8Gemv`), which is the isolated comparison the plan
  rule's `inDim` clause had never been measured against.

### The table (128 positions, width 8, split layout, ms/position)

| rail | ms/position | of the stage | of the dense half | per layer that runs it | read per position | rate |
|---|---|---|---|---|---|---|
| `indexer.pool` | 0.401 | 21.7% | 2.6% | 0.100 | 35.00 MB | 85 GB/s |
| `indexer.project` | **0.741 → 0.631** | 34.1% | 4.1% | 0.093 → 0.079 | 42.54 MB | 56 → 66 GB/s |
| `indexer.select` | **0.818** | 44.2% | 5.3% | 0.102 | 0.20 MB | — |
| `indexer.idle` | 0.000 | 0.0% | 0.0% | — | — | — |
| **sum** | 1.982 → **1.850** | | 12.9 → 12.1% | | | |

Dense device 15.24 → 15.30 in the split layout and 10.93 → 11.07 in the one-buffer layout, which is
the first sign that the whole change is smaller than the process-to-process spread. `indexer wait`
(the host round trip the split still pays) is 2.5 ms/position in the split layout against 7.7-8.8 in
the one-buffer layout: in production the host waits at `.indexer` for EVERYTHING encoded before it,
so most of that 8 is the dense half's drain and only the 2.5 is the indexer's own. `#71`'s `indexer
wait` figure is the same quantity in a different layout and is not the stage.

### What it says

1. **Three rails, roughly a third each.** Nothing owns the stage, so there was no single fix to
   find; `#73`'s row was the sum of a compressor, two projections and a score kernel.
2. **`select` is neither bandwidth nor FLOP bound.** 0.20 MB and ~1.6M multiply-adds per position,
   eight dispatches of 64-128 threads — a fortieth of the machine per dispatch. That is a latency and
   occupancy row, and its ceiling is its own 0.82 ms.
3. **`project` was one thread per row over 4,096 rows.** 42.54 MB of weight per position read at
   56 GB/s where the coalesced form measures 132-133 GB/s on the same bytes: the plan rule sent it to
   `dsv41_fp8_block32_gemv` because 1,280 columns is under its 2,048, and that clause is a
   lane-per-block argument derived on `attn.wq_b`, whose 32,768 rows fill the machine on their own.
4. **`pool` is four layers' worth of two BF16 `[512, 5120]` projections per position** at 85 GB/s.
   Already coalesced, so its rate is a shape question rather than a rule question, and the whole rail
   is 0.4 ms.
5. **The split overstates the stage.** Its three windows sum to 1.982 against `#73`'s single window
   of 1.715, i.e. the two extra submissions per owning layer cost ~0.27. The SHARES are the result;
   the sum is not the shipped wall, the same bias as `#60` and `#73`.
6. **The whole stage is ~1.85 ms of a ~78 ms position, 2.4%.** That is the ceiling on any indexer
   work, and it is why the rule change below is priced as a rounding error in situ and shipped on an
   isolated number instead.

### What shipped: an occupancy clause in the FP8 wide-grid rule

`fp8GemvWantsWideGrid` gained a second clause, `outDim * rows < 16_384`, and
`fp8WideGridThreadThreshold` names it. The isolated A/B on the published shapes (six distinct weight
regions per dispatch, best of 8, arms interleaved) is what supports it:

| shape | tokens | one-thread | one row per threadgroup | winner |
|---|---|---|---|---|
| `attn.indexer.wq_b` 4,096x1,280 | 1 | 0.058 ms (90 GB/s) | **0.040 ms (132 GB/s)** | coalesced 1.46x |
| `attn.wq_b` 32,768x1,280 | 1 | **0.135 ms (310 GB/s)** | 0.142 ms | one-thread 1.05x |
| `attn.indexer.wq_b` 4,096x1,280 | 8 | **0.107 ms (49 GB/s)** | 0.141 ms | one-thread 1.32x |
| row width 1,024-4,096, 5,120 rows | 1 | 0.055-0.139 | **0.037-0.064 (1.49-2.17x)** | coalesced at every width |
| row width 1,024 / 1,536 | 8 | **0.102 / 0.147** | 0.108 / 0.175 | one-thread below 2,048 |
| row width 2,048-4,096 | 8 | 0.182-0.334 | **0.178-0.317 (1.01-1.05x)** | coalesced at and above 2,048 |

Every cell of that table falls out of the one clause: at one position the grid is `outDim` threads
and 5,120 or 4,096 of them cannot fill the machine, so the coalesced form wins at ANY width; at a
block of eight the grid is `outDim * 8` and the old `inDim >= 2,048` crossover is the right one
again. The indexer's projections are encoded per slot (`rows: 1`), which is the row that wins 1.46x,
and the clause catches the batched case for any future caller.

In situ the change moves `indexer.project` 0.741 → 0.631 and the stage 1.982 → 1.850 ms/position,
i.e. 0.13-0.29 ms of a ~78 ms position (0.2-0.4%), which is under what two processes resolve on this
machine: quote the isolated 1.46x as the result and the in-situ figure as direction. Forcing the
coalesced form on EVERY FP8 projection costs the dense half 1.8 ms/position (15.30 → 17.08), which is
what makes the rule a rule rather than a code change.

Narrower rows also want the coalesced form at a decode step (`row width 1,024` and `1,536` at one
token, 1.49x), so the clause corrects a miscellany of shapes, not only the indexer's: no other
published FP8 projection sits in the region, because every one of them either reaches 2,048 columns
or has an `outDim` that fills the machine on its own.

### What it is not (do not re-propose)

- **The host selection.** `readIndexSelections` is inside the round trip, not the stage; the stage
  table is device windows only. A device-side top-k is a different slice (`#28`/`#30` closed the
  arithmetic).
- **The compressed-row gather.** It runs in `writeLatent`, which `#73` priced at 4.0% of the half.
- **A bandwidth wall on either projection rail.** Both are below what the same bytes reach under a
  better shape, and `pool`'s 85 GB/s is a 512-row projection rather than DRAM.
- **Empty `.indexer` windows as an explanation of `#73`'s 1.715.** `indexer.idle` measures 0.000: a
  command buffer with no dispatch reports no device window, so thirty-two of forty layers contribute
  nothing to the row. The inflation is the two extra submissions on the eight that do.
- **A slot dimension in the scores kernel, for now.** It is the select rail's only lever and its
  ceiling is that rail's own 0.82 ms/position (1.1% of a position); `#73`'s rejection of slot forms
  does not transfer (this kernel reads no weight), but neither is there a measured win to point at.
  Build it only with an isolated A/B in hand, the way this slice built the wide-grid question.
- Everything from `#51`-`#73` still stands.

### Verification

`DeepSeekV41DeviceGraphParityTests`, `DeepSeekV41DeviceParityTests`, `DeepSeekV41TurnLoopTests`,
`DeepSeekV41SlotWiringTests`, `DeepSeekV41ExpertStreamTests`, `DeepSeekV41FP8GemvKernelTests`,
`DeepSeekV41FP8SharedExpertKernelTests`, `DeepSeekV41IndexerKernelTests`: 89 tests, 0 failures
(`/tmp/test74_parity1.log`). The compressed block parity cases (`testACompressedBlockPrefillsLikeTheWalk`,
`testTheDeviceAndTheHostAgreeOnCompressedRows`, `testEveryPositionOfABlockAttendsOverItsOwnRows`)
drive the refactored sections against the host. The probe asserts the split layout returns
bit-identical logits to the one-buffer layout (measured 0.0 at a 1e-6 bar), the same bytes within
2%, that all four rail rows are reported, and that they sum to the stage row. The rule test pins
4,096x1,280 to the coalesced form, 32,768x1,280 to the one-thread form at both row counts, a 16x512
projection to neither, and the threshold constant itself.

## DSV4.1 #75: THE FRONT WAIT IS THE PREVIOUS LAYER'S MIXTURE, AND A PREFILL POSITION IS 63% GPU IDLE

### Question

`#73` and `#74` left the prefill bucket table as: wall 99.7 ms/position = fill 52.2 (the disk, closed
at its ceiling by `#71`) + mixture device 19.3-20.0 (`#72`) + dense device 13.4 (`#73`, its one
unattributed stage closed by `#74`) + indexer wait 11.3 + **front wait 43.7**. The front wait was the
largest term that is neither the disk nor a named stage, and every candidate fix starts from knowing
whether it is the GPU, the driver, or the host. The instrument already split a drain into `late start
+ device window + tail`, so the question was narrow: which of the three, and how much of it is work
another bucket already bills?

### What was added

Measurement only, off in every production path:

- `DeepSeekV41BlockTiming.frontBusyNanoseconds` — how much of `frontWaitNanoseconds` was the
  PREVIOUS layer's mixture still on the queue. `frontWait` is one `settlePending`, and the queue is in
  order, so the mixture the layer before committed is drained inside it and then the dense half is.
  Accumulated at the settle that measured the wait, not over every settle the layer made, because a
  compressed layer's dense half is two submissions and only the second one's drain is inside
  `frontWait` (`#66`'s footnote, now exact).
- `frontOwnDrainNanoseconds` (`frontWait - frontBusy`), `frontTailNanoseconds` and
  `mixtureTailNanoseconds` (the third term of a drain, which `#22` named and no field held).
- `DeepSeekV41BlockTiming.deviceUnionNanoseconds` — the UNION of every measured window, so a block's
  wall can be split into busy and idle. The half sums are sums of busy TIME; the union is the busy
  SPAN, and `wall - union` is the one number no host interval names. Plus
  `frontUnwindowedBuffers`/`mixtureUnwindowedBuffers`, so a union with a hole in it says so.
- `recordsBlockGaps` + `DeepSeekV41BlockGap`: one clock record per block LAYER pass with that layer's
  own buffers' windows attached, so the wait divides per layer instead of per span. This is the block
  side of `#48`'s `DeepSeekV41LayerGapSample`.
- **A correction to the dense device column.** The shipped layout submits twice on the eight layers
  that split at the indexer, and only the SECOND submission reached `timingBuffers`: `denseDevice`
  silently omitted the work in front of the split. Every front buffer is recorded now, which is what
  makes `drained front = late start + window + tail` tie out (it did not before: `#71` reported 18.4
  drained against 13.67 of window and 1.2 of late start).
- `testWhatTheFrontWaitIsMadeOfOnThePublishedCheckpoint` (opt-in, `/tmp/dynamoe_dsv41_front_wait`,
  102 s): 128 positions in the app's regime (`wireArena: nil`), the shipped width and half of it,
  three rounds with the arm order rotated and round 1 discarded (`#53`). It prints the decomposition,
  the per-layer partition of one block, and the union, and asserts the classification: `busy > 0`,
  `busy < frontWait`, the pickup under half a millisecond, the tail under half a millisecond, the
  mixture drain at least the busy it absorbed, `union <= wall <= sum of windows`, and that the
  settle-level `frontBusy` agrees with the records taken per layer.

### The table

Milliseconds per position, width 8, `wireArena: nil`, three arms after the discarded round:

| | arm 1 | arm 2 | arm 3 |
|---|---|---|---|
| **wall** | 78.0 | **77.7** | 78.3 |
| expert fill | 33.6 | **33.7** | 33.6 |
| indexer wait | 8.3 | **9.0** | 8.0 |
| routing + dense encode + mixture encode | 0.01 + 0.98 + 1.38 | **0.01 + 0.97 + 1.37** | 0.01 + 0.98 + 1.38 |
| **front wait** | 32.5 | **31.7** | 33.5 |
| = busy (the previous layer's mixture) | **21.1** | **20.4** | **21.9** |
| + own (the dense half and its hand-off) | 11.4 | 11.3 | 11.6 |
| every dense drain | 14.2 = 0.3 late + 13.5 window + 0.3 tail | 14.0 = 0.3 + 13.4 + 0.3 | 14.2 = 0.3 + 13.5 + 0.3 |
| every mixture drain | 27.8 = 11.9 late + 15.5 window + 0.3 tail | 27.5 = 11.8 + 15.4 + 0.3 | 28.1 = 12.2 + 15.6 + 0.3 |
| device (dense / mixture) | 29.0 (13.5 / 15.5) | 28.8 (13.4 / 15.4) | 29.1 (13.5 / 15.6) |
| **device union** | 29.0 | **28.8** | 29.1 |
| **idle (`wall - union`)** | 49.0 | **48.9** | 49.2 |

At half the width (measured in the same rounds, 83.1 / 83.0 / 82.8 ms of wall): front wait 35.4 / 35.2
/ 34.4 = busy 23.0 / 22.8 / 22.0 + own 12.4; mixture drain 29.7 = 13.3 late + 15.8 window + 0.7 tail;
dense drain 14.4 = 0.3 + 14.4 + 0.7; union 30.2 / 30.1 / 30.0 of the wall, i.e. idle 52.9 / 53.0 /
52.8. The first block of a prefill is excluded from every partition below: it is the cold one (101.6
ms/position, 68.3 GB read, 3,812 misses against 55.2 GB and 3,152 once the arena is warm).

The per-layer records, one block of 39 layer passes (the second block, `recordsBlockGaps`), in the
units the totals use — milliseconds per position across the block's 39 layers:

| term | ms/position | share of the wait |
|---|---|---|
| wait = busy + pickup + window + tail | 2.17 = 1.32 + **0.00** + 0.82 + **0.02** | 100% |
| gap between the two dense submissions, inside the span | 0.04 | 2% |
| mixture gate idle (commit to its own window start) | 0.72 | — |

Per layer pass that is a front wait of ~7.1 ms: **4.3 of the previous layer's mixture, 2.7 of the
dense half, 0.00 of pickup and 0.07 of tail.**

### What it says

- **The front wait is a settle, not a stall.** 63-65% of it (20.4-21.9 of 31.7-33.5 ms/position) is
  the mixture the layer before committed, and that same work is already billed as
  `mixtureDeviceNanoseconds`. So `#71`'s bucket list counts ~21 ms/position twice: the honest front
  wait is **11.3-11.6**, and the largest non-disk term in a prefill position is the mixture half's
  device time, not the wait for it. This was flagged as a hazard in `#60` and `#65` ("front wait
  includes draining the previous layer's mixture") and never quantified until now.
- **The hand-off itself is closed at a block's front.** The pickup is 0.00 ms/position and the tail
  0.02 over 39 layer passes — the queue starts the dense half before the host finishes asking and the
  host notices its last buffer in 70 us per layer. `#48`'s 2.1-2.3 ms/layer of hand-off idle is a
  WALK's boundary (one position, two buffers, no fill in between); a block commits 2.7 ms of dense
  work behind 4.3 ms of finished mixture, so there is nothing for the driver to be slow about. That is
  the interleaving `#52` predicted and no table had shown.
- **Both drains now decompose exactly**: dense 14.2 = 0.3 + 13.5 + 0.3 and mixture 27.8 = 11.9 + 15.5
  + 0.3. The 3.5 ms/position that used to be unexplained between `drained front` (18.4) and
  `window + late start` (14.9) was the instrument's missing pre-indexer submission, not the machine.
- **A prefill position is 37% GPU work.** The union is 28.8-29.1 of a 77.7-78.3 wall: 13.4-13.5 dense
  plus 15.4-15.6 mixture, with **no overlap at all** (the sum equals the union, so the queue never has
  two windows open). The idle is 48.9-49.0 ms/position, and it tiles: fill 33.6 + the mixture's own
  late start 11.9 + the two encodes 2.35 + routing and the indexer's host readback ~1.1 = 48.9. Three
  of those four terms are already closed or named — the fill at the disk (`#71`), the encodes
  (`#60`/`#66`), and the mixture's late start as the machine after reading (`#50`, `#65`). What is NOT
  closed is that the mixture's late start is 11.8-12.2 ms/position in a BLOCK, i.e. ~2.4 ms per layer
  pass, where a walked position pays 1.2 (`#22`) — a worse pickup after a 34 ms host gap than after a
  2 ms one, which this section read as the opposite of what `#22`'s isolated sweep predicted.
  **Withdrawn by `#76`** (prose reconciled in `#82`): that sweep measured 0.57-0.84 ms after 10-50 ms
  of host idle, so it predicted 0.55 and the block's probe measured 0.55-0.60. The block is exactly
  where the curve says it should be; what differs is the mixture buffer behind the probe, not the
  pickup.
- **Width moves the answer only through the dense half.** `busy` is 20.4-21.9 at width 8 and 22.0-23.0
  at width 4: the mixture's work per position does not depend on the block width. `own` falls 12.4 ->
  11.3, which is the dense batching. Wall 77.7-78.3 against 82.8-83.1, so width eight wins 5-6
  ms/position, of which ~1.5 is device and the rest is the fill amortising over twice as many
  positions (33.6 against 34.9).

### What is not

- **The front wait is not idle**, so there is nothing in it to reclaim. Do not propose a fix for the
  pickup (0.00) or the tail (0.02) on the strength of `#48`'s walk figure: at a block's front they are
  already zero.
- **The mixture's 11.8-12.2 ms/position pickup is the one unexplained term left**, and it is not a new
  lever until it is measured as one. `#50` (the expert buffer's resource list and CPU-dirtied arena
  memory), `#65` (a deferred bind and an event gate, a wash) and `#22` (no kernel change recovers it,
  because the readback empties the queue) all closed the WALK's version of exactly this. A block's
  regime differs in the one variable those slices could not hold: the gap is 34 ms, not 2, and the
  pickup is twice as expensive. Anyone re-proposing the gate needs that A/B first — a real mixture
  buffer committed behind a 34 ms host gap, in a block, against the shipped order — and it is a
  measurement, not a build.
- **The 12.4 -> 11.3 `own` is not a licence to widen further.** `#70` swept the width on the wall and
  eight won everywhere; this table agrees, and `#18`/`#20` explain why a wider block costs more input
  traffic than it saves weight. Nothing here reopens it.
- **`indexer wait` (8.0-9.0 ms/position) is mostly this same mixture device time**, drained by the
  pre-indexer settle. Read it beside `busy`, never as the indexer's own cost (`#74`).
- **The 11.3 in `#71`/`#72`/`#73` and the 8.0-9.0 here are the same stop, not a change in the
  indexer** (reconciled in `#82`, which found the two figures standing side by side unexplained).
  The stop bills whatever the queue had not finished when the host asked, so the number follows the
  layout and the order of what was encoded ahead of it rather than the indexer's own work: `#64`
  split the split-layout form at **2.5 ms/position of indexer and the rest a drain of the dense
  half**, and the 2.5 is what the stage actually costs (`#30`-`#32` priced its kernels separately).
  The two runs also sat at different walls, 99.7 against 77.7-78.3 ms/position, so nothing here says
  the drain shrank -- only that it was measured over less work. It is the reason `#74` forbids
  reading this row as the indexer stage, and the reason `#82` did not chase the difference.

### Verification

`/tmp/test75_front3.log` (three rounds, arms rotated, round 1 discarded, all assertions green) and
`/tmp/test75_front2.log`/`/tmp/test75_front1.log` (the two cuts before the per-settle attribution and
the probe's per-layer-per-pass units). `DeepSeekV41DeviceGraphParityTests` 37, `DeepSeekV41DeviceParityTests`,
`DeepSeekV41TurnLoopTests` 26, `DeepSeekV41SlotWiringTests`, `DeepSeekV41ExpertStreamTests`,
`DeepSeekV41FP8GemvKernelTests`, `DeepSeekV41FP8SharedExpertKernelTests`,
`DeepSeekV41IndexerKernelTests`, `DeepSeekV41LogicTests` — 89 tests, 0 failures
(`/tmp/test75_parity1.log`). Production default is `recordsBlockGaps = false`, and the only line that
changed in a shipped path is `DeepSeekV41BlockTiming.describe()`, which is diagnostic text.

## DSV4.1 #76: THE MIXTURE'S PICKUP IS A PROBE'S PICKUP PLUS TWO MILLISECONDS, AND NEITHER THE GAP NOR THE BYTES THIS LAYER READ MOVES IT

### Question

`#75`'s idle tiling left one term it could not place. A prefill position is 37% GPU work, and after
the fill (the disk, `#71`), the two encodes (`#60`/`#66`) and the indexer's host readback (`#74`) are
subtracted, the remainder is the mixture's own late start: **11.8-12.2 ms of a 77.7 ms position**,
i.e. ~2.4 ms per block layer pass. `#22` measured a walked position's hand-off at ~1.2 ms/layer after
a **2 ms** fill gap, and a block's gap is **34 ms**; `#75` read the doubling as "the opposite of what
`#22`'s isolated idle-gap sweep predicted" and asked for an A/B before anyone re-proposed the deferred
bind `#65` closed.

### What was added (measurement only, every flag off in production)

- **The walk's mixture-start probe, at a block's commit.** `#50`'s one-dispatch probe buffers are now
  built by one shared helper (`commitMixtureStartProbes`), so the walk and the block are measured by
  the same code rather than by two that look alike. `runBlockLayer` commits them immediately after the
  union fill and before the mixture, exactly where the walk commits its own.
- `DeepSeekV41BlockGap.fillBytes`/`fillMisses` — what THAT layer pass's fill read, so the pickup can
  be priced per byte the way `#50` priced the walk's (`0.088-0.117` ms of GPU idle per MB) instead of
  per layer.
- `DeepSeekV41BlockGap.probes` (a `DeepSeekV41GapProbe` per committed probe, windows filled in later),
  `mixturePickupMilliseconds`, `probePickupMilliseconds` and `fillMegabytes`.
- `timingBuffers` grew a fourth field: a probe buffer is bucket `2` and carries its index within the
  gap's `probes`, because its pickup is measured from its OWN commit stamp. Probes are deliberately
  not part of `denseDevice`/`mixtureDevice` or of the busy union — they do no work.
- `testWhatABlocksMixturePickupIsMadeOfOnThePublishedCheckpoint` (opt-in,
  `/tmp/dynamoe_dsv41_mixture_pickup`, 224 s): 128 positions in the app's regime (`wireArena: nil`),
  five arms in a rotation, four rounds with round 1 discarded (`#53`) — control, probe depth 1, probe
  depth 3, probe binding 16 MB of host-dirtied shared memory, and a width-one arm.

### The table

Milliseconds per block layer pass (8 positions), `wireArena: nil`, three measured rounds each:

| arm | mixture pickup (mean / median) | ms/position | fill MB/layer pass | misses | ms per MB | probe pickup (median) |
|---|---|---|---|---|---|---|
| control | 2.50 / 1.63, 2.40 / 1.47, 2.69 / 1.42 | 12.5, 12.0, 13.5 | 83.4 | 4 | 0.0300 / 0.0288 / 0.0323 | — |
| probe ×1 | 2.24 / 1.42, 1.99 / 1.51, 2.75 / 1.53 | 11.2, 10.0, 13.8 | 84.3 | 4 | 0.0269 / 0.0237 / 0.0327 | **0.55 / 0.57 / 0.56** |
| probe ×3 | 2.74 / 1.35, 1.97 / 1.45, 2.56 / 1.52 | 13.7, 9.9, 12.8 | 84.3 | 4 | 0.0325 / 0.0234 / 0.0304 | **0.57 / 0.57 / 0.60** |
| probe, 16 MB dirty | 2.10 / 1.46, 2.91 / 1.50, 2.44 / 1.52 | 10.5, 14.5, 12.2 | 84.3 | 4 | 0.0249 / 0.0345 / 0.0289 | **0.58 / 0.55 / 0.57** |
| width 1 | 1.01 / 0.60, 0.93 / 0.58, 1.21 / 0.58 | 40.3, 37.2, 48.3 | **43.9** | 2 | 0.0230 / 0.0212 / 0.0275 | — |

The mixture starts **2.03 / 2.45 / 1.68 ms after the first probe started** (depth-one arms),
**2.44 / 1.72 / 2.26** (depth three), **1.79 / 2.63 / 2.13** (the dirty arm). Walls: control 78.4 /
78.5 / 78.6, probe 78.7 / 78.9 / 78.7, ×3 79.3 / 78.2 / 79.1, dirty 78.5 / 79.6 / 79.1, width 1
103.8 / 103.2 / 103.9 ms/position — so a probe per layer is free within 1%, and the first control of
round 1 read 101.5 ms, which is `#75`'s cold-first-block 101.6 to the tenth.

### What it says

- **The queue is willing at a block's hand-off, and `#48`'s idle-gap curve already covers it.** A
  trivial buffer committed right there starts in **0.55-0.60 ms**, every arm, every round. That is not
  a tenth of a millisecond (which is what the same probe reads on a walk, `#50`) and it is not
  unexplained: `#48`'s isolated sweep measured **0.57-0.84 ms after 10-50 ms of host idle** and 0.07-
  0.08 ms with the queue fed. A block's gap is 34 ms, so the probe is exactly where that curve says it
  should be. `#75`'s "the opposite of what the sweep predicted" is withdrawn: the sweep predicted
  0.55, and it got 0.55.
- **The other ~2 ms is the mixture and nothing else.** Three commits in a row (depth three), a long
  resource list, and 16 MB of shared memory the host has just `memset` all leave the probe at 0.55-0.60
  and the mixture starting 1.7-2.6 ms later, so the extra wait is attached to the mixture buffer
  itself rather than to the act of committing into an empty queue. `#50`'s walk arm has the same shape
  with different constants (probe 0.10, expert half 1.5 ms after its own commit).
- **What this layer read does not predict its own pickup (r = -0.17 to +0.60, mean ≈ +0.13), and the
  previous layer's read predicts it even less (r ≈ 0.03).** That is `#50`'s conclusion at a block:
  "what predicts the gap is that the WALK is reading, not that this LAYER read". What does move it is
  the ARM's own read volume: the width-one arm hands the same code **43.9 MB per layer pass against
  84.3**, and its pickup falls from ~2.4 ms to ~1.0 — the **same 0.021-0.035 ms per MB** as the block's
  full width, and the same slope the walk's own hand-off shows across its fill levels.
- **And that slope is the finding: a block already pays 3-4x less per byte than a walk.** `#50`'s walk
  scaled at 0.088-0.117 ms of GPU idle per MB read; a block's mixture pays 0.021-0.035 on the same
  measurement. So the term is not an anomaly of the batched path — it is the reading machine's cost,
  amortised by batching exactly as the fill is. The 11.8-12.2 ms/position is the price of reading
  439 MB per position, and the only lever on it is fewer bytes, which `#58`, `#59` and `#71` closed.
- **The block's pickup reproduces `#75` to the tenth** (10.0-14.5 against 11.8-12.2 ms/position) on a
  second instrument, which is the check `#75` never got: its number came from `blockTiming`, this one
  from the per-layer records read against each layer's own commit stamp.

### What is not (do not re-propose)

- **Not the commit, the gap, the depth, the resource list or dirty memory.** One arm each above, plus
  `#50`'s four bindings and `#65`'s event gate. A deferred bind moves WHICH call hands the work over;
  the probe starts in a half millisecond either way.
- **Not this layer's bytes.** r ≈ +0.13, and the zero-read layers inside a full-fill arm idle the same
  as their neighbours (`#50`).
- **Not the mixture's shape.** `#48`'s twenty arms: 1-128 launches, a million-lane grid, 2 GiB bound,
  512 named resources, a barrier per launch, zero-copy mapped memory. `#66` priced fusion at ≤1.06 ms
  and it is device time, not start.
- **Not a wider or narrower block.** Width moves the pickup per position only through bytes read; the
  width-one arm is eight times worse per position (37-48 ms) for half the bytes per layer pass, and
  `#70` closed the width question on the wall.
- **Nothing here reopens the two-stream idea.** There is no work to feed a second queue: the next
  layer's dense half reads what this mixture writes.

### Verification

`/tmp/test76_pickup1.log` (four rounds, five arms rotated, round 1 discarded, all assertions green:
every recorded layer pass reported a mixture window, every probe reported one, a probe's median pickup
stayed under the mixture's, and four arms' last logits matched the control's to 1e-6 relative, so a
probe is inert). `DeepSeekV41DeviceGraphParityTests`, `DeepSeekV41DeviceParityTests`,
`DeepSeekV41TurnLoopTests`, `DeepSeekV41SlotWiringTests`, `DeepSeekV41ExpertStreamTests`,
`DeepSeekV41FP8GemvKernelTests`, `DeepSeekV41FP8SharedExpertKernelTests`,
`DeepSeekV41IndexerKernelTests`, `DeepSeekV41LogicTests` — 89 tests, 0 failures
(`/tmp/test76_parity1.log`). Production defaults are unchanged (`recordsBlockGaps = false`,
`recordsMixtureStartProbe = false`); the shared probe helper is the walk's old body verbatim, and the
only shipped-path change is that a probe buffer can now carry a `blockGaps` index.

## DSV4.1 #77: THE SERVER NOW ACCEPTS THE PICTURES THE APP ALREADY COULD, BECAUSE THE BLOCKER WAS A STALE SENTENCE AND NOT A MISSING CAPABILITY

### The question

`#76` closed the last unattributed term in a prefill position, which left the prefill path with
nothing worth a slice: the remaining named levers (`sharedExpert`'s rows-per-threadgroup form at
~2.4 ms/position, `select`'s slot dimension with a ceiling of 0.82, `pool`'s 0.401 at 85 GB/s
already coalesced) total **~3.5 ms of a 78 ms position, about 4.5%**, and the dominant terms are at
measured hardware ceilings (the fill at the disk's 13-14 GB/s, the mixture pickup as the reading
machine's cost). So this slice asked the next question rather than the next measurement: what is
still MISSING for the model, on a branch called `64-complete-full-deepseek-support`?

The answer was in two comments that had stopped being true. `ServerController.buildTurnPlan` refused
every image content part with

> This server is serving a local weights model, which understands text only, so image, file, and
> audio content parts cannot be processed.

and the same sentence shaped `OpenAIAdapter.Message.hasUnsupportedPart`, which counted an image part
as unsupported. Both were written when no local family had a tower. Since `#35`, `#38` and `#43` the
DeepSeek V4.1 checkpoint **does** have one: the composer attaches pictures, the engine prices the
expanded prompt before provisioning, the device tower encodes a smallest image in 0.32 s, and a real
turn reads the rows. The same model, in the same app, saw images in the chat window and refused them
over the API.

Routing agreed without reservation: `next_slice` = `api_images` **0.987 @ 0.981** against
`prefill_micro` 0.010 and `journal_reconcile` 0.003, `gap_is_real` **0.999**,
`perf_exhausted` **0.9995**. That last one is the honest justification for a slice that measures
nothing: the perf path is closed, and this is a hole rather than a percentage.

### What was there already

The engine seam was complete and unused by the server:

- `CompletionTurnRequest.images: [TurnImage]` — decoded RGB plus a content digest.
- `DeepSeekV41TurnImages.price` — prompt positions from the resize plan alone, no weight read, and
  the placeholder/image agreement check.
- `DeepSeekV41TurnImages.bind` — runs the tower and places its rows, failing closed five ways.
- `DeepSeekV41ImagePixels.decode` — any ImageIO format, EXIF-oriented, capped at 4,096 pixels a side.
- The span cache in `CompletionEngine.launchTurn`, keyed on `snapshot path | sha256`.
- `ContentView` writing one marker per attachment at the head of the user turn.

Every piece of the picture path existed. What did not exist was a decoder for the two wire spellings
and a planner that knew which model was loaded.

### What landed

**`ServerImageParts.swift` (new)** — the one form both protocols decode into:

- `ServerImagePart.Payload` is `.inline(Data, mediaType:)`, `.remoteURL(String)` or
  `.malformed(reason:)`. The refusal cases are payloads rather than thrown errors because a model
  with **no tower** should answer "this checkpoint cannot see images" before the request is rejected
  for its base64.
- `fromOpenAIURL` takes a `data:` URL, a bare base64 payload (llama.cpp and several hand-rolled
  clients send that), or a plain URL. **A scheme is checked before base64 is attempted**: every
  character of `https://host/a.png` except `:` and `/` is legal base64, so a lenient decode turns a
  URL into plausible-looking garbage and the tower reads noise. There is a test for exactly that.
- `fromAnthropicSource` takes `base64` (the only source this server can read), `url` (a reference),
  and names any other type.
- `digest(_:)` is a SHA-256 over the parts plus their order, for the session key.
- `visionMarker(modelPath:config:)` is the gate: DeepSeek V4.1 **and**
  `ModelConfig.supportsVision`, which reads the shard index for a block, the aligner and the three
  marker embeddings. A half-finished download keeps the answer "no".

**Both decoders.** `OpenAIAdapter.ContentPart` gained `imageURL` decoded from BOTH shapes (an object
in Chat Completions, a bare string in the Responses spelling — a client using the other one must not
have the whole body rejected), and `Message.imageParts`. `AnthropicAdapter.ContentBlockIn` gained the
`source` object and `Message.imageBlocks`. `hasUnsupportedPart`/`hasUnsupportedBlock` no longer count
a picture, so `file`, `input_audio` and `document` keep their explicit 400 and images do not.

**`splitMessages` returns `ServerPromptSplit`** (`systemPrompt`, `turns`, `images` keyed by TURN
INDEX) instead of a tuple. A picture belongs to the message it came with, and the turn text is what
gets rendered — a raw picture is not text — so it travels beside the turns. Four roles are refused by
name rather than moved: an image on an assistant or tool turn, and an image nested inside an
Anthropic `tool_result`.

**`ServerController.resolveTurnImages` + `attachImageMarkers`.** The gate is split from the
placement so the placement can be tested without a loaded model. The gate reads
`CompletionEngine.LoadedModelInfo.supportsVision` (a new field the server and the composer now
share) and `ModelConfig.visionSpec`, and refuses a checkpoint with no tower. The placement writes one
marker per attachment at the **head** of its own turn — where the checkpoint's own prompts put an
image relative to the question about it, and what the in-app composer writes — and decodes the pixels
into `TurnImage`s carrying `ServerImageParts.digest([part])`, which is what the engine's span cache
and the session identity both key on.

This had to run **first in the planner**, before `historyDigest`, before
`substitutingRecordedTurn`, before tokenizing: everything below has to see the expanded turn.

**The prompt budget moved to the expanded size.** The renderer emits ONE token per attachment and the
turn runs several hundred, so the number checked against `dynamoe_server_max_prompt_tokens` and the
number the client is told (`plan.promptTokenCount` → usage and the request-log row) is now
`DeepSeekV41TurnImages.price(...).promptTokens`. It costs no weight read, and it moved the
placeholder/image agreement check into the planner, where a stray marker is a 400 instead of an
engine error.

**The three correctness traps accepting images would otherwise have introduced:**

1. **The session key had to learn about the picture.** Two requests can carry the same TEXT and
   different images; both render to the same list of identical image token ids, so neither the
   anchor's text nor `findCommonPrefix`'s token-prefix equality can separate them. Without the digest
   folded into `sessionKey`, a follow-up about a second picture would splice the first one's recorded
   generation (`ServerTurnMemory`) and reuse its pinned KV prefix. Both adapters now mix
   `"images:" + digest` of the anchor message. Pinned by
   `testTheOpenAISessionKeySeparatesTwoPicturesWithTheSameText` and its Anthropic twin.
2. **The engine was silently dropping pixels for every other family.** `launchTurn`'s generic path
   never reads `request.images`; only the DeepSeek branch does. A non-vision model would have
   answered about a picture it never saw. `launchTurn` now refuses the turn by name before it touches
   anything (`CompletionEngine.swift`, the guard right after the tokenizer check).
3. **A request-log row would have shown a message with no content and no sign of a picture.** The
   summaries now name the count (`image_parts` / `image_blocks`) and never carry the payload.

**Remote URLs are refused, not fetched.** `https://...` inside an `image_url` or an Anthropic `url`
source is an outbound connection the caller chose — a privacy surface and an SSRF one — so the turn
is refused with the URL named and the fix spelled out ("send it inline as base64, which is what the
SDKs do").

### Evidence

The permanent tests, all model-free:

| suite | tests |
|---|---|
| `ServerImagePartsTests` (new) | 26 |
| `OpenAIAdapterLogicTests` | 21 |
| `AnthropicAdapterLogicTests` | 21 |
| `ReasoningRoundTripTests` | 13 |
| `ServerHardeningLogicTests` | 42 |
| `ServerRequestLogLogicTests` | 29 |
| `ServerStreamSafetyTests`, `ServerToolArgumentTests`, `PrefixAsOfDateTests` | 10 |
| **total** | **162, 0 failures** (`/tmp/test77_logic3.log`) |

The new class pins both wire spellings (data URL, bare base64, line-wrapped base64, the Responses
bare-string form), the four refusals (a URL that must not be fetched, a non-base64 payload, a
non-base64 data URL, an unknown Anthropic source type), the Anthropic `base64` and `url` sources, the
digest's order sensitivity, both session keys, the assistant-turn and `tool_result` refusals, the
marker placement (its own turn only, one per picture, in order), and the three ways
`attachImageMarkers` fails closed.

The prompt itself, at the checkpoint's own tokenizer (no weights, milliseconds):

```
🔎 [dsv41 real] API image request: 18 prompt tokens with 1 image token(s); priced 201 after
   expansion (+183), 1 estimate(s), 0.32 s of tower
```

`DeepSeekV41PublishedTurnTests.testAnAPIRequestWithAnImageCarriesOneImageToken` starts from a real
OpenAI request body with a real 2x2 PNG, runs `splitMessages` → `attachImageMarkers` →
`ChatPromptFormatter.formatConversation`, and asserts the marker heads its own turn and no other,
that the rendered prompt tokenizes to exactly one image token, and that `TurnImages.price` accepts
the result. A marker that tokenized into pieces here would leave literal text in the prompt and bind
no rows, which looks exactly like a model ignoring pictures — the same failure `#35` guarded against
for the composer, now guarded for the API surface.

### What it does not mean

- **Not a faster image turn, and not a cheaper one.** The tower is untouched (`#43`'s 0.32-6.90 s
  range stands) and the prefill of the expanded span is still the cost: `#46`'s largest picture is
  1,005 positions, and at `#75`/`#76`'s ~78 ms/position that is ~78 s of prefill behind a
  0.32-6.90 s encode. An API image request is a minutes-long turn, not a seconds-long one.
- **No capability advertisement.** `/v1/models` still returns only an id and a created stamp; there
  is no vision field in the OpenAI model list and none was invented. A client must know its model
  from the app.
- **Not an acceptance of files, audio or documents.** Those keep their `unsupported_content` 400 with
  the message narrowed to what is actually missing.
- **Not a claim about a full generated turn over the API.** The assembly, the ids, the price and the
  placement are pinned; the generation is not, because an app-hosted test has no model loaded and the
  published-turn harness drives `TurnRunner` directly rather than through the server.
- **Not a re-opening of any perf question.** Nothing in this slice dispatches a kernel.

### Verification

`xcodebuild build-for-testing` at `-O` into `/tmp/dd_dsv41_77`, then `test-without-building` for the
nine server-logic classes (`/tmp/test77_logic3.log`, 162/0). The checkpoint-gated prompt case ran with
`touch /tmp/dynamoe_dsv41_real_turn` (`/tmp/test77_apiimg1.log`, 2/0) and printed the line above.
`DeepSeekV41LogicTests`, `DeepSeekV41TurnLoopTests`, `DeepSeekV41SlotWiringTests` and
`DeepSeekV41DeviceParityTests` are green (`/tmp/test77_ds1.log`, 31/0).

One suite-order note, which is the documented pattern and not this slice: `ServerQueueLogicTests`
loses 4-6 of 13 when it shares a process with the adapter suites (`/tmp/test77_logic2.log`,
`/tmp/test77_queue2.log`) and is **13/13 alone** (`/tmp/test77_queue1.log`). It also loses them
without the new test class in the same batch, so the new file is not the cause.

## DSV4.1 #78: AN IMAGE REQUEST NOW ANSWERS OVER THE REAL SERVER, AND THE MODEL DESCRIBES THE PICTURE IT WAS SENT

### The question

`#77` let the local server accept image content parts and pinned everything it could reach without a
loaded model: the two wire spellings, the marker placement, the image-token count, the expanded prompt
budget, the session-key digest. It said so in its own section —

> **Not a claim about a full generated turn over the API.** The assembly, the ids, the price and the
> placement are pinned; the generation is not, because an app-hosted test has no model loaded and the
> published-turn harness drives `TurnRunner` directly rather than through the server.

That left the largest untested stretch in the whole image path, and not a small one. Everything below
`TurnPlan.request.images` is the code that runs in production and nothing else: `launchTurn` prices the
spans from the resize plan, provisions the KV cache for the **expanded** prompt (not the
placeholders), builds or reuses the DeepSeek runtime, calls `DeepSeekV41TurnImages.bind` which runs the
tower and places its rows, asserts the bound length equals the priced one, hands the expanded ids to the
prefix pin and the sampler, and streams the turn. A mistake anywhere along it is a refusal or a 500 —
never a wrong token — so no arithmetic test can see it and no unit test had ever executed it.

It is also worth stating plainly: `CompletionEngine.shared.configure` appears **nowhere** in the test
target before this slice. Every DeepSeek case in `DeepSeekV41PublishedTurnTests` builds its own
`DeepSeekV41Runtime` and calls `DeepSeekV41TurnRunner.run`. The app's own turn path had never run on
the published checkpoint at all, with text or with pictures.

Routing put it first without reservation: `next_slice` = `live_image_turn` **0.991 @ 0.989** against
shared-expert rows 0.0016, `select`'s slots 0.0003, `pool`'s BF16 pair 0.0001 and a docs pass 0.0069;
`gap_is_real` **0.9996**; `perf_exhausted` **0.9997** (the named remainders are ~4.5% of a position and
every dominant term is at a measured ceiling).

### What landed

One opt-in case, `testAnAPIRequestWithAnImageAnswersOverTheLocalServer`, and three helpers:

- **The load is the app's own sequence**, not a shortcut: `DynaMoeEngine(filePath:)` →
  `getSummary()` → `InferenceEngine.segmentShardsForMetal` at the device's buffer limit → one
  `makeBuffer(bytesNoCopy: .storageModeShared)` per segment → `CompletionEngine.shared.configure`.
  The `DynaMoeEngine` is held for the life of the case, because the buffers point into its mappings.
  A test that built its own summary would not be exercising the load path the server's turns run on.
- **The server is the real one** — `ServerController.shared.start()` on a private port (8791, so a
  developer's own DynaMoE on 8787 cannot turn a bind failure into a connection error), hit with
  `URLSession` exactly as a client would. Every default the server consults (`enabled`, `port`,
  `api_key`, body capture) is saved, set and restored, and the request log is pointed at an
  `inMemory()` store so an app-hosted run never appends to the developer's log.
- **The request is a real one**: `content: [{"type":"text"}, {"type":"image_url"}]` with a real PNG
  base64'd into a `data:` URL, `max_tokens: 24`, no auth.

### The check is a delta, because the server renders its own prompt

The prompt the server tokenizes includes the app's whole system block — identity, date, the
checkpoint's own instruction shape — which is 190 tokens for a one-line question. A test that
re-derived that string would be pinning a second copy of the template, and the first time the two
drifted it would fail for the wrong reason. So the same question is sent twice, plain and with one
attachment, and the counted prompt tokens must differ by exactly

```
marker + newline   (2 tokens at this checkpoint's tokenizer, measured by #35's 10 -> 12)
+ spanPositions - 1   (the span replaces the marker's single token)
```

Both numbers come from outside the server: the span is `DeepSeekV41VisionPricing.estimate` — the same
function the engine prices with — applied to the pixels `DeepSeekV41ImagePixels.decode` produces, and
the marker cost is `#35`'s. A prompt that never saw the marker counts the same as the plain one, and a
span that bound nothing costs one token; both fail this assertion by 183.

### Evidence

```
🔎 [dsv41 real] live server: 96085 tensors in 48 shard(s), mapped 48/48 segment(s), 475.2508442029357 GB
🔎 [dsv41 real] live image turn: plain 190 prompt tokens / 23 generated, pictured 375 / 23,
   span 184 position(s) over 1521 patch token(s), 0.32 s of tower
🔎 [dsv41 real] live image answer: "The user has shared an image that appears to be entirely
   red/maroon colored with no visible content. Let me"
Test Case '…testAnAPIRequestWithAnImageAnswersOverTheLocalServer' passed (117.739 seconds).
```

`375 - 190 = 185 = 2 + 184 - 1`. The 2x2 fixture is the smallest thing ImageIO decodes, and the
checkpoint's `min_pixels` floor turns it into 1,521 patch tokens on a 13x13 aligner grid, i.e. 184
prompt positions — the same span `#38` priced and `#43` encodes.

The answer is the point of the slice. The picture is a **solid red square**, and the model says so:
*"an image that appears to be entirely red/maroon colored with no visible content."* Nothing upstream
of the tower could have produced that sentence — the pixels went through `ImagePixels.decode`, the
device tower, the span bind, the expanded prompt and the cache in the order `launchTurn` prescribes,
and came out as a description of the right colour.

The first arm ran the same request at `max_tokens: 8` and stopped mid-sentence (*"The user has shared
an image that"*, 7 tokens), which is why the budget is 24: a truncated reasoning channel hides the one
observation worth printing. Two runs of the finished case agree on every number and vary only in
wording — *"entirely red/maroon colored with no visible content"* and *"entirely solid red/crimson
colored with no discernible content"* — which is the server's default sampler doing what a `min_p`
filter at temperature 0 allows. So the case asserts only that the answer is non-empty and that the
token accounting is exact; the colour is the model's evidence and not the port's contract.

Cost: **117.7 s** for the load plus both turns — a 375-position image prefill, a 190-position text
prefill and 46 decoded tokens — behind a 0.32 s encode. That is the API image turn `#77` priced at
~78 ms/position and it lands where the price said.

The row contract is asserted too: both turns log a `/v1/chat/completions` row, the image row's
`input_tokens` is the **expanded** 375 (not the 191 the placeholders would count), and its captured
body names the picture (`"image_parts":1`) rather than carrying it.

### What it does not mean

- **Not a faster image turn.** Nothing here dispatches a new kernel; the tower is `#43`'s and the
  prefill is `#75`/`#76`'s. A largest image is still ~1,005 positions of prefill.
- **Not asserted on content.** The answer describing red is evidence the pixels arrived, not a test
  the model must keep passing: a wording change is not a regression, so the case prints it and asserts
  only non-emptiness plus the token accounting.
- **Only the OpenAI protocol is driven end to end.** The Anthropic adapter's image blocks share
  `attachImageMarkers` and `ServerImageParts`, and `#77` pins their decoding and their turn placement;
  what this case adds is the engine seam beneath both, which is protocol-independent.
- **Not a queue or lifecycle change.** `TurnExecutionQueue`'s single-flight behaviour, the 503 for a
  concurrent in-app turn, and `/v1/models`' silence about vision are all as `#77` left them.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_78`, then `test-without-building` **with the same
`-derivedDataPath`** for the single case, gated on `touch /tmp/dynamoe_dsv41_real_turn` and removed in
the same command (`/tmp/test78_run2.log`, `/tmp/test78_run3.log`, 1/0 each). Bundle freshness checked
with `strings … | grep -c testAnAPIRequestWithAnImageAnswersOverTheLocalServer` (1).

One trap worth recording, because it produced a green run that proved nothing: **`-derivedDataPath`
has to be passed to `test-without-building` as well as to `build-for-testing`.** Omitted, the run
silently used the default DerivedData bundle, which predates `#77` — the suite reported
`Test Suite 'DeepSeekV41PublishedTurnTests' passed` with **0 tests executed**, and a class-wide run
listed 39 names with neither of the two newest API cases among them. The `status=0` and the "passed"
line are exactly the shape of a real pass, which is the same silent-zero trap the repo already
documents for `-destination` and for the two-part `-only-testing` form.
## DSV4.1 #79: THE ANTHROPIC ROUTE CARRIES A PICTURE TOO, AND IT RENDERS A DIFFERENT PROMPT THAN THE OPENAI ONE

### The question

The image path was proven in three pieces and one of them was still the wrong shape. `#77` pinned both
wire spellings without a model (the OpenAI `image_url` object, the Anthropic `source` object, the marker
placement, the image-token count, the expanded budget, the session digest). `#78` then ran one turn end
to end over real HTTP — but through `/v1/chat/completions` only, on the argument that the engine seam
beneath both adapters is protocol-independent.

That argument is true about the seam and false about everything above it. `/v1/messages` has its own
`Decodable` (`AnthropicMessagesRequest`, built by hand in `init(from:)` around a `ContentBlockIn`), its
own entry into the planner (`buildTurnPlan(_ apiRequest: AnthropicMessagesRequest)`), its own response
type (`AnthropicMessageResponse`, with reasoning and text as separate blocks), its own required field
(`max_tokens` is not optional here, unlike OpenAI's), and its own request-row key: the same picture is
`image_parts` on one protocol and `image_blocks` on the other. `image_blocks` had never been written by
a real request in a test — only by a hand-built `AnthropicMessagesRequest` value.

Routing was decisive without being unanimous: `next_slice` = `anthropic_image_wire` **0.778 @ 0.741**
against advertising vision in `/v1/models` 0.178, a docs pass 0.020, serve-side queueing 0.017, and the
three remaining prefill terms at 0.0032, 0.0021 and 0.0013. `perf_work_worth_doing` **0.012** — the
perf question is settled and one more A/B on a sub-1% term would not unsettle it.

### What landed

One opt-in case, `testAnAnthropicRequestWithAnImageAnswersOverTheLocalServer`, plus `postMessages` (the
wire helper) and `AnthropicTurn`. It reuses `#78`'s `withLocalServer` unchanged — same private-port
server, same saved-and-restored defaults, same in-memory store — on port **8792** so neither case can
inherit the other's listening socket if a `stop()` ever leaves one behind.

Two extractions came out of it, because the alternative was a third copy of a load sequence that has to
stay identical to the app's:

- `smallestRedSquarePNG()` — the 2x2 fixture whose floor-expanded encode costs 184 prompt positions.
- `loadPublishedCheckpointIntoTheSharedEngine(_:)` + `unloadSharedEngine()` — the app-order load
  (`DynaMoeEngine` → `getSummary()` → `segmentShardsForMetal` → one `makeBuffer(bytesNoCopy:)` per
  segment → `CompletionEngine.shared.configure`), with the `DynaMoeEngine` handed back so the caller
  holds the mappings every buffer points into. `#78`'s case now calls both instead of carrying its own
  copy, so there is one load in the file rather than two.

The request is a base64 `source` object, which is what the Anthropic SDKs send
(`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"…"}}`). No `thinking` block
is sent, and that is the only deliberate difference from `#78`'s body.

### Evidence

```
🔎 [dsv41 real] live server: 96085 tensors in 48 shard(s), mapped 48/48 segment(s), 475.2508442029357 GB
🔎 [dsv41 real] anthropic image turn: plain 165 input tokens / 23 generated, pictured 350 / 23,
   span 184 position(s) over 1521 patch token(s), 0.32 s of tower
🔎 [dsv41 real] anthropic image answer: "The image you shared appears to be completely solid red with
   no visible content, shapes, text, or objects. It" (stop max_tokens, 0 chars of thinking)
Test Case '…testAnAnthropicRequestWithAnImageAnswersOverTheLocalServer' passed (112.144 seconds).
🔎 [dsv41 real] live server: 96085 tensors in 48 shard(s), mapped 48/48 segment(s), 475.2508442029357 GB
🔎 [dsv41 real] live image turn: plain 190 prompt tokens / 23 generated, pictured 375 / 23,
   span 184 position(s) over 1521 patch token(s), 0.32 s of tower
🔎 [dsv41 real] live image answer: "The user has shared an image that appears to be entirely
   red/crimson colored with no visible content — it looks…" (finish length)
Test Case '…testAnAPIRequestWithAnImageAnswersOverTheLocalServer' passed (117.866 seconds).
	 Executed 2 tests, with 0 failures (0 unexpected) in 230.010 seconds
```

The delta holds on both routes and is the same number: `350 - 165 = 185` and `375 - 190 = 185`, each
`= 2 + 184 - 1`. The picture is a solid red square and both answers say so, on a route neither had run
before.

### The two protocols render a 25-token difference for "the same" request, and the adapters' thinking defaults are why

The one number nobody would have guessed: the plain turn is **165** input tokens on the Anthropic route
and **190** on the OpenAI one, from a byte-identical question and the same loaded checkpoint.

It is not a bug in either. `OpenAIAdapter`'s planner takes `enableThinking ?? true` — a request that says
nothing about reasoning gets thinking **on**, which is what the app itself does. `AnthropicAdapter`'s
takes `apiRequest.thinking?.type == "enabled"` — a request with no `thinking` block gets thinking **off**,
because Anthropic's protocol makes it an explicit opt-in and honouring that is the whole point of reading
the field. The 25 tokens are the family's graded-effort scaffold: the `Reasoning Effort: N (range 1-100…)`
line and the reasoning channel's opening marker are rendered for a thinking-on prompt and omitted for a
thinking-off one.

So a prompt length is not portable across the two protocols, and the delta check has to stay inside one
route — which is exactly what both cases do. It also means an unqualified "the prompt is N tokens" claim
about the server is incomplete: the same body is two different prompts, and the difference is 25 tokens
of scaffold.

### What it does not mean

- **Not the streaming path.** `anthropicStreamingResponse` is a separate writer with its own block
  lifecycle, and no case here streams.
- **Not `/v1/models`.** It still advertises no vision capability, so a client has no way to discover
  that the served model accepts pictures. That is the next completeness gap, not this one.
- **Not a queue or lifecycle change.** Single-flight is untouched: a request during an in-app turn still
  gets a 503 rather than queueing.
- **Not asserted on content.** As in `#78`, the colour is printed and the assertions are the delta, the
  non-emptiness and the row's `input_tokens`.
- **Not a second engine path.** Both protocols reach the same `attachImageMarkers`, the same
  `DeepSeekV41TurnImages.bind`, the same expanded-prompt cache provisioning. There is one image
  implementation with two front doors, and now both doors have been walked through.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_79` (status 0, only the two pre-existing
`ContentView.swift` Swift-6 actor warnings), then `test-without-building` **with the same
`-derivedDataPath`** and both cases in one run, gated on `touch /tmp/dynamoe_dsv41_real_turn` and removed
in the same command (`/tmp/test79_build1.log`, `/tmp/test79_run1.log`). Bundle freshness checked with
`strings …/DynaMoETests | grep -c testAnAnthropicRequestWithAnImageAnswersOverTheLocalServer` (1) before
running, because `#78`'s silent-zero trap is a green run that executes nothing. `Executed 2 tests, with 0
failures`, and each case's `passed` line is present, which is the only thing that distinguishes a real
run from that trap.
## DSV4.1 #80: THE STREAMING ROUTES CARRY A PICTURE TOO, AND A STREAMED REFUSAL WEARS A 200

### The question

`#78` and `#79` proved the image path end to end over real HTTP on both protocols, and both drove the
**non-streaming** route. The streaming route is a different program above the same engine seam:

- The plan is built *inside* `TurnExecutionQueue.streamTurn`'s closure, after the FIFO hands the turn the
  engine's single-flight slot — not on the request path. A picture is resolved by
  `buildTurnPlan` either way, but on a different call site, at a different time, in a `do`/`catch` whose
  failure path is a frame rather than an HTTP error.
- The expanded prompt length reaches the client through `promptTokensBox`: the OpenAI writer puts it in a
  final usage chunk with `choices: []`, the Anthropic writer in `message_start.message.usage.input_tokens`.
  Neither is the non-streaming `usage` object, and neither had ever been asked for an image's number.
- **A streamed failure wears a 200.** Both routes answer `200 text/event-stream` before the turn is
  planned, so a refused attachment arrives as an error frame and then a terminal marker. A test that only
  asserts "status 200 and some text" cannot tell a picture the model saw from a picture the server refused.

Routing: `next_slice` = `streaming_image_turn` **0.844 @ 0.819** against advertising vision in
`/v1/models` 0.140, single-flight queueing 0.007, a docs pass 0.004 and the three remaining prefill terms
at 0.0021, 0.0012 and 0.0011; `perf_still_worth_a_slice` **0.0014**.

### What landed

Two opt-in cases, `testAStreamedAPIRequestWithAnImageAnswersOverTheLocalServer` (port 8793) and
`testAStreamedAnthropicRequestWithAnImageAnswersOverTheLocalServer` (8794), plus five helpers:
`parseSSEFrames`, `postStreaming`, `streamChat`, `streamMessages` and `waitForLogRows`.

- `parseSSEFrames` splits the transcript the way the spec does: a frame is a block of lines ended by a
  blank line, `event:` names it, `data:` carries the payload (multi-line payloads joined), and a block with
  no field but a leading `:` is a comment heartbeat.
- `postStreaming` reads the chunked SSE body to EOF with `data(for:)` and asserts the two things a
  streaming route promises regardless of the turn's fate: `200` and `Content-Type: text/event-stream`.
- `waitForLogRows` polls the in-memory store, because a streaming row is recorded by the body closure's
  `defer` as the response finishes and can trail the client's EOF by a few milliseconds. `#78`'s
  single `fetch` was safe only because a non-streaming row is recorded before the route returns.
- Both cases reuse `#78`'s `withLocalServer` and `#79`'s `loadPublishedCheckpointIntoTheSharedEngine`
  unchanged, and keep the delta contract: plain turn and pictured turn, same question, same body.

Two deliberate differences from the non-streaming cases, both recorded in the code:

- **`max_tokens: 48` rather than 24.** On the OpenAI route reasoning is streamed into its own
  `reasoning_content` field (`resolvedInlineReasoning` is off by default), so the visible answer needs
  tokens the non-streaming case spent on the pre-amble it printed as content. 48 leaves room for both.
- **The error frame is collected and asserted nil.** That is the only signal a streamed refusal leaves.

### Evidence

```
🔎 [dsv41 real] live server: 96085 tensors in 48 shard(s), mapped 48/48 segment(s), 475.2508442029357 GB
🔎 [dsv41 real] streamed image turn: plain 190 prompt tokens / 47 generated over 51 frame(s),
   pictured 375 / 47 over 52, span 184 position(s)
🔎 [dsv41 real] streamed image answer: "The user has shared an image that appears to be entirely
   red/crimson colored with no visible content — it looks like a solid color block, possibly a
   placeholder or an image that failed to load properly. I should describe what I see"
   (0 chars of reasoning, 264 heartbeat(s), done true)
Test Case '…testAStreamedAPIRequestWithAnImageAnswersOverTheLocalServer' passed (122.841 seconds).
🔎 [dsv41 real] live server: 96085 tensors in 48 shard(s), mapped 48/48 segment(s), 475.2508442029357 GB
🔎 [dsv41 real] streamed anthropic image turn: plain 165 input tokens / 46 generated over
   79 frame(s), pictured 350 / 32 over 288, span 184 position(s)
🔎 [dsv41 real] streamed anthropic image answer: "The image you shared appears to be completely
   solid red with no visible content, shapes, text, or objects—just a uniform red color filling
   the entire frame." (0 chars of reasoning, 251 heartbeat(s), stopped true)
Test Case '…testAStreamedAnthropicRequestWithAnImageAnswersOverTheLocalServer' passed (115.401 seconds).
	 Executed 2 tests, with 0 failures (0 unexpected) in 238.241 seconds
```

The delta is the same 185 on both routes and on both transports: `375 - 190` and `350 - 165`, each
`= 2 + 184 - 1`. The rows are the other half of the claim and are asserted as such: both turns logged
`wasStreaming`, both `streamTermination == .completed`, and the image row's `inputTokens` is the
**expanded** count the usage frame reported (`"image_parts":1` / `"image_blocks":1` in the captured body).

### The two writers keep the connection warm in two different ways, and the first run's instrument said they did not

The first run printed `0 heartbeat(s)` for the Anthropic pictured turn and `266` for the OpenAI one, on
turns that took the same ~95 s of image prefill. That read like a real asymmetry, and it was the
instrument: OpenAI's keep-alive is `writer.write(":\(comment)\n\n")` — an SSE **comment block**, which is
what `parseSSEFrames` counts as a heartbeat — while the Anthropic writer calls `sse("ping",
AnthropicPingEvent())`, which emits a named `event: ping` frame with a JSON payload. The parser counted
those as payload frames, so the Anthropic turn's 251 keep-alives were sitting inside its 288 "frames".

Fixed before anything was written down, by counting `event: ping` as a heartbeat on that route: **264
keep-alives on the OpenAI route and 251 on the Anthropic one**, against 52 and 288 total frames. So both
writers do hold a long image prefill open, at roughly a 0.35 s cadence, and the earlier number was a
parser classifying the two mechanisms differently.

### What it does not mean

- **Not a latency claim.** These cases prove the transcript, not the cadence: the frames are asserted for
  order and content, and no timing is asserted anywhere.
- **Not `/v1/models`.** It still advertises no vision capability, so a client cannot discover that the
  served model accepts pictures. That remains the next completeness gap.
- **Not a queue or lifecycle change.** Single-flight is untouched: a concurrent request during an in-app
  turn still gets a 503 rather than queueing behind it, and the streaming routes inherit that as
  `#77` left it.
- **Not a second image path.** Both protocols still reach the same `attachImageMarkers`, the same
  `DeepSeekV41TurnImages.bind` and the same expanded-prompt cache provisioning. Four cases now walk the
  two protocols on both transports to one implementation.
- **Not deterministic text.** Two runs of the Anthropic case generated 47 and 32 tokens from the same
  prompt (`top_p` sampling), so the assertions are the delta, the terminal marker, non-emptiness and the
  row. The wording is printed as evidence and never asserted.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_80` twice (status 0 both times; only the two
pre-existing `ContentView.swift` Swift-6 actor warnings), then `test-without-building` **with the same
`-derivedDataPath`** for both cases in one run each time — `/tmp/test80_run1.log` before the heartbeat fix
and `/tmp/test80_run2.log` after it, `Executed 2 tests, with 0 failures` in both, with each case's
`passed` line present. Bundle freshness checked with
`strings … | grep -c "testAStreamedAPI…\|testAStreamedAnthropic…"` (2) before running, because `#78`'s
silent-zero trap is a green run that executes nothing.
## DSV4.1 #81: THE MODELS LIST NOW SAYS WHAT THE MODEL CAN DO, FROM THE SAME FACTS THE PLANNER GATES ON

### The question

Three consecutive slices ended by naming the same hole. `#78` proved a non-streaming image turn answers;
`#79` did it for the Anthropic route; `#80` did it for both streaming transports. Each of them then had to
write down that **a client had no way to learn any of it**: `GET /v1/models` answered with

```json
{"object":"list","data":[{"id":"<snapshot hash>","object":"model","created":1793830186,"owned_by":"dynamoe"}]}
```

An id and a timestamp. So the only way to discover that the served checkpoint reads pictures was to send
one and see whether it came back as an answer or a 400, and the only way to discover the prompt ceiling was
to exceed it, and the only way to discover whether a reasoning request would be honoured was to ask and
watch.

That is not a cosmetic gap on this branch. The capability exists, is exercised end to end on both
protocols and both transports, and was undiscoverable.

Routing: `next_slice` = `models_vision_capability` **0.986 @ 0.983** against a larger "capability plus
image-cost metadata" slice 0.011, a docs pass 0.0017, single-flight queueing 0.0014 and the three remaining
prefill terms at 0.0002, 0.0002 and 0.0001. `declare_vision_before_it_is_device_verified` **0.998**: the
capability is real for a client today (a picture is accepted and answered), only slower than text, and
`#43`'s encode is a device pass, so advertising it is not an overclaim.

### What landed

`OpenAIModelList.Model` gained three additive fields and a factory, and the route that builds it stayed six
lines:

- **`capabilities: { vision, thinking, thinking_levels }`** — copied from `LoadedModelInfo`, which is the
  same snapshot of facts `buildTurnPlan` reads. `vision` is `ModelConfig.supportsVision`, whose own gate is
  the tower's files rather than the model row's flag, so a half-downloaded checkpoint cannot advertise a
  tower it cannot run. Deliberately NOT a hand-written list: a capability the planner does not gate on
  would be marketing, and a swapped flag here is silent on the wire.
- **`context_length`** — the checkpoint's own architectural ceiling, `effectiveMaxPositionEmbeddings`
  (1,048,576 here). Not this machine's KV budget: a client asking "how much fits" wants the weights'
  answer, and the two numbers are far apart on purpose.
- **`max_prompt_tokens`** — the prompt length the server actually refuses
  (`dynamoe_server_max_prompt_tokens`, 262,144 by default), so a client can tell the weights' ceiling from
  the server's. The "no cap" sentinel `Int.max` is reported as **nil**, because a client's JSON float
  conversion would show `9.22e18` as a number rather than as "unlimited".
- `OpenAIModelList.entry(for:contextLength:maxPromptTokens:created:)` is the same three-line mapping as a
  free function of the facts, so it is testable with no loaded model.

Nothing about the id changed: it is still the snapshot directory's own name (a hash). That is a real
discoverability wart of its own, but it is what every response's `model` field carries and what the
turn-memory digest and prefix session keys are built from, so renaming it would invalidate recorded turns
across an update. Recorded here as a known limitation rather than changed inside a slice about
capabilities.

### Evidence

The live check, on the published checkpoint (`testTheServerAdvertisesWhatTheLoadedCheckpointCanDo`):

```
🔎 [dsv41 real] live server: 96085 tensors in 48 shard(s), mapped 48/48 segment(s), 475.2508442029357 GB
🔎 [dsv41 real] /v1/models: {"object":"list","data":[{"context_length":1048576,"created":1793830186,
   "object":"model","owned_by":"dynamoe","capabilities":{"vision":true,"thinking_levels":true,
   "thinking":true},"max_prompt_tokens":262144,"id":"1bae0e1088993b9b82de08f59de5be5a34e42170"}]}
Test Case '…testTheServerAdvertisesWhatTheLoadedCheckpointCanDo' passed (1.716 seconds).
```

**1.716 seconds for the whole case** — the app-order load, the server start and the GET. That is the first
place the numbers say something new about the harness itself: the load of a 475 GB, 96,085-tensor
checkpoint is ~1.5 s (it indexes shard headers through the mmap rather than reading weights), so
`#78`/`#79`/`#80`'s ~115 s per case is the TURNS and essentially none of it is the load. A capability check
therefore costs nothing and belongs in the suite rather than in a one-off.

The three assertions that matter are cross-checks against something outside the wire, not against a second
copy of the mapping: `context_length` against `modelConfig.effectiveMaxPositionEmbeddings` from the
config on disk, `thinking_levels` against `ModelConfig.supportsThinkingLevels(modelPath:)` (the planner's
own gate), and `max_prompt_tokens` against `ServerController.resolvedMaxPromptTokens` (what the server
enforces). `vision` is asserted true because this checkpoint publishes a tower, which is exactly the fact
`#78`-`#80` spend their turns proving.

Two pure cases pin the mapping without a model
(`testModelListAdvertisesTheCapabilitiesThePlannerGatesOn`,
`testModelListCapabilitiesFollowTheLoadedModelRatherThanTheWire`): the snake_case keys a client sees, that
Swift property names do not leak (`thinkingLevels`), that a tower-less model reports `vision: false` rather
than a hardcoded true, and that "no ceiling" and "no cap" both encode as absent rather than as sentinels.

### What it does not mean

- **Not a promise that the context is servable.** `context_length` is the weights' ceiling; the KV budget
  still decides what a given machine affords, and a request past `max_prompt_tokens` is still a 400 with
  `code: "context_length_exceeded"`.
- **Not an image price.** Nothing advertises that a picture costs 184-1,005 prompt positions or 0.32-6.90 s
  of tower. That was the larger slice this one beat on routing, and it needs the resize plan's own
  arithmetic rather than a constant.
- **Not an id change.** Still the snapshot hash; see above.
- **Not a per-model list.** Exactly one entry, because the app serves exactly one loaded model; no loaded
  model is still `{"object":"list","data":[]}`.
- **Not an Anthropic-shaped models route.** `GET /v1/models` answers in the OpenAI shape on both protocols,
  as it always has.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_81` (status 0, only the two pre-existing
`ContentView.swift` Swift-6 actor warnings), then `test-without-building` with the same `-derivedDataPath`:
the two pure cases from `/tmp/test81_run1.log` (`Executed 2 tests, with 0 failures`) and the live case from
`/tmp/test81_run2.log`, gated on `touch /tmp/dynamoe_dsv41_real_turn` and removed in the same command.
Bundle freshness checked with `strings … | grep -c testTheServerAdvertisesWhatTheLoadedCheckpointCanDo`
(1) before running.

The existing app-hosted smoke case (`ServerControllerSmokeTests.testModelsEndpointServesOpenAIListShape`,
0.943 s) still passes unchanged: it asserts the list shape and the id, which is exactly the part this slice
left alone.

## DSV4.1 #82: A PICTURE'S PRICE IS A RANGE, BECAUSE THE PLAN PRICES ITS SHAPE AND NOT JUST ITS PIXELS

### The question

`#81` made `GET /v1/models` say what the served checkpoint can do, and left one hole in it standing:
`vision: true` says a picture is accepted and nothing about what it costs. That is not a marketing
detail on this model. The same slot in a prompt costs a client **184 positions and 0.32 s** or
**994 positions and 6.95 s** depending only on which picture is put in it (`#35`, `#43`), and a
client choosing a request timeout, or deciding whether a near-full prompt has room for an
attachment, cannot tell from a boolean which end it is sizing for. The composer has shown this
per-image since `#38` (`ModelConfig.visionCostEstimate` at the point of attach); the wire had no
equivalent.

Routing: `next_slice` = `image_cost_metadata` **0.900 @ 0.880**, ahead of a docs-only reconciliation
0.084, single-flight queueing 0.010, and the three remaining prefill terms at 0.0021, 0.0019 and
0.0015. `docs_is_a_slice` **0.666** put the prose repairs in the mid-band, so they rode along on this
commit instead of taking one of their own. `serve_queueing_should_change` **0.059**: the single-flight
`503` stays.

### What was added

| | where | what it is |
|---|---|---|
| `DeepSeekV41VisionPricing.Bounds` + `bounds(spec:costModel:)` | `DeepSeekV41VisionCost.swift` | the cost envelope of one attachment, from the resize plan alone |
| `ModelConfig.visionCostBounds(modelPath:)` | `ModelConfig.swift` | the same per checkpoint, beside the per-image estimate the composer uses |
| `LoadedModelInfo.visionCost` | `CompletionEngine.swift` | the fact, snapshotted with the other `/v1/models` facts |
| `Capabilities.VisionCost` / `vision_cost` | `OpenAIAdapter.swift` | the wire form, six keys, additive to the OpenAI schema |

```json
"capabilities": {"vision": true, "thinking": true, "thinking_levels": true,
  "vision_cost": {"min_prompt_positions": 184, "max_prompt_positions": 1024,
                  "min_patch_tokens": 1521, "max_patch_tokens": 9216,
                  "min_encode_seconds": 0.32, "max_encode_seconds": 7.84}}
```

### The finding: a single price is not derivable, and the plan's own cap is the ceiling

`estimate(width:height:)` prices ONE picture. A client that has not been sent one needs the envelope,
and the envelope is not the price of the largest square: **the resize plan prices an aspect ratio as
well as a pixel count**, because the cap is applied in TOKEN space (`fit` trims the aligner grid until
`imageTokenCount` fits `max_image_tokens`) and the floor is applied in PIXEL space, with both sides
rounded up to a patch first. Two images of identical area therefore cost different amounts.

What makes a bound possible without a search is that `fit` trims until the **span** fits, not until
a grid area fits, so:

- no picture can put more than `max_image_tokens` positions in a prompt, and a one-row span reaches
  that exactly (`42,882x42` → a 1,021x1 grid → span **1,024**);
- patches never exceed `downsample_ratio^2` per aligner cell, so that widest span carries at most
  `3^2 * 1,024 = 9,216` of them (`42,882x42` → **9,189**);
- the smallest accepted image is a one-pixel one raised to `min_pixels`, and a square needs the
  fewest patches: `fit(1,1)` → 546x546 → **1,521** patches on a 13x13 aligner grid → **184**
  positions (`#35`'s own measurement).

Both patch ends are then priced with `DeepSeekV41VisionCostModel.measured`, the same model the
composer quotes, so the wire range cannot disagree with the per-image chip: 0.32 s and 7.84 s.

### What it does not mean

- **Not the tightest number the server can produce.** The span ceiling of 1,024 is the plan's, and
  the plan alone can reach it; the app's image decoder refuses a source wider than
  `DeepSeekV41ImagePixels.maximumSourceSide` (4,096 px a side), and the 1,021:1 strip that attains it
  is 42,882 px wide. Through the server the ceiling is the largest square, 8,649 patches and 994
  positions (`#37`) — so the advertised bound is 3% high on positions and 6.5% high on patches. It is
  a ceiling by construction; erring high is the direction that keeps a client's timeout honest.
  Deriving the decoder-limited maximum would need a search over aspect ratios for a 3% tighter number.
- **Not a per-request price.** It says what an attachment CAN cost, not what the one in a given turn
  does; the engine still prices the actual picture before it provisions anything, via
  `TurnImages.price`, and for the largest accepted image that is 994 of a 1,048,576-position context.
- **Not rounded because it is unknown.** Two decimals on the seconds, because the model is a fit to
  three measured points on one machine class rather than a measurement of the client's hardware; the
  positions and patch counts are exact integers from the plan.
- **Not advertised for a tower-less model.** A checkpoint can publish a `vision_config` (and so a
  resize plan) without shipping the tower's weights, which is exactly the half-downloaded case
  `supportsVision` refuses for the composer. `entry(for:)` gates the cost block on `supportsVision`,
  so the two cannot disagree about whether an attachment will be read.
- **Not a change to the single-flight contract, in this slice.** A request that arrives during an
  in-app turn still got a `503` rather than queueing here (0.059 against changing it), which `#83`
  then took on anyway — on the user's instruction, and it turned up the defect underneath the
  policy question: the queue was not serializing concurrent enqueues at all. Nothing about the
  cost metadata above depends on that.

### The two prose repairs that rode along

Both were standing side by side in this journal, contradicting one another:

- **`indexer wait` 11.3 against 8.0-9.0** is now reconciled at `#75`, where the second figure is
  tabulated: it is one stop billing whatever the queue had not finished, so it follows the layout and
  the order of what was encoded ahead of it. `#64` split the same stop at 2.5 ms/position of indexer
  and the rest a drain of the dense half, and the two runs sat at different walls (99.7 against
  77.7-78.3 ms/position), so the difference is not the indexer shrinking.
- **`#75`'s "the opposite of what `#22`'s sweep predicted"** now carries its withdrawal in its own
  section, not only in `#76`'s: that sweep measured 0.57-0.84 ms after 10-50 ms of host idle, so it
  predicted 0.55 and the block's probe measured 0.55-0.60.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_82` three times (status 0 each), then
`test-without-building` with the same `-derivedDataPath` every time. The full build (`build2`) reports
no error and no warning in any touched file: the only diagnostics against them are
`ModelConfig.swift:1110`'s actor-isolated `VisionSpecBox` call, which is present in `#81`'s logs too.
One warning this slice did introduce was fixed rather than recorded: passing the mapping as a
function value (`info.visionCost.map(Capabilities.VisionCost.init)`) took the init in a nonisolated
context and warned, so it is a closure. Bundle freshness verified with
`strings … | grep -c <case>` (1 for all six cases) before running.

New pure cases: `DeepSeekV41VisionCostTests.testTheAdvertisedBoundsComeFromThePlansOwnFloorAndCap`,
`testNoAdmissibleImageEscapesTheAdvertisedBounds` (ten shapes including both thin extremes, every one
asserted inside all six bounds, and the cap asserted REACHED so the ceiling is not a direction nothing
can reach), `testAnUncappedPlanAdvertisesNothing`; and in `OpenAIAdapterLogicTests` the two `#81` cases,
extended to assert the six wire keys, the two-decimal rounding and that no Swift name leaks at either
nesting level, plus the tower-less case where a resize plan exists and `vision_cost` must be absent.
`DeepSeekV41PublishedTurnTests.testTheServerAdvertisesWhatTheLoadedCheckpointCanDo` gained the live
cross-check: `min_prompt_positions` 184 and `min_patch_tokens` 1,521 against `#35`'s measurement,
`max_prompt_positions` against the checkpoint's own `max_image_tokens`, the patch ceiling against
`downsample_ratio^2 * max_image_tokens`, and the second ceiling against
`DeepSeekV41VisionCostModel.measured` (a same-number check, not an inequality, because the wire
rounds).

Measured on the published checkpoint (`/tmp/test82_run2.log`, 1.683 s for the load, the server start and
the request together — the 475 GB load is header indexing, so a capability check is cheap and belongs in
the suite):

```json
{"object":"list","data":[{"id":"1bae0e1088993b9b82de08f59de5be5a34e42170","capabilities":
 {"thinking":true,"thinking_levels":true,"vision":true,"vision_cost":
  {"max_prompt_positions":1024,"min_patch_tokens":1521,"min_prompt_positions":184,
   "max_patch_tokens":9216,"min_encode_seconds":0.32,"max_encode_seconds":7.84}},
 "context_length":1048576,"created":1793831591,"owned_by":"dynamoe","object":"model",
 "max_prompt_tokens":262144}]}
```

and `ServerControllerSmokeTests.testModelsEndpointServesOpenAIListShape` still passes unchanged
(0.948 s, `/tmp/test82_run3.log`): it asserts the list shape and the id, which is the part this slice
left alone.

## DSV4.1 #83: THE FIFO WAS NOT A FIFO UNDER A BURST, AND AN IN-APP GENERATION NOW QUEUES INSTEAD OF REFUSING

### The question

Two things were wrong with the server's turn queueing, and they were the same thing:
`TurnExecutionQueue` is documented as "a FIFO over the engine's single-flight generation
slot", and it was neither.

1. **A request that arrived while the app was generating got `503` before it had waited at
   all.** `CompletionEngine.launchTurn` refuses a second turn (`engineNotReady`), and the
   server mapped that straight to `503 overloaded`. The queue serialized *server* turns only;
   the chat UI and the agent harness call `launchTurn` directly, so the one engine slot had a
   second holder the FIFO could not see. A client that asked while a chat answer was streaming
   was told the server was overloaded.
2. **Under a burst, the queue let every request through at once.** `perform`/`streamTurn` read
   `tail`, then did several things — including constructing and configuring
   `ServerTurnCancelBox`, a `MainActor`-isolated type in this target — before writing the new
   chain element. Those calls are hops off the queue actor, so a burst of enqueues could all
   read the SAME `tail` and chain off a turn that was not the previous one. The engine then
   refused all but the first as a second active turn: **23 of 24 concurrent requests returned
   `503`**, and the queue's own bookkeeping never noticed.

Routing: the user asked for this slice after `#82`; the previous run had already put
`serve_queueing_should_change` at **0.059**, so this is work taken on the user's instruction
rather than on the routing evidence — which is the right call in hindsight, because what it
found was not the policy question that ran (should an API turn queue behind an in-app turn)
but a defect underneath it.

### What was wrong, measured

`ServerQueueLogicTests.testConcurrentEnqueuesNeverRunTwoTurnsAtOnce` is the instrument: 24
independent enqueues, each turn incrementing a lock-guarded live counter on entry and
decrementing it on exit, 1 ms of work inside, and the peak recorded.

| arm | peak concurrent turns | finished |
|---|---|---|
| pre-fix queue, no deadline | **24** | 24 |
| pre-fix queue, 5 s deadline | **24** | 24 |
| shipped queue, either arm | **1** | 24 |

That case failed **5 of 5** runs against the pre-fix queue (`XCTAssertEqual failed: ("24") is
not equal to ("1")`) and passes 11 of 11 after. It also explains a pre-existing red test that
had been failing on `main`: `testPerformInvokesSlotOpenedOnlyWhenSlotOpens` asserts the
slot-open hook does not fire while the leader holds the slot, and failed on a clean baseline
build of the parent commit — the same stale-chain read, in the single-deadline form.

### What landed

- **The chain is read and advanced with no suspension in between.** A `SlotDrain` (one-shot,
  nonisolated lock box) is allocated, the gate `Task` is built to await it, `tail = gate`
  happens immediately, and only THEN are the cancel box, the hook registry and the work task
  created. The gate waits on `drained` instead of on `Task.value`, which is what lets it exist
  before the task does. Nothing that can hop actors sits between the read and the write.
- **`DrainFlag`, `ServerTurnCancelBox` and the new `SlotDrain` are `nonisolated`.** This target
  compiles with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so leaving them bare made them
  main-actor types called from the queue actor's critical path — the warnings that had been
  sitting in this region were pointing at exactly the calls that broke the FIFO. Six of them
  are gone.
- **The queue waits for the engine slot, not just for its predecessor.** `waitForSlot` now
  finishes with a poll of `CompletionEngine.isActive`, so an in-app generation is waited out
  the same way an earlier request is. The wait is bounded by the SAME deadline and fed by the
  SAME heartbeat, so nothing about the latency promise changes; what changes is that a request
  inside its budget is served rather than refused. Past the deadline it gets `408` with a
  message that names which of the two things it waited on — "the generation queue" or "a
  generation running in the DynaMoE app" — because a client debugging a 408 needs to know.
- **`slotProbe` is injectable** (`TurnExecutionQueue(slotProbe:)`), so the wait for an in-app
  turn can be tested without loading a 475 GB model. Production passes the default.

The wait sits BEFORE the `onSlotOpened` hook, deliberately: that hook is where the request log
rows stamp `queueWaitMs`, and a request that spent 40 s behind a chat answer has been waiting,
not generating.

### What it does not mean

- **Not a new knob and not unbounded latency.** The bound is the existing
  `dynamoe_server_queue_timeout` (default 120 s, `0` = wait forever). Nothing new to configure;
  the semantics of an existing setting simply now cover the case they described.
- **Not a change to what `stop()` may cancel.** `cancelLiveWork()` still cancels exactly the
  server-owned boxes, and the app's turn is still untouchable — the queue observes the engine
  slot, it does not own it.
- **Not proof that the queue is now race-free.** The invariant is stated and pinned ("no two
  turns inside the work at once, over 24 concurrent enqueues, on both wait paths"), not proven.
  What the five consecutive green suite runs do establish is that the burst case that failed
  5 of 5 now passes 11 of 11.
- **Not the end of the single-flight `503`.** A request can still hit it if an in-app turn
  starts in the microseconds between the slot opening and `launchTurn` — a hop, not a wait.
  That window is a design seam (the queue cannot reserve the engine slot), and it is rare;
  closing it would mean giving `launchTurn` a reservation API.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_83` (status 0; the queue region's
actor-isolation warnings have dropped from ten to four, and the remaining ones are in route
handlers this slice did not touch). `test-without-building` with the same `-derivedDataPath`:

| suite | result |
|---|---|
| `ServerQueueLogicTests` (19 cases, now including 5 for the in-app wait and 1 for the burst) | **5 consecutive green runs** after the fix, 3 more after the burst test widened to both wait paths |
| `ServerStreamSafetyTests`, `ServerHardeningLogicTests`, `ServerToolArgumentTests`, `AnthropicAdapterLogicTests`, `OpenAIAdapterLogicTests`, `TurnToolMaskTests`, `ReasoningRoundTripTests` | 90 cases, 0 failures |
| `ServerControllerSmokeTests` (app-hosted, real server on 8787) | 17 cases, 0 failures |

The pre-fix arm is a real A/B, not an estimate: the parent commit's `ServerController.swift`
was restored, built into `/tmp/dd_dsv41_pre`, and the burst case run against it five times.

## DSV4.1 #84: THE SLOT IS NOW HELD, NOT JUST LOOKED AT, AND THE QUEUE'S LAST 503 WINDOW IS CLOSED

### The question

`#83` closed with the seam it could not close: "a request can still hit it if an in-app turn
starts in the microseconds between the slot opening and `launchTurn` — a hop, not a wait. That
window is a design seam (the queue cannot reserve the engine slot), and it is rare; closing it
would mean giving `launchTurn` a reservation API."

Two things in that sentence turned out to be wrong, and they are the reason this slice was
worth doing.

1. **The window is not microseconds.** Between the probe and the launch sits the queue's
   whole `work` closure: `buildTurnPlan` renders the prompt and tokenizes it, the turn memory
   is looked up, the prefix pin is consulted, images are priced. On a long API prompt that is
   not a hop, it is the preparation. The seam was as wide as a turn's setup, not as wide as a
   scheduling hop. (Unmeasured — this is a code fact about what sits between the two calls, and
   it is why the fix had to hold the slot rather than shorten the gap.)
2. **The party that paid for it was the wrong one.** The client that lost was the one that had
   already waited out the FIFO, its queue timeout, and its heartbeats, and got `503` at the
   end of it; the chat turn that caused it ran to completion. A seam that fails the caller who
   has invested the most waiting is the worst available ordering.

Routing: `next_slice` = `reservation_api` at **0.518 @ 0.438**, with `image_cost_ceiling` at
0.296 and every perf candidate under 0.12; and `measurement_slice_is_worth_it` = **0.045**,
decisively against spending the session on another stage table. The choice was mid-band and
low-confidence, so it was taken on the second answer plus the reasoning above rather than on
the probability: what is left of the perf journal is each under ~1% of a prefill position
(`#73`'s three FP8 projections are the dense half, `#74`'s `select` rail is 1.1% of a
position, `#72`'s shared-expert lever is under 1%), while this is a correctness hole in the
contract `#83` had just shipped.

### What was added

**One new primitive** (`EngineTurnSlot.swift`):

- `TurnSlotClaim` — a hold on the engine's single-flight generation slot. Only one can be
  outstanding, `release()` is idempotent, and it is a ticket rather than a flag so a claim that
  was already given back cannot release the slot out from under whoever holds it now (both the
  queue and the launch path release on their way out).
- `TurnSlotLedger` — the lock box that grants the one claim. `nonisolated`, like the queue's
  other boxes: this target builds with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, and a
  release that had to hop to the main actor could land after the next turn had already looked.

**On the engine** (`CompletionEngine.swift`):

- `reserveTurnSlot() -> TurnSlotClaim?` — nil when a turn is running or another caller holds
  the slot. This is the queue's only way in.
- `isActive` is `currentGenerationTask != nil || turnSlotLedger.isClaimed`: a turn being
  PREPARED occupies the slot exactly like a running one, which is what makes the claim
  visible to every other prober (the queue's own, `MCPManager`'s config-reload deferral, the
  UI's busy state).
- `launchTurn(_:claim:)` takes the slot on the way in — the caller's claim if it has one, or
  one of its own, in the same step, so two launches can never both see a free slot. A launch
  that holds no claim while one is outstanding is refused with its own message ("the local API
  server is starting a generation"), and a `defer` gives the claim back on every one of the
  function's many early throws, which is what keeps a refused turn from wedging the server.

**On the queue** (`ServerController.swift`):

- `slotReserve` joins `slotProbe` as an injectable `@Sendable` closure (so the whole seam is
  testable with no model, the way `#83`'s probe already was), and `SlotWaitOutcome.opened` now
  carries the claim.
- A probe that says free followed by a claim that FAILS is a retry, not a refusal: somebody
  else took the slot in that instant, so the turn goes back to waiting inside
  `waitForSlot`, still bounded by its deadline. Nothing about the request is thrown away.
- `perform`/`streamTurn` hand the claim to the work closure and release it in the same `defer`
  that drains the queue, so a claim the turn never redeemed (a timeout, a cancellation, a work
  closure that threw) goes back. `runTurn` and both streaming writers now pass it to
  `launchTurn`.

### What it says

- **Reserving converts a lost race into a wait, provided the loser is still waiting.** That is
  the whole shape of the fix: `#83` made the queue wait for the slot, and the wait was undone
  by a check-then-take pair. Taking the slot in one step is all it needed.
- **The seam could not be closed by retrying the work closure.** That is the alternative that
  looks free and is not: `work` is opaque to the queue, so "the slot was gone" arrives as an
  `engineNotReady` that is indistinguishable from "no model is loaded" or "this checkpoint
  cannot run", and re-running the closure would re-render the prompt, re-tokenize it and
  re-record the request-log box. The claim is what makes the loss distinguishable and
  non-fatal, and it makes `work` run exactly once.
- **The fix moves who can lose, it does not remove losing.** An in-app turn that arrives while
  a server request is being prepared is now refused at its own launch with a message naming
  the server, where before it would have won and the request would have failed. That is a
  real, deliberate trade and it is scoped as narrowly as it can be: the losing case is a chat
  send landing inside another turn's setup, it is immediate rather than after a wait, and it
  is retryable by pressing send again. It is also not a new class of failure for the user —
  a chat turn sent while a server turn was RUNNING already failed with `engineNotReady`.
- **Every early throw in `launchTurn` had to give the claim back.** The function has dozens of
  refusal paths (no model, no tokenizer, an image on a family with no tower, a checkpoint that
  cannot be provisioned, a prompt that priced differently than it bound). One `defer` covers
  all of them, and `testARefusedLaunchWithoutAClaimGivesTheSlotBack` is what says so.

### What it is not (do not re-propose)

- **Not a change to what a `503` means.** A launch that fails for the model's own reasons is
  still a `503` through `wireLaunchError`, and `ServerControllerSmokeTests` still pins those
  envelopes. What no longer produces a `503` is *winning the probe and losing the slot*.
- **Not a new knob.** No setting, no timeout, no queue policy: the claim is invisible to
  clients and the existing `dynamoe_server_queue_timeout` still bounds the wait.
- **Not the queue owning the engine slot.** The queue observes and holds a claim; `stop()`
  still cancels exactly the server-owned boxes and still cannot touch the app's turn.
- **Not proof of race-freedom.** The invariant is stated and pinned ("one claim at a time, and
  every path gives it back"), not proven. The 24-enqueue burst case now also runs through the
  real ledger, which is the closest thing to a proof this suite offers.
- **Not the last word on the in-app side.** A chat turn whose own preparation (prompt
  formatting in `ContentView`) straddles a server claim still loses, because that preparation
  happens before `launchTurn` and there is nowhere to hold a slot for it. Making the chat path
  reserve early means making `startAutoregressiveGeneration` async; it is not needed while the
  loss is bounded by a turn's setup, and it is the obvious next step if it ever bites.
- Everything from `#51`-`#83` still stands.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_84` (status 0). The warning set is
**identical** to `#83`'s build, line for line — 1,773 warnings, `diff` clean — so this slice
introduced none, despite touching the actor-isolation-sensitive part of `ServerController`.

`test-without-building` with the same `-derivedDataPath`:

| suite | result |
|---|---|
| `EngineTurnSlotTests` (new: the ledger's one-at-a-time rule, a stale ticket, the engine looking busy under a claim, an in-app launch refused while a server claim is held, a claim-carrying launch honoured, and a refused launch giving the slot back) | **7 cases, 0 failures** |
| `ServerQueueLogicTests` (30 cases: `#83`'s 19 plus 4 for the claim — a lost claim retried rather than reported, the claim held for the whole turn, released when the work throws, and a claim another caller holds waited for rather than refused) | **30 cases, 0 failures** |
| `OpenAIAdapterLogicTests`, `AnthropicAdapterLogicTests`, `ServerStreamSafetyTests`, `ServerToolArgumentTests`, `ServerHardeningLogicTests`, `TurnToolMaskTests`, `ChatPromptFormatterConversationTests`, `ReasoningRoundTripTests`, `PrefixAsOfDateTests` | 119 cases, 0 failures |
| `ServerControllerSmokeTests` (app-hosted, real server on 8787) | 17 cases, 0 failures — including the `503` envelopes, which is what says the claim did not swallow the model-not-loaded path |

The live arm is `DeepSeekV41PublishedTurnTests.testAnAPIRequestWithAnImageAnswersOverTheLocalServer`
(opt-in, `/tmp/dynamoe_dsv41_real_turn`): a real request through the real server, whose launch
now carries the queue's claim into `launchTurn`. It passes (117 s, `/tmp/test84_live2.log`): the
checkpoint indexed as 96,085 tensors over 48 mapped segments (475.25 GB), the server came up on
127.0.0.1:8791, and two requests answered `200` — the plain prompt in 33.1 s (queue 77 ms) and
the same prompt with one attachment in 80.9 s (queue 76 ms), 190 prompt tokens against 375, a
span of 184 positions over 1,521 patch tokens from 0.32 s of tower, the model describing the
picture it was sent. So the claim travels from the queue through `work` and `withTurnToolMask`
into the engine and back, and the queue-wait stamps the server already logged are unchanged.

## DSV4.1 #85: THE CHAT PATH RESERVES TOO, SO THE SEAM #84 CLOSED IS CLOSED FROM BOTH SIDES

### The question

`#84` gave the engine a reservation API and wired exactly one caller into it: the API server's
queue, so a chat turn could not take the slot a queued request was preparing for. `#84`'s own
journal named the asymmetry it left standing:

> A chat turn whose own preparation (prompt formatting in `ContentView`) straddles a server claim
> still loses, because that happens before `launchTurn`.

That is the same seam as `#84`'s, seen from the other side. `ContentView.startAutoregressiveGeneration`
builds the effective system block, renders the dialect, encodes the prompt and assembles the
request — `ModelConfig.buildEffectiveSystemPrompt`, `ChatPromptFormatter.formatPrompt`,
`tokenizer.encode` — and only then launches. Every one of those is a chance for the server's
`claimEngineSlot` to take the ledger, because the ledger is an `NSLock` box: a server turn claims
from its own task, not from the main actor, and it does not need the chat turn to yield. The chat
turn's preparation is synchronous on the main actor, but that does not exclude a background
claim.

So the owner of the slot changed mid-preparation and the party that paid was, again, the wrong
one: the person who typed a message and pressed send got an error bubble naming the API server,
while the server request ran.

Routing: `next_slice` = `reserved_inapp_launch` **0.663 @ 0.596**, ahead of `vision_device_encode`
0.303, `image_cost_ceiling` 0.028, `model_id` 0.004 and the three remaining prefill terms under
0.002. `slice_should_be_measured` = **0.052**, decisively against another stage table: every perf
candidate left in this journal is under ~1.2% of a prefill position against a 5-30% spread
between runs on this machine, so nothing there can be resolved. The second answer is what carried
the decision again, as it did for `#84`.

### What was added

Three lines of behaviour, one guard, and the tests that pin them.

**The chat path reserves before it prepares** (`ContentView.startAutoregressiveGeneration`, after
the empty-prompt guard and before the system block is built):

```swift
let turnSlotClaim = CompletionEngine.shared.reserveTurnSlot()
defer { turnSlotClaim?.release() }
```

with the existing launch site becoming `launchTurn(request, claim: turnSlotClaim)`.

Two properties make this three lines rather than a rework:

- **A nil reservation is not a new error path.** `reserveTurnSlot()` returns nil when a turn is
  running or another caller holds the slot, and the launch below then reports that exactly the
  way it always has — `launchTurn` with no claim is the pre-`#84` call, and it refuses with the
  same two messages. So a chat send that cannot reserve behaves as it does today; a chat send
  that CAN reserve now closes the window. No message text moved, and the queued-prompt path
  (`dequeueAndRunNextPromptIfNeeded`, which removes the prompt from the queue before sending)
  gains no new way to lose an entry.
- **`defer` covers every early return**, and `release()` is idempotent, so the claim is given
  back whether the turn launches, is refused for a missing model, is refused for an image on a
  graph without a tower, or dies in the tokenizer. `launchTurn`'s own `defer` releases the same
  claim on its way out; neither side needs to know about the other.

**A claim that was given back no longer authorises a launch** (`CompletionEngine.launchTurn`).
`claim ?? turnSlotLedger.tryClaim()` treats a non-nil claim as proof the slot is held, and it
was not: a caller that released its claim and then launched on it would skip the `tryClaim`
guard entirely and run a turn while the ledger read FREE to everyone else — the precise state the
ledger exists to prevent. The guard is one `held.isHeld` check, taken under `stateLock` (lock
order is `stateLock` then the ledger's, which is the order `reserveTurnSlot` and `isActive`
already use, so the new read cannot invert it), and the refusal names the reservation rather
than the model.

This one is defensive, not a live defect: no caller in the app releases before it launches
today. It is included because `#85` makes `launchTurn(claim:)` take claims from a second
caller type, and "the claim is what authorises the turn" is now the invariant two subsystems
depend on rather than one.

### The finding: the order is the whole feature, and the loser now waits instead of failing

Reserving is not about who asked first — it is about who holds the slot when it is spent. With
the chat path reserving:

- a server turn that wants the slot during a chat preparation gets `nil` from
  `claimEngineSlot()`, and `TurnExecutionQueue.waitForSlot` already treats a probe that says
  free followed by a failed claim as a **retry, not a refusal** (`#84`'s explicit decision). The
  request goes back to waiting with its deadline and its heartbeats intact, exactly as if it had
  found the slot busy a moment earlier;
- a chat send during a server turn is still refused, and should be — one turn at a time is the
  contract. What changed is that it is now refused for a reason that is true at the moment it is
  reported, and the refusal no longer depends on losing a race during formatting.

The invariant, stated: **the slot is taken in the same step that decides it is free,
by every caller that then spends real time preparing.** `#84` said that for the server;
`#85` makes it true for both halves of the app, which is what lets the sentence be an invariant
rather than a note about one path.

### What it is not (do not re-propose)

- **Not a wait for the chat path.** Making the in-app send block until the server's turn ends
  would put a suspension in the middle of a main-actor function that then builds a request from
  `@State`, and it would be unbounded — a server turn is a minute. Refusing immediately with a
  true reason is the correct behaviour for an interactive surface.
- **Not a fix for a send that straddles a MODEL LOAD.** `handleSendMessage` can `await`
  `ensureModelLoaded` before it reaches `startAutoregressiveGeneration`, and that suspension is
  upstream of the reserve. Closing it means reserving before the load, which would hold the slot
  across a multi-second model load and refuse the API server for the duration. Left alone
  deliberately.
- **Not a claim on the queued-prompt path.** `dequeueAndRunNextPromptIfNeeded` already requeues
  rather than running a second generation, and it re-checks `isGeneratingText` after every
  suspension. It reaches the same reserve as any other send.
- **Not a change to the 503 contract.** A request that arrives during an in-app turn still
  queues and waits (`#83`); a request whose deadline expires still gets the queue-timeout
  envelope. Nothing here touches the wire.
- **Not measured, and not measurable here.** The window is a code fact (what sits between the
  reserve and the launch), not a timing. The machine's run-to-run spread is 5-30%; a
  "chat-send-during-preparation" histogram would be noise, and the property is structural.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_85` (status 0), then `test-without-building`
with the same `-derivedDataPath` throughout. The build's warning set is **identical to `#84`'s**:
1,773 diagnostics, `diff` clean, so the two touched files introduced none. Bundle freshness
checked with `strings … | grep -c <case>` (3 for the three new cases) before running.

| suite | result |
|---|---|
| `EngineTurnSlotTests` (10 cases: `#84`'s 7 plus three for this slice — the in-app reservation keeping a server turn from claiming mid-preparation, a released claim refused rather than honoured, and an in-app claim redeemed by its launch and given back) | **10 cases, 0 failures** |
| `ServerQueueLogicTests` | 30 cases, 0 failures |
| `OpenAIAdapterLogicTests`, `AnthropicAdapterLogicTests`, `ServerStreamSafetyTests`, `ServerToolArgumentTests`, `ServerHardeningLogicTests`, `TurnToolMaskTests`, `ChatPromptFormatterConversationTests`, `ReasoningRoundTripTests`, `PrefixAsOfDateTests` | 142 cases, 0 failures |
| `ServerControllerSmokeTests` (app-hosted, real server) | 17 cases, 0 failures |

The three new cases are the ones that can be pinned without a UI driver, and each is about a
fact the ledger is supposed to guarantee rather than about the chat view: that a chat claim makes
the server's claim fail (so the queue re-waits), that the failed attempt does not take the slot,
and that a spent claim is refused instead of authorising a turn.

The live arm is `#84`'s own `DeepSeekV41PublishedTurnTests.testAnAPIRequestWithAnImageAnswersOverTheLocalServer`
(opt-in, `/tmp/dynamoe_dsv41_real_turn`), re-run because this slice changed the launch call and
added a guard on the launch path. It passes unchanged (117.5 s, `/tmp/test85_live.log`): 96,085
tensors over 48 segments, server on 127.0.0.1:8791, two requests `200` — the plain prompt in
33.0 s (queue 77 ms) and the same prompt with one attachment in 81.3 s (queue 74 ms), 190 and 375
prompt tokens, a span of 184 positions over 1,521 patch tokens from 0.32 s of tower, and the
model describing the picture. Same numbers to the tenth of a second as `#84`'s run, which is what
a four-line change to who holds a lock should look like.

## DSV4.1 #86: THE ANTHROPIC FRONT DOOR CAN NOW ANSWER WHAT A REQUEST WILL COST, AND ITS NUMBER IS THE TURN'S OWN

### The question

`#77` to `#82` spent six slices making the local server a front door a real client could walk
through: a picture on both protocols, both transports, the models list advertising the capability
and then its cost. What that left is not a bug but a hole in the protocol surface. The Anthropic
API defines two POST routes, and this server had one of them.

`POST /v1/messages/count_tokens` is the pre-flight. A client that manages its own context — Claude
Code is the one that matters — counts a conversation before sending it, and without the route
that pre-flight is a 404. So the branch's own theme, "complete full DeepSeek support" as a drop-in
local server, had exactly one cheap unbuilt piece left, and the price of it is a request with no
generation in it.

Routing agreed decisively: `next_slice` = `anthropic_count_tokens` **0.995 @ 0.994**, with
`device_mhc_stages` 0.004, `model_id` 0.001, `openai_responses` 0.0003, `docs_sweep` 0.0003,
`perf_leftovers` 0.0001 and `image_cost_ceiling` 0.0001. The scope question —
`count_tokens_scope` "the route alone" against "both protocols" and "route plus docs only" —
answered **0.947 @ 0.921** for the route alone.

One correction matters more than the routing. The first request put `vision_device_encode` on top
of the state at 0.94, because that was the largest unbuilt thing this journal used to name — and it
was **stale**: `#43` built the whole device encode (0.32 s smallest image, 6.90 s largest, 8.81x to
6.02x), `#44` and `#45` then measured its attention dead as a SIMT *and* a matrix-unit question,
and `#46` found the encode was 2% of a picture's real cost. The candidate list is only as good as
the journal it is read from, and the journal says the tower is done.

### What was added

One route, one response type, one parameter.

- **`POST /v1/messages/count_tokens`** (`ServerController.buildRouter`), registered beside
  `/v1/messages`. It authorizes, decodes the body as `AnthropicMessagesRequest`, prints the same
  request summary into the same log row, and answers `{"input_tokens": N}`.
- **`AnthropicTokenCount`** (`AnthropicAdapter.swift`) is the whole body. The protocol answers one
  field, and a type that could carry more would invite one.
- **`buildTurnPlan(_:maxTokensRequired:)`** on the Anthropic surface: the messages route requires
  `max_tokens` and the `count_tokens` body has no such field by protocol. The count path passes
  `false` and the planner takes the smallest legal budget instead of refusing.

The reason the route is nine lines of behaviour rather than an estimator is the whole point of the
slice: **it calls the messages route's own planner.** `buildTurnPlan` is where a request becomes a
prompt — the system block rendered with the session's own date, the recorded turn spliced back in
when a follow-up reproduces the history, the tokenizer run once, and an attachment's placeholder
expanded into the span the resize plan predicts. A second estimator here, even a careful one, would
answer a number the turn it precedes never matches, and that mismatch is the only way this endpoint
can be wrong.

The reply budget is safe to substitute because it never moves the prompt: `maxTokens` reaches
`TurnSampling` and nothing else in the planner.

### The finding: the number is a property of the plan, not of the route

Measured on the published checkpoint, one body through both routes (`testTheTokenCountRoutePredictsWhatItsTurnCosts`,
33.2 s, `/tmp/test86_live.log`):

```
[server.request] POST /v1/messages/count_tokens → 200 56ms  model=1bae0e10…  in=166
[server.request] POST /v1/messages/count_tokens → 200 52ms  model=1bae0e10…  in=166
🔎 [dsv41 real] count_tokens: 166 predicted, 166 reported, 23 generated
```

166 predicted, 166 reported, and two identical pre-flights agreeing in between. Three separate
claims are in that line and each one is a thing that could have been wrong:

- **the count is the turn's own `usage.input_tokens`**, not an estimate near it. Both routes render
  the same prompt because they run the same function, so the equality is structural;
- **the count is stable**, so the session date the system block carries is minted once per session
  and not per request (`PrefixCacheManager.asOfDate`). A per-request date would have moved it;
- **the body it accepts is the protocol's**, i.e. without `max_tokens`, which is the one field the
  messages route insists on and this one must not.

A pre-flight is also cheap in the way a pre-flight has to be: **56 ms and 52 ms**, against 33 s for
the turn it predicted. It runs no generation and claims no engine slot, so it answers while a turn
generates — a client compacting mid-turn is not blocked by the turn it is measuring.

### What it is not (do not re-propose)

- **Not a second estimator.** Any implementation that counts without the planner will drift from
  the turn, and drift is the entire failure mode. The shared call is the feature.
- **Not an OpenAI route.** The OpenAI protocol has no pre-flight; a client there sends and reads
  `usage`, or reads the models list `#82` taught to advertise a picture's cost range. Adding a
  `/v1/chat/completions/count` would be an invention, not a compatibility fix.
- **Not a change to `/v1/messages`.** `maxTokensRequired` defaults to `true`, so the generation
  route's refusal of a body without `max_tokens` is unchanged and still pinned
  (`testMessagesWithoutMaxTokensReturns400`).
- **Not a `/health`-style silent route.** `/health` writes no row because polling would flood the
  store; a pre-flight is a client request that happens once per turn, and its row is what lets a
  developer read the predicted count beside the turn that followed it. It inherits the Anthropic
  row key automatically, because `ServerRequestLog.proto(forPath:)` matches `/v1/messages*`.
- **Not a token budget change.** The prompt cap, the KV `turnFit` check and the 503-when-busy
  contract all live in the turn path, which this route does not enter.

### Verification

`xcodebuild build-for-testing` into `/tmp/dd_dsv41_86` (status 0, error grep with a leading space so
source echo lines do not read as errors), then `test-without-building` with the same
`-derivedDataPath` throughout. Bundle freshness checked with `strings … | grep -c <case>` (4 for the
four new cases) before running anything.

The build's warning set grew by **25 instances of categories that already fill these files**
(1,773 → 1,798 diagnostics, `diff` of the `file:line: warning` sets taken against `#85`'s build).
Every one is the project-wide `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` pattern that the two
existing route handlers already carry dozens of times — `beginRequest()`/`endRequest()` from a
nonisolated context, `loadedModelInfo` from a nonisolated context, and the main-actor-isolated
`Encodable` conformance of a wire type (`AnthropicTokenCount` joins `AnthropicMessageResponse`,
`OpenAIChatResponse`, `OpenAIErrorEnvelope` in that list). No new kind of diagnostic, so the shape
of the codebase is unchanged.

| suite | result |
|---|---|
| `AnthropicAdapterLogicTests` (25 cases: `#86`'s two are a `count_tokens` body decoding with no `max_tokens`, and the response encoding to exactly one field) | **25 cases, 0 failures** |
| `ServerControllerSmokeTests` (app-hosted, real server on 8787; `#86`'s three are the 503 model gate with a body that carries no `max_tokens`, the Anthropic-shaped 400 on a malformed body, and the credential both ways — 401 anonymous, past-the-gate with `x-api-key`) | **20 cases, 0 failures** (17 before) |
| `ServerQueueLogicTests`, `OpenAIAdapterLogicTests`, `ServerStreamSafetyTests`, `ServerToolArgumentTests`, `ServerHardeningLogicTests`, `TurnToolMaskTests`, `ChatPromptFormatterConversationTests`, `ReasoningRoundTripTests`, `PrefixAsOfDateTests`, `EngineTurnSlotTests` | 129 cases, 0 failures |
| `DeepSeekV41PublishedTurnTests.testTheTokenCountRoutePredictsWhatItsTurnCosts` (opt-in, `/tmp/dynamoe_dsv41_real_turn`, private port 8793) | **passed, 33.2 s**, 166/166, rows carry the count |

The three smoke cases are chosen so the route's state machine is pinned without a model: a planner
that still demanded `max_tokens` would answer 400 where the first case requires 503, and a
credential path that did not accept `x-api-key` would answer 401 where the third requires 503.
