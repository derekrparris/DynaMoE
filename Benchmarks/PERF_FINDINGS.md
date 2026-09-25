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
