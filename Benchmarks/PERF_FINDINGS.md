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
