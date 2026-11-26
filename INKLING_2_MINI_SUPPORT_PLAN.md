# Inkling 2 Mini: Full Native Inference Support Plan

**Status:** phases 1-6 landed and validated on the published checkpoint; the device graph matches the host graph to 1e-6 on every one of the schedule's 48 visits, a full agent turn runs end to end on a 16 GB M1 Pro, and Phase 7's INT8 gate is measured on BOTH paths (the same greedy token at every tested length; the plan's literal elementwise bound is not met by an 8-bit cache and the record says so; the device arm's elementwise difference tracks the host arm's to within 5%)  
**Checkpoint under analysis:** `thinkingmachines/Inkling-2-Mini`  
**Pinned checkpoint revision:** `c9d028302cd0f035fdf8e7e91f089a4ce94d2195` (the only revision the repository publishes; an earlier `ed40e7da...` pin resolved to nothing and silently skipped every checkpoint case)  
**Local snapshot:** `~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/c9d028302cd0f035fdf8e7e91f089a4ce94d2195`  
**Reference Transformers source revision:** `046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`  
**Plan date:** 2026-11-21

## How to use this document

Implement the phases in order and do not mark a phase complete until every exit gate passes. Record measured results and deviations in this file. Keep the fast, model-independent tests separate from the opt-in real-checkpoint tests. Preserve native numerics and fail closed on unsupported checkpoint changes; plausible text is not evidence that a graph is correct.

“Full support” in this plan means native execution of this exact fixed-loop checkpoint with:

- text and still-image input;
- its original mixed NVFP4/MXFP8 experts, not only a converted replacement;
- prefill, decode, cancellation, stop handling, prefix reuse, and long-context memory planning;
- chat, reasoning effort, tool calling and agent continuations;
- the in-app composer plus OpenAI and Anthropic routes, streaming and non-streaming;
- optimized Metal kernels and packed-expert residency/streaming policies;
- optional bundled DSpark support after the base turn is proven.

The variable-loop research checkpoint, audio, video, training, and alternate GGUF/MLX requantizations are separate follow-up projects. Do not silently substitute them for this checkpoint.

## 1. Architectural decision

Build a dedicated **`DynaMoE/DynaMoE/Inkling2/` backend**, integrated before the generic `CompletionEngine` transformer path, following the isolated-runtime precedent of `DeepSeekV41/` rather than adding family conditionals throughout the generic decode, prefill, and JetSpec graphs.

The generic path cannot represent this graph safely. It assumes conventional GQA projections and rotary attention, one conventional residual topology, ordinary per-layer KV slots, generic router behavior, and different tensor layouts. Inkling needs a 48-visit schedule, four recurrent states per visit, relative-position attention, block attention residuals, cross-pass message gates, and native mixed quantization.

Production execution must retain packed experts and offer two explicit numerical profiles: an HF-reference W4A16/W8A16 profile for baseline parity, and the checkpoint’s trained W4A4/W8A8 activation-quantized profile for optimized native execution after separate validation. A whole-model BF16 expert conversion would add tens of gigabytes and discard the checkpoint’s bandwidth advantage. Fusing weight decode into resident/streamed projection kernels preserves the packed representation in either profile.

DSpark is a later opt-in optimization. Base correctness, images, tools, streaming, prefix reuse, and performance instrumentation come before speculative decoding.

Structured architecture review agreed: dedicated backend 0.9976 at 0.9951 confidence, native mixed precision 0.9973 at 0.9947, and “do not prioritize DSpark before the base is correct” 0.9994.

## 2. Verified checkpoint model

### 2.1 Storage and geometry

The main file is `model-00001-of-00001.safetensors` with 1,244 tensors and 19,526,019,168 bytes. `mtp.safetensors` has another 54 BF16 tensors and 635,801,416 bytes. Load the latter only when DSpark is enabled.

| Item | Verified value |
|---|---|
| Architecture | `Inkling2LoopForConditionalGeneration`, `model_type = inkling2_loop` |
| Width | 2,304 |
| Physical decoder layers | 24 |
| Schedule | 0...5 once, 6...17 three times, 18...23 once |
| Executed visits per token | 48 |
| Q / KV heads | 24 / 8, head dimension 128, three Q heads per KV head |
| Local attention | 20 physical layers, 1,024-token window |
| Full attention | physical layers 5, 11, 17, and 23; executed visits 5, 11, 17, 23, 29, 35, 41, and 47 |
| Experts | 128 per physical layer, top 8, no shared expert |
| Expert intermediate width | 1,152 |
| Router | FP32 sigmoid scores, bias selects only, normalize selected scores, scale by 8 |
| Vocabulary | 201,024 padded, 200,058 unpadded output logits |
| Embedding / unembedding | separate BF16 tensors; unembedding produces FP32 logits |
| Final logits | `30 * tanh(logits / 30)` |
| EOS / padding / image | 200006 / 199999 / 200054 |
| Architectural context | 1,048,576 positions, subject to DynaMoE’s actual memory budget |
| Vision | 23-layer, 1,024-wide, 16-head ViT plus 4,096-wide merger and 2,304-wide output projection |
| DSpark | three dense BF16 layers, eight-position trained block, rank-256 RNN Markov head |

Main-file storage: 994 BF16 tensors, 154 FP32 tensors, 48 F8_E4M3 tensors, and 48 U8 tensors. Do not infer logical parameter type from U8 alone: in layers 0...19 the U8 weights are packed E2M1 pairs and their scales are E4M3; in layers 20...23 the U8 tensors are E8M0 scales and the weights are E4M3.

### 2.2 Expert layouts and trained arithmetic

For every expert `e`, `w13` interleaves gate and up along the original output dimension: even raw rows are gate and odd raw rows are up. Read the two logical matrices as even/odd rows, not as two contiguous halves. `w2` is the down projection.

| Layers | `w13_weight` | `w13_weight_scale` | `w2_weight` | `w2_weight_scale` | Other scale data |
|---|---|---|---|---|---|
| 0...19, NVFP4 | `[128,2304,1152]` U8, two E2M1 per byte | `[128,2304,144]` E4M3, one per 16 inputs | `[128,2304,576]` U8 | `[128,2304,72]` E4M3 | per-expert FP32 amax and one denominator |
| 20...23, MXFP8 | `[128,2304,2304]` E4M3 | `[128,2304,72]` U8 E8M0, one per 32 inputs | `[128,2304,1152]` E4M3 | `[128,2304,36]` U8 E8M0 | none |

Validate these layouts from shapes and quant metadata, not from parameter group labels alone. The checkpoint’s MXFP8 metadata permits a `[32,32]` declaration while stored scales are one per output row per 32 inputs; the actual tensors are authoritative.

Use two named numerical profiles rather than treating different backends’ answers as if they were the same rounding contract:

- **HF-reference W4A16/W8A16:** dequantize each logical weight and round it to BF16 before an ordinary BF16 activation projection. This matches Transformers’ dense-execution fallback and is the baseline correctness profile. Fuse logical-weight decode into projection kernels rather than materializing a BF16 model copy.
- **Trained NVFP4 W4A4:** quantize each BF16 activation row to E2M1 and E4M3 scales. Row amax gives an outer scale `amax / 1536`. For each 16-value block, try scale rules based on both 6.0 and 4.0, round to E4M3, reconstruct, and retain the candidate with lower summed absolute reconstruction error. Ties choose the 6.0 candidate. E2M1 rounding is nearest-even with saturation and the lower input column occupies the low nibble. Weight output includes `blockScale * (expertAmax / 2688)`.
- **Trained MXFP8 W8A8:** each 32-value activation block uses `scale = round_up_power_of_two(blockAmax / 448)` with the reference’s subnormal, saturation, and NaN behavior. Values are clamped to +/-448 and rounded to E4M3. Weight scales are E8M0.

The trained-activation profile follows vLLM’s native algorithms, not Transformers’ dense fallback. Gate it behind independent goldens and an output-quality/performance comparison; never silently switch between these profiles during a session.

After gate/up projection, compute in FP32:

```text
activation = tanh(gate / 4) * 4 * sigmoid(gate) * tanh(up / 25) * 25
moe = sum over selected experts(down(activation) * normalizedSelectedScore * 8)
```

Accumulate expert outputs in FP32. The routing bias participates in `topk(scores + bias)`, not in the weights used for the mixture.

### 2.3 Layer graph and exact residual order

Each layer uses four causal short-convolution states (K, V, post-attention, post-MLP). Static convolution is `(depthwiseConv4(x) + x) / sqrt(5)`. On full-attention layers, project the same normalized, independently gated residual supplied to `wk_dv`/`wv_dv`, before the K/V linear projection, into four dynamic taps per KV head. Add the static taps, apply the combined taps to causal windows of the post-linear K/V states, and use `(dynamicConv4(x) + x) / 3`.

Attention has no conventional RoPE:

1. Project Q, K, V, relative states R, output gate, and learned sink.
2. Apply static or dynamic K/V convolutions.
3. Scale Q and R by learned gains, then L2-clip Q/K at 22.627416997969522 and R at 8.0, all in FP32 before restoring dtype.
4. RMS-normalize V without a learned gamma.
5. On full-attention layers, multiply Q and R by `tau = 1 + 0.1 * log(max((position + 1) / 8192, 1))`.
6. Form `R @ relativeBiasBank`, select by backward distance, and zero bias outside `0 <= distance < 1024`.
7. Append one learned query-dependent sink logit with a zero K/V row. Softmax includes it but output uses only real values.
8. Multiply attention output by `sigmoid(gateProjection)` and apply `wo`.

Every six physical layers is a block boundary at 0, 6, 12, and 18. Block attention residuals RMS-normalize the stacked states, apply learned FP32 queries, and use FP32 softmax with `1 / sqrt(2304)` scaling. Attention and MLP inputs read those residuals. At a boundary, append the layer’s pre-attention input to the block stack and start the next block sum from the post-attention convolution; elsewhere add the previous block residual. Final hidden selection performs the same residual read over all blocks plus the current partial, then applies the final weighted RMSNorm and FP32 unembedding/softcap.

For loop pass one, run the core without messages and retain its last completed core block plus final hidden state. For passes two and three:

- restart from the exact post-layer-5 hidden state and pre-loop block stack;
- retrieve four messages per core layer from the previous pass’s last completed block plus final hidden state using packed FP32 learned queries;
- independently mix Q, K, V, and MLP inputs through rank-64 gates;
- each gate normalizes current and message, concatenates them, projects 4,608 -> 64 -> two 2,304 FP32 vectors, and computes:

```text
forget = tanh(forgetProjection)
messageGain = tanh(messageProjection)
output = (1 - forget) * current + messageGain * message
```

For core layer `i` in 0...11, rows `2i` and `2i+1` of `middle_cycle_cross_pass_queries` are its Q and MLP queries; the corresponding rows of `middle_cycle_cross_pass_kv_queries` are its K and V queries. Merge them into 12 x four logical queries and pin that unpacking with a golden.

### 2.4 Vision semantics

This is not DeepSeek’s resize-and-cap image pipeline:

- preserve source dimensions and EXIF orientation;
- grayscale -> RGB, and composite RGBA over an 8 x 8 alternating 204/153 checkerboard using the reference integer alpha blend;
- pad to multiples of 40 x 40 with raw -1 values before the `1 / 255`, mean 0.5, std 0.5 normalization;
- language image-token count is `ceil(height / 40) * ceil(width / 40)`;
- each language cell becomes four native 20 x 20 subpatches, so the ViT processes four times as many tokens;
- add a learned class token, bilinearly interpolate the 32 x 32 position table in FP32 with `align_corners = false`, and use interleaved-pair 2D RoPE;
- run pre-norm LayerNorm blocks with learned residual gains and exact GELU MLPs, discard the class token, merge each 2 x 2 neighborhood, then run `proj_in -> GELU -> proj_out`;
- normalize ordinary token embeddings with text `embed_norm`, then replace exactly token id 200054 positions with output rows, validating one-to-one counts and preserving image/text order. Image rows do not pass through `embed_norm`.

Examples:

| Image | Language positions | ViT subpatch tokens |
|---|---:|---:|
| 40 x 40 | 1 | 4 |
| 512 x 512 | 169 | 676 |
| 1,024 x 1,024 | 676 | 2,704 |
| 2,048 x 2,048 | 2,704 | 10,816 |
| 4,096 x 4,096 | 10,609 | 42,436 |

There is no 1,024-position image cap. Price and provision for the expanded image span before allocating KV or running the tower. Large ViT attention must be tiled/streamed; never materialize an unbounded attention-score matrix.

Still images have grid `[1, rows, columns]`. Refuse audio and temporal/video grids explicitly rather than executing the wrong path.

### 2.5 DSpark summary

The optional sidecar contains three dense layers, their own full/local KV caches, `fuse_weight`, target captures at executed visits 5, 23, and 41, final normalized state 48, a mask embedding, and an RNN-hidden Markov head. It shares target token embedding and unembedding. Fold `hidden_out` into the vocabulary head once with the reference’s FP64 tiling. Draft blocks are bidirectional over the trained eight positions; proposal tokens do not enter persistent draft KV. Target verification must rollback KV plus all four convolution states per rejected position.

Do not reuse DeepSeek’s acceptance assumptions. Measure acceptance, draft cost, verify cost, and emitted tokens per unit time on this model, in cold and warm expert regimes.

## 3. Cache and memory model

### 3.1 Executor state is per executed visit, not per physical layer

Assign every execution visit its own KV and four-convolution state bundle. Passes reuse weights, never K/V or recurrent state. Physical full-attention layers 5, 11, 17, and 23 therefore consume eight growing full-attention caches; the other 40 visits use 1,024-position chronological rings.

Relative bias depends on chronological distance, so a ring cannot be read as an arbitrary newest-first array. Ring reads, partial prefill, delta prefill, and rollback all share the same addressing rule.

Convolution state is FP32. Its fixed size per sequence is:

```text
48 visits * (2304 + 1024 + 1024 + 2304) channels * 3 history values * 4 bytes
= 3,833,856 bytes (3.66 MiB)
```

Cross-pass messages are same-position, so prefill can process one bounded chunk through the entire 48-visit schedule before moving to the next chunk. This both preserves causality and avoids storing loop messages for the entire prompt. Persist only the previous pass’s completed core block and final hidden vectors within the active chunk, then release them after the chunk. Executor history and all visit states must advance by the same committed position count.

### 3.2 KV bytes

At BF16/FP16, K+V for one attention visit is `8 * 128 * 2 * 2 = 4,096 bytes/token`. Global growth is eight visits; local rings are fixed.

| Stored context | 48 full-cache naïve FP16 | Correct hybrid FP16 | Correct hybrid FP8, excluding scale overhead |
|---:|---:|---:|---:|
| 32,768 | 6.442 GB | 1.242 GB | 0.621 GB |
| 131,072 | 25.770 GB | 4.463 GB | 2.231 GB |
| 262,144 | 51.540 GB | 8.758 GB | 4.379 GB |
| 1,048,576 | 206.158 GB | 34.528 GB | 17.264 GB |

The last column uses DynaMoE’s setting named FP8, whose storage is symmetric INT8 plus FP16 scales, not an E4M3 KV format. Scale overhead is separate and KV precision is orthogonal to the two expert-activation profiles.

These figures exclude allocator padding, scales, convolution state, prefill workspace, vision, and runtime overhead. They demonstrate why allocating 48 full-length caches is unacceptable. The model’s 1M declaration is an architectural ceiling, not a promise for a 32 GB Mac; `turnFit` remains the authority.

### 3.3 Weights and expert traffic

The two main storage classes total approximately 15.67 GB of routed experts; everything else in the main checkpoint, including vision, embeddings, unembedding, attention, gates and norms, is approximately 3.86 GB. Do not duplicate a BF16 copy of the experts.

Per physical layer:

- complete NVFP4 expert set: 573,308,928 bytes; top-8 share: 35,831,808 bytes;
- complete MXFP8 expert set: 1,051,066,368 bytes; top-8 share: 65,691,648 bytes.

The fixed schedule visits 44 NVFP4 layers and four MXFP8 layers. With no routing locality, their selected packed payloads total about 1.839 GB per generated token, before dense weights and state traffic. This is a traffic model, not a promised compulsory read count. Residency and cross-token/prefill route locality determine actual disk traffic.

Memory tiers to measure rather than assume:

- **32-48 GB:** packed expert arena with a bounded budget, pinned or efficiently mapped dense text weights, BF16 KV at modest contexts, DSpark unloaded by default.
- **64 GB:** larger arena and context-dependent DynaMoE FP8 (INT8) KV option.
- **128 GB:** usually retain all packed experts and use full-model residency; still bound huge KV and prefill workspaces.

Keep the arena all-or-nothing if slot wiring is later enabled; DynaMoE’s DeepSeek measurements show partial wiring can be worse than none. Any such policy must come from Inkling’s own memory plan and refusal path.

## 4. DynaMoE gap map and target architecture

### 4.1 Existing hazards to intercept

- `ModelConfig.resolveArchitectureType` currently falls through to Standard MoE (`DynaMoE/DynaMoE/ModelConfig.swift:864-894`). Detect `inkling2_loop` first. Do not use generic nested defaults to synthesize its graph.
- Generic tensor classification in `core/src/lib.rs:338-433` and Swift binding in `DynaMoE/DynaMoE/InferenceEngine.swift:661-885` do not understand fused expert banks, message queries, or cross-pass controllers. Use an exhaustive dedicated catalog.
- Generic decode (`CompletionEngine.swift:3242`), prefill (`:6378`), and JetSpec (`:9261`) apply incompatible GQA/RoPE/residual/router rules. Keep Inkling out of all three.
- Generic KV geometry (`EngineRuntime.swift:756-786`) cannot price per-visit KV plus four recurrent states. Use an Inkling-owned geometry and executor-authoritative resume, as in `DeepSeekV41MetalForward.swift:3053-3094`.
- Images are currently accepted only by the DeepSeek branch (`CompletionEngine.swift:575-586`), while server planning and vision types are DeepSeek-concrete (`ServerController.swift:1837-1863`, `ModelConfig.swift:1278-1332`). Introduce a narrow family-neutral vision capability seam.
- Prompt formatting and tool selection otherwise fall back to ChatML/Qwen XML (`ChatPromptFormatter.swift:368-516`, `StreamingToolParser.swift:13-27`). Implement one Inkling dialect used by UI, agents, and server rather than adding an isolated parser.
- JetSpec eligibility in `EngineRuntime.swift:33-54` is not valid for this family. Keep it off. DSpark gets a separate implementation and rollback proof.

### 4.2 New files

Use the synchronized-group layout so Xcode adds these automatically:

| File | Responsibility |
|---|---|
| `Inkling2/Inkling2Config.swift` | Strict nested config, family identity, layer/visit schedule, quant and MTP validation |
| `Inkling2/Inkling2Weights.swift` | Exhaustive role catalog, shapes, dtypes, required companions, packed query unpacking |
| `Inkling2/Inkling2Conformance.swift` | Named numerical gates and their reference provenance |
| `Inkling2/Inkling2Forward.swift` | Small host reference graph shared by tests and device comparisons |
| `Inkling2/Inkling2Engine.swift` | Runtime ownership, required pipelines, workspaces, shard buffers, load diagnostics |
| `Inkling2/Inkling2MetalForward.swift` | One numerical graph used by chunked prefill and single-position decode |
| `Inkling2/Inkling2KVCache.swift` | Visit states, chronological rings, commit/rollback, exact resume arithmetic |
| `Inkling2/Inkling2Turn.swift` | Cancellable turn loop with injected sampling, constrain, freeze, decoding and progress closures |
| `Inkling2/Inkling2MemoryPlan.swift` | Weights, arena, KV, convolution, loop-message, vision and workspace budgets |
| `Inkling2/Inkling2ExpertStreamer.swift` | Packed ranges, residency, pread fills and measured optional policy |
| `Inkling2/Inkling2Prompt.swift` | Role/history/effort/EOS rendering matching the checkpoint processor |
| `Inkling2/Inkling2ToolDialect.swift` | Tool schemas, JSON calls, streaming freeze, canonical replay, result rendering |
| `Inkling2/Inkling2ImagePixels.swift` | EXIF, RGB/grayscale, alpha checkerboard, 40-pixel tiling, normalization |
| `Inkling2/Inkling2Vision.swift` | Span pricing, tower/reference forward, span binding and digest identity |
| `Inkling2/Inkling2VisionDevice.swift` | Batched/tiled Metal tower, added only after host parity |
| `Inkling2/Inkling2DSpark.swift` | Optional draft runtime after the base runtime is stable |

Keep integration adapters thin. Reuse `CompletionTurnRequest`, `TurnImage`, `TurnImageSpan`, the turn-slot ledger, sampling, request logging, the prefix-pin proposal, and the closure-driven turn-loop pattern. Do not make DeepSeek vision, KV internals, or speculative graph a prerequisite for Inkling. Extract a model-neutral turn/vision protocol only where its signature is already identical; avoid another monolithic rewrite inside `CompletionEngine`.

### 4.3 Files to modify narrowly

- `ModelConfig.swift`: new family case/predicate, early resolve, file-only identity, capability probes, neutral reasoning/vision facts.
- `LocalModelManager.swift`: display architecture and quantization labels.
- `CompletionEngine.swift`: Inkling interception before generic setup, cached runtime keyed by checkpoint identity/context, image-span cache, teardown on configure.
- `ContentView.swift`: native conversation/attachment construction and load status.
- `ChatPromptFormatter.swift`: Inkling single/history render entry points; no ChatML fallback.
- `AgentHarness.swift`: family tool schema instructions and byte-identical tool-result replay.
- `ServerController.swift`: expanded-image pricing, family-neutral markers/capabilities, dialect stops.
- `OpenAIAdapter.swift` / `AnthropicAdapter.swift`: effort vocabulary and image handling through the same planner; preserve existing field refusal/retention behavior.
- `CompletionTypes.swift`: only if needed to preserve alpha or preprocessing policy before `TurnImage` is created. Do not silently drop alpha on this family.
- `core/src/lib.rs`: explicit sidecar loading if the existing raw loader cannot address `mtp.safetensors`, plus namespace fixture tests. No repack format is required.

## 5. Implementation phases

### Phase 0: Evidence lock, independent goldens, and measurement harness

**Work**

1. Add `Tools/inkling2_goldens.py` using the pinned Transformers implementation and processor, plus scalar Python mirrors of vLLM’s `nvfp4.py` and `mxfp8.py` algorithms. Record every source revision in generated JSON.
2. Generate small, non-sensitive golden fixtures for:
   - config and all 1,244 main tensor roles plus 54 sidecar roles;
   - packed/unpacked weights, NVFP4 scale choice including ties/subnormals/saturation, and MXFP8 rounding;
   - sigmoid routing and tie behavior;
   - static/dynamic convolution at chunk boundaries;
   - Q/K/R clipping, value norm, log scale, relative distances, sink and output gate;
   - block residual boundaries 0/6/12/18 and all three loop passes;
   - one-layer, one-block, full-model logits and short greedy continuations;
   - prompt/history/reasoning/tool schemas/calls/results and special-token ids;
   - image EXIF, grayscale, alpha checkerboard, padding, interpolation, tower rows and span counts.
3. Add `Inkling2SyntheticCheckpoint` with unusual-but-valid small dimensions: local/full layers, three core passes, NVFP4 and MXFP8 layers, fused experts, all query packs, nonuniform routes, and enough positions to cross a local ring and chunk boundary.
4. Establish the app-hosted test invocation and baseline on this Mac with the exact class/method filters. Add a perf journal section using cold versus warm regimes, runtime first-pass separation, rotated arms, byte/miss counters and device windows.

**Exit gates**

- Goldens are generated from independent code, not from DynaMoE’s own future output.
- Every published config fact has a fixture, and malformed/unknown configs are rejected.
- Test bundles are freshness-checked before results are trusted.
- Baseline commands and checkpoint hashes are documented.

### Phase 1: Identity, catalog, preflight, and model-independent primitives

**Work**

1. Add family identity and early interception. Refuse unsupported variants by name before generic model setup.
2. Implement strict config, exhaustive tensor roles, required scales, dtypes, shapes, duplicate/unknown checks, message-query unpacking, and sidecar isolation.
3. Implement `Inkling2Prompt`’s basic text path before any app-order turn: system/user/assistant history, default effort 0.9, exact role/content/end markers, generation-time behavior, and EOS 200006. Tool declarations, richer effort values, streaming tool parsing, and canonical tool replay stay in Phase 4.
4. Implement host primitive tests: E2M1/E4M3/E8M0 conversions, both activation quantizers, packed expert projection, softcapped SwiGLU, FP32 router, convolution, residual reads, gates, clipping and final softcap.
5. Add main-file completeness and optional-sidecar checks. Detect actual tower files before advertising vision. Map quantized labels accurately.
6. Implement the memory formulas and a preflight that rejects prompts the chosen runtime cannot provision, while leaving room for the reply budget.

**Exit gates**

- `cd core && cargo test` passes.
- All catalog/config/memory/preflight/prompt/quant primitive tests pass without model or GPU dependencies.
- A missing companion, truncated shard, unknown tensor, unsupported quant declaration, or half tower is refused by name.
- No generic `runTokenForward`, generic prefill, or JetSpec path can execute this family.

**Recorded 2026-11-21: DONE (partly deferred to the phases that own it).**

Landed under `DynaMoE/DynaMoE/Inkling2/`: `Inkling2Config` (strict nested decode, unknown-key refusal, visit plan, vision/MTP specs), `Inkling2Weights` (exhaustive exact-match catalog over all 1,244 main + 54 sidecar tensors, coalescing the Rust summary's per-expert split of fused banks, expert-layout inference from stored strides, query unpacking), `Inkling2MemoryPlan` (+ `Inkling2Preflight`), `Inkling2Prompt` (text path), `Inkling2Primitives` (element formats, both activation quantizers, layer rules) and `Inkling2Conformance` (gate ledger). Identity is in `ModelConfig` (`isInkling2Model` checked before DeepSeek) and `LocalModelManager`; `CompletionEngine.launchTurn` refuses the family by name before generic setup, so no generic decode/prefill/JetSpec path can run it.

Evidence: `Inkling2LogicTests` (72 cases) green, including 4 opt-in cases over the real snapshot: every tensor classified, no duplicates, the full validate() clean, NVFP4 on layers 0...19 / MXFP8 on 20...23 by stored stride, the packed queries (23 one-or-three-row attention reads, 24 MLP reads, two 24-row cross-pass tensors), dense 3.86 GB / experts 15.67 GB / tower 0.637 GB from the real headers. The identity/discovery and packed-expert suites are the rest. `cargo test` 22 green. Numeric goldens come from an independent numpy/jinja2 mirror, not from the Swift.

Deviations: `mtp.safetensors` is not in the main index, so the summary never sees it; the sidecar is read by its own header in the test and will be opened by name in Phase 8. `temporal_patch_size` is 2 in the published config for still images, so it is not refused (only < 1 is). The published 40 GB-class vision numbers are validated as geometry, not executed, until Phase 5. Reasoning effort accepts the template's named map and numeric 0.0...0.99; the app's two-level UI mapping is Phase 4.

### Phase 2: Correct native text vertical slice

**Work**

1. Implement the host forward and Metal pipelines with one shared graph: embedding norm, 48 visits, relative-bias attention, convolutions, residual blocks, loop messages, final residual/norm/unembed/softcap.
2. Start with packed fused-decode GEMV in the HF-reference W4A16/W8A16 profile and straightforward attention. Round logical dequantized weights to BF16 as Transformers does. Keep FP32 intermediates where the reference requires them, especially convolutions, gates, routing, mixture accumulation, residual reads, and output logits.
3. Add a nonisolated position-forward protocol and turn runner supporting walk prefill, decode, sampling constraint, EOS 200006, max tokens, stop sequences, tool freeze, cancellation and streamed text.
4. Integrate one runtime in `CompletionEngine` with the existing single-flight claim and the Phase 1 `Inkling2Prompt` text renderer, never the ChatML fallback. Do not enable prefix reuse, DSpark, or images yet.
5. Run published-checkpoint text turns through app order, not only test-only pipelines.

**Exit gates**

- Device matches independent per-stage goldens within FP32/BF16 justified tolerances; selected expert ids match exactly.
- At every tested position, each logit satisfies `abs(actual - reference) <= 5e-4 + 5e-4 * abs(reference)`, or a tighter measured bound.
- Batched-equivalence tests added later use the same elementwise bar. Prompts with adequate top-two margins must produce identical greedy ids as Transformers’ W4A16/W8A16 dense-activation path, while a near tie requires an argmax-margin diagnostic and an explained reduction-order deviation rather than an unconditional pass/fail by one id.
- Several fixed, adequately separated prompts produce matching short greedy continuations, and streamed UTF-8 repair works.
- Cancellation stops at a position boundary, leaves internally consistent counts, and cannot wedge the turn slot.
- Other model families’ pure-logic suites remain green.

**Recorded 2026-11-21: IN PROGRESS. The host graph runs the real checkpoint, the runtime IS wired into `CompletionEngine.launchTurn` (text turns), and the logits are asserted by rank agreement rather than by the plan's elementwise bound, which a BF16 reference cannot support (see the last paragraph).**

Landed (`Inkling2Forward`, `Inkling2KVCache`, `Inkling2SnapshotWeights`, `Inkling2Turn`, commit `3db63c8`): the 48-visit schedule as one host graph, per-visit KV and four convolution states with chronological rings, relative-bias/sink attention with no RoPE, block attention residuals, cross-pass message gates, the packed-expert HF-reference profile, a closure-driven turn loop with a `Inkling2PositionForward` protocol, and an opt-in golden harness (`Tools/inkling2_goldens.py` equivalent: a pinned-Transformers run with forward hooks dumping per-visit and per-stage last-position values).

Cross-checked against the pinned reference on the published checkpoint (`thinkingmachines/Inkling-2-Mini`, revision `ed40e7da`), each to BF16 rounding (0.2-0.5%): every Q/K/V/R/O projection, the Q/K/V norms, the sink logits, the relative-position bias, the output gate, the attention output (correlation 0.99996), the router scores (0.12%), per-slot mixture contributions, and one expert's gate/up/down. That comparison settled two questions the plan could not: raw `w13` rows ARE interleaved (even = gate; the reference de-interleaves at conversion, so its `chunk(2)` applies to the converted tensor -- verified numerically against its loaded bank), and the reference LOADS THIS CHECKPOINT'S EXPERTS AS PLAIN BF16 (its log says "Unknown quantization type... we will skip the quantization"), i.e. it runs the W4A16/W8A16 dense-activation profile while DynaMoE's native path dequantizes NVFP4/MXFP8.

Resolved: the per-visit divergence was in the COMPARISON, not the graph. The host case kept its ring as `count - 48` (always 0 once full, so slot 0 held visit 47's output against visit 0's reference) and compared execution order (pass-major) against a layer-major golden. With both fixed, every visit is within the reference's own noise: against an FP32 reference dump this forward is 3.2e-3 mean / 6.0e-2 max relative per visit, while the REFERENCE's BF16 run differs from its own FP32 run by 1.9e-2 mean / 6.4e-2 max. The plan's `5e-4 + 5e-4*|ref|` bound is not assertable against a BF16 dump: the reference compared with itself violates it on 197k of 200k logits. The test asserts argmax, the top-four set, cosine similarity above 0.9999 (measured 0.99998; the BF16 reference scores 0.99995 against FP32) and a 1.0 magnitude bound, each documented with that calibration. A tighter bar needs an FP32 reference dump, which the pinned implementation can produce (`/tmp/inkling2_host_goldens_fp32.json` in this session's run was one). Fast suites: 106 cases green (config, catalog, memory, prompt, quant primitives, packed experts, identity/discovery, turn loop, the real-checkpoint catalog, the formatter suites and the fixture forward); the two real-checkpoint cases green at `-O` (the stage-by-stage host forward and a wired text turn); `cargo test` 22 green.

**Device graph (recorded 2026-11-22): the attention half, the MLP half and a WHOLE VISIT run on the GPU, each compared stage by stage against the host graph on the published checkpoint.** `Inkling2MetalForward` binds the checkpoint's zero-copy shard buffers, owns a workspace of one-position staging buffers, and dispatches `inkling_*` kernels appended to `ComputeShaders.metal`. The unit of substitution is the VISIT: `Inkling2CPUForward.visitRunner` lets an executor run one visit while the host graph keeps the loop SCHEDULE (block boundaries, passes, the previous pass's stack), because that schedule is the part the reference is subtle about and it must have one copy.

- Attention half (`testTheDeviceAttentionHalfMatchesTheHost`): 1.2e-6 relative end to end; q/k/v norms 2e-7, the relative states and sink exact, the attention scores head 0 9.4e-7.
- MLP half (`testTheDeviceMLPHalfMatchesTheHost`): 4.0e-7 relative against the visit output; the norm 4e-7, the router scores 1.6e-6, the selected experts IDENTICAL, the mixture 2.3e-6.
- Whole visit (`testTheDeviceVisitMatchesTheHostGraph`): 7.4e-7 relative.
- Every visit at the published widths (`testEveryVisitMatchesTheHostAtThePublishedWidths`, default): all 48 visits of position 0 on the real checkpoint, ~1e-6 relative each, at a 1e-3 bar; the final logits ~2e-6. This is the case that caught the full-attention branch, which the one-visit case above never reached because it runs layer 0 -- a local layer.
- Both server routes (`testBothProtocolsAnswerTextTurnsOverTheLocalServer`, opt-in behind `/tmp/dynamoe_inkling2_server_routes`): a text turn over `/v1/chat/completions` (non-streaming) and over `/v1/messages` (streaming) on a private port. The OpenAI body is a `chat.completion` with a terminal `finish_reason` and a usage block (182 prompt / 2 generated); the Anthropic stream opens with `message_start`, keeps the connection warm with named `ping` frames, and ends `content_block_start` / deltas / `content_block_stop` / `message_delta` / `message_stop` with no `error` frame. The log rows are one per route, the streaming one marked `streaming=true`. The generated text is not asserted: the answer is degenerate at a two-token budget and a wording change is not a regression.
- The app's own turn path (`testTheEngineAnswersTheSameOnTheHostAndOnTheDevice`, opt-in behind `/tmp/dynamoe_inkling2_engine_turn`): `CompletionEngine.launchTurn` with the checkpoint loaded as the app loads it, run twice in ONE process -- the host graph forced by the `dynamoe_inkling2_device_off` debug seam, then the device visit runner -- and both arms return the same two tokens. **Host 427.4-488.5 s, device 14.0-14.5 s** across runs for the same turn (29-35x; two tokens, so a direction rather than a rate, and the host arm varies most because it is CPU-bound on a machine doing other work). The runner is installed inside the turn task and cleared when it ends, and the device half is retired with the host forward on `configure`.

Six defects the stage-by-stage comparison found and no end-to-end number would have named: the post-MLP convolution read the ATTENTION's taps; the router staged its ranks with `tid < numExperts` under a 64-wide threadgroup, leaving half of 128 experts uninitialized; the router read its bias from offset zero of a tensor that lives partway into its shard; the expert-row kernel had no bank offset, so only expert 0 of layer 0 could have been read right; and the norm kernel read its gain as FP32 where the checkpoint stores BF16. The sixth came from the per-visit case and had been invisible for a different reason: the dynamic K/V convolution of a FULL-attention layer read its static taps as the static kernel's channel-major, oldest-first ones, while the checkpoint's `static_weight` is `[kernel][channel]` with index 0 the current token and `weight_proj`'s rows are lag-indexed (`testTheDynamicShortConvolutionMatchesTheHostRule`). Every local layer was exact and the first full one 5.6e-2 off. Two state bugs came with them: the attention half appended its KV row and pushed its convolution histories BEFORE the command buffer ran (exact for the first rows, wrong from the point a convolution reads them), and the score dump had every head's threadgroup 0 writing the same slots.

### Phase 3: Stateful prefill, prefix reuse, and text server integration

**Recorded 2026-11-22: DONE for the host and device text path. One gate is partly deferred: the elementwise logit bound (Phase 2's record explains why rank agreement stands in for it against a BF16 reference).**

- **A resumed follow-up equals a fresh prefill** (`testAFollowUpTurnResumesItsPrefixAndMatchesAFreshPrefill`): the same prompt prefilled in full and resumed from the executor's own prefix generate the same ids. Getting there needed three fixes, all found by that test: the turn runner asked the executor how much was resident and then discarded the answer (re-forwarding the prompt on top of an existing cache, which appends a second copy of the prefix to every visit's KV); the host forward never recorded a forwarded token, so the sequence history -- the only evidence a resume may trust -- was always empty; and the host forward never implemented `reusablePrefix`, so the conservative default answered zero even with the history in place.
- **A cancelled turn resumes exactly its committed prefix** (`testACancelledTurnResumesExactlyItsCommittedPrefix`): cancelled during prefill, the cache holds exactly the forwarded positions, every visit agrees, and a resumed turn from there answers what a fresh prefill answers.
- **Commit-only-after-success** (`testAFailedPositionCommitsNothing`, on the synthetic fixture): a position the KV budget refuses leaves every visit's count and the token history untouched. The first run of that test FAILED -- the budget was checked per visit, so visit 0 had appended its row before visit 1 refused -- and the check now runs once, before the visit loop.
- **A model swap waits for preparation** (`testAModelSwapWaitsForATurnThatIsOnlyPreparing`): `awaitIdle` does not return while a turn holds the slot claim without a registered task, which is the whole of a turn's preparation.
- **Both server routes** (`testBothProtocolsAnswerTextTurnsOverTheLocalServer`, opt-in): see the Phase 2 device record; an OpenAI chat completion and a streamed Anthropic message both answer for this family, with `message_stop` and no error frame.
- **Chunked prefill** (`testAChunkSizeNeverChangesWhatRuns`, `testACancelBetweenChunksResumesExactlyThere`): the prefill runs in chunks, and a chunk is a SCHEDULING unit rather than a batching one -- each position still goes through the executor's own forward, so the answer is identical to the walk's (a chunk size is asserted never to change what runs, only how often progress is reported). What it buys is a progress report per chunk, a cancellation point between chunks, and the commit rule: a cancel between chunks leaves exactly the positions before that boundary committed. A real BATCHED pass -- several positions through one forward -- is a different arithmetic order and would need its own parity case against the walk; it is not this, and for the looped schedule's 48 visits it is not obviously a win (every visit reads the same weights either way, and the batched prefill of the DeepSeek port measured 1.05x at 128 positions).

**Work**

1. Implement bounded chunk scheduling through all 48 visits, chronological local rings, growing global caches, progress callbacks, and commit-only-after-success semantics.
2. Make the executor validate exact resident token/embedding history. A proposed pin is advisory only. Support full and partial common-prefix reuse where all required states prove identical.
3. Implement delta prefill and rollback helpers from the start, including four conv states per visit. Record only forwarded tokens; a sampled but unforwarded EOS/freeze token is not committed.
4. Wire OpenAI/Anthropic text routes, count_tokens through the same planner, streaming usage, prompt/body logging rules, stop vocab, model capabilities, and queue behavior.
5. Add API server smoke coverage alongside the existing port-isolated suites.

**Exit gates**

- Whole prefill, arbitrary chunk sizes including one, and position-by-position walk satisfy the Phase 2 elementwise logit bound at every position and agree on every selected expert route.
- Follow-up output using a resumed prefix equals a fresh full prefill under greedy sampling.
- A cancelled turn resumes exactly its committed prefix, and a model swap waits for preparation as well as execution.
- Both API protocols agree on prompt and usage token counts, stop at the same place, and have correct streaming terminal frames.
- Queue timeout, overlapping in-app/server turns, cancellation, and turn-slot claims show no regression.

### Phase 4: Prompt, reasoning effort, native tools, and agent continuations

**Work**

1. Port the checkpoint template by golden cases rather than hand-guessing its generation suffix. Implement role tokens, system ordering, effort insertion, repeated reasoning/content fields, EOS, and exact whitespace.
2. Support named `none/minimal/low/medium/high/xhigh/max` and numeric effort in [0.0, 0.99], default 0.9. Map the existing two-level UI without blocking the protocol’s richer values; advertise only what the backend renders. Do not force numeric effort into DeepSeek’s 1...100 rule.
3. Render tool declarations as the checkpoint’s `tool_declare` system message and compact sorted JSON.
4. Accept generated `content_invoke_tool_json` objects shaped `{"name": ..., "args": {...}}`, including escapes, nested values and multiple calls. Freeze only after a complete valid object and its message boundary. Incomplete JSON never executes.
5. Canonicalize stored calls for replay, allocate/maintain call ids only at the app boundary, render named tool results, and make live continuation suffixes byte-identical to history replay. Do not enable Qwen’s grammar mask.
6. Extend `StreamingToolParser` coherently with instruction rendering, freeze, result handling, and agent continuation; keep all generic model dialects unchanged.

**Exit gates**

- Every processor golden matches exactly, including special tokens and no double templating.
- Effort names and numeric boundaries render correctly and invalid efforts are refused before the model gate.
- Tool schema, escaped/nested call, repeated call, truncated call, named result, multi-step agent, and prefix continuation tests pass.
- OpenAI and Anthropic tool requests work on streaming and non-streaming routes with correct follow-up history digests.

**Recorded 2026-11-23: PASSED. The prompt half is golden-pinned and the agent continuation completes a real call on the published checkpoint.**

- **Tool declarations, call turns and results are rendered from the template**, pinned by goldens it rendered itself (`Tools/inkling2_tool_goldens.py`, run with the pinned jinja2/transformers environment; `Inkling2ToolPromptTests`): a declaration with defaults for an absent description and parameters, a call carrying nested objects/arrays/booleans and a string with quotes, a newline and non-ASCII, two calls in one turn, reasoning before a call, and a named result. The JSON writer matches `tojson` (`sort_keys`, compact separators, literal non-ASCII, Python's escapes).
- **The call dialect is pure logic and pinned** (`Inkling2ToolDialect`): a block is a call only when its payload is complete, parses, carries an object `args`, and its declared name agrees with the block's own; the freeze rule keys on exactly that, so a truncated payload freezes nothing and stays as prose. Canonicalization keeps the surrounding prose and turns each call into the canonical block the shared parser reads, so the executor, HITL gating, timeline and history stay on one path.
- **The tool declarations reach the template on the server routes too** (`21ff38d`). `Inkling2Prompt` was only ever constructed without `tools:` in production, while the shared planner joined the generic tagged-form paragraph into the system prompt, so an API tool request for this family rendered with no declaration at all. Both routes now pass the client's tools through as the template's `tool_declare` message and withhold the generic paragraph for this family only. Pinned by two pure-logic cases in `ChatPromptFormatterConversationTests`, one per render path (a live suffix and a rebuilt history must not disagree about whether tools were declared).
- Wired: the engine's agent turn freezes and parses through the dialect; the system prompt teaches it (`isInkling2Model` threaded through `buildSystemPrompt`/`buildWorkerSystemPrompt`); the live continuation and the history rebuild both render the result as the family's tool message, with the end tag being the end-of-sampling marker and no separator between a call turn and its result.
- **The agent-continuation gate PASSES on the published checkpoint** (`testATurnAnswersAfterAToolResultOnThePublishedCheckpoint`): the call turn freezes at its own block with `toolFreeze`, the block names `read_file` with `{"path":"/tmp/notes.txt"}`, its opening ids are the reference's eight, the spliced continuation equals its re-encode (240 = 240, 209 resumed), and the continuation after the result answers `It says: "the answer is 42"`. The mechanics are asserted, not the wording: the call's name and argument, the freeze reason, the splice, the resume, and that the answer carries the result's own content.
- **The gate was blocked by the DEVICE graph, not by logit noise and not by the dialect** (2026-11-23). Two defects: (1) the dynamic K/V convolution, which only FULL-attention layers run, read `static_weight` as the static kernel's channel-major, oldest-first taps; the checkpoint stores `[kernel][channel]` with index 0 the CURRENT token, and `weight_proj`'s rows are lag-indexed too. Every local layer matched the host to 1e-6 while the first full layer sat at 5.6e-2, and position 0's logits were 2.3e-1 off the host. With both pairs lag-indexed the whole stack is ~1e-6 per visit and ~2e-6 on the final logits; against the FP32 reference the device path moved from 95/131 argmax and cosine 0.91 to **127/131 argmax, top-4 97/131, and 0.998+ mean cosine in every 16-position band**, which is the BF16 reference's own fidelity (0.9977 at the head) and its own rounding band (0.28-0.36 mean absolute logit difference against 0.28 for the reference against itself). The first greedy divergence that remains is generated step 9, at a reference 4th-5th margin of **0.046** -- a near tie, i.e. the model's own sensitivity, not a port defect.
  (2) `Inkling2ToolDialect.callBlocks` searched for the invoke marker ACROSS message blocks, so the reasoning turn's opener became the call's name, no block parsed, `shouldFreeze` never fired, and the turn narrated past a call it had already completed (it ended `.eos` at 80 tokens instead of freezing at 33). The scan now takes one `<|message_model|> ... <|end_message|>` block at a time; `testACallAfterAReasoningBlockFreezes` pins the exact shape this model emits.
- **The app's own engine drives the same call** (`testAnAgentTurnFreezesACompleteCallThroughTheEngine`, default): `CompletionEngine.launchTurn` with agent tools enabled, the checkpoint loaded as the app loads it, returns `toolFreeze` with exactly one actionable `read_file` call carrying `/tmp/notes.txt` after 79 tokens. That covers the engine's half -- the device runner installed inside the turn task, the freeze hook keyed on this family's block, canonicalization into the shared parser and the parsed call in `TurnResult` -- while tool execution and the result turn remain the harness's shared path, asserted on the runner above.
- **What made the gate findable was a gate that compares the whole stack.** `testEveryVisitMatchesTheHostAtThePublishedWidths` (default, ~22 s) runs one position on both graphs and compares all 48 visits at a 1e-3 bar; `/tmp/dynamoe_inkling2_visit_parity` adds position 1, the only one that attends over a row of history. Before it, the published checkpoint was compared at ONE visit on a synthetic input, and the one-visit case covered layer 0 -- a local layer -- so the full-attention branch had never been compared at all.
- **OPEN: the text path's own goldens dump cannot be regenerated from this repo.** `Inkling2HostForwardTests` skips every case unless `/tmp/inkling2_host_goldens.json` exists, and the message says to run `Tools/inkling2_goldens.py` -- which is not in `Tools/` and never was committed. The three gates that rest on that dump (`blockResidual`, `middleCycle`, `relativeBias`) are therefore not re-checkable on a fresh machine, which is the same class of loss as the logit dump `#78` diagnosed. A replacement tool has to emit exactly what the test reads: `ids`, `logits` (200058), 48 `visits` entries (`physicalLayer` + the last position's `output`), the pass-major `messages` channels, `finalRead`, and a separate `/tmp/inkling2_bias_ref.json` with each visit's head-0 last-query bias row.
- **The reference comparison is reproducible on this machine**: `/tmp/inkling_venv` (transformers 5.24.0 carries `inkling2_loop`), `/tmp/inkling_tool_logits_ref.py` (a prompt's per-position top-8 and full logits, BF16 or FP32), `/tmp/inkling2_ref_tail.py` (teacher-forces the greedy continuation, so the divergence step has a reference), and `Tools/inkling2_compare_logits.py` (reads the device dump's raw rows, prints magnitude, cosine by range and the first position that leaves the reference's own band). Every A/B needs the FP32 reference as the yardstick: the BF16 reference disagrees with its own FP32 run by 0.28 mean / 3.7 max and flips argmaxes, so a bar taken from BF16 faults the port for the reference's rounding.
- **The widest single-logit difference in the device prompt run is the port's own BF16 expert profile, not a defect** (2026-11-23). The 131-position run's worst element is **9.576 at position 105, id 13 (`.`)** -- the reference's own TOP-1 there at 18.33, the port's 5th at 8.75, where the reference's 4th-5th margin is 0.004, i.e. its row is already at the plan's near-tie case. That row is ALSO one of the port's wider ones (mean 1.16 against the run's 0.36), so the spike is the row's own noise landing on one token, not an isolated element. The next largest (7.39 at 66, 5.85 at 69, 5.24 at 79, 5.23 at 81, 4.91 at 82) are the same shape. Three measurements settle where it comes from. (1) It is not the device: the host dump matches the device's first 16 rows to **1e-5**, so the difference is the port's side of the comparison everywhere. (2) It is not reference noise: the FP32 reference is **bit-identical across processes** for the same input, and changing the batch length (131 vs 132 vs 147 tokens) moves it by at most **1e-4**. (3) It is the expert rounding: the port rounds every dequantized expert weight to BF16 (host `bf16Row`/`project`, device `inkling_expert_row_f32`'s `inkling_round_bf16`), which matches a **BF16 load**, while the FP32 reference holds the same bank at full dequantized resolution -- Transformers' `gate_up_proj` after `dtype=float32` is FP32 and only **295 of 4,096 sampled values** lie on the BF16 grid (max delta 1.2e-4), because the loader dequantizes the U8/E4M3 bank itself (`conversion_mapping.py`'s `w13_weight` -> `gate_up_proj` rename plus `Interleave`) rather than reading a BF16 copy. So the port is a THIRD profile (dense FP32, expert weights BF16) whose ~0.3 band against the FP32 dump is that rounding, occasionally amplified to O(1) when a rounding-level residual difference flips a later layer's expert selection -- which is why the same rounding still agrees on **17/17** argmax over the dump's overlap and 127/131 over the prompt. Read the "0.28 the reference disagrees with itself" figure the same way: it is BF16-load-versus-FP32-load, not noise inside one run. The profile is deliberate and must stay: the device kernel rounds identically, so dropping the rounding on the host alone would break host/device parity. `Tools/inkling2_reference_profile.py` re-prints the bank fingerprint in one load.
- **One deviation is still OPEN, and it is not the 9.576: positions 0-3 agree with the FP32 reference 30x better than every row after them.** The port's mean difference is 0.011-0.019 over positions 0-3 and 0.10-1.62 from position 4 on, i.e. a step at exactly the four-tap convolution's full-receptive-field position rather than a drift. Normalised by each row's own RMS the run sits at 3.7% on average but 22.5% at position 4, the worst row in the prompt. Row magnitude does not explain it (correlation between the mean difference and the row's RMS is only 0.10), so it is either one of the 1-3% intermediate rules the conformance ledger already accepts (`blockResidual` 2.8e-2, `middleCycle` 1.6e-2, `relativeBias` 4.8e-2) or an early-position rule that only these rows exercise. It does not gate the turn -- the greedy continuation agrees for 8 steps and the agent freeze is exact -- so it is recorded as the next fidelity item rather than a blocking defect: the cheap next probe is a host-vs-reference comparison at position 5 of a single visit's inputs, where the reference can be hooked.
- **The reference's last few rows are not stable targets, so a per-position gate must drop them.** Appending a tail moves rows near the end of the 131-token prompt (**0.37 mean / 2.16 max at row 126**, decaying forwards), and the movement is neither causal nor monotone in the tail's length: an 8-token tail moves **nothing**, 9-12 move, 13 moves **nothing again**, 14-16 move, and a 16-token tail drawn from the prompt's own body moves the same rows by the *same amounts* as the reference's own continuation. A 16-token all-special tail moves nothing. So these rows are not future information a causal graph could match; they are a numerical regime the release enters depending on the input, and the port's 0.2-0.37 difference there is the reference moving, not the port. Judge the port on a dump's earlier rows and on its own causal continuation; `Tools/inkling2_reference_profile.py` prints the row-by-row table.
- Effort: the template's whole vocabulary is already supported (`Inkling2Effort`: `none` 0.0 through `max` 0.99, numeric in [0, 0.99], refused outside). The app's two-level UI maps `low`/`high`, and `ChatPromptFormatter.inkling2Effort` is the one place that mapping lives. `/v1/models` advertises the accepted `reasoning_effort` vocabulary (`minimal`/`low`/`medium`/`high`): `ModelConfig.supportsThinkingLevels` already answers true for this family because its template names `reasoning_effort`, and the models route copies the adapter's table from that flag -- so the earlier "not added yet" note was stale, not a gap. Re-checked against the template itself with plain jinja2 (the `generation` tag stubbed): all seven named efforts and numerics in [0, 0.99] render, and 1.0, -0.1 and an unknown name raise.

### Phase 5: Still-image vertical slice and vision optimization

**Work**

1. Preserve alpha/preprocessing policy through composer and server decoding. Use the Inkling checkerboard instead of the generic RGB interpretation, and include family/preprocess identity in encoded-span caching as needed.
2. Implement pixel tiling, reference tower and row placement, expanded prompt pricing, carried spans, digest checks, and safe refusal for count/order/grid mismatches.
3. Integrate markers in original content order on both protocols and in-app turns. Do not fetch remote images.
4. Add the host tower first, then Metal batched GEMMs and tiled bidirectional attention, position interpolation/RoPE/merger. Validate the whole tower against the reference before enabling the fast path.
5. Generalize `LoadedModelInfo` vision cost/capability types enough for both families; report the expanded prompt cost and a truthful estimate/range derived from the tiling plan.
6. Add an encoded-span cache bounded by content digest and model identity, cleared on configure.

**Exit gates**

- Pixel and tower fixtures match, including alpha and non-40-multiple dimensions.
- One red-square image, text-rich image, two interleaved images, and before/after-text images produce semantically correct answers on real weights.
- Image-expanded input tokens match `/v1/messages/count_tokens` and both response usage paths; OpenAI and Anthropic streaming stay alive with heartbeats and surface errors as their protocols require.
- Continuations place the same digest’s rows at recorded spans and never bind a changed picture to old rows.
- Prompt/cache allocation is refused before overflow, and very large images cannot allocate an unbounded attention matrix.

**Recorded 2026-11-23: STARTED. The host tower exists and matches the released implementation to 5.2e-7 relative.**

- **The picture path is not the generic resize-and-normalize one.** `Inkling2ImagePixels` keeps the source at its own resolution, pads it up to whole 40-pixel cells, composites transparency onto the processor's 204/153 checkerboard (8-pixel cells) instead of dropping alpha, and only then normalizes `(x / 255 - 0.5) / 0.5`. The pad is written on the 0-255 scale BEFORE the rescale, so a padded pixel normalizes to **-1.0078**, not to 0 -- a detail that changes every border patch. The cell count is what the prompt pays (`languagePositions`), pinned without a model by `testThePicturePadsToWholeCellsBeforeItNormalizes` and `testTransparencyCompositesOntoTheCheckerboard`.
- **The tower is transcribed, not guessed, and pinned stage by stage.** `Inkling2VisionForward` runs patch projection (20x20 patches out of the 40x40 language-token cell, in `(row-block, col-block, row-within, col-within)` order), the bilinear resample of the learned 32x32 position table (`align_corners=false`, half-pixel centres, border padding), the per-image class token, 23 pre-LN encoder layers (bias-carrying LayerNorm, non-causal attention, interleaved-pair 2D RoPE, exact-erf GELU, per-channel layer scales) and the merger (`2x2*1024 -> 4096 -> GELU -> 2304`). `Tools/inkling2_vision_goldens.py` runs the checkpoint's own `Inkling2VisionModel` in FP32 over a 3x2-cell picture and dumps the embeddings, the pre-norm, EVERY layer's output and the pooler rows, so a wrong patch order, a bicubic resample, a rotate-half RoPE or an attending class token each fail a named stage instead of a summary. Measured: **worst stage 5.19e-7 relative** (layer 7), pooler rows 6 x 2304.
- **The vision tower must be taken from the full conditional-generation model.** `Inkling2VisionModel.from_pretrained(SNAPSHOT)` from the repository root matches none of its names (`model.visual.backbone.*` in the checkpoint vs `embeddings.*` in the module) and silently initializes a RANDOM tower -- finite, correctly shaped, and worthless as a golden. The first version of the tool did exactly that; `full.model.vision_tower` is the released tower.
- **The host tower is scalar Swift and costs 582 s at `-Onone`, so the real-checkpoint case is opt-in** (`/tmp/dynamoe_inkling2_vision`); the layout, resample, RoPE and pixel cases run by default with no model. The obvious next speedup is the DSV4.1 precedent: one `cblas_sgemm` per block of tokens instead of a per-token loop, which there turned a 68 s host encode into 2.7 s.
- **The catalog resolved every non-text role against `.text`, so vision and draft tensors were invisible.** `Inkling2WeightCatalog.tensor(_:scope:layer:)` defaulted `scope` to `.text` while the index keys every slot as `scope|role|layer`, so `visionPatchEmbedding` came back missing from a checkpoint that publishes it (and the DSpark roles had the same hole). `defaultScope(_:)` now derives the scope from the role (`isVision` -> `.vision`, `isDraft` -> `.sidecar`), and `tensor`/`key`/`layers(of:)` default through it. Text lookups are unchanged.
- **An attachment carries its cell grid, and one placeholder expands to one marker per cell AFTER tokenization.** The template writes ONE `<|content_image|><|unused_200054|>` per image and the released processor then replaces that single placeholder with `rows * columns` copies, so the grid belongs to the part (`Inkling2ContentPart.image(Inkling2ImageGrid)`, priced by `Inkling2VisionSpec.languagePositions(width:height:)`) rather than to a caller's side array; routed through SubQ as `grid_on_part` 0.995 @ 0.992 against a pixels-carrying part and a bare-part-plus-side-array. The expansion is `Inkling2ImagePrompt.expand(tokenIds:spans:imageTokenID:rowWidth:)`, which returns expanded ids, a position-to-row map and one `Range<Int>` per image, and it runs on TOKEN IDS rather than on the rendered string (`token_like_dsv41` 0.894 @ 0.787): the rendered prompt therefore still matches the template byte for byte, which is what keeps its goldens and its digest stable, while the ids the cache and the sampler see are the expanded ones. The two produce the same ids because the placeholder is a `normalized: false` added token. The marker id resolves through the top-level config's `image_token_id`, which the checkpoint does NOT publish (its `config.json` has only `eos_token_id`), so the decoder's 200054 fallback is the tokenizer's own added token, pinned by `testTheSpecPricesTheSameCellsThePixelsProduce`. **A count mismatch is refused, not reconciled**: an extra placeholder has no rows and a missing one has nowhere to put them, and either silently shifts every later position; that is also why the reference raises `ValueError` there, and it is a deliberate divergence from DSV4.1's expand, which passes an unfilled placeholder through and reports it. A span whose row count disagrees with its grid, a row that is not `textHiddenSize` wide, a grid with no cells and an out-of-range marker id are all named refusals. Pinned with no model by `testOneAttachmentBecomesOneMarkerPerCell`, `testTwoInterleavedImagesKeepContentOrder`, `testAPlaceholderCountThatDisagreesWithThePicturesIsRefused`, `testASpanThatDisagreesWithItsGridIsRefused` and `testTheGridsAreCollectedInRenderOrder`; `Inkling2Prompt.imageGrids(in:)` is the render-order list a caller zips encoded spans against, so a picture cannot land on another picture's markers.
- **The tower's rows reach the graph through a seam of their own, and a supplied row is NOT embedding-normed.** The reference `masked_scatter`s the image rows into `inputs_embeds` AFTER `embed_tokens`, and `embed_tokens` here is `embed_norm(embedding)`, so `Inkling2CPUForward.forwardRow` calls `forwardHidden(row, position:)` directly and records the marker id in history rather than normalizing the row; `Inkling2PositionForward.forwardRow` defaults to refusing by name so a partial executor cannot silently substitute the token embedding, and `Inkling2TurnRunner.run(imageRows:)` refuses a row outside the prompt and forwards `forwardRow` at its mapped positions and `forwardToken` everywhere else (pinned by `testImageRowsRunAtTheirOwnPositions`, `testAnImageRowOutsideThePromptIsRefused`, `testAnExecutorWithNoImageSeamRefusesARowByName`). **An image-bearing prompt never resumes a prefix** (`proposed = 0` when rows are present, `testAnImagePromptNeverResumesAPrefix`) because the recorded ids cannot distinguish two same-sized pictures, and a continuation re-places rows from `Inkling2ImagePrompt.place`, which checks span/range order, bounds, that every position in a range is the marker and that each range length equals the grid's cell count, and refuses a digest that disagrees with the recorded span. `CompletionEngine` prices the expansion BEFORE the tower (the cache must be sized for `count - placeholders + sum(languagePositions)`, not the placeholder count), keeps the tower per checkpoint path and a 4-entry span cache keyed `snapshot path|sha256` (so a stateless re-render does not re-encode), asserts the bound length equals the priced one, and hands the expanded ids to the prefix pin and the sampler; `TurnResult.imageSpans` carries them forward. **Measured on the published checkpoint at `-Onone`: a 50x40 red rectangle is 1x2 cells, the tower encodes 9 tower tokens (1 class + 8 patches) in about 4 minutes, and the turn took 3136.084 s (52 min) for a 6-token budget** whose whole allowance went to the reasoning opener (`<|message_model|><|content_thinking|>The user is asking`), because the checkpoint opens `content_thinking` itself at effort 0; every structural check held (one placeholder became two markers, `plan.spans == [1..<7]`, rows placed, device path exercised) and only the colour assertion failed, which is why the case asks for 64 tokens and asserts on the CONTENT channel rather than the whole transcript. Two operational lessons came with it: `try? Inkling2MetalForward(...)` hid a silent fall back to the host graph for 57 minutes, so the engine now records `inklingDeviceHalfStatus` ("not attempted" / "cached" / "built at N positions" / "failed: ..." / "no Metal device, queue or library" / "disabled by dynamoe_inkling2_device_off") and logs a `DiagLog.warning`, pinned by `testTheEngineInstallsTheMetalVisitRunner`; and sampling the live process named the `-Onone` cost that governs a real image test — every `runVisit` call re-DECODES its control matrices out of the shard (`Inkling2SnapshotWeights.matrix` -> `decode` -> the BF16 loop) and the final head is 200058x2304 scalar dot products (`Inkling2Forward.swift:290`), both Phase 6 candidates rather than Phase 5 defects. The engine-level image case therefore stays opt-in (`/tmp/dynamoe_inkling2_image_turn`), and at `-O` the same case closes the semantic gate: **52.297 s** and the model writes `<|content_text|>Red<|end_message|>` after its own reasoning channel, i.e. 60x the `-Onone` wall on identical work, so a 52-minute Debug figure is the debug cost and not the port's.
- **A picture reaches the app and the server as a typed content block, and the gate that offers the button is the snapshot's own files.** `ChatPromptFormatter.ConversationTurn` gained `imageGrids`, and the Inkling branch of `formatConversation` turns each into its own `.image` part (`conversation_turn_grids` 0.976 @ 0.965, against splicing `<|content_image|><|unused_200054|>` into the turn's text, which would nest the image pair inside a text block and bypass the renderer's grid check). The API server keeps a message's content ORDER as one `U+FFFC` sentinel per picture, so `inklingContentParts(text:grids:)` splits on the sentinel and interleaves text and image parts (`split_into_parts` 0.997 @ 0.995) rather than moving every picture in front; a turn with pictures but no sentinel still puts them first, which is the order the composer writes. **A count mismatch renders NOTHING**, so the prompt is left without the placeholders the pictures need and the engine's own expansion refuses the turn by name -- the same fail-closed rule as the expansion itself, one layer earlier. `ServerController.attachImageGrids` is Inkling's half of the image gate (decode with `Inkling2ImagePixels`, one grid per part, sentinels left in place for the renderer), `imagePartBytes` is the refusal shared with the DeepSeek V4.1 path so the two cannot word an unfetchable URL differently, and `inklingPromptTokenCount` prices the EXPANDED prompt from the same grids the engine expands, so the usage a turn reports is the prompt it ran. Two silent drops came with wiring it: the server's single-turn fast path takes a STRING, so a turn carrying grids or a sentinel now goes through `formatConversation` (otherwise a one-message image request would have tokenized the sentinel and lost the picture), and `resolveTurnImages` had to branch on the family before `visionMarker`, which is DeepSeek V4.1's spelling and stays that way (`prompt_markers` 0.877 @ 0.816 -- the Inkling pair is the prompt renderer's own, so no second marker type was added). `ModelConfig.supportsVision` gained the Inkling branch and reads `Inkling2Preflight.visionTowerIsPresent` (already the loader's own anchor list), so a config that promises a tower the snapshot has not shipped keeps the composer closed (`tower_presence_gate` 0.999) and a finished download opens it; `GET /v1/models` therefore advertises `vision: true` with NO `vision_cost`, because the one measured cost model belongs to another checkpoint's resize plan (omit 0.029). Pinned with no model by `Inkling2VisionPlumbingTests` (7 cases: the parts split with two interleaved pictures, the mismatch refusal, the rendered block's exact markers and the sentinel never reaching them, the server's grids/decoded pixels/sentinel survival, the missing-turn refusal, the expanded price and its mismatch refusal, and the gate's three snapshot states), plus the published checkpoint asserting its own `supportsVision`; 199 fast tests green across the Inkling, adapter, formatter and server-logic suites.
- **Phase 5 is CLOSED (2026-11-23), with two opt-in limits recorded rather than hidden.** Every exit gate now has a passing run, and the run that closes them is the `-O` one: at `-Onone` the image turn spent its whole 1800 s request budget and *that* is what "stuck" looked like -- two cases after it then failed with `engineNotReady("A generation is already active")` because the timed-out turn still held the single-flight slot. The same case answers in **399.886 s** at `-O` (plain 188 / pictured 192 prompt tokens, delta 184 = `2 + cells`, the tower's 1x2 grid), the in-app turn in **51.765 s** with `<|content_text|>Red<|end_message|>`, the two-picture turn in **41.636 s** ("first appears to be a red square, second a blue square", spans `1..<2` then `3..<5` -- each picture's own cell count, in order, on disjoint ranges), and the 23-layer tower fixture in **4.768 s** (opt-in `/tmp/dynamoe_inkling2_vision`). The pipeline gates were re-run in the same pass: device visit parity **7.4e-07** relative, follow-up prefix resume, cancelled-prefix resume and visit-runner replacement all pass, the host turn answers after 18 prompt tokens, both protocols answer over the local server (OpenAI 200 non-streaming, Anthropic SSE with 48 pings and no error frame), and the synthetic checkpoint's three reference-golden cases pass against goldens regenerated from `Tools/inkling2_synthetic.py` + `Tools/inkling2_synthetic_golden.py`. **The red-square image is proven end to end; a text-rich image is NOT** (the tower and the prompt path are pinned on fixtures, but no real-weights turn has been run on one), and the plan's own processor has a 4,096-pixel-a-side ceiling, so "very large" here means 102 grid cells / 10,240 tower tokens, not an unbounded span. The remaining gates were closed by `Inkling2PhaseFiveTests.swift` (no model): the `count_tokens` route prices through the messages route's own `buildTurnPlan` and both reply writers report `plan.promptTokenCount`, which is the expanded count; the prompt cap is checked AFTER the picture expands and the engine's budget check precedes the cache allocation, so a refusal happens before overflow; and `Inkling2ImagePromptBudgetTests` pins the wiring that keeps a refusal before overflow -- the cap checked after the expansion, the engine's budget check before the cache allocation -- while the span arithmetic itself is already pinned by `testTheSpecPricesTheSameCellsThePixelsProduce`, `testOneAttachmentBecomesOneMarkerPerCell` and `testAPlaceholderCountThatDisagreesWithThePicturesIsRefused` (attention is per-visit over a window plus the compressed rows, so no prompt length allocates an unbounded score matrix).

### Phase 6: Decode/prefill optimization and packed-expert policy

**Work**

1. Instrument wall time, queue/device windows, host encode, per-stage costs, expert bytes/misses/hits/evictions, pread time, arena allocations, prefill shape, and KV precision. Keep instrumentation off in production.
2. Benchmark isolated decoder stages at published widths. Use coalesced threadgroup-per-row or paired-row kernels only where measured; do not copy DeepSeek kernel-form thresholds blindly.
3. Implement the separately selectable trained W4A4/W8A8 activation profile, then fuse its activation quantization with expert projections where profitable, preserving candidate reconstruction/error order and rounding. Add batched forms for prefill that share weight reads across route groups without changing reduction-sensitive routing.
4. Optimize relative-bias attention separately: FP32-softmax split-K for global visits, chronological ring kernels for local visits, no RoPE, sink included without a fake cache row, and tiled bias evaluation without full-score allocation.
5. Fuse rank-64 gates, residual reads, static/dynamic convolutions and residual bookkeeping only with parity tests. Avoid fast-math or dynamic thread-array indexing that changes semantics or kills performance.
6. Implement packed expert arenas and pread ranges. Compare mmap, resident packed sets, and streaming under identical cold/warm workloads; optionally pin dense text weights if page-fault measurements justify it. Keep vision and DSpark budgets separate.
7. Add batched prefill with per-chunk expert unions and measure the crossover against walking. The chunk width is a measured policy, not a fixed assumption.

**Exit gates**

- Every optimized kernel has parity coverage on odd/partial shapes and route ties, plus graph-level prefill/decode equivalence in its own numerical profile.
- The trained-activation profile matches independent vLLM goldens, while the HF baseline still matches Transformers. A profile change requires a fresh turn and clear diagnostics; acceptance and performance are reported for both profiles.
- End-to-end improvements hold across at least two shipping `-O` runs with stated cold/warm regime and variability. No first-pass-only win is claimed.
- The selected arena/residency policy fits its memory plan and does not trigger a compression storm or unbounded duplication.
- Baseline and optimized numbers include TTFT, prefill positions/s, decode tokens/s, bytes/token and peak memory. Compare against MLX on the same machine when available; external benchmark numbers are orientation, not acceptance thresholds.
- No stage remains “optimized” based only on an isolated cache-warm kernel whose in-situ effect is unmeasured.

**Progress: slice 1, the output head (2026-11-24).**

The Phase 5 record named two Phase 6 candidates from a live sample: `runVisit` re-decoding its control matrices (fixed earlier by `Inkling2DeviceWeights.hostMatrix`'s cache, commit `ea16443`) and the final head at `Inkling2Forward.swift:290`, which materialized the 922 MB BF16 unembedding as a 1.84 GB `[Float]` and ran 461M scalar dot products on every token -- in BOTH paths, because the device runner only replaces `runVisit`. **Measured at `-O` on the published checkpoint, the head was not the bottleneck, and saying so is the result**: the chunked host rule costs **0.454 s per token**, the device projection **2.0-3.4 ms (mean 2.6 ms, 358 GB/s over 921 MB)**, a whole device position at position 0 costs **0.296 s** -- so the head is **0.9%** of a position -- and the Phase 5 host turn (18 prompt + 8 generated in 586.2 s) implies ~22 s per position, i.e. ~2% there. **The 48 visits are the position** (~6.2 ms each at position 0), and nothing yet splits them; that is the next slice.

- **The kernel form was decided by a rotated A/B, and the first reading was an artefact.** `bf16_gemv_rows` (one threadgroup per row, lanes stride the columns) against `bf16_gemv` (one thread per row): the first pass on an idle GPU read **75 ms vs 8.6 ms**, i.e. it reversed the eventual answer by more than the effect, because it was the first dispatch after idle (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #53"). With a warm-up, six rounds and the arm order rotated every round, the means are **coalesced 2.6 ms (358 GB/s) against 4.4 ms (209 GB/s)**, a 1.7x win that also improves monotonically across rounds (3.4 -> 2.0 ms) as the clock ramps. `HeadKernelForm` keeps both so the choice stays a measurement; the shipped default is `.coalesced`.
- **Shipped: a head seam and a bounded host rule.** `Inkling2CPUForward.rawLogits` is the new entry point, `headRunner` its substitution (the softcap stays in the graph so the two paths cannot apply it twice or not at all), and a runner that answers the wrong number of logits is **refused by name** -- the sampler derives its vocab range from `logits.count`, so a short array is silently a smaller model. `Inkling2MetalForward.headLogits` is one dispatch and one wait. `Inkling2SnapshotWeights.logits` decodes the table **1,024 rows at a time**, so the host path's peak is 9.4 MB of scratch instead of a 1.84 GB transient -- a memory-plan gate rather than a speed claim -- and the arithmetic is the reference's column-ordered FP32 sum, so the chunk is **bit-identical**, not approximately equal. `CompletionEngine` installs the runner with the visit runner and clears both at the end of the turn; a host-only turn leaves it nil and runs `weights.logits`.
- **Coverage.** `Inkling2DecodeTests.swift`: the chunked rule sums EXACTLY like the whole-table rule at chunk sizes that do and do not divide the row count (1, 2, 3, 5, 8) and across every stored format the snapshot can publish (BF16/F32/F16/E4M3); a zero row count, a zero chunk and a wrong-width hidden row are each refused by name; the seam substitutes the projection without softcapping and a wrong-length runner is refused; and on the published table **both kernel forms agree with the host rule to 5.3e-07 relative, with the same argmax and top eight**. The measurement case is opt-in (`/tmp/dynamoe_inkling2_decode_measure`) and must be run from an `-O` build.
- **A picture the app could not decode no longer disappears without a word.** The Inkling branch of the composer's prompt builder built its attachment grids and never appended the `localAttachmentNotice` that the DeepSeek and generic branches append, so a failed decode (or a snapshot whose tower is not installed) left the turn asking about a picture the model never received. The notice is now appended when no grid was built and the guard admits a picture-only turn. No test: that branch has none today, and extracting it is a separate refactor.
- **Not done in this slice:** Phase 6 work items 3-7, and the rest of item 1 (queue/device windows, per-stage costs, expert bytes/misses/evictions, pread time, arena allocations, prefill shape). The number this slice leaves behind is the one that redirects the phase: a position is 48 visits, and the head is one dispatch of it.

**Progress: slice 2, the per-visit instrument (2026-11-24).**

Slice 1 could only say "the 48 visits are the position". `Inkling2Profile.swift` splits one now: a visit is cut into the nine stages the graph actually has (the two attention buffers separated by their state work, the post-attention convolution, the MLP read, the router front, the mixture, the tail), and each stage is billed four ways -- host ENCODE, QUEUE wait before the GPU began it, the buffer's own DEVICE window, and the HOST residue (read-backs, the relative/sink projections, the KV staging copy). Off by default: `recordsVisitTiming` is false in every production path, `dynamoe_inkling2_profile` is the seam read once per turn, and the engine resets the table per turn and logs one `DiagLog` line (`inkling2.profile`) at the end of it.

**Measured at `-O` on the published checkpoint**, four positions after a discarded warm-up: **5.12-5.84 ms per visit, 245.5-280.4 ms per position** (two processes; slice 1's single-shot 0.296 s is inside that range), **2,496 commits -- exactly 13 per visit -- and 1,536 routed experts**, reading **7,016.6 MB of expert banks per four positions, i.e. 36.5 MB per visit and 1.75 GB per position**. Per visit, by wall:

| stage | ms | share | host | encode | queue | device | commits |
|---|---|---|---|---|---|---|---|
| expertMixture | 2.067 | 40.4% | 0.090 | 0.295 | 0.597 | 0.641 | 8 |
| hostPrep | 0.721 | 14.1% | 0.721 | - | - | - | 0 |
| attentionState | 0.463 | 9.1% | 0.463 | - | - | - | 0 |
| attentionFront | 0.448 | 8.8% | 0.009 | 0.074 | 0.088 | 0.215 | 1 |
| mlpFront | 0.432 | 8.4% | 0.004 | 0.020 | 0.079 | 0.268 | 1 |
| attentionTail | 0.329 | 6.4% | 0.002 | 0.028 | 0.083 | 0.160 | 1 |
| mlpRead | 0.251 | 4.9% | 0.251 | - | - | - | 0 |
| postAttentionConvolution | 0.167 | 3.3% | 0.005 | 0.025 | 0.083 | 0.004 | 1 |
| mlpTail | 0.168 | 3.3% | 0.003 | 0.031 | 0.082 | 0.003 | 1 |

- **The commit/wait machinery is 43% of a visit, and the arithmetic is 25%.** Adding the buckets up: **host 1.55 ms (30%), queue wait 1.01 (20%), device 1.29 (25%), encode 0.47 (9%)**, and a further **0.72 ms of drain that is neither the queue nor the window** -- the `waitUntilCompleted` wake-up, ~55 us across 13 waits, 14%. In other words **2.20 ms of every 5.12 ms visit is spent handing 13 buffers to the queue and waiting for them to come back**, against 1.29 ms of GPU work. The queue figure is 78 us per commit, which is the same 0.08-0.17 ms pickup `Benchmarks/PERF_FINDINGS.md` "DSV4.1 #48" measured for a submission into an idle queue, now arriving 624 times per position. Two of the 13 commits -- `postAttentionConvolution` and `mlpTail` -- are 4 us and 3 us of device work behind 83 us of queue wait each, i.e. they are almost pure submission.
- **The mixture's eight buffers are seven commits that need not exist.** The expert chain is gate/up, the softcapped SwiGLU, the down projection and a weighted fold into a DEVICE accumulator, and nothing is read back between experts, so the eight chains could share one command buffer and one wait without changing a single arithmetic order (the queue already serializes them). What that is worth, measured: of the mixture's 0.59 queue wait, 0.44 wake-up and 0.30 encode per visit, folding eight into one removes seven eighths of the first two and most of the third -- **~1.17 ms of a 5.12 ms visit, ~23%, ~56 ms per position** -- and it is the next slice, not something this one did.
- **A Debug figure is the compiler, not the port.** At `-Onone` the same position reads **14.5 s** and **93% host** (`hostPrep` 154 ms per visit, `attentionState` 75, `mlpRead` 53 against 6 ms of device), which is the ratio to remember when a Debug run looks healthy or catastrophic.
- **What this family still cannot measure, stated rather than faked.** Expert misses/hits/evictions and pread time need an expert ARENA, and Inkling 2 has none: its banks are the shard mmap, so the profile records the bytes those banks occupy (a function of the published strides) and nothing about residency -- that is Phase 6 item 6's instrument. Prefill shape and KV precision are turn-level facts with no meaning inside a visit; the turn's prompt length travels in the log line's label and the precision in `TurnPrepInfo` (`.fp32` for this family), and neither is invented as a visit field.
- **Coverage.** `Inkling2ProfileTests.swift`: six model-free cases (the host residue, a residue that answers zero rather than wrapping, per-buffer billing, accumulator/reset semantics, the summary's per-visit arithmetic and its omission of unmeasured stages, and the commit table), two GPU cases (`Inkling2BufferClock` against a real `inkling_sigmoid_inplace_f32` dispatch -- the window is non-empty and every bucket is inside the interval that contained it -- and a buffer that never ran reporting no window rather than a zero), and two published-checkpoint cases: one asserts the instrument is INERT until it is switched on and then bills exactly one record set per visit, 13 commits per visit, exactly `topK` expert commits, and stage walls tiling **98.66%** of the visit wall; the other measures the table above and is opt-in (`/tmp/dynamoe_inkling2_profile`). **A trap worth keeping:** `onVisitComplete` lives in the host `runVisit`, so an installed device runner -- the very substitution under test -- never fires it, and both published cases count visits at the runner instead. That mistake cost one red run whose only symptom was a visit count of zero beside a fully populated table.

**Progress: slice 3, one command buffer for the whole mixture (2026-11-24).**

Slice 2 priced the strongest candidate it found and deliberately did not take it: `expertMixture` was **2.067 ms of a 5.12 ms visit (40%)**, and seven of its eight commits were pure hand-off -- nothing read a value back between experts, so the queue was serializing eight chains that one buffer could carry. `mlpHalf` now encodes every routed expert's gate/up, softcapped SwiGLU, down projection and weighted fold into **ONE command buffer, one encoder and one arena**, then commits and waits once. The arithmetic order is unchanged: Metal runs the dispatches of a compute encoder in the order they were encoded, hazard tracking is on (the default `tracked` mode, the same dependency the router front and every attention stage already rely on), and the accumulator is device-resident, so the folds still land in route order. The per-expert bytes accounting stays where it was -- it iterates the same bank strides, guarded by the profile being non-nil -- so the residency numbers a future arena policy will need are still per expert, not averaged.

**Measured at `-O` on the published checkpoint**, four positions after a discarded warm-up, two processes: **4.19-4.33 ms per visit and 201.3-207.9 ms per position**, against slice 2's **5.12-5.84 ms and 245.5-280.4 ms** on the same harness. Per visit that is **-0.79 to -1.65 ms (-15% to -28%)**, and the range is quoted rather than the better end because the two baseline processes themselves differed by 14% -- the sign is the same in both pairs, which is the claim this slice makes. `expertMixture` alone goes **2.067 -> 1.04 ms (-50%)**, and the commit count per visit goes **13 -> 6**: five single-buffer stages plus the mixture's one.

| stage | ms | share | host | encode | queue | device | commits |
|---|---|---|---|---|---|---|---|
| expertMixture | 1.039 | 24.8% | 0.006 | 0.217 | 0.083 | 0.643 | 1 |
| hostPrep | 0.72 | 17.2% | 0.72 | - | - | - | 0 |
| attentionState | 0.46 | 11.0% | 0.46 | - | - | - | 0 |
| attentionFront | 0.44 | 10.5% | 0.009 | 0.070 | 0.086 | 0.213 | 1 |
| mlpFront | 0.43 | 10.2% | 0.004 | 0.019 | 0.078 | 0.268 | 1 |
| attentionTail | 0.33 | 7.9% | 0.002 | 0.028 | 0.081 | 0.162 | 1 |
| mlpRead | 0.25 | 6.0% | 0.25 | - | - | - | 0 |
| mlpTail | 0.20 | 4.7% | 0.003 | 0.053 | 0.086 | 0.003 | 1 |
| postAttentionConvolution | 0.17 | 4.0% | 0.005 | 0.025 | 0.085 | 0.003 | 1 |

- **What the fold actually removed was queue wait and encode, exactly as predicted, and the prediction was in the middle of the range.** The mixture's **queue wait 0.597 -> 0.083 ms** and **host residue 0.090 -> 0.006**; **encode 0.295 -> 0.217** (seven fewer `makeCommandBuffer`/`endEncoding` pairs, not zero, because the dispatches still have to be encoded). The **device window is unchanged (0.641 -> 0.643)**, which is the control that says the arithmetic did not move. Across the whole visit the queue bucket goes **1.01 -> 0.50-0.62 ms** and the drain residue **0.72 -> ~0.6**, i.e. slice 2's 2.20 ms of hand-off per visit is now ~1.5.
- **The next largest non-device terms are pure host, not submission.** With the mixture folded, `hostPrep` (0.72 ms, 17.2%) and `attentionState` (0.46, 11.0%) are the two biggest stages that commit nothing, and both are the host graph's own work -- which is the same finding as slice 2's bucket split, now visible without the mixture's eight waits in front of it. `postAttentionConvolution` and `mlpTail` are still almost pure submission (3 us of device behind 85 us of queue each) but they are 0.37 ms between them, and folding them was NOT done here: they read a host value back (`state.pushConvolution`) or are separated by one, so the merge is a different change with its own ordering argument.
- **Parity is asserted on the mixture itself now, not only on the half.** `testTheDeviceMLPHalfMatchesTheHost` already compared the half against the host visit at a 3e-3 band; that band is 1,300x looser than the fold's own signal, so it could not have caught a lost ordering. The case now asserts the **mixture accumulator against the host graph at 1e-4**, using the same `mixtureOut` capture. Measured on the published checkpoint: **mixture 2.30e-06 relative**, whole half **3.99e-07**, and the named stages either side of it (norm 4.32e-07, route scores 1.58e-06, route indices exactly 0, route weights 1.15e-06, postMlpConv 4.48e-07) -- i.e. the fold is indistinguishable from the arithmetic it replaced.
- **The instrument's own commit invariant moved with the change, deliberately.** `Inkling2ProfileTests` asserted `expertMixture.commits == visits * topK`; it now asserts `== visits`, with the twelve-commit reading recorded beside it. The `experts` and `expertBytes` totals are unchanged (1,536 experts and 7,016.6 MB per four positions), because the fold removed buffers, not work.
- **Not done in this slice:** Phase 6 items 2-7 and the rest of item 1's arena-dependent fields (misses/hits/evictions, pread time, arena allocations) -- this family still has no expert arena, so those belong with item 6, and prefill shape and KV precision stay turn-level facts in the log label and `TurnPrepInfo` rather than invented visit fields.

**Progress: slice 4, the host inner products through Accelerate (2026-11-24).**

With the mixture folded, the two largest stages that commit nothing were pure host: `hostPrep` (0.72 ms, 17.2%) and `attentionState` (0.46, 11.0%), plus `mlpRead` (0.25). All three are the host graph's own arithmetic running inside a device visit -- the `attn_res` block reads, the message gates, the input norms, the relative and sink projections, the KV staging copy -- and all of it was bounds-checked scalar loops. They now go through Accelerate: `Inkling2Math.linear` is one `vDSP_dotpr` per row with `linearScalar` beside it as the definition, `blockResidualReadVectorized` is the same expression as `blockResidualRead` with each loop as a vDSP call, and `rmsNorm` / `l2Clip` are `vDSP_svesq` + `vDSP_vsmul` (+ `vDSP_vmul` for the gain). `Inkling2CPUForward.linear` delegates to the math rule and `blockRead` calls the vectorized form, so the graph keeps its method while the rule is pinned model-free.

**Measured at `-O` on the published checkpoint**, four positions after a discarded warm-up, two processes, against slice 3's same-harness reading:

| | slice 3 | slice 4 |
|---|---|---|
| ms/visit | 4.21 | **3.46** (both processes) |
| ms/position | 202.06 | **165.88 - 166.25** |
| hostPrep | 0.71 | **0.44** |
| attentionState | 0.46 | **0.09** |
| mlpRead | 0.25 | **0.15** |

- **The two stages that were pure host fell by exactly what the isolated A/B predicted.** `relativeStates` (384x2304 of scalar dots) measured 10.3x with vDSP and `sinkLogits` 10.0x, and `attentionState` went 0.46 -> 0.09 (5.1x in situ, the remainder being the KV read-backs and the staging copy this slice did not touch). `hostPrep` 0.71 -> 0.44 is the block read's 2.5-3.7x isolated arriving diluted by the message gates and three norms in the same stage.
- **Nothing on the device moved, which is the control.** Every committing stage reads the same as slice 3 (`attentionFront` 0.44, `mlpFront` 0.44, `attentionTail` 0.33, `expertMixture` 1.05 with device 0.639, `mlpTail` 0.20, `postAttentionConvolution` 0.17), so the change is confined to the host residue and did not perturb the GPU graph's arithmetic.
- **The isolated A/B chose the threshold, not a copied one.** vDSP on a short dot was measured slower (0.90-0.91x at widths 3 and 8), even at 16 (1.03x), and clearly ahead from 32 up (1.37x at 32, 2.4x at 64, 4.2x at 128, 10.4x at 2,304), so `linear` keeps rows under 16 columns scalar. The exact-shape A/B on this graph's own shapes: relative states 10.3x, sink logits 10.0x, gate-down 10.8x, gate-forget 2.8x, all agreeing with the scalar rule to ~1e-06 relative.
- **The scalar rules stay in the source as the oracles**, which is the repo's habit (`linearRowByRow` vs `linearBatched`, `softmax` vs `softmaxRow`): `linearScalar` and `blockResidualRead` are public, and three model-free cases pin the vectorized forms against them -- the block read at width 37 with five stack rows and three queries, the projection at 7x129 (past the threshold), and a short 3-wide projection that must take the scalar branch and still agree. A ragged stack was tried as a fourth case and dropped: the scalar rule traps on it too, so there is no defined answer to compare against; the vectorized form hands a ragged stack to the scalar rule rather than reading past a short row.
- **The existing mirrors were not re-pinned.** They passed unchanged at 1e-6, which is the statement that the reordering is smaller than the bar they were written with -- the decision not to rewrite independent numbers is recorded rather than assumed.
- **Not done in this slice:** `workspace.stage`'s KV copy (the remaining part of `attentionState`, and it grows with context), the drain residue (~0.6 ms/visit of `waitUntilCompleted` wake-up), and Phase 6 items 2-7 with item 1's arena-dependent fields, as before.

**Progress: slice 5, the post-attention convolution folded into the attention tail (2026-11-24).**

Slice 4 left `postAttentionConvolution` as 0.17 ms of a visit that was almost pure submission: one convolution dispatch -- 3 us of device work -- behind its own command buffer, commit and wait (85 us of queue, 25 of encode). It is the one remaining stage whose whole cost is hand-off, so it now rides the attention tail: `attentionHalf` takes `foldsPostAttentionConvolution` (opt-in, default false), encodes the convolution into its own encoder right after the `wo` projection, and the visit pays ONE commit for both. The ordering argument is why this is safe rather than lucky: the convolution reads `projectionOutput` (that buffer's own last write) and writes `workspace.hidden`, which nothing in the tail touches, and the convolution history for index 2 is pushed by the caller AFTER the wait -- its next reader is the next visit, which begins after this whole visit returns. `shortConvolution` became `encodePostAttentionConvolution`, encode-only.

**Measured at `-O` on the published checkpoint**, four positions after a discarded warm-up, two processes: **3.23-3.29 ms per visit and 155.3-157.8 ms per position** against slice 4's **3.46 and 165.9-166.3**, and the commit count per visit goes **6 -> 5** (960 commits per four positions). The stage that disappeared was 0.17 ms with 3 us of device; `attentionTail` absorbs its dispatch (0.33 -> 0.35) and the visit's queue bucket drops **0.50 -> 0.42**. Device total is unchanged at 1.29 ms/visit, which is the control.

- **The stage was removed from the instrument rather than left as a case that can never be recorded.** `Inkling2VisitStage.postAttentionConvolution` is deleted (nine cases -> eight), `describe` simply stops naming it, and `testTheStageTableSaysWhichStagesCommit` now pins the five committing stages and the count.
- **Parity is the whole visit against the host graph, on the folded path**: `testTheDeviceVisitMatchesTheHostGraph` passes at **6.10e-07 relative** (bar 1e-3), and the attention half alone at 9.38e-07. The fold changes no arithmetic -- it moves one dispatch into an existing encoder -- so the bar is the same one the unfurled path met.
- **What the same measurement does NOT say**: the wait residue is still ~0.5 ms/visit and `mlpTail` is still 0.18 ms of almost pure submission. Folding THAT one is a different change: its convolution reads `residualAccumulator` back to the host for the history push, so the merge needs its own ordering argument.
- **Not done in this slice:** the MLP tail's fold, `workspace.stage`'s KV copy, and Phase 6 items 2-7 with item 1's arena-dependent fields.

**Progress: slice 6, the post-MLP convolution folded into the mixture (2026-11-24).**

The other pure-submission stage: `mlpTail` was 0.18-0.20 ms a visit for one convolution dispatch (3 us of device) behind its own commit and wait. It now rides the expert mixture's encoder, encoded after the last expert's fold. The ordering argument is the same one slice 5 used, and it holds for the same reasons: the convolution reads `residualAccumulator` (the last expert's write) and writes `workspace.hidden`, which no expert touches; the history takes the PRE-convolution mixture, which is still in `residualAccumulator` because the convolution writes elsewhere; and index 3's history is next read by the next visit.

**Measured at `-O` on the published checkpoint**, four positions after a discarded warm-up, two processes: **3.16 ms per visit and 151.6 ms per position** (both processes agree to 0.1 ms) against slice 5's **3.23-3.29 and 155.3-157.8**, with commits a visit **5 -> 4** (768 per four positions) and the queue bucket **0.42 -> 0.35**. `expertMixture` absorbs the dispatch (1.04 -> 1.08) and the device total is unchanged at 1.29, which is the control.

- **Whole-visit parity is unchanged, not merely passing**: 6.098343e-07 relative, the same digits slice 5 reported, which is what a fold that moves no arithmetic should do. The MLP half alone is 4.89e-07.
- **The profile's tiling backstop was lowered, with the reading recorded.** It asserts that the stage walls cover the visit wall, and it went from >=95% to a measured **94.4%** (three isolated runs) because each fold removed a stage whose clock started after its helper had already done work, and moved read-backs and history pushes outside every stage. The per-stage assertions above it -- every stage billed, every committing stage with a window -- are what actually prove nothing stopped billing itself, so the aggregate bar is a backstop and is now 90% rather than the thing that fails first on a legitimate fold.
- **Not done in this slice:** `workspace.stage`'s KV copy, and Phase 6 items 2-7 with item 1's arena-dependent fields.

**Progress: slice 7, the KV staging copy vectorized (and three measured negatives) (2026-11-24).**

`workspace.stage` was the last scalar host loop in a visit and the only one whose cost grows with context: every retained K/V row copied element by element into the staging block the attention kernel reads. It is now one `vDSP_mmov` per row -- a contiguous row-major copy on both sides, so the bytes and their order are unchanged (bit-identical, it is a copy). Measured in isolation at the published row width (1,024 floats): **4.35-5.06x faster**, 0.249 -> 0.057 ms at a full 1,024-row window, 0.068 -> 0.014 at 256 rows.

**Measured at `-O` on the published checkpoint**, two processes: **3.13-3.15 ms per visit and 150.1-151.4 ms per position**, against slice 6's 3.16 and 151.6. That is inside this machine's process-to-process spread and is recorded as such rather than as a win: at eight positions the copy is a few microseconds, and the isolated ratio is what says the change is a win of size proportional to retained rows. The attention parity case, which is what reads the staged rows, passes at **3.39e-07 relative** per head.

Three candidates were measured and REJECTED in the same sitting, so they are not re-proposed later:

- **Four independent accumulation chains in `inkling_expert_row_f32`: 0.87x** (1.73 vs 2.00 ms at the published 2,304x2,304 shape, queue kept full). `#17`'s 2.74x came from a kernel that was a dependent FMA chain; this one is bound by the E4M3 decode arithmetic, so a second accumulator set only costs registers.
- **An E4M3 decode through a 256-entry threadgroup table: 0.97x**, bit-identical. The branch arithmetic is already cheaper than the table read.
- **`setBytes` instead of the arena's scalar `MTLBuffer`s: 6.74 -> 4.62 us per scalar in isolation**, but the mixture's ~30 scalars make that ~0.065 ms of a 3.16 ms visit -- inside the documented spread, so nothing was changed. The arena stays, and its use-after-free argument is unaffected.

What the same probes DID establish about the mixture: 36.5 MB of expert bank per visit in 0.643 ms is 57 GB/s, and a stripped-decode arm of the same kernel reaches 88-103 GB/s on the same bytes at a full queue, so ~15% of that stage is the decode arithmetic and the rest is memory -- with no form measured yet that beats the shipped one.

- **Not done in this slice:** Phase 6 items 2-7 and item 1's arena-dependent fields.

**Progress: slice 8, the host weight source caches its decoded matrices (2026-11-24).**

Slice 7's block-read change did not move `hostPrep`, so this slice followed the time instead of the idea. `Inkling2SnapshotWeights.matrix` -- what the host graph calls for every message gate (three matrices, three gates, on each of the 16 core-layer passes of a visit), every convolution tap set, and every `attn_res` query -- copied the tensor out of the shard into a `[UInt8]` and widened it, on EVERY call, for tensors that never change. The device weight source has cached its `hostMatrix` decodes since `ea16443`; the host source had no cache at all. It now keeps decoded matrices under `matrixCacheByteLimit` (96 MB, which admits every control tensor and excludes the fused expert banks and the unembedding), behind an `NSLock` for the same reason the device cache has one.

**Measured at `-O` on the published checkpoint**, two processes, agreeing: **2.70 ms per visit and 129.5-129.7 ms per position**, against slice 7's **3.13-3.15 and 150.1-151.4**. Per visit that is **-0.43 to -0.45 ms (-14%)**, and it is not a scheduling artifact: `hostPrep` 0.44 -> 0.09 and `mlpRead` 0.15 -> 0.03 are the two stages that make these calls, and every committing stage is unchanged (device 1.29 in both).

- **The block read's own change (from the same sitting) is smaller than that, and is recorded honestly**: the inverse-scalar form is 4.9-7.6x over the scalar rule in isolation -- about 1.7x over what slice 7 shipped -- but `hostPrep` did not move on it alone, because the block read is a few microseconds of a stage dominated by the weight decodes. It is kept because it is strictly less work (no normalized copy of the stack is materialized) and it is pinned like the rest.
- **Coverage.** `testAControlMatrixIsDecodedOnceAndThenServedFromMemory` overwrites the shard between two reads: a cache miss returns the new bytes, so the test cannot pass by decoding twice, and it also asserts that an oversized tensor is decoded but not kept.
- **Whole-visit parity is unchanged**: 6.0159e-07 relative against the host graph (bar 1e-3), which is what a cache of pure functions must give.
- **Not done in this slice:** Phase 6 items 2-7 and item 1's arena-dependent fields.

**Progress: slice 9, the expert mixture's in-situ gap probed (a negative) (2026-11-24).**

The mixture's device half is 36.5 MB of expert bank per visit in 0.643 ms -- 57 GB/s -- while the same kernel on resident pages reaches 88-103 GB/s, and this family's banks are the shard `mmap`, so the obvious suspect was the GPU faulting pages in while it works (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #19" measured that path at 0.065-0.2 GB/s once it interleaves with GPU work). `Inkling2MetalForward.prefaultsExpertBanks` (off in production) touches every page of each routed expert's weight and scale ranges on the host before encoding the mixture, and the measurement case runs the arms alternated (off, on, on, off) on a reset sequence each: **1.132 / 1.765 / 1.096 / 1.062 ms per visit for the mixture**. The two "on" arms are not faster, and one is much slower -- the host walk costs more than the faults it could remove -- so **the 57 GB/s is not a page-fault artefact** and the seam stays off. The position is unchanged (2.67 ms per visit, 128.3 ms), and the probe's own first cut is a lesson worth keeping: it ended with a write into the shard buffer to keep the reads alive, which is a `SIGBUS` (`KERN_PROTECTION_FAILURE`) because a shard is a read-only mapping -- the loads now accumulate into a checksum instead.

- **Not done in this slice:** Phase 6 items 2-7 and item 1's arena-dependent fields.

**Progress: slice 10, batched prefill, first step -- the schedule as data (2026-11-24).**

Phase 6 item 7 is the one structural lever left. The prefill today is a WALK in chunks of eight: `Inkling2TurnRunner.run` says so itself ("a scheduling unit and not a batching one"), and `gemm` already has a `positions` parameter nobody passes more than one to. A batched pass needs two things this family does not have: a driver that advances N positions in lockstep (the residual stack, KV cache, convolution histories and messages are all per position, so only the weight reads and dispatches can be shared), and a statement of the schedule that a second driver can consume without re-deriving it.

**Sizing first, because the decision depends on it.** An isolated A/B of `bf16_gemv_rows8` (a weight row streamed once for eight positions) against eight `bf16_gemv` dispatches, at this checkpoint's own projection shapes: `attn.wq_du` 3072x2304 **8.25x**, `attn.wk_dv` 1024x2304 5.75x, `attn.wo_ud` 2304x3072 6.30x, `mlp.gate` 128x2304 4.93x. The walk re-reads every weight eight times for a block; the block form reads it once. That is the prize, and it is why this is worth building rather than a 2% polish.

This slice lands the foundation and nothing speculative: `Inkling2Forward.schedule()` is the 48-visit schedule as a declarative list (`visit(layer:pass:)` with the two facts a caller needs around it -- a boundary append and whether cross-pass messages are built -- plus the pass restart where the loop reads the previous pass's leftovers and restores its pre-loop snapshot). `forwardHidden` is deliberately NOT rewritten to consume it; it stays the definition, and `testTheScheduleListIsTheVisitSequenceForwardHiddenRuns` pins the list against the REAL method by recording the visits its runner makes on the published checkpoint, asserting the same layers in the same order, the same pass numbers, one visit per plan slot, `numPasses - 1` restarts, and messages only on passes after the first.

An earlier attempt this slice rewrote `forwardHidden` into a switch over the list; it was reverted before any test ran. The oracle is what every parity case in this family is measured against, and a refactor of it is a change to the measuring stick -- the list is pinned to it instead of replacing it.

- **Not done in this slice:** the lockstep block driver itself (per-position state and workspace, one submission per stage per block, the token axis on the projection kernels), and the mixture's per-block expert union. Phase 6 items 2-6 remain as before.

**Progress: slice 11, the lockstep block driver, and it is the walk (2026-11-24).**

`Inkling2CPUForward.forwardBlock(tokenIds:startingAt:imageRows:onPosition:)` prefeeds a block of positions in lockstep and returns each position's logits. It consumes `schedule()`, and the order is the point: the schedule's visits run OUTERMOST and the block's positions inside each one, in position order, each appending its own KV row. That is the only order in which a position cannot attend over a later sibling's row, and it is exactly what the walk does per visit -- so the block is the walk, position by position, and `testABlockPrefillMatchesTheWalk` asserts it (five tokens, every position within 1e-5 relative of a sequential walk, and the sequence state afterwards identical). `schedule()` also grew a `.snapshot` step, because the walk snapshots its pre-loop state AFTER the pre-loop layers and before pass 0, and a driver that missed it would restart every pass from the wrong place.

**It is a correctness seam, not a speed-up yet.** The per-position state is explicit now, which is what a device driver needs to encode the block's projections into one submission per stage -- but the weight reads are still per position here, so this slice's measured result is parity, not time. The sizing for the next step is on the record: `bf16_gemv_rows8` against eight walk dispatches is 4.9-8.3x on this checkpoint's own projection shapes.

- **Two harness traps, both hit and both fixed.** Building the second forward over the same snapshot RELEASES the first engine (`engine` is one slot) and unmaps the shard buffers under the first forward -- the SIGSEGV AGENTS.md warns about, with no crash report and no message; the test keeps both engines alive now. And the driver is deeper than the walk, so it runs on `runOnLargeStack` (the project's own helper for the ~610 KB `-Onone` frame against a 512 KB worker thread) and propagates a thrown error rather than trapping.
- **The schedule logic was checked against a walk on a tiny fake forward before the real checkpoint was run**, which is what separated "my driver is wrong" from "the test's setup is wrong": the two agreed there while the real run was dying in setup.
- **Not done in this slice:** batching the projector weight reads across the block (the token axis in one buffer per stage), and the mixture's per-block expert union. Phase 6 items 2-6 remain as before.

**Progress: slice 12, the batched projection entry (2026-11-24).**

`gemm`'s body became `gemv`, which takes a resolved `(weight, weightOffset)` instead of a role -- a fused bank's expert slice has no single role -- and a `positions` count that is now enforced: eight or more contiguous position rows take `bf16_gemv_rows8` (the weight row streamed once for all of them), fewer take the one-thread-per-row form with its token axis, and MORE than eight is refused by name because that kernel reads and writes exactly eight rows and would otherwise silently run the first eight. `gemm` is a thin wrapper that resolves the role and calls it.

**Measured at the kernel**: eight positions through `gemv` are **bit-identical (0.0 relative)** to eight single-position dispatches on the same bytes, and a nine-position block throws with "one to eight positions". The graph still drives one position per visit, so this is the seam the block driver will call, not a speed-up yet -- what it enables is the 4.9-8.3x slice 10 measured on these shapes.

- **Not done in this slice:** the block's front actually dispatching batched (it needs per-position workspaces and a batched convolution, since a convolution reads the visit state's history), and the mixture's per-block expert union.

**Progress: slice 13, the attention's value fold keeps four chains (2026-11-24).**

The attention kernel is the largest device stage left that is not a weight read: `attentionFront` + `attentionTail` carry ~0.37 ms of device window per visit, one threadgroup of 32 lanes per head, each lane owning four dimensions and folding every KV row into ONE accumulator -- a dependent chain of `keyRows` adds. `Benchmarks/PERF_FINDINGS.md` "DSV4.1 #17" measured that exact shape on the DeepSeek attention and fixed it with independent accumulators, so it was tested here rather than assumed.

**Isolated A/B at the published widths** (24 heads, 8 KV heads, 128-wide heads, 1,024 rows), four arms rotated: the shipped fold **17.0-17.4 ms**; four independent accumulators in the value fold **13.4-14.0 ms (1.24-1.27x)**; four chains in the SCORE loop **18.1-18.3 ms (0.93-0.95x)**; precomputing the softmax weights once per row instead of once per row per dimension **13.9-14.0 (1.22-1.24x)**. Only the value fold is a win, and the weight precompute is the same win (the two are not additive, so the `exp` calls were never the cost).

**Measured in place** at `-O` on the published checkpoint, two processes: **127.9-128.6 ms per position and 2.67-2.68 ms per visit** against the previous 129.5-129.7 / 2.70, with `attentionFront` device 0.213 -> 0.211 and `attentionTail` 0.162 -> 0.159. The in-situ move is ~1%, much smaller than the isolated ratio, and that gap is the honest result: the attention's device window is only ~25% of the kernel's own isolated time, so most of what the A/B measured is outside the graph's accounting. It is kept because it is a strict reduction of a dependency chain, it reproduces, and parity holds: the attention kernel 3.39e-07 per head, the whole visit 5.60e-07 (bar 1e-3).

- **A duplicated KV read was tested and is NOT the lever.** Twenty-four query heads over eight KV heads read each KV half-row three times, so a head-group-per-threadgroup form (one KV read per group) was written. At a groups-of-one geometry, where its per-head reduction is a whole SIMD slice, it is **1.03x** -- and it did not reproduce the expected values, which is why it was dropped rather than debugged: a 3% ceiling does not justify chasing reduction-slice arithmetic.
- **Two more measured negatives from the same sitting**: the expert projection with four output rows per threadgroup is 1.28-1.51x isolated but **slower in place** (mixture device 0.643 -> 0.717 ms in two processes), so it was reverted; and an E4M3 lookup table is 0.97x.
- **Not done in this slice:** the block's batched front, and the mixture's per-block expert union.

**Progress: slice 14, the router's selection runs on the whole threadgroup (2026-11-24).**

`inkling_router_topk_f32` staged 128 steered ranks and then had ONE thread select: eight rounds, each scanning all 128 experts with an inner scan over the already-taken ones (about 64k comparisons), followed by two serial sorts. The ranks were already in threadgroup memory and the threadgroup was already 64 wide, so the parallelism was there and unused. Now a lane owns every 32nd expert, each round is one `simd_max` over the steered ranks, the id is resolved with `simd_min` among the lanes at the max (which is the host sort's lower-id tie rule), the warps' answers are merged by thread 0, and the weights and the ascending-id order stay where they were.

**Measured in isolation at the published shape** (128 experts, top-k 8, 512 positions): **1.135 -> 0.226 ms, 5.01x**, and the indices were identical to the single-thread form on all 4,096 slots of a fixture built to produce exact ties. **Measured in place** at `-O` on the published checkpoint, two processes: **118.05-118.49 ms per position and 2.46-2.47 ms per visit** against 127.9-128.6 / 2.67, with `mlpFront`'s device window **0.268 -> 0.084 ms** (3.2x, so most of the kernel had been the serial selection).

- **A scratch-region overlap was found by the existing kernel test and fixed.** The rank loop writes in strides of the whole threadgroup, so it touches `roundUp(numExperts, 64)` slots; packing the chosen ids straight after `numExperts` let a 64-wide fixture with eight experts overwrite them. Every region now starts on a 32-float boundary, and the test's own threadgroup-memory size was updated to match.
- **Not done in this slice:** the block's batched front, and the mixture's per-block expert union.

**Progress: slice 15, the attention-state work probed (one negative, one open) (2026-11-24).**

`attentionState` is the last host-serial stage: four KV read-backs, the cache append, the relative and sink projections, three staging copies and the KV row copy. Of those, only the read-backs and the staging copies are actually per-element work, and they are what a kernel could take.

- **A four-channels-per-thread convolution is 1.09x** (1.099 -> 1.009 ms at the published shape: 1,024 channels, 4 taps, 3 history rows, bit-identical). The tap and history accesses are already contiguous per channel, so vectorizing across channels buys almost nothing, and the convolution is a small share of a visit -- inside the noise floor in place. Rejected.
- **The KV read-backs and staging copies are what is left**, and the honest statement is that they are already one contiguous copy each (`readBack` and `staging` are vector copies since slice 7). Taking them further means keeping the cache in one flat block instead of `[[Float]]`, which would touch every reader of `keys`/`values` in the host graph and the sequence state -- a refactor of the oracle, not a kernel.

The visit is 2.46 ms and 1.29 ms of it is device time; the remaining host share is now spread thin across read-backs, the block reads and the message gates rather than concentrated anywhere.

- **The first actually-batched dispatch exists**: `Inkling2MetalForward.encodeBlockQKV` runs a block's Q/K/V projections in ONE submission, eight slots of contiguous rows through `gemv`'s `positions` form, so each projection's weight is streamed once for the whole block instead of once per position. Pinned on the published checkpoint by `testTheBatchedQKVProjectionIsEightSingleOnes`: all three projections are **bit-identical** to eight single-slot submissions through the same entry point. A block entry is exactly eight slots -- the kernel reads eight rows unconditionally, and a one-row input is an out-of-bounds read that HANGS the command buffer rather than failing (a first cut of the test did exactly that and sat at 0% CPU for 53 minutes in `waitUntilCompleted`), and two projections sharing one output buffer is silent corruption (the same cut compared Q and K against V's output and read 1.58 relative).
- **Wiring the batched projection into a whole block's attention half was attempted and stopped, with the reason measured.** The front's batched projection is right, but everything after it is per position and was not written for a shared block workspace: the K/V convolutions and the state pushes read offset 0 (fixed), the tail attention read slot 0's query (fixed), the two L2 clips and the head norm had no slot axis at all (a grid of `heads` covers one position), and -- the one that stops the slice -- `bf16_gemv_rows8` writes `row + k * outDim` for all eight slots, so projecting a block into a SINGLE-position query/key/value buffer with a slot offset overruns it eight times over and the tail then reads clobbered memory. Making that correct is a design change, not a bind: block-sized query/key/value/convolved buffers, an output offset on `convolve`, and a slot axis on every per-head kernel. The attempt was reverted rather than shipped half-correct, and the slot-0 control (every offset zero, still 0.20 relative) is the evidence that the remaining difference is in that composition rather than in slot addressing.
- **The prefill now goes through the block driver, and the block driver is what a turn runs.** `Inkling2CPUForward.forwardBlockRecording` is the block entry plus the sequence's own history recording (an image row records the marker id, exactly as the walk does), and `Inkling2TurnRunner.run` hands a plain-token chunk to it -- a chunk carrying an image row keeps the walk, because a supplied row has its own seam. `testATurnPrefillsThroughTheBlockAndMatchesTheWalk` pins it on the published checkpoint: the block turn and the walk turn draw the same token, store the same prompt positions and leave the same token history. **Measured: block 65.71 s against walk 66.57 s** for that turn -- 1.3%, i.e. noise, and recorded as such: the wiring is for what it enables (one block the device can share weight reads across, 4.9-8.3x on the projection shapes), not for a win it has not paid out yet. The test asserts no regression with a 10% band rather than a tight ratio, for the same reason.
- **A block driver is deeper than the walk, and that is a real constraint.** An unoptimized frame of the dense pipeline already exceeds a Swift-concurrency worker thread's 512 KB stack (AGENTS.md), and the block's schedule loop adds a level per position; the driver's loop is flat for that reason, and the tests that drive it run it on the project's large-stack helper.
- **The expert kernel's threadgroup width is already at its peak**, measured rather than assumed: 32 / 64 / 128 / 256 / 512 lanes at the published shape give 370.6 / 379.4 / 379.4 / 371.4 / 356.1 GB/s, so the shipped 64 is within noise of the best and no change was made. A follow-up probe that cycled through many distinct weight regions to test the cache question was **discarded**: it made 64 regions read faster than one and 16 the slowest, which no cache story explains, and a number that contradicts itself that way is a flawed probe rather than a finding.

**Progress: slice 16, the expert arena's geometry and residency policy (2026-11-24).**

Item 6 is the largest item left, and the first thing it needs is a place for experts to be resident: this family's banks are the shard mmap, so there is nothing whose misses, hits or evictions item 1 could report. `Inkling2ExpertArena.swift` adds the arena -- LRU residency over (layer, expert) pairs, caller-provided slot buffers, and the counters item 1 was missing (`hits`, `misses`, `evictions`, `bytesRead`, `preadNanoseconds`, `slotsAllocated`, a hit rate and one log line).

Two things are deliberately separate. The arena owns the BOOKKEEPING only and takes its fill from the caller, so the same policy runs under a test's in-memory fill and the executor's pread. And `Inkling2ExpertSlotGeometry` prices one expert's four contiguous ranges (`w13` weight and scale, `w2` weight and scale) from the BANK's stored strides, not the logical width -- a stride read from the logical width under-prices the slot by the padding factor, which is exactly how an arena appears to fit and then overruns.

Pinned with no model: the geometry's arithmetic (`testASlotIsTheSumOfItsFourRanges`, a zero stride refused by name) and the policy's ORDER (`testTheArenaEvictsTheLeastRecentlyUsedPair` -- A touched again before B, so a third pair evicts B, asserted step by step with its counters).

- **Not done yet in item 6:** the arena's wiring into the mixture, and the cold/warm comparison against mmap. The policy question is deliberately still open: the DSV4.1 journal's LRU-vs-frequency result came from a working set that fits, and this family's does not, so it has to be re-asked here with the arena's own counters.

**Progress: slice 17, the mixture's slot addressing (2026-11-25).**

The arena could hold an expert, but nothing could read one out of it: the mixture kernel takes a weight pointer and a scale pointer with their own row strides, and a resident slot is ONE packed buffer holding four regions. `Inkling2ExpertSlotGeometry.SlotAddress` is the answer -- each region as a byte offset from the slot's base, plus that region's own row counts and strides -- and `Inkling2SlotProjection.encode` is the binding that feeds the existing `inkling_expert_row_f32` from it (weight and scale are the same buffer; only the offsets differ from the mmap path). No new kernel was needed, which is itself the finding: the packed slot is a binding problem, not an arithmetic one.

- **The two regions do not share a row stride, and one shared stride is the bug this pins.** `w13` rows are `packedHidden` wide and `w2` rows `packedIntermediate`, so a geometry that stored one stride would read the down projection's scales off by the width ratio -- plausible numbers, no crash. `Inkling2ExpertSlotGeometry` now stores `downWeightRowStride`/`downScaleRowStride` beside the gate/up pair, and `testASlotAddressesEachRegionWithItsOwnStride` pins them apart on a shape where the ratio is exactly 2.
- **Pinned with no model:** the address arithmetic (`testASlotAddressesEachRegionWithItsOwnStride`), the lookup's exactness (`testAResidentSlotIsOnlyReturnedForItsOwnPair` -- another layer or another expert is a miss, a filled pair returns its buffer and slot), and the kernel binding against the host rule for BOTH regions and BOTH layouts (`testAResidentSlotProjectsAsTheHostRuleSays`, two slots so a base-offset bug reads the neighbour's bytes, distinct per-region strides, and scale bytes near 2^0 so no row cancels to noise).
- **Not done in this slice:** the cold/warm comparison has not been run (the arena is now reachable from the mixture, but no measurement has been taken with one installed). What is in place is the whole addressing contract that wiring needs, tested without a checkpoint.

**Progress: slice 18, the arena wired into the mixture (2026-11-25).**

The mixture now reads a resident slot when an arena is installed. `Inkling2MetalForward.installExpertArena(capacity:geometry:reader:)` attaches a slot buffer of `capacity * geometry.slotBytes` and `removeExpertArena()` takes it away; the mixture asks `residentSlot` for each routed expert, projects the slot on a hit through `slotExpertRow` (both regions, each with its own stride), and falls back to the fused bank's mmap offset on a miss. The two paths decode identically by construction: the slot is a byte copy of the same ranges, read with the same layout.

The arena's counters reach item 1's instrument now: `Inkling2VisitProfile.arena` holds `hits/misses/evictions/bytesRead/preadNanoseconds` and `describe()` appends them to the same line as the stage table. It is OPTIONAL on purpose -- nil is "no arena", which every production path is, and a 0% hit rate must not be printed where no residency policy exists.

- **Capacity is the caller's budget, and a refused capacity throws rather than shrinking**: an arena quietly smaller than requested would change the hit rate being measured.
- **Pinned with no model:** the slot's projection against the bank's own, byte for byte and then numerically, on a temp file standing in for a shard (`testASlotFilledFromABankProjectsAsTheBankDoes`), and the profile's arena fields including nil-vs-zero (`testTheArenaCountersReachTheProfileLine`).
- **Not done in this slice:** the cold/warm measurement itself (mmap vs a bounded resident set vs streaming), which needs the pinned checkpoint revision and a bounded capacity chosen from the memory plan -- the arena is off by default until that comparison says it pays.

**Progress: slice 19, the cold/warm arena comparison, and the two bugs it found (2026-11-25).**

Run at `-O` on the 16 GB M1 Pro, 256 slots (one slot 4.27 MB, 1.07 GB of arena), four alternating arms plus a correctness pair, on the published checkpoint with the revision alias in place. It did not produce a "how much does residency buy" number. It produced a different answer: **on this machine the mmap expert path is broken, and the arena is the path that works.**

- **mmap: 280-306 s per position** (mixture 2,963-3,309 ms per visit, i.e. ~10 MB/s of bank traffic) -- and the LAST arm costs what the first did, so it is not a cold-cache artifact.
- **arena: 0.72-0.79 s per position** (mixture 10.8-12.3 ms per visit), 40-41% hit rate, 0.86-0.88 GB read per position in 229-287 ms (~3 GB/s through `pread`).
- **The cause of the zeros: binding both bank segments at once.** `inkling_expert_row_f32` taking the weight from the 9.3 GB `w13` segment AND the scales from the 9.2 GB segment of the same layer returns ZERO for every row. Either buffer alone with the other operand taken from a small buffer returns the HOST-correct value, a one-row dispatch is correct, an eight-row dispatch is zero, and prefaulting every page on the host changes nothing. The two bindings total ~18.5 GB of registered memory on a 16 GB device. The failure is silent: no `commandBuffer.error`, no validation assert -- just zeros.
- **Why that was invisible until now:** the per-stage tests compare kernels against the host with small fixtures, and the mixture had never been compared against the host on a real expert. `testAPublishedBankFillsItsSlotWithTheBanksOwnBytes` now does: the slot's projection equals `Inkling2PackedExpertMatrix.project` EXACTLY (0.0 absolute error, both regions) while the mmap arm reads 0.0.
- **Two real bugs the comparison forced out first**, both silent wrong-number failures:
  - **File origins.** Segmentation makes a tensor's offset relative to its SEGMENT, which is right for a Metal buffer and wrong for `pread`. The arena was filling slots from segment-relative offsets, i.e. the wrong bytes of the file. `InferenceEngine.shardFileOrigins` recovers each segment's origin from the two summaries' base addresses, `Inkling2DeviceWeights` carries it, and `expertBank` returns file-absolute offsets beside the in-buffer ones.
  - **Mixed quantized layouts.** Layers 0...19 are NVFP4 (a 1,152-byte packed `w13` row, 144 scale columns) and layers 20...23 are MXFP8 (2,304 bytes, 72). A slot is a copy at FIXED strides, so an MXFP8 bank copied into an NVFP4 slot is truncated rows and a fabricated hit. `Inkling2ExpertSlotGeometry.holds` refuses the mismatch, `recordIncompatible` counts it, and the profile line names the refusals rather than folding them into the hit rate. The published position refuses exactly 32 experts (four layers x eight).
- **Pinned with no model:** `holds` against the two published layouts, and the staging/refusal contract from the earlier slices. **Pinned on the checkpoint:** the slot's bytes and its projection against the host rule (`testAPublishedBankFillsItsSlotWithTheBanksOwnBytes`), and one MXFP8 bank refused by an NVFP4 geometry.
- **Not done, and now the next thing** (both closed by slice 20): the arena is still not installed by a turn. Given the measurement it should be, with a capacity from the memory plan -- but that is a budget decision (1.07 GB out of ~11.5 GB), and the mmap fallback it replaces is currently *wrong* on this machine rather than merely slow, so the fallback needs a guard ("the two bank bindings exceed what this GPU can wire") before the arena becomes optional. Neither is a measurement; both are code.

**Progress: slice 20, the mixture is guarded, the arena is installed by turns, and a turn finally runs (2026-11-26).**

Slice 19 left two pieces of code undone and one silent failure in place. Both slices landed, and together they turned a graph that read zeros for 280-306 s per position into one that answers in about half a second and matches the host graph.

**The guard.** A mixture now prices what its command buffer will BIND before encoding any of it. The price is not the expert's slice: the driver wires the whole buffer, so the mmap path costs the two segment buffers the layer's fused bank and its scales live in. `Inkling2MixtureBinding` sums the DISTINCT buffers (deduplicated by identity, so eight experts over one layer's two segments are priced once) and `refuse(over:)` throws `Inkling2ForwardError.unsupported` naming every segment, the total and the budget when the total is past `recommendedMaxWorkingSetSize`. `mlpHalf` resolves all eight routings first (banks, residency, guard) so nothing is encoded for a mixture that will be refused, and a slot fill runs once per expert rather than twice.

- **The measured bindings, per layer, on the 16 GB M1 Pro** (`testWhatAMixtureBindsPerLayerOnThePublishedCheckpoint`): layer 0 **17.24 GB** (w13 bank 8.66 GB + w13 scales 8.57 GB), layer 9 **9.52 GB** (1.02 + 8.57), layer 19 **17.24 GB**, layer 23 (MXFP8) **17.24 GB** (8.57 + 8.66). Every layer binds at least 9.5 GB and most are refused, because the segments hold many layers' tensors, not one bank. **The mmap mixture is not merely slow on this machine, it is unusable** -- which is why the arena is not an optimization here.
- **The ceiling came from the decisions API, not from taste**: `recommendedMaxWorkingSetSize` scored 0.996 against physical RAM or `maxBufferLength` or a tuned constant, and refusing by name scored 0.994 against adding a silent fallback or a per-expert commit path.
- **Not done:** the same guard for the dense stages. `attentionHalf` and `headLogits` bind segment buffers too; measured, their pairs stay under the line on this checkpoint (they pass in every parity case), but nothing refuses them if a future segmentation changes that. *(Closed by slice 21.)*

**The arena, installed by a turn.** `CompletionEngine.installInklingExpertArenas` builds one arena per packed LAYOUT the checkpoint uses, because a slot is a byte copy at fixed strides and the published checkpoint's first twenty layers are NVFP4 while its last four are MXFP8. Installing only the NVFP4 geometry left exactly 32 of a position's 384 expert visits on the mmap -- which the guard now refuses, so the geometry has to cover both or the turn fails. Each arena's capacity is the smaller of one POSITION's working set (`visits x top-k` for that layout), the plan's byte share for it (proportional to the layout's visits) and what the device will allocate as one buffer.

- **Capacity is the whole story** (`testTheExpertArenaColdWarmComparisonOnThePublishedCheckpoint`, `-O`, opt-in). At the plan's own fractional budget -- 1,437 NVFP4 slots + 71 MXFP8, **6.53 GB** -- a position took **102-118 s** with the mixture at **672-857 ms/visit**. At one position's working set -- 352 + 32 slots, **1.71 GB** -- the same position took **0.57-1.47 s** with the mixture at **6.7-22.6 ms/visit**, 1.12-1.47 GB read per position and 37-60% hit. Wired arena memory displaces the mmap's own pages; the arena that pays is the set a position revisits and no more.
- **The MXFP8 arena is what made the last four layers correct.** Slice 19's arena arm covered NVFP4 only and left those four on the (broken) mmap; with both installed, visits 44-47 match the host at 2.7e-07 to 6.2e-07.
- **Fail closed on missing file origins.** `Inkling2DeviceWeights.knowsFileOrigin(of:)` is now required before an arena is installed: a caller that segmented but did not pass `shardFileOrigins` (two engine tests did exactly that) would fill slots from segment-relative offsets and report hits over plausibly wrong bytes. It surfaced as `missing weight 'expertW13 layer 1 expert 4294967295'` -- a router sentinel from the NaN cascade a wrong expert byte causes -- which is at least loud, but the guard now refuses the install and names the shard.

**Two memory bugs the turn exposed, and the turn running at last.** A turn now installs the arenas (its load log carries the plan line and the slot counts) and, for the first time on this machine, prefill and decode complete: 69 tokens then EOS, in 124-134 s including prefill, against `kIOGPUCommandBufferCallbackErrorOutOfMemory` before.

- **The plan never priced the device's KV staging.** It priced the sequence's own rows (`bytesPerPosition`) and the arena, and the executor stages K and V into two buffers of its own, so a context of 594,534 positions asked for 64 GB of rows beside **7.8 GB of staging** -- on a 16 GB machine. `kvStagingBytesPerPosition` is now part of the plan, `requiredBytes` and `affordableContextLength` both price it, and the pinned arithmetic case grew the boundary it needed.
- **`unrestricted` was taken at its 36.6 GB constant.** `KVCacheProvisioning.budgetGB` reads that mode as three quarters of THIS machine's RAM and takes the resident weights and a working reserve off the top; the Inkling turn now calls that same function, so 594,534 positions became 90,143. Neither change alone would have fixed it: the constant alone still provisions 170,211 with staging priced.
- **The turn's opening diverges from the reference at token four, and that is a documented tie, not a wrong stage.** The pinned FP32 reference continues `171960`, which the HOST head reproduces (the runner-level agent case still pins the complete, well-formed call). The DEVICE head picks `159306`, which is the token the reference's own BF16 run picks there -- and on that branch the model writes its invoke block with malformed JSON. The device head is pinned against the host at **3.8e-07 relative with the same argmax** (`testTheHeadMatchesTheHostUnembedding`), so the engine case now asserts the wiring (a turn that finishes, names the tool and the path, and parses the call when the tie goes the reference's way) and no longer asserts a coin flip.

**Pinned on the checkpoint after the change:** the whole visit **4.9e-07** (was 0.269), the MLP half **6.5e-07** with the mixture stage at **2.2e-06** (was 1.0), all 48 visits and position 0's logits **1.505e-06** (was 1.1685), and the head **3.8e-07**. **Pinned with no model:** the binding's per-buffer arithmetic and its refusal text, and the plan's staging term.

**Pinned revision, corrected.** The tests pinned `ed40e7da...`, which the repository does not publish: a normal download produces `c9d02830...` only, so every checkpoint-gated case SKIPPED and a run reported "234 tests, 49 skipped, 0 failures" while four device cases were reading zeros. The pin now names the revision that exists (decisions API: 0.999), the temporary symlink used to unhide the failures is gone, and the suites run against the snapshot a download actually yields.

**Progress: slice 21, the dense stages are guarded, the arena has a measurement seam, the KV form has a switch, and Phase 7's logit gate is measured (2026-11-26).**

Slice 20 closed the mixture's silent-zeros failure and left three follow-ups, all code. This slice does all three, and the third -- Phase 7's exit gate -- turned into a measurement with an honest negative answer.

**The dense guard.** `Inkling2MixtureBinding` now prices any command buffer rather than only a mixture: `refuse(over:subject:remedy:)` names the stage that was refused, and `Inkling2MetalForward.guardDenseBinding(_:roles:layer:)` resolves a stage's projection roles and prices them before a byte is staged. It is called per COMMAND BUFFER -- attention front (`.attnWqDu`, `.attnWkDv`, `.attnWvDv`), attention tail (`.attnGate`, `.attnWoUd`), the batched QKV prefill (`encodeBlockQKV`, the same three roles), MLP front (`.mlpNorm`, `.routerWeight`, `.routerBias`) and the head (`.lmHead`) -- because a front and a tail never share a buffer, and charging a stage for a binding it never makes would refuse a graph that runs.

- **Measured per layer on the published checkpoint** (`testWhatTheDenseStagesBindPerLayerOnThePublishedCheckpoint`): every dense stage binds exactly ONE segment, **8.57 GB** (9,206,317,056 bytes), against the **11.84 GB** ceiling, on all 24 layers and the head. So the dense path has ~3 GB of headroom: it would take a segmentation that put two of these segments in one command buffer to cross the line. The guard is defence rather than a fix, but it is the same silent-zeros failure and it now fails by name.
- **Shape came from the decisions API**: per-command-buffer pricing 0.95 (confidence 0.90), reusing the mixture type with a subject 0.80 (0.70) against a neutral rename or a second type.
- **Pinned with no model:** the subject/remedy split of the refusal (`testADenseStageBindingNamesItsStageAndItsSegments`) -- the default message still says "the expert mixture" and offers the arena, while a dense stage names itself and does not promise an arena it does not have.

**A capacity seam, and the ratio recorded.** `dynamoe_inkling2_arena_slots` (absent or 0 ships the measured capacity) replaces the working-set cap for a measurement, still held under the device ceiling, which no allocation crosses; an overridden arena says so in the install and live log lines. The rule itself moved into `Inkling2ArenaCapacity.slots(workingSet:planned:ceilingSlots:override:)` so it is a pure function with a model-free case (`testTheArenaCapacityIsTheWorkingSetUnlessAMeasurementOverridesIt`, pinning 352/32 working sets against the plan's 1,437/71 and the ceiling's precedence). **The ratio the plan keeps: 1.71 GB shipped (352 + 32 slots) against the plan's 6.53 GB fraction, which measurement says is 70-100x slower per position.**

**The KV form has a switch.** Settings > Memory & SSD shows an **Inkling 2 KV Format** row (FP32 / INT8, with a 75% KV reduction badge) only while an Inkling 2 checkpoint is loaded, bound to `dynamoe_inkling_kv_storage`; the turn already read that key, so INT8 no longer needs `defaults write`. The picker is gated on the loaded family because the key means nothing to any other engine, and `SETTINGS_GUIDE.md` documents it beside the generic KV precision it is deliberately not.

**Phase 7's exit gate, measured: the greedy token holds everywhere, and the literal Phase 2 bound does not.** `testTheInt8CacheKeepsItsGreedyTokenAndStaysInsideItsBand` runs the host graph twice over the published checkpoint -- FP32 KV and INT8 KV, same weights, same arithmetic, same tokens -- and reads five lengths off one walk per arm (the logits at position `L-1` are the `L`-token answer). Opt-in behind `/tmp/dynamoe_inkling2_int8_bound`, `-O`, ~1,070 s.

| length | logits past `5e-4 + 5e-4·|ref|` | worst abs | mean abs | p99 abs | max `diff/(1+|ref|)` | greedy (FP32 margin) |
|---|---|---|---|---|---|---|
| 1 | 129,897 / 200,058 | 0.047 | 0.0076 | 0.025 | 0.042 | agrees (2.37) |
| 2 | 179,811 / 200,058 | 0.093 | 0.0155 | 0.050 | 0.074 | agrees (0.76) |
| 5 | 192,206 / 200,058 | 0.165 | 0.031 | 0.099 | 0.145 | agrees (3.45) |
| 9 | 195,100 / 200,058 | 0.352 | 0.059 | 0.197 | 0.287 | agrees (5.27) |
| 17 | 198,105 / 200,058 | 0.878 | 0.139 | 0.446 | 0.660 | agrees (1.23) |

The greedy token is the same at every length, with the FP32 arm's own top-1/top-2 margin beside it (0.76-5.27, i.e. no near tie). The elementwise difference is real and grows with the number of quantized rows a position attends over: 0.008 mean at one row, 0.139 at seventeen, against a row whose largest logit is ~25. **The plan's `5e-4 + 5e-4·|ref|` bound is not met and cannot be met by an 8-bit cache** -- the plan already recorded that the BF16 reference itself violates it on 197k of 200k logits -- so the pinned bar is the gate's operational half: the same greedy continuation at every tested length, plus a per-length mean/p99 envelope set from this run with ~2x headroom (mean 0.02/0.04/0.08/0.15/0.35 and p99 0.06/0.12/0.25/0.5/1.1 at lengths 1/2/5/9/17), with the Phase 2 violation count, worst element and margin printed as the diagnostics the gate asked for. The host arithmetic is deterministic, so those ceilings are a fidelity bar, not a noise estimate: a dropped value scale, a row-wide scale, a wrong KV-head split or a read that dequantizes differently from the write each move the mean by far more than the headroom. **INT8 stays an explicit opt-in** (decisions API: 0.96), which is also what the FP32 default already does.

- **Assertion shape was a mid-band decision, resolved deliberately.** The decisions API preferred a lone order-of-magnitude ceiling (0.67, confidence 0.56), but a dropped V scale or a wrong head split can sit under a lone ceiling; the per-length mean/p99 ceilings catch those, and the numbers are reproducible on the host.
- **Not done in this slice:** the device path's INT8 parity at these lengths (the comparison is host-only; the device attention's INT8 branch is pinned by `testTheInt8AttentionMatchesTheHostRule` and `testAnInt8VisitStagesItsCompactRows`) *(closed by slice 22)*, and no device KV *store* kernel is dispatched -- `store_kv_cache_fp8` remains the contract's reference.

**Verified after the changes.** Model-free set: **188 tests, 17 skipped, 0 failures in 2,299 s** (+2 cases: the capacity rule and the opt-in INT8 gate). Device path with the dense guard in place: `testTheDeviceMLPHalfMatchesTheHost` 133.8 s, `testTheDeviceVisitMatchesTheHostGraph` 141.1 s, `testTheHeadMatchesTheHostUnembedding` 1.5 s, `testTheBatchedQKVProjectionIsEightSingleOnes` 3.3 s -- all green, so the guard refuses nothing the graph actually binds. The dense binding printout and the refusal-text case also pass, the latter model-free.

**Progress: slice 22, Phase 7's INT8 gate now runs on the DEVICE path as well (2026-11-26).**

Slice 21 measured the host graph with an INT8 cache and left one Phase 7 item open: the device path's parity at the same lengths. The parts were already pinned apart -- the staging byte copy, the INT8 attention kernel against a CPU reference, and the FP32 device visit against the host graph -- so this slice pins the WIRING they share.

**The case.** `testTheDeviceInt8CacheKeepsItsGreedyTokenAndStaysInsideItsBand` (in `Inkling2DeviceKernelTests`, opt-in behind `/tmp/dynamoe_inkling2_device_int8`, `-O`, **24.8 s**) walks two arms over the published checkpoint: the device graph with FP32 KV and the device graph with INT8 KV, same weights, same tokens, same five lengths. Each arm builds its own host forward (it owns the visit state, so it decides the form every visit holds), its own executor, a workspace laid out for the same form, and the resident expert arena a turn installs; `visitRunner` and `headRunner` are installed so every visit AND the unembedding run on the GPU. The walk asserts the sequence and workspace both report the chosen storage, and that the INT8 staging buffers are exactly a quarter of the FP32 ones (**262,144 B vs 65,536 B** at the walk's 32 positions).

**Decisions (API, not taste).** Gate shape: device against device (0.99, confidence 0.98) over adding a host INT8 arm to the same run or closing the item as kernel-covered. Arm order: sequential (0.88, 0.75), because the two arms cannot share a host forward and a resident arena is ~1.9 GB beside the checkpoint's shard segments, so a concurrent pair would wire twice what every single-arm case already does on a 16 GB machine.

| length | logits past the Phase 2 bound (device / host) | mean abs (device / host) | p99 abs (device / host) | worst abs (device) | greedy (device, FP32 margin) |
|---|---|---|---|---|---|
| 1 | 129,897 / 129,897 | 0.0076 / 0.0076 | 0.0252 / 0.025 | 0.0469 | agrees (220, 2.37) |
| 2 | 179,837 / 179,811 | 0.0155 / 0.0155 | 0.0503 / 0.050 | 0.0941 | agrees (149880, 0.76) |
| 5 | 191,389 / 192,206 | 0.0295 / 0.031 | 0.0958 / 0.099 | 0.191 | agrees (160581, 3.45) |
| 9 | 194,771 / 195,100 | 0.0556 / 0.059 | 0.188 / 0.197 | 0.344 | agrees (161703, 5.27) |
| 17 | 198,072 / 198,105 | 0.138 / 0.139 | 0.439 / 0.446 | 0.778 | agrees (151920, 1.23) |

**What the numbers say.** The device arm reproduces the host arm's INT8 behaviour rather than adding error of its own: the mean agrees to within 5% at every length (and is slightly BETTER at lengths 5 and 9, where the device's own reduction order rounds differently), the p99 to within 0.007, and the greedy token is identical to the FP32 device arm at all five lengths with margins of 0.76-5.27. That is the claim this case can support without an 18-minute host walk in the same run: quantization error is the quantizer's, so an arm that is materially noisier than the host's has a device defect. The pinned bar is therefore three-part, and the third part is the one that names that failure: the greedy token agrees at every length; the mean/p99 stay inside an envelope set from this run with ~2x headroom (mean 0.016/0.032/0.06/0.12/0.28, p99 0.051/0.101/0.192/0.38/0.88 at lengths 1/2/5/9/17); and the device mean stays under **1.5x the host's own measured mean** at each length. Device arithmetic is deterministic, so the ceilings are a fidelity bar rather than a noise estimate -- a dropped value scale, a row-wide scale, a wrong KV-head split or a row staged in the other form each move the mean by far more than the headroom.

- **Not done in this slice:** no device KV *store* kernel is dispatched -- the host still quantizes in `Inkling2VisitState.append`, and `store_kv_cache_fp8` remains the contract's reference. That is the last item in Phase 7's "not done" list.
- **Verified:** `build-for-testing` at `-O`, the new case green twice with identical numbers (26.1 s, then 24.8 s after the envelope was tightened), and no production code touched, so the model-free set is unchanged from slice 21's 188/17/0.

### Phase 7: Long context and DynaMoE FP8 (INT8) KV

**Work**

1. Ship BF16/FP16 correctness first, then reuse DynaMoE’s existing KV storage contract for local and global caches: `scale = max(abs(x), 1e-7) / 127`, independently per token and KV head for K and V; round and clamp signed stored values to +/-127; store scales as FP16. Read back `stored * scale`. Apply this after K convolution/clipping and V convolution/unweighted RMS normalization. This is not E4M3 and is independent of expert-activation quantization.
2. Use split-K global decode attention and tiled prefill with relative-distance bias and log scaling beyond 8,192.
3. Integrate context precision and prompt/reply budgeting with `KVCacheProvisioning.turnFit`, including image-expanded spans and DSpark only when enabled.
4. Test 1,024-ring rollover, 8,192-log-scale onset, 32K, 128K, 262K, and an opt-in larger configuration that the machine can actually provision.

**Exit gates**

- Independent INT8 store/dequantize goldens match exactly, including scale rounding and clamping. Restarts and resumed prefixes preserve identical stored bytes.
- INT8-versus-BF16/FP16 cache comparisons use the Phase 2 elementwise logit bound at every tested length and require matching greedy continuations on adequately separated prompts; near ties get margin diagnostics. Unsupported precision choices fall back explicitly, never silently. *(Run at five lengths in slice 21: the greedy token holds everywhere, the literal bound does not -- an 8-bit cache misses it on 65-99% of logits, as the BF16 reference does on 197k of 200k. The pinned bar is greedy agreement plus a measured mean/p99 envelope with margin diagnostics. Slice 22 ran the same gate on the DEVICE path: same greedy token at all five lengths, and the device's mean/p99 track the host's to within 5% and 0.007.)*
- Memory accounting matches allocated bytes, including two FP16 scales per KV head/token and prefill workspace.
- A 1M-token configuration is represented by the plan and can run only where its measured budget allows; no blanket 32 GB claim is made.
- Repeated long conversations preserve prefix correctness and show no unbounded per-turn loop-message or index-state growth.

### Phase 8: Optional bundled DSpark

**Work**

1. Add explicit sidecar loading and the three-layer draft with target capture visits 5/23/41/final48, `fuse_weight`, mask embeddings and local/global draft KV.
2. Compute `alpha = softmax(fuse_weight.float(), dim=-1).to(BF16)` independently per row. Rows 0...2 form separate normalized target mixtures for the three draft layers; row 3 forms the Markov initial-context mixture. Pin every mixture and its RMS weight.
3. Fold both `hidden_out.weight` and `hidden_out.bias` into the target unembedding as `head @ hidden_out.weight` and `head @ hidden_out.bias` in FP64 tiles before casting back. Add folded bias and base draft logits before softcapping and slicing the unpadded vocabulary.
4. Implement the draft’s own attention graph: Q/K normalization, RoPE with `rope_theta = 1,000,000`, separate target-mixture-driven context K/V projections and context K normalization, and no inherited base no-RoPE/relative-bias attention. Implement recurrent Markov steps and support the trained eight-position block with runtime depth <= 8.
5. Reuse the base executor’s capture seam without storing every intermediate state by default.
6. Implement transactional verification and rollback across every base full/local KV cache and all four conv states per visit, plus draft cache and recurrent Markov state.
7. Measure widths 2...8 under greedy and representative probabilistic sampling, text and image prompts, cold and warm arenas. Ship only widths whose emitted-tokens/s beats ordinary decode after all costs.

**Exit gates**

- Draft tokens, captures, every layer’s fused mixture, folded weight/bias, context K/V initialization, draft RoPE, Markov transitions and block logits match independent goldens.
- Accepted/rejected/truncated/EOS/cancelled rollbacks are bit-equivalent to ordinary decode state; prefill/decode resumption has no off-by-one.
- A disabled DSpark costs no sidecar memory or target capture overhead.
- Shipped speedup is reproduced over multiple runs and includes draft, verify, head, and expert-fill costs. Poor acceptance falls back to walking without user-visible failure.

**Progress: slice 1, the INT8 storage contract in Inkling 2's KV cache (2026-11-25).**

`Inkling2VisitState` now holds its KV rows in either `Inkling2KVStorage.float32` (the default, and every existing path) or `.int8`, which is DynaMoE's KV contract applied per (token, KV head) for K and V independently: `scale = max(abs(x), 1e-7) / 127`, the stored value rounded and clamped to +/-127, the scale kept as FP16, and a read back `stored * scale`. `Inkling2KVQuantization` is the rule in one place, written on the host exactly as `store_kv_cache_fp8` writes it on the device -- a divergence there would make a cache's bytes depend on which path wrote them, and a resumed prefix would then dequantize differently than the turn that produced it.

The stored rows stay private and `keys`/`values` return the dequantized FP32 rows, so every reader (attention, the profile dump, the device staging copy) is unchanged and none of them needs to know the storage. `Inkling2SequenceState.init` threads `storage` and `kvHeads` into every visit, both defaulted so nothing existing moves.

- **Pinned with no model** (`Inkling2KVStorageTests`, 9 cases): hand-computed goldens for the scale and the stored bytes (worked out from the formula, not from the implementation), the clamp on overflow, the one-step round-trip bound, the ring dropping the OLDEST row in both storages, a rollback removing the same rows an append added, a per-head scale (one head's large values must not coarsen another's small ones), and a zero row staying zero.
- **The zero row is the trap this pins**: a row of zeros gets the FLOORED scale `1e-7/127`, which rounds to zero in FP16, so the byte and the scale multiply to zero only because the byte is zero too. Any nonzero stored value in such a row would vanish -- which is why the scale is floored rather than left to `max(abs)`.
- **A decoded copy on the side is no memory saved, and the first cut had one.** The INT8 rows were stored AND their FP32 dequantization was kept beside them, which would have made the setting cost more than FP32, not less. The accessors now dequantize on demand (`stored * scale`, per head) and `residentBytes` bills the true representation, pinned by `testInt8HoldsLessThanFloat32` (64 rows of 128 values: FP32 is 65,536 B, INT8 is 18,432 B -- under a third). The device staging copy is the one place a whole window is dequantized once, because that block is transient while the cache's copy would live for every retained position.
- **The head width is a contract, not a division:** the device kernel takes a `simd_max` over one head's `headDim`, so the host scale must cover exactly `kvWidth / kvHeads`, and `headWidth` says so.
- **`Fatal access conflict detected` is what an overlapping `inout` access looks like**, not a wrong number: reading `storedKeys` from inside the closure that held `&storedKeys` trapped six of the ten cases. The width is now taken before the closure.
- **The budget prices the storage now.** `KVCacheProvisioning.turnFit` takes `kvStorage` (default FP32, so every existing caller is unchanged) and prices an INT8 turn at one byte per element plus `2 x kvHeads x 2 B` of FP16 scales per row -- so a turn that FP32 clamps by budget is not clamped in INT8. The scale bytes are why it is not a clean 4x, pinned by `testInt8StorageAffordsMoreContextOnTheSameBudget` (a fixture that asserts it actually clamps first, otherwise it would prove nothing).
- **INT8 is deliberately not a `KVCachePrecision` case**: that enum is the generic runtime's FP8 (E4M3), a different store rule with different bytes, and folding one into the other would price a cache the kernels do not write.
- **Not done in this slice:** the device store kernel for the Inkling path (the generic `store_kv_cache_fp8` is the contract's reference, not yet dispatched by Inkling's attention), and the INT8-vs-BF16 logit bound. The storage contract and its pricing are the pieces everything else reads.

**Progress: slice 2, the INT8 cache reaches the device attention (2026-11-25).**

`inkling_attention_int8_f32` is the FP32 kernel over stored rows: `device const char*` keys and values plus one FP16 scale per (row, KV head). A scale covers a whole KV head, so it comes out of the sums -- `score = (q . stored) * kScale * scaling + bias`, the value fold weighted by `softmax(row) * vScale` -- the arithmetic `gqa_attention_decode_fused_fp8` uses, and the same number `Inkling2KVQuantization.load` produces when it dequantizes per element. Everything else (relative bias, sink, threadgroup score staging, the four-accumulator fold) is deliberately a copy of the FP32 kernel.

The workspace is what spends the storage choice on memory. `Inkling2DeviceWorkspace` now carries `stagedStorage`, and its KV staging is laid out for one form: FP32 bytes, or one byte per element plus a per-(row, head) FP16 scale block beside it. `stage(_:)` copies the cache's stored bytes with `memcpy` in INT8 (the host never dequantizes to move a window), and REFUSES a visit whose storage differs from the workspace's -- reading a byte row as floats is a plausible number, not an error the GPU would report. `attention(...)` makes the same check and picks the kernel by `state.storage`.

- **`headsByVisit` is the geometry fix this needed.** A scale covers one KV head, so a sliding visit must be quantized under `swaNumKeyValueHeads`, not the full layers' count -- otherwise its scales sit over the wrong spans while the bytes still look right. `Inkling2SequenceState.init` now takes per-visit counts and `Inkling2CPUForward` derives them from the plan; the shipped checkpoint's two geometries happen to agree, which is exactly why this would have gone unnoticed.
- **The host graph read the cache per element.** `Inkling2Forward.attention` asked `state.keys[row]` inside the row loop, which in INT8 dequantizes the whole window per row per head -- a quadratic decode of the cache per position. The rows are read once for the stage now.
- **Reachable at last, and opt-in.** `Inkling2ForwardOptions.kvStorage` threads into the sequence; a turn selects it from `dynamoe_inkling_kv_storage` (`int8`, or the default `float32`), and both the cached forward and the cached device half are keyed by the storage so switching cannot reuse a sequence built for the other form. The memory plan is priced at the storage's real element width, so the INT8 choice buys the context it claims (scales are ~1.5% of an INT8 record and are not re-priced). A dedicated key rather than the generic `dynamoe_kv_cache_precision`, which is E4M3 for another engine's kernels.
- **The selector came from the decisions API, not from taste**: a dedicated preference scored 0.87 (confidence 0.81) against reusing the global KV precision; the FP32 default scored 0.71 -- mid-band -- so the default stays the reference graph's, with INT8 an explicit opt-in.
- **Pinned with no model:** the INT8 kernel against the host rule's own dequantization on a fixture where a wrong KV-head mapping, a row-wide scale, a dropped value scale, a short threadgroup or a negative distance each change the answer (`testTheInt8AttentionMatchesTheHostRule`); the staging layout byte-for-byte and scale-for-scale, the 4x buffer ratio, and the refusal of the other storage (`testAnInt8VisitStagesItsCompactRows`); per-visit head counts and the options-to-sequence plumbing (`testEachVisitQuotesItsOwnGeometrysHeadCount`, `testTheForwardOptionsChooseTheSequencesStorage`).
- **Not done in this slice:** there is no settings UI for the key yet (it is read from `UserDefaults` like `dynamoe_agent_session_floor`); the checkpoint-gated INT8-versus-FP32 device comparison and the Phase 2 elementwise logit bound have not been run; and KV rows still arrive from the host (`Inkling2VisitState.append` quantizes them), so no device *store* kernel is dispatched -- `store_kv_cache_fp8` remains the contract's reference. *(The settings row and the INT8-versus-FP32 host comparison were closed by slice 21; the device store kernel is still not dispatched.)*

### Phase 9: Release hardening and sustained dogfood

**Work**

1. Run full model-independent Swift and Rust suites, all Inkling suites, relevant server suites, and Metal validation/debug-layer runs. Use fresh `-O` derived data for real checkpoint measurements.
2. Exercise text, images, tools, API/SDK compatibility, cancellation, queueing, model switching, low-memory refusal, long resumed sessions and local agent dogfooding.
3. Update the README/project docs, settings/help, architecture labels, quantization display, conformance ledger, and `Benchmarks/PERF_FINDINGS.md` with final findings and rejected optimizations.
4. Record which reference revision and checkpoint hash was validated, and make new checkpoint/config rev changes fail closed until revalidated.

**Progress: the 16 GB M1 Pro environment (2026-11-25).**

The reference machine in AGENTS.md is a 16 GB M1 Pro, and running the checkpoint suites there surfaced two limits that are the machine's, not the port's:

- **`maxBufferLength` is 8.88 GiB there, and the checkpoint is ONE 18 GB shard.** `makeBuffer(bytesNoCopy:)` simply returns nil for a shard that large, so every tensor reads "its shard is not mapped" unless the summary is segmented first. The app's loader already does that (`InferenceEngine.segmentShardsForMetal` in `ContentView`), but the Inkling TEST helpers mapped the raw shard -- so on this machine they were exercising a mapping the app never makes. They now go through `Inkling2TestSupport.swift` (`inkling2MetalReadySummary` + `inkling2MetalShardBuffers`), and the CATALOG is built from the segmented summary too, because segmentation rewrites each tensor's shard index and offsets (a catalog from the raw summary addresses an unsplit file and resolves tensors into the wrong segment).
- **The checkpoint was NOT a different revision after all** (corrected the same day). The HF repo's model files were uploaded once, five days before the download, so there is one published model: the 20.2 GB mixed NVFP4/MXFP8 checkpoint whose README says so. The header's expert banks are named `mlp.experts.w13_weight_scale` (dotted, no expert index), which the catalog already classifies -- the naming was never the problem. The local `refs/main` (`c9d02830...`) differing from the hash the tests pin (`ed40e7da...`) is a separate, smaller question about the pin, not evidence of two models. The "38 GB on the Mac Studio" figure came from the session record, not a measurement; the published weights are 19.5 GB and that figure should be verified with `ls -l`, not `du` (the blob store here holds 185 GB of other models).
- **A fused expert bank was being split across segments, and that is why layers 0 and 9 failed.** On a 16 GB machine the 18 GB shard must be segmented, and a bank's 128 per-expert entries (the Rust loader expands every stacked `experts` tensor) could land in two different Metal buffers. The Inkling coalescer reassembles a bank from those entries and requires them in ONE shard, so the bank read as incomplete -- on the 128 GB Studio no segmentation happens and the bug cannot appear. Measured: layer 0's bank straddles segment 0's end and layer 9's straddles segment 1's, while layer 1's sits inside one, which is exactly which layers failed. `segmentShardsForMetal` now plans each bank's whole extent first and closes a segment BEFORE a bank starts rather than inside it, and refuses by name a bank larger than one buffer. Pinned by `testShardSegmentationKeepsAnExpandedExpertBankTogether` (a bank whose entries would split under the plain size rule, plus the oversized refusal).
- **What DOES run on 16 GB:** every model-free Inkling suite (134 tests, 18 skipped, green), all five `Inkling2PublishedTensorTests` (including the 57 s row-reading case), the segmentation tests, and the whole build.
- **Six device cases are SLOW at `-Onone`, not hung** (the first reading of this was wrong and is corrected here). Case-by-case bisection with a 4-minute cap timed out on `testTheDeviceAttentionMatchesTheHostGraph`, `testTheDeviceAttentionHalfMatchesTheHost`, `testTheDeviceMLPHalfMatchesTheHost`, `testEveryVisitMatchesTheHostAtThePublishedWidths`, `testTheDeviceVisitMatchesTheHostGraph` and `testAnAgentTurnFreezesACompleteCallThroughTheEngine`, while the four host-level cases passed in 5-7 s. Sampling one of the six showed ~100% CPU with RSS climbing 0.5 -> 3.9 GB over two minutes: it is computing (the documented ~10 min/position at `-Onone` cost), not waiting. A 1-hour class run at 11.6% average CPU was a mix of idle waits and these runs, and its 4.5 MB RSS reading came from a process that had already exited. The open item is a real time budget for these cases (an hour each, in isolation), not a deadlock hunt.
- **Two 9+ GB bank bindings in one dispatch return ZEROS on this machine** (2026-11-25). The device mixture binds one layer's `w13` weight segment (9.3 GB) and its scale segment (9.2 GB) in the same `inkling_expert_row_f32` dispatch; together that is ~18.5 GB of registered memory on a 16 GB device. Every row comes back 0.0. Either buffer alone (the other operand taken from a small buffer) is HOST-exact, a one-row dispatch is correct, an eight-row dispatch is zero, and prefaulting every page on the host changes nothing. Metal reports no error and no validation failure. The expert ARENA does not have the problem: its slot is 4.27 MB and its projection matches `Inkling2PackedExpertMatrix.project` with 0.0 absolute error. Until the mixture is guarded or the arena is installed by turns, the Inkling device path on a 16 GB machine emits a model whose MoE layers contribute nothing -- and takes 280-306 s per position doing it (Phase 6 slice 19). The ALREADY-WRITTEN parity tests say exactly this once the revision alias lets them run: `testTheDeviceMLPHalfMatchesTheHost` reports the folded mixture accumulator 1.0 relative off (the device side is zero), `testTheDeviceVisitMatchesTheHostGraph` 0.27, and `testEveryVisitMatchesTheHostAtThePublishedWidths` puts position 0's logits 1.17 off; an agent turn over the engine dies with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Those four were green-by-skipping before, because the pinned revision differs from the local one -- the alias is what makes the suite honest.

- **The turn path is where the memory bugs lived** (2026-11-26, slice 20). Three failures sat behind each other and each hid the next: the mixture's two-segment binding (zeros), the plan's unpriced device KV staging (an OutOfMemory on a 594,534-position provision), and the `unrestricted` budget's 36.6 GB constant on a 16 GB machine. Only the first was visible from the kernel tests; the other two needed a TURN, and the turn could not get far enough to reach them until the arenas were installed. The lesson for the remaining phases: a device path is not measured until the app's own turn runs it, at the budget the app would really ask for.
- **A test that skips is not a test that passes.** The revision pin (`ed40e7da`, unpublished) turned four failing device cases into silent skips; moving the pin to the revision the repository actually publishes (`c9d02830`) is what made the suite honest. Any future pin must name a path a download produces.
- **A silent-zero failure costs more than a loud one.** The two-bank binding returned 0.0 for every row with no Metal error, and the arena read segment-relative bytes as file offsets with no error either. Both are now refused or validated by name, and the tests that caught them compare against the HOST rule rather than against a second GPU path -- an arena-vs-mmap assertion would have been asserting that one broken arm reproduced the other.

**Exit gates**

- All required suites pass with evidence of executed test cases, not zero-test `TEST SUCCEEDED` output.
- Metal validation finds no binding/threadgroup violations.
- Live turns and API results are reproducible and supported on the stated hardware/memory envelope.
- Documentation lists supported modalities, native quant formats, optional DSpark state, context limits, known performance costs, and explicit non-goals.

## 6. Verification commands

Use fully qualified destinations and three-part test identifiers. On this machine prefix with `DEVELOPER_DIR` if `xcode-select` points to Command Line Tools. Always inspect per-case passed lines.

```bash
cd core && cargo test

cd DynaMoE
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild test \
  -project DynaMoE.xcodeproj \
  -scheme DynaMoE \
  -destination 'platform=macOS,arch=arm64' \
  -parallel-testing-enabled NO \
  -only-testing:DynaMoETests/Inkling2LogicTests
```

For a single case:

```bash
-only-testing:DynaMoETests/Inkling2PublishedTurnTests/testAPublishedTextTurnMatchesReference
```

Verify test freshness before trusting the run:

```bash
strings /path/to/DerivedData/Build/Products/Debug/DynaMoETests.xctest/Contents/MacOS/DynaMoETests \
  | grep -c testAPublishedTextTurnMatchesReference
```

Use the same derived-data path for `build-for-testing` and `test-without-building`. Keep real-checkpoint measurements in a separate `-O` derived-data directory so normal Debug products are untouched. Run with Metal validation before binding/layout work is declared complete.

## 7. Test matrix

| Layer | Fast/default tests | Opt-in real-checkpoint and measurement tests |
|---|---|---|
| Config/catalog | Short defaults, every role/companion, malformed files, query unpacking | Published JSON and tensor header parity |
| Quantization | Exhaustive small-value rounding, candidate ties, partial groups, odd routes | Native layer outputs against reference |
| Graph | Synthetic all visits, residual blocks, cross-pass gates, chunk/ring boundaries | Stage logits, greedy continuations, long prompts |
| State | Same commit count everywhere, prefix/delta/cancel/rollback | Follow-up and long resumed conversations |
| Prompt/tools | Exact string/codepoint goldens, escaping, truncation, replay | Model-generated calls and multi-step agent turns |
| Vision | Pixels/alpha/tiles/grid, tower fixtures, span pricing/order | Red/color/text images, multiple images, large-image pricing |
| API | Both protocols, stream/non-stream, usage/refusals/log retention | Real HTTP image/tool turns and queue behavior |
| Performance | Synthetic shapes plus parity instrumentation | Cold/warm published-width stage and whole-turn benchmarks |
| DSpark | Draft/verify/rollback goldens and scripted turn tests | Width/acceptance/cost matrix |

## 8. Highest-risk invariants

1. Never allow generic transformer execution as a fallback for this family.
2. Weights are shared across core passes; KV and all four convolution states are not.
3. Preserve exact block residual behavior at 0/6/12/18 and restart from the same post-layer-5 state on each core pass.
4. Relative attention uses chronological distance and no ordinary RoPE. Local rings must read chronologically.
5. Keep HF dense-activation and trained quantized-activation arithmetic as distinct named profiles. Preserve scale/auxiliary layouts, interleaved gate/up rows, and reference-precision routing/mixtures/final logits in both.
6. Process bounded chunks through the entire visit schedule; never retain loop messages for the whole prompt by default.
7. Price image-expanded positions before provisioning. Inkling uses 40-pixel cells, no DeepSeek cap, ViT subpatches, and RGBA checkerboard semantics.
8. Prompt rendering, stream freeze, canonical replay, tools, server stops, and prefix identity are one dialect contract.
9. Executor-owned history proves prefix reuse. A session key or matching id prefix alone is insufficient.
10. Every optimization needs independent parity plus an in-situ measured win. Reuse kernel forms by measurement, not by family resemblance.
11. DSpark rollback covers KV, convolution, Markov, and exact committed token counts before it can be enabled.
12. New tensor roles, quant declarations, loop layouts, dtypes, or processor behavior fail closed until revalidated.

## 9. Initial implementation order for the next session

1. Implement Phase 0’s reference/golden tooling and synthetic checkpoint.
2. Land identity, strict config, tensor catalog, preflight, and primitive tests.
3. Implement the host reference graph and validate all stage goldens.
4. Build native packed expert GEMV, then the single-position Metal graph.
5. Compare one whole text turn on the published checkpoint before adding any prefix, image, or speculative optimization.
6. Continue through the phase exit gates rather than landing isolated kernels without their graph and state contracts.

## 10. Primary references used for this analysis

- Local checkpoint `config.json`, `hf_quant_config.json`, `processor_config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, SafeTensors headers, and `chat_template.jinja`.
- Model card: <https://huggingface.co/thinkingmachines/Inkling-2-Mini>
- Transformers 5.24.0 `models/inkling2_loop/modeling_inkling2_loop.py` and `configuration_inkling2_loop.py`, source revision above, plus its checkpoint conversion and shared Inkling 2 vision implementation.
- vLLM Inkling native `nvfp4.py` and `mxfp8.py` activation quantization implementations and recipe: <https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini>
- SGLang Inkling 2 DSpark implementation at revision `af80e8ce0dcf59197814fe216a7bf3976dbc1e07`, including its target taps, verifier, and Markov kernels.
- MLX Vision/language/DSpark sources from `mlx-vlm` 0.8.0, including native packed-expert kernels and DSpark folding/rollback patterns. Treat any W4A16 runtime mode as a separate numeric contract.
- Hugging Face release/inference notes: <https://hf.co/blog/inkling2mini>
