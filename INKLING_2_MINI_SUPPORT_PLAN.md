# Inkling 2 Mini: Full Native Inference Support Plan

**Status:** architecture approved for implementation; no Inkling production code has landed yet  
**Checkpoint under analysis:** `thinkingmachines/Inkling-2-Mini`  
**Pinned checkpoint revision:** `ed40e7da625f594fc9c4990f5f66be24873c22d7`  
**Local snapshot:** `~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7`  
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
- Both server routes (`testBothProtocolsAnswerTextTurnsOverTheLocalServer`, opt-in behind `/tmp/dynamoe_inkling2_server_routes`): a text turn over `/v1/chat/completions` (non-streaming) and over `/v1/messages` (streaming) on a private port. The OpenAI body is a `chat.completion` with a terminal `finish_reason` and a usage block (182 prompt / 2 generated); the Anthropic stream opens with `message_start`, keeps the connection warm with named `ping` frames, and ends `content_block_start` / deltas / `content_block_stop` / `message_delta` / `message_stop` with no `error` frame. The log rows are one per route, the streaming one marked `streaming=true`. The generated text is not asserted: the answer is degenerate at a two-token budget and a wording change is not a regression.
- The app's own turn path (`testTheEngineAnswersTheSameOnTheHostAndOnTheDevice`, opt-in behind `/tmp/dynamoe_inkling2_engine_turn`): `CompletionEngine.launchTurn` with the checkpoint loaded as the app loads it, run twice in ONE process -- the host graph forced by the `dynamoe_inkling2_device_off` debug seam, then the device visit runner -- and both arms return the same two tokens. **Host 427.4 s, device 14.5 s** for the same turn (29x; two tokens, so a direction rather than a rate). The runner is installed inside the turn task and cleared when it ends, and the device half is retired with the host forward on `configure`.

Five defects the stage-by-stage comparison found and no end-to-end number would have named: the post-MLP convolution read the ATTENTION's taps; the router staged its ranks with `tid < numExperts` under a 64-wide threadgroup, leaving half of 128 experts uninitialized; the router read its bias from offset zero of a tensor that lives partway into its shard; the expert-row kernel had no bank offset, so only expert 0 of layer 0 could have been read right; and the norm kernel read its gain as FP32 where the checkpoint stores BF16. Two state bugs came with them: the attention half appended its KV row and pushed its convolution histories BEFORE the command buffer ran (exact for the first rows, wrong from the point a convolution reads them), and the score dump had every head's threadgroup 0 writing the same slots.

### Phase 3: Stateful prefill, prefix reuse, and text server integration

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

### Phase 7: Long context and DynaMoE FP8 (INT8) KV

**Work**

1. Ship BF16/FP16 correctness first, then reuse DynaMoE’s existing KV storage contract for local and global caches: `scale = max(abs(x), 1e-7) / 127`, independently per token and KV head for K and V; round and clamp signed stored values to +/-127; store scales as FP16. Read back `stored * scale`. Apply this after K convolution/clipping and V convolution/unweighted RMS normalization. This is not E4M3 and is independent of expert-activation quantization.
2. Use split-K global decode attention and tiled prefill with relative-distance bias and log scaling beyond 8,192.
3. Integrate context precision and prompt/reply budgeting with `KVCacheProvisioning.turnFit`, including image-expanded spans and DSpark only when enabled.
4. Test 1,024-ring rollover, 8,192-log-scale onset, 32K, 128K, 262K, and an opt-in larger configuration that the machine can actually provision.

**Exit gates**

- Independent INT8 store/dequantize goldens match exactly, including scale rounding and clamping. Restarts and resumed prefixes preserve identical stored bytes.
- INT8-versus-BF16/FP16 cache comparisons use the Phase 2 elementwise logit bound at every tested length and require matching greedy continuations on adequately separated prompts; near ties get margin diagnostics. Unsupported precision choices fall back explicitly, never silently.
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

### Phase 9: Release hardening and sustained dogfood

**Work**

1. Run full model-independent Swift and Rust suites, all Inkling suites, relevant server suites, and Metal validation/debug-layer runs. Use fresh `-O` derived data for real checkpoint measurements.
2. Exercise text, images, tools, API/SDK compatibility, cancellation, queueing, model switching, low-memory refusal, long resumed sessions and local agent dogfooding.
3. Update the README/project docs, settings/help, architecture labels, quantization display, conformance ledger, and `Benchmarks/PERF_FINDINGS.md` with final findings and rejected optimizations.
4. Record which reference revision and checkpoint hash was validated, and make new checkpoint/config rev changes fail closed until revalidated.

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
