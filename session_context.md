# DynaMoE Session Context: Qwen 3.8 Flash Next Support

This document summarizes the architecture, kernel implementations, streaming pipeline optimizations, and stability fixes implemented to support **Qwen 3.8 Flash Next** (and related Hybrid SSM / Ornith MoE architectures) in DynaMoE.

---

## 1. Architecture Overview & Model Specifications

**Qwen 3.8 Flash Next** (`qwen4_exp` / `qwen3_8`) is a hybrid sparse Mixture-of-Experts (MoE) architecture combining Gated DeltaNet (GDN) linear attention layers, Grouped-Query Attention (GQA) full attention layers, 4-stream HyperConnection residuals, and high-capacity routed experts.

| Parameter | Specification |
| :--- | :--- |
| **Total Layers** | 48 Layers |
| **Attention Pattern** | Hybrid: 3 GDN Linear Attention layers followed by 1 GQA Full Attention layer (e.g. Layers 0–2: GDN, Layer 3: Full Attention, ..., Layer 12: GDN) |
| **Hidden Dimension** | 2560 |
| **Stream Representation** | 4 parallel streams ($4 \times 2560 = 10,240$ floats per token state) |
| **HyperConnection Low-Rank** | Rank 320 bottleneck projection |
| **GDN Linear Attention** | 48 Value Heads, 16 Key Heads, Head Dimension 128 |
| **GDN Projections** | • $QKV$ Projection Dim: $(16 + 16 + 48) \times 128 = 10,240$<br>• $Z$ Gate Dim: $48 \times 128 = 6,144$ (activated via **Sigmoid**)<br>• $A$ (Decay) Dim: 48, $B$ (Learning Rate) Dim: 48 |
| **Full Attention (GQA)** | Standard GQA with RoPE rotary embeddings, supporting FP16 and FP8 KV Caching |
| **MoE Layer Configuration** | 512 Routed Experts per layer (~3.15 MB per expert in FP8/BF16), Top-10 routing per token + Shared Experts |
| **Intermediate Dimension** | 640 per expert (Max Intermediate Dim: 2560) |
| **Norm Centering** | Unit offset RMSNorm ($1.0 + \gamma$) across all layers and streams |

---

## 2. HyperConnection Multi-Stream Architecture & Metal Kernels

Qwen 3.8 Flash Next replaces standard additive residual connections with a **4-Stream HyperConnection (HC) Low-Rank Mixing & Injection Pipeline**.

```
[4 Streams: 10,240]
       │
       ▼
 1. hcNormPipeline (4 x RMSNorm with (1.0 + γ))
       │
       ├───────────────────────────────────────┬───────────────────────────────────────┐
       ▼                                       ▼                                       ▼
 2. hcDownProjPipeline                   3. hcUpBlendPipeline                   4. hcInjectScalePipeline
    (10,240 -> 320, * 1/4, SiLU)            (320 -> 10,240 sigmoid gates,          (10,240 -> 4 scales,
                                            blend & avg into 2560 sublayer input)  inject[s] = 2.0 * σ(1/4 * W_inj * normed))
                                               │
                                               ▼
                                     [Sublayer: GDN / Attn / MLP]
                                               │
                                               ▼
                                         [Sublayer Out]
                                               │
                                               ▼
                                         5. hcInjectPipeline
                                            (streams[s] += inject[s] * sublayer_out)
```

### Compute Shader Implementations (`ComputeShaders.metal`):
1. **`hyper_connection_norm_bf16`**:
   - Computes independent RMSNorm across all 4 streams with unit offset centering:
     $$\text{rms}_s = \sqrt{\frac{1}{2560} \sum_{d=0}^{2559} (X[s, d])^2 + \epsilon}, \quad \text{normed}[s, d] = \frac{X[s, d]}{\text{rms}_s} \times (1.0 + W_{\text{hc\_norm}}[s, d])$$
2. **`hyper_connection_down_proj_bf16`**:
   - Projects 10,240 normalized stream elements to a 320 rank bottleneck with $\frac{1}{4}$ scaling and $\text{SiLU}$ activation:
     $$z_r = \text{silu}\left( \frac{1}{4} \sum_{i=0}^{10239} W_{\text{down}}[r, i] \cdot \text{normed}[i] \right)$$
3. **`hyper_connection_up_proj_blend_bf16`**:
   - Computes stream gating and blends normalized streams into sublayer input:
     $$g_{s, d} = \sigma\left( \sum_{r=0}^{319} W_{\text{up}}[s \times 2560 + d, r] \cdot z_r \right), \quad \text{SublayerInput}[d] = \frac{1}{4} \sum_{s=0}^3 g_{s, d} \cdot \text{normed}[s, d]$$
4. **`hyper_connection_inject_scale_bf16`**:
   - Computes dynamic per-stream injection coefficients:
     $$\text{inject}[s] = 2.0 \times \sigma\left( \frac{1}{4} \sum_{i=0}^{10239} W_{\text{inject}}[s, i] \cdot \text{normed}[i] \right)$$
5. **`hyper_connection_inject_bf16`**:
   - Residual injection updating 4 stream states:
     $$X_{\text{streams}}[s, d] \leftarrow X_{\text{streams}}[s, d] + \text{inject}[s] \cdot y[d]$$
6. **Final Layer Mixer**:
   - At the final layer, steps 1–3 mix the 4-stream state into a 2560 hidden vector (`xFinalBuffer`) passed directly to `lm_head`.

---

## 3. Gated DeltaNet (GDN) Mathematical Formulations

- **Output Gate Correction**: Changed output gate activation from $\text{SiLU}(z)$ to $\text{Sigmoid}(z)$ in `linear_attention_recurrent_sequence` and `gdn_linear_attention_recurrent_step`.
  $$\delta v_i = \beta \cdot (v_i - S_i \cdot k)$$
  $$S_i^{new} = \alpha \cdot S_i^{old} + \delta v_i \cdot k^T$$
  $$y_i = \frac{S_i^{new} \cdot q}{\sqrt{d_{head}}}$$
  $$y_{norm} = \text{RMSNorm}(y) \odot \gamma \odot \mathbf{\sigma(z)}$$
- **L2 QK Normalization**: Head-by-head $Q$ and $K$ normalizations with dynamic head stride calculation.
- **Causal Depthwise Conv1D**: Depthwise 1D convolutions with persistent cache carryover across decode steps.

---

## 4. Prefill IO Optimization & Layer-Wise Execution

### Batched Layer-Wise Prefill (`runLayerWisePrefill`):
- **O(Layers) IO Scaling**: Processes all prompt tokens ($P$) simultaneously layer by layer.
  - Active experts for the entire prompt sequence are read from disk **once per layer** into a single pre-allocated `prefillStagingBuffer`.
  - Recurrent GDN states are updated sequentially in GPU SRAM in $< 0.1\text{ms}$.
  - Prompt ingestion speed reduced from several minutes to **$< 1$ second per layer** (~45 seconds total for 48 layers).
- **Batched HyperConnection Execution**:
  - `hcNormedBuffer_all` and `hcInjectScaleBuffer_all` allocated for $P$ tokens.
  - All HC stages (Norm, Down, Up-Blend, InjectScale, Inject) dispatched with batch height $P$.
- **Real-Time Progress & Throughput Reporting**:
  - Reports effective prompt tokens per second and dynamic ETA:
    $$\text{Layer } \ell / 48 \text{ (pct\%)} \bullet X\text{ tok/s} \bullet \text{ETA: } Y\text{s}$$

---

## 5. Key Bug Fixes & Stability Improvements

| Issue | Root Cause | Fix Implemented |
| :--- | :--- | :--- |
| **Degenerate "!!!" Thinking Output** | GDN used `silu(z)` instead of `sigmoid(z)`, and HyperConnection was missing the 4-stream RMSNorm, $(1.0+\gamma)$ centering, $\frac{1}{4}$ scaling, and dynamic `inject_scale` computation. | Corrected GDN output gate to `sigmoid(z)` and implemented the full 5-stage HyperConnection pipeline in `ComputeShaders.metal`, `ContentView.swift`, and `InferenceEngine.swift`. |
| **512-Router Shared Memory Corruption** | `moe_router_topk_512_bf16` processes 2 logits per thread (`tid * 2` and `tid * 2 + 1`). Dispatching 512 threads exceeded `sharedLogits[512]`. | Changed `threadsPerThreadgroup` dispatch size from 512 to 256 threads (`MTLSize(width: 256, height: 1, depth: 1)`). |
| **RMSNorm Unit Offset Model Tagging** | `isRMSNormUnitOffset` in `ModelConfig.swift` lacked model family identifiers for Qwen 3.8 / Next. | Added `qwen3_8`, `qwen3.8`, `qwen4`, `qwen4_exp`, `qwen3_next` to `isRMSNormUnitOffset`. |
| **`0_os_unfair_lock_recursive_abort` Crash** | Non-recursive `NSLock` instances on macOS trigger kernel aborts when re-entered during layer transitions or handle cleanup. | Replaced all `NSLock` instances across the codebase (`ExpertIOThreadPool`, `ExpertTransitionTracker`, `AgentHarness`, and UniFFI Rust bridge) with `NSRecursiveLock`. |
| **MoE Token Assignment Race Condition** | Shared CPU buffer pointers were overwritten in a loop while GPU command encoding was async. | Switched to per-dispatch `layerEnc.setBytes` embedded directly in the Metal command stream. |

---

## 6. Verification Status

- **Xcode Build Status**: Clean build verified with zero errors (`** BUILD SUCCEEDED **`).
- **Target Scheme**: `DynaMoE` (`macOS`, `arm64`).
- **Automated Tests Passed**:
  - `DynaMoETests.testQwen38FlashNextForward`: Passed cleanly across all 48 layers with 2D block-scaled FP8 kernels.
  - `DynaMoETests.testQwen38FlashNextPrefillLargePrompt`: Passed cleanly for 1,187-token prompt batch without Metal assertion failures.
- **Core Files Modified**:
  - `DynaMoE/ComputeShaders.metal`
  - `DynaMoE/DynaMoE/ContentView.swift`
  - `DynaMoE/DynaMoE/InferenceEngine.swift`
  - `DynaMoE/DynaMoE/ModelConfig.swift`
  - `DynaMoE/DynaMoE/ExpertIOThreadPool.swift`
  - `DynaMoE/DynaMoE/AgentHarness.swift`
  - `DynaMoE/DynaMoETests/DynaMoETests.swift`
