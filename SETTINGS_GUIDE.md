# DynaMoE Settings & User Guide

A comprehensive, practical guide to configuring, tuning, and operating DynaMoE on Apple Silicon. This document details every section of the **Settings** window (`⌘,`) — which presents a sidebar of sections on the left and their controls on the right, matching the macOS System Settings layout — explains the underlying mathematics and runtime mechanics, provides battle-tested presets for coding versus conversational assistance, and offers memory budgeting advice for Mac hardware configurations.

---

## Table of Contents
1. [Overview & Accessing Settings](#1-overview--accessing-settings)
2. [General: Updates & Chat History](#2-general-updates--chat-history)
3. [Models & Binary Repackaging](#3-models--binary-repackaging)
   - [Local Model Discovery](#local-model-discovery)
   - [Default & Last Used Model](#default--last-used-model)
   - [Apple Foundation Model (On-Device System Model)](#apple-foundation-model-on-device-system-model)
   - [Conversation Compaction (`/compact`)](#conversation-compaction-compact)
   - [MoE Contiguous Binary Repackaging (Flash-MoE)](#moe-contiguous-binary-repackaging-flash-moe)
   - [Dense Weight Quantization (Q4 / Q8)](#dense-weight-quantization-q4--q8)
   - [Model-Specific Profiles ("Coder" & "Assistant")](#model-specific-profiles-coder--assistant)
4. [Generation & Sampling Hyperparameters](#4-generation--sampling-hyperparameters)
   - [Active Profile Banner & Fast Switching](#active-profile-banner--fast-switching)
   - [Sampling Parameters (Math & Practical Tuning)](#sampling-parameters-math--practical-tuning)
   - [System Prompts & Conjunction Merging](#system-prompts--conjunction-merging)
   - [JetSpec Speculative Tree Acceleration](#jetspec-speculative-tree-acceleration)
5. [Memory & SSD Management](#5-memory--ssd-management)
   - [Memory Execution Modes (Auto, Full RAM, SSD Streaming)](#memory-execution-modes)
   - [Memory Budget Modes & Limits](#memory-budget-modes--limits)
   - [KV Cache Precision (FP32, FP16, FP8 E4M3/E5M2)](#kv-cache-precision)
   - [Speculative MoE Lookahead Prefetching](#speculative-moe-lookahead-prefetching)
   - [Live Diagnostics & Working Set Metrics](#live-diagnostics--working-set-metrics)
   - [Cache Maintenance: Flush vs. Pre-Fault](#cache-maintenance-flush-vs-pre-fault)
6. [Agent & Tools](#6-agent--tools)
   - [Global Agent Enablement & Sandbox Directory](#global-agent-enablement--sandbox-directory)
   - [Safety Limits (Max Output & Max Steps)](#safety-limits)
   - [Search Integrations (Headless Chrome & Brave Search)](#search-integrations)
   - [Built-In Local Tools Reference](#built-in-local-tools-reference)
   - [Model Context Protocol (MCP) Servers](#model-context-protocol-mcp-servers)
7. [Local Inference Server](#7-local-inference-server)
   - [Enabling the Server](#enabling-the-server)
   - [Live Status](#live-status)
   - [Endpoints & Authentication](#endpoints--authentication)
   - [Reasoning in Content](#reasoning-in-content)
   - [Queueing & Guardrails](#queueing--guardrails)
   - [Client Setup](#client-setup)
8. [Cloud Providers](#8-cloud-providers)
   - [API Key & Connection Test](#api-key--connection-test)
   - [Registering Cloud Models](#registering-cloud-models)
   - [Image Input (Vision-Capable Cloud Models)](#image-input-vision-capable-cloud-models)
   - [What a Cloud Turn Supports](#what-a-cloud-turn-supports)
9. [Hardware Profiles & Recommended Configurations](#9-hardware-profiles--recommended-configurations)
   - [16 GB Unified RAM (M1/M2/M3/M4)](#16-gb-unified-ram-m1m2m3m4)
   - [24 GB – 36 GB Unified RAM](#24-gb--36-gb-unified-ram)
   - [64 GB – 128 GB+ Unified RAM (M-Max / M-Ultra)](#64-gb--128-gb-unified-ram-m-max--m-ultra)
10. [Frequently Asked Questions & Troubleshooting](#10-frequently-asked-questions--troubleshooting)

---

## 1. Overview & Accessing Settings

DynaMoE provides centralized control over model inference, GPU shader dispatch, unified memory allocation, and agent autonomy.

### How to Open Settings
- **Keyboard Shortcut**: Press `⌘,` (Command + Comma) from any window.
- **Application Menu**: Navigate to **DynaMoE $\to$ Settings…** in the top macOS menu bar.
- **Chat Header**: Click the **Settings** slider icon in the upper right toolbar of the chat interface.

### How to Open this Guide in the App
- **Help Menu**: Select **Help $\to$ DynaMoE Help & Settings Guide** (or press `⌘?`).
- **Settings Window**: Click the **Help & Guide** button located in the top-right of the Settings window. Sections are selected from the sidebar on the left.

---

## 2. General: Updates & Chat History

The **General** section covers app-wide housekeeping: software updates, and how conversations and reasoning displays behave.

### Updates
- **Enable Auto-Updates**: DynaMoE checks the Sparkle update feed in the background; new releases are verified and installed safely. You can always check manually with `⌘U`.
- **Check for Updates Now**: Queries the update feed on demand, regardless of the automatic setting.
- **Current Version**: The DynaMoE release and build currently installed.

### Chat & Reasoning
- **Expand Thinking by Default**: Show the thinking/reasoning accordion expanded when a message first appears. When off, reasoning starts collapsed and can be opened manually.
- **Auto-Delete Old Conversations**: Conversations are kept between launches, with the oldest removed automatically once the limit below is reached. Turn this off to keep every conversation forever.
- **Conversations to Keep**: How many of the most-recently-updated conversations are retained (1–100). The conversation you have open is always kept.

---

## 3. Models & Binary Repackaging

The **Models** section acts as your local model registry and management hub.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Models Section                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 📦 ornith-ai/Ornith-1.5-9B                  [Active] [⋯] [Loaded] │  │
│  │    Architecture: Hybrid GDN + GQA | Precision: BF16 | 19.3 GB    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 📦 Qwen/Qwen3.8-Flash-Next-FP8              [Default] [⋯]  [Load] │  │
│  │    Architecture: MoE 512 Experts | Precision: FP8 | 48.2 GB      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Local Model Discovery
Upon startup, DynaMoE automatically scans your Hugging Face cache directory (`~/.cache/huggingface/hub`).
- Every discovered model snapshot is inspected for `config.json`, SafeTensors index files (`model.safetensors.index.json`), and tokenizer configs (`tokenizer.json`, `tokenizer_config.json`).
- DynaMoE displays each model's detected architecture (e.g. *Hybrid GatedDeltaNet + GQA*, *MoE 512 Experts*, *Dense Transformer*), parameter count, total disk footprint in GB, and quantization type.

### Default & Last Used Model
- **Make Default**: Setting a model as the default ensures it is selected automatically when DynaMoE launches fresh.
- **Automatic Fallback**: If no default model is specified, DynaMoE restores the **Last Used Model** across app restarts.

### Apple Foundation Model (On-Device System Model)

On macOS 26 or later with Apple Intelligence enabled, DynaMoE surfaces Apple's built-in on-device Foundation Model as a first-class chat backend alongside your local weights models:

- **Zero-Install**: it appears as *Apple Foundation Model (On-Device)* under **System Models** in the chat model picker and in the Models settings list. Selecting it takes effect instantly — nothing is loaded from disk, so any resident weights engine is released and the working-set memory budget is freed.
- **Availability States**: if Apple Intelligence is turned off, still downloading its model, or the Mac is ineligible, selecting the model or sending a message surfaces the exact remediation in the chat status area.
- **Profiles Still Apply**: the Coder/Assistant profile's system prompt and temperature drive the turn (temperature is clamped to the on-device model's 0–1 range; repetition/presence penalties and Top-P/Min-P/Top-K have no system-model equivalent and are ignored).
- **Plain Chat Only**: reasoning/thinking blocks, agent tools, JetSpec, and KV/prefix-cache acceleration are local-weights features; the system model runs as a direct conversationalist. Each turn carries as much conversation history as the system model's small context window can spare after the profile's persona and response budget — oldest turns drop first.

> [!NOTE]
> Token counts and tok/s shown for Apple Foundation Model turns are character-based approximations; the system API does not report per-chunk token usage.

### Conversation Compaction (`/compact`)

Typing `/compact` in the composer — on **any** backend, with optional extra emphasis such as `/compact keep the pread-streaming decisions` — manually summarizes and compacts the conversation using the Apple Foundation Model:

- The entire conversation is folded into two digests that persist with the session and are injected into every later prompt on both chat backends: a rolling **general summary** (goals, decisions, paths, commands, blockers, open threads) that merges forward at each subsequent compaction, and a **detailed recap of the most recently compacted work** at full fidelity (files, commands, errors), replaced wholesale by each new pass.
- The chat keeps a **summary card** at the top afterwards: an expandable card showing the general summary and the detailed recap, followed by a marker naming how many messages were compacted. Earlier turns vanish from the screen, but the card is display-only (the digests reach the model through the system block), so it never counts against the context window and never double-counts what the model already knows.
- The history rewrite is transactional: the conversation is only replaced once both summaries exist, so an unavailable or failed on-device model leaves everything untouched — compaction never degrades into silent truncation.
- The compacted prefix no longer matches the pinned KV state, so the next turn on a weights model pays one full re-prefill by design.
- Compaction is manual for now; automatic context-window-aware thresholding may come later.

### MoE Contiguous Binary Repackaging (Flash-MoE)
Sparse MoE models with hundreds of experts (e.g., Qwen 3.8 Flash Next with 512 routed experts) distribute their tensor weights across dozens of multi-gigabyte `.safetensors` shard files. During token-by-token generation under SSD streaming:
- Loading non-contiguous expert slices causes random file seeks across multiple shard files, severely throttling NVMe read throughput.
- **The Solution**: DynaMoE implements contiguous binary repackaging adapted from Dan Woods' Flash-MoE specification.
- Clicking **Repack Model for Fast Streaming** restructures all expert projections (`gate_proj`, `up_proj`, `down_proj`) into contiguous per-layer binary files (`packed_experts/layer_XX.bin` + `layout.json`).
- Once repacked, an 8-thread POSIX `pread` pool streams the exact active expert bytes directly into shared Metal staging buffers in a single sequential I/O read, achieving peak SSD read performance (>3.5 GB/s on Apple internal SSDs).

> [!TIP]
> Repackaging is recommended for large MoE models that will run in **SSD Streaming** mode. For models that fit entirely into physical RAM (like Ornith 1.5 9B), repackaging is not required.

### Dense Weight Quantization (Q4 / Q8)
A dense model has no routed experts to repack around, and a token reads the *whole* weight set, so decode throughput tracks the bytes on disk rather than which expert the router picked. The Models tab therefore offers **Quantize to Q8** and **Quantize to Q4** for a dense checkpoint (a model with no `num_experts`), which rewrites its weights into one contiguous quantized blob next to the snapshot:

- `dense_weights.bin` holds every tensor back to back; `dense_weights.json` records each one's dtype, shape and byte range.
- Weights are quantized to MLX-style affine 4-bit or 8-bit with a group size of 64: every group of 64 values keeps a scale and an offset, which is what the engine's GEMV kernels read. Perplexity loss is the usual 4-bit/8-bit trade; the quantization is lossy, and a repack is not reversible — delete the two files to go back to the full-precision checkpoint.
- The token embedding and output head stay at full precision: the embedding is read one row at a time and the head is what the sampler reads.
- Nanbeige 4.3's 66 GB becomes about **34 GB at Q8** and **22 GB at Q4**. A Q4 repack of that checkpoint still answers its published turn, with the same opening tokens as the full-precision run.
- The repack is a one-time, minutes-long job (it reads every shard and writes the blob). An interrupted run leaves an incomplete manifest, which the engine detects and ignores: it falls back to the original safetensors, so a half-written repack can never load as a broken model.

#### Model-Specific Profiles ("Coder" & "Assistant")
Different workloads demand opposite sampling behaviors. Coding requires precision, structural consistency, and low entropy, while assistant conversations require natural flow, topic freshness, and conversational variety.

DynaMoE automatically defaults to and persists the official publisher-tuned profiles for each model family:

#### Ornith-1.5-9B Official Profiles
- **Precise Coding & Tool Calling (`Coder`)**:
  - **Temperature**: $0.60$ (low entropy, deterministic syntax)
  - **Top-P**: $0.95$ | **Top-K**: $20$ | **Min-P**: $0.00$
  - **Repetition Penalty**: $1.00$ (disabled to preserve code braces/keywords)
  - **Presence Penalty**: $0.00$ (strict schema & JSON compliance)
  - **Max Tokens**: $8,192$ | **JetSpec**: Disabled (linear recurrent GDN)
- **General Chat / Agent Loops (`Assistant`)**:
  - **Temperature**: $1.00$ (creative conversational pacing)
  - **Top-P**: $0.95$ | **Top-K**: $20$ | **Min-P**: $0.00$
  - **Repetition Penalty**: $1.00$
  - **Presence Penalty**: $1.50$ (fresh vocabulary, prevents looping)
  - **Max Tokens**: $4,096$ | **JetSpec**: Disabled (linear recurrent GDN)

#### Qwen 3.8 Flash Next FP8 Official Profiles
- **Coding & Agentic Profile (Thinking Mode - `Coder`)**:
  - **Temperature**: $1.00$ (high-entropy exploration for SWE-bench & reasoning)
  - **Top-P**: $0.95$ | **Top-K**: $20$ | **Min-P**: $0.00$
  - **Repetition Penalty**: $1.00$
  - **Presence Penalty**: $0.00$ (code consistency)
  - **Max Tokens**: $8,192$ | **JetSpec**: Enabled (draft speculative decoding)
- **General Assistant Profile (Instruct / Direct Mode - `Assistant`)**:
  - **Temperature**: $0.70$ (balanced direct responses without reasoning overhead)
  - **Top-P**: $0.80$ | **Top-K**: $20$ | **Min-P**: $0.00$
  - **Repetition Penalty**: $1.00$
  - **Presence Penalty**: $1.50$ (fresh conversational flow)
  - **Max Tokens**: $4,096$ | **JetSpec**: Enabled (draft speculative decoding)

#### Profile Customization & Persistence
1. Open the **Models** section, click the **⋯** (ellipsis) menu on your model, and choose **"Configure Profiles…"**.
2. Switch between the **Coder** and **Assistant** segments in the inspector.
3. Customize hyperparameters specifically for that model (Temperature, Top-P, Min-P, Top-K, Repetition Penalty, Presence Penalty, Max Tokens, JetSpec, System Prompt).
4. Click **Save [Coder/Assistant] Profile** to persist these settings permanently to `UserDefaults`.
5. In the chat interface, toggle between **[ 💻 Coder ▾ ]** and **[ 💬 Assistant ▾ ]** right next to the model selector with zero friction.

#### Thinking Levels (per conversation)

Next to the profile selector, the **[ 🧠 Thinking ▾ ]** dropdown controls reasoning per conversation. For models whose chat template only has the on/off switch (most models), it offers **Thinking On / Off**. For level-capable models whose chat template exposes a graded reasoning effort, such as Spark X2.6's `reasoning_effort`, it offers **High (full reasoning)**, **Low (brief reasoning)**, and **Off (direct response)** instead. The chosen level is stored per conversation and defaults to High. Changing it mid-conversation starts a fresh prompt prefix, so the next turn re-reads the conversation before generating. Level capability is detected from the model's Spark family config and chat template rather than its name, so repacked or quantized copies of a level-capable checkpoint stay detected, and plain on/off models are unaffected.

---

## 4. Generation & Sampling Hyperparameters

The **Generation** section configures the mathematical sampling engine used during autoregressive token decoding.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Generation Section                                                    │
│  Active Profile: [ Coder ]  [ Assistant ]              [ Save ]       │
│                                                                        │
│  Temperature: 0.60 ────────●──────────────  Top-P: 0.95 ───────────●── │
│  Min-P: 0.00 ●────────────────────────────  Top-K: 20   ─────●──────── │
│  Repetition Pen: 1.00 ●───────────────────  Presence Pen: 0.00 ●────── │
│  Max Tokens: 8192 ────────────────●───────                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Active Profile Banner & Fast Switching
At the top of the Generation section, an active profile banner indicates which profile is currently governing the session. Any adjustments made here can be saved back to that profile using the **"Save"** button.

### Sampling Parameters (Math & Practical Tuning)

#### 1. Temperature ($T$)
- **What it does**: Controls the sharpness of the probability distribution by scaling raw logits $z_i$ before the Softmax function:
  $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **Low Values ($0.0 \le T \le 0.4$)**: Collapses probability mass onto the highest-scoring tokens. Ideal for deterministic coding, formal syntax, JSON generation, and mathematical reasoning.
- **Medium Values ($0.6 \le T \le 0.7$)**: Balanced distribution. Retains logical coherence while avoiding robotic repetition. Recommended for code explanations and structured assistant tasks.
- **High Values ($0.8 \le T \le 1.2$)**: Flattens distribution, encouraging creative vocabulary and diverse phrasing. May cause syntax errors if used for programming.

#### 2. Top-P (Nucleus Sampling)
- **What it does**: Truncates the vocabulary tail by dynamically keeping only the smallest subset of candidate tokens whose cumulative Softmax probability reaches threshold $P$:
  $$\sum_{w \in V^{(P)}} P(w) \ge P$$
- **Recommended Values**: $0.90\text{–}0.95$. A value of $0.95$ allows healthy vocabulary variation while discarding extremely unlikely hallucinated tokens.

#### 3. Min-P (Dynamic Truncation)
- **What it does**: Discards any candidate token whose probability is less than a fraction of the single most probable candidate ($P_{\text{max}}$):
  $$\text{Keep } w_i \iff P(w_i) \ge \text{Min-}P \times P_{\text{max}}$$
- **Why it is superior to Top-P in coding**: When the model is 99% confident in a syntax token (e.g., closing parenthesis or keyword), Min-$P$ automatically prunes all alternatives. When the model is genuinely uncertain among several valid identifiers, Min-$P$ gracefully widens the candidate pool.
- **Recommended Values**: $0.00$ (disabled) or $0.05$ (standard).

#### 4. Top-K
- **What it does**: Enforces a strict upper bound on the number of candidate tokens evaluated during sampling using an $O(\log K)$ vectorized min-heap in Apple Accelerate.
- **Recommended Values**:
  - **Coder**: $20\text{–}40$ (prevents wild token divergence in syntax).
  - **Assistant**: $40\text{–}60$ (encourages richer phrasing).

#### 5. Repetition Penalty (Multiplicative)
- **What it does**: Scales logits of previously generated tokens: divides positive logits and multiplies negative logits by factor $r$:
  $$z'_i = \begin{cases} z_i / r & \text{if } z_i > 0 \\ z_i \times r & \text{if } z_i \le 0 \end{cases}$$
- **Recommended Values**:
  - **Coder**: $1.00$ (disabled). **Never set repetition penalty $> 1.0$ when writing code**, as programming languages require repeating variable names, structural brackets, and boilerplate keywords.
  - **Assistant**: $1.05\text{–}1.12$. Moderately penalizes repetitive phrasing in prose.

#### 6. Presence Penalty (Additive)
- **What it does**: Applies an additive flat penalty subtracted from the logit of any unique token that has appeared within the recent context window (sliding window of 256 tokens):
  $$z'_i = z_i - (\text{presencePenalty} \times \mathbb{I}[w_i \in \text{recentTokens}])$$
- **Why it matters**: Unlike multiplicative repetition penalty, presence penalty applies equally regardless of token magnitude, effectively encouraging the model to introduce new topics and vocabulary without distorting grammatical punctuation.
- **Recommended Values**: $0.00$ for coding; $0.10\text{–}0.40$ for open conversation.

#### 7. Max Tokens
- **What it does**: Hard ceiling on the maximum number of new tokens generated per turn (32 to the KV cache provision ceiling, 32,768 here; a turn may provision up to the checkpoint's own `max_position_embeddings`, capped at 262,144 rows). The KV cache budget can lower the effective ceiling further for a long prompt.
- **Recommended Values**: $8192$ for Coder; $4096$ for Assistant.

---

### System Prompts & Conjunction Merging
DynaMoE features an intelligent **System Prompt Conjunction Engine**:
- A few families require a specific system instruction (Ornith, DeepSeek-R1). Nanbeige 4.3 does not: its template has a Chinese default line, but that line is only the fallback for a conversation with no system message, and DynaMoE always sends one, so the persona is deliberately not injected.
- In DynaMoE, your custom system prompt entered in Settings is never overwritten; instead, it is merged *in conjunction* with the model's mandatory instruct requirements.
- Example merged prompt:
  ```text
  [Model Instruct Persona: You are Ornith, a helpful AI assistant developed by...]
  
  [User Guidelines: Respond concisely in clean GitHub Flavored Markdown. Prioritize idiomatic Swift code.]
  ```

---

### JetSpec Speculative Tree Acceleration
JetSpec accelerates token generation by proposing structured trees of candidate tokens in parallel and scoring them all at once using **tree-causal attention masking**.

- **Enable JetSpec**: Toggles the speculative drafting engine.
- **Tree Depth ($D$)**: How many speculative sequential steps ahead to draft ($1\text{–}6$). Recommended: $2\text{–}3$.
- **Branching Factor ($B$)**: How many alternative candidate paths to expand per node ($1\text{–}4$). Recommended: $2$.
- **Max Active Expert Cap ($E_{\text{max}}$)**: Dynamic pruning threshold for MoE models. When the union of active experts across candidate tree branches exceeds $E_{\text{max}}$, low-confidence speculative branches are pruned to protect SSD streaming bandwidth. Recommended: $8\text{–}12$.

> [!IMPORTANT]
> JetSpec is designed for dense models with draft heads or MoE architectures with fast router lookahead. For linear recurrence models (such as Ornith 1.5's Gated DeltaNet layers), JetSpec is disabled by default because state updates are already $O(1)$ and state branching introduces memory overhead.

---

## 5. Memory & SSD Management

The **Memory & SSD** section controls how model weights and runtime caches are allocated across Apple Silicon's Unified Memory Architecture (UMA) and NVMe storage.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Memory & SSD Section                                                  │
│  Memory Execution Mode:  [ Auto (Smart) ]  [ Full RAM ]  [ SSD Stream ]│
│  Memory Budget Mode:     [ Balanced (16GB) ▾ ]                         │
│  KV Cache Precision:     [ FP16 ▾ ]                                    │
│  Lookahead Prefetching:  [ON]  Depth: 2 layers                         │
│                                                                        │
│  Live Working Set:                                                     │
│  • Resident RSS: 5.42 GB / 16.0 GB      • Cache Hit Rate: 98.4%        │
│  • Resident Experts: 512 / 512           • Paging Latency: 0.12 ms      │
└────────────────────────────────────────────────────────────────────────┘
```

### Memory Execution Modes
- **Auto (Smart)** *(Recommended)*: DynaMoE calculates the model footprint against total unified RAM. If sufficient RAM headroom exists ($\ge 25\%$), it pins the model in memory. If the model exceeds available RAM, it automatically engages SSD streaming.
- **Full RAM**: Locks the entire model backbone and all MoE experts into physical memory via `posix_madvise(POSIX_MADV_WILLNEED)`. Guarantees maximum token throughput without disk reads.
- **SSD Streaming**: Keeps only embedding tables, normalizations, and routing networks in RAM while streaming active MoE experts on demand from SSD storage.

### Memory Budget Modes & Limits
Sets the memory envelope for the process to avoid triggering macOS system-wide memory compression or paging:
- **Conservative (12GB)**: Best for 16GB Macs with heavy background multitasking.
- **Balanced (16GB)**: Default for 16GB–24GB Macs.
- **High Capacity (24GB+)**: Maximizes resident expert caching on 32GB+ systems.
- **Unrestricted**: No artificial resident set size (RSS) ceiling.

### KV Cache Precision
Controls the floating-point representation of the autoregressive key-value cache:
- **FP32**: Full 32-bit precision. Highest numerical fidelity, largest memory footprint.
- **FP16**: Half precision. 50% memory reduction compared to FP32 with zero noticeable degradation in generation quality. Default for most configurations.
- **FP8 (E4M3 / E5M2)**: 8-bit quantized KV cache with per-head dynamic scale factors. Reduces KV cache memory by 75%, enabling 32k+ token contexts on 16GB Macs without memory pressure.

**Long context needs FP8 KV.** The row ceiling is now derived from the checkpoint's own `max_position_embeddings` (capped at 262,144 rows) instead of a flat 32,768, so a long-context model can be driven past 32k when the machine affords it — and the KV budget, not that ceiling, is what decides. The bytes are what bind. A 52-layer dense model with 8 KV heads of 128 costs 4 KB per layer per logical token at FP16 (1024 stride x 2 tensors x 2 bytes), i.e. about 208 KB per token across the 52 layers (52 slots x 1024 stride x 2 tensors x 2 bytes). So a 66 GB checkpoint on a 128 GB Mac leaves roughly 27 GB of KV budget in Unrestricted mode: about 130k rows at FP16, versus about 260k at FP8 (the FP8 cache also carries one scaling byte per 32 values, about 1.6% on top). That FP8 figure sits within a few percent of the 262,144-row policy ceiling, so at FP8 the KV budget stops being the binding constraint before the ceiling does. Past roughly 100k rows, switch the KV cache to FP8 E4M3; below that FP16 is the better trade.

### Speculative MoE Lookahead Prefetching
When running sparse MoE models in SSD streaming mode:
- **Lookahead Prefetching**: Predicts upcoming layer expert activations and triggers asynchronous kernel paging (`POSIX_MADV_WILLNEED`) via a background thread pool before the GPU finishes the current layer.
- **Prefetch Depth**: Number of layers ahead to issue prefetch instructions ($1\text{–}4$). Recommended: $2$.

### Live Diagnostics & Working Set Metrics
- **DynaMoE Memory**: Total RAM attributable to DynaMoE — Darwin's `phys_footprint` (dirty heap, compressed memory, IOKit/GPU, including the FlashMoE pinned backbone) **plus** the clean file-backed model weights held in the Unified Memory Buffer Cache. Activity Monitor's *Memory* column shows only the footprint, so it omits those multi-GB weight pages; the cache share is reclaimable under memory pressure. The tooltip breaks the total into footprint, cache, and resident pages (RSIZE).
- **Cache Hit Rate**: Percentage of expert routing requests served directly from RAM without SSD I/O.
- **Paging Latency**: Time spent by the I/O pool reading expert slices from disk.
- **Prefetch Efficiency**: Ratio of prefetched pages that were successfully utilized.

### Cache Maintenance: Flush vs. Pre-Fault
- **Flush Cache**: Invokes `posix_madvise(POSIX_MADV_DONTNEED)` across all unpinned expert buffers, freeing system RAM immediately.
- **Pre-Fault All**: Pre-reads all SafeTensors or packed binary shards sequentially into system file cache, warming the model before timing-critical benchmarks.

---

## 6. Agent & Tools

The **Agent & Tools** section configures DynaMoE's autonomous agent engine, tool-calling loop, and external search integrations.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Agent & Tools Section                                                 │
│  Autonomous Tool Calling:  [ON]                                        │
│  Working Directory:        [/Users/username/Workspace/Projects] [Pick] │
│  Max Tool Output:          [4000 characters ▾]                         │
│  Max Agent Steps:          [15 steps ▾]                                │
│                                                                        │
│  Search Providers:                                                     │
│  • Headless Chrome: [Active - Local Browser Automation]                │
│  • Custom Chrome Binary: [/Applications/Google Chrome.app/...]         │
│  • Brave Search API Key: [••••••••••••••••••••••••••••••••]            │
└────────────────────────────────────────────────────────────────────────┘
```

### Global Agent Enablement & Sandbox Directory
- **Enable Agent Tools**: When active, DynaMoE detects JSON tool-calling syntax in model responses and automatically executes local system functions, returning the tool result back into the prompt context for multi-turn reasoning.
- **Agent Working Directory**: Defines the sandboxed local filesystem directory for file read/write operations and terminal command execution. DynaMoE restricts file modifications to this directory to protect your operating system.

### Safety Limits
- **Max Tool Output Character Length**: Caps the maximum length of `stdout`/`stderr` returned by tools (1,000 to 16,000 characters). Prevents large file dumps from exhausting model context windows.
- **Max Autonomous Steps**: Hard limit on sequential tool execution cycles (1 to 100 steps) per user prompt. Prevents infinite tool-calling loops. Past about 40 steps a local model usually exhausts the KV budget first, so the run stops with a capacity error rather than the step limit.

### Plan Mode (Read-Only Architect)
Plan Mode makes the agent investigate and propose instead of acting. It is a per-conversation mode, so one chat can be planning while another is coding.

- **Enter**: click the mode pill beside Turbo, press **⇧⌘P**, or type `/plan`. `/plan off` (or clicking the pill again) returns to Code Mode. Entering Plan Mode turns Agent Tools on, because planning needs something to read with — and turning Agent Tools off leaves Plan Mode with them, since the directive, the read tools and the handoff all belong to the agent surface. A backend with no tool loop cannot run the mode, so entry is refused with a message rather than flipping a mode its turns cannot carry: the Apple Foundation Model (plain chat), and any cloud provider registered without tool support. A cloud model that does call tools plans normally — the same directive, read tools and marker contract go out in its system prompt, with the withheld tools absent from its `tools` array.
- **What it withholds**: `file_write`, `file_edit`, `shell_run`, `git_commit`, `complete`, and every `mcp__*` tool. That is enforced in three places rather than merely asked of the model: the tools are absent from the prompt and the cloud `tools` array, they are withheld from the constrained-sampling grammar, and `executeTool` refuses them if one is somehow emitted.
- **What it keeps**: file reads, glob/grep, codebase search, symbol and reference lookup, `git_status`/`git_diff`, web search and fetch, lint diagnostics, and the tool-discovery tools. Entering the mode loads a small read core (`file_read`, `grep_search`, `find_files`, `codebase_search`) so an investigation does not open with a discovery step; the rest is one `tools_load` away, which keeps each plan prompt's schema block small. Subagents spawned in Plan Mode inherit the restriction, so delegation cannot route around it — including the Test Runner archetype, whose shell pipeline is withheld rather than lifting a command out of the task text and running it.
- **The deliverable**: the model opens its final reply with `<!-- DYNAMOE_PLAN_START -->` and ends it with `<!-- DYNAMOE_PLAN_READY -->`. That reply renders as an **Implementation Plan** card, and a handoff bar offers **Start coding**, **Code with Turbo**, or **Revise plan**. Approving switches the conversation to Code Mode, re-arms the write tools, and sends "Implement the plan." as a normal user turn. The marker lines stay in the committed history (they are stripped at render time only), so later turns keep seeing the convention the model was taught; a model that echoes an old marker in an ordinary reply can therefore raise a plan card without having planned, which is why the card is presentation and the handoff bar plus the mode pill remain the controls.
- **Turbo** has nothing to auto-approve while planning, because no write tools exist. The handoff asks explicitly, and "Code with Turbo" is how it gets turned on for the implementation turn.
- **Failure states**: a plan that never reaches the end marker shows no handoff, and the mode pill is always the way back. Dismissing the bar and pressing Return in an empty composer brings it back. The bar only describes the newest plan no later user turn has overtaken, so a follow-up question answered without markers does not resurrect an older plan. Leaving Plan Mode by any route restores the coder surface.
- **The mode belongs to the turn as well as the conversation**: the pill, `/plan` and the handoff are all refused while a turn is running, because the live turn's prompt, grammar and tool surface were built for the mode it started in. The handoff bar is hidden until the plan's turn finishes, and its approve buttons are disabled while another turn holds the engine. An approval whose send is refused (no engine slot, an unusable cloud client) puts the conversation back in Plan Mode with the transcript unchanged rather than leaving it in Code Mode with nothing sent or a dangling "Implement the plan." turn. Switching to a backend that cannot call tools — the Apple Foundation Model, or a cloud model registered without tool support — ends Plan Mode on the conversations that backend serves rather than stranding them in a mode whose pill is hidden. A `/plan` argument that is neither `on` nor `off` (a typo such as `/plan off.`) is sent as ordinary text instead of being guessed at, and `/plan` is still available while a `/compact` pass runs (only `/compact` collides with itself).

### Search Integrations
- **Headless Chrome Web Search (De Facto Built-In)**: Built-in, privacy-preserving web search executed via a local sandboxed browser instance (`--headless=new --dump-dom`). Automatically executes client-side JavaScript, renders complex web pages, and extracts structured titles, URLs, and real-time snippets with **zero API keys or external tracking**. Automatically detects Google Chrome, Chromium, Brave, or Microsoft Edge, with an optional user override for custom binary paths.
- **Brave Search API**: Optional cloud search integration. Enter your Brave Search API key to route queries through Brave's structured search API.

### Built-In Local Tools Reference
DynaMoE equips models with the following native tools:
1. `shell_run(command)`: Executes shell commands in the sandboxed directory with timeout and exit code monitoring.
2. `file_read(path, start_line, end_line)`: Reads files line-by-line with bounds checking.
3. `file_write(path, content)`: Creates or updates files with automated parent directory creation and compiler feedback.
4. `file_edit(path, target_content, replacement_content)`: High-precision search-and-replace code editing with self-healing compiler feedback.
5. `find_files(pattern, max_depth)` & `grep_search(query, regex)`: Fast glob discovery and ripgrep text/regex search.
6. `codebase_search(query)`: Hybrid BM25 & Metal GPU vector embeddings search across AST code chunks.
7. `spawn_subagent(role, goal)`: Spawns isolated background subagents for parallel research, test running, or optimization.
8. `git_status`, `git_diff`, `git_commit`: Complete Git version control management.
9. `lint_diagnostics(file_path)`: Native compiler error detection and self-healing diagnostics.
10. `web_search(query)`: Live web search via local Headless Chrome browser automation (with Brave fallback).
11. `web_fetch(url)`: Fetches web pages with client-side JavaScript execution via Headless Chrome DOM dumping and markdown extraction.

### Model Context Protocol (MCP) Servers
MCP lets DynaMoE use tools that ship as separate programs or remote services instead of compiled-in Swift. Servers are declared in one JSON file; the **MCP Servers** card in this section shows their live status and lets you add, edit, enable, restart, and remove them.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  MCP Servers                                                           │
│  Enable MCP Servers: [ON]                                              │
│  ~/Library/Application Support/DynaMoE/mcp.json                        │
│  [Open Config File] [Reveal] [Reload] [Edit as JSON]                   │
│                                                                        │
│  Configured Servers (2)                                                │
│  ● filesystem   [Ready · 11]        stdio   [ON]                       │
│      npx -y @modelcontextprotocol/server-filesystem ${workspace}       │
│      ▸ 11 tools discovered                                             │
│  ● linear       [Failed]            http    [ON]                       │
│      https://mcp.linear.app/mcp                                        │
│      HTTP 401: unauthorized                                            │
│  [+ Add Server…]  [Restart All]                                        │
└────────────────────────────────────────────────────────────────────────┘
```

#### Configuration File

The file at `~/Library/Application Support/DynaMoE/mcp.json` is the single source of truth. Editing it by hand is fully supported: a watcher picks up the change and reconciles live connections (an edit that lands mid-generation waits for the turn to end, so a running tool call is never killed).

```json
{
  "version": 1,
  "mcpServers": {
    "filesystem": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "${workspace}"],
      "env": { "LOG_LEVEL": "warn" },
      "cwd": "${workspace}",
      "startupTimeoutSec": 30,
      "toolTimeoutSec": 120
    },
    "linear": {
      "enabled": true,
      "url": "https://mcp.linear.app/mcp",
      "headers": { "Authorization": "${keychain:dynamoe.mcp.linear.Authorization}" }
    }
  }
}
```

- **Transport is inferred**: a `command` means **stdio** (DynaMoE launches the program as a child process and speaks JSON-RPC over its stdin/stdout); a `url` means **Streamable HTTP**. Declare exactly one of the two.
- **`enabled`** defaults to `true` when omitted, so a config pasted from another client works as-is.
- **Server names** may contain letters, digits, `-` and `_`. They may not contain `__` or end with `_`, because tool names are built as `mcp__<server>__<tool>`: a name ending in `_` would run into that separator and share tool names with the name without it.
- **Secrets**: any env value or header may be a `${keychain:...}` token instead of plaintext. The Settings form has a per-field **Keychain** checkbox that stores the value in the Keychain and writes only the token to disk.
- **Keychain access**: shipped builds keep these items in the data protection keychain, where access is granted by the app's code-signing identity rather than by a per-item prompt, so an update never asks again. "Is a credential stored?" is answered from item attributes, which macOS returns without a dialog, so nothing here asks for access at launch. If a token still lives in the keychain an older version wrote, the **Stored Credentials** card (Agent & Tools, and Cloud Providers) copies it forward with one system prompt per credential.
- **Other tokens**: `${env:NAME}` reads an environment variable; `${workspace}` is the agent working directory. Tokens are expanded when a server connects and are never written back, so changing the workspace reconnects every server that uses `${workspace}` instead of leaving it scoped to the previous project.

#### Tool Discovery & Loading

Discovered tools are added to the tool catalog as `mcp__<server>__<tool>` but are **not loaded into the prompt**. This is deliberate: loaded schemas cost prefill tokens on every turn, and DynaMoE's agent keeps only a small core set loaded. The model calls `tools_discover` (filter by category `mcp`), then `tools_load` with the tools a task needs.

Each server's row has a disclosure listing its tools with a **Load** button, which does the same thing without spending a model turn.

#### Safety Model

- **Every MCP tool call requires confirmation**, even for tools a server marks read-only, unless **Turbo Mode** is on. A remote server can do anything, including writing files and spending money, and its annotations are claims rather than guarantees.
- **MCP servers are unsandboxed local programs or network endpoints.** They run with the same privileges as `shell_run`. Only add servers you trust.
- Removing a server from Settings stops it and deletes its Keychain entries. Editing it out of the JSON by hand stops it but leaves the secrets, so a transient edit cannot lose a credential.
- Failures are terminal: a server that cannot start stays `Failed` with its error (and the tail of its stderr) shown until you press **Restart** or change its config. Nothing retries in a loop.

#### Troubleshooting

- **`Failed` with `env: npx: No such file or directory`**: the command is not on the app's PATH. DynaMoE injects `/opt/homebrew/bin`, `/usr/local/bin`, `~/.cargo/bin` and `~/.local/bin`, so installing via Homebrew is usually enough. Otherwise use an absolute path.
- **First start is slow**: `npx -y` downloads the package on first use. Raise `startupTimeoutSec` if the default 30 s is not enough.
- **`HTTP 404` after it worked before**: the HTTP session expired. Press **Restart**.
- **Tools exist but the model never calls them**: MCP tools are lazily loaded by design. Ask the model to discover tools, or press **Load** on the tool in this card.

---

## 7. Local Inference Server

The **Local Inference Server** section turns DynaMoE into a local API server: the model loaded in the app answers OpenAI Chat Completions and Anthropic Messages requests from any client on your Mac, with no external service and no network egress. It drives the same headless engine as the chat UI, so a client gets what the app would have generated: the same chat templates, samplers, KV-cache prefix reuse, grammar-constrained tool calls, reasoning splitting and stop-sequence handling.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Local Inference Server Section                                        │
│  Live Status:                                                          │
│  Server State:      [● Running]                                        │
│  Endpoint:          [http://127.0.0.1:8787/v1] [Copy]                  │
│  Serving Model:     [Qwen3.8-Flash-Next] [Copy]                        │
│  Engine Activity:   [Generating | Idle]                                │
│  Queued Turns:      [1]     Requests In Flight: [1]                    │
│  Total Requests:    [42]    Authentication: [Open Access | API Key]    │
│                                                                        │
│  Control:                                                              │
│  Enable Local Server:  [ON]                                            │
│  Port:                 [8787]  [Restart]                               │
│  Reasoning in Content: [OFF]                                           │
│                                                                        │
│  Connecting Clients:                                                   │
│  Available Endpoints: POST /v1/chat/completions, POST /v1/messages,    │
│                       GET /v1/models, GET /health                      │
│  Model Name:          [loaded model name] [Copy]                       │
│  Max Tokens Default:  [Generation tab value]                           │
└────────────────────────────────────────────────────────────────────────┘
```

### Enabling the Server
- **Enable Local Server**: Starts the server immediately and automatically on every subsequent launch. Toggling it off stops the server at once and drops any in-flight SSE streams (an in-progress in-app response is left alone).
- **Port**: The TCP port the server listens on (default 8787). A changed port takes effect after a restart, which the **Restart** button performs without touching the enable setting.
- The server binds to `127.0.0.1` only, so nothing outside your Mac can reach it, and only locally loaded weights are servable (the Apple system model is a separate, chat-only backend).

### Live Status
- **Server State** and **Engine Activity**: Whether the server is listening, and whether a completion turn is generating on the GPU right now.
- **Endpoint** and **Serving Model**: The base URL and the model id clients should reference (also what `GET /v1/models` returns). Both have copy buttons. Requests are rejected with `503` until a model is loaded.
- **Queued Turns**, **Requests In Flight**, **Total Requests**: The in-process queue depth (including the running turn), the completions being handled right now, and the total accepted since launch. **Authentication** reports whether an API key is required.

### Endpoints & Authentication
- `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic), each streaming (SSE) or non-streaming, plus `GET /v1/models` and `GET /health` (`status`, `model_loaded`, `model`, `engine_active`).
- Set an API key to require it as `Authorization: Bearer <key>` (OpenAI convention) or `x-api-key: <key>` (Anthropic convention). An empty key accepts any local process. Changes apply immediately.

### Reasoning in Content
- **Reasoning in Content** (OpenAI endpoint only): When on, reasoning is folded into `content` inside the model's own reasoning tags instead of being reported as `reasoning_content` deltas, so clients that ignore that extension still display it, and they echo it back on the next turn.

### Queueing & Guardrails
- **One turn at a time**: The engine is single-flight, so concurrent requests queue in arrival order and queued streaming clients receive SSE heartbeats so they never idle-timeout while waiting. A request that arrives while an in-app response is generating is answered with `503` once it reaches the front of the queue.
- **Queue deadline**: `dynamoe_server_queue_timeout` (default 120 seconds) rejects requests that wait too long with `408` (or an SSE error). `0` waits indefinitely.
- **Prompt ceiling**: `dynamoe_server_max_prompt_tokens` (default 262,144 tokens) rejects oversized prompts with `400` before prefill. `0` disables the check.
- **KV capacity**: before prefill, the turn is priced against the KV cache this machine can actually provision (RAM minus the model footprint minus a 2.5 GB working reserve, at the configured KV precision and the model's real sliding-window layout). A prompt that cannot fit is rejected with `400` and `"code": "context_length_exceeded"`; the message names the budget, the affordable rows and the reserve involved. The memory budget always wins, so a tools-enabled turn's session reserve is capped to what the machine affords rather than refusing prompts it could serve.
- **Agent session reserve**: `dynamoe_agent_session_floor` (default 8,192 rows) is the KV headroom a tools-enabled turn keeps for its multi-step loop, in-app and for API clients alike; plain chat turns reserve 2,048. Lower it on a memory-constrained Mac with `defaults write DRP.DynaMoE dynamoe_agent_session_floor -int 4096`. `0` restores the default, and values above the turn's row ceiling are clamped to it.
- **Multi-turn prefix reuse**: A client resends the whole conversation every turn, so the server records the last completed turn (reasoning included) and replays it whenever the echoed history matches, letting a follow-up extend the pinned prompt instead of re-prefilling it. Reuse is refused whenever the history, rendered system prompt or echoed answer no longer match.

### Client Setup
- Point an OpenAI-compatible client at `http://127.0.0.1:8787/v1` or an Anthropic client at `http://127.0.0.1:8787`. Any `model` string is accepted; the loaded model answers every request.
- Anthropic requests must send `max_tokens`; an OpenAI request that omits it falls back to the **Max Tokens Default** shown above (the Generation tab value).
- Reasoning-capable models separate their thinking into `reasoning_content` (OpenAI) or thinking blocks (Anthropic): pass `"enable_thinking": false` on the OpenAI endpoint to disable it, or leave `thinking` unset on the Anthropic endpoint.
- Tool schemas sent in `tools` are honored: calls return as `tool_calls` (OpenAI) or `tool_use` blocks (Anthropic), and sending the results back continues the loop. `tool_choice` accepts `auto` and `none`; forced choices are rejected with `400`, and an Anthropic request that ends with assistant text (a response prefill) is rejected as well.

---

## 8. Cloud Providers

The **Cloud Providers** section connects remote inference providers. **Synthetic** ships built in, and **Add Provider…** registers any other OpenAI-compatible endpoint at runtime. A configured provider's models appear in the chat composer under **Cloud Models**, alongside local weights and the Apple on-device model. In agent mode a cloud turn uses the provider's native function calling, while DynaMoE's local engine, KV cache, grammar mask, JetSpec, and memory/SSD machinery stay out of the path.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Cloud Providers Section                                               │
│  Synthetic (built in)                                                  │
│  API Key:          [••••••••••••]  [Save]  [Remove]                    │
│  Connection:       [Test & Sync Models] Connected ✓ 14 registered      │
│  Documentation:    [Open API Docs]                                     │
│  Provider:         Identifier synthetic · base https://api.synthetic…  │
│                                                                        │
│  Models:           [Sync Models] [Clear]  14 registered                │
│                                                                        │
│  Add a Provider                                                        │
│  OpenAI-Compatible Endpoint                          [Add Provider…]   │
│                                                                        │
│  Registered Cloud Models                                               │
│  Synthetic:        DeepSeek-V4.1-Flash-1017 (Synthetic)  ·  …          │
└────────────────────────────────────────────────────────────────────────┘
```

### Adding & Removing Providers
- **Add Provider…** opens a form for any endpoint that speaks `POST <base>/chat/completions` with an `Authorization: Bearer <key>` header (SSE streaming) and lists its catalog at `GET <base>/models`. The key never rides the URL.
- **Display name** is what the composer shows as `model (name)`. The **identifier** is a lowercase slug derived from it; it becomes the Keychain account suffix (`dynamoe.cloud.<identifier>.apikey`) and the `cloud://<identifier>/<model>` address saved with each conversation, so it cannot be changed later.
- **API base URL** includes the version segment (`https://opencode.ai/zen/v1`). `https://` is required, except that plain `http://` is accepted on `127.0.0.1`, `localhost`, and `::1` for a local server. Enter the base, not `…/chat/completions` or `…/models`. A missing scheme defaults to `https`, a trailing slash is removed, and the scheme and host are lowercased. A URL carrying `user:password@` is refused with an inline message instead of being saved or silently stripped: the key belongs in the API Key field and rides the `Authorization` header. A query string is refused for the same reason it cannot be honoured: the app appends the endpoint path, so an Azure-style `?api-version=…` would be dropped and the request would fail later. Editing a saved provider's base URL drops that provider's registered models, no-tools list, and vision marks, because every one of them is a claim about the old server; press **Test & Sync Models** to register the new endpoint's catalog.
- **Built-in providers cannot be edited or removed**, so the registry is never empty. **Remove** on a custom provider deletes its Keychain key, its registered model list, its no-tools list, and its vision marks; conversations pinned to its models then refuse the send with a reselect message rather than falling back to local weights.
- The list persists as a JSON array under `dynamoe_cloud_custom_providers`. A hand-edited entry that cannot be decoded costs only that provider; the rest of the list is kept. Stored URLs are re-validated when the list is read, so a record written by an older build (or edited by hand) with plain `http://`, an embedded `user:password@`, or a query string is dropped instead of being displayed and used for turns.
- **OpenCode Zen** (hosted gateway): name `OpenCode Zen`, identifier `opencode-zen`, base `https://opencode.ai/zen/v1`, key from `console.opencode.ai`. Its `/models` catalog lists models served on other endpoints too: the `gpt-5.x` family uses `/zen/v1/responses` and Claude models use `/zen/v1/messages` (the Anthropic wire, currently a stub), so only the OpenAI-compatible ids answer here (`qwen3-coder`, `kimi-k2`, `kimi-k2-thinking`, `glm-4.6`, `glm-4.7-free`, `grok-code`, `big-pickle`, and their current `-free` variants). The **−** button beside a registered model prunes the rest; sending to one returns the gateway's own "Model … is not supported" error, which the turn surfaces.

### API Key & Connection Test
- **API Key**: Paste a key from the provider's dashboard and press **Save**. It is written only to the Keychain (service `DRP.DynaMoE.cloud`, account `dynamoe.cloud.<provider>.apikey`) — never to UserDefaults, logs, or a config file. The field never pre-fills an existing key; the placeholder says "Stored" and typing a new value replaces it. **Remove** deletes the item from both the current and the older keychain.
- **Stored Credentials**: shows up only when a credential is still in the keychain an older version wrote; press **Import** in that card. Importing copies each one across (read back and verified) and then removes the older copy, so a new version stops asking for keychain access. macOS asks once per credential during the import; choose **Always Allow**.
- **Test & Sync Models**: Calls the provider's `GET /models` endpoint with the stored key. A good key validates the credential **and** registers the provider's current chat models, then rescans so the composer's **Cloud Models** section updates immediately. Embeddings, rerankers, speech, moderation, and image generators in the same catalog are skipped (falling back to the unfiltered list only if the filter would register nothing). A bad key returns `401`, a rate-limited endpoint `429`, and an unreachable endpoint a network error — all shown inline. The test spends no completion tokens.
- **Documentation**: Opens the provider's API reference. The link must be `http(s)`, and because it is stored in plain text with the provider record, `user:password@` and query strings are refused rather than written to disk.

### Registered Models
- **Sync Models** re-runs the catalog fetch and replaces the registered list, which is how provider-side additions and retirements reach the picker. **Clear** unregisters the provider's list.
- The **Registered Cloud Models** card lists what will appear in the composer, one provider per row. `hf:` models show their model tail (`DeepSeek-V4.1-Flash-1017 (Synthetic)`); provider aliases such as `syn:large:text` are humanized (`Syn Large Text (Synthetic)`).
- The list persists in `dynamoe_cloud_<provider>_models`. Manual overrides still work when you want a narrower or pinned set:

  ```bash
  defaults write DRP.DynaMoE dynamoe_cloud_synthetic_models \
    "hf:deepseek-ai/DeepSeek-V4.1-Flash-1017; hf:moonshotai/Kimi-K2-Thinking"
  ```

  Entries are separated by `;`, `,`, or newlines and duplicates are dropped. A hand-written value may not reach an already-running app because of preference caching — quit and relaunch, or just press **Sync Models**.
- A single-model development hook exists: `defaults write DRP.DynaMoE dynamoe_cloud_debug_model "hf:org/model"` registers one model when the main list is empty. It is one global value rather than a per-provider one, so the minus button does not clear it; remove it with `defaults delete DRP.DynaMoE dynamoe_cloud_debug_model`.

### Image Input (Vision-Capable Cloud Models)

Each row in the **Registered Cloud Models** card carries a per-model **switch** that marks it as accepting images. Vision is opt-in because no provider catalog publishes modalities: a model left off the list is treated as text-only, and the composer refuses an attachment with a notice instead of sending pixels it would never receive. The list persists in `dynamoe_cloud_<provider>_vision` (`;`/`,`/newline separated):

```bash
# Mark DeepSeek V4.1 Flash 1017 as accepting images
defaults write DRP.DynaMoE dynamoe_cloud_synthetic_vision "hf:deepseek-ai/DeepSeek-V4.1-Flash-1017"
```

To attach an image to a turn: paste a screenshot (⌘V), drag an image file onto the composer, use **+ → Attach Image…**, or type an absolute path to an image file — the path is replaced by an attachment chip. The app reads the file itself because the model has no file system access (the agent's `file_read` tool refuses binary content).

Preparation policy: EXIF orientation is baked into the pixels, metadata is stripped, the longest edge is downscaled to 1024 px, and the bytes are re-encoded as PNG for screenshots, including TIFF clipboard captures, so text stays legible, or JPEG for photos. Limits are 4 MB per image, 6 images per message, and 32 MB per source file; anything else is refused with a message. The same picture attached twice is stored once per message.

On the wire an attached user turn becomes `content: [{"type":"text",…}, {"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]`. Text-only turns keep the bare-string `content` they always had. Images are re-sent with their original message on every later request (including each agent step), so the model keeps its visual context; each later turn therefore pays that image's tokens again, and the pre-turn estimate prices each image as a flat allowance until the provider reports real usage.

A model marked capable that rejects the image part is un-marked automatically (the Settings switch turns off), the turn is retried once without the picture, and a system message records it in the transcript. Only error bodies that name the image part itself trigger this, so an unrelated `400` never disables a working model. Re-mark it only after confirming the model really accepts images. Attachment files live at `~/Library/Application Support/DynaMoE/sessions/assets/<session>/`; deleting a conversation or letting auto-deletion prune it removes its images too.

For local weights, vision is a property of the checkpoint rather than a switch: a family whose snapshot ships an image tower (`DeepSeek-V4.1-Flash-1017` today) takes images in the chat window and over the local inference server, and every other family plus the Apple on-device model is text-only, so attaching an image there is refused with a notice and the disabled attach menu items say why. The composer writes one image marker per attachment and prices the expanded prompt before the tower runs, so the chip's estimate is the encode it will pay. The local inference server answers an `image_url` / `image` content part with `400` and `"code": "unsupported_content"` when the loaded checkpoint publishes no tower, rather than silently stripping it, and the models list advertises `capabilities.vision` plus what an image costs so a client can discover either without sending one.

### What a Cloud Turn Supports
- **Streaming** SSE text and, when the model reports it, `reasoning_content` thinking deltas in the usual accordion. Token counts come from the provider's `usage` when reported and fall back to character estimation. The provider's optional `stream_options.include_usage` field is retried without it if the endpoint rejects it.
- **Sampling**: `temperature`, `top_p`, `presence_penalty`, and `max_tokens` forward; Min-P, Top-K, and the repetition penalty have no remote equivalent and are ignored (the same documented gap as the on-device model). Stop sequences are not forwarded — the provider owns templating and its own end-of-turn marker.
- **Agent tools**: with the toolbar switch on (the default for a cloud model), the request carries the loaded tool schemas and the provider's native `tool_calls` are executed through the same harness as a local agent turn — HITL approvals, Turbo Mode, empty-argument rejection, the repeated/failed-call loop guards, and the subagent report relay all apply. Tool results replay as a `tool` message per call. Turn the switch off for plain chat.
- **Models without tool support**: list them under `dynamoe_cloud_<provider>_no_tools` (`;`/`,`/newline separated); that model then runs plain chat with a visible notice instead of failing mid-loop on an HTTP 400.
- **Reasoning control**: the Thinking menu is hidden for cloud models because the OpenAI-compatible surface has no portable on/off switch — the provider's default applies, and reported reasoning still renders in the accordion.
- **History**: the conversation is sent as a standard `messages` array, with the persona as the leading system message and UI notice messages skipped.
- **Images**: only when the model is marked vision-capable. The request carries a multi-part `content` array with `image_url` data URLs; earlier images are re-sent with their message on every step so the model keeps its visual context. A model that rejects an image is un-marked and the step retries once without it.
- **Logging**: every cloud turn records a row in **Settings → Server → Request Log**, with the rendered prompt (system + messages) under the content-capture toggle. An agent run writes one row per step.
- **Timeouts**: the idle (between-bytes) timeout is 120 s, so a large prompt or a reasoning model can take its time to first token; a connection that stops sending bytes for two minutes fails with "The request timed out."
- **Not supported in this phase**: grammar masking, JetSpec, prefix pinning, and the memory budgets. Switching to a cloud model releases any resident local engine so the RAM budget is not held.

---

## 9. Hardware Profiles & Recommended Configurations

### 16 GB Unified RAM (M1/M2/M3/M4)
- **Primary Model**: `ornith-ai/Ornith-1.5-9B` (Dense Hybrid GDN; BF16 weights stream from SSD).
- **Memory Execution Mode**: `Auto (Smart)` (the BF16 model exceeds 16 GB, so weights stream from SSD).
- **KV Cache Precision**: `FP16` or `FP8 E4M3` (FP8 recommended for $>8,000$ token contexts).
- **JetSpec**: Disabled (`false`).
- **Coder Profile**: $T=0.60$, $\text{Top-}P=0.95$, $\text{Top-}K=20$, $\text{RepPen}=1.00$, $\text{PresPen}=0.00$.
- **Assistant Profile**: $T=1.00$, $\text{Top-}P=0.95$, $\text{Top-}K=20$, $\text{RepPen}=1.00$, $\text{PresPen}=1.50$.

### 24 GB – 36 GB Unified RAM
- **Primary Models**: Ornith 1.5 9B (Full RAM) or Ornith 1.5 35B A3B / Qwen 3.8 Flash Next (SSD Streaming).
- **Qwen 3.8 Flash Next Profiles**:
  - **Thinking Mode (Coder)**: $T=1.00$, $\text{Top-}P=0.95$, $\text{Top-}K=20$, $\text{RepPen}=1.00$, $\text{PresPen}=0.00$.
  - **Instruct Mode (Assistant)**: $T=0.70$, $\text{Top-}P=0.80$, $\text{Top-}K=20$, $\text{RepPen}=1.00$, $\text{PresPen}=1.50$.
- **Memory Execution Mode**: `Auto (Smart)`.
- **Memory Budget Mode**: `Balanced (16GB)` or `High Capacity (24GB)`.
- **KV Cache Precision**: `FP16`.
- **Speculative Prefetching**: Enabled (`true`), Lookahead Depth: `2`.

### 64 GB – 128 GB+ Unified RAM (M-Max / M-Ultra)
- **Primary Models**: Full MoE models resident in RAM without SSD streaming.
- **Memory Execution Mode**: `Full RAM`.
- **KV Cache Precision**: `FP16` (or `FP32` for research validation).
- **JetSpec**: Enabled (`true`), Depth: `3`, Branching: `2`, Expert Cap: `16`.

---

## 10. Frequently Asked Questions & Troubleshooting

#### Q: Why does token generation speed gradually decrease over a long thinking chain?
**A:** This occurs primarily because of the model's hybrid architecture. For example, Ornith 1.5 9B has 24 Gated DeltaNet layers (which run in constant $O(1)$ time) and 8 full Grouped-Query Attention (GQA) layers (which must scan all previous tokens in the KV cache on every step, scaling $O(T)$). Additionally, the displayed tokens/sec is a cumulative running average from token 1, meaning as instantaneous speed naturally drops at 4,000+ tokens, the displayed average will gradually slide toward 1.5–2.0 tok/s.

#### Q: Why is repetition penalty set to 1.0 in the Coder profile?
**A:** In programming, repeating variable names, structural brackets, indentations, and language keywords is required. Applying a repetition penalty $> 1.0$ forces the model to invent erroneous synonyms for identifiers and omit required syntax. Use **Presence Penalty ($0.0\text{–}0.1$)** instead if you want mild topic variety.

#### Q: What does "Repack Model for Fast Streaming" do?
**A:** It converts scattered expert weights across multiple SafeTensors shards into contiguous per-layer binary files (`packed_experts/layer_XX.bin`), enabling DynaMoE to stream active MoE experts via high-speed POSIX direct I/O without disk seek latency.

#### Q: How do I restore default settings?
**A:** Open **Settings $\to$ Models**, click the **⋯** (ellipsis) menu on your active model, choose **"Configure Profiles…"**, and click **Reset to Defaults**. This restores factory-tuned parameters for the profile you are currently editing (Coder **or** Assistant), not both at once; repeat for the other profile if needed. The running session keeps its current values until you click **Apply to Session**.
