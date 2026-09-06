# DynaMoE Settings & User Guide

A comprehensive, practical guide to configuring, tuning, and operating DynaMoE on Apple Silicon. This document details every tab in the **Settings & Diagnostics** panel (`⌘,`), explains the underlying mathematics and runtime mechanics, provides battle-tested presets for coding versus conversational assistance, and offers memory budgeting advice for Mac hardware configurations.

---

## Table of Contents
1. [Overview & Accessing Settings](#1-overview--accessing-settings)
2. [Tab 1: Models & Binary Repackaging](#2-tab-1-models--binary-repackaging)
   - [Local Model Discovery](#local-model-discovery)
   - [Default & Last Used Model](#default--last-used-model)
   - [MoE Contiguous Binary Repackaging (Flash-MoE)](#moe-contiguous-binary-repackaging-flash-moe)
   - [Model-Specific Profiles ("Coder" & "Assistant")](#model-specific-profiles-coder--assistant)
3. [Tab 2: Generation & Sampling Hyperparameters](#3-tab-2-generation--sampling-hyperparameters)
   - [Active Profile Banner & Fast Switching](#active-profile-banner--fast-switching)
   - [Sampling Parameters (Math & Practical Tuning)](#sampling-parameters-math--practical-tuning)
   - [System Prompts & Conjunction Merging](#system-prompts--conjunction-merging)
   - [JetSpec Speculative Tree Acceleration](#jetspec-speculative-tree-acceleration)
4. [Tab 3: Memory & SSD Management](#4-tab-3-memory--ssd-management)
   - [Memory Execution Modes (Auto, Full RAM, SSD Streaming)](#memory-execution-modes)
   - [Memory Budget Modes & Limits](#memory-budget-modes--limits)
   - [KV Cache Precision (FP32, FP16, FP8 E4M3/E5M2)](#kv-cache-precision)
   - [Speculative MoE Lookahead Prefetching](#speculative-moe-lookahead-prefetching)
   - [Live Diagnostics & Working Set Metrics](#live-diagnostics--working-set-metrics)
   - [Cache Maintenance: Flush vs. Pre-Fault](#cache-maintenance-flush-vs-pre-fault)
5. [Tab 4: Agent & Tools](#5-tab-4-agent--tools)
   - [Global Agent Enablement & Sandbox Directory](#global-agent-enablement--sandbox-directory)
   - [Safety Limits (Max Output & Max Steps)](#safety-limits)
   - [Search Integrations (Headless Chrome & Brave Search)](#search-integrations)
   - [Built-In Local Tools Reference](#built-in-local-tools-reference)
6. [Tab 5: Advanced Diagnostics](#6-tab-5-advanced-diagnostics)
   - [SafeTensors Sharded Metadata & Search](#safetensors-sharded-metadata--search)
   - [Direct Metal Kernel Execution Harness](#direct-metal-kernel-execution-harness)
7. [Hardware Profiles & Recommended Configurations](#7-hardware-profiles--recommended-configurations)
   - [16 GB Unified RAM (M1/M2/M3/M4)](#16-gb-unified-ram-m1m2m3m4)
   - [24 GB – 36 GB Unified RAM](#24-gb--36-gb-unified-ram)
   - [64 GB – 128 GB+ Unified RAM (M-Max / M-Ultra)](#64-gb--128-gb-unified-ram-m-max--m-ultra)
8. [Frequently Asked Questions & Troubleshooting](#8-frequently-asked-questions--troubleshooting)

---

## 1. Overview & Accessing Settings

DynaMoE provides centralized control over model inference, GPU shader dispatch, unified memory allocation, and agent autonomy.

### How to Open Settings
- **Keyboard Shortcut**: Press `⌘,` (Command + Comma) from any window.
- **Application Menu**: Navigate to **DynaMoE $\to$ Settings…** in the top macOS menu bar.
- **Chat Header**: Click the **Settings** slider icon in the upper right toolbar of the chat interface.

### How to Open this Guide in the App
- **Help Menu**: Select **Help $\to$ DynaMoE Help & Settings Guide** (or press `⌘?`).
- **Settings Header**: Click the **Help & Guide** button located in the top-right of the Settings sheet.

---

## 2. Tab 1: Models & Binary Repackaging

The **Models** tab acts as your local model registry and management hub.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Models Tab                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 📦 mlx-community/Ornith-1.5-9B-OptiQ-4bit    [Active] [Profiles] │  │
│  │    Architecture: Hybrid GDN + GQA | Precision: 4-bit | 5.4 GB    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 📦 Qwen/Qwen3.8-Flash-Next-FP8               [Default] [Repack]  │  │
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

### MoE Contiguous Binary Repackaging (Flash-MoE)
Sparse MoE models with hundreds of experts (e.g., Qwen 3.8 Flash Next with 512 routed experts) distribute their tensor weights across dozens of multi-gigabyte `.safetensors` shard files. During token-by-token generation under SSD streaming:
- Loading non-contiguous expert slices causes random file seeks across multiple shard files, severely throttling NVMe read throughput.
- **The Solution**: DynaMoE implements contiguous binary repackaging adapted from Dan Woods' Flash-MoE specification.
- Clicking **Repack Model for Fast Streaming** restructures all expert projections (`gate_proj`, `up_proj`, `down_proj`) into contiguous per-layer binary files (`packed_experts/layer_XX.bin` + `layout.json`).
- Once repacked, an 8-thread POSIX `pread` pool streams the exact active expert bytes directly into shared Metal staging buffers in a single sequential I/O read, achieving peak SSD read performance (>3.5 GB/s on Apple internal SSDs).

> [!TIP]
> Repackaging is recommended for large MoE models that will run in **SSD Streaming** mode. For models that fit entirely into physical RAM (like Ornith 1.5 9B), repackaging is not required.

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
1. Click any model card in the **Models** tab or click its **"Profiles"** button.
2. Select either the **Coder** or **Assistant** tab in the inspector.
3. Customize hyperparameters specifically for that model (Temperature, Top-P, Min-P, Top-K, Repetition Penalty, Presence Penalty, Max Tokens, JetSpec, System Prompt).
4. Click **Save Profile** to persist these settings permanently to `UserDefaults`.
5. In the chat interface, toggle between **[ 💻 Coder ▾ ]** and **[ 💬 Assistant ▾ ]** right next to the model selector with zero friction.

---

## 3. Tab 2: Generation & Sampling Hyperparameters

The **Generation** tab configures the mathematical sampling engine used during autoregressive token decoding.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Generation Tab                                                        │
│  Active Profile: [ 💻 Coder ]  [ 💬 Assistant ]   [ Save to Coder ]    │
│                                                                        │
│  Temperature: 0.60 ────────●──────────────  Top-P: 0.95 ───────────●── │
│  Min-P: 0.00 ●────────────────────────────  Top-K: 20   ─────●──────── │
│  Repetition Pen: 1.00 ●───────────────────  Presence Pen: 0.00 ●────── │
│  Max Tokens: 8192 ────────────────●───────                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Active Profile Banner & Fast Switching
At the top of the Generation tab, an active profile banner indicates which profile is currently governing the session. Any slider adjustments made here can be saved back to that profile using the **"Save to [Active Profile]"** button.

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
- **What it does**: Hard ceiling on the maximum number of new tokens generated per turn (64 to 32,768).
- **Recommended Values**: $8192$ for Coder; $4096$ for Assistant.

---

### System Prompts & Conjunction Merging
DynaMoE features an intelligent **System Prompt Conjunction Engine**:
- Many frontier models require specific system instructions embedded in their tokenizer chat templates (e.g. Ornith, DeepSeek-R1, Nanbeige).
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

## 4. Tab 3: Memory & SSD Management

The **Memory & SSD** tab controls how model weights and runtime caches are allocated across Apple Silicon's Unified Memory Architecture (UMA) and NVMe storage.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Memory & SSD Tab                                                      │
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

### Speculative MoE Lookahead Prefetching
When running sparse MoE models in SSD streaming mode:
- **Lookahead Prefetching**: Predicts upcoming layer expert activations and triggers asynchronous kernel paging (`POSIX_MADV_WILLNEED`) via a background thread pool before the GPU finishes the current layer.
- **Prefetch Depth**: Number of layers ahead to issue prefetch instructions ($1\text{–}4$). Recommended: $2$.

### Live Diagnostics & Working Set Metrics
- **Resident Process RSS**: Actual physical RAM occupied by DynaMoE.
- **Cache Hit Rate**: Percentage of expert routing requests served directly from RAM without SSD I/O.
- **Paging Latency**: Time spent by the I/O pool reading expert slices from disk.
- **Prefetch Efficiency**: Ratio of prefetched pages that were successfully utilized.

### Cache Maintenance: Flush vs. Pre-Fault
- **Flush Cache**: Invokes `posix_madvise(POSIX_MADV_DONTNEED)` across all unpinned expert buffers, freeing system RAM immediately.
- **Pre-Fault All**: Pre-reads all SafeTensors or packed binary shards sequentially into system file cache, warming the model before timing-critical benchmarks.

---

## 5. Tab 4: Agent & Tools

The **Agent & Tools** tab configures DynaMoE's autonomous agent engine, tool-calling loop, and external search integrations.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Agent & Tools Tab                                                     │
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
- **Max Autonomous Steps**: Hard limit on sequential tool execution cycles (1 to 30 steps) per user prompt. Prevents infinite tool-calling loops.

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

---

## 6. Tab 5: Advanced Diagnostics

The **Advanced Diagnostics** tab is an engineering inspection suite for validating Metal shaders, examining raw SafeTensors tensors, and testing layer forward execution independently of chat generation.

```text
┌────────────────────────────────────────────────────────────────────────┐
│  Advanced Diagnostics Tab                                              │
│  Category Filter: [ All ] [ Attention ] [ Experts ] [ Router ] [ Norm ]│
│  Search Tensors:  [ model.layers.0.mlp... ]                            │
│                                                                        │
│  Selected: model.layers.0.mlp.experts.0.gate_proj.weight               │
│  • Shape: [1024, 2048]  • Dtype: F16  • Offset: 0x0042F000 (1.42 MB)   │
│                                                                        │
│  Kernel Execution Harness:                                             │
│  [ Execute Router Top-K ]  [ Execute Full Layer ]  [ Multi-Layer Test ]│
└────────────────────────────────────────────────────────────────────────┘
```

### SafeTensors Sharded Metadata & Search
- Live search over hundreds of tensors in multi-gigabyte models.
- Filter by structural category: *Embedding*, *Attention*, *Routing*, *Experts*, *Normalization*, and *LM Head*.
- Inspect byte offsets, memory shard indices, dimensions, data types, and total tensor memory footprint.

### Direct Metal Kernel Execution Harness
Test GPU compute pipelines in isolation without running full autoregressive generation:
- **Execute MoE Router**: Runs the Top-$K$ gating kernel on the selected layer to verify softmax routing probability distribution and expert index selection.
- **Execute Full Layer**: Computes a complete forward pass through one transformer layer (RMSNorm $\to$ DeltaNet/Attention $\to$ Gated Residual $\to$ MoE/Dense MLP $\to$ PostNorm).
- **Multi-Layer Test**: Chains $N$ consecutive layers through double-buffered Metal command encoders to measure raw GPU layer compute latency in milliseconds.

---

## 7. Hardware Profiles & Recommended Configurations

### 16 GB Unified RAM (M1/M2/M3/M4)
- **Primary Model**: `mlx-community/Ornith-1.5-9B-OptiQ-4bit` (Dense Hybrid GDN).
- **Memory Execution Mode**: `Auto (Smart)` or `Full RAM`.
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

## 8. Frequently Asked Questions & Troubleshooting

#### Q: Why does token generation speed gradually decrease over a long thinking chain?
**A:** This occurs primarily because of the model's hybrid architecture. For example, Ornith 1.5 9B has 24 Gated DeltaNet layers (which run in constant $O(1)$ time) and 8 full Grouped-Query Attention (GQA) layers (which must scan all previous tokens in the KV cache on every step, scaling $O(T)$). Additionally, the displayed tokens/sec is a cumulative running average from token 1, meaning as instantaneous speed naturally drops at 4,000+ tokens, the displayed average will gradually slide toward 1.5–2.0 tok/s.

#### Q: Why is repetition penalty set to 1.0 in the Coder profile?
**A:** In programming, repeating variable names, structural brackets, indentations, and language keywords is required. Applying a repetition penalty $> 1.0$ forces the model to invent erroneous synonyms for identifiers and omit required syntax. Use **Presence Penalty ($0.0\text{–}0.1$)** instead if you want mild topic variety.

#### Q: What does "Repack Model for Fast Streaming" do?
**A:** It converts scattered expert weights across multiple SafeTensors shards into contiguous per-layer binary files (`packed_experts/layer_XX.bin`), enabling DynaMoE to stream active MoE experts via high-speed POSIX direct I/O without disk seek latency.

#### Q: How do I restore default settings?
**A:** Open **Settings $\to$ Models**, click **"Profiles"** on your active model, and click **Reset to Defaults**. This will immediately restore factory-tuned parameters for both Coder and Assistant profiles.
