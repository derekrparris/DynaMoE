# **Feasibility and Architectural Evaluation of Integrating JetSpec Parallel Tree Drafting into DynaMoE**

## **Executive Assessment and Strategic Vision**

Integrating the JetSpec speculative decoding framework into the DynaMoE local inference engine represents a transformative engineering opportunity, provided the integration accounts for system memory constraints1. Developed by researchers at UC San Diego's Hao AI Lab, JetSpec achieves up to ![][image1] latency reductions on reasoning-heavy workloads by training a lightweight causal parallel draft head over fused hidden states from a frozen target model2. This architecture generates a scored tree of candidate tokens in a single forward pass, which the target model then verifies all at once using tree-causal attention masking2.  
DynaMoE is engineered specifically for Apple Silicon's Unified Memory Architecture (UMA) to run massive Mixture-of-Experts (MoE) models that exceed physical system RAM1. It achieves high throughput by dynamically memory-mapping weight files and streaming expert parameters directly from NVMe storage to GPU memory1.  
The strategic viability of merging JetSpec into DynaMoE depends heavily on the engine's operational memory regime:

* **RAM-Resident Execution**: When model weights fit entirely within Apple Silicon unified memory, JetSpec provides immediate throughput gains1. The zero-copy UMA memory bridge in DynaMoE eliminates host-to-device transfer overhead, allowing the causal draft head to run alongside target verification with minimal latency1.  
* **NVMe SSD-Paged Execution**: When running massive MoE architectures (such as Qwen3.8-Flash-Next) under physical RAM constraints, evaluating a wide candidate token tree causes an **MoE Expert Expansion Conflict**1. Evaluating multiple candidate branches in a single target verification pass activates a larger union of routed experts per layer1. This increases NVMe read bandwidth demands and can create disk streaming bottlenecks that diminish the speedup from higher token acceptance rates1.

Implementing JetSpec into DynaMoE is strongly recommended, provided the implementation includes an expert-aware tree pruning mechanism within DynaMoE's prefetching pipeline to limit expert fan-out during SSD streaming1.

## **Fundamental Architectural Foundations**

### **The JetSpec Speculative Framework**

Speculative decoding accelerates autoregressive language models by using a lightweight drafting mechanism to propose multiple candidate tokens, which are then verified in parallel by the target model in a single forward pass2. Traditional speculative frameworks suffer from a causality-efficiency dilemma3:

* **Autoregressive Drafters**: Methods like EAGLE generate path-conditioned candidates sequentially, resulting in high acceptance rates but incurring drafting latency that scales linearly with tree depth3.  
* **Block-Diffusion Drafters**: Methods like DFlash draft an entire token block in one forward pass, but score draft positions independently2. This independent scoring generates candidate trees with inconsistent token branches, leading to lower acceptance rates during target verification2.

JetSpec resolves this trade-off by attaching a lightweight, trainable causal draft head to a frozen target language model2. The draft head reuses multi-layer hidden states (![][image2]) from the target model's forward pass to generate an entire candidate tree in a single pass2. To maintain branch consistency, JetSpec applies a **tree-causal attention mask** across draft slots3. Each tree node attends exclusively to its prefix and direct ancestors, ensuring candidate scores align with the target model's autoregressive probability distribution ![][image3]3.  
The frozen target model then verifies the candidate tree in a single forward pass using a tree-causal attention mask2. The target commits the longest accepted path while maintaining its exact output distribution2.

| Evaluation Benchmark | Target Model Architecture | Baseline Latency Speedup | Mean Accepted Tokens (τ) | Peak Engine Throughput | Primary Source |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **MATH-500** | Qwen3-8B | ![][image1] | 10.76 tokens / round | \~1,456 tok/s (NVIDIA B200) | 2 |
| **GSM8K** | Qwen3-8B | ![][image4] | 8.62 tokens / round | \~1,000 tok/s (NVIDIA H100) | 2 |
| **HumanEval** | Qwen3-8B | ![][image5] | 7.78 tokens / round | \~1,000 tok/s (NVIDIA H100) | 2 |
| **MT-Bench (Chat)** | Qwen3-8B / MoE | ![][image6] | 4.80 tokens / round | \~1,000 tok/s (NVIDIA H100) | 2 |

### **The DynaMoE Runtime Architecture**

DynaMoE is a high-performance native macOS application, local inference engine, Model Context Protocol (MCP) server, and OpenAI-compatible API host designed for Apple Silicon1. It allows users to run large MoE language models that exceed physical system RAM by dynamically memory-mapping and streaming expert weights from NVMe storage to GPU unified memory1.  
DynaMoE's core system architecture consists of three main components1:

> 1. **Rust Compute Core (dynamoe-core)**: Uses memmap2 to create zero-copy memory maps of single-shard and multi-shard SafeTensors weight files1. It handles index parsing, 3D tensor slicing, tokenization via Hugging Face tokenizers, and exposes functionality to Swift through UniFFI bindings1.  
> 2. **Metal GPU Compute Engine (ComputeShaders.metal)**: Custom Metal Shading Language (MSL) compute kernels supporting MXFP8, Q4, and Q8 affine matrix-vector multiplication (GEMV), Top\-![][image7] expert router gating, fused SwiGLU activations, GatedDeltaNet state operations, and grouped-query attention (GQA)1.  
> 3. **WorkingSetManager & Memory Bridge**: Uses MTLBuffer with .storageModeShared to pass OS page cache addresses directly into Metal GPU buffers, eliminating CPU-to-GPU copy overhead1. An asynchronous prefetching pipeline uses posix\_madvise(POSIX\_MADV\_WILLNEED) to load upcoming MoE experts from NVMe into RAM prior to layer execution, while POSIX\_MADV\_DONTNEED manages memory pressure by evicting inactive pages1.

## **Comparative Architectural Compatibility Analysis**

Integrating JetSpec into DynaMoE requires mapping JetSpec's algorithmic requirements to DynaMoE's hardware and software abstractions1.

| Operational Subsystem | JetSpec Engine Requirement | DynaMoE Baseline Capability | Integration Feasibility & Requirements |
| :---- | :---- | :---- | :---- |
| **Drafting Execution** | Trains a causal parallel draft head over fused multi-layer target hidden states (![][image2])2. | Performs sequential single-token autoregressive decoding; lacks a multi-token draft head1. | **Requires Engine Modification**: Metal MSL compute pipelines must add fused projection and parallel draft head prediction kernels1. |
| **Attention Masking** | Dynamic tree-causal attention masking for both drafting and verification stages3. | Sequential causal masking with FP8/FP16 paged KV cache decoding1. | **Requires Engine Modification**: Attention kernels must support variable-length tree sequence indexing and causal masks1. |
| **Memory Access Bridge** | High-bandwidth access to GPU memory for verification passes6. | Zero-copy MTLBuffer UMA bridge linking OS page cache addresses directly to Metal1. | **Fully Compatible**: Apple Silicon UMA provides unified address space for draft head and target execution1. |
| **MoE Dispatch** | Evaluates ![][image8] candidate tree nodes through Top\-![][image7] router gating per layer4. | Single-token Top\-![][image7] router gating (mlp.gate.weight) with pre-calculated prefetching1. | **Moderate Friction**: Tree nodes fan out expert activations, requiring batched router dispatch1. |
| **Storage Paging Pipeline** | Assumes model weights reside fully in accelerator VRAM6. | Predictive SafeTensors paging using posix\_madvise(POSIX\_MADV\_WILLNEED)1. | **High Friction**: Multi-branch verification expands the active expert set per layer during offloaded execution1. |

## **Technical Analysis: Compute, Memory, and MoE Streaming Dynamics**

Integrating JetSpec into DynaMoE introduces complex interactions between speculative candidate generation and dynamic SSD weight streaming1.

### **The MoE Expert Expansion Problem**

In a dense Transformer model, evaluating ![][image8] candidate tree tokens in a single target forward pass improves compute utilization without changing the total volume of model parameters read from memory2. However, in an MoE model (such as Qwen3.8-Flash-Next or Qwen3-30B-A3B), token routing dynamically selects a subset of ![][image7] experts per layer1.  
In standard single-token generation, the engine routes one token hidden state through the gate matrix, activating ![][image7] experts (e.g., ![][image9] out of 512 total experts)1. The I/O memory footprint for layer ![][image10] is bounded by ![][image11], where ![][image12] is the parameter size per expert1.  
In a JetSpec tree verification pass, ![][image8] candidate nodes are processed simultaneously3. Each candidate node ![][image13] routes independently to ![][image7] experts based on its hidden state ![][image14]1. The unique active expert set ![][image15] required for layer ![][image10] is the union of all routed choices across all candidate nodes:  
![][image16]  
As the candidate tree width expands, ![][image15] grows rapidly4. For example, evaluating ![][image17] candidate nodes across 4 distinct branches can increase the required active experts per layer from ![][image9] to ![][image18] or ![][image19] experts1.  
When model weights reside fully in RAM, this expansion causes minimal degradation because all expert parameters are instantly accessible in unified memory1. However, during SSD-paged execution, tripling the active expert count triples the required NVMe read bandwidth per target pass1. If disk bandwidth is the primary system bottleneck, this extra I/O overhead can negate the performance gains achieved by higher token acceptance rates1.

### **Unified Memory Architecture (UMA) Advantages**

When models fit within physical RAM, Apple Silicon's unified memory architecture offers key structural advantages for JetSpec compared to discrete PCIe GPU systems1:

* **Zero-Copy Hidden State Fusion**: The causal draft head accesses target hidden states ![][image2] directly in unified memory via MTLBuffer references, avoiding PCIe transfer overhead1.  
* **Shared Tree Mask Management**: The Rust core (dynamoe-core) updates candidate tree structures and causal mask buffers directly in shared RAM, making them immediately accessible to Metal shaders without API synchronization stalls1.  
* **Low Parameter Overhead**: JetSpec reuses the target model's non-expert layers and execution context entirely, requiring no separate draft model binary to be loaded into memory2.

## **Implementation Plan for DynaMoE**

To integrate JetSpec effectively while managing SSD streaming limitations, implementation should proceed across four modular components1.

### **1\. Tree Topology and Engine Management (dynamoe-core)**

The Rust compute core (dynamoe-core) must be extended to manage candidate tree structures, parent-child index mapping, and acceptance verification1:

* **Tree Adjacency Representation**: Maintain candidate trees as flattened adjacency arrays. For a tree with 5 nodes (1 root, 2 candidate branches, 2 continuation nodes), the core constructs a ![][image20] tree-causal mask matrix where each node attends only to itself and its direct ancestors3.  
* **Speculative Acceptance Oracle**: Implement target verification logic in Rust that evaluates output token probabilities against draft candidates to commit valid prefixes1.  
* **UniFFI Bridge Extensions**: Export tree topology buffers and mask matrices directly to Swift and Metal without copying memory1.

### **2\. Custom Metal Shading Language Kernels (ComputeShaders.metal)**

DynaMoE's Metal shader library requires two new compute kernels1:

* **jet\_draft\_head\_predict**: Reads fused hidden states ![][image2] from designated target model layers, applies layer normalization, projects them through the draft weight matrix ![][image21], and outputs candidate token probabilities across draft slots2.  
* **flash\_attn\_varlen\_tree**: Modifies DynaMoE's grouped-query attention kernel to enforce tree-causal masking during softmax computation1:

![][image22]

### **3\. Expert-Aware Prefetching (WorkingSetManager)**

To prevent NVMe streaming bottlenecks during offloaded MoE execution, DynaMoE's prefetching engine must be adapted1:

> 1. Run a lightweight router pre-pass across all ![][image8] tree nodes to identify candidate expert selections for upcoming layers1.  
> 2. Calculate the total unique active expert set ![][image15]4.  
> 3. **Dynamic Tree Pruning**: If ![][image23] exceeds a configured bandwidth threshold ![][image24] (e.g., ![][image25] experts), prune lower-probability candidate branches until the required expert count fits within I/O limits1.  
> 4. Issue posix\_madvise(POSIX\_MADV\_WILLNEED) system calls for the pruned expert set to prefetch parameters from NVMe before layer execution1.

### **4\. User Interface and Diagnostics Integration**

DynaMoE's SwiftUI interface should be expanded to display speculative decoding performance metrics1:

* Controls to adjust draft tree depth (![][image26]), tree width (![][image27]), and maximum active expert limits (![][image24]).  
* Real-time metrics tracking mean accepted tokens per round (![][image28]), draft acceptance rates, GPU compute utilization, and SSD read bandwidth1.

## **Performance Expectations Across Memory Regimes**

Analyzing JetSpec across different model architectures and system RAM configurations establishes clear operational expectations for DynaMoE1.

| System Operational Regime | Model Architecture Type | System Memory Offloading Status | Engineering Complexity | Expected Speedup | Deployment Priority |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Regime 1: Fully Resident Dense** | Dense Models (e.g., Qwen2.5-7B, LLaMA-3-8B) | 100% System RAM Resident | Moderate | **High Speedup (![][image29])** | **Priority 1**: Immediate deployment candidate; maximum performance gains2. |
| **Regime 2: Fully Resident MoE** | MoE Models (e.g., Qwen3-30B-A3B) | 100% System RAM Resident (64GB+ Macs) | High | **High Speedup (![][image30])** | **Priority 1**: High gains; expert expansion handled smoothly by unified RAM2. |
| **Regime 3: Paged MoE (Pruned Tree)** | MoE Models (e.g., Qwen3.8-Flash-Next) | Hybrid NVMe SSD Streaming | High | **Moderate Speedup (![][image31])** | **Priority 2**: Requires strict tree width limits (![][image32]) and expert caps1. |
| **Regime 4: Paged MoE (Unconstrained)** | Large MoE Models (100B+ parameters) | Heavy NVMe SSD Streaming | High | **Minimal Gains (![][image33])** | **Not Recommended**: SSD read bottlenecks offset tree verification gains1. |

## **Strategic Summary and Recommendations**

Implementing JetSpec into DynaMoE is **highly recommended**1. It directly addresses the primary performance bottleneck of local LLM inference—memory bandwidth bounds during sequential token generation—by increasing the number of accepted tokens produced per target forward pass2.  
For optimal results, implementation should follow a phased development approach:

> 1. **Phase 1**: Add JetSpec draft head prediction and tree-causal attention kernels to DynaMoE's dense model execution path1. This delivers immediate speedups of ![][image34] for fully RAM-resident models without requiring changes to the paging pipeline2.  
> 2. **Phase 2**: Implement the parent-indexed tree topology manager and speculative acceptance verification algorithms within dynamoe-core1.  
> 3. **Phase 3**: Extend WorkingSetManager with expert-aware tree pruning1. By enforcing a cap on active expert sets (![][image24]) during tree verification, DynaMoE can maintain high inference speedups even when streaming large MoE models from NVMe storage1.

### **Key Considerations for Running Qwen 3.8 Flash Next in DynaMoE**

To resolve issues when bringing up **Qwen 3.8 Flash Next** as the primary target model in DynaMoE, focus on five main architectural differences. Because this model deviates significantly from standard MoE architectures (such as Mixtral, DeepSeek, or Qwen 2.5 MoE), generic assumptions in the ingestion, dispatch, and shader pipelines can cause initialization failures or runtime panics.

#### **1\. SafeTensors Catalog Indexing and Unrecognized Keys (dynamoe-core)**

The Rust core (core/src/lib.rs) indexes model tensors via model.safetensors.index.json. Qwen 3.8 Flash Next introduces non-standard key namespaces that will cause parsing or binding errors if hardcoded tensor mapping logic expects standard attention or MLP keys:

* **N-Gram Local Embedding:** model.ple.ple\_embedding.ngram\_embedding.weight (51B parameter table across 20M entries).

* **Gated DeltaNet (GDN) Recurrent Layers:** model.layers.{i}.linear\_attn.\* (contains decay factor gates $\\beta\_t$, query/key projections, and value projections).

* **Qwen Sparse Attention (QSA) Layers:** model.layers.{i}.self\_attn.\* (contains the Multi-Query Attention indexer weights along with sparse attention projection matrices).

* **Gated Residual Streams:** model.layers.{i}.gated\_residual.\* (contains 4-branch read/write gates with bottleneck rank 320).

* **512 Fine-Grained Expert Pools:** model.layers.{i}.mlp.experts.{j}.\* indexed from expert 0 to 511\.

*Troubleshooting Action:* Update LocalModelManager.swift and lib.rs to ensure the index parser dynamically maps these structural keys without dropping unmapped tensor handles or throwing key-not-found exceptions during model initialization.

#### **2\. Prevent Out-of-Memory (OOM) Crashes on the 51B N-Gram Table**

DynaMoE creates zero-copy MTLBuffer handles for model weights using bytesNoCopy:length:options:deallocator: with the .storageModeShared flag.

* **The Problem:** The 51B parameter N-gram table occupies \~51 GB in FP8 precision. If the ingestion engine attempts to wrap the entire ngram\_embedding.weight mapped file buffer into a single contiguous MTLBuffer at startup, macOS will reject the memory allocation on a 16 GB system, causing an instant crash.

* **Troubleshooting Action:** Exempt ngram\_embedding.weight from automatic GPU buffer initialization. Keep the file virtual memory mapped in Rust via memmap2. When layer 2 generates N-gram hashes, use Rust to extract only the specific 160-dimensional vector rows required for that token, issue a page-level advisory (posix\_madvise(POSIX\_MADV\_WILLNEED)), and return small zero-copy slice pointers to Metal on demand.

#### **3\. Interleaved Heterogeneous Backbone Scheduler (InferenceEngine.swift)**

DynaMoE's standard multi-layer backbone loop (h\_0 $\\to$ h\_N) assumes every transformer layer uses identical attention and MLP execution steps.

* **The Problem:** Qwen 3.8 Flash Next uses a repeating 12-block pattern: 3 successive Gated DeltaNet (GDN) linear attention layers followed by 1 Qwen Sparse Attention (QSA) block:

$$\\text{Block Pattern} \= 3 \\times (\\text{GDN} \\to \\text{MoE}) \+ 1 \\times (\\text{QSA} \\to \\text{MoE})$$

* **Troubleshooting Action:** Modify the sequential execution pipeline in Swift (InferenceEngine.swift) to dispatch the appropriate compute path based on layer index $l$:

  * If $l \\bmod 4 \\neq 3$: Execute GDN linear attention recurrent state updates ($S\_t \= \\text{diag}(\\beta\_t)S\_{t-1} \+ K\_t V\_t^T$).

  * If $l \\bmod 4 \= 3$: Execute the two-stage QSA block-sparse compute pass (MQA indexer micro-block selection followed by sparse attention across selected blocks).

#### **4\. Updating the Top-K Gating Shader for 512 Experts (ComputeShaders.metal)**

DynaMoE's mlp.gate.weight Metal kernel multiplies incoming hidden states against router weights and applies Softmax to extract top expert indices.

* **The Problem:** Standard MoE router kernels in DynaMoE are designed for 8, 16, or 64 total experts selecting Top-2 or Top-8. Qwen 3.8 Flash Next routes across **512 fine-grained experts**, selecting **10 routed experts plus 1 unrouted shared expert** per layer. If the threadgroup reduction in MSL is hardcoded for smaller expert sizes, logit extraction will produce out-of-bounds array reads or incorrect routing probabilities.

* **Troubleshooting Action:** Rewrite the Top-$K$ gating kernel in ComputeShaders.metal to handle 512 logits using parallel threadgroup reduction. Ensure the router selects the Top-10 routed experts, computes Softmax normalization across those 10, and appends the static shared expert projection to the dispatch list.

#### **5\. Implementing 4-Branch Gated Residual Stream Blending**

Standard transformer engines update layer hidden states using single-stream residual addition ($h\_{l+1} \= h\_l \+ f(h\_l)$).

* **The Problem:** Qwen 3.8 Flash Next splits intermediate representations into 4 parallel structural streams modulated by a rank-320 bottleneck. Reverting to standard vector addition causes numerical divergence and garbled text generation.

* **Troubleshooting Action:** Implement a custom gated\_residual\_blend kernel in ComputeShaders.metal. For each layer output, the kernel reads vector states across the 4 branches, applies read gates via Sigmoid activation ($\\sigma(W\_{\\text{read}} \\cdot x\_b)$), applies scalar write scaling $\\gamma\_b$, and accumulates the combined output into the main hidden vector $h\_{l+1}$.

### **Strategic Guidance: Testing JetSpec on Ornith 1.5 9B Dense**

Testing JetSpec on **Ornith 1.5 9B** is an ideal baseline validation strategy.

┌────────────────────────────────────────────────────────────────────────┐  
│                        JetSpec Integration Paths                       │  
└────────────────────────────────────────────────────────────────────────┘  
                                    │  
         ┌──────────────────────────┴──────────────────────────┐  
         ▼                                                     ▼  
┌───────────────────────────────┐             ┌───────────────────────────────┐  
│     Ornith 1.5 9B (Dense)     │             │  Qwen 3.8 Flash Next (MoE)    │  
├───────────────────────────────┤             ├───────────────────────────────┤  
│ • 100% System RAM Resident    │             │ • NVMe SSD Expert Paging      │  
│ • No MoE Expert Expansion     │             │ • Expert Fan-out Expansion    │  
│ • Pure Compute Acceleration   │             │ • Requires Strict Tree Capping│  
│ • Expected Speedup: 5x – 9x   │             │ • Expected Speedup: 2x – 3.5x │  
└───────────────────────────────┘             └───────────────────────────────┘

#### **Why Ornith 1.5 9B is the Right First Target for JetSpec**

> 1. **Zero I/O Bottlenecks:** Ornith 1.5 9B is a \~9B parameter dense model based on Qwen 3.5. In 4-bit/6-bit precision (e.g., Q4\_K\_M or MLX 6-bit), it occupies only \~5.5 GB to \~7.3 GB of memory. It fits entirely in system RAM on a 16 GB Apple Silicon Mac without requiring NVMe offloading.

> 2. **Pure Latency Validation:** Because all parameters remain in RAM, evaluating candidate token trees in a single pass will not be constrained by SSD read speeds. This isolates the JetSpec parallel draft head and tree-causal attention implementation, allowing you to measure pure GPU compute speedups ($5\\times \- 9\\times$).

> 3. **Draft Head Integration:** JetSpec trains a lightweight causal draft head that reuses target model hidden states $h\_l$. Because Ornith 1.5 9B uses standard Qwen 3.5 dense layers, extracting multi-layer hidden states into the draft head requires minimal custom shader development compared to hybrid GDN/QSA models.

#### **Handling Reasoning Chains (\<think\> Blocks)**

Ornith 1.5 9B is a reasoning model that emits chain-of-thought tokens inside \<think\> ... \</think\> blocks before providing its final answer.

* When evaluating JetSpec on reasoning workloads, candidate tree acceptance rates ($\\tau$) are typically higher during long, structured thinking traces because structural reasoning patterns are easier for the parallel draft head to predict.

* Ensure ModelConfig.swift handles reasoning parser tags so that draft tree validation operates seamlessly across both the thinking trace and the final answer response.  


#### **Works cited**

> 1. GitHub \- derekrparris/DynaMoE: Dynamic, SSD-streamed Mixture-of, [https://github.com/derekrparris/DynaMoE](https://github.com/derekrparris/DynaMoE)  
> 2. JetSpec: Causal Parallel Tree Drafting Hits 9.64x Faster LLM Inference, [https://rits.shanghai.nyu.edu/ai/jetspec-causal-parallel-tree-drafting-hits-9-64x-faster-llm-inference/](https://rits.shanghai.nyu.edu/ai/jetspec-causal-parallel-tree-drafting-hits-9-64x-faster-llm-inference/)  
> 3. JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with, [https://arxiv.org/html/2606.18394v2](https://arxiv.org/html/2606.18394v2)  
> 4. JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with, [https://haoailab.com/blogs/parallel-tree-decoding/](https://haoailab.com/blogs/parallel-tree-decoding/)  
> 5. Qwen/Qwen3.8-Flash-Next \- Hugging Face, [https://huggingface.co/Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)  
> 6. JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with, [https://huggingface.co/papers/2606.18394](https://huggingface.co/papers/2606.18394)  
> 7. A speculative decoding method called 'JetSpec' has ... \- GIGAZINE, [https://gigazine.net/gsc\_news/en/20260626-jetspec-speedup-ai/](https://gigazine.net/gsc_news/en/20260626-jetspec-speedup-ai/)  
> 8. sgl-project/sglang \- \[Feature\] JetSpec Speculative Decoding \- GitHub, [https://github.com/sgl-project/sglang/issues/29524](https://github.com/sgl-project/sglang/issues/29524)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAWCAYAAAC/kK73AAACDUlEQVR4Xu2WPUiVYRTH/2U0BSEWtTlIqVAIgQ7RViAuYUSDINTk5haITYFDUyi4CRK0BUFNDg4ONjSkg33gVxh+fwxRDYqD5fnf8zze07nPfe+NGgLfH/y5z/k/5z33vC/vc+4FcnKOD6dFk6JfonG3Vw3XRT+g1w+6Pc8paN5fcw1a6GyIW0NcLW9F2yaeFfWb2LOPP6tfFhaZSXgfnJdiAL83cS7EY8azPBKt4B80fh5aZMT508GvBHNGndfk4kgNtOmXKF/7oTccF+LiPrTIUHGvwETws2iH5twN8W2zl2IvfGY1fkP00ZuBy6L1GNQj/cSXg1/nfMsLaE636LGoVrQq2jQ5kV5RZ1hnNU5uihac14hEXRZ5n/AoHtRyzENzvjqf3rDzdsy6UuPklmgxrNn0ltk7gnfIQhyJpA96MOmdjEkJ3kFzepwfbzryzaxJNY2T2Ly96RL40nN+fxJdFS2hcvFn0JwG59vGu0QPilsFqm2cY5pP+ovfyMI/tRQd0Jxm59trn4jeOMU5HuMUbDq+022iz2bviFSTjO847x7MOAowj4fTez+dZ9lA6fdZ2DRzLGzeH9hCkV0TT6H0wJ1A+gafiw5MfAaac9F4nu8orRO5Ap1oKTgo5qzRAi3EUcZP/mdJ8RrpH4hX0Ovik+SXp+BvBf8arKE4NvnaWJ662HPJGzk5Of8Rh7oukFGI5Z9iAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAbCAYAAACa9mScAAAA20lEQVR4XmNgGAX4wEUg/gPE/4GYC02OJLCfAWIIReAfFFMEQK6YgC5IChBhgBgiiC5BCpjEADHEFYj/AvFtKF8YWREhAAoLkKYkNLE3SHyCAGTAHixi39HENgLxTTQxMACFA0iDOJo4SKwOTYwXKo4B/BgwJRihYuZo4p5AfBdNDAxOM2AaMhuLGAicBOI4dEEQACm+ikXsFpT9Ek0cKwBJBGMR8wJiViDegiaOAcQYsEssYoCIn0MSc2eApB+KwHEgjkEXJBXAXHwZRZREUMsAcQ0TusQoGKwAAL2SNjbWVJtqAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAAAaCAYAAABIIVmfAAADN0lEQVR4Xu2ZS+hNQRzHf97vvBZYiKKUhQ1JsmFBSfJIRDZSpFDSP7EQeaQsSGGjv8fCY2PlkfIoyUaIEhErsfKW9+P3/c+Ze+d+75xz5tx7z/2fxfnUt2a+85s7c2bOmZlzrkhJSUlrWcpGTkxiowD0Vo1gs52cVc1iMye+sVEQPqiGsBnKIdVH1b9IuMh35F2oRNcyX3WNzRz5zkaBwDg1hR1sZpwY3zfQvvg8+cFGgdinusdmFjCYd9mM8E3OdtVD8vLmJxsFA2PUg80QVoipPI8LlAHinwDkZ5CXN7/YKBh/VHvYDOGJ1A+w5aKYskXkx8Vb1qoOqPpG+Q2qo9XihvjNRgqDxOxfc6L8YNUZ1dxKRGs5Ltn72IXvDgezxfiHycfm64u3oBM4ng0XE/dSNV61MSprFNxhoWxR7YzSz1U3VA/ELBHo07CorJXMlORxicVOAI5T78WcNpB/rBrpxFl2SXxDr1X9nTzi8BTZ9FOnLCt/2YhhoOqVkz8o1f5eddKtZrQ08Nt2/V/DBQmckviGNjvpXmLiJkZ5d2Is/dhIIK5NZjrlP0l43TSwrCWRuZ1nkr1Sp4TV6ZD0uLRylyyxLqiHtb9ZFkj6yS9zH1Eha6VtElbniyTHYY8IXVZA0m/F0UdMvQlckMIJqd/7sIcsJs9llDTQR1R4wWYK6ERcQ9g/TkdpxLhv0cukenS1E5/lBgiNw2ZvY/c6aQvnXa6LeXJdJkttX8/VFlfAtSX9dh07xFRYxwUB+BoaI8ZfqVoepU9GZT3FPBEuONXgKQjF16YPxNnTFk9wp2qakwc4seEovpp8Jq39I5Ie0wUCP4s58eC7z1fJthQANDSVTeWt1A48Ph8g73vLDuqsQ2j8FDGxuCZs8ng5shPBG/QV1SbyfCyU9PUfL4r72cyLy6o7bGYA5/BGJj0P8IL2RqovbD7uq5awSeTVv1iaaXC36liUxl0YQjPthYBj8yPVKi6QatvYD3yfnrGco25bwaN9ns1AsO7ifH6L/CTyngCXS6qtTh6b+U2J//OpnX2r4baYNbcddNtFpoDlayib7WQ9Gznh2/S7G3wtHstmSUlJSUmJj//OrMssrvGsNgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAWCAYAAAC/kK73AAAB4klEQVR4Xu2WzysFURTHj7IRSVlY+JH8TKwUC7FiYSdJWLBhI9n5E2yVjYWyIEsKWVhZs7EQSn70UH7EwtKPFOd0731z5jt3Ru+tlPnUtzfne87cOW/uvTNDlJLyPzhmHbJmWOOsMdYoa8TqN/pYn6xv1h7kHDWsGzI1J6zSUDZPZLA4vao6H/2sFRU3kjlPM8Q6UHGGTM2A8vJCBulhtbIaWHVW2IAPX80ma1nFUjOsYuf5zs2JIzTI3KE2ND3IxcvA22KtqtjX5Iv1WsCfgxipQEPTxdpAMwbXlOwP7WmmyewZzQeZOlzr3axT8BxNrHs0NXjhJGSJuebf7G9JqMKPbxYcvawL8JpZj+CFWLPKhVkKGhHNh9MRlsjUSTNxyJPq0h5L3ZPKeZEBa9FMYJe1b48XKWh+IlsRppxMvhMTHlzzz5hApih++nzImsP6IuuhLxSQ8asxEUM7mTudwQQi68p3wTjOWOtoUtA8Il6xiidZ9SrWSNNuTcvsXKlchLg75ZBnsX4c7VD8EwDHeWcVgid/3Ic0/QCeNI8bNktS426adV4akbhKecI1hT8V7ig4F4XIu+MWTUsH6xxNQQb6QlOxTdEXRCUFTcgmkt/BUEW02aTGF9AA5JMiJSXlr/IDb9eDZLQCDVQAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAWCAYAAAC/kK73AAABmElEQVR4Xu2VTytFQRjGX2WnbKS7IvmbWCkWYsXeQsLCio3s5FMoSx9Alja+BYqFUEJd1/9Y2CLF+zYzvOe5M3Pm3pVyfvV07zzve+Y8nTNzhqig4H9wzNpnrbAWWPOsOdasVSqHaCjaWdesL9YJqzlTrROZLKRX1efjgLL9PqZZe2pcJtM7pby6kEnGWQOsblanVSiIj1UK94s/4/FC/ckcoUHmCQ2iGSEvONZerNcP/hqMkRIamlHWDpo5xIIvk9kzmncy/bjWx1in4Dl6WfdoakIBYsSC+/C9BccE6wK8PtYjeBm2rGqlluCbZHolTIhJ1qX9L31PquZFJuxAM4HU4C1k+kaw4MGFf8YCskRpN/eREryBTE8bFgIMkXnSZSwgsq7ybh4iJbjUm9R4kdWlxhoJ7da0vJ0rVasitmEE+RaHPkd5wd9YjeCdwdghoR/Ak/C4YX+IBXevOVRfJ1NrxQJzQ7/XohA5OypoWoZZ52gKMtEnmopdqj4gPsiswzvWrf2Vw2Vb9WDYWPANNIAeNAoKCv4Q34l5dEUgL9eSAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAWCAYAAAC/kK73AAAB7klEQVR4Xu2VPUgdQRSFb0DBOkYsNESF+ANWgiJiF8HOztJGAqkFBS0EwdoyhWBjZyOoCIJFINgZYiEWagQRMf4g4g8ioqD3MDPPO+ftw7doIbgfHHbn3NnZ8+bNzIpkZLxPvqge2CzAuWpc9VlVompX/Y16OFBbFzfusaouLr8OGLzY4KGv1UDUw3Gn+mDaF6rvpv1illXXki74iGpa9YNqgQVVE5tS/DuepVo1rzqV4ge9ZyMBjDVDHmY/6R2DbBCVbIAwUJrgWALPgX8R4+0Yb001atqBTtUGm5561QGbs+I2GEgT/Fb1U9yaxaziuY6oh8PugS3VWFSN+abaJq9BdUielKt+mXaa4JeqbtNuE/fsR+OBMu8HncXlPLpU//w9Qh+ZWg4OmSZ4EngWPyhQ6z2s6xp/z32SCOFPuAAmVY3kpQleyoY8BbPtT6YNfnu/gnxLi7iZ3uUCWFKtkMKLcT/11DWPPnH9hsi3wRGs0CTAH2bTg9BhTWP52Y1dEJ6xQK/Ex1G/uH5VxgPw7OZKGgvAxxHMIPR/8hCeN2weScHDucs+txcTPKxlHImWHtUVeaBZtcemp1W1ySbA2Yo1te+F+z+mPif5HwisXQTFcYjrjcSf9gBmC3UchbiuxuUcE2wQX9nIyMh4QzwCA96KkOkOuvkAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA1UlEQVR4XmNgGAWUgtlA/AmI/yPhVygqGBi+IMmBsDeqNCaAKcQGmoD4PLogLsDIADHoFroEEFwGYl90QXwgmwFiWDiSGBMQ/wNiLiQxosBLBlQvGgLxUyQ+SQA5vKZB2ccQ0qQBkOYLDBAXakH5uCIDL4CF1x8ksSVQsXwkMaLAawbsriDLdbg0vWWAiCuiS+ACzAwQDafRJYBAlQEi9x5dAhfoZ4BoCEWXgAKYqwXRJZDBMgZIfnwHxV8ZIAkUBmQYIC4CpbXHDBC195DkR8EoGLoAALqKPUMnIoY7AAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA3klEQVR4XmNgGAWUgnlA/BmI/0PxAhRZCPjLgJAHYWdUaUyArBgb2AfEKuiC2AAjEG8H4vUMEMOCUKXBAJclGCAfiE2gbFyu+4MugAu8RWJ/YIAYxockpgbEnUh8vADZJaBwAfFvIoktA2IeJD5OAAqvzWhi6F7F5m2sADm8kMVABnRD+b+Q5PCCd+gCUABznTYQt6DJ4QS4vLCbASJ3D4g50eSwAhYg3osuCAVMDJhhhxMwA/EbID6JLoEEvgHxD3RBdLAKiD8yQNIXKF2B8h42oA/E2eiCo2AUDGkAAM4NNN65dbHtAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAaCAYAAAANIPQdAAABjklEQVR4Xu2WPS8FQRSGXxSESOg1gkYolAqVjuiI6CR0JAr/QFQUGtEQ4vM/+AMaBYmORCMKxGeoCM7JmbUzJzPu3ZDdLeZJ3mTnfc9uztxz9wOIRCJlZp30QvqydOtUAK9Wxhp249yYIn1CeuC+M5NswMcC6USbObNImrXWQwj366UGcsK5Dogz0og2C8C3oTvShDZDzEAuMm55tZC/RqPlFUULpD8ehs0DaVJ5QW7g/lJ9pGtrXQaS22nQrJvMumrs+3HNHB+lcSmYR9onDyDTBhk+4RQy0W6zznwRQzNpN6Ad0jZpi7RJ2kC2p+Q+0t5Yo24cJrkfPyxvz3hzllc0F6Rlc3yIdKM9PxW/wE8o39T+Ms3/Zpr0pLw2ZOgxVHgP8dt1UIF60lJGVYL74A8BzQD8vTvUQYqOdUB0QbJHHRTAM2lVm0QnqtjkCqRoTAeGZMqtOsiZDkgfDcp/M5mXA8j3Kr9MWVzML/4E/r/zBPlRfQWpvbTyIuiHbPQd0i8f9zoVkUgkEolEIgnfrjd0VncPZVcAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAYCAYAAAA20uedAAAAZ0lEQVR4XmNgoAsQAWJtdEEvIP4CxP+BeD+aHByAJF3QBUFgIQNEEisASfxCF4QBkGQbuiAIyDNAJLnQJUBgPgMB+36iC8IATi/EMiCMTAdidSQ5hkIGhORfZAkY+MoAUWCOLjESAABgvRYgeEzSiQAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAaCAYAAACn4zKhAAADH0lEQVR4Xu2YWchNURTHl6lkKjKkzCVleJBkSMqThBclJZ7kQR54oPiIDHkkZXiQQvKA8kR5kLx4IFGUMT4Zyjxkntff2qu7zrrnfvece7/u7Xzf/tW/s/daZ9jnrL323mcTRSKRSKSdWMvazBppbFNNuXAcYn1k/TV6mTiD6JPxQfOT7oZxkPWHNYc1iHWCdZOkTd3NeYVFP3Aa21nXvbHB7GL99EbmIlVud6HoQvIi97yDpKct9MYmgPalDTnTSDK18KwmecklxtaVJPV7GVtWlnqDowerrze2wRSS9k3wDmYia703FpEXlEzpyaxnpp6XIawP3hjozfrhjVVAb9fhspZOUQjsfHAglC+X3DUxjMqHCQQgbVzPgrZR9ZrkGR0GvNQNkowYH+o2M2plOOtzKCMAv4wvLz1Zj6g8GAPsSUVF5wP7gY4H2xpjqxUNRD0B8GBOeULSxlPOV0heUXqvb69sQE/FEPTVOzKCTlIJtO+BNzaJujpDpY/9hsQ+2jtyoAEAA1lfjC8raW1T4NvojU1gOtURhG4kL3LVO5ixJL533pGR/lQ+CSMQOkdkpVIQxlC5bx/rKMlQtYg1mEqd7DDrrKnPCEec+5R1jvWckiB7sfzFe+BfCgHHNTtI7oUV5NZgU437f2UO9pBcuNg7AnpjfNA89GN988YAAuFXTZVYQPL8a87eJ9hnGds81ltTtwF6zFrOWsYaauyzKdkpdpMMzwBbOViqg7ms+6HcQqV32xmOCETuTMCeCx6CRkNoCH7MFCz9kAGINHoKzn1o/NXABltb4EcNwajGFdYI1ioqdQj8Y6Ct6OUWbSeOOmlbUF/nbMgGXKOgXXodjnove78NJN/PUlMQOiK3Wfu90dBK5YHBWG6DgF0CG4Q0EIQjzraFdSaUV1pHZwP/Ib9NfYUp3w1HBOmSsSMIdtg8zTofyicpmTl6j02sY8YO8OEvhDIC0qkZRTKJIisw1GC7A8Pt++BvJVn16WJDMwFzxi3W3mBX8MG/U2nCRl2HcL/1cicokpOZlByOIg0GG4HbSDJnkvNFIpFIJNIw/gGJldXabeIXYwAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAaCAYAAAAXHBSTAAAB10lEQVR4Xu2XPShGURjHH19FPlahZFKMDBZMDAYWGUwGuwiLTBaDLEqIoixSbEaTZDL5GCwGksFnFPL5/Hue03vu877d3uEt76n7q3/3Of/zcc/z3nPuPS9RQkJe0shaYfV63oQXB0UJ64e1yqpidbJ+WdOsF69dUCCBdmuS+FPWDIENkslnAj6eYnBg4nFJBYlLat5WhMwcpRJzWo60CJQRSk/sItIicLoofp/lPf3WUDYpf5LqsUYcfaxRayqTlD9J3VojjmPWrjWVb4q+LIpZX6xx1r165xRdpi5e9+Ix1hHJeAPaDgyx9lkHrBn1XB/4uNaRnHKcn9WP7BpWGH+H0o9GaFek8SzJccqvA8+eB+CXmXIlq1pj3/fjcpIV5D76WSXjuGYVsp5IOj7qdcNr44B/pcJyuPHqOrS+wPOAncwna5Fkv35Qajw8RRykge0DMnk5IW7gZtYJyffOx/Z5Zy2wllhnps5h+wDnNfhmLrhktXnlbb3iSe9pjJvXauzKNaacKW4iWXLWd8DD0u+2FblgjWSp4gUBEN+RPAGAfYjym5YxmUHWIcmGL1UfYG89sE5Zw+q9kvTHuPXqgVaSe2x53r+R6VcPmhaSpPx/0AkJCQHxB7JDhVJCe/FCAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAAAaCAYAAAB/w1TuAAADaUlEQVR4Xu2a2+sNURTHl3siSijk+uBSkiQ8HsmLB3cll0IpPEgUUigehAfJJZc3vHjBk0vKLxS5xX9wErmmXCP39bVmfr8968zs2efMnMtv7E99O3O+a8/eZ8/ZZ8/aew6Rx+PxtBI9WYu16algHmukNjsz/Vh/WC9ZM1TMU0mJdYPkms2OhmpnFKuszQbxk/VZmzE81EYn5jXrB8mXCHWNhmmfEYM+RsP/OE0Sy4V3lGNlVYJ272sz4B5FL0SRmMy6RdKvayoWYuvzGrLHG8YK1npD46LhVNAJXAgbm6lFOpsjF1ndKXlwd2E90KbBSoo/ryEMJmn8OGsqazRrCGsgq4dRzgXU06ZNRREHQNifY8HxBiMG1rHmKM9kKeV0TfazVmnTwiLWbW1m4H8dAO+N47hZ4IV6r1lCledUzbfgFRXNNAMWkK3nCdq+rk1F0QbAJNYW432ZpH8jDC+tvwsovYyVk9SRfaIilyXFXpJlW170Jml7rg4oijYALrC6Ge8HkPTvreHdMY7jQI6AcxbqgCs7g9dD5H5xn5Lc823q3146mUGsHSTt4stNo2gDIK4v8EIfybTLD3ICyTnnguOaQAVPtJkA1uz4tdo0tL10MlgpHCFpe76KxVG0ARC3rg/v6fgy36hYEkjGv5JsDM1SMSeQtaNRVOTCCZJpOy8wW6D9kvI1RRoAU1jbtRkQzgKufUW5ZdqshsfU0dg21jAjlkRZGxlB+1mTwK0ka+okMCshabKxSxsKbNxM16YirQ5wheS5RxxnSPp5UwdiyJwEAlRwNTh2nXbwwAZJTF7gM2AKs3GApBxyB81GktgnHTBI+1UdJInjB5FEWh3Y2EH8sg4YjCUpg9ckEC9pM4ZcloG4b6AS3NurYSLJebtJNoH6RMNVgXratBnwnfWK9Zz1LHhFpnzWLEQyK9n6gCn3kjYNMHug/uE6YHCYdUqbBsjk8dmSvpQvJGt/bLsjB7gbDbeDRNuF3DaCsoB7+CaSgbCHZJk4LVIiHXTCZcpLo+kXI+CDNuoEtuBbpc+ZQCfw0CcLvUiejjUbTO3LtVknWuZhUFbQCXNbtBZ+aaNJ/NZGHTlKBRkAYWL0iGSJVAt9tdEksDtXb8azzpNcs0L9gwo7Waup8s8RnihrWWO06fF4PB6Px+PCX6rCzHIL7lhSAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAABGElEQVR4XmNgGAVUAoxArIouSG3wDoj/QzHNwWUGOlkEsuQiuiAtAMgif3RBaoMgBkSwgSxbBcTRCGnqAVCQgSz6A8SaUDEQfxlcBZUAthT3F4sYxQBkYBgWsYNoYrOg4vjAe3QBGPBjwNQMyrwgMXM0cU8gvosmhg7y0QVg4DQDpkWzsYiBwEkgjkMXJBaADLyKRewWlP0STRwXqGeAmANLTBgApDkYi5gXELMC8RY0cVwgHoh7gbgfXQIExBiwa17EABE/hyTmDsS3kfjYADazSAbHgTgGXRAJCDJAsgTFAOZaUOELA6AMDgMHGCClSR2SGFmglgHiKyYkMVBCiYSybRkgicEeIU1dII8uQAtwB11gFAxNAADEQD9uIvoXLAAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAdCAYAAAAKGSQrAAACcklEQVR4Xu2YTYhOURjHH58pdpJQTE2WSlFixULjK0mzIPnIwoIFCmOjrDSLSZSVlZVkZyeFBUVRiKWvMI2N73xlMvP8O+cx//nPfc0073Xdt+ZX/97n+Z9733vOvefcc+4xK48HahAX1GgFTrhmUd7jGqB8iWsb5S3BBzVseKOK8lpz2LVVvPWu5+Kddi0Tr7bcV8O569op3gLXVfFqS1G3KvLAOzXqSlEDijzQyP+nXLN04bEo0Ip2uJ6KFzxRo0q04sx8G172jWKAG7PHdUv8ua6z4lXGFEuVfqQFBDeqy7WJ8kWuF5aeGNPtWiVeZRyxVOnN4h+iWJ/iW8mL0HMq5ZONrMAG1y7KV1AMLrqmi8e0uTrVrJIYTytda1zHcj4aOtkyV9SokhhPGBOnXOdcb7LXsmDQowHrxH9G8WzXJMprzxcb+VSmWRpTwU+KW4IYT42Y47qhZp3B2wsNuqcFBMqnqtkkf7uJTXPGiucnsMVSWbNdD58kiq48SuGyq9/124a6Hwv+L9d319J8zngZyyT9X0F3/eo6aKnBWBMG8yzdiJM21L30pgGchxgvo8lUFvMaH4vx/N7Sfz7OXulggdqb46hQwDHeqDEFsB9EozgPXlHM/h3XDspLZabrvKXVRFx0I8VKka+Nwpy4N8fH82+7peNeZ2FPBEuz0jnq+kx5VHg1xUr4beJxo/DZD28feegVjf6zVHCR3TlemPPtVDYjx+h6y8nHUmxtzsPjRoXXJx7GKFYzwX6KSwPzFVb0t10HXA9dN6n8paUBjfVjgN2kH65LOccUgX2Lj3+OSGDfcLF44Lql7ocP0QkmqCOD+Bukq/HLUVoAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,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>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAABy0lEQVR4Xu2Xuy8EURSHj1chhFZJiEZC4z9Q6CVqCZUo1HoNOhpCPBOFCAmFiBCtUkeERoQC8QgRj/A77l175+ycmVmJGcX9ki+753dmZ++cjGuWyOPxeH7PLHyEn9b5QNfwQfk+2xFsp0o5fJWhoB5ekVnrRrCVDPdiw9iDTTJMkSOKXyMzCd+ceg1uO3UsJXALrpP5oq5g+5uoBaTJA+lr4TtW9ri+E1kkg7Ddvtcm/y6DjIgaBucnIqsTdSw3znueIp+0xsma4YhTZ4k2DF4v5xO27oQV+XZy3JPzvsD1sZMtw2qnzhJtGP1k8iG4CqvgPsVvtgF4v9gUmfxTCftyjVa4pLgIF+Acmf9iM3DafCwx2jCmqHDdDNeHIlNx9ws345OM2bqo6f4x2jB4rZzzXeySeyxIxK0MLLkpt8Bh0csSbRg9ZPI+kWvHh6IduEOmdwYrRS+KBjhapMWgXVwtmZz3Dpcnm8fCT3O7MrSUUv7u+E9ow2A4531IZtrxP5TBa3ggGw7P8EWGGcNr0i6ulwp7XEf+fFiB92SeL/i5gjeZMNrggAwzgn8/XcJz6wWZ/a7RPQiMkxnAqX3tDrY9Ho/H4/F4PKnzBT7di8eZdM10AAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHEAAAAdCAYAAACdSIPxAAAD40lEQVR4Xu2aR4gUQRSGnzmDooggiCJ4ERHTRcSLBz0oIopgwBVWUFDwYgLxqAcFMeDFHDCg3swIevCgYkAUFTGjqChGzPn9VNXO2zddPT0z1bO9S3/wM9WvuqtfzauqrqpuonSZzeqijZaX2pCTPfqwForjeaw/rBnCdkOkczLIL21g/qnjk6xOypaTEdqzbmojFQexO+uYsuVkhImsOmWbwHqobEAHNqcKBmlDFaxnDVO2S2QmOppaBHGwNrRUQv6ZV1g9lc1Xvs8eklrcIxP4KjqaTF4SYRgFR1lDbdrhK99nD0kt7hFHPesLGT9OqzwfFflc6qJPFH8O8jBRAStZM0UeGsIT1jhhc8SVGYpa3MPHNSrMD9pQocF3aDijmAdUoc+lLnI39/FDpNuyropjgJZ4Stm6ss4qWxrE+Z0mrckstfoJGzY/4M9PYZOMYT2lCn2Ou8i1oOvKvkOksZCXxJXnOEKm7LRJ4ksaYEYe1fijbI7frEXkz48l7qIlZPInKfsjkZ4q0gDPxKjZqAQToFoQV7e0QUMfoGy+IN5ldaSUgviRTD66OjSXTIupE+dEsYzVThstr7QhReLqVms6k/HnmbJjSbbBplMJoms5a8jc6Lw9bi4k9XWfR3tZe1i7WDtZ21nb7DXl8pWi/fkr0sGDiEkK8rBYl+jzQ24WhEb72lQsJuOLngdcZHUTx8GDuIJMHh7SEvRGxxYyby2yiq9utWQENe5tjoGs3coWPIifyZ/nKJXf1CT1b22ZSkov1jdlc8/E6awLSq/J+OyOE+OrKOy+PHCZtUAbq+Q7Fff8aojzP20wdL7XRor36QDF53uJughDJOx6fQiwhPhA0deVi3x5DA6S2QgIRQgfK8V1Aq2oodWBDRCcg82CspAVHU5mpwE30jd3DmCJgR5zyF5TKViG6CCGpqmCOIuK/zunM+I8xxTWG9ZzMsPtCzI7XYkJVVGM5wjKffsrwfR6HesxqzcVlipOYLlNr7bHOt+l+9pjPGuWkml0raxNE6pumSdURdFDHbJM9NohNn2CNdmmEUgd7K1UCCJAOf1tGt/qOLAp795bjifTcKIIVbfME7KieIuB56gs01d+VBA3UeMgziHzFgS8E3aUiWHHyXcPn73FEaKiPciUg/0/kCSI58gsggFeWYGN1DiIANfjQY/ntbQlIel5zZ6R2lAB+CxDboq7P28UmaGvXuQdt7/7Watseqz91T0R4MMrlCE5TGZz3nFPpCXY780pg9usW2TeHc4nM9tyoNe9JbNbL8Fk6I5NbyYzZELTGs4wn3sg4BoM3XiXiZlcTk5OTk5OpfwHO2MkXq3MWa8AAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABHUlEQVR4XmNgGAW0BPpA/BaI/wPxCSAWQJWGg10MEDXvgTgJTQ4FZADxJCT+EgaIRiMkMRAAibFC2ZFQ/iuENCoASYIwPrHtQLwBiQ8CWxkgavzQxMHgCQNhQ/9A+SFIYtpQsZdIYjhBFQNEsReSmDQQL0Lig4ADA0TdWTRxDBDAAFE4EV0CC9jJAFGrgy6BDHqAeDUQ/wViZzQ5dMDMADHwNLoELgDyKkjDFnQJJPCdgYC35dAFGBARBbIAGXBAxZmQxJSR2GAACg/0mAYBmJgCFnF0sAddQJ4BovACmjg2i/4BcQMDJHXUAHErEB8G4jwkNXDwgAES4LwMEG/B0i07kpqPUDFsWAGhDBUoAfFKBojNJWhyo2AUDCQAACcTSg4ylmqSAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAZCAYAAABOxhwiAAABh0lEQVR4Xu2WPS8EURiFrwiNkBCiQaEhEj9Bsnqdzi9QIlEIiQKJBolGRCESjU7rB6iVopUoFITEV3yfk7l3MvfMzO7MFlNwn+TJ7p737s7ZyZ2dNSYQ+B/MwxM4Zl+PwmM4F6+ohms4BTtgH5yF794KYR3+iBfeimrQDrTdWyGswl14BFdgqz+uDBbdhjtwXGaZsOykhgWZ0UBog50a5sDipVg2zRfvh48aWrhX6+5RoXTxJbhhojce2sd9b0V9BuCTZCz9IVkjeNwHeA6vTP4JiVmAZ5LxQ9Ykq8cgfLbPWfozMSuKnvH7jKwh7qougyvfTOks+HPIDhM6cLRoAL5M+eI9JtoerzpokpqJOuxJHsPhXUZWprgrTXrhS2JWhG+TPh5vRsx4n8mEw8WMTD8oj26TvhBZ3u35IvBYN5Jt2nxI8hieHd5iHTUTvWEkkeXRBd80tLC8/trkMQ1PJWOHS8lScKu4s0yH/XEujf7P8ObDL1CELRMd+9Y+HvjjQCAQCPwVfgEnWFrN6coDBAAAAABJRU5ErkJggg==>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAaCAYAAAAaAmTUAAACEUlEQVR4Xu2XvUscYRDGh8RUiYJpEhEsjX+CgiFFUtkE0cpCBQmIRRDslYAQySeCZcADFREFQWvBRrFRLEQkjQhWoqhBxRiimYeZNzeMu8fd6XmH7A8ebt95dtmZez+XKCEhIVceso5ZV0b76tWxTp33Uz2w67w24xWVM5KEogjJRtHN6vPBYrNO8QlnKuaPD5QC0yQJV7l4O+tSPc8I66kPlgIfSRJ+6eLnrCX1ypy36dolQxdJwp0mNsEqZ42r98J4G+a65HhFkvCgia3q7wf1mrRdSTLECk2FD2RLNUnCU9reMV6Heu+1fWG8KJpZv1mfvJEDeMdjip6rWYEH0RvPWUMm3qjeMOsN663x4vhG+RfzmrWs10/0FwtUTiDhX6y/Lo4VDt58hBfHF8q/GPTsnGnXU57FQPj3PcF75g3DGGuRZM+aoXQx4dkF/cWQBtijelgHJL0PavWeoAHXtotQRnBz3HyAt+2DBgyrSdPGCmh7Bs9jDvSyHmkMR6UA/AB6BttBAAXl1TNYqaKwL4sCPoZDIEXXi/E8IBmO2K9uvZibgGTCUAEp1mfT9sU0kJwsAr6YsACAftasXr8z8YKBfSe8EKywvpu2L2aNNarXOLnDD0epFpLnAygA8w2gsDvhB8nk36P0qeEr64Tks+KIVfP/bvmEQA9gyU+RfF60sg5JFgUcpQJbqoSEhIR7zj8Uh5F+PEyn+QAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,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>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAdCAYAAAD1uwHJAAAC6klEQVR4Xu2YS8hNURTHl2fEgJSUiZGJJGIimRgwIAMmJBRFGZh4lQyZKHlkogxFmHkNFFMiiZC+PpRCRMgzz/W31vats87Z++7v3nPvx+3+6t9d57/X2Wefffbd5+xN1F7WsMZ5U3nujW5kCmuLOd7I+sFaZbzbJu5KvnmD+eWOL7LGOq9rGM26400qd8IE1jnndQ1LWeuct4TV7zzgO2ZIme6NFjjAmu28ayQTpacTnTDDGzHqbMwN1iTnxeqP+XWSfY1Y4nySshzhbwDOsmZpHIjVH/PrJPsajRLfUzoHZZjowG7WalOGjnzCWmS8QKrOusi+RqPE8LRjfDXxSNZNcww+si45bzzrsvPaQardBVKJI0jKbzn/uInxIWRJ1Rc4Q1J3u8lpyx9SidtIypc5/5GJV5gYYE6oehtYMIF2gtS9FUglviMpX6Baz/pO5W8Bzw7WKG8qL7zRRlL3ViCVGOaDfayDrKt6/L+Q3dZYIiY5lOFjx+Lz6/zYqhvf1iixxF0kZfjstWA0BI6SrBr/VWL3ViKW+IHiZYFG5UNNdvtiifBjZeA6a7M3W+QLlUdeK6TaX6AqEUMcvv8+AHgFvqXq8waL3XwBJ0k+pOoiu402cQ7JpshP9b3g4xWJJ3ZKz2kWvEZ9J9RNU53QCi9JbqpPfy2fWPtZj1mTaeBVGwR2arxXj315iKfq8WfWdpKHNkw9T/a9ZSc2ACMkYOvEqJmp8QXWco3REb6zjtFAJwDUM01j7FUGsKgL+xaLSTq+iux7y07MAKtIzCO2zlj9VZ1wmIqdsJZkFQreGB91PjWKXSPml8hOTDCRpJ4xepzTCVdYWzXGkhscomInAJw/nGS+sl4OuXk01xtNgG01u6gKF59HMnQ3mLLz+nuCtUfjhfrrRwLAxi3qsJwmWdwFHprYgvVOR7nHukuyd7CJ9cqU4am/Zj0wHsBkel/jIyRDHlr5N0O269BhHvz1sJfxzBf06NFj0PwGuqTLy11uX3kAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAaCAYAAAAue6XIAAABk0lEQVR4Xu2VvSuFcRTHj7yMFlkUN2wsEslgsEmSv4A/QMkgNpmURQaTxa4opcSgTAaTRRkuidFr8i58T+ccnft7PDeD63nU86lPnZffo3Mf5/4uUUbG/2YbfvzQ1FBsoDqK7/055STDHIQNR2qGnSAZZjCoj7s4NcPeUnSYfjjs8i4XJ4rtazfshZOapw7b1xM4CxfhudZSxxTJYH1BPe/iGljm8sS4o+hbrCTZWePZxYlS7H5lauFOWEyCKpJB98OGg/sVLmZb4Qp8I1mRPbgBD/Uc0wPX4Ry8dHX7G/aCLM59nYhhgeRgeL8yQyS9cAW4NqoxD+T/KxxXazwDpzUeg7saM3wN3mu85erfYm/lnQo/qcn1V/gI2/QZww/XSXLO4F69y1vgGryCx67ObFLhsyXBD9sOn1zOvQaNz+CqxgMkV6NnmaIf7tfxw3bAF5dzL+diu+747j6FR5rzc7wK9p0pCbxnF/CGZAWuSb48DyS7zTHXGkn2mdeJh2zSMyMkP+18bgk263nO5ykjIyMjlk+Pw3gXwH4wQwAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAaCAYAAADYMiBQAAAChklEQVR4Xu2YTahNURTHl/BmTKSknhdvIgwkegwMlCTpZWbG0EDJQGQiA8kz0FNGeD0fM6KUEgNlZKCUkjLwFZJvku+v9/9be1/rrnvPse/t1nXv3b/619prr3Xvuevss/Y+VySTyWQyncoV6Heiup350HPvNByAvkM/oJNuriHKCjpbiuc6nTXQS/n7+19UT1d4Dy0J9gwpr1cpk0UTb/kJQ1Mf3GEUFXtIdMVPM75lovE3jS+JHaKJw86/3di9XGy2D87ddv6mVjcfEZ+0Dtpkxry73U5RsclZaLrzNVXsmLQCWgXtDONeo6zYHtaK8bwJycR+/QDaDx2BngTf/8ZG6HSBTkEnoHFoDDoOHf6TlQ5/MzfLFBj7yzv/xS7RxLXOf8/Y3H0nmXG3wjq89s46nIN+emcKH6R2FU8V7dmRr8buZliHN97p2AK9885UYr8uYiZ01TvbxGroYAPap2nJsA5lhVwOPXS+stpV0ScafMNPGDg/xdjUQuiM6JsUW8x16CJ0J8SRldAFaESqH834GfEioz1QiWgfvA6ezOoxS+q/hyQXe1Q02J+vyQbROd9C6NsabBbUfhnteDzaC+0J9jboWrAJj5Efg33Z+NsNr/+zd4ouNrtIrLjQSomrkrupT6bo538A/OLFISdii8u3KMZFONdvxgug86J98L7xk0tSndsuFole31PocdAz0b0sckxqaxS128S1HFts/lfwxYw5NyfYvGju2mS96NHSMi61NyfjsMVeCn0zY84NGDseF3l2fwTdDWPmsZXEPSNTB/bZV6I7NlvIW9HN75Nob6dN31zRfs52xCLPCzGbRTcgxh2FBkM8x4ckk8lkMplMJtNiJgABb83sKTaLQQAAAABJRU5ErkJggg==>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAAsUlEQVR4XmNgGAWkgi4g/gjE/6H4OxC/QxO7DldNBIBpwgZ+MuCWwwAghYfQBaGAhwEi34AmjgEiGCAKHdElkAA+F8PBNQbCiogyiBhFxKgBKziALogE3BggavDGHix8HNDEkcFtBogaMXQJZHCTgbCTQfJ/0QXRAUgRKJ3gAiD5J+iC6ECFAaKwGV0CCOQYIHLr0CWQQSAQn2RAxMQdID4OxWehYqBsYgrTMApGwaAHAJc1Nr9vaqJpAAAAAElFTkSuQmCC>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAAA/ElEQVR4Xu2SPwtBYRTGT7EaWJAPwIcgi8liN6AsJotPoBSr8hEYZPQZjMogg0XKaKFI5M85vefqdHrv7R1kur96ct/n13O73Qsg5B9EMCfMW+TILoe5KLdlRxyUqwr35QpG2vCGNlqYji4lK/AfB934oQvNDMw4rfoa5sVOM8IkdKnpgxkXVH/DLNhFlduos5UmmHFDdBNMDDNmlxVuLa4DKYIZ90S35N8uuzKf42BegxMZMOMpn/fC1dm1+XwXzgka01OmMAPR59kNMSVMRTgnaHzGPFVP/xRyc4tzgsYUeiqN55JauEBDv/dHbqdLV2hMX9wGuZCQkF/wARnDRfhEUQ1vAAAAAElFTkSuQmCC>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAAWUlEQVR4XmNgGAWDFnAA8Vcg/o8D+8AUMkIFFgFxBRDvRGKXAXE+TCEIgExkQuL/QGITBCBbiAIsDCQo7gfiD+iCuADIVJAGogBIsQ66IDYgz0CCe0fBIAEANQ0VXRjhPe4AAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAWCAYAAADQIfLaAAABp0lEQVR4Xu2XzSsFURjGX8k/oMTaxsfaH8DSDmVlR8lGIRY+ysJH1nZWUiSliFJSLER2UhQL2YmirJSFj+d1ZnTm9Z6Ze5XNzPurp3vnd8506znTOXOJDMMwDCPJMLKBNEfXjcgqMvQzI5025Br5RNbEmFEis+QK9HOemBFmBPnwrgfI3W+UyTSyiKwgU0hlcjgVLrxJcQvCGRlw8byllEsH6U/9G+leMiiFoEWKPDNJf1uEQ9LLviPdS/gsWpcyohM5kjLPjCNz5Ipbjj6XEjN0Xkgv+4p0r8G/vSVcF3IsXO7hw3VfOC5xRjgJz9HKviDdh5hAtqPvvAAn3lihCRXsc0/6nEvSfRq8EKfImRwoChVSgHfKLjJ0JtyS7tPoQ26QPTlQFLiwZ8VlFckHujan1LejGF6A+Azg/xk73lhh4MLGFCeL7EZqheM51YrbFS5EL/0+hHkh4jOiMLwiNd51K7kiGzzHW5a2MA/kXklj6sjNqfJciB7kQMqIfmRTyrzD21FcMqc+OfwNP52jUoIn5JHcfh66V2NeCkG7FIZhGIZhGP/KF4PyZT6UPOX5AAAAAElFTkSuQmCC>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAWCAYAAADQIfLaAAABhElEQVR4Xu2XvyuGURTHvwb/AAPDK2Vi9w8YbSiTjZLFQBbs8i+od5WF/CqKYhAppRTFIIMSZbAYLDine2+u8577dFOW955Pfes+n3Oe5Z6ne98XMAzDMIxquilfUlYwQLmFe2dV1Iw/wpuZO4RZymf0PIX8d40EB5R35G8k9/Upblk4I5MaZZvyirwhDEHv+4DuJdNSCPqlKIGwcblDOILe9wDdS2Yoa1J6hinHUjY7G5Quv84dwhv0vhvoXmOesincCOVEuKanHe6rDuQOgXu0vivoPsUCZcuveQCnUa0Y5IblDuEJet81dF8FD+KMci4LJbBC6RUudwipO+Eeuq9ignJH2ZOFEtiHO3/jhGOG1/Wf1gYWoW927q+jAA8g3AH8P2MnqhVL6qwfpXQIx31titsVLsU4Gi9hHkS4I4pFG0JLwj/D/SQNdML1tEYuxRjlUErPJGVdyhK4hNvURx9eX0R1/jrnoucA3yEvcOc5D6DndznJkhSCQSkMwzAMwzD+lW/igGfO4d69mQAAAABJRU5ErkJggg==>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAWCAYAAADQIfLaAAABkklEQVR4Xu2YsStGYRTGH0qKiRJls9mVgcVoQ5lslCwGsmBUsiv/gGRRRFkUg8hKFINsogwmYsA5vd/N67nn3q7B4j2/euq7v/O+y3m/7zv3XsBxHMdx8vRLniSfkjNJ/c9yKQOSa4S9G1RzKrIqWYuuXxAa2hW5ImYlH9H1FMJe55do03oNV6WZuqbbcCvknBKaYTfccswQ7DVvsD0zzYLoYfGfWZL0katyCIew19zB9syMZJNljWHJEcvU0CbG//UWz7CbfQXbW8xLtsmNSI7JJccFQhObuEAU/VrOYfsiFiQ7tc96ACdRLUl0QGsD27hgcA+72ZewfRl6EKcIt8dJ04LQvEYuFFA0E25h+zImJDeSfS6khD6ccePW6ZpZRH6PUvXuKEMPIJsB+pyxG9WSwhrC7EYl7eS02a2G2yNXxDjyQ1gPIpsRyfCO7yHLyagznPKAcEua0YGwpiFyRYxJDljWmJRssfyvdCLf+Cyv0TpFv51z5BR95/SI8H+u+6q87lCWWRCDLBzHcRzHcf6UL3BfaRKy4BSpAAAAAElFTkSuQmCC>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAXCAYAAACrggdNAAABeUlEQVR4Xu2VvSuFYRjGbx+rxIKUpA4mE1mIwUSHXUJZTBaLYlGKVZTBiEFGf4NRnUEGi5RFWSgfkY/77r7ferp6j96P5y3y/Oqq81y/zvPe57xfRIFAwDd1nAfOl5N7c72cJ3BX5oRbcNOOy0sNFll4Jh0sjmjoOBY5y1jmYI48/kEVqj74Tz/qHYuMrJIeYxJFHk5IN22DfpbzaQ7Z5TRjmZI90v37Ufhgk3TwYehfOWfm6sFdwjoNp6SXfAcKnyyQDj7vdEecBs6huR7HXTifk1LLOefccRrBFcII6eAbTicDCOvmxm3dRHrppeWA88LphL4w2kkHP7b1jeOiJ9KSrd8cl4Ud0vtoAEURyOBydlo5W04/ZG6bM8aZclweVkj3LaPwiRzgkfMBvTwRxcnNjc4HM6T7yzvPO7KxRM4GErkWFB4ZJT3GGvS5kA2r3S/irrEsiG7OPpZZkcHlyRaHuMB/oY8zkTAl+86vp4szmDDy3gwE/hrfoYVd93J7UYEAAAAASUVORK5CYII=>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAWCAYAAADQIfLaAAAB0klEQVR4Xu2YzyttURTHFymJgRgYMGUoSUlhwoAh8QxeKEqvpCgThpKxzJSBMpBSfpVXpDcQKYYM3uD9A+byUmJ92/u4+36de++5yuSe9alv5+7P3mey1rln73tFDMMwDOMz1ZozzZvmVlOWPZ2Xfs2LuHt/05yRkEZxBazy43o/Lv9YkZsBzVYwbhZ3r1EkT5o9cnea/+TiiCv4vmaTpZEfFPIHuWXvC4E1teQONNvk4phjQXSwKFV6xRWym/yk93XkGaxBZsklYUGzy9IzpPnDslSZF1e0dvKj3neSZ3ok04hnf63JWpGfJXHfnJBhzSW5kmZFXOFayeNJhP9JPg68VqJGIKvZ0wXBq+/Qf0YDroK5VDAjrnBt5Ee87yPPnGgu/Od1yTRi4mNFMtCIa80NT6SBaE/oIj/uPY6vuWgRtyYEx9yoEcUwrfmrOeWJNFAprmBfOR09aHZYSqYRSUEDoj3gl+Y4mEsNKNgGOTyRXEhs1g3B+EhzH4xD+N5cTMnnTRiNiPaI1BD31GOMTTICf2Pwa6bCj5sCB/5pxsjFgU3/nKUHexV+9KUKnNdf/RWFxdGVwdO5SC76ywN59FecrJKwxoIYZGEYhmEYhvGtvAO/GmXFYRSfPAAAAABJRU5ErkJggg==>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE4AAAAZCAYAAACfIRhSAAACIklEQVR4Xu2XO2hUQRiFfzHaSIKKQYMJgo1inVIhliIEjJ12EUKaQBKw8IEpfHQW2tkogYAEhSQWglgohIhWWgRCUqRLZ4pAfOD7HGYmzJ69e/du3IsxzAeH3Tn/wMyeufe/d80SiURi+zMCTUIn/Pg4NAENb8z4P5iBfkMLUIvUSuGWuQVjva+YsbXZa27PPHDS4ce7N2aUxBh0HxqHrkM7K8tbnnDYMfPQT/GaDsM6rWZBLqgh7IJa1WwyDG1JvLveV4bUELrVyOOabT64g9Camp490Dc1m8wRcwF9EP+G98PtG2A/fyxe4Bz0Ss08rkC3zS30yH8+qJiRTye0Lh5D+y5eWWRdcWw79M+LT/h7p8Trg2bFq8so9EI8LnpTvDy6oE/+O0P7EdXKhnvVfvbL+5fFD1yFpv13hjYX1f6KrIZbjxDeZkNjfymqmMPm9nrIj09Cr713yntZMLw30FstFGWHGuZOsNHg9pu7Pb9ooSC9DUjhQ+gJtAKdhR6a23/e+9wlaBF6roWicIHVDK+R4EJo5AD0Oar9C95Z/v4ZWuhpg9CzqFaYrF7QSHD7rPpBwPBCzyubZaveK8f3xAv0W/WDgOGFnlcYXh3t0bjH3MLHIq8WbdBXNT0MT5+2ZcDWEAfHNwINMnAReqmmZwB6qmY9eKuGq4w6WlmuSb3/s3z5ZYBlwsPjnj+a6828AmtxRw3hjBqJRCKRSCQSie3DH+Y9eWLO+UOAAAAAAElFTkSuQmCC>