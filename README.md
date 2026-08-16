# DynaMoE

*Dynamic, SSD-streamed Mixture-of-Experts LLM inference on Apple Silicon.*

DynaMoE is an all-in-one native macOS application, local inference engine, Model Context Protocol (MCP) server, and OpenAI-compatible API host. Inspired by research like *Flash-MoE*, DynaMoE is engineered to run Mixture-of-Experts (MoE) language models that far exceed physical system RAM on modest Apple hardware by dynamically streaming active expert weights off high-speed NVMe storage directly into Unified Memory.

---

**Key Features**

* **SSD Weight Streaming:** Memory-maps MoE checkpoints off local NVMe drives, paging active experts into Unified Memory on-demand to run massive models on constrained RAM setups.
* **All-in-One Local Harness:** Houses a native SwiftUI interface, a local OpenAI-compatible HTTP server, and an embedded MCP server inside a single macOS application bundle.
* **Pure Swift & Rust Architecture:** Employs a high-performance Rust core (`dynamoe-core`) for memory management, file parsing, and async disk I/O, cleanly linked to a SwiftUI frontend via UniFFI.
* **Apple Silicon Optimization:** Designed around zero-copy memory handles passing directly from the macOS kernel page cache to Metal GPU buffers.
* **Open & Modular:** Released under the permissive **Apache 2.0** license for transparent community collaboration.

---

**Architecture Overview**

```text
DynaMoE/
├── DynaMoE/         # Native SwiftUI macOS App (UI, MCP Harness, OpenAI API Server)
└── core/            # High-Performance Rust Engine (mmap, MoE gating, metal integration)

```

* **Frontend:** SwiftUI / Swift Package Manager (SPM) / Hummingbird
* **Backend Engine:** Rust (`dynamoe-core`), `memmap2`, `uniffi`
* **GPU Kernels:** Metal Shading Language (MSL)

---

**Project Roadmap**

* [x] Project architecture & Apache 2.0 license initialization
* [ ] UniFFI Swift/Rust cross-language binding setup
* [ ] Memory-mapped Safetensors/GGUF weight parser & expert router
* [ ] Zero-copy Metal buffer handoffs for GPU execution
* [ ] Integrated MCP tool server & local OpenAI REST API
* [ ] SwiftUI dashboard for model configuration & memory monitoring

---

**License**

Distributed under the Apache License, Version 2.0. See [`LICENSE`](https://www.google.com/search?q=LICENSE) and [`THIRD_PARTY_LICENSES.md`](https://www.google.com/search?q=THIRD_PARTY_LICENSES.md) for details.

