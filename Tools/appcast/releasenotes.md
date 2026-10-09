# DynaMoE v26.4
> High-Performance MoE Inference Engine for Apple Silicon

---

### 📦 What's New in 26.4
* **Apple Foundation Model support:** Chat with Apple's on-device Foundation Model, with live activity indicator, smoothed streaming, and a per-turn history budget so the transcript honors its context window.
* **`/compact` conversation compaction:** Manual `/compact [focus]` slash command (opencode-style) that folds the conversation into two AFM digests — a rolling general summary plus a detailed recap of recent work — injected into every later prompt on both backends. Transactional (a failed model leaves chat untouched), chunked for long histories, includes tool-call records. Auto-compact on context limits is deliberately deferred to a follow-up.
* **Settings glow-up:** Reworked settings into a sidebar-driven layout matching macOS System Settings: sliders + editable numeric fields replacing steppers, shaded group backgrounds, three-per-row sampling controls with precise inputs, and an updated settings guide.
* **Composer polish:** Chat composer default height reduced by a third; minimum fits one line with trimmed chrome.
* **Gemma 4 26B A4B support:** DynaMoE now loads and runs `google/gemma-4-26B-A4B-it` end to end: architecture detection, the interleaved sliding/global attention layers (per-head q/k norms, proportional RoPE on global layers), the dense+MoE decoder block, and the `<|channel>thought … <channel|>` reasoning template. Verified element-for-element against the HF reference forward pass.

### 🐛 Fixes
* **RAM telemetry:** "Process Heap" mislabeled RSIZE as the Activity Monitor/Xcode number (actually`phys_footprint`, overstated up to 5.2×), and Working Set RAM double-counted weight bytes (~12.9 GB shown was really ~5 GB), evicting experts early. Now shows genuine footprint + a new "Unified Memory Cache" line; budget decisions use RSIZE alone. Also removed the pread-into-read-only-mmap prefetch path that was a silent no-op (EFAULT).
* **FP8 slow at long context:** FP8 KV-cache decode never got the FP16 path's flash-decoding split-K kernels — one thread per head over the whole context. Added FP8 chunked scan kernels for Spark and Ornith, and disabled JetSpec under FP8 (its verify kernels would corrupt a 1-byte-element cache with 4-byte writes). Added fused FP8 attention benchmark coverage.
* **Agent tool-call regressions:** Fixed three tool-call regression causes from live runs, two Spark agent stalls (whitespace loop, missing turn reopen), KV splice relayout corruption on head-count padding / stride shrink, FP8 scale buffers going stale on spliced stride changes, Spark's end-of-text marker leaking into saved replies, and web_fetch now loading by default at session start.

---

[🔗 View Release Tag](https://github.com/derekrparris/DynaMoE/releases/tag/v26.3) · [📁 Repository](https://github.com/derekrparris/DynaMoE)

<sub>DynaMoE is an independent open-source inference engine. macOS, Metal, and Apple Silicon are trademarks of Apple Inc. Apache 2.0 licensed.</sub>