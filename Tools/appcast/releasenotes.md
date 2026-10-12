# DynaMoE v26.5
> High-Performance MoE Inference Engine for Apple Silicon

---

### 📦 What's New in 26.5
* **Conversation renaming:** Rename any conversation inline from the sidebar or the active chat header. Titles you set are never overwritten by auto-generated ones, truncated auto-titles are marked with an ellipsis, and both survive `/compact` and relaunches.
* **Session token odometer:** The chat header now tracks a live, cumulative token count for the whole conversation: prompt tokens plus everything the model generates (skipping pinned-prefix reuse), ticking during decoding, persisted with each chat, and reset by `/compact`. Conversations from before this feature pick the counter up automatically on their next full prompt.
* **Inline agent activity indicator:** A pulsing "Working..." indicator now appears in the message stream while the model streams a tool call's arguments or the agent executes it, across all three supported tool-calling dialects, so long tool turns read as live work instead of a frozen transcript.
* **Chrome polish:** A single centered app name in the window title bar replaces the duplicated sidebar branding, and the redundant header breadcrumb, sidebar toggle, and thumbs up/down feedback buttons are gone.

### 🐛 Fixes
* **Chat actions:** Copy and speaker buttons no longer appear on thinking-only or tool-call-only turns; read-aloud is restricted to assistant replies and speaks only the visible text, never hidden tool markup.
* **Tool-stream rendering:** Raw Qwen and Llama tool-call markup no longer leaks into the transcript mid-stream or reappears after a call parses, and tool-only turns keep their time-to-first-token, throughput, and token stats.
* **Title persistence:** A conversation renamed before the startup history load finishes is no longer reverted to its previous title.
* **Token counting:** Stopped, cancelled, and failed turns now commit their final token counts so the odometer never loses the tail; empty on-device responses no longer add a phantom token. Covered by new pure-logic unit tests.
* **Keyboard focus:** No stray focus ring lights up a header button at launch, the rename button no longer steals focus on first open, and Full Keyboard Access users keep their focus indicators with no hidden tab stop.
* **Model guidance:** The 16 GB settings profile now recommends native Ornith weights, agreeing with the README's preference for native weights over MLX quantizations.

---

[🔗 View Release Tag](https://github.com/derekrparris/DynaMoE/releases/tag/v26.5) · [📁 Repository](https://github.com/derekrparris/DynaMoE)

<sub>DynaMoE is an independent open-source inference engine. macOS, Metal, and Apple Silicon are trademarks of Apple Inc. Apache 2.0 licensed.</sub>