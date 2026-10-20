# DynaMoE v26.6
> High-Performance MoE Inference Engine for Apple Silicon

---

### 📦 What's New in 26.6
* **Local inference server:** The loaded model is now served on `127.0.0.1` through both OpenAI-compatible (`/v1/chat/completions`) and Anthropic-compatible (`/v1/messages`) APIs, so local coding agents and other tools can use DynaMoE with no external service involved. A new Server settings tab manages live status, port, API key, and a copyable endpoint; turns queue behind the engine with SSE streaming, heartbeats while queued, and protocol-correct errors; and stopping or restarting the server never interrupts an in-app response.
* **Multi-turn prefix reuse:** Follow-up API requests extend the previous turn's pinned KV prefix instead of re-prefilling the whole conversation. The server remembers each completed turn, reasoning included, and splices it back in only when the client's history matches the conversation that produced it; the prompt's date is pinned per session so the cached prefix stays valid; and reasoning round-trips for clients: Anthropic thinking blocks are signed so SDKs keep them, and the new "Reasoning in Content" setting folds reasoning into content for OpenAI-compatible clients that ignore the reasoning field. Agent loops that re-prefilled the entire prompt at every step now prefill seconds' worth of tokens.
* **Client-defined tools:** API requests that carry their own tool schemas now work: each turn teaches the engine the client's tool vocabulary (restoring the previous registry afterwards), tool calls keep their ids and JSON nulls through the round trip, and forced tool choices the engine cannot enforce are rejected with a clear 400 instead of silently degrading.
* **Request log:** Every API request and every in-app chat, agent, and Apple Foundation Model turn now leaves a trace: metadata-only lines in Console.app plus a persisted request log browsable from the Server tab, with search, error filtering, a detail inspector, JSONL export, and a queue-wait vs. generation-time split. Bodies are opt-in behind the "Store Request Content" toggle, rows read as withheld when it's off, credentials are never logged, and the log prunes itself to 30 days / 20,000 rows.
* **Persistent memory settings:** The in-app memory mode and budget now persist across relaunches, and server turns honor them, so picking Unrestricted with an FP8 KV cache actually raises the server's KV capacity instead of being silently ignored.

### 🐛 Fixes
* **Memory accounting:** Duplicated backbone buffers are no longer allocated once per model load (three resident copies were together holding 15.25 GB on a 16 GB Mac), the user-facing memory figure reports what the process actually holds (footprint plus cached weight pages) instead of resident pages that collapsed under swap pressure, and KV provisioning is clamped to the machine's RAM budget.
* **OpenAI tool calls:** Responses and stream deltas now encode tool calls under the wire format's `tool_calls` key; the previous camelCase key made them invisible to every OpenAI client, and client-defined tools were taught a call shape the engine's parser only recognized for built-in tool names.
* **SSE streaming:** Anthropic stream frames keep their `data:` prefix (standard parsers silently discarded the payloads), streaming deltas withhold suffixes that could still grow into a configured stop sequence, and a repaired snapshot is never re-sent where a client cannot replace it.
* **Oversized prompts:** A prompt that cannot fit the machine's KV cache allocation is rejected (or its `max_tokens` ratcheted) in milliseconds instead of overrunning the cache and grinding for the better part of an hour; SSE heartbeats now run for the full turn instead of going silent during prefill, and Metal allocation failures now print where they happened.
* **Queue and lifecycle correctness:** A queue-wait timeout fires at its deadline even when an earlier turn is stalled, cancelling a request cancels its queued work instead of launching against an abandoned connection, the server toggle and Restart reconcile to the persisted setting once a shutdown drains, and Restart applies a freshly typed port.
* **Under the hood:** Rust dependencies upgraded (uniffi 0.28 to 0.32, safetensors 0.4 to 0.8, tokenizers 0.20 to 0.23) and Sparkle 2.9.6 to 2.10.0, the regenerated Swift bindings gain Sendable conformances, and a bindgen change that could hang every Xcode 27 build is fixed.

---

[🔗 View Release Tag](https://github.com/derekrparris/DynaMoE/releases/tag/v26.6) · [📁 Repository](https://github.com/derekrparris/DynaMoE)

<sub>DynaMoE is an independent open-source inference engine. macOS, Metal, and Apple Silicon are trademarks of Apple Inc. Apache 2.0 licensed.</sub>
