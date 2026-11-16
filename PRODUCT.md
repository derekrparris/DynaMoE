# Product

<!-- impeccable:product-schema 1 -->

## Platform

ios

<!-- Native macOS (SwiftUI/AppKit + Metal). Recorded as `ios` because the skill's
platform enum has no macOS value; this is an Apple-native, non-web product. -->

## Users

Apple Silicon developers and tinkerers who run large language models on their
own Mac. They are comfortable with Xcode, Hugging Face cache directories, and
command-line tooling, and they evaluate the app by what it actually produces on
their hardware rather than by marketing claims.

Primary job: run models that do not fit in RAM, at usable speed, entirely on
local hardware, and connect the rest of their tooling to that model.

## Product Purpose

DynaMoE is a native macOS application that runs large Mixture-of-Experts,
hybrid SSM/attention, and dense language models on Apple Silicon by streaming
weights from NVMe through mmap and POSIX `pread` instead of loading them into
memory. It exists so a single Mac can run checkpoints several times its
physical RAM without a network dependency.

Success means: a model larger than RAM answers a prompt in the app, the same
model answers OpenAI- and Anthropic-shaped HTTP requests from other tools on
the machine, and the autonomous agent completes real coding work — all
locally, with no weights or prompts leaving the Mac.

## Positioning

Weights are never resident. Storage is treated as the model store, with a
zero-copy unified-memory bridge (mmap'd SafeTensors shards wrapped directly as
Metal buffers) and a working-set manager that evicts experts at layer
boundaries. The speculative decoder (JetSpec) is sized and pruned by the same
per-layer expert budget the streaming path enforces.

A single headless turn engine drives every surface — in-app chat, the agent
harness, and the local API server — so protocol adapters and agent tools are
different front doors on one execution path, not parallel implementations.

## Operating Context

- Build and run happen in Xcode (`DynaMoE/DynaMoE.xcodeproj`); there is no
  Makefile, no CI, and no standalone build script. A build phase runs
  `cargo build` and regenerates the UniFFI bindings on every build.
- Models are discovered from `~/.cache/huggingface/hub` at runtime and are
  never committed. Tests and benchmarks that need weights skip when the
  snapshot is absent.
- The reference development machine is an M1 Pro with 16 GB RAM, and memory
  budgets and chunk sizes are tuned for it.
- Distribution is a notarized Developer ID app with Sparkle auto-updates.
- `Benchmarks/PERF_FINDINGS.md` is the authoritative performance record and
  explicitly lists optimizations that were shipped, measured, and reverted.

## Capabilities and Constraints

Confirmed capabilities:

- MoE, hybrid GatedDeltaNet/GQA and dense transformer families, with
  architecture auto-detection from `config.json` and tensor topology.
- In-app chat with markdown, reasoning accordions, and a tool execution
  timeline; an autonomous agent with shell, file, search, git, subagent, and
  MCP tool servers; multi-turn prefix caching.
- A local server on `127.0.0.1` serving OpenAI (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`) and Anthropic (`/v1/messages`,
  `/v1/messages/count_tokens`) shapes, with one FIFO, single-flight turn slot.
- A Rust core (`dynamoe_core`) owning tokenizer, SafeTensors shard indexing,
  and speculative-tree math, exposed over UniFFI.

Constraints and technical facts:

- Apple Silicon only; macOS 26 deployment target.
- The engine is single-flight: one generation at a time, shared by chat, agent,
  and API turns.
- Secrets live in the Keychain, never in config files; remote URLs in image
  parts are refused rather than fetched.
- Request fields the server cannot honour are refused by name with a 400 rather
  than silently answered differently.
- Model weights, benchmarks, and API surface behaviour are validated against
  real checkpoints, not fixtures, whenever a claim depends on numerics.

Explicitly undecided / open:

- No binding brand or identity commitment is recorded; the current visual
  identity is provisional.

## Evidence on Hand

- `README.md` — product overview, supported architectures, acknowledgements.
- `SETTINGS_GUIDE.md` — every setting, budget, and precision choice explained.
- `AGENTS.md` — architecture, invariants, and measured findings for agents.
- `Benchmarks/PERF_FINDINGS.md` — measured performance journal, including
  rejected optimizations.
- `DESIGN.md` and `.impeccable/design.json` — the recorded interface system.
- Real checkpoints are present locally under `~/.cache/huggingface/hub`; they
  are not distributable and must never be committed.

Absences future work must not fabricate: no testimonials, no customer logos, no
published benchmark numbers beyond those measured on the reference hardware, no
pricing or licensing claims beyond `LICENSE`.

## Product Principles

1. **Weights stream; they do not load.** Every design decision preserves the
   ability to run a checkpoint larger than physical RAM off the SSD.
2. **One engine, many doors.** Chat, agent, and HTTP surfaces share one turn
   engine; a capability added to one path must not be reimplemented beside it.
3. **Refuse loudly rather than degrade silently.** An unsupported request, tool
   mode, or model capability returns a named refusal instead of a plausible
   wrong answer.
4. **Measured, not assumed.** Performance and correctness claims are attached
   to a measurement on real weights, including the ones that failed.

## Accessibility & Inclusion

No formal accessibility standard is committed. The bar is Apple platform
defaults: semantic AppKit/SwiftUI colors, system controls, and stock behaviors.
Accessibility work beyond platform defaults is not a stated requirement and
should not be invented.
