# DynaMoE Subagent Runtime

Design and working document for the model-backed subagent system. This file is
the reference for a multi-session build: read it first, check the phase
checklist, then continue at the first unchecked item.

Status: **Phases 0–5 complete for writers.** Phase 1 landed both backends;
Phase 2 landed `AgentScheduler` (capacity, priority, cancellation tree, queue
timeouts), the real run path inside `SubagentManager`, the spawn approval gate
and the extended `spawn_subagent` schema. Phase 3 landed `AgentRunStore` (one
JSON per run, schema gate, atomic digest-skipped writes, retention), the
inspector (`SubagentInspectorView`, replacing the drawer), inline spawn cards,
notifications, the start-of-turn unread-report relay on both paths, and the
Settings surface for backend, cloud model, cost limit, concurrency, retention
and notifications. Phase 4 landed `WorkspaceIsolation` (a git worktree per
writer, diffed against the commit it branched from), the approved
`apply_subagent_changes` tool, the inspector's worktree controls and
conversation cost strip, and the spawn-result hints that make the worktree
discoverable to the coordinator. Phase 5 adds the operator's own Apply Changes
control in the inspector, which lands an isolated run's work through the same
approval card and apply service the coordinator's tool uses, makes the apply
single-flight per run, and proves the path with a real local writer run (3
steps, applied `hello_from_subagent.txt`). The optional AFM backend and the
performance pass remain deferred (§16). Still open: the cheapest-capable cloud
model as the default, and when a model-backed spawn becomes the default rather
than a Settings opt-in (§16).

---

## 0. Decisions (approved by the operator)

| Decision | Choice |
|---|---|
| Backends | **Both** local Metal engine and cloud providers, available from the first delivery |
| Write capability | Subagents may write, edit, run shell and commit, behind isolation and approval |
| Backend selection | Both available; **the spawn asks** which one, with model, workspace and budgets |
| Local in-turn delegation | Allowed, gated by a **cost estimate + refusal** when the coordinator re-prefill would be too expensive |
| Delegation depth | One level. A subagent may not spawn subagents |
| Persistence | Every run is stored with its session and survives relaunch |
| Documentation | This file, kept current as the project spans sessions |

SubQ signals used for the approved shape: allow write workers 0.93, replace the
hardcoded archetype pipelines 0.88. Backend default and isolation were mid-band
and were decided by hand (see §6 and §8).

---

## 1. Why this exists

The original implementation (`SubagentManager.swift`) is not an agent loop. It
dispatches on a role archetype and runs one of four hardcoded pipelines:
semantic search plus three file reads, one shell command, or regex extraction of
paths and commands from the task text. No model is ever invoked. There is no
prompt, no reasoning, no tool loop, no synthesis, no persistence, and no
approval. A "Shader Optimizer" emits three canned recommendations regardless of
what it read.

The goal is a real delegated agent runtime:

- a headless agent loop backed by a real model,
- an isolated context window per run,
- a structured report handed back to the coordinator,
- per-run tool scopes and loop guards,
- real scheduling, budgets and cancellation,
- human approval where writes are possible,
- durable transcripts and accounting.

---

## 2. Non-negotiable constraints

These come from the app, not from taste. Breaking one of them is a regression in
the coordinator.

1. **The local engine is single-flight.** `CompletionEngine.launchTurn` refuses a
   second turn while one is active. There is one shared KV cache and one expert
   arena.
2. **There is one global KV prefix pin**, keyed by `sessionId`
   (`PrefixCacheManager`). Any turn that records a different session takes the
   pin; the next coordinator step then pays a **full re-prefill**. On the
   published DeepSeek checkpoint that is tens of seconds to minutes.
3. **`AgentHarness.shared` holds process-global run state** (loaded tools, loop
   guards, mode, session id, grammar). A subagent must not mutate it.
4. **Approval lives in the two coordinator loops**, not in `executeTool`. A
   subagent must get its own enforcement or it bypasses HITL entirely.
5. **The project defaults to `@MainActor` isolation.** A headless loop must be
   explicitly `nonisolated` and run off the main actor.
6. **A writer needs its own working tree.** Worktrees are how Phase 4 isolates
   them; an in-place overlay is still absent, and a non-git workspace can only
   fall back to per-call approval.
7. **`spawn_subagent` is not in `coreLoadedToolNames`.** It remains a discovered,
   loaded tool; that is unchanged.

---

## 3. Architecture

```
SubagentManager                      coordinator-facing API + registry + relays
      |
      +-- AgentScheduler            capacity, queueing, leases, cancellation tree
      |        +-- local engine lease (1)
      |        +-- cloud pool (N, per provider)
      |        +-- Apple FM (1, optional)
      |
      +-- AgentRun                  one delegated task
               |
               +-- HeadlessAgentLoop        backend-agnostic step loop
               |        +-- AgentRunContext        tools, guards, mode, cwd, approval
               |        +-- AgentTranscript        messages, tool records, usage
               |        +-- AgentModelBackend
               |                 +-- LocalModelBackend
               |                 +-- CloudModelBackend
               |                 +-- AppleFMBackend
               |                 +-- ScriptedBackend (tests)
               |
               +-- AgentRunStore            durable JSON per run
               +-- WorkspaceIsolation       git worktree for writers
```

### Module map (planned)

| File | Responsibility |
|---|---|
| `Subagents/AgentRunModels.swift` | `AgentRunRecord`, transcript entries, report, usage, status, backend descriptor, events |
| `Subagents/AgentRunContext.swift` | Per-run tool scope, loop guards, mode, working directory, approval policy |
| `Subagents/AgentModelBackend.swift` | Backend protocol, request/turn types, `ScriptedBackend` |
| `Subagents/CloudAgentBackend.swift` | Cloud provider backend over `CloudChatClient` |
| `Subagents/LocalAgentBackend.swift` | Local engine backend over `CompletionEngine` |
| `Subagents/AppleFMBackend.swift` | Optional text-only helper backend |
| `Subagents/HeadlessAgentLoop.swift` | The loop: render, call, parse, approve, execute, append, repeat |
| `Subagents/AgentScheduler.swift` | Capacity, queueing, engine lease, cancellation tree |
| `Subagents/AgentRunStore.swift` | Persistence, retention, corrupt-file handling |
| `Subagents/AgentRolePresets.swift` | Role presets: prompt, tools, budgets, isolation |
| `Subagents/WorkspaceIsolation.swift` | Git worktree lifecycle, base-commit snapshots, `git apply`, cleanup and stale sweep |
| `Subagents/SubagentApply.swift` | The approved path that lands a run's worktree diff, used by both `apply_subagent_changes` and the inspector's Apply Changes (`SubagentApplyControl`, single-flight per run) |
| `Subagents/SubagentManager.swift` | Rewritten coordinator API (`spawn`, status, message, cancel, relays) plus the live→record projection |
| `Subagents/SubagentInspectorView.swift` | Inspector UI (replaces the drawer), worktree controls, conversation cost strip |
| `AgentHarness.swift` | Tool implementations, including `spawn_subagent`, status/message/list and `apply_subagent_changes` |

Existing files touched:

- `AgentHarness.swift`: expose a per-run initializer, add a run-scoped approval
  policy and a worker system prompt builder. No behavior change for the shared
  instance.
- `CompletionEngine.swift` / `EngineTurnSlot.swift`: shared engine lease used by
  the server queue, the coordinator and local subagent turns.
- `ContentView.swift`: spawn approval surfacing, stop propagation, relay hooks.
- `ChatDetailView.swift` / `SettingsSheetView.swift`: inspector entry and settings.

---

## 4. Data model

All values are `nonisolated` + `Sendable` + `Codable` unless stated.

```swift
enum AgentBackendKind: String { case cloud, local, appleFoundationModel, scripted }

struct AgentBackendDescriptor {
    var kind: AgentBackendKind
    var providerId: String?       // cloud
    var modelId: String?          // upstream id or local model id
    var displayName: String
    var isRemote: Bool
}

enum AgentRunStatus: String {
    case pending, preparing, running, waitingForApproval,
         completed, failed, cancelled, blocked
}

enum AgentRolePreset: String {
    case explore, implementer, verifier, reviewer, general, custom
}

struct AgentBudget {
    var maxSteps: Int             // default 24
    var maxTokens: Int            // completion tokens across the run
    var maxWallClockSeconds: Int  // default 600
    var maxCostUSD: Double?       // cloud only
}

struct AgentRunRecord: Identifiable {
    var id: UUID
    var parentSessionId: UUID?
    var parentTurnId: UUID?
    var role: String
    var preset: AgentRolePreset
    var task: String
    var contextSummary: String?
    var backend: AgentBackendDescriptor
    var workingDirectory: String
    var isolation: AgentIsolation   // .none / .worktree(path, branch)
    var allowedTools: [String]
    var budget: AgentBudget
    var status: AgentRunStatus
    var createdAt: Date
    var startedAt: Date?
    var completedAt: Date?
    var transcript: [AgentTranscriptEntry]
    var report: AgentReport?
    var usage: AgentUsage
    var messages: [AgentMessageRecord]
    var error: String?
    var depth: Int                  // 0 = coordinator child; only 0 is allowed
    var summaryRelayed: Bool
}

enum AgentTranscriptEntryKind {
    case task, assistantText, assistantThinking, toolCall, toolResult, status, usage
}

struct AgentTranscriptEntry: Identifiable {
    var id: UUID
    var index: Int
    var timestamp: Date
    var kind: AgentTranscriptEntryKind
    var text: String?
    var toolRecord: ToolCallRecord?
}

struct AgentUsage {
    var steps: Int
    var promptTokens: Int
    var completionTokens: Int
    var cachedTokens: Int
    var costUSD: Double
    var wallClockSeconds: Double
    var toolCalls: Int
    var backendRequests: Int
}

struct AgentReport {
    var status: AgentRunStatus
    var summary: String
    var findings: [AgentFinding]      // claim + evidence + confidence
    var changedFiles: [String]
    var diff: String?
    var tests: [AgentTestOutcome]
    var risks: [String]
    var openQuestions: [String]
    var usage: AgentUsage
}

struct AgentFinding {
    var claim: String
    var evidence: String       // file:line, URL, or tool-result reference
    var confidence: Double
}

struct AgentTestOutcome {
    var command: String
    var exitCode: Int32
    var passed: Int?
    var failed: Int?
    var summary: String
}
```

### Events

```swift
enum AgentRunEvent {
    case status(AgentRunStatus, detail: String)
    case assistantDelta(entryId: UUID, text: String, isThinking: Bool)
    case toolStarted(record: ToolCallRecord)
    case toolFinished(record: ToolCallRecord)
    case usage(AgentUsage)
    case completed(AgentReport)
}
```

The UI consumes events; the store consumes the final record. No event carries
raw file bytes or secrets.

---

## 5. The loop

`HeadlessAgentLoop` is `nonisolated` and runs off the main actor. One iteration:

1. Check cancellation, wall clock, step budget, token budget and cost budget.
2. Render the run's messages in the backend's native format.
3. Call the backend with the run's tool schemas.
4. Append assistant text/thinking to the transcript; emit deltas live.
5. If no tool calls: this is the final answer. Derive the report and finish.
6. For each call: validate the name against the run's allowlist, enforce plan
   mode, enforce the approval policy, execute through the run's harness, append
   the result, record usage.
7. Apply loop guards: repeated-call fingerprint, consecutive failures, no-progress
   detection. A tripped guard ends the run with `.blocked` and a diagnostic,
   never a fabricated success.
8. Repeat.

Termination:

- `complete` tool called → `.completed` with its summary.
- Final text with no calls → `.completed` with a report derived from it.
- Step/token/time/cost budget → `.failed` (budget) with a partial report.
- Cancellation → `.cancelled`.
- Guard trip → `.blocked`.
- Backend error → `.failed` with the error, retried at most twice for retryable
  transport failures.

The loop never invents a finding. A finding must cite a tool result or the
model's own text; the report renderer refuses an empty evidence string.

---

## 6. Backends

```swift
protocol AgentModelBackend: Sendable {
    var descriptor: AgentBackendDescriptor { get }
    func runTurn(_ request: AgentBackendRequest) async throws -> AgentBackendTurn
}

struct AgentBackendRequest {
    var messages: [AgentBackendMessage]
    var tools: [ToolDefinition]
    var systemPrompt: String
    var sampling: AgentSampling
    var maxTokens: Int
    var stopSequences: [String]
    var thinking: Bool
    var sessionId: UUID
}

struct AgentBackendTurn {
    var text: String
    var thinking: String
    var toolCalls: [ParsedToolCall]
    var rawText: String
    var finishReason: String
    var usage: AgentUsage
}
```

### CloudModelBackend

- Resolves a provider profile + upstream model from settings.
- Uses `CloudChatClient.streamTurn`; maps `CloudTurnEvent` deltas to
  `assistantDelta`.
- Tool calls arrive natively; assemble fragmented arguments before executing.
- Cost from `CloudCostModel`; persisted per run.
- **No engine lease.** Parallel-safe. Default pool size 3 per provider, global
  fan-out 6.
- Retries: one retry for a transport failure, never for a 4xx.

### LocalModelBackend

- Renders the run's conversation with the loaded model family's formatter
  (reuse `ChatPromptFormatter.formatConversation` and the family flags the
  server path uses).
- Installs the run's tool schemas into `GrammarConstrainedSampler` around the
  turn (snapshot + restore), exactly as `ServerController.withTurnToolMask` does.
- Calls `CompletionEngine.launchTurn` with the run's own `sessionId` and the
  rendered prompt; consumes the stream; uses `TurnResult.parsedToolCalls` for
  calls and `generatedTokenIds` for history.
- Holds the **engine lease** for the whole run, not per step, so the run's own
  prefix stays pinned between steps. Interleaving another run or a coordinator
  turn would force a full re-prefill on its next step. *(Phase 1 ships the
  claim per step, because `launchTurn` consumes a claim as soon as it mints the
  generation task and the ledger holds one; the lifetime lease is
  `AgentScheduler`'s, §8.)*
- **Cost estimate + refusal:** before the first step, compute the coordinator's
  re-prefill cost from the pinned prefix length and the recent prefill rate.
  If the estimate exceeds `dynamoe_subagent_local_cost_limit` (default 120 s),
  the spawn is refused with the estimate unless the operator overrides it on the
  approval card.
- If the model family cannot parse tool calls reliably, the run is refused by
  name rather than run without tools.

### AppleFMBackend

Optional, text-only. No tool loop. Useful only for narrow summarization or
classification workers. Ships behind the same protocol; not a default.

### ScriptedBackend

Test-only. A queue of scripted turns, each with text + tool calls. It is how the
loop, guards, budgets, cancellation and approval are tested without a model.

---

## 7. Isolation and approval

### Run-scoped harness

Each run owns an `AgentHarness` instance (the initializer is exposed for this).
It is seeded with:

- exactly the run's allowed tools in `loadedTools` (no `tools_load`,
  `tools_discover`, `spawn_subagent`, `tools_unload`),
- `defaultWorkingDirectory` = the run's workspace,
- the run's mode (plan-mode children are read-only),
- a run-scoped approval policy.

`AgentHarness.shared` is never touched by a run. The per-run instance means
search budgets, repeated-call counts, failure counts, loaded tools and grammar
state are per run.

Tools that internally reference `AgentHarness.shared` are excluded from the
subagent surface (notably `tools_discover`, `tools_load`, `tools_unload` and the
subagent tools themselves). If a tool must be offered and uses the singleton,
it is patched to use the calling harness.

### Approval

Approval moves into `AgentHarness.runToolInTurn` as a run-scoped policy so it is
enforced on every path, including subagents:

```swift
enum AgentApprovalPolicy {
    case auto                    // read-only runs
    case spawnOnly               // writers inside an isolated worktree
    case everyStateChange        // writers on the main tree, shell, git
    case deny                    // plan mode
}
```

- **Spawn approval** is the primary card. It shows: role, backend and model,
  tool set, working directory, isolation, budgets, and whether the run can
  write. A read-only run may be auto-approved if the user enables it.
- **Per-call approval** applies to state-changing calls when the policy says
  every change, or when the workspace is not isolated.
- The card is an `ActionApprovalManager` record rendered in the run's inspector
  and the coordinator's tool timeline.
- Turbo Mode may bypass routine writes; it may **never** bypass
  `targetsApplicationPreferences`, and a run in plan mode is `.deny`.
- `subq_decide` keeps its existing refusal/approval semantics.

### Workspace isolation

- Writer in a git repository: `git worktree add` under
  `~/Library/Application Support/DynaMoE/worktrees/<runId>` on branch
  `dynamoe/agent/<shortId>`; the run's working directory is that worktree.
- On completion, produce `changedFiles` and a unified diff in the report.
- The coordinator applies changes with a new approved tool
  (`apply_subagent_changes`) or a merge; the worktree is cleaned up on cancel and
  after a successful apply.
- Non-git workspace: writers run on the main tree only with per-call approval;
  otherwise the spawn is refused.
- Read-only runs never touch a worktree.

### Recursion

`depth == 0` only. A subagent has no `spawn_subagent` tool and a spawn request
from a child is refused by name.

---

## 8. Scheduling, leases and budgets

`AgentScheduler` is an actor (`Subagents/AgentScheduler.swift`).

- **Local capacity: 1.** The scheduler admits one local run at a time. The
  engine itself is claimed **per step**, through the existing `TurnSlotClaim` /
  `TurnSlotLedger` the server queue and the chat send path share; a local turn
  is launched with `TurnPriority.background` so it can yield.
- **Interactive priority is cancel-and-report, not pause/resume.** An operator's
  send that finds a background turn active cancels it (`cancelBackgroundTurn`),
  waits for the engine to drain (`awaitIdle`) and runs the send once
  (`preemptionRetry` in `startAutoregressiveGeneration`). The subagent's run is
  cancelled at its next boundary and reports why; it is not resumed. SubQ chose
  this over `pause_and_resume` (0.81 vs 0.10) because resumability needs a
  paused status, a serializable loop state and store work that belongs with
  Phase 3.
- **Cloud capacity: N** (default 3 per provider, global fan-out 6, both
  overridable through `dynamoe_subagent_cloud_per_provider` /
  `dynamoe_subagent_cloud_global`).
- **Apple FM: 1** (reserved; the backend is deferred, §16).
- **Queue:** FIFO within a priority class (interactive > user-requested >
  speculative), and priority ordering is per CLASS, so one class's head cannot
  starve a class that still has capacity. A queued run can be cancelled while
  waiting, and the wait is bounded by `dynamoe_subagent_queue_timeout`
  (default 120 s); a timeout is a refusal that names the class.
- **Cancellation tree:** `register(runId:parentRunId:parentSessionId:cancel:)`
  records how to stop a run. `cancel(runId:)` walks descendants,
  `cancelRuns(parentSessionId:)` is what the Stop button and
  `SubagentManager.cancelSession` call, and application termination cancels
  every run and every pending approval.
- **Budgets** are enforced by the loop, not the scheduler; the scheduler enforces
  concurrency and queue timeouts.

Engine lease: the doc's original "hold a claim for the run's lifetime" was
implemented as **per-step claims** instead (SubQ: 0.012 for the lifetime lease).
A run's step is a turn, and `launchTurn` releases the claim it is handed as soon
as it mints the generation task, so a lifetime claim would mean generalizing the
ledger that the server queue and the chat send path already depend on — and
holding the only slot across a multi-minute run would block every chat turn. The
safety property invariant 1 needs (a local run never launches a turn without
holding the claim) holds unchanged. The cost is stated in `LocalAgentBackend`:
a coordinator turn between two steps moves the global prefix pin and the run's
next step re-prefills. Measuring that cost is a Phase 3 item; the scheduler is
where a lifetime lease would land if it proves worth the mechanism.

---

## 9. Roles as presets

A preset is data, not a pipeline:

| Preset | Tools | Isolation | Budget | Output |
|---|---|---|---|---|
| Explore | read/search/web/skills | none | 16 steps | findings + evidence |
| Implementer | read + write/edit/shell/git | worktree | 40 steps | diff + tests |
| Verifier | read + shell + lint | none | 24 steps | test outcomes + risks |
| Reviewer | read + git_diff | none | 16 steps | findings + risks |
| General | read-only by default | none | 20 steps | report |

Custom roles map to `general` with the task text supplying the specialization.
The model may not name a tool outside the preset unless the spawn card granted
it.

---

## 10. Persistence and retention

- One JSON file per run:
  `~/Library/Application Support/DynaMoE/agent-runs/<sessionId>/<runId>.json`
  (application support, not caches, matching the session store decision).
- `schemaVersion` on every file; unknown newer versions are ignored rather than
  decoded lossily.
- Atomic writes (temp + rename), digest-based skip when unchanged.
- Transcript tool outputs are truncated to a per-entry budget before writing.
- Retention: keep the newest 200 runs or 30 days, whichever is larger; never
  prune a run younger than 24 hours. Runs of a deleted session are pruned with
  it.
- The store is the source of truth for the inspector on relaunch. In-memory
  `AgentRun` objects are projections of the record.
- `appliedAt` (optional) records when an isolated run's worktree landed on the
  main tree. A record without it decodes as "not applied", which is also what
  every pre-Phase-4 record means. `report.diff` has its own 64 KB budget; the
  live worktree, not the stored copy, is what applying reads.

---

## 11. UI

- **Inspector** (replaces `SubagentDrawerView`): list of runs for the current
  session with status, backend, model, duration, usage and cost, plus a
  conversation strip (active/runs/tokens/spend/time) over the same merged list;
  a detail pane with live streamed text/thinking, tool timeline, artifacts,
  diff, report, Reveal Worktree, the applied date and controls (stop, steer,
  approve, Apply Changes). Apply Changes is the operator's own path to land a
  stopped isolated run: it raises the ordinary approval card and then calls the
  same `SubagentApplyService` the coordinator's `apply_subagent_changes` tool
  uses.
- **Inline spawn card** in the transcript: role, backend, live status, short
  summary, link to the inspector.
- **Notifications** for a finished background run; unread reports are injected
  at the start of the next coordinator turn on both paths (fixes the weights-path
  orphan).
- **Settings**: default backend, cloud model for subagents, read-only
  auto-approve, local cost limit, max concurrency, retention.

---

## 12. Testing

Pure-logic and fake-backend tests are the default; real-model tests are opt-in.

| Suite | What it proves |
|---|---|
| `AgentRunContextTests` | allowlist enforcement, plan mode, per-run guards, no shared-state mutation |
| `HeadlessAgentLoopTests` | step loop, tool execution, budget, cancellation, guard trips, report derivation |
| `AgentApprovalTests` | spawn and per-call approval, Turbo/app-preference rules, denial |
| `AgentReportTests` | schema, evidence requirement, truncation, sanitization |
| `AgentSchedulerTests` | capacity, priority, cancel tree, queue timeout |
| `AgentRunStoreTests` | round trip, retention, corrupt file, unknown schema |
| `CloudAgentBackendLogicTests` | event mapping, fragmented tool-call assembly, retry rules |
| `LocalAgentBackendLogicTests` | prompt rendering, mask install/restore, cost estimate and refusal |
| `WorkspaceIsolationTests` | worktree create/snapshot/apply/cleanup, committed work included, conflicts refused, non-git refusal, the apply tool and a scripted isolated run end to end, and the operator's Apply Changes path (offer gate, one approval request, declined card, applied-then-nothing) |
| Opt-in | one real local turn, one real cloud turn, one real local writer run + operator apply |

Headless invocation (from `AGENTS.md`):

```bash
cd DynaMoE && xcodebuild test-without-building -project DynaMoE.xcodeproj -scheme DynaMoE \
  -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO \
  -only-testing:DynaMoETests/<ClassName>/<testName>
```

---

## 13. Phase plan

### Phase 0 — Foundations (complete)

- [x] Design approved and captured in this file
- [x] `AgentRunModels.swift`: records, transcript, report, usage, events
- [x] `AgentModelBackend.swift`: protocol, request/turn types, scripted backend
- [x] `AgentRunContext.swift`: tool scope, guards, approval policy
- [x] Expose per-run `AgentHarness` initializer + worker system prompt builder
      (`makeRunScoped`, `configureRunScope`, `buildWorkerSystemPrompt`,
      `buildCloudWorkerSystemPrompt`, run-scoped approval in `runToolInTurn`)
- [x] `HeadlessAgentLoop.swift`: full loop over a scripted backend
- [x] Tests: loop, context, report, approval policy (`SubagentLoopTests`, 14 cases)

**Exit criteria:** a subagent run with scripted turns executes real tools in a
temp directory, honors budgets, cancels cleanly, and returns a structured
report; no production path is wired yet.

Implemented beyond the checklist, deliberately early:

- `AgentHarness.runScopedForbiddenToolNames` refuses `tools_discover`,
  `tools_load`, `tools_unload` and `spawn_subagent` at `configureRunScope`
  (`AgentRunScopeError.forbiddenTool`), so invariant 3 is enforced where the run
  scope is built rather than left to the spawn path.
- `CloudAgentBackend.swift` is written against `CloudChatClient` and builds; its
  logic tests landed in Phase 1. It still has no spawn path and no UI.

Naming drift from §4/§6 as implemented: `AgentBackendTurn.calls` (not
`toolCalls`), `AgentBudget.maxCompletionTokens` (not `maxTokens`), and
`AgentReport.backend` is a required field carried from the backend descriptor.

### Phase 1 — Real backends

- [x] `CloudAgentBackend`: provider/model resolution, streaming, native tool
      calls, usage/cost
- [x] `LocalAgentBackend`: family formatting, tool mask, parser, engine lease
- [x] Cost estimate + refusal for local in-turn
- [x] Backend logic tests; one opt-in real turn per backend
      (`CloudAgentBackendLogicTests`, 18 cases incl. the opt-in real turn;
      `LocalAgentBackendLogicTests`, 22 cases; the real-turn cases in
      `DeepSeekV41PublishedTurnTests` and `CloudAgentBackendLogicTests`).

**Exit criteria:** the same loop runs a real cloud turn and a real local turn,
producing identical transcript/report shapes.

Measured on the reference machine (2026-11-17):

- real LOCAL turn on `DeepSeek-V4.1-Flash-1017`: 28 prompt / 7 generated tokens,
  `finish eos`, 7 deltas, 17.2 s including the runtime build, answer "The
  capital of France is Paris.";
- real CLOUD turn on Syn Large Text (Synthetic): 50 prompt / 24 generated,
  0.76 s, same answer shape. Both return the same `AgentBackendTurn` fields
  (`text`, `thinking`, `calls`, `finishReason`, `usage`), which is what the exit
  criterion asks for.

Implemented in Phase 1, with the reasoning recorded:

- **The engine slot is claimed per STEP, not per run.** `launchTurn` consumes the
  `TurnSlotClaim` it is handed as soon as it mints the generation task, and the
  ledger holds exactly one claim, so a claim cannot be carried across steps
  without changing the mechanism the server queue and the chat send path depend
  on. Claiming per step was the chosen risk profile (~0.91); the cost is stated
  in the code rather than hidden: a coordinator turn landing between two steps
  takes the engine, moves the global prefix pin to its own session, and the
  run's next step re-prefills. Phase 2 kept this shape on purpose (§8): the
  lifetime lease measured 0.012 with SubQ, and holding the only slot across a
  run would block every chat turn. `AgentScheduler` (§8) owns admission,
  priority and cancellation, not a run-lifetime claim.
- **The cost refusal lives in `LocalAgentBackend`.** `make` prices the spawn and
  calls `admission(hasLoadedModel:estimate:operatorOverride:)`, so no caller can
  start a local run without consulting it; `operatorOverride` is the spawn
  card's permission and the only thing that turns a refusal into an admission
  (invariant 10). The estimate is the coordinator's DISPLACED prefix (the pin is
  only priced when another session owns it) divided by the last measured prefill
  rate, which the engine now records from its own progress line
  (`PrefillThroughput`); an unmeasured process prices with a deliberately
  pessimistic 5 tok/s floor.
- **A step re-renders the whole conversation** through
  `ChatPromptFormatter.formatConversation` (the stateless server path's shape),
  rather than splicing token ids forward like the in-app chat loop. The
  conversation-turn mapping mirrors `OpenAIAdapter.splitMessages`: a tool result
  is a `.toolResult` turn, an assistant turn keeps its reasoning closed, and a
  call whose markup is not in the text is echoed with the server's plain-text
  call record.
- **The tool mask is one implementation.** `GrammarConstrainedSampler.withTurnToolMask`
  now owns the snapshot/install/restore that `ServerController` used privately,
  and the local backend uses the same call for its turn.
- A test caught a real defect in `maskSchemas`: encoding the parameters
  dictionary inside a second `AnyCodable` has no encoding of its own, so every
  schema decoded as `{}` and the mask constrained nothing. It now encodes the
  dictionary itself.

### Phase 2 — Manager, scheduling, approval

- [x] `AgentScheduler` with local/cloud capacity and cancellation tree
- [x] `SubagentManager` rewritten to spawn `AgentRun`s
- [x] Spawn approval card + per-call policy in `executeTool`
- [x] `spawn_subagent` schema: backend, preset, budget, isolation
- [x] Stop propagation; relay on a finished background run

**Exit criteria:** the coordinator can delegate a real task on either backend
and receives a report; a stopped turn cancels its children; no write reaches the
main tree without approval.

**Measured at the exit check:** the local run answered through the manager on
the published checkpoint in 116.7 s total (`1 step(s), 906 prompt / 7
generated`, "The capital of France is Paris."), and the cloud run answered
through the manager on a discovered tool-capable model. `AgentSchedulerTests`
(9) pins local capacity 1, per-class priority, per-provider and global cloud
caps, the cancellation tree, session-scoped stop and both queue outcomes;
`AgentRunIntegrationTests` (4) drives a real manager run with a scripted
backend; `AgentSpawnPolicyTests` (14) pins the preset table, plan-mode filtering,
the write-spawn approval rule and the tool schema. 153 related tests green
(2 opt-in cloud cases skipped), zero failures.

**Implemented in Phase 2, with the reasoning recorded:**

- **`SubagentManager` grew a second execution path instead of a second
  manager.** `SubagentInstance` now carries an `AgentSpawnPlan` and runs the
  headless loop when it has one; the canned archetype pipelines stay for the
  legacy path and for the tests that pin them (SubQ: extend the instance 0.94,
  new manager 0.06). Both paths share the same published status, transcript,
  messages, `waitForCompletion` and drawer, so the UI needed only an approval
  section.
- **Event application is serialized through one `AsyncStream`.** A Task per
  event applied a late `.running` after the terminal bookkeeping, so a finished
  run could read as running; the pump is drained before `finishRun`. Cancelling
  now writes status and summary synchronously for the same reason.
- **A model-backed spawn is opt-in until the Settings default flips.** The tool
  reads `dynamoe_subagent_backend`; with none stored it keeps the legacy
  pipelines. Sending work to a model (and, for cloud, off-device) is the
  operator's choice, not something an omitted argument should make; tests that
  pin the legacy pipelines also never touch a network or a checkpoint.
- **Spawn approval reuses the existing card.** `requiresApproval` treats a
  `spawn_subagent` that can write (its explicit tools, else its preset) as
  state-changing, so the operator approves the delegation once with the role and
  tool set in front of them; read-only spawns and plan-mode spawns need no card,
  and Turbo still bypasses routine writes.
- **Per-call approval in a run lives in the Task Manager.** A run's card cannot
  ride the coordinator's tool timeline (the spawning step may already be over),
  so `SubagentApprovalCenter` renders it in the drawer from the same
  `ActionApprovalManager` continuations. Plan mode is `.deny`; a write-capable
  run on the main tree is `.everyStateChange`, an isolated writer is
  `.spawnOnly` (Phase 4), and a read-only run is `.auto`.
- **Cloud capacity is per provider AND global, and priority is per class.** A
  global head-of-queue check let one provider's waiter starve another provider
  that still had quota; the fix is in `isHeadWaiter` and is pinned by
  `testCloudCapacityIsPerProvider`.
- **`spawn_subagent` refuses what it cannot honour.** `worktree` isolation is
  validated before a backend is resolved (a read-only run and a non-git
  workspace are refused with the `none` alternative named), forbidden and
  unknown tool names are refused before any backend, and an empty tool set after
  plan-mode filtering is refused rather than starting a run with nothing to do.

### Phase 3 — Persistence and UI

- [x] `AgentRunStore` + retention
- [x] Inspector with live stream, timeline, diff and controls
- [x] Inline spawn cards, notifications, unread reports
- [x] Settings

**Exit criteria met:** a run written by one process is listed by the next after
relaunch, every run is observable down to its tool timeline and report, and a
live run can be cancelled or steered from the panel.

Implemented, with the decisions worth carrying:

- **One JSON per run, written off the main actor.** `AgentRunStore` mirrors the
  chat store: a private `ioQueue`, atomic writes, and a SHA256 digest skip so a
  caller can persist on every tool boundary. The `schemaVersion` gate ignores a
  record written by a newer build rather than decoding it lossily and rewriting
  the fields this build does not know. Tool output is truncated at a per-entry
  budget (16 KB) before the write, because a run that read a large file would
  otherwise make one record the size of that file. Retention keeps the newest
  200 runs or 30 days (whichever is larger) and never prunes anything younger
  than 24 hours; `AgentRunStore.configuredPolicy()` reads the two Settings
  values, and the launch sweep, `SubagentManager.init`, and every tenth spawn
  run it.
- **A test run never writes the operator's history.** `AgentRunStore.shared`
  picks its root from the process: Application Support in the app, a fixed
  temporary directory under XCTest. This is the request-log recorder's lesson
  applied to run records.
- **The inspector is a list → detail panel, not a table.** The old drawer was a
  stack of expanded cards; at the width an app sidebar gets, a master list with
  a back button gives each run a readable detail pane instead of squeezing
  everything into a column. The list merges stored records with live instances
  (the live one wins for its own id), groups active above finished, and marks a
  finished report the coordinator has not received as `UNREAD`. The detail shows
  backend + model, budget, duration, usage/cost, the live reasoning and answer,
  the tool timeline, findings/changed files/tests/risks/questions/diff, the
  run's own approval cards, steering, and Stop/Copy Report controls. Approvals
  render in the detail when a run is selected and above the list otherwise, so
  the same request never appears twice.
- **A live instance is a projection of the record.** `snapshotRecord()` is the
  one mapping from runtime state to `AgentRunRecord`; the loop's own transcript
  replaces the step list at the terminal boundary, so the persisted run, the
  inspector and the drawer cannot disagree about what happened. `isSummaryRelayed`
  persists on change, which is what makes "unread" survive a relaunch.
- **Notifications are a Dock bounce plus an opt-in banner.** The bounce needs no
  permission and cannot outlive the moment; the banner asks on first use, is
  skipped under XCTest (a headless runner must not meet a permission dialog),
  and names the run rather than claiming more than the summary did.
- **The start-of-turn relay has two shapes because the caches differ.** The
  cloud path initializes its trailing directive before the first request
  (`takeSubagentRelayDirective()`), keeping the provider's cached prefix intact.
  A locally rendered prompt has no such prefix to protect, so a prompt-only
  `.user` turn (`ChatMessage.isPromptOnly`) is inserted ahead of the user's
  message; the transcript skips it and a restore drops it, and every family
  template carries it without a per-family branch. A directive lives for one
  turn: the next send removes the previous one before gathering again.
- **Settings reads at use time where it can.** Backend, cloud model, cost limit
  and retention are read by the resolver and store at the next spawn/sweep.
  Scheduler capacities are read at launch: `AgentScheduler.shared` now uses
  `AgentSchedulerLimits.configured()`, which it never did before this phase, so
  the keys were dead in production.

Measured: the Phase 3 logic suites (`AgentRunStoreTests`, `SubagentRunLinkTests`,
`LocalSubagentRelayTests`, `SubagentRecordProjectionTests`) are 17 cases; the
Phase 2 suites still pass (35). App and test targets build with the new files.
At the time of Phase 3 the inspector could only show isolation as metadata; the
worktree picker and `apply_subagent_changes` landed in Phase 4.

### Phase 4 — Writers and polish

- [x] `WorkspaceIsolation` worktrees + `apply_subagent_changes`
- [x] Verifier and Reviewer presets (defined in Phase 2 and pinned by
      `AgentSpawnPolicyTests`; a real model-backed spawn of each remains opt-in)
- [x] Cost/usage dashboard (the inspector's conversation strip)
- [ ] Optional AFM backend (deferred: its tool loop is unproven and it is not
      needed for the writers exit criteria)
- [ ] Performance pass (deferred; no measurement has made this path the cost)

**Exit criteria met:** two writer runs can work in parallel in separate
worktrees without touching the main tree, and the coordinator lands one diff
with `apply_subagent_changes` behind the ordinary approval card.

Implemented, with the decisions worth carrying:

- **A worktree is created only after the backend resolves, and only for a run
  whose tools can change something.** A local cost refusal re-resolves the same
  request; creating the worktree before that would leave an orphan branch and
  directory. Read-only presets are refused by name ("a worktree would be an
  empty copy"), and a non-git workspace is refused with the `isolation: "none"`
  alternative in the message. The plan carries the real path and branch, so
  every record written from it knows where the run worked.
- **The diff is against the commit the branch was cut from, not HEAD.** The
  implementer preset can `git_commit` inside its worktree. A HEAD-relative diff
  would then contain nothing the run committed, which is exactly the work a
  writer produces. `createWorktree` records the base commit; an `AgentIsolation`
  restored from disk derives it as `git merge-base <branch> HEAD` in the main
  repository, which is the fork point even after the main tree advances.
- **`git add -A` before every snapshot, and the untracked files are included.**
  Staging is safe because the worktree is disposable and the index is not part
  of what is applied. `--name-status -z` separates the STATUS from its path with
  NUL as well as entries from each other (`A\0path\0`, `R100\0old\0new\0`); a
  tab-only parser silently reports no files, which is how the first cut passed
  every apply case while listing nothing in the inspector.
- **Applying is `git apply`, not a merge.** `git apply --check` validates the
  whole patch before anything is written, so a conflict leaves the main tree
  exactly as it was and is reported by name. A merge would bring history and
  create merge state, and the coordinator may have uncommitted work. Applied
  changes are deliberately left UNSTAGED: the coordinator (or the operator)
  decides what becomes a commit. A merge or rebase already in progress in the
  main tree is refused rather than interleaved with someone else's conflict
  resolution.
- **Lifecycle: completed and failed runs keep their worktree; cancelled runs
  lose it.** The record's diff is bounded (64 KB for a report, plus the
  per-entry transcript budgets), so the worktree is the only copy that can
  always be applied in full. `apply_subagent_changes` re-reads the LIVE diff,
  applies it, writes `appliedAt` synchronously, then removes the worktree; a
  second call is a no-op that says so. A launch sweep removes worktrees older
  than 14 days, which is the only owner of a run interrupted between creation
  and apply.
- **Approval moves to spawn for an isolated writer.** `.spawnOnly` means the
  operator approves the delegation once, with the worktree and branch in front
  of them, and the run's writes inside its own tree need no per-call card.
  `apply_subagent_changes` is itself state-changing, so landing the work always
  gets its own card. `targetsApplicationPreferences` stays absolute on every
  path.
- **The coordinator is told how to finish the job.** A spawn's result (and
  `get_subagent_status` at a terminal state) carries `isolation`, `worktree`,
  `branch` and a directive naming `apply_subagent_changes` and the run id.
  Without that, a coordinator that never read this document would present the
  report and leave the work in a branch it never mentions.
- **The dashboard is a conversation aggregate, not a second window.** The
  inspector header shows active/runs/tokens/spend/time over the same merged
  list it renders, so live usage and restored records cannot disagree. A
  settings-wide ledger would duplicate the list and drift from it.
- **`appliedAt` is an additive optional field.** Records written by an older
  build decode with no value, and the store's schema gate never rewrites a
  newer build's fields. The inspector shows the apply date when it exists.

### Phase 5 — Operator control

- [x] Inspector Apply Changes, behind `SubagentApprovalCenter` and
      `SubagentApplyService`
- [x] One apply in flight per run, so the coordinator's tool and the
      inspector's button cannot race
- [x] Opt-in real-model writer + operator apply case
      (`DeepSeekV41PublishedTurnTests.testARealLocalWriterLandsItsWorkFromTheWorktree`)
- [ ] Discard Worktree control (not built: the 14-day sweep and Reveal
      Worktree cover cleanup for now, §16)

**Exit criteria met at the logic seam:** a stopped isolated run can be landed
from the inspector without terminal git, and the card it raises is the same
approval the coordinator's tool raises. The button itself has not had a
rendered UI pass yet: `WorkspaceIsolationTests` proves the control's gate,
request and apply outcomes, not the pixels.

Implemented, with the decisions worth carrying:

- **The button is a second caller, not a second apply path.**
  `SubagentApplyControl` asks `SubagentApprovalCenter` for the same
  `AgentApprovalRequest` shape a tool call uses, then delegates to
  `SubagentApplyService`. Patch validation, record writes and worktree removal
  are not duplicated. The apply card carries `apply_subagent_changes` as its
  tool name and the run id in its arguments, so the generic card and the
  inspector's inline stack render it exactly as they render any other
  approval.
- **The card names the main tree, not the worktree.** `applyRequest(for:)` is
  pure and testable; `apply` resolves the owning repository from the worktree
  (`git rev-parse --git-common-dir`) so the destination is where the patch
  actually lands. A worktree that is gone offers nothing: `canApply` requires
  the path to exist.
- **Apply is offered only for a terminal, unapplied, isolated run.** A running
  run cannot be applied; an applied run shows its date instead of a dead
  button; a non-git/non-worktree run was never isolated. The control re-reads
  the record from the manager or the store rather than trusting the pane's
  snapshot, so a stale view cannot apply a live run.
- **One apply per run at a time.** The durable guard is `appliedAt`, but it is
  written only after the patch lands; with two callers both could pass the
  guard before either wrote it. `SubagentApplyService` now takes an in-flight
  set keyed by run id before the first `await` and refuses a concurrent
  request with a named error.
- **`SubagentInstance.appliedAt` is `@Published`.** The inspector renders a
  value snapshot and the manager forwards each instance's `objectWillChange`;
  without publishing, the pane that offered the button would keep showing it
  after a successful apply. Stored (non-live) records are refreshed through
  `onRecordChanged` to `reloadStored`.
- **A declined card changes nothing.** `SubagentApplyControl.apply` throws
  `.approvalDenied` before touching the worktree, so the record, the branch and
  the main tree are all left as they were.
- **The coordinator is told when the work is already landed.**
  `get_subagent_status` and a synchronous spawn result check `appliedAt`. They
  report `applied: true` and say the changes are already in the main tree,
  instead of naming `apply_subagent_changes`; telling the coordinator to apply
  work the operator just landed would produce a second, confusing card.

**Measured (2026-11-17):** the opt-in
`testARealLocalWriterLandsItsWorkFromTheWorktree` ran the published checkpoint
through this whole path with a model rather than `ScriptedBackend`: 3 steps,
8,403 prompt / 227 generated, 1,035 s including checkpoint load and a
2,636-position first prefill. The implementer wrote
`hello_from_subagent.txt` in its own worktree on branch
`dynamoe/agent/<short>`; the main tree had no such file while it ran; and the
operator path applied exactly `["hello_from_subagent.txt"]`, set
`appliedAt`, and removed the worktree. That closes the ScriptedBackend-only
gap Phase 4 recorded.

---

## 14. Invariants (do not break)

1. A local run never launches a turn without holding the engine lease.
2. A run never mutates `AgentHarness.shared`, the shared grammar registry
   outside its turn, or the coordinator's loaded tools.
3. `tools_load`, `tools_discover`, `tools_unload`, `spawn_subagent` and
   `apply_subagent_changes` are never offered to a run.
4. Every state-changing tool call passes through the run's approval policy,
   including `shell_run`, MCP and `subq_decide`.
5. `targetsApplicationPreferences` is absolute: no card, Turbo, or run policy may
   bypass it.
6. The loop never runs on the main actor and never blocks it.
7. The report contains no unsupported claim: every finding cites evidence.
8. The coordinator's prompt receives only the bounded report, never the
   transcript.
9. A cancelled run stops issuing backend requests and tool calls at the next
   boundary and writes `.cancelled`.
10. A local run is refused, with an estimate, when its re-prefill cost exceeds the
    configured limit, unless the operator overrides it on the spawn card.
11. A run's backend descriptor is recorded on the run and in the report; a report
    without its backend is invalid.
12. No fabricated output: when the loop cannot ground a result, it reports a
    failure or an open question.

---

## 15. Session log

Append one line per session so the next session knows where work stopped.

| Date | Session | Result |
|---|---|---|
| 2026-11-17 | phase 5 operator apply | `SubagentApplyControl` is the inspector's Apply Changes path: it offers apply only for a terminal, unapplied, isolated run whose worktree still exists, raises the ordinary `SubagentApprovalCenter` card with the main tree as its destination, and delegates to the existing `SubagentApplyService` (no second apply implementation). `SubagentApplyService` takes a per-run in-flight guard before its first `await`, because the coordinator's tool and the button can snapshot together; `SubagentInstance.appliedAt` is now `@Published` so a live run's pane redraws, and stored records refresh through `onRecordChanged` → `reloadStored`. `get_subagent_status` and a synchronous spawn now report `applied: true` instead of telling the coordinator to apply work the operator landed. The worktree note names the button. Tests: `WorkspaceIsolationTests` 18 (gate, request shape, declined card leaves everything untouched, approved apply then offers nothing, status after apply), and the opt-in `DeepSeekV41PublishedTurnTests.testARealLocalWriterLandsItsWorkFromTheWorktree` passed with a real model (3 steps, 8,403 prompt / 227 generated, 1,035 s; wrote and applied `hello_from_subagent.txt`). Related subagent suites 69 green. Build + test build green. Next: rendered UI pass, or the §16 measurements. |
| 2026-11-17 | phase 4 worktrees + apply | `WorkspaceIsolation` (worktree at Application Support/worktrees/<runId> on `dynamoe/agent/<short>`, base-commit snapshots so committed work is included, bounded display diff, `git apply` landing, merge/rebase refusal, cancel/apply cleanup, 14-day launch sweep) with 12 pure git cases; spawn resolver is asynchronous and creates a worktree only for a writable run in a git workspace, after the backend resolves; `.spawnOnly` approval for isolated writers; `apply_subagent_changes` tool + `SubagentApplyService` (idempotent via `appliedAt`, live diff, synchronous record write, worktree removal) and its state-changing card; spawn/status results name the worktree, branch and apply tool; inspector gains Reveal Worktree, applied date, and a conversation cost/usage strip; `AgentRunRecord.appliedAt` additive field, report diff budget 64 KB. Tests: `WorkspaceIsolationTests` 15, `AgentSpawnPolicyTests` new isolation/policy cases. Build + test build green. Next: Phase 5 candidates (§16). |
| 2026-11-17 | phase 3 store + inspector | `AgentRunStore` (schema gate, atomic digest-skipped writes, transcript bounds, retention) with 8 pure cases; `SubagentInstance.snapshotRecord()` + persistence on start/tool boundary/finish/cancel/relay; `SubagentInspectorView` (list/detail, live stream, timeline, report, approvals, steering) replacing `SubagentDrawerView`; inline `spawn_subagent` card linked by the result's run id; `SubagentNotifier` (Dock bounce + opt-in banner, silent under XCTest); start-of-turn unread relay on both backends (cloud trailing directive, local prompt-only turn); Settings for backend, cloud model, cost limit, concurrency, retention and notifications; `AgentScheduler.shared` now reads `configured()` limits. New logic cases: `AgentRunStoreTests` 8, `SubagentRunLinkTests` 4, `LocalSubagentRelayTests` 2, `SubagentRecordProjectionTests` 2. Build + test build green; Phase 2 suites 35/35. Next: Phase 4 (worktrees, writers). |
| 2026-11-17 | phase 2 manager | `AgentScheduler` (capacity 1 local, 3/provider + 6 global cloud, per-class priority, cancellation tree, queue timeout), `AgentRolePresets`, `AgentSpawn` resolver, `SubagentApprovalCenter`, the model-backed path in `SubagentManager`/`SubagentInstance`, the extended `spawn_subagent` schema with a write-spawn approval gate, engine `TurnPriority` + chat preemption retry, Stop propagating to children and termination cancelling everything. New tests: `AgentSchedulerTests` 9, `AgentSpawnPolicyTests` 14, `AgentRunIntegrationTests` 4. Related suites 153 green, 0 failures (2 opt-in cloud skipped). Opt-in manager runs passed on both backends (local DSV4.1 906 prompt / 7 generated in 116.7 s; cloud Syn Large Text 668 prompt / 67 generated, 1.27 s), each answering the task it was given. Next: Phase 3 (`AgentRunStore`, inspector). |
| 2026-11-17 | design | Architecture approved; decisions recorded; doc created; Phase 0 started |
| 2026-11-17 | phase 1 exit check | Both opt-in real turns ran and passed (local DSV4.1: 28 prompt / 7 generated, eos, 17.2 s; cloud Syn Large Text: 50 / 24, 0.76 s), so Phase 1's exit criteria are met. The cloud case walks the spawn's candidates and treats a provider's HTTP 4xx as "this model cannot serve the run" rather than a defect. Markers removed afterwards. |
| 2026-11-17 | phase 1 backends | `CloudAgentBackendLogicTests` (18, one opt-in) and `LocalAgentBackendLogicTests` (22) landed; `LocalAgentBackend` added (per-step engine claim, family rendering through `formatConversation`, shared tool mask, result/usage mapping); `LocalSubagentCostEstimate` + `PrefillThroughput` added and wired into `LocalAgentBackend.make` and the engine's prefill meter; `AgentRunContext.workerSystemPrompt` gained the family flags. 70 tests green (new suites + `SubagentLoopTests`, `TurnToolMaskTests`, `AgentStepBudgetTests`), `ServerControllerSmokeTests` 30 green in isolation. Next: one opt-in real turn per backend, then Phase 2. |
| 2026-11-17 | phase 0 build | Models, backend protocol + scripted backend, run context, run-scoped harness/prompt, loop and 14 tests landed; app + test targets build; `SubagentLoopTests` green. Session ended looping on compile/lock/isolation errors; picked up, fixed and closed out. Next: Phase 1 (CloudAgentBackend tests + LocalAgentBackend). |

---

## 16. Open questions

- Cloud provider defaults: Phase 3 shipped the Settings field
  (`dynamoe_subagent_cloud_model`, an upstream model id). Resolution stays
  stored value → coordinator's model when it supports tools → first discovered
  tool-capable model. Still open: whether the CHEAPEST tool-capable profile
  should lead that chain instead of the coordinator's, which needs a price
  source the registry does not carry.
- Model-backed default: decided in Phase 3 — it stays an explicit Settings
  choice. The picker offers "Role pipelines (no model)", "Local model" and
  "Cloud model"; an unset value keeps the legacy pipelines. Sending work to a
  model (and, for cloud, off the machine) is the operator's decision, not
  something an omitted `spawn_subagent` argument should infer.
- Worktree location: decided and shipped — `~/Library/Application Support/
  DynaMoE/worktrees/<runId>`, with the run's own tools rooted at that directory.
  A `codebase_search` index is keyed by workspace, so a large repository pays
  for a second index on its first worktree run; measure that before treating it
  as a cost. The inspector offers Reveal Worktree AND Apply Changes (Phase 5);
  the button is a second caller of the same approval card and apply service,
  not a bypass.
- Local prefix preservation: measured as a per-step claim, not a lifetime lease
  (§8). What is still unmeasured is how often a coordinator turn lands between a
  run's steps and forces a re-prefill. Instrument that before proposing a
  lifetime lease again.
- Interactive preemption loses the in-flight step: the run is cancelled with a
  reason rather than paused. Phase 3's `AgentRunStore` is what makes a resumable
  pause (paused status, serialized messages, step index) affordable.
- AFM tool loop: deferred. The FoundationModels framework exposes a `Tool`
  protocol; a bridge is possible but unproven, and the writers exit criteria do
  not need it.
- Do subagent turns appear in `ServerRequestLog`? (Proposed: no, a dedicated
  run store is the record; the log is for the model-serving wire.)
- Operator-initiated apply: decided and shipped in Phase 5. The inspector's
  Apply Changes button raises the same `SubagentApprovalCenter` request a tool
  call would and delegates to `SubagentApplyService`. Deliberately NOT built:
  an inspector Discard Worktree control. The 14-day sweep and Reveal Worktree
  cover cleanup for now, and a discard would need its own recorded outcome so
  the inspector could distinguish "thrown away" from "never existed".
