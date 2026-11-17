# DynaMoE Subagent Runtime

Design and working document for the model-backed subagent system. This file is
the reference for a multi-session build: read it first, check the phase
checklist, then continue at the first unchecked item.

Status: **Phase 1 complete (both backends, the local cost refusal and the
opt-in real turns); Phase 2 is next.** Only the cloud provider/model default
(§16) is still an open product question.

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
6. **There is no worktree or overlay support today.** Parallel writers need it.
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
| `Subagents/WorkspaceIsolation.swift` | Git worktree lifecycle and diff extraction |
| `Subagents/SubagentManager.swift` | Rewritten coordinator API (`spawn`, status, message, cancel, relays) |
| `Subagents/SubagentInspectorView.swift` | Inspector UI (replaces the drawer) |
| `Subagents/SubagentTools.swift` | `spawn_subagent` / status / message / cancel / apply-changes tools |

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

`AgentScheduler` is an actor.

- **Local capacity: 1.** A local run acquires the engine lease before its first
  turn and holds it until the run ends. Interactive coordinator turns have
  priority: a queued coordinator turn that needs the engine cancels or pauses
  the local run at its next step boundary (the run is resumable).
- **Cloud capacity: N** (default 3 per provider, global fan-out 6).
- **Apple FM: 1.**
- **Queue:** FIFO within a priority class (interactive > user-requested >
  speculative). A queued run can be cancelled while waiting.
- **Cancellation tree:** stopping the coordinator turn cancels every descendant
  run. Application termination cancels all runs and closes cloud streams.
- **Budgets** are enforced by the loop, not the scheduler; the scheduler enforces
  concurrency and queue timeouts.

Engine lease implementation: generalize the existing `TurnSlotClaim` /
`TurnSlotLedger` rather than add a second mechanism. The server queue, the
coordinator send path and the local backend all acquire the same claim. A local
run holds a claim for its lifetime; `launchTurn` receives it per step.

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

---

## 11. UI

- **Inspector** (replaces `SubagentDrawerView`): list of runs for the current
  session with status, backend, model, duration, usage and cost; a detail pane
  with live streamed text/thinking, tool timeline, artifacts, diff, report and
  controls (stop, steer, approve, apply changes, reveal worktree).
- **Inline spawn card** in the transcript: role, backend, live status, short
  summary, link to the inspector.
- **Notifications** for a finished background run; unread reports are injected
  at the start of the next coordinator turn on both paths (fixes the weights-path
  orphan).
- **Settings**: default backend, cloud model for subagents, read-only
  auto-approve, local cost limit, max concurrency, worktree root, retention.

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
| `WorkspaceIsolationTests` | worktree create/diff/cleanup, non-git refusal |
| Opt-in | one real local turn, one real cloud turn |

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
  run's next step re-prefills. `AgentScheduler` (§8) owns the lifetime lease.
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

- [ ] `AgentScheduler` with local/cloud capacity and cancellation tree
- [ ] `SubagentManager` rewritten to spawn `AgentRun`s
- [ ] Spawn approval card + per-call policy in `executeTool`
- [ ] `spawn_subagent` schema: backend, preset, budget, isolation
- [ ] Stop propagation; relay on a finished background run

**Exit criteria:** the coordinator can delegate a real task on either backend
and receives a report; a stopped turn cancels its children; no write reaches the
main tree without approval.

### Phase 3 — Persistence and UI

- [ ] `AgentRunStore` + retention
- [ ] Inspector with live stream, timeline, diff and controls
- [ ] Inline spawn cards, notifications, unread reports
- [ ] Settings

**Exit criteria:** runs survive relaunch, are fully observable, and can be
cancelled or steered from the UI.

### Phase 4 — Writers and polish

- [ ] `WorkspaceIsolation` worktrees + `apply_subagent_changes`
- [ ] Verifier and Reviewer presets
- [ ] Cost/usage dashboard
- [ ] Optional AFM backend
- [ ] Performance pass

**Exit criteria:** two writer runs can work in parallel without touching the
main tree, and the coordinator applies one diff with approval.

---

## 14. Invariants (do not break)

1. A local run never launches a turn without holding the engine lease.
2. A run never mutates `AgentHarness.shared`, the shared grammar registry
   outside its turn, or the coordinator's loaded tools.
3. `tools_load`, `tools_discover`, `tools_unload` and `spawn_subagent` are never
   offered to a run.
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
| 2026-11-17 | design | Architecture approved; decisions recorded; doc created; Phase 0 started |
| 2026-11-17 | phase 1 exit check | Both opt-in real turns ran and passed (local DSV4.1: 28 prompt / 7 generated, eos, 17.2 s; cloud Syn Large Text: 50 / 24, 0.76 s), so Phase 1's exit criteria are met. The cloud case walks the spawn's candidates and treats a provider's HTTP 4xx as "this model cannot serve the run" rather than a defect. Markers removed afterwards. |
| 2026-11-17 | phase 1 backends | `CloudAgentBackendLogicTests` (18, one opt-in) and `LocalAgentBackendLogicTests` (22) landed; `LocalAgentBackend` added (per-step engine claim, family rendering through `formatConversation`, shared tool mask, result/usage mapping); `LocalSubagentCostEstimate` + `PrefillThroughput` added and wired into `LocalAgentBackend.make` and the engine's prefill meter; `AgentRunContext.workerSystemPrompt` gained the family flags. 70 tests green (new suites + `SubagentLoopTests`, `TurnToolMaskTests`, `AgentStepBudgetTests`), `ServerControllerSmokeTests` 30 green in isolation. Next: one opt-in real turn per backend, then Phase 2. |
| 2026-11-17 | phase 0 build | Models, backend protocol + scripted backend, run context, run-scoped harness/prompt, loop and 14 tests landed; app + test targets build; `SubagentLoopTests` green. Session ended looping on compile/lock/isolation errors; picked up, fixed and closed out. Next: Phase 1 (CloudAgentBackend tests + LocalAgentBackend). |

---

## 16. Open questions

- Cloud provider defaults: which profile/model should a subagent use when the
  user has several? (Proposed: a dedicated "Subagent model" setting, defaulting
  to the cheapest configured profile that reports tool support.)
- Worktree location: inside Application Support avoids polluting the repo, but
  some tools assume the repo root. Validate with the existing `find_files` and
  `codebase_search` tools.
- Local prefix preservation: a run holds the engine for its lifetime. Is that
  acceptable for long runs, or should a run yield after N steps and accept a
  re-prefill? (Proposed: hold, with a yield on interactive priority.)
- AFM tool loop: the FoundationModels framework exposes a `Tool` protocol; a
  bridge is possible but unproven. Keep it optional.
- Do subagent turns appear in `ServerRequestLog`? (Proposed: no, a dedicated
  run store is the record; the log is for the model-serving wire.)
