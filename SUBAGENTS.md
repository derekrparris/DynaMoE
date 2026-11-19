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
               |        +-- AgentReportBuilder     findings/summary from the transcript
               |        +-- AgentModelBackend
               |                 +-- LocalAgentBackend
               |                 +-- CloudAgentBackend
               |                 +-- ScriptedBackend (tests)
               |
               +-- SubagentApprovalCenter   the operator's card queue
               +-- AgentRunStore            durable JSON per run
               +-- WorkspaceIsolation       git worktree for writers
```

### Module map

| File | Responsibility |
|---|---|
| `Subagents/AgentRunModels.swift` | `AgentRunRecord`, transcript entries, report, usage, status, backend descriptor, events |
| `Subagents/AgentRunContext.swift` | Per-run tool scope, loop guards, mode, working directory, approval policy |
| `Subagents/AgentModelBackend.swift` | Backend protocol, request/turn types, `ScriptedBackend` |
| `Subagents/CloudAgentBackend.swift` | Cloud provider backend over `CloudChatClient` |
| `Subagents/LocalAgentBackend.swift` | Local engine backend over `CompletionEngine`, under `TurnPriority.background` |
| `Subagents/HeadlessAgentLoop.swift` | The loop: render, call, parse, approve, execute, append, repeat, compact |
| `Subagents/AgentReportBuilder.swift` | Derives the report (findings, summary, open questions) from the run's own transcript |
| `Subagents/AgentScheduler.swift` | Capacity, queueing, engine lease, cancellation tree |
| `Subagents/AgentSpawn.swift` | The spawn resolver: preset, tools, budget, isolation, backend, policy |
| `Subagents/AgentRunStore.swift` | Persistence, retention, corrupt-file handling |
| `Subagents/AgentRolePresets.swift` | Role presets: tool set, budgets, approval policy, isolation |
| `Subagents/WorkspaceIsolation.swift` | Git worktree lifecycle, base-commit snapshots, `git apply`, cleanup and stale sweep |
| `Subagents/SubagentApply.swift` | The approved path that lands a run's worktree diff, used by both `apply_subagent_changes` and the inspector's Apply Changes (`SubagentApplyControl`, single-flight per run) |
| `Subagents/SubagentApprovalCenter.swift` | The operator's approval queue: card lifetime, the Turbo rules and the bound (32) |
| `Subagents/SubagentRelay.swift` / `Subagents/SubagentRelayWait.swift` | `[SUBAGENT_RESULT]` line format, relay marking, the bounded wait for in-flight runs |
| `Subagents/SubagentNotifier.swift` | Finish notifications (banner, frontmost and permission rules) |
| `Subagents/LocalSubagentCost.swift` | The local cost estimate and the pre-spawn refusal |
| `Subagents/SubagentManager.swift` | Coordinator API (`spawn`, status, message, cancel, relays, apply, discard) plus the live→record projection |
| `Subagents/SubagentInspectorView.swift` | Inspector UI (replaces the drawer), worktree controls, conversation cost strip |
| `AgentHarness.swift` | Tool implementations, including `spawn_subagent`, status/message/list and `apply_subagent_changes` |
| `EngineTurnSlot.swift` | `TurnSlotClaim` / `TurnSlotLedger`, the single-flight slot the server queue, chat sends, loads and subagent steps share |

Deferred (not in the tree because no file exists): the Apple Foundation Model
backend. It is a §16 open question, not a landing.

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
    var maxSteps: Int             // `.standard` is 24; a preset may raise it
    var maxCompletionTokens: Int  // completion tokens of the LAST request
    var maxWallClockSeconds: Int  // `.standard` is 600
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
    /// When an isolated run's diff was landed, and on which branch. Written by
    /// the apply service; a restored run still knows its work landed.
    var appliedAt: Date?
    var appliedToBranch: String?
    /// The GIT-IGNORED files an apply kept because a patch cannot carry them.
    /// Their presence pins the worktree against the sweep.
    var keptIgnoredFiles: [String]?
    /// The repository the worktree was cut from and the commit it branched at.
    /// Load-bearing, not documentation: a cardless `git_commit` re-verifies the
    /// worktree's linkage and HEAD against these, and refuses rather than
    /// guessing when the root is absent.
    var repositoryRoot: String?
    var baseCommit: String?
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
    /// A billed token count the app had no price for. Never a reason to report
    /// $0: the merge keeps "unpriced" distinct from "free".
    var costIsUnpriced: Bool
    /// The size of the LAST request, i.e. what the run's context window was
    /// carrying — not the sum of every step's prompt, which is not a context
    /// size and read absurdly high.
    var latestPromptTokens: Int
    var latestCompletionTokens: Int
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
- Backend error → `.failed` with the error. `CloudAgentBackend.runTurn` retries
  exactly once, and only when the failure is retryable AND the receipt proves
  nothing was sent; a spent request is never re-sent or re-billed.

The loop never invents a finding. A finding must cite a tool result or the
model's own text; the report renderer refuses an empty evidence string.

In-loop context bounds: the history is compacted only once it passes a
character budget, never on turn count. Folding rewrites bytes in the middle
of the prompt, and the local backend reuses its KV prefix exactly when each
step's prompt extends the previous one byte for byte — folding on turn count
made every step from the third on re-render the turn that had just aged, so
no multi-step local run ever reused its pin. Under the budget every step is
a pure extension; folding runs only when the budget demands it, one
deliberate re-prefill per fold cycle.

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
- The cloud request carries temperature, top-p, max tokens and the native tool
  declarations. `thinking` and `stopSequences` are LOCAL-TEMPLATE fields and are
  not mapped onto provider fields: the spawn path passes an empty stop list, and
  the reasoning channel is the local families' own convention. A future cloud
  reasoning switch belongs here, not in the request struct.
- Cost from `CloudCostModel`; persisted per run.
- **No engine lease.** Parallel-safe. Default pool size 3 per provider, global
  fan-out 6.
- Retries: exactly one, and only for a failure that proves nothing was billed —
  a transport error the transport reports as unsent, or a `429`. A 4xx never
  retries. A 5xx never retries either: the transport records `spent: true` for
  every 5xx (the provider answered, so the prompt may already have run), and a
  retry can be invoiced a second time for a turn nobody sees.

### LocalModelBackend

- Renders the run's conversation with the loaded model family's formatter
  (reuse `ChatPromptFormatter.formatConversation` and the family flags the
  server path uses).
- Installs the run's tool schemas into `GrammarConstrainedSampler` around the
  turn (snapshot + restore), exactly as `ServerController.withTurnToolMask` does.
- Calls `CompletionEngine.launchTurn` with the run's own `sessionId` and the
  rendered prompt; consumes the stream; uses `TurnResult.parsedToolCalls` for
  calls and `generatedTokenIds` for history.
- Holds the engine lease **per step**, not for the run: `launchTurn` consumes a
  claim as soon as it mints the generation task and the ledger holds one, so a
  lifetime claim would block every chat turn and server request for the run's
  whole life. `waitForEngineSlot` reserves with `TurnPriority.background`, so an
  operator's send can tell a subagent's PREPARATION (which it may stop) from a
  chat turn's or a model load's (which it may not). Interleaving another turn
  between two steps costs the run a re-prefill; measuring that is a §16 item.
- **Cost estimate + refusal:** before the first step, compute the coordinator's
  re-prefill cost from the pinned prefix length and the recent prefill rate.
  If the estimate exceeds `dynamoe_subagent_local_cost_limit` (default 120 s),
  the spawn is refused with the estimate unless the operator overrides it on the
  approval card.
- Tool calls are read from the family's rendered dialect by the same parser the
  chat turn uses; there is no separate pre-spawn capability gate. A family
  whose output the parser cannot recognise yields no calls (a final answer),
  and a `toolfreeze` turn that produced no complete call fails the run by name
  in `HeadlessAgentLoop`.

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
  write. A read-only LOCAL spawn runs without a card (its containment rules are
  below); a spawn that sends work off-device always gets one, including in plan
  mode.
- **Per-call approval** applies to state-changing calls when the policy says
  every change, when the workspace is not isolated, or when an isolated call's
  writes cannot be PROVED to stay inside its worktree. Isolation is a promise
  about the call's destination: `file_write`/`file_edit` are approved silently
  only when the resolved (symlink-resolved) path is inside the worktree, a
  `git_commit` only when it runs there AND the run's repository root is recorded
  (the linkage/HEAD re-verification needs it; without it the commit is refused,
  never assumed safe), and `shell_run` — arbitrary code with its own `cwd` —
  always gets the card. A worktree existing is not evidence that a call wrote
  inside it. A path whose components include `.git` never counts as "inside"
  even when it is: a linked worktree's `.git` is a regular FILE in the sandbox,
  and rewriting it retargets every later git call, so it gets its own card, and
  `snapshot`/`apply` re-verify the worktree's git linkage before staging or
  landing anything.
- **The READ half is enforced too, in every policy.** A read whose resolved path
  (symlinks walked, `..` applied the way the kernel applies it) is outside the
  run's workspace — or touches `.git`, where a remote URL's embedded credential
  lives — gets a card. The read also TAINTS the run: egress tools
  (`web_fetch`, `web_search`, `subq_decide`) are carded for the rest of its life
  once anything outside the workspace has been read, because a secret committed
  INSIDE the workspace is readable cardlessly and egress is the other half of
  the same leak. A run that has only read its own workspace keeps its web tools
  cardless, so research runs do not become click-through. This closes the
  prompt-injection chain that ended in "read `~/.ssh/id_ed25519`, post it out".
- The card is an `ActionApprovalManager` record rendered in the run's INSPECTOR
  (and the run list's approval stack). It is not injected into the coordinator's
  tool timeline: the coordinator is blocked waiting for the tool, so a row there
  would claim a call it is not making. The queue is bounded at 32 cards, and an
  unanswered card resolves as a refusal at the configured timeout (10 min).
- Turbo Mode may bypass routine writes; it may **never** bypass
  `targetsApplicationPreferences`, a containment-escape card (an
  out-of-workspace write or read, a `shell_run`, a `.git` metadata write inside
  the sandbox) or `apply_subagent_changes`, and a run in plan mode is `.deny`.
- `subq_decide` keeps its existing refusal/approval semantics, and its card
  compounds with the taint rule: after an out-of-workspace read it is carded
  even against the default endpoint.

### Workspace isolation

- Writer in a git repository: `git worktree add` under
  `~/Library/Application Support/DynaMoE/worktrees/<runId>` on branch
  `dynamoe/agent/<shortId>`; the run's working directory is that worktree.
- On completion, produce `changedFiles` and a unified diff in the report.
- The coordinator applies changes with a new approved tool
  (`apply_subagent_changes`) or a merge; the worktree is removed after a
  successful apply (and a spawn that never reached its loop discards the
  worktree it was given). A cancelled run's unapplied worktree is KEPT — it
  is the only full copy of the run's work, and the operator may still apply
  it from the inspector; the stale-worktree sweep protects it too.
- Applying refuses a main tree mid-merge, mid-rebase, mid-cherry-pick,
  mid-revert or mid-bisect, by name, before any byte is touched — landing a
  patch into someone else's conflict resolution or history walk is never this
  tool's call.
- A snapshot registers new files as intent-to-add (`git add -N`) rather than
  staging content: a linked worktree shares the operator's object DB, and
  staging blobbed every file the run touched into it just to read a patch
  back. The patch carries the same bytes either way. Two honest limits follow
  from reading the patch as text, and both are refused by name with the
  worktree kept — the operator retrieves or converts the work by hand: a file
  whose content is not valid UTF-8 cannot be expressed, and a nested git
  repository can only be expressed as a gitlink. `git apply` exits **0** on a
  gitlink while writing an empty directory and leaving the nested files
  unapplied, so treating its exit status as success reported a landed apply and
  deleted the only copy. `snapshot` refuses one, and `apply` checks a supplied
  change set again before touching a byte.
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
  is launched with `TurnPriority.background`, so an interactive send can cancel
  it at its next boundary. That is a cancel, not a yield-and-resume: see the
  next bullet.
- **Interactive priority is cancel-and-report, not pause/resume.** An operator's
  FRESH send that finds a background turn active cancels it
  (`cancelBackgroundTurn`), waits for the engine to drain (`awaitIdle(boundedBy:)`)
  and runs the send once (`preemptionRetry` in
  `startAutoregressiveGeneration`). The subagent's run is cancelled at its next
  boundary and reports why; it is not resumed. SubQ chose this over
  `pause_and_resume` (0.81 vs 0.10) because resumability needs a paused status,
  a serializable loop state and store work that belongs with Phase 3. A
  CONTINUATION of the coordinator's own turn NEVER preempts, at any timeout: the
  run holding the slot is usually the turn's own background child, so its step
  waits bounded (120 s) for the slot and then ends with the reason instead of
  killing the work it exists to collect. Only a fresh send falls back to
  cancellation, and only when `hasActiveBackgroundTurn` says the holder can
  actually be cancelled — a server request, a chat turn or a model load is never
  stopped to make room. A Stop is scoped to the TURN that minted it
  (`cancelTurnRuns`): runs spawned by earlier turns of the same conversation
  keep working and keep their unapplied worktrees.
- **Cloud capacity: N** (default 3 per provider, global fan-out 6, both
  overridable through `dynamoe_subagent_cloud_per_provider` /
  `dynamoe_subagent_cloud_global`).
- **Apple FM: 1** (reserved; the backend is deferred, §16).
- **Queue:** FIFO within a priority class (interactive > user-requested >
  speculative), and priority ordering is per CLASS, so one class's head cannot
  starve a class that still has capacity. A queued run can be cancelled while
  waiting, and the wait is bounded by `dynamoe_subagent_queue_timeout`
  (default 120 s; every class's ticket is held for its holder's whole life, so
  admission is floored at 1200 s — a standard 600 s run plus its 600 s approval
  park — and the configured value can only raise it); a timeout is a refusal
  that names the class. `AgentSchedulingPriority` keeps all three classes because
  the scheduler's ordering is written against them, but every spawn resolves to
  `.userRequested`: nothing in the spawn schema or Settings produces
  `.interactive` or `.speculative`, so their arms are ordering vocabulary rather
  than a behavior a caller can reach.
- **Cancellation tree:** `register(runId:parentRunId:parentSessionId:cancel:)`
  records how to stop a run. `cancel(runId:)` walks descendants.
  `cancelTurnRuns(parentTurnId:)` is what the Stop button calls, so a Stop
  reaches exactly what the current turn spawned (continuations carry the same
  id); `cancelSession(sessionId:)` / `cancelRuns(parentSessionId:)` is the
  session-wide fallback for a path that never minted a turn id.
  Application termination cancels every run and every pending approval.
- **Budgets** are enforced by the loop, not the scheduler; the scheduler enforces
  concurrency and queue timeouts. A cloud run also carries the operator's
  standing spend ceiling: `dynamoe_subagent_cloud_max_cost_usd` (Settings →
  Agent & Tools → *Cloud spend cap*, default $5, 0 clears it) is applied by
  `AgentSpawnResolver` to every cloud spawn that names no `max_cost_usd`, so a
  preset's `nil` ceiling or a read-only spawn that raises no card still has a
  bound the operator can see. A cap can only bind on a priced model, so while
  one is set a cloud run on a model with no recorded price is refused by name
  rather than run unbounded.

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
- A delete removes EVERY bucket copy of a run before tombstoning the id. A
  conversation delete re-scopes an isolated survivor by writing the new copy
  before removing the old one, so a crash between the two leaves the same run in
  two buckets; deleting only the newest left the stale copy to resurface on the
  next launch (retention-exempt whenever it claimed worktree work).
- Transcript tool outputs are truncated to a per-entry budget before writing.
- Retention: keep the newest 200 runs or 30 days, whichever is larger; never
  prune a run younger than 24 hours. Runs of a deleted session are pruned with
  it — with one deliberate exception: an UNAPPLIED isolated run's record is
  re-scoped to the unscoped bucket instead (it rides every inspector, and
  `prune` keeps skipping it), because that record is the only thing keeping
  its worktree out of the 14-day sweep. A deleted conversation may not
  silently destroy work the operator never saw; the work goes when the operator
  applies or clears it, or never.
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
  orphan). Restored reports relay oldest first, and deleting a run (clear,
  discard, or a conversation delete) withdraws its delivered and pending banner
  so a tap cannot select a run no row can show. A stopped synchronous spawn does
  not consume its unread state: the partial report lands during the loop's
  unwind and is relayed on the next turn.
- **Settings**: default backend, cloud model for subagents, the cloud spend cap
  (see §8), local cost limit,
  max concurrency (per provider and global), retention (runs and days),
  notifications, and the subagent approval timeout. The AGENT STEP CAP in the
  same tab is the coordinator's own loop budget (`AgentStepBudget`), not a
  subagent's: a run's steps come from its preset and the spawn's explicit
  `max_steps`, so changing the coordinator cap cannot silently shorten a
  delegated run.

---

## 12. Testing

Pure-logic and fake-backend tests are the default; real-model tests are opt-in.

| Suite | What it proves |
|---|---|
| `SubagentLoopTests` | step loop, tool execution, budget, cancellation, guard trips, report derivation, the per-run harness context, its guards, the READ containment rules and the egress taint |
| `AgentSpawnPolicyTests` | spawn approval and backend policy, Turbo/app-preference/containment rules, cost and isolation refusals |
| `AgentRunIntegrationTests` | the manager's run lifecycle end to end (spawn → events → terminal record) against fake backends, including a run parked on an approval card across a conversation delete and a quit path |
| `AgentSchedulerTests` | capacity, priority, cancel tree, queue timeout |
| `AgentRunStoreTests` | round trip, retention, corrupt file, unknown schema, mutate vs stale-load, fault-injected durability (write/fsync/rename) and the duplicate-copy delete after a crashed conversation delete |
| `EngineTurnSlotTests` | the claim ledger under concurrent chat/server/subagent reservations, preparation-claim cancellation, and the priority match between a claim and its launch |
| `CloudAgentBackendLogicTests` | event mapping, fragmented tool-call assembly, retry rules |
| `LocalAgentBackendLogicTests` | prompt rendering, mask install/restore, cost estimate and refusal |
| `SubagentApprovalCenterTests` | the Turbo bypass rules, timeout resolution, the request/approve round trip and the queue bound |
| `SubagentRelayWaitTests` | the bounded wait for in-flight runs (deadline, empty set, Stop) |
| `SubagentRecordProjectionTests` / `SubagentRunLinkTests` / `LocalSubagentRelayTests` | record projection (including the terminal-record guard against a lagging event), run links, the relay line format |
| `WorkspaceIsolationTests` | worktree create/snapshot/apply/cleanup, committed work included, conflicts refused, non-git refusal, the apply tool and a scripted isolated run end to end, and the operator's Apply Changes path (offer gate, one approval request, declined card, applied-then-nothing) |
| Opt-in | one real local turn, one real cloud turn, one real local writer run + operator apply, and the full `spawn_subagent` → model-backed run happy path (it needs a loaded engine, which the pure suites deliberately do not have; `AgentRunIntegrationTests` covers the same manager path with a scripted backend) |

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
- [x] Tests: loop, context, report, approval policy (`SubagentLoopTests`; the
      suite has grown with every phase — §12 names what each one covers rather
      than a count that goes stale)

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
  persists on change, which is what makes "unread" survive a relaunch. A run
  parked on an approval card publishes `.waitingForApproval` and returns to
  `.running` when the card is answered, so a blocked run is not drawn as busy.
- **Notifications are a Dock bounce plus an opt-in banner.** The bounce happens
  whenever the app is not frontmost and needs no permission; the banner is OFF
  until the operator enables it in Settings, is posted only while the app is
  not frontmost, is skipped under XCTest (a headless runner must not meet a
  permission dialog), and names the run rather than claiming more than the
  summary did.
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
- **`git add -N` before every snapshot, so untracked files are included.**
  Intent-to-add records the PATHS without writing their contents into the
  operator's shared object store, which `add -A` did for every file a run
  touched. The index is not part of what is applied. `--name-status -z` separates the STATUS from its path with
  NUL as well as entries from each other (`A\0path\0`, `R100\0old\0new\0`); a
  tab-only parser silently reports no files, which is how the first cut passed
  every apply case while listing nothing in the inspector.
- **Applying is `git apply`, not a merge.** `git apply --check` validates the
  whole patch before anything is written, so a conflict leaves the main tree
  exactly as it was and is reported by name. The apply itself is NOT atomic:
  a destination that fails partway through (read-only, full disk) is why every
  target is pre-flighted for writability and the touched files are snapshotted
  and restored if the write still fails. A merge would bring history and
  create merge state, and the coordinator may have uncommitted work. Applied
  changes are deliberately left UNSTAGED: the coordinator (or the operator)
  decides what becomes a commit. A merge or rebase already in progress in the
  main tree is refused rather than interleaved with someone else's conflict
  resolution.
- **Lifecycle: every terminal run keeps its worktree until it is applied or
  cleared — cancelled runs included.** The record's diff is bounded (64 KB for a
  report, plus the per-entry transcript budgets), so the worktree is the only
  copy that can always be applied in full. A cancelled writer's tree used to be
  discarded on the Stop, which destroyed the only copy of work the operator
  never saw; it is kept now, and the record still names it so the inspector
  offers the same keep-or-clear decision a completed run gets.
  `apply_subagent_changes` re-reads the LIVE diff, applies it, writes
  `appliedAt` synchronously, then removes the worktree — unless the tree also
  holds git-ignored files a patch cannot carry, in which case it stays and the
  apply names them; a second call is a no-op that says so. A launch sweep
  removes worktrees older than 14 days, but NEVER one whose record has not
  been applied: the record keeps only a bounded diff, so the worktree is the
  only full copy. A run whose work is entirely in git-ignored files refuses to
  apply rather than reporting "no changes" and deleting the tree. A launch
  sweep also closes records left `.running` by a crash, so they are terminal
  and clearable. Retention never deletes an unapplied isolated run's record
  either: the record is what keeps the sweep from treating that worktree as
  orphaned.
- **Approval moves to spawn for an isolated writer.** `.spawnOnly` means the
  operator approves the delegation once, with the worktree and branch in front
  of them, and the run's writes inside its own tree need no per-call card.
  `apply_subagent_changes` is itself state-changing, so landing the work always
  gets its own card, and Turbo Mode may never bypass it.
  `targetsApplicationPreferences` stays absolute on every path. A spawn's
  `backend: "cloud"` is off-device consent: when the Settings choice does not
  already name Cloud, even a read-only spawn gets a card that says so. A
  run-scoped approval waits for a BOUNDED time, so an unanswered card cannot
  hold the scheduler ticket forever.
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
- [x] Discard Run control (`SubagentManager.discard`), which resolves the
      record's worktree and clears it under the same guards as Apply

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
  pure and testable; `apply` names the RECORDED repository root, the same
  answer `WorkspaceIsolation.open` uses: a recorded root outranks the
  worktree's writable `.git` pointer, and a worktree the record no longer owns
  is refused rather than followed. Git's own answer is the fallback for an
  older record with no stored root, and the controls pass the same
  execution-unwinding gate the inspector does. A worktree that is gone offers
  nothing: `canApply` requires the path to exist.
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
3. `tools_load`, `tools_discover`, `tools_unload`, `spawn_subagent`,
   `apply_subagent_changes`, `get_subagent_status`, `send_subagent_message`
   and `list_subagents` are never offered to a run.
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
| 2026-11-19 | PR #104 review round 10 | Five merge blockers plus the should-fix batch closed. **B1 — the environment allowlist hit the shared helper.** `toolEnvironment()` is the INHERITING environment again (minus this app's own `SUBQ_API_KEY`); `toolEnvironment(scoped: true)` is the allowlist, and `runProcess`/`gitEnvironment`/`WorkspaceIsolation.git` choose it from the `executingInRunScope` task-local, so ordinary `shell_run`/git/MCP children get SSH-agent, proxies, virtualenvs and `SDKROOT` back while a subagent's children stay scrubbed. **B2 — cross-session status tools.** `list_subagents`/`get_subagent_status`/`send_subagent_message` are conversation-scoped: a foreign run answers "not found", cannot be steered, and cannot be marked relayed by the wrong chat. **B3 — Stop declined every card.** `ActionApprovalManager` gained `ActionApprovalScope` (`session`/`subagentRun`), registered when the card is rendered; `stopGeneration` cancels its own session's cards and the stopped turn's runs' cards instead of calling process-wide `cancelAll` (kept for quit). **B4 — `awaitIdle` spun when cancelled.** It returns false on `Task.isCancelled`, so the swallowed `Task.sleep` can no longer burn the main actor to the deadline. **B5 — cleanup deleted a surviving tree's branch.** `cleanup` deletes the branch only once the tree is gone and returns whether it is; `SubagentApplyOutcome.worktreeRemoved`, the apply tool and `get_subagent_status` report the truth, and `discard` now removes the tree BEFORE the record and refuses when it cannot. Should-fixes: Discard reports failure and `canDiscard` honours a repository-wide apply; build output (`.build`, `node_modules`, `__pycache__`, `.DS_Store`, …) is no longer "kept ignored work"; a final turn's abandoned tool calls reach the transcript; a failed first request counts as a step; the notifier covers `.blocked` and `withdraw` clears the queued banner; AFM marks a relay delivered only when the transcript carries it; `AgentRunStore` fsyncs the directory; `hardTrim` can no longer grow a message; overlapping scoped masks COMPOSE; the apply card waits for its destination read; Clear Finished always confirms and can be switched to "All Runs"; settings observes the manager; branches use 12 hex characters. Tests: env passthrough/scoping (including through `runProcess`), cross-conversation status/list/steer, cleanup-keeps-branch, discard failure, build-output filtering, final-turn calls, failed-step counting, fold bounds, two live mask scopes, the registered-successor stale cancel (DEBUG seam), scheduler ticket release, and the new 1200 s floor. |
| 2026-11-19 | PR #104 review round 9 | Three blockers and the should-fixes closed. **B1 — a deferred continuation wedged every composer.** A continuation does not clear `generationTask` (the next step owns that handle), so when one lost the slot and deferred, the stale handle made `releaseQueuedGenerationFlags`'s live-task guard a no-op forever; deleting the conversation then left `isGeneratingText` true with `generatingSessionId` naming a session that no longer existed, so every composer refused sends and none drew a Stop. The deferring branch nils the stale handle (the queued send publishes its own on launch) and the session-scoped cancel releases on `generatingSessionId` alone. **B2 — activations swapped the engine with no claim.** `activateAppleFoundationModel`/`activateCloudModel` ran `awaitGenerationTeardown()` and then released the shard buffers and retired the DeepSeek runtime; between the wait and the swap is a hop the background poller can claim in. Both take `reserveTurnSlot()` and refuse when nil, as the weights load does. **B3 — the read/egress half had no enforcement.** `runScopeDecision` now cards a read whose resolved (symlink-walked) path is outside the run's workspace or touches `.git` — in EVERY policy, because every policy is a rule about writes — and the card TAINTS the run, so `web_fetch`/`web_search`/`subq_decide` are carded for the rest of its life (a secret committed inside the workspace is readable cardlessly; egress is the other half). A run that only reads its workspace keeps its web tools cardless. `approvalDestination` names a read's path. A cardless `git_commit` is REFUSED when the run's repository root is unknown instead of skipping the linkage re-verification. Should-fixes: a Stop's transcript and partial report are persisted on the already-finished path; deleting the generating conversation stops its turn (the weights loop had no vanished-session exit) and `startAutoregressiveGeneration` refuses a deleted session; a CONTINUATION never preempts at any timeout (it ends the step with the reason), only a fresh send cancels, and only when `hasActiveBackgroundTurn` says the holder is cancellable; `awaitIdle(boundedBy:)` bounds the drain so an uncancellable holder cannot lock every composer; Stop cancels by the ENGINE id `launchTurn` returned instead of the priority-blind `cancelActiveTurn()` (a cloud turn has no engine turn, so the old path aborted a subagent's step); the turn slot's claim now carries `TurnPriority`, so a run in PREPARATION is visible to preemption; compaction is a hard bound (few blocks no longer returns the history unchanged, and a final oldest-first trim runs when the recent window alone is over budget); non-cancel transport failures keep streamed text (`AgentBackendError.failedAfterPartial`); a gesture-only local turn is a finished run, not `toolFreeze`; cloud `finish_reason` `content_filter`/`refusal`/unknown is a failure, and the new arms are gated on a remote backend so the local decode vocabulary keeps its own messages; `toolEnvironment()` is an allowlist; `existingRecordURL` picks the newest copy like `loadAny`; cancelled runs are relayed only with a partial report; the apply patch lives in a private 0700 UUID directory; `.blocked` survives as its own status; `SubagentStatus` gained it, the inspector's active count and Clear-Finished copy are honest, its icon buttons have accessibility labels, and the approval queue is bounded at 32. Tests: 6 read-containment/egress cases and the git-root refusal (`SubagentLoopTests`), the compaction hard bound, the approval round trip + timeout + bound, the lagging-pump terminal guard, and the cloud partial-failure payload. 338 executed across the 15 subagent/server suites, 2 opt-in skips, 0 failures. Docs: §3 module map matches the tree, §4 has the record's real fields, §6/§8 say per-step claims and no continuation preemption, §7 documents the read/egress rules and the inspector-only card, §11/§12 stopped claiming a shared step cap and name every suite. |
| 2026-11-18 | PR #104 review round 8 | The remaining review findings closed. (1) `AgentUsage.merge` takes the NEWEST non-zero `latestPromptTokens`/`latestCompletionTokens` instead of the larger one, so a compacted (smaller) later prompt still becomes the drawer's number while a usage-less turn leaves the last reporting step's values; the projection test sets the last step's generation (8 of the summed 12) and pins both merge rules at 124. (2) `clearCompleted` reads the LIVE `recordedIsolation` through `SubagentInstance.holdsUnappliedWorktreeWork`, so a pre-loop refusal whose worktree was discarded is clearable in memory on the same turn it becomes clearable on disk instead of only at relaunch. (3) The operator's apply path follows `WorkspaceIsolation.open`'s precedence: the RECORDED repository root outranks the worktree's writable `.git` pointer (a retargeted pointer can no longer make the card name another repository), git's answer is only the no-root fallback, and `SubagentApplyControl.apply` passes the same execution-unwinding gate the inspector checks. (4) Both cost-ceiling guards call one `unpricedCeilingFailure`, so the post-turn refusal names provider-omitted usage instead of claiming no price is recorded. (5) `stageRollback` removes its temp directory when a patch path is unsafe, not only when a copy fails. (6) Two post-turn harness reads in `ContentView`'s detached turn are resolved on the MainActor, removing the `failedToolCallTotalLimit` autoclosure warning and the sibling counter/guard reads beside it. Tests: 170 executed across `SubagentRecordProjectionTests`, `AgentRunStoreTests`, `AgentRunIntegrationTests`, `SubagentLoopTests`, `WorkspaceIsolationTests` and `CloudAgentBackendLogicTests`, 2 opt-in skips, 0 failures; the previously failing context-token case passes. Docs: §7 names the recorded-root precedence and the unwind gate. |
| 2026-11-18 | PR #104 review round 7 | The three merge-gating findings plus the majors. (1) A Stop during a queued send no longer spins the MainActor: the slot wait checks `Task.isCancelled` before sleeping again (`try?` swallowed the thrown sleep, so the loop ran at full speed holding the ledger lock), and both deferred-relaunch tasks check cancellation BEFORE preempting, so a Stop during the task hop cannot destroy another conversation's runs. (2) A cancelled run's `shell_run` child is killed: `ShellRunTool` spawns through `AgentHarness.runProcess`, whose cancellation handler SIGTERMs the process tree and SIGKILLs after 500 ms, and `SubagentInstance.isExecutionUnwound` (cleared where the execution body returns, which is the same actor turn that resumes `waitForCompletion`) makes the apply path refuse a terminal record whose task is still unwinding. (3) Turbo keeps the forced-cloud spawn card (`turboModeMayBypassApproval` consults `spawnsWorkOffDevice`; the Settings choice is still routine). Majors: preemption now cancels only the LOCAL TICKET HOLDER, not every queued local run (`ticketRunIds`); a refused model load rolls the conversation back to the previous model/profile/last-used id instead of leaving it rendering B while the engine serves A; `writeOwnerOnlyAtomically` checks `fsync`/`close`; the relay index builds OFF `signatureLock` and publishes with a settled-id guard; a pre-loop refusal that discards its worktree clears the record's isolation (`forgetDiscardedWorktree`), so `holdsWorktreeWork` no longer pins it forever; the containment card names the call's real destination (`approvalDestination`: a write's path, a shell's `cwd`); `.waitingForApproval` counts as active for the relay wait and the active counts; a cancelled turn keeps its partial text (`AgentBackendError.cancelledAfterPartial`, appended by the loop) and its spend is written for any terminal unwind, not only `.cancelled`; in-loop compaction counts tool-call arguments and the system block/tool schemas. Minors: `git worktree repair` runs only for a MISSING pointer (a retargeted one is refused without repair, which git applies through the pointer and would rewrite a sibling's record), a recorded repository root outranks the pointer in `open`, git reads run with `GIT_OPTIONAL_LOCKS=0`, `save()` bounds on the IO queue, an absent-file delete still tombstones, the tombstone bound covers a bulk clear, apply/cancel record writes await `mutateAsync` instead of blocking the main actor, `suffixDelta` drops a shrink instead of duplicating the answer, tool-call markup is stripped from the turn's prose, a usage-less turn no longer resets `latestPromptTokens`, `CloudTurnFailure` maps to an honest error type, `noPriceRecorded` wins the unpriced-cause merge, and a schema encode failure installs no mask. Docs: Stop is turn-scoped, local admission is 900 s, `add -N` not `add -A`, Discard Run exists. Tests: queued-writer preemption (the queued run survives), the forced-cloud Turbo card, the absent-file tombstone, the discarded-worktree record, compaction counting arguments/overhead, markup-free turn text, the shrink delta, the apply button's unwind gate, and a real `sleep 5` shell killed at a Stop in 0.3 s. `WorkspaceIsolationTests`' retarget test now expects the refusal (and asserts the pointer was not rewritten). 350 executed across the touched suites, 2 opt-in skips, 0 failures. |
| 2026-11-18 | PR #104 review round 6 | Four must-fixes and the should-fixes closed. (1) `withTurnToolMask` no longer snapshots before its lock or restores unconditionally: scoped masks are tracked by identity on a stack, and an exit rebuilds the registry from the base snapshot plus every scope STILL live, so a run whose mask outlives its engine task and a server turn that installs in the freed-slot window cannot leave either vocabulary installed for later turns (`testTheScopeThatEndsFirstDoesNotRestoreOverTheOneStillInstalled`). (2) `WorkspaceIsolation.diffStat` verifies the worktree's linkage and HEAD before its `git add -N .`, so merely opening the apply card cannot write intent-to-add entries into the operator's index through a retargeted pointer. (3) A queued send now publishes BOTH `isGeneratingText` and `generatingSessionId` when it takes over an idle flag, and a release transfers ownership to any remaining queued send instead of clearing the flag out from under it — the "composer refuses sends forever after a completed turn in another conversation" wedge. (4) A cancelled run's interrupted step is not lost: when the loop unwinds after `cancel` persisted the record, the model-backed path writes `outcome.usage` (including the ledger's billed-but-no-turn attempt) back to the record (`testACancelledRunsInFlightSpendReachesItsRecord`). Should-fixes: a worktree whose HEAD moved off its recorded branch is refused by the same guard as the linkage (`verifyWorktreeHead`, also checked by the cardless `git_commit` through a run-scoped `workspaceBranch`); an empty-change apply refuses when the branch holds commits a patch cannot carry, and a non-empty apply keeps the branch when its commit count cannot be proven empty; `deleteSession` moves undecodable record files to the unscoped bucket (so the sweep still protects their trees) and tombstones only after a successful removal; local admission covers the holder's 900 s default and names strict seriality; the local cost override approves the estimate the card showed and refuses when the pin/rate moved; an EMPTY Cloud model setting now refuses instead of falling back to a provider the operator never named; compaction skips prompt-only relay directives; a Stop landing in the hop into `launchTurn` is checked inside the hop; `max_cost_usd: 0` and non-finite ceilings are refused at the card; a billed attempt with no usage block says the provider withheld usage rather than "no price is recorded". Minors: `allowed_tools: []` is refused by name (the tool no longer collapses it to "omitted"); a local dollar ceiling gets a local-specific refusal; the Cloud model field documents the two accepted spellings; 429 honours a clamped `Retry-After` carried on the receipt; the event pump gates `.toolFinished`/`.completed` on `isFinished`; `contextTokens` adds only the LAST step's completion; `safeURL` resolves the existing ancestor and leaf so a symlink cannot leave the tree; `GitController` spawns every git child with `GIT_*` stripped; the chat apply card names the branch, destination and files. Tests: the interleaved mask scope, clearCompleted sparing an unapplied isolated run, `deleteClearableRecords`, `safeURL` traversal/leaf-directory symlinks, `invalidateIfNotPreserved`, `onCacheWillChange` (fires only past every refusal gate and before reset), a real partial `git apply` rollback (the first file is created, the second fails, the tree returns clean), claim release after a real and a failed apply, the detached-HEAD refusal, the cancelled-spend record, the stale-estimate refusal, the empty-tool-list refusal, Retry-After parsing/clamping, and the new last-step context arithmetic. 248 executed across the touched suites, 2 opt-in skips, 0 failures. |
| 2026-11-18 | PR #104 review round 5 | Eight merge blockers closed. (1) A queued relaunch clears the generating state it published in its own `defer` (guarded on session ownership, `generationTask` and other queued tasks), and `cancelDeferredRelaunches(sessionId:)` clears it immediately, so deleting a conversation during the preemption drain cannot leave every composer refusing sends. (2) `deleteSession` honours the rescue write: a failed write leaves the original record (and the whole session directory, untombstoned) in place. (3) A failed rollback keeps its snapshot directory and names the path in the error. (4) `AgentRunRecord.keptIgnoredFiles`/`holdsWorktreeWork` protect an APPLIED worktree kept for git-ignored files from the 14-day sweep, `prune`, `clearCompleted` and `deleteSession`; `noteKeptIgnoredFiles` persists it on both the live and restored paths, and the sweep keeps a UUID-named tree whose record file exists but cannot be decoded (newer schema/corrupt). (5) `pruneStaleWorktrees` probes for the record FILE before treating a tree as orphaned. (6) `.everyStateChange` marks out-of-workspace `file_write`/`file_edit`/`git_commit` as containment escapes, so Turbo cannot bypass them for an unisolated run. (7) The relay line puts `id=` first, parses only that trailer, validates the UUID, and sanitizes the model-supplied role to one line with brackets neutralized and an 80-character bound. (8) Discard is refused in `SubagentManager.discard` while the run or its repository is claimed by an apply, and the inspector disables the button and its confirmation. Majors: the apply card shows `git diff --stat` and the git-resolved destination; `WorkspaceIsolation.git` passes `-c diff.noprefix=false` and a `GIT_*`-scrubbed environment; a cancelled run persists its reason (`lastError`, surfaced as `error` for `.cancelled`); the headless loop resolves tool definitions EVERY step so `tools_load` reaches later requests; `complete` without a summary completes the run and calls after it are recorded as abandoned. Minors: legacy `spawn` honours `notifiesOnFinish`; "Clear Finished" reaches unscoped terminal instances; `replaceItemAt` re-asserts 0600; delete/prune tombstone only after a successful removal; `markRelayed` updates the relay index before the queued write (`mutateLater` alone left a duplicate-relay window); a deferred grammar rebuild is applied at turn start (holding the slot); steering that lands after the final step is recorded; unparseable cloud arguments no longer invent `_raw_arguments`; instance observers are cancelled when trimmed; §6 no longer describes the dead 5xx retry. Tests: 4 new sweep/discard cases, 2 relay-line cases, an out-of-workspace Turbo case, and the no-summary `complete` case — 220 executed, 2 opt-in skips, 0 failures. |
| 2026-11-18 | PR #104 review round 4 | (1) BLOCKER: containment resolved `..` lexically before following symlinks, so `<worktree>/link/../x` read as inside while the write landed outside; `resolvedForContainment` now walks the components in order, resolving each existing one before applying the next `..` (a DANGLING symlink is resolved from `destinationOfSymbolicLink`, because Foundation reports it as absent while an O_CREAT write still follows it), and `testASymlinkFollowedByDotDotEscapesTheWorkspace` plus `testADanglingSymlinkEscapeNeedsTheCard` pin both cases. (2) An unapplied isolated run can be DISCARDED per run from its pane (confirmed destructive action), which deletes the record and cleans the worktree and branch; before this, no path could ever remove one. (3) Plan mode no longer exempts a spawn that resolves to Cloud: its read tools still upload repository content, so `spawnResolvesToCloud` decides the card (including the Settings-resolved backend). (4) Turn identity is a task-local bound by `executeTool` from the value each turn knows, so a tool still running when the operator sends again stamps its run with its OWN turn and Stop cancels the right runs. (5) A spawn checks `Task.isCancelled` after every resolve/approval await and discards the plan's fresh worktree instead of starting a run stamped with a stopped turn. (6) `syncGrammar` refuses to rebuild the shared registry while a scoped mask owns it or a generation is reading it, and defers the rebuild to the next tool boundary; `LocalAgentBackend.collect` now awaits the engine task on the failed-turn path too. (7) `runTool` unwraps BOTH optionals, so a deadline that wins still collects a tool's completed result. Minors: role text is sanitized in both worker prompts; the dead 5xx retry arm is removed (the transport records `spent: true` for every 5xx); the measured prefill rate resets with the checkpoint; the detail pane is keyed by run id; relay marking and the clear-finished scan moved off the main actor (store reads are queue-ordered behind deletes); "Clear Finished" warns when it would discard unread reports; the legacy spawn path names an ignored `isolation: worktree`; the pane reloads on status changes, not only counts. |
| 2026-11-18 | review round 3: two blockers, majors, minors | (1) The result-delivery wait is now Stop-sensitive end to end: `SubagentRelayWait` checks `Task.isCancelled` (the old polling loop spun to its 120 s deadline under cancellation), and BOTH teardown paths guard `!Task.isCancelled` and generation ownership after the wait — the weights path before gathering and again before its relay launch, the cloud path before `takeSubagentRelayDirective`, whose loop-top check had already run. (2) A nested git repository is refused as a gitlink (`nestedRepositoryUnsupported`) in `snapshot` AND from a caller-supplied change set in `apply`, before a byte is touched; `git apply` exits 0 on one while writing an empty directory, and the old success path deleted the worktree, the only copy of the nested files. (3) `SubagentInspectorView` gained `.onChange(of: sessionId)`, clearing the old rows and detail selection before loading the new conversation. (4) `.running` is published only if the run is still unfinished (a synchronous `hasFinished()` lock read) in both execution paths, so a Stop in the scheduler's grant window cannot overwrite a cancelled record. Minors: a block-drop range ends at the first `.user` (steering notes survive compaction); `pathTouchesGitMetadata` folds case (`.GIT` on APFS); the two model-not-ready early returns release a carried `TurnSlotClaim`, and a nil re-claim after preemption re-enters the slot wait instead of launching unauthorised; manual delete and chat retention cancel that session's queued relaunches and a relaunch re-checks the session exists; `mutate` locates the record by id when its conversation was deleted (so an apply's `appliedAt` reaches the re-scoped record) and reports a refused write as nil; the relay index fallback scans on the IO queue; a mid-turn cancellation/deadline records every call it did not run; `shell_run` and `.git` writes are containment escapes in `.everyStateChange` too (Turbo can no longer bypass them); the banner is opt-in and posts only while the app is not frontmost; the cost strip marks a mixed total with `+`; `waitingForApproval` is emitted for an unanswered card; a notification tap that beats the history load is held until the load finishes. (§5 retry, §6 tool parsing, §7 gitlinks/Turbo, §11 notifications and status, §16 cloud defaults corrected.) |
| 2026-11-18 | review crash + majors | Round 3's findings closed. (1) BLOCKER: `deleteSession` re-scoped its survivors from inside `ioQueue.async` and called `saveImmediately`, whose `ioQueue.sync` libdispatch-aborted the process ("dispatch_sync called on queue already owned by current thread") — deleting a conversation that owned an unapplied isolated run crashed the app, and the PR's own delete-session test took the test process down with it. Every synchronous entry point (`saveImmediately`, `mutate`, `prune`, `sweepOrphanedTempWrites`, `flush`) now goes through `onQueue`, which uses a `DispatchSpecificKey` to run INLINE when the caller already holds the store's queue, so the class of defect cannot recur. (2) A fresh send queued behind the preemption drain now publishes the generating state (guarded like the continuation's slot wait), so Stop is drawn for the whole drain instead of the composer showing a plain Send button while a relaunch it cannot cancel is pending. (3) The AFM path marks a relayed report delivered only when `startAppleFoundationModelGeneration` actually started a turn: it returned true for an unavailable on-device model, so the report was consumed having reached no model. (4) The three single-flight invariants round 3 asked for: `LocalAgentBackend.withStepClaim` (the per-step claim is injected/observed so an early throw between grant and launch is provable without a checkpoint), stale-id no-op coverage for `cancelTurn(id:)` against a preparing claim, and the per-repository apply claim via a DEBUG seam (`repositoryKey` is now one function). Minors: the DS launch no longer drops the prefix pin before the runner can refuse — `DeepSeekV41TurnRunner.run` gained `onCacheWillChange`, fired past its own refusal gates and immediately before the resume/reset, which is where `invalidateIfNotPreserved` now runs (the generic path's rule, applied to this one). SUBAGENTS §6's retry sentence now matches the code (one retry, only when the failure proves nothing was billed). Verified: `AgentRunStoreTests` 23 (the crashing case passes, no process restart), `EngineTurnSlotTests` 11, `LocalAgentBackendLogicTests` 24, `WorkspaceIsolationTests` 36, plus the six other subagent suites and the DeepSeek turn-loop/parity suites — 284 executed, 2 skipped (opt-in live cloud), 0 failures. |
| 2026-11-18 | review majors + minors | All five review majors closed. (1) "Clear Finished" no longer disposes unapplied isolated work: `clearCompleted` skips the records the 14-day sweep protects (both its live-instance and stored-record paths — the guard `prune` already has), and `deleteSession` RE-SCOPES those records to the unscoped bucket instead of deleting them: they ride every inspector with their Apply/Clear buttons, `prune` keeps skipping them, a still-working survivor is closed as cancelled with a note, and `summaryRelayed` is set so the relay never injects a deleted conversation's report into an unrelated one; §10 records the exception (write-new-before-remove-old, so a crash between the two leaves a duplicate the sweep still protects). (2) The legacy `spawn` path passes `parentTurnId`, so a Stop reaches the children the stopped turn spawned instead of missing the filter. (3) Chat retention protects every session with a live run (`SubagentManager.liveRunSessionIds`) and cancels a session's runs before its records go — a run whose record is deleted underneath it is invisible, unstoppable, and its terminal write is dropped by the store's tombstone. (4) Manual conversation delete does the same (§10). (5) In-loop compaction is budget-gated: under the 160k-character budget the history comes back UNCHANGED so each step's prompt extends the last byte for byte and the local KV prefix pin survives the step; folding runs only when the budget demands it, one deliberate re-prefill per fold cycle instead of one per step (SubQ 0.996). Minors: `send_subagent_message` refuses a finished run instead of reporting `delivered: true` into a channel nothing drains; the Stop race that left a cancelled run's live object carrying a completed report the store never accepted is guarded (the worktree readback await is the window); the apply mid-operation refusal covers cherry-pick, revert and bisect beside merge/rebase; a send queued on the engine slot publishes the generating state so Stop is visible for the whole 120 s wait (guarded so it never repoints another session's state) and the deadline preempt announces itself like the immediate one; `snapshot` registers intent-to-add (`git add -N .`) instead of staging, so a worktree's files stop blobbing into the operator's shared object DB — verified byte-identical appliable patch, zero new blobs — and the non-UTF-8 apply refusal is documented in §7. Tests: turn-scoped cancel (the path Stop takes), deleteSession re-scope, under-budget compaction, mid-operation refusals (merge/cherry-pick/revert/bisect with a settled control), object-store no-bloat (`cat-file -e` on the content hash), and the single-flight refusal via a DEBUG seam (the in-flight window is one git call and cannot be raced deterministically). 9 suites, 137 executed, 0 failures. Mid-apply task cancellation stays covered by the two rollback cases rather than a racy interleaving. |
| 2026-11-17 | PR #104 review findings | Addressed the reviewer's three blockers plus three should-fixes. (1) `complete` is now in every run's tool set: `AgentSpawnResolver.alwaysAvailableToolNames` appends it after the preset/explicit list is filtered (`resolveTools`), still withheld in plan mode, so `endsAgentRun` is reachable and the grammar/tool mask advertises it. The worker prompt was also mismatched to the backend: cloud runs were taught the LOCAL XML dialect while their provider receives native function declarations, so `SubagentManager` now picks `buildCloudWorkerSystemPrompt` for `.cloud` and the local prompt for everything else. (2) The report is derived rather than hardcoded: `AgentReportBuilder` turns the run's own transcript into findings (one per substantive claim, evidence naming the step and the tool results in front of it), open questions (question-marked lines), and a best-effort partial summary (the LONGEST assistant text, not the last preamble), so a budget-exhausted run keeps its finished analysis; `get_subagent_status` now relays findings and open questions, and `SubagentInstance.lastError` keeps the failure reason out of the summary banner. (3) Worktree isolation is enforced per call: `.spawnOnly` auto-approves `file_write`/`file_edit` only when the resolved (symlink-resolved) path is inside the worktree and `git_commit` only when it runs there, while `shell_run` — arbitrary code with its own `cwd` — always gets the per-change card. Should-fixes: the two legacy pipeline tests pass `backend: legacy` explicitly instead of reading `dynamoe_subagent_backend` from the operator's defaults; `AgentUsage.costIsUnpriced` records that a billed token count had no price, the spawn refuses a `max_cost_usd` ceiling that no recorded price could bind, and the loop fails closed if a ceiling meets unpriced usage; `TurnPriority` and the settings copy now say preemption CANCELS rather than yields/pauses. Tests: 6 resolver/report cases in `AgentSpawnPolicyTests`, 5 loop cases (findings, partial summary, two containment cases, shell card, unpriced ceiling), 3 cloud usage cases, 1 cloud-prompt case. Subagent suites 78 green (2 opt-in cloud skipped). |
| 2026-11-17 | drawer token count | The Tasks drawer's "tokens" stat summed each run's per-step prompt tokens, and every step of a model-backed run reports the FULL prompt it was sent (correct for billing), so the operator's real 40-step review run displayed "1.4M tokens" against ~34k of context — contradicting the chat header's own written doctrine ("the prompt the model last ingested plus the tokens it generated... not a running total of what each step billed"). `AgentUsage` gained `latestPromptTokens` (set per step by the loop, never summed by `merge`) and `contextTokens` (last request + generated, falling back to the recorded sum for records that predate the field, the only number they honestly hold); the strip and every `compactDescription` site show `contextTokens`, with a help tooltip naming the rule; spend is untouched (it prices the billed sum, as it should). The decoder for `AgentUsage` is defaults-tolerant now (`decodeIfPresent` for every field) because the store's decode-failure path is "record lost" — a stored usage missing the new key must never cost the whole run; `init()` is explicit because `init(from:)` displaces the memberwise one. Tests: the loop's context-vs-billed arithmetic (billed 220 stays, context 132), a store case that rewrites a record's usage JSON without the new key and decodes it (legacy fallback 232), and a projection case pinning both semantics plus a coding round trip. 11 suites, 126 executed, 0 failures. Legacy runs keep their old number until cleared. |
| 2026-11-17 | review fixes | Addressed the build-verifier review of the PR: (1) `SubagentInstance.waitForCompletion` held `NSLock` across an await — the finished-check and the waiter registration are now one decision in a synchronous `registerCompletionWaiter` helper called from the continuation body, so a run finishing between the two cannot strand the waiter and the lock is never touched from an async context; (2) the headless-Chrome timeout path's captured `isResumed` var (a data race, error-grade in Swift 6) is a lock-guarded `ResumeOnce` claim; (3) the `runProcess` local `ProcessState` inherited default MainActor isolation, making every member mutation from the pipe readers/timer/cancel handler a Swift 6 error — it is `nonisolated` now (this also covered the `isCancelled` warning at the old 6990 the verifier missed); (4) an optional-interpolation in the test-runner report named its command `Optional(...)` — the resolved command is interpolated; (5) the subagent Cloud model setting silently fell back to the provider default when the stored id did not resolve — `SubagentBackendSettings.preferredCloudModel` is now a three-way `PreferredCloudModel` (configured / unresolvable(stored) / noneConfigured) resolved at PARSE level (keychain-free, `providers:` injectable), a spawn refuses an unresolvable stored value BY NAME instead of billing a provider the operator never chose (same rule as `CloudModelAddress.selectionIsUnresolved`), and Settings flags the value where it is typed; SUBAGENTS.md §16 documents the refusal rule. Tests: 4 new pure cases pin the three-way resolution and the spawn refusal; both opt-in live-cloud cases ran and passed (turn: 50/24 tokens, "The capital of France is Paris."; manager run: 668/136 — first cut failed at its own 96-token step cap because the tiny model rambled past it and `finishReason` is advisory to every loop in this app, a pre-existing design; the test's step cap is 256 now), markers removed after. 11 suites, 123 executed, 0 failures, the 2 live gates skipping as designed. |
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
  (`dynamoe_subagent_cloud_model`, an upstream model id). The stored value is
  the ONLY resolution: `AgentSpawn.resolveBackend` passes
  `coordinateModelId: nil`, so there is no coordinator-model or
  first-discovered fallback. An empty value refuses a cloud spawn by name
  (`noneConfigured`), and a stored value that does not name a configured cloud
  model is also refused by name (and flagged in Settings where it is typed),
  rather than silently substituted — the substituted run would be billed on a
  provider the operator never chose. `.auto` still prefers a loaded local
  checkpoint. Still open: whether a real default chain (cheapest or
  coordinator's tool-capable model) should exist; it needs a price source the
  registry does not carry and an explicit product decision.
- Cooperative cancellation boundaries: `runTurn` and `runTool` use
  `withTaskGroup`, which always awaits its children after `cancelAll()`. A
  backend stream or tool that ignores cancellation therefore keeps the run,
  and its scheduler ticket, alive past Stop and the wall clock. The model
  backends are cancellation-aware, `shell_run` spawns through
  `AgentHarness.runProcess` (SIGTERM to the process tree, SIGKILL after
  500 ms), and `git` children run under the same helper. A tool that holds a
  child process of its own must use that helper; a hard-abandonment path is
  not built.
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
  call would and delegates to `SubagentApplyService`. Also shipped: the
  inspector's Discard Run control (`SubagentManager.discard`), which removes an
  isolated run's worktree under the same single-flight/apply guards and then
  deletes its record, so the list cannot offer a run whose tree is gone.
