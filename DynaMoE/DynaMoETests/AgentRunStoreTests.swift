//
//  AgentRunStoreTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for the durable subagent run store (SUBAGENTS.md §10).
//

import XCTest
@testable import DynaMoE

final class AgentRunStoreTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dynamoe-agent-runs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root {
            try? FileManager.default.removeItem(at: root)
        }
    }

    private func makeStore() -> AgentRunStore {
        AgentRunStore(rootURL: root)
    }

    private func record(
        id: UUID = UUID(),
        sessionId: UUID?,
        createdAt: Date = Date(),
        completedAt: Date? = nil,
        status: AgentRunStatus = .completed,
        transcript: [AgentTranscriptEntry] = [],
        isolation: AgentIsolation = .none,
        appliedAt: Date? = nil,
        report: AgentReport? = nil,
        error: String? = nil,
        summaryRelayed: Bool = false
    ) -> AgentRunRecord {
        AgentRunRecord(
            id: id,
            parentSessionId: sessionId,
            role: "Codebase Researcher",
            preset: .explore,
            task: "Find the store tests.",
            backend: AgentBackendDescriptor(
                kind: .local,
                providerId: nil,
                modelId: "local",
                displayName: "Local model"
            ),
            workingDirectory: "/tmp",
            isolation: isolation,
            allowedTools: ["file_read"],
            status: status,
            createdAt: createdAt,
            startedAt: createdAt,
            completedAt: completedAt,
            transcript: transcript,
            report: report,
            error: error,
            summaryRelayed: summaryRelayed,
            appliedAt: appliedAt
        )
    }

    private func report(
        status: AgentRunStatus = .cancelled,
        summary: String
    ) -> AgentReport {
        AgentReport(
            status: status,
            summary: summary,
            findings: [],
            changedFiles: [],
            diff: nil,
            tests: [],
            risks: [],
            openQuestions: [],
            usage: AgentUsage(),
            backend: AgentBackendDescriptor(
                kind: .local,
                providerId: nil,
                modelId: "local",
                displayName: "Local model"
            )
        )
    }

    func testAnInterruptedRunIsClosedAtLaunchAndAFreshOneIsLeftAlone() throws {
        let store = makeStore()
        let session = UUID()
        let staleId = UUID()
        let freshId = UUID()
        let now = Date()
        store.saveImmediately(record(
            id: staleId,
            sessionId: session,
            createdAt: now.addingTimeInterval(-3_600),
            status: .running
        ))
        store.saveImmediately(record(
            id: freshId,
            sessionId: session,
            createdAt: now,
            status: .running
        ))

        let closed = store.reconcileInterruptedRuns(now: now, graceSeconds: 60)

        XCTAssertEqual(closed, [staleId])
        let stale = try XCTUnwrap(store.load(runId: staleId, sessionId: session))
        XCTAssertEqual(stale.status, .failed)
        XCTAssertNotNil(stale.error, "the closed record names what happened")
        XCTAssertEqual(
            try XCTUnwrap(store.load(runId: freshId, sessionId: session)).status,
            .running,
            "a run young enough to be in flight must not be closed"
        )
    }

    func testARunRoundTripsThroughDisk() throws {
        let store = makeStore()
        let sessionId = UUID()
        let saved = record(sessionId: sessionId)
        store.saveImmediately(saved)

        let loaded = store.load(sessionId: sessionId)
        XCTAssertEqual(loaded.count, 1)
        let restored = try XCTUnwrap(loaded.first)
        XCTAssertEqual(restored.id, saved.id)
        XCTAssertEqual(restored.role, saved.role)
        XCTAssertEqual(restored.backend, saved.backend)
        XCTAssertEqual(restored.status, .completed)
        XCTAssertEqual(restored.schemaVersion, AgentRunRecord.currentSchemaVersion)
        XCTAssertEqual(restored.parentSessionId, sessionId)
    }

    func testAnUnscopedRunIsStillDurable() throws {
        let store = makeStore()
        let saved = record(sessionId: nil)
        store.saveImmediately(saved)
        XCTAssertEqual(store.load(sessionId: nil).first?.id, saved.id)
        XCTAssertEqual(store.loadAll().first?.id, saved.id)
    }

    func testARecordFromANewerSchemaIsIgnored() throws {
        let store = makeStore()
        let sessionId = UUID()
        let directory = store.directory(for: sessionId)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        var future = record(sessionId: sessionId)
        future.schemaVersion = AgentRunRecord.currentSchemaVersion + 1
        let data = try JSONEncoder().encode(future)
        try data.write(to: store.fileURL(runId: future.id, sessionId: sessionId))

        XCTAssertTrue(store.load(sessionId: sessionId).isEmpty)
    }

    /// A stored usage that predates the context field must still decode: the
    /// store's decode-failure path is "the record is lost", so a missing field
    /// in one nested struct may never cost the operator the whole run.
    func testAUsageThatPredatesTheContextFieldStillDecodes() throws {
        let store = makeStore()
        let sessionId = UUID()
        let directory = store.directory(for: sessionId)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        var saved = record(sessionId: sessionId)
        var usage = AgentUsage()
        usage.steps = 2
        usage.promptTokens = 220
        usage.latestPromptTokens = 120
        usage.completionTokens = 12
        saved.usage = usage
        try JSONEncoder().encode(saved).write(to: store.fileURL(runId: saved.id, sessionId: sessionId))

        // Rewrite the stored JSON the way an older build wrote it: the usage
        // block carries no `latestPromptTokens` key at all.
        let url = store.fileURL(runId: saved.id, sessionId: sessionId)
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        var legacy = try XCTUnwrap(object)
        var legacyUsage = try XCTUnwrap(legacy["usage"] as? [String: Any])
        legacyUsage.removeValue(forKey: "latestPromptTokens")
        legacy["usage"] = legacyUsage
        try JSONSerialization.data(withJSONObject: legacy).write(to: url)

        let restored = try XCTUnwrap(store.load(sessionId: sessionId).first)
        XCTAssertEqual(restored.usage.steps, 2)
        XCTAssertEqual(restored.usage.promptTokens, 220)
        XCTAssertEqual(restored.usage.completionTokens, 12)
        XCTAssertEqual(restored.usage.latestPromptTokens, 0)
        // The legacy fallback: with no last-request size stored, the recorded
        // sum is the only number the record can honestly show.
        XCTAssertEqual(restored.usage.contextTokens, 232)
    }

    /// The context chip is "the prompt the model last ingested plus the
    /// tokens it generated". The last prompt already contains every earlier
    /// step's generation, so adding the SUMMED `completionTokens` counted
    /// them twice.
    func testContextTokensCountsOnlyTheLastStepsGeneration() {
        var usage = AgentUsage()
        usage.steps = 3
        usage.promptTokens = 300
        usage.completionTokens = 30
        usage.latestPromptTokens = 220
        usage.latestCompletionTokens = 10
        XCTAssertEqual(usage.contextTokens, 230)
        XCTAssertEqual(usage.totalTokens, 330, "billing still sees every step's sum")
    }

    func testTranscriptToolOutputIsTruncatedBeforeWriting() throws {
        let store = makeStore()
        let sessionId = UUID()
        let huge = String(repeating: "x", count: AgentRunStore.TranscriptBudget.toolOutput + 500)
        let entry = AgentTranscriptEntry(
            index: 0,
            stepIndex: 0,
            kind: .toolResult,
            text: huge,
            toolRecord: ToolCallRecord(name: "file_read", status: .success, output: huge)
        )
        store.saveImmediately(record(sessionId: sessionId, transcript: [entry]))

        let restored = try XCTUnwrap(store.load(sessionId: sessionId).first)
        let text = try XCTUnwrap(restored.transcript.first?.text)
        XCTAssertLessThan(text.count, huge.count)
        XCTAssertTrue(text.contains("truncated"))
        let output = try XCTUnwrap(restored.transcript.first?.toolRecord?.output)
        XCTAssertTrue(output.contains("truncated"))
    }

    func testRetentionKeepsTheNewestRunsAndNeverPrunesAFreshOne() throws {
        let store = makeStore()
        let sessionId = UUID()
        let now = Date()
        let old = now.addingTimeInterval(-45 * 86_400)
        // Three old runs, three young ones. The policy keeps two newest plus
        // everything inside the 30-day window.
        for index in 0..<3 {
            let r = record(
                sessionId: sessionId,
                createdAt: old.addingTimeInterval(Double(index)),
                completedAt: old.addingTimeInterval(Double(index))
            )
            store.saveImmediately(r)
        }
        for index in 0..<3 {
            let r = record(
                sessionId: sessionId,
                createdAt: now.addingTimeInterval(-Double(index)),
                completedAt: now.addingTimeInterval(-Double(index))
            )
            store.saveImmediately(r)
        }

        let removed = store.prune(
            now: now,
            policy: .init(keepNewest: 2, keepDays: 30, minimumAgeHours: 24)
        )
        XCTAssertEqual(removed.count, 3)
        let remaining = store.load(sessionId: sessionId)
        XCTAssertEqual(remaining.count, 3)
        XCTAssertTrue(remaining.allSatisfy { ($0.completedAt ?? .distantPast) > old.addingTimeInterval(2) })
    }

    func testTheFreshnessGuardKeepsARunInsideItsMinimumAge() throws {
        let store = makeStore()
        let sessionId = UUID()
        let now = Date()
        // A zero-day window would prune both on age alone; the 24-hour floor is
        // the only reason the hour-old run survives.
        let insideFloor = now.addingTimeInterval(-1 * 3_600)
        let ancient = now.addingTimeInterval(-90 * 86_400)
        for date in [ancient, insideFloor] {
            store.saveImmediately(record(sessionId: sessionId, createdAt: date, completedAt: date))
        }
        let removed = store.prune(
            now: now,
            policy: .init(keepNewest: 0, keepDays: 0, minimumAgeHours: 24)
        )
        XCTAssertEqual(removed.count, 1)
        XCTAssertEqual(store.load(sessionId: sessionId).count, 1)
    }

    /// The clamp is the only thing standing between a hand-edited defaults
    /// value and "keep forever", which is the one direction the bound exists to
    /// prevent. Nothing pinned it, so a change from `min(max(…))` to a plain
    /// read would have compiled and shipped.
    func testConfiguredRetentionClampsToItsBounds() throws {
        let suiteName = "dynamoe-retention-\(UUID().uuidString)"
        let suite = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { suite.removePersistentDomain(forName: suiteName) }

        let standard = AgentRunStore.RetentionPolicy.standard
        let policy = AgentRunStore.configuredPolicy(defaults: suite)
        XCTAssertEqual(policy.keepDays, standard.keepDays)
        XCTAssertEqual(policy.keepNewest, standard.keepNewest)

        suite.set(10_000.0, forKey: "dynamoe_subagent_retention_days")
        suite.set(1_000_000, forKey: "dynamoe_subagent_retention_runs")
        let high = AgentRunStore.configuredPolicy(defaults: suite)
        XCTAssertEqual(high.keepDays, 365, "a huge day window is capped instead of meaning 'forever'")
        XCTAssertEqual(high.keepNewest, 2_000)

        suite.set(0.01, forKey: "dynamoe_subagent_retention_days")
        suite.set(1, forKey: "dynamoe_subagent_retention_runs")
        let low = AgentRunStore.configuredPolicy(defaults: suite)
        XCTAssertEqual(low.keepDays, 1, "a sub-day window is raised to one day")
        XCTAssertEqual(low.keepNewest, 20, "a tiny run count is raised to the floor")

        // A non-positive value is not a policy at all: it keeps the default.
        suite.set(0.0, forKey: "dynamoe_subagent_retention_days")
        suite.set(-5, forKey: "dynamoe_subagent_retention_runs")
        let zero = AgentRunStore.configuredPolicy(defaults: suite)
        XCTAssertEqual(zero.keepDays, standard.keepDays)
        XCTAssertEqual(zero.keepNewest, standard.keepNewest)
    }

    func testRetentionNeverDeletesAnUnappliedIsolatedRunsRecord() throws {
        let store = makeStore()
        let sessionId = UUID()
        let now = Date()
        let ancient = now.addingTimeInterval(-90 * 86_400)
        // An unapplied isolated run's record is the only thing keeping the
        // launch sweep from treating its worktree as orphaned: prune the record
        // and the very next sweep deletes the only full copy of the work.
        // Completed AND cancelled unapplied writers are both protected; an
        // applied one prunes like any other terminal record.
        let unapplied = record(
            sessionId: sessionId,
            createdAt: ancient,
            completedAt: ancient,
            isolation: .worktree(path: "/tmp/w/x", branch: "dynamoe/run-x"),
            appliedAt: nil
        )
        let cancelledWriter = record(
            sessionId: sessionId,
            createdAt: ancient,
            completedAt: ancient,
            status: .cancelled,
            isolation: .worktree(path: "/tmp/w/y", branch: "dynamoe/run-y"),
            appliedAt: nil
        )
        let applied = record(
            sessionId: sessionId,
            createdAt: ancient,
            completedAt: ancient,
            isolation: .worktree(path: "/tmp/w/z", branch: "dynamoe/run-z"),
            appliedAt: ancient
        )
        store.saveImmediately(unapplied)
        store.saveImmediately(cancelledWriter)
        store.saveImmediately(applied)

        let removed = store.prune(
            now: now,
            policy: .init(keepNewest: 0, keepDays: 0, minimumAgeHours: 24)
        )

        XCTAssertEqual(removed, [applied.id])
        XCTAssertEqual(store.loadAny(runId: unapplied.id)?.id, unapplied.id)
        XCTAssertEqual(store.loadAny(runId: cancelledWriter.id)?.id, cancelledWriter.id)
        XCTAssertNil(store.loadAny(runId: applied.id))
    }

    func testARunCanBeFoundWithoutKnowingItsSession() throws {
        let store = makeStore()
        let scoped = record(sessionId: UUID())
        let unscoped = record(sessionId: nil)
        store.saveImmediately(scoped)
        store.saveImmediately(unscoped)

        XCTAssertEqual(store.loadAny(runId: scoped.id)?.id, scoped.id)
        XCTAssertEqual(store.loadAny(runId: unscoped.id)?.id, unscoped.id)
        XCTAssertNil(store.loadAny(runId: UUID()))
    }

    func testDeletingASessionRemovesItsRuns() throws {
        let store = makeStore()
        let sessionId = UUID()
        store.saveImmediately(record(sessionId: sessionId))
        store.flush()
        XCTAssertEqual(store.load(sessionId: sessionId).count, 1)

        store.deleteSession(sessionId)
        store.flush()
        XCTAssertTrue(store.load(sessionId: sessionId).isEmpty)
        XCTAssertTrue(store.loadAll().isEmpty)
    }

    /// A deleted conversation's unapplied isolated work is NOT disposed with
    /// it: the record is the only thing keeping that run's worktree out of the
    /// launch sweep's reach, so it is re-scoped to the unscoped bucket —
    /// visible in every inspector, skipped by `prune` — rather than deleted.
    /// A still-working survivor is closed as cancelled instead of left
    /// claiming to work for a conversation that no longer exists, and the
    /// relay never sees its report as undelivered.
    func testDeletingASessionKeepsItsUnappliedIsolatedRunsReachable() throws {
        let store = makeStore()
        let session = UUID()
        let unapplied = record(
            sessionId: session,
            isolation: .worktree(path: "/tmp/w/x", branch: "dynamoe/run-x"),
            appliedAt: nil
        )
        let working = record(
            sessionId: session,
            status: .running,
            isolation: .worktree(path: "/tmp/w/y", branch: "dynamoe/run-y"),
            appliedAt: nil
        )
        let applied = record(
            sessionId: session,
            isolation: .worktree(path: "/tmp/w/z", branch: "dynamoe/run-z"),
            appliedAt: Date()
        )
        let plain = record(sessionId: session)
        store.saveImmediately(unapplied)
        store.saveImmediately(working)
        store.saveImmediately(applied)
        store.saveImmediately(plain)

        store.deleteSession(session)
        store.flush()

        XCTAssertTrue(store.load(sessionId: session).isEmpty, "the session bucket is gone")
        XCTAssertNil(store.loadAny(runId: applied.id), "an applied isolated run goes with its conversation")
        XCTAssertNil(store.loadAny(runId: plain.id), "a run with nothing to protect goes with its conversation")

        let keptUnapplied = try XCTUnwrap(store.loadAny(runId: unapplied.id))
        XCTAssertNil(keptUnapplied.parentSessionId, "the survivor is re-scoped to the unscoped bucket")
        XCTAssertTrue(keptUnapplied.isolation.isIsolated)
        XCTAssertNil(keptUnapplied.appliedAt)
        XCTAssertEqual(keptUnapplied.status, .completed, "a terminal survivor keeps its own status")
        XCTAssertTrue(keptUnapplied.summaryRelayed, "the relay never injects a deleted conversation's report elsewhere")

        let keptWorking = try XCTUnwrap(store.loadAny(runId: working.id))
        XCTAssertEqual(keptWorking.status, .cancelled, "a still-working survivor is closed, not left claiming to run")
        XCTAssertNotNil(keptWorking.error, "the closed survivor names what happened to it")

        XCTAssertEqual(
            Set(store.load(sessionId: nil).map(\.id)),
            [unapplied.id, working.id],
            "exactly the unapplied isolated survivors ride the unscoped bucket"
        )
    }

    /// A record this build cannot decode is the ONLY thing that keeps a
    /// UUID-named worktree out of the launch sweep (`pruneStaleWorktrees`
    /// probes `hasRecordFile`). Deleting a conversation used to remove the
    /// whole directory, so a schema downgrade plus a conversation delete handed
    /// that tree to the 14-day clock. The undecodable file moves to the
    /// unscoped bucket instead.
    func testDeletingASessionKeepsARecordThisBuildCannotDecode() throws {
        let store = makeStore()
        let session = UUID()
        let runId = UUID()
        let directory = store.directory(for: session)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let undecodable = directory.appendingPathComponent("\(runId.uuidString).json")
        // Not a valid record in this schema, but a real file a newer build may
        // have written. Its CONTENT is never read by the sweep.
        try Data(#"{"schemaVersion":9999,"id":"\#(runId.uuidString)"}"#.utf8).write(to: undecodable)

        store.deleteSession(session)
        store.flush()

        XCTAssertFalse(FileManager.default.fileExists(atPath: undecodable.path))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: store.directory(for: nil).appendingPathComponent("\(runId.uuidString).json").path
            ),
            "an undecodable record must survive the conversation delete"
        )
        XCTAssertTrue(store.hasRecordFile(runId: runId))
    }

    /// `deleteClearableRecords` is the bulk Clear button's store half. It used
    /// to have no direct coverage, so a predicate regression could delete a
    /// protected record through the UI path alone.
    func testDeleteClearableRecordsDeletesOnlyWhatThePredicateAccepts() throws {
        let store = makeStore()
        let session = UUID()
        let keep = record(sessionId: session)
        let drop = record(sessionId: session)
        store.saveImmediately(keep)
        store.saveImmediately(drop)
        store.flush()

        store.deleteClearableRecords(sessionId: session) { $0.id == drop.id }
        store.flush()

        XCTAssertNotNil(store.loadAny(runId: keep.id))
        XCTAssertNil(store.loadAny(runId: drop.id))
    }

    /// An apply that completes AFTER its conversation was deleted must land on
    /// the re-scoped record. The apply path still holds the old session id, and
    /// the store's tombstone refused a write to the old path — so the worktree
    /// was cleaned up while the surviving record still said unapplied, and
    /// `prune` then skipped that record forever.
    func testAMutateAfterASessionDeleteLandsOnTheRescopedRecord() throws {
        let store = makeStore()
        let session = UUID()
        let runId = UUID()
        let unapplied = record(
            id: runId,
            sessionId: session,
            isolation: .worktree(path: "/tmp/w/x", branch: "dynamoe/run-x"),
            appliedAt: nil
        )
        store.saveImmediately(unapplied)
        store.deleteSession(session)
        store.flush()

        let appliedAt = Date()
        let returned = store.mutate(runId: runId, sessionId: session) { current in
            current.appliedAt = appliedAt
            current.appliedToBranch = "main"
        }

        XCTAssertNotNil(returned, "the re-scoped record must still be mutable by id")
        let reloaded = try XCTUnwrap(store.loadAny(runId: runId))
        XCTAssertEqual(reloaded.appliedAt, appliedAt)
        XCTAssertEqual(reloaded.appliedToBranch, "main")
        XCTAssertNil(reloaded.parentSessionId)
    }

    /// A mutation that cannot be written reports nil instead of the record it
    /// read. The apply path consumes this answer; a false success cleaned up
    /// worktrees for records that still said unapplied.
    func testAMutateReportsARefusedWriteAsNil() throws {
        let store = makeStore()
        let session = UUID()
        let saved = record(sessionId: session)
        store.saveImmediately(saved)
        store.delete(runId: saved.id, sessionId: session)
        store.flush()

        XCTAssertNil(
            store.mutate(runId: saved.id, sessionId: session) { current in
                current.status = .completed
            },
            "a deleted run is not a successful mutation"
        )
    }

    func testDeletingOneRunLeavesItsSessionAlone() throws {
        let store = makeStore()
        let sessionId = UUID()
        let keep = record(sessionId: sessionId)
        let drop = record(sessionId: sessionId)
        store.saveImmediately(keep)
        store.saveImmediately(drop)

        store.delete(runId: drop.id, sessionId: sessionId)
        store.flush()
        let remaining = store.load(sessionId: sessionId)
        XCTAssertEqual(remaining.map(\.id), [keep.id])
    }

    /// A store is a record of work, not a second line of defence: task,
    /// conversation excerpt, thinking and tool output all live in it, and an
    /// Application Support write is world-readable by default.
    func testARecordAndItsDirectoryAreOwnerOnly() throws {
        let store = makeStore()
        let sessionId = UUID()
        let run = record(sessionId: sessionId)
        store.saveImmediately(run)

        let fileURL = store.fileURL(runId: run.id, sessionId: sessionId)
        let fileAttributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        XCTAssertEqual(fileAttributes[.posixPermissions] as? NSNumber, 0o600)
        let directoryAttributes = try FileManager.default.attributesOfItem(
            atPath: store.directory(for: sessionId).path
        )
        XCTAssertEqual(directoryAttributes[.posixPermissions] as? NSNumber, 0o700)
    }

    /// `file_write` stores its body in `arguments["content"]` and again in
    /// `rawArguments`. Both are bounded now, along with the fields that had no
    /// limit at all.
    func testBoundingCoversArgumentsContextAndMessages() {
        let huge = String(repeating: "x", count: 50_000)
        var record = record(sessionId: UUID())
        record.contextSummary = huge
        record.messages = [AgentMessageRecord(sender: "user", content: huge)]
        record.transcript = [AgentTranscriptEntry(
            index: 0,
            stepIndex: 0,
            kind: .toolCall,
            toolRecord: ToolCallRecord(
                name: "file_write",
                arguments: ["path": "Sources/File.swift", "content": huge],
                rawArguments: huge
            )
        )]

        let bounded = AgentRunStore.bounded(record)
        let call = bounded.transcript[0].toolRecord
        XCTAssertEqual(call?.arguments["path"], "Sources/File.swift")
        XCTAssertLessThan(call?.arguments["content"]?.count ?? .max, 2_200)
        XCTAssertLessThan(call?.rawArguments.count ?? .max, 8_500)
        XCTAssertLessThan(bounded.contextSummary?.count ?? .max, 16_700)
        XCTAssertLessThan(bounded.messages[0].content.count, 8_500)
    }

    /// A save captured before its delete used to be queued behind it and write
    /// the file back. The tombstone is what makes a delete final.
    func testAQueuedSaveCannotResurrectADeletedRecord() throws {
        let store = makeStore()
        let sessionId = UUID()
        let run = record(sessionId: sessionId)
        store.saveImmediately(run)
        store.flush()

        store.delete(runId: run.id, sessionId: sessionId)
        store.save(run)
        store.flush()

        XCTAssertNil(
            store.load(runId: run.id, sessionId: sessionId),
            "a queued save must not recreate a deleted record"
        )
    }

    /// A delete for a run that left no file still has to be final: the file
    /// may be absent because the record was already removed, while a save
    /// captured before the delete is still queued behind it.
    func testADeleteWithNoFileStillStopsAQueuedSave() throws {
        let store = makeStore()
        let sessionId = UUID()
        let run = record(sessionId: sessionId)

        store.delete(runId: run.id, sessionId: sessionId)
        store.save(run)
        store.flush()

        XCTAssertNil(
            store.load(runId: run.id, sessionId: sessionId),
            "an absent file is not permission for a queued save to create it"
        )
    }

    /// Retention's clock starts at a run's start, so an old record can still be
    /// live. Pruning it deletes the file out from under its own instance.
    func testRetentionNeverPrunesALiveRecord() throws {
        let store = makeStore()
        let sessionId = UUID()
        let now = Date()
        let ancient = now.addingTimeInterval(-90 * 86_400)
        let live = record(
            sessionId: sessionId,
            createdAt: ancient,
            completedAt: nil,
            status: .running
        )
        store.saveImmediately(live)

        let removed = store.prune(
            now: now,
            policy: .init(keepNewest: 0, keepDays: 0, minimumAgeHours: 0)
        )
        XCTAssertTrue(removed.isEmpty)
        XCTAssertNotNil(store.load(runId: live.id, sessionId: sessionId))
    }

    /// The report relay asks this index instead of decoding a conversation's
    /// whole run history on every send, so it has to be complete after a
    /// relaunch and has to follow a record that is delivered or deleted.
    func testTheRelayIndexFindsUnrelayedRecordsOnDiskAndFollowsChanges() throws {
        let store = makeStore()
        let sessionId = UUID()
        // A terminal record still needs something to DELIVER to be owed; the
        // index asks `relaySummaryText`, so a bare completed status is settled.
        let pending = record(
            sessionId: sessionId,
            report: report(status: .completed, summary: "Found the scheduler.")
        )
        var delivered = record(
            sessionId: sessionId,
            report: report(status: .completed, summary: "Already shown.")
        )
        delivered.summaryRelayed = true
        store.saveImmediately(pending)
        store.saveImmediately(delivered)

        // A fresh store answers from disk, which is the relaunch case. The
        // launch prewarm is what builds that index off the main thread; a main
        // -thread lookup before it finishes answers from what is built and the
        // next send picks the record up, so the test prewarms explicitly.
        let restored = AgentRunStore(rootURL: root)
        restored.prepareRelayIndex()
        restored.flush()
        XCTAssertEqual(restored.unrelayedRunIds(sessionId: sessionId), [pending.id])

        var marked = pending
        marked.summaryRelayed = true
        store.saveImmediately(marked)
        XCTAssertFalse(store.unrelayedRunIds(sessionId: sessionId).contains(pending.id))

        store.delete(runId: delivered.id, sessionId: sessionId)
        store.flush()
        XCTAssertFalse(store.unrelayedRunIds(sessionId: sessionId).contains(delivered.id))
    }

    /// `markRelayed` must update the index BEFORE its durable write lands.
    /// The relay asks the fast path on the next send, and a write still behind
    /// the IO queue left the run listed as owed — its report injected twice.
    func testMarkingRelayedIsVisibleBeforeItsWriteLands() throws {
        let store = makeStore()
        let sessionId = UUID()
        let run = record(
            sessionId: sessionId,
            report: report(status: .completed, summary: "Found the scheduler.")
        )
        store.saveImmediately(run)
        // Build the index so the answer comes from the in-memory fast path.
        store.prepareRelayIndex()
        XCTAssertTrue(store.unrelayedRunIds(sessionId: sessionId).contains(run.id))

        store.markRelayed(runId: run.id, sessionId: sessionId)
        XCTAssertFalse(
            store.unrelayedRunIds(sessionId: sessionId).contains(run.id),
            "the relay index must learn a delivery before the queued write drains"
        )
        store.flush()
        XCTAssertFalse(store.unrelayedRunIds(sessionId: sessionId).contains(run.id))
    }

    // MARK: - Mutate

    /// `mutate` is the one sanctioned read-modify-write: a caller that loaded
    /// a record seconds ago must not be able to revert fields other writers
    /// changed in between (the apply path's own comment names the defect).
    func testMutateAppliesAgainstTheCurrentRecordAndReturnsIt() throws {
        let store = makeStore()
        let sessionId = UUID()
        let saved = record(id: UUID(), sessionId: sessionId, status: .running)
        store.saveImmediately(saved)

        let returned = store.mutate(runId: saved.id, sessionId: sessionId) { current in
            current.status = .completed
            current.completedAt = Date()
        }
        XCTAssertEqual(try XCTUnwrap(returned).status, .completed)
        XCTAssertEqual(
            try XCTUnwrap(store.load(runId: saved.id, sessionId: sessionId)).status,
            .completed,
            "the mutation lands against the record ON DISK, not a caller's stale copy"
        )
        XCTAssertNil(store.mutate(runId: UUID(), sessionId: sessionId) { _ in },
                     "a record that does not decode returns nil and writes nothing")
    }

    /// Two mutates through different callers keep EACH OTHER's fields: this is
    /// the property `markSubagentSummariesRelayed` relies on — an apply
    /// completing mid-relay must not be reverted by the relay's own write, and
    /// the relay's mark must not be erased by a later apply.
    func testAMutateDoesNotRevertFieldsAnotherWriterChanged() throws {
        let store = makeStore()
        let sessionId = UUID()
        let saved = record(sessionId: sessionId)
        store.saveImmediately(saved)

        store.mutate(runId: saved.id, sessionId: sessionId) { current in
            current.summaryRelayed = true
        }
        let appliedAt = Date()
        store.mutate(runId: saved.id, sessionId: sessionId) { current in
            current.appliedAt = appliedAt
            current.appliedToBranch = "main"
        }

        let reloaded = try XCTUnwrap(store.load(runId: saved.id, sessionId: sessionId))
        XCTAssertTrue(reloaded.summaryRelayed, "the relay mark survives the apply's write")
        XCTAssertEqual(reloaded.appliedAt, appliedAt, "the apply's fields survive the relay mark")
        XCTAssertEqual(reloaded.appliedToBranch, "main")
    }

    /// A synchronous entry point reached from the store's OWN queue must run
    /// inline. `deleteSession` re-scopes its survivors from inside
    /// `ioQueue.async` and called `saveImmediately`, whose `ioQueue.sync`
    /// libdispatch-aborted the process ("dispatch_sync called on queue already
    /// owned by current thread") — the PR's own delete-session test died with
    /// it. This pins the mechanism directly, so a future caller on the queue
    /// fails a test instead of a run.
    func testASynchronousWriteFromInsideTheStoreQueueRunsInline() throws {
        let store = makeStore()
        let sessionId = UUID()
        let saved = record(sessionId: sessionId)
        store.saveImmediately(saved)

        var nestedWriteRan = false
        let returned = store.mutate(runId: saved.id, sessionId: sessionId) { current in
            // Already on `ioQueue` here.
            current.summaryRelayed = true
            store.saveImmediately(current)
            nestedWriteRan = true
        }
        XCTAssertTrue(nestedWriteRan)
        XCTAssertNotNil(returned)
        XCTAssertTrue(
            try XCTUnwrap(store.load(runId: saved.id, sessionId: sessionId)).summaryRelayed,
            "the nested write landed against the same record"
        )
        // And the queue is still usable afterwards, i.e. the inline path did
        // not leave the store wedged.
        store.flush()
        XCTAssertEqual(store.load(sessionId: sessionId).map(\.id), [saved.id])
    }

    // MARK: - Retention vs in-flight saves

    /// Retention deletes then tombstones; a save that passes its
    /// `wasDeleted` check in flight must not be able to resurrect the file
    /// afterwards. The tombstone a prune sets is what blocks it.
    func testAPrunedRunCannotBeResurrectedByALaterSave() throws {
        let store = makeStore()
        let sessionId = UUID()
        let now = Date()
        var ancient = record(sessionId: sessionId, createdAt: now.addingTimeInterval(-400 * 86_400))
        ancient.completedAt = now.addingTimeInterval(-399 * 86_400)
        store.saveImmediately(ancient)

        let removed = store.prune(now: now, policy: .init(keepNewest: 0, keepDays: 30, minimumAgeHours: 24))
        XCTAssertEqual(removed, [ancient.id])

        // The save that raced the prune: whatever it observed, the record is
        // gone and stays gone.
        var resurrection = ancient
        resurrection.status = .running
        store.saveImmediately(resurrection)
        store.flush()
        XCTAssertNil(store.load(runId: ancient.id, sessionId: sessionId))
        let resurrectedAnywhere = store.loadAll().contains { $0.id == ancient.id }
        XCTAssertFalse(resurrectedAnywhere, "no bucket may hold a pruned record")
    }

    // MARK: - Orphaned temp writes

    func testSweepOrphanedTempWritesRemovesOldTempsAndKeepsFreshOnes() throws {
        let store = makeStore()
        let sessionId = UUID()
        let saved = record(sessionId: sessionId)
        store.saveImmediately(saved)
        let directory = root.appendingPathComponent(sessionId.uuidString, isDirectory: true)
        let oldTemp = directory.appendingPathComponent(".\(saved.id.uuidString).json.new-aaa", isDirectory: false)
        try "partial".write(to: oldTemp, atomically: true, encoding: .utf8)
        let freshTemp = directory.appendingPathComponent(".\(UUID().uuidString).json.new-bbb", isDirectory: false)
        try "partial".write(to: freshTemp, atomically: true, encoding: .utf8)
        // An old DOTLESS file is a record, not a temp: never touched.
        let age = Date().addingTimeInterval(-7_200)
        try FileManager.default.setAttributes([.modificationDate: age], ofItemAtPath: oldTemp.path)
        try FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: freshTemp.path)

        let removed = store.sweepOrphanedTempWrites(now: Date(), olderThan: 3_600)

        XCTAssertEqual(removed, 1, "only the aged temp is removed")
        XCTAssertFalse(FileManager.default.fileExists(atPath: oldTemp.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: freshTemp.path), "an in-flight temp is never touched")
        XCTAssertNotNil(store.load(runId: saved.id, sessionId: sessionId), "records are never swept")
    }

    // MARK: - Durability under an injected fault

    /// Every failure point in the durable write either keeps the PREVIOUS
    /// record intact or leaves nothing behind; a retry after the fault clears
    /// must land, and the digest cache must not have recorded the failed bytes
    /// as already written.
    func testAWriteThatFailsAtEachStageKeepsThePriorRecordAndARetryLands() throws {
        for fault in [AgentRunStore.StoreWriteFault.write, .fsync, .rename] {
            let store = makeStore()
            let sessionId = UUID()
            var saved = record(sessionId: sessionId)
            saved.task = "the first version"
            XCTAssertTrue(store.saveImmediately(saved))

            var updated = saved
            updated.task = "the second version"
            store.injectWriteFaultForTesting { fault }
            XCTAssertFalse(store.saveImmediately(updated), "\(fault) must be reported as a failed write")
            XCTAssertEqual(
                store.load(runId: saved.id, sessionId: sessionId)?.task,
                "the first version",
                "\(fault) must not damage the record that was already durable"
            )
            // A stale temp must not be left behind either: it is what a later
            // sweep has to clean up and what a confused writer could reuse.
            let directory = root.appendingPathComponent(sessionId.uuidString, isDirectory: true)
            let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?
                .filter { $0.hasPrefix(".\(saved.id.uuidString).json.new-") } ?? []
            XCTAssertTrue(leftovers.isEmpty, "\(fault) left a temp file behind: \(leftovers)")

            store.injectWriteFaultForTesting(nil)
            XCTAssertTrue(store.saveImmediately(updated), "the retry must land once the fault is gone")
            XCTAssertEqual(
                store.load(runId: saved.id, sessionId: sessionId)?.task,
                "the second version",
                "the retry must actually write, not be skipped by the digest cache"
            )
            store.flush()
        }
    }

    /// The directory flush happens AFTER the rename, so its failure must not
    /// be reported as a failed write. Callers treat `false` as "nothing
    /// happened": a conversation-delete rescue kept files it had already
    /// moved, and an apply stamp claimed the record update failed after it had
    /// landed. The record is in place; only its crash durability is reduced.
    func testADirectoryFlushFailureAfterTheRenameIsStillASuccessfulWrite() throws {
        let store = makeStore()
        let sessionId = UUID()
        var saved = record(sessionId: sessionId)
        saved.task = "the first version"
        XCTAssertTrue(store.saveImmediately(saved))

        var updated = saved
        updated.task = "the second version"
        store.injectWriteFaultForTesting { .directoryFsync }
        XCTAssertTrue(
            store.saveImmediately(updated),
            "the bytes are already in place; a directory fsync failure is a warning, not a failed write"
        )
        XCTAssertEqual(
            store.load(runId: saved.id, sessionId: sessionId)?.task,
            "the second version",
            "and the record that landed is the new one, not the prior copy"
        )

        store.injectWriteFaultForTesting(nil)
        store.flush()
    }

    // MARK: - What the relay owes

    /// A cancelled run's `error` is the stop reason, not a report. Keeping it
    /// in the relay index made every later send pay a decode for it forever,
    /// and the inspector badge stayed UNREAD forever, because the relay's own
    /// cancelled rule refuses to inject a bare stop reason. Once terminal with
    /// no partial there is nothing left to wait for, so the record settles;
    /// a partial that lands later re-opens it.
    func testACancelledRunWithNothingToRelayIsNotOwed() throws {
        let store = makeStore()
        let sessionId = UUID()
        let cancelled = record(
            sessionId: sessionId,
            completedAt: Date(),
            status: .cancelled,
            error: "Cancelled by user or coordinator."
        )
        XCTAssertNil(cancelled.relaySummaryText, "the stop reason is not a report")
        XCTAssertTrue(store.saveImmediately(cancelled))
        store.flush()
        XCTAssertFalse(
            store.unrelayedRunIds(sessionId: sessionId).contains(cancelled.id),
            "a settled run must not be decoded on every send"
        )

        // The loop's unwind writes the partial it managed to produce.
        var withPartial = cancelled
        withPartial.report = report(summary: "Halfway through the analysis.")
        XCTAssertEqual(withPartial.relaySummaryText, "Halfway through the analysis.")
        XCTAssertTrue(store.saveImmediately(withPartial))
        store.flush()
        XCTAssertTrue(
            store.unrelayedRunIds(sessionId: sessionId).contains(cancelled.id),
            "the partial the run earned is owed once it lands"
        )

        // Delivered, it settles for good.
        var delivered = withPartial
        delivered.summaryRelayed = true
        XCTAssertNil(delivered.relaySummaryText)
        XCTAssertTrue(store.saveImmediately(delivered))
        store.flush()
        XCTAssertFalse(store.unrelayedRunIds(sessionId: sessionId).contains(cancelled.id))
    }

    /// Non-cancelled runs keep the old fallback: a failure with no report
    /// still has the failure text to show the coordinator.
    func testAFailedRunWithOnlyAnErrorIsStillOwed() {
        let failed = record(
            sessionId: UUID(),
            completedAt: Date(),
            status: .failed,
            error: "The provider refused the request."
        )
        XCTAssertEqual(failed.relaySummaryText, "The provider refused the request.")
    }

    // MARK: - Duplicate copies after a crashed conversation delete

    /// A conversation delete writes the rescued copy BEFORE removing the old
    /// one, so a crash in between leaves the same run in two buckets. Deleting
    /// one copy and tombstoning the id used to leave the other on disk: after a
    /// relaunch the tombstone was gone and the stale record resurfaced. A
    /// delete must remove the RECORD, not one file of it.
    func testDeletingARunRemovesEveryDuplicateCopy() throws {
        let store = makeStore()
        let sessionId = UUID()
        var saved = record(sessionId: sessionId)
        saved.isolation = .worktree(path: "/tmp/gone", branch: "dynamoe/agent/dup")
        saved.status = .cancelled
        XCTAssertTrue(store.saveImmediately(saved))

        // The crash window: the re-scoped copy exists in the unscoped bucket
        // while the original is still in its conversation's bucket.
        var rescued = saved
        rescued.parentSessionId = nil
        XCTAssertTrue(store.saveImmediately(rescued))
        XCTAssertNotNil(store.loadAny(runId: saved.id))

        store.deleteImmediately(runId: saved.id, sessionId: sessionId)

        XCTAssertNil(store.loadAny(runId: saved.id), "no bucket may hold the deleted run")
        XCTAssertNil(store.load(runId: saved.id, sessionId: sessionId))
        XCTAssertNil(store.load(runId: saved.id, sessionId: nil))
        XCTAssertFalse(
            store.hasRecordFile(runId: saved.id),
            "the launch sweep must not see a file for a deleted run"
        )
    }

    /// The rescue write is the one step of a conversation delete that can
    /// fail, and the failure branch exists because removing the original then
    /// would hand an unapplied worktree to the 14-day sweep. Inject the fault
    /// and prove the branch: both the record and its directory stay, no
    /// half-rescue appears in the unscoped bucket, and a retry after the fault
    /// clears completes the delete.
    func testAFailedRescueKeepsTheConversationsFilesAndARetryCompletes() throws {
        let store = makeStore()
        let sessionId = UUID()
        var saved = record(sessionId: sessionId)
        saved.isolation = .worktree(path: "/tmp/still-here", branch: "dynamoe/agent/rescue")
        saved.status = .cancelled
        saved.task = "the run that must not lose its tree"
        XCTAssertTrue(store.saveImmediately(saved))
        let directory = root.appendingPathComponent(sessionId.uuidString, isDirectory: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))

        store.injectWriteFaultForTesting { .write }
        store.deleteSession(sessionId)
        store.flush()

        XCTAssertNotNil(
            store.load(runId: saved.id, sessionId: sessionId),
            "a failed rescue must leave the only copy of the record where it was"
        )
        XCTAssertNil(
            store.load(runId: saved.id, sessionId: nil),
            "the failed rescue must not leave a half-written copy in the unscoped bucket"
        )
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: directory.path),
            "the conversation's directory stays so its worktree keeps a record"
        )

        store.injectWriteFaultForTesting(nil)
        store.deleteSession(sessionId)
        store.flush()

        // The retry RESCUES the worktree-holding record: it now lives in the
        // unscoped bucket, which is the whole point (the tree keeps a record
        // the sweep can see). The conversation's copy and directory are gone.
        XCTAssertNil(
            store.load(runId: saved.id, sessionId: sessionId),
            "the conversation's copy must be removed once the rescue lands"
        )
        XCTAssertEqual(
            store.load(runId: saved.id, sessionId: nil)?.task,
            "the run that must not lose its tree",
            "the rescued record carries the run whose worktree it protects"
        )
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: directory.path),
            "and the conversation's directory is removed"
        )
    }
}
