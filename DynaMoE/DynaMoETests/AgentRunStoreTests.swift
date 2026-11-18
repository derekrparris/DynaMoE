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
        transcript: [AgentTranscriptEntry] = []
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
            allowedTools: ["file_read"],
            status: status,
            createdAt: createdAt,
            startedAt: createdAt,
            completedAt: completedAt,
            transcript: transcript
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
        let pending = record(sessionId: sessionId)
        var delivered = record(sessionId: sessionId)
        delivered.summaryRelayed = true
        store.saveImmediately(pending)
        store.saveImmediately(delivered)

        // A fresh store answers from disk, which is the relaunch case.
        let restored = AgentRunStore(rootURL: root)
        XCTAssertEqual(restored.unrelayedRunIds(sessionId: sessionId), [pending.id])

        var marked = pending
        marked.summaryRelayed = true
        store.saveImmediately(marked)
        XCTAssertFalse(store.unrelayedRunIds(sessionId: sessionId).contains(pending.id))

        store.delete(runId: delivered.id, sessionId: sessionId)
        store.flush()
        XCTAssertFalse(store.unrelayedRunIds(sessionId: sessionId).contains(delivered.id))
    }
}
