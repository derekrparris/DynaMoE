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
}
