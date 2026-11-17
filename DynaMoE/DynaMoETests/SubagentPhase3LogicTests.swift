//
//  SubagentPhase3LogicTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for Phase 3: the transcript card's run link, the local
//  start-of-turn relay, and the live instance's record projection.
//

import XCTest
@testable import DynaMoE

final class SubagentRunLinkTests: XCTestCase {
    private let runId = UUID(uuidString: "8C1E2F44-0A3B-4C5D-8E6F-112233445566")!

    func testTheSpawnResultNamesItsRun() {
        let output = #"{"tool":"spawn_subagent","status":"success","data":{"subagent_id":"\#(runId.uuidString)","role":"Researcher"}}"#
        XCTAssertEqual(SubagentRunLink.runId(fromOutput: output), runId)
    }

    func testTheLinkSurvivesTheModelFacingRendering() {
        // The record stores the rendered form, where quotes are escaped inside
        // a wrapper element.
        let output = "<tool_response>{\\\"subagent_id\\\": \\\"\(runId.uuidString)\\\"}</tool_response>"
        XCTAssertEqual(SubagentRunLink.runId(fromOutput: output), runId)
    }

    func testAnOutputWithoutTheFieldIsNotLinked() {
        XCTAssertNil(SubagentRunLink.runId(fromOutput: "Subagent spawned in background."))
        XCTAssertNil(SubagentRunLink.runId(fromOutput: #"{"subagent_id":"not-a-uuid"}"#))
    }

    func testTheCardFallsBackToTheErrorField() {
        let record = ToolCallRecord(
            name: "spawn_subagent",
            status: .error,
            error: #"{"subagent_id":"\#(runId.uuidString)"}"#
        )
        XCTAssertEqual(SubagentRunLink.runId(from: record), runId)
    }
}

final class LocalSubagentRelayTests: XCTestCase {
    func testTheDirectiveCarriesEveryReportAndTheCoordinatorRule() {
        let directive = LocalSubagentRelay.directive(
            reports: ["[SUBAGENT_RESULT] role=Researcher id=1\nfound it", "[SUBAGENT_RESULT] role=Tester id=2\npassed"]
        )
        XCTAssertTrue(directive.contains("found it"))
        XCTAssertTrue(directive.contains("passed"))
        XCTAssertTrue(directive.contains("the user only talks to you"))
        XCTAssertTrue(directive.hasPrefix("<system>"))
        XCTAssertTrue(directive.hasSuffix("</system>"))
    }

    func testReportsAreKeptInTheOrderTheyWereGathered() throws {
        let directive = LocalSubagentRelay.directive(reports: ["FIRST", "SECOND"])
        let first = try XCTUnwrap(directive.range(of: "FIRST"))
        let second = try XCTUnwrap(directive.range(of: "SECOND"))
        XCTAssertLessThan(first.lowerBound, second.lowerBound)
    }
}

final class SubagentRecordProjectionTests: XCTestCase {
    func testALegacyInstanceBecomesAScriptedRecord() {
        let instance = SubagentInstance(
            role: "Codebase Researcher",
            taskDescription: "Find the store.",
            parentSessionId: UUID()
        )
        let record = instance.snapshotRecord()
        XCTAssertEqual(record.id, instance.id)
        XCTAssertEqual(record.preset, .explore)
        XCTAssertEqual(record.backend.kind, .scripted)
        XCTAssertEqual(record.status, .pending)
        XCTAssertEqual(record.depth, 0)
        XCTAssertEqual(record.transcript.count, 1)
        XCTAssertEqual(record.transcript.first?.kind, .task)
        XCTAssertEqual(record.transcript.first?.text, "Find the store.")
        XCTAssertEqual(record.schemaVersion, AgentRunRecord.currentSchemaVersion)
    }

    func testCancelledInstanceRecordsTheTerminalState() {
        let instance = SubagentInstance(role: "Test Runner", taskDescription: "Run the suite.")
        instance.cancel(reason: "Stopped by the operator.")
        let record = instance.snapshotRecord()
        XCTAssertEqual(record.status, .cancelled)
        XCTAssertNotNil(record.completedAt)
    }

    func testTheUsageTokenCountIsContextNotTheBilledSum() throws {
        var usage = AgentUsage()
        usage.steps = 2
        usage.promptTokens = 220
        usage.latestPromptTokens = 120
        usage.completionTokens = 12
        // The drawer's number: the last request's size plus everything
        // generated, not 220 + 12.
        XCTAssertEqual(usage.contextTokens, 132)
        // The billed total stays available for anything that prices it.
        XCTAssertEqual(usage.totalTokens, 232)
        XCTAssertTrue(usage.compactDescription.contains("132 tok"))

        // A legacy record never stored the last request's size, so its
        // recorded sum is the only honest number it can show.
        var legacy = AgentUsage()
        legacy.promptTokens = 1_352_593
        legacy.completionTokens = 7_478
        XCTAssertEqual(legacy.contextTokens, 1_360_071)

        // The new field survives a coding round trip.
        let decoded = try JSONDecoder().decode(AgentUsage.self, from: JSONEncoder().encode(usage))
        XCTAssertEqual(decoded, usage)
    }
}
