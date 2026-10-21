//
//  AgentActivityTests.swift
//  DynaMoETests
//
//  Phase 7: `AgentActivity`, the "agent operation is running" signal the MCP
//  manager uses to defer config reloads.
//
//  Pure logic: no model, no real server, no disk.
//

import XCTest
@testable import DynaMoE

final class AgentActivityTests: XCTestCase {

    func testCountsAreBalancedPerKind() {
        let activity = AgentActivity.shared
        XCTAssertFalse(activity.isBusy)
        XCTAssertTrue(activity.snapshot.isEmpty)

        activity.begin(.toolCall)
        activity.begin(.toolCall)
        XCTAssertTrue(activity.isBusy)
        XCTAssertEqual(activity.snapshot["toolCall"], 2)

        activity.end(.toolCall)
        XCTAssertTrue(activity.isBusy, "one call is still running")

        activity.end(.toolCall)
        XCTAssertFalse(activity.isBusy)
        XCTAssertTrue(activity.snapshot.isEmpty)
    }

    func testExtraEndsDoNotGoNegative() {
        let activity = AgentActivity.shared
        activity.end(.approval)
        activity.end(.approval)
        XCTAssertFalse(activity.isBusy)
        XCTAssertTrue(activity.snapshot.isEmpty)
    }
}

@MainActor
final class AgentActivityIntegrationTests: XCTestCase {

    private func waitUntil(
        _ description: String,
        timeout: TimeInterval = 3,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ condition: @MainActor () -> Bool
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertTrue(condition(), "timed out waiting for \(description)", file: file, line: line)
    }

    func testPendingApprovalCountsAsAgentActivity() async throws {
        let id = UUID()
        let waiter = Task { await ActionApprovalManager.shared.waitForApproval(id: id) }

        try await waitUntil("the approval to be pending") { AgentActivity.shared.isBusy }
        XCTAssertTrue(MCPManager.shared.isAgentOperationActive)

        // The busy flag is set before the continuation is registered, so give
        // the waiter a moment to park before answering it.
        try await Task.sleep(nanoseconds: 20_000_000)
        ActionApprovalManager.shared.approve(id: id)

        let approved = await waiter.value
        XCTAssertTrue(approved)
        try await waitUntil("the approval to release") { !AgentActivity.shared.isBusy }
        XCTAssertFalse(MCPManager.shared.isAgentOperationActive)
    }

    func testRunningToolCallCountsAsAgentActivity() async throws {
        let harness = AgentHarness.shared
        let tool = ParkingTool()
        harness.registerTools([tool])
        defer { harness.unregisterTools(prefix: "testing__") }

        let call = ParsedToolCall(
            name: "testing__park",
            arguments: [:],
            rawArguments: "{}",
            rawText: "{}"
        )
        let execution = Task { await harness.executeTool(call: call) }

        try await waitUntil("the tool call to start") { AgentActivity.shared.isBusy }
        XCTAssertTrue(MCPManager.shared.isAgentOperationActive)

        tool.releaseGate()
        _ = await execution.value

        try await waitUntil("the tool call to finish") { !AgentActivity.shared.isBusy }
        XCTAssertFalse(MCPManager.shared.isAgentOperationActive)
    }
}

/// A tool that blocks until the test lets it return, so the activity counter
/// can be observed in the middle of a call.
private final class ParkingTool: AgentTool, @unchecked Sendable {
    let definition = ToolDefinition(
        name: "testing__park",
        description: "Blocks until released.",
        parameters: ["type": AnyCodable("object")]
    )

    private let gate: AsyncStream<Void>
    private let gateContinuation: AsyncStream<Void>.Continuation

    init() {
        let pair = AsyncStream<Void>.makeStream()
        gate = pair.stream
        gateContinuation = pair.continuation
    }

    func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        for await _ in gate { break }
        return (#"{"ok":true}"#, nil, nil, true)
    }

    func releaseGate() {
        gateContinuation.yield(())
    }
}
