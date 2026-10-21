//
//  MCPTests.swift
//  DynaMoE
//
//  Pure-logic and protocol-level coverage for the MCP client. Nothing here
//  needs a model, a GPU, or a network: the transport is a mock that answers
//  JSON-RPC from an in-memory handler, so the initialize handshake, tool
//  discovery, tool calls, error mapping, and timeouts are all exercised
//  without spawning a server.
//

import XCTest
@testable import DynaMoE

// MARK: - Mock Transport

private final class MockMCPTransport: MCPTransport, @unchecked Sendable {
    let messages: AsyncStream<Data>
    private let continuation: AsyncStream<Data>.Continuation
    private let lock = NSLock()
    private var recorded: [[String: Any]] = []
    private var closed = false

    /// Called synchronously from `send` (off the main actor) for every frame
    /// the client writes. The closure decides whether and how to answer.
    var responder: (([String: Any], MockMCPTransport) -> Void)?

    private var parkedSendNanos: UInt64 = 0

    /// When greater than zero, `send` parks this long before recording the
    /// frame and aborts entirely if the sending task is cancelled while
    /// parked, simulating a write that has not completed yet.
    var sendParkNanos: UInt64 {
        get {
            lock.lock()
            defer { lock.unlock() }
            return parkedSendNanos
        }
        set {
            lock.lock()
            parkedSendNanos = newValue
            lock.unlock()
        }
    }

    init() {
        let stream = AsyncStream<Data>.makeStream()
        self.messages = stream.stream
        self.continuation = stream.continuation
    }

    var sentFrames: [[String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    var stderrTail: String { "" }
    private var recordedClosureReason: String?

    var closureReason: String? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return recordedClosureReason
        }
        set {
            lock.lock()
            recordedClosureReason = newValue
            lock.unlock()
        }
    }

    func start() async throws {}

    func send(_ payload: Data) async throws {
        guard let object = try? JSONSerialization.jsonObject(with: payload) as? [String: Any] else {
            XCTFail("client wrote a non-object frame")
            return
        }

        let park = sendParkNanos
        if park > 0 {
            try? await Task.sleep(nanoseconds: park)
            if Task.isCancelled { return }
        }

        lock.lock()
        recorded.append(object)
        lock.unlock()

        // Notifications carry no id and are never answered.
        guard object["id"] != nil else { return }
        responder?(object, self)
    }

    func close() {
        lock.lock()
        if closed {
            lock.unlock()
            return
        }
        closed = true
        lock.unlock()
        continuation.finish()
    }

    var isClosed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return closed
    }

    func reply(id: Any, result: [String: Any]) {
        push(["jsonrpc": "2.0", "id": id, "result": result])
    }

    func replyError(id: Any, code: Int, message: String) {
        push(["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]])
    }

    func pushNotification(method: String) {
        push(["jsonrpc": "2.0", "method": method])
    }

    func push(_ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        continuation.yield(data)
    }
}

private final class AtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    func set() {
        lock.lock()
        value = true
        lock.unlock()
    }

    var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

private final class CloseCallbackBox: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    private var closedClient: MCPClient?
    private var detail = ""

    func record(client: MCPClient, detail: String) {
        lock.lock()
        fired = true
        closedClient = client
        self.detail = detail
        lock.unlock()
    }

    var hasFired: Bool {
        lock.lock()
        defer { lock.unlock() }
        return fired
    }

    var client: MCPClient? {
        lock.lock()
        defer { lock.unlock() }
        return closedClient
    }

    var closureDetail: String {
        lock.lock()
        defer { lock.unlock() }
        return detail
    }
}

// MARK: - Helpers

private func resolvedServer(
    key: String = "testserver",
    kind: MCPServerConfig.TransportKind = .stdio,
    startupTimeout: Double = 2,
    toolTimeout: Double = 2
) -> MCPResolvedServer {
    MCPResolvedServer(
        key: key,
        kind: kind,
        command: kind == .stdio ? "echo" : nil,
        arguments: [],
        environment: [:],
        workingDirectory: nil,
        endpoint: kind == .http ? URL(string: "https://example.invalid/mcp") : nil,
        headers: [:],
        startupTimeoutSec: startupTimeout,
        toolTimeoutSec: toolTimeout,
        rawConfig: MCPServerConfig(command: kind == .stdio ? "echo" : nil, url: kind == .http ? "https://example.invalid/mcp" : nil)
    )
}

private func makeDescriptor(
    name: String,
    description: String = "A test tool.",
    properties: [String: Any] = ["query": ["type": "string"]],
    readOnly: Bool = false,
    destructive: Bool = false
) -> MCPToolDescriptor {
    let json: [String: Any] = [
        "name": name,
        "description": description,
        "inputSchema": [
            "type": "object",
            "properties": properties,
            "required": ["query"]
        ],
        "annotations": [
            "readOnlyHint": readOnly,
            "destructiveHint": destructive
        ]
    ]
    return MCPToolDescriptor(json: MCPJSONValue(any: json))!
}

// MARK: - Config Model

final class MCPConfigModelTests: XCTestCase {

    func testDecodesMinimalConfigWithDefaults() throws {
        let json = """
        {
          "mcpServers": {
            "filesystem": { "command": "npx", "args": ["-y", "server"] }
          }
        }
        """
        let document = try JSONDecoder().decode(MCPConfigDocument.self, from: Data(json.utf8))
        XCTAssertEqual(document.version, MCPConfigDocument.currentVersion)
        let server = try XCTUnwrap(document.mcpServers["filesystem"])
        XCTAssertTrue(server.enabled, "an absent 'enabled' must mean on")
        XCTAssertEqual(server.endpointDescription, "npx -y server")
        XCTAssertEqual(server.transportKind, .stdio)
        XCTAssertEqual(server.resolvedStartupTimeoutSec, MCPServerConfig.defaultStartupTimeoutSec)
        XCTAssertEqual(server.resolvedToolTimeoutSec, MCPServerConfig.defaultToolTimeoutSec)
    }

    func testDecodesServerUrlAlias() throws {
        let json = """
        { "mcpServers": { "remote": { "serverUrl": "https://mcp.example.com/mcp" } } }
        """
        let document = try JSONDecoder().decode(MCPConfigDocument.self, from: Data(json.utf8))
        let server = try XCTUnwrap(document.mcpServers["remote"])
        XCTAssertEqual(server.url, "https://mcp.example.com/mcp")
        XCTAssertEqual(server.transportKind, .http)
    }

    func testRoundTripsThroughEncoder() throws {
        var document = MCPConfigDocument()
        document["fs"] = MCPServerConfig(
            enabled: false,
            command: "npx",
            args: ["-y", "server"],
            env: ["LOG_LEVEL": "warn"],
            cwd: "/tmp",
            startupTimeoutSec: 12,
            toolTimeoutSec: 34
        )
        let data = try JSONEncoder().encode(document)
        let decoded = try JSONDecoder().decode(MCPConfigDocument.self, from: data)
        XCTAssertEqual(decoded, document)
    }

    func testTransportKindRequiresExactlyOneEndpoint() {
        XCTAssertEqual(MCPServerConfig(command: "npx").transportKind, .stdio)
        XCTAssertEqual(MCPServerConfig(url: "https://x/y").transportKind, .http)
        XCTAssertNil(MCPServerConfig(command: "npx", url: "https://x/y").transportKind)
        XCTAssertNil(MCPServerConfig().transportKind)
        XCTAssertNil(MCPServerConfig(command: "   ").transportKind)
    }

    func testUnknownKeysAreIgnored() throws {
        let json = """
        { "mcpServers": { "a": { "command": "x", "futureField": 3 } } }
        """
        let document = try JSONDecoder().decode(MCPConfigDocument.self, from: Data(json.utf8))
        XCTAssertNotNil(document.mcpServers["a"])
    }

    func testOversizedTimeoutsDecodeButResolveToTimerRange() throws {
        let json = """
        {
          "mcpServers": {
            "huge": { "command": "npx", "startupTimeoutSec": 1e300, "toolTimeoutSec": 1e300 }
          }
        }
        """
        let document = try JSONDecoder().decode(MCPConfigDocument.self, from: Data(json.utf8))
        let server = try XCTUnwrap(document.mcpServers["huge"])
        XCTAssertEqual(server.startupTimeoutSec, 1e300)
        XCTAssertEqual(server.resolvedStartupTimeoutSec, MCPServerConfig.maximumTimeoutSec)
        XCTAssertEqual(server.resolvedToolTimeoutSec, MCPServerConfig.maximumTimeoutSec)
        // The whole point of the clamp: this conversion must not trap.
        XCTAssertLessThanOrEqual(server.resolvedToolTimeoutSec * 1_000_000_000, Double(UInt64.max))
    }

    func testUnusableTimeoutsFallBackToDefaults() {
        for value in [Double.nan, .infinity, -1, 0] {
            let server = MCPServerConfig(command: "npx", startupTimeoutSec: value, toolTimeoutSec: value)
            XCTAssertEqual(server.resolvedStartupTimeoutSec, MCPServerConfig.defaultStartupTimeoutSec, "\(value)")
            XCTAssertEqual(server.resolvedToolTimeoutSec, MCPServerConfig.defaultToolTimeoutSec, "\(value)")
        }
    }

    func testTimerSecondsFloorsAndClamps() {
        XCTAssertEqual(MCPServerConfig.timerSeconds(forTimeout: 2), 2)
        XCTAssertEqual(MCPServerConfig.timerSeconds(forTimeout: 0), 0.1)
        XCTAssertEqual(MCPServerConfig.timerSeconds(forTimeout: Double.nan), 0.1)
        XCTAssertEqual(MCPServerConfig.timerSeconds(forTimeout: 1e300), MCPServerConfig.maximumTimeoutSec)
    }
}

// MARK: - Validation

final class MCPConfigValidatorTests: XCTestCase {

    func testRejectsInvalidServerNameCharacters() {
        let issues = MCPConfigValidator.validate(server: MCPServerConfig(command: "npx"), key: "bad key!")
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("only contain") })
    }

    func testRejectsDoubleUnderscoreInServerName() {
        let issues = MCPConfigValidator.validate(server: MCPServerConfig(command: "npx"), key: "a__b")
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("cannot contain '__'") })
    }

    func testRejectsTrailingUnderscoreInServerName() {
        // `a` and `a_` both pass the character check, but `a_`'s tool `echo`
        // and `a`'s tool `_echo` both become `mcp__a___echo`, and the prefix
        // teardown for `a` would delete `a_`'s tools too.
        for key in ["a_", "_", "filesystem_"] {
            let issues = MCPConfigValidator.validate(server: MCPServerConfig(command: "npx"), key: key)
            XCTAssertTrue(
                issues.contains { $0.severity == .error && $0.message.contains("cannot end with '_'") },
                "\(key): \(issues.map(\.message).joined(separator: " | "))"
            )
        }
    }

    func testAcceptsUnderscoresInsideServerName() {
        for key in ["_a", "a_b", "a-b", "a1"] {
            let issues = MCPConfigValidator.validate(server: MCPServerConfig(command: "npx"), key: key)
            XCTAssertFalse(issues.contains { $0.severity == .error }, "\(key): \(issues.map(\.message).joined(separator: " | "))")
        }
    }

    func testTrailingUnderscoreServerIsRefusedByTheRuntime() {
        // Validation is the only gate between a hand-edited file and a live
        // connection, so the resolver must refuse the key before it can
        // register a colliding tool name.
        let (resolved, issues) = MCPConfigResolver.resolve(
            key: "a_",
            config: MCPServerConfig(command: "npx"),
            workspacePath: nil
        )
        XCTAssertNil(resolved)
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("cannot end with '_'") })
    }

    func testRejectsBothEndpoints() {
        let issues = MCPConfigValidator.validate(
            server: MCPServerConfig(command: "npx", url: "https://example.com/mcp"),
            key: "both"
        )
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("keep exactly one") })
    }

    func testRejectsNoEndpoint() {
        let issues = MCPConfigValidator.validate(server: MCPServerConfig(), key: "empty")
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("either 'command'") })
    }

    func testRejectsNonHTTPURLScheme() {
        let issues = MCPConfigValidator.validate(server: MCPServerConfig(url: "ftp://example.com/x"), key: "ftp")
        XCTAssertTrue(issues.contains { $0.severity == .error && $0.message.contains("Unsupported URL scheme") })
    }

    func testWarnsOnCleartextRemoteHTTP() {
        let issues = MCPConfigValidator.validate(server: MCPServerConfig(url: "http://example.com/mcp"), key: "plain")
        XCTAssertTrue(issues.contains { $0.severity == .warning && $0.message.contains("Plain http") })
    }

    func testLoopbackHTTPIsNotWarned() {
        for host in ["http://localhost:3000/mcp", "http://127.0.0.1:3000/mcp"] {
            let issues = MCPConfigValidator.validate(server: MCPServerConfig(url: host), key: "local")
            XCTAssertFalse(issues.contains { $0.severity == .warning }, "\(host) should not warn")
        }
    }

    func testRejectsNonPositiveTimeouts() {
        let issues = MCPConfigValidator.validate(
            server: MCPServerConfig(command: "npx", startupTimeoutSec: 0, toolTimeoutSec: -1),
            key: "timeouts"
        )
        XCTAssertEqual(issues.filter { $0.severity == .error }.count, 2)
    }

    func testRejectsTimeoutTooTallToConvert() {
        // `1e300` is valid JSON, passes a `> 0` check, and then traps when the
        // deadline converts it to `UInt64` nanoseconds.
        let issues = MCPConfigValidator.validate(
            server: MCPServerConfig(command: "npx", startupTimeoutSec: 1e300),
            key: "timeouts"
        )
        XCTAssertTrue(
            issues.contains { $0.severity == .error && $0.message.contains("out of range") },
            issues.map(\.message).joined(separator: " | ")
        )
    }

    func testRejectsNonFiniteTimeouts() {
        for value in [Double.infinity, .nan] {
            let issues = MCPConfigValidator.validate(
                server: MCPServerConfig(command: "npx", startupTimeoutSec: value, toolTimeoutSec: value),
                key: "timeouts"
            )
            XCTAssertEqual(
                issues.filter { $0.severity == .error && $0.message.contains("finite") }.count,
                2,
                "\(value)"
            )
        }
    }

    func testAcceptsTimeoutAtTimerLimit() {
        let issues = MCPConfigValidator.validate(
            server: MCPServerConfig(command: "npx", startupTimeoutSec: MCPServerConfig.maximumTimeoutSec),
            key: "timeouts"
        )
        XCTAssertFalse(issues.contains { $0.severity == .error }, issues.map(\.message).joined(separator: " | "))
    }

    func testValidServerHasNoIssues() {
        let issues = MCPConfigValidator.validate(
            server: MCPServerConfig(command: "npx", args: ["-y", "server"]),
            key: "good"
        )
        XCTAssertTrue(issues.isEmpty, "unexpected issues: \(issues.map(\.message))")
    }
}

// MARK: - Key-Name Validation

final class MCPConfigKeyValidationTests: XCTestCase {

    private func issues(command: String = "npx", env: [String: String]? = nil, headers: [String: String]? = nil) -> [MCPConfigIssue] {
        MCPConfigValidator.validate(
            server: MCPServerConfig(
                command: command,
                env: env,
                url: command.isEmpty ? "https://example.com/mcp" : nil,
                headers: headers
            ),
            key: "trello"
        )
    }

    func testRejectsHeaderNameWithComma() {
        // The exact shape that produced an opaque "HTTP 400 http1.codec_error":
        // two header names joined into one field.
        let found = issues(command: "", headers: ["API_TOKEN, Authorization": "Bearer x"])
        XCTAssertTrue(
            found.contains { $0.severity == .error && $0.message.contains("Header name") },
            found.map(\.message).joined(separator: " | ")
        )
    }

    func testRejectsHeaderNameWithSpace() {
        let found = issues(command: "", headers: ["Authori zation": "Bearer x"])
        XCTAssertTrue(found.contains { $0.severity == .error && $0.message.contains("Header name") })
    }

    func testRejectsHeaderValueWithLineBreak() {
        let found = issues(command: "", headers: ["Authorization": "Bearer x\r\nX-Injected: 1"])
        XCTAssertTrue(found.contains { $0.severity == .error && $0.message.contains("line break") })
    }

    func testAcceptsStandardHeaderNames() {
        let found = issues(command: "", headers: [
            "Authorization": "Bearer x",
            "X-Api-Key": "y",
            "X-Custom_Header": "z"
        ])
        XCTAssertFalse(found.contains { $0.severity == .error }, found.map(\.message).joined(separator: " | "))
    }

    func testRejectsInvalidEnvironmentNames() {
        XCTAssertTrue(issues(env: ["1TOKEN": "x"]).contains { $0.message.contains("Environment variable name") })
        XCTAssertTrue(issues(env: ["API-TOKEN": "x"]).contains { $0.message.contains("Environment variable name") })
        XCTAssertTrue(issues(env: ["API TOKEN": "x"]).contains { $0.message.contains("Environment variable name") })
    }

    func testAcceptsEnvironmentNames() {
        XCTAssertTrue(issues(env: ["API_TOKEN": "x"]).isEmpty)
        XCTAssertTrue(issues(env: ["_PRIVATE": "x"]).isEmpty)
        XCTAssertTrue(issues(env: ["A1": "x"]).isEmpty)
    }

    func testHeaderNamePredicate() {
        XCTAssertTrue(MCPConfigValidator.isValidHeaderName("Authorization"))
        XCTAssertTrue(MCPConfigValidator.isValidHeaderName("X-Api-Key"))
        XCTAssertFalse(MCPConfigValidator.isValidHeaderName("API_TOKEN, Authorization"))
        XCTAssertFalse(MCPConfigValidator.isValidHeaderName(""))
        XCTAssertFalse(MCPConfigValidator.isValidHeaderName("Has:Colon"))
    }
}

// MARK: - HTTP Error Summary

final class MCPHTTPErrorSummaryTests: XCTestCase {

    func testStripsHTMLDownToTheUsefulText() {
        let body = """
        <!DOCTYPE html>
        <html><head><title>Bad Request</title>
        <meta name="details" content="http1.codec_error"></head>
        <body><svg><path d="M5,3 8,2"/></svg></body></html>
        """
        let summary = MCPHTTPTransport.summarizeErrorBody(body)
        XCTAssertTrue(summary.contains("Bad Request"), summary)
        XCTAssertFalse(summary.contains("<"), summary)
        XCTAssertFalse(summary.contains("path d="), summary)
        XCTAssertLessThan(summary.count, 60)
    }

    func testPlainTextBodyIsPreservedAndCollapsed() {
        let summary = MCPHTTPTransport.summarizeErrorBody("unauthorized\n\n  token expired  ")
        XCTAssertEqual(summary, "unauthorized token expired")
    }

    func testSummaryIsBounded() {
        let summary = MCPHTTPTransport.summarizeErrorBody(String(repeating: "x", count: 5000), limit: 50)
        XCTAssertEqual(summary.count, 53)
        XCTAssertTrue(summary.hasSuffix("..."))
    }
}

// MARK: - Token Expansion

final class MCPTokenExpanderTests: XCTestCase {

    func testExpandsWorkspaceToken() {
        let resolution = MCPTokenExpander.expand("/repos/${workspace}/app", workspacePath: "/tmp/ws")
        XCTAssertEqual(resolution.value, "/repos//tmp/ws/app")
        XCTAssertTrue(resolution.unresolved.isEmpty)
    }

    func testUnresolvedKeychainTokenIsReported() {
        let account = "dynamoe.mcp.testserver.unresolved-\(UUID().uuidString)"
        let resolution = MCPTokenExpander.expand("Bearer ${keychain:\(account)}", workspacePath: nil)
        XCTAssertEqual(resolution.unresolved, ["${keychain:\(account)}"])
    }

    func testUnknownTokenIsReported() {
        let resolution = MCPTokenExpander.expand("${nonsense}", workspacePath: nil)
        XCTAssertEqual(resolution.unresolved, ["${nonsense}"])
    }

    func testPlainTextIsUntouched() {
        let resolution = MCPTokenExpander.expand("https://example.com/a?b=c", workspacePath: nil)
        XCTAssertEqual(resolution.value, "https://example.com/a?b=c")
        XCTAssertTrue(resolution.unresolved.isEmpty)
    }

    func testPureKeychainTokenDetection() {
        let token = MCPTokenExpander.keychainToken(account: "dynamoe.mcp.fs.token")
        XCTAssertTrue(MCPTokenExpander.isPureKeychainToken(token))
        XCTAssertEqual(MCPTokenExpander.keychainAccount(in: token), "dynamoe.mcp.fs.token")
        XCTAssertFalse(MCPTokenExpander.isPureKeychainToken("Bearer \(token)"))
        XCTAssertNil(MCPTokenExpander.keychainAccount(in: "plain"))
    }

    func testResolverFailsOnUnresolvedCredential() {
        let account = "dynamoe.mcp.resolver.\(UUID().uuidString)"
        let config = MCPServerConfig(command: "npx", env: ["TOKEN": "${keychain:\(account)}"])
        let result = MCPConfigResolver.resolve(key: "resolver", config: config, workspacePath: nil)
        XCTAssertNil(result.server)
        XCTAssertTrue(result.issues.contains { $0.severity == .error && $0.message.contains("Unresolved credential") })
    }

    func testResolverExpandsStdioServer() {
        let config = MCPServerConfig(
            command: "npx",
            args: ["-y", "${workspace}/server.js"],
            env: ["MODE": "test"]
        )
        let result = MCPConfigResolver.resolve(key: "fs", config: config, workspacePath: "/tmp/ws")
        let server = try? XCTUnwrap(result.server)
        XCTAssertEqual(server?.kind, .stdio)
        XCTAssertEqual(server?.arguments, ["-y", "/tmp/ws/server.js"])
        XCTAssertEqual(server?.environment, ["MODE": "test"])
    }
}

// MARK: - Framing

final class MCPFramingTests: XCTestCase {

    func testLineBufferAssemblesSplitFrames() {
        var buffer = MCPLineBuffer(limit: 4096)
        XCTAssertTrue(buffer.append(Data("{\"a\":1}".utf8)).isEmpty)
        let lines = buffer.append(Data("\n{\"b\":".utf8))
        XCTAssertEqual(lines, ["{\"a\":1}"])
        let rest = buffer.append(Data("2}\n".utf8))
        XCTAssertEqual(rest, ["{\"b\":2}"])
    }

    func testLineBufferStripsCarriageReturn() {
        var buffer = MCPLineBuffer(limit: 4096)
        XCTAssertEqual(buffer.append(Data("hello\r\n".utf8)), ["hello"])
    }

    func testLineBufferFlushesUnterminatedTail() {
        var buffer = MCPLineBuffer(limit: 4096)
        _ = buffer.append(Data("tail".utf8))
        XCTAssertEqual(buffer.flush(), "tail")
        XCTAssertNil(buffer.flush())
    }

    func testLineBufferRefusesOversizedFrame() {
        var buffer = MCPLineBuffer(limit: 8)
        _ = buffer.append(Data(String(repeating: "x", count: 32).utf8))
        XCTAssertTrue(buffer.overflowed)
        XCTAssertTrue(buffer.append(Data("\n".utf8)).isEmpty)
    }

    func testLineBufferRefusesOversizedCompleteFrame() {
        var buffer = MCPLineBuffer(limit: 8)
        XCTAssertTrue(
            buffer.append(Data("123456789\n".utf8)).isEmpty,
            "a nine-byte frame must not slip through just because it is newline-terminated"
        )
        XCTAssertTrue(buffer.overflowed)
        XCTAssertTrue(buffer.append(Data("ok\n".utf8)).isEmpty, "an overflowed buffer stays refused")
    }

    func testLineBufferAcceptsFrameExactlyAtLimit() {
        var buffer = MCPLineBuffer(limit: 8)
        XCTAssertEqual(buffer.append(Data("12345678\n".utf8)), ["12345678"])
        XCTAssertFalse(buffer.overflowed)
    }

    func testLineBufferOverflowKeepsEarlierLinesAndDropsTheRest() {
        var buffer = MCPLineBuffer(limit: 8)
        let lines = buffer.append(Data("ok\n123456789\nok2\n".utf8))
        XCTAssertEqual(lines, ["ok"], "frames completed before the oversized one are still delivered")
        XCTAssertTrue(buffer.overflowed)
        XCTAssertFalse(lines.contains("ok2"), "frames after the oversized one are dropped")
    }

    func testLineBufferCountsCarriageReturnTowardLimit() {
        var buffer = MCPLineBuffer(limit: 8)
        XCTAssertTrue(
            buffer.append(Data("12345678\r\n".utf8)).isEmpty,
            "the CRLF pair must not buy an oversized frame an extra byte"
        )
        XCTAssertTrue(buffer.overflowed)
    }

    func testSSEParserJoinsMultipleDataLines() {
        var parser = MCPSSEFrameParser()
        XCTAssertNil(parser.consume(line: "event: message"))
        XCTAssertNil(parser.consume(line: "data: {\"a\":"))
        XCTAssertNil(parser.consume(line: "data: 1}"))
        let frame = parser.consume(line: "")
        XCTAssertEqual(frame?.event, "message")
        XCTAssertEqual(frame?.data, "{\"a\":\n1}")
    }

    func testSSEParserIgnoresCommentsAndBlankFrames() {
        var parser = MCPSSEFrameParser()
        XCTAssertNil(parser.consume(line: ": keep-alive"))
        XCTAssertNil(parser.consume(line: ""))
        XCTAssertNil(parser.finish())
    }

    func testSSEParserFlushDispatchesBufferedData() {
        var parser = MCPSSEFrameParser()
        _ = parser.consume(line: "data: {\"x\":1}")
        let frame = parser.finish()
        XCTAssertEqual(frame?.data, "{\"x\":1}")
    }
}

// MARK: - Wire Types

final class MCPWireTests: XCTestCase {

    func testIncomingMessageDecodesResponse() throws {
        let data = Data("{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"tools\":[]}}".utf8)
        let message = try XCTUnwrap(MCPIncomingMessage(data: data))
        XCTAssertEqual(message.id, 7)
        XCTAssertNil(message.method)
        XCTAssertNotNil(message.result)
        XCTAssertNil(message.errorCode)
    }

    func testIncomingMessageDecodesError() throws {
        let data = Data("{\"jsonrpc\":\"2.0\",\"id\":9,\"error\":{\"code\":-32601,\"message\":\"nope\"}}".utf8)
        let message = try XCTUnwrap(MCPIncomingMessage(data: data))
        XCTAssertEqual(message.id, 9)
        XCTAssertEqual(message.errorCode, -32601)
        XCTAssertEqual(message.errorMessage, "nope")
    }

    func testIncomingMessageDecodesNotificationWithoutID() throws {
        let data = Data("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}".utf8)
        let message = try XCTUnwrap(MCPIncomingMessage(data: data))
        XCTAssertNil(message.id)
        XCTAssertNil(message.rawID)
        XCTAssertEqual(message.method, "notifications/tools/list_changed")
    }

    func testIncomingMessagePreservesNonNumericStringID() throws {
        let data = Data("{\"jsonrpc\":\"2.0\",\"id\":\"health-check\",\"method\":\"ping\"}".utf8)
        let message = try XCTUnwrap(MCPIncomingMessage(data: data))
        XCTAssertEqual(message.rawID, .string("health-check"))
        XCTAssertNil(message.id, "a non-numeric string id must not masquerade as a numeric one")
        XCTAssertEqual(message.method, "ping")
    }

    func testIncomingMessagePreservesNumericStringIDVerbatim() throws {
        let data = Data("{\"jsonrpc\":\"2.0\",\"id\":\"007\",\"method\":\"ping\"}".utf8)
        let message = try XCTUnwrap(MCPIncomingMessage(data: data))
        XCTAssertEqual(message.rawID, .string("007"), "the raw id keeps its original spelling")
        XCTAssertEqual(message.id, 7, "permissive matching still sees the numeric value")
    }

    func testResponseBuildersEchoRawIDTypeAndValue() throws {
        let success = try MCPWire.successResponse(id: .string("health-check"), result: [:])
        let successObject = try XCTUnwrap(JSONSerialization.jsonObject(with: success) as? [String: Any])
        XCTAssertEqual(successObject["id"] as? String, "health-check")

        let failure = try MCPWire.errorResponse(id: .string("007"), code: -32601, message: "nope")
        let failureObject = try XCTUnwrap(JSONSerialization.jsonObject(with: failure) as? [String: Any])
        XCTAssertEqual(failureObject["id"] as? String, "007", "the id is not reinterpreted as a number")

        let intEcho = try MCPWire.successResponse(id: .int(42), result: [:])
        let intObject = try XCTUnwrap(JSONSerialization.jsonObject(with: intEcho) as? [String: Any])
        XCTAssertEqual(intObject["id"] as? Int, 42)
    }

    func testIncomingMessageRejectsNonObject() {
        XCTAssertNil(MCPIncomingMessage(data: Data("[1,2,3]".utf8)))
        XCTAssertNil(MCPIncomingMessage(data: Data("not json".utf8)))
    }

    func testJSONValueDistinguishesBooleansFromIntegers() {
        XCTAssertEqual(MCPJSONValue(any: true), .bool(true))
        XCTAssertEqual(MCPJSONValue(any: 1), .int(1))
        XCTAssertEqual(MCPJSONValue(any: 1.5), .double(1.5))
        XCTAssertEqual(MCPJSONValue(any: "s"), .string("s"))
        XCTAssertEqual(MCPJSONValue(any: NSNull()), .null)
        if case .array(let values) = MCPJSONValue(any: [1, "a"]) {
            XCTAssertEqual(values.count, 2)
        } else {
            XCTFail("expected an array value")
        }
    }

    func testRequestEncodingShape() throws {
        let data = try MCPWire.request(id: 3, method: "tools/list", params: ["cursor": "x"])
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["jsonrpc"] as? String, "2.0")
        XCTAssertEqual(object["id"] as? Int, 3)
        XCTAssertEqual(object["method"] as? String, "tools/list")
        XCTAssertEqual((object["params"] as? [String: Any])?["cursor"] as? String, "x")
    }

    func testNotificationEncodingHasNoID() throws {
        let data = try MCPWire.notification(method: "notifications/initialized", params: nil)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(object["id"])
        XCTAssertEqual(object["method"] as? String, "notifications/initialized")
    }
}

// MARK: - Tool Bridge

final class MCPToolBridgeTests: XCTestCase {

    func testQualifiedNameRoundTrip() {
        let qualified = MCPToolBridge.qualifiedName(serverKey: "filesystem", toolName: "read_file")
        XCTAssertEqual(qualified, "mcp__filesystem__read_file")
        let parsed = MCPToolBridge.parse(qualifiedName: qualified)
        XCTAssertEqual(parsed?.serverKey, "filesystem")
        XCTAssertEqual(parsed?.toolName, "read_file")
    }

    func testParseRejectsMalformedNames() {
        XCTAssertNil(MCPToolBridge.parse(qualifiedName: "shell_run"))
        XCTAssertNil(MCPToolBridge.parse(qualifiedName: "mcp__onlyserver"))
        XCTAssertNil(MCPToolBridge.parse(qualifiedName: "mcp____tool"))
    }

    func testLeadingUnderscoreToolKeepsItsServerBoundary() {
        // `mcp__a__` + `_echo` is the exact string a server key `a_` would
        // produce for `echo`, which is why keys may not end with `_`.
        let qualified = MCPToolBridge.qualifiedName(serverKey: "a", toolName: "_echo")
        XCTAssertEqual(qualified, "mcp__a___echo")
        XCTAssertEqual(
            MCPToolBridge.qualifiedName(serverKey: "a_", toolName: "echo"),
            qualified,
            "the colliding shape the key rule prevents"
        )
        let parsed = MCPToolBridge.parse(qualifiedName: qualified)
        XCTAssertEqual(parsed?.serverKey, "a")
        XCTAssertEqual(parsed?.toolName, "_echo")
    }

    func testDefinitionCarriesServerPrefixAndSchema() {
        let transport = MockMCPTransport()
        let client = MCPClient(config: resolvedServer(), transport: transport)
        let descriptor = makeDescriptor(name: "read_file", description: "Reads a file.")
        let bridge = MCPToolBridge(serverKey: "filesystem", descriptor: descriptor, client: client)

        XCTAssertEqual(bridge.definition.function.name, "mcp__filesystem__read_file")
        XCTAssertTrue(bridge.definition.function.description.contains("[MCP: filesystem]"))
        XCTAssertEqual(bridge.definition.type, "function")

        let parameters = bridge.definition.function.parameters
        XCTAssertEqual(parameters["type"]?.value as? String, "object")
        let properties = try? XCTUnwrap(parameters["properties"]?.value as? [String: Any])
        XCTAssertNotNil(properties?["query"])
        let required = parameters["required"]?.value as? [String]
        XCTAssertEqual(required, ["query"])
    }

    func testDestructiveHintIsSurfaceInDescription() {
        let client = MCPClient(config: resolvedServer(), transport: MockMCPTransport())
        let descriptor = makeDescriptor(name: "write", readOnly: false, destructive: true)
        let bridge = MCPToolBridge(serverKey: "fs", descriptor: descriptor, client: client)
        XCTAssertTrue(bridge.definition.function.description.contains("may modify state"))
    }

    func testSchemaDefaultsTypeAndProperties() {
        let parameters = MCPToolBridge.parameters(from: .object([:]))
        XCTAssertEqual(parameters["type"]?.value as? String, "object")
        XCTAssertNotNil(parameters["properties"])
    }

    func testSchemaConvertsNestedStructures() {
        let schema: MCPJSONValue = .object([
            "type": .string("object"),
            "properties": .object([
                "paths": .object([
                    "type": .string("array"),
                    "items": .object(["type": .string("string")])
                ]),
                "flag": .object(["type": .string("boolean")])
            ])
        ])
        let parameters = MCPToolBridge.parameters(from: schema)
        let properties = parameters["properties"]?.value as? [String: Any]
        let paths = properties?["paths"] as? [String: Any]
        XCTAssertEqual(paths?["type"] as? String, "array")
        let items = paths?["items"] as? [String: Any]
        XCTAssertEqual(items?["type"] as? String, "string")
    }

    func testClipHonorsBudget() {
        let long = String(repeating: "a", count: 100)
        let clipped = MCPToolBridge.clip(long, to: 10)
        XCTAssertTrue(clipped.didClip)
        XCTAssertEqual(clipped.text.count, 10)
        XCTAssertEqual(clipped.originalCount, 100)

        let untouched = MCPToolBridge.clip("short", to: 10)
        XCTAssertFalse(untouched.didClip)
        XCTAssertEqual(untouched.text, "short")
    }

    /// A client whose server answers `tools/call` with `result`.
    private func makeExecutingClient(replyingToCall result: [String: Any]) async throws -> MCPClient {
        let transport = MockMCPTransport()
        transport.responder = { request, transport in
            guard let id = request["id"] else { return }
            switch request["method"] as? String {
            case "initialize":
                transport.reply(id: id, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": ["tools": [:]],
                    "serverInfo": ["name": "mock-server", "version": "1"]
                ])
            case "tools/list":
                transport.reply(id: id, result: [
                    "tools": [["name": "list", "inputSchema": ["type": "object"]]]
                ])
            case "tools/call":
                transport.reply(id: id, result: result)
            default:
                transport.replyError(id: id, code: -32601, message: "unsupported")
            }
        }
        let client = MCPClient(config: resolvedServer(), transport: transport)
        try await client.start()
        return client
    }

    func testStructuredOnlyResultStaysWithinTheCharacterBudget() async throws {
        // `MCPCallResult.text` is already the serialized structured content for
        // a content-less reply; appending the object again would restore the
        // unbounded payload the character budget exists to cap.
        let blob = String(repeating: "x", count: 5_000)
        let client = try await makeExecutingClient(replyingToCall: [
            "content": [],
            "structuredContent": ["blob": blob]
        ])
        let bridge = MCPToolBridge(
            serverKey: "filesystem",
            descriptor: makeDescriptor(name: "list"),
            client: client
        )

        let output = try await bridge.execute(arguments: [:], workingDirectory: nil, maxOutputLength: 400)

        XCTAssertEqual(output.stdout?.count, 400)
        XCTAssertLessThan(
            output.resultJSON.count,
            1_200,
            "the tool result must stay near the 400-character budget, got \(output.resultJSON.count) characters"
        )
        XCTAssertFalse(output.resultJSON.contains("structured_content"))
        XCTAssertTrue(output.resultJSON.contains("truncated"))

        await client.stop()
    }

    func testStructuredOnlyResultReachesTheModelThroughText() async throws {
        let client = try await makeExecutingClient(replyingToCall: [
            "content": [],
            "structuredContent": ["count": 3]
        ])
        let bridge = MCPToolBridge(
            serverKey: "filesystem",
            descriptor: makeDescriptor(name: "list"),
            client: client
        )

        let output = try await bridge.execute(arguments: [:], workingDirectory: nil, maxOutputLength: 4_000)

        XCTAssertTrue(output.stdout?.contains("\"count\"") ?? false, "got \(output.stdout ?? "nil")")
        XCTAssertTrue(output.resultJSON.contains("count"))
        XCTAssertFalse(output.resultJSON.contains("structured_content"))

        await client.stop()
    }
}

// MARK: - Call Result Mapping

final class MCPCallResultTests: XCTestCase {

    func testJoinsTextContentBlocks() {
        let json: [String: MCPJSONValue] = [
            "content": .array([
                .object(["type": .string("text"), "text": .string("first")]),
                .object(["type": .string("text"), "text": .string("second")])
            ])
        ]
        let result = MCPCallResult(json: json)
        XCTAssertEqual(result.text, "first\n\nsecond")
        XCTAssertEqual(result.contentBlockCount, 2)
        XCTAssertFalse(result.isError)
    }

    func testSummarizesImageBlocks() {
        let base64 = String(repeating: "A", count: 4096)
        let json: [String: MCPJSONValue] = [
            "content": .array([
                .object([
                    "type": .string("image"),
                    "mimeType": .string("image/png"),
                    "data": .string(base64)
                ])
            ])
        ]
        let result = MCPCallResult(json: json)
        XCTAssertEqual(result.imageBlockCount, 1)
        XCTAssertTrue(result.text.contains("[image image/png"))
        XCTAssertTrue(result.text.contains("KB"), "expected a size hint, got \(result.text)")
    }

    func testFlagsServerReportedError() {
        let json: [String: MCPJSONValue] = [
            "isError": .bool(true),
            "content": .array([.object(["type": .string("text"), "text": .string("boom")])])
        ]
        let result = MCPCallResult(json: json)
        XCTAssertTrue(result.isError)
        XCTAssertEqual(result.text, "boom")
    }

    func testFallsBackToStructuredContent() {
        let json: [String: MCPJSONValue] = [
            "structuredContent": .object(["count": .int(3)])
        ]
        let result = MCPCallResult(json: json)
        XCTAssertEqual(result.contentBlockCount, 0)
        XCTAssertTrue(result.text.contains("\"count\""))
    }

    func testEmptyResultProducesEmptyText() {
        let result = MCPCallResult(json: [:])
        XCTAssertTrue(result.text.isEmpty)
        XCTAssertFalse(result.isError)
    }
}

// MARK: - Client Protocol Exchange

final class MCPClientProtocolTests: XCTestCase {

    private func makeClient(
        transport: MockMCPTransport,
        startupTimeout: Double = 2,
        toolTimeout: Double = 2,
        onToolsChanged: (@Sendable () -> Void)? = nil,
        onClosed: (@Sendable (MCPClient, String) -> Void)? = nil
    ) -> MCPClient {
        MCPClient(
            config: resolvedServer(startupTimeout: startupTimeout, toolTimeout: toolTimeout),
            transport: transport,
            onToolsChanged: onToolsChanged,
            onClosed: onClosed
        )
    }

    private func installHandshake(_ transport: MockMCPTransport, tools: [[String: Any]]) {
        transport.responder = { request, transport in
            let id = request["id"] as Any
            switch request["method"] as? String {
            case "initialize":
                transport.reply(id: id, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": ["tools": [:]],
                    "serverInfo": ["name": "mock-server", "version": "0.1.0"],
                    "instructions": "Be brief."
                ])
            case "tools/list":
                transport.reply(id: id, result: ["tools": tools])
            default:
                transport.replyError(id: id, code: -32601, message: "unsupported")
            }
        }
    }

    func testInitializeNegotiatesAndStoresServerInfo() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let client = makeClient(transport: transport)

        try await client.start()

        let negotiated = await client.negotiatedProtocolVersion
        let name = await client.serverName
        let instructions = await client.serverInstructions
        XCTAssertEqual(negotiated, "2025-06-18")
        XCTAssertEqual(name, "mock-server")
        XCTAssertEqual(instructions, "Be brief.")

        let methods = transport.sentFrames.compactMap { $0["method"] as? String }
        XCTAssertEqual(methods.first, "initialize")
        XCTAssertTrue(methods.contains("notifications/initialized"))
        let initialize = try XCTUnwrap(transport.sentFrames.first)
        let params = try XCTUnwrap(initialize["params"] as? [String: Any])
        XCTAssertEqual(params["protocolVersion"] as? String, MCPWire.protocolVersion)
        let clientInfo = try XCTUnwrap(params["clientInfo"] as? [String: Any])
        XCTAssertEqual(clientInfo["name"] as? String, "DynaMoE")

        await client.stop()
    }

    func testOversizedConfiguredTimeoutsDoNotTrapTheDeadline() async throws {
        // The deadline sleeps with `UInt64` nanoseconds; a config value like
        // `1e300` used to trap here and crash the test host.
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let client = makeClient(transport: transport, startupTimeout: 1e300, toolTimeout: 1e300)

        try await client.start()

        let name = await client.serverName
        XCTAssertEqual(name, "mock-server")
        await client.stop()
    }

    func testListToolsDiscoversNamesAndSchemas() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [
            [
                "name": "read_file",
                "description": "Read a file.",
                "inputSchema": ["type": "object", "properties": ["path": ["type": "string"]]]
            ],
            [
                "name": "list_dir",
                "description": "List a directory.",
                "inputSchema": ["type": "object", "properties": [:]],
                "annotations": ["readOnlyHint": true]
            ]
        ])
        let client = makeClient(transport: transport)
        try await client.start()

        let tools = try await client.listTools()
        XCTAssertEqual(tools.map(\.name), ["read_file", "list_dir"])
        XCTAssertTrue(tools[1].readOnlyHint)
        XCTAssertEqual(tools[0].inputSchema["properties"]?["path"]?["type"]?.stringValue, "string")

        await client.stop()
    }

    func testListToolsFollowsPaginationCursor() async throws {
        let transport = MockMCPTransport()
        transport.responder = { request, transport in
            guard let id = request["id"] else { return }
            switch request["method"] as? String {
            case "initialize":
                transport.reply(id: id, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": [:],
                    "serverInfo": ["name": "m", "version": "1"]
                ])
            case "tools/list":
                let params = request["params"] as? [String: Any]
                let cursor = params?["cursor"] as? String
                if cursor == nil {
                    transport.reply(id: id, result: [
                        "tools": [["name": "one", "inputSchema": ["type": "object"]]],
                        "nextCursor": "page-2"
                    ])
                } else {
                    transport.reply(id: id, result: [
                        "tools": [["name": "two", "inputSchema": ["type": "object"]]]
                    ])
                }
            default:
                transport.replyError(id: id, code: -32601, message: "unsupported")
            }
        }
        let client = makeClient(transport: transport)
        try await client.start()

        let tools = try await client.listTools()
        XCTAssertEqual(tools.map(\.name), ["one", "two"])

        let listCalls = transport.sentFrames.filter { ($0["method"] as? String) == "tools/list" }
        XCTAssertEqual(listCalls.count, 2)
        let secondParams = try XCTUnwrap(listCalls[1]["params"] as? [String: Any])
        XCTAssertEqual(secondParams["cursor"] as? String, "page-2")

        await client.stop()
    }

    func testCallToolPassesArgumentsAndReturnsContent() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [["name": "echo", "inputSchema": ["type": "object"]]])
        transport.responder = { request, transport in
            guard let id = request["id"] else { return }
            switch request["method"] as? String {
            case "initialize":
                transport.reply(id: id, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": [:],
                    "serverInfo": ["name": "m", "version": "1"]
                ])
            case "tools/call":
                let params = request["params"] as? [String: Any]
                let name = params?["name"] as? String ?? "?"
                let arguments = params?["arguments"] as? [String: Any] ?? [:]
                transport.reply(id: id, result: [
                    "content": [["type": "text", "text": "\(name):\(arguments["value"] ?? "")"]]
                ])
            default:
                transport.replyError(id: id, code: -32601, message: "unsupported")
            }
        }
        let client = makeClient(transport: transport)
        try await client.start()

        let result = try await client.callTool(name: "echo", arguments: ["value": "hello"])
        XCTAssertEqual(result.text, "echo:hello")
        XCTAssertFalse(result.isError)

        await client.stop()
    }

    func testRemoteErrorIsSurfaced() async throws {
        let transport = MockMCPTransport()
        transport.responder = { request, transport in
            guard let id = request["id"] else { return }
            if (request["method"] as? String) == "initialize" {
                transport.reply(id: id, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": [:],
                    "serverInfo": ["name": "m", "version": "1"]
                ])
            } else {
                transport.replyError(id: id, code: -32000, message: "tool exploded")
            }
        }
        let client = makeClient(transport: transport)
        try await client.start()

        do {
            _ = try await client.callTool(name: "boom", arguments: [:])
            XCTFail("expected a remote error")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("tool exploded"), error.localizedDescription)
        }
        await client.stop()
    }

    func testRequestTimesOutWhenServerNeverAnswers() async throws {
        let transport = MockMCPTransport()
        let client = makeClient(transport: transport, startupTimeout: 0.3, toolTimeout: 0.3)
        do {
            try await client.start()
            XCTFail("expected a timeout")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("timed out"), error.localizedDescription)
        }
        await client.stop()
    }

    func testToolsListChangedNotificationInvokesCallback() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let flag = AtomicFlag()
        let client = makeClient(transport: transport) { flag.set() }
        try await client.start()

        transport.pushNotification(method: "notifications/tools/list_changed")

        var waited = 0
        while !flag.isSet && waited < 40 {
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        XCTAssertTrue(flag.isSet, "the list-changed callback never fired")
        await client.stop()
    }

    func testServerRequestGetsMethodNotSupportedReply() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let client = makeClient(transport: transport)
        try await client.start()

        transport.push(["jsonrpc": "2.0", "id": 4242, "method": "roots/list"])

        var waited = 0
        while waited < 40 {
            let answered = transport.sentFrames.contains { frame in
                (frame["id"] as? Int) == 4242 && (frame["error"] as? [String: Any])?["code"] as? Int == -32601
            }
            if answered { break }
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        let answered = transport.sentFrames.contains { frame in
            (frame["id"] as? Int) == 4242 && (frame["error"] as? [String: Any])?["code"] as? Int == -32601
        }
        XCTAssertTrue(answered, "server request was not answered")
        await client.stop()
    }

    func testPingRequestIsAnsweredWithEmptyResult() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let client = makeClient(transport: transport)
        try await client.start()

        transport.push(["jsonrpc": "2.0", "id": 777, "method": "ping"])

        var waited = 0
        while waited < 40 {
            let answered = transport.sentFrames.contains { ($0["id"] as? Int) == 777 && $0["result"] != nil }
            if answered { break }
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        let answered = transport.sentFrames.contains { ($0["id"] as? Int) == 777 && $0["result"] != nil }
        XCTAssertTrue(answered, "ping was not answered")
        await client.stop()
    }

    func testStringIDServerRequestsAreAnsweredWithVerbatimEcho() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let client = makeClient(transport: transport)
        try await client.start()

        transport.push(["jsonrpc": "2.0", "id": "health-check", "method": "ping"])
        transport.push(["jsonrpc": "2.0", "id": "007", "method": "roots/list"])

        func echoed(_ frame: [String: Any], id: String, needsResult: Bool) -> Bool {
            guard (frame["id"] as? String) == id else { return false }
            if needsResult { return frame["result"] != nil }
            return (frame["error"] as? [String: Any])?["code"] as? Int == -32601
        }

        var waited = 0
        while waited < 40 {
            let pingAnswered = transport.sentFrames.contains { echoed($0, id: "health-check", needsResult: true) }
            let unsupportedAnswered = transport.sentFrames.contains { echoed($0, id: "007", needsResult: false) }
            if pingAnswered && unsupportedAnswered { break }
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        XCTAssertTrue(
            transport.sentFrames.contains { echoed($0, id: "health-check", needsResult: true) },
            "ping with a string id was not answered with the same string id"
        )
        XCTAssertTrue(
            transport.sentFrames.contains { echoed($0, id: "007", needsResult: false) },
            "unsupported method with a string id was not answered with the same string id"
        )
        XCTAssertFalse(
            transport.sentFrames.contains { ($0["id"] as? Int) == 7 },
            "the \"007\" id must not be reinterpreted as numeric 7 in the reply"
        )
        await client.stop()
    }

    func testStreamEndFailsPendingRequests() async throws {
        let transport = MockMCPTransport()
        let client = makeClient(transport: transport, startupTimeout: 5)
        let task = Task { try await client.start() }
        try await Task.sleep(nanoseconds: 100_000_000)
        transport.close()
        do {
            try await task.value
            XCTFail("expected the connection to close")
        } catch {
            XCTAssertTrue(
                error.localizedDescription.contains("closed the connection"),
                error.localizedDescription
            )
        }
    }

    func testCancellingDispatchedRequestNotifiesServer() async throws {
        let transport = MockMCPTransport()
        transport.responder = { request, transport in
            guard request["id"] != nil else { return }
            if (request["method"] as? String) == "initialize" {
                transport.reply(id: request["id"] as Any, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": [:],
                    "serverInfo": ["name": "m", "version": "1"]
                ])
            }
            // tools/call is deliberately never answered.
        }
        let client = makeClient(transport: transport)
        try await client.start()

        let task = Task { try await client.callTool(name: "slow_tool", arguments: [:]) }

        var requestID: Int?
        var waited = 0
        while waited < 40 {
            if let frame = transport.sentFrames.first(where: { ($0["method"] as? String) == "tools/call" }) {
                requestID = frame["id"] as? Int
                break
            }
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        let dispatchedID = try XCTUnwrap(requestID, "the tools/call request never reached the server")

        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected the call to be cancelled")
        } catch {
            XCTAssertTrue(error is CancellationError, "expected a CancellationError, got \(error)")
        }

        var notified = false
        waited = 0
        while waited < 40 {
            notified = transport.sentFrames.contains { frame in
                (frame["method"] as? String) == "notifications/cancelled"
                    && ((frame["params"] as? [String: Any])?["requestId"] as? Int) == dispatchedID
            }
            if notified { break }
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        XCTAssertTrue(notified, "cancelling a dispatched request never notified the server")
        await client.stop()
    }

    func testTimedOutRequestNotifiesServerButCompletedOnesDoNot() async throws {
        let transport = MockMCPTransport()
        transport.responder = { request, transport in
            guard let id = request["id"] else { return }
            if (request["method"] as? String) == "initialize" {
                transport.reply(id: id, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": [:],
                    "serverInfo": ["name": "m", "version": "1"]
                ])
            }
            // tools/call is deliberately never answered.
        }
        let client = makeClient(transport: transport, toolTimeout: 0.3)
        try await client.start()

        let initializeID = transport.sentFrames.first(where: { ($0["method"] as? String) == "initialize" })?["id"] as? Int

        do {
            _ = try await client.callTool(name: "slow_tool", arguments: [:])
            XCTFail("expected a timeout")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("timed out"), error.localizedDescription)
        }

        var noticedID: Int?
        var waited = 0
        while waited < 40 {
            if let frame = transport.sentFrames.first(where: { ($0["method"] as? String) == "notifications/cancelled" }) {
                noticedID = (frame["params"] as? [String: Any])?["requestId"] as? Int
                break
            }
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        let notifiedID = try XCTUnwrap(noticedID, "the timed-out request never notified the server")

        let timedOutID = transport.sentFrames.first(where: { ($0["method"] as? String) == "tools/call" })?["id"] as? Int
        XCTAssertEqual(notifiedID, timedOutID, "the notice must name the abandoned request")
        XCTAssertNotEqual(notifiedID, initializeID, "a completed request must not be reported cancelled")

        let noticeCount = transport.sentFrames.filter { ($0["method"] as? String) == "notifications/cancelled" }.count
        XCTAssertEqual(noticeCount, 1, "exactly one abandonment notice is expected")
        await client.stop()
    }

    func testCancellingBeforeSendDispatchesStopsTheWrite() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let client = makeClient(transport: transport)
        try await client.start()

        // Parks the tools/call write for 10 s: the send is queued, not
        // dispatched, when the test cancels the call.
        transport.sendParkNanos = 10_000_000_000

        let task = Task { try await client.callTool(name: "queued_tool", arguments: [:]) }
        try await Task.sleep(nanoseconds: 200_000_000)
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected the call to be cancelled")
        } catch {
            XCTAssertTrue(error is CancellationError, "expected a CancellationError, got \(error)")
        }

        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertFalse(
            transport.sentFrames.contains { ($0["method"] as? String) == "tools/call" },
            "a cancelled queued request must not be written after cancellation"
        )
        XCTAssertFalse(
            transport.sentFrames.contains { ($0["method"] as? String) == "notifications/cancelled" },
            "a request that never reached the server must not be reported cancelled"
        )
        await client.stop()
    }

    func testCancellingInitializeSendsNoCancellationNotice() async throws {
        let transport = MockMCPTransport()
        // No responder: initialize is dispatched but never answered.
        let client = makeClient(transport: transport, startupTimeout: 5)
        let task = Task { try await client.start() }
        try await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertTrue(
            transport.sentFrames.contains { ($0["method"] as? String) == "initialize" },
            "the initialize request never reached the server"
        )
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected the handshake to be cancelled")
        } catch {
            XCTAssertTrue(error is CancellationError, "expected a CancellationError, got \(error)")
        }

        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertFalse(
            transport.sentFrames.contains { ($0["method"] as? String) == "notifications/cancelled" },
            "abandoned initialization must not be reported cancelled"
        )
        await client.stop()
    }

    func testServerExitNotifiesCloseCallbackWithIdentityAndDetail() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let box = CloseCallbackBox()
        let client = makeClient(
            transport: transport,
            onClosed: { closedClient, detail in
                box.record(client: closedClient, detail: detail)
            }
        )
        try await client.start()

        transport.closureReason = "server exited (exit code 1)"
        transport.close()

        var waited = 0
        while !box.hasFired && waited < 40 {
            try await Task.sleep(nanoseconds: 50_000_000)
            waited += 1
        }
        XCTAssertTrue(box.hasFired, "the close callback never fired")
        XCTAssertTrue(box.client === client, "the callback must identify the client that closed")
        XCTAssertEqual(box.closureDetail, "server exited (exit code 1)")
    }

    func testDeliberateStopDoesNotNotifyCloseCallback() async throws {
        let transport = MockMCPTransport()
        installHandshake(transport, tools: [])
        let box = CloseCallbackBox()
        let client = makeClient(
            transport: transport,
            onClosed: { closedClient, detail in
                box.record(client: closedClient, detail: detail)
            }
        )
        try await client.start()

        await client.stop()
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertFalse(box.hasFired, "an explicit stop must not be reported as a server exit")
    }
}

// MARK: - Stdio Transport

final class MCPStdioTransportTests: XCTestCase {

    func testMissingBinarySurfacesClosure() async throws {
        let transport = MCPStdioTransport(
            command: "dynamoe-definitely-not-a-real-binary",
            arguments: [],
            environment: [:],
            workingDirectory: nil,
            label: "missing"
        )
        try await transport.start()

        let finished = expectation(description: "message stream finished")
        Task {
            for await _ in transport.messages {}
            finished.fulfill()
        }
        await fulfillment(of: [finished], timeout: 10)

        XCTAssertNotNil(transport.closureReason)
    }
}

// MARK: - End To End (real stdio process)

/// Drives the whole stack against a real long-lived child process: the
/// persistent stdin/stdout framing, stderr draining, the initialize handshake,
/// tool discovery, and a tool call. This is the path a mock cannot cover.
final class MCPStdioEndToEndTests: XCTestCase {

    private static let serverScript = #"""
    import json
    import sys

    def send(payload):
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")

        # Chatty stderr: an unread stderr pipe would block this process at 64 KB.
        sys.stderr.write("noise from the test server\n")
        sys.stderr.flush()

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "smoke", "version": "1.0.0"},
                "instructions": "smoke instructions"
            }})
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": [{
                "name": "echo",
                "description": "Echoes the input.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"]
                },
                "annotations": {"readOnlyHint": True}
            }]}})
        elif method == "tools/call":
            params = message.get("params") or {}
            value = (params.get("arguments") or {}).get("value", "")
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "content": [{"type": "text", "text": "echo:" + str(value)}]
            }})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "unsupported"}})
    """#

    func testRealServerHandshakeDiscoveryAndCall() async throws {
        let python = "/usr/bin/python3"
        guard FileManager.default.isExecutableFile(atPath: python) else {
            throw XCTSkip("python3 is not available on this host")
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-mcp-e2e-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let scriptURL = directory.appendingPathComponent("server.py")
        try Self.serverScript.write(to: scriptURL, atomically: true, encoding: .utf8)

        let resolved = MCPResolvedServer(
            key: "smoke",
            kind: .stdio,
            command: python,
            arguments: [scriptURL.path],
            environment: [:],
            workingDirectory: nil,
            endpoint: nil,
            headers: [:],
            startupTimeoutSec: 15,
            toolTimeoutSec: 15,
            rawConfig: MCPServerConfig(command: python, args: [scriptURL.path])
        )

        let client = try MCPClient(config: resolved)
        do {
            try await client.start()
            let negotiated = await client.negotiatedProtocolVersion
            XCTAssertEqual(negotiated, "2025-06-18")
            let name = await client.serverName
            XCTAssertEqual(name, "smoke")

            let tools = try await client.listTools()
            XCTAssertEqual(tools.map(\.name), ["echo"])
            XCTAssertTrue(tools[0].readOnlyHint)

            let result = try await client.callTool(name: "echo", arguments: ["value": "hi"])
            XCTAssertEqual(result.text, "echo:hi")
            XCTAssertFalse(result.isError)

            // The transport must have drained stderr while the server ran.
            let stderr = await client.stderrTail
            XCTAssertTrue(stderr.contains("noise from the test server"), stderr)
        } catch {
            await client.stop()
            throw error
        }
        await client.stop()
    }
}

// MARK: - Config Store

final class MCPConfigStoreTests: XCTestCase {

    private func temporaryFile() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-mcp-tests-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("mcp.json")
    }

    func testSaveAndLoadRoundTrip() throws {
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        var document = MCPConfigDocument()
        document["fs"] = MCPServerConfig(command: "npx", args: ["-y", "server"])
        try MCPConfigStore.save(document, to: url)

        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let loaded = try MCPConfigStore.load(from: url)
        XCTAssertEqual(loaded, document)
    }

    func testLoadOrSeedCreatesTemplate() throws {
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let seeded = try MCPConfigStore.loadOrSeed(at: url)
        XCTAssertTrue(seeded.mcpServers.isEmpty)
        XCTAssertEqual(seeded.version, MCPConfigDocument.currentVersion)
    }

    func testSaveRawJSONRejectsInvalidJSON() {
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        XCTAssertThrowsError(try MCPConfigStore.saveRawJSON("{ not json", to: url))
    }

    func testSaveRawJSONAcceptsValidDocument() throws {
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let text = """
        { "mcpServers": { "fs": { "command": "npx" } } }
        """
        let document = try MCPConfigStore.saveRawJSON(text, to: url)
        XCTAssertNotNil(document.mcpServers["fs"])
        XCTAssertNotNil(MCPConfigStore.modificationToken(at: url))
    }

    func testMalformedFileReportsError() {
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? Data("[]".utf8).write(to: url)
        XCTAssertThrowsError(try MCPConfigStore.load(from: url))
    }
}

// MARK: - Keychain Store

final class MCPSecretStoreTests: XCTestCase {

    func testAccountNameIsDeterministic() {
        XCTAssertEqual(
            MCPSecretStore.account(serverKey: "linear", field: "Authorization"),
            "dynamoe.mcp.linear.Authorization"
        )
    }

    func testSetGetDeleteRoundTrip() throws {
        let account = "dynamoe.mcp.tests.\(UUID().uuidString)"
        defer { MCPSecretStore.delete(account: account) }

        let secret = "sk-test-\(UUID().uuidString)"
        do {
            try MCPSecretStore.set(secret, account: account)
        } catch {
            throw XCTSkip("Keychain is not reachable from this test host: \(error.localizedDescription)")
        }
        XCTAssertEqual(MCPSecretStore.get(account: account), secret)
        XCTAssertTrue(MCPSecretStore.exists(account: account))

        MCPSecretStore.delete(account: account)
        XCTAssertNil(MCPSecretStore.get(account: account))
    }

    func testRefusesEmptySecret() {
        XCTAssertThrowsError(try MCPSecretStore.set("", account: "dynamoe.mcp.tests.empty"))
    }
}

// MARK: - Agent Harness Integration

final class MCPAgentHarnessIntegrationTests: XCTestCase {

    private func makeBridge(serverKey: String = "testserver", tool: String = "echo") -> MCPToolBridge {
        let client = MCPClient(config: resolvedServer(key: serverKey), transport: MockMCPTransport())
        return MCPToolBridge(serverKey: serverKey, descriptor: makeDescriptor(name: tool), client: client)
    }

    func testBatchRegistrationAndPrefixUnregistration() {
        let harness = AgentHarness.shared
        let before = harness.tools.count
        let prefix = "mcp__testserver__"

        let bridge = makeBridge()
        harness.registerTools([bridge])
        XCTAssertNotNil(harness.tools["mcp__testserver__echo"])
        XCTAssertEqual(harness.tools.count, before + 1)
        // Lazy by design: registration must not load it into the prompt.
        XCTAssertNil(harness.loadedTools["mcp__testserver__echo"])

        let removed = harness.unregisterTools(prefix: prefix)
        XCTAssertEqual(removed, 1)
        XCTAssertNil(harness.tools["mcp__testserver__echo"])
        XCTAssertEqual(harness.tools.count, before)
    }

    func testLoadedStateSurvivesReRegistration() throws {
        let harness = AgentHarness.shared
        defer { harness.unregisterTools(prefix: "mcp__testserver__") }

        let first = makeBridge()
        harness.registerTools([first])
        _ = try harness.loadTool(named: "mcp__testserver__echo")
        XCTAssertNotNil(harness.loadedTools["mcp__testserver__echo"])

        let second = makeBridge()
        harness.registerTools([second])
        XCTAssertNotNil(harness.loadedTools["mcp__testserver__echo"])
    }

    func testMCPToolsAreAlwaysStateChanging() {
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "mcp__filesystem__read_file"))
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "mcp__linear__list_issues"))
        XCTAssertTrue(AgentHarness.isStateChanging(toolName: "shell_run"))
        XCTAssertFalse(AgentHarness.isStateChanging(toolName: "web_fetch"))
    }

    func testEmptyRequiredArgumentsUsesSchemaForMCPTools() {
        let harness = AgentHarness.shared
        defer { harness.unregisterTools(prefix: "mcp__testserver__") }

        harness.registerTools([makeBridge()])

        // `query` is required by the descriptor schema, so an all-empty call is
        // caught even though the name is absent from the static table.
        XCTAssertTrue(
            AgentHarness.hasEmptyRequiredArguments(toolName: "mcp__testserver__echo", arguments: ["query": ""])
        )
        XCTAssertFalse(
            AgentHarness.hasEmptyRequiredArguments(toolName: "mcp__testserver__echo", arguments: ["query": "x"])
        )
    }

    func testCatalogCategoryIsMCP() {
        XCTAssertEqual(makeBridge().catalogCategory, "mcp")
    }

    func testToolsDiscoverListsMCPCategory() {
        let harness = AgentHarness.shared
        defer { harness.unregisterTools(prefix: "mcp__testserver__") }
        harness.registerTools([makeBridge()])

        let definition = ToolDiscoverTool().definition
        let category = definition.function.parameters["properties"]?.value as? [String: Any]
        let enumValues = category?["category"] as? [String: Any]
        let values = (enumValues?["enum"] as? [String]) ?? []
        XCTAssertTrue(values.contains("mcp"))
    }
}

// MARK: - Manager Gating

@MainActor
final class MCPManagerGatingTests: XCTestCase {

    func testMasterSwitchDefaultsToEnabledWhenUnset() {
        let key = MCPManager.enabledKey
        let original = UserDefaults.standard.object(forKey: key)
        defer {
            if let original {
                UserDefaults.standard.set(original, forKey: key)
            } else {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }

        UserDefaults.standard.removeObject(forKey: key)
        XCTAssertTrue(MCPManager.isEnabledRequested)

        UserDefaults.standard.set(false, forKey: key)
        XCTAssertFalse(MCPManager.isEnabledRequested)
    }

    func testConfigPathPointsInsideApplicationSupport() {
        let path = MCPConfigStore.defaultPathDescription
        XCTAssertTrue(path.hasSuffix("mcp.json"), path)
        XCTAssertTrue(path.contains("DynaMoE"), path)
    }

    func testRawJSONRoundTripsThroughManagerDocument() {
        let text = MCPManager.shared.rawJSONText()
        XCTAssertFalse(text.isEmpty)
        XCTAssertNotNil(try? JSONDecoder().decode(MCPConfigDocument.self, from: Data(text.utf8)))
    }
}

// MARK: - Connection Attempt Ownership

/// Parks the first handshake so a restart or config edit can land while the
/// transport is still starting; the responses can be released afterwards to
/// prove the obsolete attempt leaves the replacement alone.
private final class ParkedMCPHandshake: @unchecked Sendable {
    private let lock = NSLock()
    private var parked: [(id: Any, transport: MockMCPTransport)] = []
    let tools: [[String: Any]]

    init(tools: [[String: Any]] = [["name": "echo", "inputSchema": ["type": "object"]]]) {
        self.tools = tools
    }

    var parkedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return parked.count
    }

    /// Keeps `initialize` unanswered and answers everything else the way a
    /// healthy server would.
    func attach(to transport: MockMCPTransport) {
        let tools = self.tools
        transport.responder = { [weak self] request, transport in
            guard let id = request["id"] else { return }
            switch request["method"] as? String {
            case "initialize":
                self?.park(id: id, transport: transport)
            case "tools/list":
                transport.reply(id: id, result: ["tools": tools])
            default:
                transport.replyError(id: id, code: -32601, message: "unsupported")
            }
        }
    }

    private func park(id: Any, transport: MockMCPTransport) {
        lock.lock()
        parked.append((id, transport))
        lock.unlock()
    }

    func releaseParkedInitialize() {
        lock.lock()
        let requests = parked
        parked = []
        lock.unlock()
        for request in requests {
            request.transport.reply(id: request.id, result: [
                "protocolVersion": "2025-06-18",
                "capabilities": ["tools": [:]],
                "serverInfo": ["name": "parked-server", "version": "1"]
            ])
        }
    }
}

@MainActor
final class MCPManagerAttemptOwnershipTests: XCTestCase {
    private var savedEnabled: Any?
    private var savedAgentTools: Any?

    override func setUp() async throws {
        savedEnabled = UserDefaults.standard.object(forKey: MCPManager.enabledKey)
        savedAgentTools = UserDefaults.standard.object(forKey: MCPManager.agentToolsKey)
        UserDefaults.standard.set(true, forKey: MCPManager.enabledKey)
        UserDefaults.standard.set(true, forKey: MCPManager.agentToolsKey)
    }

    override func tearDown() async throws {
        MCPManager.shared.transportFactory = nil
        await MCPManager.shared.shutdown()
        AgentHarness.shared.unregisterTools(prefix: "mcp__race__")
        if let savedEnabled {
            UserDefaults.standard.set(savedEnabled, forKey: MCPManager.enabledKey)
        } else {
            UserDefaults.standard.removeObject(forKey: MCPManager.enabledKey)
        }
        if let savedAgentTools {
            UserDefaults.standard.set(savedAgentTools, forKey: MCPManager.agentToolsKey)
        } else {
            UserDefaults.standard.removeObject(forKey: MCPManager.agentToolsKey)
        }
    }

    private func document(command: String) -> MCPConfigDocument {
        var document = MCPConfigDocument()
        document["race"] = MCPServerConfig(command: command)
        return document
    }

    /// First attempt parks its `initialize`; every later attempt completes a
    /// normal handshake so the test can observe the replacement.
    private func installTransportFactory(
        parking gate: ParkedMCPHandshake,
        into record: @escaping (MockMCPTransport) -> Void
    ) {
        var attempts = 0
        let tools = gate.tools
        MCPManager.shared.transportFactory = { _ in
            let transport = MockMCPTransport()
            if attempts == 0 {
                gate.attach(to: transport)
            } else {
                transport.responder = { request, transport in
                    guard let id = request["id"] else { return }
                    switch request["method"] as? String {
                    case "initialize":
                        transport.reply(id: id, result: [
                            "protocolVersion": "2025-06-18",
                            "capabilities": ["tools": [:]],
                            "serverInfo": ["name": "replacement-server", "version": "1"]
                        ])
                    case "tools/list":
                        transport.reply(id: id, result: ["tools": tools])
                    default:
                        transport.replyError(id: id, code: -32601, message: "unsupported")
                    }
                }
            }
            attempts += 1
            record(transport)
            return transport
        }
    }

    private func waitUntil(
        _ description: String,
        timeout: TimeInterval = 3,
        _ condition: @MainActor () -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(condition(), "timed out waiting for \(description)", file: file, line: line)
    }

    func testConfigEditDuringHandshakeStartsTheReplacement() async throws {
        let manager = MCPManager.shared
        let gate = ParkedMCPHandshake()
        var transports: [MockMCPTransport] = []
        installTransportFactory(parking: gate) { transports.append($0) }

        await manager.installDocumentForTesting(document(command: "first"))
        try await waitUntil("the first handshake to park") { gate.parkedCount == 1 }

        // The config changes while the first server is still handshaking.
        await manager.installDocumentForTesting(document(command: "second"))
        try await waitUntil("the replacement connection") {
            transports.count == 2 && manager.servers.first?.status == .ready
        }
        XCTAssertEqual(manager.servers.first?.remoteToolNames, ["echo"])
        XCTAssertNotNil(AgentHarness.shared.tools["mcp__race__echo"])

        // The abandoned handshake finishes late; it must not publish, and it
        // must not unregister the replacement's tools.
        gate.releaseParkedInitialize()
        try await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(transports.count, 2, "the abandoned attempt must not start a third connection")
        XCTAssertEqual(manager.servers.first?.status, .ready)
        XCTAssertNotNil(
            AgentHarness.shared.tools["mcp__race__echo"],
            "the abandoned attempt unregistered the replacement's tools"
        )
        XCTAssertTrue(transports[0].isClosed, "the abandoned attempt leaked its transport")
        XCTAssertNil(transports[0].closureReason, "a deliberate teardown must not read as a server exit")
    }

    func testRestartDuringHandshakeKeepsTheReplacementBinding() async throws {
        let manager = MCPManager.shared
        let gate = ParkedMCPHandshake()
        var transports: [MockMCPTransport] = []
        installTransportFactory(parking: gate) { transports.append($0) }

        await manager.installDocumentForTesting(document(command: "first"))
        try await waitUntil("the first handshake to park") { gate.parkedCount == 1 }

        manager.restart(key: "race")
        try await waitUntil("the replacement connection") {
            transports.count == 2 && manager.servers.first?.status == .ready
        }

        // The restarted attempt unwinds after the replacement is live.
        gate.releaseParkedInitialize()
        try await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(manager.servers.first?.status, .ready)
        XCTAssertNotNil(
            AgentHarness.shared.tools["mcp__race__echo"],
            "the superseded attempt unregistered the replacement's tools"
        )

        // A later restart must still be able to cancel the live attempt: the
        // superseded task must not have cleared its bookkeeping. Here the live
        // attempt is already connected, so restart closes its transport and
        // starts a third one.
        manager.restart(key: "race")
        try await waitUntil("the restart to start a third attempt") { transports.count == 3 }
        XCTAssertTrue(transports[1].isClosed, "restart did not stop the live replacement")
    }

    func testToolListRefreshUpdatesTheLiveConnection() async throws {
        let manager = MCPManager.shared
        let server = MutableToolMockServer(tools: [["name": "echo", "inputSchema": ["type": "object"]]])
        manager.transportFactory = { _ in server.transport }

        await manager.installDocumentForTesting(document(command: "first"))
        try await waitUntil("the connection") { manager.servers.first?.status == .ready }
        XCTAssertEqual(manager.servers.first?.remoteToolNames, ["echo"])

        server.tools = [["name": "echo2", "inputSchema": ["type": "object"]]]
        server.announceToolListChanged()

        try await waitUntil("the refreshed tool list") { manager.servers.first?.remoteToolNames == ["echo2"] }
        XCTAssertNotNil(AgentHarness.shared.tools["mcp__race__echo2"])
        XCTAssertNil(AgentHarness.shared.tools["mcp__race__echo"])
    }

    func testToolListRefreshFromAReplacedClientIsDiscarded() async throws {
        let manager = MCPManager.shared
        let server = MutableToolMockServer(tools: [["name": "echo2", "inputSchema": ["type": "object"]]])
        manager.transportFactory = { _ in server.transport }

        await manager.installDocumentForTesting(document(command: "first"))
        try await waitUntil("the connection") { manager.servers.first?.status == .ready }

        // The client a restart already closed: it still answers `tools/list`,
        // but it no longer owns the server's connection.
        let replaced = MCPClient(config: resolvedServer(key: "race"), transport: MockMCPTransport())
        let applied = manager.applyToolRefresh(
            [makeDescriptor(name: "stale")],
            from: replaced,
            serverKey: "race"
        )

        XCTAssertFalse(applied, "a replaced client must not publish a tool list")
        XCTAssertEqual(manager.servers.first?.remoteToolNames, ["echo2"])
        XCTAssertNotNil(AgentHarness.shared.tools["mcp__race__echo2"])
        XCTAssertNil(AgentHarness.shared.tools["mcp__race__stale"])
    }
}

/// A mock MCP server whose tool list changes between refreshes.
private final class MutableToolMockServer: @unchecked Sendable {
    let transport = MockMCPTransport()
    private let lock = NSLock()
    private var currentTools: [[String: Any]]
    private let serverName: String

    init(tools: [[String: Any]], serverName: String = "mutable-server") {
        self.currentTools = tools
        self.serverName = serverName
        transport.responder = { [weak self] request, transport in
            guard let id = request["id"] else { return }
            switch request["method"] as? String {
            case "initialize":
                transport.reply(id: id, result: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": ["tools": [:]],
                    "serverInfo": ["name": self?.serverName ?? "mutable-server", "version": "1"]
                ])
            case "tools/list":
                transport.reply(id: id, result: ["tools": self?.tools ?? []])
            default:
                transport.replyError(id: id, code: -32601, message: "unsupported")
            }
        }
    }

    var tools: [[String: Any]] {
        get {
            lock.lock()
            defer { lock.unlock() }
            return currentTools
        }
        set {
            lock.lock()
            currentTools = newValue
            lock.unlock()
        }
    }

    func announceToolListChanged() {
        transport.pushNotification(method: "notifications/tools/list_changed")
    }
}
